#!/usr/bin/env python3
"""
Jetson Nano ONNX Pipeline
Optimized real-time speaker verification using ONNX models
"""

import os
import sys
import time
import json
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JetsonONNXPipeline:
    """Optimized ONNX-based speaker verification for Jetson Nano"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.config = self._load_config(config_path)
        
        # Model paths
        self.onnx_model_path = self.base_dir / "onnx_models" / "titanet-l-dynamic-quantized.onnx"
        if not self.onnx_model_path.exists():
            self.onnx_model_path = self.base_dir / "onnx_models" / "titanet-l.onnx"
        
        # Runtime components
        self.onnx_session = None
        self.vad_model = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.embeddings_db = {}
        
        # Performance monitoring
        self.performance_stats = {
            'total_processed': 0,
            'avg_inference_time': 0,
            'avg_embedding_time': 0,
            'memory_usage': []
        }
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration"""
        default_config = {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 1.0,
                'vad_threshold': 0.5,
                'min_speech_duration': 1.0,
                'max_speech_duration': 10.0
            },
            'inference': {
                'onnx_providers': ['CPUExecutionProvider'],
                'batch_size': 1,
                'embedding_threshold': 0.85,
                'max_concurrent_streams': 2
            },
            'jetson': {
                'enable_gpu': False,  # ONNX GPU support limited on Jetson Nano
                'cpu_threads': 4,
                'memory_limit_mb': 512,
                'enable_tensorrt': False  # For future TensorRT conversion
            },
            'realtime': {
                'enable_realtime': True,
                'buffer_duration': 3.0,
                'overlap_duration': 0.5,
                'max_latency_ms': 500
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config {config_path}: {e}")
        
        return default_config
    
    def _check_jetson_environment(self) -> Dict:
        """Check Jetson Nano environment and capabilities"""
        env_info = {
            'jetson_detected': False,
            'cuda_available': False,
            'tensorrt_available': False,
            'cpu_count': os.cpu_count(),
            'memory_gb': 0
        }
        
        try:
            # Check if running on Jetson
            with open('/proc/device-tree/model', 'r') as f:
                model_info = f.read().strip()
                if 'jetson' in model_info.lower():
                    env_info['jetson_detected'] = True
                    logger.info(f"Detected Jetson device: {model_info}")
        except:
            pass
        
        # Check memory
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        env_info['memory_gb'] = mem_kb / 1024 / 1024
                        break
        except:
            pass
        
        # Check CUDA
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                env_info['cuda_available'] = True
        except:
            pass
        
        return env_info
    
    def initialize_onnx_session(self) -> bool:
        """Initialize ONNX Runtime session"""
        try:
            import onnxruntime as ort
            
            if not self.onnx_model_path.exists():
                logger.error(f"ONNX model not found: {self.onnx_model_path}")
                return False
            
            logger.info(f"Loading ONNX model: {self.onnx_model_path}")
            
            # Configure session options for Jetson Nano
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config['jetson']['cpu_threads']
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create session
            providers = self.config['inference']['onnx_providers']
            self.onnx_session = ort.InferenceSession(
                str(self.onnx_model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Get model info
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            
            logger.info(f"✓ ONNX session created successfully")
            logger.info(f"  Input: {input_info.name} {input_info.shape}")
            logger.info(f"  Output: {output_info.name} {output_info.shape}")
            logger.info(f"  Providers: {self.onnx_session.get_providers()}")
            
            return True
            
        except ImportError:
            logger.error("ONNX Runtime not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            return False
    
    def initialize_vad(self) -> bool:
        """Initialize VAD (Voice Activity Detection)"""
        try:
            import torch
            
            logger.info("Loading Silero VAD model...")
            
            # Load VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            
            self.vad_model = model
            self.vad_utils = utils
            
            # Extract utility functions
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.VADIterator = utils[3]
            self.collect_chunks = utils[4]
            
            logger.info("✓ VAD initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            return False
    
    def extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding using ONNX model"""
        if self.onnx_session is None:
            logger.error("ONNX session not initialized")
            return None
        
        try:
            start_time = time.time()
            
            # Prepare input
            if len(audio.shape) == 1:
                audio_input = audio.reshape(1, -1).astype(np.float32)
            else:
                audio_input = audio.astype(np.float32)
            
            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: audio_input}
            embedding = self.onnx_session.run(None, ort_inputs)[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            # Update performance stats
            self.performance_stats['total_processed'] += 1
            self.performance_stats['avg_inference_time'] = (
                (self.performance_stats['avg_inference_time'] * (self.performance_stats['total_processed'] - 1) + inference_time) /
                self.performance_stats['total_processed']
            )
            
            logger.debug(f"Embedding extracted: {embedding.shape}, time: {inference_time:.2f}ms")
            
            return embedding.flatten()
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None
    
    def detect_speech(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments using VAD"""
        if self.vad_model is None:
            logger.warning("VAD not initialized, returning full audio")
            return [(0.0, len(audio) / self.config['audio']['sample_rate'])]
        
        try:
            import torch
            
            # Convert to tensor if needed
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio)
            else:
                audio_tensor = audio
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=self.config['audio']['vad_threshold'],
                sampling_rate=self.config['audio']['sample_rate'],
                min_speech_duration_ms=int(self.config['audio']['min_speech_duration'] * 1000),
                max_speech_duration_s=self.config['audio']['max_speech_duration']
            )
            
            # Convert to time ranges
            speech_segments = []
            for segment in speech_timestamps:
                start_time = segment['start'] / self.config['audio']['sample_rate']
                end_time = segment['end'] / self.config['audio']['sample_rate']
                speech_segments.append((start_time, end_time))
            
            logger.debug(f"Detected {len(speech_segments)} speech segments")
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            return [(0.0, len(audio) / self.config['audio']['sample_rate'])]
    
    def load_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file"""
        try:
            import librosa
            
            audio, sr = librosa.load(
                audio_path,
                sr=self.config['audio']['sample_rate'],
                mono=True
            )
            
            logger.info(f"Loaded audio: {len(audio)} samples, {len(audio)/sr:.2f}s")
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None
    
    def register_speaker(self, speaker_id: str, audio_path: str) -> bool:
        """Register a speaker with their voice sample"""
        logger.info(f"Registering speaker: {speaker_id}")
        
        # Load audio
        audio = self.load_audio_file(audio_path)
        if audio is None:
            return False
        
        # Detect speech
        speech_segments = self.detect_speech(audio)
        if not speech_segments:
            logger.error("No speech detected in registration audio")
            return False
        
        # Extract embeddings from all speech segments
        embeddings = []
        sample_rate = self.config['audio']['sample_rate']
        
        for start_time, end_time in speech_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            speech_audio = audio[start_sample:end_sample]
            
            # Skip too short segments
            if len(speech_audio) < sample_rate * 0.5:  # Less than 0.5 seconds
                continue
            
            embedding = self.extract_embedding(speech_audio)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            logger.error("No valid embeddings extracted")
            return False
        
        # Store average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        self.embeddings_db[speaker_id] = {
            'embedding': avg_embedding,
            'num_segments': len(embeddings),
            'registration_time': time.time()
        }
        
        logger.info(f"✓ Speaker {speaker_id} registered with {len(embeddings)} segments")
        
        return True
    
    def verify_speaker(self, audio: np.ndarray, return_scores: bool = False) -> Dict:
        """Verify speaker identity"""
        if not self.embeddings_db:
            return {'verified': False, 'error': 'No registered speakers'}
        
        # Extract embedding from input audio
        embedding = self.extract_embedding(audio)
        if embedding is None:
            return {'verified': False, 'error': 'Could not extract embedding'}
        
        # Compare with registered speakers
        scores = {}
        for speaker_id, speaker_data in self.embeddings_db.items():
            registered_embedding = speaker_data['embedding']
            
            # Cosine similarity
            similarity = np.dot(embedding, registered_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(registered_embedding)
            )
            
            scores[speaker_id] = float(similarity)
        
        # Find best match
        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]
        
        verified = best_score >= self.config['inference']['embedding_threshold']
        
        result = {
            'verified': verified,
            'speaker_id': best_speaker if verified else 'unknown',
            'confidence': best_score,
            'threshold': self.config['inference']['embedding_threshold']
        }
        
        if return_scores:
            result['all_scores'] = scores
        
        return result
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """Process audio file for speaker verification"""
        logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio = self.load_audio_file(audio_path)
        if audio is None:
            return {'error': 'Could not load audio file'}
        
        # Detect speech
        speech_segments = self.detect_speech(audio)
        if not speech_segments:
            return {'error': 'No speech detected'}
        
        # Process each speech segment
        results = []
        sample_rate = self.config['audio']['sample_rate']
        
        for i, (start_time, end_time) in enumerate(speech_segments):
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            speech_audio = audio[start_sample:end_sample]
            
            # Skip too short segments
            if len(speech_audio) < sample_rate * 0.5:
                continue
            
            segment_result = self.verify_speaker(speech_audio, return_scores=True)
            segment_result.update({
                'segment_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
            
            results.append(segment_result)
        
        return {'segments': results}

def main():
    """Main function"""
    print("🚀 Jetson Nano ONNX Speaker Verification Pipeline")
    print("=" * 55)
    
    # Initialize pipeline
    pipeline = JetsonONNXPipeline()
    
    # Check Jetson environment
    env_info = pipeline._check_jetson_environment()
    print(f"\n🖥️  Environment Information:")
    print(f"   Jetson detected: {env_info['jetson_detected']}")
    print(f"   CPU cores: {env_info['cpu_count']}")
    print(f"   Memory: {env_info['memory_gb']:.1f} GB")
    print(f"   CUDA available: {env_info['cuda_available']}")
    
    # Initialize components
    print(f"\n🔧 Initializing components...")
    
    if not pipeline.initialize_onnx_session():
        logger.error("Failed to initialize ONNX session")
        return False
    
    if not pipeline.initialize_vad():
        logger.error("Failed to initialize VAD")
        return False
    
    print(f"\n✅ Pipeline initialized successfully!")
    
    # Interactive mode or test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print(f"\n🧪 Test Mode")
        
        # Test with sample audio
        base_dir = Path(__file__).parent
        test_audio_dir = base_dir.parent / "audio-test"
        
        if test_audio_dir.exists():
            audio_files = list(test_audio_dir.glob("*.wav"))
            if audio_files:
                test_file = audio_files[0]
                print(f"Testing with: {test_file}")
                
                result = pipeline.process_audio_file(str(test_file))
                print(f"Result: {result}")
            else:
                print("No test audio files found")
        else:
            print("No test audio directory found")
    
    else:
        print(f"\n💡 Interactive Mode")
        print(f"Commands:")
        print(f"  register <speaker_id> <audio_file>  - Register speaker")
        print(f"  verify <audio_file>                 - Verify speaker")
        print(f"  stats                               - Show performance stats")
        print(f"  quit                                - Exit")
        
        while True:
            try:
                cmd = input(f"\n> ").strip()
                
                if cmd == 'quit':
                    break
                elif cmd == 'stats':
                    print(f"Performance stats: {pipeline.performance_stats}")
                elif cmd.startswith('register '):
                    parts = cmd.split(' ', 2)
                    if len(parts) == 3:
                        speaker_id, audio_file = parts[1], parts[2]
                        success = pipeline.register_speaker(speaker_id, audio_file)
                        print(f"Registration {'successful' if success else 'failed'}")
                    else:
                        print("Usage: register <speaker_id> <audio_file>")
                elif cmd.startswith('verify '):
                    parts = cmd.split(' ', 1)
                    if len(parts) == 2:
                        audio_file = parts[1]
                        result = pipeline.process_audio_file(audio_file)
                        print(f"Verification result: {result}")
                    else:
                        print("Usage: verify <audio_file>")
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\n👋 Pipeline shutdown complete")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)