"""
Pipeline đơn giản cho Jetson
"""

import torch
import os
import numpy as np
from speaker_verification_pipeline import SpeakerVerificationPipeline
from jetson_config import get_jetson_config

class JetsonSpeakerPipeline:
    """Pipeline đơn giản cho Jetson"""
    
    def __init__(self):
        self.config = get_jetson_config()
        self.pipeline = None
        self.model = None
    
    def _optimize_for_jetson(self):
        """Apply Jetson-specific optimizations"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            try:
                # Enable half precision if supported and requested
                if self.jetson_config.precision == "fp16" and hasattr(self.speaker_model, 'half'):
                    logger.info("Enabling FP16 precision for Jetson")
                    self.speaker_model = self.speaker_model.half()
                
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    logger.info("Compiling model for Jetson optimization")
                    self.speaker_model = torch.compile(self.speaker_model, mode="reduce-overhead")
                
                # Enable TensorRT optimization if available
                if self.jetson_config.enable_tensorrt:
                    try:
                        import torch_tensorrt
                        logger.info("TensorRT available for optimization")
                    except ImportError:
                        logger.info("TensorRT not available, using standard CUDA")
                
            except Exception as e:
                logger.warning(f"Some Jetson optimizations failed: {e}")
    
    def _load_models(self):
        """Jetson-optimized model loading with memory management"""
        try:
            logger.info("Loading models for Jetson...")
            
            # Clear any existing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load TitaNet-L with memory mapping
            logger.info("Loading TitaNet-L model...")
            self.speaker_model = EncDecSpeakerLabelModel.restore_from(
                self.config.titanet_model_path,
                map_location=self.device
            )
            self.speaker_model = self.speaker_model.eval().to(self.device)
            
            # Optimize model for inference
            if hasattr(torch.jit, 'optimize_for_inference'):
                self.speaker_model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.speaker_model)
                )
            
            logger.info("TitaNet-L loaded successfully")
            
            # Load Silero VAD with memory optimization
            if self.config.use_vad:
                logger.info("Loading Silero VAD v6...")
                
                # Load with reduced memory footprint
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                self.vad_model = self.vad_model.to(self.device)
                
                # Optimize VAD for inference
                self.vad_model.eval()
                
                logger.info("Silero VAD v6 loaded successfully")
            else:
                self.vad_model = None
                self.vad_utils = None
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Jetson model loading completed")
            
        except Exception as e:
            logger.error(f"Jetson model loading failed: {e}")
            raise
    
    @contextmanager
    def _memory_management(self):
        """Context manager for memory-efficient processing"""
        # Pre-processing cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            yield
        finally:
            # Post-processing cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Memory-optimized embedding extraction for Jetson"""
        with self._memory_management():
            try:
                # Check cache first
                if self.jetson_config.cache_embeddings:
                    cache_key = f"{audio_path}_{os.path.getmtime(audio_path)}"
                    if cache_key in self.embedding_cache:
                        logger.debug(f"Using cached embedding for {audio_path}")
                        return self.embedding_cache[cache_key]
                
                # Preprocess audio with memory optimization
                waveform, sample_rate = self.preprocess_audio(audio_path)
                
                # Check duration
                duration = len(waveform) / sample_rate
                if duration < self.config.min_audio_duration:
                    logger.warning(f"Audio too short: {duration:.2f}s < {self.config.min_audio_duration}s")
                    return None
                
                if duration > self.config.max_audio_duration:
                    logger.warning(f"Audio too long: {duration:.2f}s, truncating to {self.config.max_audio_duration}s")
                    max_samples = int(self.config.max_audio_duration * sample_rate)
                    waveform = waveform[:max_samples]
                
                # Apply VAD with memory optimization
                waveform = self.apply_vad(waveform, sample_rate)
                
                # Save processed audio temporarily
                from datetime import datetime
                temp_path = os.path.join(self.config.temp_dir, f"temp_{datetime.now().timestamp()}.wav")
                
                # Convert to int16 for WAV format
                from scipy.io import wavfile
                waveform_int16 = (waveform * 32767).astype(np.int16)
                wavfile.write(temp_path, sample_rate, waveform_int16)
                
                # Extract embedding with mixed precision if available
                with torch.no_grad():
                    if self.jetson_config.precision == "fp16" and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            embedding = self.speaker_model.get_embedding(temp_path)
                    else:
                        embedding = self.speaker_model.get_embedding(temp_path)
                    
                    embedding = embedding.cpu().numpy().squeeze()
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Cache embedding if enabled
                if self.jetson_config.cache_embeddings:
                    # Manage cache size
                    if len(self.embedding_cache) >= self.jetson_config.max_cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self.embedding_cache))
                        del self.embedding_cache[oldest_key]
                    
                    self.embedding_cache[cache_key] = embedding
                
                return embedding
                
            except Exception as e:
                logger.error(f"Jetson embedding extraction failed for {audio_path}: {e}")
                return None
    
    def apply_vad(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Jetson-optimized VAD processing"""
        if not self.config.use_vad or self.vad_model is None:
            return waveform
        
        try:
            with self._memory_management():
                # Convert numpy to torch tensor for VAD
                audio_tensor = torch.from_numpy(waveform).float()
                
                # Move to device if using CUDA
                if self.config.device == "cuda":
                    audio_tensor = audio_tensor.to(self.device)
                
                # Get speech timestamps with optimized settings
                get_speech_timestamps = self.vad_utils.get('get_speech_timestamps')
                collect_chunks = self.vad_utils.get('collect_chunks')
                
                with torch.no_grad():
                    timestamps = get_speech_timestamps(
                        audio_tensor,
                        self.vad_model,
                        sampling_rate=sample_rate,
                        threshold=self.config.vad_threshold,
                        min_speech_duration_ms=self.config.vad_min_speech_duration,
                        max_speech_duration_s=self.config.vad_max_speech_duration // 1000
                    )
                
                if not timestamps:
                    logger.warning("No speech detected by VAD")
                    return waveform
                
                # Collect speech chunks
                speech_audio = collect_chunks(timestamps, audio_tensor)
                
                if speech_audio is None or len(speech_audio) == 0:
                    logger.warning("VAD returned empty audio")
                    return waveform
                
                # Convert back to numpy
                return speech_audio.cpu().numpy()
                
        except Exception as e:
            logger.warning(f"VAD failed on Jetson, using original audio: {e}")
            return waveform
    
    def get_jetson_stats(self) -> Dict:
        """Get Jetson performance statistics"""
        stats = {}
        
        # CPU and Memory
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
        stats['memory_percent'] = psutil.virtual_memory().percent
        stats['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
        
        # GPU stats (if CUDA available)
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_utilization'] = torch.cuda.utilization()
        
        # Temperature (Jetson specific)
        try:
            temp_zones = ['/sys/class/thermal/thermal_zone0/temp',
                         '/sys/class/thermal/thermal_zone1/temp']
            temps = []
            for zone in temp_zones:
                if os.path.exists(zone):
                    with open(zone, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        temps.append(temp)
            if temps:
                stats['temperature_c'] = max(temps)
        except:
            stats['temperature_c'] = None
        
        # Embedding cache stats
        stats['cache_size'] = len(self.embedding_cache)
        stats['cache_hit_ratio'] = getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Jetson cache cleared")

# Convenience function for Jetson deployment
def create_jetson_pipeline(jetson_model: str = None, model_path: str = None) -> JetsonSpeakerPipeline:
    """Create Jetson-optimized pipeline with auto-detection"""
    from jetson_config import create_jetson_config, get_jetson_model
    
    if jetson_model is None:
        jetson_model = get_jetson_model()
        logger.info(f"Auto-detected Jetson model: {jetson_model}")
    
    config = create_jetson_config(jetson_model, model_path)
    return JetsonSpeakerPipeline(config)

if __name__ == "__main__":
    # Example usage for Jetson
    print("Creating Jetson-optimized speaker verification pipeline...")
    
    try:
        pipeline = create_jetson_pipeline()
        print("✓ Pipeline created successfully")
        
        # Show performance stats
        stats = pipeline.get_jetson_stats()
        print(f"\nJetson Performance Stats:")
        print(f"  CPU Usage: {stats['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {stats['memory_percent']:.1f}%")
        print(f"  Available Memory: {stats['available_memory_gb']:.2f}GB")
        if stats['temperature_c']:
            print(f"  Temperature: {stats['temperature_c']:.1f}°C")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory Used: {stats['gpu_memory_allocated']:.2f}GB")
            print(f"  GPU Memory Cached: {stats['gpu_memory_cached']:.2f}GB")
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")