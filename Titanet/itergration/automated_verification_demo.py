#!/usr/bin/env python3
"""
Example: Automated Speaker Verification with Voice-to-Text Integration
Demo script showing how to integrate the speaker verification pipeline
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from speaker_verification_pipeline import SpeakerVerificationPipeline, VerificationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceToTextMock:
    """Mock voice-to-text model for demonstration"""
    def transcribe(self, audio_path: str) -> str:
        # Replace this with your actual voice-to-text model
        # Examples: OpenAI Whisper, Google Speech-to-Text, etc.
        return f"[TRANSCRIPTION FROM {Path(audio_path).name}] This is a mock transcription."

class AutomatedVerificationSystem:
    """Automated system combining speaker verification with voice-to-text"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize speaker verification pipeline
        self.speaker_pipeline = SpeakerVerificationPipeline(self.config)
        
        # Initialize voice-to-text model (replace with your model)
        self.voice_to_text = VoiceToTextMock()
        
        logger.info("Automated verification system initialized")
    
    def _load_config(self, config_path: str = None) -> VerificationConfig:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return VerificationConfig(**config_dict)
        else:
            # Production-ready defaults
            return VerificationConfig(
                titanet_model_path="/home/edabk408/NgocDat/Titanet/titanet-l.nemo",
                use_vad=True,
                vad_threshold=0.5,
                similarity_threshold=0.65,  # Adjust based on your EER tests
                min_audio_duration=1.0,
                max_audio_duration=60.0,
                device="cuda",
                temp_dir="/tmp/automated_verification",
                enrollment_db_path="./production_speakers.json"
            )
    
    def setup_speakers(self, speakers_data: Dict[str, List[str]]) -> None:
        """Setup speakers for the system"""
        logger.info(f"Setting up {len(speakers_data)} speakers...")
        
        for speaker_id, audio_files in speakers_data.items():
            success = self.speaker_pipeline.enroll_speaker(speaker_id, audio_files)
            if success:
                logger.info(f"Speaker {speaker_id} enrolled successfully")
            else:
                logger.error(f"Failed to enroll speaker {speaker_id}")
    
    def process_audio_sequential(self, audio_path: str, expected_speaker: str = None) -> Dict:
        """Process audio with sequential verification then transcription"""
        logger.info(f"Processing {audio_path} (sequential mode)")
        
        # Step 1: Speaker verification
        verification_result = self.speaker_pipeline.verify_speaker(audio_path, expected_speaker)
        
        # Step 2: Transcription (only if verified or if no speaker expected)
        transcription = None
        if verification_result.get('verified', False) or expected_speaker is None:
            try:
                transcription = self.voice_to_text.transcribe(audio_path)
                logger.info("Transcription completed")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
        else:
            logger.info("Skipping transcription - speaker not verified")
        
        return {
            'audio_path': audio_path,
            'processing_mode': 'sequential',
            'verification': verification_result,
            'transcription': transcription,
            'success': verification_result.get('success', False)
        }
    
    def process_audio_parallel(self, audio_path: str, expected_speaker: str = None) -> Dict:
        """Process audio with parallel verification and transcription"""
        import concurrent.futures
        import time
        
        logger.info(f"Processing {audio_path} (parallel mode)")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks simultaneously
            verification_future = executor.submit(
                self.speaker_pipeline.verify_speaker, audio_path, expected_speaker
            )
            transcription_future = executor.submit(
                self.voice_to_text.transcribe, audio_path
            )
            
            # Wait for results
            verification_result = verification_future.result()
            transcription = transcription_future.result()
        
        processing_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {processing_time:.2f}s")
        
        return {
            'audio_path': audio_path,
            'processing_mode': 'parallel',
            'processing_time': processing_time,
            'verification': verification_result,
            'transcription': transcription,
            'success': verification_result.get('success', False)
        }
    
    def process_audio_vad_shared(self, audio_path: str, expected_speaker: str = None) -> Dict:
        """Process audio with shared VAD preprocessing"""
        import tempfile
        import torchaudio
        
        logger.info(f"Processing {audio_path} (shared VAD mode)")
        
        try:
            # Step 1: Shared VAD preprocessing
            waveform, sample_rate = self.speaker_pipeline.preprocess_audio(audio_path)
            vad_audio = self.speaker_pipeline.apply_vad(waveform, sample_rate)
            
            # Save VAD-processed audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, vad_audio, sample_rate)
                vad_audio_path = tmp_file.name
            
            # Step 2: Parallel processing with VAD-cleaned audio
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                verification_future = executor.submit(
                    self.speaker_pipeline.verify_speaker, vad_audio_path, expected_speaker
                )
                transcription_future = executor.submit(
                    self.voice_to_text.transcribe, vad_audio_path
                )
                
                verification_result = verification_future.result()
                transcription = transcription_future.result()
            
            # Cleanup
            Path(vad_audio_path).unlink()
            
            return {
                'audio_path': audio_path,
                'processing_mode': 'vad_shared',
                'verification': verification_result,
                'transcription': transcription,
                'vad_applied': True,
                'success': verification_result.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"VAD shared processing failed: {e}")
            # Fallback to sequential processing
            return self.process_audio_sequential(audio_path, expected_speaker)
    
    def batch_process(self, audio_files: List[str], expected_speakers: List[str] = None, 
                     mode: str = "parallel") -> List[Dict]:
        """Batch process multiple audio files"""
        logger.info(f"Batch processing {len(audio_files)} files in {mode} mode")
        
        results = []
        expected_speakers = expected_speakers or [None] * len(audio_files)
        
        for i, audio_path in enumerate(audio_files):
            expected_speaker = expected_speakers[i] if i < len(expected_speakers) else None
            
            if mode == "sequential":
                result = self.process_audio_sequential(audio_path, expected_speaker)
            elif mode == "vad_shared":
                result = self.process_audio_vad_shared(audio_path, expected_speaker)
            else:  # parallel
                result = self.process_audio_parallel(audio_path, expected_speaker)
            
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        enrollment_stats = self.speaker_pipeline.get_enrollment_stats()
        
        return {
            'pipeline_config': {
                'use_vad': self.config.use_vad,
                'vad_threshold': self.config.vad_threshold,
                'similarity_threshold': self.config.similarity_threshold,
                'device': self.config.device
            },
            'enrollment_stats': enrollment_stats
        }

def demo_usage():
    """Demonstrate the automated verification system"""
    
    # Initialize system
    system = AutomatedVerificationSystem()
    
    # Example: Setup speakers (you would use real audio files)
    example_speakers = {
        "user001": ["path/to/user001_enroll1.wav", "path/to/user001_enroll2.wav"],
        "user002": ["path/to/user002_enroll1.wav", "path/to/user002_enroll2.wav"]
    }
    
    # Uncomment to enroll speakers
    # system.setup_speakers(example_speakers)
    
    # Example: Process single audio file
    test_audio = "path/to/test_audio.wav"
    
    print("=== Sequential Processing ===")
    result_seq = system.process_audio_sequential(test_audio, "user001")
    print(json.dumps(result_seq, indent=2))
    
    print("\n=== Parallel Processing ===")
    result_par = system.process_audio_parallel(test_audio, "user001")
    print(json.dumps(result_par, indent=2))
    
    print("\n=== VAD Shared Processing ===")
    result_vad = system.process_audio_vad_shared(test_audio, "user001")
    print(json.dumps(result_vad, indent=2))
    
    # Example: Batch processing
    test_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    expected_speakers = ["user001", "user002", "user001"]
    
    print("\n=== Batch Processing ===")
    batch_results = system.batch_process(test_files, expected_speakers, mode="parallel")
    
    for i, result in enumerate(batch_results):
        print(f"File {i+1}: Verified={result['verification'].get('verified', False)}, "
              f"Transcription={'Yes' if result['transcription'] else 'No'}")
    
    # System stats
    print("\n=== System Statistics ===")
    stats = system.get_system_stats()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    demo_usage()