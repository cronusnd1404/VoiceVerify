# Speaker Verification Pipeline using TitaNet-L and Silero VAD v6
# Integrates with voice-to-text systems for enrollment and verification
import os
import torch
import librosa
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from scipy.io import wavfile
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationConfig:
    """Configuration for speaker verification pipeline"""
    # Model paths
    titanet_model_path: str = "/home/edabk408/NgocDat/Titanet/integration/titanet-l.nemo"

    # VAD settings
    use_vad: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_duration: int = 250  # ms
    vad_max_speech_duration: int = 30000  # ms
    
    # Verification settings
    similarity_threshold: float = 0.6  # adjustable based on your EER tests
    min_audio_duration: float = 1.0  # seconds
    max_audio_duration: float = 60.0  # seconds
    
    # Processing settings
    target_sample_rate: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    
    # Storage
    temp_dir: str = "/home/edabk408/Titanet/integration/temp"
    enrollment_db_path: str = "./speaker_enrollments.json"
    
    # Output settings
    output_format: str = "json"  # json, csv, or both
    save_embeddings: bool = True
    save_vad_segments: bool = False

class SpeakerVerificationPipeline:
    """Production speaker verification pipeline"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self._load_models()
        
        # Create directories
        os.makedirs(config.temp_dir, exist_ok=True)
        
        # Load enrollment database
        self.enrollment_db = self._load_enrollment_db()
        
        logger.info(f"Speaker verification pipeline initialized on {self.device}")
    
    def _load_models(self):
        """Load TitaNet-L and Silero VAD models"""
        try:
            # Load TitaNet-L
            logger.info("Loading TitaNet-L model...")
            self.speaker_model = EncDecSpeakerLabelModel.restore_from(
                self.config.titanet_model_path
            )
            self.speaker_model = self.speaker_model.eval().to(self.device)
            logger.info("TitaNet-L loaded successfully")
            
            # Load Silero VAD
            if self.config.use_vad:
                logger.info("Loading Silero VAD v6...")
                # Load Silero VAD - returns model and utils tuple
                self.vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                
                # Unpack utils tuple: (get_speech_timestamps, collect_chunks, read_audio, VADIterator, collect_chunks)
                try:
                    self.get_speech_timestamps, self.collect_chunks, _, _, _ = utils
                    logger.info("VAD functions extracted successfully")
                except ValueError as e:
                    logger.warning(f"Could not unpack VAD utils: {e}, trying alternative approach")
                    # Fallback: try accessing by index
                    self.get_speech_timestamps = utils[0] if len(utils) > 0 else None
                    self.collect_chunks = utils[1] if len(utils) > 1 else None
                
                self.vad_model = self.vad_model.to(self.device)
                logger.info("Silero VAD v6 loaded successfully")
                logger.info(f"VAD functions loaded: get_speech_timestamps={self.get_speech_timestamps is not None}, collect_chunks={self.collect_chunks is not None}")
            else:
                self.vad_model = None
                self.get_speech_timestamps = None
                self.collect_chunks = None
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _load_enrollment_db(self) -> Dict:
        """Load speaker enrollment database"""
        if os.path.exists(self.config.enrollment_db_path):
            with open(self.config.enrollment_db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_enrollment_db(self):
        """Save speaker enrollment database"""
        with open(self.config.enrollment_db_path, 'w') as f:
            json.dump(self.enrollment_db, f, indent=2)
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Preprocess audio: load, resample, convert to mono"""
        # Load audio with librosa (automatically converts to mono and resamples)
        waveform, sample_rate = librosa.load(
            audio_path, 
            sr=self.config.target_sample_rate,  # Automatically resample to target rate
            mono=True,  # Convert to mono
            dtype=np.float32
        )
        
        # Ensure 1D array (librosa.load already returns 1D for mono=True)
        waveform = waveform.flatten()
        
        return waveform, sample_rate
    
    def apply_vad(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply VAD to extract speech segments"""
        if not self.config.use_vad or self.vad_model is None:
            return waveform
        
        try:
            # Convert numpy to torch tensor for VAD
            audio_tensor = torch.from_numpy(waveform).float()
            
            # Check if VAD functions are available
            if self.get_speech_timestamps is None:
                logger.warning("VAD functions not available, using full audio")
                return waveform
            
            timestamps = self.get_speech_timestamps(
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
            if self.collect_chunks is not None:
                speech_audio = self.collect_chunks(timestamps, audio_tensor)
            else:
                # Fallback: manually extract chunks
                speech_audio = []
                for ts in timestamps:
                    # Silero VAD returns timestamps in samples, not milliseconds
                    if isinstance(ts, dict):
                        start_sample = int(ts['start'])
                        end_sample = int(ts['end'])
                    else:
                        # Handle alternative timestamp format
                        start_sample = int(ts.start) if hasattr(ts, 'start') else 0
                        end_sample = int(ts.end) if hasattr(ts, 'end') else len(audio_tensor)
                    
                    chunk = audio_tensor[start_sample:end_sample]
                    speech_audio.append(chunk)
                speech_audio = torch.cat(speech_audio) if speech_audio else audio_tensor
            
            if speech_audio is None or len(speech_audio) == 0:
                logger.warning("VAD returned empty audio")
                return waveform
            
            # Convert back to numpy
            return speech_audio.numpy()
            
        except Exception as e:
            logger.warning(f"VAD failed, using original audio: {e}")
            return waveform
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file"""
        try:
            # Preprocess audio
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
            
            # Apply VAD
            waveform = self.apply_vad(waveform, sample_rate)
            
            # Save processed audio temporarily
            temp_path = os.path.join(self.config.temp_dir, f"temp_{datetime.now().timestamp()}.wav")
            # Use scipy.io.wavfile to write audio (no external dependencies needed)
            # Convert to int16 for WAV format
            waveform_int16 = (waveform * 32767).astype(np.int16)
            wavfile.write(temp_path, sample_rate, waveform_int16)
            
            # Extract embedding using NeMo
            with torch.no_grad():
                embedding = self.speaker_model.get_embedding(temp_path)
                embedding = embedding.cpu().numpy().squeeze()
            
            # Clean up temp file
            os.remove(temp_path)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed for {audio_path}: {e}")
            return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norms = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norms == 0:
            return 0.0
        return float(dot_product / norms)
    
    def enroll_speaker(self, speaker_id: str, audio_paths: List[str]) -> bool:
        """Enroll a speaker with multiple audio samples"""
        embeddings = []
        
        logger.info(f"Enrolling speaker {speaker_id} with {len(audio_paths)} audio files")
        
        for audio_path in audio_paths:
            embedding = self.extract_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                logger.warning(f"Failed to extract embedding from {audio_path}")
        
        if not embeddings:
            logger.error(f"No valid embeddings extracted for speaker {speaker_id}")
            return False
        
        # Store enrollment data
        self.enrollment_db[speaker_id] = {
            "embeddings": embeddings,
            "enrollment_date": datetime.now().isoformat(),
            "num_samples": len(embeddings),
            "audio_paths": audio_paths
        }
        
        self._save_enrollment_db()
        logger.info(f"Speaker {speaker_id} enrolled successfully with {len(embeddings)} embeddings")
        return True
    
    def verify_speaker(self, audio_path: str, claimed_speaker_id: str = None) -> Dict:
        """Verify speaker identity against enrollment database"""
        # Extract embedding from test audio
        test_embedding = self.extract_embedding(audio_path)
        
        if test_embedding is None:
            return {
                "success": False,
                "error": "Failed to extract embedding from test audio",
                "audio_path": audio_path
            }
        
        results = {
            "success": True,
            "audio_path": audio_path,
            "test_embedding": test_embedding.tolist() if self.config.save_embeddings else None,
            "verification_time": datetime.now().isoformat(),
            "speakers": {}
        }
        
        # Compare with all enrolled speakers (or specific speaker if claimed)
        speakers_to_check = [claimed_speaker_id] if claimed_speaker_id else list(self.enrollment_db.keys())
        
        for speaker_id in speakers_to_check:
            if speaker_id not in self.enrollment_db:
                continue
                
            enrolled_embeddings = self.enrollment_db[speaker_id]["embeddings"]
            similarities = []
            
            # Compare with all enrollment samples
            for enrolled_emb in enrolled_embeddings:
                enrolled_emb = np.array(enrolled_emb)
                similarity = self.cosine_similarity(test_embedding, enrolled_emb)
                similarities.append(similarity)
            
            # Calculate statistics
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Verification decision
            is_verified = max_similarity >= self.config.similarity_threshold
            
            results["speakers"][speaker_id] = {
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "std_similarity": std_similarity,
                "num_comparisons": len(similarities),
                "is_verified": is_verified,
                "threshold_used": self.config.similarity_threshold
            }
        
        # Overall verification result
        if claimed_speaker_id:
            results["claimed_speaker"] = claimed_speaker_id
            results["verified"] = results["speakers"].get(claimed_speaker_id, {}).get("is_verified", False)
        else:
            # Find best match
            best_speaker = None
            best_similarity = 0.0
            
            for speaker_id, data in results["speakers"].items():
                if data["max_similarity"] > best_similarity:
                    best_similarity = data["max_similarity"]
                    best_speaker = speaker_id
            
            results["best_match"] = {
                "speaker_id": best_speaker,
                "similarity": best_similarity,
                "verified": best_similarity >= self.config.similarity_threshold
            }
        
        return results
    
    def batch_verify(self, audio_paths: List[str], claimed_speakers: List[str] = None) -> List[Dict]:
        """Batch verification of multiple audio files"""
        results = []
        claimed_speakers = claimed_speakers or [None] * len(audio_paths)
        
        for i, audio_path in enumerate(audio_paths):
            claimed_speaker = claimed_speakers[i] if i < len(claimed_speakers) else None
            result = self.verify_speaker(audio_path, claimed_speaker)
            results.append(result)
        
        return results
    
    def get_enrollment_stats(self) -> Dict:
        """Get statistics about enrolled speakers"""
        total_speakers = len(self.enrollment_db)
        total_samples = sum(data["num_samples"] for data in self.enrollment_db.values())
        
        return {
            "total_enrolled_speakers": total_speakers,
            "total_enrollment_samples": total_samples,
            "speakers": {
                speaker_id: {
                    "num_samples": data["num_samples"],
                    "enrollment_date": data["enrollment_date"]
                }
                for speaker_id, data in self.enrollment_db.items()
            }
        }

# Convenience functions for integration with voice-to-text systems

def create_pipeline(config_path: str = None) -> SpeakerVerificationPipeline:
    """Create speaker verification pipeline with optional config file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VerificationConfig(**config_dict)
    else:
        config = VerificationConfig()
    
    return SpeakerVerificationPipeline(config)

def quick_verify(audio_path: str, speaker_id: str = None, 
                pipeline: SpeakerVerificationPipeline = None) -> Dict:
    """Quick verification function for integration"""
    if pipeline is None:
        pipeline = create_pipeline()
    
    return pipeline.verify_speaker(audio_path, speaker_id)

class RealTimeSpeakerRecognition:
    """Real-time speaker recognition from live microphone input"""
    
    def __init__(self, pipeline: SpeakerVerificationPipeline = None, 
                 chunk_duration=2.0, overlap_duration=0.5):
        """
        Initialize real-time speaker recognition
        chunk_duration: Duration of each audio chunk to process (seconds)
        overlap_duration: Overlap between chunks (seconds)
        """
        self.pipeline = pipeline or create_pipeline()
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = self.pipeline.config.target_sample_rate
        
        # Setup microphone
        self.setup_microphone()
        
        # Speech activity tracking
        self.is_recording = False
        self.current_speech_buffer = []
        self.speech_start_time = None
        self.min_speech_duration = 1.0  # Minimum speech duration to process
        self.max_speech_duration = 10.0  # Maximum speech duration
        
        # Logging
        self.conversation_log = []
        
    def setup_microphone(self):
        """Setup microphone for recording"""
        try:
            import sounddevice as sd
            self.sd = sd
            self.recording_available = True
            print("✓ Real-time microphone available")
        except ImportError:
            try:
                import pyaudio
                self.pyaudio = pyaudio
                self.recording_available = True
                self.use_pyaudio = True
                print("✓ Real-time microphone available (PyAudio)")
            except ImportError:
                self.recording_available = False
                print("❌ No microphone libraries available")
    
    def vad_segment_audio(self, audio_chunk):
        """Use VAD to detect speech segments in audio chunk"""
        if not self.pipeline.config.use_vad or self.pipeline.vad_model is None:
            return [audio_chunk]  # Return whole chunk if no VAD
        
        try:
            audio_tensor = torch.from_numpy(audio_chunk).float()
            
            get_speech_timestamps = self.pipeline.get_speech_timestamps
            
            if get_speech_timestamps is None:
                logger.warning("VAD functions not available, using full audio chunk")
                return [audio_chunk]  # Return the audio chunk directly
            
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.pipeline.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.pipeline.config.vad_threshold,
                min_speech_duration_ms=500,  # 0.5s minimum
                max_speech_duration_s=self.max_speech_duration
            )
            
            if not timestamps:
                return []  # No speech detected
            
            # Extract speech segments
            segments = []
            for timestamp in timestamps:
                start_sample = int(timestamp['start'])
                end_sample = int(timestamp['end'])
                segment = audio_chunk[start_sample:end_sample]
                
                # Only process segments longer than minimum duration
                segment_duration = len(segment) / self.sample_rate
                if segment_duration >= self.min_speech_duration:
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            print(f"VAD segmentation failed: {e}")
            return [audio_chunk]
    
    def identify_speaker(self, audio_segment):
        """Identify speaker from audio segment"""
        try:
            # Save segment temporarily
            temp_path = os.path.join(
                self.pipeline.config.temp_dir, 
                f"temp_segment_{datetime.now().timestamp()}.wav"
            )
            
            # Convert to int16 and save
            from scipy.io import wavfile
            audio_int16 = (audio_segment * 32767).astype(np.int16)
            wavfile.write(temp_path, self.sample_rate, audio_int16)
            
            # Extract embedding
            embedding = self.pipeline.extract_embedding(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            if embedding is None:
                return "Unknown", 0.0
            
            # Compare with enrolled speakers
            best_speaker = "Unknown"
            best_similarity = 0.0
            
            for speaker_id, speaker_data in self.pipeline.enrollment_db.items():
                enrolled_embeddings = speaker_data["embeddings"]
                
                similarities = []
                for enrolled_emb in enrolled_embeddings:
                    enrolled_emb = np.array(enrolled_emb)
                    similarity = self.pipeline.cosine_similarity(embedding, enrolled_emb)
                    similarities.append(similarity)
                
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_speaker = speaker_id if max_similarity >= self.pipeline.config.similarity_threshold else "Unknown"
            
            return best_speaker, best_similarity
            
        except Exception as e:
            print(f"Speaker identification failed: {e}")
            return "Unknown", 0.0
    
    def log_speaker_activity(self, speaker_name, confidence, start_time, duration):
        """Log speaker activity"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speaker": speaker_name,
            "confidence": confidence,
            "start_time": start_time,
            "duration": duration,
            "status": "Enrolled" if speaker_name != "Unknown" else "Stranger"
        }
        
        self.conversation_log.append(log_entry)
        
        # Print real-time log
        status_icon = "👤" if speaker_name != "Unknown" else "❓"
        confidence_str = f"({confidence:.2f})" if speaker_name != "Unknown" else ""
        
        print(f"[{log_entry['timestamp']}] {status_icon} {speaker_name} {confidence_str} - {duration:.1f}s")
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio for speaker recognition"""
        # Segment audio using VAD
        speech_segments = self.vad_segment_audio(audio_chunk)
        
        for segment in speech_segments:
            # Get segment duration
            segment_duration = len(segment) / self.sample_rate
            
            # Identify speaker
            speaker_name, confidence = self.identify_speaker(segment)
            
            # Log the activity
            self.log_speaker_activity(
                speaker_name, 
                confidence, 
                datetime.now().strftime("%H:%M:%S"),
                segment_duration
            )
    
    def start_continuous_recognition(self, duration_minutes=None):
        """Start continuous speaker recognition"""
        if not self.recording_available:
            print("❌ Microphone not available")
            return
        
        print("🎤 Starting real-time speaker recognition...")
        print("💡 Speak into the microphone. The system will identify speakers continuously.")
        print("🔴 Press Ctrl+C to stop")
        
        if duration_minutes:
            print(f"⏱️  Will run for {duration_minutes} minutes")
        
        # Check if we have enrolled speakers
        if not self.pipeline.enrollment_db:
            print("⚠️  No speakers enrolled. All voices will be marked as 'Unknown'")
            print("   Use voice_embedding_tool.py to enroll speakers first")
        else:
            enrolled_speakers = list(self.pipeline.enrollment_db.keys())
            print(f"📋 Enrolled speakers: {', '.join(enrolled_speakers)}")
        
        print("\n" + "="*60)
        print("🎧 SPEAKER RECOGNITION LOG")
        print("="*60)
        
        try:
            if hasattr(self, 'use_pyaudio'):
                self._continuous_recognition_pyaudio(duration_minutes)
            else:
                self._continuous_recognition_sounddevice(duration_minutes)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Recognition stopped by user")
        except Exception as e:
            print(f"\n\n❌ Error during recognition: {e}")
        finally:
            self.save_conversation_log()
    
    def _continuous_recognition_sounddevice(self, duration_minutes):
        """Continuous recognition using sounddevice"""
        import time
        
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Recording buffer
        audio_buffer = np.array([], dtype=np.float32)
        
        start_time = time.time()
        
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer
            
            # Add new audio to buffer
            audio_buffer = np.concatenate([audio_buffer, indata[:, 0]])
            
            # Process when we have enough audio
            if len(audio_buffer) >= chunk_samples:
                # Extract chunk for processing
                chunk_to_process = audio_buffer[:chunk_samples]
                
                # Keep overlap for next iteration
                overlap_samples = int(self.overlap_duration * self.sample_rate)
                audio_buffer = audio_buffer[chunk_samples - overlap_samples:]
                
                # Process in background (you might want to use threading for real-time)
                self.process_audio_chunk(chunk_to_process)
        
        # Start recording
        with self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(0.1 * self.sample_rate)  # 100ms blocks
        ):
            if duration_minutes:
                time.sleep(duration_minutes * 60)
            else:
                while True:
                    time.sleep(0.1)
    
    def _continuous_recognition_pyaudio(self, duration_minutes):
        """Continuous recognition using pyaudio"""
        import time
        
        CHUNK = 1024
        FORMAT = self.pyaudio.paInt16
        
        p = self.pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        audio_buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration limit
                if duration_minutes and (time.time() - start_time) > duration_minutes * 60:
                    break
                
                # Read audio
                data = stream.read(CHUNK)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # Process when buffer is full
                if len(audio_buffer) >= chunk_samples:
                    chunk_to_process = audio_buffer[:chunk_samples]
                    
                    # Keep overlap
                    overlap_samples = int(self.overlap_duration * self.sample_rate)
                    audio_buffer = audio_buffer[chunk_samples - overlap_samples:]
                    
                    # Process chunk
                    self.process_audio_chunk(chunk_to_process)
                
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def save_conversation_log(self):
        """Save conversation log to file"""
        if not self.conversation_log:
            return
        
        log_file = os.path.join(
            self.pipeline.config.temp_dir,
            f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        
        print(f"\n💾 Conversation log saved: {log_file}")
        
        # Print summary
        print("\n📊 CONVERSATION SUMMARY:")
        speakers = {}
        total_duration = 0
        
        for entry in self.conversation_log:
            speaker = entry['speaker']
            duration = entry['duration']
            
            if speaker not in speakers:
                speakers[speaker] = {'count': 0, 'total_duration': 0}
            
            speakers[speaker]['count'] += 1
            speakers[speaker]['total_duration'] += duration
            total_duration += duration
        
        for speaker, stats in speakers.items():
            percentage = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            icon = "👤" if speaker != "Unknown" else "❓"
            print(f"  {icon} {speaker}: {stats['count']} segments, {stats['total_duration']:.1f}s ({percentage:.1f}%)")
    
    def get_enrolled_speakers(self):
        """Get list of enrolled speakers"""
        return list(self.pipeline.enrollment_db.keys())

# Convenience function for real-time recognition
def start_real_time_recognition(duration_minutes=None, chunk_duration=2.0):
    """Start real-time speaker recognition with enrolled speakers"""
    pipeline = create_pipeline()
    recognizer = RealTimeSpeakerRecognition(pipeline, chunk_duration=chunk_duration)
    recognizer.start_continuous_recognition(duration_minutes)
    return recognizer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "realtime":
            # Real-time speaker recognition mode
            duration = None
            if len(sys.argv) > 2:
                try:
                    duration = float(sys.argv[2])
                    print(f"Running for {duration} minutes")
                except ValueError:
                    print("Invalid duration, running indefinitely")
            
            start_real_time_recognition(duration_minutes=duration)
            
        elif sys.argv[1] == "test":
            # Test mode
            config = VerificationConfig(
                similarity_threshold=0.65,
                use_vad=True,
                vad_threshold=0.5
            )
            
            pipeline = SpeakerVerificationPipeline(config)
            print("Pipeline initialized successfully")
            print(f"Enrolled speakers: {list(pipeline.enrollment_db.keys())}")
            
        else:
            print("Available modes:")
            print("  python3 speaker_verification_pipeline.py realtime [duration_minutes]")
            print("  python3 speaker_verification_pipeline.py test")
    else:
        # Default example usage
        config = VerificationConfig(
            similarity_threshold=0.65,
            use_vad=True,
            vad_threshold=0.5
        )
        
        pipeline = SpeakerVerificationPipeline(config)
        
        print("Speaker Verification Pipeline")
        print("Usage examples:")
        print("1. Real-time recognition:")
        print("   python3 speaker_verification_pipeline.py realtime")
        print("   python3 speaker_verification_pipeline.py realtime 5  # Run for 5 minutes")
        print("")
        print("2. Enroll speakers first using:")
        print("   python3 voice_embedding_tool.py interactive")
        print("   > enroll")
        print("")
        print("3. Then run real-time recognition:")
        print("   python3 speaker_verification_pipeline.py realtime")
        
        # Example: Enroll a speaker
        # pipeline.enroll_speaker("user001", ["enrollment1.wav", "enrollment2.wav"])
        
        # Example: Verify a speaker
        # result = pipeline.verify_speaker("test_audio.wav", "user001")
        # print(json.dumps(result, indent=2))