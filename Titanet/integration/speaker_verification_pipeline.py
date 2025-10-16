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
    titanet_model_path: str = "/home/edabk/Titanet/integration/titanet-l.nemo"
    
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
    temp_dir: str = "/home/edabk/Titanet/integration/temp"
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
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self.vad_model = self.vad_model.to(self.device)
                logger.info("Silero VAD v6 loaded successfully")
            else:
                self.vad_model = None
                self.vad_utils = None
                
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
            
            # Get speech timestamps
            get_speech_timestamps = self.vad_utils.get('get_speech_timestamps')
            collect_chunks = self.vad_utils.get('collect_chunks')
            
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

if __name__ == "__main__":
    # Example usage
    config = VerificationConfig(
        similarity_threshold=0.65,
        use_vad=True,
        vad_threshold=0.5
    )
    
    pipeline = SpeakerVerificationPipeline(config)
    
    # Example: Enroll a speaker
    # pipeline.enroll_speaker("user001", ["enrollment1.wav", "enrollment2.wav"])
    
    # Example: Verify a speaker
    # result = pipeline.verify_speaker("test_audio.wav", "user001")
    # print(json.dumps(result, indent=2))