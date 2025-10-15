#!/usr/bin/env python3
"""
Jetson-optimized configuration for Speaker Verification Pipeline
Optimized for NVIDIA Jetson devices (Xavier NX, Orin NX, etc.)
"""

from dataclasses import dataclass
from speaker_verification_pipeline import VerificationConfig

@dataclass
class JetsonConfig(VerificationConfig):
    """Optimized configuration for Jetson deployment"""
    
    # Model paths (adjust for your Jetson setup)
    titanet_model_path: str = "/home/jetson/models/titanet-l.nemo"
    
    # Performance optimizations for Jetson
    device: str = "cuda"  # Use GPU acceleration
    batch_size: int = 1   # Reduce batch size for memory efficiency
    
    # Audio processing optimizations
    target_sample_rate: int = 16000
    max_audio_duration: float = 30.0  # Reduce max duration to save memory
    min_audio_duration: float = 1.0
    
    # VAD optimizations
    use_vad: bool = True
    vad_threshold: float = 0.6  # Slightly higher for edge device stability
    vad_min_speech_duration: int = 250
    vad_max_speech_duration: int = 20000  # Reduce for memory
    
    # Verification settings optimized for Jetson
    similarity_threshold: float = 0.65
    
    # Memory management
    temp_dir: str = "/tmp/speaker_verification"
    enrollment_db_path: str = "/home/jetson/data/speaker_enrollments.json"
    save_embeddings: bool = False  # Reduce memory usage
    save_vad_segments: bool = False
    
    # Jetson-specific settings
    num_threads: int = 4  # Limit CPU threads
    precision: str = "fp16"  # Use half precision for speed
    enable_tensorrt: bool = True  # Enable TensorRT optimization
    cache_embeddings: bool = True  # Cache frequently used embeddings
    max_cache_size: int = 100  # Maximum cached embeddings

# Jetson hardware profiles
JETSON_PROFILES = {
    "jetson_nano": {
        "device": "cpu",  # GPU memory too limited
        "batch_size": 1,
        "max_audio_duration": 15.0,
        "num_threads": 2,
        "precision": "fp32"
    },
    "jetson_xavier_nx": {
        "device": "cuda",
        "batch_size": 1,
        "max_audio_duration": 30.0,
        "num_threads": 4,
        "precision": "fp16"
    },
    "jetson_orin_nx": {
        "device": "cuda", 
        "batch_size": 2,
        "max_audio_duration": 45.0,
        "num_threads": 6,
        "precision": "fp16"
    },
    "jetson_agx_orin": {
        "device": "cuda",
        "batch_size": 4,
        "max_audio_duration": 60.0,
        "num_threads": 8,
        "precision": "fp16"
    }
}

def create_jetson_config(jetson_model: str = "jetson_xavier_nx", 
                        model_path: str = None) -> JetsonConfig:
    """Create Jetson configuration based on hardware model"""
    
    if jetson_model not in JETSON_PROFILES:
        raise ValueError(f"Unknown Jetson model: {jetson_model}. "
                        f"Available: {list(JETSON_PROFILES.keys())}")
    
    profile = JETSON_PROFILES[jetson_model]
    
    config = JetsonConfig()
    
    # Apply hardware-specific settings
    config.device = profile["device"]
    config.batch_size = profile["batch_size"]  
    config.max_audio_duration = profile["max_audio_duration"]
    config.num_threads = profile["num_threads"]
    config.precision = profile["precision"]
    
    # Set model path if provided
    if model_path:
        config.titanet_model_path = model_path
    
    return config

def get_jetson_model() -> str:
    """Auto-detect Jetson model from hardware info"""
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            model_info = f.read().strip()
        
        if 'Jetson AGX Orin' in model_info:
            return "jetson_agx_orin"
        elif 'Jetson Orin NX' in model_info:
            return "jetson_orin_nx"  
        elif 'Jetson Xavier NX' in model_info:
            return "jetson_xavier_nx"
        elif 'Jetson Nano' in model_info:
            return "jetson_nano"
        else:
            print(f"Unknown Jetson model: {model_info}")
            return "jetson_xavier_nx"  # Default fallback
            
    except Exception as e:
        print(f"Could not detect Jetson model: {e}")
        return "jetson_xavier_nx"  # Default fallback

if __name__ == "__main__":
    # Auto-detect Jetson model and create config
    jetson_model = get_jetson_model()
    print(f"Detected Jetson model: {jetson_model}")
    
    config = create_jetson_config(jetson_model)
    print(f"Created config for {jetson_model}:")
    print(f"  - Device: {config.device}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max audio duration: {config.max_audio_duration}s")
    print(f"  - Precision: {config.precision}")