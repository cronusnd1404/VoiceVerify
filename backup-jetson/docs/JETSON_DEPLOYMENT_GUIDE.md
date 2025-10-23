# Jetson Deployment Guide for Speaker Verification Pipeline

This guide covers the complete deployment of the TitaNet-L Speaker Verification Pipeline on NVIDIA Jetson devices.

## 📋 Prerequisites

### Jetson Hardware Requirements
- **Jetson Xavier NX / AGX Xavier** (Recommended)
- **Jetson Orin Nano / Orin NX** (Best performance)
- **Jetson Nano** (Limited, CPU-only mode)
- At least **8GB RAM** (16GB recommended)
- **32GB+ storage** (for models and dependencies)

### Software Requirements
- **JetPack 4.6+ or 5.0+** (includes CUDA, cuDNN)
- **Python 3.8+**
- **Docker** (optional but recommended)

## 🚀 Installation Steps

### 1. Prepare Jetson Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    python3-pip \
    python3-dev \
    libsndfile1-dev \
    libsox-fmt-all \
    sox \
    ffmpeg \
    portaudio19-dev

# Install pip packages for Jetson
sudo pip3 install --upgrade pip setuptools wheel
```

### 2. Install PyTorch for Jetson

```bash
# For JetPack 5.x (Jetson Orin series)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For JetPack 4.6 (Xavier series)
# Download prebuilt wheel from NVIDIA
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Install Audio Processing Libraries

```bash
# Install librosa and dependencies
pip3 install librosa==0.10.1
pip3 install scipy numpy soundfile

# Test librosa installation
python3 -c "import librosa; print(f'Librosa version: {librosa.__version__}')"
```

### 4. Install NeMo Toolkit

```bash
# Install NeMo with specific constraints for Jetson
pip3 install nemo-toolkit[asr]==1.20.0

# Alternative: Install from source for better compatibility
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip3 install -e .
```

### 5. Optimize for Jetson Performance

#### Enable Max Performance Mode
```bash
# Set Jetson to max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current power mode
sudo nvpmodel -q
```

#### Configure Memory and Swap
```bash
# Increase swap space (especially for Jetson Nano)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Configure shared memory for PyTorch
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## ⚙️ Model Optimization for Jetson

### 1. Create Jetson-Optimized Configuration

```python
# jetson_config.py
from dataclasses import dataclass
from speaker_verification_pipeline import VerificationConfig

@dataclass
class JetsonConfig(VerificationConfig):
    """Optimized configuration for Jetson deployment"""
    
    # Model paths (adjust for your Jetson setup)
    titanet_model_path: str = "/home/jetson/models/titanet-l.nemo"
    
    # Performance optimizations
    device: str = "cuda"  # Use GPU acceleration
    batch_size: int = 1   # Reduce batch size for memory efficiency
    
    # Audio processing optimizations
    target_sample_rate: int = 16000
    max_audio_duration: float = 30.0  # Reduce max duration
    
    # VAD optimizations
    use_vad: bool = True
    vad_threshold: float = 0.6  # Slightly higher for edge device
    
    # Memory management
    temp_dir: str = "/tmp/speaker_verification"
    save_embeddings: bool = False  # Reduce memory usage
    
    # Jetson-specific settings
    num_threads: int = 4  # Limit CPU threads
    precision: str = "fp16"  # Use half precision for speed
```

### 2. Create Jetson-Optimized Pipeline

```python
# jetson_speaker_pipeline.py
import torch
import gc
from typing import Optional
from speaker_verification_pipeline import SpeakerVerificationPipeline

class JetsonSpeakerPipeline(SpeakerVerificationPipeline):
    """Jetson-optimized speaker verification pipeline"""
    
    def __init__(self, config):
        # Set CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        super().__init__(config)
        
        # Enable half precision if supported
        if hasattr(self.speaker_model, 'half'):
            self.speaker_model = self.speaker_model.half()
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Memory-optimized embedding extraction"""
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process with original method
            embedding = super().extract_embedding(audio_path)
            
            # Clean up after processing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Jetson embedding extraction failed: {e}")
            # Force cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def _load_models(self):
        """Jetson-optimized model loading"""
        try:
            # Load with reduced memory usage
            logger.info("Loading models for Jetson...")
            
            # Load TitaNet-L with optimizations
            self.speaker_model = EncDecSpeakerLabelModel.restore_from(
                self.config.titanet_model_path,
                map_location=self.device
            )
            self.speaker_model = self.speaker_model.eval().to(self.device)
            
            # Optimize for inference
            torch.jit.optimize_for_inference(self.speaker_model)
            
            # Load VAD if needed
            if self.config.use_vad:
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self.vad_model = self.vad_model.to(self.device)
            else:
                self.vad_model = None
                self.vad_utils = None
            
            logger.info("Jetson models loaded successfully")
            
        except Exception as e:
            logger.error(f"Jetson model loading failed: {e}")
            raise
```

### 3. Performance Monitoring Script

```python
# jetson_monitor.py
import psutil
import time
import subprocess
from typing import Dict

class JetsonMonitor:
    """Monitor Jetson performance during inference"""
    
    def get_jetson_stats(self) -> Dict:
        """Get Jetson-specific performance stats"""
        stats = {}
        
        try:
            # GPU stats (tegrastats)
            tegra_output = subprocess.check_output(['tegrastats', '--interval', '1000'], 
                                                 timeout=2).decode()
            stats['tegra_raw'] = tegra_output
            
        except:
            stats['tegra_raw'] = "Not available"
        
        # CPU and Memory
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
        stats['memory_percent'] = psutil.virtual_memory().percent
        stats['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
        
        # Temperature (if available)
        try:
            with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                temp_raw = int(f.read().strip())
                stats['cpu_temp_c'] = temp_raw / 1000.0
        except:
            stats['cpu_temp_c'] = None
        
        return stats
    
    def log_performance(self, operation: str, duration: float, stats: Dict):
        """Log performance metrics"""
        print(f"\n=== {operation} Performance ===")
        print(f"Duration: {duration:.2f}s")
        print(f"CPU Usage: {stats['cpu_percent']:.1f}%")
        print(f"Memory Usage: {stats['memory_percent']:.1f}%")
        print(f"Available Memory: {stats['available_memory_gb']:.2f}GB")
        if stats['cpu_temp_c']:
            print(f"CPU Temperature: {stats['cpu_temp_c']:.1f}°C")
```

## 🐳 Docker Deployment (Recommended)

### 1. Create Jetson Dockerfile

```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    sox \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    librosa==0.10.1 \
    scipy \
    numpy \
    soundfile \
    nemo-toolkit[asr]==1.20.0

# Copy application
WORKDIR /app
COPY . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV CUDA_VISIBLE_DEVICES=0

# Expose API port
EXPOSE 8000

# Run application
CMD ["python3", "jetson_speaker_api.py"]
```

### 2. Build and Run Docker Container

```bash
# Build Jetson container
sudo docker build -f Dockerfile.jetson -t speaker-verification-jetson .

# Run with GPU support
sudo docker run --runtime nvidia --gpus all \
    -v /home/jetson/models:/app/models \
    -v /tmp:/tmp \
    -p 8000:8000 \
    speaker-verification-jetson
```

## 🔧 Optimization Tips

### 1. Memory Optimization
```bash
# Reduce system memory usage
sudo systemctl disable graphical.target
sudo systemctl set-default multi-user.target

# Optimize Python memory
export PYTHONOPTIMIZE=1
export MALLOC_TRIM_THRESHOLD_=100000
```

### 2. CUDA Optimization
```python
# Add to your pipeline initialization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()

# Use autocast for mixed precision
from torch.cuda.amp import autocast

with autocast():
    embedding = self.speaker_model.get_embedding(audio_path)
```

### 3. Audio Processing Optimization
```python
# Optimize librosa for Jetson
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['OMP_NUM_THREADS'] = '4'

# Use faster resampling
import librosa
librosa.load(audio_path, sr=16000, res_type='kaiser_fast')
```

## 📊 Performance Benchmarks

### Expected Performance (Jetson Orin NX)
- **Model Loading**: 15-30 seconds
- **Embedding Extraction**: 0.5-2.0 seconds per 10s audio
- **Speaker Verification**: 0.1-0.3 seconds
- **Memory Usage**: 2-4GB RAM
- **Power Consumption**: 15-25W

### Expected Performance (Jetson Xavier NX)
- **Model Loading**: 20-40 seconds
- **Embedding Extraction**: 1.0-3.0 seconds per 10s audio
- **Speaker Verification**: 0.2-0.5 seconds
- **Memory Usage**: 3-5GB RAM
- **Power Consumption**: 20-30W

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   ```bash
   # Increase swap space
   sudo swapoff /swapfile
   sudo fallocate -l 16G /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   nvcc --version
   python3 -c "import torch; print(torch.cuda.is_available())"
   
   # Reinstall CUDA-compatible PyTorch
   pip3 uninstall torch torchvision torchaudio
   # Install appropriate version for your JetPack
   ```

3. **Model Loading Timeout**
   ```python
   # Add timeout handling
   import signal
   
   def timeout_handler(signum, frame):
       raise TimeoutError("Model loading timeout")
   
   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(300)  # 5 minute timeout
   ```

4. **Audio Processing Issues**
   ```bash
   # Install additional audio codecs
   sudo apt install ubuntu-restricted-extras
   sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
   ```

## 📝 Deployment Checklist

- [ ] Jetson firmware updated to latest JetPack
- [ ] CUDA and cuDNN properly installed
- [ ] PyTorch with CUDA support verified
- [ ] All Python dependencies installed
- [ ] Models downloaded and accessible
- [ ] Performance mode enabled (`nvpmodel -m 0`)
- [ ] Sufficient swap space configured
- [ ] Temperature monitoring setup
- [ ] Docker environment tested (if using)
- [ ] API endpoints functional
- [ ] Performance benchmarks recorded

## 📖 Next Steps

1. **Deploy the optimized pipeline**: Use the Jetson-specific configuration
2. **Setup monitoring**: Implement performance and temperature monitoring
3. **Create API service**: Build REST API for integration
4. **Test thoroughly**: Validate performance under load
5. **Production hardening**: Add error handling and recovery mechanisms

This guide provides a complete roadmap for deploying your speaker verification system on Jetson devices with optimal performance and reliability.