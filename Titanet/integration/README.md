# 🎯 TitaNet-L Speaker Verification - Clean Integration

This is the cleaned integration workspace with only essential files for production deployment.

## 📁 Current Structure

```
integration/
├── 🧠 speaker_verification_pipeline.py    # Main pipeline (NeMo + VAD + Real-time)
├── 🎤 voice_embedding_tool.py             # Interactive enrollment/verification
├── ⚙️  jetson_config.py                   # Jetson Nano configuration
├── 🤖 titanet-l.nemo                      # TitaNet-L model (400MB)
├── 📁 temp/                               # Temporary files
├── 📁 tests/                              # Test files
└── 📦 backup-jetson/                      # Legacy files & documentation
```

## ✅ Core Components

### 1. Main Pipeline (`speaker_verification_pipeline.py`)
- **TitaNet-L**: Speaker embedding extraction
- **Silero VAD**: Voice activity detection  
- **Real-time Recognition**: Live microphone processing
- **Enrollment/Verification**: Speaker database management
- **Usage**: `python3 speaker_verification_pipeline.py realtime`

### 2. Interactive Tool (`voice_embedding_tool.py`)
- **Microphone Recording**: Direct voice capture
- **Speaker Enrollment**: Interactive enrollment process
- **Voice Comparison**: Real-time verification
- **Usage**: `python3 voice_embedding_tool.py interactive`

### 3. Jetson Config (`jetson_config.py`)
- **Hardware Detection**: Jetson Nano specifications
- **Path Configuration**: Updated for `/home/edabk/Titanet/integration`
- **Resource Management**: Memory and CPU optimization

## 🚀 Quick Start

### Enroll a Speaker
```bash
python3 voice_embedding_tool.py interactive
> enroll
# Follow prompts to record reference voice
```

### Real-time Recognition
```bash
python3 speaker_verification_pipeline.py realtime
# Speaks into microphone, system identifies speakers
```

### Voice Verification
```bash
python3 voice_embedding_tool.py interactive  
> verify
# Choose enrolled speaker and verify against live voice
```

## 🎯 Next: ONNX Optimization

With the clean workspace, we're ready to:

1. **Export to ONNX** - Convert NeMo model for Jetson compatibility
2. **Quantization** - Reduce 400MB model to ~100MB  
3. **Jetson Deployment** - Lightweight pipeline for ARM64

## 📊 Performance (Current NeMo Setup)
- **Model Size**: 400MB (titanet-l.nemo)
- **Memory Usage**: ~2-3GB RAM
- **Load Time**: 30-45 seconds
- **Inference**: ~200ms per embedding

## 📊 Target (ONNX + Quantization)
- **Model Size**: ~100MB (quantized ONNX)
- **Memory Usage**: ~500MB-1GB RAM  
- **Load Time**: 3-5 seconds
- **Inference**: ~50-100ms per embedding

## 🔧 Dependencies

Current requirements:
```
torch>=1.11.0
librosa>=0.9.0
nemo-toolkit>=1.15.0
silero-vad
sounddevice
scipy
```

After ONNX conversion:
```
onnxruntime-gpu  # or onnxruntime for CPU
librosa>=0.9.0
sounddevice
scipy
# No more NeMo dependency!
```