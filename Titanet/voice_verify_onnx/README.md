# 🚀 Voice Verify ONNX

ONNX conversion and deployment pipeline for TitaNet-L Speaker Verification.

## 📁 Structure

```
voice_verify_onnx/
├── 📄 README.md                        # This file
├── 🧠 titanet-l.nemo                   # Original NeMo model (copied from integration)
├── 📋 speaker_verification_pipeline.py # Reference pipeline (for compatibility)
│
├── ⚡ ONNX Conversion Tools
│   ├── export_to_onnx.py               # NeMo → ONNX converter
│   ├── quantize_onnx.py                # ONNX quantization
│   ├── test_onnx_inference.py          # Performance testing
│   └── install_onnx_deps.sh            # Dependency installer
│
├── 🤖 Jetson Deployment
│   ├── jetson_pipeline_onnx.py         # Lightweight Jetson pipeline
│   └── voice_embedding_onnx.py         # ONNX-based embedding tool
│
├── 📁 Output Directories
│   ├── onnx_models/                    # Exported ONNX models
│   ├── temp/                           # Temporary files
│   └── tests/                          # Test files
│
└── 📚 Documentation
    ├── DEPLOYMENT_GUIDE.md             # Complete deployment guide
    └── PERFORMANCE_BENCHMARKS.md       # Performance comparison
```

## 🎯 Goals

1. **Convert** NeMo TitaNet-L to optimized ONNX format
2. **Quantize** model from 400MB to ~100MB  
3. **Deploy** on Jetson Nano with <1GB memory usage
4. **Maintain** speaker verification accuracy >95%

## 🚀 Quick Start

### 1. Install Dependencies
```bash
chmod +x install_onnx_deps.sh
./install_onnx_deps.sh
```

### 2. Export to ONNX
```bash
python3 export_to_onnx.py
```

### 3. Quantize Model
```bash
python3 quantize_onnx.py
```

### 4. Test Performance
```bash
python3 test_onnx_inference.py
```

### 5. Deploy on Jetson
```bash
python3 jetson_pipeline_onnx.py
```

## 📊 Expected Results

| Model | Size | Memory | Inference | Platform |
|-------|------|--------|-----------|----------|
| NeMo | 400MB | 2-4GB | 200ms | Desktop only |
| ONNX | 400MB | 1GB | 100ms | Desktop + Jetson |
| ONNX Quantized | 100MB | 500MB | 50ms | All platforms |

## 🔗 Integration

This ONNX pipeline is designed to be compatible with the existing `integration/` workspace:
- Uses same enrollment database format
- Produces compatible embeddings  
- Maintains API compatibility
- Can replace NeMo pipeline directly

## 📞 Support

For ONNX-specific issues, check files in this directory.
For general speaker verification, see `../integration/` directory.


# 1. Cài dependencies
cd /home/edabk408/NgocDat/Titanet/voice_verify_onnx
./install_onnx_deps.sh

# 2. Convert NeMo → ONNX
python3 export_to_onnx.py

# 3. Quantize để giảm 75% kích thước
python3 quantize_onnx.py

# 4. Test accuracy & performance  
python3 test_onnx_inference.py

# 5. Deploy trên Jetson Nano
python3 jetson_pipeline_onnx.py