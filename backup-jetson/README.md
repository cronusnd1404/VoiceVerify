# 📦 Backup Jetson Files

This folder contains legacy files and documentation from the Jetson development phase.

## 📁 Structure

### `docs/` - Documentation Files
- `JETSON_COMPLETE_GUIDE.md` - Complete Jetson setup guide
- `JETSON_DEPLOYMENT_GUIDE.md` - Deployment instructions  
- `JETSON_PATH_UPDATE.md` - Path configuration guide
- `PRODUCTION_READY_SUMMARY.md` - Production deployment summary
- `REALTIME_RECOGNITION_GUIDE.md` - Real-time recognition guide
- `VOICE_EMBEDDING_GUIDE.md` - Voice embedding usage guide

### `legacy-scripts/` - Old Implementation Files
- `jetson_pipeline_new.py` - Legacy Jetson pipeline
- `jetson_speaker_pipeline.py` - Old speaker pipeline version
- `jetson_monitor.py` - Legacy system monitoring
- `realtime_speaker_demo.py` - Old real-time demo (now integrated in main pipeline)
- `create_directories.sh` - Directory creation script
- `jetson_setup.sh` - Legacy Jetson setup script

### `__pycache__/` - Python Cache Files
- Compiled Python bytecode files (can be deleted)

## ✅ Current Active Files (in parent directory)

The main integration folder now contains only essential files:

1. **`speaker_verification_pipeline.py`** - Main pipeline with real-time recognition
2. **`voice_embedding_tool.py`** - Interactive enrollment/verification tool
3. **`jetson_config.py`** - Jetson configuration (updated paths)
4. **`titanet-l.nemo`** - TitaNet-L model file (400MB)
5. **`temp/`** - Temporary files directory
6. **`tests/`** - Test files directory

## 🎯 Next Steps

With the cleaned workspace, we can now focus on:

1. **ONNX Export** - Convert NeMo model to ONNX format
2. **Quantization** - Reduce model size for Jetson Nano
3. **Lightweight Deployment** - Create Jetson-optimized pipeline using ONNX

## 🗑️ Cleanup

You can safely delete the `__pycache__/` folder:
```bash
rm -rf __pycache__/
```

The documentation and legacy scripts are preserved here for reference.