#!/bin/bash
"""
Install ONNX dependencies for TitaNet-L quantization and deployment
Compatible with Ubuntu/Debian and Jetson Nano
"""

echo "🚀 Installing ONNX Dependencies for Voice Verify"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python version: $python_version"

# Detect platform
if [ -f "/etc/nv_tegra_release" ]; then
    print_status "Detected Jetson platform"
    JETSON_PLATFORM=true
else
    print_status "Detected desktop/server platform"
    JETSON_PLATFORM=false
fi

# Update system
print_status "Updating system packages..."
sudo apt update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y python3-pip python3-dev build-essential

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Core dependencies
print_status "Installing core packages..."
pip3 install numpy==1.24.3
pip3 install protobuf==3.20.3
pip3 install packaging sympy

# ONNX installation
print_status "Installing ONNX..."
if [ "$JETSON_PLATFORM" = true ]; then
    pip3 install onnx==1.13.1  # Jetson compatible version
else
    pip3 install onnx==1.15.0  # Latest for desktop
fi

# ONNXRuntime installation
print_status "Installing ONNXRuntime..."
if [ "$JETSON_PLATFORM" = true ]; then
    # Jetson-specific installation
    print_status "Installing ONNXRuntime for Jetson..."
    pip3 install onnxruntime==1.8.1
else
    # Desktop/Server installation
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU detected, installing ONNXRuntime GPU version..."
        pip3 install onnxruntime-gpu==1.16.3
    else
        print_status "Installing CPU version..."
        pip3 install onnxruntime==1.16.3
    fi
    
    # Install quantization tools (desktop only)
    print_status "Installing quantization tools..."
    pip3 install onnxruntime-tools
fi

# Audio processing libraries
print_status "Installing audio processing libraries..."
pip3 install librosa==0.10.1
pip3 install scipy==1.10.1
pip3 install soundfile

# Test installations
print_status "Testing installations..."
python3 -c "
import sys

def test_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f'✓ {package_name or module_name} imported successfully')
        return True
    except ImportError as e:
        print(f'✗ {package_name or module_name} import failed: {e}')
        return False

print('Testing dependencies:')
test_import('numpy')
test_import('onnx')
test_import('onnxruntime')
test_import('librosa')
test_import('scipy')

# Test quantization tools
try:
    from onnxruntime.quantization import quantize_dynamic
    print('✓ ONNX quantization tools available')
except ImportError:
    print('⚠ Quantization tools not available (normal on Jetson)')

# Test ONNX providers
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f'✓ Available providers: {providers}')
except Exception as e:
    print(f'⚠ Provider check failed: {e}')
"

echo ""
print_status "Installation completed!"

if [ "$JETSON_PLATFORM" = true ]; then
    echo ""
    print_status "Jetson Setup Complete!"
    print_warning "Note: Model quantization should be done on desktop/server"
    echo "Next steps:"
    echo "1. Copy quantized models from desktop"
    echo "2. Run: python3 jetson_pipeline_onnx.py"
else
    echo ""
    print_status "Desktop Setup Complete!"
    echo "Next steps:"
    echo "1. Export model: python3 export_to_onnx.py"
    echo "2. Quantize: python3 quantize_onnx.py"
    echo "3. Test: python3 test_onnx_inference.py"
fi