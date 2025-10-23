#!/bin/bash

# Script setup đơn giản cho Jetson
echo "=== Jetson Setup Script ==="

# Kiểm tra quyền sudo
if ! sudo -n true 2>/dev/null; then
    echo "❌ Cần quyền sudo để chạy script này"
    exit 1
fi

echo "✅ Bắt đầu setup Jetson..."

# Check if running on Jetson
check_jetson() {
    log "Checking if running on Jetson device..."
    if [ -f "/sys/firmware/devicetree/base/model" ]; then
        MODEL=$(cat /sys/firmware/devicetree/base/model)
        if [[ "$MODEL" == *"Jetson"* ]]; then
            success "Detected Jetson device: $MODEL"
            export JETSON_MODEL="$MODEL"
        else
            error "Not running on a Jetson device: $MODEL"
            exit 1
        fi
    else
        error "Cannot detect device type"
        exit 1
    fi
}

# Check JetPack version
check_jetpack() {
    log "Checking JetPack version..."
    if [ -f "/etc/nv_tegra_release" ]; then
        JETPACK_VERSION=$(cat /etc/nv_tegra_release)
        success "JetPack version: $JETPACK_VERSION"
    else
        warning "Cannot detect JetPack version"
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
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
        python3-venv \
        libsndfile1-dev \
        libsox-fmt-all \
        sox \
        ffmpeg \
        portaudio19-dev \
        libasound2-dev \
        pkg-config \
        libffi-dev \
        libssl-dev
    success "System dependencies installed"
}

# Setup Python environment
setup_python() {
    log "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        success "Created Python virtual environment"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    success "Python environment ready"
}

# Install PyTorch for Jetson
install_pytorch() {
    log "Installing PyTorch for Jetson..."
    
    source venv/bin/activate
    
    # Check JetPack version and install appropriate PyTorch
    if grep -q "R35" /etc/nv_tegra_release 2>/dev/null; then
        log "Installing PyTorch for JetPack 5.x (L4T R35)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif grep -q "R32" /etc/nv_tegra_release 2>/dev/null; then
        log "Installing PyTorch for JetPack 4.6 (L4T R32)"
        # Download prebuilt wheel for JetPack 4.6
        TORCH_WHEEL="torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl"
        if [ ! -f "$TORCH_WHEEL" ]; then
            wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/$TORCH_WHEEL
        fi
        pip install $TORCH_WHEEL
    else
        warning "Unknown JetPack version, trying generic PyTorch installation"
        pip install torch torchvision torchaudio
    fi
    
    # Verify PyTorch installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    success "PyTorch installed for Jetson"
}

# Install audio processing libraries
install_audio_libs() {
    log "Installing audio processing libraries..."
    
    source venv/bin/activate
    
    pip install \
        librosa==0.10.1 \
        scipy \
        numpy \
        soundfile \
        resampy
    
    success "Audio processing libraries installed"
}

# Install NeMo toolkit
install_nemo() {
    log "Installing NeMo toolkit..."
    
    source venv/bin/activate
    
    # Install dependencies first
    pip install \
        omegaconf \
        hydra-core \
        pytorch-lightning \
        transformers
    
    # Install NeMo
    pip install nemo-toolkit[asr]==1.20.0
    
    success "NeMo toolkit installed"
}

# Setup model directory
setup_models() {
    log "Setting up model directory..."
    
    # Create model directory
    MODEL_DIR="/home/edabk/Titanet/integration"
    mkdir -p "$MODEL_DIR"
    
    # Copy TitaNet model if it exists
    if [ -f "titanet-l.nemo" ]; then
        cp titanet-l.nemo "$MODEL_DIR/"
        success "TitaNet-L model copied to $MODEL_DIR"
    else
        warning "TitaNet-L model not found. Please copy titanet-l.nemo to $MODEL_DIR/"
    fi
    
    # Create data directory
    DATA_DIR="/home/edabk/Titanet/integration/data"
    mkdir -p "$DATA_DIR"
    success "Model and data directories created"
}

# Configure Jetson performance
configure_performance() {
    log "Configuring Jetson performance..."
    
    # Set maximum performance mode
    if command -v nvpmodel &> /dev/null; then
        sudo nvpmodel -m 0
        success "Set nvpmodel to maximum performance mode"
    else
        warning "nvpmodel not available"
    fi
    
    # Enable jetson_clocks
    if command -v jetson_clocks &> /dev/null; then
        sudo jetson_clocks
        success "Enabled jetson_clocks"
    else
        warning "jetson_clocks not available"
    fi
    
    # Increase swap space if needed
    SWAP_SIZE=$(swapon --show=SIZE --noheadings --bytes | head -1)
    if [ -z "$SWAP_SIZE" ] || [ "$SWAP_SIZE" -lt 8589934592 ]; then  # 8GB
        log "Creating 8GB swap file..."
        sudo fallocate -l 8G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Make swap permanent
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        success "Swap space configured"
    else
        success "Sufficient swap space already available"
    fi
}

# Create systemd service for auto-start
create_service() {
    log "Creating systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/speaker-verification.service"
    WORK_DIR="$(pwd)"
    USER="$(whoami)"
    
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Speaker Verification Pipeline
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment=PATH=$WORK_DIR/venv/bin
ExecStart=$WORK_DIR/venv/bin/python jetson_speaker_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    success "Systemd service created"
    log "To enable auto-start: sudo systemctl enable speaker-verification"
}

# Test installation
test_installation() {
    log "Testing installation..."
    
    source venv/bin/activate
    
    # Test PyTorch
    python -c "import torch; print('PyTorch OK')" || error "PyTorch test failed"
    
    # Test librosa
    python -c "import librosa; print('Librosa OK')" || error "Librosa test failed"
    
    # Test NeMo
    python -c "import nemo; print('NeMo OK')" || error "NeMo test failed"
    
    # Test Jetson configuration
    if [ -f "jetson_config.py" ]; then
        python jetson_config.py || error "Jetson config test failed"
    fi
    
    success "All components tested successfully"
}

# Main installation function
main() {
    echo "Starting Jetson setup..."
    
    check_jetson
    check_jetpack
    update_system
    install_system_deps
    setup_python
    install_pytorch
    install_audio_libs
    install_nemo
    setup_models
    configure_performance
    create_service
    test_installation
    
    echo ""
    echo "🎉 Jetson setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Copy your TitaNet-L model to /home/edabk/Titanet/integration/"
    echo "2. Activate the environment: source venv/bin/activate"
    echo "3. Run the pipeline: python jetson_speaker_pipeline.py"
    echo "4. Optional: Enable auto-start: sudo systemctl enable speaker-verification"
    echo ""
    echo "Performance monitoring:"
    echo "- Run: python jetson_monitor.py"
    echo "- Check system status: tegrastats"
    echo ""
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Jetson Speaker Verification Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --skip-update  Skip system package updates"
    echo ""
    echo "This script will:"
    echo "1. Verify Jetson hardware and JetPack version"
    echo "2. Install system dependencies"
    echo "3. Setup Python virtual environment"
    echo "4. Install PyTorch optimized for Jetson"
    echo "5. Install audio processing libraries"
    echo "6. Install NeMo toolkit"
    echo "7. Configure performance optimizations"
    echo "8. Create systemd service for auto-start"
    echo "9. Test all installations"
    exit 0
fi

# Skip updates if requested
if [ "$1" = "--skip-update" ]; then
    update_system() {
        log "Skipping system package updates"
    }
fi

# Run main installation
main