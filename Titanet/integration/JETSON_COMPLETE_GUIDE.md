# 🚀 Hướng Dẫn Triển Khai Speaker Verification trên Jetson - Tổng Hợp

## 📋 Tổng Quan

Hệ thống Speaker Verification với TitaNet-L được tối ưu hóa cho các thiết bị NVIDIA Jetson, bao gồm:
- **jetson_setup.sh**: Script tự động cài đặt toàn bộ hệ thống
- **jetson_config.py**: Cấu hình tối ưu cho từng loại Jetson
- **jetson_speaker_pipeline.py**: Pipeline đã được tối ưu hóa
- **jetson_monitor.py**: Giám sát hiệu suất và tối ưu hóa

## 🎯 Hướng Dẫn Sử Dụng Nhanh

### Bước 1: Cài Đặt Tự Động
```bash
# Tải về và chạy script cài đặt
chmod +x jetson_setup.sh
./jetson_setup.sh

# Hoặc với tùy chọn bỏ qua cập nhật hệ thống
./jetson_setup.sh --skip-update
```

### Bước 2: Kích Hoạt Môi Trường
```bash
# Kích hoạt Python virtual environment
source venv/bin/activate

# Kiểm tra cài đặt
python jetson_config.py
```

### Bước 3: Sao Chép Model
```bash
# Tạo thư mục models
mkdir -p ~/models

# Sao chép TitaNet-L model
cp titanet-l.nemo ~/models/
```

### Bước 4: Chạy Pipeline
```python
# Chạy pipeline tối ưu hóa
python jetson_speaker_pipeline.py

# Hoặc sử dụng trong code:
from jetson_speaker_pipeline import create_jetson_pipeline

# Tự động phát hiện loại Jetson
pipeline = create_jetson_pipeline()

# Hoặc chỉ định cụ thể
pipeline = create_jetson_pipeline(jetson_model="jetson_orin_nx")
```

## 🔧 Cấu Hình Theo Từng Loại Jetson

### Jetson Nano (RAM hạn chế)
```python
from jetson_config import create_jetson_config

config = create_jetson_config("jetson_nano")
# - Sử dụng CPU (device="cpu") 
# - Batch size = 1
# - Max audio = 15 giây
# - FP32 precision
```

### Jetson Xavier NX (Cân bằng)
```python
config = create_jetson_config("jetson_xavier_nx")
# - Sử dụng CUDA (device="cuda")
# - Batch size = 1 
# - Max audio = 30 giây
# - FP16 precision
```

### Jetson Orin NX (Hiệu suất cao)
```python
config = create_jetson_config("jetson_orin_nx")
# - Sử dụng CUDA (device="cuda")
# - Batch size = 2
# - Max audio = 45 giây  
# - FP16 precision + TensorRT
```

### Jetson AGX Orin (Tối đa)
```python
config = create_jetson_config("jetson_agx_orin")
# - Sử dụng CUDA (device="cuda")
# - Batch size = 4
# - Max audio = 60 giây
# - FP16 precision + TensorRT
```

## 📊 Giám Sát và Tối Ưu Hóa

### Khởi Chạy Monitor
```python
from jetson_monitor import JetsonMonitor

# Tạo monitor
monitor = JetsonMonitor()

# Hiển thị thông tin hệ thống
info = monitor.get_jetson_info()
print(f"Model: {info['model']}")
print(f"JetPack: {info['jetpack_version']}")

# Bắt đầu giám sát
monitor.start_monitoring(interval=2.0)

# Dừng giám sát
monitor.stop_monitoring()

# Tạo báo cáo hiệu suất
report = monitor.generate_performance_report()
```

### Monitor trong Pipeline
```python
from jetson_speaker_pipeline import create_jetson_pipeline

pipeline = create_jetson_pipeline()

# Kiểm tra stats
stats = pipeline.get_jetson_stats()
print(f"CPU: {stats['cpu_percent']:.1f}%")
print(f"Memory: {stats['memory_percent']:.1f}%")  
print(f"Temperature: {stats['temperature_c']:.1f}°C")

# Xóa cache khi cần thiết
pipeline.clear_cache()
```

## ⚡ Tối Ưu Hóa Hiệu Suất

### 1. Cài Đặt Performance Mode
```bash
# Chế độ hiệu suất tối đa
sudo nvpmodel -m 0
sudo jetson_clocks

# Kiểm tra chế độ hiện tại
sudo nvpmodel -q
```

### 2. Tối Ưu Memory
```bash
# Tăng swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Làm swap vĩnh viễn
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Tối Ưu CUDA
```python
import torch

# Tối ưu CUDA trong code
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Mixed precision training
from torch.cuda.amp import autocast
with autocast():
    embedding = model.get_embedding(audio_path)
```

## 🔄 Sử Dụng Thực Tế

### Ví Dụ Enrollment (Đăng Ký)
```python
from jetson_speaker_pipeline import create_jetson_pipeline

# Tạo pipeline
pipeline = create_jetson_pipeline()

# Đăng ký speaker
audio_files = [
    "/path/to/speaker1_sample1.wav",
    "/path/to/speaker1_sample2.wav", 
    "/path/to/speaker1_sample3.wav"
]

success = pipeline.enroll_speaker("speaker_001", audio_files)
if success:
    print("✓ Speaker enrolled successfully")
```

### Ví Dụ Verification (Xác Thực)
```python
# Xác thực speaker
result = pipeline.verify_speaker(
    "/path/to/test_audio.wav", 
    claimed_speaker_id="speaker_001"
)

if result["success"]:
    verified = result["verified"]
    similarity = result["speakers"]["speaker_001"]["max_similarity"]
    print(f"Verified: {verified}, Similarity: {similarity:.3f}")
```

### Ví Dụ Batch Processing
```python
# Xử lý nhiều file cùng lúc
audio_files = [
    "/path/to/test1.wav",
    "/path/to/test2.wav", 
    "/path/to/test3.wav"
]

results = pipeline.batch_verify(audio_files)

for i, result in enumerate(results):
    if result["success"]:
        best_match = result.get("best_match", {})
        speaker = best_match.get("speaker_id", "Unknown")
        similarity = best_match.get("similarity", 0)
        print(f"File {i+1}: {speaker} ({similarity:.3f})")
```

## 🐳 Triển Khai Docker (Tùy Chọn)

### Build Docker Image
```bash
# Tạo Dockerfile cho Jetson
cat > Dockerfile.jetson << EOF
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install dependencies
RUN apt-get update && apt-get install -y \\
    libsndfile1-dev sox ffmpeg portaudio19-dev

# Install Python packages  
RUN pip3 install librosa scipy numpy soundfile nemo-toolkit[asr]

# Copy application
WORKDIR /app
COPY . .

ENV PYTHONPATH="/app"
EXPOSE 8000

CMD ["python3", "jetson_speaker_pipeline.py"]
EOF

# Build image
sudo docker build -f Dockerfile.jetson -t speaker-verification-jetson .
```

### Chạy Container
```bash
# Chạy với GPU support
sudo docker run --runtime nvidia --gpus all \\
    -v ~/models:/app/models \\
    -v /tmp:/tmp \\
    -p 8000:8000 \\
    speaker-verification-jetson
```

## 🚨 Xử Lý Sự Cố

### Lỗi Thường Gặp

#### 1. Out of Memory
```bash
# Giải pháp: Tăng swap space
sudo swapoff /swapfile
sudo fallocate -l 16G /swapfile  
sudo mkswap /swapfile
sudo swapon /swapfile

# Hoặc giảm batch size trong config
config.batch_size = 1
config.max_audio_duration = 15.0
```

#### 2. CUDA không khả dụng
```bash
# Kiểm tra CUDA
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Cài lại PyTorch cho Jetson
pip uninstall torch torchvision torchaudio
# Chạy lại jetson_setup.sh
```

#### 3. Model loading quá chậm
```python
# Tăng timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Model loading timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 phút timeout
```

#### 4. Nhiệt độ cao
```python
# Giám sát nhiệt độ
from jetson_monitor import JetsonMonitor

monitor = JetsonMonitor()
stats = monitor.get_current_metrics()

if stats.temperature_c > 80:
    print("⚠️ High temperature! Consider cooling")
    # Giảm tần số xử lý hoặc batch size
```

## 📈 Benchmark Hiệu Suất

### Jetson Orin NX (Dự kiến)
- **Model Loading**: 15-30 giây
- **Embedding Extraction**: 0.5-2.0 giây/10s audio
- **Speaker Verification**: 0.1-0.3 giây
- **RAM Usage**: 2-4GB
- **Power**: 15-25W

### Jetson Xavier NX (Dự kiến)
- **Model Loading**: 20-40 giây
- **Embedding Extraction**: 1.0-3.0 giây/10s audio
- **Speaker Verification**: 0.2-0.5 giây
- **RAM Usage**: 3-5GB
- **Power**: 20-30W

## 🔧 Tùy Chỉnh Nâng Cao

### Tạo Config Tùy Chỉnh
```python
from jetson_config import JetsonConfig

# Tạo config tùy chỉnh
class MyJetsonConfig(JetsonConfig):
    # Tùy chỉnh cho use case cụ thể
    similarity_threshold: float = 0.7  # Ngưỡng nghiêm ngặt hơn
    use_vad: bool = True
    vad_threshold: float = 0.6
    max_audio_duration: float = 20.0  # Giới hạn audio ngắn hơn

config = MyJetsonConfig()
```

### Cache Optimization
```python
# Tối ưu cache cho frequent speakers
pipeline.jetson_config.cache_embeddings = True
pipeline.jetson_config.max_cache_size = 200

# Xóa cache khi cần
pipeline.clear_cache()
```

## 🎛️ API Service (Tùy Chọn)

### Tạo REST API
```python
# jetson_speaker_api.py
from flask import Flask, request, jsonify
from jetson_speaker_pipeline import create_jetson_pipeline

app = Flask(__name__)
pipeline = create_jetson_pipeline()

@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.json
    speaker_id = data['speaker_id']
    audio_paths = data['audio_paths']
    
    success = pipeline.enroll_speaker(speaker_id, audio_paths)
    return jsonify({'success': success})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    audio_path = data['audio_path']
    speaker_id = data.get('speaker_id')
    
    result = pipeline.verify_speaker(audio_path, speaker_id)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Auto-start Service
```bash
# Enable systemd service (đã được tạo bởi setup script)
sudo systemctl enable speaker-verification
sudo systemctl start speaker-verification

# Kiểm tra status
sudo systemctl status speaker-verification
```

## 📝 Checklist Triển Khai

- [ ] ✅ Chạy `jetson_setup.sh` thành công
- [ ] ✅ Kiểm tra PyTorch + CUDA hoạt động
- [ ] ✅ Sao chép TitaNet-L model vào `/home/user/models/`
- [ ] ✅ Test pipeline với `python jetson_speaker_pipeline.py`
- [ ] ✅ Cấu hình performance mode (`nvpmodel -m 0`)
- [ ] ✅ Thiết lập swap space đủ lớn (8GB+)
- [ ] ✅ Test monitor với `python jetson_monitor.py`
- [ ] ✅ Kiểm tra nhiệt độ và hiệu suất
- [ ] ✅ Cấu hình auto-start nếu cần
- [ ] ✅ Test API endpoints nếu sử dụng

## 🆘 Hỗ Trợ và Debug

### Chạy Diagnostic
```python
# Kiểm tra tổng thể hệ thống
from jetson_monitor import JetsonMonitor

monitor = JetsonMonitor()

# System info
print("=== System Information ===")
info = monitor.get_jetson_info()
for key, value in info.items():
    print(f"{key}: {value}")

# Performance check
print("\\n=== Performance Check ===")
metrics = monitor.get_current_metrics("diagnostic")
print(f"CPU: {metrics.cpu_percent:.1f}%")
print(f"Memory: {metrics.memory_percent:.1f}%")
print(f"Temperature: {metrics.temperature_c}°C")

# Recommendations
print("\\n=== Recommendations ===")
recommendations = monitor.get_optimization_recommendations()
for rec in recommendations:
    print(f"• {rec}")
```

### Log Files
```bash
# Xem logs của systemd service
sudo journalctl -u speaker-verification -f

# Xem performance logs
tail -f /tmp/jetson_performance.json
```

## 🎉 Kết Luận

Với hướng dẫn này, bạn có thể:

1. **Cài đặt tự động**: Sử dụng `jetson_setup.sh` để cài đặt toàn bộ hệ thống
2. **Tối ưu hiệu suất**: Sử dụng các config được tối ưu cho từng loại Jetson
3. **Giám sát hệ thống**: Theo dõi nhiệt độ, RAM, CPU usage
4. **Triển khai production**: API service và auto-start capability
5. **Debug và troubleshoot**: Tools để chẩn đoán và giải quyết vấn đề

Hệ thống đã sẵn sàng cho production deployment trên các thiết bị Jetson! 🚀