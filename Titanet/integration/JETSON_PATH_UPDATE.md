# Hướng dẫn triển khai Jetson với đường dẫn mới

## Đường dẫn mới: `/home/edabk/Titanet/integration`

### 1. Tạo thư mục trên Jetson

```bash
# Chạy script tạo thư mục
chmod +x create_directories.sh
./create_directories.sh
```

### 2. Copy files từ development machine

```bash
# Từ máy phát triển, copy toàn bộ thư mục integration
scp -r /home/edabk408/NgocDat/Titanet/integration/* edabk@jetson_ip:/home/edabk/Titanet/integration/
```

### 3. Cấu trúc thư mục trên Jetson

```
/home/edabk/Titanet/integration/
├── jetson_config.py           # Cấu hình Jetson
├── jetson_monitor.py          # Monitor hệ thống
├── jetson_pipeline_new.py     # Pipeline mới (đơn giản)
├── jetson_simple_pipeline.py  # Pipeline đơn giản nhất
├── speaker_verification_pipeline.py # Pipeline chính
├── voice_embedding_tool.py    # Tool embedding
├── titanet-l.nemo            # Model file
├── jetson_setup.sh           # Script setup
├── create_directories.sh     # Script tạo thư mục
├── temp/                     # Thư mục tạm
├── logs/                     # Log files
├── data/                     # Dữ liệu
└── dataset/                  # Dataset
    ├── test/
    └── train/
```

### 4. Test trên Jetson

```bash
# Chuyển vào thư mục
cd /home/edabk/Titanet/integration

# Test cấu hình
python3 jetson_config.py

# Test monitor
python3 jetson_monitor.py

# Test pipeline đơn giản
python3 jetson_simple_pipeline.py
```

### 5. Đường dẫn đã được cập nhật

- **Model path**: `/home/edabk/Titanet/integration/titanet-l.nemo`
- **Temp directory**: `/home/edabk/Titanet/integration/temp`
- **Log directory**: `/home/edabk/Titanet/integration/logs`
- **Data directory**: `/home/edabk/Titanet/integration/data`
- **Test audio**: `/home/edabk/Titanet/integration/test.wav`

### 6. Chạy setup (nếu cần)

```bash
chmod +x jetson_setup.sh
sudo ./jetson_setup.sh
```

### Lưu ý:
- Tất cả đường dẫn đã được sửa từ `/home/edabk408/NgocDat/Titanet` thành `/home/edabk/Titanet/integration`
- Code đã được đơn giản hóa để tránh lỗi dependencies
- Monitor đã được đơn giản hóa không cần psutil
- Pipeline có 2 phiên bản: `jetson_pipeline_new.py` và `jetson_simple_pipeline.py`