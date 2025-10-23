# Hướng dẫn sử dụng Voice Embedding Tool với Microphone

## Tính năng mới: Ghi âm trực tiếp từ microphone

### Cài đặt thư viện cần thiết

```bash
# Cài đặt sounddevice (khuyến nghị)
pip3 install sounddevice scipy

# Hoặc cài đặt pyaudio (backup option)
pip3 install pyaudio
```

### Các chế độ sử dụng

#### 1. Test Microphone
```bash
python3 voice_embedding_tool.py test_mic
```

#### 2. Enrollment (Đăng ký giọng nói)
```bash
# Chế độ nhanh
python3 voice_embedding_tool.py enroll

# Chế độ interactive
python3 voice_embedding_tool.py interactive
> enroll
```

#### 3. Verification (Xác thực giọng nói)
```bash
# Chế độ nhanh
python3 voice_embedding_tool.py verify /path/to/reference.pkl

# Chế độ interactive
python3 voice_embedding_tool.py interactive
> verify
```

#### 4. Interactive Mode (Chế độ tương tác)
```bash
python3 voice_embedding_tool.py interactive
```

### Commands trong Interactive Mode

#### Ghi âm và xử lý
- `test_mic` - Test microphone
- `record [duration] [save_audio] [save_embedding]` - Ghi âm và extract embedding
- `record_compare <reference> [duration] [threshold]` - Ghi âm và so sánh ngay

#### Enrollment/Verification
- `enroll` - Đăng ký giọng nói mới (guided)
- `verify` - Xác thực giọng nói (guided)

#### File processing
- `extract <audio_file> [save_path]` - Extract embedding từ file
- `compare <test_audio> <reference> [threshold]` - So sánh 2 file
- `batch <reference> <audio_dir> [threshold]` - So sánh nhiều file

### Workflow thực tế

#### A. Đăng ký người dùng mới
```bash
python3 voice_embedding_tool.py interactive
> enroll
Enter speaker name: john_doe
# Hệ thống sẽ hướng dẫn ghi âm 10 giây
# Kết quả: john_doe_reference.wav và john_doe_reference.pkl
```

#### B. Xác thực người dùng
```bash
python3 voice_embedding_tool.py interactive
> verify
Available references:
  1. john_doe
  2. jane_smith
Choose reference: 1
# Hệ thống sẽ hướng dẫn ghi âm 5 giây và so sánh
```

#### C. Ghi âm và xử lý custom
```bash
python3 voice_embedding_tool.py interactive
> record 8 true true
# Ghi âm 8 giây, lưu cả audio và embedding

> record_compare john_doe_reference.pkl 5 0.7
# Ghi âm 5 giây và so sánh với threshold 0.7
```

### Cấu trúc file được tạo

```
/home/edabk/Titanet/integration/
├── data/
│   ├── john_doe_reference.wav
│   ├── john_doe_reference.pkl
│   ├── jane_smith_reference.wav
│   └── jane_smith_reference.pkl
└── temp/
    ├── recorded_voice_20231016_143022.wav
    └── recorded_voice_20231016_143022_embedding.pkl
```

### Thông số có thể tuỳ chỉnh

- **Duration**: Thời gian ghi âm (giây) - mặc định 5s cho verify, 10s cho enroll
- **Threshold**: Ngưỡng so sánh (0.0-1.0) - mặc định 0.65
- **Sample rate**: 16000 Hz (tối ưu cho TitaNet-L)
- **Format**: WAV, mono channel, 16-bit

### Troubleshooting

#### Lỗi microphone không hoạt động
```bash
# Kiểm tra microphone
python3 voice_embedding_tool.py test_mic

# Kiểm tra audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

#### Lỗi thư viện audio
```bash
# Cài đặt lại sounddevice
pip3 uninstall sounddevice
pip3 install sounddevice scipy

# Hoặc sử dụng pyaudio
pip3 install pyaudio
```

#### Permission issues
```bash
# Thêm user vào audio group
sudo usermod -a -G audio $USER
# Logout và login lại
```

### Tips sử dụng hiệu quả

1. **Enrollment**: Ghi âm trong môi trường yên tĩnh, nói rõ ràng
2. **Verification**: Giữ khoảng cách và âm lượng tương tự khi enrollment
3. **Threshold**: Giảm threshold nếu bị reject nhiều, tăng nếu có false positive
4. **Audio quality**: Đảm bảo microphone chất lượng tốt và ít nhiễu