# Hướng dẫn sử dụng Real-Time Speaker Recognition

## Tính năng mới: Nhận dạng người nói thời gian thực

Hệ thống có thể thu âm liên tục từ microphone, tự động phân đoạn bằng VAD (Voice Activity Detection), và nhận dạng người nói trong từng đoạn hội thoại.

### Workflow hoạt động:

```
🎤 Microphone Input
    ↓
📊 VAD Segmentation (phân đoạn giọng nói)
    ↓
🧠 Speaker Embedding Extraction
    ↓
🔍 Compare with Enrolled Speakers
    ↓
📝 Log Speaker Identity + Timestamp
```

## Cài đặt và thiết lập

### 1. Cài đặt thư viện cần thiết:
```bash
# Audio recording
pip3 install sounddevice scipy

# Hoặc backup option
pip3 install pyaudio
```

### 2. Đăng ký giọng nói trước:
```bash
# Chạy voice embedding tool
python3 voice_embedding_tool.py interactive

# Đăng ký từng người
> enroll
Enter speaker name: nguyen_van_a

> enroll  
Enter speaker name: tran_thi_b

> quit
```

### 3. Chạy real-time recognition:
```bash
# Cách 1: Sử dụng demo script (khuyến nghị)
python3 realtime_speaker_demo.py

# Cách 2: Sử dụng trực tiếp
python3 speaker_verification_pipeline.py realtime

# Cách 3: Giới hạn thời gian (5 phút)
python3 speaker_verification_pipeline.py realtime 5
```

## Cách sử dụng Demo Script

### Khởi chạy:
```bash
python3 realtime_speaker_demo.py
```

### Cấu hình tham số:
- **Thời gian ghi**: Nhập số phút hoặc Enter để không giới hạn
- **Độ dài đoạn**: Thời gian mỗi đoạn phân tích (mặc định 2 giây)

### Output mẫu:
```
🎧 SPEAKER RECOGNITION LOG
============================================================
[2023-10-16 14:30:15] 👤 nguyen_van_a (0.87) - 3.2s
[2023-10-16 14:30:19] ❓ Unknown - 2.1s
[2023-10-16 14:30:22] 👤 tran_thi_b (0.91) - 4.5s
[2023-10-16 14:30:28] 👤 nguyen_van_a (0.82) - 2.8s
```

### Giải thích log:
- **Timestamp**: Thời gian nhận dạng
- **👤/❓**: Icon cho người đã đăng ký / người lạ
- **Tên**: Tên người nói hoặc "Unknown"
- **(0.87)**: Độ tin cậy (chỉ hiện với người đã đăng ký)
- **3.2s**: Độ dài đoạn nói

## Thông số kỹ thuật

### Cấu hình mặc định:
- **Sample rate**: 16000 Hz
- **Chunk duration**: 2.0 giây
- **Overlap duration**: 0.5 giây
- **Min speech duration**: 1.0 giây
- **Max speech duration**: 10.0 giây
- **Similarity threshold**: 0.65
- **VAD threshold**: 0.5

### Tối ưu hóa:
- **Chunk duration**: 
  - Ngắn (1-2s): Phản hồi nhanh, có thể ít chính xác
  - Dài (3-5s): Chính xác hơn, phản hồi chậm hơn

- **Similarity threshold**:
  - Thấp (0.5-0.6): Ít strict, có thể nhận nhầm
  - Cao (0.7-0.8): Strict hơn, có thể từ chối người đúng

## File outputs

### Conversation Log:
```json
{
  "timestamp": "2023-10-16 14:30:15",
  "speaker": "nguyen_van_a",
  "confidence": 0.87,
  "start_time": "14:30:15",
  "duration": 3.2,
  "status": "Enrolled"
}
```

### Summary Report:
```
📊 CONVERSATION SUMMARY:
  👤 nguyen_van_a: 5 segments, 12.3s (45.2%)
  👤 tran_thi_b: 3 segments, 8.7s (32.0%)
  ❓ Unknown: 2 segments, 6.2s (22.8%)
```

## Use Cases

### 1. Meeting Transcription:
```bash
# Ghi cuộc họp 30 phút
python3 realtime_speaker_demo.py
# Nhập: 30 phút, chunk 3 giây
```

### 2. Security Monitoring:
```bash
# Giám sát liên tục
python3 realtime_speaker_demo.py  
# Nhập: không giới hạn thời gian
```

### 3. Interview Analysis:
```bash
# Phân tích cuộc phỏng vấn
python3 realtime_speaker_demo.py
# Nhập: 60 phút, chunk 2 giây
```

## Troubleshooting

### Lỗi microphone:
```bash
# Test microphone
python3 voice_embedding_tool.py test_mic

# Kiểm tra devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Nhận dạng không chính xác:
1. **Enroll lại với audio chất lượng tốt hơn**
2. **Giảm similarity threshold**
3. **Tăng chunk duration**
4. **Kiểm tra VAD threshold**

### Performance issues:
1. **Sử dụng GPU** (CUDA) nếu có
2. **Giảm batch_size** trong config
3. **Tăng chunk duration** để giảm tần suất xử lý

## Advanced Usage

### Custom Configuration:
```python
from speaker_verification_pipeline import VerificationConfig, RealTimeSpeakerRecognition

config = VerificationConfig(
    similarity_threshold=0.7,
    vad_threshold=0.4,
    use_vad=True
)

pipeline = SpeakerVerificationPipeline(config)
recognizer = RealTimeSpeakerRecognition(
    pipeline=pipeline,
    chunk_duration=3.0,
    overlap_duration=1.0
)

recognizer.start_continuous_recognition(duration_minutes=10)
```

### Integration với other systems:
```python
# Callback function cho real-time processing
def speaker_detected_callback(speaker_name, confidence, timestamp):
    # Gửi thông tin đến hệ thống khác
    print(f"Speaker detected: {speaker_name} at {timestamp}")

# Modify the recognition system để add callback
```

Hệ thống sẵn sàng để nhận dạng người nói thời gian thực!