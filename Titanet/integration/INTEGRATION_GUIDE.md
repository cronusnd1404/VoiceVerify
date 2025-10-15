# Speaker Verification Integration API

Dễ dàng tích hợp TitaNet-L + Silero VAD v6 vào hệ thống voice-to-text hiện tại

## Cài đặt nhanh

```python
from speaker_verification_pipeline import create_pipeline, quick_verify

# Tạo pipeline với cấu hình mặc định
pipeline = create_pipeline()

# Đăng ký speaker (chỉ cần làm 1 lần)
pipeline.enroll_speaker("user001", ["enroll1.wav", "enroll2.wav"])

# Xác thực speaker trong quá trình voice-to-text
result = quick_verify("input_audio.wav", "user001", pipeline)
print(f"Verified: {result['verified']}")
```

## Tích hợp với Voice-to-Text

### Phương án 1: Pipeline tuần tự
```python
def process_audio_with_verification(audio_path, claimed_speaker=None):
    # 1. Speaker verification trước
    verification_result = pipeline.verify_speaker(audio_path, claimed_speaker)
    
    if verification_result.get('verified', False):
        # 2. Chạy voice-to-text nếu speaker được xác thực
        transcription = your_voice_to_text_model(audio_path)
        return {
            'transcription': transcription,
            'speaker_verified': True,
            'speaker_info': verification_result
        }
    else:
        return {
            'transcription': None,
            'speaker_verified': False,
            'speaker_info': verification_result
        }
```

### Phương án 2: Pipeline song song (nhanh hơn)
```python
import asyncio
import concurrent.futures

async def process_audio_parallel(audio_path, claimed_speaker=None):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Chạy song song speaker verification và voice-to-text
        verification_future = executor.submit(
            pipeline.verify_speaker, audio_path, claimed_speaker
        )
        transcription_future = executor.submit(
            your_voice_to_text_model, audio_path
        )
        
        # Đợi cả hai hoàn thành
        verification_result = await asyncio.wrap_future(verification_future)
        transcription = await asyncio.wrap_future(transcription_future)
    
    return {
        'transcription': transcription,
        'speaker_verified': verification_result.get('verified', False),
        'speaker_info': verification_result
    }
```

### Phương án 3: VAD chia sẻ (tối ưu nhất)
```python
def process_audio_shared_vad(audio_path, claimed_speaker=None):
    # 1. Áp dụng VAD một lần, dùng chung cho cả hai model
    waveform, sr = pipeline.preprocess_audio(audio_path)
    vad_audio = pipeline.apply_vad(waveform, sr)
    
    # Lưu audio sau VAD
    temp_path = "/tmp/vad_processed.wav"
    torchaudio.save(temp_path, vad_audio, sr)
    
    # 2. Speaker verification với audio đã qua VAD
    verification_result = pipeline.verify_speaker(temp_path, claimed_speaker)
    
    # 3. Voice-to-text với cùng audio đã qua VAD (sạch hơn)
    transcription = your_voice_to_text_model(temp_path)
    
    return {
        'transcription': transcription,
        'speaker_verified': verification_result.get('verified', False),
        'speaker_info': verification_result,
        'vad_applied': True
    }
```

## Cấu hình cho Production

### Config file (config.json)
```json
{
    "titanet_model_path": "/path/to/titanet-l.nemo",
    "use_vad": true,
    "vad_threshold": 0.5,
    "similarity_threshold": 0.65,
    "min_audio_duration": 1.0,
    "max_audio_duration": 60.0,
    "device": "cuda",
    "batch_size": 4,
    "temp_dir": "/tmp/speaker_verification",
    "enrollment_db_path": "./speaker_enrollments.json"
}
```

### Khởi tạo với config
```python
pipeline = create_pipeline("config.json")
```

## Batch Processing

```python
# Xử lý nhiều file cùng lúc
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
claimed_speakers = ["user001", "user002", "user001"]

results = pipeline.batch_verify(audio_files, claimed_speakers)

for result in results:
    if result['verified']:
        # Chạy voice-to-text cho file này
        transcription = your_voice_to_text_model(result['audio_path'])
        print(f"Speaker verified, transcription: {transcription}")
```

## Enrollment Management

```python
# Đăng ký speaker mới
success = pipeline.enroll_speaker(
    speaker_id="user001", 
    audio_paths=["enroll1.wav", "enroll2.wav", "enroll3.wav"]
)

# Xem thống kê enrollment
stats = pipeline.get_enrollment_stats()
print(f"Total speakers: {stats['total_enrolled_speakers']}")

# Load/save enrollment database
# Database tự động lưu vào speaker_enrollments.json
```

## Monitoring & Logging

```python
import logging

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speaker_verification.log'),
        logging.StreamHandler()
    ]
)

# Pipeline sẽ tự động log các hoạt động
pipeline = create_pipeline()
```

## Performance Tips

1. **GPU Acceleration**: Đặt `device: "cuda"` trong config
2. **Batch Processing**: Sử dụng `batch_verify()` cho nhiều file
3. **VAD Sharing**: Dùng VAD chung cho cả speaker verification và voice-to-text
4. **Caching**: Embedding sẽ được cache trong enrollment database
5. **Async Processing**: Chạy song song với voice-to-text khi có thể

## Error Handling

```python
try:
    result = pipeline.verify_speaker("audio.wav", "user001")
    if not result['success']:
        print(f"Verification failed: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"Pipeline error: {e}")
    # Fallback: chạy voice-to-text mà không cần verification
    transcription = your_voice_to_text_model("audio.wav")
```

## Integration Examples

### FastAPI Web Service
```python
from fastapi import FastAPI, File, UploadFile
import tempfile

app = FastAPI()
pipeline = create_pipeline()

@app.post("/verify-and-transcribe/")
async def verify_and_transcribe(
    audio: UploadFile = File(...),
    speaker_id: str = None
):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        tmp_file.write(await audio.read())
        tmp_file.flush()
        
        # Verification
        verification = pipeline.verify_speaker(tmp_file.name, speaker_id)
        
        if verification.get('verified', False):
            # Transcription
            transcription = your_voice_to_text_model(tmp_file.name)
            return {
                "verified": True,
                "transcription": transcription,
                "speaker_info": verification
            }
        else:
            return {
                "verified": False, 
                "message": "Speaker not verified",
                "speaker_info": verification
            }
```

### Real-time Streaming
```python
import pyaudio
import wave

def real_time_verification(speaker_id, duration=5):
    # Record audio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, 
                       rate=16000, input=True, frames_per_buffer=1024)
    
    frames = []
    for _ in range(int(16000/1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        wf = wave.open(tmp_file.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Verify
        result = pipeline.verify_speaker(tmp_file.name, speaker_id)
        return result['verified']
```