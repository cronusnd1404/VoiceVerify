#!/usr/bin/env python3
"""
Enhanced Speaker Verification with Dynamic Threshold
"""
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# ===== LOAD TITANET-L =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncDecSpeakerLabelModel.restore_from("titanet-l.nemo", map_location=device)
model = model.to(device)
model.eval()

# ===== HÀM LẤY EMBEDDING =====
@torch.no_grad()
def get_embedding_from_wav(wav, sr):
    # Đảm bảo mono
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    
    # Resample về 16kHz
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # Chuẩn hóa amplitude
    wav = wav / (wav.abs().max() + 1e-7)
    
    # Chuẩn input: [B, T]
    audio_signal = wav.squeeze(0).unsqueeze(0).to(device)   # [1, T]
    audio_length = torch.tensor([audio_signal.shape[-1]], device=device)

    emb, _ = model.forward(
        input_signal=audio_signal,
        input_signal_length=audio_length
    )
    
    # L2 normalize embedding
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze().cpu().numpy()


def get_embedding(file, start=0, end=None):
    wav, sr = torchaudio.load(file)
    
    # Cắt đoạn audio nếu cần
    if end:
        wav = wav[:, int(start*sr): int(end*sr)]
    elif start > 0:
        wav = wav[:, int(start*sr):]
    
    return get_embedding_from_wav(wav, sr)

# ===== KIỂM TRA SILERO VAD VERSION =====
def check_silero_version():
    """Kiểm tra và hiển thị thông tin Silero VAD version"""
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False
        )
        
        # Check v6 features
        has_reset_states = hasattr(model, 'reset_states')
        has_vad_iterator = len(utils) >= 5
        
        print(f"🔍 Silero VAD Info:")
        print(f"   • Model loaded: ✅")
        print(f"   • Has reset_states (v6): {'✅' if has_reset_states else '❌'}")
        print(f"   • Has VADIterator: {'✅' if has_vad_iterator else '❌'}")
        print(f"   • Utils functions: {len(utils)}")
        
        if has_reset_states and has_vad_iterator:
            print(f"   • Version: v6 compatible ✅")
            return True
        else:
            print(f"   • Version: Older version ⚠️")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Silero: {e}")
        return False

# ===== HÀM PHÂN TÍCH THÔNG MINH =====
def smart_speaker_verification(enroll_file, verify_file, enroll_duration=30):
    print(f"🎯 SMART SPEAKER VERIFICATION")
    print(f"📁 Enroll: {enroll_file}")
    print(f"📁 Verify: {verify_file}")
    print("="*60)
    
    # 1. Tạo embedding reference
    print(f"🔄 Đang tạo embedding reference từ file: {enroll_file}")
    wav_check, sr_check = torchaudio.load(enroll_file)
    duration = wav_check.shape[1] / sr_check
    print(f"📏 Độ dài file enroll: {duration:.2f}s")

    max_duration = min(enroll_duration, duration - 1)
    enroll_emb = get_embedding(enroll_file, 1, max_duration)
    print(f"✅ Đã tạo embedding reference (shape: {enroll_emb.shape})")
    
    # 2. Phân tích file verify
    print(f"\n🔍 Đang phân tích file: {verify_file}")
    wav, sr = torchaudio.load(verify_file)
    verify_duration = wav.shape[1] / sr
    print(f"📏 Độ dài file verify: {verify_duration:.2f}s")

    # Preprocessing
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # 3. VAD với Silero VAD v6
    print("🔄 Loading Silero VAD v6...")
    silero_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # Sử dụng tham số tối ưu cho v6
    segments = get_speech_timestamps(
        wav.squeeze(), 
        silero_model, 
        sampling_rate=sr,
        threshold=0.5,                    # Threshold hơi cao hơn cho v6
        min_speech_duration_ms=1000,      # Tăng lên 1s để lọc tốt hơn
        min_silence_duration_ms=200,      # Silence ngắn hơn
        window_size_samples=1536,         # v6 default window size
        speech_pad_ms=30                  # Padding cho speech segments
    )

    print(f"🔍 VAD phát hiện {len(segments)} đoạn nói")
    
    # Optional: Sử dụng VADIterator cho real-time processing (v6 feature)
    if len(segments) > 20:  # Chỉ dùng streaming cho file dài
        print("📡 Sử dụng VADIterator cho streaming processing...")
        vad_iterator = VADIterator(silero_model, threshold=0.5, sampling_rate=sr)
        # Process audio in chunks for better memory usage
        chunk_size = sr * 30  # 30 seconds chunks
        refined_segments = []
        
        for i in range(0, wav.shape[1], chunk_size):
            chunk = wav[:, i:min(i+chunk_size, wav.shape[1])].squeeze()
            if len(chunk) > sr:  # Skip chunks < 1 second
                chunk_segments = vad_iterator(chunk)
                # Adjust timestamps for global position
                for seg in chunk_segments:
                    refined_segments.append({
                        'start': seg['start'] + i,
                        'end': seg['end'] + i
                    })
        
        vad_iterator.reset_states()  # v6 feature
        segments = refined_segments if refined_segments else segments
        print(f"📡 Streaming VAD đã tinh chỉnh: {len(segments)} đoạn")
    
    # 4. Tính toán cosine cho tất cả đoạn
    cosine_scores = []
    segment_info = []
    
    for i, seg in enumerate(segments, 1):
        start_s = seg["start"] / sr
        end_s = seg["end"] / sr
        duration = end_s - start_s
        
        # Bỏ qua đoạn quá ngắn
        if duration < 1.0:  # Tăng threshold lên 1s
            continue
        
        seg_wav = wav[:, seg["start"]:seg["end"]]
        emb = get_embedding_from_wav(seg_wav, sr)

        cos_sim = F.cosine_similarity(
            torch.tensor(enroll_emb).unsqueeze(0),
            torch.tensor(emb).unsqueeze(0)
        ).item()
        
        cosine_scores.append(cos_sim)
        segment_info.append((i, start_s, end_s, duration, cos_sim))
    
    if not cosine_scores:
        print("⚠️ Không có đoạn nào đủ dài để phân tích!")
        return
    
    # 5. Sử dụng threshold cố định
    scores_array = np.array(cosine_scores)
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    
    # Threshold cố định = 0.6
    dynamic_threshold = 0.6
    
    print(f"\n📊 PHÂN TÍCH PHÂN PHỐI:")
    print(f"   • Số đoạn hợp lệ: {len(cosine_scores)}")
    print(f"   • Cosine trung bình: {mean_score:.3f}")
    print(f"   • Độ lệch chuẩn: {std_score:.3f}")
    print(f"   • Min/Max: {scores_array.min():.3f} / {scores_array.max():.3f}")
    print(f"   • Threshold cố định: {dynamic_threshold:.3f}")
    
    # 6. Phân loại với threshold động
    same_speaker = []
    different_speaker = []
    
    print(f"\n🎯 KẾT QUẢ PHÂN LOẠI:")
    for i, start_s, end_s, duration, cos_sim in segment_info:
        if cos_sim >= dynamic_threshold:
            label = "✅ Giọng bạn"
            same_speaker.append((i, cos_sim))
        else:
            label = "❌ Người khác" 
            different_speaker.append((i, cos_sim))
        
        print(f"Đoạn {i}: {start_s:.1f}s–{end_s:.1f}s ({duration:.1f}s) | Cosine={cos_sim:.3f} | {label}")
    
    # 7. Tổng kết
    print(f"\n📈 TỔNG KẾT:")
    print(f"   • Giọng bạn: {len(same_speaker)} đoạn ({len(same_speaker)/len(segment_info)*100:.1f}%)")
    print(f"   • Người khác: {len(different_speaker)} đoạn ({len(different_speaker)/len(segment_info)*100:.1f}%)")
    
    if same_speaker:
        same_scores = [score for _, score in same_speaker]
        print(f"   • Độ tin cậy giọng bạn: {np.mean(same_scores):.3f} ± {np.std(same_scores):.3f}")
    
    # 8. Lưu kết quả
    np.save("enroll_emb.npy", enroll_emb)
    
    # Lưu segments để tránh phải chạy VAD lại
    segments_data = {
        'segments': segments,
        'file': verify_file,
        'sr': sr,
        'duration': verify_duration
    }
    np.save("vad_segments.npy", segments_data, allow_pickle=True)
    
    print(f"\n💾 Đã lưu:")
    print(f"   • embedding reference: enroll_emb.npy")
    print(f"   • VAD segments: vad_segments.npy")
    
    return {
        'same_speaker_segments': same_speaker,
        'different_speaker_segments': different_speaker,
        'threshold': dynamic_threshold,
        'stats': {
            'mean': mean_score,
            'std': std_score,
            'min': scores_array.min(),
            'max': scores_array.max()
        }
    }

# ===== MAIN =====
if __name__ == "__main__":
    print("🚀 ENHANCED SPEAKER VERIFICATION WITH SILERO VAD v6")
    print("="*60)
    
    # Kiểm tra Silero VAD version
    is_v6 = check_silero_version()
    
    # Cấu hình
    enroll_file = "Việt Anh_24.9.wav"
    verify_file = "conversation-test.wav"
    
    print(f"\n📁 Files:")
    print(f"   • Enroll: {enroll_file}")
    print(f"   • Verify: {verify_file}")
    print("="*60)
    
    # Chạy phân tích
    result = smart_speaker_verification(enroll_file, verify_file, enroll_duration=30)
    
    print(f"\n🔧 KHUYẾN NGHỊ ĐIỀU CHỈNH:")
    print(f"   • Threshold hiện tại: {result['threshold']:.3f} (cố định)")
    print(f"   • Nếu quá nhiều false positive: Tăng lên 0.70-0.75")
    print(f"   • Nếu quá nhiều false negative: Giảm xuống 0.60-0.65")
    print(f"   • Threshold động sẽ là: {result['stats']['mean'] + 0.2 * result['stats']['std']:.3f}")