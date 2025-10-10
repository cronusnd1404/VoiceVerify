import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# ===== LOAD TITANET-S =====
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

# ===== 1. ENROLL GIỌNG BẠN =====
enroll_file = "Việt Anh_24.9.wav"   # đổi thành file giọng bạn
print(f"🔄 Đang tạo embedding từ file: {enroll_file}")

# Kiểm tra độ dài file
wav_check, sr_check = torchaudio.load(enroll_file)
duration = wav_check.shape[1] / sr_check
print(f"📏 Độ dài file: {duration:.2f}s")

# Lấy embedding từ đoạn đầu (tối đa 30s để tránh quá dài)
max_duration = min(30, duration - 1)  # Lấy tối đa 30s, tránh phần cuối
enroll_emb = get_embedding(enroll_file, 1, max_duration)  # Bắt đầu từ giây thứ 1
np.save("enroll_emb.npy", enroll_emb)
print(f"✅ Đã lưu embedding giọng bạn vào enroll_emb.npy (shape: {enroll_emb.shape})")
print(f"📊 Norm của embedding: {np.linalg.norm(enroll_emb):.3f}")

# ===== 2. VERIFY VỚI FILE MỚI (CÓ 2 SPEAKERS) =====
verify_file = "conversation-test.wav"   # đổi thành file hội thoại
print(f"\n🔍 Đang phân tích file: {verify_file}")

wav, sr = torchaudio.load(verify_file)
verify_duration = wav.shape[1] / sr
print(f"📏 Độ dài file verify: {verify_duration:.2f}s")

# Preprocessing nhất quán
if wav.shape[0] > 1:
    wav = wav[:1, :]
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

# ===== Silero VAD =====
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, _, _, _, _) = utils
segments = get_speech_timestamps(
    wav.squeeze(), silero_model, sampling_rate=sr,
    threshold=0.4, min_speech_duration_ms=250, min_silence_duration_ms=150
)

print(f"🔍 VAD phát hiện {len(segments)} đoạn nói")

# ===== So sánh từng đoạn với giọng bạn =====
cosine_scores = []
same_speaker_segments = []
different_speaker_segments = []

for i, seg in enumerate(segments, 1):
    start_s = seg["start"] / sr
    end_s   = seg["end"] / sr
    duration = end_s - start_s
    
    # Bỏ qua đoạn quá ngắn
    if duration < 0.5:
        print(f"Đoạn {i}: {start_s:.2f}s – {end_s:.2f}s | ⏭️ Bỏ qua (quá ngắn: {duration:.2f}s)")
        continue
    
    seg_wav = wav[:, seg["start"]:seg["end"]]
    emb = get_embedding_from_wav(seg_wav, sr)

    # Tính cosine similarity với embedding đã normalize
    cos_sim = F.cosine_similarity(
        torch.tensor(enroll_emb).unsqueeze(0),
        torch.tensor(emb).unsqueeze(0)
    ).item()
    
    cosine_scores.append(cos_sim)
    
    # Lowered threshold và thêm thông tin chi tiết
    if cos_sim > 0.4:  # Threshold thấp hơn
        label = "✅ Có thể là giọng bạn"
        same_speaker_segments.append((i, cos_sim))
    else:
        label = "❌ Người khác"
        different_speaker_segments.append((i, cos_sim))
    
    print(f"Đoạn {i}: {start_s:.2f}s – {end_s:.2f}s ({duration:.1f}s) | Cosine={cos_sim:.3f} | {label}")

# ===== THỐNG KÊ =====
if cosine_scores:
    print(f"\n📊 THỐNG KÊ:")
    print(f"   • Số đoạn phân tích: {len(cosine_scores)}")
    print(f"   • Cosine trung bình: {np.mean(cosine_scores):.3f}")
    print(f"   • Cosine cao nhất: {max(cosine_scores):.3f}")
    print(f"   • Cosine thấp nhất: {min(cosine_scores):.3f}")
    print(f"   • Độ lệch chuẩn: {np.std(cosine_scores):.3f}")
    
    print(f"\n🎯 KẾT QUẢ PHÂN LOẠI:")
    print(f"   • Có thể là giọng bạn: {len(same_speaker_segments)} đoạn")
    print(f"   • Người khác: {len(different_speaker_segments)} đoạn")
