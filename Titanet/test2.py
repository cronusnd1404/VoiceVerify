import torch
import torchaudio
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sklearn.cluster import KMeans

# ===== LOAD MODELS =====
print("🔄 Loading models...")
titanet = EncDecSpeakerLabelModel.from_pretrained("titanet_small")

# Load Silero VAD
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# ===== HÀM VAD TÁCH ĐOẠN SPEECH =====
def get_speech_segments(wav, sr):
    wav = wav.squeeze()  # [1, N] -> [N]
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    return speech_timestamps

# ===== HÀM LẤY EMBEDDING =====
def extract_embedding(segment_wav, sr):
    if sr != 16000:
        segment_wav = torchaudio.functional.resample(segment_wav, sr, 16000)
        sr = 16000
    emb = titanet.get_embedding(segment_wav.unsqueeze(0), sr)  # thêm batch dim
    return emb.squeeze().cpu().numpy()

# ===== MAIN =====
file = "conversation-test.wav"
wav, sr = torchaudio.load(file)

# Nếu stereo -> lấy 1 kênh
if wav.shape[0] > 1:
    wav = wav[0:1, :]

# 🔥 Resample về 16kHz
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

# Lấy các đoạn có speech bằng VAD
segments = get_speech_segments(wav, sr)

embeddings = []
segment_infos = []

for seg in segments:
    start = int(seg['start'] * sr)
    end   = int(seg['end'] * sr)
    segment_wav = wav[:, start:end]

    # Bỏ đoạn quá ngắn < 0.25s
    if segment_wav.shape[1] < sr * 0.1:
        continue

    emb = extract_embedding(segment_wav, sr)
    embeddings.append(emb)
    segment_infos.append((seg['start'], seg['end']))

# ===== KIỂM TRA VÀ CLUSTER =====
if len(embeddings) < 2:
    print("⚠️ Không đủ đoạn để phân biệt người nói (file quá ngắn hoặc VAD không phát hiện).")
else:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    print("\n===== KẾT QUẢ PHÂN LOẠI SPEAKER =====")
    for i, (start, end) in enumerate(segment_infos):
        print(f"Đoạn {i+1}: {start:.2f}s – {end:.2f}s --> Speaker {labels[i]}")
