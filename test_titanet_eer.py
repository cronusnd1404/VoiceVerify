import os
import torch
import torchaudio
import pandas as pd
import itertools
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve

from nemo.collections.asr.models import EncDecSpeakerLabelModel

# ========= CONFIG =========
# Đường dẫn dữ liệu theo cấu trúc workspace hiện tại
TEST_CSV = "/home/edabk408/NgocDat/Titanet/dataset/test/test.csv"   # đường dẫn file CSV test
AUDIO_DIR = "/home/edabk408/NgocDat/Titanet/dataset/test"           # folder chứa file wav
# Sử dụng model TitaNet-L (.nemo) local thay vì tên pretrained
MODEL_PATH = "/home/edabk408/NgocDat/Titanet/titanet-l.nemo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Giới hạn thực tế để tránh nổ bộ nhớ khi tạo cặp nC2
MAX_FILES = 10000              # tối đa số file dùng để trích xuất embedding (None để dùng tất cả)
NUM_GENUINE = 2000             # số cặp cùng speaker (enroll vs probe)
NUM_IMPOSTOR = 2000            # số cặp khác speaker (enroll vs probe)
RANDOM_SEED = 42

# Thiết lập enroll/probe per speaker
ENROLL_PER_SPK = 2             # số file enroll mỗi speaker
PROBE_PER_SPK = 5              # số file probe mỗi speaker

# VAD cấu hình
USE_VAD = True                 # bật/tắt VAD
VAD_TYPE = "silero_v6"        # loại VAD: "silero_v6" hoặc "energy"
VAD_FRAME_MS = 25              # (chỉ dùng cho energy VAD) độ dài frame (ms)
VAD_ENERGY_THRESHOLD = 0.01    # (energy VAD) ngưỡng RMS tối thiểu để coi là speech
VAD_PAD_FRAMES = 2             # (energy VAD) pad thêm số frame trước/sau segment speech
TMP_VAD_DIR = "/home/edabk408/NgocDat/Titanet/tmp_vad"  # folder tạm lưu file sau VAD

# s-norm cấu hình
USE_S_NORM = True              # bật/tắt s-norm
COHORT_SIZE = 200              # số mẫu cohort từ các speaker khác
EPS = 1e-6

# ========= 1. LOAD MODEL =========
print("Loading TitaNet-L model from .nemo...")
model = EncDecSpeakerLabelModel.restore_from(MODEL_PATH)
model = model.eval().to(DEVICE)

# Load Silero VAD v6 nếu cần
silero_utils = None
silero_model = None
if USE_VAD and VAD_TYPE == "silero_v6":
    try:
        silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        print("Loaded Silero VAD v6")
    except Exception as e:
        print(f"Silero VAD load failed, fallback to energy VAD. Error: {e}")
        VAD_TYPE = "energy"

# ========= 2. LOAD CSV =========
df = pd.read_csv(TEST_CSV)
# Đảm bảo các cột cần thiết tồn tại và đặt tên chuẩn
required_cols = {"filename", "speaker"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV thiếu cột: cần {required_cols}, hiện có {set(df.columns)}")
print(f"Loaded {len(df)} rows from {TEST_CSV}")

# Lọc các file tồn tại và (tuỳ chọn) giới hạn số lượng
df["path"] = df["filename"].apply(lambda fn: os.path.join(AUDIO_DIR, fn))
df = df[df["path"].apply(os.path.isfile)]
if MAX_FILES is not None and len(df) > MAX_FILES:
    random.seed(RANDOM_SEED)
    df = df.sample(n=MAX_FILES, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Using {len(df)} valid audio files for embedding extraction")

# ========= 3. CHỌN ENROLL / PROBE PER SPEAKER =========
random.seed(RANDOM_SEED)
spk_groups = df.groupby("speaker")
selection = []  # [(filename, speaker, path, role)] role in {"enroll","probe"}
for spk, g in spk_groups:
    files_spk = g.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle
    enroll = files_spk.head(ENROLL_PER_SPK)
    remaining = files_spk.iloc[ENROLL_PER_SPK:]
    probe = remaining.head(PROBE_PER_SPK)
    for _, r in enroll.iterrows():
        selection.append((r["filename"], spk, r["path"], "enroll"))
    for _, r in probe.iterrows():
        selection.append((r["filename"], spk, r["path"], "probe"))

# Tạo đường dẫn sau VAD nếu bật
if USE_VAD:
    os.makedirs(TMP_VAD_DIR, exist_ok=True)

def apply_vad_and_save(in_path: str, out_path: str):
    """Đọc audio, chuyển mono 16k, áp dụng VAD (Silero v6 hoặc energy), rồi lưu WAV."""
    wav, sr = torchaudio.load(in_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    if VAD_TYPE == "silero_v6" and silero_utils is not None:
        # Silero expects 1D tensor
        audio_1d = wav.squeeze(0)
        get_speech_timestamps = silero_utils.get('get_speech_timestamps', None)
        collect_chunks = silero_utils.get('collect_chunks', None)
        if get_speech_timestamps is None or collect_chunks is None:
            # fallback
            pass
        else:
            try:
                timestamps = get_speech_timestamps(audio_1d, silero_model, sampling_rate=sr)
                out_1d = collect_chunks(timestamps, audio_1d)
                if out_1d is None or out_1d.numel() == 0:
                    # if no speech detected, save original
                    torchaudio.save(out_path, wav, sr)
                    return out_path
                out_wav = out_1d.unsqueeze(0)  # [1, N]
                torchaudio.save(out_path, out_wav, sr)
                return out_path
            except Exception:
                # fallback below
                pass

    # Fallback: energy VAD
    frame_len = int(sr * VAD_FRAME_MS / 1000)
    if frame_len <= 0:
        frame_len = 400
    total_samples = wav.shape[1]
    frames = []
    for start in range(0, total_samples, frame_len):
        end = min(start + frame_len, total_samples)
        frame = wav[:, start:end]
        if frame.numel() == 0:
            continue
        rms = torch.sqrt(torch.mean(frame**2) + 1e-12).item()
        frames.append((start, end, rms))
    speech_mask = [i for i, (_, _, rms) in enumerate(frames) if rms >= VAD_ENERGY_THRESHOLD]
    if not speech_mask:
        torchaudio.save(out_path, wav, sr)
        return out_path
    segments = []
    for idx_f in speech_mask:
        start, end, _ = frames[idx_f]
        segments.append((start, end))
    pad = VAD_PAD_FRAMES * frame_len
    merged = []
    for s, e in segments:
        s2 = max(0, s - pad)
        e2 = min(total_samples, e + pad)
        merged.append((s2, e2))
    merged.sort()
    compact = []
    cur_s, cur_e = merged[0]
    for s, e in merged[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            compact.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    compact.append((cur_s, cur_e))
    out_wav = torch.cat([wav[:, s:e] for (s, e) in compact], dim=1)
    if out_wav.shape[1] == 0:
        out_wav = wav
    torchaudio.save(out_path, out_wav, sr)
    return out_path

# ========= 4. EXTRACT EMBEDDINGS =========
embeddings = {}  # filename -> (embedding, speaker, role)
for fn, spk, path, role in tqdm(selection, desc="Extracting embeddings", total=len(selection)):
    src_path = path
    if USE_VAD:
        vad_path = os.path.join(TMP_VAD_DIR, f"{spk}__{role}__{fn}")
        try:
            src_path = apply_vad_and_save(path, vad_path)
        except Exception:
            src_path = path
    try:
        with torch.no_grad():
            emb = model.get_embedding(src_path).cpu().numpy().squeeze()
        embeddings[fn] = (emb, spk, role)
    except Exception:
        continue

# ========= 4. TẠO CẶP (SAME / DIFFERENT) =========
files = list(embeddings.keys())
print(f"Prepared embeddings for {len(files)} files (after enroll/probe). Building trial pairs...")

# Tách danh sách enroll và probe per speaker
enroll_by_spk = {}
probe_by_spk = {}
for fn in files:
    emb, spk, role = embeddings[fn]
    if role == "enroll":
        enroll_by_spk.setdefault(spk, []).append(fn)
    elif role == "probe":
        probe_by_spk.setdefault(spk, []).append(fn)

# Xây cohort cho s-norm (chọn một tập các embedding từ speaker khác)
cohort_probe_embs = []
cohort_enroll_embs = []
if USE_S_NORM:
    other_fns = [fn for fn in files if fn not in set().union(*probe_by_spk.values(), *enroll_by_spk.values())]
    # Nếu other_fns rỗng (do tất cả đều là enroll/probe đã chọn), dùng toàn bộ files làm nguồn và lọc cross-speaker khi tính
    pool_fns = files
    random.shuffle(pool_fns)
    for fn in pool_fns[:COHORT_SIZE]:
        emb, spk, role = embeddings[fn]
        if role == "probe":
            cohort_probe_embs.append((emb, spk))
        elif role == "enroll":
            cohort_enroll_embs.append((emb, spk))
    if not cohort_probe_embs:
        cohort_probe_embs = [(embeddings[fn][0], embeddings[fn][1]) for fn in files[:COHORT_SIZE]]
    if not cohort_enroll_embs:
        cohort_enroll_embs = [(embeddings[fn][0], embeddings[fn][1]) for fn in files[:COHORT_SIZE]]

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)

def s_norm_score(raw_score, emb_enroll, spk_enroll, emb_probe, spk_probe):
    # z-norm w.r.t enroll: compare enroll against cohort probes of different speakers
    z_scores = []
    for c_emb, c_spk in cohort_probe_embs:
        if c_spk == spk_enroll:
            continue
        s = cosine(emb_enroll, c_emb)
        if s is not None:
            z_scores.append(s)
    # t-norm w.r.t probe: compare cohort enrolls against probe
    t_scores = []
    for c_emb, c_spk in cohort_enroll_embs:
        if c_spk == spk_probe:
            continue
        s = cosine(c_emb, emb_probe)
        if s is not None:
            t_scores.append(s)
    if len(z_scores) < 3 or len(t_scores) < 3:
        return raw_score
    mu_z, sigma_z = np.mean(z_scores), np.std(z_scores) + EPS
    mu_t, sigma_t = np.mean(t_scores), np.std(t_scores) + EPS
    return 0.5 * (((raw_score - mu_z) / sigma_z) + ((raw_score - mu_t) / sigma_t))

pairs, labels = [], []

# Genuine: enroll vs probe cùng speaker
genuine_candidates = []
for spk in enroll_by_spk.keys():
    e_fns = enroll_by_spk.get(spk, [])
    p_fns = probe_by_spk.get(spk, [])
    for ef in e_fns:
        for pf in p_fns:
            genuine_candidates.append((ef, pf))
random.shuffle(genuine_candidates)
for (ef, pf) in genuine_candidates[:NUM_GENUINE]:
    emb_e, spk_e, _ = embeddings[ef]
    emb_p, spk_p, _ = embeddings[pf]
    raw = cosine(emb_e, emb_p)
    if raw is None:
        continue
    score = s_norm_score(raw, emb_e, spk_e, emb_p, spk_p) if USE_S_NORM else raw
    pairs.append(score)
    labels.append(1)

# Impostor: enroll vs probe khác speaker (sampling)
spk_list = list(enroll_by_spk.keys())
impostor_count = 0
while impostor_count < NUM_IMPOSTOR and len(spk_list) >= 2:
    spk_a, spk_b = random.sample(spk_list, 2)
    e_fns = enroll_by_spk.get(spk_a, [])
    p_fns = probe_by_spk.get(spk_b, [])
    if not e_fns or not p_fns:
        continue
    ef = random.choice(e_fns)
    pf = random.choice(p_fns)
    emb_e, spk_e, _ = embeddings[ef]
    emb_p, spk_p, _ = embeddings[pf]
    raw = cosine(emb_e, emb_p)
    if raw is None:
        continue
    score = s_norm_score(raw, emb_e, spk_e, emb_p, spk_p) if USE_S_NORM else raw
    pairs.append(score)
    labels.append(0)
    impostor_count += 1

pairs, labels = np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.int32)

# ========= 5. TÍNH EER =========
fpr, tpr, thresholds = roc_curve(labels, pairs)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
eer = fpr[np.nanargmin(np.abs(fnr - fpr))] * 100

print(f"\n✅ Equal Error Rate (EER): {eer:.2f}%")
print(f"Threshold: {eer_threshold:.4f}")

# ========= 6. LƯU KẾT QUẢ =========
out_df = pd.DataFrame({
    "score": pairs,
    "label": labels
})
out_df.to_csv("eer_scores.csv", index=False)
print("Scores saved to eer_scores.csv")
