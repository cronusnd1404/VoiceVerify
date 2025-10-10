#!/usr/bin/env python3
"""
Script debug để kiểm tra tại sao cosine similarity thấp
"""
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# ===== LOAD TITANET-L =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Sử dụng device: {device}")

model = EncDecSpeakerLabelModel.restore_from("titanet-l.nemo", map_location=device)
model = model.to(device)
model.eval()
print("✅ Đã load TitaNet-L model")

# ===== HÀM LẤY EMBEDDING CÓ DEBUG =====
@torch.no_grad()
def get_embedding_debug(wav, sr, label=""):
    print(f"  🔍 [{label}] Input shape: {wav.shape}, SR: {sr}")
    
    # Đảm bảo mono
    if wav.shape[0] > 1:
        wav = wav[:1, :]
        print(f"  📻 [{label}] Converted to mono: {wav.shape}")
    
    # Resample về 16kHz
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
        print(f"  🔄 [{label}] Resampled to 16kHz: {wav.shape}")

    # Thống kê audio
    print(f"  📊 [{label}] Audio stats - Min: {wav.min():.6f}, Max: {wav.max():.6f}, Mean: {wav.mean():.6f}")
    
    # Chuẩn hóa amplitude
    max_val = wav.abs().max()
    if max_val > 0:
        wav = wav / (max_val + 1e-7)
        print(f"  🎚️ [{label}] Normalized by max value: {max_val:.6f}")
    
    # Chuẩn input: [B, T]
    audio_signal = wav.squeeze(0).unsqueeze(0).to(device)   # [1, T]
    audio_length = torch.tensor([audio_signal.shape[-1]], device=device)
    print(f"  🎯 [{label}] Model input: {audio_signal.shape}, length: {audio_length.item()}")

    emb, _ = model.forward(
        input_signal=audio_signal,
        input_signal_length=audio_length
    )
    
    print(f"  🧠 [{label}] Raw embedding shape: {emb.shape}")
    print(f"  📏 [{label}] Raw embedding norm: {emb.norm(dim=1).item():.6f}")
    
    # L2 normalize embedding
    emb = F.normalize(emb, p=2, dim=1)
    emb_np = emb.squeeze().cpu().numpy()
    
    print(f"  ✅ [{label}] Final embedding shape: {emb_np.shape}, norm: {np.linalg.norm(emb_np):.6f}")
    return emb_np

# ===== TEST: CÙNG FILE, KHÁC ĐOẠN =====
def test_same_file_different_segments():
    print("\n" + "="*60)
    print("🧪 TEST 1: Cùng file, khác đoạn thời gian")
    print("="*60)
    
    file = "Việt Anh_24.9.wav"
    wav, sr = torchaudio.load(file)
    duration = wav.shape[1] / sr
    print(f"📁 File: {file} | Duration: {duration:.2f}s")
    
    # Lấy 3 đoạn khác nhau từ cùng file
    segment1 = wav[:, int(2*sr):int(7*sr)]    # 2-7s
    segment2 = wav[:, int(8*sr):int(13*sr)]   # 8-13s  
    segment3 = wav[:, int(15*sr):int(20*sr)]  # 15-20s
    
    emb1 = get_embedding_debug(segment1, sr, "Đoạn 2-7s")
    emb2 = get_embedding_debug(segment2, sr, "Đoạn 8-13s")
    emb3 = get_embedding_debug(segment3, sr, "Đoạn 15-20s")
    
    # So sánh
    cos12 = F.cosine_similarity(torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0)).item()
    cos13 = F.cosine_similarity(torch.tensor(emb1).unsqueeze(0), torch.tensor(emb3).unsqueeze(0)).item()
    cos23 = F.cosine_similarity(torch.tensor(emb2).unsqueeze(0), torch.tensor(emb3).unsqueeze(0)).item()
    
    print(f"\n📊 KẾT QUẢ SO SÁNH (cùng người):")
    print(f"   • Đoạn 1 vs Đoạn 2: {cos12:.4f}")
    print(f"   • Đoạn 1 vs Đoạn 3: {cos13:.4f}")
    print(f"   • Đoạn 2 vs Đoạn 3: {cos23:.4f}")
    print(f"   • Trung bình: {(cos12+cos13+cos23)/3:.4f}")
    
    return emb1  # Trả về embedding đầu tiên làm reference

# ===== TEST: KHÁC FILE, CÙNG NGƯỜI =====
def test_different_files_same_person(ref_emb):
    print("\n" + "="*60)
    print("🧪 TEST 2: So sánh với embedding từ test1")
    print("="*60)
    
    # Load embedding đã lưu trước đó
    if os.path.exists("enroll_emb.npy"):
        saved_emb = np.load("enroll_emb.npy")
        cos_saved = F.cosine_similarity(torch.tensor(ref_emb).unsqueeze(0), torch.tensor(saved_emb).unsqueeze(0)).item()
        print(f"📁 So sánh với enroll_emb.npy: {cos_saved:.4f}")
    else:
        print("⚠️ Không tìm thấy enroll_emb.npy")

# ===== TEST: KHÁC NGƯỜI =====
def test_different_person(ref_emb):
    print("\n" + "="*60)
    print("🧪 TEST 3: So sánh với người khác")
    print("="*60)
    
    other_files = ["speaker1.wav", "speaker2.wav", "my_voice.wav", "my_voice2.wav"]
    
    for file in other_files:
        if os.path.exists(file):
            try:
                wav, sr = torchaudio.load(file)
                # Lấy 5s đầu
                segment = wav[:, :int(5*sr)]
                emb = get_embedding_debug(segment, sr, f"File {file}")
                
                cos_sim = F.cosine_similarity(torch.tensor(ref_emb).unsqueeze(0), torch.tensor(emb).unsqueeze(0)).item()
                print(f"📊 {file}: {cos_sim:.4f}")
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {file}: {e}")
        else:
            print(f"⚠️ Không tìm thấy {file}")

# ===== MAIN =====
if __name__ == "__main__":
    import os
    
    print("🚀 Bắt đầu debug embedding...")
    
    ref_emb = test_same_file_different_segments()
    test_different_files_same_person(ref_emb)
    test_different_person(ref_emb)
    
    print("\n🎯 KHUYẾN NGHỊ:")
    print("   • Cosine > 0.7: Rất có thể cùng người")
    print("   • Cosine 0.4-0.7: Có thể cùng người")  
    print("   • Cosine < 0.4: Có thể khác người")
    print("   • Nếu cùng người mà cosine < 0.3: Có vấn đề về chất lượng audio hoặc model")