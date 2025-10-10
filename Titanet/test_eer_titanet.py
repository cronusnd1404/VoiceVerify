#!/usr/bin/env python3
"""
Script để test EER của mô hình TitaNet-L trên dataset tiếng Việt
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import random
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import warnings
warnings.filterwarnings("ignore")

def load_titanet_model(model_path):
    """Load TitaNet model từ file .nemo"""
    print(f"Loading TitaNet model from {model_path}...")
    model = EncDecSpeakerLabelModel.restore_from(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model loaded successfully!")
    return model

def load_dataset_info(csv_path, audio_dir):
    """Load thông tin dataset từ file CSV"""
    print(f"Loading dataset info from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Tạo full path cho audio files
    df['audio_path'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))
    
    # Kiểm tra xem file có tồn tại không
    existing_files = []
    for idx, row in df.iterrows():
        if os.path.exists(row['audio_path']):
            existing_files.append(idx)
    
    df = df.iloc[existing_files].reset_index(drop=True)
    print(f"Found {len(df)} valid audio files")
    print(f"Number of unique speakers: {df['speaker'].nunique()}")
    
    return df

def extract_embeddings(model, audio_paths, batch_size=32):
    """Trích xuất embeddings cho list audio files"""
    print(f"Extracting embeddings for {len(audio_paths)} files...")
    embeddings = []
    
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = audio_paths[i:i+batch_size]
        batch_embeddings = []
        
        for audio_path in batch_paths:
            try:
                # Load audio file
                audio, sr = sf.read(audio_path)
                
                # Convert to torch tensor
                audio_tensor = torch.FloatTensor(audio)
                if torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()
                
                # Extract embedding
                with torch.no_grad():
                    embedding = model.get_embedding(audio_tensor.unsqueeze(0))
                    if torch.cuda.is_available():
                        embedding = embedding.cpu()
                    batch_embeddings.append(embedding.squeeze().numpy())
                    
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Tạo zero embedding nếu có lỗi
                batch_embeddings.append(np.zeros(192))  # TitaNet-L có embedding size 192
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def create_trial_pairs(df, num_genuine=5000, num_impostor=5000):
    """Tạo các cặp thử nghiệm genuine và impostor"""
    print(f"Creating {num_genuine} genuine pairs and {num_impostor} impostor pairs...")
    
    genuine_pairs = []
    impostor_pairs = []
    labels = []
    
    # Tạo genuine pairs (cùng speaker)
    speakers = df['speaker'].unique()
    
    for _ in range(num_genuine):
        # Chọn random speaker
        speaker = random.choice(speakers)
        speaker_files = df[df['speaker'] == speaker].index.tolist()
        
        if len(speaker_files) >= 2:
            # Chọn 2 file khác nhau từ cùng speaker
            file1, file2 = random.sample(speaker_files, 2)
            genuine_pairs.append((file1, file2))
            labels.append(1)  # genuine
    
    # Tạo impostor pairs (khác speaker)
    for _ in range(num_impostor):
        # Chọn 2 speaker khác nhau
        speaker1, speaker2 = random.sample(list(speakers), 2)
        
        files1 = df[df['speaker'] == speaker1].index.tolist()
        files2 = df[df['speaker'] == speaker2].index.tolist()
        
        if files1 and files2:
            file1 = random.choice(files1)
            file2 = random.choice(files2)
            impostor_pairs.append((file1, file2))
            labels.append(0)  # impostor
    
    # Combine pairs
    all_pairs = genuine_pairs + impostor_pairs
    
    print(f"Created {len(genuine_pairs)} genuine pairs and {len(impostor_pairs)} impostor pairs")
    return all_pairs, labels

def compute_cosine_similarity(embedding1, embedding2):
    """Tính cosine similarity giữa 2 embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def compute_scores(embeddings, pairs):
    """Tính similarity scores cho tất cả các pairs"""
    print("Computing similarity scores...")
    scores = []
    
    for file1_idx, file2_idx in tqdm(pairs, desc="Computing scores"):
        emb1 = embeddings[file1_idx]
        emb2 = embeddings[file2_idx]
        score = compute_cosine_similarity(emb1, emb2)
        scores.append(score)
    
    return np.array(scores)

def compute_eer(labels, scores):
    """Tính Equal Error Rate (EER)"""
    print("Computing EER...")
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # Compute EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # Find threshold at EER
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer, thresh

def main():
    parser = argparse.ArgumentParser(description='Test EER of TitaNet-L on Vietnamese dataset')
    parser.add_argument('--model_path', type=str, default='titanet-l.nemo',
                      help='Path to TitaNet-L model file')
    parser.add_argument('--test_csv', type=str, default='dataset/test/test.csv',
                      help='Path to test CSV file')
    parser.add_argument('--test_audio_dir', type=str, default='dataset/test',
                      help='Path to test audio directory')
    parser.add_argument('--num_genuine', type=int, default=5000,
                      help='Number of genuine pairs for testing')
    parser.add_argument('--num_impostor', type=int, default=5000,
                      help='Number of impostor pairs for testing')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for embedding extraction')
    parser.add_argument('--save_embeddings', type=str, default=None,
                      help='Path to save extracted embeddings (optional)')
    parser.add_argument('--load_embeddings', type=str, default=None,
                      help='Path to load pre-extracted embeddings (optional)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=== TitaNet-L EER Testing on Vietnamese Dataset ===")
    
    # Load dataset info
    df = load_dataset_info(args.test_csv, args.test_audio_dir)
    
    if args.load_embeddings and os.path.exists(args.load_embeddings):
        print(f"Loading pre-extracted embeddings from {args.load_embeddings}")
        embeddings = np.load(args.load_embeddings)
    else:
        # Load model
        model = load_titanet_model(args.model_path)
        
        # Extract embeddings
        embeddings = extract_embeddings(model, df['audio_path'].tolist(), args.batch_size)
        
        # Save embeddings if requested
        if args.save_embeddings:
            print(f"Saving embeddings to {args.save_embeddings}")
            np.save(args.save_embeddings, embeddings)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create trial pairs
    pairs, labels = create_trial_pairs(df, args.num_genuine, args.num_impostor)
    
    # Compute similarity scores
    scores = compute_scores(embeddings, pairs)
    
    # Compute EER
    eer, threshold = compute_eer(labels, scores)
    
    print("\n=== RESULTS ===")
    print(f"Number of test files: {len(df)}")
    print(f"Number of speakers: {df['speaker'].nunique()}")
    print(f"Number of genuine pairs: {sum(labels)}")
    print(f"Number of impostor pairs: {len(labels) - sum(labels)}")
    print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
    print(f"EER Threshold: {threshold:.4f}")
    
    # Additional statistics
    genuine_scores = scores[np.array(labels) == 1]
    impostor_scores = scores[np.array(labels) == 0]
    
    print(f"\nScore Statistics:")
    print(f"Genuine scores - Mean: {genuine_scores.mean():.4f}, Std: {genuine_scores.std():.4f}")
    print(f"Impostor scores - Mean: {impostor_scores.mean():.4f}, Std: {impostor_scores.std():.4f}")
    
    # Save results
    results = {
        'eer': eer,
        'threshold': threshold,
        'num_files': len(df),
        'num_speakers': df['speaker'].nunique(),
        'num_genuine_pairs': sum(labels),
        'num_impostor_pairs': len(labels) - sum(labels),
        'genuine_mean': genuine_scores.mean(),
        'genuine_std': genuine_scores.std(),
        'impostor_mean': impostor_scores.mean(),
        'impostor_std': impostor_scores.std()
    }
    
    results_file = 'titanet_eer_results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()