# TitaNet-L EER Testing Script

Script để test Equal Error Rate (EER) của mô hình TitaNet-L trên dataset tiếng Việt.

## Yêu cầu

### Python Packages
```bash
pip install torch torchaudio
pip install numpy pandas scikit-learn scipy tqdm soundfile
pip install nemo_toolkit[all]
```

### Files cần thiết
- `titanet-l.nemo`: Mô hình TitaNet-L đã train
- `dataset/test/test.csv`: File CSV chứa mapping filename -> speaker_id
- `dataset/test/`: Thư mục chứa các file audio .wav

## Cách sử dụng

### Script đơn giản (khuyến nghị)
```bash
python test_eer_simple.py
```

Script này sẽ:
- Load mô hình TitaNet-L từ file `titanet-l.nemo`
- Đọc dataset từ `dataset/test/test.csv`
- Trích xuất embeddings từ các file audio
- Tạo các cặp thử nghiệm genuine/impostor
- Tính EER và lưu kết quả vào `eer_results.txt`

### Script đầy đủ (nhiều tùy chọn hơn)
```bash
# Chạy với cài đặt mặc định
python test_eer_titanet.py

# Chỉ định đường dẫn cụ thể
python test_eer_titanet.py \
    --model_path titanet-l.nemo \
    --test_csv dataset/test/test.csv \
    --test_audio_dir dataset/test \
    --num_genuine 5000 \
    --num_impostor 5000

# Lưu embeddings để sử dụng lại
python test_eer_titanet.py --save_embeddings test_embeddings.npy

# Load embeddings đã lưu (để test nhanh hơn)
python test_eer_titanet.py --load_embeddings test_embeddings.npy
```

## Tham số

### test_eer_titanet.py
- `--model_path`: Đường dẫn tới file mô hình .nemo (mặc định: titanet-l.nemo)
- `--test_csv`: Đường dẫn tới file CSV test (mặc định: dataset/test/test.csv)
- `--test_audio_dir`: Thư mục chứa file audio test (mặc định: dataset/test)
- `--num_genuine`: Số cặp genuine để test (mặc định: 5000)
- `--num_impostor`: Số cặp impostor để test (mặc định: 5000)
- `--batch_size`: Batch size cho việc trích xuất embedding (mặc định: 32)
- `--save_embeddings`: Đường dẫn lưu embeddings (tùy chọn)
- `--load_embeddings`: Đường dẫn load embeddings đã lưu (tùy chọn)

## Format Dataset

File CSV cần có format:
```csv
filename,speaker
000000.wav,id00005
000001.wav,id00005
000002.wav,id00010
...
```

## Kết quả

Script sẽ in ra:
- Số lượng file và speaker trong dataset
- Số cặp genuine/impostor được test
- EER (Equal Error Rate) và threshold tương ứng
- Thống kê điểm số genuine và impostor

Kết quả sẽ được lưu vào file:
- `eer_results.txt` (script đơn giản)
- `titanet_eer_results.txt` (script đầy đủ)

## Ví dụ kết quả
```
=== RESULTS ===
Dataset: 1000 files, 50 speakers
Test pairs: 2000 (1000 genuine, 1000 impostor)
EER: 0.0850 (8.50%)
EER Threshold: 0.7234
Genuine scores - Mean: 0.8456 ± 0.1234
Impostor scores - Mean: 0.3456 ± 0.2134
```

## Lưu ý

1. Script sẽ tự động phát hiện và sử dụng GPU nếu có sẵn
2. Với dataset lớn, việc trích xuất embedding có thể mất thời gian
3. Bạn có thể lưu embeddings lần đầu và tái sử dụng cho các lần test tiếp theo
4. EER càng thấp thì mô hình càng tốt (EER < 5% được coi là tốt cho speaker verification)

## Troubleshooting

### Lỗi import NeMo
```bash
pip install nemo_toolkit[all]
```

### Lỗi CUDA/GPU
Nếu gặp lỗi CUDA, script sẽ tự động chuyển sang CPU.

### Lỗi file not found
Kiểm tra đường dẫn tới model file và dataset có đúng không.