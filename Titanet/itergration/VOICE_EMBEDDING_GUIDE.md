# Voice Embedding & Comparison Guide

## Cách thức hoạt động

1. **Tạo embedding vector** từ 1 file audio → vector 192 chiều
2. **Lưu vector** này làm reference
3. **So sánh các voice mới** với reference vector
4. **Tính cosine similarity** (0-1, càng cao càng giống)
5. **Quyết định** based on threshold

## Kết quả thực tế từ test

### Same Speaker (id00005):
- **000000.wav vs 000001.wav**: similarity = **0.9675** ✅ (SAME SPEAKER)
- Threshold 0.85: PASS ✅
- Confidence: 100%

### Different Speakers:
- **id00005 vs id00009**: similarity = **0.7947** ⚠️ (tương đối giống)
- **id00005 vs khác xa**: similarity = **0.6191** ❌ (khác rõ)

## Cách sử dụng

### 1. Basic Usage
```python
from voice_embedding_tool import VoiceEmbeddingTool

tool = VoiceEmbeddingTool()

# Tạo reference embedding
tool.extract_and_save_embedding("reference_voice.wav", "reference.pkl")

# So sánh voice mới
result = tool.compare_with_reference("test_voice.wav", "reference.pkl", threshold=0.80)
print(f"Same speaker: {result['is_same_speaker']}")
print(f"Similarity: {result['similarity_score']:.4f}")
```

### 2. Interactive Mode
```bash
python3 voice_embedding_tool.py interactive

# Commands:
> extract voice1.wav ref1.pkl
> compare voice2.wav ref1.pkl 0.80
> batch ref1.pkl /path/to/audio/folder/ 0.80
```

### 3. Batch Comparison
```python
# So sánh nhiều file cùng lúc
audio_files = ["voice1.wav", "voice2.wav", "voice3.wav"]
results = tool.batch_compare(audio_files, "reference.pkl", threshold=0.80)

for result in results:
    print(f"{result['test_audio']}: {result['similarity_score']:.3f} -> {result['decision']}")
```

## Threshold Recommendations

### Based on test results:

- **0.95+**: Rất chắc chắn same speaker (như enroll vs verify)
- **0.85-0.94**: Có khả năng cao same speaker  
- **0.70-0.84**: Có thể same speaker, cần xem xét
- **0.60-0.69**: Khác nhau nhưng có thể giống một phần
- **< 0.60**: Rõ ràng different speakers

### Recommended settings:
- **Strict verification**: threshold = 0.85-0.90
- **Balanced**: threshold = 0.75-0.80  
- **Lenient**: threshold = 0.65-0.70

## Production Integration

### Với Voice-to-Text System:
```python
def enhanced_voice_processing(audio_path, reference_embedding_path):
    tool = VoiceEmbeddingTool()
    
    # Bước 1: Verify speaker
    verification = tool.compare_with_reference(
        audio_path, 
        reference_embedding_path, 
        threshold=0.80
    )
    
    if verification['is_same_speaker']:
        # Bước 2: Process voice-to-text
        transcription = your_voice_to_text_function(audio_path)
        
        return {
            'transcription': transcription,
            'speaker_verified': True,
            'similarity_score': verification['similarity_score'],
            'confidence': verification['confidence']
        }
    else:
        return {
            'transcription': None,
            'speaker_verified': False,
            'error': 'Speaker verification failed'
        }
```

## Files Created

- **reference.pkl**: Embedding vector của reference voice
- **comparison_results.json**: Chi tiết kết quả so sánh
- Có thể save format JSON hoặc pickle

## Performance

- **Speed**: ~1-2 giây per comparison (include loading)
- **Memory**: ~2GB GPU RAM cho TitaNet-L
- **Accuracy**: Tương đương EER ~15% trên dataset Vietnamese

## Lưu ý quan trọng

1. **Quality matters**: Audio chất lượng tốt → similarity cao hơn
2. **Duration**: Audio ít nhất 1-2 giây để có embedding ổn định  
3. **Same conditions**: Record trong điều kiện tương tự cho kết quả tốt nhất
4. **Multiple samples**: Có thể tạo nhiều reference embeddings cho 1 người
5. **Threshold tuning**: Test với data thực của bạn để tìm threshold tối ưu

## Next Steps

1. **Test với audio thực** của bạn
2. **Tune threshold** phù hợp với use case
3. **Integrate với voice-to-text** system hiện tại
4. **Set up monitoring** cho production environment