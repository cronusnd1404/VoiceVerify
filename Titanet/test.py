import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

# Load pre-trained VAD and speaker embedding models
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="/home/edabk408/NgocDat/Titanet/titanet-s.nemo")

# Load your audio file
audio_file_path = "M_1017_11y8m_1.wav" # Replace with your audio file path
audio, sample_rate = sf.read(audio_file_path)

# Ensure audio is mono and at the correct sample rate (usually 16000 Hz for most models)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
if sample_rate != 16000:
    # Use your preferred resampling method here. For simplicity, we'll assume the audio is already at 16000 Hz.
    print("Warning: Audio is not at 16000 Hz. Resampling is required for optimal performance.")
    # You might want to use a library like librosa for resampling:
    import librosa
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

# Perform VAD to find speech segments
vad_segments = vad_model.vad_infer(audio, sample_rate)

# For each speech segment, extract speaker embedding
speaker_embeddings = []
for start, end in vad_segments:
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = audio[start_sample:end_sample]
    
    # Extract embedding
    embedding = speaker_model.extract_speaker_embeddings(segment)
    speaker_embeddings.append(embedding)

# Perform speaker clustering (e.g., using a simple KMeans or Agglomerative Clustering)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Let's use Agglomerative Clustering for simplicity
# You might need to experiment with the number of clusters (n_clusters)
num_speakers = 2
clustering = AgglomerativeClustering(n_clusters=num_speakers, affinity='cosine', linkage='average')
labels = clustering.fit_predict(np.vstack(speaker_embeddings))

# Map labels to speakers and print the results
speaker_map = {}
for i, (start, end) in enumerate(vad_segments):
    label = labels[i]
    if label not in speaker_map:
        speaker_map[label] = f"Người {len(speaker_map) + 1}"
    
    print(f"{speaker_map[label]} nói từ {start:.2f}s đến {end:.2f}s")