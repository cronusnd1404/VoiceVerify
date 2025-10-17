"""Simple tool for voice embedding and comparison using TitaNet-L and Silero VAD v6"""
import json
import numpy as np
import sys
from pathlib import Path
import pickle
from datetime import datetime
import os
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from speaker_verification_pipeline import create_pipeline

class MicrophoneRecorder:
    """Ghi âm trực tiếp từ microphone"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.setup_recording()
    
    def setup_recording(self):
        """Thiết lập ghi âm"""
        try:
            import sounddevice as sd
            self.sd = sd
            self.recording_available = True
            print("✓ Microphone recording available")
        except ImportError:
            try:
                import pyaudio
                import wave
                self.pyaudio = pyaudio
                self.wave = wave
                self.recording_available = True
                self.use_pyaudio = True
                print("✓ Microphone recording available (using PyAudio)")
            except ImportError:
                self.recording_available = False
                print("⚠️ No audio recording libraries found. Install sounddevice or pyaudio.")
    
    def record_audio(self, duration=5, save_path=None):
        """
        Ghi âm từ microphone
        duration: thời gian ghi âm (giây)
        save_path: đường dẫn lưu file (tự động tạo nếu None)
        """
        if not self.recording_available:
            return {'success': False, 'error': 'Recording not available'}
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"/home/edabk408/Titanet/integration/temp/recorded_voice_{timestamp}.wav"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            if hasattr(self, 'use_pyaudio'):
                return self._record_with_pyaudio(duration, save_path)
            else:
                return self._record_with_sounddevice(duration, save_path)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _record_with_sounddevice(self, duration, save_path):
        """Ghi âm bằng sounddevice"""
        print(f"🎤 Bắt đầu ghi âm {duration} giây...")
        print("Nói vào microphone...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("🔴 Đang ghi âm!")
        
        # Record
        recording = self.sd.rec(int(duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, dtype='float32')
        self.sd.wait()  # Wait until recording is finished
        
        # Save
        import scipy.io.wavfile as wavfile
        # Convert float32 to int16
        recording_int16 = (recording * 32767).astype(np.int16)
        wavfile.write(save_path, self.sample_rate, recording_int16.flatten())
        
        print(f"✓ Ghi âm hoàn tất: {save_path}")
        return {'success': True, 'file_path': save_path, 'duration': duration}
    
    def _record_with_pyaudio(self, duration, save_path):
        """Ghi âm bằng pyaudio"""
        CHUNK = 1024
        FORMAT = self.pyaudio.paInt16
        CHANNELS = 1
        RATE = self.sample_rate
        
        p = self.pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        print(f"🎤 Bắt đầu ghi âm {duration} giây...")
        print("Nói vào microphone...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("🔴 Đang ghi âm!")
        
        frames = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("✓ Ghi âm hoàn tất")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save
        wf = self.wave.open(save_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"✓ File đã lưu: {save_path}")
        return {'success': True, 'file_path': save_path, 'duration': duration}
    
    def test_microphone(self):
        """Test microphone hoạt động"""
        if not self.recording_available:
            print("❌ Microphone không khả dụng")
            return False
        
        print("🎤 Test microphone...")
        result = self.record_audio(duration=2, save_path="/tmp/mic_test.wav")
        
        if result['success']:
            print("✅ Microphone hoạt động bình thường")
            # Cleanup test file
            try:
                os.remove("/tmp/mic_test.wav")
            except:
                pass
            return True
        else:
            print(f"❌ Lỗi microphone: {result['error']}")
            return False

class VoiceEmbeddingTool:
    """Simple tool for voice embedding and comparison"""
    
    def __init__(self, config_path="production_config.json"):
        """Initialize the embedding tool"""
        print("Initializing voice embedding tool...")
        self.pipeline = create_pipeline(config_path)
        self.mic_recorder = MicrophoneRecorder()
        print("✓ Pipeline ready")
    
    def extract_and_save_embedding(self, audio_path: str, save_path: str = None) -> dict:
        """
        Extract embedding from audio and save it
        Returns: dict with embedding info
        """
        print(f"Extracting embedding from: {audio_path}")
        
        # Extract embedding
        embedding = self.pipeline.extract_embedding(audio_path)
        
        if embedding is None:
            return {
                'success': False,
                'error': 'Failed to extract embedding'
            }
        
        # Prepare embedding data
        embedding_data = {
            'embedding': embedding.tolist(),  # Convert numpy to list for JSON
            'audio_path': audio_path,
            'extracted_at': datetime.now().isoformat(),
            'embedding_shape': embedding.shape,
            'embedding_norm': float(np.linalg.norm(embedding))
        }
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.json'):
                with open(save_path, 'w') as f:
                    json.dump(embedding_data, f, indent=2)
            else:  # Save as pickle for numpy compatibility
                with open(save_path, 'wb') as f:
                    pickle.dump(embedding_data, f)
            
            print(f"✓ Embedding saved to: {save_path}")
        
        print(f"✓ Embedding extracted: shape={embedding.shape}, norm={embedding_data['embedding_norm']:.3f}")
        
        return {
            'success': True,
            'embedding_data': embedding_data
        }
    
    def load_reference_embedding(self, embedding_path: str) -> dict:
        """Load reference embedding from file"""
        try:
            if embedding_path.endswith('.json'):
                with open(embedding_path, 'r') as f:
                    data = json.load(f)
                    # Convert list back to numpy array
                    data['embedding'] = np.array(data['embedding'])
            else:  # pickle file
                with open(embedding_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data['embedding'], list):
                        data['embedding'] = np.array(data['embedding'])
            
            print(f"✓ Reference embedding loaded from: {embedding_path}")
            return {'success': True, 'data': data}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def compare_with_reference(self, test_audio_path: str, reference_embedding_path: str, 
                             threshold: float = 0.65) -> dict:
        """
        Compare test audio with reference embedding
        """
        print(f"\nComparing: {test_audio_path}")
        print(f"Reference: {reference_embedding_path}")
        
        # Load reference embedding
        ref_result = self.load_reference_embedding(reference_embedding_path)
        if not ref_result['success']:
            return {
                'success': False,
                'error': f"Failed to load reference: {ref_result['error']}"
            }
        
        reference_embedding = ref_result['data']['embedding']
        
        # Extract test embedding
        test_embedding = self.pipeline.extract_embedding(test_audio_path)
        if test_embedding is None:
            return {
                'success': False,
                'error': 'Failed to extract test embedding'
            }
        
        # Calculate cosine similarity
        similarity = self.pipeline.cosine_similarity(test_embedding, reference_embedding)
        
        # Make decision
        is_same_speaker = similarity >= threshold
        
        result = {
            'success': True,
            'test_audio': test_audio_path,
            'reference_info': {
                'path': reference_embedding_path,
                'original_audio': ref_result['data'].get('audio_path', 'Unknown'),
                'extracted_at': ref_result['data'].get('extracted_at', 'Unknown')
            },
            'similarity_score': float(similarity),
            'threshold': threshold,
            'is_same_speaker': is_same_speaker,
            'confidence': min(similarity / threshold, 1.0) if is_same_speaker else 0.0,
            'decision': 'SAME SPEAKER' if is_same_speaker else 'DIFFERENT SPEAKER'
        }
        
        # Print results
        print(f"Similarity score: {similarity:.4f}")
        print(f"Threshold: {threshold}")
        print(f"Decision: {result['decision']} ({'✓' if is_same_speaker else '✗'})")
        print(f"Confidence: {result['confidence']:.2%}")
        
        return result
    
    def batch_compare(self, test_audio_files: list, reference_embedding_path: str, 
                     threshold: float = 0.65) -> list:
        """Compare multiple test files against reference"""
        results = []
        
        print(f"\nBatch comparing {len(test_audio_files)} files against reference...")
        
        for i, audio_file in enumerate(test_audio_files, 1):
            print(f"\n[{i}/{len(test_audio_files)}] Processing: {Path(audio_file).name}")
            result = self.compare_with_reference(audio_file, reference_embedding_path, threshold)
            results.append(result)
        
        # Summary
        if results:
            successful = [r for r in results if r.get('success', False)]
            same_speaker = [r for r in successful if r.get('is_same_speaker', False)]
            
            print(f"\n=== Batch Results Summary ===")
            print(f"Total files: {len(test_audio_files)}")
            print(f"Successfully processed: {len(successful)}")
            print(f"Same speaker: {len(same_speaker)}")
            print(f"Different speaker: {len(successful) - len(same_speaker)}")
            
            if successful:
                similarities = [r['similarity_score'] for r in successful]
                print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
                print(f"Average similarity: {np.mean(similarities):.3f}")
        
        return results
    
    def record_and_extract_embedding(self, duration=5, save_audio=True, save_embedding=True):
        """
        Ghi âm trực tiếp và trích xuất embedding
        """
        print(f"\n=== Ghi âm và trích xuất embedding ===")
        
        # Record audio
        record_result = self.mic_recorder.record_audio(duration=duration)
        if not record_result['success']:
            return {'success': False, 'error': f"Recording failed: {record_result['error']}"}
        
        audio_path = record_result['file_path']
        
        # Extract embedding
        print("🧠 Đang trích xuất embedding...")
        embedding_result = self.extract_and_save_embedding(
            audio_path, 
            save_path=audio_path.replace('.wav', '_embedding.pkl') if save_embedding else None
        )
        
        if not embedding_result['success']:
            return {'success': False, 'error': f"Embedding extraction failed: {embedding_result['error']}"}
        
        # Cleanup audio file if not saving
        if not save_audio:
            try:
                os.remove(audio_path)
                print("🗑️ Audio file cleaned up")
            except:
                pass
        
        result = {
            'success': True,
            'audio_path': audio_path if save_audio else None,
            'embedding_data': embedding_result['embedding_data'],
            'duration': duration
        }
        
        print(f"✅ Hoàn tất! Embedding shape: {embedding_result['embedding_data']['embedding_shape']}")
        return result
    
    def record_and_compare_live(self, reference_embedding_path, duration=5, threshold=0.65):
        """
        Ghi âm trực tiếp và so sánh với reference ngay lập tức
        """
        print(f"\n=== So sánh giọng nói trực tiếp ===")
        print(f"Reference: {reference_embedding_path}")
        
        # Record audio
        record_result = self.mic_recorder.record_audio(duration=duration)
        if not record_result['success']:
            return {'success': False, 'error': f"Recording failed: {record_result['error']}"}
        
        # Compare immediately
        comparison_result = self.compare_with_reference(
            record_result['file_path'], 
            reference_embedding_path, 
            threshold
        )
        
        # Cleanup temporary audio file
        try:
            os.remove(record_result['file_path'])
        except:
            pass
        
        return comparison_result

def demo_usage():
    """Demonstrate the voice embedding tool"""
    
    # Initialize tool
    tool = VoiceEmbeddingTool()
    
    # Check if we have test audio
    test_audio_dir = Path("/Home/edabk408/NgocDat/Titanet/integration/dataset/test")
    if not test_audio_dir.exists():
        print("! Test audio directory not found")
        return
    
    audio_files = list(test_audio_dir.glob("*.wav"))[:5]  # Use first 5 files
    if len(audio_files) < 2:
        print("! Need at least 2 audio files for demo")
        return
    
    reference_audio = str(audio_files[0])
    test_audios = [str(f) for f in audio_files[1:]]
    
    print("=== Voice Embedding & Comparison Demo ===")
    
    # Step 1: Extract and save reference embedding
    print(f"\n1. Creating reference embedding from: {Path(reference_audio).name}")
    reference_embedding_path = "reference_voice.pkl"
    
    result = tool.extract_and_save_embedding(reference_audio, reference_embedding_path)
    if not result['success']:
        print(f"✗ Failed: {result['error']}")
        return
    
    # Step 2: Compare other voices against reference
    print(f"\n2. Comparing {len(test_audios)} test voices against reference...")
    
    for test_audio in test_audios:
        print(f"\n--- Testing: {Path(test_audio).name} ---")
        comparison_result = tool.compare_with_reference(
            test_audio, 
            reference_embedding_path,
            threshold=0.65  # Adjust based on your needs
        )
        
        if comparison_result['success']:
            print(f"Result: {comparison_result['decision']}")
        else:
            print(f"✗ Error: {comparison_result['error']}")
    
    # Step 3: Batch comparison
    print(f"\n3. Batch comparison summary...")
    batch_results = tool.batch_compare(test_audios, reference_embedding_path)
    
    # Save detailed results
    results_file = "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    print(f"\n✓ Detailed results saved to: {results_file}")

def interactive_mode():
    """Interactive mode for manual testing"""
    tool = VoiceEmbeddingTool()
    
    print("=== Interactive Voice Comparison Tool ===")
    print("Commands:")
    print("1. extract <audio_file> [save_path] - Extract embedding")
    print("2. compare <test_audio> <reference_embedding> [threshold] - Compare voices")
    print("3. batch <reference_embedding> <audio_dir> [threshold] - Batch compare")
    print("4. record [duration] [save_audio] [save_embedding] - Record and extract embedding")
    print("5. record_compare <reference_embedding> [duration] [threshold] - Record and compare live")
    print("6. test_mic - Test microphone")
    print("7. enroll - Interactive enrollment (record reference voice)")
    print("8. verify - Interactive verification (record and compare)")
    print("9. quit - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
            
            if command[0] == "quit":
                break
                
            elif command[0] == "test_mic":
                tool.mic_recorder.test_microphone()
                
            elif command[0] == "record":
                duration = int(command[1]) if len(command) > 1 else 5
                save_audio = command[2].lower() == 'true' if len(command) > 2 else True
                save_embedding = command[3].lower() == 'true' if len(command) > 3 else True
                
                result = tool.record_and_extract_embedding(duration, save_audio, save_embedding)
                if not result['success']:
                    print(f"✗ Error: {result['error']}")
                    
            elif command[0] == "record_compare":
                if len(command) < 2:
                    print("Usage: record_compare <reference_embedding> [duration] [threshold]")
                    continue
                
                reference_emb = command[1]
                duration = int(command[2]) if len(command) > 2 else 5
                threshold = float(command[3]) if len(command) > 3 else 0.65
                
                result = tool.record_and_compare_live(reference_emb, duration, threshold)
                if not result['success']:
                    print(f"✗ Error: {result['error']}")
                    
            elif command[0] == "enroll":
                print("\n=== Enrollment Mode ===")
                print("This will record your voice as reference")
                speaker_name = input("Enter speaker name: ").strip()
                if not speaker_name:
                    print("Speaker name required")
                    continue
                
                duration = 10  # Longer for better reference
                print(f"Recording reference voice for '{speaker_name}' ({duration}s)")
                
                result = tool.record_and_extract_embedding(duration, save_audio=True, save_embedding=True)
                if result['success']:
                    # Rename files with speaker name
                    audio_path = result['audio_path']
                    if audio_path:
                        new_audio_path = f"/home/edabk408/Titanet/integration/data/{speaker_name}_reference.wav"
                        new_embedding_path = f"/home/edabk408/Titanet/integration/data/{speaker_name}_reference.pkl"

                        os.makedirs(os.path.dirname(new_audio_path), exist_ok=True)
                        
                        try:
                            os.rename(audio_path, new_audio_path)
                            os.rename(audio_path.replace('.wav', '_embedding.pkl'), new_embedding_path)
                            print(f"✅ Reference enrolled for '{speaker_name}'")
                            print(f"   Audio: {new_audio_path}")
                            print(f"   Embedding: {new_embedding_path}")
                        except Exception as e:
                            print(f"✗ Error saving: {e}")
                else:
                    print(f"✗ Enrollment failed: {result['error']}")
                    
            elif command[0] == "verify":
                print("\n=== Verification Mode ===")
                
                # List available references
                data_dir = Path("/home/edabk408/Titanet/integration/data")
                if data_dir.exists():
                    reference_files = list(data_dir.glob("*_reference.pkl"))
                    if reference_files:
                        print("Available references:")
                        for i, ref_file in enumerate(reference_files, 1):
                            speaker_name = ref_file.stem.replace('_reference', '')
                            print(f"  {i}. {speaker_name}")
                        
                        try:
                            choice = input("Choose reference (number or speaker name): ").strip()
                            
                            if choice.isdigit():
                                choice_idx = int(choice) - 1
                                if 0 <= choice_idx < len(reference_files):
                                    reference_path = str(reference_files[choice_idx])
                                    speaker_name = reference_files[choice_idx].stem.replace('_reference', '')
                                else:
                                    print("Invalid choice")
                                    continue
                            else:
                                reference_path = str(data_dir / f"{choice}_reference.pkl")
                                speaker_name = choice
                                if not Path(reference_path).exists():
                                    print(f"Reference not found for '{choice}'")
                                    continue
                            
                            duration = 5
                            threshold = 0.65
                            
                            print(f"Verifying against '{speaker_name}' ({duration}s)")
                            result = tool.record_and_compare_live(reference_path, duration, threshold)
                            
                            if result['success']:
                                print(f"\n🎯 VERIFICATION RESULT:")
                                print(f"   Speaker: {speaker_name}")
                                print(f"   Decision: {result['decision']}")
                                print(f"   Confidence: {result['confidence']:.1%}")
                                if result['is_same_speaker']:
                                    print("   ✅ VERIFIED - Same speaker")
                                else:
                                    print("   ❌ REJECTED - Different speaker")
                            else:
                                print(f"✗ Verification failed: {result['error']}")
                                
                        except ValueError:
                            print("Invalid input")
                    else:
                        print("No references found. Use 'enroll' first.")
                else:
                    print("No data directory found. Use 'enroll' first.")
            
            elif command[0] == "extract":
                if len(command) < 2:
                    print("Usage: extract <audio_file> [save_path]")
                    continue
                
                audio_file = command[1] 
                save_path = command[2] if len(command) > 2 else f"embedding_{Path(audio_file).stem}.pkl"
                
                result = tool.extract_and_save_embedding(audio_file, save_path)
                if not result['success']:
                    print(f"✗ Error: {result['error']}")
                
            elif command[0] == "compare":
                if len(command) < 3:
                    print("Usage: compare <test_audio> <reference_embedding> [threshold]")
                    continue
                
                test_audio = command[1]
                reference_emb = command[2]
                threshold = float(command[3]) if len(command) > 3 else 0.65
                
                result = tool.compare_with_reference(test_audio, reference_emb, threshold)
                if not result['success']:
                    print(f"✗ Error: {result['error']}")
                
            elif command[0] == "batch":
                if len(command) < 3:
                    print("Usage: batch <reference_embedding> <audio_dir> [threshold]")
                    continue
                
                reference_emb = command[1]
                audio_dir = Path(command[2])
                threshold = float(command[3]) if len(command) > 3 else 0.65
                
                if not audio_dir.exists():
                    print(f"✗ Directory not found: {audio_dir}")
                    continue
                
                audio_files = list(audio_dir.glob("*.wav"))
                if not audio_files:
                    print("✗ No WAV files found in directory")
                    continue
                
                tool.batch_compare([str(f) for f in audio_files], reference_emb, threshold)
                
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_mode()
        elif sys.argv[1] == "enroll":
            # Quick enrollment mode
            tool = VoiceEmbeddingTool()
            speaker_name = input("Enter speaker name for enrollment: ").strip()
            if speaker_name:
                print(f"Recording 10 seconds for '{speaker_name}'...")
                result = tool.record_and_extract_embedding(duration=10, save_audio=True, save_embedding=True)
                if result['success']:
                    print(f"✅ Enrollment completed for '{speaker_name}'")
        elif sys.argv[1] == "verify":
            # Quick verification mode  
            tool = VoiceEmbeddingTool()
            if len(sys.argv) > 2:
                reference_path = sys.argv[2]
                print(f"Recording 5 seconds for verification...")
                result = tool.record_and_compare_live(reference_path, duration=5)
                if result['success']:
                    print(f"Result: {result['decision']} (confidence: {result['confidence']:.1%})")
            else:
                print("Usage: python voice_embedding_tool.py verify <reference_embedding_path>")
        elif sys.argv[1] == "test_mic":
            # Test microphone
            recorder = MicrophoneRecorder()
            recorder.test_microphone()
        else:
            print("Available modes: interactive, enroll, verify, test_mic")
    else:
        demo_usage()