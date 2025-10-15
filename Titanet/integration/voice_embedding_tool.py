"""Simple tool for voice embedding and comparison using TitaNet-L and Silero VAD v6"""
import json
import numpy as np
import sys
from pathlib import Path
import pickle
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from speaker_verification_pipeline import create_pipeline

class VoiceEmbeddingTool:
    """Simple tool for voice embedding and comparison"""
    
    def __init__(self, config_path="production_config.json"):
        """Initialize the embedding tool"""
        print("Initializing voice embedding tool...")
        self.pipeline = create_pipeline(config_path)
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

def demo_usage():
    """Demonstrate the voice embedding tool"""
    
    # Initialize tool
    tool = VoiceEmbeddingTool()
    
    # Check if we have test audio
    test_audio_dir = Path("/home/edabk408/NgocDat/Titanet/dataset/test")
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
    print("4. quit - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
            
            if command[0] == "quit":
                break
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
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo_usage()