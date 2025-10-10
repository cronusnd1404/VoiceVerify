#!/usr/bin/env python3
"""
Simple test script showing TitaNet-L + Silero VAD v6 integration
Run this to test the complete pipeline with your dataset
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from speaker_verification_pipeline import create_pipeline

def test_integration():
    """Test the complete integration pipeline"""
    
    print("=== TitaNet-L + Silero VAD v6 Integration Test ===\n")
    
    # 1. Create pipeline
    print("1. Creating speaker verification pipeline...")
    try:
        pipeline = create_pipeline("production_config.json")
        print("✓ Pipeline created successfully")
        print(f"   - Device: {pipeline.config.device}")
        print(f"   - VAD enabled: {pipeline.config.use_vad}")
        print(f"   - Similarity threshold: {pipeline.config.similarity_threshold}")
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False
    
    # 2. Test with sample audio from your dataset
    test_audio_dir = Path("/home/edabk408/NgocDat/Titanet/dataset/test")
    if not test_audio_dir.exists():
        print("! Test audio directory not found, skipping audio tests")
        return True
    
    audio_files = list(test_audio_dir.glob("*.wav"))[:6]  # Use first 6 files
    if not audio_files:
        print("! No audio files found in test directory")
        return True
    
    print(f"\n2. Testing with {len(audio_files)} audio files from dataset...")
    
    # 3. Enroll speakers (simulate 3 speakers with 2 samples each)
    print("\n3. Enrolling speakers...")
    speakers = {}
    for i in range(0, min(6, len(audio_files)), 2):
        speaker_id = f"test_speaker_{i//2 + 1}"
        audio_samples = [str(audio_files[i])]
        if i+1 < len(audio_files):
            audio_samples.append(str(audio_files[i+1]))
        
        speakers[speaker_id] = audio_samples
        
        print(f"   Enrolling {speaker_id} with {len(audio_samples)} samples...")
        success = pipeline.enroll_speaker(speaker_id, audio_samples)
        
        if success:
            print(f"   ✓ {speaker_id} enrolled successfully")
        else:
            print(f"   ✗ {speaker_id} enrollment failed")
    
    # 4. Test verification
    print("\n4. Testing speaker verification...")
    
    if audio_files and speakers:
        test_audio = str(audio_files[0])
        expected_speaker = list(speakers.keys())[0]
        
        print(f"   Testing audio: {Path(test_audio).name}")
        print(f"   Expected speaker: {expected_speaker}")
        
        # Verification against specific speaker
        result = pipeline.verify_speaker(test_audio, expected_speaker)
        
        if result.get('success', False):
            verified = result.get('verified', False)
            speaker_data = result.get('speakers', {}).get(expected_speaker, {})
            similarity = speaker_data.get('max_similarity', 0.0)
            
            print(f"   Result: {'✓ VERIFIED' if verified else '✗ NOT VERIFIED'}")
            print(f"   Similarity score: {similarity:.3f}")
            print(f"   Threshold: {speaker_data.get('threshold_used', 0.0):.3f}")
        else:
            print(f"   ✗ Verification failed: {result.get('error', 'Unknown error')}")
        
        # Open identification (find best match)
        print("\n   Testing open identification...")
        result = pipeline.verify_speaker(test_audio)  # No claimed speaker
        
        if result.get('success', False) and 'best_match' in result:
            best_match = result['best_match']
            print(f"   Best match: {best_match.get('speaker_id', 'Unknown')}")
            print(f"   Similarity: {best_match.get('similarity', 0.0):.3f}")
            print(f"   Verified: {'✓' if best_match.get('verified', False) else '✗'}")
    
    # 5. Show enrollment statistics
    print("\n5. Final enrollment statistics:")
    stats = pipeline.get_enrollment_stats()
    print(f"   Total enrolled speakers: {stats['total_enrolled_speakers']}")
    print(f"   Total enrollment samples: {stats['total_enrollment_samples']}")
    
    for speaker_id, data in stats['speakers'].items():
        print(f"   - {speaker_id}: {data['num_samples']} samples")
    
    print("\n=== Integration Test Complete ===")
    print("\nNext steps for production:")
    print("1. Replace the enrollment demo with your actual speakers")
    print("2. Integrate with your voice-to-text model")
    print("3. Adjust similarity_threshold based on your requirements")
    print("4. Set up proper error handling and logging")
    
    return True

def show_integration_example():
    """Show code example for voice-to-text integration"""
    
    example_code = '''
# Example: Integration with your voice-to-text system
from speaker_verification_pipeline import create_pipeline

# Initialize once at startup
speaker_pipeline = create_pipeline("production_config.json")

# Your existing voice-to-text function
def your_voice_to_text(audio_path):
    # Your existing implementation
    return transcribed_text

# Enhanced function with speaker verification
def enhanced_voice_to_text(audio_path, expected_speaker=None):
    # Step 1: Verify speaker first
    verification_result = speaker_pipeline.verify_speaker(audio_path, expected_speaker)
    
    # Step 2: Only transcribe if speaker is verified (optional)
    if verification_result.get('verified', False) or expected_speaker is None:
        transcription = your_voice_to_text(audio_path)
        
        return {
            'transcription': transcription,
            'speaker_verified': verification_result.get('verified', False),
            'speaker_id': expected_speaker or verification_result.get('best_match', {}).get('speaker_id'),
            'similarity_score': verification_result.get('speakers', {}).get(expected_speaker, {}).get('max_similarity', 0.0)
        }
    else:
        return {
            'transcription': None,
            'error': 'Speaker verification failed',
            'speaker_verified': False
        }

# Usage
result = enhanced_voice_to_text("input_audio.wav", "user001")
print(f"Transcription: {result['transcription']}")
print(f"Speaker verified: {result['speaker_verified']}")
'''
    
    print("=== Voice-to-Text Integration Example ===")
    print(example_code)

if __name__ == "__main__":
    if test_integration():
        print("\n" + "="*50)
        show_integration_example()
    else:
        print("Integration test failed. Please check the errors above.")
        sys.exit(1)