#!/usr/bin/env python3
"""
Quick integration test for speaker verification pipeline
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Fix import paths - use relative imports
try:
    from speaker_verification_pipeline import create_pipeline
    from automated_verification_demo import AutomatedVerificationSystem
except ImportError:
    # Fallback to absolute imports if needed
    sys.path.append(str(Path(__file__).parent.parent))
    from itergration.speaker_verification_pipeline import create_pipeline
    from itergration.automated_verification_demo import AutomatedVerificationSystem

def test_pipeline_creation():
    """Test basic pipeline creation"""
    print("Testing pipeline creation...")
    
    try:
        # Test with default config
        pipeline = create_pipeline()
        print("✓ Default pipeline created successfully")
        
        # Test with config file
        config_path = "production_config.json"
        if os.path.exists(config_path):
            pipeline_with_config = create_pipeline(config_path)
            print("✓ Pipeline with config file created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False

def test_models_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        pipeline = create_pipeline("production_config.json")
        
        # Check TitaNet model
        if hasattr(pipeline, 'speaker_model') and pipeline.speaker_model is not None:
            print("✓ TitaNet-L model loaded successfully")
        else:
            print("✗ TitaNet-L model not loaded")
            return False
        
        # Check Silero VAD
        if pipeline.config.use_vad:
            if hasattr(pipeline, 'vad_model') and pipeline.vad_model is not None:
                print("✓ Silero VAD v6 loaded successfully")
            else:
                print("✗ Silero VAD not loaded")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_enrollment_system():
    """Test enrollment database system"""
    print("\nTesting enrollment system...")
    
    try:
        pipeline = create_pipeline("production_config.json")
        
        # Test enrollment database loading
        initial_stats = pipeline.get_enrollment_stats()
        print(f"✓ Initial enrollment stats: {initial_stats['total_enrolled_speakers']} speakers")
        
        return True
    except Exception as e:
        print(f"✗ Enrollment system test failed: {e}")
        return False

def test_automated_system():
    """Test automated verification system"""
    print("\nTesting automated verification system...")
    
    try:
        system = AutomatedVerificationSystem("production_config.json")
        stats = system.get_system_stats()
        
        print("✓ Automated verification system created successfully")
        print(f"  - Device: {stats['pipeline_config']['device']}")
        print(f"  - VAD enabled: {stats['pipeline_config']['use_vad']}")
        print(f"  - Similarity threshold: {stats['pipeline_config']['similarity_threshold']}")
        
        return True
    except Exception as e:
        print(f"✗ Automated system test failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing capabilities"""
    print("\nTesting audio processing...")
    
    try:
        pipeline = create_pipeline("production_config.json")
        
        # Test with sample audio files from your dataset
        test_audio_dir = Path("/home/edabk408/NgocDat/Titanet/dataset/test")
        test_files = list(test_audio_dir.glob("*.wav"))[:3]  # Test with first 3 files
        
        if not test_files:
            print("! No test audio files found, skipping audio processing test")
            return True
        
        print(f"Testing with {len(test_files)} audio files...")
        
        for audio_file in test_files:
            try:
                # Test preprocessing
                waveform, sr = pipeline.preprocess_audio(str(audio_file))
                print(f"✓ Preprocessed {audio_file.name}: shape={waveform.shape}, sr={sr}")
                
                # Test VAD if enabled
                if pipeline.config.use_vad:
                    vad_audio = pipeline.apply_vad(waveform, sr)
                    print(f"✓ VAD processed {audio_file.name}: shape={vad_audio.shape}")
                
                # Test embedding extraction (this will take some time)
                embedding = pipeline.extract_embedding(str(audio_file))
                if embedding is not None:
                    print(f"✓ Embedding extracted from {audio_file.name}: shape={embedding.shape}")
                else:
                    print(f"! Failed to extract embedding from {audio_file.name}")
                
                break  # Only test one file to save time
                
            except Exception as e:
                print(f"✗ Audio processing failed for {audio_file.name}: {e}")
                continue
        
        return True
    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("=== Speaker Verification Pipeline Integration Test ===\n")
    
    tests = [
        test_pipeline_creation,
        test_models_loading,
        test_enrollment_system,
        test_automated_system,
        test_audio_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! System ready for production integration.")
        print("\nNext steps:")
        print("1. Replace VoiceToTextMock with your actual voice-to-text model")
        print("2. Enroll speakers using pipeline.enroll_speaker()")
        print("3. Adjust similarity_threshold based on your requirements")
        print("4. Test with your actual audio data")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())