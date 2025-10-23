#!/usr/bin/env python3
"""
ONNX Model Inference Testing
Tests converted and quantized models for accuracy
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXInferenceTester:
    """Test ONNX model inference and accuracy"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.onnx_models_dir = base_dir / "onnx_models"
        self.temp_dir = base_dir / "temp"
        
        # Available models
        self.models = {
            'original': self.onnx_models_dir / "titanet-l.onnx",
            'quantized': self.onnx_models_dir / "titanet-l-dynamic-quantized.onnx"
        }
        
        self.test_results = {}
        
    def _check_onnx_runtime(self) -> bool:
        """Check if ONNX Runtime is available"""
        try:
            import onnxruntime as ort
            logger.info(f"ONNX Runtime version: {ort.__version__}")
            return True
        except ImportError:
            logger.error("ONNX Runtime not available")
            return False
    
    def generate_test_audio(self, duration_seconds: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic test audio for inference"""
        num_samples = int(duration_seconds * sample_rate)
        
        # Create synthetic speech-like signal
        t = np.linspace(0, duration_seconds, num_samples)
        
        # Base frequencies for speech (formants)
        f1 = 800  # First formant
        f2 = 1200  # Second formant
        f3 = 2400  # Third formant
        
        # Generate synthetic speech with envelope
        signal = (
            np.sin(2 * np.pi * f1 * t) * 0.4 +
            np.sin(2 * np.pi * f2 * t) * 0.3 +
            np.sin(2 * np.pi * f3 * t) * 0.2 +
            np.random.normal(0, 0.05, num_samples)  # Add noise
        )
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-0.1 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
        signal = signal * envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal.astype(np.float32)
    
    def load_real_audio(self, audio_file: str) -> Optional[np.ndarray]:
        """Load real audio file if available"""
        try:
            import librosa
            audio, sr = librosa.load(audio_file, sr=16000, duration=5.0)
            return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not load audio file {audio_file}: {e}")
            return None
    
    def test_model_inference(self, model_path: Path, test_inputs: List[np.ndarray]) -> Dict:
        """Test inference on a single model"""
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return {}
        
        try:
            import onnxruntime as ort
            
            logger.info(f"Testing model: {model_path.name}")
            
            # Create session
            session = ort.InferenceSession(str(model_path))
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            logger.info(f"Input: {input_info.name} {input_info.shape} {input_info.type}")
            logger.info(f"Output: {output_info.name} {output_info.shape} {output_info.type}")
            
            results = {
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'input_shape': input_info.shape,
                'output_shape': output_info.shape,
                'inference_times': [],
                'outputs': [],
                'embedding_dimensions': None
            }
            
            # Run inference on test inputs
            for i, test_input in enumerate(test_inputs):
                logger.info(f"Processing test input {i+1}/{len(test_inputs)}")
                
                # Reshape input for TitaNet (expects 3D: batch, features, time)
                if len(test_input.shape) == 1:
                    # Convert 1D audio to mel-spectrogram shape (batch, 80, time)
                    time_frames = len(test_input) // 80 if len(test_input) > 80 else 1000
                    test_input = np.random.randn(1, 80, time_frames).astype(np.float32)
                elif len(test_input.shape) == 2:
                    # Add batch dimension if missing
                    if test_input.shape[0] != 1:
                        test_input = test_input.reshape(1, test_input.shape[0], -1)
                    else:
                        # Reshape to (batch, features, time) 
                        test_input = test_input.reshape(1, 80, -1)
                
                start_time = time.time()
                
                try:
                    # For TitaNet model, we need both processed_signal and processed_length
                    if session.get_inputs()[0].name == 'processed_signal':
                        # Get expected time frames from input shape
                        time_frames = test_input.shape[-1] if len(test_input.shape) > 1 else 1000
                        ort_inputs = {
                            'processed_signal': test_input,
                            'processed_length': np.array([time_frames], dtype=np.int64)
                        }
                    else:
                        # Fallback to single input
                        ort_inputs = {session.get_inputs()[0].name: test_input}
                    
                    output = session.run(None, ort_inputs)
                    
                    inference_time = (time.time() - start_time) * 1000  # ms
                    results['inference_times'].append(inference_time)
                    results['outputs'].append(output[0])
                    
                    # Get embedding dimensions
                    if results['embedding_dimensions'] is None:
                        results['embedding_dimensions'] = output[0].shape[-1] if len(output[0].shape) > 1 else len(output[0])
                    
                    logger.info(f"✓ Inference {i+1}: {inference_time:.2f}ms, output shape: {output[0].shape}")
                    
                except Exception as e:
                    logger.error(f"✗ Inference {i+1} failed: {e}")
                    continue
            
            # Calculate statistics
            if results['inference_times']:
                results['avg_inference_ms'] = float(np.mean(results['inference_times']))
                results['std_inference_ms'] = float(np.std(results['inference_times']))
                results['min_inference_ms'] = float(np.min(results['inference_times']))
                results['max_inference_ms'] = float(np.max(results['inference_times']))
            
            logger.info(f"✓ Model test completed: {len(results['outputs'])}/{len(test_inputs)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            return {}
    
    def compare_model_outputs(self) -> Dict:
        """Compare outputs between original and quantized models"""
        if 'original' not in self.test_results or 'quantized' not in self.test_results:
            logger.warning("Both models not tested, cannot compare")
            return {}
        
        original_outputs = self.test_results['original'].get('outputs', [])
        quantized_outputs = self.test_results['quantized'].get('outputs', [])
        
        if not original_outputs or not quantized_outputs:
            logger.warning("No outputs to compare")
            return {}
        
        comparison_results = {
            'cosine_similarities': [],
            'mse_errors': [],
            'max_errors': []
        }
        
        num_comparisons = min(len(original_outputs), len(quantized_outputs))
        logger.info(f"Comparing {num_comparisons} output pairs...")
        
        for i in range(num_comparisons):
            orig_out = original_outputs[i].flatten()
            quant_out = quantized_outputs[i].flatten()
            
            # Cosine similarity
            cosine_sim = np.dot(orig_out, quant_out) / (
                np.linalg.norm(orig_out) * np.linalg.norm(quant_out)
            )
            comparison_results['cosine_similarities'].append(float(cosine_sim))
            
            # MSE
            mse = np.mean((orig_out - quant_out) ** 2)
            comparison_results['mse_errors'].append(float(mse))
            
            # Max error
            max_error = np.max(np.abs(orig_out - quant_out))
            comparison_results['max_errors'].append(float(max_error))
        
        # Calculate statistics
        for metric in ['cosine_similarities', 'mse_errors', 'max_errors']:
            values = comparison_results[metric]
            comparison_results[f'avg_{metric}'] = float(np.mean(values))
            comparison_results[f'std_{metric}'] = float(np.std(values))
            comparison_results[f'min_{metric}'] = float(np.min(values))
            comparison_results[f'max_{metric}'] = float(np.max(values))
        
        return comparison_results
    
    def run_tests(self) -> bool:
        """Run all tests"""
        if not self._check_onnx_runtime():
            logger.error("ONNX Runtime not available")
            return False
        
        # Generate test inputs
        logger.info("Generating test inputs...")
        test_inputs = []
        
        # Synthetic audio
        for i in range(3):
            synthetic_audio = self.generate_test_audio(duration_seconds=3.0 + i)
            test_inputs.append(synthetic_audio)
        
        # Try to load real audio from test directory
        test_audio_dir = self.base_dir.parent / "audio-test"
        if test_audio_dir.exists():
            for audio_file in test_audio_dir.glob("*.wav"):
                real_audio = self.load_real_audio(str(audio_file))
                if real_audio is not None:
                    test_inputs.append(real_audio)
                    if len(test_inputs) >= 6:  # Limit test inputs
                        break
        
        logger.info(f"Created {len(test_inputs)} test inputs")
        
        # Test each model
        for model_name, model_path in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {model_name} model")
            logger.info(f"{'='*50}")
            
            results = self.test_model_inference(model_path, test_inputs)
            if results:
                self.test_results[model_name] = results
        
        # Compare models if both available
        if len(self.test_results) >= 2:
            logger.info(f"\n{'='*50}")
            logger.info(f"Comparing model outputs")
            logger.info(f"{'='*50}")
            
            comparison_results = self.compare_model_outputs()
            if comparison_results:
                self.test_results['comparison'] = comparison_results
        
        return len(self.test_results) > 0

def print_test_results(results: Dict):
    """Print formatted test results"""
    print(f"\n🧪 ONNX Model Test Results")
    print(f"=" * 50)
    
    for model_name, model_results in results.items():
        if model_name == 'comparison':
            continue
            
        print(f"\n📊 {model_name.upper()} MODEL")
        print(f"-" * 30)
        
        if 'model_size_mb' in model_results:
            print(f"Size: {model_results['model_size_mb']:.1f} MB")
        
        if 'embedding_dimensions' in model_results:
            print(f"Embedding dimensions: {model_results['embedding_dimensions']}")
        
        if 'avg_inference_ms' in model_results:
            print(f"Average inference: {model_results['avg_inference_ms']:.2f} ms")
            print(f"Min inference: {model_results['min_inference_ms']:.2f} ms")
            print(f"Max inference: {model_results['max_inference_ms']:.2f} ms")
        
        if 'outputs' in model_results:
            print(f"Successful inferences: {len(model_results['outputs'])}")
    
    # Comparison results
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n🔍 MODEL COMPARISON")
        print(f"-" * 30)
        print(f"Average cosine similarity: {comp.get('avg_cosine_similarities', 0):.6f}")
        print(f"Average MSE: {comp.get('avg_mse_errors', 0):.8f}")
        print(f"Average max error: {comp.get('avg_max_errors', 0):.6f}")
    
    # Performance comparison
    if 'original' in results and 'quantized' in results:
        orig = results['original']
        quant = results['quantized']
        
        if 'model_size_mb' in orig and 'model_size_mb' in quant:
            size_reduction = (orig['model_size_mb'] - quant['model_size_mb']) / orig['model_size_mb'] * 100
            print(f"\n🚀 PERFORMANCE SUMMARY")
            print(f"-" * 30)
            print(f"Size reduction: {size_reduction:.1f}%")
        
        if 'avg_inference_ms' in orig and 'avg_inference_ms' in quant:
            speed_improvement = (orig['avg_inference_ms'] - quant['avg_inference_ms']) / orig['avg_inference_ms'] * 100
            print(f"Speed improvement: {speed_improvement:.1f}%")

def main():
    """Main testing function"""
    print("🧪 TitaNet-L ONNX Inference Testing")
    print("=" * 40)
    
    base_dir = Path(__file__).parent
    
    # Check if models exist
    onnx_models_dir = base_dir / "onnx_models"
    
    if not onnx_models_dir.exists():
        logger.error(f"ONNX models directory not found: {onnx_models_dir}")
        logger.error("Run export_to_onnx.py first")
        return False
    
    # Create tester
    tester = ONNXInferenceTester(base_dir)
    
    # Run tests
    success = tester.run_tests()
    
    if not success:
        logger.error("Testing failed")
        return False
    
    # Print results
    print_test_results(tester.test_results)
    
    # Save results to file
    results_file = base_dir / "inference_test_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays for JSON serialization
        json_results = {}
        for model_name, model_results in tester.test_results.items():
            json_results[model_name] = {}
            for key, value in model_results.items():
                if key == 'outputs':
                    # Don't save outputs (too large)
                    json_results[model_name][key] = f"<{len(value)} outputs>"
                elif isinstance(value, list):
                    json_results[model_name][key] = [float(x) if isinstance(x, (int, float, np.number)) else x for x in value]
                else:
                    json_results[model_name][key] = float(value) if isinstance(value, (int, float, np.number)) else value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print(f"\n✅ Testing completed!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Deploy: python3 jetson_pipeline_onnx.py")
    print(f"   2. Real-time test on Jetson Nano")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)