#!/usr/bin/env python3
"""
ONNX Model Quantization for TitaNet-L
Optimizes models for Jetson Nano deployment
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXQuantizer:
    """Quantize ONNX models for optimized inference"""
    
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.model_dir = Path(onnx_model_path).parent
        self.quantization_available = self._check_quantization_tools()
        
    def _check_quantization_tools(self) -> bool:
        """Check if ONNX quantization tools are available"""
        try:
            from onnxruntime.quantization import quantize_dynamic
            return True
        except ImportError:
            logger.warning("ONNX quantization tools not available")
            return False
    
    def dynamic_quantization(self, output_path: Optional[str] = None) -> bool:
        """Dynamic quantization - quantize weights only"""
        if not self.quantization_available:
            logger.error("Quantization tools not available")
            return False
        
        if output_path is None:
            model_name = Path(self.onnx_model_path).stem
            output_path = str(self.model_dir / f"{model_name}-dynamic-quantized.onnx")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            logger.info("Starting dynamic quantization...")
            logger.info(f"Input: {self.onnx_model_path}")
            logger.info(f"Output: {output_path}")
            
            quantize_dynamic(
                self.onnx_model_path,
                output_path,
                weight_type=QuantType.QUInt8
            )
            
            # Compare file sizes
            original_size = os.path.getsize(self.onnx_model_path) / (1024 * 1024)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (original_size - quantized_size) / original_size * 100
            
            logger.info("✓ Dynamic quantization completed")
            logger.info(f"Size reduction: {original_size:.1f}MB → {quantized_size:.1f}MB ({reduction:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return False
    
    def benchmark_models(self, quantized_model_path: str) -> Dict:
        """Benchmark original vs quantized model"""
        try:
            import onnxruntime as ort
            
            logger.info("Benchmarking models...")
            
            results = {
                'original': {'size_mb': 0, 'inference_times': []},
                'quantized': {'size_mb': 0, 'inference_times': []},
                'accuracy_comparison': []
            }
            
            # Model sizes
            results['original']['size_mb'] = os.path.getsize(self.onnx_model_path) / (1024 * 1024)
            results['quantized']['size_mb'] = os.path.getsize(quantized_model_path) / (1024 * 1024)
            
            # Create sessions
            original_session = ort.InferenceSession(self.onnx_model_path)
            quantized_session = ort.InferenceSession(quantized_model_path)
            
            # Test data (5 seconds audio at 16kHz)
            test_inputs = []
            for _ in range(5):
                test_audio = np.random.randn(1, 80000).astype(np.float32)
                test_inputs.append(test_audio)
            
            # Benchmark inference times
            num_runs = 10
            
            for session, key in [(original_session, 'original'), (quantized_session, 'quantized')]:
                logger.info(f"Benchmarking {key} model...")
                
                for _ in range(num_runs):
                    for test_input in test_inputs:
                        start_time = time.time()
                        
                        ort_inputs = {session.get_inputs()[0].name: test_input}
                        _ = session.run(None, ort_inputs)
                        
                        inference_time = (time.time() - start_time) * 1000  # ms
                        results[key]['inference_times'].append(inference_time)
            
            # Accuracy comparison
            for test_input in test_inputs:
                # Original output
                ort_inputs = {original_session.get_inputs()[0].name: test_input}
                original_output = original_session.run(None, ort_inputs)[0]
                
                # Quantized output
                ort_inputs = {quantized_session.get_inputs()[0].name: test_input}
                quantized_output = quantized_session.run(None, ort_inputs)[0]
                
                # Cosine similarity
                cosine_sim = np.dot(original_output.flatten(), quantized_output.flatten()) / (
                    np.linalg.norm(original_output) * np.linalg.norm(quantized_output)
                )
                results['accuracy_comparison'].append(float(cosine_sim))
            
            # Calculate statistics
            for key in ['original', 'quantized']:
                times = results[key]['inference_times']
                results[key]['avg_inference_ms'] = float(np.mean(times))
                results[key]['std_inference_ms'] = float(np.std(times))
            
            results['avg_cosine_similarity'] = float(np.mean(results['accuracy_comparison']))
            
            # Performance metrics
            size_reduction = (results['original']['size_mb'] - results['quantized']['size_mb']) / results['original']['size_mb'] * 100
            speed_improvement = (results['original']['avg_inference_ms'] - results['quantized']['avg_inference_ms']) / results['original']['avg_inference_ms'] * 100
            
            results['size_reduction_percent'] = float(size_reduction)
            results['speed_improvement_percent'] = float(speed_improvement)
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}

def main():
    """Main quantization function"""
    print("🔧 TitaNet-L ONNX Quantization")
    print("=" * 40)
    
    # Find ONNX model
    base_dir = Path(__file__).parent
    onnx_model_path = base_dir / "onnx_models" / "titanet-l.onnx"
    
    if not onnx_model_path.exists():
        logger.error(f"ONNX model not found: {onnx_model_path}")
        logger.error("Run export_to_onnx.py first")
        return False
    
    # Create quantizer
    quantizer = ONNXQuantizer(str(onnx_model_path))
    
    if not quantizer.quantization_available:
        logger.error("Quantization tools not available")
        logger.error("Install with: pip install onnxruntime-tools")
        return False
    
    print(f"\n📥 Input: {onnx_model_path}")
    
    # Dynamic quantization
    print(f"\n🔄 Dynamic quantization...")
    quantized_path = str(base_dir / "onnx_models" / "titanet-l-dynamic-quantized.onnx")
    
    success = quantizer.dynamic_quantization(quantized_path)
    
    if not success:
        logger.error("Quantization failed")
        return False
    
    # Benchmark
    print(f"\n📊 Benchmarking...")
    results = quantizer.benchmark_models(quantized_path)
    
    if results:
        print(f"\n🚀 Quantization Results:")
        print(f"   Original size: {results['original']['size_mb']:.1f} MB")
        print(f"   Quantized size: {results['quantized']['size_mb']:.1f} MB")
        print(f"   Size reduction: {results['size_reduction_percent']:.1f}%")
        print(f"   Original inference: {results['original']['avg_inference_ms']:.1f} ms")
        print(f"   Quantized inference: {results['quantized']['avg_inference_ms']:.1f} ms")
        print(f"   Speed improvement: {results['speed_improvement_percent']:.1f}%")
        print(f"   Output similarity: {results['avg_cosine_similarity']:.4f}")
        
        # Save results
        results_file = base_dir / "quantization_results.json"
        with open(results_file, 'w') as f:
            # Convert lists for JSON serialization
            json_results = results.copy()
            for model_type in ['original', 'quantized']:
                if 'inference_times' in json_results[model_type]:
                    json_results[model_type]['inference_times'] = [
                        float(x) for x in json_results[model_type]['inference_times']
                    ]
            json_results['accuracy_comparison'] = [
                float(x) for x in json_results['accuracy_comparison']
            ]
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n✅ Quantization completed!")
    print(f"📁 Quantized model: {quantized_path}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Test: python3 test_onnx_inference.py")
    print(f"   2. Deploy: python3 jetson_pipeline_onnx.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)