#!/usr/bin/env python3
"""
Export TitaNet-L NeMo model to ONNX format
Designed for Jetson Nano deployment and optimization
"""

import torch
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from datetime import datetime

# NeMo imports
import nemo.collections.asr as nemo_asr

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TitaNetONNXExporter:
    """Export TitaNet-L model from NeMo to ONNX format"""
    
    def __init__(self, nemo_model_path: str):
        self.nemo_model_path = nemo_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_nemo_model(self):
        """Load NeMo TitaNet-L model"""
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            
            logger.info(f"Loading NeMo model from: {self.nemo_model_path}")
            model = EncDecSpeakerLabelModel.restore_from(self.nemo_model_path)
            model.eval()
            model = model.to(self.device)
            
            logger.info("✓ NeMo model loaded successfully")
            return model
            
        except ImportError:
            logger.error("NeMo toolkit not found. Install with: pip install nemo-toolkit")
            raise
        except Exception as e:
            logger.error(f"Failed to load NeMo model: {e}")
            raise
    
    def create_dummy_input(self, sample_rate: int = 16000, duration: float = 5.0) -> tuple:
        """Create dummy audio input for ONNX export (NeMo needs both audio and length)"""
        num_samples = int(sample_rate * duration)
        # Create realistic audio-like input
        dummy_audio = torch.randn(1, num_samples, dtype=torch.float32).to(self.device)
        dummy_length = torch.tensor([num_samples], dtype=torch.long).to(self.device)
        
        logger.info(f"Created dummy inputs: audio shape={dummy_audio.shape}, length={dummy_length.shape}, duration={duration}s")
        return (dummy_audio, dummy_length)


def export_to_onnx(model_path: str, output_path: str, opset_version: int = 17) -> bool:
    """
    Export NeMo TitaNet model to ONNX format with preprocessing bypass.
    
    Args:
        model_path: Path to the .nemo model file
        output_path: Path where ONNX model will be saved
        opset_version: ONNX opset version (17 for STFT support)
    
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        # Load model
        print(f"Loading model from {model_path}")
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path, map_location=torch.device('cuda:0'))
        model.eval()
        
        # Simple approach: override the problematic forward method
        print("Creating ONNX-compatible wrapper for the model...")
        
        # Create a wrapper that bypasses preprocessing
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, nemo_model):
                super().__init__()
                self.encoder = nemo_model.encoder
                self.decoder = nemo_model.decoder
                
            def forward(self, processed_signal, processed_length):
                # Skip preprocessing, go directly to encoder
                encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_length)
                # Get embeddings from decoder  
                logits, embeddings = self.decoder(encoder_output=encoded, length=encoded_len)
                return embeddings  # Return only embeddings for speaker verification
        
        export_model = ONNXWrapper(model)
        export_model.eval()
        
        # Create input for wrapped model (expects pre-processed features)
        def create_wrapper_input():
            # Features from preprocessor output: (batch, features, time)
            features = torch.randn(1, 80, 1000, device='cuda:0')
            length = torch.tensor([1000], device='cuda:0')
            return (features, length)
        
        dummy_input = create_wrapper_input()
        input_names = ['processed_signal', 'processed_length']
        output_names = ['embeddings']
        print("Input shape: processed_signal (1, 80, 1000), length (1,)")
        
        print(f"Exporting to ONNX with opset version {opset_version}...")
        
        # Export with appropriate configuration
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'processed_signal': {0: 'batch_size', 2: 'time_frames'},
                'processed_length': {0: 'batch_size'},
                'embeddings': {0: 'batch_size'}
            },
            verbose=False  # Reduce output verbosity
        )
        
        print(f"✅ Export successful! Model saved to: {output_path}")
        print("⚠️  Note: This model expects pre-processed mel-spectrogram features as input")
        print("   Input shape: (batch_size, 80, time_frames)")
        print("   You'll need to handle audio preprocessing (mel-spectrogram) separately in your pipeline")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False
    
    def verify_onnx_model(self, onnx_path: str, test_input: torch.Tensor, original_model) -> bool:
        """Verify ONNX model produces similar outputs to original"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get original model output
            with torch.no_grad():
                if isinstance(test_input, tuple):
                    original_output = original_model(*test_input)
                else:
                    original_output = original_model(test_input)
                    
                if hasattr(original_output, 'cpu'):
                    original_np = original_output.cpu().numpy()
                else:
                    original_np = original_output
            
            # Get ONNX model output  
            if isinstance(test_input, tuple):
                # NeMo model with audio and length
                ort_inputs = {
                    ort_session.get_inputs()[0].name: test_input[0].cpu().numpy(),
                    ort_session.get_inputs()[1].name: test_input[1].cpu().numpy()
                }
            else:
                # Single input
                ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
                
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare outputs
            diff = np.abs(ort_outputs[0] - original_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            logger.info(f"Output comparison - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            
            # Verification threshold
            if max_diff < 1e-3:
                return True
            else:
                logger.warning(f"Large output difference detected: {max_diff}")
                return False
                
        except ImportError:
            logger.error("ONNX or ONNXRuntime not installed")
            return False
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def get_model_info(self, onnx_path: str) -> dict:
        """Get information about exported ONNX model"""
        try:
            import onnx
            
            model = onnx.load(onnx_path)
            
            # Model size
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            # Input/output info
            input_info = []
            for inp in model.graph.input:
                shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" 
                        for dim in inp.type.tensor_type.shape.dim]
                input_info.append({
                    'name': inp.name,
                    'shape': shape,
                    'type': inp.type.tensor_type.elem_type
                })
            
            output_info = []
            for out in model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" 
                        for dim in out.type.tensor_type.shape.dim]
                output_info.append({
                    'name': out.name,
                    'shape': shape,
                    'type': out.type.tensor_type.elem_type
                })
            
            info = {
                'file_size_mb': file_size,
                'opset_version': model.opset_import[0].version,
                'inputs': input_info,
                'outputs': output_info,
                'num_nodes': len(model.graph.node)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

def main():
    """Main export function"""
    print("🚀 TitaNet-L ONNX Export Tool")
    print("=" * 50)
    
    # Paths
    base_dir = Path(__file__).parent
    nemo_model_path = base_dir / "titanet-l.nemo"
    output_dir = base_dir / "onnx_models"
    output_dir.mkdir(exist_ok=True)
    
    onnx_output_path = output_dir / "titanet-l.onnx"
    
    # Check if NeMo model exists
    if not nemo_model_path.exists():
        logger.error(f"NeMo model not found: {nemo_model_path}")
        return False
    
    # Export to ONNX
    print(f"\n📥 Input: {nemo_model_path}")
    print(f"📤 Output: {onnx_output_path}")
    print(f"⚙️  Device: cuda")
    
    success = export_to_onnx(
        str(nemo_model_path),
        str(onnx_output_path),
        opset_version=17
    )
    
    if success:
        print(f"\n✅ Export successful!")
        
        # Show basic file info
        if onnx_output_path.exists():
            file_size = onnx_output_path.stat().st_size / (1024 * 1024)
            print(f"\n📊 Model Information:")
            print(f"   File size: {file_size:.1f} MB")
            print(f"   ONNX opset: 17")
            print(f"   Path: {onnx_output_path}")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Quantize model: python3 quantize_onnx.py")
        print(f"   2. Test inference: python3 test_onnx_inference.py")
        print(f"   3. Deploy to Jetson: python3 jetson_pipeline_onnx.py")
        
    else:
        print(f"\n❌ Export failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)