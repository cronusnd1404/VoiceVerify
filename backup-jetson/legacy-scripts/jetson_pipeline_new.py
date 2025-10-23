"""
Pipeline đơn giản cho Jetson
"""

import torch
import os
import numpy as np
from speaker_verification_pipeline import SpeakerVerificationPipeline
from jetson_config import get_jetson_config

class JetsonSpeakerPipeline:
    """Pipeline đơn giản cho Jetson"""
    
    def __init__(self):
        self.config = get_jetson_config()
        self.pipeline = None
        self.model = None
        
    def setup_model(self):
        """Setup model với cấu hình Jetson"""
        try:
            # Tạo pipeline với config đơn giản
            pipeline_config = type('Config', (), {})()
            pipeline_config.titanet_model_path = self.config.get_model_path()
            pipeline_config.device = self.config.device
            pipeline_config.target_sample_rate = self.config.sample_rate
            pipeline_config.similarity_threshold = self.config.similarity_threshold
            pipeline_config.temp_dir = self.config.temp_dir
            
            self.pipeline = SpeakerVerificationPipeline(pipeline_config)
            print(f"✅ Đã setup model thành công trên {self.config.device}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi setup model: {e}")
            return False
    
    def preprocess_audio(self, audio_path):
        """Tiền xử lý audio đơn giản"""
        try:
            if self.pipeline:
                return self.pipeline.preprocess_audio(audio_path)
            else:
                print("❌ Pipeline chưa được setup")
                return None
        except Exception as e:
            print(f"❌ Lỗi preprocess audio: {e}")
            return None
    
    def extract_embedding(self, audio_path):
        """Trích xuất embedding"""
        try:
            if self.pipeline:
                return self.pipeline.extract_embedding(audio_path)
            else:
                print("❌ Pipeline chưa được setup")
                return None
        except Exception as e:
            print(f"❌ Lỗi extract embedding: {e}")
            return None
    
    def verify_speaker(self, test_audio, enrolled_audio):
        """Xác thực speaker"""
        try:
            if self.pipeline:
                return self.pipeline.verify_speaker(test_audio, enrolled_audio)
            else:
                print("❌ Pipeline chưa được setup")
                return None
        except Exception as e:
            print(f"❌ Lỗi verify speaker: {e}")
            return None
    
    def get_memory_usage(self):
        """Lấy thông tin memory usage"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_max = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                return {
                    'gpu_current_mb': gpu_memory,
                    'gpu_max_mb': gpu_max
                }
        except:
            pass
        return {'gpu_current_mb': 0, 'gpu_max_mb': 0}
    
    def clear_cache(self):
        """Xóa cache để giải phóng memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Đã xóa cache")
        except Exception as e:
            print(f"❌ Lỗi xóa cache: {e}")

def create_jetson_pipeline():
    """Tạo pipeline Jetson đơn giản"""
    pipeline = JetsonSpeakerPipeline()
    
    # Setup model
    if pipeline.setup_model():
        print("🎉 Jetson Pipeline sẵn sàng!")
        return pipeline
    else:
        print("❌ Không thể tạo pipeline")
        return None

if __name__ == "__main__":
    print("=== Test Jetson Speaker Pipeline ===")
    
    # Tạo pipeline
    pipeline = create_jetson_pipeline()
    
    if pipeline:
        # Kiểm tra memory
        memory_info = pipeline.get_memory_usage()
        print(f"GPU Memory: {memory_info['gpu_current_mb']:.1f}MB")
        
        # Test với file audio nếu có
        test_audio = "/home/edabk/Titanet/integration/test.wav"  # Thay đổi path này
        
        if os.path.exists(test_audio):
            print(f"Testing với {test_audio}...")
            embedding = pipeline.extract_embedding(test_audio)
            if embedding is not None:
                print(f"✅ Embedding shape: {embedding.shape}")
            else:
                print("❌ Không thể extract embedding")
        else:
            print(f"File test không tồn tại: {test_audio}")
        
        # Clear cache
        pipeline.clear_cache()
    
    print("Hoàn thành test!")