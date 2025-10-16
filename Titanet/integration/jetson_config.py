"""
Cấu hình đơn giản cho Jetson
"""

import os
import json

# Cấu hình cơ bản cho Jetson
class JetsonConfig:
    def __init__(self):
        self.batch_size = 1
        self.num_workers = 2
        self.sample_rate = 16000
        self.max_audio_duration = 30.0
        self.min_audio_duration = 1.0
        self.similarity_threshold = 0.65
        self.device = "cuda" if self.has_cuda() else "cpu"
        self.temp_dir = "/home/edabk/Titanet/integration/temp"
        
    def has_cuda(self):
        """Kiểm tra có CUDA không"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
            
    def get_model_path(self):
        """Lấy đường dẫn model"""
        model_path = "/home/edabk/Titanet/integration/titanet-l.nemo"
        return model_path if os.path.exists(model_path) else None
        
    def create_temp_dir(self):
        """Tạo thư mục tạm"""
        os.makedirs(self.temp_dir, exist_ok=True)
        return self.temp_dir

def get_jetson_config():
    """Tạo cấu hình Jetson đơn giản"""
    return JetsonConfig()

if __name__ == "__main__":
    config = get_jetson_config()
    print(f"Jetson Config:")
    print(f"  - Device: {config.device}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Sample rate: {config.sample_rate}")
    print(f"  - Model path: {config.get_model_path()}")