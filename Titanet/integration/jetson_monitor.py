"""
Monitor đơn giản cho Jetson
"""

import os
import time
import json
import subprocess
from datetime import datetime

class JetsonMonitor:
    """Monitor cơ bản cho Jetson"""
    
    def __init__(self):
        self.log_file = "/home/edabk/Titanet/integration/logs/jetson_monitor.json"
        
    def get_jetson_info(self) -> Dict:
        """Get Jetson hardware information"""
        info = {}
        
        try:
            # Get Jetson model
            with open('/sys/firmware/devicetree/base/model', 'r') as f:
                info['model'] = f.read().strip()
        except:
            info['model'] = "Unknown Jetson"
        
        try:
            # Get JetPack version
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'R' in line and 'REVISION' in line:
                        info['jetpack_version'] = line.strip()
                        break
        except:
            info['jetpack_version'] = "Unknown"
        
        # Get CPU info
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_freq_mhz'] = psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
        
        # Get memory info
        mem = psutil.virtual_memory()
        info['total_memory_gb'] = mem.total / (1024**3)
        
    def get_cpu_usage(self):
        """Lấy thông tin CPU usage đơn giản"""
        try:
            with open('/proc/loadavg', 'r') as f:
                load = f.read().strip().split()[0]
                return float(load) * 25  # Rough percentage
        except:
            return 0.0
    
    def get_memory_usage(self):
        """Lấy thông tin memory usage"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                
            mem_total = 0
            mem_free = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    mem_free = int(line.split()[1])
                    
            if mem_total > 0:
                mem_used = mem_total - mem_free
                return (mem_used / mem_total) * 100
        except:
            pass
        return 0.0
    
    def get_temperature(self):
        """Lấy nhiệt độ CPU"""
        try:
            # Thử đọc nhiệt độ từ thermal zone
            temp_files = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp'
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        return temp
        except:
            pass
        return None
    
    def get_current_status(self):
        """Lấy trạng thái hiện tại"""
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage(),
            'temperature': self.get_temperature()
        }
        return status
    
    def check_system_health(self):
        """Kiểm tra sức khỏe hệ thống"""
        status = self.get_current_status()
        warnings = []
        
        if status['cpu_usage'] > 80:
            warnings.append(f"CPU cao: {status['cpu_usage']:.1f}%")
        
        if status['memory_usage'] > 85:
            warnings.append(f"Memory cao: {status['memory_usage']:.1f}%")
        
        if status['temperature'] and status['temperature'] > 75:
            warnings.append(f"Nhiệt độ cao: {status['temperature']:.1f}°C")
        
        return status, warnings
    
    def log_status(self, status):
        """Ghi log trạng thái"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(status) + '\n')
        except:
            pass
    
    def print_status(self):
        """In trạng thái ra màn hình"""
        status, warnings = self.check_system_health()
        
        print(f"\n=== Jetson Status [{status['timestamp']}] ===")
        print(f"CPU Usage: {status['cpu_usage']:.1f}%")
        print(f"Memory Usage: {status['memory_usage']:.1f}%")
        
        if status['temperature']:
            print(f"Temperature: {status['temperature']:.1f}°C")
        
        if warnings:
            print("\n⚠️ Cảnh báo:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("✅ Hệ thống hoạt động bình thường")
        
        self.log_status(status)
        return status


def monitor_loop(duration=60):
    """Chạy monitor trong khoảng thời gian nhất định"""
    monitor = JetsonMonitor()
    
    print(f"Bắt đầu monitor Jetson trong {duration} giây...")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        monitor.print_status()
        time.sleep(5)  # Check every 5 seconds
    
    print("\nKết thúc monitor")

if __name__ == "__main__":
    monitor = JetsonMonitor()
    
    # Test một lần
    print("=== Test Jetson Monitor ===")
    monitor.print_status()
    
    # Có thể chạy loop nếu cần
    # monitor_loop(30)  # Monitor 30 giây