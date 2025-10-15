#!/usr/bin/env python3
"""
Jetson Performance Monitor and Optimization Script
Monitors system performance and provides optimization recommendations
"""

import psutil
import time
import subprocess
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    temperature_c: Optional[float]
    gpu_memory_mb: Optional[float] 
    power_draw_w: Optional[float]
    operation: str = "idle"

class JetsonMonitor:
    """Monitor and optimize Jetson performance"""
    
    def __init__(self, log_file: str = "/tmp/jetson_performance.json"):
        self.log_file = log_file
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
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
        
        return info
    
    def get_thermal_info(self) -> Dict:
        """Get Jetson thermal information"""
        thermal_info = {}
        
        try:
            # Common thermal zones for Jetson devices
            thermal_zones = [
                ('/sys/class/thermal/thermal_zone0/temp', 'CPU'),
                ('/sys/class/thermal/thermal_zone1/temp', 'GPU'),
                ('/sys/class/thermal/thermal_zone2/temp', 'PLL'),
                ('/sys/devices/virtual/thermal/thermal_zone0/temp', 'PMIC'),
            ]
            
            for zone_path, zone_name in thermal_zones:
                if os.path.exists(zone_path):
                    with open(zone_path, 'r') as f:
                        temp_raw = int(f.read().strip())
                        thermal_info[zone_name] = temp_raw / 1000.0
        except Exception as e:
            thermal_info['error'] = str(e)
        
        return thermal_info
    
    def get_power_info(self) -> Optional[float]:
        """Get power consumption information"""
        try:
            # Try different power monitoring paths for different Jetson models
            power_paths = [
                '/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input',
                '/sys/class/hwmon/hwmon0/power1_input',
                '/sys/class/hwmon/hwmon1/power1_input',
            ]
            
            for path in power_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        power_mw = int(f.read().strip())
                        return power_mw / 1000.0  # Convert to watts
        except:
            pass
        
        return None
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage via nvidia-smi or tegrastats"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        try:
            # Try tegrastats as fallback
            result = subprocess.run(['tegrastats', '--interval', '1000'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                # Parse tegrastats output for GPU memory
                # Format usually like: "GR3D_FREQ 0% EMC_FREQ 0% ... RAM 1234/4096MB"
                for line in result.stdout.split('\n'):
                    if 'RAM' in line:
                        parts = line.split('RAM')[1].strip().split('/')
                        if len(parts) >= 1:
                            used_mb = int(parts[0].replace('MB', ''))
                            return used_mb
        except:
            pass
        
        return None
    
    def get_current_metrics(self, operation: str = "idle") -> PerformanceMetrics:
        """Get current performance metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Temperature (get max from all zones)
        thermal_info = self.get_thermal_info()
        max_temp = None
        if thermal_info:
            temps = [temp for temp in thermal_info.values() if isinstance(temp, (int, float))]
            max_temp = max(temps) if temps else None
        
        # GPU memory
        gpu_memory = self.get_gpu_memory_usage()
        
        # Power consumption
        power_draw = self.get_power_info()
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            temperature_c=max_temp,
            gpu_memory_mb=gpu_memory,
            power_draw_w=power_draw,
            operation=operation
        )
    
    def log_metrics(self, metrics: PerformanceMetrics):
        """Log metrics to file and memory"""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries to avoid memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Save to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics.__dict__) + '\n')
        except Exception as e:
            print(f"Failed to log metrics: {e}")
    
    def monitor_performance(self, interval: float = 2.0):
        """Continuous performance monitoring"""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics("monitoring")
                self.log_metrics(metrics)
                
                # Check for critical conditions
                self._check_critical_conditions(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval: float = 2.0):
        """Start background monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self.monitor_performance, 
                args=(interval,)
            )
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print(f"Started Jetson monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            print("Stopped Jetson monitoring")
    
    def _check_critical_conditions(self, metrics: PerformanceMetrics):
        """Check for critical performance conditions"""
        warnings = []
        
        # Temperature warnings
        if metrics.temperature_c and metrics.temperature_c > 80:
            warnings.append(f"High temperature: {metrics.temperature_c:.1f}°C")
        
        # Memory warnings
        if metrics.memory_percent > 90:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # CPU warnings
        if metrics.cpu_percent > 95:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if warnings:
            print(f"⚠️  JETSON WARNING [{metrics.timestamp}]: {', '.join(warnings)}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Analyze metrics and provide optimization recommendations"""
        if not self.metrics_history:
            return ["No metrics available for analysis"]
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        recommendations = []
        
        # Analyze temperature
        avg_temp = sum(m.temperature_c for m in recent_metrics if m.temperature_c) / len(recent_metrics)
        if avg_temp > 70:
            recommendations.extend([
                "High temperature detected. Consider:",
                "- Enabling jetson_clocks with lower power mode",
                "- Improving cooling/ventilation", 
                "- Reducing batch sizes or model complexity"
            ])
        
        # Analyze memory usage
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        if avg_memory > 80:
            recommendations.extend([
                "High memory usage detected. Consider:",
                "- Reducing batch sizes",
                "- Clearing model caches regularly",
                "- Using FP16 precision",
                "- Increasing swap space"
            ])
        
        # Analyze CPU usage
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 80:
            recommendations.extend([
                "High CPU usage detected. Consider:",
                "- Reducing number of processing threads",
                "- Using GPU acceleration where possible",
                "- Optimizing audio preprocessing"
            ])
        
        if not recommendations:
            recommendations.append("Performance looks good! No optimizations needed.")
        
        return recommendations
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics_history[-50:]  # Last 50 measurements
        
        report = {
            "jetson_info": self.get_jetson_info(),
            "measurement_period": {
                "start": recent_metrics[0].timestamp if recent_metrics else None,
                "end": recent_metrics[-1].timestamp if recent_metrics else None,
                "num_measurements": len(recent_metrics)
            },
            "performance_summary": {},
            "recommendations": self.get_optimization_recommendations()
        }
        
        if recent_metrics:
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            temp_values = [m.temperature_c for m in recent_metrics if m.temperature_c]
            
            report["performance_summary"] = {
                "cpu_usage": {
                    "avg": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                },
                "memory_usage": {
                    "avg": sum(memory_values) / len(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values)
                }
            }
            
            if temp_values:
                report["performance_summary"]["temperature"] = {
                    "avg": sum(temp_values) / len(temp_values),
                    "max": max(temp_values),
                    "min": min(temp_values)
                }
        
        return report

def optimize_jetson_performance():
    """Apply Jetson performance optimizations"""
    print("Applying Jetson performance optimizations...")
    
    optimizations = []
    
    try:
        # Set maximum performance mode
        result = subprocess.run(['sudo', 'nvpmodel', '-m', '0'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            optimizations.append("✓ Set nvpmodel to max performance mode")
        else:
            optimizations.append("✗ Failed to set nvpmodel")
    except:
        optimizations.append("✗ nvpmodel not available")
    
    try:
        # Enable jetson_clocks
        result = subprocess.run(['sudo', 'jetson_clocks'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            optimizations.append("✓ Enabled jetson_clocks")
        else:
            optimizations.append("✗ Failed to enable jetson_clocks")
    except:
        optimizations.append("✗ jetson_clocks not available")
    
    # Set environment variables for optimization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    optimizations.append("✓ Set CUDA environment variables")
    
    return optimizations

if __name__ == "__main__":
    monitor = JetsonMonitor()
    
    print("=== Jetson Performance Monitor ===")
    
    # Show system information
    info = monitor.get_jetson_info()
    print(f"\nJetson Information:")
    print(f"  Model: {info['model']}")
    print(f"  JetPack: {info['jetpack_version']}")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  Total Memory: {info['total_memory_gb']:.1f}GB")
    
    # Show current metrics
    print(f"\nCurrent Performance:")
    metrics = monitor.get_current_metrics("startup")
    print(f"  CPU Usage: {metrics.cpu_percent:.1f}%")
    print(f"  Memory Usage: {metrics.memory_percent:.1f}%")
    if metrics.temperature_c:
        print(f"  Temperature: {metrics.temperature_c:.1f}°C")
    if metrics.power_draw_w:
        print(f"  Power Draw: {metrics.power_draw_w:.1f}W")
    
    # Apply optimizations
    print(f"\nApplying optimizations:")
    optimizations = optimize_jetson_performance()
    for opt in optimizations:
        print(f"  {opt}")
    
    # Start monitoring for demo
    print(f"\nStarting performance monitoring for 30 seconds...")
    monitor.start_monitoring(interval=2.0)
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted")
    
    monitor.stop_monitoring()
    
    # Generate report
    print(f"\nPerformance Report:")
    report = monitor.generate_performance_report()
    
    if "performance_summary" in report:
        summary = report["performance_summary"]
        if "cpu_usage" in summary:
            cpu = summary["cpu_usage"]
            print(f"  CPU: avg={cpu['avg']:.1f}%, max={cpu['max']:.1f}%")
        if "memory_usage" in summary:
            mem = summary["memory_usage"]  
            print(f"  Memory: avg={mem['avg']:.1f}%, max={mem['max']:.1f}%")
        if "temperature" in summary:
            temp = summary["temperature"]
            print(f"  Temperature: avg={temp['avg']:.1f}°C, max={temp['max']:.1f}°C")
    
    print(f"\nRecommendations:")
    for rec in report.get("recommendations", []):
        print(f"  {rec}")