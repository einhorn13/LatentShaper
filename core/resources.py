# core/resources.py

import psutil
import torch
from typing import Dict, Any

class ResourceMonitor:
    """
    Monitors System RAM and GPU VRAM usage.
    """

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """
        Returns a dictionary with current memory stats.
        """
        stats = {
            "ram_total": 0, "ram_used": 0, "ram_percent": 0.0,
            "vram_total": 0, "vram_used": 0, "vram_percent": 0.0,
            "gpu_name": "CPU"
        }

        # 1. System RAM (psutil)
        try:
            vm = psutil.virtual_memory()
            stats["ram_total"] = vm.total
            stats["ram_used"] = vm.used
            stats["ram_percent"] = vm.percent
        except Exception as e:
            print(f"Error reading RAM: {e}")

        # 2. GPU VRAM (torch)
        if torch.cuda.is_available():
            try:
                # Use current_device() to ensure we monitor the active GPU
                device_id = torch.cuda.current_device()
                stats["gpu_name"] = torch.cuda.get_device_name(device_id)
                
                # Returns (free, total) in bytes for the given device
                free, total = torch.cuda.mem_get_info(device_id)
                
                used = total - free
                stats["vram_total"] = total
                stats["vram_used"] = used
                stats["vram_percent"] = (used / total) * 100 if total > 0 else 0.0
            except Exception as e:
                print(f"Error reading VRAM: {e}")
        
        return stats

    @staticmethod
    def format_bytes(size: int) -> str:
        """Helper to format bytes to GB/MB."""
        power = 2**30 # 1024**3 (GiB)
        n = size / power
        if n >= 1:
            return f"{n:.1f} GB"
        
        power /= 1024 # MiB
        n = size / power
        return f"{n:.0f} MB"