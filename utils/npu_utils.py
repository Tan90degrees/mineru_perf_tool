# Copyright (c) Opendatalab. All rights reserved.
"""
NPU device management utilities.
"""

import os
import subprocess
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import torch
    import torch_npu
    TORCH_NPU_AVAILABLE = True
except ImportError:
    TORCH_NPU_AVAILABLE = False


class NPUManager:
    """Manage NPU devices."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if NPU is available."""
        if not TORCH_NPU_AVAILABLE:
            return False
        
        try:
            return torch.npu.is_available()
        except Exception:
            return False
    
    @staticmethod
    def get_device_count() -> int:
        """Get number of available NPU devices."""
        if not TORCH_NPU_AVAILABLE:
            return 0
        
        try:
            return torch.npu.device_count()
        except Exception:
            return 0
    
    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        """Get NPU device name."""
        if not TORCH_NPU_AVAILABLE:
            return "NPU not available"
        
        try:
            return torch.npu.get_device_name(device_id)
        except Exception as e:
            logger.warning(f"Failed to get device name: {e}")
            return "Unknown"
    
    @staticmethod
    def get_device_capability(device_id: int = 0) -> tuple:
        """Get NPU device capability."""
        if not TORCH_NPU_AVAILABLE:
            return (0, 0)
        
        try:
            return torch.npu.get_device_capability(device_id)
        except Exception:
            return (0, 0)
    
    @staticmethod
    def get_memory_info(device_id: int = 0) -> Dict[str, int]:
        """
        Get NPU memory information.
        
        Returns:
            Dictionary with memory info in bytes
        """
        if not TORCH_NPU_AVAILABLE:
            return {"total": 0, "allocated": 0, "free": 0}
        
        try:
            total = torch.npu.get_device_properties(device_id).total_memory
            allocated = torch.npu.memory_allocated(device_id)
            free = total - allocated
            
            return {
                "total": total,
                "allocated": allocated,
                "free": free,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "free_gb": free / (1024**3),
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {"total": 0, "allocated": 0, "free": 0}
    
    @staticmethod
    def get_utilization(device_id: int = 0) -> float:
        """
        Get NPU utilization percentage using npu-smi.
        
        Returns:
            Utilization percentage (0-100)
        """
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(device_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse npu-smi output for utilization
                # This depends on the actual output format
                for line in result.stdout.split('\n'):
                    if 'utilization' in line.lower() or 'aicore' in line.lower():
                        # Try to extract percentage
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            return float(match.group(1))
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Failed to get NPU utilization: {e}")
            return 0.0
    
    @staticmethod
    def set_visible_devices(devices: List[int]) -> None:
        """
        Set visible NPU devices.
        
        Args:
            devices: List of device IDs to make visible
        """
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        logger.info(f"Set visible NPU devices: {devices}")
    
    @staticmethod
    def reset_memory(device_id: int = 0) -> None:
        """Reset NPU memory."""
        if not TORCH_NPU_AVAILABLE:
            return
        
        try:
            torch.npu.empty_cache()
            logger.debug(f"Reset NPU {device_id} memory")
        except Exception as e:
            logger.warning(f"Failed to reset memory: {e}")
    
    @staticmethod
    def synchronize(device_id: int = 0) -> None:
        """Synchronize NPU device."""
        if not TORCH_NPU_AVAILABLE:
            return
        
        try:
            torch.npu.synchronize(device_id)
            logger.debug(f"Synchronized NPU {device_id}")
        except Exception as e:
            logger.warning(f"Failed to synchronize: {e}")


def get_npu_info() -> Dict[str, Any]:
    """
    Get comprehensive NPU information.
    
    Returns:
        Dictionary with NPU information
    """
    info = {
        "available": NPUManager.is_available(),
        "device_count": NPUManager.get_device_count(),
        "devices": [],
    }
    
    if not info["available"]:
        return info
    
    for device_id in range(info["device_count"]):
        device_info = {
            "device_id": device_id,
            "name": NPUManager.get_device_name(device_id),
            "capability": NPUManager.get_device_capability(device_id),
            "memory": NPUManager.get_memory_info(device_id),
            "utilization": NPUManager.get_utilization(device_id),
        }
        info["devices"].append(device_info)
    
    return info


def check_npu_health() -> Dict[str, bool]:
    """
    Check NPU health status.
    
    Returns:
        Dictionary with health check results
    """
    health = {
        "npu_available": NPUManager.is_available(),
        "torch_npu_installed": TORCH_NPU_AVAILABLE,
    }
    
    if not health["npu_available"]:
        return health
    
    # Check each device
    device_count = NPUManager.get_device_count()
    for device_id in range(device_count):
        try:
            # Try to get memory info
            mem_info = NPUManager.get_memory_info(device_id)
            health[f"device_{device_id}_memory"] = mem_info["total"] > 0
        except Exception as e:
            health[f"device_{device_id}_memory"] = False
            logger.warning(f"Device {device_id} health check failed: {e}")
    
    return health
