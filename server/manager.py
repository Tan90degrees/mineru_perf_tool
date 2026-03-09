# Copyright (c) Opendatalab. All rights reserved.
"""
Server manager for multi-card multi-instance deployment.
"""

import time
from typing import List, Dict, Optional, Any
from loguru import logger

from .instance import ServiceInstance


class ServerManager:
    """Manages multiple MinerU service instances across devices."""
    
    def __init__(
        self,
        devices: List[int],
        instances_per_device: int = 1,
        base_port: int = 8000,
        backend: str = "hybrid-auto-engine",
        device_mode: str = "npu",
        max_concurrent_requests: int = 10,
    ):
        """
        Initialize server manager.
        
        Args:
            devices: List of device IDs to use
            instances_per_device: Number of instances per device
            base_port: Starting port number
            backend: MinerU backend type
            device_mode: Device mode (npu/cuda/cpu)
            max_concurrent_requests: Maximum concurrent requests per instance
        """
        self.devices = devices
        self.instances_per_device = instances_per_device
        self.base_port = base_port
        self.backend = backend
        self.device_mode = device_mode
        self.max_concurrent_requests = max_concurrent_requests
        
        self.instances: List[ServiceInstance] = []
        self.instance_urls: List[str] = []
        self.current_index = 0
        
    def start_all(self) -> bool:
        """Start all service instances."""
        logger.info(f"Starting {len(self.devices)} devices × {self.instances_per_device} instances")
        
        instance_id = 0
        for device in self.devices:
            for i in range(self.instances_per_device):
                port = self.base_port + instance_id
                
                instance = ServiceInstance(
                    device=device,
                    port=port,
                    instance_id=instance_id,
                    backend=self.backend,
                    device_mode=self.device_mode,
                    max_concurrent_requests=self.max_concurrent_requests,
                )
                
                if instance.start():
                    self.instances.append(instance)
                    self.instance_urls.append(instance.url)
                    instance_id += 1
                else:
                    logger.error(f"Failed to start instance {instance_id}")
                    self.stop_all()
                    return False
        
        logger.info(f"All {len(self.instances)} instances started successfully")
        return True
    
    def stop_all(self) -> None:
        """Stop all service instances."""
        logger.info("Stopping all instances")
        
        for instance in self.instances:
            instance.stop()
        
        self.instances.clear()
        self.instance_urls.clear()
        self.current_index = 0
    
    def get_next_url(self) -> str:
        """Get next instance URL in round-robin fashion."""
        if not self.instance_urls:
            raise RuntimeError("No instances available")
        
        url = self.instance_urls[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instance_urls)
        return url
    
    def get_all_urls(self) -> List[str]:
        """Get all instance URLs."""
        return self.instance_urls.copy()
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all instances."""
        results = {}
        for instance in self.instances:
            results[f"instance_{instance.instance_id}"] = instance.health_check()
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status."""
        return {
            "total_instances": len(self.instances),
            "devices": self.devices,
            "instances_per_device": self.instances_per_device,
            "instances": [inst.get_info() for inst in self.instances],
            "urls": self.instance_urls,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
        return False
