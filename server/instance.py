# Copyright (c) Opendatalab. All rights reserved.
"""
Service instance management for single MinerU process.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import requests


class ServiceInstance:
    """Manages a single MinerU FastAPI service instance."""
    
    def __init__(
        self,
        device: int,
        port: int,
        instance_id: int = 0,
        backend: str = "hybrid-auto-engine",
        device_mode: str = "npu",
        max_concurrent_requests: int = 10,
        host: str = "127.0.0.1",
    ):
        """
        Initialize service instance.
        
        Args:
            device: NPU/GPU device ID
            port: Port number for this instance
            instance_id: Instance identifier
            backend: MinerU backend type
            device_mode: Device mode (npu/cuda/cpu)
            max_concurrent_requests: Maximum concurrent requests
            host: Host address
        """
        self.device = device
        self.port = port
        self.instance_id = instance_id
        self.backend = backend
        self.device_mode = device_mode
        self.max_concurrent_requests = max_concurrent_requests
        self.host = host
        
        self.process: Optional[subprocess.Popen] = None
        self.url = f"http://{host}:{port}"
        self.is_running = False
        
    def start(self) -> bool:
        """Start the service instance."""
        if self.is_running:
            logger.warning(f"Instance {self.instance_id} on device {self.device} is already running")
            return True
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["MINERU_DEVICE_MODE"] = f"{self.device_mode}:{self.device}"
            env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(self.max_concurrent_requests)
            
            # For NPU, set visible devices
            if self.device_mode == "npu":
                env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device)
            elif self.device_mode == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(self.device)
            
            # Start FastAPI server
            cmd = [
                sys.executable, "-m", "mineru.cli.fast_api",
                "--host", self.host,
                "--port", str(self.port),
            ]
            
            logger.info(f"Starting instance {self.instance_id} on device {self.device}, port {self.port}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to be ready
            if self._wait_for_ready(timeout=60):
                self.is_running = True
                logger.info(f"Instance {self.instance_id} started successfully")
                return True
            else:
                logger.error(f"Instance {self.instance_id} failed to start")
                self.stop()
                return False
                
        except Exception as e:
            logger.exception(f"Failed to start instance {self.instance_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the service instance."""
        if not self.is_running or self.process is None:
            return True
        
        try:
            logger.info(f"Stopping instance {self.instance_id}")
            
            # Try graceful shutdown
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                self.process.kill()
                self.process.wait(timeout=5)
            
            self.is_running = False
            self.process = None
            logger.info(f"Instance {self.instance_id} stopped")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to stop instance {self.instance_id}: {e}")
            return False
    
    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for service to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.url}/docs", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(1)
        
        return False
    
    def health_check(self) -> bool:
        """Check if service is healthy."""
        if not self.is_running:
            return False
        
        try:
            response = requests.get(f"{self.url}/docs", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get instance information."""
        return {
            "instance_id": self.instance_id,
            "device": self.device,
            "port": self.port,
            "url": self.url,
            "backend": self.backend,
            "device_mode": self.device_mode,
            "is_running": self.is_running,
            "pid": self.process.pid if self.process else None,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
