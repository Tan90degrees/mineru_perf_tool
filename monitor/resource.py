# Copyright (c) Opendatalab. All rights reserved.
"""
Resource monitoring for CPU, memory, and NPU/GPU devices.
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not installed, CPU/memory monitoring disabled")

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_npu
except ImportError:
    torch_npu = None


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    
    # Device-specific metrics
    device_metrics: Dict[int, Dict[str, Any]] = None  # device_id -> metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "device_metrics": self.device_metrics or {},
        }


class ResourceMonitor:
    """Monitor system and device resources."""
    
    def __init__(
        self,
        devices: List[int],
        device_mode: str = "npu",
        interval: float = 1.0,
    ):
        """
        Initialize resource monitor.
        
        Args:
            devices: List of device IDs to monitor
            device_mode: Device type (npu/cuda/cpu)
            interval: Monitoring interval in seconds
        """
        self.devices = devices
        self.device_mode = device_mode
        self.interval = interval
        
        self.snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    def _get_cpu_memory(self) -> tuple:
        """Get CPU and memory usage."""
        if psutil is None:
            return 0.0, 0.0, 0.0, 0.0
        
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        return (
            cpu_percent,
            memory.percent,
            memory.used / (1024**3),  # GB
            memory.available / (1024**3),  # GB
        )
    
    def _get_device_metrics(self, device_id: int) -> Dict[str, Any]:
        """Get device-specific metrics."""
        metrics = {
            "device_id": device_id,
            "device_mode": self.device_mode,
            "available": False,
            "memory_allocated_gb": 0.0,
            "memory_reserved_gb": 0.0,
            "memory_free_gb": 0.0,
            "utilization_percent": 0.0,
        }
        
        if self.device_mode == "cpu":
            metrics["available"] = True
            return metrics
        
        if torch is None:
            return metrics
        
        try:
            if self.device_mode == "npu":
                if torch_npu is None:
                    return metrics
                
                # NPU device
                device = torch.device(f"npu:{device_id}")
                if torch.npu.is_available() and device_id < torch.npu.device_count():
                    metrics["available"] = True
                    metrics["memory_allocated_gb"] = torch.npu.memory_allocated(device) / (1024**3)
                    metrics["memory_reserved_gb"] = torch.npu.memory_reserved(device) / (1024**3)
                    # NPU utilization would need npu-smi command
                    metrics["utilization_percent"] = self._get_npu_utilization(device_id)
                    
            elif self.device_mode in ["cuda", "mps"]:
                # CUDA or MPS device
                device = torch.device(f"{self.device_mode}:{device_id}")
                if self.device_mode == "cuda" and torch.cuda.is_available():
                    if device_id < torch.cuda.device_count():
                        metrics["available"] = True
                        metrics["memory_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
                        metrics["memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
                        metrics["utilization_percent"] = 0.0  # Would need nvidia-smi
                        
        except Exception as e:
            logger.warning(f"Failed to get device {device_id} metrics: {e}")
        
        return metrics
    
    def _get_npu_utilization(self, device_id: int) -> float:
        """Get NPU utilization using npu-smi command."""
        try:
            import subprocess
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(device_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse output for utilization
            # This is a simplified version, actual parsing depends on npu-smi output format
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get NPU utilization: {e}")
            return 0.0
    
    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        cpu_percent, mem_percent, mem_used, mem_avail = self._get_cpu_memory()
        
        device_metrics = {}
        for device_id in self.devices:
            device_metrics[device_id] = self._get_device_metrics(device_id)
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=mem_percent,
            memory_used_gb=mem_used,
            memory_available_gb=mem_avail,
            device_metrics=device_metrics,
        )
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(f"Resource monitor started with interval {self.interval}s")
        
        while self._running:
            snapshot = self.take_snapshot()
            self.snapshots.append(snapshot)
            
            # Log periodically
            if len(self.snapshots) % 10 == 0:
                logger.debug(
                    f"Resource snapshot: CPU={snapshot.cpu_percent:.1f}%, "
                    f"Memory={snapshot.memory_percent:.1f}%, "
                    f"Memory Used={snapshot.memory_used_gb:.2f}GB"
                )
            
            await asyncio.sleep(self.interval)
    
    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitor started")
    
    async def stop(self) -> None:
        """Stop monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Resource monitor stopped. Collected {len(self.snapshots)} snapshots")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from collected snapshots."""
        if not self.snapshots:
            return {"error": "No snapshots collected"}
        
        # Calculate averages
        avg_cpu = sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)
        avg_mem = sum(s.memory_percent for s in self.snapshots) / len(self.snapshots)
        max_mem = max(s.memory_percent for s in self.snapshots)
        
        # Device-specific summaries
        device_summaries = {}
        for device_id in self.devices:
            device_snaps = [s.device_metrics.get(device_id, {}) for s in self.snapshots if s.device_metrics]
            if device_snaps:
                device_summaries[f"device_{device_id}"] = {
                    "avg_memory_allocated_gb": sum(d.get("memory_allocated_gb", 0) for d in device_snaps) / len(device_snaps),
                    "max_memory_allocated_gb": max(d.get("memory_allocated_gb", 0) for d in device_snaps),
                    "avg_utilization_percent": sum(d.get("utilization_percent", 0) for d in device_snaps) / len(device_snaps),
                }
        
        return {
            "monitoring_duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "snapshot_count": len(self.snapshots),
            "interval_seconds": self.interval,
            "cpu_avg_percent": avg_cpu,
            "memory_avg_percent": avg_mem,
            "memory_max_percent": max_mem,
            "devices": device_summaries,
        }
    
    def export_snapshots(self) -> List[Dict[str, Any]]:
        """Export all snapshots as dictionaries."""
        return [s.to_dict() for s in self.snapshots]
