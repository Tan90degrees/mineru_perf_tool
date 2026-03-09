# Copyright (c) Opendatalab. All rights reserved.
"""
Metrics collection and aggregation.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from .resource import ResourceMonitor, ResourceSnapshot


@dataclass
class TestMetrics:
    """Collected metrics for a test run."""
    test_id: str
    start_time: float
    end_time: float = 0.0
    
    # Throughput metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    qps: float = 0.0
    
    # Latency metrics (seconds)
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # Data transfer
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    throughput_mbps: float = 0.0
    
    # Resource metrics
    cpu_avg_percent: float = 0.0
    memory_avg_percent: float = 0.0
    memory_max_percent: float = 0.0
    device_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Resource snapshots
    resource_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "duration_seconds": self.end_time - self.start_time if self.end_time > 0 else 0,
            "throughput": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "qps": self.qps,
                "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            },
            "latency": {
                "avg": self.avg_latency,
                "min": self.min_latency,
                "max": self.max_latency,
                "p50": self.p50_latency,
                "p90": self.p90_latency,
                "p95": self.p95_latency,
                "p99": self.p99_latency,
            },
            "data_transfer": {
                "total_bytes_sent": self.total_bytes_sent,
                "total_bytes_received": self.total_bytes_received,
                "throughput_mbps": self.throughput_mbps,
            },
            "resources": {
                "cpu_avg_percent": self.cpu_avg_percent,
                "memory_avg_percent": self.memory_avg_percent,
                "memory_max_percent": self.memory_max_percent,
                "devices": self.device_metrics,
            },
        }


class MetricsCollector:
    """Collect and aggregate metrics from various sources."""
    
    def __init__(
        self,
        test_id: str,
        devices: List[int],
        device_mode: str = "npu",
        monitor_interval: float = 1.0,
    ):
        """
        Initialize metrics collector.
        
        Args:
            test_id: Unique test identifier
            devices: List of device IDs to monitor
            device_mode: Device type (npu/cuda/cpu)
            monitor_interval: Resource monitoring interval
        """
        self.test_id = test_id
        self.devices = devices
        self.device_mode = device_mode
        
        self.metrics = TestMetrics(test_id=test_id, start_time=time.time())
        self.resource_monitor = ResourceMonitor(
            devices=devices,
            device_mode=device_mode,
            interval=monitor_interval,
        )
        
        self._started = False
    
    def start(self) -> None:
        """Start metrics collection."""
        if self._started:
            return
        
        self.metrics.start_time = time.time()
        self.resource_monitor.start()
        self._started = True
        logger.info(f"Metrics collection started for test {self.test_id}")
    
    async def stop(self) -> None:
        """Stop metrics collection."""
        if not self._started:
            return
        
        self.metrics.end_time = time.time()
        await self.resource_monitor.stop()
        
        # Aggregate resource metrics
        resource_summary = self.resource_monitor.get_summary()
        self.metrics.cpu_avg_percent = resource_summary.get("cpu_avg_percent", 0.0)
        self.metrics.memory_avg_percent = resource_summary.get("memory_avg_percent", 0.0)
        self.metrics.memory_max_percent = resource_summary.get("memory_max_percent", 0.0)
        self.metrics.device_metrics = resource_summary.get("devices", {})
        
        # Store snapshots for detailed analysis
        self.metrics.resource_snapshots = self.resource_monitor.export_snapshots()
        
        self._started = False
        logger.info(f"Metrics collection stopped for test {self.test_id}")
    
    def update_from_benchmark(self, benchmark_summary: Dict[str, Any]) -> None:
        """Update metrics from benchmark summary."""
        self.metrics.total_requests = benchmark_summary.get("total_requests", 0)
        self.metrics.successful_requests = benchmark_summary.get("successful_requests", 0)
        self.metrics.failed_requests = benchmark_summary.get("failed_requests", 0)
        self.metrics.qps = benchmark_summary.get("qps", 0.0)
        
        self.metrics.avg_latency = benchmark_summary.get("avg_latency", 0.0)
        self.metrics.min_latency = benchmark_summary.get("min_latency", 0.0)
        self.metrics.max_latency = benchmark_summary.get("max_latency", 0.0)
        self.metrics.p50_latency = benchmark_summary.get("p50_latency", 0.0)
        self.metrics.p90_latency = benchmark_summary.get("p90_latency", 0.0)
        self.metrics.p95_latency = benchmark_summary.get("p95_latency", 0.0)
        self.metrics.p99_latency = benchmark_summary.get("p99_latency", 0.0)
        
        self.metrics.total_bytes_sent = benchmark_summary.get("total_bytes_sent", 0)
        self.metrics.total_bytes_received = benchmark_summary.get("total_bytes_received", 0)
        self.metrics.throughput_mbps = benchmark_summary.get("throughput_mbps", 0.0)
    
    def get_metrics(self) -> TestMetrics:
        """Get current metrics."""
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary."""
        return self.metrics.to_dict()
