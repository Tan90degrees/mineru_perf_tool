# Copyright (c) Opendatalab. All rights reserved.
"""
Configuration management for MinerU performance testing tool.
"""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    
    # Device settings
    devices: List[int] = field(default_factory=lambda: [0])
    instances_per_device: int = 1
    
    # Client settings
    concurrency: int = 10
    batch_size: int = 1
    timeout: int = 300  # seconds
    
    # Backend settings
    backend: str = "hybrid-auto-engine"  # pipeline, vlm-auto-engine, hybrid-auto-engine
    device_mode: str = "npu"  # cpu, cuda, npu, mps
    
    # Test settings
    input_dir: Optional[str] = None
    output_dir: str = "./benchmark_output"
    duration: int = 60  # seconds, 0 means run until all files processed
    
    # OmniDocBench mode
    mode: str = "throughput"  # throughput, omnidocbench
    omnidocbench_path: Optional[str] = None
    omnidocbench_config: Optional[str] = None
    
    # Monitoring settings
    monitor_interval: float = 1.0  # seconds
    enable_resource_monitor: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.devices:
            raise ValueError("devices cannot be empty")
        
        if self.instances_per_device < 1:
            raise ValueError("instances_per_device must be >= 1")
        
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.backend not in ["pipeline", "vlm-auto-engine", "hybrid-auto-engine", 
                                  "vlm-http-client", "hybrid-http-client"]:
            raise ValueError(f"Invalid backend: {self.backend}")
        
        if self.mode == "omnidocbench" and not self.omnidocbench_path:
            raise ValueError("omnidocbench_path is required when mode is 'omnidocbench'")


@dataclass
class GridSearchParams:
    """Parameters for grid search optimization."""
    
    devices_range: List[int] = field(default_factory=lambda: [1, 2, 4])
    instances_per_device_range: List[int] = field(default_factory=lambda: [1, 2])
    concurrency_range: List[int] = field(default_factory=lambda: [10, 20, 50])
    batch_size_range: List[int] = field(default_factory=lambda: [1, 5, 10])
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        combinations = []
        for devices_count in self.devices_range:
            # For devices_count, we use first N devices
            devices = list(range(devices_count))
            
            for instances, concurrency, batch_size in itertools.product(
                self.instances_per_device_range,
                self.concurrency_range,
                self.batch_size_range
            ):
                combinations.append({
                    "devices": devices,
                    "instances_per_device": instances,
                    "concurrency": concurrency,
                    "batch_size": batch_size,
                })
        
        return combinations


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.benchmark = BenchmarkConfig()
        self.grid_search: Optional[GridSearchParams] = None
        self.server: Dict[str, Any] = {}
        self.monitor: Dict[str, Any] = {}
        self.report: Dict[str, Any] = {}
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Load benchmark config
        if 'benchmark' in data:
            for key, value in data['benchmark'].items():
                if hasattr(self.benchmark, key):
                    setattr(self.benchmark, key, value)
        
        # Load grid search params
        if 'grid_search' in data:
            self.grid_search = GridSearchParams()
            for key, value in data['grid_search'].items():
                if hasattr(self.grid_search, key):
                    setattr(self.grid_search, key, value)
        
        # Load other sections
        self.server = data.get('server', {})
        self.monitor = data.get('monitor', {})
        self.report = data.get('report', {})
    
    def save_to_file(self, config_file: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'benchmark': {
                'devices': self.benchmark.devices,
                'instances_per_device': self.benchmark.instances_per_device,
                'concurrency': self.benchmark.concurrency,
                'batch_size': self.benchmark.batch_size,
                'timeout': self.benchmark.timeout,
                'backend': self.benchmark.backend,
                'device_mode': self.benchmark.device_mode,
                'input_dir': self.benchmark.input_dir,
                'output_dir': self.benchmark.output_dir,
                'duration': self.benchmark.duration,
                'mode': self.benchmark.mode,
                'omnidocbench_path': self.benchmark.omnidocbench_path,
                'monitor_interval': self.benchmark.monitor_interval,
                'enable_resource_monitor': self.benchmark.enable_resource_monitor,
            },
            'server': self.server,
            'monitor': self.monitor,
            'report': self.report,
        }
        
        if self.grid_search:
            data['grid_search'] = {
                'devices_range': self.grid_search.devices_range,
                'instances_per_device_range': self.grid_search.instances_per_device_range,
                'concurrency_range': self.grid_search.concurrency_range,
                'batch_size_range': self.grid_search.batch_size_range,
            }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> None:
        """Validate all configurations."""
        self.benchmark.validate()
