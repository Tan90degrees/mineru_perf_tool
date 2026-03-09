# Copyright (c) Opendatalab. All rights reserved.
"""
MinerU Performance Testing Tool

A comprehensive throughput and accuracy benchmarking tool for MinerU document parsing service.
Supports NPU/GPU/CPU environments with multi-card multi-instance deployment.
"""

__version__ = "1.0.0"
__author__ = "OpenDataLab"

from .config import Config, BenchmarkConfig, GridSearchParams
from .cli import main

__all__ = [
    "Config",
    "BenchmarkConfig", 
    "GridSearchParams",
    "main",
]
