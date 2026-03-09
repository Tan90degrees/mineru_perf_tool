# Copyright (c) Opendatalab. All rights reserved.
"""Client module for benchmarking MinerU service."""

from .benchmark import BenchmarkClient
from .batch import BatchProcessor

__all__ = ["BenchmarkClient", "BatchProcessor"]
