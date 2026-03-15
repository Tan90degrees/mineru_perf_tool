from pydantic import BaseModel
from typing import List, Optional

class BenchmarkConfig(BaseModel):
    # NPU cards to use, e.g. [0, 1] means use card 0 and 1
    cards: List[int] = [0]
    
    # Grid search parameters
    instances_per_card: List[int] = [1]
    batch_size_list: List[int] = [1]
    
    # MinerU processing parameters
    backend: str = "pipeline" # pipeline, vlm-auto-engine, hybrid-auto-engine
    parse_method: str = "auto"
    lang: str = "ch"
    
    # Input data
    data_dir: str
    output_dir: str = "./benchmark_results"
    
    # Server configuration
    server_port_start: int = 8000
    host: str = "127.0.0.1"
    
    # Test control
    warmup_requests: int = 2
    max_requests_per_test: Optional[int] = None # If None, process all files in data_dir
    mock_mode: bool = False # Use mock HTTP server for testing validation
    ipc_mode: bool = False  # Use direct IPC workers (no HTTP, no network overhead)
