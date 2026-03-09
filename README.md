# MinerU Performance Testing Tool

A comprehensive benchmarking tool for MinerU document parsing service, supporting throughput and accuracy evaluation with NPU/GPU/CPU environments.

## Features

- **Multi-device Support**: Run benchmarks on NPU, GPU, or CPU devices
- **Multi-instance Deployment**: Support for multiple instances per device
- **Multiple Backends**: Pipeline, VLM, and Hybrid mode support
- **Throughput Testing**: Measure QPS, latency distribution, and resource utilization
- **Accuracy Evaluation**: Integration with OmniDocBench for precision metrics
- **Grid Search Optimization**: Automatically find optimal configuration parameters
- **Comprehensive Reports**: JSON, CSV, and Markdown reports with visualization

## Installation

```bash
# Basic installation
pip install -e .

# With NPU support
pip install -e .[npu]

# With CUDA support
pip install -e .[cuda]

# For development
pip install -e .[dev]
```

## Quick Start

### 1. Check Environment

```bash
mineru-perf check --device-mode npu
```

### 2. Run Single Benchmark

```bash
mineru-perf run \
    --input-dir ./data \
    --output-dir ./output \
    --devices 0,1 \
    --concurrency 20 \
    --backend hybrid-auto-engine
```

### 3. Grid Search for Optimal Configuration

```bash
mineru-perf grid-search \
    --input-dir ./data \
    --output-dir ./results \
    --devices-range 1,2,4 \
    --concurrency-range 10,20,50
```

### 4. Accuracy Evaluation

```bash
mineru-perf evaluate \
    --prediction-dir ./predictions \
    --omnidocbench-path ./OmniDocBench \
    --ground-truth ./OmniDocBench/OmniDocBench.json
```

## Configuration

Configuration files are located in `configs/` directory:

- `default.yaml` - Default benchmark configuration
- `omnidocbench.yaml` - Accuracy evaluation configuration
- `grid_search.yaml` - Grid search optimization configuration

## Project Structure

```
mineru_perf_tool/
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── server/             # Server management
│   ├── manager.py      # Multi-instance manager
│   └── instance.py     # Single instance wrapper
├── client/             # Benchmark client
│   ├── benchmark.py    # Async benchmark client
│   └── batch.py        # File batching
├── monitor/            # Resource monitoring
│   ├── resource.py     # CPU/Memory/GPU monitor
│   └── collector.py    # Metrics collection
├── evaluator/          # Evaluation modules
│   ├── accuracy.py     # OmniDocBench integration
│   └── throughput.py   # Throughput metrics
├── optimizer/          # Grid search optimization
│   ├── grid_search.py  # Grid search engine
│   └── result.py       # Result aggregation
├── report/             # Report generation
│   ├── generator.py    # Report generator
│   └── visualization.py # Visualization plots
├── utils/              # Utilities
│   ├── file_utils.py   # File operations
│   └── npu_utils.py    # NPU device utilities
└── configs/            # Configuration files
    ├── default.yaml
    ├── omnidocbench.yaml
    └── grid_search.yaml
```

## Usage Examples

### Basic Throughput Test

```bash
# Test with default settings
mineru-perf run -i ./pdfs -o ./output

# Test with specific devices and concurrency
mineru-perf run \
    -i ./pdfs \
    -o ./output \
    --devices 0,1 \
    --concurrency 50 \
    --backend pipeline
```

### Grid Search Optimization

```bash
# Search for optimal configuration
mineru-perf grid-search \
    -i ./pdfs \
    -o ./results \
    --devices-range 1,2,4 \
    --instances-range 1,2 \
    --concurrency-range 10,20,50 \
    --batch-size-range 1,5
```

### OmniDocBench Evaluation

```bash
# Run accuracy evaluation
mineru-perf evaluate \
    -p ./predictions \
    -o ./OmniDocBench \
    -g ./OmniDocBench/OmniDocBench.json \
    --match-method quick_match
```

## Metrics Collected

### Throughput Metrics
- QPS (Queries Per Second)
- Success Rate
- Latency Distribution (P50, P90, P95, P99)
- Data Transfer Rate

### Resource Metrics
- CPU Usage (%)
- Memory Usage (%)
- NPU/GPU Memory Allocated
- NPU/GPU Utilization

### Accuracy Metrics (OmniDocBench)
- Text Edit Distance
- Formula CDM Score
- Table TEDS Score
- Reading Order Accuracy

## Requirements

- Python 3.8+
- MinerU (installed separately)
- For NPU: torch-npu
- For GPU: PyTorch with CUDA

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
