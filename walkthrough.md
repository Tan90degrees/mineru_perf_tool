# Walkthrough: MinerU Throughput Evaluation Tool

## Changes Made
This project implements a fully automated, asynchronous grid-search throughput evaluation tool for MinerU. The code is located in `mineru_perf_tool`.
The framework handles starting, stopping, and routing requests to one or more MinerU instances dynamically based on grid search parameters. 

**Module Details:**
- [config.py](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/config.py): Defines [BenchmarkConfig](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/config.py#4-30) the schema for tracking configuration and the input parameters.
- [server_manager.py](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/server_manager.py): Responsible for allocating specific free ports and isolating instances to NPU cards via the `ASCEND_RT_VISIBLE_DEVICES` environment variable, ensuring resources aren't overloaded or leaked across tests.
- [client.py](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/client.py): Uses `aiohttp` to perform robust asynchronous evaluation concurrently. Files are loaded from the given directory and requests properly utilize the [files](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/client.py#144-146) array to process `batch_sizes` per request correctly over `/file_parse`.
- [benchmark.py](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/benchmark.py): The evaluation coordinator that reads CLI arguments, runs nested loops mapping (instances_per_card * concurrency * batch_size), collects metrics generated from the client class (`total_time`, `avg_latency`, `p90`, `success_rate`) and writes out to a CSV automatically. 
- [requirements.txt](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/requirements.txt): Outlines the 3 basic dependencies (`pydantic`, `aiohttp`, `pandas`).

## How to use

Ensure dependencies are installed and MinerU is configured accurately on the NPU container/host.
```bash
cd mineru_perf_tool
pip install -r requirements.txt
python benchmark.py --data_dir /path/to/pdfs \
  --cards 0 1 \
  --instances_per_card 1 2 \
  --concurrency 1 2 4 \
  --batch_size 1 2 \
  --backend pipeline --lang ch \
  --output_csv results.csv 
```

## Validation Results
Since the execution requires actual NPU and MinerU packages (`mineru.cli.fast_api`), I introduced a test mock server ([mock_server.py](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/mineru_perf_tool/mock_server.py)) and a `--mock` validation dimension to trace the logic of the grid-search implementation without needing an NPU. 
The mock mode correctly provisions instances onto separate ports, dispatches requests synchronously mapping [(instances x concurrency x batch_size)](file:///C:/Users/imyut/WorkSpace/github.com/mineru_perf_2/MinerU/projects/multi_gpu_v2/client.py#37-79) boundaries, evaluates the results, cleans up instances appropriately to prevent process leakage, and logs correct throughput logic into CSV formatting. 

To run the mock parameter test on any Windows or CPU system, use:
```bash
python benchmark.py --data_dir /path/to/pdfs \
  --cards 0 1 \
  --instances_per_card 1 2 \
  --concurrency 1 2 4 \
  --batch_size 1 2 \
  --backend pipeline \
  --mock \
  --output_csv mock_results.csv 
```
