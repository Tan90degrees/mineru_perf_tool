#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
"""
CLI entry point for MinerU Performance Testing Tool.
"""

import click
import asyncio
import sys
from pathlib import Path
from loguru import logger
from typing import List

from .config import Config, BenchmarkConfig, GridSearchParams
from .server import ServerManager
from .client import BenchmarkClient, BatchProcessor
from .monitor import MetricsCollector
from .evaluator import AccuracyEvaluator, ThroughputEvaluator
from .optimizer import GridSearchOptimizer
from .report import ReportGenerator


@click.group(invoke_without_command=True)
@click.option('--version', '-v', is_flag=True, help='Show version')
@click.pass_context
def main(ctx, version):
    """MinerU Performance Testing Tool - Benchmark throughput and accuracy for MinerU."""
    if version:
        from . import __version__
        click.echo(f"mineru-perf-tool version {__version__}")
        return
    
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True), 
              help='Input directory containing PDF/image files')
@click.option('--output-dir', '-o', default='./benchmark_output', type=click.Path(),
              help='Output directory for results')
@click.option('--devices', '-d', default='0', type=str,
              help='NPU device IDs, comma-separated (e.g., 0,1,2,3)')
@click.option('--instances-per-device', default=1, type=int,
              help='Number of instances per device')
@click.option('--concurrency', '-c', default=10, type=int,
              help='Request concurrency')
@click.option('--batch-size', '-b', default=1, type=int,
              help='Batch size (files per request)')
@click.option('--backend', type=click.Choice(['pipeline', 'vlm-auto-engine', 'hybrid-auto-engine', 
                                               'vlm-http-client', 'hybrid-http-client']),
              default='hybrid-auto-engine', help='MinerU backend type')
@click.option('--device-mode', type=click.Choice(['cpu', 'cuda', 'npu', 'mps']),
              default='npu', help='Device mode')
@click.option('--duration', type=int, default=0,
              help='Test duration in seconds (0 = process all files)')
@click.option('--timeout', type=int, default=300,
              help='Request timeout in seconds')
@click.option('--config', '-f', type=click.Path(exists=True),
              help='YAML configuration file')
@click.option('--monitor/--no-monitor', default=True,
              help='Enable/disable resource monitoring')
@click.option('--verbose', '-V', is_flag=True, help='Enable verbose logging')
def run(input_dir, output_dir, devices, instances_per_device, concurrency, batch_size,
       backend, device_mode, duration, timeout, config, monitor, verbose):
    """
    Run single benchmark test.
    
    Example:
        mineru-perf run -i ./data -o ./output --devices 0,1 --concurrency 20
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Parse devices
    device_list = [int(d.strip()) for d in devices.split(',')]
    
    # Load or create config
    if config:
        cfg = Config(config_file=config)
    else:
        cfg = Config()
        cfg.benchmark.devices = device_list
        cfg.benchmark.instances_per_device = instances_per_device
        cfg.benchmark.concurrency = concurrency
        cfg.benchmark.batch_size = batch_size
        cfg.benchmark.backend = backend
        cfg.benchmark.device_mode = device_mode
        cfg.benchmark.duration = duration
        cfg.benchmark.timeout = timeout
        cfg.benchmark.input_dir = str(input_dir)
        cfg.benchmark.output_dir = str(output_dir)
        cfg.benchmark.enable_resource_monitor = monitor
    
    # Validate configuration
    try:
        cfg.validate()
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    
    # Run benchmark
    click.echo(f"Starting benchmark on devices {cfg.benchmark.devices}")
    click.echo(f"Backend: {cfg.benchmark.backend}, Concurrency: {cfg.benchmark.concurrency}")
    
    try:
        results = asyncio.run(_run_benchmark(cfg))
        
        # Generate report
        report_gen = ReportGenerator(output_dir=str(output_dir))
        report_path = report_gen.generate_json_report(results)
        
        click.echo(f"\nBenchmark completed!")
        click.echo(f"QPS: {results['throughput']['qps']:.2f}")
        click.echo(f"Avg Latency: {results['throughput']['avg_latency']:.4f}s")
        click.echo(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def _run_benchmark(cfg: Config) -> dict:
    """Run benchmark with given configuration."""
    
    # Initialize server manager
    base_port = 8000
    server_manager = ServerManager(
        devices=cfg.benchmark.devices,
        instances_per_device=cfg.benchmark.instances_per_device,
        base_port=base_port,
        backend=cfg.benchmark.backend,
        device_mode=cfg.benchmark.device_mode,
        max_concurrent_requests=cfg.benchmark.concurrency,
    )
    
    # Initialize metrics collector
    test_id = f"benchmark_{int(asyncio.get_event_loop().time())}"
    metrics_collector = MetricsCollector(
        test_id=test_id,
        devices=cfg.benchmark.devices,
        device_mode=cfg.benchmark.device_mode,
        monitor_interval=cfg.benchmark.monitor_interval,
    )
    
    try:
        # Start servers
        logger.info("Starting server instances...")
        server_manager.start_all()
        
        # Start monitoring
        if cfg.benchmark.enable_resource_monitor:
            metrics_collector.start()
        
        # Prepare batches
        batch_processor = BatchProcessor(
            input_dir=cfg.benchmark.input_dir,
            batch_size=cfg.benchmark.batch_size,
        )
        
        file_info = batch_processor.get_file_info()
        logger.info(f"Files to process: {file_info}")
        
        # Run benchmark
        server_urls = server_manager.get_all_urls()
        
        async with BenchmarkClient(
            server_urls=server_urls,
            concurrency=cfg.benchmark.concurrency,
            timeout=cfg.benchmark.timeout,
            backend=cfg.benchmark.backend,
        ) as client:
            
            def progress_callback(completed, total, result):
                if completed % 10 == 0:
                    click.echo(f"Progress: {completed}/{total} requests completed")
            
            batches = list(batch_processor.generate_batches())
            results = await client.run_benchmark(
                file_batches=batches,
                progress_callback=progress_callback,
            )
            
            metrics_collector.update_from_benchmark(results)
        
        await metrics_collector.stop()
        
        # Compile final results
        return {
            "test_id": test_id,
            "config": vars(cfg.benchmark),
            "throughput": results,
            "resources": metrics_collector.get_summary(),
        }
        
    finally:
        # Stop servers
        logger.info("Stopping server instances...")
        server_manager.stop_all()


@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Input directory containing files')
@click.option('--output-dir', '-o', default='./grid_search_output', type=click.Path(),
              help='Output directory for results')
@click.option('--devices-range', default='1,2,4', type=str,
              help='Device count range (comma-separated)')
@click.option('--instances-range', default='1,2', type=str,
              help='Instances per device range (comma-separated)')
@click.option('--concurrency-range', default='10,20,50', type=str,
              help='Concurrency range (comma-separated)')
@click.option('--batch-size-range', default='1,5,10', type=str,
              help='Batch size range (comma-separated)')
@click.option('--backend', type=click.Choice(['pipeline', 'vlm-auto-engine', 'hybrid-auto-engine']),
              default='hybrid-auto-engine', help='MinerU backend type')
@click.option('--device-mode', type=click.Choice(['cpu', 'cuda', 'npu', 'mps']),
              default='npu', help='Device mode')
@click.option('--config', '-f', type=click.Path(exists=True),
              help='YAML configuration file')
@click.option('--verbose', '-V', is_flag=True, help='Enable verbose logging')
def grid_search(input_dir, output_dir, devices_range, instances_range, 
                concurrency_range, batch_size_range, backend, device_mode, 
                config, verbose):
    """
    Run grid search for optimal configuration.
    
    Example:
        mineru-perf grid-search -i ./data -o ./results \\
            --devices-range 1,2,4 --concurrency-range 10,20,50
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Parse ranges
    grid_params = GridSearchParams(
        devices_range=[int(d) for d in devices_range.split(',')],
        instances_per_device_range=[int(i) for i in instances_range.split(',')],
        concurrency_range=[int(c) for c in concurrency_range.split(',')],
        batch_size_range=[int(b) for b in batch_size_range.split(',')],
    )
    
    # Create base config
    base_config = BenchmarkConfig(
        backend=backend,
        device_mode=device_mode,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )
    
    # Create optimizer
    optimizer = GridSearchOptimizer(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        grid_params=grid_params,
        base_config=base_config,
    )
    
    # Generate test cases
    test_cases = optimizer.generate_test_cases()
    click.echo(f"Grid search: {len(test_cases)} test cases")
    
    try:
        # Run grid search
        results = optimizer.run()
        
        # Display results
        click.echo("\n=== Grid Search Results ===")
        click.echo(f"Total tests: {results['total_tests']}")
        
        optimal = results.get('optimal_config', {})
        if optimal:
            click.echo(f"\nOptimal configuration:")
            click.echo(f"  Devices: {optimal.get('test_case', {}).get('devices')}")
            click.echo(f"  Instances per device: {optimal.get('test_case', {}).get('instances_per_device')}")
            click.echo(f"  Concurrency: {optimal.get('test_case', {}).get('concurrency')}")
            click.echo(f"  Batch size: {optimal.get('test_case', {}).get('batch_size')}")
            click.echo(f"  QPS: {optimal.get('metric_value', 0):.2f}")
        
    except Exception as e:
        logger.exception(f"Grid search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--prediction-dir', '-p', required=True, type=click.Path(exists=True),
              help='Directory containing prediction markdown files')
@click.option('--omnidocbench-path', '-o', required=True, type=click.Path(exists=True),
              help='Path to OmniDocBench directory')
@click.option('--ground-truth', '-g', type=click.Path(exists=True),
              help='Path to ground truth JSON file')
@click.option('--output-dir', default='./eval_output', type=click.Path(),
              help='Output directory for evaluation results')
@click.option('--match-method', type=click.Choice(['quick_match', 'simple_match', 'no_split']),
              default='quick_match', help='Matching method')
@click.option('--verbose', '-V', is_flag=True, help='Enable verbose logging')
def evaluate(prediction_dir, omnidocbench_path, ground_truth, output_dir, 
             match_method, verbose):
    """
    Run OmniDocBench accuracy evaluation.
    
    Example:
        mineru-perf evaluate -p ./predictions -o ./OmniDocBench -g ./OmniDocBench.json
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    click.echo("Starting accuracy evaluation...")
    
    try:
        # Create evaluator
        evaluator = AccuracyEvaluator(
            omnidocbench_path=str(omnidocbench_path),
            output_dir=str(output_dir),
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            prediction_dir=str(prediction_dir),
            ground_truth_path=str(ground_truth) if ground_truth else None,
            match_method=match_method,
        )
        
        if results.get("success"):
            # Display results
            click.echo("\n=== Evaluation Results ===")
            
            overall_score = evaluator.get_overall_score(results)
            click.echo(f"Overall Score: {overall_score:.2f}")
            
            eval_results = results.get("results", {})
            
            for component, metrics in eval_results.items():
                if isinstance(metrics, dict):
                    click.echo(f"\n{component}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            click.echo(f"  {metric}: {value:.4f}")
            
        else:
            click.echo(f"Evaluation failed: {results.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--device-mode', type=click.Choice(['cpu', 'cuda', 'npu', 'mps']),
              default='npu', help='Device mode to check')
def check(device_mode):
    """Check environment and device availability."""
    from .utils import NPUManager, get_npu_info, check_npu_health
    
    click.echo("=== Environment Check ===\n")
    
    # Check device availability
    click.echo(f"Device mode: {device_mode}")
    
    if device_mode == "npu":
        info = get_npu_info()
        health = check_npu_health()
        
        click.echo(f"NPU available: {info['available']}")
        click.echo(f"Device count: {info['device_count']}")
        
        if info['available']:
            click.echo("\nDevice details:")
            for device in info['devices']:
                click.echo(f"  Device {device['device_id']}: {device['name']}")
                click.echo(f"    Memory: {device['memory']['total_gb']:.2f} GB total, "
                          f"{device['memory']['free_gb']:.2f} GB free")
                click.echo(f"    Utilization: {device['utilization']:.1f}%")
            
            click.echo("\nHealth check:")
            for key, value in health.items():
                click.echo(f"  {key}: {'✓' if value else '✗'}")
    
    elif device_mode == "cuda":
        try:
            import torch
            click.echo(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                click.echo(f"Device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    click.echo(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            click.echo("PyTorch not installed")
    
    elif device_mode == "cpu":
        import multiprocessing
        click.echo(f"CPU cores: {multiprocessing.cpu_count()}")
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            click.echo(f"Memory: {mem.total / (1024**3):.2f} GB total, "
                      f"{mem.available / (1024**3):.2f} GB available")
        except ImportError:
            click.echo("psutil not installed")


if __name__ == '__main__':
    main()
