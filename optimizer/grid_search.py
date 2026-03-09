# Copyright (c) Opendatalab. All rights reserved.
"""
Grid search optimization for parameter tuning.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from loguru import logger
from dataclasses import asdict

from ..config import GridSearchParams, BenchmarkConfig
from ..server import ServerManager
from ..client import BenchmarkClient, BatchProcessor
from ..monitor import MetricsCollector
from .result import ResultAggregator


class GridSearchOptimizer:
    """
    Grid search optimizer for finding optimal configuration.
    
    Searches across:
    - Number of devices
    - Instances per device
    - Request concurrency
    - Batch size
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        grid_params: GridSearchParams,
        base_config: BenchmarkConfig,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            input_dir: Input directory with test files
            output_dir: Output directory for results
            grid_params: Grid search parameters
            base_config: Base benchmark configuration
            progress_callback: Progress callback function
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.grid_params = grid_params
        self.base_config = base_config
        self.progress_callback = progress_callback
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.result_aggregator = ResultAggregator(output_dir=str(self.output_dir))
        
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate all test case configurations."""
        return self.grid_params.generate_combinations()
    
    async def run_single_test(
        self,
        test_case: Dict[str, Any],
        test_id: str,
    ) -> Dict[str, Any]:
        """
        Run a single test with given configuration.
        
        Args:
            test_case: Test case parameters
            test_id: Unique test identifier
            
        Returns:
            Test results
        """
        logger.info(f"Running test {test_id}: {test_case}")
        
        # Update config for this test
        config = BenchmarkConfig(**asdict(self.base_config))
        config.devices = test_case["devices"]
        config.instances_per_device = test_case["instances_per_device"]
        config.concurrency = test_case["concurrency"]
        config.batch_size = test_case["batch_size"]
        
        # Start server instances
        base_port = 8000
        server_manager = ServerManager(
            devices=config.devices,
            instances_per_device=config.instances_per_device,
            base_port=base_port,
            backend=config.backend,
            device_mode=config.device_mode,
            max_concurrent_requests=config.concurrency,
        )
        
        test_output_dir = self.output_dir / test_id
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Start servers
            server_manager.start_all()
            
            # Initialize monitoring
            metrics_collector = MetricsCollector(
                test_id=test_id,
                devices=config.devices,
                device_mode=config.device_mode,
                monitor_interval=config.monitor_interval,
            )
            metrics_collector.start()
            
            # Prepare batches
            batch_processor = BatchProcessor(
                input_dir=str(self.input_dir),
                batch_size=config.batch_size,
            )
            
            # Run benchmark
            server_urls = server_manager.get_all_urls()
            
            async with BenchmarkClient(
                server_urls=server_urls,
                concurrency=config.concurrency,
                timeout=config.timeout,
                backend=config.backend,
            ) as client:
                batches = list(batch_processor.generate_batches())
                
                results = await client.run_benchmark(
                    file_batches=batches,
                    progress_callback=lambda c, t, r: self._on_progress(test_id, c, t, r),
                )
                
                metrics_collector.update_from_benchmark(results)
            
            await metrics_collector.stop()
            
            # Compile results
            test_result = {
                "test_id": test_id,
                "test_case": test_case,
                "config": asdict(config),
                "throughput": results,
                "resources": metrics_collector.get_summary(),
                "timestamp": time.time(),
            }
            
            # Save results
            self.result_aggregator.add_result(test_result)
            
            logger.info(f"Test {test_id} completed: QPS={results.get('qps', 0):.2f}")
            
            return test_result
            
        except Exception as e:
            logger.exception(f"Test {test_id} failed: {e}")
            return {
                "test_id": test_id,
                "test_case": test_case,
                "error": str(e),
                "timestamp": time.time(),
            }
            
        finally:
            # Stop servers
            server_manager.stop_all()
    
    def _on_progress(self, test_id: str, completed: int, total: int, result: Any) -> None:
        """Handle progress updates."""
        if self.progress_callback:
            self.progress_callback(test_id, completed, total, result)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test cases in the grid search.
        
        Returns:
            Aggregated results
        """
        test_cases = self.generate_test_cases()
        total_tests = len(test_cases)
        
        logger.info(f"Starting grid search with {total_tests} test cases")
        
        results = []
        for i, test_case in enumerate(test_cases):
            test_id = f"test_{i+1:04d}"
            
            logger.info(f"Running test {i+1}/{total_tests}")
            
            result = await self.run_single_test(test_case, test_id)
            results.append(result)
            
            # Progress update
            if self.progress_callback:
                self.progress_callback(
                    test_id="overall",
                    completed=i + 1,
                    total=total_tests,
                    result=result,
                )
        
        # Find optimal configuration
        optimal_config = self.result_aggregator.find_optimal(results)
        
        # Generate report
        report = self.result_aggregator.generate_report()
        
        logger.info(f"Grid search completed. Optimal config: {optimal_config}")
        
        return {
            "total_tests": total_tests,
            "results": results,
            "optimal_config": optimal_config,
            "report": report,
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Synchronous entry point for grid search.
        
        Returns:
            Grid search results
        """
        return asyncio.run(self.run_all_tests())
