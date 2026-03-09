# Copyright (c) Opendatalab. All rights reserved.
"""
Throughput evaluation and analysis.
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from ..client import BenchmarkClient, BatchProcessor
from ..monitor import MetricsCollector


class ThroughputEvaluator:
    """
    Evaluate MinerU throughput performance.
    
    Provides:
    - Request throughput (QPS)
    - Latency distribution
    - Resource utilization during load
    - Scalability analysis
    """
    
    def __init__(
        self,
        server_urls: List[str],
        concurrency: int = 10,
        batch_size: int = 1,
        timeout: int = 300,
        backend: str = "hybrid-auto-engine",
        devices: List[int] = None,
        device_mode: str = "npu",
        monitor_interval: float = 1.0,
    ):
        """
        Initialize throughput evaluator.
        
        Args:
            server_urls: List of MinerU server URLs
            concurrency: Maximum concurrent requests
            batch_size: Number of files per batch
            timeout: Request timeout in seconds
            backend: MinerU backend type
            devices: List of device IDs to monitor
            device_mode: Device type (npu/cuda/cpu)
            monitor_interval: Resource monitoring interval
        """
        self.server_urls = server_urls
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.timeout = timeout
        self.backend = backend
        self.devices = devices or [0]
        self.device_mode = device_mode
        self.monitor_interval = monitor_interval
    
    async def evaluate(
        self,
        input_dir: str,
        duration: int = 0,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run throughput evaluation.
        
        Args:
            input_dir: Input directory containing files
            duration: Test duration in seconds (0 = run until all files processed)
            progress_callback: Optional progress callback
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting throughput evaluation: concurrency={self.concurrency}, batch_size={self.batch_size}")
        
        # Initialize batch processor
        batch_processor = BatchProcessor(
            input_dir=input_dir,
            batch_size=self.batch_size,
        )
        
        file_info = batch_processor.get_file_info()
        logger.info(f"Files to process: {file_info}")
        
        # Initialize metrics collector
        test_id = f"throughput_{int(time.time())}"
        metrics_collector = MetricsCollector(
            test_id=test_id,
            devices=self.devices,
            device_mode=self.device_mode,
            monitor_interval=self.monitor_interval,
        )
        
        # Start monitoring
        metrics_collector.start()
        
        # Run benchmark
        results = {}
        try:
            async with BenchmarkClient(
                server_urls=self.server_urls,
                concurrency=self.concurrency,
                timeout=self.timeout,
                backend=self.backend,
            ) as client:
                # Prepare batches
                batches = list(batch_processor.generate_batches())
                
                if duration > 0:
                    # Duration-based test
                    results = await self._run_duration_test(
                        client=client,
                        batch_processor=batch_processor,
                        duration=duration,
                        progress_callback=progress_callback,
                    )
                else:
                    # File-based test
                    results = await client.run_benchmark(
                        file_batches=batches,
                        progress_callback=progress_callback,
                    )
                
                # Update metrics
                metrics_collector.update_from_benchmark(results)
                
        finally:
            # Stop monitoring
            await metrics_collector.stop()
        
        # Compile final results
        final_results = {
            "test_id": test_id,
            "config": {
                "server_urls": self.server_urls,
                "concurrency": self.concurrency,
                "batch_size": self.batch_size,
                "timeout": self.timeout,
                "backend": self.backend,
                "devices": self.devices,
                "device_mode": self.device_mode,
            },
            "file_info": file_info,
            "throughput": results,
            "resources": metrics_collector.get_summary(),
            "metrics": metrics_collector.get_summary(),
        }
        
        logger.info(f"Throughput evaluation completed: QPS={results.get('qps', 0):.2f}")
        
        return final_results
    
    async def _run_duration_test(
        self,
        client: BenchmarkClient,
        batch_processor: BatchProcessor,
        duration: int,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run duration-based throughput test.
        
        Args:
            client: Benchmark client
            batch_processor: Batch processor
            duration: Test duration in seconds
            progress_callback: Progress callback
            
        Returns:
            Test results
        """
        logger.info(f"Running duration-based test for {duration} seconds")
        
        start_time = time.time()
        completed_requests = 0
        total_requests = 0
        
        # Reset batch processor
        batch_processor.reset()
        
        while time.time() - start_time < duration:
            # Get next batch
            batch = batch_processor.get_next_batch()
            if not batch:
                # Wrap around if we run out of files
                batch_processor.reset()
                batch = batch_processor.get_next_batch()
            
            if not batch:
                break
            
            # Send requests for this batch
            tasks = []
            for file_path in batch:
                task = client.send_request(file_path)
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            
            completed_requests += sum(1 for r in results if r.success)
            total_requests += len(results)
            
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(
                    elapsed,
                    duration,
                    {
                        "completed": completed_requests,
                        "total": total_requests,
                    }
                )
        
        # Get final stats
        return client.get_stats()
    
    def analyze_scalability(
        self,
        input_dir: str,
        concurrency_range: List[int],
        duration: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Analyze scalability across different concurrency levels.
        
        Args:
            input_dir: Input directory
            concurrency_range: List of concurrency levels to test
            duration: Test duration per level
            
        Returns:
            List of results for each concurrency level
        """
        logger.info(f"Running scalability analysis: {concurrency_range}")
        
        results = []
        for concurrency in concurrency_range:
            logger.info(f"Testing concurrency={concurrency}")
            
            # Create evaluator with this concurrency
            evaluator = ThroughputEvaluator(
                server_urls=self.server_urls,
                concurrency=concurrency,
                batch_size=self.batch_size,
                timeout=self.timeout,
                backend=self.backend,
                devices=self.devices,
                device_mode=self.device_mode,
            )
            
            # Run evaluation
            result = asyncio.run(evaluator.evaluate(
                input_dir=input_dir,
                duration=duration,
            ))
            
            results.append(result)
        
        return results
