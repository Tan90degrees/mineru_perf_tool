# Copyright (c) Opendatalab. All rights reserved.
"""
Asynchronous benchmark client for MinerU service.
"""

import os
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class RequestResult:
    """Result of a single request."""
    success: bool
    file_name: str
    latency: float  # seconds
    status_code: int = 0
    error_message: str = ""
    response_size: int = 0  # bytes
    
    # Detailed timing
    upload_time: float = 0.0
    process_time: float = 0.0
    download_time: float = 0.0


@dataclass
class BenchmarkStats:
    """Statistics for benchmark run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    latencies: List[float] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.latencies:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": 0.0,
                "qps": 0.0,
                "avg_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "p50_latency": 0.0,
                "p90_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "total_bytes_sent": self.total_bytes_sent,
                "total_bytes_received": self.total_bytes_received,
                "throughput_mbps": 0.0,
            }
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        def percentile(data: List[float], p: float) -> float:
            """Calculate percentile."""
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0,
            "qps": self.successful_requests / sum(self.latencies) if sum(self.latencies) > 0 else 0.0,
            "avg_latency": sum(self.latencies) / n,
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "p50_latency": percentile(sorted_latencies, 50),
            "p90_latency": percentile(sorted_latencies, 90),
            "p95_latency": percentile(sorted_latencies, 95),
            "p99_latency": percentile(sorted_latencies, 99),
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "throughput_mbps": (self.total_bytes_sent + self.total_bytes_received) / (1024 * 1024) / sum(self.latencies) if sum(self.latencies) > 0 else 0.0,
        }


class BenchmarkClient:
    """Asynchronous benchmark client for MinerU service."""
    
    def __init__(
        self,
        server_urls: List[str],
        concurrency: int = 10,
        timeout: int = 300,
        backend: str = "hybrid-auto-engine",
        lang: str = "ch",
        method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        return_md: bool = True,
    ):
        """
        Initialize benchmark client.
        
        Args:
            server_urls: List of MinerU server URLs
            concurrency: Maximum concurrent requests
            timeout: Request timeout in seconds
            backend: MinerU backend type
            lang: Language for OCR
            method: Parse method (auto/txt/ocr)
            formula_enable: Enable formula parsing
            table_enable: Enable table parsing
            return_md: Return markdown content
        """
        self.server_urls = server_urls
        self.concurrency = concurrency
        self.timeout = timeout
        self.backend = backend
        self.lang = lang
        self.method = method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.return_md = return_md
        
        self.stats = BenchmarkStats()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.current_url_index = 0
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
        return False
    
    def _get_next_url(self) -> str:
        """Get next server URL in round-robin fashion."""
        url = self.server_urls[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.server_urls)
        return url
    
    async def send_request(self, file_path: Path) -> RequestResult:
        """
        Send a single file parse request.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            RequestResult object
        """
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Prepare multipart form data
                url = self._get_next_url()
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    file_size = len(file_content)
                
                # Create form data
                data = aiohttp.FormData()
                data.add_field(
                    'files',
                    file_content,
                    filename=file_path.name,
                    content_type='application/octet-stream'
                )
                
                # Add other form fields
                data.add_field('lang_list', self.lang)
                data.add_field('backend', self.backend)
                data.add_field('parse_method', self.method)
                data.add_field('formula_enable', str(self.formula_enable).lower())
                data.add_field('table_enable', str(self.table_enable).lower())
                data.add_field('return_md', str(self.return_md).lower())
                
                upload_time = time.time()
                process_start = time.time()
                
                # Send request
                async with self._session.post(f"{url}/file_parse", data=data) as response:
                    process_time = time.time() - process_start
                    
                    if response.status == 200:
                        result = await response.json()
                        download_time = time.time()
                        
                        latency = download_time - start_time
                        response_size = len(str(result))  # Approximate size
                        
                        # Update stats
                        self.stats.total_requests += 1
                        self.stats.successful_requests += 1
                        self.stats.total_bytes_sent += file_size
                        self.stats.total_bytes_received += response_size
                        self.stats.latencies.append(latency)
                        
                        return RequestResult(
                            success=True,
                            file_name=file_path.name,
                            latency=latency,
                            status_code=response.status,
                            response_size=response_size,
                            upload_time=upload_time - start_time,
                            process_time=process_time,
                            download_time=download_time - process_start,
                        )
                    else:
                        latency = time.time() - start_time
                        error_text = await response.text()
                        
                        self.stats.total_requests += 1
                        self.stats.failed_requests += 1
                        self.stats.total_bytes_sent += file_size
                        self.stats.latencies.append(latency)
                        
                        return RequestResult(
                            success=False,
                            file_name=file_path.name,
                            latency=latency,
                            status_code=response.status,
                            error_message=error_text,
                        )
                        
            except asyncio.TimeoutError:
                latency = time.time() - start_time
                self.stats.total_requests += 1
                self.stats.failed_requests += 1
                self.stats.latencies.append(latency)
                
                return RequestResult(
                    success=False,
                    file_name=file_path.name,
                    latency=latency,
                    error_message="Request timeout",
                )
                
            except Exception as e:
                latency = time.time() - start_time
                self.stats.total_requests += 1
                self.stats.failed_requests += 1
                self.stats.latencies.append(latency)
                
                logger.exception(f"Request failed for {file_path.name}")
                
                return RequestResult(
                    success=False,
                    file_name=file_path.name,
                    latency=latency,
                    error_message=str(e),
                )
    
    async def run_benchmark(
        self,
        file_batches: List[List[Path]],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark on file batches.
        
        Args:
            file_batches: List of file batches to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Benchmark summary
        """
        logger.info(f"Starting benchmark with {len(file_batches)} batches, concurrency={self.concurrency}")
        
        tasks = []
        for batch in file_batches:
            for file_path in batch:
                task = self.send_request(file_path)
                tasks.append(task)
        
        # Run all tasks with progress tracking
        completed = 0
        total = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total, result)
        
        return self.stats.get_summary()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = BenchmarkStats()
