import os
import time
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics

from config import BenchmarkConfig

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.latencies = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def add_result(self, latency: float, success_count: int, error_count: int):
        self.latencies.append(latency)
        self.success_count += success_count
        self.error_count += error_count

    def get_summary(self) -> Dict[str, Any]:
        total_time = self.end_time - self.start_time
        total_files = self.success_count + self.error_count
        throughput = total_files / total_time if total_time > 0 else 0
        
        if len(self.latencies) > 0:
            avg_latency = statistics.mean(self.latencies)
            p90 = statistics.quantiles(self.latencies, n=10)[8] if len(self.latencies) > 1 else self.latencies[0]
            p99 = statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) > 1 else self.latencies[0]
        else:
            avg_latency = p90 = p99 = 0
            
        success_rate = (self.success_count / total_files * 100) if total_files > 0 else 0

        return {
            "total_files": total_files,
            "success_rate": round(success_rate, 2),
            "throughput_fps": round(throughput, 2),
            "avg_latency_s": round(avg_latency, 2),
            "p90_latency_s": round(p90, 2),
            "p99_latency_s": round(p99, 2),
            "total_time_s": round(total_time, 2),
        }


class BenchmarkClient:
    def __init__(self, config: BenchmarkConfig, endpoints: List[str]):
        self.config = config
        self.endpoints = endpoints
        self.current_endpoint_idx = 0
        self.metrics = MetricsCollector()
        self.files_to_process = self._scan_directory(self.config.data_dir)

    def _scan_directory(self, data_dir: str) -> List[str]:
        supported_exts = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        for root, _, filenames in os.walk(data_dir):
            for file in filenames:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_exts:
                    files.append(str(file_path))
        return files

    def _get_next_endpoint(self) -> str:
        # Load balancing across endpoints via round-robin
        endpoint = self.endpoints[self.current_endpoint_idx]
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.endpoints)
        return endpoint

    async def _send_request(self, worker_name: int, session: aiohttp.ClientSession, file_paths: List[str]) -> Tuple[float, int, int]:
        endpoint = self._get_next_endpoint()
        url = f"{endpoint}/file_parse"
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        opened_files = []
        
        logger.info(f"[Worker {worker_name}] Sending {len(file_paths)} files to {endpoint}...")
        try:
            # Need to use aiohttp FormData to match FastAPI UploadFile and Form
            data = aiohttp.FormData()
            
            for file_path in file_paths:
                f = open(file_path, 'rb')
                opened_files.append(f)
                data.add_field('files',
                               f,
                               filename=os.path.basename(file_path))
                               
                # Form data uses lists for lang_list in fast_api
                data.add_field('lang_list', self.config.lang)
                
            data.add_field('output_dir', self.config.output_dir)
            data.add_field('backend', self.config.backend)
            data.add_field('parse_method', self.config.parse_method)
            # Ensure correct types
            data.add_field('formula_enable', 'True')
            data.add_field('table_enable', 'True')
            
            async with session.post(url, data=data) as response:
                # status 200 means success
                if response.status == 200:
                    _ = await response.json()
                    success_count = len(file_paths)
                    logger.info(f"[Worker {worker_name}] Successfully processed {len(file_paths)} files in {time.time() - start_time:.2f}s")
                else:
                    error_text = await response.text()
                    logger.error(f"[Worker {worker_name}] Error from {endpoint}: {response.status} - {error_text}")
                    error_count = len(file_paths)
                    
        except Exception as e:
            logger.error(f"[Worker {worker_name}] Batch request failed: {e}")
            error_count = len(file_paths)
        finally:
            for f in opened_files:
                try:
                    f.close()
                except Exception:
                    pass
            
        latency = time.time() - start_time
        return latency, success_count, error_count

    async def _worker(self, name: int, session: aiohttp.ClientSession, queue: asyncio.Queue):
        logger.info(f"[Worker {name}] Started.")
        while True:
            file_batch = await queue.get()
            if file_batch is None:
                # Terminate worker
                logger.info(f"[Worker {name}] Received termination signal. Exiting.")
                queue.task_done()
                break
                
            latency, success_count, error_count = await self._send_request(name, session, file_batch)
            self.metrics.add_result(latency, success_count, error_count)
            queue.task_done()


    def _chunk_files(self, files: List[str], batch_size: int) -> List[List[str]]:
        return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    async def run_benchmark(self, concurrency: int, batch_size: int) -> Dict[str, Any]:
        """Runs the load test and returns the metrics."""
        self.metrics = MetricsCollector()
        
        files = self.files_to_process
        if self.config.max_requests_per_test and len(files) > self.config.max_requests_per_test:
            files = files[:self.config.max_requests_per_test]
            
        if not files:
            logger.warning("No files to process!")
            return self.metrics.get_summary()

        logger.info(f"Starting test with {len(files)} files, concurrency={concurrency}, batch_size={batch_size}.")
        
        # We need a large timeout as MinerU inference can take a while per file (especially VLM)
        timeout = aiohttp.ClientTimeout(total=3600)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Warm up
            if self.config.warmup_requests > 0:
                logger.info(f"Starting warmup with {self.config.warmup_requests} requests...")
                warmup_files = files[:self.config.warmup_requests]
                await asyncio.gather(*(self._send_request("warmup", session, [f]) for f in warmup_files))
                logger.info("Warmup complete.")

            queue = asyncio.Queue()
            
            batches = self._chunk_files(files, batch_size)
            for batch in batches:
                queue.put_nowait(batch)

            # Start workers
            workers = []
            for i in range(concurrency):
                workers.append(asyncio.create_task(self._worker(i, session, queue)))
                # queue termination signals
                queue.put_nowait(None)

            self.metrics.start()
            await queue.join()
            await asyncio.gather(*workers)
            self.metrics.stop()

        summary = self.metrics.get_summary()
        logger.info(f"Benchmark run complete: {summary}")
        return summary
