import os
import time
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics
import tqdm

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


    async def _send_request(self, worker_name: str, endpoint: str, session: aiohttp.ClientSession, file_paths: List[str]) -> Tuple[float, int, int]:
        url = f"{endpoint}/file_parse"
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        opened_files = []
        
        logger.info(f"[Worker {worker_name}] Sending {len(file_paths)} files to {endpoint}...")
        try:
            data = aiohttp.FormData()
            
            for file_path in file_paths:
                f = open(file_path, 'rb')
                opened_files.append(f)
                data.add_field('files', f, filename=os.path.basename(file_path))
                data.add_field('lang_list', self.config.lang)
                
            data.add_field('output_dir', self.config.output_dir)
            data.add_field('backend', self.config.backend)
            data.add_field('parse_method', self.config.parse_method)
            data.add_field('formula_enable', 'True')
            data.add_field('table_enable', 'True')
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    _ = await response.json()
                    success_count = len(file_paths)
                    logger.info(f"[Worker {worker_name}] Done: {len(file_paths)} files in {time.time() - start_time:.2f}s")
                else:
                    error_text = await response.text()
                    logger.error(f"[Worker {worker_name}] Error {response.status}: {error_text}")
                    error_count = len(file_paths)
                    
        except Exception as e:
            logger.error(f"[Worker {worker_name}] Failed: {e}")
            error_count = len(file_paths)
        finally:
            for f in opened_files:
                try:
                    f.close()
                except Exception:
                    pass
            
        latency = time.time() - start_time
        return latency, success_count, error_count

    async def _dedicated_worker(self, name: str, endpoint: str, session: aiohttp.ClientSession,
                                queue: asyncio.Queue, pbar: tqdm.tqdm):
        """
        Dedicated worker bound to a single endpoint.
        Continuously pulls batches from the shared queue and sends them to its
        assigned endpoint one at a time. Since each MinerU instance is internally
        serial, this fully saturates the instance without leaving it idle.
        """
        logger.info(f"[Worker {name}] Started, bound to {endpoint}.")
        while True:
            file_batch = await queue.get()
            if file_batch is None:
                logger.info(f"[Worker {name}] No more tasks. Exiting.")
                queue.task_done()
                break
                
            latency, success_count, error_count = await self._send_request(name, endpoint, session, file_batch)
            self.metrics.add_result(latency, success_count, error_count)
            pbar.update(success_count + error_count)
            queue.task_done()

    def _chunk_files(self, files: List[str], batch_size: int) -> List[List[str]]:
        return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    async def run_benchmark(self, batch_size: int) -> Dict[str, Any]:
        """
        Run the load test using a dedicated-worker-per-instance strategy.
        One worker is bound to each endpoint, pulling tasks from a shared queue.
        This fully saturates every instance regardless of each endpoint's serial nature.
        """
        self.metrics = MetricsCollector()
        
        files = self.files_to_process
        if self.config.max_requests_per_test and len(files) > self.config.max_requests_per_test:
            files = files[:self.config.max_requests_per_test]
            
        if not files:
            logger.warning("No files to process!")
            return self.metrics.get_summary()

        num_instances = len(self.endpoints)
        logger.info(f"Starting test with {len(files)} files, {num_instances} instances, batch_size={batch_size}.")
        
        timeout = aiohttp.ClientTimeout(total=3600)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Warm up — send one request to each instance
            if self.config.warmup_requests > 0:
                logger.info(f"Starting warmup ({self.config.warmup_requests} requests per instance)...")
                warmup_tasks = []
                for ep_idx, endpoint in enumerate(self.endpoints):
                    warmup_batch = files[:self.config.warmup_requests]
                    warmup_tasks.append(
                        self._send_request(f"warmup-{ep_idx}", endpoint, session, warmup_batch)
                    )
                await asyncio.gather(*warmup_tasks)
                logger.info("Warmup complete.")

            # Shared work queue — all batches go in, each worker competes for them
            queue = asyncio.Queue()
            batches = self._chunk_files(files, batch_size)
            for batch in batches:
                queue.put_nowait(batch)

            # One termination sentinel per worker
            for _ in range(num_instances):
                queue.put_nowait(None)

            # Start one dedicated worker per endpoint
            with tqdm.tqdm(total=len(files), desc=f"Benchmarking ({num_instances} instances)") as pbar:
                workers = [
                    asyncio.create_task(
                        self._dedicated_worker(f"inst-{i}", ep, session, queue, pbar)
                    )
                    for i, ep in enumerate(self.endpoints)
                ]

                self.metrics.start()
                await queue.join()
                await asyncio.gather(*workers)
                self.metrics.stop()

        summary = self.metrics.get_summary()
        logger.info(f"Benchmark run complete: {summary}")
        return summary

