"""
ipc_client.py — BenchmarkClient for IPC mode.

Instead of sending HTTP requests, puts file-path tasks directly onto
multiprocessing queues and waits for results on the response queue.
Achieves the same throughput measurement semantics as client.py but
without any network overhead.
"""
import os
import time
import asyncio
import logging
import uuid
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
        self.start_time = 0.0
        self.end_time = 0.0

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

        if self.latencies:
            avg_latency = statistics.mean(self.latencies)
            p90 = statistics.quantiles(self.latencies, n=10)[8] if len(self.latencies) > 1 else self.latencies[0]
            p99 = statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) > 1 else self.latencies[0]
        else:
            avg_latency = p90 = p99 = 0.0

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


class LocalBenchmarkClient:
    """
    Sends file-path batches directly to worker processes via multiprocessing
    queues instead of HTTP.  One dedicated worker coroutine per instance,
    mirroring the strategy in client.py.
    """

    def __init__(self, config: BenchmarkConfig, ipc_endpoints: List[Tuple[int, object, object]]):
        """
        ipc_endpoints: list of (worker_id, req_queue_proxy, res_queue_proxy)
        """
        self.config = config
        self.ipc_endpoints = ipc_endpoints
        self.metrics = MetricsCollector()
        self.files_to_process = self._scan_directory(config.data_dir)

    def _scan_directory(self, data_dir: str) -> List[str]:
        supported_exts = {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        for root, _, filenames in os.walk(data_dir):
            for fname in filenames:
                fp = Path(root) / fname
                if fp.suffix.lower() in supported_exts:
                    files.append(str(fp))
        return files

    def _chunk_files(self, files: List[str], batch_size: int) -> List[List[str]]:
        return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    # ------------------------------------------------------------------
    # Send one batch to a specific worker (blocking inside a thread)
    # ------------------------------------------------------------------
    def _send_batch_sync(self, worker_id: int, req_queue, res_queue,
                         file_paths: List[str]) -> Tuple[float, int, int]:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "file_paths": file_paths,
            "output_dir": self.config.output_dir,
            "backend": self.config.backend,
            "lang": self.config.lang,
            "parse_method": self.config.parse_method,
            "formula_enable": True,
            "table_enable": True,
        }

        t0 = time.time()
        logger.info(f"[IPC inst-{worker_id}] Sending {len(file_paths)} files (task {task_id[:8]}...)")
        req_queue.put(task)

        # Wait for the matching response (simple: single-task-in-flight per worker)
        while True:
            result = res_queue.get()
            if result.get("task_id") == task_id:
                break
            # Wrong task_id — put it back and retry (shouldn't normally happen)
            res_queue.put(result)
            time.sleep(0.01)

        latency = time.time() - t0
        success = result.get("success_count", 0)
        errors = result.get("error_count", 0)
        if result.get("error"):
            logger.error(f"[IPC inst-{worker_id}] Task failed: {result['error']}")
        else:
            logger.info(f"[IPC inst-{worker_id}] Done: {success} files in {latency:.2f}s")
        return latency, success, errors

    async def _dedicated_worker(self, worker_id: int, req_queue, res_queue,
                                 queue: asyncio.Queue, pbar: tqdm.tqdm):
        """Async coroutine: pulls batches from shared queue, dispatches to IPC worker."""
        loop = asyncio.get_event_loop()
        while True:
            file_batch = await queue.get()
            if file_batch is None:
                queue.task_done()
                break

            # Run the blocking IPC call in a thread pool so we don't block the event loop
            latency, success, errors = await loop.run_in_executor(
                None,
                self._send_batch_sync,
                worker_id, req_queue, res_queue, file_batch,
            )
            self.metrics.add_result(latency, success, errors)
            pbar.update(success + errors)
            queue.task_done()

    async def run_benchmark(self, batch_size: int) -> Dict[str, Any]:
        self.metrics = MetricsCollector()

        files = self.files_to_process
        if self.config.max_requests_per_test and len(files) > self.config.max_requests_per_test:
            files = files[:self.config.max_requests_per_test]

        if not files:
            logger.warning("No files to process!")
            return self.metrics.get_summary()

        num_instances = len(self.ipc_endpoints)
        logger.info(f"[IPC] Starting benchmark: {len(files)} files, {num_instances} workers, batch_size={batch_size}")

        # Warmup — send one batch per worker
        if self.config.warmup_requests > 0:
            logger.info("Warmup...")
            warmup_tasks = []
            loop = asyncio.get_event_loop()
            for wid, rq, rsq in self.ipc_endpoints:
                warmup_files = files[:self.config.warmup_requests]
                warmup_tasks.append(
                    loop.run_in_executor(None, self._send_batch_sync, wid, rq, rsq, warmup_files)
                )
            await asyncio.gather(*warmup_tasks)
            logger.info("Warmup complete.")

        # Build shared task queue
        task_queue: asyncio.Queue = asyncio.Queue()
        for batch in self._chunk_files(files, batch_size):
            task_queue.put_nowait(batch)
        for _ in range(num_instances):
            task_queue.put_nowait(None)  # poison pills

        with tqdm.tqdm(total=len(files), desc=f"IPC Benchmarking ({num_instances} instances)") as pbar:
            workers = [
                asyncio.create_task(
                    self._dedicated_worker(wid, rq, rsq, task_queue, pbar)
                )
                for wid, rq, rsq in self.ipc_endpoints
            ]
            self.metrics.start()
            await task_queue.join()
            await asyncio.gather(*workers)
            self.metrics.stop()

        summary = self.metrics.get_summary()
        logger.info(f"[IPC] Benchmark complete: {summary}")
        return summary
