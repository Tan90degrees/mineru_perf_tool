"""
ipc_server_manager.py — Manages a pool of IPC worker processes.

Starts a MinerUQueueManager (from ipc_queue_manager.py) to host the
shared queues, then spawns ipc_worker.py subprocesses that connect to it.

Cross-platform notes:
  - Uses module-level picklable callables (via ipc_queue_manager) — no lambdas.
  - Works on both Windows (spawn) and Linux (fork/spawn).
  - Subprocess stdout/stderr are piped to avoid blocking on Windows.
"""
import os
import sys
import time
import socket
import subprocess
import logging
from typing import List, Tuple

from config import BenchmarkConfig
from ipc_queue_manager import MinerUQueueManager, connect_manager

logger = logging.getLogger(__name__)

AUTHKEY = b"mineru-ipc-bench-2025"


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IPCServerManager:
    """
    Manages MinerU worker processes that communicate via multiprocessing
    queues instead of HTTP, avoiding all network overhead.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._manager: MinerUQueueManager = None
        self._manager_port: int = None
        self._processes: List[subprocess.Popen] = []
        # List of (worker_id, req_queue_proxy, res_queue_proxy)
        self.ipc_endpoints: List[Tuple[int, object, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_workers(self, num_cards: int, instances_per_card: int):
        """Start the queue manager server, then spawn all worker subprocesses."""
        self._start_manager()

        cards = self.config.cards[:num_cards]
        worker_id = 0
        for card_id in cards:
            for _ in range(instances_per_card):
                self._launch_worker(worker_id, card_id)
                worker_id += 1

        total = worker_id
        logger.info(f"Waiting for {total} IPC workers to connect...")
        # Give workers time to start up and connect
        self._poll_workers_alive(total, timeout=60)
        # Extra grace period for model loading (real mode)
        if not self.config.mock_mode:
            logger.info("Waiting for MinerU models to load (this may take a while)...")
            time.sleep(5)
        logger.info("All IPC workers ready.")

    def stop_all(self):
        """Send shutdown signals, wait for workers, stop manager."""
        # Send poison pills to all workers
        for wid, req_q, _ in self.ipc_endpoints:
            try:
                req_q.put(None)
            except Exception:
                pass

        # Wait / force kill
        for p in self._processes:
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._processes.clear()
        self.ipc_endpoints.clear()

        # Stop the manager
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

        logger.info("All IPC workers stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_manager(self):
        self._manager_port = _find_free_port()
        self._manager = MinerUQueueManager(
            address=("127.0.0.1", self._manager_port),
            authkey=AUTHKEY,
        )
        self._manager.start()
        logger.info(f"IPC Queue Manager started on port {self._manager_port}")

    def _launch_worker(self, worker_id: int, card_id: int):
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(card_id)

        cmd = [
            sys.executable, "ipc_worker.py",
            "--worker_id", str(worker_id),
            "--manager_host", "127.0.0.1",
            "--manager_port", str(self._manager_port),
            "--authkey", AUTHKEY.decode(),
            "--card_id", str(card_id),
            "--backend", self.config.backend,
            "--parse_method", self.config.parse_method,
            "--lang", self.config.lang,
            "--output_dir", self.config.output_dir,
        ]
        if self.config.mock_mode:
            cmd.append("--mock")

        logger.info(f"Launching IPC worker {worker_id} on card {card_id}")
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._processes.append(p)

        # Connect from the main process to get proxy queue handles
        client_mgr = connect_manager("127.0.0.1", self._manager_port, AUTHKEY)
        req_proxy = client_mgr.get_req_queue(worker_id)
        res_proxy = client_mgr.get_res_queue(worker_id)
        self.ipc_endpoints.append((worker_id, req_proxy, res_proxy))

    def _poll_workers_alive(self, num_workers: int, timeout: int):
        """Verify all worker processes are still alive (haven't crashed at startup)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            all_ok = True
            for i, p in enumerate(self._processes):
                if p.poll() is not None:
                    out = p.stdout.read() if p.stdout else ""
                    raise RuntimeError(
                        f"IPC worker {i} exited early (code {p.returncode}):\n{out}"
                    )
            if all_ok:
                time.sleep(2)  # Workers are alive — give them time to connect+load
                return
        raise TimeoutError("IPC workers did not start within the timeout.")
