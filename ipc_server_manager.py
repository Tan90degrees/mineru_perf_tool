"""
ipc_server_manager.py — Manages a pool of IPC worker processes.

Starts a multiprocessing.managers.BaseManager to host the shared queues,
then spawns ipc_worker.py subprocesses that connect to it.

The client-facing interface is:
  - manager.ipc_endpoints: list of (worker_id, req_queue, res_queue) tuples
  - ipc_stop_all(): kill all workers + shutdown manager
"""
import os
import sys
import time
import subprocess
import logging
import multiprocessing
import multiprocessing.managers
from typing import List, Tuple

from config import BenchmarkConfig

logger = logging.getLogger(__name__)

AUTHKEY = b"mineru-ipc-bench"


class _IPCQueueServer(multiprocessing.managers.BaseManager):
    pass


class IPCServerManager:
    """
    Manages MinerU worker processes that communicate via multiprocessing queues
    instead of HTTP.  One (req_queue, res_queue) pair per worker instance.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._manager: _IPCQueueServer = None
        self._manager_port: int = None
        self._processes: List[subprocess.Popen] = []
        # List of (worker_id, req_queue_proxy, res_queue_proxy)
        self.ipc_endpoints: List[Tuple[int, object, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_workers(self, num_cards: int, instances_per_card: int):
        """Starts the queue manager and all worker subprocesses."""
        self._start_manager()

        cards = self.config.cards[:num_cards]
        worker_id = 0
        for card_id in cards:
            for _ in range(instances_per_card):
                self._launch_worker(worker_id, card_id)
                worker_id += 1

        logger.info(f"Waiting for {worker_id} IPC workers to be ready...")
        self._wait_for_workers(worker_id)
        logger.info("All IPC workers ready.")

    def stop_all(self):
        """Terminate all workers and shut down the manager."""
        # Send poison pills
        for wid, req_q, _ in self.ipc_endpoints:
            try:
                req_q.put(None)
            except Exception:
                pass

        # Wait / kill subprocesses
        for p in self._processes:
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._processes.clear()
        self.ipc_endpoints.clear()

        # Shut down manager
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

        logger.info("All IPC workers stopped.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_manager(self):
        """Start the BaseManager that hosts all queues."""
        # Pick a free port
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self._manager_port = s.getsockname()[1]
        s.close()

        # Register queue factories for each potential worker slot
        # We pre-register up to 32 slots; unused ones are harmless
        max_slots = 32
        queues: dict = {}
        for i in range(max_slots):
            req_q: multiprocessing.Queue = multiprocessing.Queue()
            res_q: multiprocessing.Queue = multiprocessing.Queue()
            queues[i] = (req_q, res_q)

        class _Server(_IPCQueueServer):
            pass

        for i in range(max_slots):
            rq, rsq = queues[i]
            _Server.register(f"get_req_queue_{i}", callable=lambda q=rq: q)
            _Server.register(f"get_res_queue_{i}", callable=lambda q=rsq: q)

        self._manager = _Server(address=("127.0.0.1", self._manager_port), authkey=AUTHKEY)
        self._manager.start()
        self._queues = queues  # keep strong references
        logger.info(f"IPC Queue Manager started on port {self._manager_port}")

    def _launch_worker(self, worker_id: int, card_id: int):
        """Spawn one ipc_worker.py subprocess."""
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

        logger.info(f"Launching IPC worker {worker_id} on card {card_id}")
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._processes.append(p)

        # Build client-side proxy handles using our manager
        client = _IPCQueueServer(address=("127.0.0.1", self._manager_port), authkey=AUTHKEY)
        client.register(f"get_req_queue_{worker_id}")
        client.register(f"get_res_queue_{worker_id}")
        client.connect()

        req_proxy = getattr(client, f"get_req_queue_{worker_id}")()
        res_proxy = getattr(client, f"get_res_queue_{worker_id}")()
        self.ipc_endpoints.append((worker_id, req_proxy, res_proxy))

    def _wait_for_workers(self, num_workers: int, timeout: int = 120):
        """
        Poll worker subprocesses; they're considered 'ready' once they've
        connected to the manager (their process stays alive for > 2s).
        """
        deadline = time.time() + timeout
        alive = [False] * num_workers
        while time.time() < deadline:
            for i, p in enumerate(self._processes):
                if p.poll() is not None:
                    out, _ = p.communicate()
                    raise RuntimeError(
                        f"Worker {i} died at startup:\n{out}"
                    )
                alive[i] = True
            if all(alive):
                time.sleep(3)  # Give workers time to load models
                return
            time.sleep(1)
        raise TimeoutError("IPC workers did not start in time.")
