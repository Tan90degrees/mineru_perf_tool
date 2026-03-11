import os
import subprocess
import time
import socket
import logging
from typing import List, Dict

from config import BenchmarkConfig

logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.processes: List[subprocess.Popen] = []
        self.active_endpoints: List[str] = []

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.config.host, port)) == 0

    def _find_available_port(self, start_port: int) -> int:
        port = start_port
        while self._is_port_in_use(port) or port in [int(ep.split(":")[-1]) for ep in self.active_endpoints]:
            port += 1
        return port

    def start_servers(self, num_cards: int, instances_per_card: int, concurrency: int):
        """Starts MinerU API server instances."""
        logger.info(f"Starting {num_cards * instances_per_card} servers over {num_cards} cards.")
        self.processes.clear()
        self.active_endpoints.clear()
        
        cards_to_use = self.config.cards[:num_cards]
        current_port = self.config.server_port_start

        for card_id in cards_to_use:
            for instance_idx in range(instances_per_card):
                port = self._find_available_port(current_port)
                current_port = port + 1
                
                env = os.environ.copy()
                env["ASCEND_RT_VISIBLE_DEVICES"] = str(card_id)
                # env["MINERU_DEVICE_MODE"] = f"npu:{card_id}" # Fallback for MinerU's config_reader
                # Ensure each instance can handle at least the number of concurrency/clients it will be assigned
                env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(concurrency) 
                
                import sys
                if self.config.mock_mode:
                    cmd = [
                        sys.executable, "mock_server.py",
                        "--host", self.config.host,
                        "--port", str(port)
                    ]
                else:
                    cmd = [
                        sys.executable, "-m", "mineru.cli.fast_api",
                        "--host", self.config.host,
                        "--port", str(port)
                    ]
                
                logger.info(f"Starting server on card {card_id}, port {port} with concurrency {concurrency}")
                
                # Start the subprocess
                try:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    self.processes.append(process)
                    self.active_endpoints.append(f"http://{self.config.host}:{port}")
                except Exception as e:
                    logger.error(f"Failed to start server on port {port}: {e}")
                    self.stop_all()
                    raise

        self._wait_for_servers()

    def _wait_for_servers(self, timeout: int = 60):
        """Waits for all servers to become responsive."""
        logger.info("Waiting for servers to be ready...")
        import urllib.request
        start_time = time.time()
        
        for endpoint in self.active_endpoints:
            # MinerU fast_api has docs at /docs
            health_url = f"{endpoint}/docs"
            ready = False
            while time.time() - start_time < timeout:
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as resp:
                        if resp.getcode() == 200:
                            ready = True
                            logger.info(f"Server at {endpoint} is ready.")
                            break
                except Exception:
                    pass
                time.sleep(2)
            
            if not ready:
                logger.error(f"Timeout waiting for server at {endpoint}")
                self.stop_all()
                raise RuntimeError(f"Server {endpoint} failed to start.")

    def stop_all(self):
        """Stops all running server processes."""
        logger.info("Stopping all servers...")
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.processes.clear()
        self.active_endpoints.clear()
        logger.info("All servers stopped.")
