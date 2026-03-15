"""
ipc_queue_manager.py — Shared queue manager definition.

IMPORTANT: This module is imported by BOTH the manager server process
and the worker client processes. All functions here must be picklable
(i.e., module-level named functions, NOT lambdas or closures) because
on Windows, BaseManager.start() uses spawn and pickles the callables.

Usage pattern:
  Server side:
    from ipc_queue_manager import MinerUQueueManager
    mgr = MinerUQueueManager(address=('127.0.0.1', port), authkey=KEY)
    mgr.start()  # spawns manager server process

  Client/worker side:
    from ipc_queue_manager import connect_manager
    mgr = connect_manager(host, port, KEY)
    req_q = mgr.get_req_queue(worker_id)
    res_q = mgr.get_res_queue(worker_id)
"""
import multiprocessing
from multiprocessing.managers import BaseManager

# -----------------------------------------------------------------
# Module-level queue storage — lives in the manager *server* process.
# Starts empty; queues are created lazily on first access.
# This works because BaseManager.start() imports this module inside
# the spawned manager process, giving it its own fresh namespace.
# -----------------------------------------------------------------
_req_queues: dict = {}
_res_queues: dict = {}


def _get_req_queue(worker_id: int) -> multiprocessing.Queue:
    """Called inside the manager process to return (or create) a request queue."""
    if worker_id not in _req_queues:
        _req_queues[worker_id] = multiprocessing.Queue()
    return _req_queues[worker_id]


def _get_res_queue(worker_id: int) -> multiprocessing.Queue:
    """Called inside the manager process to return (or create) a result queue."""
    if worker_id not in _res_queues:
        _res_queues[worker_id] = multiprocessing.Queue()
    return _res_queues[worker_id]


class MinerUQueueManager(BaseManager):
    pass


# Register with module-level (picklable) functions — critical for Windows
MinerUQueueManager.register("get_req_queue", callable=_get_req_queue)
MinerUQueueManager.register("get_res_queue", callable=_get_res_queue)


def connect_manager(host: str, port: int, authkey: bytes) -> MinerUQueueManager:
    """
    Connect to an already-running MinerUQueueManager server.
    Workers call this to get proxy handles to their queues.
    """
    # On the *client* side, register WITHOUT callable (just needs the name)
    class _Client(MinerUQueueManager):
        pass

    # Re-register without callable so we can connect without needing the server callable
    _Client.register("get_req_queue")
    _Client.register("get_res_queue")

    mgr = _Client(address=(host, port), authkey=authkey)
    mgr.connect()
    return mgr
