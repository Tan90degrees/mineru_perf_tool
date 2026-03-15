"""
ipc_worker.py — Direct-call MinerU worker process.

Each worker process:
  1. Imports and warm-ups MinerU once (model loaded into NPU memory).
  2. Loops on a multiprocessing.Queue receiving ParseRequest dicts.
  3. Calls do_parse() directly — no HTTP, no socket, no network overhead.
  4. Puts a ParseResponse dict onto the result queue when done.

Usage (internal, launched by server_manager.py in "ipc" mode):
    python ipc_worker.py --worker_id 0 --req_queue_file /tmp/req_0 ...

Because multiprocessing.Queue cannot be pickled across process boundaries
easily with spawn (Windows), we use a manager-based Queue passed through
a shared file-descriptor or the Manager itself. However the cleanest cross-
platform approach is to pass the Manager address via env-var and reconnect
inside the worker.  We use multiprocessing.managers.BaseManager with an
authkey so the worker can register and consume the same queues.
"""
import os
import sys
import time
import logging
import argparse
import multiprocessing
import multiprocessing.managers
import traceback
from pathlib import Path

logger = logging.getLogger("ipc_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] ipc_worker - %(message)s")


# ---------------------------------------------------------------------------
# Shared Manager definition (must be importable by both manager and workers)
# ---------------------------------------------------------------------------
class QueueManager(multiprocessing.managers.BaseManager):
    pass


def _get_req_queue():
    """Placeholder — overridden by manager registration."""
    raise NotImplementedError

def _get_res_queue():
    raise NotImplementedError


QueueManager.register("get_req_queue", callable=_get_req_queue)
QueueManager.register("get_res_queue", callable=_get_res_queue)


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------
def worker_loop(worker_id: int, manager_address: tuple, authkey: bytes,
                card_id: int, backend: str, parse_method: str,
                lang: str, output_dir: str):
    """
    Connect to the QueueManager, pull tasks, call do_parse directly.
    """
    # Set NPU device visibility for this process
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card_id)

    logger.info(f"[{worker_id}] Connecting to queue manager at {manager_address}")
    manager = QueueManager(address=manager_address, authkey=authkey)
    manager.connect()

    req_queue = manager.get_req_queue()
    res_queue = manager.get_res_queue()

    logger.info(f"[{worker_id}] Connected. Loading MinerU (backend={backend})...")

    # Pre-import to trigger model load once
    try:
        from mineru.cli.common import do_parse, read_fn
        logger.info(f"[{worker_id}] MinerU imported successfully. Waiting for tasks...")
    except Exception as e:
        logger.error(f"[{worker_id}] Failed to import MinerU: {e}")
        res_queue.put({"worker_id": worker_id, "task_id": "startup", "error": str(e)})
        return

    while True:
        try:
            task = req_queue.get(timeout=5)
        except Exception:
            # Timeout or empty — check for poison pill later
            continue

        if task is None:
            # Poison pill: shut down
            logger.info(f"[{worker_id}] Received shutdown signal.")
            break

        task_id = task.get("task_id", "?")
        file_paths = task.get("file_paths", [])
        task_output_dir = task.get("output_dir", output_dir)
        task_backend = task.get("backend", backend)
        task_lang = task.get("lang", lang)
        task_parse_method = task.get("parse_method", parse_method)
        task_formula = task.get("formula_enable", True)
        task_table = task.get("table_enable", True)

        logger.info(f"[{worker_id}] Task {task_id}: processing {len(file_paths)} files")
        t0 = time.time()

        try:
            pdf_file_names = []
            pdf_bytes_list = []
            for fp in file_paths:
                pdf_bytes = read_fn(Path(fp))
                pdf_bytes_list.append(pdf_bytes)
                pdf_file_names.append(Path(fp).stem)

            import uuid
            unique_out = os.path.join(task_output_dir, str(uuid.uuid4()))
            os.makedirs(unique_out, exist_ok=True)

            do_parse(
                output_dir=unique_out,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=[task_lang] * len(pdf_file_names),
                backend=task_backend,
                parse_method=task_parse_method,
                formula_enable=task_formula,
                table_enable=task_table,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
            )

            elapsed = time.time() - t0
            logger.info(f"[{worker_id}] Task {task_id}: done in {elapsed:.2f}s")
            res_queue.put({
                "worker_id": worker_id,
                "task_id": task_id,
                "success_count": len(file_paths),
                "error_count": 0,
                "latency": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{worker_id}] Task {task_id} failed: {e}\n{traceback.format_exc()}")
            res_queue.put({
                "worker_id": worker_id,
                "task_id": task_id,
                "success_count": 0,
                "error_count": len(file_paths),
                "latency": elapsed,
                "error": str(e),
            })


def main():
    parser = argparse.ArgumentParser(description="MinerU IPC Worker")
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--manager_host", type=str, default="127.0.0.1")
    parser.add_argument("--manager_port", type=int, required=True)
    parser.add_argument("--authkey", type=str, required=True)
    parser.add_argument("--card_id", type=int, default=0)
    parser.add_argument("--backend", type=str, default="pipeline")
    parser.add_argument("--parse_method", type=str, default="auto")
    parser.add_argument("--lang", type=str, default="ch")
    parser.add_argument("--output_dir", type=str, default="./ipc_output")
    args = parser.parse_args()

    worker_loop(
        worker_id=args.worker_id,
        manager_address=(args.manager_host, args.manager_port),
        authkey=args.authkey.encode(),
        card_id=args.card_id,
        backend=args.backend,
        parse_method=args.parse_method,
        lang=args.lang,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
