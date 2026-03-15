"""
ipc_worker.py — Direct-call MinerU worker process.

Connects to a MinerUQueueManager, pulls ParseRequest tasks from its
private request queue, and processes them via do_parse() (or a mock
sleep in --mock mode), then pushes results to the response queue.

Cross-platform notes:
  - On Windows, this process is always started with spawn (not fork),
    so all code here must be importable without side-effects at module level.
  - The guard `if __name__ == '__main__': main()` is mandatory on Windows.
"""
import os
import sys
import time
import logging
import argparse
import traceback
import uuid
from pathlib import Path

logger = logging.getLogger("ipc_worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ipc_worker - %(message)s",
)


def worker_loop(
    worker_id: int,
    manager_host: str,
    manager_port: int,
    authkey: bytes,
    card_id: int,
    backend: str,
    parse_method: str,
    lang: str,
    output_dir: str,
    mock: bool,
):
    """Main processing loop for one worker process."""

    # Set NPU visibility before importing MinerU so it picks up the right card
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card_id)

    # Connect to the shared queue manager
    from ipc_queue_manager import connect_manager
    logger.info(f"[{worker_id}] Connecting to queue manager at {manager_host}:{manager_port}")
    manager = connect_manager(manager_host, manager_port, authkey)

    req_queue = manager.get_req_queue(worker_id)
    res_queue = manager.get_res_queue(worker_id)
    logger.info(f"[{worker_id}] Queue connected. mock={mock}, card={card_id}, backend={backend}")

    # Pre-load MinerU (real mode only)
    if mock:
        read_fn = None
        do_parse = None
        logger.info(f"[{worker_id}] Mock mode — skipping MinerU import.")
    else:
        try:
            from mineru.cli.common import do_parse, read_fn
            logger.info(f"[{worker_id}] MinerU loaded. Ready for tasks.")
        except Exception as e:
            logger.error(f"[{worker_id}] Failed to import MinerU: {e}")
            res_queue.put({
                "worker_id": worker_id,
                "task_id": "startup",
                "success_count": 0,
                "error_count": 0,
                "latency": 0.0,
                "error": str(e),
            })
            return

    # Main task loop
    while True:
        try:
            task = req_queue.get(timeout=5)
        except Exception:
            continue  # Timeout — keep waiting

        if task is None:
            logger.info(f"[{worker_id}] Shutdown signal received.")
            break

        task_id = task.get("task_id", "?")
        file_paths = task.get("file_paths", [])
        task_output_dir = task.get("output_dir", output_dir)
        task_backend = task.get("backend", backend)
        task_lang = task.get("lang", lang)
        task_parse_method = task.get("parse_method", parse_method)
        task_formula = task.get("formula_enable", True)
        task_table = task.get("table_enable", True)

        logger.info(f"[{worker_id}] Task {task_id[:8]}: {len(file_paths)} files")
        t0 = time.time()

        try:
            if mock:
                # Simulate 0.5 s processing per batch (like the HTTP mock server)
                time.sleep(0.5)
                success_count = len(file_paths)
                error_count = 0
            else:
                pdf_file_names = []
                pdf_bytes_list = []
                for fp in file_paths:
                    pdf_bytes = read_fn(Path(fp))
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(Path(fp).stem)

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
                success_count = len(file_paths)
                error_count = 0

            elapsed = time.time() - t0
            logger.info(f"[{worker_id}] Task {task_id[:8]} done in {elapsed:.2f}s")
            res_queue.put({
                "worker_id": worker_id,
                "task_id": task_id,
                "success_count": success_count,
                "error_count": error_count,
                "latency": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{worker_id}] Task {task_id[:8]} failed: {e}\n{traceback.format_exc()}")
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
    parser.add_argument("--mock", action="store_true", help="Skip real MinerU, simulate processing")
    args = parser.parse_args()

    worker_loop(
        worker_id=args.worker_id,
        manager_host=args.manager_host,
        manager_port=args.manager_port,
        authkey=args.authkey.encode(),
        card_id=args.card_id,
        backend=args.backend,
        parse_method=args.parse_method,
        lang=args.lang,
        output_dir=args.output_dir,
        mock=args.mock,
    )


# CRITICAL: Required on Windows to prevent recursive spawning
if __name__ == "__main__":
    main()
