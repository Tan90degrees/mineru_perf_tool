import argparse
import asyncio
import csv
import os
import logging

from config import BenchmarkConfig
from server_manager import ServerManager
from ipc_server_manager import IPCServerManager
from client import BenchmarkClient
from ipc_client import LocalBenchmarkClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# CSV column order — fixed across all runs / modes
CSV_FIELDNAMES = [
    "mode", "cards", "instances_per_card", "total_instances",
    "batch_size", "backend",
    "total_files", "success_rate", "throughput_fps",
    "avg_latency_s", "p90_latency_s", "p99_latency_s", "total_time_s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MinerU Throughput Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)    HTTP + real MinerU fast_api server
  --mock       HTTP + mock server (no MinerU needed, for tool validation)
  --ipc        Direct IPC + real MinerU do_parse (no HTTP, max throughput)
  --ipc --mock Direct IPC + simulated processing (no MinerU needed, for validation)
""",
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing PDF or image files")
    parser.add_argument("--cards", type=int, nargs="+", default=[0],
                        help="NPU card IDs to use")
    parser.add_argument("--instances_per_card", type=int, nargs="+", default=[1],
                        help="Number of instances per card (grid search dimension)")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1],
                        help="Files per request batch (grid search dimension)")
    parser.add_argument("--backend", type=str, default="pipeline",
                        choices=["pipeline", "vlm-auto-engine", "hybrid-auto-engine"],
                        help="MinerU backend")
    parser.add_argument("--lang", type=str, default="ch",
                        help="Language for MinerU OCR")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup requests sent before timing starts")
    parser.add_argument("--max_requests", type=int, default=None,
                        help="Cap total files per test (useful for quick runs)")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
                        help="Results CSV (append mode — safe to reuse across runs)")
    parser.add_argument("--mock", action="store_true",
                        help="Use simulated processing instead of real MinerU")
    parser.add_argument("--ipc", action="store_true",
                        help="Use direct IPC workers instead of HTTP server")
    return parser.parse_args()


def open_csv_writer(output_csv: str):
    """Open in append mode; header written only for new files."""
    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    f = open(output_csv, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    return f, writer


async def run_http_benchmark(config, num_cards, num_instances, batch_size,
                             server_manager: ServerManager) -> dict:
    server_manager.start_servers(num_cards, num_instances)
    client = BenchmarkClient(config, server_manager.active_endpoints)
    try:
        return await client.run_benchmark(batch_size)
    finally:
        server_manager.stop_all()


async def run_ipc_benchmark(config, num_cards, num_instances, batch_size,
                            ipc_manager: IPCServerManager) -> dict:
    ipc_manager.start_workers(num_cards, num_instances)
    client = LocalBenchmarkClient(config, ipc_manager.ipc_endpoints)
    try:
        return await client.run_benchmark(batch_size)
    finally:
        ipc_manager.stop_all()


async def main():
    args = parse_args()

    config = BenchmarkConfig(
        cards=args.cards,
        instances_per_card=args.instances_per_card,
        batch_size_list=args.batch_size,
        backend=args.backend,
        lang=args.lang,
        data_dir=args.data_dir,
        warmup_requests=args.warmup,
        max_requests_per_test=args.max_requests,
        mock_mode=args.mock,
        ipc_mode=args.ipc,
    )

    if args.ipc and args.mock:
        mode_label = "ipc-mock"
    elif args.ipc:
        mode_label = "ipc"
    elif args.mock:
        mode_label = "http-mock"
    else:
        mode_label = "http"

    logger.info(f"MinerU Benchmark — backend: {config.backend}, mode: {mode_label}")
    logger.info(f"Cards: {config.cards} | Instances/card: {config.instances_per_card} | Batch sizes: {config.batch_size_list}")
    logger.info(f"Appending results to: {args.output_csv}")

    server_manager = ServerManager(config) if not args.ipc else None
    ipc_manager = IPCServerManager(config) if args.ipc else None

    csv_file, csv_writer = open_csv_writer(args.output_csv)

    try:
        num_cards = len(config.cards)
        for num_instances in config.instances_per_card:
            for batch_size in config.batch_size_list:
                total_instances = num_cards * num_instances
                logger.info(f"\n{'='*50}")
                logger.info(f"Test: {num_instances} inst/card × {num_cards} cards = {total_instances} inst | batch={batch_size} | mode={mode_label}")
                logger.info(f"{'='*50}")

                try:
                    if args.ipc:
                        metrics = await run_ipc_benchmark(
                            config, num_cards, num_instances, batch_size, ipc_manager
                        )
                    else:
                        metrics = await run_http_benchmark(
                            config, num_cards, num_instances, batch_size, server_manager
                        )

                    row = {
                        "mode": mode_label,
                        "cards": num_cards,
                        "instances_per_card": num_instances,
                        "total_instances": total_instances,
                        "batch_size": batch_size,
                        "backend": config.backend,
                        **metrics,
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()
                    logger.info(f"Saved → {args.output_csv}  throughput={metrics.get('throughput_fps')} fps")

                except Exception as e:
                    logger.error(f"Benchmark run failed: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        if server_manager:
            server_manager.stop_all()
        if ipc_manager:
            ipc_manager.stop_all()
        csv_file.close()
        logger.info("All workers stopped. Done.")


if __name__ == "__main__":
    asyncio.run(main())
