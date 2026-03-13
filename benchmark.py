import argparse
import asyncio
import csv
import os
import logging

from config import BenchmarkConfig
from server_manager import ServerManager
from client import BenchmarkClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# CSV column order — fixed so all runs share the same schema
CSV_FIELDNAMES = [
    "cards", "instances_per_card", "total_instances",
    "batch_size", "backend",
    "total_files", "success_rate", "throughput_fps",
    "avg_latency_s", "p90_latency_s", "p99_latency_s", "total_time_s",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MinerU Throughput Evaluation Tool")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing PDF or Image files")
    parser.add_argument("--cards", type=int, nargs="+", default=[0], help="List of NPU card IDs to use")
    parser.add_argument("--instances_per_card", type=int, nargs="+", default=[1], help="List of # instances per card to test")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1], help="List of request batch sizes to test")
    parser.add_argument("--backend", type=str, default="pipeline", choices=["pipeline", "vlm-auto-engine", "hybrid-auto-engine"], help="Backend to evaluate")
    parser.add_argument("--lang", type=str, default="ch", help="Language parameter for MinerU")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup requests")
    parser.add_argument("--max_requests", type=int, default=None, help="Max requests per test (for quick testing)")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output CSV file (results are appended across runs)")
    parser.add_argument("--mock", action="store_true", help="Use mock MinerU server for tool testing validation")
    return parser.parse_args()


def open_csv_writer(output_csv: str):
    """
    Open the CSV in append mode.
    Write the header only when creating a brand-new file.
    Returns (file_handle, csv.DictWriter).
    """
    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    f = open(output_csv, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    return f, writer


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
        mock_mode=args.mock
    )

    logger.info(f"Starting grid search benchmark for MinerU ({config.backend})")
    logger.info(f"Cards: {config.cards}")
    logger.info(f"Instances per card options: {config.instances_per_card}")
    logger.info(f"Batch size options: {config.batch_size_list}")
    logger.info(f"Results will be appended to: {args.output_csv}")

    server_manager = ServerManager(config)
    csv_file, csv_writer = open_csv_writer(args.output_csv)

    try:
        num_cards = len(config.cards)

        for num_instances in config.instances_per_card:
            for batch_size in config.batch_size_list:
                total_instances = num_cards * num_instances
                logger.info(f"\n{'='*50}")
                logger.info(f"Running Test - Instances/Card: {num_instances}, Total Instances: {total_instances}, Batch Size: {batch_size}")
                logger.info(f"{'='*50}")

                # Start servers
                try:
                    server_manager.start_servers(num_cards, num_instances)
                except Exception as e:
                    logger.error(f"Skipping configuration due to server startup error: {e}")
                    continue

                # Initialize client
                client = BenchmarkClient(config, server_manager.active_endpoints)

                # Run benchmark
                try:
                    metrics = await client.run_benchmark(batch_size)

                    row = {
                        "cards": num_cards,
                        "instances_per_card": num_instances,
                        "total_instances": total_instances,
                        "batch_size": batch_size,
                        "backend": config.backend,
                        **metrics
                    }
                    # Write and flush immediately — Ctrl+C won't lose this row
                    csv_writer.writerow(row)
                    csv_file.flush()
                    logger.info(f"Result saved → {args.output_csv}")

                except Exception as e:
                    logger.error(f"Benchmark run failed: {e}")

                # Stop servers before next config
                server_manager.stop_all()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Always kill servers and close CSV, even on Ctrl+C or crash
        server_manager.stop_all()
        csv_file.close()
        logger.info("Benchmark finished. All servers stopped.")


if __name__ == "__main__":
    asyncio.run(main())
