import argparse
import asyncio
import csv
import sys
import logging
from typing import List

from config import BenchmarkConfig
from server_manager import ServerManager
from client import BenchmarkClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="MinerU Throughput Evaluation Tool")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing PDF or Image files")
    parser.add_argument("--cards", type=int, nargs="+", default=[0], help="List of NPU card IDs to use")
    parser.add_argument("--instances_per_card", type=int, nargs="+", default=[1], help="List of # instances per card to test")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4], help="List of request concurrency levels to test")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1], help="List of request batch sizes to test")
    parser.add_argument("--backend", type=str, default="pipeline", choices=["pipeline", "vlm-auto-engine", "hybrid-auto-engine"], help="Backend to evaluate")
    parser.add_argument("--lang", type=str, default="ch", help="Language parameter for MinerU")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup requests")
    parser.add_argument("--max_requests", type=int, default=None, help="Max requests per test (for quick testing)")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output file for results")
    parser.add_argument("--mock", action="store_true", help="Use mock MinerU server for tool testing validation")
    return parser.parse_args()

async def main():
    args = parse_args()

    config = BenchmarkConfig(
        cards=args.cards,
        instances_per_card=args.instances_per_card,
        concurrency_list=args.concurrency,
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
    logger.info(f"Concurrency options: {config.concurrency_list}")
    logger.info(f"Batch size options: {config.batch_size_list}")

    server_manager = ServerManager(config)
    results = []

    try:
        # Number of actual cards used is fixed to the provided list
        num_cards = len(config.cards)
        
        for num_instances in config.instances_per_card:
            for concurrency in config.concurrency_list:
                for batch_size in config.batch_size_list:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Running Test - Instances/Card: {num_instances}, Total Concurrency: {concurrency}, Batch Size: {batch_size}")
                    logger.info(f"{'='*50}")

                    # Start servers
                    try:
                        server_manager.start_servers(num_cards, num_instances, concurrency)
                    except Exception as e:
                        logger.error(f"Skipping configuration due to server startup error: {e}")
                        continue
                    
                    # Initialize client
                    client = BenchmarkClient(config, server_manager.active_endpoints)
                    
                    # Run benchmark
                    try:
                        metrics = await client.run_benchmark(concurrency, batch_size)
                        
                        # Store result
                        row = {
                            "cards": num_cards,
                            "instances_per_card": num_instances,
                            "total_instances": num_cards * num_instances,
                            "concurrency": concurrency,
                            "batch_size": batch_size,
                            "backend": config.backend,
                            **metrics
                        }
                        results.append(row)
                        
                    except Exception as e:
                        logger.error(f"Benchmark run failed: {e}")
                    
                    # Turn off servers for next config
                    server_manager.stop_all()

    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down servers...")
        server_manager.stop_all()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        server_manager.stop_all()

    # Write report
    if not results:
        logger.warning("No results to write.")
        sys.exit(0)
        
    logger.info(f"Writing results to {args.output_csv}")
    keys = results[0].keys()
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("Benchmark complete!")

if __name__ == "__main__":
    asyncio.run(main())
