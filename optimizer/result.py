# Copyright (c) Opendatalab. All rights reserved.
"""
Result aggregation and analysis for grid search.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class ResultAggregator:
    """Aggregate and analyze grid search results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize result aggregator.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        self.result_file = self.output_dir / "grid_search_results.json"
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a test result."""
        self.results.append(result)
        self._save_results()
    
    def _save_results(self) -> None:
        """Save results to JSON file."""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "results": self.results,
            }, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved {len(self.results)} results to {self.result_file}")
    
    def find_optimal(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        metric: str = "qps",
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Find optimal configuration based on metric.
        
        Args:
            results: List of results (uses self.results if None)
            metric: Metric to optimize (qps, avg_latency, success_rate)
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Optimal configuration
        """
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if "error" not in r and r.get("throughput")]
        
        if not successful_results:
            logger.warning("No successful results found")
            return {}
        
        # Sort by metric
        sorted_results = sorted(
            successful_results,
            key=lambda r: r.get("throughput", {}).get(metric, 0),
            reverse=higher_is_better,
        )
        
        optimal = sorted_results[0]
        
        return {
            "test_id": optimal["test_id"],
            "test_case": optimal["test_case"],
            "metric_value": optimal.get("throughput", {}).get(metric),
            "throughput": optimal.get("throughput", {}),
            "resources": optimal.get("resources", {}),
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        if not self.results:
            return {"error": "No results available"}
        
        # Summary statistics
        successful = [r for r in self.results if "error" not in r]
        failed = [r for r in self.results if "error" in r]
        
        # QPS statistics
        qps_values = [r.get("throughput", {}).get("qps", 0) for r in successful]
        
        # Latency statistics
        latencies = [r.get("throughput", {}).get("avg_latency", 0) for r in successful]
        
        # Resource usage
        cpu_values = [r.get("resources", {}).get("cpu_avg_percent", 0) for r in successful]
        memory_values = [r.get("resources", {}).get("memory_avg_percent", 0) for r in successful]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful),
                "failed_tests": len(failed),
            },
            "qps": {
                "max": max(qps_values) if qps_values else 0,
                "min": min(qps_values) if qps_values else 0,
                "avg": sum(qps_values) / len(qps_values) if qps_values else 0,
            },
            "latency": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "avg": sum(latencies) / len(latencies) if latencies else 0,
            },
            "resources": {
                "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "memory_avg": sum(memory_values) / len(memory_values) if memory_values else 0,
            },
            "optimal_config": self.find_optimal(),
            "failed_tests": [{"test_id": r["test_id"], "error": r["error"]} for r in failed],
        }
        
        return report
    
    def export_csv(self, filename: str = "results.csv") -> str:
        """Export results to CSV file."""
        if not self.results:
            return ""
        
        csv_path = self.output_dir / filename
        
        # Prepare CSV data
        rows = []
        for result in self.results:
            row = {
                "test_id": result.get("test_id", ""),
                "devices": result.get("test_case", {}).get("devices", []),
                "instances_per_device": result.get("test_case", {}).get("instances_per_device", 0),
                "concurrency": result.get("test_case", {}).get("concurrency", 0),
                "batch_size": result.get("test_case", {}).get("batch_size", 0),
                "qps": result.get("throughput", {}).get("qps", 0),
                "avg_latency": result.get("throughput", {}).get("avg_latency", 0),
                "p50_latency": result.get("throughput", {}).get("p50_latency", 0),
                "p90_latency": result.get("throughput", {}).get("p90_latency", 0),
                "p95_latency": result.get("throughput", {}).get("p95_latency", 0),
                "p99_latency": result.get("throughput", {}).get("p99_latency", 0),
                "success_rate": result.get("throughput", {}).get("success_rate", 0),
                "cpu_avg_percent": result.get("resources", {}).get("cpu_avg_percent", 0),
                "memory_avg_percent": result.get("resources", {}).get("memory_avg_percent", 0),
                "error": result.get("error", ""),
            }
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Exported results to {csv_path}")
        
        return str(csv_path)
    
    def get_heatmap_data(
        self,
        x_param: str = "concurrency",
        y_param: str = "instances_per_device",
        metric: str = "qps",
    ) -> Dict[str, Any]:
        """
        Generate heatmap data for visualization.
        
        Args:
            x_param: Parameter for X axis
            y_param: Parameter for Y axis
            metric: Metric for color
            
        Returns:
            Heatmap data
        """
        if not self.results:
            return {"x_values": [], "y_values": [], "data": []}
        
        # Extract unique values
        x_values = sorted(list(set(
            r.get("test_case", {}).get(x_param, 0)
            for r in self.results
            if "test_case" in r
        )))
        
        y_values = sorted(list(set(
            r.get("test_case", {}).get(y_param, 0)
            for r in self.results
            if "test_case" in r
        )))
        
        # Create 2D array
        data = [[0.0] * len(x_values) for _ in range(len(y_values))]
        
        for result in self.results:
            if "test_case" not in result or "throughput" not in result:
                continue
            
            x_val = result["test_case"].get(x_param)
            y_val = result["test_case"].get(y_param)
            
            if x_val in x_values and y_val in y_values:
                x_idx = x_values.index(x_val)
                y_idx = y_values.index(y_val)
                data[y_idx][x_idx] = result["throughput"].get(metric, 0)
        
        return {
            "x_param": x_param,
            "y_param": y_param,
            "metric": metric,
            "x_values": x_values,
            "y_values": y_values,
            "data": data,
        }
