# Copyright (c) Opendatalab. All rights reserved.
"""
Report generator for benchmark results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: str = "benchmark_report.json",
    ) -> str:
        """
        Generate JSON report.
        
        Args:
            results: Benchmark results
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }
        
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON report: {report_path}")
        return str(report_path)
    
    def generate_csv_report(
        self,
        results: List[Dict[str, Any]],
        filename: str = "benchmark_report.csv",
    ) -> str:
        """
        Generate CSV report.
        
        Args:
            results: List of benchmark results
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        if not results:
            logger.warning("No results to export")
            return ""
        
        # Flatten results for CSV
        rows = []
        for result in results:
            flat_result = self._flatten_dict(result)
            rows.append(flat_result)
        
        # Write CSV
        csv_path = self.output_dir / filename
        fieldnames = list(rows[0].keys())
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Generated CSV report: {csv_path}")
        return str(csv_path)
    
    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        filename: str = "benchmark_report.md",
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            results: Benchmark results
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        md_lines = [
            "# MinerU Performance Benchmark Report",
            "",
            f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "",
        ]
        
        # Add configuration
        config = results.get("config", {})
        if config:
            md_lines.append("```json")
            md_lines.append(json.dumps(config, indent=2))
            md_lines.append("```")
            md_lines.append("")
        
        # Add throughput metrics
        throughput = results.get("throughput", {})
        if throughput:
            md_lines.extend([
                "## Throughput Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            
            for key, value in throughput.items():
                if isinstance(value, (int, float)):
                    md_lines.append(f"| {key} | {value:.4f} |")
            
            md_lines.append("")
        
        # Add latency metrics
        latency = results.get("throughput", {})
        if latency:
            md_lines.extend([
                "## Latency Distribution",
                "",
                "| Percentile | Latency (s) |",
                "|------------|-------------|",
            ])
            
            for p in ["avg", "min", "max", "p50", "p90", "p95", "p99"]:
                key = f"{p}_latency" if p not in ["avg", "min", "max"] else f"{p}_latency"
                if key in latency:
                    md_lines.append(f"| {p.upper()} | {latency[key]:.4f} |")
            
            md_lines.append("")
        
        # Add resource metrics
        resources = results.get("resources", {})
        if resources:
            md_lines.extend([
                "## Resource Utilization",
                "",
                "| Resource | Average | Maximum |",
                "|----------|---------|---------|",
            ])
            
            cpu_avg = resources.get("cpu_avg_percent", 0)
            mem_avg = resources.get("memory_avg_percent", 0)
            md_lines.append(f"| CPU | {cpu_avg:.1f}% | - |")
            md_lines.append(f"| Memory | {mem_avg:.1f}% | - |")
            
            md_lines.append("")
        
        # Write report
        md_path = self.output_dir / filename
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Generated Markdown report: {md_path}")
        return str(md_path)
    
    def generate_grid_search_report(
        self,
        grid_results: Dict[str, Any],
        filename: str = "grid_search_report.md",
    ) -> str:
        """
        Generate grid search report.
        
        Args:
            grid_results: Grid search results
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        md_lines = [
            "# Grid Search Optimization Report",
            "",
            f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Add summary
        summary = grid_results.get("report", {}).get("summary", {})
        if summary:
            md_lines.extend([
                "## Summary",
                "",
                f"- Total Tests: {summary.get('total_tests', 0)}",
                f"- Successful: {summary.get('successful_tests', 0)}",
                f"- Failed: {summary.get('failed_tests', 0)}",
                "",
            ])
        
        # Add optimal configuration
        optimal = grid_results.get("optimal_config", {})
        if optimal:
            md_lines.extend([
                "## Optimal Configuration",
                "",
                "```json",
                json.dumps(optimal, indent=2),
                "```",
                "",
            ])
        
        # Add top 5 results
        results = grid_results.get("results", [])
        if results:
            # Sort by QPS
            sorted_results = sorted(
                [r for r in results if "error" not in r],
                key=lambda r: r.get("throughput", {}).get("qps", 0),
                reverse=True,
            )[:5]
            
            md_lines.extend([
                "## Top 5 Configurations by QPS",
                "",
                "| Rank | Devices | Instances | Concurrency | Batch Size | QPS | Avg Latency |",
                "|------|---------|-----------|-------------|------------|-----|-------------|",
            ])
            
            for i, result in enumerate(sorted_results, 1):
                test_case = result.get("test_case", {})
                throughput = result.get("throughput", {})
                md_lines.append(
                    f"| {i} | {len(test_case.get('devices', []))} | "
                    f"{test_case.get('instances_per_device', 0)} | "
                    f"{test_case.get('concurrency', 0)} | "
                    f"{test_case.get('batch_size', 0)} | "
                    f"{throughput.get('qps', 0):.2f} | "
                    f"{throughput.get('avg_latency', 0):.4f} |"
                )
            
            md_lines.append("")
        
        # Write report
        md_path = self.output_dir / filename
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Generated grid search report: {md_path}")
        return str(md_path)
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '_',
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
