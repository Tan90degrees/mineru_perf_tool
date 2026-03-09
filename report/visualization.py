# Copyright (c) Opendatalab. All rights reserved.
"""
Visualization module for benchmark results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn not installed, visualization disabled")


class Visualizer:
    """Generate visualizations for benchmark results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Visualization features disabled - matplotlib not available")
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        title: str = "Latency Distribution",
        filename: str = "latency_distribution.png",
    ) -> Optional[str]:
        """
        Plot latency distribution histogram.
        
        Args:
            latencies: List of latency values
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to generated plot or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not latencies:
            logger.warning("No latency data to plot")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(latencies, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Latency (s)')
        plt.ylabel('Count')
        plt.title('Latency Distribution')
        plt.grid(True, alpha=0.3)
        
        # CDF
        plt.subplot(1, 2, 2)
        sorted_latencies = sorted(latencies)
        cumulative = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        plt.plot(sorted_latencies, cumulative * 100)
        plt.xlabel('Latency (s)')
        plt.ylabel('Cumulative %')
        plt.title('Latency CDF')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated latency distribution plot: {plot_path}")
        return str(plot_path)
    
    def plot_throughput_timeline(
        self,
        timestamps: List[float],
        qps_values: List[float],
        title: str = "Throughput Over Time",
        filename: str = "throughput_timeline.png",
    ) -> Optional[str]:
        """
        Plot throughput over time.
        
        Args:
            timestamps: List of timestamps
            qps_values: List of QPS values
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to generated plot or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not timestamps or not qps_values:
            logger.warning("No throughput data to plot")
            return None
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, qps_values, linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('QPS')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add average line
        avg_qps = sum(qps_values) / len(qps_values)
        plt.axhline(y=avg_qps, color='r', linestyle='--', label=f'Average: {avg_qps:.2f}')
        plt.legend()
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated throughput timeline plot: {plot_path}")
        return str(plot_path)
    
    def plot_heatmap(
        self,
        heatmap_data: Dict[str, Any],
        title: str = "Performance Heatmap",
        filename: str = "performance_heatmap.png",
    ) -> Optional[str]:
        """
        Plot performance heatmap.
        
        Args:
            heatmap_data: Heatmap data from ResultAggregator.get_heatmap_data()
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to generated plot or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        x_param = heatmap_data.get("x_param", "")
        y_param = heatmap_data.get("y_param", "")
        metric = heatmap_data.get("metric", "qps")
        x_values = heatmap_data.get("x_values", [])
        y_values = heatmap_data.get("y_values", [])
        data = heatmap_data.get("data", [])
        
        if not data or not x_values or not y_values:
            logger.warning("No heatmap data to plot")
            return None
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            data,
            xticklabels=x_values,
            yticklabels=y_values,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': metric},
        )
        
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"{title}\n({metric} by {x_param} and {y_param})")
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated heatmap: {plot_path}")
        return str(plot_path)
    
    def plot_resource_usage(
        self,
        snapshots: List[Dict[str, Any]],
        title: str = "Resource Usage Over Time",
        filename: str = "resource_usage.png",
    ) -> Optional[str]:
        """
        Plot resource usage over time.
        
        Args:
            snapshots: List of resource snapshots
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to generated plot or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not snapshots:
            logger.warning("No resource data to plot")
            return None
        
        # Extract data
        timestamps = [s.get("timestamp", 0) for s in snapshots]
        cpu_values = [s.get("cpu_percent", 0) for s in snapshots]
        memory_values = [s.get("memory_percent", 0) for s in snapshots]
        
        # Normalize timestamps to relative time
        if timestamps:
            start_time = timestamps[0]
            timestamps = [t - start_time for t in timestamps]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU usage
        axes[0].plot(timestamps, cpu_values, linewidth=2, color='blue')
        axes[0].set_ylabel('CPU %')
        axes[0].set_title('CPU Usage')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Memory usage
        axes[1].plot(timestamps, memory_values, linewidth=2, color='green')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Memory %')
        axes[1].set_title('Memory Usage')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated resource usage plot: {plot_path}")
        return str(plot_path)
    
    def plot_comparison_bar(
        self,
        results: List[Dict[str, Any]],
        metric: str = "qps",
        title: str = "Performance Comparison",
        filename: str = "comparison_bar.png",
    ) -> Optional[str]:
        """
        Plot comparison bar chart.
        
        Args:
            results: List of benchmark results
            metric: Metric to compare
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to generated plot or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not results:
            logger.warning("No comparison data to plot")
            return None
        
        # Extract data
        labels = []
        values = []
        
        for result in results:
            test_case = result.get("test_case", {})
            throughput = result.get("throughput", {})
            
            label = f"D:{len(test_case.get('devices', []))} " \
                    f"I:{test_case.get('instances_per_device', 0)} " \
                    f"C:{test_case.get('concurrency', 0)} " \
                    f"B:{test_case.get('batch_size', 0)}"
            
            labels.append(label)
            values.append(throughput.get(metric, 0))
        
        # Create bar chart
        plt.figure(figsize=(14, 6))
        x_pos = np.arange(len(labels))
        
        bars = plt.bar(x_pos, values, alpha=0.8)
        
        # Color bars based on value
        max_val = max(values) if values else 0
        for bar, val in zip(bars, values):
            if val == max_val:
                bar.set_color('green')
            elif val >= max_val * 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            plt.text(i, v + max(values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated comparison bar chart: {plot_path}")
        return str(plot_path)
