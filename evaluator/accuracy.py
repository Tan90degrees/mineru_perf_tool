# Copyright (c) Opendatalab. All rights reserved.
"""
Accuracy evaluation using OmniDocBench.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class AccuracyEvaluator:
    """
    Evaluate MinerU parsing accuracy using OmniDocBench.
    
    Integrates with OmniDocBench evaluation framework for:
    - End-to-end evaluation (text, formula, table, reading order)
    - Layout detection
    - Formula recognition
    - OCR text recognition
    - Table recognition
    """
    
    def __init__(
        self,
        omnidocbench_path: str,
        output_dir: str = "./eval_output",
        config_template: Optional[str] = None,
    ):
        """
        Initialize accuracy evaluator.
        
        Args:
            omnidocbench_path: Path to OmniDocBench directory
            output_dir: Output directory for evaluation results
            config_template: Path to OmniDocBench config template
        """
        self.omnidocbench_path = Path(omnidocbench_path)
        self.output_dir = Path(output_dir)
        self.config_template = config_template or self._get_default_config()
        
        # Verify OmniDocBench installation
        self._verify_installation()
        
    def _verify_installation(self) -> None:
        """Verify OmniDocBench is properly installed."""
        if not self.omnidocbench_path.exists():
            raise FileNotFoundError(f"OmniDocBench not found at {self.omnidocbench_path}")
        
        required_files = [
            "pdf_validation.py",
            "OmniDocBench.json",
        ]
        
        for file_name in required_files:
            file_path = self.omnidocbench_path / file_name
            if not file_path.exists():
                # Try to find OmniDocBench.json in root or dataset subdirectory
                if file_name == "OmniDocBench.json":
                    # Check common locations
                    possible_paths = [
                        self.omnidocbench_path / "OmniDocBench.json",
                        self.omnidocbench_path / "dataset" / "OmniDocBench.json",
                        self.omnidocbench_path / "demo_data" / "omnidocbench_demo" / "OmniDocBench_demo.json",
                    ]
                    if any(p.exists() for p in possible_paths):
                        continue
                
                logger.warning(f"Required file not found: {file_path}")
    
    def _get_default_config(self) -> str:
        """Get default OmniDocBench config template."""
        config_path = self.omnidocbench_path / "configs" / "end2end.yaml"
        if config_path.exists():
            return str(config_path)
        return ""
    
    def _create_config(
        self,
        prediction_dir: str,
        output_config: str,
        ground_truth_path: Optional[str] = None,
        match_method: str = "quick_match",
    ) -> str:
        """
        Create OmniDocBench evaluation config.
        
        Args:
            prediction_dir: Directory containing prediction markdown files
            output_config: Output config file path
            ground_truth_path: Path to ground truth JSON file
            match_method: Matching method (quick_match/simple_match/no_split)
            
        Returns:
            Path to created config file
        """
        if not ground_truth_path:
            # Try to find ground truth
            possible_paths = [
                self.omnidocbench_path / "OmniDocBench.json",
                self.omnidocbench_path / "dataset" / "OmniDocBench.json",
            ]
            for path in possible_paths:
                if path.exists():
                    ground_truth_path = str(path)
                    break
        
        if not ground_truth_path or not Path(ground_truth_path).exists():
            raise FileNotFoundError("Ground truth file not found. Please specify ground_truth_path")
        
        config_content = f"""end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
        - BLEU
        - METEOR
    display_formula:
      metric:
        - Edit_dist
        - CDM_plain
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {ground_truth_path}
    prediction:
      data_path: {prediction_dir}
    match_method: {match_method}
"""
        
        config_path = Path(output_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Created evaluation config: {config_path}")
        return str(config_path)
    
    def run_evaluation(
        self,
        prediction_dir: str,
        ground_truth_path: Optional[str] = None,
        match_method: str = "quick_match",
    ) -> Dict[str, Any]:
        """
        Run OmniDocBench evaluation on predictions.
        
        Args:
            prediction_dir: Directory containing prediction markdown files
            ground_truth_path: Path to ground truth JSON file
            match_method: Matching method
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Running accuracy evaluation")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config
        config_path = self.output_dir / "eval_config.yaml"
        self._create_config(
            prediction_dir=prediction_dir,
            output_config=str(config_path),
            ground_truth_path=ground_truth_path,
            match_method=match_method,
        )
        
        # Run OmniDocBench evaluation
        result_file = self.output_dir / "eval_result.json"
        
        cmd = [
            "python",
            str(self.omnidocbench_path / "pdf_validation.py"),
            "--config",
            str(config_path),
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.omnidocbench_path),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
            
            # Parse results
            results = self._parse_results()
            
            logger.info("Accuracy evaluation completed successfully")
            return {
                "success": True,
                "results": results,
                "stdout": result.stdout,
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timeout")
            return {
                "success": False,
                "error": "Evaluation timeout after 1 hour",
            }
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _parse_results(self) -> Dict[str, Any]:
        """Parse OmniDocBench results."""
        results = {}
        
        # Find result files
        result_dir = self.omnidocbench_path / "result"
        if not result_dir.exists():
            result_dir = self.output_dir
        
        # Look for metric result JSON
        for file_path in result_dir.glob("*_metric_result.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.update(data)
            except Exception as e:
                logger.warning(f"Failed to parse result file {file_path}: {e}")
        
        return results
    
    def get_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall score from evaluation results.
        
        Overall = ((1 - Text Edit Distance) * 100 + Table TEDS + Formula CDM) / 3
        """
        if not results.get("success"):
            return 0.0
        
        eval_results = results.get("results", {})
        
        # Extract metrics
        text_edit_dist = eval_results.get("text_block", {}).get("Edit_dist", 1.0)
        table_teds = eval_results.get("table", {}).get("TEDS", 0.0)
        formula_cdm = eval_results.get("display_formula", {}).get("CDM", 0.0)
        
        overall = ((1 - text_edit_dist) * 100 + table_teds * 100 + formula_cdm) / 3
        
        return overall
