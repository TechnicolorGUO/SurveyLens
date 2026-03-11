"""
Analysis utilities for evaluation results.

This module provides functions to read evaluation summary JSON files
and aggregate results by different dimensions (system, category, etc.),
then export to CSV for further analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


logger = logging.getLogger(__name__)


class EvaluationResultsAnalyzer:
    """Analyzer for evaluation summary JSON files."""

    def __init__(
        self,
        results_path: str,
        ams_metric: str = "f1",
        score_source: str = "files",
    ):
        """
        Initialize analyzer with evaluation results JSON.
        
        Args:
            results_path: Path to evaluation_summary JSON file
            ams_metric: Which metric to extract from AMS dict
                (f1, precision, recall, thresholded_ams)
            score_source: "files" to aggregate from file-level scores,
                "averages" to aggregate from per-category averages
        """
        self.results_path = Path(results_path)
        self.ams_metric = ams_metric
        self.score_source = score_source
        self.data = self._load_results()
        self.timestamp = self._extract_timestamp()
        self.eval_type = self._detect_eval_type()
        self.has_float_scores = False
        self.has_ams_dict_scores = False
        self._scan_score_formats()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        with open(self.results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Loaded results from {self.results_path}")
        return data
    
    def _extract_timestamp(self) -> str:
        """
        Extract timestamp from filename or JSON, or generate a new one.
        
        Returns:
            Timestamp string in format YYYYMMDD_HHMMSS
        """
        # Try to extract from filename (e.g., evaluation_summary_20251230_191134.json)
        filename = self.results_path.stem
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        if match:
            logger.info(f"Extracted timestamp from filename: {match.group(1)}")
            return match.group(1)
        
        # Try to extract from JSON content
        timestamp_from_json = self.data.get("generated_at")
        if timestamp_from_json:
            logger.info(f"Extracted timestamp from JSON: {timestamp_from_json}")
            return timestamp_from_json
        
        # Generate new timestamp
        new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Generated new timestamp: {new_timestamp}")
        return new_timestamp
    
    def _detect_eval_type(self) -> str:
        """
        Detect whether this is qualitative or quantitative evaluation.
        
        Qualitative: scores are dicts with 'score' and 'notes' keys
        Quantitative: scores are direct numbers (similarity scores)
        
        Returns:
            "qualitative" or "quantitative"
        """
        # 1) Filename hint (most reliable for the current pipelines)
        filename = self.results_path.name.lower()
        if "quantitative" in filename:
            logger.info("Detected evaluation type from filename: quantitative")
            return "quantitative"
        if "qualitative" in filename:
            logger.info("Detected evaluation type from filename: qualitative")
            return "qualitative"

        # 2) Scan all files until we find a non-None score
        by_system = self.data.get("by_system", {})
        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                for file_entry in files:
                    file_scores = file_entry.get("scores", {})
                    for aspect in ["outline", "content", "reference"]:
                        score_data = file_scores.get(aspect)
                        if score_data is None:
                            continue
                        # If it's a dict with "score" key, it's qualitative
                        if isinstance(score_data, dict) and "score" in score_data:
                            logger.info("Detected evaluation type: qualitative (LLM-based)")
                            return "qualitative"
                        # AMS quantitative format uses precision/recall/f1 dict
                        if isinstance(score_data, dict) and (
                            "f1" in score_data
                            or {"precision", "recall", "f1"}.issubset(score_data.keys())
                        ):
                            logger.info("Detected evaluation type: quantitative (AMS metrics)")
                            return "quantitative"
                        # If it's a direct number, it's quantitative
                        if isinstance(score_data, (int, float)):
                            logger.info("Detected evaluation type: quantitative (embedding similarity)")
                            return "quantitative"

        # Default to qualitative if can't determine
        logger.warning("Could not determine evaluation type, defaulting to qualitative")
        return "qualitative"

    def _iter_score_data(self) -> List[Any]:
        """Iterate over score data entries for all aspects."""
        results: List[Any] = []
        by_system = self.data.get("by_system", {})
        for _, categories in by_system.items():
            for _, cat_data in categories.items():
                if self.score_source == "averages":
                    avg_scores = cat_data.get("averages", {})
                    for aspect in ["outline", "content", "reference"]:
                        results.append(avg_scores.get(aspect))
                else:
                    files = cat_data.get("files", [])
                    for file_entry in files:
                        file_scores = file_entry.get("scores", {})
                        for aspect in ["outline", "content", "reference"]:
                            results.append(file_scores.get(aspect))
        return results

    def _iter_diagnostic_data(self) -> List[Any]:
        """Iterate over diagnostic data entries for all aspects."""
        results: List[Any] = []
        by_system = self.data.get("by_system", {})
        for _, categories in by_system.items():
            for _, cat_data in categories.items():
                files = cat_data.get("files", [])
                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    for aspect in ["outline", "content", "reference"]:
                        results.append(diagnostics.get(aspect))
        return results

    def _scan_score_formats(self) -> None:
        """Detect whether results contain float scores or AMS dict scores."""
        for score_data in self._iter_score_data():
            if score_data is None:
                continue
            if isinstance(score_data, (int, float)):
                self.has_float_scores = True
            elif isinstance(score_data, dict):
                if any(
                    isinstance(score_data.get(k), (int, float))
                    for k in ["f1", "precision", "recall", "thresholded_ams", "bms", "t_ams"]
                ):
                    self.has_ams_dict_scores = True

    def _detect_available_metrics(self) -> List[str]:
        """Detect which AMS metrics are present in the data."""
        if self.eval_type != "quantitative":
            return []
        metrics = set()
        for score_data in self._iter_score_data():
            if score_data is None:
                continue
            if isinstance(score_data, (int, float)):
                metrics.add("ams")
                continue
            if isinstance(score_data, dict):
                for key in ["f1", "precision", "recall", "thresholded_ams", "bms", "t_ams"]:
                    if isinstance(score_data.get(key), (int, float)):
                        metrics.add(key)
        ordered = ["f1", "precision", "recall", "thresholded_ams", "bms", "t_ams", "ams"]
        return [m for m in ordered if m in metrics]

    def _detect_available_diagnostics(self) -> List[str]:
        """Detect which diagnostic metrics are present in the data."""
        metrics = set()
        for diag_data in self._iter_diagnostic_data():
            if not isinstance(diag_data, dict):
                continue
            for key, value in diag_data.items():
                if isinstance(value, (int, float)):
                    metrics.add(key)
        ordered = ["t_ams", "redundancy", "dup_rate"]
        return [m for m in ordered if m in metrics] + sorted(m for m in metrics if m not in ordered)

    def _extract_diagnostic_value(self, diag_data: Any, metric: str) -> Optional[float]:
        """Extract a numeric diagnostic metric value."""
        if not isinstance(diag_data, dict):
            return None
        value = diag_data.get(metric)
        if isinstance(value, (int, float)):
            return float(value)
        return None
    
    def _extract_score(self, score_data: Any) -> Optional[float]:
        """
        Extract numeric score from score data.
        
        Handles both formats:
        - Qualitative: {"score": 3.5, "notes": "..."}
        - Quantitative: 0.85
        
        Args:
            score_data: Score data in either format
            
        Returns:
            Numeric score or None
        """
        if score_data is None:
            return None
        
        # Quantitative format (direct number)
        if isinstance(score_data, (int, float)):
            if self.eval_type == "quantitative":
                if self.ams_metric == "ams" or not self.has_ams_dict_scores:
                    return score_data
                return None
            return score_data
        
        # Qualitative format (dict with "score" key)
        if isinstance(score_data, dict):
            score = score_data.get("score")
            if isinstance(score, (int, float)):
                return score
            # AMS quantitative format (dict with precision/recall/f1/thresholded_ams/bms/t_ams)
            metric_value = score_data.get(self.ams_metric)
            if isinstance(metric_value, (int, float)):
                return metric_value
        
        return None

    def _compute_aspect_average(self, aspect_data: Any) -> tuple[Optional[float], Optional[int]]:
        """
        Compute average score for a single aspect entry.

        Supports:
        - per-aspect scoring: aspect_data["score"]
        - per-criterion scoring: mean of aspect_data["criteria"][*]["score"]
        """
        if not isinstance(aspect_data, dict):
            return None, None

        criteria = aspect_data.get("criteria")
        if isinstance(criteria, list):
            scores = [
                c.get("score")
                for c in criteria
                if isinstance(c, dict) and isinstance(c.get("score"), (int, float))
            ]
            if scores:
                return sum(scores) / len(scores), len(scores)

        score = aspect_data.get("score")
        if isinstance(score, (int, float)):
            return float(score), None

        return None, None
    
    def aggregate_by_system(self) -> List[Dict[str, Any]]:
        """
        Aggregate scores by system (averaging across all categories).
        
        Returns:
            List of dicts with system-level averages:
            [{"system": "Gemini", "outline": 3.5, "content": 4.0, "reference": 3.8, "count": 100}, ...]
        """
        results = []
        by_system = self.data.get("by_system", {})
        
        for system, categories in by_system.items():
            scores_sum = defaultdict(float)
            scores_count = defaultdict(int)
            total_files = 0
            
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                total_files += len(files)
                
                for aspect in ["outline", "content", "reference"]:
                    if self.score_source == "averages":
                        avg_scores = cat_data.get("averages", {})
                        score = self._extract_score(avg_scores.get(aspect))
                        if score is not None and len(files) > 0:
                            scores_sum[aspect] += score * len(files)
                            scores_count[aspect] += len(files)
                    else:
                        for file_entry in files:
                            file_scores = file_entry.get("scores", {})
                            score = self._extract_score(file_scores.get(aspect))
                            if score is not None:
                                scores_sum[aspect] += score
                                scores_count[aspect] += 1
            
            result = {"system": system, "count": total_files}
            for aspect in ["outline", "content", "reference"]:
                if scores_count[aspect] > 0:
                    result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
                else:
                    result[aspect] = None
            
            # Calculate overall average
            valid_scores = [v for k, v in result.items() 
                          if k in ["outline", "content", "reference"] and v is not None]
            if valid_scores:
                result["average"] = round(sum(valid_scores) / len(valid_scores), 3)
            else:
                result["average"] = None
            
            results.append(result)
        
        return sorted(results, key=lambda x: x["system"])
    
    def aggregate_by_category(self) -> List[Dict[str, Any]]:
        """
        Aggregate scores by category/domain (averaging across all systems).
        
        Returns:
            List of dicts with category-level averages:
            [{"category": "Biology", "outline": 3.2, "content": 3.9, "reference": 3.5, "count": 50}, ...]
        """
        category_scores_sum = defaultdict(lambda: defaultdict(float))
        category_scores_count = defaultdict(lambda: defaultdict(int))
        category_counts = defaultdict(int)
        
        by_system = self.data.get("by_system", {})
        
        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                category_counts[category] += len(files)
                
                for aspect in ["outline", "content", "reference"]:
                    if self.score_source == "averages":
                        avg_scores = cat_data.get("averages", {})
                        score = self._extract_score(avg_scores.get(aspect))
                        if score is not None and len(files) > 0:
                            category_scores_sum[category][aspect] += score * len(files)
                            category_scores_count[category][aspect] += len(files)
                    else:
                        for file_entry in files:
                            file_scores = file_entry.get("scores", {})
                            score = self._extract_score(file_scores.get(aspect))
                            if score is not None:
                                category_scores_sum[category][aspect] += score
                                category_scores_count[category][aspect] += 1
        
        results = []
        for category in sorted(category_scores_sum.keys()):
            result = {"category": category, "count": category_counts[category]}
            
            for aspect in ["outline", "content", "reference"]:
                if category_scores_count[category][aspect] > 0:
                    total = category_scores_sum[category][aspect]
                    count = category_scores_count[category][aspect]
                    result[aspect] = round(total / count, 3)
                else:
                    result[aspect] = None
            
            # Calculate overall average
            valid_scores = [v for k, v in result.items() 
                          if k in ["outline", "content", "reference"] and v is not None]
            if valid_scores:
                result["average"] = round(sum(valid_scores) / len(valid_scores), 3)
            else:
                result["average"] = None
            
            results.append(result)
        
        return results
    
    def aggregate_by_system_category(self) -> List[Dict[str, Any]]:
        """
        Aggregate scores by system-category combination.
        
        Returns:
            List of dicts with system+category averages:
            [{"system": "Gemini", "category": "Biology", "outline": 3.5, ...}, ...]
        """
        results = []
        by_system = self.data.get("by_system", {})
        
        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                scores_sum = defaultdict(float)
                scores_count = defaultdict(int)
                
                for aspect in ["outline", "content", "reference"]:
                    if self.score_source == "averages":
                        avg_scores = cat_data.get("averages", {})
                        score = self._extract_score(avg_scores.get(aspect))
                        if score is not None and len(files) > 0:
                            scores_sum[aspect] += score * len(files)
                            scores_count[aspect] += len(files)
                    else:
                        for file_entry in files:
                            file_scores = file_entry.get("scores", {})
                            score = self._extract_score(file_scores.get(aspect))
                            if score is not None:
                                scores_sum[aspect] += score
                                scores_count[aspect] += 1
                
                result = {
                    "system": system,
                    "category": category,
                    "count": len(files)
                }
                
                for aspect in ["outline", "content", "reference"]:
                    if scores_count[aspect] > 0:
                        result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
                    else:
                        result[aspect] = None
                
                # Calculate overall average
                valid_scores = [v for k, v in result.items() 
                              if k in ["outline", "content", "reference"] and v is not None]
                if valid_scores:
                    result["average"] = round(sum(valid_scores) / len(valid_scores), 3)
                else:
                    result["average"] = None
                
                results.append(result)
        
        return sorted(results, key=lambda x: (x["system"], x["category"]))
    
    def aggregate_overall(self) -> Dict[str, Any]:
        """
        Aggregate scores across all systems and categories.
        
        Returns:
            Dict with overall averages:
            {"outline": 3.5, "content": 4.0, "reference": 3.8, "count": 1000}
        """
        scores_sum = defaultdict(float)
        scores_count = defaultdict(int)
        total_files = 0
        
        by_system = self.data.get("by_system", {})
        
        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                total_files += len(files)
                
                for aspect in ["outline", "content", "reference"]:
                    if self.score_source == "averages":
                        avg_scores = cat_data.get("averages", {})
                        score = self._extract_score(avg_scores.get(aspect))
                        if score is not None and len(files) > 0:
                            scores_sum[aspect] += score * len(files)
                            scores_count[aspect] += len(files)
                    else:
                        for file_entry in files:
                            file_scores = file_entry.get("scores", {})
                            score = self._extract_score(file_scores.get(aspect))
                            if score is not None:
                                scores_sum[aspect] += score
                                scores_count[aspect] += 1
        
        result = {"count": total_files}
        for aspect in ["outline", "content", "reference"]:
            if scores_count[aspect] > 0:
                result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
            else:
                result[aspect] = None
        
        # Calculate overall average
        valid_scores = [v for k, v in result.items() 
                       if k in ["outline", "content", "reference"] and v is not None]
        if valid_scores:
            result["average"] = round(sum(valid_scores) / len(valid_scores), 3)
        else:
            result["average"] = None
        
        return result

    def aggregate_diagnostics_by_system(self, metric: str) -> List[Dict[str, Any]]:
        """Aggregate diagnostic metrics by system."""
        results = []
        by_system = self.data.get("by_system", {})

        for system, categories in by_system.items():
            scores_sum = defaultdict(float)
            scores_count = defaultdict(int)
            total_files = 0

            for _, cat_data in categories.items():
                files = cat_data.get("files", [])
                total_files += len(files)

                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    for aspect in ["outline", "content", "reference"]:
                        value = self._extract_diagnostic_value(diagnostics.get(aspect), metric)
                        if value is not None:
                            scores_sum[aspect] += value
                            scores_count[aspect] += 1

            result = {"system": system, "count": total_files}
            for aspect in ["outline", "content", "reference"]:
                if scores_count[aspect] > 0:
                    result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
                else:
                    result[aspect] = None

            valid_scores = [
                v for k, v in result.items()
                if k in ["outline", "content", "reference"] and v is not None
            ]
            result["average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
            results.append(result)

        return sorted(results, key=lambda x: x["system"])

    def aggregate_diagnostics_by_category(self, metric: str) -> List[Dict[str, Any]]:
        """Aggregate diagnostic metrics by category."""
        category_scores_sum = defaultdict(lambda: defaultdict(float))
        category_scores_count = defaultdict(lambda: defaultdict(int))
        category_counts = defaultdict(int)

        by_system = self.data.get("by_system", {})

        for _, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                category_counts[category] += len(files)

                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    for aspect in ["outline", "content", "reference"]:
                        value = self._extract_diagnostic_value(diagnostics.get(aspect), metric)
                        if value is not None:
                            category_scores_sum[category][aspect] += value
                            category_scores_count[category][aspect] += 1

        results = []
        for category in sorted(category_scores_sum.keys()):
            result = {"category": category, "count": category_counts[category]}
            for aspect in ["outline", "content", "reference"]:
                if category_scores_count[category][aspect] > 0:
                    total = category_scores_sum[category][aspect]
                    count = category_scores_count[category][aspect]
                    result[aspect] = round(total / count, 3)
                else:
                    result[aspect] = None

            valid_scores = [
                v for k, v in result.items()
                if k in ["outline", "content", "reference"] and v is not None
            ]
            result["average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
            results.append(result)

        return results

    def aggregate_diagnostics_by_system_category(self, metric: str) -> List[Dict[str, Any]]:
        """Aggregate diagnostic metrics by system-category combination."""
        results = []
        by_system = self.data.get("by_system", {})

        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                scores_sum = defaultdict(float)
                scores_count = defaultdict(int)

                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    for aspect in ["outline", "content", "reference"]:
                        value = self._extract_diagnostic_value(diagnostics.get(aspect), metric)
                        if value is not None:
                            scores_sum[aspect] += value
                            scores_count[aspect] += 1

                result = {"system": system, "category": category, "count": len(files)}
                for aspect in ["outline", "content", "reference"]:
                    if scores_count[aspect] > 0:
                        result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
                    else:
                        result[aspect] = None

                valid_scores = [
                    v for k, v in result.items()
                    if k in ["outline", "content", "reference"] and v is not None
                ]
                result["average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
                results.append(result)

        return sorted(results, key=lambda x: (x["system"], x["category"]))

    def aggregate_diagnostics_overall(self, metric: str) -> Dict[str, Any]:
        """Aggregate diagnostic metrics across all systems and categories."""
        scores_sum = defaultdict(float)
        scores_count = defaultdict(int)
        total_files = 0

        by_system = self.data.get("by_system", {})

        for _, categories in by_system.items():
            for _, cat_data in categories.items():
                files = cat_data.get("files", [])
                total_files += len(files)

                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    for aspect in ["outline", "content", "reference"]:
                        value = self._extract_diagnostic_value(diagnostics.get(aspect), metric)
                        if value is not None:
                            scores_sum[aspect] += value
                            scores_count[aspect] += 1

        result = {"count": total_files}
        for aspect in ["outline", "content", "reference"]:
            if scores_count[aspect] > 0:
                result[aspect] = round(scores_sum[aspect] / scores_count[aspect], 3)
            else:
                result[aspect] = None

        valid_scores = [
            v for k, v in result.items()
            if k in ["outline", "content", "reference"] and v is not None
        ]
        result["average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
        return result

    def get_detailed_diagnostics(self, metric: str) -> List[Dict[str, Any]]:
        """Get per-file diagnostic metrics for each aspect."""
        results = []
        by_system = self.data.get("by_system", {})

        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                for file_entry in files:
                    diagnostics = file_entry.get("diagnostics", {})
                    result = {
                        "system": system,
                        "category": category,
                        "file": file_entry.get("file", ""),
                    }
                    for aspect in ["outline", "content", "reference"]:
                        value = self._extract_diagnostic_value(diagnostics.get(aspect), metric)
                        result[aspect] = value

                    valid_scores = [
                        v for k, v in result.items()
                        if k in ["outline", "content", "reference"] and v is not None
                    ]
                    result["average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
                    results.append(result)

        return results
    
    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """
        Get detailed file-level results.
        
        Returns:
            List of dicts with per-file scores:
            [{"system": "Gemini", "category": "Biology", "file": "...", 
              "outline": 3, "content": 4, "reference": 3}, ...]
        """
        results = []
        by_system = self.data.get("by_system", {})
        
        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])
                
                for file_entry in files:
                    file_scores = file_entry.get("scores", {})
                    result = {
                        "system": system,
                        "category": category,
                        "file": file_entry.get("file", ""),
                    }
                    
                    for aspect in ["outline", "content", "reference"]:
                        score = self._extract_score(file_scores.get(aspect))
                        result[aspect] = score
                    
                    # Calculate average for this file
                    valid_scores = [v for k, v in result.items() 
                                  if k in ["outline", "content", "reference"] and v is not None]
                    if valid_scores:
                        result["average"] = round(sum(valid_scores) / len(valid_scores), 3)
                    else:
                        result["average"] = None
                    
                    results.append(result)
        
        return results

    def get_aspect_level_results(self) -> List[Dict[str, Any]]:
        """
        Get aspect-level averages for per-aspect/per-criterion scoring outputs.

        Returns:
            List of dicts with per-aspect averages:
            [{"system": "...", "category": "...", "file": "...",
              "aspect_group": "reference", "aspect_name": "...",
              "aspect_average": 3.2, "criteria_count": 5}, ...]
        """
        results = []
        by_system = self.data.get("by_system", {})

        for system, categories in by_system.items():
            for category, cat_data in categories.items():
                files = cat_data.get("files", [])

                for file_entry in files:
                    file_scores = file_entry.get("scores", {})
                    for aspect_group in ["outline", "content", "reference"]:
                        aspect_payload = file_scores.get(aspect_group)
                        if not isinstance(aspect_payload, dict):
                            continue
                        aspects = aspect_payload.get("aspects")
                        if not isinstance(aspects, list):
                            continue

                        for aspect_data in aspects:
                            if not isinstance(aspect_data, dict):
                                continue
                            aspect_name = (
                                aspect_data.get("aspect_name")
                                or aspect_data.get("name")
                                or aspect_data.get("aspect")
                                or ""
                            )
                            avg, count = self._compute_aspect_average(aspect_data)
                            if avg is None:
                                continue
                            results.append(
                                {
                                    "system": system,
                                    "category": category,
                                    "file": file_entry.get("file", ""),
                                    "aspect_group": aspect_group,
                                    "aspect_name": aspect_name,
                                    "aspect_average": round(avg, 3),
                                    "criteria_count": count,
                                }
                            )

        return results
    
    def export_to_csv(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Export aggregated data to CSV file.
        
        Args:
            data: List of dicts to export
            output_path: Path to output CSV file
        """
        if not data:
            logger.warning("No data to export")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(data[0].keys())
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Exported {len(data)} rows to {output_path}")
    
    def export_all(self, output_dir: str) -> None:
        """
        Export all aggregation types to separate CSV files.
        
        Args:
            output_dir: Base directory to save CSV files (will create timestamped subdirectory)
        """
        # Create timestamped subdirectory with evaluation type
        output_dir = Path(output_dir) / f"analysis_{self.eval_type}_{self.timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {self.eval_type} results to: {output_dir}")

        metrics = (
            self._detect_available_metrics()
            if self.eval_type == "quantitative"
            else ["score"]
        )
        if not metrics:
            logger.warning("No quantitative metrics detected; nothing to export.")
            return

        original_metric = self.ams_metric
        for metric in metrics:
            self.ams_metric = metric
            suffix = f"_{metric}" if len(metrics) > 1 else ""

            # Export by system
            by_system = self.aggregate_by_system()
            self.export_to_csv(by_system, output_dir / f"aggregated_by_system{suffix}.csv")

            # Export by category
            by_category = self.aggregate_by_category()
            self.export_to_csv(by_category, output_dir / f"aggregated_by_category{suffix}.csv")

            # Export by system-category
            by_sys_cat = self.aggregate_by_system_category()
            self.export_to_csv(
                by_sys_cat,
                output_dir / f"aggregated_by_system_category{suffix}.csv",
            )

            # Export overall
            overall = self.aggregate_overall()
            self.export_to_csv([overall], output_dir / f"aggregated_overall{suffix}.csv")

            # Export detailed results
            detailed = self.get_detailed_results()
            self.export_to_csv(detailed, output_dir / f"detailed_results{suffix}.csv")

        diagnostic_metrics = self._detect_available_diagnostics()
        for diag_metric in diagnostic_metrics:
            suffix = f"_{diag_metric}" if len(diagnostic_metrics) > 1 else ""

            by_system = self.aggregate_diagnostics_by_system(diag_metric)
            self.export_to_csv(by_system, output_dir / f"diagnostics_by_system{suffix}.csv")

            by_category = self.aggregate_diagnostics_by_category(diag_metric)
            self.export_to_csv(by_category, output_dir / f"diagnostics_by_category{suffix}.csv")

            by_sys_cat = self.aggregate_diagnostics_by_system_category(diag_metric)
            self.export_to_csv(
                by_sys_cat,
                output_dir / f"diagnostics_by_system_category{suffix}.csv",
            )

            overall = self.aggregate_diagnostics_overall(diag_metric)
            self.export_to_csv([overall], output_dir / f"diagnostics_overall{suffix}.csv")

            detailed = self.get_detailed_diagnostics(diag_metric)
            self.export_to_csv(detailed, output_dir / f"diagnostics_detailed{suffix}.csv")

        if self.eval_type == "qualitative":
            # Export aspect-level averages (per-aspect/per-criterion scoring)
            aspect_results = self.get_aspect_level_results()
            self.export_to_csv(aspect_results, output_dir / "aspect_averages.csv")

        self.ams_metric = original_metric
        logger.info(f"Exported all aggregations to {output_dir}")

    def export_diagnostics(self, output_dir: str) -> None:
        """
        Export diagnostics aggregations to separate CSV files.

        Args:
            output_dir: Base directory to save CSV files (will create timestamped subdirectory)
        """
        output_dir = Path(output_dir) / f"analysis_{self.eval_type}_{self.timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        diagnostic_metrics = self._detect_available_diagnostics()
        if not diagnostic_metrics:
            logger.warning("No diagnostic metrics detected; nothing to export.")
            return

        for diag_metric in diagnostic_metrics:
            suffix = f"_{diag_metric}" if len(diagnostic_metrics) > 1 else ""

            by_system = self.aggregate_diagnostics_by_system(diag_metric)
            self.export_to_csv(by_system, output_dir / f"diagnostics_by_system{suffix}.csv")

            by_category = self.aggregate_diagnostics_by_category(diag_metric)
            self.export_to_csv(by_category, output_dir / f"diagnostics_by_category{suffix}.csv")

            by_sys_cat = self.aggregate_diagnostics_by_system_category(diag_metric)
            self.export_to_csv(
                by_sys_cat,
                output_dir / f"diagnostics_by_system_category{suffix}.csv",
            )

            overall = self.aggregate_diagnostics_overall(diag_metric)
            self.export_to_csv([overall], output_dir / f"diagnostics_overall{suffix}.csv")

            detailed = self.get_detailed_diagnostics(diag_metric)
            self.export_to_csv(detailed, output_dir / f"diagnostics_detailed{suffix}.csv")

        logger.info(f"Exported diagnostics aggregations to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results and export to CSV."
    )
    parser.add_argument(
        "results_json",
        help="Path to evaluation_summary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Directory to save CSV files (default: results/analysis)"
    )
    parser.add_argument(
        "--aggregation",
        choices=["system", "category", "system-category", "overall", "detailed", "aspect", "diagnostics", "all"],
        default="all",
        help="Type of aggregation to perform (default: all)"
    )
    parser.add_argument(
        "--ams-metric",
        choices=["f1", "precision", "recall", "thresholded_ams", "bms", "t_ams"],
        default="f1",
        help="When scores are AMS dicts, which metric to aggregate (default: f1)"
    )
    parser.add_argument(
        "--score-source",
        choices=["files", "averages"],
        default="files",
        help="Aggregate from file-level scores or category averages (default: files)"
    )
    parser.add_argument(
        "--output-file",
        help="Specific output CSV file (used with non-'all' aggregation types)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    analyzer = EvaluationResultsAnalyzer(
        args.results_json,
        ams_metric=args.ams_metric,
        score_source=args.score_source,
    )
    
    if args.aggregation == "all":
        analyzer.export_all(args.output_dir)
        output_location = Path(args.output_dir) / f"analysis_{analyzer.eval_type}_{analyzer.timestamp}"
    elif args.aggregation == "diagnostics":
        if args.output_file:
            logger.warning("--output-file is ignored for diagnostics aggregation.")
        analyzer.export_diagnostics(args.output_dir)
        output_location = Path(args.output_dir) / f"analysis_{analyzer.eval_type}_{analyzer.timestamp}"
    else:
        # Determine output file
        if args.output_file:
            output_path = args.output_file
            output_location = Path(args.output_file).parent
        else:
            # Create timestamped subdirectory for single aggregation type too
            output_dir = Path(args.output_dir) / f"analysis_{analyzer.eval_type}_{analyzer.timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_location = output_dir
            
            if args.aggregation == "system":
                output_path = output_dir / "aggregated_by_system.csv"
                data = analyzer.aggregate_by_system()
            elif args.aggregation == "category":
                output_path = output_dir / "aggregated_by_category.csv"
                data = analyzer.aggregate_by_category()
            elif args.aggregation == "system-category":
                output_path = output_dir / "aggregated_by_system_category.csv"
                data = analyzer.aggregate_by_system_category()
            elif args.aggregation == "overall":
                output_path = output_dir / "aggregated_overall.csv"
                data = [analyzer.aggregate_overall()]
            elif args.aggregation == "detailed":
                output_path = output_dir / "detailed_results.csv"
                data = analyzer.get_detailed_results()
            elif args.aggregation == "aspect":
                output_path = output_dir / "aspect_averages.csv"
                data = analyzer.get_aspect_level_results()
            else:
                logger.error(f"Unknown aggregation type: {args.aggregation}")
                return
            
            analyzer.export_to_csv(data, output_path)
    
    print(f"Analysis complete. Results saved to {output_location}")


if __name__ == "__main__":
    main()
