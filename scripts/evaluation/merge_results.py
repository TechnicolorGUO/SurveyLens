"""
Merge multiple analysis result directories into aggregated CSVs with mean/std.

Given N analysis directories (each produced by analyze_results.py), this script
creates one merged CSV per shared filename. For numeric columns, each column is
expanded into "<col>_mean" and "<col>_std" across runs. Non-numeric columns are
kept as identifiers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple analysis result directories into mean/std CSVs."
    )
    parser.add_argument(
        "analysis_paths",
        nargs="+",
        help="Paths to analysis directories (e.g., results/analysis/analysis_*).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save merged CSVs. "
            "Default: results/analysis/merged_<timestamp>."
        ),
    )
    return parser.parse_args()


def list_csv_files(analysis_dir: Path) -> List[str]:
    return sorted(p.name for p in analysis_dir.glob("*.csv") if p.is_file())


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, fieldnames


def _parse_float(value: str) -> Tuple[bool, float]:
    if value is None:
        return False, 0.0
    text = str(value).strip()
    if text == "":
        return False, 0.0
    try:
        num = float(text)
    except ValueError:
        return False, 0.0
    if math.isnan(num) or math.isinf(num):
        return False, 0.0
    return True, num


def detect_numeric_columns(
    all_rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]
) -> List[str]:
    numeric_cols: List[str] = []
    for col in fieldnames:
        is_numeric = True
        seen_value = False
        for row in all_rows:
            raw = row.get(col, "")
            ok, _ = _parse_float(raw)
            if raw is None or str(raw).strip() == "":
                continue
            seen_value = True
            if not ok:
                is_numeric = False
                break
        if is_numeric and seen_value:
            numeric_cols.append(col)
    return numeric_cols


def _compute_mean_std(values: List[float]) -> Tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def merge_csvs(
    csv_name: str,
    analysis_dirs: Sequence[Path],
    output_dir: Path,
) -> None:
    datasets: List[Tuple[List[Dict[str, str]], List[str]]] = []
    all_rows_for_detection: List[Dict[str, str]] = []
    fieldnames_order: List[str] = []
    seen_fields = set()

    for analysis_dir in analysis_dirs:
        csv_path = analysis_dir / csv_name
        rows, fieldnames = read_csv_rows(csv_path)
        datasets.append((rows, fieldnames))
        all_rows_for_detection.extend(rows)
        for name in fieldnames:
            if name not in seen_fields:
                seen_fields.add(name)
                fieldnames_order.append(name)

    numeric_cols = detect_numeric_columns(all_rows_for_detection, fieldnames_order)
    key_cols = [c for c in fieldnames_order if c not in numeric_cols]

    merged: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    key_order: List[Tuple[str, ...]] = []

    for rows, _ in datasets:
        for row in rows:
            key = tuple(row.get(c, "") for c in key_cols)
            if key not in merged:
                merged[key] = {c: [] for c in numeric_cols}
                key_order.append(key)
            for col in numeric_cols:
                raw = row.get(col, "")
                ok, num = _parse_float(raw)
                if ok:
                    merged[key][col].append(num)

    output_rows: List[Dict[str, Any]] = []
    for key in key_order:
        row_out: Dict[str, Any] = {col: key[idx] for idx, col in enumerate(key_cols)}
        for col in numeric_cols:
            mean_val, std_val = _compute_mean_std(merged[key].get(col, []))
            row_out[f"{col}_mean"] = None if mean_val is None else round(mean_val, 6)
            row_out[f"{col}_std"] = None if std_val is None else round(std_val, 6)
        output_rows.append(row_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / csv_name
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)

    logger.info("Merged %s -> %s (%d rows)", csv_name, output_path, len(output_rows))


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    analysis_dirs = [Path(p) for p in args.analysis_paths]
    for d in analysis_dirs:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Analysis path not found or not a directory: {d}")

    common_csvs = None
    for d in analysis_dirs:
        csvs = set(list_csv_files(d))
        if common_csvs is None:
            common_csvs = csvs
        else:
            common_csvs = common_csvs & csvs

    if not common_csvs:
        logger.warning("No common CSV files found among analysis directories.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/analysis") / f"merged_{timestamp}"
    logger.info("Merging %d CSV files into %s", len(common_csvs), output_dir)

    for csv_name in sorted(common_csvs):
        merge_csvs(csv_name, analysis_dirs, output_dir)

    print(f"Merged results saved to {output_dir}")

# use case: python merge_results.py results/analysis/analysis_* --output-dir results/analysis/merged


if __name__ == "__main__":
    main()

