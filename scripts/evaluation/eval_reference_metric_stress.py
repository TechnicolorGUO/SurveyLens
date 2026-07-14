"""Controlled stress tests for reference-alignment metrics.

This script reuses the reference embeddings already persisted for
``eval_metric_meta.py``.  It does not call an embedding API.  Two controlled
perturbations are evaluated:

1. duplicate the generated reference with the highest human-reference
   similarity; and
2. append low-similarity references drawn from other survey topics.

The output also contains an accuracy--coverage analysis over the existing
expert pairwise judgments.  Stress-test results are diagnostic metric-property
checks, not substitutes for human meta-evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from eval_metric_meta import ChromaBaselines, MetaEvalConfig, _merge_quantitative_results


LOGGER = logging.getLogger("reference_metric_stress")
STRESS_METRICS = (
    "plain_maxsim",
    "embedding_f1",
    "same_backbone_paper_tau_maxsim",
    "same_backbone_ra_align_f1",
)


@dataclass
class StressConfig:
    meta_config_path: str
    meta_result_path: str
    output_path: str = "results/meta_evaluation/reference_metric_stress.json"
    categories: Optional[List[str]] = None
    systems: Optional[List[str]] = None
    duplication_factors: Optional[List[int]] = None
    distractor_fractions: Optional[List[float]] = None
    distractor_source_surveys: int = 8
    distractor_pool_multiplier: int = 4
    max_surveys: Optional[int] = None
    random_seed: int = 42
    bootstrap_samples: int = 10000
    confidence_coverages: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.duplication_factors is None:
            self.duplication_factors = [2, 4, 8]
        if self.distractor_fractions is None:
            self.distractor_fractions = [0.1, 0.25, 0.5]
        if self.confidence_coverages is None:
            self.confidence_coverages = [0.2, 0.4, 0.6, 0.8, 1.0]
        if any(value < 2 for value in self.duplication_factors):
            raise ValueError("duplication_factors must all be >= 2")
        if any(not 0.0 < value <= 1.0 for value in self.distractor_fractions):
            raise ValueError("distractor_fractions must be in (0, 1]")
        if any(not 0.0 < value <= 1.0 for value in self.confidence_coverages):
            raise ValueError("confidence_coverages must be in (0, 1]")

    @classmethod
    def from_json(cls, path: str) -> "StressConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass(frozen=True)
class SurveyTarget:
    system: str
    category: str
    topic: str
    generated_file: str
    human_file: str


def _pair_credit(pair: Dict[str, Any], metric: str) -> Optional[float]:
    if pair.get("human_choice") == "tie":
        return None
    values = (pair.get("scores") or {}).get(metric) or {}
    left, right = values.get("a"), values.get("b")
    if left is None or right is None:
        return None
    if abs(float(left) - float(right)) <= 1e-12:
        return 0.5
    prediction = "a" if float(left) > float(right) else "b"
    return 1.0 if prediction == pair.get("human_choice") else 0.0


def _metric_margin(pair: Dict[str, Any], metric: str) -> Optional[float]:
    values = (pair.get("scores") or {}).get(metric) or {}
    left, right = values.get("a"), values.get("b")
    if left is None or right is None or pair.get("human_choice") == "tie":
        return None
    return abs(float(left) - float(right))


def _accuracy_at_coverage(
    pairs: Sequence[Dict[str, Any]], metric: str, coverage: float
) -> Dict[str, Any]:
    usable = [
        pair
        for pair in pairs
        if _pair_credit(pair, metric) is not None
        and _metric_margin(pair, metric) is not None
    ]
    ranked = sorted(
        usable,
        key=lambda pair: float(_metric_margin(pair, metric) or 0.0),
        reverse=True,
    )
    count = max(1, int(round(len(ranked) * coverage))) if ranked else 0
    selected = ranked[:count]
    credits = [float(_pair_credit(pair, metric)) for pair in selected]
    return {
        "requested_coverage": coverage,
        "n": len(selected),
        "effective_coverage": len(selected) / len(ranked) if ranked else 0.0,
        "accuracy_with_metric_ties_as_half": (
            statistics.fmean(credits) if credits else None
        ),
        "minimum_selected_margin": (
            min(float(_metric_margin(pair, metric) or 0.0) for pair in selected)
            if selected
            else None
        ),
    }


def _bootstrap_accuracy_ci(
    pairs: Sequence[Dict[str, Any]],
    metric: str,
    coverage: float,
    samples: int,
    seed: int,
) -> List[Optional[float]]:
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if _pair_credit(pair, metric) is not None:
            clusters[str(pair.get("annotator_id") or "unknown")].append(pair)
    keys = sorted(clusters)
    if not keys or samples <= 0:
        return [None, None]
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(samples):
        sampled: List[Dict[str, Any]] = []
        for _ in keys:
            sampled.extend(clusters[rng.choice(keys)])
        value = _accuracy_at_coverage(sampled, metric, coverage).get(
            "accuracy_with_metric_ties_as_half"
        )
        if value is not None:
            estimates.append(float(value))
    if not estimates:
        return [None, None]
    estimates.sort()
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return [estimates[low_index], estimates[high_index]]


def accuracy_coverage_report(
    meta_result: Dict[str, Any], config: StressConfig
) -> Dict[str, Any]:
    pairs = list(meta_result.get("pairs") or [])
    # Use a common subset so every row in the curve compares identical expert
    # judgments.  This avoids the unequal-n issue in the aggregate JSON report.
    common_pairs = [
        pair
        for pair in pairs
        if pair.get("human_choice") != "tie"
        and all(_pair_credit(pair, metric) is not None for metric in STRESS_METRICS)
    ]
    metrics: Dict[str, Any] = {}
    for metric_index, metric in enumerate(STRESS_METRICS):
        points = []
        for coverage_index, coverage in enumerate(config.confidence_coverages or []):
            point = _accuracy_at_coverage(common_pairs, metric, coverage)
            point["ci95"] = _bootstrap_accuracy_ci(
                common_pairs,
                metric,
                coverage,
                config.bootstrap_samples,
                config.random_seed + 1000 * metric_index + coverage_index,
            )
            points.append(point)
        metrics[metric] = points
    return {"common_pair_count": len(common_pairs), "metrics": metrics}


def _entry_lookup(quantitative: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for system, categories in quantitative.items():
        for category, data in categories.items():
            for entry in data.get("files", []):
                file_path = str(entry.get("file") or "")
                if file_path:
                    lookup[(system, category, file_path)] = entry
    return lookup


def _targets_from_meta(
    meta_result: Dict[str, Any],
    quantitative: Dict[str, Any],
    config: StressConfig,
) -> List[SurveyTarget]:
    lookup = _entry_lookup(quantitative)
    targets: Dict[Tuple[str, str, str], SurveyTarget] = {}
    allowed_categories = set(config.categories or [])
    allowed_systems = set(config.systems or [])
    for pair in meta_result.get("pairs", []):
        category = str(pair.get("category") or "")
        if allowed_categories and category not in allowed_categories:
            continue
        topic = str(pair.get("topic") or "")
        for side in ("a", "b"):
            system = str(pair.get(f"system_{side}") or "")
            generated_file = str(pair.get(f"file_{side}") or "")
            if allowed_systems and system not in allowed_systems:
                continue
            entry = lookup.get((system, category, generated_file))
            human_file = str(((entry or {}).get("alignment") or {}).get("human_file") or "")
            if generated_file and human_file:
                key = (system, category, generated_file)
                targets[key] = SurveyTarget(
                    system=system,
                    category=category,
                    topic=topic,
                    generated_file=generated_file,
                    human_file=human_file,
                )
    ordered = sorted(
        targets.values(),
        key=lambda item: (item.category, item.system, item.topic, item.generated_file),
    )
    if config.max_surveys is not None:
        ordered = ordered[: max(0, config.max_surveys)]
    return ordered


def _embeddings_for_target(
    chroma: ChromaBaselines, target: SurveyTarget
) -> Tuple[List[List[float]], List[List[float]]]:
    generated = chroma._embeddings(
        chroma._collection_name(target.category, "reference", target.system),
        target.generated_file,
        "reference",
    )
    human = chroma._embeddings(
        chroma._collection_name(target.category, "reference", "Human"),
        target.human_file,
        "reference",
    )
    return generated, human


def _best_generated_index(
    chroma: ChromaBaselines,
    generated: Sequence[Sequence[float]],
    human: Sequence[Sequence[float]],
) -> int:
    matrix = chroma._cosine_matrix(generated, human)
    maxima = [max(row) if row else 0.0 for row in matrix]
    return max(range(len(maxima)), key=maxima.__getitem__)


def _condition_record(
    target: SurveyTarget,
    attack: str,
    severity: float,
    base_scores: Dict[str, float],
    attacked_scores: Dict[str, float],
    original_count: int,
    added_count: int,
) -> Dict[str, Any]:
    return {
        "system": target.system,
        "category": target.category,
        "topic": target.topic,
        "generated_file": target.generated_file,
        "human_file": target.human_file,
        "attack": attack,
        "severity": severity,
        "original_reference_count": original_count,
        "added_reference_count": added_count,
        "base_scores": {metric: base_scores[metric] for metric in STRESS_METRICS},
        "attacked_scores": {
            metric: attacked_scores[metric] for metric in STRESS_METRICS
        },
        "score_deltas": {
            metric: attacked_scores[metric] - base_scores[metric]
            for metric in STRESS_METRICS
        },
    }


def _distractor_vectors(
    chroma: ChromaBaselines,
    target: SurveyTarget,
    human: Sequence[Sequence[float]],
    other_targets: Sequence[SurveyTarget],
    needed: int,
    config: StressConfig,
    rng: random.Random,
) -> List[List[float]]:
    candidates = [
        other
        for other in other_targets
        if other.topic != target.topic and other.generated_file != target.generated_file
    ]
    rng.shuffle(candidates)
    vectors: List[List[float]] = []
    desired_pool = max(needed, needed * config.distractor_pool_multiplier)
    for source in candidates[: config.distractor_source_surveys]:
        source_generated, _ = _embeddings_for_target(chroma, source)
        vectors.extend(source_generated)
        if len(vectors) >= desired_pool:
            break
    if not vectors:
        return []
    # Select the lowest-similarity cross-topic entries.  This is deliberately a
    # controlled contamination test and must not be described as naturally
    # occurring human-judged irrelevance.
    matrix = chroma._cosine_matrix(vectors, human)
    ranked = sorted(
        range(len(vectors)),
        key=lambda index: max(matrix[index]) if matrix[index] else 0.0,
    )
    return [vectors[index] for index in ranked[:needed]]


def run_stress_tests(
    targets: Sequence[SurveyTarget],
    chroma: ChromaBaselines,
    config: StressConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    diagnostics: Dict[str, int] = defaultdict(int)
    for target_index, target in enumerate(
        tqdm(targets, desc="Reference metric stress tests", unit="survey")
    ):
        try:
            generated, human = _embeddings_for_target(chroma, target)
            if not generated or not human:
                diagnostics["missing_cached_embeddings"] += 1
                continue
            base_scores = chroma._scores_from_embeddings(generated, human, "reference")
            best_index = _best_generated_index(chroma, generated, human)
            for factor in config.duplication_factors or []:
                added = [generated[best_index]] * (factor - 1)
                attacked = list(generated) + added
                attacked_scores = chroma._scores_from_embeddings(
                    attacked, human, "reference"
                )
                records.append(
                    _condition_record(
                        target,
                        "duplicate_best_aligned_reference",
                        float(factor),
                        base_scores,
                        attacked_scores,
                        len(generated),
                        len(added),
                    )
                )

            maximum_needed = max(
                (int(math.ceil(len(generated) * fraction))
                 for fraction in config.distractor_fractions or []),
                default=0,
            )
            rng = random.Random(config.random_seed + target_index)
            distractors = _distractor_vectors(
                chroma,
                target,
                human,
                targets,
                maximum_needed,
                config,
                rng,
            )
            if maximum_needed and len(distractors) < maximum_needed:
                diagnostics["insufficient_distractor_pool"] += 1
            for fraction in config.distractor_fractions or []:
                count = int(math.ceil(len(generated) * fraction))
                added = distractors[:count]
                if len(added) < count:
                    continue
                attacked_scores = chroma._scores_from_embeddings(
                    list(generated) + added, human, "reference"
                )
                records.append(
                    _condition_record(
                        target,
                        "low_similarity_cross_topic_injection",
                        float(fraction),
                        base_scores,
                        attacked_scores,
                        len(generated),
                        len(added),
                    )
                )
            diagnostics["evaluated_surveys"] += 1
        except Exception as exc:
            diagnostics["stress_error"] += 1
            LOGGER.warning("Stress test failed for %s: %s", target.generated_file, exc)
    diagnostics["target_surveys"] = len(targets)
    return records, dict(diagnostics)


def _summarize_values(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean_delta": None,
            "median_delta": None,
            "decreased_fraction": None,
            "unchanged_fraction": None,
            "increased_fraction": None,
        }
    epsilon = 1e-12
    return {
        "n": len(values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "decreased_fraction": sum(value < -epsilon for value in values) / len(values),
        "unchanged_fraction": sum(abs(value) <= epsilon for value in values) / len(values),
        "increased_fraction": sum(value > epsilon for value in values) / len(values),
    }


def summarize_stress(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)
    per_target: Dict[Tuple[str, str], Dict[float, Dict[str, float]]] = defaultdict(dict)
    for record in records:
        attack = str(record["attack"])
        severity = float(record["severity"])
        target_key = str(record["generated_file"])
        for metric in STRESS_METRICS:
            delta = float(record["score_deltas"][metric])
            grouped[(attack, severity, metric)].append(delta)
            per_target[(attack, target_key)].setdefault(severity, {})[metric] = float(
                record["attacked_scores"][metric]
            )

    summary: Dict[str, Any] = {}
    for (attack, severity, metric), values in sorted(grouped.items()):
        summary.setdefault(attack, {}).setdefault(str(severity), {})[metric] = (
            _summarize_values(values)
        )

    monotonic: Dict[str, Dict[str, Any]] = {}
    for attack in sorted({key[0] for key in per_target}):
        for metric in STRESS_METRICS:
            checks: List[bool] = []
            for (candidate_attack, _), by_severity in per_target.items():
                if candidate_attack != attack:
                    continue
                ordered = [
                    values[metric]
                    for _, values in sorted(by_severity.items())
                    if metric in values
                ]
                if len(ordered) >= 2:
                    checks.append(
                        all(
                            ordered[index + 1] <= ordered[index] + 1e-12
                            for index in range(len(ordered) - 1)
                        )
                    )
            monotonic.setdefault(attack, {})[metric] = {
                "n": len(checks),
                "nonincreasing_fraction": (
                    sum(checks) / len(checks) if checks else None
                ),
            }
    return {"by_condition": summary, "monotonicity": monotonic}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test reference alignment metrics using cached embeddings"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override output_path in the config")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = StressConfig.from_json(args.config)
    if args.output:
        config.output_path = args.output
    meta_config = MetaEvalConfig.from_json(config.meta_config_path)
    if not meta_config.chroma_db_dir:
        raise ValueError("chroma_db_dir is missing from the meta-evaluation config")
    # Stress tests are intentionally cache-only so this script can never cause
    # unexpected API cost or mix embeddings from a different model.
    meta_config.embedding_source = "chroma_only"
    meta_config.enable_embedding_baselines = True

    with open(config.meta_result_path, "r", encoding="utf-8") as handle:
        meta_result = json.load(handle)
    quantitative = _merge_quantitative_results(meta_config.quantitative_result_files)
    targets = _targets_from_meta(meta_result, quantitative, config)
    chroma = ChromaBaselines(meta_config.chroma_db_dir, meta_config)

    records, diagnostics = run_stress_tests(targets, chroma, config)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "protocol_notes": {
            "embedding_access": "cache_only; no API calls",
            "duplicate_attack": (
                "Duplicates the generated reference with the highest cosine "
                "similarity to any human reference."
            ),
            "distractor_attack": (
                "Adds the lowest-similarity cached reference vectors sampled "
                "from different survey topics. These are controlled synthetic "
                "distractors, not human-adjudicated irrelevant papers."
            ),
            "interpretation": (
                "Stress tests evaluate monotonicity and resistance to metric "
                "gaming; they do not measure expert correlation."
            ),
        },
        "accuracy_coverage": accuracy_coverage_report(meta_result, config),
        "stress_diagnostics": diagnostics,
        "stress_summary": summarize_stress(records),
        "stress_records": records,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Saved reference metric stress report to %s", output)
    print(
        json.dumps(
            {
                "output": str(output),
                "accuracy_coverage": report["accuracy_coverage"],
                "stress_diagnostics": diagnostics,
                "stress_summary": report["stress_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
