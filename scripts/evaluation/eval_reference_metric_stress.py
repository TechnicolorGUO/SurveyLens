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
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from eval_metric_meta import ChromaBaselines, MetaEvalConfig, _merge_quantitative_results
from compute_reference_rouge_bertscore import (
    Config as TextMetricConfig,
    _build_scorer,
    _score_chunk_sets,
    entry_aware_rouge_n_f1,
)
from eval_multicomponent_alignment_proxy import extract_entries


LOGGER = logging.getLogger("reference_metric_stress")
EMBEDDING_STRESS_METRICS = (
    "plain_maxsim",
    "embedding_f1",
    "same_backbone_paper_tau_maxsim",
    "same_backbone_ra_align_f1",
)
TEXT_STRESS_METRICS = (
    "entry_aware_rouge2_f1",
    "chunked_bertscore_f1",
)
STRESS_METRICS = EMBEDDING_STRESS_METRICS + TEXT_STRESS_METRICS


@dataclass
class StressConfig:
    meta_config_path: str
    meta_result_path: str
    output_path: str = "results/meta_evaluation/reference_metric_stress.json"
    categories: Optional[List[str]] = None
    systems: Optional[List[str]] = None
    duplication_factors: Optional[List[int]] = None
    collapse_fractions: Optional[List[float]] = None
    distractor_fractions: Optional[List[float]] = None
    distractor_source_surveys: int = 8
    distractor_pool_multiplier: int = 4
    max_surveys: Optional[int] = None
    random_seed: int = 42
    bootstrap_samples: int = 10000
    confidence_coverages: Optional[List[float]] = None
    enable_entry_aware_rouge2: bool = False
    enable_chunked_bertscore: bool = False
    bertscore_model_type: str = "roberta-large"
    bertscore_lang: str = "en"
    bertscore_num_layers: Optional[int] = None
    bertscore_batch_size: int = 16
    bertscore_device: Optional[str] = None
    bertscore_rescale_with_baseline: bool = False
    bertscore_use_fast_tokenizer: bool = False
    chunk_max_words: int = 200

    def __post_init__(self) -> None:
        if self.duplication_factors is None:
            self.duplication_factors = [2, 4, 8]
        if self.collapse_fractions is None:
            self.collapse_fractions = [0.25, 0.5, 1.0]
        if self.distractor_fractions is None:
            self.distractor_fractions = [0.1, 0.25, 0.5]
        if self.confidence_coverages is None:
            self.confidence_coverages = [0.2, 0.4, 0.6, 0.8, 1.0]
        if any(value < 2 for value in self.duplication_factors):
            raise ValueError("duplication_factors must all be >= 2")
        if any(not 0.0 < value <= 1.0 for value in self.collapse_fractions):
            raise ValueError("collapse_fractions must be in (0, 1]")
        if any(not 0.0 < value <= 1.0 for value in self.distractor_fractions):
            raise ValueError("distractor_fractions must be in (0, 1]")
        if any(not 0.0 < value <= 1.0 for value in self.confidence_coverages):
            raise ValueError("confidence_coverages must be in (0, 1]")
        if self.bertscore_batch_size <= 0:
            raise ValueError("bertscore_batch_size must be positive")
        if self.chunk_max_words <= 0:
            raise ValueError("chunk_max_words must be positive")

    @classmethod
    def from_json(cls, path: str) -> "StressConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def text_metric_config(self) -> TextMetricConfig:
        """Expose the shared BERTScore settings without duplicating scoring code."""
        return TextMetricConfig(
            v2_config_file="unused-by-stress-test",
            bertscore_model_type=self.bertscore_model_type,
            bertscore_lang=self.bertscore_lang,
            bertscore_num_layers=self.bertscore_num_layers,
            bertscore_batch_size=self.bertscore_batch_size,
            bertscore_device=self.bertscore_device,
            bertscore_rescale_with_baseline=self.bertscore_rescale_with_baseline,
            bertscore_use_fast_tokenizer=self.bertscore_use_fast_tokenizer,
            chunk_max_words=self.chunk_max_words,
        )


def _configured_stress_metrics(config: StressConfig) -> Tuple[str, ...]:
    metrics = list(EMBEDDING_STRESS_METRICS)
    if config.enable_entry_aware_rouge2:
        metrics.append("entry_aware_rouge2_f1")
    if config.enable_chunked_bertscore:
        metrics.append("chunked_bertscore_f1")
    return tuple(metrics)


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
        and all(
            _pair_credit(pair, metric) is not None
            for metric in EMBEDDING_STRESS_METRICS
        )
    ]
    metrics: Dict[str, Any] = {}
    for metric_index, metric in enumerate(EMBEDDING_STRESS_METRICS):
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


def _many_to_one_collapse(
    chroma: ChromaBaselines,
    generated: Sequence[Sequence[float]],
    generated_entries: Sequence[str],
    human: Sequence[Sequence[float]],
    fraction: float,
) -> Tuple[List[List[float]], List[str], int, int]:
    """Replace weak entries with copies of one best-aligned generated entry.

    The generated-set cardinality is fixed.  At severity 1.0 every entry is a
    copy of the same best-aligned entry.  Replacing the weakest entries first
    creates an adversarial many-to-one collapse without the length increase of
    the append-only duplication attack.
    """
    if len(generated) != len(generated_entries):
        raise ValueError("generated vectors and entries must have equal lengths")
    if not generated:
        return [], [], 0, -1
    matrix = chroma._cosine_matrix(generated, human)
    maxima = [max(row) if row else 0.0 for row in matrix]
    best_index = max(range(len(maxima)), key=maxima.__getitem__)
    replace_count = min(
        len(generated) - 1,
        int(math.ceil(len(generated) * fraction)),
    )
    weakest = sorted(
        (index for index in range(len(generated)) if index != best_index),
        key=lambda index: (maxima[index], index),
    )[:replace_count]
    attacked_vectors = [list(vector) for vector in generated]
    attacked_entries = list(generated_entries)
    for index in weakest:
        attacked_vectors[index] = list(generated[best_index])
        attacked_entries[index] = generated_entries[best_index]
    return attacked_vectors, attacked_entries, len(weakest), best_index


@lru_cache(maxsize=None)
def _reference_entries(path: str) -> Tuple[str, ...]:
    return tuple(extract_entries(path)["reference"])


def _text_scores(
    generated: Sequence[str],
    human: Sequence[str],
    config: StressConfig,
    bert_scorer: Optional[Any],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if config.enable_entry_aware_rouge2:
        scores["entry_aware_rouge2_f1"] = entry_aware_rouge_n_f1(
            generated, human, 2
        )
    if config.enable_chunked_bertscore:
        if bert_scorer is None:
            raise RuntimeError("Chunked BERTScore is enabled but no scorer was loaded")
        value, _ = _score_chunk_sets(
            bert_scorer,
            generated,
            human,
            config.text_metric_config(),
        )
        scores["chunked_bertscore_f1"] = value
    return scores


def _condition_record(
    target: SurveyTarget,
    attack: str,
    severity: float,
    base_scores: Dict[str, float],
    attacked_scores: Dict[str, float],
    original_count: int,
    added_count: int,
    metrics: Sequence[str],
    replaced_count: int = 0,
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
        "replaced_reference_count": replaced_count,
        "base_scores": {metric: base_scores[metric] for metric in metrics},
        "attacked_scores": {
            metric: attacked_scores[metric] for metric in metrics
        },
        "score_deltas": {
            metric: attacked_scores[metric] - base_scores[metric]
            for metric in metrics
        },
    }


def _distractor_items(
    chroma: ChromaBaselines,
    target: SurveyTarget,
    human: Sequence[Sequence[float]],
    other_targets: Sequence[SurveyTarget],
    needed: int,
    config: StressConfig,
    rng: random.Random,
) -> List[Tuple[List[float], str]]:
    candidates = [
        other
        for other in other_targets
        if other.topic != target.topic and other.generated_file != target.generated_file
    ]
    rng.shuffle(candidates)
    items: List[Tuple[List[float], str]] = []
    desired_pool = max(needed, needed * config.distractor_pool_multiplier)
    for source in candidates[: config.distractor_source_surveys]:
        source_generated, _ = _embeddings_for_target(chroma, source)
        source_entries = _reference_entries(source.generated_file)
        if len(source_generated) != len(source_entries):
            continue
        items.extend(
            (list(vector), entry)
            for vector, entry in zip(source_generated, source_entries)
        )
        if len(items) >= desired_pool:
            break
    if not items:
        return []
    # Select the lowest-similarity cross-topic entries.  This is deliberately a
    # controlled contamination test and must not be described as naturally
    # occurring human-judged irrelevance.
    matrix = chroma._cosine_matrix([vector for vector, _ in items], human)
    ranked = sorted(
        range(len(items)),
        key=lambda index: max(matrix[index]) if matrix[index] else 0.0,
    )
    return [items[index] for index in ranked[:needed]]


def run_stress_tests(
    targets: Sequence[SurveyTarget],
    chroma: ChromaBaselines,
    config: StressConfig,
    bert_scorer: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    diagnostics: Dict[str, int] = defaultdict(int)
    metrics = _configured_stress_metrics(config)
    text_metrics_enabled = (
        config.enable_entry_aware_rouge2 or config.enable_chunked_bertscore
    )
    for target_index, target in enumerate(
        tqdm(targets, desc="Reference metric stress tests", unit="survey")
    ):
        try:
            generated, human = _embeddings_for_target(chroma, target)
            if not generated or not human:
                diagnostics["missing_cached_embeddings"] += 1
                continue
            generated_entries = list(_reference_entries(target.generated_file))
            human_entries = list(_reference_entries(target.human_file))
            if text_metrics_enabled and (
                len(generated_entries) != len(generated)
                or len(human_entries) != len(human)
            ):
                diagnostics["text_embedding_count_mismatch"] += 1
                continue
            base_scores = chroma._scores_from_embeddings(generated, human, "reference")
            base_scores.update(
                _text_scores(generated_entries, human_entries, config, bert_scorer)
            )
            best_index = _best_generated_index(chroma, generated, human)
            for factor in config.duplication_factors or []:
                added = [generated[best_index]] * (factor - 1)
                attacked = list(generated) + added
                attacked_scores = chroma._scores_from_embeddings(
                    attacked, human, "reference"
                )
                attacked_entries = generated_entries + [
                    generated_entries[best_index]
                ] * (factor - 1)
                attacked_scores.update(
                    _text_scores(
                        attacked_entries,
                        human_entries,
                        config,
                        bert_scorer,
                    )
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
                        metrics,
                    )
                )

            for fraction in config.collapse_fractions or []:
                collapsed_vectors, collapsed_entries, replaced_count, _ = (
                    _many_to_one_collapse(
                        chroma,
                        generated,
                        generated_entries,
                        human,
                        float(fraction),
                    )
                )
                collapsed_scores = chroma._scores_from_embeddings(
                    collapsed_vectors, human, "reference"
                )
                collapsed_scores.update(
                    _text_scores(
                        collapsed_entries,
                        human_entries,
                        config,
                        bert_scorer,
                    )
                )
                records.append(
                    _condition_record(
                        target,
                        "fixed_length_many_to_one_collapse",
                        float(fraction),
                        base_scores,
                        collapsed_scores,
                        len(generated),
                        0,
                        metrics,
                        replaced_count=replaced_count,
                    )
                )

            maximum_needed = max(
                (int(math.ceil(len(generated) * fraction))
                 for fraction in config.distractor_fractions or []),
                default=0,
            )
            rng = random.Random(config.random_seed + target_index)
            distractors = _distractor_items(
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
                    list(generated) + [vector for vector, _ in added],
                    human,
                    "reference",
                )
                attacked_scores.update(
                    _text_scores(
                        generated_entries + [entry for _, entry in added],
                        human_entries,
                        config,
                        bert_scorer,
                    )
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
                        metrics,
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


def summarize_stress(
    records: Sequence[Dict[str, Any]], metrics: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    selected_metrics = tuple(metrics or STRESS_METRICS)
    grouped: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)
    per_target: Dict[Tuple[str, str], Dict[float, Dict[str, float]]] = defaultdict(dict)
    for record in records:
        attack = str(record["attack"])
        severity = float(record["severity"])
        target_key = str(record["generated_file"])
        for metric in selected_metrics:
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
        for metric in selected_metrics:
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
    bert_scorer = (
        _build_scorer(config.text_metric_config())
        if config.enable_chunked_bertscore
        else None
    )
    stress_metrics = _configured_stress_metrics(config)

    records, diagnostics = run_stress_tests(
        targets, chroma, config, bert_scorer=bert_scorer
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "protocol_notes": {
            "embedding_access": "cache_only; no API calls",
            "duplicate_attack": (
                "Duplicates the generated reference with the highest cosine "
                "similarity to any human reference."
            ),
            "many_to_one_collapse_attack": (
                "Keeps generated-set cardinality fixed while replacing the "
                "weakest 25%, 50%, or all replaceable entries with copies of "
                "one best-aligned generated entry. At severity 1.0 every "
                "generated entry is identical."
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
            "text_embedding_alignment": (
                "Reference texts are paired with cached embeddings by their "
                "original processed-file order; targets with count mismatches "
                "are excluded and reported in stress_diagnostics."
            ),
            "bertscore_aggregation": (
                "Long reference lists are split at entry boundaries; "
                "candidate-side precision and human-side recall are max-matched "
                "across chunks and combined as F1."
                if config.enable_chunked_bertscore
                else None
            ),
        },
        "stress_metrics": list(stress_metrics),
        "bertscore_hash": getattr(bert_scorer, "hash", None),
        "accuracy_coverage": accuracy_coverage_report(meta_result, config),
        "stress_diagnostics": diagnostics,
        "stress_summary": summarize_stress(records, stress_metrics),
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
