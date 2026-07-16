"""Add entry-aware ROUGE-2 and chunked BERTScore to frozen V2 pairs.

ROUGE-2 is computed within entry boundaries, so no artificial bigram is created
between adjacent headings, sections, or bibliography entries.  Long components
can exceed the sequence limit of standard document-level BERTScore, so
BERTScore is computed over entry-preserving chunks.  Candidate-side precision
and human-reference-side recall are max-matched across chunks and combined as
F1.

The script incrementally merges metrics into the existing V2 ``pairs.jsonl``.
It never changes the frozen pair selection or the LLM/human labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from eval_multicomponent_alignment_proxy import (
    COMPONENTS,
    Config as V2Config,
    Pair,
    _file_sha256,
    _write_jsonl,
    extract_entries,
    load_pairs,
)


LOGGER = logging.getLogger("reference_rouge_bertscore")
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class Config:
    v2_config_file: str
    components: Optional[List[str]] = None
    bertscore_model_type: str = "roberta-large"
    bertscore_lang: str = "en"
    bertscore_num_layers: Optional[int] = None
    bertscore_batch_size: int = 16
    bertscore_device: Optional[str] = None
    bertscore_rescale_with_baseline: bool = False
    bertscore_use_fast_tokenizer: bool = False
    chunk_max_words: int = 200
    bertscore_full_matrix_max_pairs: int = 512
    bertscore_chunk_top_k: int = 5
    checkpoint_every_sides: int = 5
    force_recompute: bool = False
    rouge_metric_name: str = "entry_aware_rouge2_f1"
    bertscore_metric_name: str = "chunked_bertscore_f1"

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = ["reference"]
        invalid = set(self.components) - set(COMPONENTS)
        if invalid:
            raise ValueError(f"Unknown components: {sorted(invalid)}")
        if self.bertscore_batch_size <= 0:
            raise ValueError("bertscore_batch_size must be positive")
        if self.chunk_max_words <= 0:
            raise ValueError("chunk_max_words must be positive")
        if self.bertscore_full_matrix_max_pairs <= 0:
            raise ValueError("bertscore_full_matrix_max_pairs must be positive")
        if self.bertscore_chunk_top_k <= 0:
            raise ValueError("bertscore_chunk_top_k must be positive")
        if self.checkpoint_every_sides <= 0:
            raise ValueError("checkpoint_every_sides must be positive")
        if not self.rouge_metric_name or not self.bertscore_metric_name:
            raise ValueError("metric names must be non-empty")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text.casefold())


def _entry_ngrams(entries: Sequence[str], n: int) -> Counter:
    """Count n-grams inside entries without crossing entry boundaries."""
    counts: Counter = Counter()
    for entry in entries:
        tokens = _tokens(entry)
        counts.update(
            tuple(tokens[index : index + n])
            for index in range(len(tokens) - n + 1)
        )
    return counts


def entry_aware_rouge_n_f1(
    candidate: Sequence[str], reference: Sequence[str], n: int
) -> float:
    left = _entry_ngrams(candidate, n)
    right = _entry_ngrams(reference, n)
    left_count, right_count = sum(left.values()), sum(right.values())
    if not left_count or not right_count:
        return 0.0
    overlap = sum((left & right).values())
    precision, recall = overlap / left_count, overlap / right_count
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _clean_entry(value: str) -> str:
    return " ".join(str(value).split())


def chunk_entries(
    entries: Sequence[str], max_words: int
) -> List[Tuple[str, int]]:
    """Create non-empty, entry-preserving chunks with approximate word weights.

    A single unusually long entry is split so the BERTScore model never receives
    an unbounded item.  Normal reference titles remain intact.
    """
    units: List[Tuple[str, int]] = []
    for raw in entries:
        entry = _clean_entry(raw)
        words = entry.split()
        if not words:
            continue
        for start in range(0, len(words), max_words):
            part = " ".join(words[start : start + max_words])
            units.append((part, len(words[start : start + max_words])))

    chunks: List[Tuple[str, int]] = []
    current: List[str] = []
    current_words = 0
    for text, weight in units:
        if current and current_words + weight > max_words:
            chunks.append(("\n".join(current), current_words))
            current, current_words = [], 0
        current.append(text)
        current_words += weight
    if current:
        chunks.append(("\n".join(current), current_words))
    return chunks


def aggregate_chunk_bertscore(
    precision: Sequence[float],
    recall: Sequence[float],
    candidate_weights: Sequence[int],
    reference_weights: Sequence[int],
) -> float:
    """Aggregate a row-major candidate-chunk x reference-chunk score matrix."""
    candidate_count, reference_count = len(candidate_weights), len(reference_weights)
    expected = candidate_count * reference_count
    if len(precision) != expected or len(recall) != expected:
        raise ValueError("BERTScore matrix shape does not match chunk counts")
    if not candidate_count or not reference_count:
        return 0.0

    candidate_best = [
        max(precision[i * reference_count : (i + 1) * reference_count])
        for i in range(candidate_count)
    ]
    reference_best = [
        max(recall[i * reference_count + j] for i in range(candidate_count))
        for j in range(reference_count)
    ]
    p_weight = sum(candidate_weights)
    r_weight = sum(reference_weights)
    p = sum(score * weight for score, weight in zip(candidate_best, candidate_weights)) / p_weight
    r = sum(score * weight for score, weight in zip(reference_best, reference_weights)) / r_weight
    return 2.0 * p * r / (p + r) if p + r else 0.0


def _chunk_pair_indices(
    candidate_chunks: Sequence[Tuple[str, int]],
    reference_chunks: Sequence[Tuple[str, int]],
    full_matrix_max_pairs: int,
    top_k: int,
) -> Tuple[List[Tuple[int, int]], str]:
    candidate_count, reference_count = len(candidate_chunks), len(reference_chunks)
    if candidate_count * reference_count <= full_matrix_max_pairs:
        return (
            [(i, j) for i in range(candidate_count) for j in range(reference_count)],
            "full_matrix",
        )

    candidate_tokens = [set(_tokens(text)) for text, _ in candidate_chunks]
    reference_tokens = [set(_tokens(text)) for text, _ in reference_chunks]

    def similarity(i: int, j: int) -> float:
        union = candidate_tokens[i] | reference_tokens[j]
        return (
            len(candidate_tokens[i] & reference_tokens[j]) / len(union)
            if union
            else 0.0
        )

    selected = set()
    for i in range(candidate_count):
        ranked = sorted(
            range(reference_count),
            key=lambda j: (-similarity(i, j), j),
        )[: min(top_k, reference_count)]
        selected.update((i, j) for j in ranked)
    for j in range(reference_count):
        ranked = sorted(
            range(candidate_count),
            key=lambda i: (-similarity(i, j), i),
        )[: min(top_k, candidate_count)]
        selected.update((i, j) for i in ranked)
    return sorted(selected), "bidirectional_lexical_top_k"


def aggregate_sparse_chunk_bertscore(
    pair_indices: Sequence[Tuple[int, int]],
    precision: Sequence[float],
    recall: Sequence[float],
    candidate_weights: Sequence[int],
    reference_weights: Sequence[int],
) -> float:
    if not candidate_weights or not reference_weights:
        return 0.0
    if len(pair_indices) != len(precision) or len(pair_indices) != len(recall):
        raise ValueError("Sparse BERTScore values do not match selected chunk pairs")
    candidate_best = [-math.inf] * len(candidate_weights)
    reference_best = [-math.inf] * len(reference_weights)
    for (i, j), p, r in zip(pair_indices, precision, recall):
        candidate_best[i] = max(candidate_best[i], p)
        reference_best[j] = max(reference_best[j], r)
    if any(not math.isfinite(value) for value in candidate_best + reference_best):
        raise ValueError("Every candidate and reference chunk must be selected")
    p = sum(value * weight for value, weight in zip(candidate_best, candidate_weights)) / sum(candidate_weights)
    r = sum(value * weight for value, weight in zip(reference_best, reference_weights)) / sum(reference_weights)
    return 2.0 * p * r / (p + r) if p + r else 0.0


def _verify_hashes(pair: Pair) -> None:
    expected = pair.file_hashes
    actual = {
        "reference": _file_sha256(pair.human_file),
        "a": _file_sha256(pair.file_a),
        "b": _file_sha256(pair.file_b),
    }
    if not expected:
        raise RuntimeError(f"Pair {pair.pair_id} has no frozen file hashes")
    if expected != actual:
        raise RuntimeError(
            f"Frozen input changed for pair {pair.pair_id}: "
            f"expected={expected}, actual={actual}"
        )


def _build_scorer(config: Config) -> Any:
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "bert-score is required; install it with "
            "`python3 -m pip install bert-score`"
        ) from exc

    kwargs: Dict[str, Any] = {
        "model_type": config.bertscore_model_type,
        "lang": config.bertscore_lang,
        "batch_size": config.bertscore_batch_size,
        "rescale_with_baseline": config.bertscore_rescale_with_baseline,
        "use_fast_tokenizer": config.bertscore_use_fast_tokenizer,
    }
    if config.bertscore_num_layers is not None:
        kwargs["num_layers"] = config.bertscore_num_layers
    if config.bertscore_device:
        kwargs["device"] = config.bertscore_device
    return BERTScorer(**kwargs)


def _score_chunk_sets(
    scorer: Any,
    candidate_entries: Sequence[str],
    reference_entries: Sequence[str],
    config: Config,
) -> Tuple[float, Dict[str, int]]:
    candidate_chunks = chunk_entries(candidate_entries, config.chunk_max_words)
    reference_chunks = chunk_entries(reference_entries, config.chunk_max_words)
    if not candidate_chunks or not reference_chunks:
        return 0.0, {
            "candidate_chunks": len(candidate_chunks),
            "reference_chunks": len(reference_chunks),
        }

    pair_indices, pairing_strategy = _chunk_pair_indices(
        candidate_chunks,
        reference_chunks,
        config.bertscore_full_matrix_max_pairs,
        config.bertscore_chunk_top_k,
    )
    candidates = [candidate_chunks[i][0] for i, _ in pair_indices]
    references = [reference_chunks[j][0] for _, j in pair_indices]

    precision, recall, _ = scorer.score(
        candidates,
        references,
        batch_size=config.bertscore_batch_size,
        verbose=False,
    )
    p_values = [float(value) for value in precision.detach().cpu().tolist()]
    r_values = [float(value) for value in recall.detach().cpu().tolist()]
    score = aggregate_sparse_chunk_bertscore(
        pair_indices,
        p_values,
        r_values,
        [weight for _, weight in candidate_chunks],
        [weight for _, weight in reference_chunks],
    )
    if not math.isfinite(score):
        raise RuntimeError("BERTScore returned a non-finite score")
    return score, {
        "candidate_chunks": len(candidate_chunks),
        "reference_chunks": len(reference_chunks),
        "full_chunk_pairs": len(candidate_chunks) * len(reference_chunks),
        "evaluated_chunk_pairs": len(pair_indices),
        "pairing_strategy": pairing_strategy,
    }


def compute(config: Config, limit_pairs: Optional[int] = None) -> Dict[str, Any]:
    v2_config = V2Config.from_json(config.v2_config_file)
    pairs = load_pairs(v2_config)
    selected = pairs[:limit_pairs] if limit_pairs else pairs
    output_dir = Path(v2_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_cache: Dict[str, Dict[str, List[str]]] = {}

    def entries(path: str, component: str) -> List[str]:
        if path not in entry_cache:
            entry_cache[path] = extract_entries(path)
        return entry_cache[path][component]

    for pair in tqdm(selected, desc="Entry-aware ROUGE-2", unit="pair", dynamic_ncols=True):
        _verify_hashes(pair)
        for component in config.components or []:
            human = entries(pair.human_file, component)
            metric = pair.metrics[component].setdefault(config.rouge_metric_name, {})
            metric["a"] = entry_aware_rouge_n_f1(
                entries(pair.file_a, component), human, 2
            )
            metric["b"] = entry_aware_rouge_n_f1(
                entries(pair.file_b, component), human, 2
            )
    _write_jsonl(output_dir / "pairs.jsonl", (asdict(pair) for pair in pairs))

    scorer = _build_scorer(config)
    errors: List[Dict[str, str]] = []
    chunk_audit: List[Dict[str, Any]] = []
    completed_sides = 0
    total_sides = len(selected) * len(config.components or []) * 2
    progress = tqdm(total=total_sides, desc="Chunked BERTScore", unit="side", dynamic_ncols=True)
    since_checkpoint = 0
    for pair in selected:
        for component in config.components or []:
            human = entries(pair.human_file, component)
            metric = pair.metrics[component].setdefault(config.bertscore_metric_name, {})
            for side, path in (("a", pair.file_a), ("b", pair.file_b)):
                if not config.force_recompute and isinstance(metric.get(side), (int, float)):
                    completed_sides += 1
                    progress.update(1)
                    continue
                try:
                    score, audit = _score_chunk_sets(
                        scorer, entries(path, component), human, config
                    )
                    metric[side] = score
                    completed_sides += 1
                    chunk_audit.append({
                        "pair_id": pair.pair_id,
                        "component": component,
                        "side": side,
                        **audit,
                    })
                except Exception as exc:
                    errors.append({
                        "pair_id": pair.pair_id,
                        "component": component,
                        "side": side,
                        "error": str(exc),
                    })
                    LOGGER.exception(
                        "Failed BERTScore for %s/%s/%s",
                        pair.pair_id,
                        component,
                        side,
                    )
                progress.update(1)
                since_checkpoint += 1
                progress.set_postfix(completed=completed_sides, errors=len(errors))
                if since_checkpoint >= config.checkpoint_every_sides:
                    _write_jsonl(output_dir / "pairs.jsonl", (asdict(item) for item in pairs))
                    since_checkpoint = 0
    progress.close()
    _write_jsonl(output_dir / "pairs.jsonl", (asdict(pair) for pair in pairs))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_pairs": len(selected),
        "components": config.components,
        "total_sides": total_sides,
        "completed_sides": completed_sides,
        "errors": errors,
        "rouge_metric_name": config.rouge_metric_name,
        "bertscore_metric_name": config.bertscore_metric_name,
        "bertscore_model_type": config.bertscore_model_type,
        "bertscore_num_layers": config.bertscore_num_layers,
        "bertscore_hash": getattr(scorer, "hash", None),
        "bertscore_rescale_with_baseline": config.bertscore_rescale_with_baseline,
        "chunk_max_words": config.chunk_max_words,
        "chunk_audit": chunk_audit,
        "config": asdict(config),
    }
    status_path = output_dir / "multicomponent_rouge_bertscore_status.json"
    status_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    report = compute(Config.from_json(args.config), args.limit_pairs)
    print(
        json.dumps(
            {
                "selected_pairs": report["selected_pairs"],
                "completed_sides": report["completed_sides"],
                "error_count": len(report["errors"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
