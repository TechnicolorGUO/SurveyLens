"""Compute same-backbone embedding baselines for frozen V2 proxy pairs.

The script reuses ChromaBaselines from eval_metric_meta.py, persists missing
entry embeddings, verifies frozen input hashes, and incrementally merges scores
into the V2 pairs.jsonl so interrupted server runs can safely resume.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from eval_metric_meta import ChromaBaselines, MetaEvalConfig
from eval_multicomponent_alignment_proxy import (
    COMPONENTS,
    Config as V2Config,
    Pair,
    _file_sha256,
    _load_env,
    _write_jsonl,
    load_pairs,
)


LOGGER = logging.getLogger("multicomponent_embedding_metrics")


@dataclass
class EmbeddingConfig:
    v2_config_file: str
    chroma_db_dir: str = "chromadb_meta_selective"
    components: Optional[List[str]] = None
    embedding_model: str = "text-embedding-v3"
    embedding_api_base: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key_env: str = "OPENROUTER_API_KEY"
    env_file: str = ".env"
    embedding_batch_size: int = 8
    embedding_max_batch_chars: int = 200000
    embedding_max_input_chars: int = 8000
    embedding_request_timeout: float = 120.0
    embedding_max_retries: int = 5
    persist_missing_embeddings: bool = True
    outline_threshold: float = 0.7
    content_threshold: float = 0.7
    reference_threshold: float = 0.8
    outline_lambda: float = 1.0
    content_lambda: float = 1.0
    reference_lambda: float = 1.0

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = list(COMPONENTS)
        invalid = set(self.components) - set(COMPONENTS)
        if invalid:
            raise ValueError(f"Unknown components: {sorted(invalid)}")

    @classmethod
    def from_json(cls, path: str) -> "EmbeddingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def meta_config(self) -> MetaEvalConfig:
        return MetaEvalConfig(
            quantitative_result_files=[],
            components=list(self.components or COMPONENTS),
            chroma_db_dir=self.chroma_db_dir,
            enable_embedding_baselines=True,
            embedding_baselines=[
                "plain_maxsim",
                "embedding_f1",
                "same_backbone_threshold_gated_maxsim",
                "same_backbone_paper_tau_maxsim",
                "same_backbone_ra_align_f1",
            ],
            embedding_source="chroma_or_api",
            embedding_model=self.embedding_model,
            embedding_api_base=self.embedding_api_base,
            embedding_api_key_env=self.embedding_api_key_env,
            embedding_batch_size=self.embedding_batch_size,
            embedding_max_batch_chars=self.embedding_max_batch_chars,
            embedding_max_input_chars=self.embedding_max_input_chars,
            embedding_request_timeout=self.embedding_request_timeout,
            embedding_max_retries=self.embedding_max_retries,
            persist_missing_embeddings=self.persist_missing_embeddings,
            outline_threshold=self.outline_threshold,
            content_threshold=self.content_threshold,
            reference_threshold=self.reference_threshold,
            outline_lambda=self.outline_lambda,
            content_lambda=self.content_lambda,
            reference_lambda=self.reference_lambda,
        )


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
            f"Frozen input changed for pair {pair.pair_id}: expected={expected}, actual={actual}"
        )


def compute(
    config: EmbeddingConfig,
    limit_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    _load_env(config.env_file)
    v2_config = V2Config.from_json(config.v2_config_file)
    pairs = load_pairs(v2_config)
    selected = pairs[:limit_pairs] if limit_pairs else pairs
    chroma = ChromaBaselines(config.chroma_db_dir, config.meta_config())
    metric_names = list(config.meta_config().embedding_baselines)
    errors: List[Dict[str, str]] = []
    completed = 0

    progress = tqdm(selected, desc="Embedding baselines", unit="pair", dynamic_ncols=True)
    for pair in progress:
        _verify_hashes(pair)
        pair_ok = True
        for component in config.components or COMPONENTS:
            entry_a = {
                "file": pair.file_a,
                "alignment": {"human_file": pair.human_file},
            }
            entry_b = {
                "file": pair.file_b,
                "alignment": {"human_file": pair.human_file},
            }
            try:
                scores_a = chroma.scores(entry_a, pair.system_a, pair.category, component)
                scores_b = chroma.scores(entry_b, pair.system_b, pair.category, component)
                for metric in metric_names:
                    a, b = scores_a.get(metric), scores_b.get(metric)
                    if a is None or b is None:
                        raise RuntimeError(f"Missing {metric} score")
                    pair.metrics[component][metric] = {
                        "a": float(a),
                        "b": float(b),
                    }
            except Exception as exc:  # preserve successful components and continue auditably
                pair_ok = False
                errors.append(
                    {
                        "pair_id": pair.pair_id,
                        "component": component,
                        "error": str(exc),
                    }
                )
                LOGGER.exception("Failed %s/%s", pair.pair_id, component)
        if pair_ok:
            completed += 1
        # Chroma already persists embeddings. Persist metric progress too.
        _write_jsonl(
            Path(v2_config.output_dir) / "pairs.jsonl",
            (asdict(item) for item in pairs),
        )
        progress.set_postfix(completed=completed, errors=len(errors))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.embedding_model,
        "chroma_db_dir": config.chroma_db_dir,
        "components": config.components,
        "selected_pairs": len(selected),
        "completed_pairs": completed,
        "errors": errors,
        "config": asdict(config),
    }
    report_path = Path(v2_config.output_dir) / "embedding_metric_status.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
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
    report = compute(EmbeddingConfig.from_json(args.config), args.limit_pairs)
    print(json.dumps({
        "selected_pairs": report["selected_pairs"],
        "completed_pairs": report["completed_pairs"],
        "error_count": len(report["errors"]),
    }, indent=2))


if __name__ == "__main__":
    main()
