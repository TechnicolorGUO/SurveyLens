"""Meta-evaluate automatic survey metrics against expert pairwise judgments.

The script consumes already-computed quantitative evaluation JSON files and the
de-identified pairwise labels in ``human_annotation``.  Optionally, it can read
persisted ChromaDB embeddings to add plain MaxSim and bidirectional embedding-F1
baselines without making embedding API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


LOGGER = logging.getLogger("metric_meta_eval")
COMPONENTS = ("outline", "content", "reference")
CHOICES = {
    "A_better": "a",
    "B_better": "b",
    "A": "a",
    "B": "b",
    "tie": "tie",
    "Tie": "tie",
}


@dataclass
class MetaEvalConfig:
    quantitative_result_files: List[str]
    human_annotation_dir: str = "human_annotation"
    output_dir: str = "results/meta_evaluation"

    # Existing metrics saved by eval_quantitative.py.
    existing_metrics: List[str] = None  # type: ignore[assignment]

    # Local lexical baseline computed from processed generated/Human JSON files.
    enable_rouge_1_baseline: bool = True

    # Optional persisted embeddings. No API calls are made by this script.
    chroma_db_dir: Optional[str] = None
    enable_embedding_baselines: bool = False
    embedding_baselines: List[str] = None  # type: ignore[assignment]
    embedding_source: str = "chroma_only"  # chroma_only or chroma_or_api
    embedding_model: Optional[str] = None
    embedding_api_base: Optional[str] = None
    embedding_api_key_env: str = "OPENAI_API_KEY"
    embedding_batch_size: int = 32
    embedding_max_batch_chars: int = 200000
    embedding_request_timeout: float = 120.0
    embedding_max_retries: int = 5
    persist_missing_embeddings: bool = True

    # Same-backbone recomputation of the proposed metrics. These must be
    # computed from the same similarity matrix as the embedding baselines.
    outline_threshold: float = 0.7
    content_threshold: float = 0.7
    reference_threshold: float = 0.8
    outline_lambda: float = 1.0
    content_lambda: float = 1.0
    reference_lambda: float = 1.0

    # Pair/file matching and statistical settings.
    min_topic_match_ratio: float = 0.55
    metric_tie_epsilon: float = 1e-12
    exclude_human_ties: bool = True
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 42
    bootstrap_cluster: str = "annotator"  # annotator or annotator_topic

    # Secondary system-ranking analysis.
    elo_initial: float = 1500.0
    elo_k_factor: float = 32.0
    elo_shuffle_seed: int = 42

    def __post_init__(self) -> None:
        if self.existing_metrics is None:
            self.existing_metrics = ["ra_align_f1", "threshold_gated_maxsim"]
        if self.embedding_baselines is None:
            self.embedding_baselines = [
                "plain_maxsim",
                "embedding_f1",
                "same_backbone_threshold_gated_maxsim",
                "same_backbone_ra_align_f1",
            ]

    @classmethod
    def from_json(cls, path: str) -> "MetaEvalConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass
class PairRecord:
    pair_id: str
    annotator_id: str
    category: str
    component: str
    topic: str
    system_a: str
    system_b: str
    human_choice: str
    file_a: str
    file_b: str
    topic_match_a: float
    topic_match_b: float
    scores: Dict[str, Dict[str, Optional[float]]]


def _normalize(value: str) -> str:
    value = value.casefold().replace("×", "x")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _canonical_system(value: str) -> str:
    aliases = {
        "autosurvey": "autosurvey",
        "auto survey": "autosurvey",
        "autosurvey2": "autosurvey2",
        "auto survey2": "autosurvey2",
        "llmxmapreduce v2": "llmxmapreduce v2",
        "llm x mapreduce v2": "llmxmapreduce v2",
        "gemini 3 pro": "gemini 3 pro preview",
        "qwen3 max": "qwen3 max",
    }
    normalized = _normalize(value)
    return aliases.get(normalized, normalized)


def _category_from_annotator(annotator_id: str) -> str:
    token = annotator_id.split("_")[1] if "_" in annotator_id else annotator_id
    aliases = {
        "CS": "Computer Science",
        "EnvSci": "Environmental Science",
    }
    return aliases.get(token, token)


def _topic_from_prompt(prompt: str, component: str) -> str:
    text = prompt.strip()
    text = re.sub(r"^\s*(?:主题|topic)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(
        rf"\s*(?:的)?\s*{re.escape(component)}\s*$", "", text, flags=re.I
    )
    return text.strip()


def _topic_from_file(path: str) -> str:
    name = Path(path).name
    name = re.sub(r"_split\.json$", "", name, flags=re.I)
    return name.replace("_", " ")


def _iter_label_items(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))
    elif isinstance(payload, dict):
        # Exported files use numeric string keys plus trailing metadata keys.
        numeric = sorted(
            ((int(k), v) for k, v in payload.items() if str(k).isdigit()),
            key=lambda item: item[0],
        )
        yield from (value for _, value in numeric if isinstance(value, dict))
        # Cross-discipline exports store pairwise judgments in Part C, while
        # the CS exports use the numeric-key format above.
        part_c = payload.get("part_c", [])
        if isinstance(part_c, list):
            yield from (value for value in part_c if isinstance(value, dict))


def load_annotations(directory: str) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*_labels.json")):
        with open(path, "r", encoding="utf-8") as handle:
            for item in _iter_label_items(json.load(handle)):
                mapping = item.get("mapping") or {}
                choice = CHOICES.get(str(item.get("choice")))
                component = str(item.get("dataset_type") or "").lower()
                if (
                    choice
                    and component in COMPONENTS
                    and mapping.get("response_a_model")
                    and mapping.get("response_b_model")
                ):
                    annotations.append(item)
    return annotations


def _merge_quantitative_results(paths: Sequence[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for system, categories in payload.get("by_system", {}).items():
            target = merged.setdefault(system, {})
            for category, data in categories.items():
                if category in target:
                    raise ValueError(
                        f"Duplicate system/category in result inputs: {system}/{category}. "
                        "Use the merged result alone or non-overlapping files."
                    )
                target[category] = data
    return merged


def _system_lookup(results: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for system in results:
        key = _canonical_system(system)
        if key in lookup and lookup[key] != system:
            raise ValueError(f"Ambiguous system aliases: {lookup[key]} and {system}")
        lookup[key] = system
    return lookup


def _category_lookup(categories: Iterable[str]) -> Dict[str, str]:
    return {_normalize(category): category for category in categories}


def _find_file_entry(
    results: Dict[str, Any], system: str, category: str, topic: str
) -> Tuple[Optional[Dict[str, Any]], float]:
    files = results.get(system, {}).get(category, {}).get("files", [])
    normalized_topic = _normalize(topic)
    best: Optional[Dict[str, Any]] = None
    best_ratio = 0.0
    for entry in files:
        candidate = _normalize(_topic_from_file(str(entry.get("file", ""))))
        ratio = SequenceMatcher(None, normalized_topic, candidate).ratio()
        if normalized_topic == candidate:
            ratio = 1.0
        elif normalized_topic in candidate or candidate in normalized_topic:
            ratio = max(ratio, 0.95)
        if ratio > best_ratio:
            best, best_ratio = entry, ratio
    return best, best_ratio


def _existing_score(entry: Dict[str, Any], component: str, metric: str) -> Optional[float]:
    value: Any = None
    if metric == "ra_align_f1":
        value = (entry.get("scores", {}).get(component) or {}).get("bms")
    elif metric == "threshold_gated_maxsim":
        value = (entry.get("diagnostics", {}).get(component) or {}).get("t_ams")
    elif metric == "redundancy_index":
        value = (entry.get("diagnostics", {}).get(component) or {}).get("redundancy")
    elif metric == "entry_count_ratio":
        value = (entry.get("entry_counts", {}).get(component) or {}).get("ratio")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _component_text(path: str, component: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    parts: List[str] = []
    if component == "outline":
        for item in payload.get("outline", []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parts.append(str(item[1]))
            elif isinstance(item, dict):
                parts.append(str(item.get("title") or item.get("heading") or ""))
            elif isinstance(item, str):
                parts.append(item)
    elif component == "content":
        for item in payload.get("content", []):
            if isinstance(item, dict):
                parts.append(f"{item.get('heading', '')}: {item.get('content', '')}")
            elif isinstance(item, str):
                parts.append(item)
    elif component == "reference":
        for item in payload.get("references", []):
            if isinstance(item, dict):
                parts.append(str(item.get("title") or item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
    text = "\n".join(part.strip() for part in parts if part and part.strip())
    return text or None


def _rouge_1_f1(candidate: str, reference: str) -> float:
    """Whitespace/punctuation-tokenized ROUGE-1 F1 without external packages."""
    left = re.findall(r"\w+", candidate.casefold())
    right = re.findall(r"\w+", reference.casefold())
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _rouge_score(entry: Dict[str, Any], component: str) -> Optional[float]:
    human_file = (entry.get("alignment") or {}).get("human_file")
    if not human_file:
        return None
    generated = _component_text(str(entry.get("file") or ""), component)
    human = _component_text(str(human_file), component)
    return _rouge_1_f1(generated, human) if generated and human else None


class ChromaBaselines:
    """Read-only embedding baselines over existing ChromaDB collections."""

    def __init__(self, path: str, config: MetaEvalConfig):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for embedding baselines") from exc
        if not Path(path).exists() and config.embedding_source == "chroma_only":
            raise FileNotFoundError(f"ChromaDB path does not exist: {path}")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.config = config
        self._embedding_client: Any = None
        # The same survey appears in many human pairwise comparisons. Cache
        # both embeddings and final per-survey scores to avoid repeated Chroma
        # reads and O(n^2) similarity/Hungarian computations.
        self._embeddings_cache: Dict[
            Tuple[str, str, str], List[List[float]]
        ] = {}
        self._scores_cache: Dict[
            Tuple[str, str, str, str, str], Dict[str, Optional[float]]
        ] = {}

    @staticmethod
    def _collection_name(category: str, component: str, system: str) -> str:
        category = category.replace(" ", "_").replace(".", "").replace("-", "_")
        return f"{category}_{component}_{system}"

    @staticmethod
    def _cosine_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
        # Reference lists can contain hundreds of entries. NumPy turns what
        # would otherwise be tens of millions of Python-level operations into
        # one matrix multiplication. Keep a dependency-free fallback.
        try:
            import numpy as np

            left = np.asarray(a, dtype=np.float32)
            right = np.asarray(b, dtype=np.float32)
            left_norm = np.linalg.norm(left, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right, axis=1, keepdims=True)
            left_norm[left_norm == 0] = 1.0
            right_norm[right_norm == 0] = 1.0
            similarities = (left @ right.T) / (left_norm * right_norm.T)
            return np.maximum(similarities, 0.0).tolist()
        except ImportError:
            pass

        matrix: List[List[float]] = []
        for left in a:
            norm_l = math.sqrt(sum(x * x for x in left)) or 1.0
            row: List[float] = []
            for right in b:
                norm_r = math.sqrt(sum(x * x for x in right)) or 1.0
                sim = sum(x * y for x, y in zip(left, right)) / (norm_l * norm_r)
                row.append(max(0.0, sim))
            matrix.append(row)
        return matrix

    def _threshold(self, component: str) -> float:
        return float(getattr(self.config, f"{component}_threshold"))

    def _lambda(self, component: str) -> float:
        return float(getattr(self.config, f"{component}_lambda"))

    @classmethod
    def _redundancy_weights(
        cls, generated: Sequence[Sequence[float]], lambda_value: float
    ) -> List[float]:
        """Match eval_quantitative.py's generated-side redundancy weights."""
        if len(generated) <= 1:
            return [1.0 for _ in generated]
        matrix = cls._cosine_matrix(generated, generated)
        weights: List[float] = []
        for row_index, row in enumerate(matrix):
            nearest_other = max(
                (similarity for col_index, similarity in enumerate(row) if col_index != row_index),
                default=0.0,
            )
            weights.append(math.exp(-lambda_value * nearest_other))
        return weights

    @staticmethod
    def _hungarian_pairs(weight_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """Maximum-weight one-to-one matching, including zero-weight pairs."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required to recompute same_backbone_ra_align_f1"
            ) from exc
        if not weight_matrix or not weight_matrix[0]:
            return []
        rows, cols = len(weight_matrix), len(weight_matrix[0])
        size = max(rows, cols)
        max_weight = max(max(row) for row in weight_matrix)
        padded_cost = [
            [
                max_weight - weight_matrix[row][col]
                if row < rows and col < cols
                else max_weight
                for col in range(size)
            ]
            for row in range(size)
        ]
        row_indices, col_indices = linear_sum_assignment(padded_cost)
        return [
            (int(row), int(col))
            for row, col in zip(row_indices, col_indices)
            if row < rows and col < cols
        ]

    def _scores_from_embeddings(
        self,
        generated: List[List[float]],
        human: List[List[float]],
        component: str,
    ) -> Dict[str, float]:
        """Compute vanilla and proposed metrics from one shared cosine matrix."""
        matrix = self._cosine_matrix(generated, human)
        generated_max = [max(row) if row else 0.0 for row in matrix]
        human_max = [max(row[col] for row in matrix) for col in range(len(human))]

        precision = statistics.fmean(generated_max)
        recall = statistics.fmean(human_max)
        embedding_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        threshold = self._threshold(component)
        gated_maxsim = statistics.fmean(
            similarity if similarity >= threshold else 0.0
            for similarity in generated_max
        )

        # This reproduces _compute_bms_redundancy in eval_quantitative.py:
        # Hungarian assignment is optimized by max(0, sim - threshold), while
        # only matched pairs meeting the threshold contribute to P/R.
        matching_weights = [
            [max(0.0, similarity - threshold) for similarity in row]
            for row in matrix
        ]
        matches = self._hungarian_pairs(matching_weights)
        redundancy_weights = self._redundancy_weights(
            generated, self._lambda(component)
        )
        precision_sum = 0.0
        recall_sum = 0.0
        for generated_index, human_index in matches:
            if matrix[generated_index][human_index] >= threshold:
                precision_sum += redundancy_weights[generated_index]
                recall_sum += 1.0
        ra_precision = precision_sum / len(generated)
        ra_recall = recall_sum / len(human)
        ra_f1 = (
            2 * ra_precision * ra_recall / (ra_precision + ra_recall)
            if ra_precision + ra_recall
            else 0.0
        )

        return {
            "plain_maxsim": precision,
            "embedding_f1": embedding_f1,
            "same_backbone_threshold_gated_maxsim": gated_maxsim,
            "same_backbone_ra_align_f1": ra_f1,
        }

    @staticmethod
    def _path_variants(path: str) -> List[str]:
        raw = path.replace("\\", "/")
        variants = [raw, str(Path(raw)), str(Path(raw).resolve())]
        return list(dict.fromkeys(variants))

    def _collection(self, collection_name: str):
        try:
            return self.client.get_collection(collection_name)
        except Exception:
            if self.config.embedding_source == "chroma_only":
                raise
            return self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

    @staticmethod
    def _entry_texts(file_path: str, component: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        texts: List[str] = []
        if component == "outline":
            for item in payload.get("outline", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    texts.append(str(item[1]))
                elif isinstance(item, dict):
                    texts.append(str(item.get("title") or item.get("heading") or ""))
                elif isinstance(item, str):
                    texts.append(item)
        elif component == "content":
            for item in payload.get("content", []):
                if isinstance(item, dict):
                    texts.append(f"{item.get('heading', '')}: {item.get('content', '')}")
                elif isinstance(item, str):
                    texts.append(item)
        elif component == "reference":
            for item in payload.get("references", []):
                if isinstance(item, dict):
                    texts.append(str(item.get("title") or item.get("text") or ""))
                elif isinstance(item, str):
                    texts.append(item)
        return [text.strip() for text in texts if text and text.strip()]

    def _client_for_api(self):
        if self._embedding_client is not None:
            return self._embedding_client
        if not self.config.embedding_model:
            raise ValueError("embedding_model is required for chroma_or_api mode")
        import os
        from openai import OpenAI

        api_key = os.environ.get(self.config.embedding_api_key_env)
        if not api_key:
            raise ValueError(
                f"Embedding API key environment variable is missing: "
                f"{self.config.embedding_api_key_env}"
            )
        self._embedding_client = OpenAI(
            api_key=api_key,
            base_url=self.config.embedding_api_base,
            timeout=self.config.embedding_request_timeout,
            max_retries=self.config.embedding_max_retries,
        )
        return self._embedding_client

    def _embed_missing(self, texts: List[str]) -> List[List[float]]:
        client = self._client_for_api()
        embeddings: List[List[float]] = []
        size = max(1, self.config.embedding_batch_size)
        batches: List[List[str]] = []
        batch: List[str] = []
        chars = 0
        for text in texts:
            if batch and (
                len(batch) >= size
                or chars + len(text) > self.config.embedding_max_batch_chars
            ):
                batches.append(batch)
                batch, chars = [], 0
            batch.append(text)
            chars += len(text)
        if batch:
            batches.append(batch)
        for batch in tqdm(
            batches,
            desc="Embedding batches",
            unit="batch",
            leave=False,
        ):
            response = client.embeddings.create(
                model=self.config.embedding_model, input=batch
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([list(map(float, item.embedding)) for item in ordered])
        return embeddings

    def _embeddings(
        self, collection_name: str, file_path: str, component: str
    ) -> List[List[float]]:
        cache_key = (collection_name, str(file_path), component)
        cached = self._embeddings_cache.get(cache_key)
        if cached is not None:
            return cached
        collection = self._collection(collection_name)
        for candidate in self._path_variants(file_path):
            where: Dict[str, Any] = {"file": candidate}
            if self.config.embedding_model:
                where = {
                    "$and": [
                        {"file": candidate},
                        {"embedding_model": str(self.config.embedding_model)},
                    ]
                }
            result = collection.get(where=where, include=["embeddings"])
            embeddings = result.get("embeddings", [])
            if embeddings is not None and len(embeddings) > 0:
                normalized = [list(map(float, row)) for row in embeddings]
                self._embeddings_cache[cache_key] = normalized
                return normalized
        if self.config.embedding_source != "chroma_or_api":
            return []
        texts = self._entry_texts(file_path, component)
        if not texts:
            self._embeddings_cache[cache_key] = []
            return []
        LOGGER.info("Embedding %d missing %s entries from %s", len(texts), component, file_path)
        embeddings = self._embed_missing(texts)
        if self.config.persist_missing_embeddings:
            file_hash = hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:16]
            collection.add(
                ids=[f"meta_{file_hash}_{index}" for index in range(len(texts))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "file": file_path.replace("\\", "/"),
                        "component": component,
                        "source": "meta_selective_api",
                        "embedding_model": str(self.config.embedding_model),
                    }
                    for _ in texts
                ],
            )
        self._embeddings_cache[cache_key] = embeddings
        return embeddings

    def scores(
        self, entry: Dict[str, Any], system: str, category: str, component: str
    ) -> Dict[str, Optional[float]]:
        human_file = str((entry.get("alignment") or {}).get("human_file") or "")
        cache_key = (
            system,
            category,
            component,
            str(entry.get("file") or ""),
            human_file,
        )
        cached = self._scores_cache.get(cache_key)
        if cached is not None:
            return cached
        generated = self._embeddings(
            self._collection_name(category, component, system), entry["file"], component
        )
        if not human_file:
            result = {metric: None for metric in self.config.embedding_baselines}
            self._scores_cache[cache_key] = result
            return result
        human = self._embeddings(
            self._collection_name(category, component, "Human"), human_file, component
        )
        if not generated or not human:
            result = {metric: None for metric in self.config.embedding_baselines}
            self._scores_cache[cache_key] = result
            return result
        result = self._scores_from_embeddings(generated, human, component)
        self._scores_cache[cache_key] = result
        return result


def build_pairs(config: MetaEvalConfig) -> Tuple[List[PairRecord], Dict[str, int]]:
    results = _merge_quantitative_results(config.quantitative_result_files)
    systems = _system_lookup(results)
    all_categories = {category for categories in results.values() for category in categories}
    categories = _category_lookup(all_categories)
    annotations = load_annotations(config.human_annotation_dir)

    chroma: Optional[ChromaBaselines] = None
    if config.enable_embedding_baselines:
        if not config.chroma_db_dir:
            raise ValueError("chroma_db_dir is required when embedding baselines are enabled")
        chroma = ChromaBaselines(config.chroma_db_dir, config)

    pairs: List[PairRecord] = []
    diagnostics: Dict[str, int] = defaultdict(int)
    for item in tqdm(annotations, desc="Matching human annotations", unit="pair"):
        annotator = str(item.get("annotator_id") or "unknown")
        component = str(item["dataset_type"]).lower()
        category_key = _normalize(_category_from_annotator(annotator))
        category = categories.get(category_key)
        if not category:
            diagnostics["unmatched_category"] += 1
            continue
        mapping = item["mapping"]
        raw_a, raw_b = str(mapping["response_a_model"]), str(mapping["response_b_model"])
        system_a = systems.get(_canonical_system(raw_a))
        system_b = systems.get(_canonical_system(raw_b))
        if not system_a or not system_b:
            diagnostics["unmatched_system"] += 1
            continue
        topic = _topic_from_prompt(str(item.get("prompt") or ""), component)
        entry_a, ratio_a = _find_file_entry(results, system_a, category, topic)
        entry_b, ratio_b = _find_file_entry(results, system_b, category, topic)
        if (
            entry_a is None
            or entry_b is None
            or ratio_a < config.min_topic_match_ratio
            or ratio_b < config.min_topic_match_ratio
        ):
            diagnostics["unmatched_topic"] += 1
            continue

        score_map: Dict[str, Dict[str, Optional[float]]] = {}
        for metric in config.existing_metrics:
            score_map[metric] = {
                "a": _existing_score(entry_a, component, metric),
                "b": _existing_score(entry_b, component, metric),
            }
        if config.enable_rouge_1_baseline:
            score_map["rouge_1"] = {
                "a": _rouge_score(entry_a, component),
                "b": _rouge_score(entry_b, component),
            }
        if chroma:
            try:
                baseline_a = chroma.scores(entry_a, system_a, category, component)
                baseline_b = chroma.scores(entry_b, system_b, category, component)
                for metric in config.embedding_baselines:
                    score_map[metric] = {
                        "a": baseline_a.get(metric),
                        "b": baseline_b.get(metric),
                    }
            except Exception as exc:  # keep existing-result evaluation usable
                LOGGER.warning("Embedding baseline failed for pair %s: %s", item.get("id"), exc)
                diagnostics["embedding_baseline_error"] += 1

        choice = CHOICES[str(item["choice"])]
        pairs.append(
            PairRecord(
                pair_id=str(item.get("id") or len(pairs)),
                annotator_id=annotator,
                category=category,
                component=component,
                topic=topic,
                system_a=system_a,
                system_b=system_b,
                human_choice=choice,
                file_a=str(entry_a["file"]),
                file_b=str(entry_b["file"]),
                topic_match_a=ratio_a,
                topic_match_b=ratio_b,
                scores=score_map,
            )
        )
        diagnostics["matched"] += 1
    diagnostics["annotations_total"] = len(annotations)
    return pairs, dict(diagnostics)


def _pair_credit(pair: PairRecord, metric: str, epsilon: float) -> Optional[float]:
    values = pair.scores.get(metric, {})
    a, b = values.get("a"), values.get("b")
    if a is None or b is None or pair.human_choice == "tie":
        return None
    if abs(a - b) <= epsilon:
        return 0.5
    predicted = "a" if a > b else "b"
    return 1.0 if predicted == pair.human_choice else 0.0


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    pairs: List[PairRecord], metric: str, config: MetaEvalConfig
) -> Tuple[Optional[float], Optional[float]]:
    clusters: Dict[str, List[PairRecord]] = defaultdict(list)
    for pair in pairs:
        key = pair.annotator_id
        if config.bootstrap_cluster == "annotator_topic":
            key = f"{key}\0{pair.topic}"
        clusters[key].append(pair)
    keys = list(clusters)
    if not keys or config.bootstrap_samples <= 0:
        return None, None
    rng = random.Random(config.bootstrap_seed)
    estimates: List[float] = []
    for _ in range(config.bootstrap_samples):
        sampled: List[PairRecord] = []
        for _ in keys:
            sampled.extend(clusters[rng.choice(keys)])
        credits = [
            credit
            for pair in sampled
            if (credit := _pair_credit(pair, metric, config.metric_tie_epsilon)) is not None
        ]
        if credits:
            estimates.append(statistics.fmean(credits))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _concordance(
    pairs: List[PairRecord], metric: str, config: MetaEvalConfig
) -> Dict[str, Any]:
    eligible = []
    human_ties = 0
    missing = 0
    for pair in pairs:
        values = pair.scores.get(metric, {})
        if values.get("a") is None or values.get("b") is None:
            missing += 1
            continue
        if pair.human_choice == "tie":
            human_ties += 1
            if config.exclude_human_ties:
                continue
            # Human ties are not forced into a directional concordance claim.
            continue
        eligible.append(pair)
    credits = [
        credit
        for pair in eligible
        if (credit := _pair_credit(pair, metric, config.metric_tie_epsilon)) is not None
    ]
    low, high = _bootstrap_ci(eligible, metric, config)
    return {
        "concordance": statistics.fmean(credits) if credits else None,
        "ci95": [low, high],
        "n": len(credits),
        "correct": sum(1 for value in credits if value == 1.0),
        "metric_ties": sum(1 for value in credits if value == 0.5),
        "human_ties_excluded": human_ties,
        "missing_metric_scores": missing,
    }


def _rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(a) != len(b):
        return None
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return numerator / (denom_a * denom_b) if denom_a and denom_b else None


def _human_elo(pairs: List[PairRecord], config: MetaEvalConfig) -> Dict[str, float]:
    ratings: Dict[str, float] = defaultdict(lambda: config.elo_initial)
    shuffled = list(pairs)
    random.Random(config.elo_shuffle_seed).shuffle(shuffled)
    for pair in shuffled:
        if pair.human_choice not in {"a", "b", "tie"}:
            continue
        ra, rb = ratings[pair.system_a], ratings[pair.system_b]
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        actual_a = 1.0 if pair.human_choice == "a" else 0.0 if pair.human_choice == "b" else 0.5
        delta = config.elo_k_factor * (actual_a - expected_a)
        ratings[pair.system_a] = ra + delta
        ratings[pair.system_b] = rb - delta
    return dict(ratings)


def _spearman_with_human_elo(
    pairs: List[PairRecord], metric: str, config: MetaEvalConfig
) -> Dict[str, Any]:
    usable = [
        pair
        for pair in pairs
        if pair.scores.get(metric, {}).get("a") is not None
        and pair.scores.get(metric, {}).get("b") is not None
    ]
    human = _human_elo(usable, config)
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for pair in usable:
        metric_values[pair.system_a].append(float(pair.scores[metric]["a"]))
        metric_values[pair.system_b].append(float(pair.scores[metric]["b"]))
    systems = sorted(set(human).intersection(metric_values))
    metric_means = [statistics.fmean(metric_values[system]) for system in systems]
    human_values = [human[system] for system in systems]
    rho = _pearson(_rankdata(metric_means), _rankdata(human_values))
    return {"spearman_rho": rho, "n_systems": len(systems), "systems": systems}


def evaluate(config: MetaEvalConfig) -> Dict[str, Any]:
    pairs, matching = build_pairs(config)
    metrics = list(config.existing_metrics)
    if config.enable_rouge_1_baseline:
        metrics.append("rouge_1")
    if config.enable_embedding_baselines:
        metrics.extend(metric for metric in config.embedding_baselines if metric not in metrics)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "matching": matching,
        "metrics": {},
    }
    groups: Dict[str, List[PairRecord]] = {"overall": pairs}
    groups.update({component: [p for p in pairs if p.component == component] for component in COMPONENTS})
    for metric in metrics:
        report["metrics"][metric] = {}
        for group_name, group_pairs in groups.items():
            result = _concordance(group_pairs, metric, config)
            result.update(_spearman_with_human_elo(group_pairs, metric, config))
            report["metrics"][metric][group_name] = result

    # Keep an auditable matched-pair manifest separate from raw survey responses.
    report["pairs"] = [asdict(pair) for pair in pairs]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-evaluate quantitative metrics against expert pairwise labels"
    )
    parser.add_argument("--config", required=True, help="Path to meta-evaluation JSON config")
    parser.add_argument("--output", help="Optional explicit output JSON path")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = MetaEvalConfig.from_json(args.config)
    report = evaluate(config)
    output = Path(args.output) if args.output else Path(config.output_dir) / "metric_meta_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Saved meta-evaluation to %s", output)
    print(json.dumps({"output": str(output), "matching": report["matching"], "metrics": report["metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
