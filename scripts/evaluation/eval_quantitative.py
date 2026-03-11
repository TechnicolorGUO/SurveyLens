"""
Quantitative evaluation using embedding similarity with ChromaDB.

This module evaluates survey quality by computing embedding similarities between
system-generated surveys and human-written surveys (ground truth).

For each system's survey, we:
1. Extract outline/content/reference entries
2. Generate embeddings for each entry
3. Find the most similar Human entry for each generated entry
4. Calculate average similarity score

The similarity scores indicate how close the system output is to human quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from dotenv import load_dotenv

load_dotenv()

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from openai import OpenAI
from tqdm import tqdm

# Disable verbose logs early
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Ensure we can import pipeline classes
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing_pipeline import SurveyData  # type: ignore

# ----------------------------- Config ----------------------------- #


@dataclass
class QuantitativeEvalConfig:
    """Configuration for quantitative evaluation."""

    processed_dir: str = "results/processed"
    output_dir: str = "results/evaluation"
    chroma_db_dir: str = "chromadb_quantitative"

    # Systems to evaluate (e.g., ["Autosurvey", "Gemini"])
    systems: Optional[List[str]] = None
    # Categories to evaluate (e.g., ["Biology", "Computer Science"])
    categories: Optional[List[str]] = None

    # Embedding model
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_base: Optional[str] = None

    # Batch size for embedding generation
    batch_size: int = 10

    # Which parts to evaluate
    eval_outline: bool = True
    eval_content: bool = True
    eval_reference: bool = True

    # Resume from previous run
    resume_from: Optional[str] = None

    # Use Average Maximum Similarity (AMS) mode
    # When True:
    # - For reference/content: compute average of maximum similarity for each entry
    # - For outline: compute F1 score based on Precision (G->H) and Recall (H->G)
    use_ams: bool = False
    
    # Use bidirectional F1 for all aspects (outline/content/reference) in AMS mode
    # When True and use_ams=True:
    # - All aspects compute F1 score with precision (G→H) and recall (H→G)
    # When False and use_ams=True:
    # - Only outline computes F1, content/reference compute unidirectional AMS
    use_bidirectional_for_all: bool = False
    
    # Use threshold-based matching for precision/recall/F1 in AMS mode
    # When True:
    # - Count a match only if max similarity >= threshold
    # - Precision denominator is total generated entries
    # - Recall denominator is total Human entries
    use_threshold: bool = False
    outline_threshold: float = 0.7
    content_threshold: float = 0.7
    reference_threshold: float = 0.7
    
    # Use thresholded AMS for content/reference in AMS mode
    # When True:
    # - For each entry, max similarity below threshold is treated as 0
    # - Thresholds are aspect-specific
    use_thresholded_ams: bool = False
    
    # Persist system embeddings to ChromaDB before evaluation
    persist_system_embeddings: bool = False
    
    # Use Hungarian matching for one-to-one assignment in AMS F1 scoring
    use_hungarian_matching: bool = False

    # Use redundancy-aware 1-1 BMS (Precision/Recall/BMS with redundancy penalty)
    use_bms: bool = False
    outline_lambda: float = 1.0
    content_lambda: float = 1.0
    reference_lambda: float = 1.0
    
    # Include hit pairs (texts + similarity) in results for P/R numerators
    include_hit_pairs: bool = False
    
    # Force rebuild Human index even if collections already exist
    force_rebuild_human_index: bool = False

    # Topic-to-file mapping for Human alignment
    topic_matches_path: str = "results/topic_matches.json"

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "QuantitativeEvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop legacy top-N keys if present in older configs.
        data.pop("outline_top_n", None)
        data.pop("content_top_n", None)
        data.pop("reference_top_n", None)
        return cls(**data)


# ----------------------------- Embedding utilities ----------------------------- #


class EmbeddingClient:
    """Client for generating embeddings via OpenAI API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or os.environ.get("OPENAI_API_BASE"),
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            embeddings = [data.embedding for data in response.data]
            self.logger.debug(
                f"Generated {len(embeddings)} embeddings "
                f"(tokens: {response.usage.total_tokens})"
            )
            return embeddings
        except Exception as exc:
            self.logger.error(f"Embedding error: {exc}")
            raise


# ----------------------------- ChromaDB helpers ----------------------------- #


def get_or_create_collection(
    client: chromadb.PersistentClient, name: str
) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine distance metric."""
    try:
        return client.get_collection(name)
    except Exception:
        # Use cosine distance to ensure similarity scores are in [0, 1] range
        # cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )


def collection_name_for(category: str, aspect: str, system: str = "Human") -> str:
    """
    Build collection name for a given category/aspect/system.

    Example: Biology_outline_Human
    """
    # Sanitize category name
    category = category.replace(" ", "_").replace(".", "").replace("-", "_")
    return f"{category}_{aspect}_{system}"


# ----------------------------- Data extraction ----------------------------- #


def extract_outline_texts(survey: SurveyData) -> List[str]:
    """
    Extract outline text representations.

    Each outline item title is treated as a separate entry.
    """
    outline_list = survey.outline.to_list()
    # Extract just the titles
    return [title for level, title in outline_list]


def extract_content_texts(survey: SurveyData) -> List[str]:
    """
    Extract content section texts.

    Each section is represented as: "heading: content..."
    """
    texts = []
    for section in survey.content.sections:
        text = f"{section.heading}: {section.content}"
        texts.append(text)
    return texts


def extract_reference_texts(survey: SurveyData) -> List[str]:
    """
    Extract reference texts.

    Each reference entry is a text string.
    """
    texts = []
    for ref in survey.references.entries:
        title = (ref.title or "").strip()
        text = (ref.text or "").strip()
        # Prefer title when available; fall back to raw text.
        texts.append(title or text)
    return texts


# ----------------------------- Index builder ----------------------------- #


class HumanIndexBuilder:
    """
    Builds ChromaDB indices for Human surveys (ground truth).
    """

    def __init__(
        self, config: QuantitativeEvalConfig, embedding_client: EmbeddingClient
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create ChromaDB client with telemetry disabled
        settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_db_dir,
            settings=settings,
        )

    def build_indices(self) -> None:
        """
        Build ChromaDB collections for all Human surveys.

        For each category, we create three collections:
        - {category}_outline_Human
        - {category}_content_Human
        - {category}_reference_Human
        """
        human_dir = Path(self.config.processed_dir) / "Human"
        if not human_dir.exists():
            self.logger.warning(f"Human directory not found: {human_dir}")
            return

        # Get all categories
        categories = [d.name for d in human_dir.iterdir() if d.is_dir()]
        if self.config.categories:
            categories = [c for c in categories if c in self.config.categories]

        self.logger.info(f"Building indices for {len(categories)} categories")

        for category in categories:
            self.logger.info(f"Processing category: {category}")
            self._build_category_index(category)

    def _build_category_index(self, category: str) -> None:
        """Build index for a single category."""
        category_dir = Path(self.config.processed_dir) / "Human" / category
        files = list(category_dir.glob("*_split.json"))

        if not files:
            self.logger.warning(f"No files found in {category_dir}")
            return

        outline_name = collection_name_for(category, "outline")
        content_name = collection_name_for(category, "content")
        reference_name = collection_name_for(category, "reference")

        # Rebuild if requested or if existing collections lack file metadata
        if self.config.force_rebuild_human_index or (
            self._needs_metadata_rebuild(outline_name)
            or self._needs_metadata_rebuild(content_name)
            or self._needs_metadata_rebuild(reference_name)
        ):
            self.logger.info(f"Rebuilding Human index for category: {category}")
            self._reset_collection(outline_name)
            self._reset_collection(content_name)
            self._reset_collection(reference_name)

        # Get or create collections
        outline_coll = get_or_create_collection(self.chroma_client, outline_name)
        content_coll = get_or_create_collection(self.chroma_client, content_name)
        reference_coll = get_or_create_collection(self.chroma_client, reference_name)

        # Check which collections need to be built
        need_outline = outline_coll.count() == 0
        need_content = content_coll.count() == 0
        need_reference = reference_coll.count() == 0
        
        if not (need_outline or need_content or need_reference):
            self.logger.info(f"Category {category} already fully indexed, skipping")
            return
        
        self.logger.info(
            f"Building indices for {category}: "
            f"outline={need_outline}, content={need_content}, reference={need_reference}"
        )

        # Process all files and store per-file metadata for filtering
        self.logger.info(f"Processing {len(files)} files in {category}")
        for file_path in tqdm(files, desc=f"Reading {category} files", leave=False):
            survey = self._load_survey(file_path)
            file_str = str(file_path).replace("\\", "/")
            file_hash = hashlib.md5(file_str.encode("utf-8")).hexdigest()[:12]

            if need_outline and self.config.eval_outline:
                outline_texts = extract_outline_texts(survey)
                if outline_texts:
                    prefix = f"Human_{category}_outline_{file_hash}"
                    self._add_to_collection(
                        outline_coll,
                        outline_texts,
                        prefix,
                        file_str,
                    )
            if need_content and self.config.eval_content:
                content_texts = extract_content_texts(survey)
                if content_texts:
                    prefix = f"Human_{category}_content_{file_hash}"
                    self._add_to_collection(
                        content_coll,
                        content_texts,
                        prefix,
                        file_str,
                    )
            if need_reference and self.config.eval_reference:
                reference_texts = extract_reference_texts(survey)
                if reference_texts:
                    prefix = f"Human_{category}_reference_{file_hash}"
                    self._add_to_collection(
                        reference_coll,
                        reference_texts,
                        prefix,
                        file_str,
                    )

    def _add_to_collection(
        self,
        collection: chromadb.Collection,
        texts: List[str],
        prefix: str,
        file_str: Optional[str] = None,
    ) -> None:
        """Add texts with embeddings to a collection."""
        self.logger.info(f"Adding {len(texts)} items to {collection.name}")

        batch_size = self.config.batch_size
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Embedding {prefix}",
            leave=False,
        ):
            batch_texts = texts[i : i + batch_size]
            try:
                embeddings = self.embedding_client.embed_texts_batch(batch_texts)
                ids = [f"{prefix}_{i + j}" for j in range(len(batch_texts))]
                if file_str:
                    metadatas = [{"file": file_str} for _ in batch_texts]
                    collection.add(
                        embeddings=embeddings,
                        documents=batch_texts,
                        ids=ids,
                        metadatas=metadatas,
                    )
                else:
                    collection.add(
                        embeddings=embeddings, documents=batch_texts, ids=ids
                    )
            except Exception as exc:
                self.logger.error(f"Failed to add batch {i}: {exc}")

    def _needs_metadata_rebuild(self, collection_name: str) -> bool:
        """Return True if collection exists but lacks 'file' metadata on entries."""
        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            return False
        try:
            if collection.count() == 0:
                return False
        except Exception:
            # If count fails, fall back to metadata check below.
            pass
        try:
            sample = collection.get(limit=1, include=["metadatas"])
        except Exception as exc:
            self.logger.warning(
                f"Failed metadata check for {collection_name}: {exc}"
            )
            return False
        metadatas = sample.get("metadatas") or []
        if not metadatas:
            return False
        return not isinstance(metadatas[0], dict) or "file" not in metadatas[0]

    def _reset_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            # If it doesn't exist or cannot be deleted, we'll recreate as needed.
            pass

    def _load_survey(self, json_path: Path) -> SurveyData:
        """Load survey from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SurveyData.from_dict(data)


class SystemIndexBuilder:
    """
    Builds ChromaDB indices for system-generated surveys.
    """

    def __init__(
        self, config: QuantitativeEvalConfig, embedding_client: EmbeddingClient
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(self.__class__.__name__)

        settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_db_dir,
            settings=settings,
        )

    def build_indices(self, systems: Optional[List[str]] = None) -> None:
        """Build ChromaDB collections for selected systems."""
        processed_dir = Path(self.config.processed_dir)
        system_names = systems or self.config.systems or [
            d.name for d in processed_dir.iterdir() if d.is_dir()
        ]
        system_names = [s for s in system_names if s != "Human"]

        if not system_names:
            self.logger.warning("No systems found to index")
            return

        for system in system_names:
            self.logger.info(f"Indexing system: {system}")
            self._build_system_index(system)

    def _build_system_index(self, system: str) -> None:
        system_dir = Path(self.config.processed_dir) / system
        if not system_dir.exists():
            self.logger.warning(f"System directory not found: {system_dir}")
            return

        categories = self.config.categories or [
            d.name for d in system_dir.iterdir() if d.is_dir()
        ]

        for category in categories:
            self.logger.info(f"  Category: {category}")
            self._build_category_index(system, category)

    def _build_category_index(self, system: str, category: str) -> None:
        category_dir = Path(self.config.processed_dir) / system / category
        if not category_dir.exists():
            self.logger.warning(f"Directory not found: {category_dir}")
            return

        files = list(category_dir.glob("*_split.json"))
        if not files:
            self.logger.warning(f"No files found in {category_dir}")
            return

        outline_coll = get_or_create_collection(
            self.chroma_client, collection_name_for(category, "outline", system)
        )
        content_coll = get_or_create_collection(
            self.chroma_client, collection_name_for(category, "content", system)
        )
        reference_coll = get_or_create_collection(
            self.chroma_client, collection_name_for(category, "reference", system)
        )

        for file_path in tqdm(files, desc=f"Indexing {system}/{category}", leave=False):
            survey = self._load_survey(file_path)
            file_str = str(file_path).replace("\\", "/")
            file_hash = hashlib.md5(file_str.encode("utf-8")).hexdigest()[:12]

            if self.config.eval_outline:
                if not self._file_indexed(outline_coll, file_str):
                    texts = extract_outline_texts(survey)
                    self._add_file_entries(
                        outline_coll,
                        texts,
                        f"{system}_{category}_outline_{file_hash}",
                        file_str,
                    )

            if self.config.eval_content:
                if not self._file_indexed(content_coll, file_str):
                    texts = extract_content_texts(survey)
                    self._add_file_entries(
                        content_coll,
                        texts,
                        f"{system}_{category}_content_{file_hash}",
                        file_str,
                    )

            if self.config.eval_reference:
                if not self._file_indexed(reference_coll, file_str):
                    texts = extract_reference_texts(survey)
                    self._add_file_entries(
                        reference_coll,
                        texts,
                        f"{system}_{category}_reference_{file_hash}",
                        file_str,
                    )

    def _file_indexed(self, collection: chromadb.Collection, file_str: str) -> bool:
        try:
            # ChromaDB no longer accepts "ids" in include; ids are returned by default.
            existing = collection.get(where={"file": file_str})
            return bool(existing.get("ids"))
        except Exception as exc:
            self.logger.warning(f"Failed to check file index: {exc}")
            return False

    def _add_file_entries(
        self,
        collection: chromadb.Collection,
        texts: List[str],
        prefix: str,
        file_str: str,
    ) -> None:
        if not texts:
            return

        batch_size = self.config.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                embeddings = self.embedding_client.embed_texts_batch(batch_texts)
                ids = [f"{prefix}_{i + j}" for j in range(len(batch_texts))]
                metadatas = [{"file": file_str} for _ in batch_texts]
                collection.add(
                    embeddings=embeddings,
                    documents=batch_texts,
                    ids=ids,
                    metadatas=metadatas,
                )
            except Exception as exc:
                self.logger.error(f"Failed to add system embeddings batch {i}: {exc}")

    def _load_survey(self, json_path: Path) -> SurveyData:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SurveyData.from_dict(data)


# ----------------------------- Evaluator ----------------------------- #


class QuantitativeEvaluator:
    """
    Evaluates system surveys by computing embedding similarities with Human surveys.
    """

    def __init__(
        self, config: QuantitativeEvalConfig, embedding_client: EmbeddingClient
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create ChromaDB client with telemetry disabled
        settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_db_dir,
            settings=settings,
        )
        self.output_file: Optional[Path] = None
        self.previous_results: Dict[str, Any] = self._load_previous_results()
        self.topic_matches: Optional[Dict[str, Any]] = self._load_topic_matches()
        self._human_embeddings_cache: Dict[
            Tuple[str, str], Tuple[Optional[List[List[float]]], bool]
        ] = {}
        self._human_data_cache: Dict[
            Tuple[str, str], Tuple[Optional[List[List[float]]], Optional[List[str]], bool]
        ] = {}
        # Repo root (…/ASG-Bench) for stable relative/absolute matching.
        self.repo_root = Path(__file__).resolve().parents[2]

    def _load_previous_results(self) -> Dict[str, Any]:
        """Load previous evaluation results if resume_from is specified."""
        if not self.config.resume_from:
            return {"by_system": {}}
        resume_path = Path(self.config.resume_from)
        if not resume_path.exists():
            self.logger.warning(f"Resume file not found: {resume_path}")
            return {"by_system": {}}
        try:
            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("by_system", {})
                return data
            self.logger.warning(
                f"Resume file is not a JSON object: {resume_path}"
            )
        except Exception as exc:
            self.logger.warning(f"Failed to load resume file: {exc}")
        return {"by_system": {}}

    def _load_topic_matches(self) -> Optional[Dict[str, Any]]:
        """Load topic-to-file mapping for Human alignment."""
        if not self.config.topic_matches_path:
            return None
        topic_path = Path(self.config.topic_matches_path)
        if not topic_path.exists():
            self.logger.warning(f"Topic matches file not found: {topic_path}")
            return None
        try:
            with open(topic_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning(f"Failed to load topic matches: {exc}")
            return None

    def _is_file_already_evaluated(
        self, file_path: Path, system: str, category: str
    ) -> bool:
        """Check if a file has already been evaluated."""
        system_data = self.previous_results.get("by_system", {}).get(system, {})
        category_data = system_data.get(category, {})
        files_data = category_data.get("files", [])

        file_str = str(file_path).replace("\\", "/")
        for entry in files_data:
            if entry.get("file", "").replace("\\", "/") == file_str:
                scores = entry.get("scores", {})
                if scores and any(scores.get(k) is not None for k in ["outline", "content", "reference"]):
                    return True
        return False

    def _save_summary_incremental(self, summary: Dict[str, Any]) -> None:
        """Save evaluation summary incrementally."""
        if not self.output_file:
            return
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.error(f"Failed to save: {exc}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Run quantitative evaluation for all configured systems.

        Returns:
            Summary dictionary with scores
        """
        # Determine systems to evaluate
        systems = self.config.systems or self._get_systems()
        systems = [s for s in systems if s != "Human"]  # Exclude Human

        # Initialize summary
        if self.config.resume_from and self.previous_results.get("by_system"):
            summary = self.previous_results.copy()
        else:
            summary = {"by_system": {}, "total": 0}

        # Set up output file
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary["generated_at"] = timestamp
        self.output_file = (
            Path(self.config.output_dir)
            / f"quantitative_evaluation_{timestamp}.json"
        )

        for system in systems:
            self.logger.info(f"Evaluating system: {system}")
            categories = self.config.categories or self._get_categories(system)

            if system not in summary["by_system"]:
                summary["by_system"][system] = {}

            for category in categories:
                self.logger.info(f"  Category: {category}")
                cat_results = self._evaluate_category(system, category, summary)

        # Final save
        summary["generated_at"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary["total"] = sum(
            len(cat_data.get("files", []))
            for sys_data in summary["by_system"].values()
            for cat_data in sys_data.values()
        )
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved evaluation to {self.output_file}")
        return summary

    def _evaluate_category(
        self, system: str, category: str, summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate all files in a system/category."""
        category_dir = Path(self.config.processed_dir) / system / category
        if not category_dir.exists():
            self.logger.warning(f"Directory not found: {category_dir}")
            return []

        files = list(category_dir.glob("*_split.json"))

        # Load existing results
        existing_results = []
        if self.config.resume_from:
            system_data = self.previous_results.get("by_system", {}).get(system, {})
            category_data = system_data.get(category, {})
            existing_results = category_data.get("files", [])

        results = list(existing_results)
        category_start = time.time()

        for file_path in tqdm(files, desc=f"Evaluating {system}/{category}"):
            if self._is_file_already_evaluated(file_path, system, category):
                continue

            try:
                result = self._evaluate_file(file_path, system, category)
                results.append(result)

                # Update summary incrementally
                if system not in summary["by_system"]:
                    summary["by_system"][system] = {}

                duration = round(time.time() - category_start, 2)
                summary["by_system"][system][category] = {
                    "files": results,
                    "averages": self._compute_averages(results),
                    "diagnostics_averages": self._compute_diagnostics_averages(results),
                    "total_duration_seconds": duration,
                }
                summary["total"] = sum(
                    len(cat_data.get("files", []))
                    for sys_data in summary["by_system"].values()
                    for cat_data in sys_data.values()
                )
                self._save_summary_incremental(summary)

            except Exception as exc:
                self.logger.exception(f"Failed to evaluate {file_path}: {exc}")

        return results

    def _evaluate_file(self, file_path: Path, system: str, category: str) -> Dict[str, Any]:
        """
        Evaluate a single survey file.

        For each aspect (outline/content/reference):
        1. Extract texts
        2. Generate embeddings
        3. Query ChromaDB for the most similar Human entry for each generated entry
        4. Compute mean similarity (or F1 for outline in AMS mode)
        """
        start = time.time()
        self.logger.debug(f"Evaluating {file_path.name}")

        survey = self._load_survey(file_path)
        human_survey = self._load_human_survey(file_path, system, category)
        scores: Dict[str, Any] = {}

        # Outline
        if self.config.eval_outline:
            if self.config.use_bms:
                scores["outline"] = self._compute_bms_redundancy(
                    survey, file_path, system, category, "outline"
                )
            elif self.config.use_ams:
                scores["outline"] = self._compute_aspect_f1(
                    survey, file_path, system, category, "outline"
                )
            else:
                scores["outline"] = self._compute_similarity(
                    survey, file_path, system, category, "outline"
                )
        else:
            scores["outline"] = None

        # Content
        if self.config.eval_content:
            if self.config.use_bms:
                scores["content"] = self._compute_bms_redundancy(
                    survey, file_path, system, category, "content"
                )
            elif self.config.use_ams and self.config.use_bidirectional_for_all:
                scores["content"] = self._compute_aspect_f1(
                    survey, file_path, system, category, "content"
                )
            elif self.config.use_ams:
                scores["content"] = self._compute_ams_similarity(
                    survey, file_path, system, category, "content"
                )
            else:
                scores["content"] = self._compute_similarity(
                    survey, file_path, system, category, "content"
                )
        else:
            scores["content"] = None

        # Reference
        if self.config.eval_reference:
            if self.config.use_bms:
                scores["reference"] = self._compute_bms_redundancy(
                    survey, file_path, system, category, "reference"
                )
            elif self.config.use_ams and self.config.use_bidirectional_for_all:
                scores["reference"] = self._compute_aspect_f1(
                    survey, file_path, system, category, "reference"
                )
            elif self.config.use_ams:
                scores["reference"] = self._compute_ams_similarity(
                    survey, file_path, system, category, "reference"
                )
            else:
                scores["reference"] = self._compute_similarity(
                    survey, file_path, system, category, "reference"
                )
        else:
            scores["reference"] = None

        diagnostics: Dict[str, Any] = {}
        if self.config.eval_outline:
            diagnostics["outline"] = {
                "t_ams": self._compute_t_ams_similarity(
                    survey, file_path, system, category, "outline"
                ),
                "redundancy": self._compute_redundancy_index(
                    survey, file_path, system, category, "outline"
                ),
                "dup_rate": self._compute_dup_rate(
                    survey, file_path, system, category, "outline"
                ),
            }
        if self.config.eval_content:
            diagnostics["content"] = {
                "t_ams": self._compute_t_ams_similarity(
                    survey, file_path, system, category, "content"
                ),
                "redundancy": self._compute_redundancy_index(
                    survey, file_path, system, category, "content"
                ),
                "dup_rate": self._compute_dup_rate(
                    survey, file_path, system, category, "content"
                ),
            }
        if self.config.eval_reference:
            diagnostics["reference"] = {
                "t_ams": self._compute_t_ams_similarity(
                    survey, file_path, system, category, "reference"
                ),
                "redundancy": self._compute_redundancy_index(
                    survey, file_path, system, category, "reference"
                ),
                "dup_rate": self._compute_dup_rate(
                    survey, file_path, system, category, "reference"
                ),
            }

        entry_counts: Dict[str, Any] = {}
        for aspect in ["outline", "content", "reference"]:
            if not getattr(self.config, f"eval_{aspect}", False):
                continue
            generated_count = self._count_entries(survey, aspect)
            human_count = (
                self._count_entries(human_survey, aspect) if human_survey else None
            )
            ratio = (
                generated_count / human_count
                if isinstance(human_count, int) and human_count > 0
                else None
            )
            entry_counts[aspect] = {
                "generated": generated_count,
                "human": human_count,
                "ratio": ratio,
            }

        duration = round(time.time() - start, 2)
        human_file = self._human_file_str(file_path, system, category) or None
        mapped = bool(human_file)
        scoped_hit = False
        if self.config.eval_outline:
            scoped_hit = scoped_hit or self._get_scoped_hit_cached(
                file_path, system, category, "outline"
            )
        if self.config.eval_content:
            scoped_hit = scoped_hit or self._get_scoped_hit_cached(
                file_path, system, category, "content"
            )
        if self.config.eval_reference:
            scoped_hit = scoped_hit or self._get_scoped_hit_cached(
                file_path, system, category, "reference"
            )
        return {
            "file": str(file_path),
            "category": category,
            "alignment": {
                "human_file": human_file,
                "mapped": mapped,
                "scoped_hit": scoped_hit,
            },
            "scores": scores,
            "diagnostics": diagnostics,
            "entry_counts": entry_counts,
            "duration_seconds": duration,
        }

    def _compute_similarity(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[float]:
        """
        Compute similarity score for a given aspect.

        Args:
            survey: Survey to evaluate
            category: Category name
            aspect: "outline", "content", or "reference"
        Returns:
            Mean similarity score (0-1) or None if error
        """
        # Extract texts
        if aspect == "outline":
            texts = extract_outline_texts(survey)
        elif aspect == "content":
            texts = extract_content_texts(survey)
        elif aspect == "reference":
            texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        if not texts:
            self.logger.warning(f"No {aspect} texts found for {category}")
            return None
        
        self.logger.debug(f"Extracted {len(texts)} {aspect} texts")

        # Get Human collection
        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
            count = collection.count()
            if count == 0:
                self.logger.error(
                    f"Collection {coll_name} exists but is empty! "
                    f"Did you run --build-index for {category}?"
                )
                return None
            self.logger.debug(f"Using collection {coll_name} with {count} entries")
        except Exception as exc:
            self.logger.error(f"Collection not found: {coll_name}: {exc}")
            return None

        # Generate embeddings (or reuse persisted system embeddings)
        embeddings = self._get_generated_embeddings(
            texts, file_path, system, category, aspect
        )
        if embeddings is None:
            try:
                embeddings = self.embedding_client.embed_texts_batch(texts)
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        # Fetch Human embeddings (file-scoped when possible) and score locally.
        human_embeddings: Optional[List[List[float]]] = None
        human_documents: Optional[List[str]] = None
        if self.config.include_hit_pairs:
            human_embeddings, human_documents, _ = self._get_human_data_for_file(
                collection, file_path, system, category, aspect
            )
        else:
            human_embeddings, _ = self._get_human_embeddings_for_file(
                collection, file_path, system, category, aspect
            )
        if not human_embeddings:
            try:
                include_fields = (
                    ["embeddings", "documents"]
                    if self.config.include_hit_pairs
                    else ["embeddings"]
                )
                all_human = collection.get(include=include_fields)
                human_embeddings = all_human.get("embeddings", [])
                if self.config.include_hit_pairs:
                    human_documents = all_human.get("documents", [])
            except Exception as exc:
                self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
                return None

        if not human_embeddings:
            self.logger.warning(f"No Human embeddings available for {category}/{aspect}")
            return None

        sim_matrix = self._cosine_similarity_matrix(embeddings, human_embeddings)
        if sim_matrix is None or not sim_matrix:
            self.logger.warning(f"No valid similarity matrix for {aspect} in {category}")
            return None

        # Max similarity per generated entry, then average.
        all_scores = [max(row) if row else 0.0 for row in sim_matrix]
        if not all_scores:
            self.logger.warning(
                f"No valid similarity scores for {aspect} in {category}."
            )
            return None

        return sum(all_scores) / len(all_scores)

    def _compute_ams_similarity(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[float]:
        """
        Compute Average Maximum Similarity (AMS) for a given aspect.
        
        For each generated entry, find the maximum similarity with any Human entry,
        then compute the average of these maximum similarities.
        
        Args:
            survey: Survey to evaluate
            category: Category name
            aspect: "content" or "reference"
            
        Returns:
            Average of maximum similarities (0-1) or None if error
        """
        # Extract texts
        if aspect == "content":
            texts = extract_content_texts(survey)
        elif aspect == "reference":
            texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"AMS only supports content/reference, got: {aspect}")

        if not texts:
            self.logger.warning(f"No {aspect} texts found for {category}")
            return None
        
        self.logger.debug(f"Computing AMS for {len(texts)} {aspect} texts")

        # Get Human collection
        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
            count = collection.count()
            if count == 0:
                self.logger.error(f"Collection {coll_name} is empty!")
                return None
            self.logger.debug(f"Using collection {coll_name} with {count} entries")
        except Exception as exc:
            self.logger.error(f"Collection not found: {coll_name}: {exc}")
            return None

        # Generate embeddings for all generated texts (or reuse persisted system embeddings)
        embeddings = self._get_generated_embeddings(
            texts, file_path, system, category, aspect
        )
        if embeddings is None:
            try:
                embeddings = self.embedding_client.embed_texts_batch(texts)
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        threshold = self._get_threshold_for_aspect(aspect)

        # Fetch Human embeddings (file-scoped when possible) and score locally.
        human_embeddings: Optional[List[List[float]]] = None
        human_documents: Optional[List[str]] = None
        if self.config.include_hit_pairs:
            human_embeddings, human_documents, _ = self._get_human_data_for_file(
                collection, file_path, system, category, aspect
            )
        else:
            human_embeddings, _ = self._get_human_embeddings_for_file(
                collection, file_path, system, category, aspect
            )
        if not human_embeddings:
            try:
                include_fields = (
                    ["embeddings", "documents"]
                    if self.config.include_hit_pairs
                    else ["embeddings"]
                )
                all_human = collection.get(include=include_fields)
                human_embeddings = all_human.get("embeddings", [])
                if self.config.include_hit_pairs:
                    human_documents = all_human.get("documents", [])
            except Exception as exc:
                self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
                return None

        if not human_embeddings:
            self.logger.warning(f"No Human embeddings available for {category}/{aspect}")
            return None

        sim_matrix = self._cosine_similarity_matrix(embeddings, human_embeddings)
        if sim_matrix is None or not sim_matrix:
            self.logger.warning(f"No valid similarity matrix for {aspect} in {category}")
            return None

        max_similarities = [max(row) if row else 0.0 for row in sim_matrix]
        if not max_similarities:
            self.logger.warning(f"No valid similarity scores for {aspect} in {category}")
            return None

        if self.config.use_thresholded_ams:
            max_similarities = [
                sim if sim >= threshold else 0.0 for sim in max_similarities
            ]

        ams = sum(max_similarities) / len(max_similarities)
        self.logger.debug(
            f"AMS for {aspect}: {ams:.4f} (from {len(max_similarities)} entries)"
        )
        return ams

    def _compute_bms_redundancy(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute redundancy-aware 1-1 BMS with Hungarian matching.

        - Redundancy penalty on generated side: w(s)=exp(-lambda * max_{s'!=s} sim(s,s'))
        - Matching weight: a_tau(s,r)=max(0, sim(s,r)-tau)
        - Precision: sum w(s)*1[sim>=tau] / |S|
        - Recall: sum 1[sim>=tau] / |R|
        - BMS: harmonic mean of precision and recall
        """
        if aspect == "outline":
            generated_texts = extract_outline_texts(survey)
            lambda_val = self.config.outline_lambda
        elif aspect == "content":
            generated_texts = extract_content_texts(survey)
            lambda_val = self.config.content_lambda
        elif aspect == "reference":
            generated_texts = extract_reference_texts(survey)
            lambda_val = self.config.reference_lambda
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        if not generated_texts:
            self.logger.warning(f"No {aspect} texts found in generated survey")
            return None

        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
            count = collection.count()
            if count == 0:
                self.logger.error(f"Collection {coll_name} is empty!")
                return None
        except Exception as exc:
            self.logger.error(f"Collection not found: {coll_name}: {exc}")
            return None

        generated_embeddings = self._get_generated_embeddings(
            generated_texts, file_path, system, category, aspect
        )
        if generated_embeddings is None:
            try:
                generated_embeddings = self.embedding_client.embed_texts_batch(
                    generated_texts
                )
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        human_embeddings: Optional[List[List[float]]] = None
        human_documents: Optional[List[str]] = None
        if self.config.include_hit_pairs:
            human_embeddings, human_documents, _ = self._get_human_data_for_file(
                collection, file_path, system, category, aspect
            )
        else:
            human_embeddings, _ = self._get_human_embeddings_for_file(
                collection, file_path, system, category, aspect
            )
        if not human_embeddings:
            try:
                include_fields = (
                    ["embeddings", "documents"]
                    if self.config.include_hit_pairs
                    else ["embeddings"]
                )
                all_human = collection.get(include=include_fields)
                human_embeddings = all_human.get("embeddings", [])
                if self.config.include_hit_pairs:
                    human_documents = all_human.get("documents", [])
            except Exception as exc:
                self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
                return None

        if not human_embeddings:
            self.logger.warning(f"No Human embeddings available for {category}/{aspect}")
            return None

        sim_matrix = self._cosine_similarity_matrix(
            generated_embeddings, human_embeddings
        )
        if sim_matrix is None:
            return None

        threshold = self._get_threshold_for_aspect(aspect)
        weight_matrix = [
            [max(0.0, sim - threshold) for sim in row] for row in sim_matrix
        ]

        matches = self._hungarian_match_pairs(weight_matrix)
        if matches is None:
            self.logger.warning(
                "Hungarian matching unavailable; falling back to greedy 1-1 matching"
            )
            matches = self._greedy_match_pairs(weight_matrix)

        if matches is None:
            return None

        redundancy_weights = self._redundancy_weights(
            generated_embeddings, lambda_val
        )

        hit_pairs: Optional[List[Dict[str, Any]]] = None
        if self.config.include_hit_pairs:
            hit_pairs = []

        precision_sum = 0.0
        recall_sum = 0.0
        for gen_idx, human_idx in matches:
            sim = sim_matrix[gen_idx][human_idx]
            if sim >= threshold:
                precision_sum += redundancy_weights[gen_idx]
                recall_sum += 1.0
                if hit_pairs is not None and gen_idx < len(generated_texts):
                    human_text = (
                        human_documents[human_idx]
                        if human_documents and human_idx < len(human_documents)
                        else ""
                    )
                    hit_pairs.append(
                        {
                            "generated": generated_texts[gen_idx],
                            "human": human_text,
                            "similarity": sim,
                        }
                    )

        precision = precision_sum / len(generated_embeddings)
        recall = recall_sum / len(human_embeddings)
        if precision + recall > 0:
            bms = 2 * (precision * recall) / (precision + recall)
        else:
            bms = 0.0

        self.logger.debug(
            f"{aspect.capitalize()} BMS - Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, BMS: {bms:.4f}"
        )
        result: Dict[str, Any] = {"precision": precision, "recall": recall, "bms": bms}
        if hit_pairs is not None:
            result["hit_pairs"] = hit_pairs
        return result

    def _compute_aspect_f1(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute F1 score for an aspect.
        
        Default (max-sim):
        - Precision (G→H): For each generated entry, find max similarity with Human entries
        - Recall (H→G): For each Human entry, find max similarity with generated entries
        - F1: Harmonic mean of precision and recall
        
        Optional (Hungarian, 1-1 matching):
        - Match pairs with 1-1 assignment
        - Precision/Recall numerator is the sum (or hit count) over matched pairs
        - Unmatched items on either side are treated as 0 via the full denominators
        
        Args:
            survey: Survey to evaluate
            category: Category name
            aspect: "outline", "content", or "reference"
            
        Returns:
            Dict with precision, recall, and f1 scores, or None if error.
            When use_thresholded_ams=True, includes thresholded_ams as well.
        """
        # Extract generated texts based on aspect
        if aspect == "outline":
            generated_texts = extract_outline_texts(survey)
        elif aspect == "content":
            generated_texts = extract_content_texts(survey)
        elif aspect == "reference":
            generated_texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"Unknown aspect: {aspect}")
            
        if not generated_texts:
            self.logger.warning(f"No {aspect} texts found in generated survey")
            return None
        
        self.logger.debug(f"Computing {aspect} F1 for {len(generated_texts)} generated entries")

        # Get Human collection
        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
            count = collection.count()
            if count == 0:
                self.logger.error(f"Collection {coll_name} is empty!")
                return None
            self.logger.debug(f"Using collection {coll_name} with {count} entries")
        except Exception as exc:
            self.logger.error(f"Collection not found: {coll_name}: {exc}")
            return None

        # Generate embeddings for generated texts (or reuse persisted system embeddings)
        generated_embeddings = self._get_generated_embeddings(
            generated_texts, file_path, system, category, aspect
        )
        if generated_embeddings is None:
            try:
                generated_embeddings = self.embedding_client.embed_texts_batch(
                    generated_texts
                )
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        # Compute Recall (H→G): For each Human entry, find max similarity with generated
        # We need to get all Human texts and their embeddings
        human_embeddings: Optional[List[List[float]]] = None
        human_documents: Optional[List[str]] = None
        if self.config.include_hit_pairs:
            human_embeddings, human_documents, _ = self._get_human_data_for_file(
                collection, file_path, system, category, aspect
            )
        else:
            human_embeddings, _ = self._get_human_embeddings_for_file(
                collection, file_path, system, category, aspect
            )
        if not human_embeddings:
            try:
                include_fields = (
                    ["embeddings", "documents"]
                    if self.config.include_hit_pairs
                    else ["embeddings"]
                )
                all_human = collection.get(include=include_fields)
                human_embeddings = all_human.get("embeddings", [])
                if self.config.include_hit_pairs:
                    human_documents = all_human.get("documents", [])
            except Exception as exc:
                self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
                return None

        if not human_embeddings:
            self.logger.warning(f"No Human embeddings available for {category}/{aspect}")
            return None

        threshold = self._get_threshold_for_aspect(aspect)

        sim_matrix: Optional[List[List[float]]] = None
        if self.config.use_hungarian_matching:
            sim_matrix = self._cosine_similarity_matrix(
                generated_embeddings, human_embeddings
            )
            if sim_matrix is None or not sim_matrix:
                self.logger.warning(
                    f"No valid similarity matrix for {aspect} in {category}"
                )
                return None
            match_pairs = self._hungarian_match_pairs_from_sim(sim_matrix)
            if match_pairs is None:
                self.logger.warning(
                    "Hungarian matching unavailable, falling back to max-sim scoring"
                )
            else:
                match_sims = [0.0 for _ in range(len(generated_embeddings))]
                for gen_idx, human_idx in match_pairs:
                    if gen_idx < len(sim_matrix) and human_idx < len(sim_matrix[gen_idx]):
                        match_sims[gen_idx] = max(0.0, sim_matrix[gen_idx][human_idx])

                matched_similarity_sum = sum(match_sims)

                if self.config.use_threshold:
                    hits = sum(1 for score in match_sims if score >= threshold)
                    precision = hits / len(generated_embeddings)
                    recall = hits / len(human_embeddings)
                else:
                    # Unmatched items are implicitly 0 via the full denominators.
                    precision = matched_similarity_sum / len(generated_embeddings)
                    recall = matched_similarity_sum / len(human_embeddings)

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                self.logger.debug(
                    f"{aspect.capitalize()} F1 (Hungarian) - "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )

                result: Dict[str, Any] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

                if self.config.include_hit_pairs:
                    pairs_list: List[Dict[str, Any]] = []
                    for gen_idx, human_idx in match_pairs:
                        if gen_idx >= len(generated_texts):
                            continue
                        sim = 0.0
                        if gen_idx < len(sim_matrix) and human_idx < len(sim_matrix[gen_idx]):
                            sim = sim_matrix[gen_idx][human_idx]
                        if self.config.use_threshold and sim < threshold:
                            continue
                        human_text = (
                            human_documents[human_idx]
                            if human_documents and human_idx < len(human_documents)
                            else ""
                        )
                        pairs_list.append(
                            {
                                "generated": generated_texts[gen_idx],
                                "human": human_text,
                                "similarity": sim,
                            }
                        )
                    result["hit_pairs"] = {
                        "precision": pairs_list,
                        "recall": list(pairs_list),
                    }

                if self.config.use_thresholded_ams:
                    thresholded_scores = [
                        score if score >= threshold else 0.0 for score in match_sims
                    ]
                    result["thresholded_ams"] = (
                        sum(thresholded_scores) / len(generated_embeddings)
                        if generated_embeddings
                        else 0.0
                    )

                return result

        # Fallback: max-sim precision/recall via full cosine similarity matrix
        if sim_matrix is None:
            sim_matrix = self._cosine_similarity_matrix(
                generated_embeddings, human_embeddings
            )
        if sim_matrix is None or not sim_matrix:
            self.logger.warning(f"No valid similarity matrix for {aspect} in {category}")
            return None

        precision_scores = [max(row) if row else 0.0 for row in sim_matrix]
        if not precision_scores:
            self.logger.warning(f"No valid precision scores for {aspect} in {category}")
            return None

        hit_pairs: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if self.config.include_hit_pairs:
            precision_pairs: List[Dict[str, Any]] = []
            for gen_idx, row in enumerate(sim_matrix):
                if not row or gen_idx >= len(generated_texts):
                    continue
                best_idx = max(range(len(row)), key=row.__getitem__)
                sim = row[best_idx]
                if self.config.use_threshold and sim < threshold:
                    continue
                human_text = (
                    human_documents[best_idx]
                    if human_documents and best_idx < len(human_documents)
                    else ""
                )
                precision_pairs.append(
                    {
                        "generated": generated_texts[gen_idx],
                        "human": human_text,
                        "similarity": sim,
                    }
                )
            hit_pairs = {"precision": precision_pairs, "recall": []}

        if self.config.use_threshold:
            precision_hits = sum(1 for score in precision_scores if score >= threshold)
            precision = precision_hits / len(precision_scores)
        else:
            precision = sum(precision_scores) / len(precision_scores)

        # Column-wise max for recall (H→G)
        num_human = len(human_embeddings)
        recall_scores = []
        for col_idx in range(num_human):
            col_max = 0.0
            best_gen_idx = -1
            for row_idx, row in enumerate(sim_matrix):
                if col_idx < len(row):
                    if row[col_idx] > col_max:
                        col_max = row[col_idx]
                        best_gen_idx = row_idx
            recall_scores.append(col_max)
            if self.config.include_hit_pairs and best_gen_idx >= 0:
                if not (self.config.use_threshold and col_max < threshold):
                    human_text = (
                        human_documents[col_idx]
                        if human_documents and col_idx < len(human_documents)
                        else ""
                    )
                    if best_gen_idx < len(generated_texts):
                        hit_pairs["recall"].append(
                            {
                                "generated": generated_texts[best_gen_idx],
                                "human": human_text,
                                "similarity": col_max,
                            }
                        )

        if not recall_scores:
            self.logger.warning(f"No valid recall scores for {aspect} in {category}")
            return None

        if self.config.use_threshold:
            recall_hits = sum(1 for score in recall_scores if score >= threshold)
            recall = recall_hits / len(recall_scores)
        else:
            recall = sum(recall_scores) / len(recall_scores)

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        self.logger.debug(
            f"{aspect.capitalize()} F1 scores - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        result: Dict[str, Any] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if hit_pairs is not None:
            result["hit_pairs"] = hit_pairs

        if self.config.use_thresholded_ams:
            thresholded_scores = [
                score if score >= threshold else 0.0 for score in precision_scores
            ]
            result["thresholded_ams"] = (
                sum(thresholded_scores) / len(precision_scores)
                if precision_scores
                else 0.0
            )

        return result

    def _compute_t_ams_similarity(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[float]:
        """
        Compute T-AMS: average max similarity per generated entry,
        with thresholding phi(u)=u if u>=tau else 0.
        """
        if aspect == "outline":
            texts = extract_outline_texts(survey)
        elif aspect == "content":
            texts = extract_content_texts(survey)
        elif aspect == "reference":
            texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        if not texts:
            self.logger.warning(f"No {aspect} texts found for {category}")
            return None

        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
            if collection.count() == 0:
                self.logger.error(f"Collection {coll_name} is empty!")
                return None
        except Exception as exc:
            self.logger.error(f"Collection not found: {coll_name}: {exc}")
            return None

        embeddings = self._get_generated_embeddings(
            texts, file_path, system, category, aspect
        )
        if embeddings is None:
            try:
                embeddings = self.embedding_client.embed_texts_batch(texts)
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        human_embeddings, _ = self._get_human_embeddings_for_file(
            collection, file_path, system, category, aspect
        )
        if not human_embeddings:
            try:
                all_human = collection.get(include=["embeddings"])
                human_embeddings = all_human.get("embeddings", [])
            except Exception as exc:
                self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
                return None

        if not human_embeddings:
            self.logger.warning(f"No Human embeddings available for {category}/{aspect}")
            return None

        sim_matrix = self._cosine_similarity_matrix(embeddings, human_embeddings)
        if sim_matrix is None or not sim_matrix:
            self.logger.warning(f"No valid similarity matrix for {aspect} in {category}")
            return None

        threshold = self._get_threshold_for_aspect(aspect)
        scores = []
        for row in sim_matrix:
            if not row:
                continue
            similarity = max(row)
            if similarity < threshold:
                similarity = 0.0
            scores.append(similarity)

        if not scores:
            self.logger.warning(f"No valid T-AMS scores for {aspect} in {category}")
            return None
        return sum(scores) / len(scores)

    def _compute_redundancy_index(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[float]:
        """Compute redundancy index as mean max intra-set similarity."""
        if aspect == "outline":
            texts = extract_outline_texts(survey)
        elif aspect == "content":
            texts = extract_content_texts(survey)
        elif aspect == "reference":
            texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        if not texts:
            self.logger.warning(f"No {aspect} texts found for {category}")
            return None

        embeddings = self._get_generated_embeddings(
            texts, file_path, system, category, aspect
        )
        if embeddings is None:
            try:
                embeddings = self.embedding_client.embed_texts_batch(texts)
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        if len(embeddings) <= 1:
            return 0.0

        sim_matrix = self._cosine_similarity_matrix(embeddings, embeddings)
        if sim_matrix is None:
            return None

        deltas = []
        for i, row in enumerate(sim_matrix):
            max_sim = 0.0
            for j, sim in enumerate(row):
                if i == j:
                    continue
                max_sim = max(max_sim, sim)
            deltas.append(max_sim)

        return sum(deltas) / len(deltas) if deltas else None

    def _compute_dup_rate(
        self,
        survey: SurveyData,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[float]:
        """
        Compute DupRate@gamma: proportion of entries whose nearest-neighbor
        similarity (within the generated set) meets or exceeds the aspect threshold.
        """
        if aspect == "outline":
            texts = extract_outline_texts(survey)
        elif aspect == "content":
            texts = extract_content_texts(survey)
        elif aspect == "reference":
            texts = extract_reference_texts(survey)
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        if not texts:
            self.logger.warning(f"No {aspect} texts found for {category}")
            return None

        embeddings = self._get_generated_embeddings(
            texts, file_path, system, category, aspect
        )
        if embeddings is None:
            try:
                embeddings = self.embedding_client.embed_texts_batch(texts)
            except Exception as exc:
                self.logger.error(f"Embedding generation failed: {exc}")
                return None

        if len(embeddings) <= 1:
            return 0.0

        sim_matrix = self._cosine_similarity_matrix(embeddings, embeddings)
        if sim_matrix is None:
            return None

        threshold = self._get_threshold_for_aspect(aspect)
        hits = 0
        for i, row in enumerate(sim_matrix):
            max_sim = 0.0
            for j, sim in enumerate(row):
                if i == j:
                    continue
                max_sim = max(max_sim, sim)
            if max_sim >= threshold:
                hits += 1

        return hits / len(embeddings)

    def _get_threshold_for_aspect(self, aspect: str) -> float:
        if aspect == "outline":
            return self.config.outline_threshold
        if aspect == "content":
            return self.config.content_threshold
        if aspect == "reference":
            return self.config.reference_threshold
        raise ValueError(f"Unknown aspect: {aspect}")

    def _get_generated_embeddings(
        self,
        texts: List[str],
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Optional[List[List[float]]]:
        """Load persisted system embeddings for a file if enabled."""
        if not self.config.persist_system_embeddings:
            return None

        coll_name = collection_name_for(category, aspect, system)
        try:
            collection = self.chroma_client.get_collection(coll_name)
        except Exception as exc:
            self.logger.warning(f"System collection not found: {coll_name}: {exc}")
            return None

        file_str = str(file_path).replace("\\", "/")
        try:
            results = collection.get(where={"file": file_str}, include=["embeddings"])
        except Exception as exc:
            self.logger.warning(f"Failed to load system embeddings: {exc}")
            return None

        embeddings = results.get("embeddings") if results else None
        if embeddings:
            return embeddings

        self.logger.warning(
            f"No persisted embeddings for {system}/{category}/{aspect}: {file_path.name}"
        )
        return None

    def _redundancy_weights(
        self, generated_embeddings: List[List[float]], lambda_val: float
    ) -> List[float]:
        if len(generated_embeddings) <= 1:
            return [1.0 for _ in generated_embeddings]

        sim_matrix = self._cosine_similarity_matrix(
            generated_embeddings, generated_embeddings
        )
        if sim_matrix is None:
            return [1.0 for _ in generated_embeddings]

        weights = []
        for i, row in enumerate(sim_matrix):
            max_sim = 0.0
            for j, sim in enumerate(row):
                if i == j:
                    continue
                max_sim = max(max_sim, sim)
            weights.append(math.exp(-lambda_val * max_sim))
        return weights

    def _hungarian_match_pairs(
        self, weight_matrix: List[List[float]]
    ) -> Optional[List[Tuple[int, int]]]:
        if not weight_matrix or not weight_matrix[0]:
            return None
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:
            self.logger.warning(f"scipy not available for Hungarian matching: {exc}")
            return None

        m = len(weight_matrix)
        n = len(weight_matrix[0])
        size = max(m, n)
        max_weight = max(max(row) for row in weight_matrix) if weight_matrix else 0.0

        padded = []
        for i in range(size):
            row = []
            for j in range(size):
                if i < m and j < n:
                    row.append(max_weight - weight_matrix[i][j])
                else:
                    row.append(max_weight)
            padded.append(row)

        try:
            row_ind, col_ind = linear_sum_assignment(padded)
        except Exception as exc:
            self.logger.warning(f"Hungarian matching failed: {exc}")
            return None

        pairs = []
        for r, c in zip(row_ind, col_ind):
            if r < m and c < n:
                pairs.append((r, c))
        return pairs

    def _hungarian_match_pairs_from_sim(
        self, sim_matrix: List[List[float]]
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute one-to-one matching pairs from a similarity matrix."""
        if not sim_matrix or not sim_matrix[0]:
            return None
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:
            self.logger.warning(f"scipy not available for Hungarian matching: {exc}")
            return None

        cost_matrix = [[1.0 - s for s in row] for row in sim_matrix]
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception as exc:
            self.logger.warning(f"Hungarian matching failed: {exc}")
            return None

        pairs: List[Tuple[int, int]] = []
        for r, c in zip(row_ind, col_ind):
            if r < len(sim_matrix) and c < len(sim_matrix[r]):
                pairs.append((r, c))
        return pairs

    def _greedy_match_pairs(
        self, weight_matrix: List[List[float]]
    ) -> Optional[List[Tuple[int, int]]]:
        if not weight_matrix or not weight_matrix[0]:
            return None
        m = len(weight_matrix)
        n = len(weight_matrix[0])
        pairs = []
        used_rows = set()
        used_cols = set()
        all_pairs = []
        for i in range(m):
            for j in range(n):
                all_pairs.append((weight_matrix[i][j], i, j))
        all_pairs.sort(reverse=True, key=lambda x: x[0])
        for weight, i, j in all_pairs:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            pairs.append((i, j))
        return pairs

    def _hungarian_match_similarities(
        self,
        generated_embeddings: List[List[float]],
        human_embeddings: List[List[float]],
    ) -> Optional[List[float]]:
        """Compute one-to-one matching similarities using Hungarian algorithm."""
        if not generated_embeddings or not human_embeddings:
            return None

        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:
            self.logger.warning(f"scipy not available for Hungarian matching: {exc}")
            return None

        sim_matrix = self._cosine_similarity_matrix(
            generated_embeddings, human_embeddings
        )
        if sim_matrix is None:
            return None

        # Convert to cost matrix (maximize similarity -> minimize 1 - sim)
        cost_matrix = [[1.0 - s for s in row] for row in sim_matrix]
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception as exc:
            self.logger.warning(f"Hungarian matching failed: {exc}")
            return None

        match_sims = [0.0 for _ in range(len(generated_embeddings))]
        for r, c in zip(row_ind, col_ind):
            if 0 <= r < len(match_sims):
                match_sims[r] = max(0.0, sim_matrix[r][c])

        return match_sims

    def _cosine_similarity_matrix(
        self,
        generated_embeddings: List[List[float]],
        human_embeddings: List[List[float]],
    ) -> Optional[List[List[float]]]:
        """Compute cosine similarity matrix between generated and human embeddings."""
        try:
            import numpy as np
        except Exception:
            np = None

        if np is not None:
            gen = np.array(generated_embeddings, dtype=np.float32)
            hum = np.array(human_embeddings, dtype=np.float32)
            gen_norm = np.linalg.norm(gen, axis=1, keepdims=True)
            hum_norm = np.linalg.norm(hum, axis=1, keepdims=True)
            gen_norm[gen_norm == 0] = 1.0
            hum_norm[hum_norm == 0] = 1.0
            sim = (gen @ hum.T) / (gen_norm * hum_norm.T)
            sim = np.maximum(sim, 0.0)
            return sim.tolist()

        sim_matrix: List[List[float]] = []
        for gen_embedding in generated_embeddings:
            row = []
            norm_g = sum(g * g for g in gen_embedding) ** 0.5
            for human_embedding in human_embeddings:
                norm_h = sum(h * h for h in human_embedding) ** 0.5
                if norm_g > 0 and norm_h > 0:
                    dot_product = sum(
                        h * g for h, g in zip(human_embedding, gen_embedding)
                    )
                    similarity = dot_product / (norm_h * norm_g)
                    row.append(max(0.0, similarity))
                else:
                    row.append(0.0)
            sim_matrix.append(row)

        return sim_matrix

    def _compute_averages(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average scores across files."""
        totals = {"outline": 0.0, "content": 0.0, "reference": 0.0}
        counts = {"outline": 0, "content": 0, "reference": 0}
        
        # For F1 scores (AMS bidirectional) and BMS scores (redundancy-aware 1-1)
        f1_totals = {
            "outline": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "content": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "reference": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
        f1_counts = {"outline": 0, "content": 0, "reference": 0}
        bms_totals = {
            "outline": {"precision": 0.0, "recall": 0.0, "bms": 0.0},
            "content": {"precision": 0.0, "recall": 0.0, "bms": 0.0},
            "reference": {"precision": 0.0, "recall": 0.0, "bms": 0.0},
        }
        bms_counts = {"outline": 0, "content": 0, "reference": 0}
        thresholded_ams_totals = {"outline": 0.0, "content": 0.0, "reference": 0.0}
        thresholded_ams_counts = {"outline": 0, "content": 0, "reference": 0}

        for entry in files:
            scores = entry.get("scores", {})
            for aspect in totals:
                score = scores.get(aspect)
                if isinstance(score, dict):
                    if "f1" in score and isinstance(score["f1"], (int, float)):
                        f1_totals[aspect]["precision"] += score.get("precision", 0.0)
                        f1_totals[aspect]["recall"] += score.get("recall", 0.0)
                        f1_totals[aspect]["f1"] += score["f1"]
                        f1_counts[aspect] += 1
                    if "bms" in score and isinstance(score["bms"], (int, float)):
                        bms_totals[aspect]["precision"] += score.get("precision", 0.0)
                        bms_totals[aspect]["recall"] += score.get("recall", 0.0)
                        bms_totals[aspect]["bms"] += score["bms"]
                        bms_counts[aspect] += 1
                    if "thresholded_ams" in score and isinstance(
                        score["thresholded_ams"], (int, float)
                    ):
                        thresholded_ams_totals[aspect] += score["thresholded_ams"]
                        thresholded_ams_counts[aspect] += 1
                elif isinstance(score, (int, float)):
                    totals[aspect] += score
                    counts[aspect] += 1

        result = {}
        for aspect in totals:
            if bms_counts[aspect] > 0:
                result[aspect] = {
                    "precision": bms_totals[aspect]["precision"] / bms_counts[aspect],
                    "recall": bms_totals[aspect]["recall"] / bms_counts[aspect],
                    "bms": bms_totals[aspect]["bms"] / bms_counts[aspect],
                }
            elif f1_counts[aspect] > 0:
                result[aspect] = {
                    "precision": f1_totals[aspect]["precision"] / f1_counts[aspect],
                    "recall": f1_totals[aspect]["recall"] / f1_counts[aspect],
                    "f1": f1_totals[aspect]["f1"] / f1_counts[aspect],
                }
                if thresholded_ams_counts[aspect] > 0:
                    result[aspect]["thresholded_ams"] = (
                        thresholded_ams_totals[aspect] / thresholded_ams_counts[aspect]
                    )
            else:
                result[aspect] = (
                    totals[aspect] / counts[aspect] if counts[aspect] else None
                )
        
        return result

    def _compute_diagnostics_averages(
        self, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute average diagnostic scores (T-AMS and Redundancy) across files."""
        totals = {
            "outline": {"t_ams": 0.0, "redundancy": 0.0, "dup_rate": 0.0},
            "content": {"t_ams": 0.0, "redundancy": 0.0, "dup_rate": 0.0},
            "reference": {"t_ams": 0.0, "redundancy": 0.0, "dup_rate": 0.0},
        }
        counts = {
            "outline": {"t_ams": 0, "redundancy": 0, "dup_rate": 0},
            "content": {"t_ams": 0, "redundancy": 0, "dup_rate": 0},
            "reference": {"t_ams": 0, "redundancy": 0, "dup_rate": 0},
        }

        for entry in files:
            diagnostics = entry.get("diagnostics", {})
            for aspect, values in diagnostics.items():
                if not isinstance(values, dict):
                    continue
                t_ams = values.get("t_ams")
                if isinstance(t_ams, (int, float)):
                    totals[aspect]["t_ams"] += t_ams
                    counts[aspect]["t_ams"] += 1
                redundancy = values.get("redundancy")
                if isinstance(redundancy, (int, float)):
                    totals[aspect]["redundancy"] += redundancy
                    counts[aspect]["redundancy"] += 1
                dup_rate = values.get("dup_rate")
                if isinstance(dup_rate, (int, float)):
                    totals[aspect]["dup_rate"] += dup_rate
                    counts[aspect]["dup_rate"] += 1

        result = {}
        for aspect in totals:
            result[aspect] = {
                "t_ams": (
                    totals[aspect]["t_ams"] / counts[aspect]["t_ams"]
                    if counts[aspect]["t_ams"]
                    else None
                ),
                "redundancy": (
                    totals[aspect]["redundancy"] / counts[aspect]["redundancy"]
                    if counts[aspect]["redundancy"]
                    else None
                ),
                "dup_rate": (
                    totals[aspect]["dup_rate"] / counts[aspect]["dup_rate"]
                    if counts[aspect]["dup_rate"]
                    else None
                ),
            }
        return result

    def _load_survey(self, json_path: Path) -> SurveyData:
        """Load survey from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SurveyData.from_dict(data)

    def _resolve_human_path(self, file_str: str) -> Optional[Path]:
        """Resolve a Human file path string to an existing Path when possible."""
        if not file_str:
            return None
        raw = file_str.strip()
        if not raw:
            return None
        try:
            raw_path = Path(raw)
        except Exception:
            return None

        candidates: List[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(self.repo_root / raw_path)
            candidates.append(Path.cwd() / raw_path)

        candidates.append(raw_path)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists():
                return resolved
        return None

    def _load_human_survey(
        self, file_path: Path, system: str, category: str
    ) -> Optional[SurveyData]:
        """Load the aligned Human survey, if a mapping exists."""
        human_file = self._human_file_str(file_path, system, category)
        if not human_file:
            return None
        resolved = self._resolve_human_path(human_file)
        if not resolved or not resolved.exists():
            self.logger.warning(f"Human file not found: {human_file}")
            return None
        try:
            return self._load_survey(resolved)
        except Exception as exc:
            self.logger.warning(f"Failed to load Human survey {resolved}: {exc}")
            return None

    def _count_entries(self, survey: SurveyData, aspect: str) -> int:
        if aspect == "outline":
            return len(extract_outline_texts(survey))
        if aspect == "content":
            return len(extract_content_texts(survey))
        if aspect == "reference":
            return len(extract_reference_texts(survey))
        raise ValueError(f"Unknown aspect: {aspect}")

    def _normalize_path(self, path_str: str) -> str:
        """Normalize paths for robust matching across separators/case."""
        if not path_str:
            return ""
        normalized = os.path.normcase(os.path.normpath(path_str.strip()))
        normalized = normalized.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _path_variants(self, path_str: str) -> List[str]:
        """Generate comparable path variants (raw/abs/relative) for matching."""
        if not path_str:
            return []
        variants = set()
        raw = path_str.strip()

        def add(candidate: str) -> None:
            if candidate:
                variants.add(self._normalize_path(candidate))

        add(raw)
        add(raw.replace("\\", "/"))
        add(os.path.normpath(raw))
        add(os.path.abspath(raw))

        # If path is relative, also resolve it against repo root for consistency.
        if not os.path.isabs(raw):
            try:
                add(str((self.repo_root / raw).resolve()))
            except Exception:
                pass

        # If path is under repo root, add a repo-root relative variant.
        try:
            abs_path = Path(raw)
            if not abs_path.is_absolute():
                abs_path = Path(os.path.abspath(raw))
            rel_path = abs_path.resolve().relative_to(self.repo_root)
            add(str(rel_path))
        except Exception:
            pass

        return [v for v in variants if v]

    def _human_file_str(self, file_path: Path, system: str, category: str) -> str:
        """Map a system file path to its corresponding Human file path string."""
        if not self.topic_matches:
            self.logger.error("Topic matches not loaded; cannot align Human file.")
            return ""
        category_matches = self.topic_matches.get(category, {})
        if not category_matches:
            self.logger.error(
                "No topic matches for category '%s'. Available categories: %s",
                category,
                ", ".join(sorted(self.topic_matches.keys())),
            )
            return ""
        file_str = str(file_path).replace("\\", "/")
        file_variants = set(self._path_variants(file_str))
        total_entries = 0
        system_entries = 0
        sample_system_paths: List[str] = []
        for topic, topic_entries in category_matches.items():
            for entry in topic_entries:
                if not isinstance(entry, dict):
                    continue
                total_entries += 1
                entry_path = entry.get(system)
                if not isinstance(entry_path, str):
                    continue
                system_entries += 1
                if len(sample_system_paths) < 5:
                    sample_system_paths.append(entry_path)
                entry_variants = set(self._path_variants(entry_path))
                if file_variants.intersection(entry_variants):
                    human_path = ""
                    for topic_entry in topic_entries:
                        if isinstance(topic_entry, dict) and isinstance(
                            topic_entry.get("Human"), str
                        ):
                            human_path = topic_entry["Human"]
                            break
                    if human_path:
                        self.logger.debug(
                            "Mapped Human file: topic=%s | system=%s | file=%s | human=%s",
                            topic,
                            system,
                            file_str,
                            human_path,
                        )
                    else:
                        self.logger.error(
                            "Matched system file but no Human entry in topic: %s",
                            topic,
                        )
                    return str(human_path).replace("\\", "/") if human_path else ""
        self.logger.error(
            f"No Human mapping found for system file {file_str} in category {category}."
        )
        self.logger.debug(
            "Mapping debug: file_variants=%s | total_entries=%d | system_entries=%d | "
            "sample_system_paths=%s",
            sorted(file_variants),
            total_entries,
            system_entries,
            sample_system_paths,
        )
        return ""

    def _get_scoped_hit_cached(
        self, file_path: Path, system: str, category: str, aspect: str
    ) -> bool:
        """Check whether file-scoped embeddings were found for the aligned Human file."""
        human_file = self._human_file_str(file_path, system, category)
        if not human_file:
            return False
        cache_key = (
            collection_name_for(category, aspect, "Human"),
            self._normalize_path(human_file),
        )
        cached = self._human_embeddings_cache.get(cache_key)
        if cached is not None:
            return cached[1]
        coll_name = collection_name_for(category, aspect, "Human")
        try:
            collection = self.chroma_client.get_collection(coll_name)
        except Exception:
            return False
        _, scoped_hit = self._get_human_embeddings_for_file(
            collection, file_path, system, category, aspect
        )
        return scoped_hit

    def _get_human_data_for_file(
        self,
        collection: chromadb.Collection,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Tuple[Optional[List[List[float]]], Optional[List[str]], bool]:
        """Fetch Human embeddings/documents scoped to the matching Human file."""
        file_str = self._human_file_str(file_path, system, category)
        if not file_str:
            return None, None, False
        cache_key = (collection.name, self._normalize_path(file_str))
        cached = self._human_data_cache.get(cache_key)
        if cached is not None:
            return cached
        # Human index stores raw file paths with only slash normalization.
        # Try raw variants first, then normalized path variants for compatibility.
        raw_candidates: List[str] = []
        raw = file_str.strip()
        if raw:
            raw_candidates.append(raw)
            raw_candidates.append(raw.replace("\\", "/"))
        candidates = list(dict.fromkeys(raw_candidates + self._path_variants(file_str)))
        for candidate in candidates:
            try:
                scoped = collection.get(
                    where={"file": candidate}, include=["embeddings", "documents"]
                )
                scoped_embeddings = scoped.get("embeddings", [])
                scoped_documents = scoped.get("documents", [])
                if scoped_embeddings:
                    result = (scoped_embeddings, scoped_documents, True)
                    self._human_data_cache[cache_key] = result
                    return result
            except Exception as exc:
                self.logger.debug(
                    f"Failed to filter Human data for {candidate}: {exc}"
                )
                continue

        self.logger.warning(
            f"No Human embeddings for file {file_str} in {collection.name}. "
            "Falling back to full collection; rebuild Human index to enable "
            "file-scoped recall."
        )

        try:
            all_human_data = collection.get(include=["embeddings", "documents"])
            human_embeddings = all_human_data.get("embeddings", [])
            human_documents = all_human_data.get("documents", [])
            if not human_embeddings:
                self.logger.error(
                    f"No embeddings found in Human collection {collection.name}"
                )
                result = (None, None, False)
                self._human_data_cache[cache_key] = result
                return result
            result = (human_embeddings, human_documents, False)
            self._human_data_cache[cache_key] = result
            return result
        except Exception as exc:
            self.logger.error(f"Failed to retrieve Human embeddings: {exc}")
            result = (None, None, False)
            self._human_data_cache[cache_key] = result
            return result

    def _get_human_embeddings_for_file(
        self,
        collection: chromadb.Collection,
        file_path: Path,
        system: str,
        category: str,
        aspect: str,
    ) -> Tuple[Optional[List[List[float]]], bool]:
        """Fetch Human embeddings scoped to the matching Human file when available."""
        embeddings, _, scoped_hit = self._get_human_data_for_file(
            collection, file_path, system, category, aspect
        )
        file_str = self._human_file_str(file_path, system, category)
        if file_str:
            cache_key = (collection.name, self._normalize_path(file_str))
            self._human_embeddings_cache[cache_key] = (embeddings, scoped_hit)
        return embeddings, scoped_hit

    def _get_systems(self) -> List[str]:
        """Get all available systems."""
        processed_dir = Path(self.config.processed_dir)
        return [d.name for d in processed_dir.iterdir() if d.is_dir()]

    def _get_categories(self, system: str) -> List[str]:
        """Get all categories for a system."""
        system_dir = Path(self.config.processed_dir) / system
        return [d.name for d in system_dir.iterdir() if d.is_dir()]


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation using embedding similarity"
    )
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--save-config", help="Save default config and exit")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build Human index before evaluation",
    )
    parser.add_argument("--system", action="append", help="Systems to evaluate")
    parser.add_argument("--category", action="append", help="Categories to evaluate")
    parser.add_argument("--model", help="Embedding model name")
    parser.add_argument("--resume-from", help="Resume from previous evaluation")
    parser.add_argument(
        "--use-ams",
        action="store_true",
        help="Use Average Maximum Similarity mode. "
        "For content/reference: average of max similarity per entry (unidirectional). "
        "For outline: F1 score with Precision (G→H) and Recall (H→G). "
        "This ignores any top-N selection and averages over all entries.",
    )
    parser.add_argument(
        "--use-bidirectional-for-all",
        dest="use_bidirectional_for_all",
        action="store_true",
        help="When used with --use-ams, compute bidirectional F1 (precision/recall/f1) "
        "for all aspects (outline/content/reference) instead of just outline. "
        "Without this flag, only outline computes F1, content/reference use unidirectional AMS.",
    )
    parser.add_argument(
        "--use-threshold",
        action="store_true",
        help="When used with AMS F1 scoring, count a match only if max similarity "
        "meets the configured threshold for each aspect.",
    )
    parser.add_argument(
        "--use-thresholded-ams",
        action="store_true",
        help="When used with AMS (content/reference), set max similarities below the "
        "aspect threshold to 0 before averaging.",
    )
    parser.add_argument(
        "--persist-system-embeddings",
        action="store_true",
        help="Persist system embeddings to ChromaDB before evaluation and reuse them.",
    )
    parser.add_argument(
        "--use-hungarian-matching",
        action="store_true",
        help="Use Hungarian algorithm for one-to-one matching in AMS F1 scoring.",
    )
    parser.add_argument(
        "--use-bms",
        action="store_true",
        help="Use redundancy-aware 1-1 BMS (precision/recall/BMS) as main metric.",
    )
    parser.add_argument(
        "--rebuild-human-index",
        action="store_true",
        help="Force rebuild Human index even if collections already exist.",
    )
    parser.add_argument("--outline-lambda", type=float, help="Redundancy lambda for outline")
    parser.add_argument("--content-lambda", type=float, help="Redundancy lambda for content")
    parser.add_argument("--reference-lambda", type=float, help="Redundancy lambda for reference")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for alignment/matching diagnostics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Save default config if requested
    if args.save_config:
        cfg = QuantitativeEvalConfig()
        cfg.to_json(args.save_config)
        print(f"Default config saved to {args.save_config}")
        return

    # Load config
    config = QuantitativeEvalConfig()
    if args.config:
        config = QuantitativeEvalConfig.from_json(args.config)

    # Override with CLI args
    if args.system:
        config.systems = args.system
    if args.category:
        config.categories = args.category
    if args.model:
        config.embedding_model = args.model
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.use_ams:
        config.use_ams = True
    if args.use_bidirectional_for_all:
        config.use_bidirectional_for_all = True
    if args.use_threshold:
        config.use_threshold = True
    if args.use_thresholded_ams:
        config.use_thresholded_ams = True
    if args.persist_system_embeddings:
        config.persist_system_embeddings = True
    if args.use_hungarian_matching:
        config.use_hungarian_matching = True
    if args.use_bms:
        config.use_bms = True
    if args.rebuild_human_index:
        config.force_rebuild_human_index = True
    if args.outline_lambda is not None:
        config.outline_lambda = args.outline_lambda
    if args.content_lambda is not None:
        config.content_lambda = args.content_lambda
    if args.reference_lambda is not None:
        config.reference_lambda = args.reference_lambda

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Disable verbose HTTP logs from httpx and openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Initialize embedding client
    embedding_model = config.embedding_model or os.environ.get("MODEL")
    if not embedding_model:
        raise ValueError("No embedding model specified (use --model or MODEL env var)")

    embedding_client = EmbeddingClient(
        model=embedding_model,
        api_key=config.embedding_api_key,
        api_base=config.embedding_api_base,
    )

    # Build Human index if requested
    if args.build_index:
        logging.info("Building Human index...")
        index_builder = HumanIndexBuilder(config, embedding_client)
        index_builder.build_indices()
        logging.info("Index building complete")

    # Build system embeddings if requested
    if config.persist_system_embeddings:
        logging.info("Building system embeddings...")
        system_index_builder = SystemIndexBuilder(config, embedding_client)
        system_index_builder.build_indices()
        logging.info("System embeddings complete")

    # Run evaluation
    logging.info("Starting evaluation...")
    evaluator = QuantitativeEvaluator(config, embedding_client)
    summary = evaluator.evaluate()
    
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(json.dumps(summary.get("by_system", {}), indent=2))


if __name__ == "__main__":
    main()
