"""V2 meta-evaluation for outline/content/reference alignment.

The same metric-blind survey pairs are judged separately for all three
components. Component ratings are then aggregated with the same equal-weight
mean used by the repository's result analysis. LLM labels are explicitly pilot
artifacts and can later serve as an LLM-judge baseline against human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


LOGGER = logging.getLogger("multicomponent_alignment_proxy")
COMPONENTS = ("outline", "content", "reference")
CHOICES = {"A", "B", "tie", "abstain"}


@dataclass
class Config:
    quantitative_result_file: str
    topic_matches_file: str = "results/topic_matches.json"
    output_dir: str = "results/multicomponent_alignment_proxy"
    sample_size: int = 75
    random_seed: int = 9057
    categories: Optional[List[str]] = None
    max_pairs_per_topic: int = 2
    max_entries: Dict[str, int] = field(
        default_factory=lambda: {"outline": 200, "content": 320, "reference": 250}
    )
    max_combined_chars: Dict[str, int] = field(
        default_factory=lambda: {
            "outline": 50000,
            "content": 180000,
            "reference": 100000,
        }
    )
    metric_tie_epsilon: float = 1e-12
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 42
    aggregate_rating_epsilon: float = 1e-12
    surveygen_title_similarity_threshold: float = 0.95

    model: str = "gpt-5.6-luna"
    endpoint: str = "responses"
    api_key_env: str = "OHMYGPT_API_KEY"
    api_base_env: str = "OHMYGPT_BASE_URL"
    temperature: float = 0.0
    max_retries: int = 4
    request_timeout: float = 180.0
    order_repetitions: int = 2
    env_file: str = ".env"

    def __post_init__(self) -> None:
        if self.sample_size <= 0 or self.max_pairs_per_topic <= 0:
            raise ValueError("sample_size and max_pairs_per_topic must be positive")
        if self.endpoint not in {"responses", "chat_completions"}:
            raise ValueError("endpoint must be responses or chat_completions")
        if self.order_repetitions not in {1, 2}:
            raise ValueError("order_repetitions must be 1 or 2")
        for component in COMPONENTS:
            if self.max_entries.get(component, 0) <= 0:
                raise ValueError(f"Missing positive max_entries for {component}")
            if self.max_combined_chars.get(component, 0) <= 0:
                raise ValueError(f"Missing positive max_combined_chars for {component}")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass
class Candidate:
    system: str
    file: str
    entries: Dict[str, List[str]]
    metrics: Dict[str, Dict[str, float]]


@dataclass
class Pair:
    pair_id: str
    category: str
    topic: str
    human_file: str
    system_a: str
    file_a: str
    system_b: str
    file_b: str
    entry_counts: Dict[str, Dict[str, int]]
    input_chars: Dict[str, int]
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    file_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Label:
    pair_id: str
    component: str
    winner: str
    alignment_a: Optional[float]
    alignment_b: Optional[float]
    status: str
    source: str
    confidence: str = ""
    reason_codes: List[str] = field(default_factory=list)
    reason: str = ""
    order_consistent: Optional[bool] = None
    model: str = ""
    annotator_id: str = ""


def _load_env(path: str) -> None:
    if not Path(path).exists():
        return
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _safe_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def extract_entries(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    outline: List[str] = []
    for item in data.get("outline", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1]).strip()
        elif isinstance(item, dict):
            text = str(item.get("title") or item.get("heading") or "").strip()
        else:
            text = str(item).strip() if isinstance(item, str) else ""
        if text:
            outline.append(text)
    content: List[str] = []
    for item in data.get("content", []):
        if isinstance(item, dict):
            heading = str(item.get("heading") or "").strip()
            body = str(item.get("content") or "").strip()
            text = f"{heading}: {body}".strip(": ")
        else:
            text = str(item).strip() if isinstance(item, str) else ""
        if text:
            content.append(text)
    reference: List[str] = []
    for item in data.get("references", []):
        if isinstance(item, dict):
            text = str(item.get("title") or item.get("text") or "").strip()
        else:
            text = str(item).strip() if isinstance(item, str) else ""
        if text:
            reference.append(text)
    return {"outline": outline, "content": content, "reference": reference}


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _rouge1_f1(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = Counter(_tokens("\n".join(left))), Counter(_tokens("\n".join(right)))
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0
    overlap = sum((a & b).values())
    precision, recall = overlap / na, overlap / nb
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _rouge_n_f1(left: Sequence[str], right: Sequence[str], n: int) -> float:
    def ngrams(entries: Sequence[str]) -> Counter:
        tokens = _tokens("\n".join(entries))
        return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    a, b = ngrams(left), ngrams(right)
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0
    overlap = sum((a & b).values())
    precision, recall = overlap / na, overlap / nb
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _token_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(_tokens("\n".join(left))), set(_tokens("\n".join(right)))
    return len(a & b) / len(a | b) if a or b else 0.0


def _normalize_entry(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _entry_exact_f1(left: Sequence[str], right: Sequence[str]) -> float:
    a = {_normalize_entry(value) for value in left if _normalize_entry(value)}
    b = {_normalize_entry(value) for value in right if _normalize_entry(value)}
    overlap = len(a & b)
    precision = overlap / len(a) if a else 0.0
    recall = overlap / len(b) if b else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _reference_set_scores(
    generated: Sequence[str], human: Sequence[str], threshold: float
) -> Dict[str, float]:
    """Reuse the citation-set baseline implementation from eval_metric_meta.py."""
    from eval_metric_meta import _citation_set_scores, _normalize_reference_title

    normalized_generated = [
        value for title in generated if (value := _normalize_reference_title(title))
    ]
    normalized_human = [
        value for title in human if (value := _normalize_reference_title(title))
    ]
    return _citation_set_scores(normalized_generated, normalized_human, threshold)


def _path_key(path: str) -> str:
    return str(Path(path).resolve())


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metric_index(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for categories in payload.get("by_system", {}).values():
        for category_data in categories.values():
            for item in category_data.get("files", []):
                component_metrics: Dict[str, Dict[str, float]] = {}
                for component in COMPONENTS:
                    ra = _safe_float((item.get("scores", {}).get(component) or {}).get("bms"))
                    gated = _safe_float(
                        (item.get("diagnostics", {}).get(component) or {}).get("t_ams")
                    )
                    if ra is not None and gated is not None:
                        component_metrics[component] = {
                            "ra_align_f1": ra,
                            "threshold_gated_maxsim": gated,
                        }
                index[_path_key(str(item.get("file") or ""))] = component_metrics
    return index


def _topic_system_files(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, str]]]:
    for category, topics in payload.items():
        if not isinstance(topics, dict):
            continue
        for topic, records in topics.items():
            systems: Dict[str, str] = {}
            for record in records if isinstance(records, list) else []:
                if isinstance(record, dict):
                    for system, path in record.items():
                        systems[str(system)] = str(path)
            yield str(category), str(topic), systems


def build_eligible_pool(config: Config) -> Tuple[Dict[str, Dict[str, List[Tuple[Candidate, Candidate, Candidate]]]], Dict[str, int]]:
    with open(config.topic_matches_file, "r", encoding="utf-8") as handle:
        topic_matches = json.load(handle)
    metric_index = load_metric_index(config.quantitative_result_file)
    allowed = set(config.categories or [])
    entry_cache: Dict[str, Dict[str, List[str]]] = {}
    rejection: Counter[str] = Counter()
    pool: Dict[str, Dict[str, List[Tuple[Candidate, Candidate, Candidate]]]] = defaultdict(dict)

    def candidate(system: str, path: str) -> Optional[Candidate]:
        if not Path(path).exists():
            rejection["missing_file"] += 1
            return None
        entries = entry_cache.setdefault(path, extract_entries(path))
        if any(not entries[component] for component in COMPONENTS):
            rejection["empty_component"] += 1
            return None
        if any(len(entries[c]) > config.max_entries[c] for c in COMPONENTS):
            rejection["entry_cap"] += 1
            return None
        metrics = metric_index.get(_path_key(path), {})
        if any(component not in metrics for component in COMPONENTS):
            rejection["missing_metric"] += 1
            return None
        return Candidate(system=system, file=path, entries=entries, metrics=metrics)

    for category, topic, systems in _topic_system_files(topic_matches):
        if allowed and category not in allowed:
            continue
        human_path = systems.get("Human")
        if not human_path:
            rejection["missing_human"] += 1
            continue
        if not Path(human_path).exists():
            rejection["missing_human_file"] += 1
            continue
        human_entries = entry_cache.setdefault(human_path, extract_entries(human_path))
        if any(not human_entries[c] for c in COMPONENTS):
            rejection["human_empty_component"] += 1
            continue
        if any(len(human_entries[c]) > config.max_entries[c] for c in COMPONENTS):
            rejection["human_entry_cap"] += 1
            continue
        human = Candidate("Human", human_path, human_entries, {})
        candidates = [
            value for system, path in sorted(systems.items()) if system != "Human"
            and (value := candidate(system, path)) is not None
        ]
        possible: List[Tuple[Candidate, Candidate, Candidate]] = []
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                combined = {
                    component: sum(
                        len("\n".join(item.entries[component]))
                        for item in (human, left, right)
                    )
                    for component in COMPONENTS
                }
                if any(combined[c] > config.max_combined_chars[c] for c in COMPONENTS):
                    rejection["combined_char_cap"] += 1
                    continue
                possible.append((human, left, right))
        if possible:
            pool[category][topic] = possible
        else:
            rejection["topic_without_pair"] += 1
    return dict(pool), dict(rejection)


def _make_pair(category: str, topic: str, human: Candidate, left: Candidate, right: Candidate, rng: random.Random) -> Pair:
    if rng.random() < 0.5:
        left, right = right, left
    identity = "|".join(
        [category, topic] + sorted([f"{left.system}:{_path_key(left.file)}", f"{right.system}:{_path_key(right.file)}"])
    )
    pair_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for component in COMPONENTS:
        metrics[component] = {}
        names = sorted(set(left.metrics[component]) & set(right.metrics[component]))
        for name in names:
            metrics[component][name] = {
                "a": left.metrics[component][name],
                "b": right.metrics[component][name],
            }
        metrics[component]["rouge1_f1"] = {
            "a": _rouge1_f1(left.entries[component], human.entries[component]),
            "b": _rouge1_f1(right.entries[component], human.entries[component]),
        }
    return Pair(
        pair_id=pair_id,
        category=category,
        topic=topic,
        human_file=human.file,
        system_a=left.system,
        file_a=left.file,
        system_b=right.system,
        file_b=right.file,
        entry_counts={
            c: {"reference": len(human.entries[c]), "a": len(left.entries[c]), "b": len(right.entries[c])}
            for c in COMPONENTS
        },
        input_chars={
            c: sum(len("\n".join(item.entries[c])) for item in (human, left, right))
            for c in COMPONENTS
        },
        metrics=metrics,
        file_hashes={
            "reference": _file_sha256(human.file),
            "a": _file_sha256(left.file),
            "b": _file_sha256(right.file),
        },
    )


def sample_pairs(pool: Dict[str, Dict[str, List[Tuple[Candidate, Candidate, Candidate]]]], config: Config) -> List[Pair]:
    rng = random.Random(config.random_seed)
    categories = sorted(pool)
    base, remainder = divmod(config.sample_size, len(categories))
    targets = {category: base + (index < remainder) for index, category in enumerate(categories)}
    capacities = {
        category: sum(
            min(
                config.max_pairs_per_topic,
                len({tuple(sorted((left.system, right.system))) for _, left, right in triples}),
            )
            for triples in topics.values()
        )
        for category, topics in pool.items()
    }
    deficit = 0
    for category in categories:
        if targets[category] > capacities[category]:
            deficit += targets[category] - capacities[category]
            targets[category] = capacities[category]
    while deficit:
        receivers = [category for category in categories if targets[category] < capacities[category]]
        if not receivers:
            raise RuntimeError(
                f"Eligible pool capacity is below sample_size={config.sample_size}"
            )
        receiver = min(receivers, key=lambda category: (targets[category], category))
        targets[receiver] += 1
        deficit -= 1
    system_use: Counter[str] = Counter()
    system_pair_use: Counter[Tuple[str, str]] = Counter()
    sampled: List[Pair] = []
    for category in categories:
        candidates: List[Tuple[str, Candidate, Candidate, Candidate]] = []
        for topic, triples in pool[category].items():
            for human, left, right in triples:
                candidates.append((topic, human, left, right))
        rng.shuffle(candidates)
        topic_use: Counter[str] = Counter()
        selected_keys: set = set()
        for _ in range(targets[category]):
            eligible = [
                item for item in candidates
                if topic_use[item[0]] < config.max_pairs_per_topic
                and tuple(sorted((item[2].system, item[3].system))) + (item[0],) not in selected_keys
            ]
            if not eligible:
                raise RuntimeError(f"Cannot sample target {targets[category]} for {category}")
            eligible.sort(
                key=lambda item: (
                    topic_use[item[0]],
                    system_pair_use[tuple(sorted((item[2].system, item[3].system)))],
                    system_use[item[2].system] + system_use[item[3].system],
                )
            )
            topic, human, left, right = eligible[0]
            sampled.append(_make_pair(category, topic, human, left, right, rng))
            key = tuple(sorted((left.system, right.system)))
            selected_keys.add(key + (topic,))
            topic_use[topic] += 1
            system_pair_use[key] += 1
            system_use[left.system] += 1
            system_use[right.system] += 1
    return sampled


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(config: Config) -> Tuple[List[Pair], Dict[str, Any]]:
    pool, rejection = build_eligible_pool(config)
    pairs = sample_pairs(pool, config)
    output = Path(config.output_dir)
    _write_jsonl(output / "pairs.jsonl", (asdict(pair) for pair in pairs))
    manifest = {
        "protocol_version": "alignment-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "UNLABELED",
        "sampling": "topic_matches canonical mapping; metadata-only eligibility; discipline-stratified",
        "n_pairs": len(pairs),
        "category_counts": dict(Counter(pair.category for pair in pairs)),
        "topic_counts": dict(Counter(f"{pair.category}/{pair.topic}" for pair in pairs)),
        "system_counts": dict(Counter(s for pair in pairs for s in (pair.system_a, pair.system_b))),
        "eligible_topics": {category: len(topics) for category, topics in pool.items()},
        "rejections": rejection,
        "config": asdict(config),
        "warning": "LLM proxy labels are not human evidence.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(output / "human_labels_template.csv", "w", encoding="utf-8", newline="") as handle:
        fields = ["pair_id", "category", "topic", "component", "winner", "alignment_a", "alignment_b", "confidence", "reason_codes", "reason", "annotator_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            for component in COMPONENTS:
                writer.writerow({"pair_id": pair.pair_id, "category": pair.category, "topic": pair.topic, "component": component})
    return pairs, manifest


def enrich_metrics(pairs: Sequence[Pair], config: Config) -> List[Pair]:
    """Add dependency-free lexical and citation baselines without changing pairs."""
    entry_cache: Dict[str, Dict[str, List[str]]] = {}
    progress = tqdm(pairs, desc="Computing local baselines", unit="pair", dynamic_ncols=True)
    for pair in progress:
        human = entry_cache.setdefault(pair.human_file, extract_entries(pair.human_file))
        left = entry_cache.setdefault(pair.file_a, extract_entries(pair.file_a))
        right = entry_cache.setdefault(pair.file_b, extract_entries(pair.file_b))
        for component in COMPONENTS:
            metrics = pair.metrics[component]
            metrics["rouge2_f1"] = {
                "a": _rouge_n_f1(left[component], human[component], 2),
                "b": _rouge_n_f1(right[component], human[component], 2),
            }
            metrics["token_jaccard"] = {
                "a": _token_jaccard(left[component], human[component]),
                "b": _token_jaccard(right[component], human[component]),
            }
            metrics["entry_exact_f1"] = {
                "a": _entry_exact_f1(left[component], human[component]),
                "b": _entry_exact_f1(right[component], human[component]),
            }
        for side, entries in (("a", left["reference"]), ("b", right["reference"])):
            scores = _reference_set_scores(
                entries,
                human["reference"],
                config.surveygen_title_similarity_threshold,
            )
            for metric in (
                "citation_normalized_exact_f1",
                "surveyforge_sam_r",
                "surveygen_style_f1",
            ):
                pair.metrics["reference"].setdefault(metric, {})[side] = scores[metric]
    _write_jsonl(Path(config.output_dir) / "pairs.jsonl", (asdict(pair) for pair in pairs))
    return list(pairs)


def load_pairs(config: Config) -> List[Pair]:
    return [Pair(**row) for row in _read_jsonl(Path(config.output_dir) / "pairs.jsonl")]


def build_prompt(pair: Pair, component: str, swapped: bool) -> str:
    reference = extract_entries(pair.human_file)[component]
    left = extract_entries(pair.file_a)[component]
    right = extract_entries(pair.file_b)[component]
    if swapped:
        left, right = right, left
    component_note = {
        "outline": "Each entry is an outline heading. Judge topical/section coverage and precision, not writing style.",
        "content": "Each entry is a complete section represented as heading plus content. Judge substantive coverage, precision, and non-redundancy, not prose fluency.",
        "reference": "Each entry is a citation title or raw bibliography entry. Judge coverage, precision, and duplicate/near-duplicate citations.",
    }[component]
    data = {
        "protocol_version": "alignment-v2",
        "pair_id": pair.pair_id,
        "component": component,
        "reference": {"entries": reference},
        "candidate_a": {"entries": left},
        "candidate_b": {"entries": right},
    }
    return f"""You are a blinded annotator of human-reference alignment. {component_note}
Compare candidate A and B only by alignment to the supplied human reference. Do not judge
standalone survey quality, system identity, prestige, or list length by itself. Rate each
candidate on the same 0-4 scale: 0=no recognizable alignment, 1=weak, 2=partial,
3=strong, 4=near-complete and precise. Use abstain only for malformed or genuinely
unjudgeable input; tie means equal alignment. Winner must agree with the two ratings.

INPUT JSON:
{json.dumps(data, ensure_ascii=False)}

Return JSON only:
{{"winner":"A|B|tie|abstain","alignment_a":0,"alignment_b":0,
"confidence":"high|medium|low","reason_codes":["coverage","precision","redundancy"],
"reason":"one concise sentence"}}"""


def _parse_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Response is not an object")
    winner = str(value.get("winner", "")).strip()
    winner = winner if winner in {"tie", "abstain"} else winner.upper()
    if winner not in CHOICES:
        raise ValueError(f"Invalid winner {winner}")
    a, b = _safe_float(value.get("alignment_a")), _safe_float(value.get("alignment_b"))
    if a is None or b is None or not (0 <= a <= 4 and 0 <= b <= 4):
        raise ValueError("alignment_a/b must be numbers in [0,4]")
    expected = "A" if a > b else "B" if b > a else "tie"
    if winner != "abstain" and winner != expected:
        raise ValueError(f"Winner {winner} conflicts with ratings {a}/{b}")
    confidence = str(value.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    codes = value.get("reason_codes", [])
    return {
        "winner": winner,
        "alignment_a": a,
        "alignment_b": b,
        "confidence": confidence,
        "reason_codes": [str(code) for code in codes] if isinstance(codes, list) else [],
        "reason": str(value.get("reason", "")).strip(),
    }


def _response_text(response: Any) -> str:
    direct = str(getattr(response, "output_text", "") or "")
    if direct:
        return direct
    payload = response.model_dump() if hasattr(response, "model_dump") else response if isinstance(response, dict) else {}
    texts: List[str] = []
    for item in payload.get("output", []) if isinstance(payload, dict) else []:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts)


class Judge:
    def __init__(self, config: Config):
        _load_env(config.env_file)
        key = os.environ.get(config.api_key_env)
        if not key:
            raise RuntimeError(f"Missing {config.api_key_env}")
        from openai import OpenAI
        kwargs: Dict[str, Any] = {"api_key": key, "timeout": config.request_timeout}
        base = os.environ.get(config.api_base_env)
        if base:
            kwargs["base_url"] = base
        self.client, self.config = OpenAI(**kwargs), config

    def call(self, pair: Pair, component: str, swapped: bool) -> Dict[str, Any]:
        prompt = build_prompt(pair, component, swapped)
        last: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                if self.config.endpoint == "responses":
                    response = self.client.responses.create(model=self.config.model, input=prompt)
                    raw = _response_text(response)
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"},
                    )
                    raw = str(response.choices[0].message.content or "")
                parsed = _parse_object(raw)
                if swapped:
                    parsed["winner"] = "B" if parsed["winner"] == "A" else "A" if parsed["winner"] == "B" else parsed["winner"]
                    parsed["alignment_a"], parsed["alignment_b"] = parsed["alignment_b"], parsed["alignment_a"]
                return {**parsed, "raw": raw, "swapped": swapped, "model": self.config.model}
            except Exception as exc:
                last = exc
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"Non-retryable API error: {exc}") from exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Judge failed for {pair.pair_id}/{component}: {last}")


def merge_orders(pair_id: str, component: str, rows: Sequence[Dict[str, Any]], model: str) -> Label:
    if any(row["winner"] == "abstain" for row in rows):
        status, winner, consistent = "abstained", "abstain", False
    elif len({row["winner"] for row in rows}) != 1:
        status, winner, consistent = "order_unstable", "abstain", False
    else:
        status, winner, consistent = "accepted", str(rows[0]["winner"]), True
    confidences = [str(row.get("confidence", "low")) for row in rows]
    rank = {"low": 0, "medium": 1, "high": 2}
    return Label(
        pair_id=pair_id,
        component=component,
        winner=winner,
        alignment_a=statistics.fmean(float(row["alignment_a"]) for row in rows) if status == "accepted" else None,
        alignment_b=statistics.fmean(float(row["alignment_b"]) for row in rows) if status == "accepted" else None,
        status=status,
        source="llm_proxy",
        confidence=min(confidences, key=lambda value: rank.get(value, 0)),
        reason_codes=sorted({code for row in rows for code in row.get("reason_codes", [])}),
        reason=" | ".join(str(row.get("reason", "")) for row in rows if row.get("reason")),
        order_consistent=consistent,
        model=model,
    )


def run_labeling(pairs: Sequence[Pair], config: Config) -> List[Label]:
    output = Path(config.output_dir)
    raw_path = output / "proxy_raw.jsonl"
    existing = _read_jsonl(raw_path)
    cache = {(row["pair_id"], row["component"], bool(row["swapped"])): row for row in existing}
    all_raw = list(existing)
    judge = Judge(config)
    labels: List[Label] = []
    tasks = [(pair, component) for pair in pairs for component in COMPONENTS]
    progress = tqdm(tasks, desc="V2 proxy labeling", unit="component", dynamic_ncols=True)
    for pair, component in progress:
        rows: List[Dict[str, Any]] = []
        hits = 0
        for swapped in ([False, True] if config.order_repetitions == 2 else [False]):
            key = (pair.pair_id, component, swapped)
            row = cache.get(key)
            if row is None:
                row = {"pair_id": pair.pair_id, "component": component, **judge.call(pair, component, swapped)}
                cache[key] = row
                all_raw.append(row)
                _write_jsonl(raw_path, all_raw)
            else:
                hits += 1
            rows.append(row)
        label = merge_orders(pair.pair_id, component, rows, config.model)
        labels.append(label)
        progress.set_postfix(component=component, status=label.status, cached=f"{hits}/{len(rows)}")
    _write_jsonl(output / "proxy_labels.jsonl", (asdict(label) for label in labels))
    return labels


def load_labels(path: Path) -> List[Label]:
    if path.suffix.lower() == ".jsonl":
        return [Label(**row) for row in _read_jsonl(path)]
    labels: List[Label] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            winner = str(row.get("winner") or "").strip()
            winner = winner if winner in {"tie", "abstain"} else winner.upper()
            a, b = _safe_float(row.get("alignment_a")), _safe_float(row.get("alignment_b"))
            if winner not in CHOICES or a is None or b is None:
                continue
            labels.append(Label(
                pair_id=str(row["pair_id"]), component=str(row["component"]), winner=winner,
                alignment_a=a, alignment_b=b, status="accepted", source="human",
                confidence=str(row.get("confidence") or ""),
                reason_codes=[x.strip() for x in str(row.get("reason_codes") or "").split("|") if x.strip()],
                reason=str(row.get("reason") or ""), annotator_id=str(row.get("annotator_id") or ""),
            ))
    return labels


def aggregate_labels(labels: Sequence[Label], epsilon: float) -> List[Label]:
    by_pair: Dict[str, Dict[str, Label]] = defaultdict(dict)
    for label in labels:
        by_pair[label.pair_id][label.component] = label
    aggregates: List[Label] = []
    for pair_id, components in by_pair.items():
        if any(c not in components or components[c].status != "accepted" for c in COMPONENTS):
            continue
        a = statistics.fmean(float(components[c].alignment_a) for c in COMPONENTS)  # type: ignore[arg-type]
        b = statistics.fmean(float(components[c].alignment_b) for c in COMPONENTS)  # type: ignore[arg-type]
        winner = "tie" if abs(a - b) <= epsilon else "A" if a > b else "B"
        aggregates.append(Label(
            pair_id=pair_id, component="aggregate", winner=winner,
            alignment_a=a, alignment_b=b, status="accepted",
            source=components[COMPONENTS[0]].source,
            confidence=min((components[c].confidence for c in COMPONENTS), key=lambda x: {"": -1, "low": 0, "medium": 1, "high": 2}.get(x, -1)),
            reason="Equal-weight mean of accepted component ratings",
            order_consistent=True,
        ))
    return aggregates


def _kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _metric_scores(pair: Pair, level: str, metric: str) -> Optional[Tuple[float, float]]:
    if level in COMPONENTS:
        scores = pair.metrics.get(level, {}).get(metric)
        return (float(scores["a"]), float(scores["b"])) if scores else None
    parts = [pair.metrics.get(c, {}).get(metric) for c in COMPONENTS]
    if any(part is None for part in parts):
        return None
    return (
        statistics.fmean(float(part["a"]) for part in parts if part),
        statistics.fmean(float(part["b"]) for part in parts if part),
    )


def analyze(pairs: Sequence[Pair], labels: Sequence[Label], config: Config) -> Dict[str, Any]:
    all_labels = list(labels) + aggregate_labels(labels, config.aggregate_rating_epsilon)
    lookup = {(label.pair_id, label.component): label for label in all_labels if label.status == "accepted"}
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    metrics = sorted({name for pair in pairs for component in COMPONENTS for name in pair.metrics[component]})
    results: Dict[str, Any] = {}
    rng = random.Random(config.bootstrap_seed)
    for level in (*COMPONENTS, "aggregate"):
        results[level] = {}
        for metric in metrics:
            rows: List[Tuple[Pair, Label, float, float]] = []
            for pair in pairs:
                label = lookup.get((pair.pair_id, level))
                scores = _metric_scores(pair, level, metric)
                if label and scores:
                    rows.append((pair, label, scores[0], scores[1]))
            if not rows:
                continue
            directional = [row for row in rows if row[1].winner in {"A", "B"}]
            credits = [
                0.5 if abs(a - b) <= config.metric_tie_epsilon
                else 1.0 if ("A" if a > b else "B") == label.winner else 0.0
                for _, label, a, b in directional
            ]
            clusters: Dict[Tuple[str, str], List[Tuple[Pair, Label, float, float]]] = defaultdict(list)
            for row in directional:
                clusters[(row[0].category, row[0].topic)].append(row)
            cluster_values = list(clusters.values())
            boot: List[float] = []
            for _ in range(config.bootstrap_samples):
                sampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values] if cluster_values else []
                sample_rows = [row for cluster in sampled for row in cluster]
                sample_credits = [
                    0.5 if abs(a - b) <= config.metric_tie_epsilon
                    else 1.0 if ("A" if a > b else "B") == label.winner else 0.0
                    for _, label, a, b in sample_rows
                ]
                if sample_credits:
                    boot.append(statistics.fmean(sample_credits))
            results[level][metric] = {
                "pairwise_concordance": statistics.fmean(credits) if credits else None,
                "concordance_ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
                "kendall_tau_b": _kendall_tau_b(
                    [a - b for _, _, a, b in rows],
                    [float(label.alignment_a) - float(label.alignment_b) for _, label, _, _ in rows],
                ),
                "decision_coverage": sum(abs(a - b) > config.metric_tie_epsilon for _, _, a, b in rows) / len(rows) if rows else None,
                "n_available": len(rows),
                "n_directional": len(directional),
                "judge_ties": sum(label.winner == "tie" for _, label, _, _ in rows),
            }
    sources = Counter(label.source for label in labels)
    if "llm_proxy" in sources:
        evidence_status = "PIPELINE_VALIDATION_ONLY"
    elif set(sources) == {"human"}:
        evidence_status = "HUMAN_LABEL_ANALYSIS"
    else:
        evidence_status = "NON_HUMAN_TEST_ANALYSIS"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": evidence_status,
        "label_sources": dict(sources),
        "n_pairs": len(pairs),
        "component_status": {
            component: dict(Counter(label.status for label in labels if label.component == component))
            for component in COMPONENTS
        },
        "aggregate_labels": len(aggregate_labels(labels, config.aggregate_rating_epsilon)),
        "metrics": results,
        "warning": "LLM proxy labels are not human meta-evaluation evidence." if "llm_proxy" in sources else None,
    }


def write_report(report: Dict[str, Any], config: Config) -> None:
    output = Path(config.output_dir)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Multi-component alignment meta-evaluation", "", f"Evidence status: **{report['evidence_status']}**", ""]
    for level, metrics in report["metrics"].items():
        lines.extend([f"## {level.title()}", "", "| Metric | Concordance | 95% CI | Kendall tau-b | Coverage | n |", "|---|---:|---:|---:|---:|---:|"])
        for metric, values in metrics.items():
            f = lambda x: "NA" if x is None else f"{x:.3f}"
            ci = values["concordance_ci95"]
            lines.append(f"| {metric} | {f(values['pairwise_concordance'])} | [{f(ci[0])}, {f(ci[1])}] | {f(values['kendall_tau_b'])} | {f(values['decision_coverage'])} | {values['n_available']} |")
        lines.append("")
    if report.get("warning"):
        lines.append(f"> {report['warning']}")
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["prepare", "label", "metrics", "analyze", "all"], default="all")
    parser.add_argument("--labels")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        help="Label only the first N frozen pairs for a resumable smoke test",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = Config.from_json(args.config)
    if args.phase in {"prepare", "all"}:
        pairs, manifest = prepare(config)
        LOGGER.info("Prepared %d pairs: %s", len(pairs), manifest["category_counts"])
    else:
        pairs = load_pairs(config)
    if args.phase in {"label", "all"}:
        label_pairs = pairs[: args.limit_pairs] if args.limit_pairs else pairs
        run_labeling(label_pairs, config)
    if args.phase in {"metrics", "all"}:
        pairs = enrich_metrics(pairs, config)
        LOGGER.info("Saved additional local baselines for %d frozen pairs", len(pairs))
    if args.phase in {"analyze", "all"}:
        label_path = Path(args.labels) if args.labels else Path(config.output_dir) / "proxy_labels.jsonl"
        report = analyze(pairs, load_labels(label_path), config)
        write_report(report, config)
        LOGGER.info("Saved %s analysis", report["evidence_status"])


if __name__ == "__main__":
    main()
