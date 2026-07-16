"""Dual-axis proxy/human meta-evaluation on frozen SurveyLens pairs.

Each blinded call returns two construct-aligned judgments in one response:

* global_set_alignment: set-level coverage, precision, and redundancy; and
* local_correspondence: average best-entry semantic match strength while
  deliberately ignoring whether distinct human entries are collectively covered.

The script preserves the frozen pair selection and metric scores, supports A/B
order swapping and resumable labeling, and analyzes each axis independently.
LLM proxy labels are pipeline-validation artifacts, not human evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from eval_multicomponent_alignment_proxy import (
    CHOICES,
    COMPONENTS,
    Pair,
    _file_sha256,
    _kendall_tau_b,
    _load_env,
    _percentile,
    _response_text,
    _safe_float,
    extract_entries,
)


LOGGER = logging.getLogger("dual_axis_alignment_proxy")
AXES = ("global_set_alignment", "local_correspondence")


@dataclass
class Config:
    source_pairs_file: str
    output_dir: str = "results/dual_axis_alignment_proxy"
    components: Optional[List[str]] = None
    model: str = "gpt-5.6-luna"
    endpoint: str = "responses"
    api_key_env: str = "OHMYGPT_API_KEY"
    api_base_env: str = "OHMYGPT_BASE_URL"
    temperature: float = 0.0
    max_retries: int = 4
    request_timeout: float = 180.0
    order_repetitions: int = 2
    env_file: str = ".env"
    metric_tie_epsilon: float = 1e-12
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 42

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = ["reference"]
        invalid = set(self.components) - set(COMPONENTS)
        if invalid:
            raise ValueError(f"Unknown components: {sorted(invalid)}")
        if self.endpoint not in {"responses", "chat_completions"}:
            raise ValueError("endpoint must be responses or chat_completions")
        if self.order_repetitions not in {1, 2}:
            raise ValueError("order_repetitions must be 1 or 2")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass
class AxisLabel:
    pair_id: str
    component: str
    axis: str
    winner: str
    rating_a: Optional[float]
    rating_b: Optional[float]
    status: str
    source: str
    confidence: str = ""
    reason_codes: Optional[List[str]] = None
    reason: str = ""
    order_consistent: Optional[bool] = None
    model: str = ""
    annotator_id: str = ""

    def __post_init__(self) -> None:
        if self.reason_codes is None:
            self.reason_codes = []


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pairs(config: Config) -> List[Pair]:
    return [Pair(**row) for row in _read_jsonl(Path(config.source_pairs_file))]


def _verify_hashes(pair: Pair) -> None:
    expected = pair.file_hashes
    actual = {
        "reference": _file_sha256(pair.human_file),
        "a": _file_sha256(pair.file_a),
        "b": _file_sha256(pair.file_b),
    }
    if not expected or expected != actual:
        raise RuntimeError(
            f"Frozen files changed for pair {pair.pair_id}: "
            f"expected={expected}, actual={actual}"
        )


def prepare(pairs: Sequence[Pair], config: Config) -> Dict[str, Any]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": "dual-axis-alignment-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pairs_file": config.source_pairs_file,
        "n_pairs": len(pairs),
        "components": config.components,
        "axes": list(AXES),
        "calls_expected": len(pairs)
        * len(config.components or [])
        * config.order_repetitions,
        "warning": "LLM proxy labels are not human evidence.",
        "config": asdict(config),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "pair_id", "category", "topic", "component", "axis", "winner",
        "rating_a", "rating_b", "confidence", "reason_codes", "reason",
        "annotator_id",
    ]
    with open(output / "human_dual_axis_template.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            for component in config.components or []:
                for axis in AXES:
                    writer.writerow({
                        "pair_id": pair.pair_id,
                        "category": pair.category,
                        "topic": pair.topic,
                        "component": component,
                        "axis": axis,
                    })
    return manifest


def build_prompt(pair: Pair, component: str, swapped: bool) -> str:
    reference = extract_entries(pair.human_file)[component]
    left = extract_entries(pair.file_a)[component]
    right = extract_entries(pair.file_b)[component]
    if swapped:
        left, right = right, left
    component_note = {
        "outline": "Entries are outline headings; ignore prose style.",
        "content": "Entries are complete sections; ignore prose fluency.",
        "reference": "Entries are citation titles or raw bibliography entries.",
    }[component]
    data = {
        "protocol_version": "dual-axis-alignment-v1",
        "pair_id": pair.pair_id,
        "component": component,
        "human_reference": {"entries": reference},
        "candidate_a": {"entries": left},
        "candidate_b": {"entries": right},
    }
    return f"""You are a blinded annotator comparing two generated entry sets with one
human-reference entry set. {component_note} Judge two DISTINCT axes. Do not use system
identity, prestige, standalone writing quality, or list length by itself.

AXIS 1 — global_set_alignment:
Rate how well the candidate set as a whole covers DISTINCT human-reference information
with precision and without duplicate/near-duplicate entries. Many candidate entries
collapsing onto one human entry must receive a low global score even if each local match
is strong.

AXIS 2 — local_correspondence:
For each candidate entry, consider the semantic strength of its BEST matching human
entry, then judge the candidate's average best-match strength. Deliberately IGNORE
whether different candidate entries cover distinct human entries, and do not penalize
repetition or missing global coverage on this axis.

For each axis, rate A and B independently on the same 0-4 scale:
0=none, 1=weak, 2=partial/moderate, 3=strong, 4=near-complete/very strong.
Winner must agree with the two ratings. Use abstain only for malformed or genuinely
unjudgeable input; tie means equal ratings. Keep the two axes independent.

INPUT JSON:
{json.dumps(data, ensure_ascii=False)}

Return JSON only:
{{
  "global_set_alignment": {{
    "winner": "A|B|tie|abstain", "rating_a": 0, "rating_b": 0,
    "confidence": "high|medium|low",
    "reason_codes": ["coverage", "precision", "redundancy", "many_to_one"],
    "reason": "one concise sentence"
  }},
  "local_correspondence": {{
    "winner": "A|B|tie|abstain", "rating_a": 0, "rating_b": 0,
    "confidence": "high|medium|low",
    "reason_codes": ["best_match_strength", "specificity"],
    "reason": "one concise sentence"
  }}
}}"""


def _parse_axis(value: Any, axis: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{axis} must be an object")
    winner = str(value.get("winner", "")).strip()
    winner = winner if winner in {"tie", "abstain"} else winner.upper()
    if winner not in CHOICES:
        raise ValueError(f"Invalid {axis} winner {winner}")
    a = _safe_float(value.get("rating_a"))
    b = _safe_float(value.get("rating_b"))
    if a is None or b is None or not (0 <= a <= 4 and 0 <= b <= 4):
        raise ValueError(f"{axis} ratings must be numbers in [0,4]")
    expected = "A" if a > b else "B" if b > a else "tie"
    if winner != "abstain" and winner != expected:
        raise ValueError(f"{axis} winner {winner} conflicts with ratings {a}/{b}")
    confidence = str(value.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    codes = value.get("reason_codes", [])
    return {
        "winner": winner,
        "rating_a": a,
        "rating_b": b,
        "confidence": confidence,
        "reason_codes": [str(code) for code in codes] if isinstance(codes, list) else [],
        "reason": str(value.get("reason", "")).strip(),
    }


def parse_response(text: str) -> Dict[str, Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Response must be an object")
    return {axis: _parse_axis(value.get(axis), axis) for axis in AXES}


def normalize_swapped(parsed: Dict[str, Dict[str, Any]]) -> None:
    for axis in AXES:
        item = parsed[axis]
        item["winner"] = (
            "B" if item["winner"] == "A"
            else "A" if item["winner"] == "B"
            else item["winner"]
        )
        item["rating_a"], item["rating_b"] = item["rating_b"], item["rating_a"]


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
                    response = self.client.responses.create(
                        model=self.config.model, input=prompt
                    )
                    raw = _response_text(response)
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"},
                    )
                    raw = str(response.choices[0].message.content or "")
                parsed = parse_response(raw)
                if swapped:
                    normalize_swapped(parsed)
                return {
                    **parsed,
                    "raw": raw,
                    "swapped": swapped,
                    "model": self.config.model,
                }
            except Exception as exc:
                last = exc
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"Non-retryable API error: {exc}") from exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Judge failed for {pair.pair_id}/{component}: {last}")


def merge_axis_orders(
    pair_id: str,
    component: str,
    axis: str,
    rows: Sequence[Dict[str, Any]],
    model: str,
) -> AxisLabel:
    items = [row[axis] for row in rows]
    if any(item["winner"] == "abstain" for item in items):
        status, winner, consistent = "abstained", "abstain", False
    elif len({item["winner"] for item in items}) != 1:
        status, winner, consistent = "order_unstable", "abstain", False
    else:
        status, winner, consistent = "accepted", str(items[0]["winner"]), True
    rank = {"low": 0, "medium": 1, "high": 2}
    confidences = [str(item.get("confidence", "low")) for item in items]
    return AxisLabel(
        pair_id=pair_id,
        component=component,
        axis=axis,
        winner=winner,
        rating_a=(statistics.fmean(float(item["rating_a"]) for item in items) if status == "accepted" else None),
        rating_b=(statistics.fmean(float(item["rating_b"]) for item in items) if status == "accepted" else None),
        status=status,
        source="llm_proxy",
        confidence=min(confidences, key=lambda value: rank.get(value, 0)),
        reason_codes=sorted({code for item in items for code in item.get("reason_codes", [])}),
        reason=" | ".join(str(item.get("reason", "")) for item in items if item.get("reason")),
        order_consistent=consistent,
        model=model,
    )


def run_labeling(
    pairs: Sequence[Pair], config: Config
) -> List[AxisLabel]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "dual_axis_proxy_raw.jsonl"
    existing = _read_jsonl(raw_path)
    cache = {
        (row["pair_id"], row["component"], bool(row["swapped"])): row
        for row in existing
    }
    all_raw = list(existing)
    judge = Judge(config)
    labels: List[AxisLabel] = []
    tasks = [(pair, component) for pair in pairs for component in config.components or []]
    progress = tqdm(tasks, desc="Dual-axis proxy labeling", unit="component", dynamic_ncols=True)
    for pair, component in progress:
        _verify_hashes(pair)
        rows: List[Dict[str, Any]] = []
        hits = 0
        orders = [False, True] if config.order_repetitions == 2 else [False]
        for swapped in orders:
            key = (pair.pair_id, component, swapped)
            row = cache.get(key)
            if row is None:
                row = {
                    "pair_id": pair.pair_id,
                    "component": component,
                    **judge.call(pair, component, swapped),
                }
                cache[key] = row
                all_raw.append(row)
                _write_jsonl(raw_path, all_raw)
            else:
                hits += 1
            rows.append(row)
        axis_labels = [
            merge_axis_orders(pair.pair_id, component, axis, rows, config.model)
            for axis in AXES
        ]
        labels.extend(axis_labels)
        progress.set_postfix(
            component=component,
            global_status=axis_labels[0].status,
            local_status=axis_labels[1].status,
            cached=f"{hits}/{len(rows)}",
        )
    _write_jsonl(
        output / "dual_axis_proxy_labels.jsonl",
        (asdict(label) for label in labels),
    )
    return labels


def load_labels(path: Path) -> List[AxisLabel]:
    if path.suffix.lower() == ".jsonl":
        return [AxisLabel(**row) for row in _read_jsonl(path)]
    labels: List[AxisLabel] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            winner = str(row.get("winner") or "").strip()
            winner = winner if winner in {"tie", "abstain"} else winner.upper()
            a, b = _safe_float(row.get("rating_a")), _safe_float(row.get("rating_b"))
            axis = str(row.get("axis") or "")
            if winner not in CHOICES or a is None or b is None or axis not in AXES:
                continue
            labels.append(AxisLabel(
                pair_id=str(row["pair_id"]),
                component=str(row["component"]),
                axis=axis,
                winner=winner,
                rating_a=a,
                rating_b=b,
                status="accepted",
                source="human",
                confidence=str(row.get("confidence") or ""),
                reason_codes=[x.strip() for x in str(row.get("reason_codes") or "").split("|") if x.strip()],
                reason=str(row.get("reason") or ""),
                annotator_id=str(row.get("annotator_id") or ""),
            ))
    return labels


def _metric_scores(pair: Pair, component: str, metric: str) -> Optional[Tuple[float, float]]:
    values = pair.metrics.get(component, {}).get(metric)
    if not values:
        return None
    return float(values["a"]), float(values["b"])


def analyze(
    pairs: Sequence[Pair], labels: Sequence[AxisLabel], config: Config
) -> Dict[str, Any]:
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    accepted = {
        (label.pair_id, label.component, label.axis): label
        for label in labels
        if label.status == "accepted"
    }
    report_metrics: Dict[str, Any] = {}
    rng_seed = config.bootstrap_seed
    for axis in AXES:
        report_metrics[axis] = {}
        for component in config.components or []:
            names = sorted({
                name
                for pair in pairs
                for name in pair.metrics.get(component, {})
            })
            component_results: Dict[str, Any] = {}
            for metric_index, metric in enumerate(names):
                rows: List[Tuple[Pair, AxisLabel, float, float]] = []
                for pair in pairs:
                    label = accepted.get((pair.pair_id, component, axis))
                    scores = _metric_scores(pair, component, metric)
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
                clusters: Dict[Tuple[str, str], List[Tuple[Pair, AxisLabel, float, float]]] = defaultdict(list)
                for row in directional:
                    clusters[(row[0].category, row[0].topic)].append(row)
                cluster_values = list(clusters.values())
                import random
                rng = random.Random(rng_seed + metric_index)
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
                component_results[metric] = {
                    "pairwise_concordance": statistics.fmean(credits) if credits else None,
                    "concordance_ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
                    "kendall_tau_b": _kendall_tau_b(
                        [a - b for _, _, a, b in rows],
                        [float(label.rating_a) - float(label.rating_b) for _, label, _, _ in rows],
                    ),
                    "decision_coverage": sum(abs(a - b) > config.metric_tie_epsilon for _, _, a, b in rows) / len(rows),
                    "n_available": len(rows),
                    "n_directional": len(directional),
                    "judge_ties": sum(label.winner == "tie" for _, label, _, _ in rows),
                }
            report_metrics[axis][component] = component_results
    sources = Counter(label.source for label in labels)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "PIPELINE_VALIDATION_ONLY" if "llm_proxy" in sources else "HUMAN_LABEL_ANALYSIS",
        "label_sources": dict(sources),
        "n_pairs": len(pairs),
        "status": {
            axis: {
                component: dict(Counter(
                    label.status for label in labels
                    if label.axis == axis and label.component == component
                ))
                for component in config.components or []
            }
            for axis in AXES
        },
        "metrics": report_metrics,
        "warning": "LLM proxy labels are not human meta-evaluation evidence." if "llm_proxy" in sources else None,
    }


def write_report(report: Dict[str, Any], config: Config) -> None:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dual_axis_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Dual-axis alignment meta-evaluation",
        "",
        f"Evidence status: **{report['evidence_status']}**",
        "",
    ]
    for axis, components in report["metrics"].items():
        for component, metrics in components.items():
            lines.extend([
                f"## {axis} / {component}", "",
                "| Metric | Concordance | 95% CI | Kendall tau-b | Coverage | n |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for metric, values in metrics.items():
                f = lambda x: "NA" if x is None else f"{x:.3f}"
                ci = values["concordance_ci95"]
                lines.append(
                    f"| {metric} | {f(values['pairwise_concordance'])} | "
                    f"[{f(ci[0])}, {f(ci[1])}] | {f(values['kendall_tau_b'])} | "
                    f"{f(values['decision_coverage'])} | {values['n_available']} |"
                )
            lines.append("")
    if report.get("warning"):
        lines.append(f"> {report['warning']}")
    (output / "dual_axis_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["prepare", "label", "analyze", "all"], default="all")
    parser.add_argument("--labels")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = Config.from_json(args.config)
    pairs = load_pairs(config)
    if args.phase in {"prepare", "all"}:
        manifest = prepare(pairs, config)
        LOGGER.info("Prepared dual-axis protocol: %s", manifest)
    if args.phase in {"label", "all"}:
        run_labeling(pairs[: args.limit_pairs] if args.limit_pairs else pairs, config)
    if args.phase in {"analyze", "all"}:
        label_path = Path(args.labels) if args.labels else Path(config.output_dir) / "dual_axis_proxy_labels.jsonl"
        report = analyze(pairs, load_labels(label_path), config)
        write_report(report, config)
        LOGGER.info("Saved %s dual-axis analysis", report["evidence_status"])


if __name__ == "__main__":
    main()
