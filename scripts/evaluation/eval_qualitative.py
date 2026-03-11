"""
Evaluation scaffold for survey quality using domain criteria and LLM scoring.

This module loads processed survey JSON (produced by data_processing_pipeline),
hydrates them into SurveyData objects, pulls domain-specific criteria
(outline/content/reference) from a configured directory, and builds prompts for
an LLM to return quantitative scores.

The goal is to provide a thin, inspectable framework; plug in your own prompt
template or LLM client credentials via config without changing code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure we can import pipeline classes when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing_pipeline import (  # type: ignore
    SurveyData,
    DataProcessingConfig,
    LLM_AVAILABLE,
    OpenAI,
    robust_json_parse,
)


# ----------------------------- Config objects ----------------------------- #


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    processed_dir: str = "results/processed"
    output_dir: str = "results/evaluation"
    criteria_base_dir: str = "outputs/criteria"
    criteria_filename: str = "merged_aspects.json"

    include_criteria_description: bool = True
    include_criteria_example: bool = True

    # When true, ask the model to score each aspect separately and average them.
    per_aspect_scoring: bool = False
    
    # When true, ask the model to score each criterion (rubric) separately and average them.
    # This provides the most granular scoring: aspect > criterion > overall
    per_criterion_scoring: bool = False
    
    # When true, use binary scoring (0 or 1) for each criterion and sum them for aspect score.
    # Each aspect is evaluated separately with one API call per aspect.
    binary_scoring: bool = False

    systems: Optional[List[str]] = None  # e.g., ["Gemini", "Qwen"]
    categories: Optional[List[str]] = None  # e.g., ["Biology"]
    max_files_per_category: Optional[int] = None

    llm_model: Optional[str] = None
    llm_temperature: float = 0.0
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None

    # Limit total characters from content sections included in prompt
    max_total_tokens_in_prompt: int = 32768

    # Which parts to evaluate
    eval_outline: bool = True
    eval_content: bool = True
    eval_reference: bool = True

    # Resume from a previous evaluation summary to skip already evaluated files
    resume_from: Optional[str] = None

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "EvaluationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ----------------------------- Helper classes ----------------------------- #


@dataclass
class CriteriaSet:
    """Domain criteria broken down by aspect."""

    outline: List[Dict[str, Any]]
    content: List[Dict[str, Any]]
    reference: List[Dict[str, Any]]

    @classmethod
    def load(
        cls,
        base_dir: str,
        category: str,
        filename: str,
        *,
        include_description: bool = True,
        include_example: bool = True,
    ) -> "CriteriaSet":
        path = Path(base_dir) / category / filename
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        def _filter(aspects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            filtered = []
            for aspect in aspects:
                criteria = []
                for criterion in aspect.get("expanded_criteria", []):
                    item = {"criterion_name": criterion.get("criterion_name")}
                    if include_description:
                        item["description"] = criterion.get("description")
                    if include_example:
                        item["example"] = criterion.get("example")
                    criteria.append(item)
                filtered.append(
                    {
                        "aspect_name": aspect.get("aspect_name"),
                        "expanded_criteria": criteria,
                    }
                )
            return filtered

        return cls(
            outline=_filter(payload.get("outline", [])),
            content=_filter(payload.get("content", [])),
            reference=_filter(payload.get("reference", [])),
        )


class QuantitativeEvaluator:
    """
    Basic evaluator that builds prompts using survey data + domain criteria
    and sends them to an LLM for scoring.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_client = self._init_llm_client()
        self.output_file: Optional[Path] = None
        self.previous_results: Dict[str, Any] = self._load_previous_results()

    def _init_llm_client(self):
        if not LLM_AVAILABLE:
            self.logger.warning("OpenAI client unavailable; LLM scoring disabled.")
            return None

        api_key = os.environ.get("API_KEY") or self.config.llm_api_key
        base_url = os.environ.get("BASE_URL") or self.config.llm_api_base

        if not api_key:
            self.logger.warning("No API key found; set API_KEY env or config.llm_api_key.")
            return None

        self.logger.info("LLM client initialized.")
        return OpenAI(api_key=api_key, base_url=base_url)

    def _load_previous_results(self) -> Dict[str, Any]:
        """Load previous evaluation results if resume_from is specified."""
        if not self.config.resume_from:
            return {"by_system": {}}

        resume_path = Path(self.config.resume_from)
        if not resume_path.exists():
            self.logger.warning(f"Resume file not found: {resume_path}, starting fresh.")
            return {"by_system": {}}

        try:
            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded previous results from {resume_path}")
            return data
        except Exception as exc:
            self.logger.warning(f"Failed to load resume file {resume_path}: {exc}")
            return {"by_system": {}}

    def _is_file_already_evaluated(self, file_path: Path, system: str, category: str) -> bool:
        """Check if a file has already been evaluated in previous results."""
        if not self.previous_results.get("by_system"):
            return False

        system_data = self.previous_results["by_system"].get(system, {})
        category_data = system_data.get(category, {})
        files_data = category_data.get("files", [])

        # Normalize file path for comparison
        file_str = str(file_path).replace("\\", "/")

        for entry in files_data:
            entry_file = entry.get("file", "").replace("\\", "/")
            if entry_file == file_str:
                # Check if it has valid scores
                scores = entry.get("scores", {})
                if scores and any(
                    scores.get(aspect, {}).get("score") is not None
                    for aspect in ["outline", "content", "reference"]
                ):
                    return True
        return False

    def _save_summary_incremental(self, summary: Dict[str, Any]) -> None:
        """Save the evaluation summary incrementally after each file."""
        if not self.output_file:
            return

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.error(f"Failed to save incremental summary: {exc}")

    # ----------------------------- I/O helpers ----------------------------- #

    def _load_survey(self, json_path: Path) -> SurveyData:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return SurveyData.from_dict(raw)

    def _trim_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Limit total characters from content sections included in the prompt.
        Sections are added in order until the cumulative content reaches the
        configured character budget. The last section may be truncated to fit.
        """
        budget_tokens = self.config.max_total_tokens_in_prompt
        budget = budget_tokens * 2
        used = 0
        trimmed = []

        for section in sections:
            if used >= budget:
                break

            content = section.get("content", "") or ""
            remaining = budget - used

            if len(content) > remaining:
                content = content[:remaining] + "... [truncated]"

            used += len(content)
            trimmed.append(
                {
                    "heading": section.get("heading"),
                    "level": section.get("level"),
                    "content": content,
                }
            )

        return trimmed

    # ----------------------------- Prompt & scoring ----------------------------- #

    def _build_prompt(
        self, survey: SurveyData, criteria: CriteriaSet, category: str, aspect: str
    ) -> str:
        if self.config.per_criterion_scoring:
            return self._build_prompt_per_criterion(survey, criteria, category, aspect)
        if self.config.per_aspect_scoring:
            return self._build_prompt_per_aspect(survey, criteria, category, aspect)

        stats = survey.get_statistics()
        outline = survey.outline.to_list()
        content = [c.to_dict() for c in survey.content.sections]
        references = [r.to_dict() for r in survey.references.entries]

        if aspect == "outline":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific evaluation criteria for OUTLINE (JSON):
{json.dumps(criteria.outline, ensure_ascii=False)}

Survey outline (cleaned):
{json.dumps(outline, ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the outline quality based on the criteria.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        if aspect == "content":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific evaluation criteria for CONTENT (JSON):
{json.dumps(criteria.content, ensure_ascii=False)}

Survey content sections (trimmed for length):
{json.dumps(self._trim_sections(content), ensure_ascii=False)}

Quick stats: {json.dumps(stats, ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the content quality based on the criteria.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        if aspect == "reference":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific evaluation criteria for REFERENCES (JSON):
{json.dumps(criteria.reference, ensure_ascii=False)}

Survey reference titles (cleaned):
{json.dumps([r.get('text') for r in references], ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the reference quality based on the criteria.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        raise ValueError(f"Unsupported aspect: {aspect}")

    def _build_prompt_per_aspect(
        self, survey: SurveyData, criteria: CriteriaSet, category: str, aspect: str
    ) -> str:
        """
        Build a prompt that asks the model to score each criterion aspect
        individually (1-5) and provide a note; the caller will average them.
        """
        stats = survey.get_statistics()
        outline = survey.outline.to_list()
        content = [c.to_dict() for c in survey.content.sections]
        references = [r.to_dict() for r in survey.references.entries]

        aspect_map = {
            "outline": criteria.outline,
            "content": criteria.content,
            "reference": criteria.reference,
        }
        if aspect not in aspect_map:
            raise ValueError(f"Unsupported aspect: {aspect}")

        criteria_payload = aspect_map[aspect]

        common_instructions = """
For EACH aspect below, give:
- an integer score from 1 to 5
- a short justification in one sentence

Return JSON exactly in this format:
{"aspects": [{"aspect_name": "<string>", "score": <int>, "notes": "<string>"}]}
"""

        if aspect == "outline":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific outline aspects and their criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey outline (cleaned):
{json.dumps(outline, ensure_ascii=False)}

{common_instructions}
"""

        if aspect == "content":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific content aspects and their criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey content sections (trimmed for length):
{json.dumps(self._trim_sections(content), ensure_ascii=False)}

Quick stats: {json.dumps(stats, ensure_ascii=False)}

{common_instructions}
"""

        if aspect == "reference":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific reference aspects and their criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey reference titles (cleaned):
{json.dumps([r.get('text') for r in references], ensure_ascii=False)}

{common_instructions}
"""

        raise ValueError(f"Unsupported aspect: {aspect}")

    def _build_prompt_per_criterion(
        self, survey: SurveyData, criteria: CriteriaSet, category: str, aspect: str
    ) -> str:
        """
        Build a prompt that asks the model to score each individual criterion (rubric)
        within each aspect, providing the most granular evaluation.
        """
        stats = survey.get_statistics()
        outline = survey.outline.to_list()
        content = [c.to_dict() for c in survey.content.sections]
        references = [r.to_dict() for r in survey.references.entries]

        aspect_map = {
            "outline": criteria.outline,
            "content": criteria.content,
            "reference": criteria.reference,
        }
        if aspect not in aspect_map:
            raise ValueError(f"Unsupported aspect: {aspect}")

        criteria_payload = aspect_map[aspect]

        common_instructions = """
For EACH criterion below (within each aspect), give:
- an integer score from 1 to 5
- a short justification in one sentence

Return JSON exactly in this format:
{
  "aspects": [
    {
      "aspect_name": "<string>",
      "criteria": [
        {"criterion_name": "<string>", "score": <int>, "notes": "<string>"}
      ]
    }
  ]
}
"""

        if aspect == "outline":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific outline aspects and their detailed criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey outline (cleaned):
{json.dumps(outline, ensure_ascii=False)}

{common_instructions}
"""

        if aspect == "content":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific content aspects and their detailed criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey content sections (trimmed for length):
{json.dumps(self._trim_sections(content), ensure_ascii=False)}

Quick stats: {json.dumps(stats, ensure_ascii=False)}

{common_instructions}
"""

        if aspect == "reference":
            return f"""
You are a domain reviewer for category: {category}.

Domain-specific reference aspects and their detailed criteria (JSON):
{json.dumps(criteria_payload, ensure_ascii=False)}

Survey reference titles (cleaned):
{json.dumps([r.get('text') for r in references], ensure_ascii=False)}

{common_instructions}
"""

        raise ValueError(f"Unsupported aspect: {aspect}")

    def _build_prompt_binary_criterion(
        self, survey: SurveyData, single_aspect: Dict[str, Any], category: str, aspect_type: str
    ) -> str:
        """
        Build a prompt for binary scoring (0 or 1) of a single aspect's criteria.
        Each criterion is evaluated independently and gets 0 (not fulfilled) or 1 (fulfilled).
        """
        stats = survey.get_statistics()
        outline = survey.outline.to_list()
        content = [c.to_dict() for c in survey.content.sections]
        references = [r.to_dict() for r in survey.references.entries]

        aspect_name = single_aspect.get("aspect_name", "Unknown Aspect")
        criteria = single_aspect.get("expanded_criteria", [])

        common_instructions = """
For EACH criterion below, evaluate whether it is fulfilled:
- Score 0 if the criterion is NOT met or only partially met
- Score 1 if the criterion is FULLY met
- Provide a brief explanation (1-2 sentences) justifying your decision

Return JSON exactly in this format:
{
  "aspect_name": "<string>",
  "criteria": [
    {
      "criterion_name": "<string>",
      "score": <0 or 1>,
      "fulfilled": <true or false>,
      "explanation": "<string explaining why the criterion is/isn't fulfilled>"
    }
  ]
}
"""

        if aspect_type == "outline":
            return f"""
You are a domain reviewer for category: {category}.

You are evaluating the aspect: "{aspect_name}"

Criteria to evaluate (with descriptions and examples):
{json.dumps(criteria, ensure_ascii=False, indent=2)}

Survey outline (cleaned):
{json.dumps(outline, ensure_ascii=False)}

{common_instructions}
"""

        if aspect_type == "content":
            return f"""
You are a domain reviewer for category: {category}.

You are evaluating the aspect: "{aspect_name}"

Criteria to evaluate (with descriptions and examples):
{json.dumps(criteria, ensure_ascii=False, indent=2)}

Survey content sections (trimmed for length):
{json.dumps(self._trim_sections(content), ensure_ascii=False)}

Quick stats: {json.dumps(stats, ensure_ascii=False)}

{common_instructions}
"""

        if aspect_type == "reference":
            return f"""
You are a domain reviewer for category: {category}.

You are evaluating the aspect: "{aspect_name}"

Criteria to evaluate (with descriptions and examples):
{json.dumps(criteria, ensure_ascii=False, indent=2)}

Survey reference titles (cleaned):
{json.dumps([r.get('text') for r in references], ensure_ascii=False)}

{common_instructions}
"""

        raise ValueError(f"Unsupported aspect type: {aspect_type}")

    def _score_with_llm(self, prompt: str) -> Dict[str, Any]:
        if not self.llm_client or not self.config.llm_model:
            return {"score": None, "notes": "LLM disabled"}

        completion = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm_temperature,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        parsed = robust_json_parse(raw)
        if isinstance(parsed, dict):
            result = dict(parsed)
            result["_raw_response"] = raw
            return result

        # Fallback: extract first numeric value if JSON parsing failed.
        match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
        if match:
            try:
                num = float(match.group(0))
                if num.is_integer():
                    num = int(num)
                return {
                    "score": num,
                    "notes": "parsed from raw numeric fallback",
                    "raw_excerpt": raw[:200],
                    "_raw_response": raw,
                }
            except ValueError:
                pass

        return {"score": None, "notes": f"parse_error: {raw[:200]}", "_raw_response": raw}

    def _score_aspects_with_llm(self, prompt: str, expected_scores: int) -> Dict[str, Any]:
        """
        Ask the LLM for per-aspect scores and compute the mean as the overall score.
        """
        base = self._score_with_llm(prompt)

        aspects = base.get("aspects") if isinstance(base, dict) else None
        numeric_scores = []
        if isinstance(aspects, list):
            # Filter out non-dict elements (LLM may return malformed items)
            aspects = [a for a in aspects if isinstance(a, dict)]
            numeric_scores = [
                a.get("score") for a in aspects if isinstance(a.get("score"), (int, float))
            ]
            if numeric_scores:
                overall = sum(numeric_scores) / len(numeric_scores)
                return {
                    "score": overall,
                    "notes": f"mean of {len(numeric_scores)} aspect scores",
                    "aspects": aspects,
                }

        raw_response = base.get("_raw_response") if isinstance(base, dict) else None
        fallback_avg = self._average_first_numeric_tokens(raw_response, expected_scores)
        if fallback_avg is not None:
            return {
                "score": fallback_avg,
                "notes": f"mean of first {expected_scores} raw numeric tokens",
                "raw_excerpt": raw_response[:200] if raw_response else None,
                "aspects": [],
            }

        return {
            "score": None,
            "notes": base.get("notes", "missing aspect scores")
            if isinstance(base, dict)
            else "missing aspect scores",
            "aspects": aspects if isinstance(aspects, list) else [],
        }

    def _score_criteria_with_llm(self, prompt: str, expected_criteria_count: int) -> Dict[str, Any]:
        """
        Ask the LLM for per-criterion scores within aspects and compute the mean as the overall score.
        This provides the most granular evaluation.
        """
        base = self._score_with_llm(prompt)

        aspects = base.get("aspects") if isinstance(base, dict) else None
        all_criterion_scores = []
        
        if isinstance(aspects, list):
            # Collect all criterion scores from all aspects
            for aspect_data in aspects:
                if not isinstance(aspect_data, dict):
                    continue
                criteria_list = aspect_data.get("criteria", [])
                if isinstance(criteria_list, list):
                    for criterion in criteria_list:
                        score = criterion.get("score") if isinstance(criterion, dict) else None
                        if isinstance(score, (int, float)):
                            all_criterion_scores.append(score)
            
            if all_criterion_scores:
                overall = sum(all_criterion_scores) / len(all_criterion_scores)
                return {
                    "score": overall,
                    "notes": f"mean of {len(all_criterion_scores)} criterion scores",
                    "aspects": aspects,
                }

        raw_response = base.get("_raw_response") if isinstance(base, dict) else None
        fallback_avg = self._average_first_numeric_tokens(raw_response, expected_criteria_count)
        if fallback_avg is not None:
            return {
                "score": fallback_avg,
                "notes": f"mean of first {expected_criteria_count} raw numeric tokens",
                "raw_excerpt": raw_response[:200] if raw_response else None,
                "aspects": [],
            }

        return {
            "score": None,
            "notes": base.get("notes", "missing criterion scores")
            if isinstance(base, dict)
            else "missing criterion scores",
            "aspects": aspects if isinstance(aspects, list) else [],
        }

    def _score_binary_criteria_with_llm(self, prompt: str, expected_criteria_count: int) -> Dict[str, Any]:
        """
        Ask the LLM for binary scores (0 or 1) for each criterion and sum them as the aspect score.
        Returns the sum (not average) of all criterion scores.
        """
        base = self._score_with_llm(prompt)

        aspect_name = base.get("aspect_name") if isinstance(base, dict) else None
        criteria = base.get("criteria") if isinstance(base, dict) else None
        
        if isinstance(criteria, list):
            binary_scores = []
            for criterion in criteria:
                if not isinstance(criterion, dict):
                    continue
                score = criterion.get("score")
                # Ensure score is 0 or 1
                if isinstance(score, (int, float)):
                    binary_score = 1 if score > 0.5 else 0
                    criterion["score"] = binary_score
                    criterion["fulfilled"] = binary_score == 1
                    binary_scores.append(binary_score)
            
            if binary_scores:
                total_score = sum(binary_scores)
                return {
                    "score": total_score,
                    "max_score": len(binary_scores),
                    "notes": f"sum of {len(binary_scores)} binary criterion scores (0 or 1 each)",
                    "aspect_name": aspect_name,
                    "criteria": criteria,
                }

        # Fallback: try to extract binary values
        raw_response = base.get("_raw_response") if isinstance(base, dict) else None
        if raw_response:
            matches = re.findall(r"\b[01]\b", raw_response)
            if len(matches) >= expected_criteria_count:
                binary_scores = [int(m) for m in matches[:expected_criteria_count]]
                return {
                    "score": sum(binary_scores),
                    "max_score": expected_criteria_count,
                    "notes": f"sum of {expected_criteria_count} binary scores from raw response",
                    "raw_excerpt": raw_response[:200],
                    "criteria": [],
                }

        return {
            "score": None,
            "max_score": expected_criteria_count,
            "notes": base.get("notes", "missing binary criterion scores")
            if isinstance(base, dict)
            else "missing binary criterion scores",
            "criteria": criteria if isinstance(criteria, list) else [],
        }

    def _average_first_numeric_tokens(
        self, raw_response: Optional[str], expected_count: int
    ) -> Optional[float]:
        if expected_count <= 0 or not raw_response:
            return None

        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", raw_response)
        if len(matches) < expected_count:
            return None

        numbers: List[float] = []
        for token in matches[:expected_count]:
            try:
                numbers.append(float(token))
            except ValueError:
                continue

        if len(numbers) < expected_count:
            return None

        average = sum(numbers) / expected_count
        if 0 <= average <= 5:
            return average

        return None

    def _evaluate_binary_aspect_group(
        self, survey: SurveyData, aspects: List[Dict[str, Any]], category: str, aspect_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate all aspects in a group (e.g., all content aspects) using binary scoring.
        Each aspect is evaluated separately with its own API call.
        Returns aggregated results with aspect-level and overall scores.
        """
        all_aspects_results = []
        total_score = 0
        total_max_score = 0

        for aspect in aspects:
            aspect_name = aspect.get("aspect_name", "Unknown")
            criteria_count = len(aspect.get("expanded_criteria", []))
            
            if criteria_count == 0:
                self.logger.warning(f"Aspect '{aspect_name}' has no criteria, skipping")
                continue

            try:
                # Build prompt for this single aspect
                prompt = self._build_prompt_binary_criterion(survey, aspect, category, aspect_type)
                
                # Score this aspect
                aspect_result = self._score_binary_criteria_with_llm(prompt, criteria_count)
                
                # Accumulate scores
                score = aspect_result.get("score")
                max_score = aspect_result.get("max_score", criteria_count)
                
                if isinstance(score, (int, float)):
                    total_score += score
                    total_max_score += max_score
                
                all_aspects_results.append(aspect_result)
                
                self.logger.info(
                    f"Evaluated aspect '{aspect_name}': {score}/{max_score} criteria fulfilled"
                )
                
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(f"Failed to evaluate aspect '{aspect_name}': {exc}")
                all_aspects_results.append({
                    "aspect_name": aspect_name,
                    "score": None,
                    "max_score": criteria_count,
                    "notes": f"error: {str(exc)}",
                    "criteria": [],
                })

        return {
            "score": total_score,
            "max_score": total_max_score,
            "notes": f"sum of binary scores across {len(all_aspects_results)} aspects",
            "aspects": all_aspects_results,
        }

    # ----------------------------- Public API ----------------------------- #

    def evaluate_file(self, file_path: Path, category: str) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Evaluating {file_path}")
        survey = self._load_survey(file_path)
        criteria = CriteriaSet.load(
            self.config.criteria_base_dir,
            category,
            self.config.criteria_filename,
            include_description=self.config.include_criteria_description,
            include_example=self.config.include_criteria_example,
        )

        # Count expected scores for aspects
        expected_aspects = {
            "outline": len(criteria.outline),
            "content": len(criteria.content),
            "reference": len(criteria.reference),
        }
        
        # Count expected scores for criteria (rubrics)
        expected_criteria = {
            "outline": sum(len(a.get("expanded_criteria", [])) for a in criteria.outline),
            "content": sum(len(a.get("expanded_criteria", [])) for a in criteria.content),
            "reference": sum(len(a.get("expanded_criteria", [])) for a in criteria.reference),
        }

        scores: Dict[str, Dict[str, Any]] = {}

        # Binary scoring mode: evaluate each aspect separately
        if self.config.binary_scoring:
            if self.config.eval_outline:
                scores["outline"] = self._evaluate_binary_aspect_group(
                    survey, criteria.outline, category, "outline"
                )
            else:
                scores["outline"] = {"score": None, "notes": "skipped by config"}

            if self.config.eval_content:
                scores["content"] = self._evaluate_binary_aspect_group(
                    survey, criteria.content, category, "content"
                )
            else:
                scores["content"] = {"score": None, "notes": "skipped by config"}

            if self.config.eval_reference:
                scores["reference"] = self._evaluate_binary_aspect_group(
                    survey, criteria.reference, category, "reference"
                )
            else:
                scores["reference"] = {"score": None, "notes": "skipped by config"}
        else:
            # Non-binary scoring modes
            if self.config.eval_outline:
                prompt = self._build_prompt(survey, criteria, category, "outline")
                if self.config.per_criterion_scoring:
                    scores["outline"] = self._score_criteria_with_llm(
                        prompt, expected_criteria["outline"]
                    )
                elif self.config.per_aspect_scoring:
                    scores["outline"] = self._score_aspects_with_llm(
                        prompt, expected_aspects["outline"]
                    )
                else:
                    scores["outline"] = self._score_with_llm(prompt)
            else:
                scores["outline"] = {"score": None, "notes": "skipped by config"}

            if self.config.eval_content:
                prompt = self._build_prompt(survey, criteria, category, "content")
                if self.config.per_criterion_scoring:
                    scores["content"] = self._score_criteria_with_llm(
                        prompt, expected_criteria["content"]
                    )
                elif self.config.per_aspect_scoring:
                    scores["content"] = self._score_aspects_with_llm(
                        prompt, expected_aspects["content"]
                    )
                else:
                    scores["content"] = self._score_with_llm(prompt)
            else:
                scores["content"] = {"score": None, "notes": "skipped by config"}

            if self.config.eval_reference:
                prompt = self._build_prompt(survey, criteria, category, "reference")
                if self.config.per_criterion_scoring:
                    scores["reference"] = self._score_criteria_with_llm(
                        prompt, expected_criteria["reference"]
                    )
                elif self.config.per_aspect_scoring:
                    scores["reference"] = self._score_aspects_with_llm(
                        prompt, expected_aspects["reference"]
                    )
                else:
                    scores["reference"] = self._score_with_llm(prompt)
            else:
                scores["reference"] = {"score": None, "notes": "skipped by config"}

        duration = time.time() - start_time
        
        return {
            "file": str(file_path),
            "category": category,
            "scores": scores,
            "prompt_tokens": None,  # can be filled when using usage info
            "duration_seconds": round(duration, 2),
        }

    def evaluate_category(
        self, system: str, category: str, summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        category_start_time = time.time()
        base = Path(self.config.processed_dir) / system / category
        if not base.exists():
            self.logger.warning(f"Category path missing: {base}")
            return []

        files = list(base.rglob("*_split.json"))
        if self.config.max_files_per_category:
            files = files[: self.config.max_files_per_category]

        # Load existing results from previous run if available
        existing_results = []
        if self.config.resume_from:
            system_data = self.previous_results.get("by_system", {}).get(system, {})
            category_data = system_data.get(category, {})
            existing_results = category_data.get("files", [])

        results = list(existing_results)  # Start with previous results

        for fp in files:
            # Skip if already evaluated
            if self._is_file_already_evaluated(fp, system, category):
                self.logger.info(f"Skipping already evaluated file: {fp}")
                continue

            try:
                result = self.evaluate_file(fp, category)
                results.append(result)

                # Update summary and save incrementally
                if system not in summary["by_system"]:
                    summary["by_system"][system] = {}
                
                category_duration = round(time.time() - category_start_time, 2)
                
                summary["by_system"][system][category] = {
                    "files": results,
                    "averages": self._compute_aspect_averages(results),
                    "total_duration_seconds": category_duration,
                }
                summary["total"] = sum(
                    len(cat_data.get("files", []))
                    for sys_data in summary["by_system"].values()
                    for cat_data in sys_data.values()
                )
                self._save_summary_incremental(summary)
                self.logger.info(f"Saved incremental results after evaluating {fp}")

            except Exception as exc:  # noqa: BLE001
                self.logger.exception(f"Failed to eval {fp}: {exc}")

        return results

    def evaluate(self) -> Dict[str, Any]:
        systems = (
            self.config.systems
            if self.config.systems is not None
            else DataProcessingConfig().get_systems(self.config.processed_dir)
        )

        # Initialize summary from previous results or start fresh
        if self.config.resume_from and self.previous_results.get("by_system"):
            summary: Dict[str, Any] = self.previous_results.copy()
            summary["total"] = sum(
                len(cat_data.get("files", []))
                for sys_data in summary.get("by_system", {}).values()
                for cat_data in sys_data.values()
            )
            self.logger.info("Resuming from previous results")
        else:
            summary = {"by_system": {}, "total": 0}

        # Set up output file for incremental saving
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary["generated_at"] = timestamp
        self.output_file = Path(self.config.output_dir) / f"evaluation_summary_{timestamp}.json"

        for system in systems:
            cats = (
                self.config.categories
                if self.config.categories is not None
                else DataProcessingConfig().get_categories_in_system(
                    system, self.config.processed_dir
                )
            )
            if system not in summary["by_system"]:
                summary["by_system"][system] = {}

            for cat in cats:
                cat_results = self.evaluate_category(system, cat, summary)
                # Results are already saved incrementally in evaluate_category

        # Final save with updated timestamp
        summary["generated_at"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved final evaluation summary to {self.output_file}")
        return summary

    @staticmethod
    def _compute_aspect_averages(
        files: List[Dict[str, Any]]
    ) -> Dict[str, Optional[float]]:
        totals: Dict[str, float] = {"outline": 0.0, "content": 0.0, "reference": 0.0}
        counts: Dict[str, int] = {"outline": 0, "content": 0, "reference": 0}
        for entry in files:
            scores = entry.get("scores", {})
            for aspect in totals:
                score = scores.get(aspect, {}).get("score")
                if isinstance(score, (int, float)):
                    totals[aspect] += score
                    counts[aspect] += 1

        averages: Dict[str, Optional[float]] = {}
        for aspect, total in totals.items():
            averages[aspect] = total / counts[aspect] if counts[aspect] else None
        return averages


# ----------------------------- CLI entrypoint ----------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation scaffold for processed surveys."
    )
    parser.add_argument("--config", help="Path to evaluation config JSON.")
    parser.add_argument("--save-config", help="Save a default config to path and exit.")
    parser.add_argument("--system", action="append", help="Systems to evaluate.")
    parser.add_argument("--category", action="append", help="Categories to evaluate.")
    parser.add_argument("--criteria-base", help="Override criteria base dir.")
    parser.add_argument("--processed-dir", help="Override processed dir.")
    parser.add_argument("--output-dir", help="Override output dir.")
    parser.add_argument("--model", help="LLM model.")
    parser.add_argument(
        "--resume-from",
        help="Resume from a previous evaluation summary JSON file, skipping already evaluated files.",
    )
    parser.add_argument(
        "--per-aspect",
        dest="per_aspect_scoring",
        action="store_true",
        default=None,
        help="Score each criterion aspect separately and average the result.",
    )
    parser.add_argument(
        "--per-criterion",
        dest="per_criterion_scoring",
        action="store_true",
        default=None,
        help="Score each individual criterion (rubric) separately and average the result. Most granular scoring level.",
    )
    parser.add_argument(
        "--binary",
        dest="binary_scoring",
        action="store_true",
        default=None,
        help="Use binary scoring (0 or 1) for each criterion. Each aspect evaluated separately. Scores are summed, not averaged.",
    )
    parser.add_argument(
        "--no-criteria-description",
        dest="include_criteria_description",
        action="store_false",
        default=None,
        help="Exclude description fields from criteria when building prompts.",
    )
    parser.add_argument(
        "--no-criteria-example",
        dest="include_criteria_example",
        action="store_false",
        default=None,
        help="Exclude example fields from criteria when building prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_config:
        cfg = EvaluationConfig()
        cfg.to_json(args.save_config)
        print(f"Default config written to {args.save_config}")
        return

    config = EvaluationConfig()
    if args.config:
        config = EvaluationConfig.from_json(args.config)

    if args.system:
        config.systems = args.system
    if args.category:
        config.categories = args.category
    if args.criteria_base:
        config.criteria_base_dir = args.criteria_base
    if args.processed_dir:
        config.processed_dir = args.processed_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model:
        config.llm_model = args.model
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.per_aspect_scoring is not None:
        config.per_aspect_scoring = args.per_aspect_scoring
    if args.per_criterion_scoring is not None:
        config.per_criterion_scoring = args.per_criterion_scoring
    if args.binary_scoring is not None:
        config.binary_scoring = args.binary_scoring
    if args.include_criteria_description is not None:
        config.include_criteria_description = args.include_criteria_description
    if args.include_criteria_example is not None:
        config.include_criteria_example = args.include_criteria_example

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = QuantitativeEvaluator(config)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

