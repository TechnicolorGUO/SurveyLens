from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure we can import pipeline classes when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing_pipeline import (  # type: ignore
    SurveyData,
    DataProcessingConfig,
)
from eval_qualitative import EvaluationConfig, QuantitativeEvaluator

CRITERIA = {
    'Coverage': {
        'description': 'Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics.',
        'score 1': 'The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.',
        'score 2': 'The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.',
        'score 3': 'The survey is generally comprehensive in coverage but still misses a few key points that are not fully discussed.',
        'score 4': 'The survey covers most key areas of the topic comprehensively, with only very minor topics left out.',
        'score 5': 'The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information.'
    },
    'Structure': {
        'description': 'Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected.',
        'score 1': 'The survey lacks logic, with no clear connections between sections, making it difficult to understand the overall framework.',
        'score 2': 'The survey has weak logical flow with some content arranged in a disordered or unreasonable manner.',
        'score 3': 'The survey has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections.',
        'score 4': 'The survey has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts.',
        'score 5': 'The survey is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adjacent sections smooth without redundancy.'
    },
    'Relevance': {
        'description': 'Relevance measures how well the content of the survey aligns with the research topic and maintains a clear focus.',
        'score 1': 'The content is outdated or unrelated to the field it purports to review, offering no alignment with the topic.',
        'score 2': 'The survey is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.',
        'score 3': 'The survey is generally on topic, despite a few unrelated details.',
        'score 4': 'The survey is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.',
        'score 5': 'The survey is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic.'
    },
    'Language': {
        'description': 'Language assesses the academic formality, clarity, and correctness of the writing, including grammar, terminology, and tone.',
        'score 1': 'The language is highly informal, contains frequent grammatical errors, imprecise terminology, and numerous colloquial expressions. The writing lacks academic tone and professionalism.',
        'score 2': 'The writing style is somewhat informal, with several grammatical errors or ambiguous expressions. Academic terminology is inconsistently used.',
        'score 3': 'The language is mostly formal and generally clear, with only occasional minor grammatical issues or slightly informal phrasing.',
        'score 4': 'The language is clear, formal, and mostly error-free, with only rare lapses in academic tone or minor imprecisions.',
        'score 5': 'The writing is exemplary in academic formality and clarity, using precise terminology throughout, flawless grammar, and a consistently scholarly tone.'
    },
    'Criticalness': {
        'description': 'Criticalness evaluates the depth of critical analysis, the originality of insights, and the clarity and justification of proposed future research directions.',
        'score 1': 'The survey lacks critical analysis and fails to identify gaps, weaknesses, or areas for improvement. Offers no original insights and does not propose any meaningful future research directions.',
        'score 2': 'The survey provides only superficial critique, with limited identification of weaknesses. Original insights are minimal and future directions are vague or generic.',
        'score 3': 'The survey demonstrates moderate critical analysis, identifying some gaps or weaknesses. Offers some original perspectives and suggests future directions, though they may lack depth or specificity.',
        'score 4': 'The survey presents a strong critique, clearly identifying significant gaps and weaknesses, and proposes well-justified future research directions. Provides some novel insights, though a few aspects could be further developed.',
        'score 5': 'The survey excels in critical analysis, incisively evaluating methodologies, results, and assumptions. Provides highly original insights and proposes clear, actionable, and innovative future research directions, all rigorously justified.'
    },
    'Outline': {
        'description': (
            'Outline evaluates the clarity, logical hierarchy, and organization of the survey structure based on its section titles. '
            'Note: The outline is now provided as a plain list of section titles'
            'Please focus your evaluation on the semantic coherence, logical grouping, and progression reflected by the section titles themselves.'
        ),
        'score 1': 'The outline is chaotic or confusing, with unclear relationships and significant structural gaps. Section titles are vague, repetitive, or lack logical flow.',
        'score 2': 'The outline shows basic attempts at organization but contains multiple misplaced or poorly grouped sections. The progression is unclear or disjointed. Section titles are sometimes ambiguous.',
        'score 3': 'The outline demonstrates a generally reasonable structure, with some minor misalignments or grouping issues. Most section titles are clear, and topic coverage is mostly logical.',
        'score 4': 'The outline is well-structured, with clearly grouped section titles and a coherent progression of topics. Minor issues may exist but do not significantly affect readability or understanding.',
        'score 5': 'The outline is exceptionally clear, logically organized, and easy to follow. Section titles are concise and informative, and the structure fully represents the topic\'s breadth and depth.'
    },
    "Reference": {
        "description": (
            "Reference relevance evaluates whether the references listed in the References section are closely related to the survey's topic. "
            "A high-quality References section should primarily include publications, articles, or works that are directly relevant to the subject matter. "
            "The score depends on the proportion of irrelevant or tangential entries as identified by the model. "
            "Additionally, the formatting of the references should adhere to standard citation guidelines (e.g., APA, MLA, Chicago), ensuring consistency, accuracy, and completeness. "
            "Poor formatting, missing information, or inconsistencies in style will negatively impact the score."
        ),
        "score 1": "Most references (over 60%) are irrelevant or only marginally related to the topic and/or the references are poorly formatted, with significant inconsistencies or missing details.",
        "score 2": "A significant portion (40-60%) of references are not closely related to the topic and/or the references show notable formatting issues, such as missing key information or inconsistent citation styles.",
        "score 3": "Some references (20-40%) are not relevant to the topic, but the majority are appropriate. Formatting may have minor issues, but does not significantly detract from the overall quality.",
        "score 4": "A small number (5-20%) of references are not well aligned, but most are relevant. The formatting is mostly consistent, with only occasional minor errors.",
        "score 5": "Nearly all references (over 95%) are relevant and directly related to the topic. The formatting is consistent, accurate, and adheres to standard citation guidelines."
    }
}


class AblationEvaluator(QuantitativeEvaluator):
    """
    Evaluator that reuses the qualitative evaluation flow but uses
    the ablation rubric prompts defined in this file.
    """

    def _format_rubric(self, name: str, rubric: Dict[str, Any]) -> str:
        lines = [f"{name}:", f"- Description: {rubric.get('description', '')}"]
        for score in range(1, 6):
            key = f"score {score}"
            if key in rubric:
                lines.append(f"- {key}: {rubric[key]}")
        return "\n".join(lines)

    def _build_prompt_ablation(
        self, survey: SurveyData, category: str, aspect: str
    ) -> str:
        stats = survey.get_statistics()
        outline = survey.outline.to_list()
        content = [c.to_dict() for c in survey.content.sections]
        references = [r.to_dict() for r in survey.references.entries]

        if aspect == "outline":
            rubric = self._format_rubric("Outline", CRITERIA["Outline"])
            return f"""
You are a domain reviewer for category: {category}.

Rubric for OUTLINE:
{rubric}

Survey outline (cleaned):
{json.dumps(outline, ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the outline quality based on the rubric.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        if aspect == "content":
            rubric = "\n\n".join(
                self._format_rubric(name, CRITERIA[name])
                for name in ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
            )
            return f"""
You are a domain reviewer for category: {category}.

Rubrics for CONTENT:
{rubric}

Survey content sections (trimmed for length):
{json.dumps(self._trim_sections(content), ensure_ascii=False)}

Quick stats: {json.dumps(stats, ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the content quality based on the rubrics.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        if aspect == "reference":
            rubric = self._format_rubric("Reference", CRITERIA["Reference"])
            return f"""
You are a domain reviewer for category: {category}.

Rubric for REFERENCES:
{rubric}

Survey reference titles (cleaned):
{json.dumps([r.get('text') for r in references], ensure_ascii=False)}

Scoring request:
- Provide a 1-5 score for the reference quality based on the rubric.
- Add a short justification.
- Return JSON exactly in this format:
{{"score": <int>, "notes": "<string>"}}
"""

        raise ValueError(f"Unsupported aspect: {aspect}")

    def evaluate_file(self, file_path: Path, category: str) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Evaluating {file_path}")
        survey = self._load_survey(file_path)

        if self.config.per_aspect_scoring or self.config.per_criterion_scoring or self.config.binary_scoring:
            self.logger.warning(
                "Ablation evaluator ignores per-aspect/per-criterion/binary flags; "
                "using single-score rubric prompts."
            )

        scores: Dict[str, Dict[str, Any]] = {}

        if self.config.eval_outline:
            prompt = self._build_prompt_ablation(survey, category, "outline")
            scores["outline"] = self._score_with_llm(prompt)
        else:
            scores["outline"] = {"score": None, "notes": "skipped by config"}

        if self.config.eval_content:
            prompt = self._build_prompt_ablation(survey, category, "content")
            scores["content"] = self._score_with_llm(prompt)
        else:
            scores["content"] = {"score": None, "notes": "skipped by config"}

        if self.config.eval_reference:
            prompt = self._build_prompt_ablation(survey, category, "reference")
            scores["reference"] = self._score_with_llm(prompt)
        else:
            scores["reference"] = {"score": None, "notes": "skipped by config"}

        duration = time.time() - start_time
        return {
            "file": str(file_path),
            "category": category,
            "scores": scores,
            "prompt_tokens": None,
            "duration_seconds": round(duration, 2),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation evaluation scaffold using fixed rubric prompts."
    )
    parser.add_argument("--config", help="Path to evaluation config JSON.")
    parser.add_argument("--save-config", help="Save a default config to path and exit.")
    parser.add_argument("--system", action="append", help="Systems to evaluate.")
    parser.add_argument("--category", action="append", help="Categories to evaluate.")
    parser.add_argument("--processed-dir", help="Override processed dir.")
    parser.add_argument("--output-dir", help="Override output dir.")
    parser.add_argument("--model", help="LLM model.")
    parser.add_argument(
        "--resume-from",
        help="Resume from a previous evaluation summary JSON file, skipping already evaluated files.",
    )
    return parser.parse_args()


def main() -> None:
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
    if args.processed_dir:
        config.processed_dir = args.processed_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model:
        config.llm_model = args.model
    if args.resume_from:
        config.resume_from = args.resume_from

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = AblationEvaluator(config)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
