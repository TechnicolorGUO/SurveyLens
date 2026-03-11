"""
Preference-based evaluation with ELO scoring for survey quality comparison.

在单个系统（如Human）的每个领域内，对所有survey进行两两比较，
计算ELO评分用于排名。

例如：Human/Biology/ 下有10篇survey，进行 C(10,2)=45 次两两比较。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure we can import pipeline classes when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing_pipeline import (
    SurveyData,
    LLM_AVAILABLE,
    OpenAI,
    robust_json_parse,
)


# ----------------------------- Config ----------------------------- #


@dataclass
class PreferenceEvalConfig:
    """Configuration for preference-based evaluation."""

    # 输入路径，如 results/processed/Human
    input_dir: str = "results/processed/Human"
    output_dir: str = "results/evaluation"

    # 要评估的领域（None = 所有领域）
    categories: Optional[List[str]] = None

    # 是否双循环（A vs B 和 B vs A）
    double_round_robin: bool = True

    # 每个输入的最大token数（用于截断）
    max_tokens_per_input: int = 32768

    # 评估哪些维度
    eval_outline: bool = True
    eval_content: bool = True
    eval_reference: bool = True

    # ELO设置
    initial_elo: float = 1500.0
    k_factor: float = 32.0

    # LLM设置
    llm_model: Optional[str] = None
    llm_temperature: float = 0.0
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None

    # 从之前的评估结果恢复
    resume_from: Optional[str] = None

    # 随机种子
    random_seed: Optional[int] = 42

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "PreferenceEvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class SurveyEntry:
    """代表一个survey文件"""
    category: str
    file_path: Path
    survey_name: str
    survey_data: Optional[SurveyData] = None

    def get_id(self) -> str:
        """返回唯一标识符"""
        return f"{self.category}/{self.survey_name}"


@dataclass
class ComparisonResult:
    """一次比较的结果"""
    survey_a_id: str
    survey_b_id: str
    aspect: str  # "outline", "content", "reference"
    winner: str  # "A", "B", "tie"
    reason: str
    raw_response: Optional[str] = None


@dataclass
class EloRating:
    """ELO评分"""
    rating: float = 1500.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0


# ----------------------------- ELO Calculator ----------------------------- #


class EloCalculator:
    """计算和更新ELO评分"""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """计算A的期望得分"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        rating_a: EloRating,
        rating_b: EloRating,
        winner: str,
    ) -> Tuple[EloRating, EloRating]:
        """根据比赛结果更新ELO评分"""
        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1.0 - expected_a

        if winner == "A":
            actual_a, actual_b = 1.0, 0.0
            rating_a.wins += 1
            rating_b.losses += 1
        elif winner == "B":
            actual_a, actual_b = 0.0, 1.0
            rating_a.losses += 1
            rating_b.wins += 1
        else:  # tie
            actual_a, actual_b = 0.5, 0.5
            rating_a.ties += 1
            rating_b.ties += 1

        rating_a.rating += self.k_factor * (actual_a - expected_a)
        rating_b.rating += self.k_factor * (actual_b - expected_b)

        rating_a.games += 1
        rating_b.games += 1

        return rating_a, rating_b


# ----------------------------- Preference Evaluator ----------------------------- #


class PreferenceEvaluator:
    """在同一领域内对surveys进行两两比较并计算ELO评分"""

    def __init__(self, config: PreferenceEvalConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_client = self._init_llm_client()
        self.elo_calculator = EloCalculator(
            k_factor=config.k_factor,
            initial_rating=config.initial_elo,
        )

        if config.random_seed is not None:
            random.seed(config.random_seed)

        # 存储结构: category -> [SurveyEntry, ...]
        self.surveys: Dict[str, List[SurveyEntry]] = {}
        self.comparison_results: List[ComparisonResult] = []
        # ELO评分: aspect -> survey_id -> EloRating
        self.elo_ratings: Dict[str, Dict[str, EloRating]] = {
            "outline": {},
            "content": {},
            "reference": {},
        }

        self.previous_results = self._load_previous_results()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        if not LLM_AVAILABLE:
            self.logger.warning("OpenAI client unavailable; LLM comparisons disabled.")
            return None

        api_key = os.environ.get("API_KEY") or self.config.llm_api_key
        base_url = os.environ.get("BASE_URL") or self.config.llm_api_base

        # 从环境变量读取MODEL
        if not self.config.llm_model:
            self.config.llm_model = os.environ.get("MODEL")

        if not api_key:
            self.logger.warning("No API key found; set API_KEY env or config.llm_api_key.")
            return None

        self.logger.info(f"LLM client initialized. Model: {self.config.llm_model}")
        return OpenAI(api_key=api_key, base_url=base_url)

    def _load_previous_results(self) -> Dict[str, Any]:
        """加载之前的评估结果"""
        if not self.config.resume_from:
            return {}

        resume_path = Path(self.config.resume_from)
        if not resume_path.exists():
            self.logger.warning(f"Resume file not found: {resume_path}")
            return {}

        try:
            with open(resume_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded previous results from {resume_path}")

            # 恢复ELO评分
            for aspect in ["outline", "content", "reference"]:
                aspect_ratings = data.get("elo_ratings", {}).get(aspect, {})
                for survey_id, rating_data in aspect_ratings.items():
                    self.elo_ratings[aspect][survey_id] = EloRating(
                        rating=rating_data.get("rating", self.config.initial_elo),
                        games=rating_data.get("games", 0),
                        wins=rating_data.get("wins", 0),
                        losses=rating_data.get("losses", 0),
                        ties=rating_data.get("ties", 0),
                    )

            # 恢复比较结果
            for comp_data in data.get("comparisons", []):
                self.comparison_results.append(ComparisonResult(
                    survey_a_id=comp_data["survey_a_id"],
                    survey_b_id=comp_data["survey_b_id"],
                    aspect=comp_data["aspect"],
                    winner=comp_data["winner"],
                    reason=comp_data.get("reason", ""),
                ))

            return data
        except Exception as exc:
            self.logger.warning(f"Failed to load resume file: {exc}")
            return {}

    # ----------------------------- Survey Loading ----------------------------- #

    def _load_survey_data(self, json_path: Path) -> SurveyData:
        """从JSON文件加载survey数据"""
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return SurveyData.from_dict(raw)

    def _discover_surveys(self) -> None:
        """发现输入目录下的所有surveys"""
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.logger.info(f"Discovering surveys in: {input_dir}")

        # 获取所有领域目录
        for category_dir in input_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            # 如果指定了领域，只处理指定的
            if self.config.categories and category not in self.config.categories:
                continue

            self.surveys[category] = []

            # 查找所有 *_split.json 文件
            for json_file in category_dir.glob("*_split.json"):
                survey_name = json_file.stem.replace("_split", "")
                entry = SurveyEntry(
                    category=category,
                    file_path=json_file,
                    survey_name=survey_name,
                )
                self.surveys[category].append(entry)

            self.logger.info(f"Found {len(self.surveys[category])} surveys in {category}")

    # ----------------------------- Content Preparation ----------------------------- #

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """截断文本"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"

    def _prepare_outline_text(self, survey: SurveyData) -> str:
        """准备大纲文本"""
        outline_items = survey.outline.to_list()
        lines = []
        for level, title in outline_items:
            indent = "  " * (level - 1)
            lines.append(f"{indent}- {title}")
        return "\n".join(lines)

    def _prepare_content_text(self, survey: SurveyData) -> str:
        """准备内容文本（不截断，截断在 _build_comparison_prompt 中统一处理）"""
        sections = survey.content.sections
        text_parts = []
        for section in sections:
            section_text = f"## {section.heading}\n{section.content}"
            text_parts.append(section_text)
        return "\n\n".join(text_parts)

    def _prepare_reference_text(self, survey: SurveyData) -> str:
        """准备参考文献文本"""
        references = survey.references.entries
        lines = []
        for i, ref in enumerate(references, 1):
            lines.append(f"[{i}] {ref.text or ref.title}")
        return "\n".join(lines)

    # ----------------------------- Prompt Building ----------------------------- #

    def _build_comparison_prompt(
        self,
        survey_a: SurveyEntry,
        survey_b: SurveyEntry,
        aspect: str,
    ) -> str:
        """构建比较prompt"""
        # 加载survey数据
        if survey_a.survey_data is None:
            survey_a.survey_data = self._load_survey_data(survey_a.file_path)
        if survey_b.survey_data is None:
            survey_b.survey_data = self._load_survey_data(survey_b.file_path)

        # 根据维度准备文本
        if aspect == "outline":
            text_a = self._prepare_outline_text(survey_a.survey_data)
            text_b = self._prepare_outline_text(survey_b.survey_data)
            aspect_description = "survey outline structure"
            criteria = """
- Substantive Integrity: Evaluates the depth, breadth, and scholarly merit of the content. This aspect aggregates criteria related to topical coverage, relevance, and content value, ensuring the outline comprehensively addresses the subject matter, maintains strict topical alignment, and incorporates meaningful scholarly elements like research gaps and conceptual frameworks rather than mere enumeration.
- Structural Coherence: Assesses the logical architecture and organizational flow of the outline. This aspect synthesizes criteria regarding structure, structural organization, and logical organization, focusing on the adherence to field-appropriate schemas (e.g., IMRaD), the natural progression of topics, and the clarity of relationships between sections to facilitate information localization and coherent argumentation.
- Formal Precision: Examines the technical execution and refinement of the outline's hierarchy and presentation. This aspect encompasses descriptiveness, topic uniqueness, structural balance, and hierarchical consistency, ensuring precise section titling, distinct non-redundant content, balanced distribution of subsections, and clear parent-child relationships that align with academic conventions."""

        elif aspect == "content":
            text_a = self._prepare_content_text(survey_a.survey_data)
            text_b = self._prepare_content_text(survey_b.survey_data)
            aspect_description = "survey content quality"
            criteria = """
- Scope and Relevance: Evaluates the breadth and pertinence of the content, ensuring it comprehensively covers key subtopics, representative works, and temporal or geographic diversity while maintaining strict alignment with the central research theme and avoiding extraneous information.
- Structural Coherence: Assesses the logical organization and flow of the work, focusing on the hierarchy of sections, the smoothness of transitions, the progression of arguments, and the overall narrative consistency that binds disparate elements into a unified whole.
- Synthesis and Integration: Measures the ability to move beyond sequential summarization to construct a cohesive intellectual framework, characterized by the identification of cross-paper connections, the recognition of overarching patterns, and the depth of comparative analysis.
- Critical Insight and Novelty: Examines the depth of intellectual contribution, including the rigorous critique of existing methodologies and limitations, the generation of original taxonomies or frameworks, and the identification of significant research gaps and future directions.
- Scholarly Communication: Reviews the quality of expression and academic rigor, encompassing clarity, fluency, and conciseness of language, as well as the precision of terminology, veracity of claims, and adherence to formal citation and formatting standards."""

        elif aspect == "reference":
            text_a = self._prepare_reference_text(survey_a.survey_data)
            text_b = self._prepare_reference_text(survey_b.survey_data)
            aspect_description = "reference quality"
            criteria = """
- Bibliometric Comprehensiveness: Evaluates the extent and depth of the literature review, focusing on the quantitative coverage of the corpus, the inclusion of seminal works, and the balanced distribution of citations across relevant thematic clusters.
- Evidential Integrity: Assesses the reliability and precision of the scholarly discourse, ensuring that all claims are substantiated by appropriate evidence, generalizations are valid, and the attribution of ideas to their original authors is factually correct.
- Referential Pertinence and Compliance: Examines the quality of the bibliography by determining the thematic alignment of cited works with the research topic and verifying adherence to established academic standards regarding citation formatting, consistency, and completeness."""

        else:
            raise ValueError(f"Unknown aspect: {aspect}")

        # 截断
        max_chars = self.config.max_tokens_per_input * 2
        text_a = self._truncate_text(text_a, max_chars)
        text_b = self._truncate_text(text_b, max_chars)

        prompt = f"""You are an expert academic reviewer comparing two survey papers.

Your task is to evaluate the {aspect_description} and determine which survey is better.

Evaluation criteria:
{criteria}

=== Survey A: {survey_a.survey_name} ===
{text_a}

=== Survey B: {survey_b.survey_name} ===
{text_b}

Instructions:
1. Carefully analyze both surveys based on the criteria above.
2. Determine which survey has better {aspect_description}.
3. If both are roughly equal in quality, you may declare a tie.

Return your response in the following JSON format:
{{
    "winner": "<A, B, or tie>",
    "reason": "<Brief explanation of your decision in 1-2 sentences>"
}}

Important: The "winner" field must be exactly one of: "A", "B", or "tie"."""

        return prompt

    # ----------------------------- LLM Comparison ----------------------------- #

    def _compare_with_llm(self, prompt: str) -> Dict[str, Any]:
        """调用LLM进行比较"""
        if not self.llm_client or not self.config.llm_model:
            return {"winner": "tie", "reason": "LLM disabled", "raw": None}

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content
            parsed = robust_json_parse(raw)

            if isinstance(parsed, dict):
                winner = parsed.get("winner", "tie")
                if winner.upper() == "A":
                    winner = "A"
                elif winner.upper() == "B":
                    winner = "B"
                else:
                    winner = "tie"

                return {
                    "winner": winner,
                    "reason": parsed.get("reason", ""),
                    "raw": raw,
                }

            return {"winner": "tie", "reason": f"parse_error: {raw[:200]}", "raw": raw}

        except Exception as exc:
            self.logger.exception(f"LLM comparison failed: {exc}")
            return {"winner": "tie", "reason": f"error: {str(exc)}", "raw": None}

    def _is_comparison_done(
        self, survey_a_id: str, survey_b_id: str, aspect: str
    ) -> bool:
        """检查比较是否已完成"""
        for comp in self.comparison_results:
            if (
                comp.survey_a_id == survey_a_id
                and comp.survey_b_id == survey_b_id
                and comp.aspect == aspect
            ):
                return True
        return False

    # ----------------------------- Main Evaluation Logic ----------------------------- #

    def _get_or_create_elo(self, survey_id: str, aspect: str) -> EloRating:
        """获取或创建ELO评分"""
        if survey_id not in self.elo_ratings[aspect]:
            self.elo_ratings[aspect][survey_id] = EloRating(
                rating=self.config.initial_elo
            )
        return self.elo_ratings[aspect][survey_id]

    def _perform_comparison(
        self,
        survey_a: SurveyEntry,
        survey_b: SurveyEntry,
        aspect: str,
    ) -> ComparisonResult:
        """执行单次比较并更新ELO"""
        survey_a_id = survey_a.get_id()
        survey_b_id = survey_b.get_id()

        # 检查是否已完成
        if self._is_comparison_done(survey_a_id, survey_b_id, aspect):
            self.logger.debug(f"Skipping: {survey_a_id} vs {survey_b_id} ({aspect})")
            for comp in self.comparison_results:
                if (
                    comp.survey_a_id == survey_a_id
                    and comp.survey_b_id == survey_b_id
                    and comp.aspect == aspect
                ):
                    return comp
            return ComparisonResult(
                survey_a_id=survey_a_id,
                survey_b_id=survey_b_id,
                aspect=aspect,
                winner="tie",
                reason="previously completed",
            )

        # 随机交换 A/B 顺序以消除位置偏差
        swap_order = random.choice([True, False])
        if swap_order:
            prompt = self._build_comparison_prompt(survey_b, survey_a, aspect)
        else:
            prompt = self._build_comparison_prompt(survey_a, survey_b, aspect)
        
        result = self._compare_with_llm(prompt)
        
        # 如果交换了顺序，需要把结果映射回来
        if swap_order:
            if result["winner"] == "A":
                result["winner"] = "B"
            elif result["winner"] == "B":
                result["winner"] = "A"
            self.logger.debug(f"Swapped order, mapped result: {result['winner']}")

        # 创建比较结果
        comp_result = ComparisonResult(
            survey_a_id=survey_a_id,
            survey_b_id=survey_b_id,
            aspect=aspect,
            winner=result["winner"],
            reason=result["reason"],
            raw_response=result.get("raw"),
        )

        # 更新ELO评分
        rating_a = self._get_or_create_elo(survey_a_id, aspect)
        rating_b = self._get_or_create_elo(survey_b_id, aspect)
        self.elo_calculator.update_ratings(rating_a, rating_b, result["winner"])

        # 存储结果
        self.comparison_results.append(comp_result)

        swap_indicator = "(swapped)" if swap_order else ""
        self.logger.info(
            f"[{aspect}] {survey_a.survey_name[:30]} vs {survey_b.survey_name[:30]} -> {result['winner']} {swap_indicator}"
        )

        return comp_result

    def _generate_pairs(self, category: str) -> List[Tuple[SurveyEntry, SurveyEntry]]:
        """生成一个领域内的所有比较配对"""
        surveys = self.surveys.get(category, [])
        pairs = []

        # 生成所有两两组合
        for survey_a, survey_b in combinations(surveys, 2):
            pairs.append((survey_a, survey_b))
            # 双循环：交换顺序再比较一次
            if self.config.double_round_robin:
                pairs.append((survey_b, survey_a))

        # 随机打乱
        random.shuffle(pairs)
        return pairs

    def evaluate_category(self, category: str) -> Dict[str, Any]:
        """评估一个领域的所有surveys"""
        self.logger.info(f"=" * 50)
        self.logger.info(f"Evaluating category: {category}")
        start_time = time.time()

        pairs = self._generate_pairs(category)
        n_surveys = len(self.surveys.get(category, []))
        self.logger.info(f"Surveys: {n_surveys}, Pairs: {len(pairs)}")

        category_results = {
            "category": category,
            "n_surveys": n_surveys,
            "n_pairs": len(pairs),
            "comparisons": [],
        }

        aspects_to_eval = []
        if self.config.eval_outline:
            aspects_to_eval.append("outline")
        if self.config.eval_content:
            aspects_to_eval.append("content")
        if self.config.eval_reference:
            aspects_to_eval.append("reference")

        for i, (survey_a, survey_b) in enumerate(pairs):
            self.logger.info(f"Pair {i + 1}/{len(pairs)}")

            for aspect in aspects_to_eval:
                comp_result = self._perform_comparison(survey_a, survey_b, aspect)
                category_results["comparisons"].append({
                    "survey_a": survey_a.get_id(),
                    "survey_b": survey_b.get_id(),
                    "aspect": aspect,
                    "winner": comp_result.winner,
                    "reason": comp_result.reason,
                })

        category_results["duration_seconds"] = round(time.time() - start_time, 2)
        return category_results

    def evaluate(self) -> Dict[str, Any]:
        """运行完整评估"""
        self.logger.info("Starting preference evaluation")
        self.logger.info(f"Input directory: {self.config.input_dir}")

        # 发现surveys
        self._discover_surveys()

        if not self.surveys:
            self.logger.error("No surveys found!")
            return {}

        # 评估每个领域
        all_results = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "config": asdict(self.config),
            "categories": {},
            "comparisons": [],
            "elo_ratings": {},
        }

        for category in self.surveys.keys():
            cat_result = self.evaluate_category(category)
            all_results["categories"][category] = cat_result
            # 增量保存
            self._save_results(all_results)

        # 整理最终ELO评分
        for aspect in ["outline", "content", "reference"]:
            all_results["elo_ratings"][aspect] = {
                survey_id: {
                    "rating": rating.rating,
                    "games": rating.games,
                    "wins": rating.wins,
                    "losses": rating.losses,
                    "ties": rating.ties,
                }
                for survey_id, rating in self.elo_ratings[aspect].items()
            }

        # 存储所有比较结果
        all_results["comparisons"] = [
            {
                "survey_a_id": comp.survey_a_id,
                "survey_b_id": comp.survey_b_id,
                "aspect": comp.aspect,
                "winner": comp.winner,
                "reason": comp.reason,
            }
            for comp in self.comparison_results
        ]

        # 保存最终结果
        self._save_results(all_results)

        return all_results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存评估结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = results.get("generated_at", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        output_file = output_dir / f"preference_eval_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved results to {output_file}")

        # 保存ELO CSV
        self._save_elo_csv(results, output_dir, timestamp)

    def _save_elo_csv(
        self, results: Dict[str, Any], output_dir: Path, timestamp: str
    ) -> None:
        """保存ELO评分为CSV"""
        for aspect in ["outline", "content", "reference"]:
            ratings = results.get("elo_ratings", {}).get(aspect, {})
            if not ratings:
                continue

            csv_file = output_dir / f"elo_{aspect}_{timestamp}.csv"
            lines = ["survey_id,category,survey_name,elo,games,wins,losses,ties"]

            # 按ELO降序排列
            sorted_ratings = sorted(
                ratings.items(),
                key=lambda x: x[1].get("rating", 0),
                reverse=True,
            )

            for survey_id, rating_data in sorted_ratings:
                parts = survey_id.split("/")
                category = parts[0] if len(parts) > 0 else ""
                survey_name = parts[1] if len(parts) > 1 else survey_id

                lines.append(
                    f'"{survey_id}","{category}","{survey_name}",'
                    f'{rating_data.get("rating", 1500):.2f},'
                    f'{rating_data.get("games", 0)},'
                    f'{rating_data.get("wins", 0)},'
                    f'{rating_data.get("losses", 0)},'
                    f'{rating_data.get("ties", 0)}'
                )

            with open(csv_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self.logger.info(f"Saved ELO CSV to {csv_file}")


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在同一领域内对surveys进行两两比较并计算ELO评分"
    )
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--save-config", help="保存默认配置到指定路径")
    parser.add_argument("--input-dir", help="输入目录，如 results/processed/Human")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--category", action="append", help="要评估的领域（可多次指定）")
    parser.add_argument("--model", help="LLM模型名称")
    parser.add_argument("--max-tokens", type=int, help="每个输入的最大token数")
    parser.add_argument("--single-round", action="store_true", help="单循环（不交换顺序）")
    parser.add_argument("--resume-from", help="从之前的评估结果恢复")
    parser.add_argument("--k-factor", type=float, help="ELO K因子")
    parser.add_argument("--no-outline", action="store_true", help="跳过大纲评估")
    parser.add_argument("--no-content", action="store_true", help="跳过内容评估")
    parser.add_argument("--no-reference", action="store_true", help="跳过参考文献评估")
    parser.add_argument("--seed", type=int, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_config:
        cfg = PreferenceEvalConfig()
        cfg.to_json(args.save_config)
        print(f"Default config written to {args.save_config}")
        return

    config = PreferenceEvalConfig()
    if args.config:
        config = PreferenceEvalConfig.from_json(args.config)

    # 应用命令行参数覆盖
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.category:
        config.categories = args.category
    if args.model:
        config.llm_model = args.model
    if args.max_tokens:
        config.max_tokens_per_input = args.max_tokens
    if args.single_round:
        config.double_round_robin = False
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.k_factor:
        config.k_factor = args.k_factor
    if args.no_outline:
        config.eval_outline = False
    if args.no_content:
        config.eval_content = False
    if args.no_reference:
        config.eval_reference = False
    if args.seed is not None:
        config.random_seed = args.seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    evaluator = PreferenceEvaluator(config)
    results = evaluator.evaluate()

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)

    for aspect in ["outline", "content", "reference"]:
        ratings = results.get("elo_ratings", {}).get(aspect, {})
        if ratings:
            print(f"\n{aspect.upper()} ELO 排名:")
            sorted_ratings = sorted(
                ratings.items(),
                key=lambda x: x[1].get("rating", 0),
                reverse=True,
            )
            for i, (survey_id, rating) in enumerate(sorted_ratings, 1):
                print(
                    f"  {i}. {survey_id}: {rating.get('rating', 0):.2f} "
                    f"({rating.get('wins', 0)}W-{rating.get('losses', 0)}L-{rating.get('ties', 0)}T)"
                )

    print(f"\n总比较次数: {len(results.get('comparisons', []))}")
    print(f"结果保存至: {config.output_dir}")


if __name__ == "__main__":
    main()
