import importlib.util
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
MODULE_PATH = EVAL_DIR / "eval_reference_metric_stress.py"
SPEC = importlib.util.spec_from_file_location("eval_reference_metric_stress", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pair(metric, left, right, choice="a", annotator="ann"):
    return {
        "human_choice": choice,
        "annotator_id": annotator,
        "scores": {metric: {"a": left, "b": right}},
    }


def test_accuracy_at_coverage_orders_by_score_margin():
    metric = "plain_maxsim"
    pairs = [
        _pair(metric, 0.9, 0.1, "a"),
        _pair(metric, 0.6, 0.5, "b"),
    ]
    top_half = MODULE._accuracy_at_coverage(pairs, metric, 0.5)
    assert top_half["n"] == 1
    assert top_half["accuracy_with_metric_ties_as_half"] == 1.0


def test_stress_summary_reports_direction_and_monotonicity():
    records = []
    for severity, attacked in [(2.0, 0.8), (4.0, 0.7), (8.0, 0.6)]:
        base = {metric: 0.9 for metric in MODULE.STRESS_METRICS}
        scores = {metric: attacked for metric in MODULE.STRESS_METRICS}
        records.append(
            {
                "generated_file": "survey.json",
                "attack": "duplicate_best_aligned_reference",
                "severity": severity,
                "score_deltas": {
                    metric: scores[metric] - base[metric]
                    for metric in MODULE.STRESS_METRICS
                },
                "attacked_scores": scores,
            }
        )
    report = MODULE.summarize_stress(records)
    metric = "same_backbone_ra_align_f1"
    assert (
        report["by_condition"]["duplicate_best_aligned_reference"]["2.0"][metric][
            "decreased_fraction"
        ]
        == 1.0
    )
    assert (
        report["monotonicity"]["duplicate_best_aligned_reference"][metric][
            "nonincreasing_fraction"
        ]
        == 1.0
    )
