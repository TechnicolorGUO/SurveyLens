"""
Bradley-Terry model for fitting aspect weights from pairwise preferences.

Given:
  - Pairwise comparisons (winner/loser) from preference evaluation
  - Per-survey aspect scores from qualitative evaluation

Fits weights w_k for each aspect such that:
  P(survey_i beats survey_j) = sigmoid( sum_k w_k * (score_i_k - score_j_k) )

This tells us which scoring aspects are most predictive of human/LLM preferences.

Key design: pairwise comparisons are per-component (outline / content / reference),
so we fit separate BT models for each component using only that component's
preference pairs and features:

  For each component (outline, content, reference):
    - Use only pairwise comparisons where aspect == component
    - Use only features (aspects / criteria) under that component
    - Fit BT weights per domain

Supports two feature levels within each component:
  - "aspect"    : N features (individual sub-aspect scores, e.g. 3-5 per component)
  - "criterion" : M features (individual rubric/criterion scores, e.g. ~15 per component)
  - "both"      : run aspect + criterion (default)

Usage:
  python scripts/evaluation/bt.py --config scripts/config/bt_config.json
"""

import json
import os
import sys
import argparse
import numpy as np
from scipy.optimize import minimize
from datetime import datetime


# ============================================================
# Data Loading
# ============================================================

def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_survey_id(filepath, category):
    """
    Extract a canonical survey ID from a file path.
    e.g. "results/processed/Human/Biology/Survey Name_split.json"
      -> "Biology/Survey Name"
    """
    basename = os.path.basename(filepath.replace('\\', '/'))
    for suffix in ['_split.json', '.json']:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    return f"{category}/{basename}"


def extract_domain_from_survey_id(survey_id):
    """
    Extract domain/category from survey ID.
    e.g. "Education/Survey Name" -> "Education"
    """
    if '/' in survey_id:
        return survey_id.split('/', 1)[0]
    return None


def clean_aspect_name(name):
    """
    Clean up LLM-generated aspect names that may contain trailing junk.
    e.g. "Critical Insight and Novelty', "  ->  "Critical Insight and Novelty"
    """
    return name.strip().rstrip("'\",; ").strip()


def _get_files_from_cat(cat_data):
    """Extract file list from category data (handles list / dict formats)."""
    if isinstance(cat_data, list):
        return cat_data
    elif isinstance(cat_data, dict):
        return cat_data.get('files', [])
    return []


def load_evaluation_scores(eval_summary_path, system, categories=None,
                           feature_groups=None, feature_level="aspect",
                           drop_missing=True):
    """
    Load scores from evaluation_summary JSON.

    feature_level:
      "group"     -> one score per group (outline/content/reference)
      "aspect"    -> one score per sub-aspect
      "criterion" -> one score per rubric criterion (finest granularity)

    drop_missing:
      If True, drop surveys that have null / 0 scores in any feature group.

    Returns:
      survey_scores : dict  {survey_id: {feature_name: score, ...}}
      feature_names : list  sorted feature names
      dropped       : list  survey_ids that were dropped due to missing scores
    """
    with open(eval_summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    by_system = data.get('by_system', {})
    if system not in by_system:
        raise ValueError(
            f"System '{system}' not found. Available: {list(by_system.keys())}"
        )

    system_data = by_system[system]
    if feature_groups is None:
        feature_groups = ['outline', 'content', 'reference']

    survey_scores = {}
    feature_names_set = set()

    for cat_name, cat_data in system_data.items():
        if categories and cat_name not in categories:
            continue

        for file_entry in _get_files_from_cat(cat_data):
            filepath = file_entry.get('file', '')
            category = file_entry.get('category', cat_name)
            survey_id = extract_survey_id(filepath, category)

            scores = {}
            has_missing = False

            for group in feature_groups:
                group_data = file_entry.get('scores', {}).get(group, None)
                if group_data is None:
                    has_missing = True
                    continue

                if feature_level == "group":
                    # --- Group-level: one score per group ---
                    fname = group
                    if isinstance(group_data, dict):
                        val = group_data.get('score', None)
                    elif isinstance(group_data, (int, float)):
                        val = group_data
                    else:
                        val = None

                    if val is None or (isinstance(val, (int, float)) and val == 0):
                        has_missing = True
                    scores[fname] = float(val) if val is not None else 0.0
                    feature_names_set.add(fname)

                elif feature_level == "criterion":
                    # --- Criterion-level: one score per rubric ---
                    group_has_valid = False
                    if isinstance(group_data, dict) and 'aspects' in group_data:
                        for asp in group_data['aspects']:
                            if not isinstance(asp, dict):
                                continue
                            asp_name = clean_aspect_name(
                                asp.get('aspect_name', ''))
                            if 'criteria' in asp:
                                for crit in asp['criteria']:
                                    crit_name = clean_aspect_name(
                                        crit.get('criterion_name', ''))
                                    fname = (f"{group}/{asp_name}"
                                             f"/{crit_name}")
                                    val = crit.get('score', None)
                                    val = float(val) if val is not None \
                                        else 0.0
                                    if fname in scores and \
                                            scores[fname] > 0:
                                        continue
                                    scores[fname] = val
                                    feature_names_set.add(fname)
                                    if val > 0:
                                        group_has_valid = True
                            else:
                                # No criteria -> fall back to aspect score
                                fname = f"{group}/{asp_name}"
                                val = float(asp.get('score') or 0)
                                if fname in scores and scores[fname] > 0:
                                    continue
                                scores[fname] = val
                                feature_names_set.add(fname)
                                if val > 0:
                                    group_has_valid = True

                    elif isinstance(group_data, dict) and 'score' in group_data:
                        fname = f"{group}/overall"
                        val = float(group_data['score']) \
                            if group_data['score'] is not None else 0.0
                        scores[fname] = val
                        feature_names_set.add(fname)
                        if val > 0:
                            group_has_valid = True

                    elif isinstance(group_data, (int, float)):
                        fname = f"{group}/overall"
                        scores[fname] = float(group_data)
                        feature_names_set.add(fname)
                        group_has_valid = True

                    if not group_has_valid:
                        has_missing = True

                else:
                    # --- Aspect-level: one score per sub-aspect ---
                    group_has_valid = False
                    if isinstance(group_data, dict) and 'aspects' in group_data:
                        for asp in group_data['aspects']:
                            if not isinstance(asp, dict):
                                continue
                            raw_name = asp.get('aspect_name', '')
                            fname = f"{group}/{clean_aspect_name(raw_name)}"
                            if 'criteria' in asp:
                                c_scores = [c['score'] for c in asp['criteria']
                                            if c.get('score') is not None]
                                val = float(np.mean(c_scores)) if c_scores else 0.0
                            else:
                                val = float(asp.get('score') or 0)
                            # Dedup: keep valid score over invalid
                            if fname in scores and scores[fname] > 0:
                                continue
                            scores[fname] = val
                            feature_names_set.add(fname)
                            if val > 0:
                                group_has_valid = True

                    elif isinstance(group_data, dict) and 'score' in group_data:
                        fname = f"{group}/overall"
                        val = float(group_data['score']) \
                            if group_data['score'] is not None else 0.0
                        scores[fname] = val
                        feature_names_set.add(fname)
                        if val > 0:
                            group_has_valid = True

                    elif isinstance(group_data, (int, float)):
                        fname = f"{group}/overall"
                        scores[fname] = float(group_data)
                        feature_names_set.add(fname)
                        group_has_valid = True

                    if not group_has_valid:
                        has_missing = True

            survey_scores[survey_id] = (scores, has_missing)

    # Separate valid / dropped surveys
    dropped = []
    clean_scores = {}
    for sid, (scores, has_missing) in survey_scores.items():
        if drop_missing and has_missing:
            dropped.append(sid)
        else:
            clean_scores[sid] = scores

    feature_names = sorted(feature_names_set)
    return clean_scores, feature_names, dropped


def load_criterion_grouped_scores(eval_summary_path, system, categories,
                                  component, drop_missing=True):
    """
    Load criterion-level scores grouped by parent aspect, using criterion
    *indices* (position within each survey's criterion list) rather than
    criterion names — this avoids name fragmentation across surveys.

    Returns:
      survey_criteria : {survey_id: {aspect_fname: [crit_score_0, ..., crit_score_K-1]}}
      aspect_feature_names : sorted aspect-level feature names (e.g. "outline/Formal Precision")
      min_K : minimum number of criteria across all (survey, aspect) combos
      dropped : list of dropped survey_ids
    """
    with open(eval_summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    by_system = data.get('by_system', {})
    if system not in by_system:
        raise ValueError(
            f"System '{system}' not found. Available: {list(by_system.keys())}"
        )

    system_data = by_system[system]
    survey_criteria = {}   # {survey_id: {aspect_fname: [scores]}}
    aspect_names_set = set()
    dropped = []
    criteria_counts = []    # track K per (survey, aspect)

    for cat_name, cat_data in system_data.items():
        if categories and cat_name not in categories:
            continue
        for file_entry in _get_files_from_cat(cat_data):
            filepath = file_entry.get('file', '')
            category = file_entry.get('category', cat_name)
            survey_id = extract_survey_id(filepath, category)

            group_data = file_entry.get('scores', {}).get(component, None)
            if group_data is None:
                if drop_missing:
                    dropped.append(survey_id)
                continue

            has_missing = False
            grouped = {}

            if isinstance(group_data, dict) and 'aspects' in group_data:
                for asp in group_data['aspects']:
                    if not isinstance(asp, dict):
                        continue
                    asp_name = clean_aspect_name(asp.get('aspect_name', ''))
                    asp_fname = f"{component}/{asp_name}"
                    aspect_names_set.add(asp_fname)

                    if 'criteria' in asp:
                        crit_vals = []
                        for crit in asp['criteria']:
                            val = crit.get('score', None)
                            crit_vals.append(float(val) if val is not None else 0.0)
                        grouped[asp_fname] = crit_vals
                        criteria_counts.append(len(crit_vals))
                        if not any(v > 0 for v in crit_vals):
                            has_missing = True
                    else:
                        val = float(asp.get('score') or 0)
                        grouped[asp_fname] = [val]
                        criteria_counts.append(1)
                        if val == 0:
                            has_missing = True
            else:
                has_missing = True

            if drop_missing and has_missing:
                dropped.append(survey_id)
            else:
                survey_criteria[survey_id] = grouped

    aspect_feature_names = sorted(aspect_names_set)
    min_K = min(criteria_counts) if criteria_counts else 0

    return survey_criteria, aspect_feature_names, min_K, dropped


def expand_by_criterion(survey_criteria, aspect_feature_names, pairs, K):
    """
    Criterion-expansion data augmentation for aspect-level BT fitting.

    For each original pair (A, B), create K augmented pairs (A_k, B_k)
    where k = 0..K-1. In augmented pair k, each aspect's score is
    its k-th criterion score (instead of the mean).

    Returns:
      augmented_scores : {virtual_survey_id: {aspect_fname: score}}
      augmented_pairs  : [(winner_virtual, loser_virtual), ...]
      expansion_K      : actual K used
    """
    # Build virtual surveys: each real survey -> K virtual surveys
    augmented_scores = {}
    sid_to_virtuals = {}

    for sid, grouped in survey_criteria.items():
        virtuals = []
        for k in range(K):
            virt_id = f"{sid}__aug{k}"
            virt_scores = {}
            for asp_fname in aspect_feature_names:
                crit_scores_list = grouped.get(asp_fname, [])
                if k < len(crit_scores_list):
                    virt_scores[asp_fname] = crit_scores_list[k]
                else:
                    # Wrap around for safety
                    virt_scores[asp_fname] = crit_scores_list[k % len(crit_scores_list)] \
                        if crit_scores_list else 0.0
            augmented_scores[virt_id] = virt_scores
            virtuals.append(virt_id)
        sid_to_virtuals[sid] = virtuals

    # Expand pairs: each original pair -> K augmented pairs
    augmented_pairs = []
    for winner, loser in pairs:
        w_virts = sid_to_virtuals.get(winner, [])
        l_virts = sid_to_virtuals.get(loser, [])
        n = min(K, len(w_virts), len(l_virts))
        for k in range(n):
            augmented_pairs.append((w_virts[k], l_virts[k]))

    return augmented_scores, augmented_pairs, K


def load_preference_pairs(pref_eval_path, preference_aspects=None,
                          categories=None):
    """
    Load pairwise comparison outcomes from preference evaluation JSON.

    Reads from both:
      - categories -> {cat: {comparisons: [...]}}   (round 1)
      - top-level comparisons list                   (round 2, double round-robin)

    Returns:
      pairs : list of (winner_survey_id, loser_survey_id)
    """
    with open(pref_eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = []

    def _process_comparison(comp):
        aspect = comp.get('aspect', '')
        if preference_aspects and aspect not in preference_aspects:
            return None
        survey_a = comp.get('survey_a') or comp.get('survey_a_id', '')
        survey_b = comp.get('survey_b') or comp.get('survey_b_id', '')
        winner = comp.get('winner', '').strip().upper()
        if winner == 'A':
            return (survey_a, survey_b)
        elif winner == 'B':
            return (survey_b, survey_a)
        return None

    # Round 1: inside categories
    for cat_name, cat_data in data.get('categories', {}).items():
        if categories and cat_name not in categories:
            continue
        for comp in cat_data.get('comparisons', []):
            result = _process_comparison(comp)
            if result:
                pairs.append(result)

    # Round 2: top-level comparisons (double round-robin)
    for comp in data.get('comparisons', []):
        if categories:
            sa = comp.get('survey_a') or comp.get('survey_a_id', '')
            cat = sa.split('/')[0] if '/' in sa else ''
            if cat not in categories:
                continue
        result = _process_comparison(comp)
        if result:
            pairs.append(result)

    return pairs


# ============================================================
# Bradley-Terry Core
# ============================================================

def build_feature_matrix(survey_scores, feature_names, survey_ids):
    """Build feature matrix X[n_surveys, n_features]."""
    n = len(survey_ids)
    d = len(feature_names)
    X = np.zeros((n, d))
    for i, sid in enumerate(survey_ids):
        scores = survey_scores.get(sid, {})
        for j, fname in enumerate(feature_names):
            X[i, j] = scores.get(fname, 0.0)
    return X


def negative_log_likelihood(weights, X, pair_indices, alpha=0.01):
    """Bradley-Terry NLL with L2 regularization."""
    scores = X @ weights
    nll = 0.0
    for w_idx, l_idx in pair_indices:
        diff = scores[w_idx] - scores[l_idx]
        nll += np.logaddexp(0, -diff)
    nll += alpha * np.sum(weights ** 2)
    return nll


def negative_log_likelihood_grad(weights, X, pair_indices, alpha=0.01):
    """Gradient of NLL."""
    scores = X @ weights
    grad = np.zeros_like(weights)
    for w_idx, l_idx in pair_indices:
        diff = scores[w_idx] - scores[l_idx]
        sig_neg = 1.0 / (1.0 + np.exp(diff))
        grad -= sig_neg * (X[w_idx] - X[l_idx])
    grad += 2.0 * alpha * weights
    return grad


def fit_bradley_terry(X, pair_indices, n_features, alpha=0.01, max_iter=1000):
    """Fit BT model via L-BFGS-B. Returns OptimizeResult."""
    initial_weights = np.ones(n_features) / n_features
    bounds = [(0, None)] * n_features
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_weights,
        args=(X, pair_indices, alpha),
        jac=negative_log_likelihood_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': max_iter}
    )
    return result


def compute_accuracy(X, weights, pair_indices):
    """Pairwise prediction accuracy."""
    scores = X @ weights
    correct = sum(1 for w, l in pair_indices if scores[w] > scores[l])
    return correct, len(pair_indices)


def compute_log_loss(X, weights, pair_indices):
    """Average log-loss per pair."""
    scores = X @ weights
    total = sum(
        np.logaddexp(0, -(scores[w] - scores[l]))
        for w, l in pair_indices
    )
    return total / len(pair_indices) if pair_indices else 0.0


# ============================================================
# Single-level fitting routine
# ============================================================

def run_fitting(survey_scores, feature_names, valid_pairs, alpha,
                max_iter, normalize, label=""):
    """
    Run one round of BT fitting.

    Returns dict with all results, or None if insufficient data.
    """
    # Build index
    all_sids = set()
    for w, l in valid_pairs:
        all_sids.add(w)
        all_sids.add(l)
    survey_ids = sorted(all_sids & set(survey_scores.keys()))

    pair_filtered = [(w, l) for w, l in valid_pairs
                     if w in survey_scores and l in survey_scores]

    if len(survey_ids) < 2 or len(pair_filtered) < 1:
        print(f"  [{label}] Insufficient data: "
              f"{len(survey_ids)} surveys, {len(pair_filtered)} pairs. Skipping.")
        return None

    id_to_idx = {sid: i for i, sid in enumerate(survey_ids)}
    X = build_feature_matrix(survey_scores, feature_names, survey_ids)
    pair_indices = [(id_to_idx[w], id_to_idx[l]) for w, l in pair_filtered]

    print(f"\n  [{label}] Feature matrix: {X.shape[0]} surveys x "
          f"{X.shape[1]} features, {len(pair_indices)} pairs")

    # Check feature variance and statistics
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    zero_var = [feature_names[j] for j in range(len(feature_names))
                if stds[j] < 1e-8]
    if zero_var:
        print(f"  [{label}] Warning: zero-variance features "
              f"(will get 0 weight): {zero_var}")
    
    # Check feature correlation (for pairs of features)
    if len(feature_names) > 1:
        corr_matrix = np.corrcoef(X.T)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.95:  # Very high correlation
                    high_corr_pairs.append(
                        (feature_names[i], feature_names[j], corr_matrix[i, j])
                    )
        if high_corr_pairs:
            print(f"  [{label}] Warning: highly correlated features "
                  f"(may cause one to get 0 weight):")
            for f1, f2, corr in high_corr_pairs[:5]:  # Show top 5
                print(f"    {f1} <-> {f2}: {corr:.3f}")

    # Fit
    result = fit_bradley_terry(X, pair_indices, len(feature_names),
                               alpha, max_iter)
    if not result.success:
        print(f"  [{label}] Warning: optimization did not converge: "
              f"{result.message}")

    raw_w = result.x
    w_sum = np.sum(raw_w)
    norm_w = raw_w / w_sum if (normalize and w_sum > 0) else raw_w
    
    # Diagnose zero weights
    zero_weight_features = [feature_names[j] for j in range(len(feature_names))
                           if abs(raw_w[j]) < 1e-6]
    if zero_weight_features:
        print(f"  [{label}] Features with zero/near-zero weights: {len(zero_weight_features)}")
        for fname in zero_weight_features[:10]:  # Show first 10
            idx = feature_names.index(fname)
            f_mean = means[idx]
            f_std = stds[idx]
            print(f"    - {fname}: mean={f_mean:.3f}, std={f_std:.3f}")
        if len(zero_weight_features) > 10:
            print(f"    ... ({len(zero_weight_features) - 10} more)")

    # Metrics
    correct, total = compute_accuracy(X, raw_w, pair_indices)
    accuracy = correct / total if total > 0 else 0.0
    avg_ll = compute_log_loss(X, raw_w, pair_indices)

    # Weight entries
    weight_entries = []
    for fname, rw, nw in zip(feature_names, raw_w, norm_w):
        weight_entries.append({
            'feature': fname,
            'raw_weight': float(rw),
            'normalized_weight': float(nw)
        })
    weight_entries.sort(key=lambda x: x['normalized_weight'], reverse=True)

    # Per-survey scores
    final_scores = X @ raw_w

    # Print
    print(f"\n  {'=' * 60}")
    print(f"  [{label}] RESULTS: Learned Weights")
    print(f"  {'=' * 60}")
    for rank, entry in enumerate(weight_entries, 1):
        bar = '#' * int(entry['normalized_weight'] * 50)
        print(f"  {rank:2d}. {entry['feature']:50s} "
              f"{entry['normalized_weight']:.4f}  {bar}")

    print(f"\n  Prediction accuracy : {correct}/{total} = {accuracy:.2%}")
    print(f"  Average log-loss    : {avg_ll:.4f}")

    print(f"\n  --- Per-Survey Weighted Scores ---")
    scored = sorted(
        [(sid, float(final_scores[id_to_idx[sid]])) for sid in survey_ids],
        key=lambda x: x[1], reverse=True
    )
    for rank, (sid, sc) in enumerate(scored, 1):
        print(f"  {rank:2d}. {sc:.4f}  {sid}")

    return {
        'level': label,
        'n_surveys': len(survey_ids),
        'n_pairs': len(pair_filtered),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'weights': weight_entries,
        'prediction_accuracy': accuracy,
        'average_log_loss': avg_ll,
        'optimization_success': bool(result.success),
        'optimization_message': str(getattr(result, 'message', '')),
        'survey_scores': [
            {'survey_id': sid, 'weighted_score': float(final_scores[id_to_idx[sid]])}
            for sid in survey_ids
        ]
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bradley-Terry aspect weight fitting'
    )
    parser.add_argument(
        '--config', type=str,
        default='scripts/config/bt_config.json',
        help='Path to config JSON file'
    )
    args = parser.parse_args()

    config = load_config(args.config)

    pref_eval_file = config['preference_eval_file']
    eval_summary_file = config['evaluation_summary_file']
    output_dir = config.get('output_dir', 'results/evaluation')
    system = config.get('system', 'Human')
    categories = config.get('categories', None)
    components = config.get('components',
                            ['outline', 'content', 'reference'])
    feature_level = config.get('feature_level', 'both')
    fitting_mode = config.get('fitting_mode', 'domain')  # "domain" | "global" | "both"
    drop_missing = config.get('drop_missing_scores', True)
    alpha = config.get('regularization_alpha', 0.01)
    normalize = config.get('normalize_weights', True)
    max_iter = config.get('max_iter', 1000)
    seed = config.get('random_seed', 42)
    augment = config.get('augment_with_criteria', False)

    np.random.seed(seed)

    print("=" * 65)
    print("  Bradley-Terry Aspect Weight Fitting (Per-Component)")
    print("=" * 65)
    print(f"  Preference eval  : {pref_eval_file}")
    print(f"  Eval summary     : {eval_summary_file}")
    print(f"  System           : {system}")
    print(f"  Categories       : {categories or 'all'}")
    print(f"  Components       : {components}")
    print(f"  Feature level    : {feature_level}")
    print(f"  Fitting mode     : {fitting_mode}")
    print(f"  Drop missing     : {drop_missing}")
    print(f"  Regularization   : {alpha}")
    print(f"  Max iterations   : {max_iter}")
    print(f"  Augment criteria : {augment}")
    print()

    # Determine which feature levels to run within each component
    levels_to_run = []
    if feature_level in ("aspect", "both", "all"):
        levels_to_run.append("aspect")
    if feature_level in ("criterion", "both", "all"):
        levels_to_run.append("criterion")
    if not levels_to_run:
        print(f"  [WARN] feature_level='{feature_level}' not recognised "
              f"for per-component mode. Defaulting to 'aspect'.")
        levels_to_run = ["aspect"]

    all_results = {}

    # ── Outer loop: iterate over components ──────────────────────
    for component in components:
        print(f"\n{'#' * 65}")
        print(f"# Component: {component.upper()}")
        print(f"{'#' * 65}")

        # Load preference pairs for *this component only*
        print(f"\n  [{component}] Loading preference pairs...")
        component_pairs = load_preference_pairs(
            pref_eval_file, preference_aspects=[component],
            categories=categories
        )
        print(f"  Loaded {len(component_pairs)} comparison pairs "
              f"for '{component}' (ties excluded)")

        if not component_pairs:
            print(f"  [WARN] No preference pairs for '{component}'. "
                  f"Skipping.")
            continue

        component_results = {}

        # ── Inner loop: feature granularity ──────────────────────
        for level in levels_to_run:
            print(f"\n  {'-' * 60}")
            print(f"  [{component}] Feature level: {level.upper()}")
            if augment and level == "aspect":
                print(f"  [{component}] *** Criterion-expansion augmentation ENABLED ***")
            print(f"  {'-' * 60}")

            # ----------------------------------------------------------
            # Load scores: with or without criterion-expansion augment
            # ----------------------------------------------------------
            use_augment = (augment and level == "aspect")
            aug_K = 0  # will be set if augmenting

            if use_augment:
                # Load criterion-level scores grouped by aspect
                survey_criteria, feature_names, aug_K, dropped = \
                    load_criterion_grouped_scores(
                        eval_summary_file, system, categories,
                        component, drop_missing=drop_missing
                    )
                print(f"  Loaded {len(survey_criteria)} surveys, "
                      f"{len(feature_names)} aspect features, "
                      f"K={aug_K} criteria per aspect")
                # Also build a plain survey_scores dict for domain grouping
                # (original survey IDs, before expansion)
                survey_scores = {}
                for sid, grouped in survey_criteria.items():
                    survey_scores[sid] = {
                        asp: np.mean(crit_list) if crit_list else 0.0
                        for asp, crit_list in grouped.items()
                    }
            else:
                # Standard loading (no augmentation)
                survey_scores, feature_names, dropped = \
                    load_evaluation_scores(
                        eval_summary_file, system, categories,
                        feature_groups=[component],
                        feature_level=level, drop_missing=drop_missing
                    )
                print(f"  Loaded {len(survey_scores)} surveys, "
                      f"{len(feature_names)} features")

            if dropped:
                print(f"  Dropped {len(dropped)} surveys with "
                      f"missing scores:")
                for d in dropped:
                    print(f"    - {d}")
            print(f"  Features:")
            if len(feature_names) <= 20:
                for i, f in enumerate(feature_names):
                    print(f"    [{i}] {f}")
            else:
                for i, f in enumerate(feature_names[:10]):
                    print(f"    [{i}] {f}")
                print(f"    ... ({len(feature_names) - 20} more) ...")
                for i, f in enumerate(feature_names[-10:],
                                      len(feature_names) - 10):
                    print(f"    [{i}] {f}")

            # Match surveys with preference pairs
            pref_sids = set()
            for w, l in component_pairs:
                pref_sids.add(w)
                pref_sids.add(l)
            common = pref_sids & set(survey_scores.keys())
            print(f"\n  Pref surveys: {len(pref_sids)}, "
                  f"Eval surveys: {len(survey_scores)}, "
                  f"Matched: {len(common)}")

            if not common:
                print(f"  [ERROR] No matching surveys. Skipping.")
                continue

            valid_pairs = [(w, l) for w, l in component_pairs
                           if w in common and l in common]
            print(f"  Valid pairs (original): {len(valid_pairs)}")

            # Group by domain (using original survey IDs)
            domain_to_surveys = {}
            domain_to_pairs = {}

            for sid in common:
                domain = extract_domain_from_survey_id(sid)
                if domain:
                    if domain not in domain_to_surveys:
                        domain_to_surveys[domain] = []
                    domain_to_surveys[domain].append(sid)

            for w, l in valid_pairs:
                w_domain = extract_domain_from_survey_id(w)
                l_domain = extract_domain_from_survey_id(l)
                if w_domain and l_domain and w_domain == l_domain:
                    if w_domain not in domain_to_pairs:
                        domain_to_pairs[w_domain] = []
                    domain_to_pairs[w_domain].append((w, l))

            print(f"\n  Found {len(domain_to_surveys)} domains:")
            for domain, sids in sorted(domain_to_surveys.items()):
                n_pairs = len(domain_to_pairs.get(domain, []))
                aug_info = f" -> {n_pairs * aug_K} augmented" \
                    if use_augment else ""
                print(f"    {domain}: {len(sids)} surveys, "
                      f"{n_pairs} pairs{aug_info}")

            level_results = {}

            # ── Global fitting ────────────────────────────────
            if fitting_mode in ("global", "both"):
                if use_augment:
                    # Expand surveys + pairs via criterion augmentation
                    global_criteria = {
                        sid: survey_criteria[sid]
                        for sid in common if sid in survey_criteria
                    }
                    aug_scores, aug_pairs, _ = expand_by_criterion(
                        global_criteria, feature_names,
                        valid_pairs, aug_K
                    )
                    fit_scores = aug_scores
                    fit_pairs = aug_pairs
                else:
                    fit_scores = {
                        sid: survey_scores[sid]
                        for sid in common if sid in survey_scores
                    }
                    fit_pairs = valid_pairs

                print(f"\n  {'=' * 60}")
                print(f"  [{component}/{level}] GLOBAL "
                      f"({len(fit_scores)} survey points, "
                      f"{len(fit_pairs)} pairs)")
                print(f"  {'=' * 60}")

                result = run_fitting(
                    fit_scores, feature_names,
                    fit_pairs, alpha, max_iter, normalize,
                    label=f"{component}/{level}/_global"
                )
                if result:
                    result['domain'] = '_global'
                    if use_augment:
                        result['augmented'] = True
                        result['augmentation_K'] = aug_K
                        result['n_original_pairs'] = len(valid_pairs)
                    level_results['_global'] = result

            # ── Per-domain fitting ────────────────────────────
            if fitting_mode in ("domain", "both"):
                for domain in sorted(domain_to_surveys.keys()):
                    domain_surveys = domain_to_surveys[domain]
                    domain_pairs = domain_to_pairs.get(domain, [])

                    if len(domain_surveys) < 2 \
                            or len(domain_pairs) < 1:
                        print(
                            f"\n  [{component}/{level}/{domain}] "
                            f"Skipping: insufficient data "
                            f"({len(domain_surveys)} surveys, "
                            f"{len(domain_pairs)} pairs)")
                        continue

                    if use_augment:
                        domain_criteria = {
                            sid: survey_criteria[sid]
                            for sid in domain_surveys
                            if sid in survey_criteria
                        }
                        aug_scores, aug_pairs, _ = expand_by_criterion(
                            domain_criteria, feature_names,
                            domain_pairs, aug_K
                        )
                        fit_scores = aug_scores
                        fit_pairs = aug_pairs
                    else:
                        fit_scores = {
                            sid: survey_scores[sid]
                            for sid in domain_surveys
                            if sid in survey_scores
                        }
                        fit_pairs = domain_pairs

                    print(f"\n  {'=' * 60}")
                    print(f"  [{component}/{level}] "
                          f"Domain: {domain}"
                          + (f"  (augmented: {len(fit_pairs)} pairs)"
                             if use_augment else ""))
                    print(f"  {'=' * 60}")

                    result = run_fitting(
                        fit_scores, feature_names,
                        fit_pairs, alpha, max_iter, normalize,
                        label=f"{component}/{level}/{domain}"
                    )
                    if result:
                        result['domain'] = domain
                        if use_augment:
                            result['augmented'] = True
                            result['augmentation_K'] = aug_K
                            result['n_original_pairs'] = len(domain_pairs)
                        level_results[domain] = result

            if level_results:
                component_results[level] = level_results

        if component_results:
            all_results[component] = component_results

    if not all_results:
        print("\n[ERROR] No results produced. Check data alignment.")
        sys.exit(1)

    # --- Save ---
    print(f"\n{'=' * 65}")
    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir,
                               f'bt_weights_{timestamp}.json')

    output_data = {
        'generated_at': timestamp,
        'config': config,
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_file}")
    print("\nDone.")


if __name__ == '__main__':
    main()
