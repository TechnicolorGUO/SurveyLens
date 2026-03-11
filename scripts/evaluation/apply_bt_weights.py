"""
Apply BT-learned weights to rescore evaluation summaries.

Workflow:
  1. bt.py learns weights from Human survey preferences
     -> "what humans actually care about when judging surveys"
  2. This script applies those weights to ANY system's eval scores
     -> produces a rescored JSON with original (equal-weight) and
        BT-weighted scores side by side.

Usage:
  python scripts/evaluation/apply_bt_weights.py \
      --config scripts/config/apply_bt_weights_config.json
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime


# ============================================================
# Helpers (shared with bt.py)
# ============================================================

def clean_aspect_name(name):
    """Clean LLM-generated aspect names with trailing junk."""
    return name.strip().rstrip("'\",; ").strip()


def normalize_aspect_name(aspect_name):
    """
    Normalize aspect names to handle common variations.
    
    Maps known variations to canonical names.
    """
    # Define mappings for known variations
    name_mappings = {
        'Scope and Reliance': 'Scope and Relevance',
        # Add more mappings as needed
    }
    
    # Check if this aspect name has a known mapping
    for variant, canonical in name_mappings.items():
        if variant in aspect_name:
            return aspect_name.replace(variant, canonical)
    
    return aspect_name


def extract_survey_id(filepath, category):
    """Extract canonical survey ID: 'Category/SurveyName'."""
    basename = os.path.basename(filepath.replace('\\', '/'))
    for suffix in ['_split.json', '.json']:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    return f"{category}/{basename}"


def _get_files_from_cat(cat_data):
    """Extract file list from category data (handles list / dict)."""
    if isinstance(cat_data, list):
        return cat_data
    elif isinstance(cat_data, dict):
        return cat_data.get('files', [])
    return []


# ============================================================
# Load weights
# ============================================================

def load_bt_weights(bt_weights_path):
    """
    Load BT weights from bt_weights JSON.

    Supports three formats:
      - Legacy:    results[level] = {weights: [...]}
      - Domain:    results[level][domain] = {weights: [...]}
      - Component: results[component][level][domain] = {weights: [...]}

    Component format is flattened to composite keys "component/level"
    so downstream code works uniformly.

    Returns:
      weight_maps : dict  {key: {domain: {feature: weight}}}
                    or {key: {feature: weight}} (legacy only)
                    key is "level" or "component/level"
      meta        : dict  metadata (accuracy, loss, etc.)
      is_domain_specific : bool  whether weights are domain-specific
    """
    with open(bt_weights_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', {})
    weight_maps = {}
    meta = {}
    is_domain_specific = False

    COMPONENT_NAMES = {'outline', 'content', 'reference'}

    # Detect component format by checking first-level keys
    first_keys = set(results.keys())
    is_component_format = bool(first_keys & COMPONENT_NAMES)

    if is_component_format:
        # ── Component format: results[component][level][domain] ──
        is_domain_specific = True
        for component, comp_data in results.items():
            if not isinstance(comp_data, dict):
                continue
            for level, level_data in comp_data.items():
                if not isinstance(level_data, dict):
                    continue
                composite_key = f"{component}/{level}"
                weight_maps[composite_key] = {}
                meta[composite_key] = {}
                for domain, domain_res in level_data.items():
                    if not isinstance(domain_res, dict) \
                            or 'weights' not in domain_res:
                        continue
                    wm = {}
                    for w in domain_res['weights']:
                        wm[w['feature']] = w['normalized_weight']
                    weight_maps[composite_key][domain] = wm
                    meta[composite_key][domain] = {
                        'prediction_accuracy':
                            domain_res.get('prediction_accuracy'),
                        'average_log_loss':
                            domain_res.get('average_log_loss'),
                        'n_features':
                            domain_res.get('n_features'),
                        'source_config': data.get('config', {})
                    }
    else:
        # ── Legacy / Domain formats ──
        for level, res in results.items():
            if isinstance(res, dict) and 'weights' not in res:
                # Domain-specific
                is_domain_specific = True
                weight_maps[level] = {}
                meta[level] = {}
                for domain, domain_res in res.items():
                    if not isinstance(domain_res, dict) \
                            or 'weights' not in domain_res:
                        continue
                    wm = {}
                    for w in domain_res['weights']:
                        wm[w['feature']] = w['normalized_weight']
                    weight_maps[level][domain] = wm
                    meta[level][domain] = {
                        'prediction_accuracy':
                            domain_res.get('prediction_accuracy'),
                        'average_log_loss':
                            domain_res.get('average_log_loss'),
                        'n_features':
                            domain_res.get('n_features'),
                        'source_config': data.get('config', {})
                    }
            else:
                # Legacy: domain-agnostic
                wm = {}
                for w in res['weights']:
                    wm[w['feature']] = w['normalized_weight']
                weight_maps[level] = wm
                meta[level] = {
                    'prediction_accuracy':
                        res.get('prediction_accuracy'),
                    'average_log_loss':
                        res.get('average_log_loss'),
                    'n_features': res.get('n_features'),
                    'source_config': data.get('config', {})
                }

    return weight_maps, meta, is_domain_specific


# ============================================================
# Extract scores from a single file entry
# ============================================================

def extract_scores(file_entry, feature_groups, compute_aspect_from_criterion=False):
    """
    Extract group-level, aspect-level, and criterion-level scores
    from one survey entry.

    Args:
      compute_aspect_from_criterion: If True, always compute aspect scores
        by averaging criterion scores, even if aspect has its own score.

    Returns:
      group_scores     : {group_name: score_or_None}
      aspect_scores    : {group/aspect_name: score_or_None}
      criterion_scores : {group/aspect_name/criterion_name: score_or_None}
    """
    group_scores = {}
    aspect_scores = {}
    criterion_scores = {}

    for group in feature_groups:
        gd = file_entry.get('scores', {}).get(group, None)
        if gd is None:
            group_scores[group] = None
            continue

        if isinstance(gd, dict):
            group_scores[group] = gd.get('score', None)
            if 'aspects' in gd:
                for asp in gd['aspects']:
                    if not isinstance(asp, dict):
                        continue
                    raw_name = asp.get('aspect_name', '')
                    asp_name = clean_aspect_name(raw_name)
                    afname = f"{group}/{asp_name}"
                    
                    # Extract criterion scores first
                    if 'criteria' in asp:
                        # Filter out non-dict items (handle malformed data)
                        valid_criteria = [c for c in asp['criteria'] 
                                         if isinstance(c, dict)]
                        c_sc = [c['score'] for c in valid_criteria
                                if c.get('score') is not None]
                        # Always extract individual criterion scores
                        for crit in valid_criteria:
                            crit_name = clean_aspect_name(
                                crit.get('criterion_name', ''))
                            cfname = f"{group}/{asp_name}/{crit_name}"
                            val = crit.get('score', None)
                            if cfname not in criterion_scores or \
                                    criterion_scores[cfname] is None:
                                criterion_scores[cfname] = \
                                    float(val) if val is not None else None
                        
                        # Compute aspect score from criteria if requested or if no direct aspect score
                        if compute_aspect_from_criterion or c_sc:
                            aspect_scores[afname] = \
                                float(np.mean(c_sc)) if c_sc else None
                        else:
                            # Fall back to aspect's own score if available
                            val = asp.get('score', None)
                            if afname not in aspect_scores or \
                                    aspect_scores[afname] is None:
                                aspect_scores[afname] = \
                                    float(val) if val is not None else None
                    else:
                        # No criteria, use aspect's own score
                        val = asp.get('score', None)
                        if afname not in aspect_scores or \
                                aspect_scores[afname] is None:
                            aspect_scores[afname] = \
                                float(val) if val is not None else None
        elif isinstance(gd, (int, float)):
            group_scores[group] = float(gd)

    return group_scores, aspect_scores, criterion_scores


# ============================================================
# Apply weights to one survey
# ============================================================

def apply_weights_to_survey(group_scores, aspect_scores, criterion_scores,
                            weight_maps, domain=None, is_domain_specific=False):
    """
    Compute BT-weighted scores for one survey.
    
    Args:
      group_scores, aspect_scores, criterion_scores: score dicts
      weight_maps: {level: {domain?: {feature: weight}}} or {level: {feature: weight}}
      domain: domain name (required if is_domain_specific=True)
      is_domain_specific: whether weights are domain-specific

    Returns:
      weighted : {level: {weighted_overall, has_missing, breakdown}}
    """
    weighted = {}

    for level, wm_level in weight_maps.items():
        # Select the appropriate weight map based on domain
        if is_domain_specific:
            if domain is None:
                # Skip this level if domain is required but not provided
                weighted[level] = {
                    'weighted_overall': None,
                    'has_missing': True,
                    'breakdown': {},
                    'error': 'Domain required but not provided'
                }
                continue
            
            if domain not in wm_level:
                # Domain not found in weights - skip or use fallback
                weighted[level] = {
                    'weighted_overall': None,
                    'has_missing': True,
                    'breakdown': {},
                    'error': f'Weights not found for domain: {domain}'
                }
                continue
            
            wm = wm_level[domain]
        else:
            # Old format: wm_level is directly the weight map
            wm = wm_level
        
        breakdown = {}
        total = 0.0
        has_missing = False
        missing_count = 0
        total_count = len(wm)

        # Determine the granularity from the level key.
        # With component format, level is e.g. "outline/aspect" or
        # "content/criterion"; extract the suffix after the last '/'.
        granularity = level.rsplit('/', 1)[-1] if '/' in level else level

        for fname, w in wm.items():
            if granularity == "group":
                raw_sc = group_scores.get(fname, None)
            elif granularity == "criterion":
                raw_sc = criterion_scores.get(fname, None)
            else:  # aspect
                # Try exact match first
                raw_sc = aspect_scores.get(fname, None)
                # If not found, try with normalized name
                if raw_sc is None:
                    normalized_fname = normalize_aspect_name(fname)
                    if normalized_fname != fname:
                        raw_sc = aspect_scores.get(normalized_fname, None)

            if raw_sc is None:
                missing_count += 1
                # Only mark as missing if significant portion is missing
                # or if the missing feature has significant weight
                if w > 0.01:  # Feature with >1% weight
                    has_missing = True
                contribution = 0.0
            else:
                contribution = w * raw_sc

            breakdown[fname] = {
                'weight': round(w, 6),
                'score': round(raw_sc, 4) if raw_sc is not None else None,
                'contribution': round(contribution, 6)
            }
            total += contribution
        
        # Additional check: if more than 30% features are missing, mark as missing
        if missing_count / total_count > 0.3:
            has_missing = True

        weighted[level] = {
            'weighted_overall': round(total, 6),
            'has_missing': has_missing,
            'breakdown': breakdown
        }

    return weighted


# ============================================================
# Main rescoring logic
# ============================================================

def rescore_summary(eval_summary_path, weight_maps, systems=None,
                    categories=None, feature_groups=None, is_domain_specific=False,
                    compute_aspect_from_criterion=False):
    """
    Apply BT weights to all surveys in an evaluation summary.

    Args:
      is_domain_specific: whether weights are domain-specific
      compute_aspect_from_criterion: If True, always compute aspect scores
        by averaging criterion scores

    Returns:
      list of rescored entry dicts, grouped by (system, category).
    """
    with open(eval_summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if feature_groups is None:
        feature_groups = ['outline', 'content', 'reference']

    by_system = data.get('by_system', {})
    all_entries = []

    for sys_name, sys_data in by_system.items():
        if systems and sys_name not in systems:
            continue

        for cat_name, cat_data in sys_data.items():
            if categories and cat_name not in categories:
                continue

            for file_entry in _get_files_from_cat(cat_data):
                filepath = file_entry.get('file', '')
                category = file_entry.get('category', cat_name)
                survey_id = extract_survey_id(filepath, category)

                group_scores, aspect_scores, criterion_scores = \
                    extract_scores(file_entry, feature_groups, 
                                 compute_aspect_from_criterion=compute_aspect_from_criterion)

                # Original overall = equal-weight avg of group scores
                valid_orig = [v for v in group_scores.values()
                              if v is not None]
                original_overall = float(np.mean(valid_orig)) \
                    if valid_orig else None

                # BT-weighted scores (use category as domain)
                weighted = apply_weights_to_survey(
                    group_scores, aspect_scores, criterion_scores,
                    weight_maps, domain=category, is_domain_specific=is_domain_specific
                )

                all_entries.append({
                    'file': filepath,
                    'survey_id': survey_id,
                    'system': sys_name,
                    'category': category,
                    'scores_original': {
                        **{g: round(group_scores[g], 4)
                           if group_scores.get(g) is not None else None
                           for g in feature_groups},
                        'overall': round(original_overall, 4)
                            if original_overall is not None else None
                    },
                    'scores_bt_weighted': weighted
                })

    return all_entries


def compute_ranks(entries, weight_maps):
    """Add rank fields to each entry, per system+category group."""
    from itertools import groupby

    # Group by (system, category)
    entries_sorted = sorted(entries, key=lambda e: (e['system'], e['category']))
    for key, group_iter in groupby(
            entries_sorted, key=lambda e: (e['system'], e['category'])):
        group = list(group_iter)

        # Rank by original overall
        ranked = sorted(
            group,
            key=lambda e: (e['scores_original']['overall'] is not None,
                           e['scores_original']['overall'] or 0),
            reverse=True
        )
        for rank, entry in enumerate(ranked, 1):
            entry.setdefault('ranks', {})
            entry['ranks']['original'] = rank

        # Rank by each weighted level
        for level in weight_maps:
            ranked = sorted(
                group,
                key=lambda e: (
                    not e['scores_bt_weighted']
                        .get(level, {}).get('has_missing', True),
                    e['scores_bt_weighted']
                        .get(level, {}).get('weighted_overall', 0)
                ),
                reverse=True
            )
            for rank, entry in enumerate(ranked, 1):
                entry['ranks'][f'bt_{level}'] = rank


# ============================================================
# Pretty-print
# ============================================================

def print_summary_table(entries, weight_maps):
    """Print a comparison table to stdout."""
    from itertools import groupby

    entries_sorted = sorted(entries,
                            key=lambda e: (e['system'], e['category']))

    for key, group_iter in groupby(
            entries_sorted, key=lambda e: (e['system'], e['category'])):
        sys_name, cat_name = key
        group = list(group_iter)

        print(f"\n  === {sys_name} / {cat_name} "
              f"({len(group)} surveys) ===")

        # Header
        levels = list(weight_maps.keys())
        header = f"  {'#':>3s}  {'Survey':<52s}  {'Orig':>5s}"
        for lv in levels:
            header += f"  {'BT-' + lv:>10s}"
        header += f"  {'Rank(orig->BT)':>15s}"
        print(header)
        print("  " + "-" * len(header))

        # Sort by first available BT level descending
        first_lv = levels[0]
        group.sort(
            key=lambda e: e['scores_bt_weighted']
                .get(first_lv, {}).get('weighted_overall', 0),
            reverse=True
        )

        for i, entry in enumerate(group, 1):
            short = entry['survey_id'].split('/')[-1][:48]
            orig = entry['scores_original']['overall']
            orig_s = f"{orig:.2f}" if orig is not None else " N/A"

            row = f"  {i:3d}  {short:<52s}  {orig_s:>5s}"
            for lv in levels:
                ws = entry['scores_bt_weighted'] \
                    .get(lv, {}).get('weighted_overall')
                missing = entry['scores_bt_weighted'] \
                    .get(lv, {}).get('has_missing', False)
                if ws is not None and not missing:
                    row += f"  {ws:>10.4f}"
                else:
                    row += f"  {'N/A*':>10s}"
            # Rank change
            r_o = entry.get('ranks', {}).get('original', '?')
            r_n = entry.get('ranks', {}).get(f'bt_{first_lv}', '?')
            if isinstance(r_o, int) and isinstance(r_n, int):
                delta = r_o - r_n
                arrow = f"+{delta}" if delta > 0 else str(delta)
                row += f"  {r_o:>2d} -> {r_n:<2d} ({arrow:>3s})"
            else:
                row += f"  {r_o} -> {r_n}"
            print(row)

        # N/A* note
        any_missing = any(
            entry['scores_bt_weighted'].get(lv, {}).get('has_missing', False)
            for entry in group for lv in levels
        )
        if any_missing:
            print("  * N/A = survey has missing scores in some groups")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apply BT-learned weights to rescore evaluation summaries'
    )
    parser.add_argument(
        '--config', type=str,
        default='scripts/config/apply_bt_weights_config.json',
        help='Path to config JSON file'
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    bt_weights_file = config['bt_weights_file']
    eval_summary_files = config.get('evaluation_summary_files', [])
    output_dir = config.get('output_dir', 'results/evaluation')
    systems = config.get('systems', None)
    categories = config.get('categories', None)
    feature_groups = config.get('feature_groups',
                                ['outline', 'content', 'reference'])
    weight_level = config.get('weight_level', 'both')
    compute_aspect_from_criterion = config.get('compute_aspect_from_criterion', False)

    print("=" * 65)
    print("  Apply BT Weights to Evaluation Summaries")
    print("=" * 65)
    print(f"  BT weights file  : {bt_weights_file}")
    print(f"  Eval summaries   : {eval_summary_files}")
    print(f"  Systems          : {systems or 'all'}")
    print(f"  Categories       : {categories or 'all'}")
    print(f"  Weight level     : {weight_level}")
    print(f"  Compute aspect from criterion: {compute_aspect_from_criterion}")
    print()

    # --- Load weights ---
    print("[1] Loading BT weights...")
    weight_maps, weight_meta, is_domain_specific = load_bt_weights(bt_weights_file)
    available_levels = list(weight_maps.keys())
    print(f"  Available levels: {available_levels}")
    print(f"  Domain-specific: {is_domain_specific}")

    # Filter to requested levels
    # Keys may be plain ("aspect") or composite ("outline/aspect")
    def _level_of(key):
        """Extract the level part from a weight-map key."""
        return key.split('/')[-1] if '/' in key else key

    if weight_level == "both":
        keep = {'group', 'aspect'}
        weight_maps = {k: v for k, v in weight_maps.items()
                       if _level_of(k) in keep}
    elif weight_level == "all":
        pass  # Keep everything
    else:
        weight_maps = {k: v for k, v in weight_maps.items()
                       if _level_of(k) == weight_level}
    
    if not weight_maps:
        print(f"[ERROR] No weights found for level '{weight_level}'.")
        print(f"  Available levels: {available_levels}")
        sys.exit(1)

    # Print weight information
    for level, wm_level in weight_maps.items():
        if is_domain_specific:
            # Domain-specific format
            print(f"\n  [{level}] Domain-specific weights:")
            for domain, wm in sorted(wm_level.items()):
                print(f"    Domain: {domain} ({len(wm)} features):")
                sorted_w = sorted(wm.items(), key=lambda x: x[1], reverse=True)
                for fname, w in sorted_w[:5]:  # Show top 5
                    if w > 0:
                        print(f"      {fname:<48s} {w:.4f}")
                if len(sorted_w) > 5:
                    print(f"      ... ({len(sorted_w) - 5} more)")
        else:
            # Old format
            print(f"\n  [{level}] {len(wm_level)} features:")
            sorted_w = sorted(wm_level.items(), key=lambda x: x[1], reverse=True)
            for fname, w in sorted_w:
                if w > 0:
                    print(f"    {fname:<50s} {w:.4f}")

    # --- Rescore each eval summary ---
    all_entries = []

    for eval_file in eval_summary_files:
        print(f"\n[2] Rescoring: {eval_file}")
        entries = rescore_summary(
            eval_file, weight_maps, systems, categories, feature_groups,
            is_domain_specific=is_domain_specific,
            compute_aspect_from_criterion=compute_aspect_from_criterion
        )
        print(f"  Rescored {len(entries)} surveys")
        all_entries.extend(entries)

    if not all_entries:
        print("[ERROR] No surveys rescored. Check file paths and filters.")
        sys.exit(1)

    # --- Compute ranks ---
    compute_ranks(all_entries, weight_maps)

    # --- Print table ---
    print(f"\n{'=' * 65}")
    print("  Results: Original vs BT-Weighted Scores")
    print(f"{'=' * 65}")
    print_summary_table(all_entries, weight_maps)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir,
                               f'rescored_summary_{timestamp}.json')

    # Format weights for output
    if is_domain_specific:
        bt_weights_output = {
            level: {
                domain: [{'feature': k, 'normalized_weight': v}
                        for k, v in sorted(wm.items(),
                                           key=lambda x: x[1], reverse=True)]
                for domain, wm in wm_level.items()
            }
            for level, wm_level in weight_maps.items()
        }
    else:
        bt_weights_output = {
            level: [{'feature': k, 'normalized_weight': v}
                    for k, v in sorted(wm.items(),
                                       key=lambda x: x[1], reverse=True)]
            for level, wm in weight_maps.items()
        }
    
    output_data = {
        'generated_at': timestamp,
        'config': config,
        'bt_weights_source': bt_weights_file,
        'bt_weights_meta': weight_meta,
        'is_domain_specific': is_domain_specific,
        'bt_weights': bt_weights_output,
        'surveys': all_entries
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {output_file}")
    print("\nDone.")


if __name__ == '__main__':
    main()
