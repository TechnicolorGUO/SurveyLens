#!/usr/bin/env python3
"""
Run metric extraction on `original/<system>/...` markdowns with the same
column names as dataset-level outputs, while using cleaned JSON refs as an
additional reference source.

Reference strategy:
  final_reference = max(reference_from_md, reference_from_json)
This follows the "prefer not under-counting" requirement.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


METRIC_COLUMNS = [
    "file_name",
    "discipline",
    "img",
    "tab",
    "eq",
    "para",
    "words",
    "sent",
    "citation",
    "reference",
    "characters",
]

BOOK_REVIEW_PATTERNS = (
    "book review",
    "books review",
    "book-review",
    "review of the book",
    "review essay",
)

DISCIPLINE_MAP = {
    "biology": "Biology",
    "business": "Business",
    "computer science": "Computer Science",
    "education": "Education",
    "environmental science": "Environmental Science",
    "engineering": "Engineering",
    "medicine": "Medicine",
    "physics": "Physics",
    "psychology": "Psychology",
    "sociology": "Sociology",
}

REF_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*\s*)?"
    r"(references|bibliography|reference list|works cited|literature cited|references and notes)"
    r"(?:\s*\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)

REF_END_HEADING_RE = re.compile(r"^\s*#{1,6}\s+[A-Za-z]")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_book_review(title: str) -> bool:
    t = norm_text(title).replace("_", " ")
    return any(p in t for p in BOOK_REVIEW_PATTERNS)


def canonical_discipline(name: str) -> str:
    key = norm_text(name)
    return DISCIPLINE_MAP.get(key, name.strip())


def load_metric_module(module_name: str = "complete_all_metrics_v2"):
    return importlib.import_module(module_name)


def _flatten_reference_item(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        parts = []
        for k in ("text", "reference", "raw", "content", "title"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            return " ".join(parts)
        return json.dumps(item, ensure_ascii=False)
    return str(item).strip()


def _extract_numeric_prefixes(text: str) -> set[int]:
    nums = set()
    if not text:
        return nums
    patterns = [
        r"^\s*\[(\d{1,4})\]",
        r"^\s*(\d{1,4})[.)]\s+",
        r"^\s*[\(（](\d{1,4})[\)）]\s*",
    ]
    for pat in patterns:
        for n in re.findall(pat, text, flags=re.MULTILINE):
            try:
                val = int(n)
            except ValueError:
                continue
            if _valid_ref_num(val):
                nums.add(val)
    return nums


def _valid_ref_num(n: int) -> bool:
    if n <= 0 or n > 2000:
        return False
    # Avoid treating publication years as reference indices.
    if 1900 <= n <= 2099:
        return False
    return True


def _extract_reference_section_aux(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        s2 = re.sub(r"^[*_`~]+|[*_`~]+$", "", s).strip()
        if REF_HEADING_RE.match(s) or REF_HEADING_RE.match(s2):
            start = i + 1
            break
    if start is None:
        # Fallback to tail if no heading is found.
        return "\n".join(lines[int(len(lines) * 0.70):])
    for j in range(start, len(lines)):
        if REF_END_HEADING_RE.match(lines[j].strip()):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def _conservative_md_lineprefix_ref_count(text: str) -> int:
    """
    Conservative auxiliary estimator based only on line-start numbering.
    Used as an extra "not under-counting" guardrail.
    """
    section = _extract_reference_section_aux(text)
    if not section.strip():
        return 0

    nums = set()
    patterns = [
        r"^\s*\[(\d{1,4})\]",
        r"^\s*(\d{1,4})[\.．]\s+",
        r"^\s*(\d{1,4})\)\s+",
        r"^\s*[\(（](\d{1,4})[\)）]\s*",
    ]
    for pat in patterns:
        for n in re.findall(pat, section, flags=re.MULTILINE):
            try:
                v = int(n)
            except ValueError:
                continue
            if _valid_ref_num(v):
                nums.add(v)

    if not nums:
        return 0

    uniq = sorted(nums)
    max_num = uniq[-1]
    cluster = _trust_max_cluster(nums)
    if cluster:
        return cluster
    if max_num <= len(uniq) + 40:
        return max_num
    return len(uniq)


def _trust_max_cluster(nums: set[int]) -> int:
    nums = {n for n in nums if _valid_ref_num(n)}
    if not nums:
        return 0
    arr = sorted(nums)
    for candidate in reversed(arr[-120:]):
        if candidate < 20:
            continue
        local = [n for n in arr if candidate - 12 <= n <= candidate]
        hits = len(local)
        adjacent_pairs = sum(1 for i in range(1, len(local)) if local[i] - local[i - 1] == 1)
        if hits >= 4 and adjacent_pairs >= 1:
            return candidate
    return 0


def count_json_references(json_path: Path) -> int:
    if not json_path.exists():
        return 0
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return 0

    refs = None
    if isinstance(data, dict):
        for key in ("references", "reference", "refs", "bibliography"):
            if key in data:
                refs = data[key]
                break
    elif isinstance(data, list):
        refs = data

    if refs is None:
        return 0

    if isinstance(refs, str):
        entries = [ln.strip() for ln in refs.splitlines() if ln.strip()]
    elif isinstance(refs, list):
        entries = [_flatten_reference_item(x) for x in refs]
        entries = [x for x in entries if x]
    else:
        return 0

    # Count non-empty entries and optionally trust numeric prefix clusters.
    # Guardrails are used to avoid inflated counts from noisy year tokens.
    entry_count = 0
    nums = set()
    for e in entries:
        if len(e.strip()) < 3:
            continue
        entry_count += 1
        nums |= _extract_numeric_prefixes(e)

    trusted_max = _trust_max_cluster(nums)
    # Only accept trusted_max when it is reasonably close to list length.
    # This keeps "prefer not under-counting" but avoids extreme over-counting.
    if trusted_max and trusted_max <= entry_count + 80:
        return max(entry_count, trusted_max)
    return entry_count


def collect_system_rows(metric_mod, system_dir: Path, progress: bool = True):
    rows = []
    ref_debug_rows = []
    md_files = sorted(system_dir.rglob("*.md"))
    iterator = tqdm(md_files, desc=f"{system_dir.name}", leave=False) if progress else md_files
    for md_path in iterator:
        file_stem = md_path.stem
        if is_book_review(file_stem):
            continue

        discipline = canonical_discipline(md_path.parent.name)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        json_path = md_path.with_name(f"{md_path.stem}_split.json")

        ref_md = metric_mod.count_references(text)
        ref_json = count_json_references(json_path)
        ref_aux_lineprefix = _conservative_md_lineprefix_ref_count(text)
        ref_final = max(ref_md, ref_json, ref_aux_lineprefix)

        row = {
            "file_name": file_stem,
            "discipline": discipline,
            "img": metric_mod.count_images(text),
            "tab": metric_mod.count_tables(text),
            "eq": metric_mod.count_equations(text),
            "para": metric_mod.count_paragraphs(text),
            "words": metric_mod.count_words(text),
            "sent": metric_mod.count_sentences(text),
            "citation": metric_mod.count_citations_extended(text),
            "reference": ref_final,
            "characters": len(text),
        }
        rows.append(row)

        ref_debug_rows.append(
            {
                "system": system_dir.name,
                "discipline": discipline,
                "file_name": file_stem,
                "reference": ref_final,
                "reference_md": ref_md,
                "reference_json": ref_json,
                "reference_aux_lineprefix": ref_aux_lineprefix,
                "json_exists": int(json_path.exists()),
                "md_path": str(md_path),
                "json_path": str(json_path) if json_path.exists() else "",
            }
        )
    return rows, ref_debug_rows


def summarize_by_system(df_all: pd.DataFrame):
    metrics = ["img", "tab", "eq", "para", "words", "sent", "citation", "reference", "characters"]

    by_subject = (
        df_all.groupby(["system", "discipline"], as_index=False)
        .agg(**{m: (m, "mean") for m in metrics}, paper_count=("file_name", "count"))
    )

    by_system = (
        df_all.groupby("system", as_index=False)
        .agg(**{m: (m, "mean") for m in metrics}, paper_count=("file_name", "count"))
        .sort_values("system")
    )
    return by_subject, by_system


def normalize_to_human(by_subject: pd.DataFrame, by_system: pd.DataFrame):
    metric_cols = ["img", "tab", "eq", "para", "words", "sent", "citation", "reference", "characters"]

    human_subject = by_subject[by_subject["system"] == "Human"][
        ["discipline"] + metric_cols
    ].rename(columns={m: f"{m}_human" for m in metric_cols})
    sub = by_subject.merge(human_subject, on="discipline", how="left")
    for m in metric_cols:
        denom = sub[f"{m}_human"].replace(0, pd.NA)
        sub[f"{m}_ratio_to_human"] = sub[m] / denom
    sub = sub.drop(columns=[f"{m}_human" for m in metric_cols])

    human_sys = by_system[by_system["system"] == "Human"]
    if len(human_sys) == 1:
        hs = human_sys.iloc[0]
        for m in metric_cols:
            denom = hs[m] if hs[m] != 0 else pd.NA
            by_system[f"{m}_ratio_to_human"] = by_system[m] / denom
    else:
        for m in metric_cols:
            by_system[f"{m}_ratio_to_human"] = pd.NA

    return sub, by_system


def run_independent_ref_spot_check(df_all: pd.DataFrame, ref_debug: pd.DataFrame, sample_n: int, seed: int):
    """
    Independent check (different rule path):
    compare final reference against max(md_ref, json_ref).
    Because final is defined as max(md_ref, json_ref), this check catches pipeline mismatch bugs.
    """
    random.seed(seed)
    if len(df_all) == 0:
        return pd.DataFrame()

    n = min(sample_n, len(df_all))
    idx = random.sample(list(df_all.index), n)
    samp = df_all.loc[idx, ["system", "discipline", "file_name", "reference"]].copy()
    dbg = ref_debug[["system", "discipline", "file_name", "reference_md", "reference_json"]].copy()
    chk = samp.merge(dbg, on=["system", "discipline", "file_name"], how="left")
    chk["independent_max"] = chk[["reference_md", "reference_json"]].max(axis=1)
    chk["delta_final_minus_ind"] = chk["reference"] - chk["independent_max"]
    chk["status"] = chk["delta_final_minus_ind"].apply(
        lambda x: "OK" if pd.notna(x) and x >= 0 else "UNDERCOUNT_POSSIBLE"
    )
    return chk.sort_values(["status", "delta_final_minus_ind", "system", "discipline", "file_name"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="original",
        help="Root folder that contains per-system subfolders.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Optional list of system folder names. Default: all systems under root.",
    )
    parser.add_argument(
        "--outdir",
        default="original_metrics_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=60,
        help="Sample size for independent reference spot check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260208,
        help="Random seed for spot check sampling.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metric_mod = load_metric_module("complete_all_metrics_v2")

    if args.systems:
        system_dirs = [root / s for s in args.systems]
    else:
        system_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    system_dirs = [p for p in system_dirs if p.exists() and p.is_dir()]
    if not system_dirs:
        raise SystemExit(f"No system folders found under: {root}")

    all_rows = []
    all_ref_debug = []

    for sdir in system_dirs:
        rows, ref_debug_rows = collect_system_rows(metric_mod, sdir, progress=not args.quiet)
        if not rows:
            continue
        all_rows.extend([{**r, "system": sdir.name} for r in rows])
        all_ref_debug.extend(ref_debug_rows)

        # Per-system file with same column names
        sys_df = pd.DataFrame(rows)[METRIC_COLUMNS]
        sys_df.to_csv(outdir / f"{sdir.name}_COMPLETE_EXTENDED_CITATION_DATA.csv", index=False, encoding="utf-8")

    df_all = pd.DataFrame(all_rows)
    if len(df_all) == 0:
        raise SystemExit("No markdown files were processed.")

    # Aggregate files
    all_cols = ["system"] + METRIC_COLUMNS
    df_all = df_all[all_cols].sort_values(["system", "discipline", "file_name"]).reset_index(drop=True)
    df_all.to_csv(outdir / "ORIGINAL_ALL_SYSTEMS_COMPLETE_EXTENDED_CITATION_DATA.csv", index=False, encoding="utf-8")

    df_ref_debug = pd.DataFrame(all_ref_debug).sort_values(["system", "discipline", "file_name"]).reset_index(drop=True)
    df_ref_debug.to_csv(outdir / "ORIGINAL_ALL_SYSTEMS_REFERENCE_DEBUG.csv", index=False, encoding="utf-8")

    by_subject, by_system = summarize_by_system(df_all)
    by_subject.to_csv(outdir / "ORIGINAL_SYSTEM_SUBJECT_MEANS.csv", index=False, encoding="utf-8")
    by_system.to_csv(outdir / "ORIGINAL_SYSTEM_OVERALL_MEANS.csv", index=False, encoding="utf-8")

    by_subject_norm, by_system_norm = normalize_to_human(by_subject.copy(), by_system.copy())
    by_subject_norm.to_csv(
        outdir / "ORIGINAL_SYSTEM_SUBJECT_MEANS_RATIO_TO_HUMAN.csv",
        index=False,
        encoding="utf-8",
    )
    by_system_norm.to_csv(
        outdir / "ORIGINAL_SYSTEM_OVERALL_MEANS_RATIO_TO_HUMAN.csv",
        index=False,
        encoding="utf-8",
    )

    # Spot check
    check_df = run_independent_ref_spot_check(df_all, df_ref_debug, args.sample_n, args.seed)
    check_df.to_csv(outdir / "ORIGINAL_REFERENCE_SPOT_CHECK.csv", index=False, encoding="utf-8")

    print("=" * 80)
    print("Completed original-system metric extraction")
    print(f"Processed systems: {len(system_dirs)}")
    print(f"Processed papers: {len(df_all)}")
    print(f"Output dir: {outdir}")
    print("-" * 80)
    print("Top-level outputs:")
    print("  - ORIGINAL_ALL_SYSTEMS_COMPLETE_EXTENDED_CITATION_DATA.csv")
    print("  - ORIGINAL_ALL_SYSTEMS_REFERENCE_DEBUG.csv")
    print("  - ORIGINAL_SYSTEM_SUBJECT_MEANS.csv")
    print("  - ORIGINAL_SYSTEM_SUBJECT_MEANS_RATIO_TO_HUMAN.csv")
    print("  - ORIGINAL_SYSTEM_OVERALL_MEANS.csv")
    print("  - ORIGINAL_SYSTEM_OVERALL_MEANS_RATIO_TO_HUMAN.csv")
    print("  - ORIGINAL_REFERENCE_SPOT_CHECK.csv")
    print("-" * 80)
    if len(check_df) > 0:
        print(check_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
