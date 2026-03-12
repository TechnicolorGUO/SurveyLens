#!/usr/bin/env python3
"""
DeepSeek-based all-metrics spot check.

This script samples rows from a metrics CSV, asks a large model to independently
estimate all core metrics from markdown, and compares them with current stats.

Checked metrics:
- img
- tab
- eq
- para
- words
- sent
- citation
- reference
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


METRIC_KEYS = ["img", "tab", "eq", "para", "words", "sent", "citation", "reference"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        default=".env.deepseek",
        help="Env file containing DEEPSEEK_API_KEY and settings.",
    )
    parser.add_argument(
        "--csv",
        default="original_metrics_output_full_v2/ORIGINAL_ALL_SYSTEMS_COMPLETE_EXTENDED_CITATION_DATA.csv",
        help="Metrics CSV with columns including file_name and metric columns.",
    )
    parser.add_argument(
        "--md-root",
        default="original",
        help="Root dir of markdown files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size. If omitted, reads LLM_SAMPLE_SIZE from env or uses 30.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260208,
        help="Random seed.",
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.6,
        help="Delay between API calls.",
    )
    parser.add_argument(
        "--max-ref-chars",
        type=int,
        default=None,
        help="Max chars sent to model from the reference section.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=None,
        help="Max chars sent to model from the main body text.",
    )
    parser.add_argument(
        "--out",
        default="LLM_DEEPSEEK_ALL_METRICS_SAMPLE_CHECK.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --out file and skip already processed rows.",
    )
    return parser.parse_args()


def resolve_env_file(env_file_arg: str) -> Path:
    """
    Resolve env path robustly:
    1) absolute path as-is
    2) relative to current working directory
    3) relative to this script directory
    """
    p = Path(env_file_arg).expanduser()
    if p.is_absolute() and p.exists():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    script_candidate = (Path(__file__).resolve().parent / p).resolve()
    if script_candidate.exists():
        return script_candidate
    return cwd_candidate


def clean_heading(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^\s*>+\s*", "", s)
    s = re.sub(r"^\s*[-*]\s*", "", s)
    s = re.sub(r"^[*_`~]+|[*_`~]+$", "", s).strip()
    return s


def extract_references_and_body(text: str):
    """
    Returns (reference_section, body_text_without_reference_section).
    """
    lines = text.splitlines()
    heading_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\d+\.?\s*)?"
        r"(references|bibliography|reference list|works cited|literature cited|references and notes)\b"
        r"\s*:?\s*$",
        re.IGNORECASE,
    )
    end_heading_re = re.compile(r"^\s*#{1,6}\s+[A-Za-z]")
    tail_heading_re = re.compile(
        r"^\s*(?:#\s*)?(appendix|acknowledg(?:e)?ments?|funding|conflicts?\s+of\s+interest|"
        r"supplementary|supporting\s+information|author\s+contributions?|endnotes?)\b",
        re.IGNORECASE,
    )

    candidate_idxs = []
    for i, line in enumerate(lines):
        raw = line.strip()
        norm = clean_heading(line)
        if heading_re.match(raw) or heading_re.match(norm):
            candidate_idxs.append(i)

    if candidate_idxs:
        idx = candidate_idxs[-1]
        start = idx + 1
        if start < len(lines) and re.match(r"^\s*[-=]{3,}\s*$", lines[start]):
            start += 1
        while start < len(lines) and not lines[start].strip():
            start += 1

        end = len(lines)
        for j in range(start, len(lines)):
            line = lines[j]
            if end_heading_re.match(line) or tail_heading_re.match(line):
                end = j
                break

        ref_section = "\n".join(lines[start:end]).strip()
        body_text = "\n".join(lines[:idx] + lines[end:]).strip()
        return ref_section, body_text

    # fallback when no heading: tail as references, full text as body
    tail = lines[-700:] if len(lines) > 700 else lines
    ref_section = "\n".join(tail).strip()
    body_text = text
    return ref_section, body_text


def truncate_preserve_head_tail(text: str, max_chars: int):
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head_len = max_chars // 2
    tail_len = max_chars - head_len - 32
    if tail_len < 0:
        tail_len = 0
    return text[:head_len] + "\n\n[... TRUNCATED ...]\n\n" + text[-tail_len:]


def build_md_index(md_root: Path):
    idx_with_system = {}
    idx_by_name = {}
    for p in md_root.rglob("*.md"):
        stem = p.stem
        idx_by_name.setdefault(stem, []).append(p)
        if len(p.parts) >= 3:
            try:
                root_pos = p.parts.index(md_root.name)
                if len(p.parts) > root_pos + 1:
                    system = p.parts[root_pos + 1]
                    idx_with_system[(system, stem)] = p
            except ValueError:
                pass
    return idx_with_system, idx_by_name


def resolve_md_path(row, idx_with_system, idx_by_name):
    name = str(row["file_name"])
    system = row["system"] if "system" in row and pd.notna(row["system"]) else None
    if system:
        p = idx_with_system.get((str(system), name))
        if p:
            return p
    candidates = idx_by_name.get(name, [])
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0] if candidates else None


def sample_rows(df: pd.DataFrame, sample_size: int, seed: int):
    random.seed(seed)
    n = min(sample_size, len(df))
    if n <= 0:
        return df.head(0)

    if "discipline" not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    per_group = max(1, n // max(1, df["discipline"].nunique()))
    chunks = []
    for _, g in df.groupby("discipline"):
        take = min(len(g), per_group)
        chunks.append(g.sample(n=take, random_state=seed))
    sampled = pd.concat(chunks, ignore_index=True)

    if len(sampled) < n:
        remaining = df.drop(sampled.index, errors="ignore")
        extra_n = min(n - len(sampled), len(remaining))
        if extra_n > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra_n, random_state=seed)],
                ignore_index=True,
            )
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    return sampled.reset_index(drop=True)


def parse_json_from_text(text: str):
    text = (text or "").strip()
    if not text:
        return None

    # Remove code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def coerce_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        return int(round(float(m.group(0))))
    return None


def normalize_llm_metrics(payload):
    if not isinstance(payload, dict):
        return None

    out = {}
    for k in METRIC_KEYS:
        out[k] = coerce_int(payload.get(k))

    # Optional confidence map or scalar.
    conf = payload.get("confidence")
    if isinstance(conf, dict):
        for k in METRIC_KEYS:
            out[f"confidence_{k}"] = conf.get(k)
    else:
        for k in METRIC_KEYS:
            out[f"confidence_{k}"] = conf

    note = payload.get("note") or payload.get("notes")
    out["note"] = note

    if any(out[k] is None for k in METRIC_KEYS):
        return None
    return out


def ask_llm_all_metrics(client: OpenAI, model: str, file_name: str, body_text: str, ref_text: str):
    prompt = (
        "You are validating paper metrics for one markdown review paper.\n"
        "Count ALL metrics below and return strict JSON only.\n\n"
        "Required JSON keys (integers):\n"
        "img, tab, eq, para, words, sent, citation, reference\n"
        "Optional: confidence (object with same keys, 0-1), note (short string).\n\n"
        "Definitions:\n"
        "- img: number of figures/images/charts explicitly present or clearly labeled in the markdown body.\n"
        "- tab: number of tables explicitly present or clearly labeled in the markdown body.\n"
        "- eq: number of equations (display or inline) in body.\n"
        "- para: number of substantive paragraphs in body.\n"
        "- words: body word count (approximate but reasonable).\n"
        "- sent: body sentence count (approximate but reasonable).\n"
        "- citation: in-text citation count in body (not reference list entries).\n"
        "- reference: number of reference entries in reference section.\n\n"
        "Rules for reference:\n"
        "1) If numbered references exist, trust largest reliable index when sequence is reasonably continuous.\n"
        "2) If unnumbered APA style, count unique entries.\n"
        "3) Do not count years or in-text citations as reference entries.\n\n"
        f"Paper: {file_name}\n\n"
        "BODY TEXT:\n"
        f"{body_text}\n\n"
        "REFERENCE SECTION:\n"
        f"{ref_text}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict metrics counter. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content or ""
    data = parse_json_from_text(content)
    normalized = normalize_llm_metrics(data)
    if not normalized:
        return None, content
    return normalized, content


def metric_tolerance(metric: str, stat_val: int):
    if metric in {"img", "tab", "eq", "citation", "reference"}:
        return 0
    if metric == "para":
        return max(2, int(round(0.05 * max(1, stat_val))))
    if metric == "words":
        return max(80, int(round(0.05 * max(1, stat_val))))
    if metric == "sent":
        return max(10, int(round(0.08 * max(1, stat_val))))
    return 0


def compare_metric(metric: str, stat_val: int, llm_val: int):
    tol = metric_tolerance(metric, stat_val)
    delta = llm_val - stat_val
    if abs(delta) <= tol:
        status = "close"
    elif delta > 0:
        status = "llm_higher"
    else:
        status = "stat_higher"
    return delta, tol, status


def main():
    args = parse_args()
    env_path = resolve_env_file(args.env_file)
    load_dotenv(env_path)
    load_dotenv(".env", override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    sample_size = args.sample_size or int(os.getenv("LLM_SAMPLE_SIZE", "30"))
    max_ref_chars = args.max_ref_chars or int(os.getenv("LLM_MAX_REF_CHARS", "30000"))
    max_body_chars = args.max_body_chars or int(os.getenv("LLM_MAX_BODY_CHARS", "60000"))

    if not api_key:
        raise SystemExit(
            f"DEEPSEEK_API_KEY is empty. Loaded env file: {env_path}. "
            "Fill it in that file or pass --env-file /absolute/path/.env.deepseek then rerun."
        )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    md_root = Path(args.md_root)
    if not md_root.exists():
        raise SystemExit(f"Markdown root not found: {md_root}")

    df = pd.read_csv(csv_path)
    missing = {"file_name"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    metric_missing = [m for m in METRIC_KEYS if m not in df.columns]
    if metric_missing:
        raise SystemExit(f"CSV missing metric columns: {metric_missing}")

    sample_df = sample_rows(df, sample_size, args.seed)
    idx_with_system, idx_by_name = build_md_index(md_root)
    client = OpenAI(api_key=api_key, base_url=base_url)

    out_path = Path(args.out)
    rows = []
    done_keys = set()
    if args.resume and out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            rows = existing.to_dict(orient="records")
            for r in rows:
                done_keys.add((str(r.get("system", "")), str(r.get("file_name", ""))))
            print(f"Resume enabled: loaded {len(rows)} existing rows from {out_path}")
        except Exception as e:
            print(f"Resume load failed, starting fresh: {e}")
            rows = []
            done_keys = set()
    total = len(sample_df)

    for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        row_system = str(row["system"]) if "system" in row and pd.notna(row["system"]) else ""
        row_file = str(row["file_name"])
        if (row_system, row_file) in done_keys:
            print(f"[{idx}/{total}] skip(resume): {row_file}")
            continue

        md_path = resolve_md_path(row, idx_with_system, idx_by_name)
        base_row = {
            "system": row_system,
            "discipline": row["discipline"] if "discipline" in row else "",
            "file_name": row_file,
            "md_path": str(md_path) if md_path else "",
        }

        if not md_path:
            out = {
                **base_row,
                "overall_status": "md_not_found",
                "under_count_metrics": "",
                "raw_llm": "",
                "note": "markdown path not found",
            }
            for m in METRIC_KEYS:
                out[f"{m}_stat"] = int(row[m]) if pd.notna(row[m]) else None
                out[f"{m}_llm"] = None
                out[f"{m}_delta"] = None
                out[f"{m}_tol"] = None
                out[f"{m}_status"] = "md_not_found"
                out[f"confidence_{m}"] = None
            rows.append(out)
            done_keys.add((row_system, row_file))
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
            print(f"[{idx}/{total}] md_not_found: {row['file_name']}")
            continue

        text = md_path.read_text(encoding="utf-8", errors="replace")
        ref_section, body_text = extract_references_and_body(text)
        ref_section = truncate_preserve_head_tail(ref_section, max_ref_chars)
        body_text = truncate_preserve_head_tail(body_text, max_body_chars)

        parsed, raw_content = ask_llm_all_metrics(
            client=client,
            model=model,
            file_name=str(row["file_name"]),
            body_text=body_text,
            ref_text=ref_section,
        )

        if parsed is None:
            out = {
                **base_row,
                "overall_status": "llm_parse_error",
                "under_count_metrics": "",
                "raw_llm": raw_content[:1200],
                "note": "cannot parse llm json",
            }
            for m in METRIC_KEYS:
                out[f"{m}_stat"] = int(row[m]) if pd.notna(row[m]) else None
                out[f"{m}_llm"] = None
                out[f"{m}_delta"] = None
                out[f"{m}_tol"] = None
                out[f"{m}_status"] = "llm_parse_error"
                out[f"confidence_{m}"] = None
            rows.append(out)
            done_keys.add((row_system, row_file))
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
            print(f"[{idx}/{total}] llm_parse_error: {row['file_name']}")
            time.sleep(max(0.0, args.delay_sec))
            continue

        out = {
            **base_row,
            "raw_llm": raw_content[:1200],
            "note": parsed.get("note"),
        }
        under_count_metrics = []
        metric_statuses = []

        for m in METRIC_KEYS:
            stat_val = int(row[m]) if pd.notna(row[m]) else 0
            llm_val = int(parsed[m])
            delta, tol, status = compare_metric(m, stat_val, llm_val)

            out[f"{m}_stat"] = stat_val
            out[f"{m}_llm"] = llm_val
            out[f"{m}_delta"] = delta
            out[f"{m}_tol"] = tol
            out[f"{m}_status"] = status
            out[f"confidence_{m}"] = parsed.get(f"confidence_{m}")

            metric_statuses.append(status)
            if status == "llm_higher":
                under_count_metrics.append(m)

        if under_count_metrics:
            overall = "possible_undercount"
        elif all(s == "close" for s in metric_statuses):
            overall = "all_close"
        elif any(s == "stat_higher" for s in metric_statuses):
            overall = "mixed_drift"
        else:
            overall = "review_needed"

        out["overall_status"] = overall
        out["under_count_metrics"] = ";".join(under_count_metrics)
        rows.append(out)
        done_keys.add((row_system, row_file))

        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        print(
            f"[{idx}/{total}] done: {row['file_name']} "
            f"(overall={overall}, under={out['under_count_metrics'] or '-'})"
        )

        time.sleep(max(0.0, args.delay_sec))

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        status_order = {
            "possible_undercount": 0,
            "mixed_drift": 1,
            "all_close": 2,
            "review_needed": 3,
            "llm_parse_error": 4,
            "md_not_found": 5,
        }
        out_df["_ord"] = out_df["overall_status"].map(status_order).fillna(99)
        out_df = out_df.sort_values(["_ord", "system", "discipline", "file_name"]).drop(columns=["_ord"])

    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print("=" * 80)
    print(f"Sample size: {len(out_df)}")
    print(f"Output: {out_path}")
    print("-" * 80)
    if len(out_df) > 0:
        print(out_df["overall_status"].value_counts(dropna=False).to_string())
        under = out_df[out_df["overall_status"] == "possible_undercount"]
        if len(under) > 0:
            print("-" * 80)
            print("Top possible undercount rows:")
            cols = [
                "system",
                "discipline",
                "file_name",
                "under_count_metrics",
                "reference_stat",
                "reference_llm",
                "citation_stat",
                "citation_llm",
            ]
            cols = [c for c in cols if c in under.columns]
            print(under[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
