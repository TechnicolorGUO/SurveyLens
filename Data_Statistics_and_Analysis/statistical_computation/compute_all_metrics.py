#!/usr/bin/env python3
"""
完整统计脚本 V2 - 扩展Citation匹配规则
支持所有常见的引用格式
"""

import re
import csv
import unicodedata
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 用户指定：这些论文不参与统计
EXCLUDED_PAPER_PREFIXES = {
    "Medicine_25_",
    "Medicine_31_",
    "Medicine_50_",
    "Medicine_53_",
    "Medicine_61_",
}

SUMMARY_CSV_PATH = Path("summary2.csv")
BOOK_REVIEW_PATTERNS = (
    "book review",
    "books review",
    "book-review",
    "review of the book",
    "review essay",
)

_SUMMARY_TITLE_MAP = None


def is_excluded_paper(stem):
    return any(stem.startswith(prefix) for prefix in EXCLUDED_PAPER_PREFIXES)


def _parse_paper_stem(stem):
    m = re.match(r'^(.+?)_(\d+?)_(.+)$', stem)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    m = re.match(r'^([A-Za-z ]+?)(\d+)_(.+)$', stem)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def _extract_subject(paper_path, stem):
    """提取学科名，优先从路径读一级学科目录，失败时回退到文件名前缀"""
    if paper_path and len(paper_path.parts) >= 3:
        # Dataset%20final/Dataset final/<Subject>/...
        subject_dir = paper_path.parts[2].strip()
        if subject_dir:
            return subject_dir
    subject, _, _ = _parse_paper_stem(stem)
    return subject or "Unknown"


def _norm_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace("_", " ")
    s = re.sub(r"[’'`´]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_summary_title_map():
    global _SUMMARY_TITLE_MAP
    if _SUMMARY_TITLE_MAP is not None:
        return _SUMMARY_TITLE_MAP

    mapping = {}
    if not SUMMARY_CSV_PATH.exists():
        _SUMMARY_TITLE_MAP = mapping
        return _SUMMARY_TITLE_MAP

    with open(SUMMARY_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = _norm_text((row.get("Subject") or "").strip())
            rank_str = (row.get("Rank") or "").strip()
            title = _norm_text((row.get("Title") or "").strip())
            if not subject or not rank_str.isdigit() or not title:
                continue
            key = (subject, int(rank_str))
            # 如果summary2有重复键，保留第一次出现
            if key not in mapping:
                mapping[key] = title

    _SUMMARY_TITLE_MAP = mapping
    return _SUMMARY_TITLE_MAP


def _title_matches_summary(file_title, summary_title):
    if not file_title or not summary_title:
        return False
    if file_title == summary_title:
        return True
    if file_title in summary_title or summary_title in file_title:
        return True
    f_tokens = set(file_title.split())
    s_tokens = set(summary_title.split())
    if not f_tokens or not s_tokens:
        return False
    jaccard = len(f_tokens & s_tokens) / len(f_tokens | s_tokens)
    return jaccard >= 0.85


def _is_book_review_title(title):
    t = _norm_text(title)
    return any(p in t for p in BOOK_REVIEW_PATTERNS)


def get_exclusion_reason(stem):
    if is_excluded_paper(stem):
        return "manual_excluded"

    subject, rank, title = _parse_paper_stem(stem)
    if subject is None:
        return "parse_failed"

    if _is_book_review_title(title):
        return "book_review"

    summary_map = _load_summary_title_map()
    key = (_norm_text(subject), rank)
    summary_title = summary_map.get(key)
    if not summary_title:
        # 用户要求：summary2里缺失映射时不排除
        return None

    file_title = _norm_text(title)
    if not _title_matches_summary(file_title, summary_title):
        return "summary_mismatch"

    return None


def count_images(text):
    """统计图片数量（偏保守上界，宁可多算）"""
    if not text:
        return 0
    md_images = len(re.findall(r'!\[[^\]]*\]\([^)]+\)', text))
    html_images = len(re.findall(r'<img[^>]*>', text, re.IGNORECASE))
    latex_images = len(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}', text))
    inserted_images = md_images + html_images + latex_images

    # 图号引用兜底（Fig. 1 / Figure 2 / Fig S1 / Fig. IV）
    fig_refs = re.findall(
        r'\b(?:fig(?:ure)?s?|chart|graph|plot|diagram|schematic|illustration)s?\.?\s*'
        r'[:\-\.]?\s*\(?([A-Za-z]{0,2}\d+[A-Za-z]?|[IVXLC]{1,6})\)?',
        text,
        re.IGNORECASE,
    )
    unique_fig_refs = len(set(ref.strip().lower() for ref in fig_refs))

    return max(inserted_images, unique_fig_refs)


def count_tables(text):
    """统计表格数量（Markdown/HTML/LaTeX + Table编号引用）"""
    if not text:
        return 0

    # 旧规则基线（保证不比旧版少）
    old_pipe_block_count = 0
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if '|' in s and len(s) > 5:
            if not in_table:
                old_pipe_block_count += 1
                in_table = True
        else:
            in_table = False

    lines = text.splitlines()
    markdown_tables = 0
    i = 0
    sep_re = re.compile(r'^\s*\|?\s*:?-{2,}:?(?:\s*\|\s*:?-{2,}:?)+\s*\|?\s*$')
    while i < len(lines) - 1:
        head = lines[i].strip()
        sep = lines[i + 1].strip()
        if "|" in head and sep_re.match(sep):
            markdown_tables += 1
            i += 2
            while i < len(lines) and "|" in lines[i]:
                i += 1
            continue
        i += 1

    html_tables = len(re.findall(r'<table[^>]*>', text, re.IGNORECASE))
    latex_tables = len(
        re.findall(r'\\begin\{(?:table|tabular|tabulary|longtable)\*?\}', text, re.IGNORECASE)
    )

    # Table引用兜底（Table 1 / Table1 / Tab. 2 / Table IV）
    table_refs = re.findall(
        r'\b(?:table|tab\.)\s*[:\-\.]?\s*\(?([A-Za-z]{0,2}\d+[A-Za-z]?|[IVXLC]{1,6})\)?',
        text,
        re.IGNORECASE,
    )
    unique_table_refs = len(set(ref.strip().lower() for ref in table_refs))
    old_style_count = max(old_pipe_block_count, html_tables)
    new_style_count = markdown_tables + html_tables + latex_tables
    return max(new_style_count, unique_table_refs, old_style_count)


def count_equations(text):
    """统计公式数量（环境优先，避免重复，偏上界）"""
    if not text:
        return 0

    # 旧规则基线（保证不比旧版少）
    old_simple_count = 0
    old_simple_count += len(re.findall(r'\$[^\$]{1,100}\$', text))
    old_simple_count += len(re.findall(r'\$\$[^\$]{1,200}\$\$', text))

    working = text
    equation_count = 0

    block_patterns = [
        r'\\begin\{(?:equation|align|gather|multline|eqnarray|array|aligned)\*?\}[\s\S]{1,8000}?\\end\{(?:equation|align|gather|multline|eqnarray|array|aligned)\*?\}',
        r'\$\$[\s\S]{1,5000}?\$\$',
        r'\\\[[\s\S]{1,5000}?\\\]',
    ]
    for pat in block_patterns:
        matches = re.findall(pat, working, flags=re.DOTALL)
        equation_count += len(matches)
        working = re.sub(pat, ' ', working, flags=re.DOTALL)

    inline_patterns = [
        r'\\\([^\n]{1,500}?\\\)',
        r'(?<!\\)\$[^$\n]{1,500}(?<!\\)\$',
    ]
    for pat in inline_patterns:
        matches = re.findall(pat, working, flags=re.DOTALL)
        equation_count += len(matches)
        working = re.sub(pat, ' ', working, flags=re.DOTALL)

    # Eq. / Equation / Formula 编号引用兜底
    eq_refs = re.findall(
        r'\b(?:Eq(?:uation)?s?|Formula)s?\.?\s*[:\-\.]?\s*\(?([A-Za-z]{0,2}\d+[A-Za-z]?|[IVXLC]{1,6})\)?',
        text,
        re.IGNORECASE,
    )
    unique_eq_refs = len(set(ref.lower() for ref in eq_refs))

    return max(equation_count, unique_eq_refs, old_simple_count)


def count_paragraphs(text):
    """统计段落数量（空行块优先，无空行时行级兜底）"""
    if not text:
        return 0

    # 旧规则基线（保证不比旧版少）
    old_paragraphs = text.split('\n\n')
    old_count = len([p for p in old_paragraphs if p.strip() and len(p.strip()) > 20])

    # 方法1：按空行块统计
    blocks = re.split(r'\n\s*\n+', text)
    block_count = 0
    for b in blocks:
        s = b.strip()
        if len(s) < 20:
            continue
        if not re.search(r'[A-Za-z]', s):
            continue
        block_count += 1

    # 方法2：无空行/密集文本兜底（偏上界）
    line_count = 0
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 60:
            continue
        if s.startswith('#'):
            continue
        if '|' in s:
            continue
        if re.match(r'^\s*(\[\d+\]|\d+[.)]|[-*•])', s):
            continue
        if re.search(r'[A-Za-z]', s):
            line_count += 1

    # 大多数论文的空行分段是可信的；仅在空行分段太少时才启用行级兜底
    if block_count >= 20:
        return max(block_count, old_count)
    return max(block_count, line_count, old_count)


def count_citations_extended(text):
    """
    扩展的Citations统计 - 支持所有引用格式
    """
    if not text:
        return 0
    
    # 移除公式和代码块避免误识别
    text = re.sub(r'\$[^\$]{1,100}\$', '', text)
    text = re.sub(r'```[^`]+```', '', text, flags=re.DOTALL)
    
    all_cites = set()
    
    # 1. APA格式 - 完整匹配（包含作者名）
    # (Smith, 2020), (Smith & Jones, 2020), (Smith et al., 2020)
    apa_patterns = re.findall(r'\([^)]{1,200}\d{4}[^)]{0,50}\)', text)
    for citation in apa_patterns:
        # 检查是否包含作者名（大写字母开头）
        if re.search(r'[A-Z][a-z]+', citation) and re.search(r'\d{4}', citation):
            years = re.findall(r'\d{4}', citation)
            author_part = citation.split(',')[0] if ',' in citation else citation[:20]
            for year in years:
                all_cites.add(f"APA_{author_part[:15]}_{year}")
    
    # 2. IEEE格式 - 方括号数字
    # [1], [2], [123]
    ieee = re.findall(r'\[(\d{1,3})\]', text)
    for num in ieee:
        all_cites.add(f"IEEE_{num}")
    
    # 3. 括号数字格式 - (1), (2), (157)  ✨ NEW
    # 用于Numbered References的引用
    # 但要排除 (2018)这样的年份
    paren_numbers = re.findall(r'\((\d{1,3})\)', text)
    for num in paren_numbers:
        # 只保留1-999的数字（不是年份）
        if int(num) < 1980:  # 年份通常>=1980
            all_cites.add(f"NUM_{num}")
    
    # 4. 纯年份格式（谨慎匹配）
    # (2018), (2007) - 只匹配独立的年份，不在其他文本中
    year_only_pattern = r'(?<![A-Za-z])\((\d{4})\)(?![A-Za-z])'
    year_only = re.findall(year_only_pattern, text)
    for year in year_only:
        # 只保留1980-2025之间的年份（学术论文范围）
        if 1980 <= int(year) <= 2025:
            all_cites.add(f"YEAR_{year}")
    
    return len(all_cites)


def count_words(text):
    """统计单词数量"""
    if not text:
        return 0
    clean = re.sub(r'```[\s\S]*?```', ' ', text)
    split_words = len(clean.split())
    regex_words = len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?", clean))
    return max(split_words, regex_words)


def count_sentences(text):
    """统计句子数量（缩写保护 + 双路径，偏上界）"""
    if not text:
        return 0

    # 路径1：原始正则（快速）
    regex_count = len(re.findall(r'[.!?]+(?:\s+|$)', text))

    # 路径2：保护学术缩写后切分
    protected = text
    protected = re.sub(r'\b([A-Z])\.', r'\1<prd>', protected)  # 作者姓名缩写
    for abbr in [
        r'e\.g\.', r'i\.e\.', r'et al\.', r'Fig\.', r'Figs\.', r'Eq\.', r'Eqs\.',
        r'Dr\.', r'Prof\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'vs\.', r'ca\.', r'vol\.',
        r'Sec\.', r'Sect\.', r'App\.', r'Ref\.', r'Refs\.'
    ]:
        protected = re.sub(abbr, lambda m: m.group(0).replace('.', '<prd>'), protected, flags=re.IGNORECASE)
    protected = re.sub(r'(\d)\.(\d)', r'\1<prd>\2', protected)  # 小数点

    parts = re.split(r'[.!?]+', protected)
    split_count = 0
    for part in parts:
        s = part.replace('<prd>', '.').strip()
        if len(s) < 5:
            continue
        if not re.search(r'[A-Za-z]', s):
            continue
        split_count += 1

    return max(regex_count, split_count)


def _is_reference_heading(line):
    """判断是否为References标题（避免误匹配Reference standard等）"""
    if not line:
        return False
    # 兼容加粗/斜体/下划线样式标题，如 "**References**"
    line_norm = line.strip()
    line_norm = re.sub(r'^\s*>+\s*', '', line_norm)  # blockquote
    line_norm = re.sub(r'^\s*[-*]\s*', '', line_norm)  # list marker
    line_norm = re.sub(r'^\s*[*_`~]+\s*', '', line_norm)
    line_norm = re.sub(r'\s*[*_`~]+\s*$', '', line_norm)

    heading_re = re.compile(
        r'^\s*#{1,6}\s*(?:\d+\.?\s*)?'
        r'(References|Bibliography|Works?\s+Cited|Literature\s+Cited|Reference\s+List|References\s+and\s+Notes)\b'
        r'\s*:?.*$',
        re.IGNORECASE
    )
    plain_re = re.compile(
        r'^\s*(?:\d+\.?\s*)?'
        r'(References|Bibliography|Works?\s+Cited|Literature\s+Cited|Reference\s+List|References\s+and\s+Notes)\b'
        r'\s*:?.*$',
        re.IGNORECASE
    )
    return bool(
        heading_re.match(line) or plain_re.match(line)
        or heading_re.match(line_norm) or plain_re.match(line_norm)
    )


def _extract_references_section(text):
    """定位References章节（支持Markdown标题与纯文本标题）"""
    if not text:
        return ""

    lines = text.splitlines()

    candidate_idxs = []
    for i, line in enumerate(lines):
        if _is_reference_heading(line):
            candidate_idxs.append(i)

    if not candidate_idxs:
        return ""

    # 通常References在文末，优先取最后一个匹配
    idx = candidate_idxs[-1]
    start = idx + 1

    # 处理Setext风格标题（References + 下划线）
    if start < len(lines) and re.match(r'^\s*[-=]{3,}\s*$', lines[start]):
        start += 1

    # 跳过空行
    while start < len(lines) and not lines[start].strip():
        start += 1

    end_heading_re = re.compile(r'^\s*#{1,6}\s+.+')
    tail_heading_re = re.compile(
        r'^\s*(?:#\s*)?(Appendix|Acknowledg(?:e)?ments?|Funding|Conflicts?\s+of\s+Interest|'
        r'Supplementary|Supporting\s+Information|Author\s+Contributions?|Endnotes?)\b',
        re.IGNORECASE
    )

    end = len(lines)
    for j in range(start, len(lines)):
        line = lines[j]
        if end_heading_re.match(line) or tail_heading_re.match(line):
            end = j
            break

    return "\n".join(lines[start:end]).strip()


def _extract_references_section_fallback(text):
    """无标题时的保守兜底：在尾部查找参考文献块"""
    if not text:
        return ""
    lines = text.splitlines()
    tail = lines[-600:] if len(lines) > 600 else lines
    non_empty = [ln for ln in tail if ln.strip()]
    if not non_empty:
        return ""

    # 先尝试在尾部定位“编号相对连续”的参考文献起点
    num_line_re = re.compile(r'^\s*(\d{1,4})(?:[.)]|\s|\u00A0)')
    start_idx = None
    window = 80
    for i in range(0, max(1, len(tail) - window)):
        nums = []
        for ln in tail[i:i + window]:
            m = num_line_re.match(ln)
            if not m:
                continue
            n = int(m.group(1))
            if not _filter_ref_num(n):
                continue
            nums.append(n)
        uniq = sorted(set(nums))
        if len(uniq) < 8:
            continue
        near = 0
        for n in uniq:
            if (n - 1 in uniq) or (n + 1 in uniq):
                near += 1
        if near >= 6 and (uniq[-1] - uniq[0] >= 7):
            start_idx = i
            break

    if start_idx is not None:
        candidate = "\n".join(tail[start_idx:]).strip()
        if candidate:
            return candidate

    # 无编号APA风格：在尾部查找高密度“作者+年份”区域
    apa_like_re = re.compile(r'^\s*[A-Z][^\\n]{0,260}\b(19|20)\d{2}[a-z]?\b')
    start_idx_apa = None
    win = 120
    for i in range(0, max(1, len(tail) - win)):
        chunk = tail[i:i + win]
        apa_like = 0
        for ln in chunk:
            s = ln.strip()
            if not s:
                continue
            if not apa_like_re.match(s):
                continue
            low = s.lower()
            if ',' in s or 'et al' in low or 'doi' in low or 'http' in low:
                apa_like += 1
        if apa_like >= 20:
            start_idx_apa = i
            break
    if start_idx_apa is not None:
        candidate = "\n".join(tail[start_idx_apa:]).strip()
        if candidate:
            return candidate

    ref_like = 0
    for ln in non_empty:
        if re.match(r'^\s*\[\d{1,4}\]', ln):
            ref_like += 1
        elif re.match(r'^\s*\d{1,4}[.)．]\s*', ln):
            ref_like += 1
        elif re.match(r'^\s*\d{1,4}\s+[A-Za-z]', ln):
            ref_like += 1
        elif re.match(r'^\s*[A-Z].*\(\d{4}\)', ln):
            ref_like += 1

    ratio = ref_like / max(1, len(non_empty))
    if ref_like >= 8 and ratio >= 0.2:
        return "\n".join(tail).strip()
    return ""


def _validate_numbered(nums, max_ratio=3.0, max_extra=10, require_low_start=False):
    """对编号序列做简单校验，避免误报"""
    if not nums:
        return 0
    nums = sorted(set(int(n) for n in nums))
    count = len(nums)
    max_num = nums[-1]
    min_num = nums[0]
    if require_low_start and min_num > 3:
        return 0
    if max_num > count * max_ratio + max_extra:
        return 0
    return count


def _filter_ref_num(num, max_num=2000):
    """过滤可能是年份或异常的大数字"""
    if num <= 0 or num > max_num:
        return False
    if 1900 <= num <= 2099:
        return False
    return True


def _normalize_ocr_numbers(text):
    """修正常见OCR数字误识别（只用于编号检测）"""
    if not text:
        return text
    # 行首或空白后：l/I -> 1
    text = re.sub(r'(?m)^(\s*)[lI](?=\d)', r'\g<1>1', text)
    text = re.sub(r'(?<=\s)[lI](?=\d)', '1', text)
    # O/o -> 0（数字邻近）
    text = re.sub(r'(?<=\d)[oO](?=\d)', '0', text)
    text = re.sub(r'(?<=\s)[oO](?=\d)', '0', text)
    text = re.sub(r'(?<=\d)[oO](?=\s|[.)])', '0', text)
    # 行首常见 lO/IO -> 10
    text = re.sub(r'(?m)^(\s*)[lI][oO](?=\s|[.)])', r'\g<1>10', text)
    return text


def _collect_reference_numbers(section):
    """收集References中的编号（含行首与行内的编号）"""
    nums = set()
    section_norm = _normalize_ocr_numbers(section)

    # 行首编号
    patterns_line = [
        r'^\s*\[(\d{1,4})\]',
        r'^\s*(\d{1,4})[\.．]\s*',
        r'^\s*(\d{1,4})\)\s',
        r'^\s*[\(（](\d{1,3})[\)）]',
        r'^\s*(\d{1,4})\s*(?:[A-Za-z][^\n]{0,120},|https?://)',  # 支持无空格，过滤作者单位噪声
    ]
    for pat in patterns_line:
        for n in re.findall(pat, section_norm, re.MULTILINE):
            try:
                n = int(n)
            except ValueError:
                continue
            if _filter_ref_num(n):
                nums.add(n)

    # 行内多条参考文献（无换行情况）
    patterns_inline = [
        r'(?<!\d)(\d{1,4})[\.．]\s*[A-Za-z][^\n]{0,120},',
        r'(?<!\d)(\d{1,4})\)\s+[A-Za-z][^\n]{0,120},',
    ]
    for pat in patterns_inline:
        for n in re.findall(pat, section_norm):
            try:
                n = int(n)
            except ValueError:
                continue
            if _filter_ref_num(n):
                nums.add(n)

    # 行内方括号编号（支持单行多条参考文献：[1] ... [2] ...）
    for n in re.findall(r'\[(\d{1,4})\]', section_norm):
        try:
            n = int(n)
        except ValueError:
            continue
        if _filter_ref_num(n):
            nums.add(n)

    return nums


def _fallback_quality_ok(section):
    """无标题兜底时，要求有足够的文献特征，避免把目录/编号列表当作参考文献"""
    if not section:
        return False
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
    if len(lines) < 20:
        return False

    prefix_re = re.compile(r'^(\[\d{1,4}\]|\d{1,4}[.)]\s|\d{1,4}\s*[A-Za-z]|[A-Z].*\(\d{4}\))')
    year_re = re.compile(r'\b(19|20)\d{2}[a-z]?\b')

    pref_cnt = 0
    bib_cnt = 0
    for ln in lines:
        if not prefix_re.match(ln):
            continue
        pref_cnt += 1
        low = ln.lower()
        has_year = bool(year_re.search(ln))
        has_hint = ('doi' in low) or ('et al' in low) or ('http://' in low) or ('https://' in low)
        has_author_mark = (',' in ln and len(ln) > 20)
        if has_year or has_hint or has_author_mark:
            bib_cnt += 1

    if pref_cnt >= 8 and bib_cnt >= 6:
        return True

    # 纯APA无编号：只要尾部文献密度足够高也认为有效
    apa_dense = 0
    for ln in lines:
        if len(ln) < 40 or not re.match(r'^[A-Z]', ln):
            continue
        if not year_re.search(ln):
            continue
        low = ln.lower()
        if ',' in ln or 'et al' in low or 'doi' in low or 'http' in low:
            apa_dense += 1
    return apa_dense >= 20


def _trust_max_sequence(
    nums,
    min_max=20,
    window=12,
    min_hits=4,
    min_adjacent_pairs=1,
    top_search=120,
):
    """若高位序号存在局部连续簇，信任该簇的最大序号（偏向取大值）"""
    if not nums:
        return None

    arr = sorted(set(int(n) for n in nums if _filter_ref_num(int(n))))
    if not arr:
        return None

    # 从高位往下找第一个满足“局部连续”条件的候选值。
    candidates = arr[-top_search:]
    for candidate in reversed(candidates):
        if candidate < min_max:
            continue
        local = [n for n in arr if candidate - window <= n <= candidate]
        hits = len(local)
        adjacent_pairs = sum(1 for i in range(1, len(local)) if local[i] - local[i - 1] == 1)
        if hits >= min_hits and adjacent_pairs >= min_adjacent_pairs:
            return candidate

    return None


def _count_apa_unumbered_fallback(section):
    """统计无编号APA风格参考文献（仅用于fallback）"""
    if not section:
        return 0
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
    year_re = re.compile(r'\b(19|20)\d{2}[a-z]?\b')
    count = 0
    for ln in lines:
        if len(ln) < 40:
            continue
        if not re.match(r'^[A-Z]', ln):
            continue
        if not year_re.search(ln):
            continue
        low = ln.lower()
        if ',' in ln or 'et al' in low or 'doi' in low or 'http' in low:
            count += 1
    return count


def _normalize_ocr_year_tokens(text):
    """纠正常见OCR年份误识别，如 2O21/20l9 -> 2021/2019"""
    if not text:
        return text

    def _repl(m):
        token = m.group(0)
        normalized = token.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        if normalized.isdigit():
            y = int(normalized)
            if 1800 <= y <= 2100:
                return normalized
        return token

    return re.sub(r"\b[12][0-9OoIl]{3}\b", _repl, text)


def _is_apa_reference_fragment(fragment):
    """判断文本片段是否像一条APA参考文献"""
    if not fragment:
        return False
    s = fragment.strip()
    if len(s) < 35:
        return False

    year_re = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
    if not year_re.search(s):
        return False

    low = s.lower()
    has_author_head = bool(re.search(r"^(?:\*\s*)?[A-Z][A-Za-z'’\-]{1,50},\s*[A-Z]", s))
    has_hint = (
        "," in s
        or "et al" in low
        or "doi" in low
        or "http" in low
        or "journal" in low
        or "proceedings" in low
        or "press" in low
    )
    return has_author_head or has_hint


def _count_apa_references_robust(section):
    """
    统计无编号APA参考文献（鲁棒版）
    兼容：
    1) OCR年份误识别（2O21）
    2) 单行多条文献粘连（无换行）
    """
    if not section:
        return 0

    text = _normalize_ocr_year_tokens(section)
    text = text.replace(r"\*", "*")
    text = text.replace("–", "-")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0

    # 基础：按行计数
    line_count = sum(1 for ln in lines if _is_apa_reference_fragment(ln))

    # 增强：对单行多条文献进行切分
    split_pat = re.compile(
        r"(?<=[.;])\s+(?=(?:\*\s*)?[A-Z][A-Za-z'’\-]{1,40},\s*[A-Z])"
    )
    split_count = 0
    for ln in lines:
        fragments = split_pat.split(ln)
        split_count += sum(1 for frag in fragments if _is_apa_reference_fragment(frag))

    return max(line_count, split_count)


def count_references(text):
    """
    统计References - 增强版 (Robust)
    支持：
    1. IEEE [1]
    2. Numbered Dot 1.
    3. Numbered Space 1 Author (如 Engineering_11)
    4. Numbered Paren 1) 或 (1)/(（1）)（支持混合括号，如 Biology_14）
    5. APA
    6. References标题支持Markdown与纯文本
    """
    section = _extract_references_section(text)
    used_fallback = False
    if not section:
        section = _extract_references_section_fallback(text)
        used_fallback = True
    if not section:
        return 0
    if used_fallback and not _fallback_quality_ok(section):
        return 0

    section_norm = _normalize_ocr_numbers(section)

    # 1. IEEE格式 [1]
    ieee_nums = re.findall(r'^\s*\[(\d{1,4})\]', section_norm, re.MULTILINE)
    ieee = len(set(ieee_nums))

    # 2. Numbered Dot 1.
    numbered_dot_nums = re.findall(r'^\s*(\d{1,4})[\.．]\s*', section_norm, re.MULTILINE)
    numbered_dot = _validate_numbered(numbered_dot_nums, max_ratio=2.5, max_extra=20, require_low_start=True)

    # 3. Numbered Space 1 Author (行首)
    numbered_space_nums = re.findall(r'^\s*(\d{1,4})\s*[A-Za-z]', section_norm, re.MULTILINE)
    numbered_space = _validate_numbered(numbered_space_nums, max_ratio=2.5, max_extra=20, require_low_start=True)

    # 4. Numbered Paren 1) / (1) / （1）
    numbered_paren_close_nums = re.findall(r'^\s*(\d{1,4})\)\s', section_norm, re.MULTILINE)
    numbered_paren_close = _validate_numbered(numbered_paren_close_nums, max_ratio=2.5, max_extra=20, require_low_start=True)

    # 括号编号只在“行首括号编号”证据足够时启用，避免把APA年份误判成编号
    paren_start_nums = re.findall(r'^\s*[\(（](\d{1,3})[\)）]', section_norm, re.MULTILINE)
    paren_start_nums = [n for n in paren_start_nums if int(n) < 300]
    if len(set(paren_start_nums)) >= 5:
        paren_nums = re.findall(r'[\(（](\d{1,3})[\)）]', section_norm)
        paren_nums = [n for n in paren_nums if int(n) < 300]
        paren_count = _validate_numbered(paren_nums, max_ratio=3.0, max_extra=10, require_low_start=True)
        if paren_count == 0:
            paren_count = _validate_numbered(paren_start_nums, max_ratio=3.0, max_extra=10, require_low_start=True)
    else:
        paren_count = 0

    # 5. APA格式（含OCR年份修正 + 单行多条文献切分）
    apa_robust = _count_apa_references_robust(section)
    apa_unumbered = _count_apa_unumbered_fallback(section) if used_fallback else 0
    apa_count = max(apa_robust, apa_unumbered)

    # 6. 最大序号连续性校验（优先信任）
    nums_all = _collect_reference_numbers(section_norm)
    inline_numbered_count = _validate_numbered(
        nums_all,
        max_ratio=3.0,
        max_extra=20,
        require_low_start=True,
    )
    trusted_max = _trust_max_sequence(nums_all)
    inline_bracket_hits = len(set(re.findall(r"\[(\d{1,4})\]", section_norm)))
    strong_numbered_evidence = (
        ieee >= 5
        or numbered_dot >= 5
        or numbered_space >= 5
        or numbered_paren_close >= 5
        or len(set(paren_start_nums)) >= 5
        or inline_bracket_hits >= 8
    )
    if trusted_max is not None and strong_numbered_evidence:
        return trusted_max

    return max(
        ieee,
        numbered_dot,
        numbered_space,
        numbered_paren_close,
        paren_count,
        inline_numbered_count,
        apa_count,
    )


def process_paper(paper_path):
    """处理单篇论文，收集所有指标"""
    try:
        if get_exclusion_reason(paper_path.stem):
            return None

        with open(paper_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        return {
            'file_name': paper_path.stem[:50],
            'discipline': _extract_subject(paper_path, paper_path.stem),
            'img': count_images(text),
            'tab': count_tables(text),
            'eq': count_equations(text),
            'para': count_paragraphs(text),
            'words': count_words(text),
            'sent': count_sentences(text),
            'citation': count_citations_extended(text),
            'reference': count_references(text),
            'characters': len(text),
        }
    except Exception as e:
        print(f"Error processing {paper_path.name}: {e}")
        return None


def main():
    print("=" * 80)
    print("📊 完整统计 V2 - 扩展Citation匹配")
    print("=" * 80)
    
    base_dir = Path("Dataset%20final/Dataset final")
    papers = [p for p in base_dir.rglob("*.md") 
              if not p.name.startswith('.') and p.parent.name == 'auto']
    included = []
    excluded_by_reason = {
        "manual_excluded": [],
        "book_review": [],
        "summary_missing": [],
        "summary_mismatch": [],
        "parse_failed": [],
    }
    for p in papers:
        reason = get_exclusion_reason(p.stem)
        if reason:
            excluded_by_reason.setdefault(reason, []).append(p)
        else:
            included.append(p)

    papers = included
    excluded_count = sum(len(v) for v in excluded_by_reason.values())
    print(f"\n总论文数(过滤后): {len(papers)}")
    print(f"排除论文数: {excluded_count}")
    for reason in ["manual_excluded", "book_review", "summary_missing", "summary_mismatch", "parse_failed"]:
        if excluded_by_reason.get(reason):
            print(f"  - {reason}: {len(excluded_by_reason[reason])}")
    
    print("\n📌 Citation支持的格式:")
    print("  1. APA完整格式: (Smith, 2020), (Smith & Jones, 2020), (Smith et al., 2020)")
    print("  2. IEEE格式: [1], [2], [123]")
    print("  3. 纯年份格式: (2018), (2007) ✨ NEW")
    print("  4. 复合引用: (Smith, 2020; Jones, 2021)")
    
    results = []
    for paper in tqdm(papers, desc="处理中"):
        result = process_paper(paper)
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv("COMPLETE_EXTENDED_CITATION_DATA.csv", index=False, encoding='utf-8')
    print(f"\n📁 保存至: COMPLETE_EXTENDED_CITATION_DATA.csv")

    # 按用户要求：ref<20(含0)不参与统计
    df_ref_ge20 = df[df["reference"] >= 20].copy()
    df_ref_ge20.to_csv("COMPLETE_EXTENDED_CITATION_DATA_REF_GE20.csv", index=False, encoding='utf-8')

    subject_summary = (
        df_ref_ge20
        .groupby("discipline", as_index=False)
        .agg(
            reference_mean=("reference", "mean"),
            citation_mean=("citation", "mean"),
        )
        .sort_values("reference_mean", ascending=False)
    )
    subject_summary.to_csv("SUBJECT_REFERENCE_MEAN_REF_GE20.csv", index=False, encoding='utf-8')
    
    print(f"\n✅ 完成！成功处理: {len(results)}/{len(papers)} 篇")
    print(f"📁 过滤数据(ref>=20): COMPLETE_EXTENDED_CITATION_DATA_REF_GE20.csv")
    print(f"📁 学科均值(ref>=20): SUBJECT_REFERENCE_MEAN_REF_GE20.csv")
    print(f"📊 参与均值统计论文数(ref>=20): {len(df_ref_ge20)} / {len(df)}")
    
    # 统计摘要
    print("\n" + "=" * 80)
    print("📈 Citation统计对比:")
    print("=" * 80)
    
    print(f"\nCitation平均值: {df['citation'].mean():.1f}")
    print(f"Reference平均值: {df['reference'].mean():.1f}")
    print(f"Citation/Reference比例: {100*df['citation'].mean()/df['reference'].mean():.1f}%")
    if len(df_ref_ge20) > 0:
        print(f"Reference平均值(ref>=20): {df_ref_ge20['reference'].mean():.1f}")


if __name__ == "__main__":
    main()
