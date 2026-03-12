#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计Dataset final中所有文档的各项指标
包括：图片数量、公式数量、表格数量、引用数量、outline数量、reference数量、句子数量
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def count_images_equations_tables_from_json(content_list_path, middle_path=None):
    """从JSON文件准确统计图片、公式、表格数量"""
    stats = {'images': 0, 'equations': 0, 'tables': 0, 'citations': 0}

    try:
        # 从content_list.json获取文本内容
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)

        full_text = ""
        table_count = 0
        image_count = 0

        for item in content_data:
            if item.get('type') == 'text' and 'text' in item:
                full_text += item['text'] + " "
            elif item.get('type') == 'table':
                table_count += 1
            elif item.get('type') == 'image':
                image_count += 1

        # 统计公式（更准确的LaTeX检测）
        latex_patterns = [
            r'\$\$.*?\$\$',  # 块级公式
            r'\$.*?\$',     # 行内公式（添加此模式）
            r'\\[.*?\\]',   # LaTeX块级
            r'\\\([^)]*?\\\)', # LaTeX行内
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{gather\}.*?\\end\{gather\}',
            r'\\begin\{multline\}.*?\\end\{multline\}',
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}'
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, full_text, re.DOTALL)
            stats['equations'] += len(matches)
            # 移除匹配的公式，避免重复计数
            full_text = re.sub(pattern, '', full_text, flags=re.DOTALL)

        # 统计图片和表格
        stats['images'] = image_count  # 从JSON中直接获取的图片数量
        stats['tables'] = table_count  # 从JSON中直接获取的表格数量

        # 额外统计文本中的LaTeX表格（以防万一）
        table_patterns = [
            r'\\begin\{tabular\}.*?\\end\{tabular\}',
            r'\\begin\{table\}.*?\\end\{table\}',
            r'\\begin\{tabulary\}.*?\\end\{tabulary\}',
            r'\\begin\{longtable\}.*?\\end\{longtable\}'
        ]

        for pattern in table_patterns:
            matches = re.findall(pattern, full_text, re.DOTALL)
            stats['tables'] += len(matches)
            full_text = re.sub(pattern, '', full_text, flags=re.DOTALL)

        # 统计图片（包括JSON中的图片和文本引用）
        stats['images'] = image_count  # 从JSON中直接获取的图片数量

        # 额外统计文本中的图片引用（Fig. Figure等）
        image_patterns = [
            r'\bFig\.\s*\d+', r'\bFigs\.\s*\d+',
            r'\bFigure\s+\d+', r'\bFigures\s+\d+',
            r'!\[.*?\]\(.*?\)',  # Markdown图片
            r'\\includegraphics'  # LaTeX图片
        ]

        for pattern in image_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            stats['images'] += len(matches)

        # 统计引用
        citation_patterns = [
            r'\[\d+\]', r'\[\d+,\s*\d+\]',  # [1], [1,2]
            r'\(\w+\s+\d{4}\)',  # (Author 2020)
            r'\b[A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\)'  # Smith (2020)
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, full_text)
            stats['citations'] += len(matches)

        return stats, full_text

    except Exception as e:
        print(f"JSON统计出错 {content_list_path}: {e}")
        return stats, ""

def count_outline_references_from_json(content_list_path):
    """从JSON统计大纲和参考文献"""
    try:
        with open(content_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        outline_items = 0
        references = 0

        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '').strip()

                # 检测大纲项目（数字编号或 bullet points）
                if re.match(r'^\d+\.|\*\s+|\-\s+|•\s+', text):
                    outline_items += 1

                # 检测参考文献条目
                if re.match(r'^\[\d+\]|\d+\.\s+[A-Z]', text):
                    references += 1

        return outline_items, references

    except Exception as e:
        print(f"统计大纲引用出错 {content_list_path}: {e}")
        return 0, 0

def count_images_in_markdown(content):
    """统计markdown内容中的图片数量 - 修复版本"""
    total_images = 0

    # 1. 统计实际的图片插入（Markdown语法和HTML）
    image_insertion_patterns = [
        r'!\[.*?\]\(.*?\)',  # Markdown图片: ![alt](url)
        r'<img[^>]*>',       # HTML img标签
        r'\\includegraphics', # LaTeX includegraphics
    ]

    for pattern in image_insertion_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        total_images += len(matches)

    # 2. 统计图片引用（Fig., Figure等）
    # 注意：这里可能与实际图片插入有重叠，但通常学术论文中
    # Fig. 引用通常对应实际插入的图片
    fig_patterns = [
        r'\bFig\.\s*\d+',      # Fig. 1
        r'\bFigure\s+\d+',     # Figure 1
        r'\bFig\.s\s*\d+',     # Figs. 1,2
        r'\bFigures\s+\d+',    # Figures 1,2
    ]

    # 计算图片引用数量，但避免与已统计的图片重复
    fig_references = 0
    for pattern in fig_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        fig_references += len(matches)

    # 如果图片引用明显多于实际插入，补充差额
    # （假设每个Fig引用都对应一个实际图片）
    if fig_references > total_images:
        total_images = fig_references

    return total_images

def count_equations_in_markdown(content):
    """统计markdown内容中的公式数量 - 修复版本"""
    total_equations = 0

    # 按照优先级处理，避免重复统计
    # 1. 先处理块级公式（优先级最高）
    block_patterns = [
        r'\$\$.*?\$\$',           # $$...$$ 块级公式
        r'\\\[.*?\\\]',          # \[...\] LaTeX块公式
        r'\\begin\{equation\}.*?\\end\{equation\}',  # equation环境
        r'\\begin\{align\}.*?\\end\{align\}',        # align环境
        r'\\begin\{gather\}.*?\\end\{gather\}',      # gather环境
        r'\\begin\{multline\}.*?\\end\{multline\}',  # multline环境
        r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # eqnarray环境
        r'\\begin\{array\}.*?\\end\{array\}',        # array环境
        r'\\begin\{aligned\}.*?\\end\{aligned\}',    # aligned环境
    ]

    # 逐个处理块级公式，避免重叠
    for pattern in block_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        total_equations += len(matches)
        # 移除已统计的公式，避免被后续模式重复统计
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # 2. 再处理行内公式（优先级较低）
    inline_patterns = [
        r'\\\([^)]*?\\\)',        # \(...\) LaTeX行内公式
        r'(?<!\\)\$[^$]+(?<!\\)\$',  # $...$ 行内公式（排除转义的\$）
    ]

    for pattern in inline_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        total_equations += len(matches)
        # 移除已统计的公式
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    return total_equations

def count_tables_in_markdown(content):
    """统计markdown内容中的表格数量 - 修复版本"""
    total_tables = 0

    # 1. 统计LaTeX表格环境
    latex_table_patterns = [
        r'\\begin\{tabular\}.*?\\end\{tabular\}',     # tabular环境
        r'\\begin\{table\}.*?\\end\{table\}',         # table环境
        r'\\begin\{tabulary\}.*?\\end\{tabulary\}',   # tabulary环境
        r'\\begin\{longtable\}.*?\\end\{longtable\}', # longtable环境
    ]

    for pattern in latex_table_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        total_tables += len(matches)
        # 移除已统计的表格
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # 2. 统计Markdown表格
    # Markdown表格的标准特征是分割线：|---| 或 |:---:| 等
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 检查是否是分割线（包含|和至少一个-）
        if '|' in line and re.search(r'-{2,}', line):
            # 验证前后行是否是有效的表格行
            prev_line = lines[i-1].strip() if i > 0 else ""
            next_line = lines[i+1].strip() if i < len(lines)-1 else ""

            # 检查上一行是否是表头或数据行
            if '|' in prev_line and prev_line.count('|') == line.count('|'):
                total_tables += 1
                # 跳过这个表格的所有行（直到下一个空行或非表格行）
                i += 1
                while i < len(lines) and '|' in lines[i].strip():
                    i += 1
                continue

        i += 1

    return total_tables

def count_sentences_in_markdown(content):
    """统计markdown内容中的句子数量 - 修复版本"""
    # 移除markdown标记
    content = re.sub(r'#+\s*', '', content)  # 移除标题
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # 移除链接
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 移除图片
    content = re.sub(r'\*\*.*?\*\*', '', content)  # 移除粗体
    content = re.sub(r'\*.*?\*', '', content)  # 移除斜体
    content = re.sub(r'`.*?`', '', content)  # 移除代码

    # 移除LaTeX公式（避免公式中的标点影响分割）
    content = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)
    content = re.sub(r'\\\$.*?\$', '', content)
    content = re.sub(r'\\\[.*?\\\]', '', content, flags=re.DOTALL)
    content = re.sub(r'\\\([^)]*?\\\)', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', content, flags=re.DOTALL)

    # 智能句子分割：避免被常见缩写误导
    # 1. 保护常见缩写词
    protected_abbrevs = [
        r'\bFig\.', r'\bFigs\.', r'\bEq\.', r'\bEqs\.',
        r'\be\.g\.', r'\bi\.e\.', r'\bet\s+al\.', r'\bvs\.',
        r'\bDr\.', r'\bProf\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.',
        r'\bca\.', r'\bvs\.', r'\bno\.', r'\bvol\.', r'\bVol\.',
        r'\bCh\.', r'\bSec\.', r'\bSect\.', r'\bApp\.',
        r'\bRef\.', r'\bRefs\.', r'\bCf\.', r'\bcp\.',
        r'\bspp\.', r'\bvar\.', r'\bsubsp\.', r'\bsubvar\.',
        r'\bgen\.', r'\bspec\.', r'\bsp\.'
    ]

    # 临时替换缩写词中的点号
    abbrev_placeholders = {}
    for i, abbrev in enumerate(protected_abbrevs):
        placeholder = f"ABBREV{i}PLACEHOLDER"
        abbrev_placeholders[placeholder] = abbrev.replace(r'\.', 'ABBREV_DOT')
        content = re.sub(abbrev, placeholder, content, flags=re.IGNORECASE)

    # 2. 保护数字后的点号（避免切分版本号、页码等）
    # 3.14 -> 3ABBREV_DECIMAL14, p. 123 -> pABBREV_DOT123
    content = re.sub(r'(\d)\.(\d)', r'\1ABBREV_DECIMAL\2', content)
    content = re.sub(r'(\w)\.(\d)', r'\1ABBREV_DOT\2', content)

    # 3. 句子分割：基于句号、问号、感叹号
    # 使用更简单的分割策略，避免复杂的lookbehind
    sentences = re.split(r'[.!?]+', content)

    # 清理和过滤句子
    clean_sentences = []
    for sentence in sentences:
        # 恢复占位符
        for placeholder, original in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, original.replace('ABBREV_DOT', '.'))

        sentence = sentence.replace('ABBREV_DECIMAL', '.')
        sentence = sentence.replace('ABBREV_DOT', '.')

        # 清理空白字符
        sentence = sentence.strip()

        # 过滤条件：非空、长度合适、不只是标点
        if (sentence and
            len(sentence) >= 5 and  # 至少5个字符
            len(sentence) <= 1000 and  # 不超过1000字符（避免异常长句子）
            not re.match(r'^[^\w]*$', sentence) and  # 不只是标点
            not sentence.isdigit()):  # 不只是数字

            clean_sentences.append(sentence)

    return len(clean_sentences)

def count_words_in_markdown(content):
    """统计markdown内容中的单词数量"""
    # 移除markdown标记
    content = re.sub(r'#+\s*', '', content)  # 移除标题
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # 移除链接
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 移除图片
    content = re.sub(r'\*\*.*?\*\*', '', content)  # 移除粗体
    content = re.sub(r'\*.*?\*', '', content)  # 移除斜体
    content = re.sub(r'`.*?`', '', content)  # 移除代码

    # 移除LaTeX公式
    content = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)  # 块级公式
    content = re.sub(r'\$.*?\$', '', content)  # 行内公式
    content = re.sub(r'\\[.*?\\]', '', content, flags=re.DOTALL)  # LaTeX块公式
    content = re.sub(r'\\\(.*?\\\)', '', content, flags=re.DOTALL)  # LaTeX行公式
    content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', content, flags=re.DOTALL)  # LaTeX环境

    # 移除标点符号和特殊字符，保留字母、数字和空格
    content = re.sub(r'[^\w\s]', ' ', content)

    # 分割单词
    words = content.split()
    # 过滤太短的词（少于2个字符）
    words = [word for word in words if len(word) >= 2]

    return len(words)

def count_sections_in_markdown(content):
    """统计markdown内容中的章节数量（以#开头的标题）"""
    lines = content.split('\n')
    section_count = 0
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            section_count += 1
    return section_count

def count_paragraphs_in_markdown(content):
    """统计markdown内容中的段落数量（通过空行分隔）"""
    # 移除markdown标题和代码块
    content = re.sub(r'^#.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

    # 分割成段落（连续的非空行）
    paragraphs = []
    current_paragraph = []

    for line in content.split('\n'):
        line = line.strip()
        if line:  # 非空行
            current_paragraph.append(line)
        else:  # 空行
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []

    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    # 过滤太短的段落
    paragraphs = [p for p in paragraphs if len(p.strip()) > 10]
    return len(paragraphs)

def count_characters_in_markdown(content):
    """统计markdown内容中的字符数量（清理后的纯文本）"""
    # 移除markdown标记
    content = re.sub(r'#+\s*', '', content)  # 移除标题
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # 移除链接
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 移除图片
    content = re.sub(r'\*\*.*?\*\*', '', content)  # 移除粗体
    content = re.sub(r'\*.*?\*', '', content)  # 移除斜体
    content = re.sub(r'`.*?`', '', content)  # 移除代码

    # 移除LaTeX公式
    content = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)  # 块级公式
    content = re.sub(r'\$.*?\$', '', content)  # 行内公式
    content = re.sub(r'\\[.*?\\]', '', content, flags=re.DOTALL)  # LaTeX块公式
    content = re.sub(r'\\\(.*?\\\)', '', content, flags=re.DOTALL)  # LaTeX行公式
    content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', content, flags=re.DOTALL)  # LaTeX环境

    # 移除多余的空白字符
    content = re.sub(r'\s+', ' ', content).strip()

    return len(content)

def count_links_in_markdown(content):
    """统计markdown内容中的链接数量"""
    # 匹配markdown链接 [text](url)
    link_pattern = r'\[.*?\]\(.*?\)'
    links = re.findall(link_pattern, content)
    return len(links)

def count_lists_in_markdown(content):
    """统计markdown内容中的列表项数量"""
    lines = content.split('\n')
    list_item_count = 0

    for line in lines:
        line = line.strip()
        # 检查是否有列表标记
        if re.match(r'^[-*+]\s', line) or re.match(r'^\d+\.\s', line):
            list_item_count += 1
        # 检查特殊符号·
        elif '·' in line and len(line.split('·')[0].strip()) < 10:  # 避免误计句子中的·
            list_item_count += 1

    return list_item_count

def analyze_json_file(json_path):
    """分析JSON文件的结构化数据"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 统计outline数量
        outline_count = 0
        if 'outline' in data:
            outline = data['outline']
            if isinstance(outline, list):
                outline_count = len(outline)

        # 统计references数量
        references_count = 0
        if 'references' in data:
            references = data['references']
            if isinstance(references, list):
                references_count = len(references)

        return outline_count, references_count

    except Exception as e:
        print(f"  警告: 解析JSON文件 {json_path} 时出错: {e}")
        return 0, 0

def analyze_markdown_file(md_path):
    """分析markdown文件的内容"""
    try:
        with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # 统计各项指标
        image_count = count_images_in_markdown(content)
        equation_count = count_equations_in_markdown(content)
        table_count = count_tables_in_markdown(content)
        sentence_count = count_sentences_in_markdown(content)
        word_count = count_words_in_markdown(content)
        section_count = count_sections_in_markdown(content)
        paragraph_count = count_paragraphs_in_markdown(content)
        character_count = count_characters_in_markdown(content)
        link_count = count_links_in_markdown(content)
        list_count = count_lists_in_markdown(content)

        return image_count, equation_count, table_count, sentence_count, word_count, section_count, paragraph_count, character_count, link_count, list_count

    except Exception as e:
        import traceback
        print(f"  警告: 读取markdown文件 {md_path} 时出错: {e}")
        print(f"    错误详情: {traceback.format_exc()}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def find_file_pairs(original_dir):
    """查找所有markdown文件"""
    file_pairs = []

    for root, dirs, files in os.walk(original_dir):
        # 只处理包含.md文件的目录
        for file in files:
            if file.endswith('.md') and not file.startswith('.'):
                md_path = os.path.join(root, file)
                base_name = file.replace('.md', '')

                # 计算相对路径作为学科标识
                rel_path = os.path.relpath(root, original_dir)
                # 取第一级目录作为学科名称
                discipline = rel_path.split(os.sep)[0] if os.sep in rel_path else 'Unknown'

                file_pairs.append((base_name, discipline, None, md_path))

    return file_pairs

def main():
    print("=" * 80)
    print("📊 开始统计 Dataset final 数据集各项指标")
    print("=" * 80 + "\n")

    # 设置路径
    base_dir = Path(__file__).parent
    original_dir = base_dir / "Dataset%20final" / "Dataset final"

    if not original_dir.exists():
        print(f"❌ 错误: 目录不存在 {original_dir}")
        return

    # 查找所有文件对
    print("📂 正在扫描文件...")
    file_pairs = find_file_pairs(original_dir)
    print(f"✅ 找到 {len(file_pairs)} 个文档\n")

    # 统计结果
    results = []

    for idx, (base_name, subdir, json_path, md_path) in enumerate(tqdm(file_pairs, desc="统计文档")):
        # 使用修正的JSON统计方法
        if json_path:
            # 使用JSON文件进行准确统计
            stats, full_text = count_images_equations_tables_from_json(json_path, None)
            image_count = stats['images']
            equation_count = stats['equations']
            table_count = stats['tables']
            citation_count = stats['citations']

            # 从JSON统计大纲和参考文献
            outline_count, references_count = count_outline_references_from_json(json_path)

            # 统计其他文本指标
            sentence_count = count_sentences_in_markdown(full_text)
            word_count = count_words_in_markdown(full_text)
            character_count = count_characters_in_markdown(full_text)
            section_count, paragraph_count = count_sections_paragraphs(full_text)
            link_count, list_count = count_links_lists(full_text)
        else:
            # 如果没有JSON文件，使用Markdown分析（降级方案）
            image_count, equation_count, table_count, sentence_count, word_count, section_count, paragraph_count, character_count, link_count, list_count = analyze_markdown_file(md_path)
            outline_count, references_count = 0, 0
            citation_count = 0

        # 调试输出
        if len(results) < 3:  # 只打印前3个文件的调试信息
            print(f"    调试: {base_name[:30]}... 图片={image_count}, 公式={equation_count}, 表格={table_count}, 句子={sentence_count}, 单词={word_count}, 章节={section_count}, 段落={paragraph_count}")

        # 记录结果
        result = {
            'file_name': base_name,
            'directory': subdir,
            'images': image_count,
            'equations': equation_count,
            'tables': table_count,
            'citations': citation_count if 'citation_count' in locals() else 0,
            'references': references_count,
            'outline_items': outline_count,
            'sentences': sentence_count,
            'words': word_count,
            'sections': section_count,
            'paragraphs': paragraph_count,
            'characters': character_count,
            'links': link_count,
            'lists': list_count
        }
        results.append(result)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 计算总计
    totals = df[['images', 'equations', 'tables', 'citations', 'references', 'outline_items', 'sentences', 'words', 'sections', 'paragraphs', 'characters', 'links', 'lists']].sum()

    # 保存详细结果
    output_file = "dataset_statistics_detailed.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n📄 详细统计已保存到: {output_file}")

    # 保存汇总结果
    summary_file = "dataset_statistics_summary.csv"
    summary_df = pd.DataFrame([totals], index=['Total'])
    summary_df.to_csv(summary_file)
    print(f"📄 汇总统计已保存到: {summary_file}")

    # 打印汇总报告
    print("\n" + "=" * 80)
    print("📊 数据集统计汇总报告")
    print("=" * 80)
    print(f"总文档数量: {len(results)}")
    print(f"总图片数量: {totals['images']}")
    print(f"总公式数量: {totals['equations']}")
    print(f"总表格数量: {totals['tables']}")
    print(f"总引用数量: {totals['citations']}")
    print(f"总参考文献数量: {totals['references']}")
    print(f"总大纲项目数量: {totals['outline_items']}")
    print(f"总句子数量: {totals['sentences']}")
    print(f"总单词数量: {totals['words']}")
    print(f"总章节数量: {totals['sections']}")
    print(f"总段落数量: {totals['paragraphs']}")
    print(f"总字符数量: {totals['characters']}")
    print(f"总链接数量: {totals['links']}")
    print(f"总列表项数量: {totals['lists']}")

    # 按学科统计
    print("\n📈 按学科统计:")
    discipline_stats = df.groupby('directory')[['images', 'equations', 'tables', 'citations', 'references', 'outline_items', 'sentences', 'words', 'sections', 'paragraphs', 'characters', 'links', 'lists']].sum()
    for discipline, stats in discipline_stats.iterrows():
        print(f"  {discipline}: 文档{len(df[df['directory']==discipline])}个, "
              f"图片{int(stats['images'])}, 公式{int(stats['equations'])}, "
              f"表格{int(stats['tables'])}, 引用{int(stats['citations'])}, "
              f"参考文献{int(stats['references'])}, 大纲{int(stats['outline_items'])}, "
              f"句子{int(stats['sentences'])}, 单词{int(stats['words'])}")

    print("\n✅ 统计完成！")

if __name__ == "__main__":
    main()