#!/usr/bin/env python3
"""
最终增强版统计脚本
结合JSON和Markdown数据，提供最准确的统计结果
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

        # 统计表格（包括JSON中的表格和LaTeX表格）
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

def get_paper_structure_from_middle(middle_path):
    """从middle.json获取论文结构信息"""
    try:
        with open(middle_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        structure = {
            'pages': len(data.get('pdf_info', [])),
            'sections': 0,
            'paragraphs': 0
        }

        for page in data.get('pdf_info', []):
            for block in page.get('preproc_blocks', []):
                if block.get('type') == 'title':
                    structure['sections'] += 1
                if 'lines' in block:
                    structure['paragraphs'] += len(block['lines'])

        return structure

    except Exception as e:
        print(f"解析middle.json出错 {middle_path}: {e}")
        return {'pages': 0, 'sections': 0, 'paragraphs': 0}

def enhanced_sentence_count(text):
    """增强的句子计数，处理学术文本"""
    if not text:
        return 0

    # 保护学术缩写
    protected_abbrevs = [
        r'\bFig\.', r'\bFigs\.', r'\bEq\.', r'\bEqs\.', r'\be\.g\.', r'\bi\.e\.',
        r'\bet\s+al\.', r'\bvs\.', r'\bDr\.', r'\bProf\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.',
        r'\bca\.', r'\bvol\.', r'\bCh\.', r'\bSec\.', r'\bSect\.', r'\bApp\.', r'\bRef\.', r'\bRefs\.',
        r'\bCf\.', r'\bcp\.', r'\bspp\.', r'\bvar\.', r'\bsubsp\.', r'\bgen\.', r'\bspec\.', r'\bsp\.'
    ]

    abbrev_placeholders = {}
    for i, abbrev in enumerate(protected_abbrevs):
        placeholder = f"ABBREV{i}PLACEHOLDER"
        abbrev_placeholders[placeholder] = abbrev.replace(r'\.', 'ABBREV_DOT')
        text = re.sub(abbrev, placeholder, text, flags=re.IGNORECASE)

    # 处理小数点
    text = re.sub(r'(\d)\.(\d)', r'\1ABBREV_DECIMAL\2', text)
    text = re.sub(r'(\w)\.(\d)', r'\1ABBREV_DOT\2', text)

    # 分割句子
    sentences = re.split(r'[.!?]+', text)
    clean_sentences = []

    for sentence in sentences:
        # 恢复缩写
        for placeholder, original in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, original.replace('ABBREV_DOT', '.'))
        sentence = sentence.replace('ABBREV_DECIMAL', '.')
        sentence = sentence.replace('ABBREV_DOT', '.')

        sentence = sentence.strip()
        # 过滤有效句子
        if (sentence and len(sentence) >= 5 and len(sentence) <= 2000 and
            not re.match(r'^[^\w]*$', sentence) and not sentence.isdigit()):
            clean_sentences.append(sentence)

    return len(clean_sentences)

def process_paper_batch(paper_batch):
    """批量处理论文"""
    batch_results = []
    for paper in paper_batch:
        try:
            result = {
                'file_name': paper['name'][:50],
                'discipline': paper['discipline']
            }

            # 1. 从JSON获取准确的图片、公式、表格、引用统计
            if 'content_list' in paper['files']:
                counts, full_text = count_images_equations_tables_from_json(
                    paper['files']['content_list'], paper['files'].get('middle')
                )
                result.update(counts)

                # 使用JSON文本进行句子和单词统计
                result['sentences'] = enhanced_sentence_count(full_text)

                # 统计单词
                clean_text = re.sub(r'[^\w\s]', ' ', full_text)
                words = clean_text.split()
                words = [w for w in words if len(w) >= 2]
                result['words'] = len(words)

                # 字符数
                result['characters'] = len(re.sub(r'\s+', ' ', full_text).strip())

                # 统计链接数
                link_pattern = r'\[.*?\]\(.*?\)'
                links = re.findall(link_pattern, full_text)
                result['links'] = len(links)

                # 统计列表数
                lines = full_text.split('\n')
                list_count = 0
                for line in lines:
                    line = line.strip()
                    if re.match(r'^[-*+]\s', line) or re.match(r'^\d+\.\s', line):
                        list_count += 1
                    elif '·' in line and len(line.split('·')[0].strip()) < 10:
                        list_count += 1
                result['lists'] = list_count

            # 2. 获取大纲和参考文献信息
            if 'content_list' in paper['files']:
                outline_items, references = count_outline_references_from_json(paper['files']['content_list'])
                result['outline_items'] = outline_items
                result['references'] = references

            # 3. 获取结构信息
            if 'middle' in paper['files']:
                structure = get_paper_structure_from_middle(paper['files']['middle'])
                result.update(structure)

            # 4. 补充Markdown统计（如果JSON不可用）
            if 'markdown' in paper['files'] and not result.get('words'):
                try:
                    with open(paper['files']['markdown'], 'r', encoding='utf-8', errors='replace') as f:
                        md_content = f.read()
                    result['words'] = len([w for w in re.sub(r'[^\w\s]', ' ', md_content).split() if len(w) >= 2])
                    result['sentences'] = enhanced_sentence_count(md_content)
                    result['characters'] = len(re.sub(r'\s+', ' ', md_content).strip())
                except Exception as e:
                    pass

            batch_results.append(result)

        except Exception as e:
            continue

    return batch_results

def main():
    print("=" * 80)
    print("🎯 最终增强版统计脚本 (批处理优化)")
    print("结合JSON和Markdown数据的最准确统计 - 处理全部1000篇论文")
    print("=" * 80)

    base_dir = Path(__file__).parent / "Dataset%20final" / "Dataset final"

    # 找到所有论文
    papers = []
    for discipline in os.listdir(base_dir):
        discipline_path = base_dir / discipline
        if not os.path.isdir(discipline_path):
            continue

        for paper_dir in os.listdir(discipline_path):
            paper_path = discipline_path / paper_dir
            if not os.path.isdir(paper_path):
                continue

            auto_dir = paper_path / f"{paper_dir}/auto"
            if not os.path.exists(auto_dir):
                continue

            files = {}
            for file in os.listdir(auto_dir):
                if file.endswith('.md'):
                    files['markdown'] = auto_dir / file
                elif file.endswith('_content_list.json'):
                    files['content_list'] = auto_dir / file
                elif file.endswith('_middle.json'):
                    files['middle'] = auto_dir / file

            if files.get('content_list'):  # 优先使用JSON
                papers.append({
                    'name': paper_dir,
                    'discipline': discipline,
                    'files': files
                })

    print(f"找到 {len(papers)} 篇论文，开始批处理...")

    results = []

    # 使用批处理来提高效率
    batch_size = 50  # 每批处理50篇论文
    for i in tqdm(range(0, len(papers), batch_size), desc="批处理论文"):
        batch = papers[i:i + batch_size]
        batch_results = process_paper_batch(batch)
        results.extend(batch_results)

    print(f"\n成功处理了 {len(results)}/{len(papers)} 篇论文")

    # 保存结果
    df = pd.DataFrame(results)
    output_file = "final_enhanced_dataset_statistics_full.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n📄 最终统计结果已保存到: {output_file}")

    # 显示汇总统计
    print(f"\n📊 汇总统计 (全部{len(results)}篇论文):")
    numeric_cols = ['images', 'equations', 'tables', 'citations', 'sentences', 'words', 'characters', 'pages', 'sections', 'paragraphs', 'outline_items', 'references', 'links', 'lists']

    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            if col in ['words', 'characters', 'sentences']:
                print(f"  {col}: {mean_val:.0f}")
            else:
                print(f"  {col}: {mean_val:.1f}")

    # 按学科统计
    print("\n📈 按学科统计:")
    discipline_stats = df.groupby('discipline').agg({
        'words': 'mean',
        'sentences': 'mean',
        'pages': 'mean',
        'citations': 'mean'
    }).round(0)

    for discipline, stats in discipline_stats.iterrows():
        print(f"  {discipline}: 单词{stats['words']:.0f}, 句子{stats['sentences']:.0f}, 页面{stats['pages']:.0f}, 引用{stats['citations']:.0f}")

    print("\n✅ 最终增强统计完成！")
    print("💡 使用JSON数据和批处理确保了统计的准确性和效率")

if __name__ == "__main__":
    main()