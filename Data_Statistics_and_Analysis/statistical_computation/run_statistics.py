#!/usr/bin/env python3
"""
完整统计脚本 - 使用改进的方法统计所有论文
生成 improved_paper_stats.csv
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from improved_statistics import (
    count_citations_improved,
    count_references_improved,
    count_images_improved,
    count_tables_improved,
    count_equations_improved,
    count_outline_items_improved
)


def count_sentences(text):
    """统计句子数"""
    if not text:
        return 0
    
    # 保护学术缩写
    protected_abbrevs = [
        r'\bFig\.', r'\bFigs\.', r'\bEq\.', r'\bEqs\.', r'\be\.g\.', r'\bi\.e\.',
        r'\bet\s+al\.', r'\bvs\.', r'\bDr\.', r'\bProf\.', r'\bMr\.', r'\bMrs\.', 
        r'\bMs\.', r'\bvol\.', r'\bCh\.', r'\bSec\.', r'\bRef\.', r'\bRefs\.'
    ]
    
    for i, abbrev in enumerate(protected_abbrevs):
        placeholder = f"ABBREV{i}PLACEHOLDER"
        text = re.sub(abbrev, placeholder, text, flags=re.IGNORECASE)
    
    # 处理小数点
    text = re.sub(r'(\d)\.(\d)', r'\1DECIMAL\2', text)
    
    # 分割句子
    sentences = re.split(r'[.!?]+', text)
    clean_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) >= 5 and len(sentence) <= 2000:
            if not re.match(r'^[^\w]*$', sentence) and not sentence.isdigit():
                clean_sentences.append(sentence)
    
    return len(clean_sentences)


def count_words(text):
    """统计单词数"""
    if not text:
        return 0
    
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    words = clean_text.split()
    words = [w for w in words if len(w) >= 2]
    return len(words)


def count_characters(text):
    """统计字符数"""
    if not text:
        return 0
    return len(re.sub(r'\s+', ' ', text).strip())


def count_links(text):
    """统计超链接数"""
    link_pattern = r'\[.*?\]\(.*?\)'
    links = re.findall(link_pattern, text)
    return len(links)


def count_lists(text):
    """统计列表项数"""
    lines = text.split('\n')
    list_count = 0
    for line in lines:
        line = line.strip()
        if re.match(r'^[-*+]\s', line) or re.match(r'^\d+\.\s', line):
            list_count += 1
        elif '·' in line and len(line.split('·')[0].strip()) < 10:
            list_count += 1
    return list_count


def process_paper(paper_path):
    """处理单篇论文"""
    try:
        # 读取文件
        with open(paper_path, 'r', encoding='utf-8', errors='replace') as f:
            full_text = f.read()
        
        # 基本信息
        discipline = paper_path.parts[-4]
        file_name = paper_path.stem
        
        # 尝试加载JSON数据
        auto_dir = paper_path.parent
        content_list_path = None
        for f in os.listdir(auto_dir):
            if f.endswith('_content_list.json'):
                content_list_path = auto_dir / f
                break
        
        # 统计所有指标
        result = {
            'file_name': file_name[:50],
            'discipline': discipline,
            'images': count_images_improved(content_list_path, full_text) if content_list_path else 0,
            'equations': count_equations_improved(full_text),
            'tables': count_tables_improved(content_list_path, full_text) if content_list_path else 0,
            'citations': count_citations_improved(full_text),
            'references': count_references_improved(full_text),
            'sentences': count_sentences(full_text),
            'words': count_words(full_text),
            'characters': count_characters(full_text),
            'sections': count_outline_items_improved(full_text),
            'paragraphs': len([p for p in full_text.split('\n\n') if len(p.strip()) > 50]),
            'links': count_links(full_text),
            'lists': count_lists(full_text),
        }
        
        return result
        
    except Exception as e:
        print(f"\n❌ 处理失败 {paper_path.name}: {e}")
        return None


def main():
    print("=" * 100)
    print("📊 完整论文统计 - 使用改进的统计方法")
    print("=" * 100)
    
    # 查找所有论文
    base_dir = Path("Dataset%20final/Dataset final")
    all_papers = list(base_dir.rglob("*.md"))
    all_papers = [p for p in all_papers if not p.name.startswith('.') and p.parent.name == 'auto']
    
    print(f"\n找到 {len(all_papers)} 篇论文")
    
    # 按学科统计
    disciplines = {}
    for paper in all_papers:
        disc = paper.parts[-4]
        disciplines[disc] = disciplines.get(disc, 0) + 1
    
    print(f"涵盖 {len(disciplines)} 个学科:")
    for disc, count in sorted(disciplines.items()):
        print(f"  - {disc}: {count} 篇")
    
    print(f"\n开始统计...")
    
    # 批量处理
    results = []
    for paper in tqdm(all_papers, desc="处理进度"):
        result = process_paper(paper)
        if result:
            results.append(result)
    
    print(f"\n✅ 成功处理 {len(results)}/{len(all_papers)} 篇论文")
    
    # 保存到CSV
    df = pd.DataFrame(results)
    output_file = "improved_paper_stats.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n📄 数据已保存到: {output_file}")
    
    # 显示汇总统计
    print("\n" + "=" * 100)
    print("📈 汇总统计")
    print("=" * 100)
    
    print(f"\n整体平均值:")
    numeric_cols = ['images', 'equations', 'tables', 'citations', 'references', 
                    'sentences', 'words', 'characters', 'sections', 'paragraphs', 'links', 'lists']
    
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            if col in ['words', 'characters']:
                print(f"  {col:15s}: 平均={mean_val:8.0f}, 中位数={median_val:8.0f}")
            else:
                print(f"  {col:15s}: 平均={mean_val:6.1f}, 中位数={median_val:6.1f}")
    
    # 按学科统计
    print(f"\n按学科统计 (平均值):")
    print(f"{'学科':20s} {'Refs':>6s} {'Cites':>6s} {'Eqs':>6s} {'Sents':>7s} {'Words':>8s}")
    print("─" * 70)
    
    discipline_stats = df.groupby('discipline').agg({
        'references': 'mean',
        'citations': 'mean',
        'equations': 'mean',
        'sentences': 'mean',
        'words': 'mean'
    }).round(1)
    
    for discipline, stats in discipline_stats.iterrows():
        print(f"{discipline:20s} {stats['references']:6.1f} {stats['citations']:6.1f} "
              f"{stats['equations']:6.1f} {stats['sentences']:7.1f} {stats['words']:8.0f}")
    
    print("\n" + "=" * 100)
    print(f"✅ 统计完成！时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
