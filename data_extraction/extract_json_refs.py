#!/usr/bin/env python3
"""
从JSON文件中提取References - 使用大模型方法
读取每篇论文的_content_list.json，提取并统计References
"""

import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def count_references_from_json(json_path):
    """从JSON文件中提取并统计References"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 拼接所有text内容
        full_text = ""
        for item in data:
            if isinstance(item, dict) and item.get('type') == 'text':
                full_text += item.get('text', '') + '\n'
        
        # 查找References章节
        ref_match = re.search(r'(?:^|\n)(References?|Bibliography)\s*\n', 
                             full_text, re.IGNORECASE | re.MULTILINE)
        
        if not ref_match:
            return 0, "无References章节", []
        
        # 提取References章节
        refs_section = full_text[ref_match.end():]
        
        # 方法1: IEEE格式 [1], [2], ...
        ieee_refs = re.findall(r'^\[(\d+)\]', refs_section, re.MULTILINE)
        ieee_count = len(set(ieee_refs))
        
        # 方法2: Numbered格式 1. Author, 2. Author, ...
        numbered_refs = re.findall(r'^\d+\.\s+[A-Z]', refs_section, re.MULTILINE)
        numbered_count = len(numbered_refs)
        
        # 方法3: APA格式 - Author (Year). Title...
        apa_refs = []
        for line in refs_section.split('\n'):
            line = line.strip()
            if (line and len(line) > 30 and 
                line[0].isupper() and 
                '(' in line and ')' in line and
                re.search(r'\d{4}', line)):
                apa_refs.append(line)
        apa_count = len(apa_refs)
        
        # 选择最大值
        counts = {
            'IEEE': ieee_count,
            'Numbered': numbered_count,
            'APA': apa_count
        }
        
        best_method = max(counts, key=counts.get)
        best_count = counts[best_method]
        
        # 提取示例（前3条）
        samples = []
        if best_method == 'IEEE' and ieee_refs:
            samples = [f"[{n}]" for n in list(set(ieee_refs))[:3]]
        elif best_method == 'APA' and apa_refs:
            samples = [ref[:80] for ref in apa_refs[:3]]
        elif best_method == 'Numbered':
            lines = refs_section.split('\n')
            for line in lines[:10]:
                if re.match(r'^\d+\.\s+[A-Z]', line):
                    samples.append(line[:80])
                    if len(samples) >= 3:
                        break
        
        return best_count, best_method, samples
        
    except Exception as e:
        print(f"Error processing {json_path.name}: {e}")
        return 0, f"错误: {str(e)}", []


def process_all_papers():
    """处理所有论文的JSON文件"""
    print("=" * 80)
    print("📚 从JSON文件提取References")
    print("=" * 80)
    
    base_dir = Path("Dataset%20final/Dataset final")
    
    # 查找所有_content_list.json文件
    json_files = list(base_dir.rglob("*_content_list.json"))
    json_files = [f for f in json_files if f.parent.name == 'auto']
    
    print(f"\n找到 {len(json_files)} 个JSON文件\n")
    
    results = []
    
    for json_path in tqdm(json_files, desc="处理中"):
        # 提取学科和文件名
        parts = json_path.parts
        discipline = parts[-4]
        file_name = json_path.stem.replace('_content_list', '')
        
        # 统计References
        ref_count, method, samples = count_references_from_json(json_path)
        
        results.append({
            'file_name': file_name[:50],
            'discipline': discipline,
            'json_reference_count': ref_count,
            'detection_method': method,
            'json_path': str(json_path)
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    df.to_csv('JSON_REFERENCES_COUNTS.csv', index=False, encoding='utf-8')
    
    print(f"\n✅ 完成！处理了 {len(df)} 篇论文")
    print(f"\n📁 保存至: JSON_REFERENCES_COUNTS.csv")
    
    # 统计摘要
    print("\n" + "=" * 80)
    print("📊 统计摘要")
    print("=" * 80)
    
    print(f"\nReferences平均值: {df['json_reference_count'].mean():.1f}")
    print(f"References中位数: {df['json_reference_count'].median():.1f}")
    print(f"References最大值: {df['json_reference_count'].max()}")
    print(f"References最小值: {df['json_reference_count'].min()}")
    
    # 检测方法统计
    print("\n检测方法分布:")
    method_counts = df['detection_method'].value_counts()
    for method, count in method_counts.items():
        print(f"  {method:15s}: {count:4d} 篇 ({100*count/len(df):.1f}%)")
    
    # 没有找到References的论文
    no_refs = df[df['json_reference_count'] == 0]
    if len(no_refs) > 0:
        print(f"\n⚠️  有 {len(no_refs)} 篇论文未找到References章节")
    
    return df


if __name__ == "__main__":
    df = process_all_papers()
