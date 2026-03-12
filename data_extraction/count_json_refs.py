#!/usr/bin/env python3
"""
统计Human_json_cleaned_v3中的References数量
按学科计算平均值
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def count_references_from_cleaned_json(json_path):
    """从清洗好的JSON文件中统计references数量"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # references是一个列表
        references = data.get('references', [])
        return len(references)
        
    except Exception as e:
        print(f"Error processing {json_path.name}: {e}")
        return 0


def main():
    print("=" * 80)
    print("📚 统计Human_json_cleaned_v3中的References")
    print("=" * 80)
    
    base_dir = Path("Dataset%20final/Human_json_cleaned_v3")
    
    # 收集所有JSON文件
    json_files = list(base_dir.rglob("*.json"))
    
    print(f"\n找到 {len(json_files)} 个JSON文件\n")
    
    results = []
    
    for json_path in tqdm(json_files, desc="处理中"):
        # 提取学科（父目录名）
        subject = json_path.parent.name
        
        # 文件名
        file_name = json_path.stem.replace('_split', '')
        
        # 统计references
        ref_count = count_references_from_cleaned_json(json_path)
        
        results.append({
            'subject': subject,
            'file_name': file_name[:50],
            'reference_count': ref_count,
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存详细结果
    df.to_csv('CLEANED_JSON_REFERENCES.csv', index=False, encoding='utf-8')
    
    print(f"\n✅ 完成！处理了 {len(df)} 篇论文")
    print(f"\n📁 保存至: CLEANED_JSON_REFERENCES.csv")
    
    # 统计摘要
    print("\n" + "=" * 80)
    print("📊 统计摘要")
    print("=" * 80)
    
    print(f"\nReferences平均值: {df['reference_count'].mean():.1f}")
    print(f"References中位数: {df['reference_count'].median():.1f}")
    print(f"References最大值: {df['reference_count'].max()}")
    print(f"References最小值: {df['reference_count'].min()}")
    
    # 按学科统计
    print("\n" + "=" * 80)
    print("📈 按学科统计")
    print("=" * 80)
    
    subject_stats = df.groupby('subject').agg({
        'reference_count': ['mean', 'median', 'min', 'max', 'count']
    }).round(1)
    
    subject_stats.columns = ['平均值', '中位数', '最小值', '最大值', '论文数']
    subject_stats = subject_stats.sort_values('平均值', ascending=False)
    
    print("\n各学科References统计:")
    print(subject_stats)
    
    # 保存学科汇总
    subject_summary = df.groupby('subject')['reference_count'].mean().round(1)
    subject_summary = subject_summary.sort_values(ascending=False)
    subject_summary.to_csv('CLEANED_JSON_SUBJECT_MEAN_REFS.csv', header=['reference_mean'])
    
    print(f"\n✅ 学科平均值已保存至: CLEANED_JSON_SUBJECT_MEAN_REFS.csv")
    
    # 无references的论文
    no_refs = df[df['reference_count'] == 0]
    if len(no_refs) > 0:
        print(f"\n⚠️  有 {len(no_refs)} 篇论文没有references ({100*len(no_refs)/len(df):.1f}%)")
    
    return df, subject_stats


if __name__ == "__main__":
    df, subject_stats = main()
