#!/usr/bin/env python3
"""
提取所有学术论文的特征数据并生成详细的Markdown表格
"""

import os
import sys
import re
import pandas as pd
from pathlib import Path

# 导入现有函数
sys.path.append('.')
from final_scientific_analysis import (
    read_md, count_md_features, extract_features_from_md,
    calculate_citation_coverage, calculate_structure_gini,
    extract_citations_robust, count_sentences
)

def extract_all_paper_data(base_path):
    """
    提取所有论文的详细数据

    Args:
        base_path (str): survey_papers_with_pdf文件夹的路径

    Returns:
        pd.DataFrame: 包含所有论文数据的DataFrame
    """
    all_data = []

    # 获取所有学科文件夹
    subject_dirs = [d for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]

    print(f"发现 {len(subject_dirs)} 个学科文件夹")

    for subject in sorted(subject_dirs):
        subject_path = os.path.join(base_path, subject)
        print(f"处理学科: {subject}")

        # 获取该学科的所有md文件
        md_files = [f for f in os.listdir(subject_path)
                   if f.endswith('.md') and not f.startswith('.')]

        for md_file in sorted(md_files):
            md_path = os.path.join(subject_path, md_file)

            try:
                # 提取特征
                features = extract_features_from_md(md_path)

                if features is None:
                    print(f"  ⚠️  跳过文件: {md_file} (无法读取)")
                    continue

                # 计算密度特征
                sentence_count = max(features.get('Sentence_count', 1), 1)
                density_features = {
                    'Images_density': round(features.get('Images_count', 0) / sentence_count * 100, 4),
                    'Equations_density': round(features.get('Equations_count', 0) / sentence_count * 100, 4),
                    'Tables_density': round(features.get('Tables_count', 0) / sentence_count * 100, 4),
                    'Citations_density': round(features.get('Citations_count', 0) / sentence_count * 100, 4),
                }

                # 整理数据
                paper_data = {
                    'Subject': subject,
                    'Paper': md_file.replace('.md', ''),

                    # 密度特征
                    **density_features,

                    # 计数特征
                    'Outline_no': features.get('Outline_count', 0),
                    'Reference_no': features.get('Reference_count', 0),
                    'Sentence_no': features.get('Sentence_count', 0),

                    # 原始计数
                    'Images_count': features.get('Images_count', 0),
                    'Equations_count': features.get('Equations_count', 0),
                    'Tables_count': features.get('Tables_count', 0),
                    'Citations_count': features.get('Citations_count', 0),

                    # 深度指标
                    'Structure_Gini': round(features.get('Structure_Gini', 0), 4),
                    'Cit_Coverage': round(features.get('Cit_Coverage', 0), 4),
                    'Info_Density': round(features.get('Info_Density', 0), 4),
                }

                all_data.append(paper_data)
                print(f"  ✅ {md_file}")

            except Exception as e:
                print(f"  ❌ 处理失败 {md_file}: {str(e)}")
                continue

    df = pd.DataFrame(all_data)

    # 按学科和论文名称排序
    if not df.empty:
        df = df.sort_values(['Subject', 'Paper']).reset_index(drop=True)

    return df

def generate_markdown_table(df, output_path):
    """
    生成详细的Markdown表格

    Args:
        df (pd.DataFrame): 论文数据
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("# 学术论文特征数据汇总\n\n")
        f.write("## 概述\n\n")
        f.write(f"- **总论文数**: {len(df)}\n")
        f.write(f"- **学科数**: {df['Subject'].nunique()}\n")
        f.write(f"- **每个学科论文数**: {len(df) // df['Subject'].nunique() if df['Subject'].nunique() > 0 else 0}\n\n")

        # 统计摘要
        f.write("## 统计摘要\n\n")

        # 密度特征统计
        f.write("### 密度特征 (每句平均)\n\n")
        density_cols = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
        density_stats = df[density_cols].describe()

        f.write("| 特征 | 平均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
        f.write("|------|--------|--------|--------|--------|--------|\n")

        for col in density_cols:
            mean_val = density_stats.loc['mean', col]
            std_val = density_stats.loc['std', col]
            min_val = density_stats.loc['min', col]
            max_val = density_stats.loc['max', col]
            median_val = density_stats.loc['50%', col]

            f.write(f"| {col.replace('_density', '').replace('_', ' ').title()} | ")
            f.write(f"{mean_val:.4f} | ")
            f.write(f"{std_val:.4f} | ")
            f.write(f"{min_val:.4f} | ")
            f.write(f"{max_val:.4f} | ")
            f.write(f"{median_val:.4f} |")
            f.write(" |\n")

        f.write("\n")

        # 计数特征统计
        f.write("### 计数特征\n\n")
        count_cols = ['Outline_no', 'Reference_no', 'Sentence_no']
        count_stats = df[count_cols].describe()

        f.write("| 特征 | 平均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
        f.write("|------|--------|--------|--------|--------|--------|\n")

        for col in count_cols:
            mean_val = count_stats.loc['mean', col]
            std_val = count_stats.loc['std', col]
            min_val = count_stats.loc['min', col]
            max_val = count_stats.loc['max', col]
            median_val = count_stats.loc['50%', col]

            f.write(f"| {col.replace('_no', '').replace('_', ' ').title()} | ")
            f.write(f"{mean_val:.1f} | ")
            f.write(f"{std_val:.1f} | ")
            f.write(f"{min_val:.0f} | ")
            f.write(f"{max_val:.0f} | ")
            f.write(f"{median_val:.0f} |")
            f.write(" |\n")

        f.write("\n")

        # 深度指标统计
        f.write("### 深度指标\n\n")
        depth_cols = ['Structure_Gini', 'Cit_Coverage', 'Info_Density']
        depth_stats = df[depth_cols].describe()

        f.write("| 特征 | 平均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
        f.write("|------|--------|--------|--------|--------|--------|\n")

        for col in depth_cols:
            mean_val = depth_stats.loc['mean', col]
            std_val = depth_stats.loc['std', col]
            min_val = depth_stats.loc['min', col]
            max_val = depth_stats.loc['max', col]
            median_val = depth_stats.loc['50%', col]

            f.write(f"| {col.replace('_', ' ').title()} | ")
            f.write(f"{mean_val:.4f} | ")
            f.write(f"{std_val:.4f} | ")
            f.write(f"{min_val:.4f} | ")
            f.write(f"{max_val:.4f} | ")
            f.write(f"{median_val:.4f} |")
            f.write(" |\n")

        f.write("\n")

        # 详细数据表格
        f.write("## 详细数据\n\n")
        f.write("### 完整特征表格\n\n")

        # 选择要显示的列
        display_cols = [
            'Subject', 'Paper',
            'Images_density', 'Equations_density', 'Tables_density', 'Citations_density',
            'Outline_no', 'Reference_no', 'Sentence_no',
            'Images_count', 'Equations_count', 'Tables_count', 'Citations_count',
            'Structure_Gini', 'Cit_Coverage', 'Info_Density'
        ]

        # 按学科分组显示
        for subject in sorted(df['Subject'].unique()):
            subject_data = df[df['Subject'] == subject]

            f.write(f"#### {subject}\n\n")

            f.write("| 论文 | 图片密度 | 公式密度 | 表格密度 | 引用密度 | 大纲数 | 参考文献数 | 句子数 | 图片数 | 公式数 | 表格数 | 引用数 | 结构基尼 | 引用覆盖率 | 信息密度 |\n")
            f.write("|------|----------|----------|----------|----------|--------|------------|--------|--------|--------|--------|--------|----------|------------|----------|\n")

            for _, row in subject_data.iterrows():
                f.write(f"| {row['Paper']} | ")
                f.write(f"{row['Images_density']:.4f} | ")
                f.write(f"{row['Equations_density']:.4f} | ")
                f.write(f"{row['Tables_density']:.4f} | ")
                f.write(f"{row['Citations_density']:.4f} | ")
                f.write(f"{int(row['Outline_no'])} | ")
                f.write(f"{int(row['Reference_no'])} | ")
                f.write(f"{int(row['Sentence_no'])} | ")
                f.write(f"{int(row['Images_count'])} | ")
                f.write(f"{int(row['Equations_count'])} | ")
                f.write(f"{int(row['Tables_count'])} | ")
                f.write(f"{int(row['Citations_count'])} | ")
                f.write(f"{row['Structure_Gini']:.4f} | ")
                f.write(f"{row['Cit_Coverage']:.4f} | ")
                f.write(f"{row['Info_Density']:.4f} |")
                f.write(" |\n")

            f.write("\n")

        # 技术说明
        f.write("## 技术说明\n\n")
        f.write("### 特征定义\n\n")
        f.write("- **密度特征**: 每句平均特征数量\n")
        f.write("- **计数特征**: 文档中的绝对数量\n")
        f.write("- **深度指标**:\n")
        f.write("  - **结构基尼**: 章节长度分布的均衡度 (0-1, 越接近0越均衡)\n")
        f.write("  - **引用覆盖率**: 包含引用的句子比例 (%)\n")
        f.write("  - **信息密度**: 图片+表格+公式的综合密度\n\n")

        f.write("### 数据来源\n\n")
        f.write("- **学科**: 10个主要学术领域\n")
        f.write("- **每学科论文数**: 10篇\n")
        f.write("- **总计**: 100篇学术论文\n")
        f.write("- **数据提取**: 自动解析Markdown格式学术论文\n\n")

        f.write("### 生成时间\n\n")
        from datetime import datetime
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("- **数据版本**: v2.0 (优化后的引用和公式提取)\n\n")

def main():
    """主函数"""
    print("=" * 60)
    print("  学术论文数据提取工具")
    print("  生成详细的Markdown数据汇总")
    print("=" * 60)

    base_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/survey_papers_with_pdf"
    output_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/paper_data_summary.md"

    if not os.path.exists(base_path):
        print(f"❌ 错误: 数据路径不存在 - {base_path}")
        return

    print("\n开始提取论文数据...")
    print("-" * 40)

    # 提取数据
    df = extract_all_paper_data(base_path)

    if df.empty:
        print("❌ 错误: 未提取到任何数据")
        return

    print(f"\n✅ 成功提取 {len(df)} 篇论文的数据")

    # 生成Markdown文件
    print(f"\n生成Markdown汇总文件...")
    generate_markdown_table(df, output_path)

    print(f"✅ Markdown文件已保存: {output_path}")

    # 显示统计信息
    print(f"\n📊 数据概览:")
    print(f"- 总论文数: {len(df)}")
    print(f"- 学科数: {df['Subject'].nunique()}")
    print(f"- 平均引用密度: {df['Citations_density'].mean():.2f}")
    print(f"- 平均公式密度: {df['Equations_density'].mean():.2f}")
    print(f"- 平均图片密度: {df['Images_density'].mean():.2f}")

    print("\n" + "=" * 60)
    print("  数据提取完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
