#!/usr/bin/env python3
"""
学科指纹画廊可视化系统 v5.0
多图布局雷达图，展示各学科的六维写作风格指纹

核心特色：
- 10个学科的雷达图画廊（2x5布局）
- 六维指标：图像/公式/表格/引用密度 + 结构均衡度 + 长度指数
- 统计层叠：均值线 + 标准差阴影
- 统一HUSL色彩系统
- 极简美学设计
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
import warnings
warnings.filterwarnings('ignore')

# 学科列表和颜色定义
SUBJECTS = ['Business', 'Education', 'Sociology', 'Physics', 'Engineering',
           'Computer Science', 'Medicine', 'Psychology', 'Environmental Science', 'Biology']

# HUSL颜色系统（备用）
HUSL_COLORS = [
    '#3B4252', '#BF616A', '#A3BE8C', '#EBCB8B', '#81A1C1',
    '#B48EAD', '#88C0D0', '#5E81AC', '#D08770', '#8FBCBB'
]

SUBJECT_COLORS = dict(zip(SUBJECTS, HUSL_COLORS))

def load_and_process_data():
    """
    加载和处理论文数据
    """
    try:
        # 导入现有数据处理函数
        import sys
        sys.path.append('.')
        from disciplinary_profile_analysis_v3 import process_all_md_files

        # 使用与v3脚本相同的数据路径
        base_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/survey_papers_with_pdf"

        if os.path.exists(base_path):
            df = process_all_md_files(base_path)
            if not df.empty and len(df) > 0:
                print(f"✅ 成功加载真实数据：{len(df)} 篇论文")
                return df

    except Exception as e:
        print(f"⚠️  无法加载真实数据: {e}")

    # 使用模拟数据作为演示
    print("✅ 使用模拟数据演示功能")
    np.random.seed(42)

    data = []
    for subject in SUBJECTS:
        for i in range(10):  # 每个学科10篇论文
            if subject == 'Physics':
                row = {
                    'Subject': subject,
                    'Images_density': np.random.beta(0.5, 2) * 2,
                    'Equations_density': np.random.beta(2, 1) * 60,
                    'Tables_density': np.random.beta(0.5, 2) * 2,
                    'Citations_density': np.random.beta(1.5, 1) * 25,
                    'Sentence_no': max(500, int(np.random.normal(2000, 400))),
                    'Structure_Gini': min(1, max(0, np.random.beta(2, 1) * 0.3 + 0.65))
                }
            elif subject == 'Business':
                row = {
                    'Subject': subject,
                    'Images_density': np.random.beta(1, 1) * 1.5,
                    'Equations_density': np.random.beta(0.5, 2) * 5,
                    'Tables_density': np.random.beta(2, 1) * 3,
                    'Citations_density': np.random.beta(2, 1) * 30,
                    'Sentence_no': max(300, int(np.random.normal(1500, 300))),
                    'Structure_Gini': min(1, max(0, np.random.beta(1.8, 1.2) * 0.25 + 0.6))
                }
            elif subject == 'Computer Science':
                row = {
                    'Subject': subject,
                    'Images_density': np.random.beta(1, 1) * 1,
                    'Equations_density': np.random.beta(1, 1) * 8,
                    'Tables_density': np.random.beta(1, 1) * 1.5,
                    'Citations_density': np.random.beta(1.8, 1.2) * 22,
                    'Sentence_no': max(250, int(np.random.normal(1300, 250))),
                    'Structure_Gini': min(1, max(0, np.random.beta(1.6, 1.4) * 0.25 + 0.62))
                }
            elif subject == 'Biology':
                row = {
                    'Subject': subject,
                    'Images_density': np.random.beta(1.5, 1) * 2,
                    'Equations_density': np.random.beta(0.8, 1.5) * 10,
                    'Tables_density': np.random.beta(1.2, 1) * 2.5,
                    'Citations_density': np.random.beta(1.2, 1.5) * 18,
                    'Sentence_no': max(400, int(np.random.normal(1700, 350))),
                    'Structure_Gini': min(1, max(0, np.random.beta(1.4, 1.6) * 0.25 + 0.64))
                }
            else:
                row = {
                    'Subject': subject,
                    'Images_density': np.random.beta(1, 1) * 1.5,
                    'Equations_density': np.random.beta(1, 1) * 12,
                    'Tables_density': np.random.beta(1, 1) * 2,
                    'Citations_density': np.random.beta(1.5, 1.5) * 22,
                    'Sentence_no': max(350, int(np.random.normal(1600, 320))),
                    'Structure_Gini': min(1, max(0, np.random.beta(1.5, 1.5) * 0.25 + 0.63))
                }

            data.append(row)

    df = pd.DataFrame(data)
    return df

def create_radar_chart(ax, subject_data, subject_name, color, global_max_values):
    """
    为单个学科创建定向高亮版雷达图 v5.1
    """
    # 六维指标
    categories = ['Images\nDensity', 'Equations\nDensity', 'Tables\nDensity',
                 'Citations\nDensity', 'Structure\nBalance', 'Length\nIndex']

    # 准备数据 - 使用全局最大值进行归一化
    values = []
    raw_values_list = []

    # 计算各指标的归一化值（基于全局最大值）
    for category in ['Images_density', 'Equations_density', 'Tables_density',
                    'Citations_density']:
        raw_values = subject_data[category].values
        raw_values_list.append(raw_values)

        # 使用全局最大值进行归一化
        global_max = global_max_values[category]
        if global_max > 0:
            normalized = raw_values.mean() / global_max  # 使用均值比上全局最大值
        else:
            normalized = 0.5
        values.append(min(normalized, 1.0))  # 确保不超过1

    # Structure Balance (1 - Gini系数)
    gini_values = subject_data['Structure_Gini'].values
    raw_values_list.append(1 - gini_values)  # 存储原始balance值
    balance_mean = (1 - gini_values).mean()
    global_max = global_max_values['Structure_Balance']
    if global_max > 0:
        normalized = balance_mean / global_max
    else:
        normalized = 0.5
    values.append(min(normalized, 1.0))

    # Length Index (句子总数)
    length_values = subject_data['Sentence_no'].values
    raw_values_list.append(length_values)
    length_mean = length_values.mean()
    global_max = global_max_values['Sentence_no']
    if global_max > 0:
        normalized = length_mean / global_max
    else:
        normalized = 0.5
    values.append(min(normalized, 1.0))

    # 计算标准差用于阴影
    std_values = []
    for i, category in enumerate(['Images_density', 'Equations_density', 'Tables_density',
                                 'Citations_density']):
        raw_values = raw_values_list[i]
        if len(raw_values) > 1:
            # 标准差基于原始值计算，然后归一化
            std_raw = raw_values.std()
            global_max = global_max_values[category]
            if global_max > 0:
                std_normalized = std_raw / global_max
            else:
                std_normalized = 0.05
            std_values.append(std_normalized)
        else:
            std_values.append(0.05)

    # Structure Balance标准差
    balance_values = raw_values_list[4]
    if len(balance_values) > 1:
        std_raw = balance_values.std()
        global_max = global_max_values['Structure_Balance']
        if global_max > 0:
            std_normalized = std_raw / global_max
        else:
            std_normalized = 0.05
        std_values.append(std_normalized)
    else:
        std_values.append(0.05)

    # Length Index标准差
    length_values = raw_values_list[5]
    if len(length_values) > 1:
        std_raw = length_values.std()
        global_max = global_max_values['Sentence_no']
        if global_max > 0:
            std_normalized = std_raw / global_max
        else:
            std_normalized = 0.05
        std_values.append(std_normalized)
    else:
        std_values.append(0.05)

    # 雷达图设置
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  # 闭合图形

    # 扩展值用于闭合
    values += values[:1]
    std_values += std_values[:1]

    # 绘制阴影区域（标准差范围）- 使用浅灰色
    upper_values = np.array(values) + np.array(std_values)
    lower_values = np.array(values) - np.array(std_values)
    lower_values = np.maximum(lower_values, 0)  # 不低于0

    ax.fill_between(angles, lower_values, upper_values, alpha=0.2, color='lightgray',
                   label='Standard Deviation Range')

    # 创建强度编码填充 - 得分越高，颜色越饱和
    for i in range(len(categories)):
        # 计算该维度的强度（得分越高，透明度越高）
        intensity = values[i]
        alpha_intensity = 0.15 + 0.5 * intensity  # 强度越高，越不透明

        # 在极坐标系统中使用fill_between创建扇形
        angle1 = angles[i]
        angle2 = angles[i+1]
        r = values[i]

        # 创建角度序列
        theta = np.linspace(angle1, angle2, 50)

        # 使用极坐标的fill_between方法
        ax.fill_between(theta, 0, r, alpha=alpha_intensity, color=color,
                       edgecolor=color, linewidth=1)

    # 绘制均值线
    ax.plot(angles, values, '-', linewidth=2.5, color=color, alpha=0.9)

    # 查找最强项并添加高亮标记
    max_idx = np.argmax(values[:-1])  # 排除闭合点的最后一个值
    max_value = values[max_idx]
    max_angle = angles[max_idx]

    # 计算原始值用于标注
    raw_max_values = [
        raw_values_list[0].mean(),  # Images_density
        raw_values_list[1].mean(),  # Equations_density
        raw_values_list[2].mean(),  # Tables_density
        raw_values_list[3].mean(),  # Citations_density
        raw_values_list[4].mean(),  # Structure_Balance
        raw_values_list[5].mean(),  # Sentence_no
    ]

    # 添加发光高亮圆点 - 在极坐标系统中
    # 在最高点位置添加标记
    highlight_r = max_value
    highlight_theta = max_angle

    # 外圈白色描边
    ax.scatter(highlight_theta, highlight_r, s=120, color='white', alpha=1.0, zorder=10)
    # 内圈彩色填充
    ax.scatter(highlight_theta, highlight_r, s=80, color=color, alpha=1.0, zorder=11,
              edgecolor='white', linewidth=3)

    # 添加数值标签 - 转换为数据坐标
    raw_value = raw_max_values[max_idx]
    if categories[max_idx] in ['Images\nDensity', 'Equations\nDensity', 'Tables\nDensity', 'Citations\nDensity']:
        label_text = '.1f'
    elif categories[max_idx] == 'Structure\nBalance':
        label_text = '.2f'
    else:  # Length Index
        label_text = '.0f'

    # 在极坐标系统中添加标签
    # 将极坐标转换为数据坐标进行标注
    offset_r = max_value + 0.1  # 稍微向外偏移
    offset_theta = max_angle

    # 使用annotate在正确的位置添加标签
    ax.annotate(label_text,
                xy=(offset_theta, offset_r), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9),
                fontweight='bold', zorder=12)

    # 极简轴向设置 - 只保留外环和主轴线
    ax.set_thetagrids([n * 60 for n in range(6)], labels=['']*6)  # 隐藏角度标签
    ax.set_rgrids([])  # 隐藏径向网格线

    # 只保留最外圆环
    ax.spines['polar'].set_visible(False)  # 隐藏外框
    # 手动绘制最外圆环
    outer_circle = plt.Circle((0, 0), 1, fill=False, color='lightgray', linewidth=1, alpha=0.5)
    ax.add_artist(outer_circle)

    # 绘制主轴线
    for angle in angles[:-1]:
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color='lightgray',
               linewidth=1, alpha=0.5)

    # 设置范围
    ax.set_ylim(0, 1)
    ax.set_rlim(0, 1)

    # 移除网格和刻度
    ax.grid(False)

    # 添加学科名称 - 拼写修正
    corrected_name = subject_name.replace('Computer Somnce', 'Computer Science').replace('Merdikane', 'Medicine')
    ax.set_title(f'{corrected_name}', fontsize=10, fontweight='bold',
                color=color, pad=5)

    # 设置极简背景
    ax.set_facecolor('white')

def create_fingerprint_gallery(df, output_path="disciplinary_fingerprint_gallery_v5.1.pdf"):
    """
    创建学科指纹画廊 v5.1 - 定向高亮版
    10个雷达图的画廊布局，带径向渐变和最强项高亮
    """

    # 计算全局最大值用于归一化
    global_max_values = {}
    for category in ['Images_density', 'Equations_density', 'Tables_density',
                    'Citations_density']:
        global_max_values[category] = df[category].max()

    # Structure Balance的最大值
    global_max_values['Structure_Balance'] = (1 - df['Structure_Gini']).max()

    # Sentence number的最大值
    global_max_values['Sentence_no'] = df['Sentence_no'].max()

    # 设置图形
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')

    # 2x5布局
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3,
                         top=0.9, bottom=0.1, left=0.05, right=0.95)

    # 主标题
    fig.suptitle('Disciplinary Fingerprint Gallery v5.1 - Directed Highlight Edition\n' +
                'Six-Dimensional Writing Style Signatures with Peak Feature Highlighting',
                fontsize=16, fontweight='bold', y=0.98)

    # 为每个学科创建雷达图
    for i, subject in enumerate(SUBJECTS):
        row = i // 5
        col = i % 5

        # 获取该学科的数据
        subject_data = df[df['Subject'] == subject]
        color = SUBJECT_COLORS[subject]

        # 创建子图
        ax = fig.add_subplot(gs[row, col], polar=True)
        create_radar_chart(ax, subject_data, subject, color, global_max_values)

    # 添加全局图例
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Mean Values'),
        plt.Line2D([0], [0], color='gray', linewidth=0, marker='o',
                  markersize=0, label='Standard Deviation Range',
                  markerfacecolor='gray', alpha=0.3)
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
              fontsize=10, ncol=2, frameon=False)

    # 保存高分辨率PDF
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    print(f"✅ 学科指纹画廊 v5.1 已保存: {output_path}")
    return output_path

def main():
    """
    主函数：生成v5.0学科指纹画廊
    """
    print("=" * 80)
    print("🎨 学科指纹画廊可视化系统 v5.1 - 定向高亮版")
    print("核心特色：径向渐变填充 + 最强项发光高亮 + 全局归一化")
    print("=" * 80)

    # 加载数据
    print("🔄 加载和处理数据...")
    df = load_and_process_data()
    print(f"✅ 数据加载完成：{len(df)} 篇论文，{len(df['Subject'].unique())} 个学科")

    # 生成指纹画廊
    print("🎨 生成学科指纹画廊...")
    output_path = create_fingerprint_gallery(df)

    print("=" * 80)
    print("🎉 分析完成！")
    print(f"📄 输出文件: {output_path} (600 DPI 高分辨率PDF)")
    print("✨ 包含10个学科的六维雷达图画廊")
    print("🔍 渐变填充：得分强度 | 发光圆点：最强项 | 灰影：波动范围")
    print("=" * 80)

if __name__ == "__main__":
    main()
