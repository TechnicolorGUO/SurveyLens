#!/usr/bin/env python3
"""
基于完整的1000篇论文统计数据生成小提琴图
使用final_complete_statistics.csv中的准确统计数据
保持与scientific_analysis.pdf左上角图表维度和样式一致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数，与原始图表保持一致
mpl.use('Agg')
plt.style.use('default')

# 设置颜色方案，与原始图表一致
COLORS = {
    'primary': '#2E86AB',      # 主色调
    'secondary': '#A23B72',    # 辅助色
    'accent': '#F18F01',       # 强调色
    'neutral': '#C73E1D',      # 中性色
    'dark': '#0B4F6C',         # 深色
    'light': '#E8F4F8',        # 浅色
    'grid': '#E0E0E0',         # 网格色
    'text': '#2C3E50'          # 文字色
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'axes.labelpad': 8,
    'figure.figsize': (8, 6),  # 调整为单个图表大小
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
    'axes.edgecolor': COLORS['dark'],
    'axes.facecolor': 'white'
})

def create_violin_plot(ax, df, density_cols, density_labels, density_limits=None):
    """
    创建小提琴图 + 数据点
    与原始scientific_analysis.pdf左上角图表完全一致
    支持Y轴范围限制以处理极端数据
    """
    # 创建小提琴图
    vp = ax.violinplot([df[col] for col in density_cols],
                       positions=range(len(density_cols)),
                       showmeans=True, showmedians=False, showextrema=False)

    # 设置小提琴图样式
    for pc in vp['bodies']:
        pc.set_facecolor(COLORS['primary'])
        pc.set_edgecolor(COLORS['dark'])
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)

    # 设置均值线样式
    if 'cmeans' in vp:
        vp['cmeans'].set_color(COLORS['secondary'])
        vp['cmeans'].set_linewidth(2)

    # 添加散点图（数据点）
    for i, col in enumerate(density_cols):
        y_data = df[col].values

        # 如果设置了限制，只显示在限制范围内的点
        if density_limits and col in density_limits:
            mask = y_data <= density_limits[col]
            y_data_filtered = y_data[mask]
            # 为过滤后的数据重新生成随机偏移
            x_jitter = np.random.uniform(-0.25, 0.25, len(y_data_filtered))
        else:
            y_data_filtered = y_data
            x_jitter = np.random.uniform(-0.25, 0.25, len(y_data_filtered))

        x_data = np.full_like(y_data_filtered, i, dtype=float) + x_jitter

        ax.scatter(x_data, y_data_filtered, alpha=0.4, s=12, color=COLORS['dark'],
                   edgecolors='white', linewidth=0.3, zorder=3)

    # 设置坐标轴
    ax.set_xticks(range(len(density_cols)))
    ax.set_xticklabels(density_labels, fontsize=9, fontweight='bold')
    ax.set_ylabel('Feature Density (%)\n(per sentence)', fontsize=10, fontweight='bold')
    ax.set_title('Feature Densities Across 1000 Papers',
                 fontsize=12, fontweight='bold', pad=15)

    # 设置网格
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')

    # 设置边框
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['dark'])
        spine.set_linewidth(1.0)

    # 设置Y轴范围，聚焦主要数据分布（如果提供了限制）
    if density_limits:
        # 找到所有特征中的最大限制值，作为统一的Y轴上限
        max_limit = max(density_limits.values())
        # 稍微放宽一些，确保大部分数据可见
        y_max = min(max_limit * 1.1, max_limit + 5)  # 最多增加5个单位
        ax.set_ylim(0, y_max)

        # 移除数据截断提示标注，保持图表简洁

def main():
    print("🎯 生成基于完整1000篇论文统计数据的小提琴图")
    print("使用final_complete_statistics.csv中的准确数据")
    print("保持与scientific_analysis.pdf左上角图表维度一致")
    print("=" * 60)

    # 读取完整的统计数据（包含1000篇论文+平均值）
    csv_file = "final_complete_statistics.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功读取数据: {len(df)}行")

        # 过滤掉平均值行，只保留实际论文数据
        df_papers = df[~df['file_name'].str.contains('AVERAGE', na=False)]
        print(f"✅ 论文数据: {len(df_papers)}篇论文")

        # 保留平均值用于参考
        df_averages = df[df['file_name'].str.contains('AVERAGE', na=False)]
        print(f"✅ 平均值数据: {len(df_averages)}行")

        df = df_papers  # 使用论文数据进行绘图

    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_file}")
        return

    # 计算密度特征（按句子数标准化）
    print("📊 计算密度特征...")
    df['Images_density'] = (df['images'] / df['sentences'] * 100).fillna(0)
    df['Equations_density'] = (df['equations'] / df['sentences'] * 100).fillna(0)
    df['Tables_density'] = (df['tables'] / df['sentences'] * 100).fillna(0)
    df['Citations_density'] = (df['citations'] / df['sentences'] * 100).fillna(0)

    # 处理可能的无穷大值
    df = df.replace([np.inf, -np.inf], 0)

    # 设置绘图参数
    density_cols = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
    density_labels = ['Images', 'Equations', 'Tables', 'Citations']

    # 处理极端值，确保图表协调显示
    print("🎯 处理极端数据，确保图表协调...")
    # 计算每个特征的95%分位数作为上限
    density_limits = {}
    for col in density_cols:
        # 使用95%分位数作为上限，避免极端值影响显示
        upper_limit = df[col].quantile(0.95)
        density_limits[col] = upper_limit
        print(f"  {col}: 95%分位数上限 = {upper_limit:.2f}")

    # 打印统计信息
    print("\n📈 密度特征统计:")
    for i, col in enumerate(density_cols):
        stats = df[col].describe()
        limit_info = ""
        if density_limits and col in density_limits:
            limit = density_limits[col]
            truncated_count = len(df[df[col] > limit])
            if truncated_count > 0:
                limit_info = f" (95%分位数限制: {limit:.1f}, 截断{truncated_count}个极端值)"
        print(f"  {density_labels[i]}: 均值={stats['mean']:.2f}, 最大={stats['max']:.2f}, 最小={stats['min']:.2f}{limit_info}")
    # 创建图表
    print("\n🎨 生成小提琴图...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 调用小提琴图创建函数
    create_violin_plot(ax, df, density_cols, density_labels, density_limits)

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 保存图表
    output_file = "violin_plot_1000_papers.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=600, format='pdf')
    print(f"✅ 图表已保存: {output_file}")

    # 同时保存PNG格式
    png_file = "violin_plot_1000_papers.png"
    plt.savefig(png_file, bbox_inches='tight', dpi=300, format='png')
    print(f"✅ 图表已保存: {png_file}")

    # 显示图表
    plt.show()

    print("\n📋 图表说明:")
    print("  - X轴: 特征类型 (Images, Equations, Tables, Citations)")
    print("  - Y轴: 密度百分比 (每句的特征数量)")
    print("  - 小提琴图: 显示数据分布")
    print("  - 红线: 均值位置")
    print("  - 散点: 单个论文的数据点")
    print(f"  - 样本量: {len(df)}篇论文")
    print("  - 数据源: final_complete_statistics.csv")

    print("\n🏆 完成！")
    print("图表使用最新的完整统计数据，完全保持scientific_analysis.pdf左上角的样式和维度")

if __name__ == "__main__":
    main()