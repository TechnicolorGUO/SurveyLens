#!/usr/bin/env python3
"""
科学论文特征分析脚本
使用matplotlib创建符合顶级期刊审美的2x2子图组合

依赖库:
- numpy
- matplotlib
- pandas

安装依赖:
pip install numpy matplotlib pandas

运行方法:
python survey_paper_analysis_final.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import pandas as pd

# 设置Nature风格的matplotlib参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 10)

# 设置随机种子以保证结果可重现
np.random.seed(42)

def generate_mock_data(n_samples=100):
    """
    生成模拟的论文特征数据

    Parameters:
    -----------
    n_samples : int
        样本数量

    Returns:
    --------
    pd.DataFrame
        包含7个特征的DataFrame
    """
    # 定义数据分布参数（基于学术论文的合理范围）
    data_params = {
        'Images_density': {'mean': 0.8, 'std': 0.5, 'min': 0, 'max': 3},
        'Equations_density': {'mean': 1.2, 'std': 0.8, 'min': 0, 'max': 4},
        'Tables_density': {'mean': 0.6, 'std': 0.4, 'min': 0, 'max': 2},
        'Citations_density': {'mean': 2.5, 'std': 1.0, 'min': 0.5, 'max': 6},
        'Outline_no': {'mean': 45, 'std': 15, 'min': 20, 'max': 80},
        'Reference_no': {'mean': 85, 'std': 25, 'min': 30, 'max': 150},
        'Sentence_no': {'mean': 1200, 'std': 400, 'min': 500, 'max': 2000}
    }

    # 生成数据
    data = {}
    for feature, params in data_params.items():
        # 使用截断正态分布生成数据
        raw_data = np.random.normal(params['mean'], params['std'], n_samples)
        # 截断到合理范围内
        data[feature] = np.clip(raw_data, params['min'], params['max'])

    return pd.DataFrame(data)

def create_scientific_plots(df, save_path='survey_paper_analysis.pdf'):
    """
    创建2x2子图组合的科学图表

    Parameters:
    -----------
    df : pd.DataFrame
        包含特征数据的DataFrame
    save_path : str
        保存路径
    """
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # 图A：Citations_density 与 Reference_no 的相关性散点图，带线性回归拟合线
    ax1 = axes[0, 0]
    x_a = df['Citations_density']
    y_a = df['Reference_no']

    # 散点图
    ax1.scatter(x_a, y_a, alpha=0.7, s=50, color='#2E86AB',
                edgecolors='white', linewidth=0.5)

    # 线性回归 (使用numpy实现)
    # 计算斜率和截距
    x_mean = np.mean(x_a)
    y_mean = np.mean(y_a)
    slope = np.sum((x_a - x_mean) * (y_a - y_mean)) / np.sum((x_a - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # 计算相关系数
    r_value = np.corrcoef(x_a, y_a)[0, 1]

    x_line = np.linspace(x_a.min(), x_a.max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, color='#F24236', linewidth=2, alpha=0.8,
             label='.3f')

    # 设置标签和标题
    ax1.set_xlabel(r'$Density_{citations}$', fontsize=12)
    ax1.set_ylabel(r'$Reference_{no}$', fontsize=12)
    ax1.set_title('A', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 图B：Equations_density 的分布直方图，使用步进状显示
    ax2 = axes[0, 1]
    n_bins = 15
    ax2.hist(df['Equations_density'], bins=n_bins, alpha=0.7,
             color='#F24236', edgecolor='white', linewidth=0.5,
             histtype='step', fill=True)

    ax2.set_xlabel(r'$Density_{equations}$', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('B', fontsize=14, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 图C：Images_density 与 Tables_density 的对比箱线图
    ax3 = axes[1, 0]
    box_data = [df['Images_density'], df['Tables_density']]
    box_labels = [r'$Density_{images}$', r'$Density_{tables}$']

    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#2E86AB'),
                    capprops=dict(color='#2E86AB'))

    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('C', fontsize=14, fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 移除第4个子图（右下角）
    axes[1, 1].set_visible(False)

    # 设置整体标题
    fig.suptitle('Scientific Paper Feature Analysis', fontsize=16,
                 fontweight='bold', y=0.98)

    # 保存为PDF，DPI=300
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"图表已成功保存为: {save_path}")

def main():
    """主函数"""
    print("开始生成科学论文特征分析图表...")

    # 生成模拟数据
    print("生成模拟数据...")
    df = generate_mock_data(100)
    print(f"生成了 {len(df)} 条模拟数据")
    print("\n数据统计信息:")
    print(df.describe())

    # 创建图表
    print("\n创建图表...")
    save_path = '/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/survey_paper_analysis.pdf'
    create_scientific_plots(df, save_path)

    print("分析完成!")

if __name__ == "__main__":
    main()
