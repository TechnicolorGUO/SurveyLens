#!/usr/bin/env python3
"""
学术指纹矩阵 v4.3 最终版：主图视觉化、附录数据化
升级为学术出版标准：移除数值标注、强化色彩指纹、生成附录数据表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 1. 核心数据 (基于paper_data_summary.md的真实Mean值)
data = {
    'Subject': ['Physics', 'Computer Science', 'Engineering', 'Education', 'Business',
                'Sociology', 'Medicine', 'Psychology', 'Environmental Science', 'Biology'],
    'Images (%)': [0.54, 0.75, 0.67, 0.50, 0.24, 0.37, 0.33, 0.20, 0.38, 0.31],
    'Equations (%)': [26.46, 3.11, 7.49, 2.95, 1.70, 4.50, 7.26, 6.45, 8.82, 4.96],
    'Tables (%)': [0.09, 0.58, 0.25, 0.96, 0.97, 0.14, 0.38, 0.37, 0.41, 0.16],
    'Citations (%)': [23.28, 23.51, 20.60, 22.23, 20.75, 19.73, 16.45, 16.25, 15.28, 10.52],
    'Outline (Count)': [73, 34, 29, 24, 29, 21, 33, 32, 24, 30],
    'References (Count)': [63, 187, 240, 97, 108, 73, 142, 122, 159, 116],
    'Sentences (Count)': [6261, 1770, 1991, 968, 1138, 741, 1196, 997, 1419, 1195]
}

df = pd.DataFrame(data).set_index('Subject')
# 按引用密度排序
df = df.sort_values('Citations (%)', ascending=False)

# 2. 绘图配置
cols = df.columns
cmaps = ['Oranges', 'Purples', 'Greens', 'Blues', 'Reds', 'YlOrBr', 'Greys']

# 分别控制热力图和色标区域的布局
fig = plt.figure(figsize=(19, 10))  # 缩小宽度，让文字更突出

# 热力图区域：稍微增加行高 (wspace=0.0)
gs_heatmap = gridspec.GridSpec(1, len(cols), figure=fig,
                              left=0.06, right=0.94, top=0.88, bottom=0.30,
                              wspace=0.0)  # 热力图列间无缝，稍微增加行高

# 色标区域：行高缩短一倍 (wspace=0.0，确保与热力图完美对齐)
gs_cbar = gridspec.GridSpec(1, len(cols), figure=fig,
                           left=0.06, right=0.94, top=0.26, bottom=0.20,
                           wspace=0.0)  # 色标与热力图使用相同wspace，确保完美对齐

# 3. 逐列绘制热力图
for i, col in enumerate(cols):
    # 热力图主体 - 使用独立的热力图GridSpec
    ax = fig.add_subplot(gs_heatmap[0, i])
    col_data = df[[col]].values

    # 独立归一化每一列
    vmin, vmax = col_data.min(), col_data.max()
    if vmax > vmin:
        normalized_data = (col_data - vmin) / (vmax - vmin)
    else:
        normalized_data = np.full_like(col_data, 0.5)

    # 创建热力图
    cmap = plt.get_cmap(cmaps[i])
    cmap = LinearSegmentedColormap.from_list(
        f"{col}_cmap",
        [(0, cmap(0.2)), (0.5, cmap(0.6)), (1, cmap(0.9))]
    )

    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto')

    # 移除数值标注：强化纯粹的色彩指纹表达 (annot=False)
    # 颜色深度将成为唯一的表达语言

    # 添加极致细微的白色水平分割线 (linewidths=0.2)
    # 增强质感但不破坏视觉连续性
    for j in range(1, len(df)):
        ax.axhline(y=j-0.5, color='white', linewidth=0.2, alpha=0.8)

    # 设置轴标签
    if i == 0:
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=26, fontweight='bold')
    else:
        ax.set_yticks([])

    ax.set_xticks([])

    # 设置标题（放在顶部）
    title_text = col.replace(' (%)', '\n(%)').replace(' (Count)', '\n(Count)')
    ax.set_title(title_text, fontsize=26, fontweight='bold', pad=12)

    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) if i > 0 else None

    # 核心修正：手动定位色标确保与热力图列完美对齐
    # 计算每列的精确位置：left + i * (total_width / n_cols) + small_margin
    col_width = (0.94 - 0.06) / len(cols)  # 每列宽度
    col_left = 0.06 + i * col_width + col_width * 0.05  # 左侧位置（留5%边距）
    col_right = 0.06 + (i + 1) * col_width - col_width * 0.05  # 右侧位置（留5%边距）
    cbar_width = col_right - col_left  # 色标宽度
    cbar_height = 0.03  # 色标高度（3%，再缩短一倍）
    cbar_bottom = 0.22  # 色标底部位置（相应上移保持间距）

    cax = fig.add_axes([col_left, cbar_bottom, cbar_width, cbar_height])  # 精确手动定位

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # 将色标放入 cax，取消单独计算 add_axes
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')

    # 精细化调整色标刻度 - 优化标签位置避免重叠
    # 将刻度稍微向内移动，避免标签与色标边缘重叠
    tick_offset = (vmax - vmin) * 0.1  # 10%的偏移量
    cb.set_ticks([vmin + tick_offset, vmax - tick_offset])
    if i < 4:
        cb.set_ticklabels([f'{vmin:.0f}', f'{vmax:.0f}'], fontsize=20, fontweight='bold')
    else:
        cb.set_ticklabels([f'{vmin:.0f}', f'{vmax:.0f}'], fontsize=20, fontweight='bold')

    cb.outline.set_visible(False)  # 移除色标边框让它更 Fancy
    cax.tick_params(axis='x', which='both', length=0, pad=2)  # 移除刻度小线段，适当pad

# 移除主图标题：实现纯粹的视觉指纹表达
# plt.suptitle('Fig. X: The Disciplinary Fingerprint Matrix (Visual Patterns)\n' +
#             '(Color-coded patterns across 7 dimensions, n=100 papers)',
#             fontsize=16, fontweight='bold', y=0.98, family='sans-serif')

plt.savefig('disciplinary_fingerprint_matrix_v43_visual.pdf', bbox_inches='tight', dpi=600)
plt.close()

# 2. 生成附录数据表 (Appendix Data Table)
print("\n📊 Generating Appendix Data Table...")

# 准备附录数据表
appendix_data = df.copy()

# 格式化数据：百分比维度保留2位小数并带%，计数值保留整数
for col in appendix_data.columns:
    if '(%)' in col:
        appendix_data[col] = appendix_data[col].apply(lambda x: f"{x:.2f}%")
    elif '(Count)' in col:
        appendix_data[col] = appendix_data[col].astype(int)

# 保存为CSV文件
appendix_filename = 'Appendix_Table_Disciplinary_Means.csv'
appendix_data.to_csv(appendix_filename, index=True)

print("✅ Academic Fingerprint Matrix v4.3 completed!")
print("📄 Main Figure: disciplinary_fingerprint_matrix_v43_visual.pdf")
print(f"📄 Appendix Table: {appendix_filename}")
print("\nTable S1: Quantitative Stylometric Metrics Across Ten Disciplines")

# 打印数据验证
print("\n📊 Appendix Data Table Preview:")
print(appendix_data)
