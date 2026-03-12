import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 0. 配置顶刊风格 (Nature Style Settings)
# ==========================================
# 设置全局字体为一种干净的无衬线字体 (类似 Arial/Helvetica)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线宽
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['font.size'] = 8        # 全局基础字号
plt.rcParams['axes.labelsize'] = 9   # 坐标轴标签字号
plt.rcParams['xtick.labelsize'] = 8  # 刻度字号
plt.rcParams['ytick.labelsize'] = 8

# 定义一个符合顶刊审美的感知均匀色系 (Viridis的变体)
nature_palette = ["#440154", "#31688e", "#35b779", "#fde725"]
sns.set_palette(nature_palette)

# ==========================================
# 1. 生成模拟数据 (基于你的维度)
# ==========================================
np.random.seed(42)
N = 300
data = pd.DataFrame({
    'Discipline': np.random.choice(['Physics', 'Biology', 'CS', 'Sociology'], N),
    # 使用对数正态分布模拟密度，更符合真实情况
    'Equations_density': np.random.lognormal(mean=1, sigma=0.8, size=N),
    'Images_density': np.random.lognormal(mean=1.5, sigma=0.6, size=N),
    'Citations_density': np.random.lognormal(mean=2, sigma=0.5, size=N),
    'Sentence_no': np.random.normal(loc=5000, scale=1500, size=N).astype(int),
})
# 确保句子数无负数
data['Sentence_no'] = data['Sentence_no'].clip(lower=1000) 

# ==========================================
# 2. 创建画布与网格布局 (Grid layout)
# ==========================================
fig = plt.figure(figsize=(10, 7), dpi=150) # 高DPI方便查看细节
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2], height_ratios=[1.2, 1])
plt.subplots_adjust(wspace=0.35, hspace=0.35) # 增加间距，让图表“呼吸”

# ==========================================
# 3. 子图 A：雨云图 (Raincloud-ish Plot)
# 展示不同学科的密度分布。比箱线图更高级。
# ==========================================
ax1 = fig.add_subplot(gs[0, :2])
# 添加顶刊 Panel 标签
ax1.text(-0.08, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

# 绘制半边小提琴图 (展示密度分布)
sns.violinplot(x='Discipline', y='Equations_density', data=data, ax=ax1,
               inner=None, color="lightgray", linewidth=0, saturation=0.5, split=True)
# 移除小提琴图的右半边，制造“云”的效果 (Hack trick)
for item in ax1.collections:
    if isinstance(item, matplotlib.collections.PolyCollection):
        item.set_alpha(0.6)

# 叠加原始数据点 (展示真实样本量)
sns.stripplot(x='Discipline', y='Equations_density', data=data, ax=ax1,
              palette=nature_palette, alpha=0.6, jitter=0.15, size=3)

# 叠加一个细长的箱线图 (展示统计量)
sns.boxplot(x='Discipline', y='Equations_density', data=data, ax=ax1,
            width=0.1, boxprops={'zorder': 2, 'facecolor':'none', 'linewidth':1},
            whiskerprops={'linewidth':1}, medianprops={'linewidth':1.5, 'color':'black'},
            showfliers=False)

ax1.set_ylabel("Eq. Density (per 1k words)")
ax1.set_xlabel("")
sns.despine(ax=ax1) # 去掉上方和右侧边框

# ==========================================
# 4. 子图 B：联合密度分布 (Joint KDE Plot)
# 展示句子数量与引用密度的关系。比普通散点图更高级。
# ==========================================
# 注意：Jointplot通常独立创建 figure，这里我们用稍微复杂的方法嵌入到现有网格中
ax_joint = fig.add_subplot(gs[1, :2])
ax_joint.text(-0.08, 1.05, 'b', transform=ax_joint.transAxes, fontsize=16, fontweight='bold', va='top')

# 绘制核心的 KDE 等高线图
sns.kdeplot(x='Sentence_no', y='Citations_density', data=data, ax=ax_joint,
            fill=True, cmap="viridis", thresh=0.05, levels=8, alpha=0.7)
# 叠加散点图 (低透明度)
sns.scatterplot(x='Sentence_no', y='Citations_density', data=data, ax=ax_joint,
                s=15, color='black', alpha=0.2, edgecolor=None)

ax_joint.set_xlabel("Total Sentences (N)")
ax_joint.set_ylabel("Cit. Density")
ax_joint.set_xlim(0, 10000)
sns.despine(ax=ax_joint)

# ==========================================
# 5. 子图 C：学术三线表 (Academic Three-line Table)
# 严格符合规范的表格绘制
# ==========================================
ax_table = fig.add_subplot(gs[:, 2])
ax_table.axis('off') # 关闭坐标轴
ax_table.text(0, 1.02, 'c', transform=ax_table.transAxes, fontsize=16, fontweight='bold', va='top')
ax_table.set_title("Statistical Summary (Mean ± SD)", fontsize=9, pad=12, loc='left')

# 计算汇总数据
summary = data.groupby('Discipline')[['Equations_density', 'Images_density']].agg(['mean', 'std']).round(1)
table_data = []
for idx, row in summary.iterrows():
    # 格式化为 "Mean ± SD" 字符串
    eq_str = f"{row[('Equations_density', 'mean')]:.1f} ± {row[('Equations_density', 'std')]:.1f}"
    img_str = f"{row[('Images_density', 'mean')]:.1f} ± {row[('Images_density', 'std')]:.1f}"
    table_data.append([idx, eq_str, img_str])

col_labels = ['Discipline', 'Eq. Density', 'Img. Density']

# 绘制表格
table = ax_table.table(cellText=table_data,
                       colLabels=col_labels,
                       loc='center',
                       cellLoc='center',
                       edges='open') # 'open' 模式初始不显示线条

# 手动设置三线表的线条样式
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.8) # 调整表格长宽比

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0) # 默认无边框
    cell.set_text_props(fontfamily='sans-serif') # 统一字体
    
    # 设置顶线 (Top Line, row 0 上方)
    if row == 0:
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2) # 顶线较粗
        cell.set_text_props(weight='bold') # 表头加粗
        cell.visible_edges = "T" # 只显示上边框

    # 设置表头下线 (Header Line, row 0 下方/ row 1 上方)
    if row == 1:
        cell.set_edgecolor('black')
        cell.set_linewidth(0.8) # 中线较细
        cell.visible_edges = "T"
        
# 设置底线 (Bottom Line, 最后一行下方)
for col in range(len(col_labels)):
    cell = table[len(table_data), col]
    cell.set_edgecolor('black')
    cell.set_linewidth(1.2) # 底线较粗
    cell.visible_edges = "B" # 只显示下边框

# ==========================================
# 6. 完成并显示
# ==========================================
# plt.tight_layout() # GridSpec布局下通常不需要，但可以微调
# 保存为顶刊投稿格式
fig.savefig('Nature_Benchmark_Figure.pdf', dpi=300, bbox_inches='tight', format='pdf')
fig.savefig('Nature_Benchmark_Figure.png', dpi=300, bbox_inches='tight', format='png')
print("图表已保存为 Nature_Benchmark_Figure.pdf 和 Nature_Benchmark_Figure.png")
plt.close()  # 关闭图形以释放内存