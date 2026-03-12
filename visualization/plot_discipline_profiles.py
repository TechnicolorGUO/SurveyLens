#!/usr/bin/env python3
"""
学术指纹级跨学科论文特征分析系统 v3.0
从100篇Markdown论文中提取特征，生成符合Nature标准的学科对比可视化

核心改进：
- 统计逻辑重构：使用中位数替代均值，消除离群值偏置
- 学科指纹雷达图：对比代表性学科的特征差异
- HUSL调色盘：符合Nature审美的10个学科色彩系统
- 稳健统计可视化：离群值标注和置信区间

依赖库:
- numpy, pandas, matplotlib, seaborn
- husl (pip install husl)
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在无GUI环境中使用Agg后端
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# 尝试导入必要的库
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib实现")

try:
    import husl
    HAS_HUSL = True
except ImportError:
    HAS_HUSL = False
    print("警告: husl未安装，将使用备用颜色方案")

# 尝试导入scipy用于统计计算
try:
    from scipy import stats
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，将使用简化统计方法")

# 尝试导入joypy用于山脊图
try:
    import joypy
    HAS_JOYPY = True
except ImportError:
    HAS_JOYPY = False
    print("警告: joypy未安装，将使用seaborn实现山脊图")

# mpl.use('Agg')  # 注释掉非GUI后端设置，让图表可以显示

# Nature级别的高分辨率设置 - 优化版
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 14,  # 再次增大基础字体
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'axes.linewidth': 1.2,  # 略微加粗边框
    'axes.labelpad': 10,
    'figure.figsize': (22, 16),  # 增大画布尺寸让图表更好地填充页面
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

# 学科HUSL色彩系统 - 符合Nature审美
SUBJECTS = ['Biology', 'Business', 'Computer Science', 'Education',
           'Engineering', 'Environmental Science', 'Medicine', 'Physics',
           'Psychology', 'Sociology']

# 生成10个学科的husl颜色，使用均匀分布的色相
def generate_subject_colors():
    """生成10个学科的HUSL色彩方案"""
    if HAS_HUSL:
        # HUSL色彩空间：均匀分布的色相，固定饱和度和亮度
        hues = np.linspace(15, 345, 10)  # 避开红色区域，得到更好的区分度
        colors = {}
        for i, subject in enumerate(SUBJECTS):
            # HUSL格式: (hue, saturation, lightness)
            husl_color = husl.husl_to_hex(hues[i], 70, 55)  # 饱和度70%，亮度55%
            colors[subject] = husl_color
        return colors
    else:
        # 备用颜色方案
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        return {subject: base_colors[i] for i, subject in enumerate(SUBJECTS)}

SUBJECT_COLORS = generate_subject_colors()

# 基础颜色
BASE_COLORS = {
    'primary': '#2E3440',      # 深蓝灰
    'secondary': '#5E81AC',    # 中等蓝色
    'accent': '#A3BE8C',       # 柔和绿色
    'dark': '#1a1a1a',         # 深灰
    'light': '#ECEFF4',        # 浅灰
    'warning': '#D08770',      # 橙红色
}

def extract_citations_robust(text):
    """
    健壮的学术引用提取函数 - v3.0增强版

    支持多种引用格式，包含作者-年份型和数值型引用
    """
    # 上标数字到普通数字的映射
    superscript_to_digit = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }

    citations_found = []
    total_citations = 0

    # 数值型引用正则表达式
    bracket_pattern = r'\[(\d+(?:,\s*\d+)*)(?:-(\d+))?\]'
    superscript_digits = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    superscript_pattern = r'([' + superscript_digits + r']+(?:,[' + superscript_digits + r']+)*(?:-[' + superscript_digits + r']+)?)(?=\W|$)'

    # 处理数值型引用
    for pattern in [bracket_pattern, superscript_pattern]:
        for match in re.finditer(pattern, text):
            citation_str = match.group(0)
            citations_found.append(citation_str)

            if citation_str.startswith('['):
                # 处理方括号格式
                if match.group(2):
                    start = int(re.findall(r'\d+', citation_str)[0])
                    end = int(re.findall(r'\d+', citation_str)[1])
                    total_citations += (end - start + 1) if start <= end else 1
                else:
                    numbers = re.findall(r'\d+', citation_str)
                    total_citations += len(numbers)
            else:
                # 处理上标格式
                normal_text = ''.join(superscript_to_digit.get(c, c) for c in citation_str)
                if '-' in normal_text:
                    parts = normal_text.split('-')
                    if len(parts) == 2:
                        try:
                            start, end = int(parts[0]), int(parts[1])
                            total_citations += (end - start + 1) if start <= end else 1
                        except ValueError:
                            total_citations += 1
                else:
                    numbers = re.findall(r'\d+', normal_text)
                    total_citations += len(numbers)

    # 作者-年份型引用
    author_year_pattern = r'\(([^)]+)\)'
    for match in re.finditer(author_year_pattern, text):
        content = match.group(1)
        if re.search(r'\d{4}', content) and re.search(r'[A-Za-z]', content):
            citations_found.append(match.group(0))
            individual_citations = re.split(r';\s*', content)
            total_citations += len(individual_citations)

    return total_citations, citations_found

def read_md(md_path: str) -> str:
    """读取markdown文件"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        try:
            with open(md_path, "r", encoding="latin-1") as f:
                return f.read()
        except:
            print(f"无法读取文件: {md_path}")
            return ""

def count_sentences(text: str) -> int:
    """统计句子数量"""
    sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def calculate_structure_gini(content: str, total_sentences: int) -> float:
    """
    计算结构均衡度（基尼系数）
    使用基尼系数衡量章节长度分布的不均衡程度
    """
    if total_sentences == 0:
        return 0.0

    headings = []
    for match in re.finditer(r'^#{1,6} .+$', content, re.MULTILINE):
        headings.append(match.start())

    if len(headings) < 2:
        return 0.0

    section_lengths = []
    headings.append(len(content))

    for i in range(len(headings) - 1):
        section_content = content[headings[i]:headings[i+1]]
        section_sentences = count_sentences(section_content)
        if section_sentences > 0:
            section_lengths.append(section_sentences)

    if len(section_lengths) < 2:
        return 0.0

    section_lengths = np.array(section_lengths, dtype=float)
    n = len(section_lengths)

    if n < 2:
        return 0.0

    sorted_lengths = np.sort(section_lengths)
    mean_length = np.mean(section_lengths)

    if mean_length == 0:
        return 0.0

    gini_sum = 0.0
    for i in range(n):
        gini_sum += (2 * (i + 1) - n - 1) * sorted_lengths[i]

    gini = gini_sum / (n * n * mean_length)
    return max(0.0, min(1.0, gini))

def calculate_citation_coverage(content: str, total_sentences: int) -> float:
    """计算引文覆盖率"""
    if total_sentences == 0:
        return 0.0

    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    citation_sentences = 0

    for sentence in sentences:
        _, citation_strings = extract_citations_robust(sentence)
        if citation_strings:
            citation_sentences += 1

    return (citation_sentences / total_sentences) * 100

def count_md_features(md_content: str) -> dict:
    """
    统计markdown文件中的各种特征 - v3.0增强版

    改进：
    - 支持单字符行内公式 ($x$, $n$)
    - 更精确的图片和表格检测
    """
    # 图片提取 - 避免重复计数
    img_md = re.findall(r'!\[.*?\]\(.*?\)', md_content)
    img_html_full = re.findall(r'<img [^>]*src=[\'"].*?[\'"][^>]*>', md_content, re.IGNORECASE)
    img_html_simple = []
    for simple_match in re.findall(r'<img [^>]*>', md_content, re.IGNORECASE):
        if not any(simple_match in full_match for full_match in img_html_full):
            img_html_simple.append(simple_match)

    image_count = len(set(img_md + img_html_full + img_html_simple))

    # 公式提取 - v3.0增强版，支持单字符公式
    block_eq = []
    block_eq.extend([m for m in re.findall(r'\$\$.*?\$\$', md_content, re.DOTALL) if m.strip('$\n\t ')])
    block_eq.extend(re.findall(r'\\\[.*?\\\]', md_content, re.DOTALL))
    block_eq.extend(re.findall(r'\\begin\{.*?\}.*?\\end\{.*?\}', md_content, re.DOTALL))

    # 行内公式：支持单字符公式，如$x$, $n$
    inline_eq = [eq for eq in re.findall(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', md_content) if eq.strip()]

    equation_count = len(block_eq) + len(inline_eq)

    # 表格提取
    md_table_pattern = r'\|.*\|.*\n\|[\s\-\|:]+\|.*\n(?:\|.*\|.*\n)*'
    md_tables = re.findall(md_table_pattern, md_content)
    html_tables = re.findall(r'<table[\s\S]*?</table>', md_content, re.IGNORECASE)
    table_count = len(md_tables) + len(html_tables)

    sentence_count = count_sentences(md_content)

    return {
        'images': image_count,
        'equations': equation_count,
        'tables': table_count,
        'sentences': sentence_count
    }

def extract_features_from_md(md_file_path: str) -> dict:
    """从markdown文件中提取所有特征 - v3.0增强版"""
    content = read_md(md_file_path)
    if not content:
        return None

    md_features = count_md_features(content)

    # 使用增强版引用提取
    citation_count, citation_strings = extract_citations_robust(content)

    # 大纲提取
    outline_pattern = r'^#{1,6} .+$'
    outline_count = len(re.findall(outline_pattern, content, re.MULTILINE))

    # 参考文献提取 - v3.0增强版，支持多种标题格式
    references_section = re.search(
        r'#+\s*(?:References?|Bibliography|Works\s+Cited|Literature\s+Cited|References?\s+and\s+Notes?).*?(?=#+ |\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    reference_count = 0
    if references_section:
        references_text = references_section.group(0)
        # 更通用的参考文献计数
        reference_items = [line.strip() for line in references_text.split('\n')
                          if line.strip() and not line.strip().startswith('#')]
        reference_count = len(reference_items)

    sentence_count = md_features['sentences']

    # 高阶指标计算
    structure_gini = calculate_structure_gini(content, sentence_count)
    cit_coverage = calculate_citation_coverage(content, sentence_count)
    info_density = (md_features['images'] + md_features['equations'] + md_features['tables']) / max(sentence_count, 1) * 100

    return {
        'Images_count': md_features['images'],
        'Equations_count': md_features['equations'],
        'Tables_count': md_features['tables'],
        'Citations_count': citation_count,
        'Outline_count': outline_count,
        'Reference_count': reference_count,
        'Sentence_count': sentence_count,
        'Structure_Gini': structure_gini,
        'Cit_Coverage': cit_coverage,
        'Info_Density': info_density
    }

def calculate_density_features(features_dict: dict) -> dict:
    """计算密度特征和深度指标"""
    sentence_count = max(features_dict.get('Sentence_count', 1), 1)
    return {
        'Images_density': features_dict.get('Images_count', 0) / sentence_count * 100,
        'Equations_density': features_dict.get('Equations_count', 0) / sentence_count * 100,
        'Tables_density': features_dict.get('Tables_count', 0) / sentence_count * 100,
        'Citations_density': features_dict.get('Citations_count', 0) / sentence_count * 100,
        'Outline_no': features_dict.get('Outline_count', 0),
        'Reference_no': features_dict.get('Reference_count', 0),
        'Sentence_no': features_dict.get('Sentence_count', 0),
        'Structure_Gini': features_dict.get('Structure_Gini', 0.0),
        'Cit_Coverage': features_dict.get('Cit_Coverage', 0.0),
        'Info_Density': features_dict.get('Info_Density', 0.0)
    }

def process_all_md_files(base_path: str) -> pd.DataFrame:
    """处理所有markdown文件并提取特征"""
    all_features = []

    print("开始处理10个学科的论文特征提取...")

    for subject in SUBJECTS:
        subject_path = os.path.join(base_path, subject)
        if not os.path.exists(subject_path):
            print(f"警告: 学科路径不存在 - {subject_path}")
            continue

        print(f"处理学科: {subject}")
        md_files = [f for f in os.listdir(subject_path) if f.endswith('.md')]

        for md_file in md_files:
            md_file_path = os.path.join(subject_path, md_file)
            features = extract_features_from_md(md_file_path)

            if features:
                density_features = calculate_density_features(features)
                density_features['Subject'] = subject
                density_features['Paper_ID'] = f"{subject} {md_file.replace('.md', '')}"
                all_features.append(density_features)

    df = pd.DataFrame(all_features)
    print(f"成功处理了 {len(df)} 篇论文")
    print(f"数据维度: {df.shape}")
    return df

def create_radar_chart(ax, df):
    """
    创建学科指纹雷达图 - v3.0新增功能

    选择5个代表性学科，展示它们在5个核心维度的归一化得分
    """
    # 选择代表性学科
    representative_subjects = ['Physics', 'Business', 'Sociology', 'Medicine', 'Computer Science']

    # 核心特征维度
    radar_features = ['Images_density', 'Equations_density', 'Tables_density',
                     'Citations_density', 'Structure_Gini']
    radar_labels = ['Images', 'Equations', 'Tables', 'Citations', 'Structure\nBalance']

    # 计算每个学科的中位数（使用中位数而非均值）
    subject_medians = df.groupby('Subject')[radar_features].median()

    # 归一化处理（0-1缩放）
    normalized_data = {}
    for feature in radar_features:
        feature_data = subject_medians[feature]
        min_val, max_val = feature_data.min(), feature_data.max()
        if max_val > min_val:
            normalized_data[feature] = (feature_data - min_val) / (max_val - min_val)
        else:
            normalized_data[feature] = feature_data / feature_data.max() if feature_data.max() > 0 else 0

    normalized_df = pd.DataFrame(normalized_data)

    # 雷达图参数
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 绘制雷达图
    for subject in representative_subjects:
        if subject in normalized_df.index:
            values = normalized_df.loc[subject].values.tolist()
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, 'o-', linewidth=2, markersize=6,
                   color=SUBJECT_COLORS[subject], label=subject, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=SUBJECT_COLORS[subject])

    # 设置标签 - 优化版
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=13, fontweight='medium')
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=12)
    ax.set_title('Disciplinary Fingerprints\n(Normalized Scores)', fontsize=17, pad=30, weight='bold',
                fontfamily='sans-serif')

    # 网格线优化
    ax.grid(True, alpha=0.4, linewidth=0.8)
    ax.spines['polar'].set_visible(False)

    # 图例优化 - 放在雷达图正下方，避免与数据重叠
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=10,
             frameon=True, fancybox=True, shadow=True, ncol=3)

def create_grouped_boxplot(ax, df):
    """
    创建分组箱线图 - v3.0改进版

    按学科分组展示特征密度，包含离群值标注
    """
    density_cols = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
    density_labels = ['Images', 'Equations', 'Tables', 'Citations']

    # 准备数据
    plot_data = []
    positions = []
    colors = []
    labels = []

    subject_order = SUBJECTS.copy()

    pos = 0
    for subject in subject_order:
        subject_data = df[df['Subject'] == subject]
        if len(subject_data) > 0:
            for i, col in enumerate(density_cols):
                values = subject_data[col].values
                if len(values) > 0:
                    plot_data.append(values)
                    positions.append(pos + i * 0.8)
                    colors.append(SUBJECT_COLORS[subject])
                    # 只在第一个特征位置显示学科名称
                    if i == 0:
                        labels.append(subject)
                    else:
                        labels.append('')  # 其他位置不显示标签

            pos += len(density_cols) + 1  # 学科间距

    # 创建分组箱线图
    bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                   widths=0.6, showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.6))

    # 设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.0)

    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.0)

    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)

    # 标注离群值
    for i, (fliers, pos_val) in enumerate(zip(bp['fliers'], positions)):
        if len(fliers.get_xdata()) > 0:
            # 计算IQR和离群值
            data = plot_data[i]
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [y for y in data if y < lower_bound or y > upper_bound]
            if outliers:
                # 标注最大离群值
                max_outlier = max(outliers)
                ax.annotate(f'{max_outlier:.1f}', xy=(pos_val, max_outlier),
                           xytext=(pos_val + 0.3, max_outlier + 0.5),
                           fontsize=10, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 设置标签 - 优化版
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12, fontweight='medium')
    ax.set_ylabel('Feature Density (%)\n(per sentence)', fontsize=14, fontweight='medium')
    ax.set_title('Feature Densities by Subject\n(Grouped Boxplot with Outliers)', fontsize=17, pad=30, weight='bold')
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', alpha=0.4, linewidth=0.8)

    # 移除统计说明注释，保持图表简洁

def create_simple_bar_chart(ax, df):
    """
    创建简单的柱状图 - 显示每个学科的Citation Density和Reference Count
    """
    # 计算每个学科的均值
    subject_means = df.groupby('Subject')[['Citations_density', 'Reference_no']].mean()

    # 按citation density排序
    subject_means = subject_means.sort_values('Citations_density', ascending=False)

    # 获取学科名称和数据
    subjects = subject_means.index.tolist()
    citation_means = subject_means['Citations_density'].values
    reference_means = subject_means['Reference_no'].values

    # 设置柱状图位置
    x = np.arange(len(subjects))
    width = 0.35

    # 绘制citation density柱状图
    bars1 = ax.bar(x - width/2, citation_means, width, label='Citation Density (%)',
                   color=[SUBJECT_COLORS[s] for s in subjects], alpha=0.8, edgecolor='white', linewidth=1)

    # 绘制reference count柱状图 (右侧Y轴)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, reference_means, width, label='Reference Count',
                    color=[SUBJECT_COLORS[s] for s in subjects], alpha=0.6, edgecolor='black', linewidth=1, hatch='/')

    # 添加数值标签
    for bar, value in zip(bars1, citation_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar, value in zip(bars2, reference_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 设置标签和标题
    ax.set_xlabel('Discipline', fontsize=14, fontweight='medium')
    ax.set_ylabel('Citation Density (%)\n(per sentence)', fontsize=14, fontweight='medium', color='black')
    ax2.set_ylabel('Reference Count\n(number of references)', fontsize=14, fontweight='medium', color='black')
    ax.set_title('Citation Patterns by Subject\n(Mean Values Comparison)', fontsize=17, pad=30, weight='bold')

    # 设置X轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=11, fontweight='medium')

    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 创建图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.98, 0.90),
             fontsize=10, frameon=True, fancybox=True, shadow=True)

def create_median_heatmap(ax, df, density_cols):
    """
    创建中位数热力图 - v3.0改进版

    使用中位数而非均值进行学科对比
    """
    # 计算每个学科的中位数
    subject_medians = df.groupby('Subject')[density_cols].median()

    # 按引用密度排序
    subject_medians = subject_medians.sort_values('Citations_density', ascending=False)

    # 创建热力图
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(subject_medians, cmap='Blues', annot=True, fmt='.2f',
                   linewidths=0.8, linecolor='#ffffff', cbar_kws={'shrink': 0.85, 'label': 'Median Density (%)'},
                   ax=ax, annot_kws={'size': 12, 'weight': 'medium'}, vmin=subject_medians.min().min() * 0.8)

        # 确保标签正确显示，保持原始空格
        current_labels = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels(current_labels, rotation=0, ha='right', fontsize=13, fontweight='medium')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('Subject Comparison\n(Median Values)', fontsize=17, pad=30, weight='bold')
    else:
        # matplotlib备用方案
        im = ax.imshow(subject_medians.values, cmap='Blues', aspect='auto', alpha=0.8)

        for i in range(len(subject_medians.index)):
            for j in range(len(density_cols)):
                value = subject_medians.values[i, j]
                # 根据背景色调整文字颜色以提高可读性
                text_color = 'white' if value > subject_medians.values.mean() else 'black'
                ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                       color=text_color, fontsize=10, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, aspect=15, pad=0.03)
        cbar.set_label('Median Density (%)', fontsize=12, fontweight='medium')
        cbar.ax.tick_params(labelsize=11)

        ax.set_xticks(np.arange(len(density_cols)))
        ax.set_yticks(np.arange(len(subject_medians.index)))
        ax.set_xticklabels(['Images', 'Equations', 'Tables', 'Citations'], fontsize=11, rotation=60, ha='right', fontweight='medium')
        ax.set_yticklabels([s.replace(' ', '\n') for s in subject_medians.index], fontsize=11, fontweight='medium')
        ax.set_title('Subject Comparison\n(Median Values)', fontsize=17, pad=30, weight='bold')

    ax.tick_params(axis='both', labelsize=8)

def create_robust_summary_table(ax, df):
    """
    创建稳健统计汇总表 - v3.0改进版

    使用中位数和IQR替代均值和标准差
    """
    def get_robust_stats(col, fmt='.2f'):
        median_val = df[col].median()
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return f"{median_val:{fmt}} (IQR: {iqr:{fmt}})"

    table_data = [
        ['Feature', 'Median (IQR)'],
        ['Density Features (%)', ''],
        ['Images', get_robust_stats('Images_density')],
        ['Equations', get_robust_stats('Equations_density')],
        ['Tables', get_robust_stats('Tables_density')],
        ['Citations', get_robust_stats('Citations_density')],
        ['Count Features', ''],
        ['Outlines', f"{df['Outline_no'].median():.0f}"],
        ['References', f"{df['Reference_no'].median():.0f}"],
        ['Sentences', f"{df['Sentence_no'].median():.0f}"],
        ['Depth Metrics', ''],
        ['Structure Balance', get_robust_stats('Structure_Gini', '.3f')],
        ['Citation Coverage (%)', get_robust_stats('Cit_Coverage', '.1f')],
        ['Info Density (%)', get_robust_stats('Info_Density', '.1f')]
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0.05, 0.12, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # 清除默认边框
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor('none')
        if j == 0:
            is_bold = (i == 0 or table_data[i][1] == '')
            cell.set_text_props(ha='left', weight='bold' if is_bold else 'normal')
        else:
            cell.set_text_props(ha='right')

    # 绘制三线表
    plt.gcf().canvas.draw()

    def get_row_y_bounds(row_idx):
        cell = table[row_idx, 0]
        bbox = cell.get_window_extent(ax.figure.canvas.get_renderer())
        bbox_ax = bbox.transformed(ax.transAxes.inverted())
        return bbox_ax.y1, bbox_ax.y0

    y_top_limit, y_header_below = get_row_y_bounds(0)
    _, y_bottom_limit = get_row_y_bounds(len(table_data) - 1)

    ax.axhline(y=y_top_limit, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, clip_on=False)
    ax.axhline(y=y_header_below, xmin=0.05, xmax=0.95, color='black', linewidth=0.8, clip_on=False)
    ax.axhline(y=y_bottom_limit, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, clip_on=False)

    ax.text(0.05, y_bottom_limit - 0.12, f"n = {len(df)} papers across 10 disciplines\nRobust statistics: median and IQR",
            transform=ax.transAxes, fontsize=11, style='italic', ha='left')

    ax.set_title('Robust Statistical Summary\n(Median & IQR)', fontsize=17, pad=30, weight='bold',
                fontfamily='sans-serif')
    ax.axis('off')

def create_disciplinary_dashboard(df: pd.DataFrame):
    """
    创建学科特征分析仪表板 - v3.0完整版

    包含6个核心图表：
    1. 学科指纹雷达图
    2. 分组箱线图（带离群值标注）
    3. 学科上色散点图（带回归和CI）
    4. 中位数热力图
    5. KDE分布图
    6. 稳健统计汇总表
    """
    fig = plt.figure(figsize=(24, 18))  # 进一步增大画布尺寸

    # 创建子图布局 - 最大化图表空间
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45,  # 增加两行图之间的垂直间距
                         top=0.88, bottom=0.12, left=0.06, right=0.97,  # 大幅减少边距
                         height_ratios=[1.15, 1], width_ratios=[1, 1, 1])  # 顶部行稍高

    # 1. 学科指纹雷达图
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    create_radar_chart(ax1, df)

    # 2. 分组箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    create_grouped_boxplot(ax2, df)

    # 3. 学科对比柱状图
    ax3 = fig.add_subplot(gs[0, 2])
    create_simple_bar_chart(ax3, df)

    # 4. 中位数热力图
    ax4 = fig.add_subplot(gs[1, 0])
    density_cols = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
    create_median_heatmap(ax4, df, density_cols)

    # 5. 方程式密度云雨图
    ax5 = fig.add_subplot(gs[1, 1])
    create_raincloud_plot(ax5, df)

    # 6. 稳健统计汇总表
    ax6 = fig.add_subplot(gs[1, 2])
    create_robust_summary_table(ax6, df)

    # 主标题 - 适应新布局 (已移除)
    # fig.suptitle('Academic Fingerprint Analysis: Cross-Disciplinary Feature Comparison (v3.0)',
    #              fontsize=18, fontweight='bold', y=0.94, ha='center', fontfamily='sans-serif',
    #              color='#2E3440')

    # 移除副标题

    plt.tight_layout(pad=1.2, rect=[0.05, 0.08, 0.98, 0.90])

    # 保存高分辨率PDF
    output_path = "disciplinary_profile_analysis.pdf"

    try:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf',
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        print(f"✅ 高分辨率PDF已保存: {output_path} (600 DPI, 向量格式)")

        # 同时保存PNG版本用于预览
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png',
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        print(f"✅ PNG预览图已保存: {png_path} (300 DPI)")

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        plt.savefig(output_path, format='pdf', facecolor='white', bbox_inches='tight')
        print(f"⚠️ 使用备用方法保存PDF: {output_path}")

    # 显示图表窗口
    plt.show()

def create_raincloud_plot(ax, df):
    """
    创建分学科小提琴分布图 (Disciplinary Violin Plot)
    采用葫芦型设计，垂直布局，更清晰地展示学科差异
    """
    # 1. 数据排序
    df_copy = df.copy()
    # 按方程式密度均值降序排列，数学密集学科在上方
    median_order = df_copy.groupby('Subject')['Equations_density'].mean().sort_values(ascending=False).index.tolist()

    # 2. 绘制小提琴图 (葫芦型) - 垂直布局
    for i, subject in enumerate(median_order):
        subject_data = df_copy[df_copy['Subject'] == subject]['Equations_density']

        if len(subject_data) > 1 and HAS_SCIPY:
            try:
                # 计算KDE用于小提琴形状
                kde = stats.gaussian_kde(subject_data)
                x_range = np.linspace(0, 60, 200)
                kde_values = kde(x_range)

                # 创建葫芦形：左右对称分布
                # 归一化KDE值作为宽度
                width_scale = 0.8  # 控制小提琴的宽度
                violin_width = (kde_values / kde_values.max()) * width_scale

                # 计算每个学科的垂直中心位置
                y_center = i * 2.5  # 增加间距使图更清晰

                # 绘制左侧小提琴 - 降低透明度让轮廓更清晰
                ax.fill_betweenx(x_range, y_center - violin_width, y_center,
                                color=SUBJECT_COLORS[subject], alpha=0.4, zorder=1)

                # 绘制右侧小提琴 - 降低透明度让轮廓更清晰
                ax.fill_betweenx(x_range, y_center, y_center + violin_width,
                                color=SUBJECT_COLORS[subject], alpha=0.4, zorder=1)

                # 绘制轮廓线 - 增强可见性
                # 黑色粗线作为外轮廓
                ax.plot(y_center - violin_width, x_range, color='black',
                       linewidth=2.5, alpha=1.0, zorder=3)
                ax.plot(y_center + violin_width, x_range, color='black',
                       linewidth=2.5, alpha=1.0, zorder=3)

                # 彩色细线作为内轮廓
                ax.plot(y_center - violin_width, x_range, color=SUBJECT_COLORS[subject],
                       linewidth=1.5, alpha=0.95, zorder=4)
                ax.plot(y_center + violin_width, x_range, color=SUBJECT_COLORS[subject],
                       linewidth=1.5, alpha=0.95, zorder=4)

                # 添加顶部和底部的特别强调线
                max_idx = np.argmax(kde_values)
                min_idx = np.argmin(kde_values)
                ax.plot([y_center - violin_width[max_idx], y_center + violin_width[max_idx]],
                       [x_range[max_idx], x_range[max_idx]], color='black', linewidth=3, alpha=1.0, zorder=5)
                ax.plot([y_center - violin_width[min_idx], y_center + violin_width[min_idx]],
                       [x_range[min_idx], x_range[min_idx]], color='black', linewidth=2, alpha=1.0, zorder=5)

            except Exception as e:
                print(f"警告: {subject} 小提琴图绘制失败: {e}")
                # 备用方案：绘制简单的箱线图
                y_center = i * 2.5
                median_val = subject_data.median()
                q1 = subject_data.quantile(0.25)
                q3 = subject_data.quantile(0.75)

                # 绘制箱体
                ax.add_patch(plt.Rectangle((y_center - 0.3, q1), 0.6, q3-q1,
                                         fill=True, color=SUBJECT_COLORS[subject], alpha=0.6, zorder=2))
                # 绘制中位数线
                ax.plot([y_center - 0.3, y_center + 0.3], [median_val, median_val],
                       color='white', linewidth=2, zorder=3)

    # 3. 绘制数据点 (可选，增加透明度)
    for i, subject in enumerate(median_order):
        subject_data = df_copy[df_copy['Subject'] == subject]['Equations_density']
        y_center = i * 2.5

        # 轻微的水平抖动
        np.random.seed(42)
        jitter = np.random.normal(0, 0.1, len(subject_data))

        ax.scatter([y_center + j for j in jitter], subject_data,
                  color=SUBJECT_COLORS[subject], alpha=0.4, s=15, zorder=4,
                  edgecolor='white', linewidth=0.5)

    # 3.5. 添加均值连线和标记（保留所有数据点）
    means = []
    centers = []
    for i, subject in enumerate(median_order):
        subject_data = df_copy[df_copy['Subject'] == subject]['Equations_density']
        subject_mean = subject_data.mean()
        y_center = i * 2.5
        means.append(subject_mean)
        centers.append(y_center)

        # 显示所有原始数据点（葫芦内部的采样点）
        np.random.seed(42)
        jitter = np.random.normal(0, 0.15, len(subject_data))  # 轻微水平抖动
        ax.scatter([y_center + j for j in jitter], subject_data.values,
                  color=SUBJECT_COLORS[subject], alpha=0.6, s=25, zorder=4,
                  edgecolor='white', linewidth=0.5)

        # 绘制均值点标记（蓝色系）
        ax.scatter(y_center, subject_mean, color='navy', s=60, zorder=6,
                  edgecolor='white', linewidth=2, marker='D')

        # 添加数值标签
        ax.text(y_center, subject_mean + 1, f'{subject_mean:.1f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='navy', alpha=0.9))

    # 绘制连接均值的直线（蓝色系）
    ax.plot(centers, means, color='navy', linewidth=3, linestyle='-', alpha=0.9, zorder=5)

    # 4. 设置轴标签和标题
    ax.set_xticks([i * 2.5 for i in range(len(median_order))])
    ax.set_xticklabels(median_order, rotation=45, ha='right', fontsize=9, fontweight='medium')
    ax.set_ylabel('Equations Density (%)', fontsize=12, fontweight='medium')
    # 移除x轴标签
    ax.set_title('Disciplinary Equations Distribution\n(Violin Plot by Mean: n=10 per group)',
                 fontsize=17, pad=30, weight='bold')

    # 5. 设置范围和网格
    ax.set_xlim(-0.5, (len(median_order) - 1) * 2.5 + 0.5)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 统计信息已通过数值标签显示，无需额外标注

    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def perform_robust_statistical_analysis(df: pd.DataFrame):
    """
    执行稳健统计分析 - v3.0新增

    检测离群值并提供统计建议
    """
    print("\n" + "="*80)
    print("📊 学术指纹分析 v3.0 - 稳健统计分析报告")
    print("="*80)

    # 分析句子数特征的离群值
    sentence_col = 'Sentence_no'
    if sentence_col in df.columns:
        sentences = df[sentence_col]
        median_val = sentences.median()
        Q1 = sentences.quantile(0.25)
        Q3 = sentences.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = sentences[(sentences < lower_bound) | (sentences > upper_bound)]
        outlier_papers = df.loc[outliers.index, 'Paper_ID'].tolist()

        print("🔍 句子数特征稳健统计分析:")
        print(f"   中位数: {median_val:,.0f} 句")
        print(f"   四分位距(IQR): {IQR:,.0f}")
        print(f"   离群值界限: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
        print(f"   检测到离群值: {len(outliers)} 个论文")

        if len(outliers) > 0:
            max_outlier = sentences.max()
            max_outlier_paper = df.loc[sentences.idxmax(), 'Paper_ID']
            print(f"   最大离群值: {max_outlier:,.0f} 句 (论文: {max_outlier_paper})")
            print(f"   最大值是中位数的 {max_outlier/median_val:.1f} 倍")

        print("\n💡 统计学严谨性建议:")
        print(f"   • 采用中位数({median_val:,.0f})而非均值作为中心趋势度量")
        print(f"   • 使用IQR({IQR:,.0f})而非标准差作为离散度度量")
        print(f"   • 变异系数(CV): {sentences.std()/sentences.mean():.2f} > 1，数据高度异质")

        # 学科特征对比（使用中位数）
        print("\n🏷️ 学科特征对比 (基于中位数):")
        subject_medians = df.groupby('Subject')[['Images_density', 'Equations_density',
                                                'Tables_density', 'Citations_density']].median()
        print("   引用密度排名 (从高到低):")
        citation_ranking = subject_medians['Citations_density'].sort_values(ascending=False)
        for i, (subject, value) in enumerate(citation_ranking.items(), 1):
            print(f"   {i}. {subject}: {value:.1f}%")

def main():
    """主函数 - v3.0完整工作流"""
    print("=" * 80)
    print("  学术指纹级跨学科论文特征分析系统 v3.0")
    print("  核心改进：稳健统计、学科指纹雷达图、HUSL色彩系统")
    print("=" * 80)

    base_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/survey_papers_with_pdf"

    if not os.path.exists(base_path):
        print(f"❌ 错误: 数据路径不存在 - {base_path}")
        return

    print("\n🔄 开始数据提取和处理...")
    print("-" * 60)

    df = process_all_md_files(base_path)

    if df.empty:
        print("❌ 错误: 未找到任何有效数据")
        return

    print("\n📈 数据概览:")
    print(f"   总论文数: {len(df)}")
    print(f"   学科数: {df['Subject'].nunique()}")
    print(f"   每个学科平均论文数: {len(df) // df['Subject'].nunique()}")

    # 执行稳健统计分析
    perform_robust_statistical_analysis(df)

    print("\n📊 统计摘要:")
    print("密度特征:")
    print(df[['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']].describe())
    print("\n计数特征:")
    print(df[['Outline_no', 'Reference_no', 'Sentence_no']].describe())
    print("\n深度指标:")
    print(df[['Structure_Gini', 'Cit_Coverage', 'Info_Density']].describe())

    print("\n🎨 生成学科特征分析仪表板...")
    print("-" * 60)

    create_disciplinary_dashboard(df)

    print("\n" + "=" * 80)
    print("🎉 分析完成！")
    print("📄 生成文件: disciplinary_profile_analysis.pdf (600 DPI 向量格式)")
    print("🖼️ 预览文件: disciplinary_profile_analysis.png (300 DPI)")
    print("✨ 包含6个核心可视化：雷达图、箱线图、柱状图、热力图、分布图、统计表")
    print("=" * 80)

if __name__ == "__main__":
    main()
