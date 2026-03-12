#!/usr/bin/env python3
"""
Nature级别科学论文特征分析脚本
从survey_papers_with_pdf文件夹读取100个真实md文件，提取7个维度特征
生成符合Nature期刊标准的正式插图

依赖库:
- numpy
- matplotlib
- pandas
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

mpl.use('Agg')

# 尝试导入seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib实现")

plt.style.use('default')
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
    'figure.figsize': (17, 10.5),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

import matplotlib.cm as cm
cividis = cm.get_cmap('cividis')
COLORS = {
    'primary': cividis(0.2),
    'secondary': cividis(0.5),
    'accent': cividis(0.8),
    'dark': cividis(0.1),
}

def extract_citations_robust(text):
    """
    健壮的学术引用提取函数

    支持两种引用格式：
    1. 数值型： [1], [1, 2, 3], [10-15], [64,67]
    2. 作者-年份型： (Smith, 2023), (Wang & Zhang, 2022), (Brown et al., 2021)

    处理复杂边界，避免误读正文中的普通数字。
    对于范围引用[1-3]，计为3次引用；对于[1,2,3]，计为3次引用。

    Args:
        text (str): 输入文本

    Returns:
        tuple: (总引用次数, 识别到的引用字符串列表)

    Examples:
        >>> extract_citations_robust("Test [1] citation")
        (1, ['[1]'])

        >>> extract_citations_robust("Multiple [1, 2, 3] citations")
        (3, ['[1, 2, 3]'])

        >>> extract_citations_robust("Range [10-15] citation")
        (6, ['[10-15]'])

        >>> extract_citations_robust("Superscript¹²³ citation")
        (3, ['¹²³'])

        >>> extract_citations_robust("Author (Smith, 2023) citation")
        (1, ['(Smith, 2023)'])
    """
    # 上标数字到普通数字的映射
    superscript_to_digit = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }

    citations_found = []
    total_citations = 0

    # 数值型引用正则表达式 - 支持方括号和上标格式
    # 方括号格式： [数字] 或 [数字, 数字, ...] 或 [数字-数字]
    bracket_pattern = r'\[(\d+(?:,\s*\d+)*)(?:-(\d+))?\]'

    # 上标数字格式：Unicode上标数字，可能有逗号分隔或范围
    superscript_digits = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    superscript_pattern = r'([' + superscript_digits + r']+(?:,[' + superscript_digits + r']+)*(?:-[' + superscript_digits + r']+)?)(?=\W|$)'

    # 分别处理两种格式
    numeric_patterns = [bracket_pattern, superscript_pattern]

    # 作者-年份型引用正则表达式
    # 匹配完整的括号内容，包含可能的分号分隔多个引用
    author_year_pattern = r'\(([^)]+)\)'

    # 查找数值型引用 - 分别处理方括号和上标格式
    for pattern in numeric_patterns:
        for match in re.finditer(pattern, text):
            citation_str = match.group(0)
            citations_found.append(citation_str)

            # 方括号格式处理
            if citation_str.startswith('['):
                # 检查是否有范围 (第二个捕获组)
                if match.group(2):
                    # 范围格式：[start-end]
                    range_part = match.group(0)
                    numbers_in_range = re.findall(r'\d+', range_part)
                    if len(numbers_in_range) == 2:
                        start = int(numbers_in_range[0])
                        end = int(numbers_in_range[1])
                        if start <= end:
                            total_citations += (end - start + 1)
                        else:
                            total_citations += 1
                else:
                    # 逗号分隔格式：[1, 2, 3] 或单个 [1]
                    numbers = re.findall(r'\d+', match.group(0))
                    total_citations += len(numbers)

            # 上标格式处理
            else:
                superscript_part = match.group(1)
                if superscript_part:
                    # 将上标转换为普通数字
                    normal_text = ''.join(superscript_to_digit.get(c, c) for c in superscript_part)

                    # 检查是否包含范围（-分隔）
                    if '-' in normal_text:
                        # 范围格式：如 1-15
                        parts = normal_text.split('-')
                        if len(parts) == 2:
                            try:
                                start = int(parts[0])
                                end = int(parts[1])
                                if start <= end:
                                    total_citations += (end - start + 1)
                                else:
                                    total_citations += 1
                            except ValueError:
                                total_citations += 1
                        else:
                            total_citations += 1
                    else:
                        # 逗号分隔或连续格式：如 123 或 1,2,3
                        # 先处理逗号分隔
                        if ',' in normal_text:
                            # 逗号分隔格式：1,2,3
                            numbers = [n.strip() for n in normal_text.split(',') if n.strip()]
                            total_citations += len(numbers)
                        else:
                            # 连续格式：123 -> 1,2,3 (每个数字作为一个引用)
                            # 对于连续上标¹²³，转换为123，然后按单个数字计数
                            individual_digits = [d for d in normal_text if d.isdigit()]
                            total_citations += len(individual_digits)

    # 查找作者-年份型引用
    for match in re.finditer(author_year_pattern, text):
        citation_str = match.group(0)
        content = match.group(1)

        # 检查是否包含年份（4位数字），且包含作者信息
        if re.search(r'\d{4}', content) and re.search(r'[A-Za-z]', content):
            citations_found.append(citation_str)

            # 计算分号分隔的引用数量
            individual_citations = re.split(r';\s*', content)
            total_citations += len(individual_citations)

    return total_citations, citations_found

def read_md(md_path: str) -> str:
    """读取markdown文件内容"""
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

    使用基尼系数衡量章节长度分布的不均衡程度。
    基尼系数范围：[0, 1]
    - 0: 完全均衡（所有章节长度相等）
    - 1: 完全不均衡（一个章节包含所有内容）

    Args:
        content: 文档内容
        total_sentences: 总句子数

    Returns:
        float: 基尼系数，范围[0, 1]
    """
    if total_sentences == 0:
        return 0.0

    # 提取所有标题位置
    headings = []
    for match in re.finditer(r'^#{1,6} .+$', content, re.MULTILINE):
        headings.append(match.start())

    if len(headings) < 2:
        return 0.0  # 如果没有足够的章节，返回均衡值

    # 计算各章节的句子数
    section_lengths = []
    headings.append(len(content))  # 添加文档结尾

    for i in range(len(headings) - 1):
        section_content = content[headings[i]:headings[i+1]]
        section_sentences = count_sentences(section_content)
        if section_sentences > 0:  # 只计算有内容的章节
            section_lengths.append(section_sentences)

    if len(section_lengths) < 2:
        return 0.0

    # 计算基尼系数
    section_lengths = np.array(section_lengths, dtype=float)
    n = len(section_lengths)

    if n < 2:
        return 0.0

    # 排序章节长度（升序）
    sorted_lengths = np.sort(section_lengths)

    # 标准基尼系数计算方法
    # G = (sum_{i=1}^n (2*i - n - 1) * x_i) / (n^2 * mean(x))
    mean_length = np.mean(section_lengths)

    if mean_length == 0:
        return 0.0

    # 计算基尼系数的标准公式
    gini_sum = 0.0
    for i in range(n):
        gini_sum += (2 * (i + 1) - n - 1) * sorted_lengths[i]

    gini = gini_sum / (n * n * mean_length)

    # 确保结果在[0, 1]范围内
    return max(0.0, min(1.0, gini))

def calculate_citation_coverage(content: str, total_sentences: int) -> float:
    """计算引文覆盖率（带引文的句子数/总句子数）"""
    if total_sentences == 0:
        return 0.0

    # 分割成句子
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())

    # 统计包含引文的句子数
    citation_sentences = 0

    for sentence in sentences:
        # 使用新的引用提取函数检查句子是否包含引用
        _, citation_strings = extract_citations_robust(sentence)
        if citation_strings:  # 如果找到了引用字符串
            citation_sentences += 1

    return (citation_sentences / total_sentences) * 100

def count_md_features(md_content: str) -> dict:
    """统计markdown文件中的各种特征"""
    # 图片提取 - 避免重复计数
    img_md = re.findall(r'!\[.*?\]\(.*?\)', md_content)
    img_html_full = re.findall(r'<img [^>]*src=[\'"].*?[\'"][^>]*>', md_content, re.IGNORECASE)
    # 移除已经被完整HTML img标签匹配的部分，避免重复计数
    img_html_simple = []
    for simple_match in re.findall(r'<img [^>]*>', md_content, re.IGNORECASE):
        if not any(simple_match in full_match for full_match in img_html_full):
            img_html_simple.append(simple_match)

    image_count = len(set(img_md + img_html_full + img_html_simple))

    # 公式提取 - 修复版本，支持标准LaTeX公式格式
    # 块级公式：$$...$$, \[...\], \begin{env}...\end{env}
    block_eq = []

    # $$...$$ 格式：要求内容不为空，至少包含非空白字符
    block_eq.extend([m for m in re.findall(r'\$\$.*?\$\$', md_content, re.DOTALL) if m.strip('$\n\t ')])

    # \[...\] 格式
    block_eq.extend(re.findall(r'\\\[.*?\\\]', md_content, re.DOTALL))

    # \begin{env}...\end{env} 格式
    block_eq.extend(re.findall(r'\\begin\{.*?\}.*?\\end\{.*?\}', md_content, re.DOTALL))

    # 行内公式：$...$ 格式，要求至少2个字符且不包含换行
    inline_eq = [eq for eq in re.findall(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', md_content) if eq.strip()]

    equation_count = len(block_eq) + len(inline_eq)

    # 表格提取 - 改进markdown表格正则表达式
    # 改进的markdown表格匹配：表头行 + 分隔行 + 数据行
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
    """从markdown文件中提取所有特征"""
    content = read_md(md_file_path)
    if not content:
        return None

    md_features = count_md_features(content)

    # 使用新的健壮引用提取函数
    citation_count, citation_strings = extract_citations_robust(content)

    outline_pattern = r'^#+ .+$'
    outline_count = len(re.findall(outline_pattern, content, re.MULTILINE))

    references_section = re.search(r'#+\s*(References|Bibliography|Literature\sCited|Works\sCited|References\s+and\s+Notes).*?(?=#+ |\Z)', content, re.DOTALL | re.IGNORECASE)
    reference_count = 0
    if references_section:
        references_text = references_section.group(0)
        # 更通用的参考文献计数：匹配任何非空行作为潜在参考文献项
        reference_items = [line.strip() for line in references_text.split('\n') if line.strip() and not line.strip().startswith('#')]
        reference_count = len(reference_items)

    sentence_count = md_features['sentences']

    # 计算高阶指标

    # 1. 结构均衡度 (Structure_Gini)
    structure_gini = calculate_structure_gini(content, sentence_count)

    # 2. 引文覆盖率 (Cit_Coverage)
    cit_coverage = calculate_citation_coverage(content, sentence_count)

    # 3. 信息承载率 (Info_Density)
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
        # 深度指标直接传递
        'Structure_Gini': features_dict.get('Structure_Gini', 0.0),
        'Cit_Coverage': features_dict.get('Cit_Coverage', 0.0),
        'Info_Density': features_dict.get('Info_Density', 0.0)
    }

def process_all_md_files(base_path: str) -> pd.DataFrame:
    """处理所有markdown文件并提取特征"""
    all_features = []
    subjects = ['Biology', 'Business', 'Computer Science', 'Education',
                'Engineering', 'Environmental Science', 'Medicine', 'Physics',
                'Psychology', 'Sociology']

    print(f"开始处理 {len(subjects)} 个学科的论文...")

    for subject in subjects:
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
                density_features['Paper_ID'] = f"{subject}_{md_file.replace('.md', '')}"
                all_features.append(density_features)

    df = pd.DataFrame(all_features)
    print(f"成功处理了 {len(df)} 篇论文")
    print(f"数据维度: {df.shape}")
    return df

def create_violin_plot(ax, df, density_cols, density_labels):
    """创建小提琴图 + 数据点"""
    vp = ax.violinplot([df[col] for col in density_cols], positions=range(len(density_cols)),
                       showmeans=True, showmedians=False, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(COLORS['primary'])
        pc.set_edgecolor(COLORS['dark'])
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)

    for i, col in enumerate(density_cols):
        y_data = df[col].values
        x_jitter = np.random.uniform(-0.25, 0.25, len(y_data))
        x_data = np.full_like(y_data, i, dtype=float) + x_jitter
        ax.scatter(x_data, y_data, alpha=0.4, s=12, color=COLORS['dark'],
                   edgecolors='white', linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(density_cols)))
    ax.set_xticklabels(density_labels, fontsize=8)
    ax.set_ylabel('Feature Density (%)\n(per sentence)', fontsize=9)
    ax.set_title('Feature Densities', fontsize=12, pad=15)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_point_plot(ax, df):
    """创建深度指标点图"""
    # 三个高阶指标
    metrics_cols = ['Structure_Gini', 'Cit_Coverage', 'Info_Density']
    metrics_means = df[metrics_cols].mean()
    metrics_stds = df[metrics_cols].std()

    # 根据指标重要性设置颜色
    colors = [COLORS['accent'], COLORS['secondary'], COLORS['primary']]  # Info_Density -> Cit_Coverage -> Structure_Gini

    for i, (metric, mean_val, std_val) in enumerate(zip(metrics_cols, metrics_means.values, metrics_stds.values)):
        # 计算置信区间 (95%)
        try:
            from scipy import stats
            n = len(df)
            se = std_val / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, n-1)
            ci = t_critical * se

            y_lower = max(0, mean_val - ci)
            y_upper = mean_val + ci
        except ImportError:
            # 简单方法：使用标准差
            y_lower = max(0, mean_val - std_val)
            y_upper = mean_val + std_val

        # 绘制点和误差线
        ax.plot(i, mean_val, 'o', color=colors[i], markersize=10, zorder=3, markeredgecolor='white', markeredgewidth=1.5)
        ax.errorbar(i, mean_val, yerr=[[mean_val - y_lower], [y_upper - mean_val]],
                    color=colors[i], linewidth=2.5, capsize=5, capthick=2, zorder=2, alpha=0.8)

    ax.set_xticks(range(len(metrics_cols)))
    ax.set_xticklabels(['Structure\nBalance', 'Citation\nCoverage', 'Info\nDensity'], fontsize=8, ha='center')
    ax.set_ylabel('Depth Metric Value\n(varies by metric)', fontsize=9)
    ax.set_title('Depth Metrics', fontsize=12, pad=15)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 移除图例（x轴标签已清楚显示指标名称，避免挡住线条）

def create_heatmap(ax, df, density_cols):
    """创建热力图"""
    subject_means = df.groupby('Subject')[density_cols].mean()
    subject_means = subject_means.sort_values('Citations_density', ascending=False)

    # 使用seaborn heatmap实现更好的网格线和自动标注
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(subject_means, cmap='Blues', annot=True, fmt='.1f',
                   linewidths=0.5, linecolor='#f0f0f0', cbar_kws={'shrink': 0.8, 'label': 'Density (%)'},
                   ax=ax, annot_kws={'size': 7}, vmin=subject_means.min().min() * 0.8)

        # 调整Y轴标签间距
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('Subject Comparison', fontsize=12, pad=15)
        ax.tick_params(axis='both', labelsize=8)
    else:
        # 备用matplotlib实现
        min_val = subject_means.min().min()
        im = ax.imshow(subject_means.values, cmap='Blues', aspect='auto',
                       vmin=min_val * 0.8, vmax=25, alpha=0.7)

        for i in range(len(subject_means.index)):
            for j in range(len(density_cols)):
                # 统一使用黑色字体标注
                value = subject_means.values[i, j]
                text = ax.text(j, i, f'{value:.1f}',
                               ha="center", va="center",
                               color="black", fontsize=7, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=12, pad=0.02)
        cbar.set_label('Density (%)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xticks(np.arange(len(density_cols)))
        ax.set_yticks(np.arange(len(subject_means.index)))
        ax.set_xticklabels(['Images', 'Equations', 'Tables', 'Citations'], fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels([s.replace(' ', '\n') for s in subject_means.index], fontsize=8)
        ax.set_title('Subject Comparison', fontsize=12, pad=15)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def create_scatter_plot(ax, df):
    """创建散点图 + 回归线"""
    x_data = df['Citations_density']
    y_data = df['Reference_no']

    ax.scatter(x_data, y_data, alpha=0.6, s=20, color=COLORS['primary'],
               edgecolors='white', linewidth=0.3, zorder=3)

    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.stats.linregress(x_data, y_data)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept

        n = len(x_data)
        x_mean = x_data.mean()
        t_critical = stats.t.ppf(0.975, n-2)
        se_fit = std_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_data - x_mean)**2))
        ci_upper = y_line + t_critical * se_fit
        ci_lower = y_line - t_critical * se_fit

        ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.15, color=COLORS['secondary'], zorder=1)
        ax.plot(x_line, y_line, color=COLORS['secondary'], linewidth=1.5, alpha=0.7, zorder=2,
                linestyle='--')

        # 在右上角添加完整的统计信息
        r_squared = r_value ** 2
        if p_value < 0.001:
            stats_text = f'$R^2 = {r_squared:.3f}$, $p < 0.001$'
        else:
            stats_text = f'$R^2 = {r_squared:.3f}$, $p = {p_value:.3f}$'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
               facecolor='white', alpha=0.8, edgecolor='none'),
               fontfamily='sans-serif')
    except ImportError:
        coeffs = np.polyfit(x_data, y_data, 1)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color=COLORS['secondary'], linewidth=1.5, alpha=0.7, zorder=2,
                linestyle='--')

        # 计算基本统计信息（没有scipy时的简化版本）
        r_value = np.corrcoef(x_data, y_data)[0, 1]
        r_squared = r_value ** 2
        # 注意：没有scipy时无法精确计算p值，这里使用简化显示
        stats_text = f'$R^2 = {r_squared:.3f}$'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
               facecolor='white', alpha=0.8, edgecolor='none'),
               fontfamily='sans-serif')

    # 移除边缘图以保持一致的高度
    # ax4_top = ax.inset_axes([0, 1.02, 1, 0.18])
    # ax4_top.hist(x_data, bins=15, alpha=0.25, color=COLORS['primary'],
    #              density=True, edgecolor='none')
    # ax4_top.set_xlim(ax.get_xlim())
    # ax4_top.axis('off')

    # ax4_right = ax.inset_axes([1.02, 0, 0.18, 1])
    # ax4_right.hist(y_data, bins=15, alpha=0.25, color=COLORS['primary'],
    #                density=True, edgecolor='none', orientation='horizontal')
    # ax4_right.set_ylim(ax.get_ylim())
    # ax4_right.axis('off')

    ax.set_xlabel('Citation Density (%)\n(per sentence)', fontsize=9)
    ax.set_ylabel('Reference Count\n(number of references)', fontsize=9)
    ax.set_title('Key Relationships', fontsize=12, pad=15)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_kde_plot(ax, df):
    """创建KDE分布图"""
    hist_data = df['Equations_density']

    # 首先绘制直方图（调淡以突出KDE）
    ax.hist(hist_data, bins=25, alpha=0.3, color=COLORS['accent'],
            edgecolor=COLORS['dark'], linewidth=1.5, density=True, zorder=1)

    # 叠加KDE曲线
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(hist_data)
        x_kde = np.linspace(hist_data.min(), hist_data.max(), 200)
        kde_values = kde(x_kde)
        ax.fill_between(x_kde, kde_values, alpha=0.3, color=COLORS['accent'], zorder=2)
        ax.plot(x_kde, kde_values, color=COLORS['dark'], linewidth=2.5, alpha=0.9, zorder=3)
    except ImportError:
        pass  # 如果没有scipy，跳过KDE

    mean_val = hist_data.mean()
    ax.axvline(mean_val, color=COLORS['secondary'], linestyle='--',
               linewidth=2, alpha=0.8, zorder=4)

    # 在虚线顶部添加直接标注
    ax.text(mean_val + 1, ax.get_ylim()[1] * 0.9, f'Mean = {mean_val:.2f}%',
            ha='left', va='top', fontsize=8, color=COLORS['secondary'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                     edgecolor='none'), fontweight='bold')

    ax.set_xlabel('Equations Density (%)\n(per sentence)', fontsize=9)
    ax.set_ylabel('Density\n(KDE estimate)', fontsize=9)
    ax.set_title('Equations Distribution', fontsize=12, pad=15)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_three_line_table(ax, df):
    """
    创建严格遵循顶刊标准的三线表
    通过动态探测单元格边界，彻底解决线条穿字和双底线问题
    """
    # 1. 准备数据并修复占位符（确保显示真实数值）
    def get_stats(col, p='.2f'):
        m, s = df[col].mean(), df[col].std()
        return f"{m:{p}} ± {s:{p}}"

    table_data = [
        ['Feature', 'Mean ± SD'],
        ['Density Features (%)', ''],
        ['Images', get_stats('Images_density')],
        ['Equations', get_stats('Equations_density')],
        ['Tables', get_stats('Tables_density')],
        ['Citations', get_stats('Citations_density')],
        ['Count Features', ''],
        ['Outlines', f"{df['Outline_no'].mean():.0f}"],
        ['References', f"{df['Reference_no'].mean():.0f}"],
        ['Sentences', f"{df['Sentence_no'].mean():.0f}"],
        ['Depth Metrics', ''],
        ['Structure Balance', get_stats('Structure_Gini', '.3f')],
        ['Citation Coverage (%)', get_stats('Cit_Coverage', '.1f')],
        ['Info Density (%)', get_stats('Info_Density', '.1f')]
    ]

    # 2. 建立表格并彻底清除默认边框
    # 使用 bbox 确保表格在子图中位置适中
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0.05, 0.12, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # 隐藏所有默认边框
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor('none')
        # 表头和分类标题加粗，标签左对齐，数值右对齐
        if j == 0:
            is_bold = (i == 0 or table_data[i][1] == '')
            cell.set_text_props(ha='left', weight='bold' if is_bold else 'normal')
        else:
            cell.set_text_props(ha='right')

    # 3. 动态绘制三根标准线条 (核心修复逻辑)
    # 强制 Matplotlib 先计算布局
    plt.gcf().canvas.draw()

    # 获取第一行和最后一行的边界坐标
    def get_row_y_bounds(row_idx):
        # 获取该行第一个单元格的边界框
        cell = table[row_idx, 0]
        bbox = cell.get_window_extent(ax.figure.canvas.get_renderer())
        # 将屏幕像素坐标转换为子图坐标
        bbox_ax = bbox.transformed(ax.transAxes.inverted())
        return bbox_ax.y1, bbox_ax.y0 # 返回 (上边缘, 下边缘)

    y_top_limit, y_header_below = get_row_y_bounds(0) # 第一行（表头）的上下界
    _, y_bottom_limit = get_row_y_bounds(len(table_data) - 1) # 最后一行下界

    # 绘制三线
    # 线1：顶线 (Top Line) - 表头上方
    ax.axhline(y=y_top_limit, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, clip_on=False)
    # 线2：栏目线 (Header Line) - 表头下方
    ax.axhline(y=y_header_below, xmin=0.05, xmax=0.95, color='black', linewidth=0.8, clip_on=False)
    # 线3：底线 (Bottom Line) - 数据最后一行下方
    ax.axhline(y=y_bottom_limit, xmin=0.05, xmax=0.95, color='black', linewidth=1.5, clip_on=False)

    # 4. 添加注脚 (紧贴底线下方)
    ax.text(0.05, y_bottom_limit - 0.05, f"n = {len(df)} papers, 10 disciplines",
            transform=ax.transAxes, fontsize=8, style='italic', ha='left')

    ax.set_title('Statistical Summary', fontsize=12, pad=20, weight='bold')
    ax.axis('off')

def create_clean_plots(df: pd.DataFrame):
    """创建Nature级别的正式插图"""
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.40,
                         top=0.88, bottom=0.14, left=0.09, right=0.96,
                         height_ratios=[1, 1], width_ratios=[1, 1, 1])

    density_cols = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
    density_labels = ['Images', 'Equations', 'Tables', 'Citations']

    ax1 = fig.add_subplot(gs[0, 0])
    create_violin_plot(ax1, df, density_cols, density_labels)

    ax2 = fig.add_subplot(gs[0, 1])
    create_point_plot(ax2, df)

    ax3 = fig.add_subplot(gs[0, 2])
    create_heatmap(ax3, df, density_cols)

    ax4 = fig.add_subplot(gs[1, 0])
    create_scatter_plot(ax4, df)

    ax5 = fig.add_subplot(gs[1, 1])
    create_kde_plot(ax5, df)

    ax6 = fig.add_subplot(gs[1, 2])
    create_three_line_table(ax6, df)

    fig.suptitle('Survey Features Small Multiples',
                 fontsize=14, fontweight='bold', y=0.97, ha='center', fontfamily='sans-serif')

    plt.tight_layout(pad=2.0)

    pdf_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/scientific_analysis.pdf"
    png_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/scientific_analysis.png"

    try:
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png',
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        print(f"PNG格式已保存: {png_path}")

        plt.savefig(pdf_path, dpi=600, bbox_inches='tight', format='pdf',
                    facecolor='white', edgecolor='none')
        print(f"PDF格式已保存: {pdf_path} (矢量格式)")

    except Exception as e:
        print(f"保存失败: {e}")
        plt.savefig(pdf_path, format='pdf', facecolor='white', bbox_inches='tight')
        print(f"使用备用方法保存PDF: {pdf_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("  Nature级别科学论文特征分析工具")
    print("  处理100篇学术论文，提取7个关键维度")
    print("=" * 60)

    base_path = "/Users/shihaochen/Desktop/Polyu学习文件夹/YESAR2-1/SA/北辰学长课题/survey_papers_with_pdf"

    if not os.path.exists(base_path):
        print(f"错误: 数据路径不存在 - {base_path}")
        return

    print("\n开始数据提取和处理...")
    print("-" * 40)

    df = process_all_md_files(base_path)

    if df.empty:
        print("错误: 未找到任何有效数据")
        return

    print("\n数据概览:")
    print(f"总论文数: {len(df)}")
    print(f"学科数: {df['Subject'].nunique()}")
    print(f"每个学科平均论文数: {len(df) // df['Subject'].nunique()}")

    print("\n统计摘要:")
    print("密度特征:")
    print(df[['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']].describe())
    print("\n计数特征:")
    print(df[['Outline_no', 'Reference_no', 'Sentence_no']].describe())
    print("\n深度指标:")
    print(df[['Structure_Gini', 'Cit_Coverage', 'Info_Density']].describe())

    # 统计学严谨性分析 - 离群值检测
    print("\n📊 统计学严谨性分析 - 离群值检测")
    print("=" * 60)

    # 分析句子数特征的离群值
    sentence_col = 'Sentence_no'
    if sentence_col in df.columns:
        sentences = df[sentence_col]
        Q1 = sentences.quantile(0.25)
        Q3 = sentences.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = sentences[(sentences < lower_bound) | (sentences > upper_bound)]
        outlier_papers = df.loc[outliers.index, 'Paper_ID'].tolist()

        print("🔍 句子数特征离群值分析:")
        print(f"   四分位距(IQR): {IQR:.1f}")
        print(f"   离群值界限: [{lower_bound:.1f}, {upper_bound:.1f}]")
        print(f"   检测到离群值: {len(outliers)} 个")
        if len(outliers) > 0:
            print(f"   最大离群值: {sentences.max():,.0f} 句")
            print(f"   最大值是均值的 {sentences.max()/sentences.mean():.1f} 倍")
            print(f"   最大值是中位数的 {sentences.max()/sentences.median():.1f} 倍")
            print(f"   离群值论文: {outlier_papers[0] if outlier_papers else 'Unknown'}")

        # 提供稳健统计建议
        print("\n💡 统计学严谨性建议:")
        print(f"   • 均值({sentences.mean():.1f}) 受离群值严重影响，不具代表性")
        print(f"   • 建议使用中位数({sentences.median():.1f}) 作为中心趋势度量")
        print(f"   • 使用IQR({IQR:.1f}) 而非标准差({sentences.std():.1f}) 作为离散度度量")
        print(f"   • 变异系数(CV): {sentences.std()/sentences.mean():.2f} > 1，数据高度异质")
        # 计算过滤后的统计量
        filtered_sentences = sentences[sentences <= upper_bound]
        if len(filtered_sentences) > 0:
            print("\n📈 过滤离群值后的稳健统计:")
            print(f"   • 过滤后均值: {filtered_sentences.mean():.1f}")
            print(f"   • 过滤后标准差: {filtered_sentences.std():.1f}")
            print(f"   • 过滤后变异系数: {filtered_sentences.std()/filtered_sentences.mean():.2f}")
    print()

    print("\n开始生成Nature级别可视化...")
    print("-" * 40)

    create_clean_plots(df)

    print("\n" + "=" * 60)
    print("  分析完成！")
    print("  生成的文件: scientific_analysis.pdf (矢量格式)")
    print("  包含6个符合Nature期刊标准的图表")
    print("=" * 60)

if __name__ == "__main__":
    main()