import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def read_md(md_path: str) -> str:
    """
    Read and return the contents of a markdown file.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        str: Contents of the markdown file
    """
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
    """
    Count the number of sentences in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of sentences
    """
    sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
    sentences = [s for s in sentences if s]
    return len(sentences)

def count_md_features(md_content: str) -> dict[str, int]:
    """
    Count images, equations, tables, and sentences in markdown content.
    
    Args:
        md_content (str): Markdown text
        
    Returns:
        dict[str, int]: Counts of {'images': int, 'equations': int, 'tables': int, 'sentences': int}
    """
    img_md = re.findall(r'!\[.*?\]\(.*?\)', md_content)
    img_html = re.findall(r'<img [^>]*src=[\'"].*?[\'"][^>]*>', md_content, re.IGNORECASE)
    img_html2 = re.findall(r'<img [^>]*>', md_content, re.IGNORECASE)
    image_count = len(set(img_md + img_html + img_html2))

    block_eq = re.findall(r'\$\$.*?\$\$', md_content, re.DOTALL)
    block_eq += re.findall(r'\\\[.*?\\\]', md_content, re.DOTALL)
    block_eq += re.findall(r'\\begin\{.*?\}.*?\\end\{.*?\}', md_content, re.DOTALL)
    inline_eq = re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', md_content)
    equation_count = len(block_eq) + len(inline_eq)

    md_tables = re.findall(
        r'(?:\|[^\n]*\n)+\|[\s\-:|]+\|(?:\n\|[^\n]*)*', md_content)
    html_tables = re.findall(r'<table[\s\S]*?</table>', md_content, re.IGNORECASE)
    table_count = len(md_tables) + len(html_tables)

    sentence_count = count_sentences(md_content)
    return {
        'images': image_count,
        'equations': equation_count,  
        'tables': table_count,
        'sentences': sentence_count
    }

def extract_features_from_md(md_file_path):
    """
    从Markdown文件中提取特征，完全基于Markdown内容
    
    Parameters:
    -----------
    md_file_path : str
        Markdown文件的路径
        
    Returns:
    --------
    dict
        包含提取的特征的字典
    """
    # 读取Markdown文件内容
    content = read_md(md_file_path)
    if not content:
        return None
    
    # 使用count_md_features函数提取基本特征
    md_features = count_md_features(content)
    
    # 计算引用数量 (使用[@...]格式或数字引用[1])
    citation_pattern = r'\[@[^\]]+\]|\[\d+(,\s*\d+)*\]'
    citation_count = len(re.findall(citation_pattern, content))
    
    # 计算大纲编号数量 (使用#、##、###等标题)
    outline_pattern = r'^#+ .+$'
    outline_count = len(re.findall(outline_pattern, content, re.MULTILINE))
    
    # 计算参考文献数量 (References部分)
    references_section = re.search(r'#+ References.*?(?=#+ |\Z)', content, re.DOTALL | re.IGNORECASE)
    reference_count = 0
    if references_section:
        references_text = references_section.group(0)
        # 计算参考文献条目 (通常以数字开头或作者名开头)
        reference_items = re.findall(r'^\[?\d+\]?\.|\d+\.|\* |\- |^[A-Z][a-zA-Z]+.*\d{4}', references_text, re.MULTILINE)
        reference_count = len(reference_items)
    
    # 使用count_md_features中的句子数量
    sentence_count = md_features['sentences']
    
    return {
        'Images_count': md_features['images'],
        'Equations_count': md_features['equations'],  # 保持内部变量名
        'Tables_count': md_features['tables'],
        'Citations_count': citation_count,
        'Outline_count': outline_count,
        'Reference_count': reference_count,
        'Sentence_count': sentence_count
    }

def calculate_density_features(features_dict):
    """
    计算密度特征 (将计数特征除以句子数量并乘以100得到百分比)
    
    Parameters:
    -----------
    features_dict : dict
        包含计数特征的字典
        
    Returns:
    --------
    dict
        包含密度特征的字典
    """
    sentence_count = features_dict.get('Sentence_count', 1)  # 避免除以0
    
    return {
        'Images_density': features_dict.get('Images_count', 0) / sentence_count * 100,
        'Equations_density': features_dict.get('Equations_count', 0) / sentence_count * 100,  # 保持内部变量名
        'Tables_density': features_dict.get('Tables_count', 0) / sentence_count * 100,
        'Citations_density': features_dict.get('Citations_count', 0) / sentence_count * 100,
        'Outline_no': features_dict.get('Outline_count', 0),
        'Reference_no': features_dict.get('Reference_count', 0),
        'Sentence_no': features_dict.get('Sentence_count', 0)
    }

def process_subject_directory(subject_path):
    """
    处理一个学科目录，计算该学科所有论文的平均特征
    
    Parameters:
    -----------
    subject_path : str
        学科目录的路径
        
    Returns:
    --------
    dict
        包含该学科平均特征的字典
    """
    # 获取所有Markdown文件
    md_files = [f for f in os.listdir(subject_path) if f.endswith('.md')]
    
    all_features = []
    
    for md_file in md_files:  # 处理所有文件
        md_file_path = os.path.join(subject_path, md_file)
        
        features = extract_features_from_md(md_file_path)
        
        if features:
            density_features = calculate_density_features(features)
            all_features.append(density_features)
    
    # 计算该学科的平均特征
    if all_features:
        avg_features = {}
        for key in all_features[0].keys():
            avg_features[key] = np.mean([f[key] for f in all_features])
        return avg_features
    else:
        return None

def plot_survey_small_multiples(data, save_path=None):
    """
    Plot small multiples horizontal bar plots showing categories for each metric.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the survey features
    save_path : str, optional
        Path to save the figure. If None, the figure will be displayed.
    """
    # Set the style for academic publication
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # Define features and categories
    density_features = ['Images_density', 'Equations_density', 'Tables_density', 'Citations_density']
    count_features = ['Outline_no', 'Reference_no', 'Sentence_no']
    
    # Get categories from data index
    categories = data.index.tolist()
    
    # 定义学科颜色
    subject_colors = [
        '#C5C8E6',  # 非常淡的薰衣草色
        '#B8BCE0',  # 淡薰衣草灰
        '#ABB0DA',  # 淡灰紫色
        '#9EA4D4',  # 淡紫灰色
        '#9198CE',  # 淡蓝紫色
        '#848CC8',  # 淡紫蓝色
        '#7780C2',  # 淡靛蓝色
        '#6A74BC',  # 淡深蓝色
        '#5D68B6',  # 淡藏青色
        '#505CB0'   # 最深的但仍然柔和的蓝紫色
    ]
    
    # 确保颜色数量与学科数量匹配
    if len(categories) != len(subject_colors):
        # 如果学科数量不等于10，使用默认颜色
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(categories)))
    else:
        # 使用指定的颜色
        colors = subject_colors
    
    # Create figure with subplots in two rows
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
    
    # 添加大标题 - 上移与图表分开
    fig.suptitle('Survey Features Small Multiples', fontsize=20, fontweight='bold', 
                 fontfamily='Times New Roman', y=0.995)
    
    # 展平axes数组以便迭代
    axes = axes.flatten()
    
    # Plot density features
    for i, feature in enumerate(density_features):
        # Get values for this feature (已经是百分比形式)
        values = data[feature].values
        
        # Create horizontal bar plot with specified colors
        bars = axes[i].barh(categories, values, color=colors, height=0.7)
        
        # 处理特征名称，将Equation替换为Expression
        feature_name = feature.replace('_', ' ').title()
        # 单独替换Equation为Expression
        if 'Equations Density' in feature_name:
            feature_name = feature_name.replace('Equations Density', 'Expression Density')
        
        axes[i].set_title(f'{feature_name}\n(Count/Sentence)÷100', pad=15, fontsize=14, 
                          fontfamily='Times New Roman')
        
        # Add value labels at the end of bars - 保留两位小数（百分比显示）
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.2f}%',
                       ha='left', va='center', fontsize=12, fontfamily='Times New Roman')
        
        # Remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        # Add grid lines
        axes[i].xaxis.grid(True, linestyle='--', alpha=0.3)
        axes[i].yaxis.grid(True, linestyle='--', alpha=0.3)
        axes[i].set_axisbelow(True)
        
        # Invert y-axis to have categories in the same order
        axes[i].invert_yaxis()
        
        # Adjust y-axis ticks to reduce spacing
        axes[i].set_yticks(np.arange(len(categories)))
        axes[i].set_yticklabels(categories, fontsize=12, fontfamily='Times New Roman')
        
        # Adjust x-axis tick labels to show percentage
        axes[i].tick_params(axis='x', labelsize=12)
        for label in axes[i].get_xticklabels():
            label.set_fontfamily('Times New Roman')
        
        # 设置x轴标签为百分比
        axes[i].set_xlabel('Percentage (%)', fontsize=12, fontfamily='Times New Roman')
        
        # Remove y-axis margin
        axes[i].margins(y=0.1)
    
    # Plot count features
    for i, feature in enumerate(count_features):
        # Get values for this feature
        values = data[feature].values
        
        # Create horizontal bar plot with specified colors
        bars = axes[i+4].barh(categories, values, color=colors, height=0.7)
        
        # Customize the plot - 保持子图标题位置不变
        feature_name = feature.replace('_', ' ').title()
        axes[i+4].set_title(f'{feature_name}', pad=15, fontsize=16, 
                            fontfamily='Times New Roman')
        
        # Add value labels at the end of bars - 保留两位小数
        for bar in bars:
            width = bar.get_width()
            axes[i+4].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.2f}',
                       ha='left', va='center', fontsize=12, fontfamily='Times New Roman')
        
        # Remove top and right spines
        axes[i+4].spines['top'].set_visible(False)
        axes[i+4].spines['right'].set_visible(False)
        
        # Add grid lines
        axes[i+4].xaxis.grid(True, linestyle='--', alpha=0.3)
        axes[i+4].yaxis.grid(True, linestyle='--', alpha=0.3)
        axes[i+4].set_axisbelow(True)
        
        # Invert y-axis to have categories in the same order
        axes[i+4].invert_yaxis()
        
        # Adjust y-axis ticks to reduce spacing
        axes[i+4].set_yticks(np.arange(len(categories)))
        axes[i+4].set_yticklabels(categories, fontsize=12, fontfamily='Times New Roman')
        
        # Adjust x-axis tick labels
        axes[i+4].tick_params(axis='x', labelsize=12)
        for label in axes[i+4].get_xticklabels():
            label.set_fontfamily('Times New Roman')
        
        # Remove y-axis margin
        axes[i+4].margins(y=0.1)
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 调整子图区域顶部边距
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='pdf')
    else:
        plt.show()

# 主程序
def main():
    # 设置数据集路径（请根据实际路径修改）
    base_path = "/Users/joanna/Desktop/Dataset_Md_2"
    
    # 指定要处理的学科列表
    target_subjects = [
        "Medicine", "Environmental Science", "Sociology", "Education",
        "Engineering", "Physics", "Psychology", "Business",
        "Computer Science", "Biology"
    ]
    
    # 存储所有学科的特征
    subject_features = {}
    
    # 处理每个学科
    for subject in target_subjects:
        subject_path = os.path.join(base_path, subject)
        if os.path.exists(subject_path):
            print(f"处理学科: {subject}")
            features = process_subject_directory(subject_path)
            if features:
                subject_features[subject] = features
                print(f"{subject}学科的特征提取完成")
            else:
                print(f"无法处理{subject}学科")
        else:
            print(f"学科路径不存在: {subject_path}")
    
    # 创建DataFrame
    if subject_features:
        df = pd.DataFrame.from_dict(subject_features, orient='index')
        df = df[['Images_density', 'Equations_density', 'Tables_density', 'Citations_density',
                'Outline_no', 'Reference_no', 'Sentence_no']]
        
        # 确保学科顺序与target_subjects一致
        df = df.reindex(target_subjects)
        
        # 打印数据
        print("所有学科特征数据:")
        print(df)
        
        # 保存数据到CSV
        csv_path = os.path.join(os.path.expanduser("~"), "Desktop", "all_subjects_features.csv")
        df.to_csv(csv_path)
        print(f"数据已保存至: {csv_path}")
        
        # 绘制图表
        save_path = os.path.join(os.path.expanduser("~"), "Desktop", "all_subjects_feature_comparison.pdf")
        plot_survey_small_multiples(df, save_path=save_path)
    else:
        print("未能提取任何学科的特征数据")

if __name__ == "__main__":
    main()
    
    
    
    

