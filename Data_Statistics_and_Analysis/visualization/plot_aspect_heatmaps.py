"""
Optimized heatmaps for Content, Outline, and Reference aspects.
Key improvements:
1. Better color scheme (sequential instead of diverging)
2. More compact layout with shared colorbar
3. Better label formatting
4. Clean professional academic style
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_aspect_heatmaps_optimized(    
    csv_path: Path,
    save_path: Path = None,
    figsize: tuple = (16, 6),
    dpi: int = 600,
    exclude_systems: list = None,
    style: str = 'modern',  # 'modern', 'academic', 'vibrant'
):
    """
    Optimized three-panel heatmap visualization.
    
    Parameters:
    -----------
    style : str
        'modern' - Clean blue gradient
        'academic' - Subtle warm tones
        'vibrant' - Bold colorful scheme
    """
    # Read data
    df = pd.read_csv(csv_path)
    df = df[df['system'].notna() & df['category'].notna()]
    
    # 设置要排除的系统，默认为['human']
    if exclude_systems is None:
        exclude_systems = ['human']
    
    # 删除指定系统（包括大小写变体）
    for system_to_exclude in exclude_systems:
        mask = df['system'].astype(str).str.lower() == system_to_exclude.lower()
        excluded_count = mask.sum()
        if excluded_count > 0:
            print(f"Excluding '{system_to_exclude}': removed {excluded_count} rows")
            df = df[~mask]
    
    if len(df) == 0:
        raise ValueError("No data after filtering!")
    
    # 修改系统名称
    df.loc[df['system'] == 'Qwen', 'system'] = 'Qwen_DR'
    df.loc[df['system'] == 'Gemini', 'system'] = 'Gemini_DR'
    
    # Define aspects
    aspects = {
        'Content': 'bt_content_aspect_avg',
        'Outline': 'bt_outline_aspect_avg',
        'Reference': 'bt_reference_aspect_avg'
    }
    
    # 系统名称缩写映射（让x轴更整洁）
    system_abbr = {
        'AutoSurvey': 'AS',
        'SurveyForge': 'SF',
        'AutoSurvey2': 'AS2',
        'InteractiveSurvey': 'IS',
        'LLMxMapReduce_V2': 'LxMR',
        'SurveyX': 'SX',
        'SciSage': 'SS',
        'Qwen_DR': 'Qw_DR',
        'Gemini_DR': 'Ge_DR',
        'gemini-3-pro-preview': 'Ge3',
        'qwen3-max': 'Qw3'
    }
    
    # 学科领域缩写映射
    category_abbr = {
        'Biology': 'Bio',
        'Business': 'Bus',
        'Computer Science': 'CS',
        'Education': 'Edu',
        'Engineering': 'Eng',
        'Environmental Science': 'Env',
        'Medicine': 'Med',
        'Physics': 'Phy',
        'Psychology': 'Psy',
        'Sociology': 'Soc'
    }
    
    # Get unique values
    systems = sorted(df['system'].unique())
    categories = sorted(df['category'].unique())
    
    print(f"Systems after filtering: {systems}")
    print(f"Categories: {categories}")
    
    # 设置配色方案
    if style == 'modern':
        # 现代蓝色渐变 - 适合学术论文
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
        cmap = LinearSegmentedColormap.from_list('modern_blue', colors)
    elif style == 'academic':
        # 学术风格 - 温暖色调
        colors = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04']
        cmap = LinearSegmentedColormap.from_list('academic_warm', colors)
    elif style == 'vibrant':
        # 鲜艳风格 - viridis变体
        cmap = plt.cm.viridis
    else:
        cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    # Set up plot style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    # 创建图形 - 使用 gridspec 精确控制布局
    fig = plt.figure(figsize=figsize)
    
    # 使用 gridspec 创建更紧凑的布局
    # 3个热力图 + 1个颜色条
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    cbar_ax = fig.add_subplot(gs[3])
    
    axes = [ax1, ax2, ax3]
    
    # Global color range
    vmin = df[[col for col in aspects.values()]].min().min()
    vmax = df[[col for col in aspects.values()]].max().max()
    
    print(f"Score range: {vmin:.2f} - {vmax:.2f}")
    
    # 系统顺序
    desired_order = [
        'AutoSurvey', 
        'SurveyForge', 
        'AutoSurvey2', 
        'InteractiveSurvey', 
        'LLMxMapReduce_V2', 
        'SurveyX', 
        'SciSage', 
        'Qwen_DR', 
        'Gemini_DR', 
        'gemini-3-pro-preview', 
        'qwen3-max'
    ]
    
    heatmaps = []
    
    for idx, (aspect_name, col_name) in enumerate(aspects.items()):
        ax = axes[idx]
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values=col_name,
            index='category',
            columns='system',
            aggfunc='mean'
        )
        
        # 确保所有列都存在
        for system in desired_order:
            if system not in pivot_data.columns:
                pivot_data[system] = np.nan
        
        pivot_data = pivot_data[desired_order]
        pivot_data = pivot_data.reindex(index=sorted(categories), fill_value=np.nan)
        
        # 使用缩写作为列名
        pivot_data.columns = [system_abbr.get(s, s) for s in pivot_data.columns]
        pivot_data.index = [category_abbr.get(c, c) for c in pivot_data.index]
        
        # Create heatmap
        im = sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar=False,
            linewidths=0.5,
            linecolor='white',
            square=True
        )
        
        heatmaps.append(im)
        
        # Title styling
        ax.set_title(aspect_name, fontweight='bold', fontsize=13, pad=8)
        ax.set_xlabel('')
        
        if idx == 0:
            ax.set_ylabel('Field', fontweight='medium', fontsize=12)
            ax.yaxis.set_tick_params(labelsize=10)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        
        # X轴标签 - 垂直显示更节省空间
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
    
    # 添加共享颜色条
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Score', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = csv_path.parent / 'aspect_heatmaps_optimized.pdf'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print(f"Saved: {save_path}")
    
    png_path = save_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved preview: {png_path}")
    
    plt.close(fig)
    return save_path, png_path


def plot_aspect_heatmaps_vertical(    
    csv_path: Path,
    save_path: Path = None,
    figsize: tuple = (8, 14),
    dpi: int = 600,
    exclude_systems: list = None,
):
    """
    Alternative: 垂直排列的三个热力图（如果系统太多水平放不下）
    """
    # Read data
    df = pd.read_csv(csv_path)
    df = df[df['system'].notna() & df['category'].notna()]
    
    if exclude_systems is None:
        exclude_systems = ['human']
    
    for system_to_exclude in exclude_systems:
        mask = df['system'].astype(str).str.lower() == system_to_exclude.lower()
        df = df[~mask]
    
    df.loc[df['system'] == 'Qwen', 'system'] = 'Qwen_DR'
    df.loc[df['system'] == 'Gemini', 'system'] = 'Gemini_DR'
    
    aspects = {
        'Content': 'bt_content_aspect_avg',
        'Outline': 'bt_outline_aspect_avg',
        'Reference': 'bt_reference_aspect_avg'
    }
    
    categories = sorted(df['category'].unique())
    
    # 颜色方案
    colors = ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    cmap = LinearSegmentedColormap.from_list('blues', colors)
    
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
    })
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.03], hspace=0.25, wspace=0.05)
    
    vmin = df[[col for col in aspects.values()]].min().min()
    vmax = df[[col for col in aspects.values()]].max().max()
    
    desired_order = [
        'AutoSurvey', 'SurveyForge', 'AutoSurvey2', 'InteractiveSurvey', 
        'LLMxMapReduce_V2', 'SurveyX', 'SciSage', 
        'Qwen_DR', 'Gemini_DR', 'gemini-3-pro-preview', 'qwen3-max'
    ]
    
    for idx, (aspect_name, col_name) in enumerate(aspects.items()):
        ax = fig.add_subplot(gs[idx, 0])
        
        pivot_data = df.pivot_table(
            values=col_name,
            index='category',
            columns='system',
            aggfunc='mean'
        )
        
        for system in desired_order:
            if system not in pivot_data.columns:
                pivot_data[system] = np.nan
        
        pivot_data = pivot_data[desired_order]
        pivot_data = pivot_data.reindex(index=categories, fill_value=np.nan)
        
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=False,
            cbar=False,
            linewidths=0.5,
            linecolor='white',
            square=True
        )
        
        ax.set_title(f'{aspect_name}', fontweight='bold', fontsize=12, pad=8)
        ax.set_ylabel('Field of Study', fontsize=10)
        ax.set_xlabel('')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # 颜色条
    cbar_ax = fig.add_subplot(gs[:, 1])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Score', fontsize=10)
    
    if save_path is None:
        save_path = csv_path.parent / 'aspect_heatmaps_vertical.pdf'
    
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    fig.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    # 使用当前目录
    root = Path(__file__).resolve().parent
    csv_path = root / "aggregated_by_system_category.csv"
    
    print("=" * 60)
    print("Generating optimized heatmaps...")
    print("=" * 60)
    
    # 方案1: 现代蓝色风格（推荐）
    print("\n🎨 Style: Modern Blue")
    plot_aspect_heatmaps_optimized(
        csv_path=csv_path,
        save_path=root / "aspect_heatmaps_modern.pdf",
        style='modern',
        exclude_systems=['human']
    )
    
    # 方案2: 学术暖色风格
    print("\n🎨 Style: Academic Warm")
    plot_aspect_heatmaps_optimized(
        csv_path=csv_path,
        save_path=root / "aspect_heatmaps_academic.pdf",
        style='academic',
        exclude_systems=['human']
    )
    
    # 方案3: Viridis 鲜艳风格
    print("\n🎨 Style: Vibrant (Viridis)")
    plot_aspect_heatmaps_optimized(
        csv_path=csv_path,
        save_path=root / "aspect_heatmaps_vibrant.pdf",
        style='vibrant',
        exclude_systems=['human']
    )
    
    print("\n" + "=" * 60)
    print("✅ All heatmaps generated successfully!")
    print("=" * 60)
