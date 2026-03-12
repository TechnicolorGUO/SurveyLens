"""
Generate publication-quality bar charts comparing systems across different aspects.
Designed for top-tier AI conference aesthetics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Ensure text is editable in PDF
matplotlib.rcParams['ps.fonttype'] = 42

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')


def plot_system_comparison_bar(
    csv_path: Path,
    save_path: Path = None,
    aspect_type: str = "all",  # "all", "overall", or "aspects"
    figsize: tuple = (12, 5),
    dpi: int = 600,
):
    """
    Create publication-quality bar charts comparing systems.
    
    Parameters:
    -----------
    csv_path : Path
        Path to the aggregated_by_system.csv file
    save_path : Path, optional
        Path to save the figure. If None, will be saved in the same directory as CSV.
    aspect_type : str
        "all" - show both overall and aspect-specific plots
        "overall" - show only overall average
        "aspects" - show only aspect-specific comparison
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution for saved figure
    """
    # System name mapping
    system_name_map = {
        "Gemini": "Gemini_DR",
        "Qwen": "Qwen_DR",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "qwen3-max": "Qwen 3 Max",
        "Autosurvey": "AutoSurvey",
    }
    
    # Desired system order
    system_order = [
        "AutoSurvey",
        "SurveyForge",
        "AutoSurvey2",
        "InteractiveSurvey",
        "LLMxMapReduce_V2",
        "SurveyX",
        "SciSage",
        "Qwen 3 Max",
        "Gemini 3 Pro",
        "Qwen_DR",
        "Gemini_DR",
    ]
    
    # Read data
    df = pd.read_csv(csv_path)
    df = df[df['system'].notna()]  # Remove empty rows
    
    # Remove human data
    df = df[~df['system'].str.lower().isin(['human', 'humans'])]
    
    # Apply system name mapping
    df['system'] = df['system'].apply(lambda x: system_name_map.get(x, x))
    
    # Filter to only include systems in the desired order
    df = df[df['system'].isin(system_order)]
    
    # Create a categorical column for custom sorting
    df['system_order'] = df['system'].apply(lambda x: system_order.index(x) if x in system_order else len(system_order))
    df = df.sort_values('system_order')
    
    # Clean system names for better display
    system_names = df['system'].values
    
    # Define metrics
    overall_col = 'original_overall_avg'
    aspect_cols = {
        'Content': 'bt_content_aspect_avg',
        'Outline': 'bt_outline_aspect_avg',
        'Reference': 'bt_reference_aspect_avg'
    }
    
    # Set up the plot style - optimized for two-column format
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.labelsize': 20,
        'axes.titlesize': 21,
        'xtick.labelsize': 18,  # Increased for better readability in two-column layout
        'ytick.labelsize': 18,  # Increased for better readability
        'legend.fontsize': 16,
        'figure.titlesize': 22,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Color palette - professional grayscale with accent colors
    colors = {
        'Content': '#2E86AB',      # Blue
        'Outline': '#A23B72',       # Purple
        'Reference': '#F18F01',     # Orange
        'Overall': '#6C757D'        # Gray
    }
    
    # Create figure(s)
    if aspect_type == "all":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle('System Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    elif aspect_type == "overall":
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    else:  # aspects
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    
    # Plot 1: Overall Average
    if aspect_type in ["all", "overall"]:
        ax = axes[0] if aspect_type == "all" else axes[0]
        
        # Use custom order (already sorted by system_order)
        df_sorted = df.copy()
        
        bars = ax.barh(
            range(len(df_sorted)),
            df_sorted[overall_col].values,
            color=colors['Overall'],
            edgecolor='white',
            linewidth=0.5,
            height=0.7
        )
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['system'].values)
        ax.set_xlabel('', fontweight='medium')
        ax.set_title('Overall Performance', fontweight='medium', pad=10)
        ax.set_xlim(0, max(df[overall_col].values) * 1.1)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            value = row[overall_col]
            ax.text(value + max(df[overall_col].values) * 0.01, i, 
                   f'{value:.2f}', 
                   va='center', fontsize=8, fontweight='medium')
    
    # Plot 2: Aspect-specific comparison
    if aspect_type in ["all", "aspects"]:
        ax = axes[1] if aspect_type == "all" else axes[0]
        
        # Prepare data for grouped bar chart
        x = np.arange(len(system_names))
        width = 0.25
        offset = width * 1.5
        
        # Use custom order (already sorted by system_order)
        df_sorted = df.copy()
        system_names_sorted = df_sorted['system'].values
        
        bars = []
        for i, (aspect_name, col_name) in enumerate(aspect_cols.items()):
            pos = x - offset + i * width
            values = [df_sorted[df_sorted['system'] == sys][col_name].values[0] 
                     if len(df_sorted[df_sorted['system'] == sys][col_name].values) > 0 
                     else 0 for sys in system_names_sorted]
            
            bar = ax.bar(
                pos,
                values,
                width,
                label=aspect_name,
                color=colors[aspect_name],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.9
            )
            bars.append(bar)
        
        ax.set_xlabel('', fontweight='medium')
        ax.set_ylabel('', fontweight='medium')
        ax.set_title('Performance by Aspect', fontweight='medium', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(system_names_sorted, rotation=25, ha='right', rotation_mode='anchor')
        ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
                 edgecolor='gray', framealpha=0.9)
        ax.set_ylim(0, max([df[col].max() for col in aspect_cols.values()]) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    if save_path is None:
        save_path = csv_path.parent / 'system_comparison_bar.pdf'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {save_path}")
    
    # Also save as PNG for preview
    png_path = save_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved preview to {png_path}")
    
    plt.close(fig)


def plot_system_comparison_grouped(
    csv_path: Path,
    save_path: Path = None,
    figsize: tuple = (14, 6),
    dpi: int = 600,
):
    """
    Create a grouped bar chart showing all aspects side by side.
    More compact version suitable for papers.
    """
    # System name mapping
    system_name_map = {
        "Gemini": "Gemini_DR",
        "Qwen": "Qwen_DR",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "qwen3-max": "Qwen 3 Max",
        "Autosurvey": "AutoSurvey",
    }
    
    # Desired system order
    system_order = [
        "AutoSurvey",
        "SurveyForge",
        "AutoSurvey2",
        "InteractiveSurvey",
        "LLMxMapReduce_V2",
        "SurveyX",
        "SciSage",
        "Qwen 3 Max",
        "Gemini 3 Pro",
        "Qwen_DR",
        "Gemini_DR",
    ]
    
    # Read data
    df = pd.read_csv(csv_path)
    df = df[df['system'].notna()]
    
    # Remove human data
    df = df[~df['system'].str.lower().isin(['human', 'humans'])]
    
    # Apply system name mapping
    df['system'] = df['system'].apply(lambda x: system_name_map.get(x, x))
    
    # Filter to only include systems in the desired order
    df = df[df['system'].isin(system_order)]
    
    # Create a categorical column for custom sorting
    df['system_order'] = df['system'].apply(lambda x: system_order.index(x) if x in system_order else len(system_order))
    df_sorted = df.sort_values('system_order')
    system_names = df_sorted['system'].values
    
    # Set up the plot style - optimized for two-column format
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.labelsize': 20,
        'axes.titlesize': 21,
        'xtick.labelsize': 18,  # Increased for better readability in two-column layout
        'ytick.labelsize': 18,  # Increased for better readability
        'legend.fontsize': 16,
        'figure.titlesize': 22,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Color palette
    colors = {
        'Overall': '#1f77b4',      # Blue
        'Content': '#2ca02c',      # Green
        'Outline': '#d62728',      # Red
        'Reference': '#ff7f0e',    # Orange
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    x = np.arange(len(system_names))
    width = 0.2
    
    metrics = {
        'Overall': 'original_overall_avg',
        'Content': 'bt_content_aspect_avg',
        'Outline': 'bt_outline_aspect_avg',
        'Reference': 'bt_reference_aspect_avg'
    }
    
    bars = []
    overall_positions = []
    overall_values = []
    
    # Use a more saturated/darker version of blue for Overall to make it stand out
    overall_color = '#0d5a9c'  # Darker, more saturated blue
    
    for i, (metric_name, col_name) in enumerate(metrics.items()):
        pos = x - width * 1.5 + i * width
        values = df_sorted[col_name].values
        
        # Special styling for Overall metric - use darker color and full opacity
        if metric_name == 'Overall':
            # Store positions and values for adding markers later
            overall_positions = pos
            overall_values = values
            # Use darker blue and full opacity for distinction
            bar = ax.bar(
                pos,
                values,
                width,  # Same width as others
                label=metric_name,
                color=overall_color,  # Darker, more saturated blue
                edgecolor='none',  # No edge
                linewidth=0,
                alpha=1.0,  # Full opacity
                zorder=3  # Ensure Overall bars are on top
            )
        else:
            # Regular styling for other metrics
            bar = ax.bar(
                pos,
                values,
                width,
                label=metric_name,
                color=colors[metric_name],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.85,
                zorder=2  # Regular bars stay below
            )
        bars.append(bar)
    
    # Add subtle dot markers on top of Overall bars for emphasis
    # Draw these AFTER all bars to ensure they're visible
    for j, (pos_val, height_val) in enumerate(zip(overall_positions, overall_values)):
        if height_val > 0:
            ax.plot(pos_val, height_val, marker='o', markersize=6, 
                   color=overall_color, markeredgecolor='white', 
                   markeredgewidth=1.2, zorder=100)  # Very high zorder to stay on top
    
    # Add gray line connecting Overall bar centers
    ax.plot(overall_positions, overall_values, 
           color='gray', linestyle='-', linewidth=2, 
           alpha=0.6, zorder=99, label='_nolegend_')  # _nolegend_ to exclude from legend
    
    ax.set_xlabel('', fontweight='medium')
    ax.set_ylabel('', fontweight='medium')
    ax.set_title('System Performance Comparison Across All Metrics', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(system_names, rotation=25, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), frameon=True, fancybox=False, shadow=False, 
             edgecolor='gray', framealpha=0.9, ncol=4)
    ax.set_ylim(0, max([df[col].max() for col in metrics.values()]) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = csv_path.parent / 'system_comparison_grouped.pdf'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {save_path}")
    
    # Also save as PNG for preview
    png_path = save_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved preview to {png_path}")
    
    plt.close(fig)


def plot_system_comparison_stacked(
    csv_path: Path,
    save_path: Path = None,
    figsize: tuple = (10, 6),
    dpi: int = 600,
):
    """
    Create a stacked bar chart showing Content, Outline, Reference stacked up.
    The total height naturally equals the Overall score.
    """
    # System name mapping
    system_name_map = {
        "Gemini": "Gemini_DR",
        "Qwen": "Qwen_DR",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "qwen3-max": "Qwen 3 Max",
        "Autosurvey": "AutoSurvey",
    }
    
    # Desired system order
    system_order = [
        "AutoSurvey",
        "SurveyForge",
        "AutoSurvey2",
        "InteractiveSurvey",
        "LLMxMapReduce_V2",
        "SurveyX",
        "SciSage",
        "Qwen 3 Max",
        "Gemini 3 Pro",
        "Qwen_DR",
        "Gemini_DR",
    ]
    
    # Read data
    df = pd.read_csv(csv_path)
    df = df[df['system'].notna()]
    
    # Remove human data
    df = df[~df['system'].str.lower().isin(['human', 'humans'])]
    
    # Apply system name mapping
    df['system'] = df['system'].apply(lambda x: system_name_map.get(x, x))
    
    # Filter to only include systems in the desired order
    df = df[df['system'].isin(system_order)]
    
    # Create a categorical column for custom sorting
    df['system_order'] = df['system'].apply(lambda x: system_order.index(x) if x in system_order else len(system_order))
    df_sorted = df.sort_values('system_order')
    system_names = df_sorted['system'].values
    
    # Set up the plot style - optimized for two-column format
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.labelsize': 20,
        'axes.titlesize': 21,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.titlesize': 22,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Color palette - use the same colors as grouped chart
    colors = {
        'Content': '#2ca02c',      # Green
        'Outline': '#d62728',      # Red
        'Reference': '#ff7f0e',    # Orange
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for stacking
    metrics = ['bt_content_aspect_avg', 'bt_outline_aspect_avg', 'bt_reference_aspect_avg']
    labels = ['Content', 'Outline', 'Reference']
    
    # Get values
    content_values = df_sorted['bt_content_aspect_avg'].values
    outline_values = df_sorted['bt_outline_aspect_avg'].values
    reference_values = df_sorted['bt_reference_aspect_avg'].values
    
    y = np.arange(len(system_names))
    
    # Create stacked horizontal bars
    ax.barh(y, content_values, height=0.7, label='Content', 
            color=colors['Content'], edgecolor='white', linewidth=0.5)
    ax.barh(y, outline_values, height=0.7, left=content_values, label='Outline',
            color=colors['Outline'], edgecolor='white', linewidth=0.5)
    ax.barh(y, reference_values, height=0.7, 
            left=content_values + outline_values, label='Reference',
            color=colors['Reference'], edgecolor='white', linewidth=0.5)
    
    # Add total (overall) score at the end of each bar
    overall_values = df_sorted['original_overall_avg'].values
    for i, (overall, total) in enumerate(zip(overall_values, 
                                              content_values + outline_values + reference_values)):
        ax.text(total + 0.1, i, f'{overall:.2f}', 
               va='center', fontsize=9, fontweight='medium')
    
    ax.set_yticks(y)
    ax.set_yticklabels(system_names)
    ax.set_xlabel('', fontweight='medium')
    ax.set_title('System Performance: Stacked Aspect Scores', 
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False,
             edgecolor='gray', framealpha=0.9)
    ax.set_xlim(0, max(overall_values) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = csv_path.parent / 'system_comparison_stacked.pdf'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {save_path}")
    
    # Also save as PNG for preview
    png_path = save_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved preview to {png_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    csv_path = current_dir / "aggregated_by_system.csv"
    figures_dir = current_dir / "figures"
    
    # Generate all three versions
    print("Generating publication-quality bar charts...")
    
    # 1. Overall comparison only
    plot_system_comparison_bar(
        csv_path=csv_path,
        save_path=figures_dir / "system_comparison_overall.pdf",
        aspect_type="overall"
    )
    
    # 2. Aspect-specific comparison only
    plot_system_comparison_bar(
        csv_path=csv_path,
        save_path=figures_dir / "system_comparison_aspects.pdf",
        aspect_type="aspects"
    )
    
    # 3. Combined (overall + aspects)
    plot_system_comparison_bar(
        csv_path=csv_path,
        save_path=figures_dir / "system_comparison_all.pdf",
        aspect_type="all"
    )
    
    # 4. Grouped bar chart (all metrics together)
    plot_system_comparison_grouped(
        csv_path=csv_path,
        save_path=figures_dir / "system_comparison_grouped.pdf"
    )
    
    # 5. Stacked bar chart (aspects stacked to show overall)
    plot_system_comparison_stacked(
        csv_path=csv_path,
        save_path=figures_dir / "system_comparison_stacked.pdf"
    )
    
    print("\nAll plots generated successfully!")
