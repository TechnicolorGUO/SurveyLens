from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SYSTEM_ORDER = [
    "SurveyX",
    "SurveyForge",
    "InteractiveSurvey",
    "Autosurvey",
    "LLMxMapReduce_V2",
    "Qwen",
    "SciSage",
    "Gemini",
    "Autosurvey2",
]


def _load_elo_scores(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.set_index("model")
    return df["elo"].reindex(SYSTEM_ORDER)


def plot_elo_scores_premium(
    outline_csv: Path,
    content_csv: Path,
    reference_csv: Path,
    save_path: Path,
) -> None:
    """
    Generate publication-ready Elo Score bar plots with advanced visual styling.
    """
    elo_scores = {
        "Outline": _load_elo_scores(outline_csv),
        "Content": _load_elo_scores(content_csv),
        "Reference": _load_elo_scores(reference_csv),
    }

    # Set up publication-quality matplotlib parameters with even larger fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # Adjust layout to leave space for horizontal colorbar at bottom right
    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.05, top=0.9, bottom=0.25)

    # Get global min/max for consistent colormap scaling
    all_scores = pd.concat(elo_scores.values())
    global_min = all_scores.min()
    global_max = all_scores.max()

    # Create colormap normalizer
    norm = plt.Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.cm.Spectral_r  # High scores = more prominent colors

    for idx, (title, series) in enumerate(elo_scores.items()):
        ax = axes[idx]

        # Get colors for each bar based on score values
        colors = [cmap(norm(score)) for score in series.values]

        # Create bars with premium styling
        bars = ax.bar(
            range(len(SYSTEM_ORDER)),
            series.values,
            color=colors,
            width=0.5,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )

        # Add value annotations on top of each bar
        for bar, score in zip(bars, series.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,  # Position above bar
                f'{score:.0f}',  # Format as integer
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
                color='black',
                zorder=4
            )

        # Professional axis styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(-0.5, len(SYSTEM_ORDER) - 0.5)

        # Set Y-axis limits to focus on data range (not from 0)
        y_margin = (global_max - global_min) * 0.1
        ax.set_ylim(global_min - y_margin, global_max + y_margin + 30)  # Extra space for annotations

        # Y-axis label only on leftmost plot
        if idx == 0:
            ax.set_ylabel("Elo Score", fontsize=15, fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel("")

        # X-axis labels with rotation for readability
        if idx == 0:  # Only show labels on first subplot
            ax.set_xticks(range(len(SYSTEM_ORDER)))
            ax.set_xticklabels(
                SYSTEM_ORDER,
                rotation=45,
                ha='right',
                rotation_mode='anchor',
                fontsize=13
            )
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Premium spine styling (despine effect)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['bottom'].set_linewidth(0.8)

        # Subtle Y-axis grid only
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)
        ax.grid(axis='x', visible=False)

        # Clean tick styling
        ax.tick_params(axis='y', labelsize=13, colors='#333333')
        ax.tick_params(axis='x', labelsize=13, colors='#333333')

    # Add horizontal colorbar at bottom right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Position horizontal colorbar at bottom right (moved further right)
    cbar_ax = fig.add_axes([0.75, 0.08, 0.20, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elo Score', fontsize=14, fontweight='bold', labelpad=5)
    cbar.ax.tick_params(labelsize=12)
    # Position label below the colorbar
    cbar.ax.xaxis.set_label_position('bottom')

    fig.tight_layout()
    # Ensure tight bounding box for publication
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Publication-ready Elo Score plot saved to {save_path}")

    # Print statistics for verification
    print("\n📊 Data Statistics:")
    for task, scores in elo_scores.items():
        print(f"  {task}: Range {scores.min():.1f} - {scores.max():.1f}, Mean {scores.mean():.1f}")


if __name__ == "__main__":
    # 使用当前目录中的elo文件
    current_dir = Path(__file__).resolve().parent
    plot_elo_scores_premium(
        outline_csv=current_dir / "elo_scores_outline.csv",
        content_csv=current_dir / "elo_scores_content.csv",
        reference_csv=current_dir / "elo_scores_reference.csv",
        save_path=current_dir / "elo_scores_premium.pdf",
    )
