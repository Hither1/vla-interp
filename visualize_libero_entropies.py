#!/usr/bin/env python3
"""
Visualize entropy distributions for success vs failure cases.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_visualizations(csv_file="libero_entropy_analysis.csv", output_dir="entropy_plots"):
    """Create comprehensive visualizations of entropy analysis."""

    # Load data
    df = pd.read_csv(csv_file)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # 1. Overall distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    ax = axes[0, 0]
    success_df = df[df['success'] == True]
    failure_df = df[df['success'] == False]

    ax.hist(success_df['action_entropy_kde'].dropna(), bins=30, alpha=0.6, label='Success', color='green', density=True)
    ax.hist(failure_df['action_entropy_kde'].dropna(), bins=30, alpha=0.6, label='Failure', color='red', density=True)
    ax.set_xlabel('Action Entropy', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Action Entropy Distribution: Success vs Failure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[0, 1]
    data_to_plot = [
        success_df['action_entropy_kde'].dropna(),
        failure_df['action_entropy_kde'].dropna()
    ]
    bp = ax.boxplot(data_to_plot, labels=['Success', 'Failure'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Entropy: Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Violin plot
    ax = axes[1, 0]
    plot_df = df[df['action_entropy_kde'].notna()].copy()
    plot_df['outcome'] = plot_df['success'].map({True: 'Success', False: 'Failure'})
    sns.violinplot(data=plot_df, x='outcome', y='action_entropy_kde', ax=ax, palette=['lightgreen', 'lightcoral'])
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Entropy: Violin Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # CDF comparison
    ax = axes[1, 1]
    success_entropy = success_df['action_entropy_kde'].dropna().sort_values()
    failure_entropy = failure_df['action_entropy_kde'].dropna().sort_values()

    ax.plot(success_entropy, np.linspace(0, 1, len(success_entropy)), label='Success', color='green', linewidth=2)
    ax.plot(failure_entropy, np.linspace(0, 1, len(failure_entropy)), label='Failure', color='red', linewidth=2)
    ax.set_xlabel('Action Entropy', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_entropy_comparison.png", bbox_inches='tight')
    print(f"✓ Saved {output_dir}/overall_entropy_comparison.png")
    plt.close()

    # 2. Per-suite comparison
    suites = df['suite'].unique()
    n_suites = len(suites)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, suite in enumerate(sorted(suites)):
        if idx >= len(axes):
            break

        ax = axes[idx]
        suite_df = df[df['suite'] == suite]
        success_suite = suite_df[suite_df['success'] == True]['action_entropy_kde'].dropna()
        failure_suite = suite_df[suite_df['success'] == False]['action_entropy_kde'].dropna()

        data_to_plot = []
        labels = []
        if len(success_suite) > 0:
            data_to_plot.append(success_suite)
            labels.append(f'Success (n={len(success_suite)})')
        if len(failure_suite) > 0:
            data_to_plot.append(failure_suite)
            labels.append(f'Failure (n={len(failure_suite)})')

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor('lightgreen' if 'Success' in labels[i] else 'lightcoral')

        ax.set_ylabel('Action Entropy', fontsize=10)
        ax.set_title(f'Suite: {suite}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_suites, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_suite_entropy.png", bbox_inches='tight')
    print(f"✓ Saved {output_dir}/per_suite_entropy.png")
    plt.close()

    # 3. Entropy vs other metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Entropy vs trajectory length
    ax = axes[0, 0]
    success_mask = df['success'] == True
    ax.scatter(df[success_mask]['num_actions'], df[success_mask]['action_entropy_kde'],
               alpha=0.5, c='green', label='Success', s=20)
    ax.scatter(df[~success_mask]['num_actions'], df[~success_mask]['action_entropy_kde'],
               alpha=0.5, c='red', label='Failure', s=20)
    ax.set_xlabel('Number of Actions', fontsize=12)
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Entropy vs Trajectory Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Entropy vs smoothness
    ax = axes[0, 1]
    smoothness_df = df[df['smoothness'].notna()]
    success_mask = smoothness_df['success'] == True
    ax.scatter(smoothness_df[success_mask]['smoothness'], smoothness_df[success_mask]['action_entropy_kde'],
               alpha=0.5, c='green', label='Success', s=20)
    ax.scatter(smoothness_df[~success_mask]['smoothness'], smoothness_df[~success_mask]['action_entropy_kde'],
               alpha=0.5, c='red', label='Failure', s=20)
    ax.set_xlabel('Smoothness', fontsize=12)
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Entropy vs Smoothness', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Entropy vs mean action delta norm
    ax = axes[1, 0]
    delta_df = df[df['mean_action_delta_norm'].notna()]
    success_mask = delta_df['success'] == True
    ax.scatter(delta_df[success_mask]['mean_action_delta_norm'], delta_df[success_mask]['action_entropy_kde'],
               alpha=0.5, c='green', label='Success', s=20)
    ax.scatter(delta_df[~success_mask]['mean_action_delta_norm'], delta_df[~success_mask]['action_entropy_kde'],
               alpha=0.5, c='red', label='Failure', s=20)
    ax.set_xlabel('Mean Action Delta Norm', fontsize=12)
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Entropy vs Mean Action Delta', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Success rate by entropy bins
    ax = axes[1, 1]
    entropy_bins = pd.cut(df['action_entropy_kde'], bins=10)
    success_by_bin = df.groupby(entropy_bins)['success'].agg(['mean', 'count'])
    bin_centers = [interval.mid for interval in success_by_bin.index]

    ax.bar(range(len(bin_centers)), success_by_bin['mean'], alpha=0.7, color='steelblue')
    ax.set_xlabel('Action Entropy Bins', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate by Action Entropy Bins', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels([f'{x:.2f}' for x in bin_centers], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_correlations.png", bbox_inches='tight')
    print(f"✓ Saved {output_dir}/entropy_correlations.png")
    plt.close()

    # 4. Summary statistics table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    summary_data = []
    for suite in sorted(df['suite'].unique()):
        suite_df = df[df['suite'] == suite]
        success_df = suite_df[suite_df['success'] == True]
        failure_df = suite_df[suite_df['success'] == False]

        row = [
            suite,
            len(success_df),
            f"{success_df['action_entropy_kde'].mean():.4f} ± {success_df['action_entropy_kde'].std():.4f}" if len(success_df) > 0 else "N/A",
            len(failure_df),
            f"{failure_df['action_entropy_kde'].mean():.4f} ± {failure_df['action_entropy_kde'].std():.4f}" if len(failure_df) > 0 else "N/A",
        ]
        summary_data.append(row)

    # Add overall row
    success_all = df[df['success'] == True]
    failure_all = df[df['success'] == False]
    summary_data.append([
        'OVERALL',
        len(success_all),
        f"{success_all['action_entropy_kde'].mean():.4f} ± {success_all['action_entropy_kde'].std():.4f}",
        len(failure_all),
        f"{failure_all['action_entropy_kde'].mean():.4f} ± {failure_all['action_entropy_kde'].std():.4f}"
    ])

    table = ax.table(cellText=summary_data,
                     colLabels=['Suite', 'Success\nCount', 'Success\nEntropy (mean±std)', 'Failure\nCount', 'Failure\nEntropy (mean±std)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.25, 0.12, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style overall row
    for i in range(5):
        table[(len(summary_data), i)].set_facecolor('#E8F5E9')
        table[(len(summary_data), i)].set_text_props(weight='bold')

    plt.title('Action Entropy Summary by Suite', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/entropy_summary_table.png", bbox_inches='tight', dpi=150)
    print(f"✓ Saved {output_dir}/entropy_summary_table.png")
    plt.close()

    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_visualizations()
