#!/usr/bin/env python3
"""
Analyze correlation between entropy and task-level success rates.
For both success and failure cases, does entropy correlate with the overall success rate of that task?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def analyze_entropy_task_correlation(csv_file="libero_entropy_analysis.csv", output_dir="entropy_plots"):
    """Analyze correlation between entropy and task success rates."""

    # Load data
    df = pd.read_csv(csv_file)

    # Calculate task-level success rates
    task_stats = df.groupby(['suite', 'task_id', 'task_description']).agg({
        'success': ['sum', 'count', 'mean'],
        'action_entropy_kde': 'mean'
    }).reset_index()

    # Flatten column names
    task_stats.columns = ['suite', 'task_id', 'task_description', 'num_success', 'num_total', 'success_rate', 'mean_entropy']

    # Merge task success rates back to individual trajectories
    df = df.merge(
        task_stats[['suite', 'task_id', 'success_rate']],
        on=['suite', 'task_id'],
        how='left',
        suffixes=('', '_task')
    )

    # Separate success and failure cases
    success_df = df[df['success'] == True].copy()
    failure_df = df[df['success'] == False].copy()

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. SUCCESS CASES: Entropy vs Task Success Rate
    ax = axes[0, 0]
    if len(success_df) > 0:
        ax.scatter(success_df['success_rate'], success_df['action_entropy_kde'],
                   alpha=0.4, c='green', s=30)

        # Add trend line
        if len(success_df) > 1:
            z = np.polyfit(success_df['success_rate'].dropna(),
                          success_df['action_entropy_kde'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(success_df['success_rate'].min(),
                               success_df['success_rate'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        # Calculate correlation
        corr, p_val = stats.pearsonr(success_df['success_rate'].dropna(),
                                     success_df['action_entropy_kde'].dropna())

        ax.set_xlabel('Task Success Rate', fontsize=12)
        ax.set_ylabel('Action Entropy KDE', fontsize=12)
        ax.set_title(f'Success Cases: Entropy vs Task Success Rate\nr={corr:.3f}, p={p_val:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 2. FAILURE CASES: Entropy vs Task Success Rate
    ax = axes[0, 1]
    if len(failure_df) > 0:
        ax.scatter(failure_df['success_rate'], failure_df['action_entropy_kde'],
                   alpha=0.6, c='red', s=50, marker='x')

        # Add trend line if enough points
        if len(failure_df) > 2:
            z = np.polyfit(failure_df['success_rate'].dropna(),
                          failure_df['action_entropy_kde'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(failure_df['success_rate'].min(),
                               failure_df['success_rate'].max(), 100)
            ax.plot(x_line, p(x_line), "darkred", linestyle='--', alpha=0.8,
                   linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        # Calculate correlation
        if len(failure_df) > 2:
            corr, p_val = stats.pearsonr(failure_df['success_rate'].dropna(),
                                        failure_df['action_entropy_kde'].dropna())
            title = f'Failure Cases: Entropy vs Task Success Rate\nr={corr:.3f}, p={p_val:.4f}'
        else:
            title = f'Failure Cases: Entropy vs Task Success Rate\n(n={len(failure_df)}, too few for correlation)'

        ax.set_xlabel('Task Success Rate', fontsize=12)
        ax.set_ylabel('Action Entropy KDE', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if len(failure_df) > 2:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 3. Box plot: Entropy by task success rate bins (SUCCESS)
    ax = axes[0, 2]
    if len(success_df) > 0:
        success_df['success_rate_bin'] = pd.cut(success_df['success_rate'],
                                                bins=[0, 0.7, 0.85, 0.95, 1.0],
                                                labels=['<70%', '70-85%', '85-95%', '95-100%'])

        bins_with_data = success_df.groupby('success_rate_bin')['action_entropy_kde'].count()
        bins_to_plot = bins_with_data[bins_with_data > 0].index

        if len(bins_to_plot) > 0:
            data_to_plot = [success_df[success_df['success_rate_bin'] == b]['action_entropy_kde'].dropna()
                           for b in bins_to_plot]
            bp = ax.boxplot(data_to_plot, labels=[str(b) for b in bins_to_plot], patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor('lightgreen')

            ax.set_xlabel('Task Success Rate Bin', fontsize=12)
            ax.set_ylabel('Action Entropy KDE', fontsize=12)
            ax.set_title('Success Cases: Entropy by Task Success Rate', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

    # 4. Per-suite analysis (SUCCESS)
    ax = axes[1, 0]
    suite_colors = {'10': 'tab:blue', 'goal': 'tab:orange', 'object': 'tab:green', 'spatial': 'tab:red'}

    if len(success_df) > 0:
        for suite in success_df['suite'].unique():
            suite_data = success_df[success_df['suite'] == suite]
            ax.scatter(suite_data['success_rate'], suite_data['action_entropy_kde'],
                      alpha=0.5, label=suite, s=30, c=suite_colors.get(suite, 'gray'))

            # Correlation per suite
            if len(suite_data) > 2:
                corr, _ = stats.pearsonr(suite_data['success_rate'].dropna(),
                                        suite_data['action_entropy_kde'].dropna())
                print(f"Success - Suite {suite}: r={corr:.3f}")

        ax.set_xlabel('Task Success Rate', fontsize=12)
        ax.set_ylabel('Action Entropy KDE', fontsize=12)
        ax.set_title('Success Cases by Suite', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 5. Per-suite analysis (FAILURE)
    ax = axes[1, 1]

    if len(failure_df) > 0:
        for suite in failure_df['suite'].unique():
            suite_data = failure_df[failure_df['suite'] == suite]
            ax.scatter(suite_data['success_rate'], suite_data['action_entropy_kde'],
                      alpha=0.7, label=f"{suite} (n={len(suite_data)})",
                      s=80, marker='x', c=suite_colors.get(suite, 'gray'))

            # Correlation per suite (if enough points)
            if len(suite_data) > 2:
                corr, _ = stats.pearsonr(suite_data['success_rate'].dropna(),
                                        suite_data['action_entropy_kde'].dropna())
                print(f"Failure - Suite {suite}: r={corr:.3f}")

        ax.set_xlabel('Task Success Rate', fontsize=12)
        ax.set_ylabel('Action Entropy KDE', fontsize=12)
        ax.set_title('Failure Cases by Suite', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 6. Average entropy per task vs success rate
    ax = axes[1, 2]

    # Calculate average entropy per task (combining success and failure)
    task_entropy = df.groupby(['suite', 'task_id']).agg({
        'action_entropy_kde': 'mean',
        'success_rate': 'first',
        'task_description': 'first'
    }).reset_index()

    ax.scatter(task_entropy['success_rate'], task_entropy['action_entropy_kde'],
              alpha=0.6, c='purple', s=60)

    # Add trend line
    if len(task_entropy) > 1:
        z = np.polyfit(task_entropy['success_rate'].dropna(),
                      task_entropy['action_entropy_kde'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(task_entropy['success_rate'].min(),
                           task_entropy['success_rate'].max(), 100)
        ax.plot(x_line, p(x_line), "darkviolet", linestyle='--', alpha=0.8,
               linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

    # Calculate correlation
    corr, p_val = stats.pearsonr(task_entropy['success_rate'].dropna(),
                                task_entropy['action_entropy_kde'].dropna())

    ax.set_xlabel('Task Success Rate', fontsize=12)
    ax.set_ylabel('Mean Action Entropy KDE', fontsize=12)
    ax.set_title(f'Task-Level: Mean Entropy vs Success Rate\nr={corr:.3f}, p={p_val:.4f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_task_correlation.png", bbox_inches='tight', dpi=150)
    print(f"\n✓ Saved {output_dir}/entropy_task_correlation.png")
    plt.close()

    # Print detailed statistics
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: Entropy vs Task Success Rate")
    print("="*70)

    print("\n1. SUCCESS CASES:")
    print("-" * 70)
    if len(success_df) > 1:
        corr, p_val = stats.pearsonr(success_df['success_rate'].dropna(),
                                     success_df['action_entropy_kde'].dropna())
        print(f"   Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
        spearman_corr, spearman_p = stats.spearmanr(success_df['success_rate'].dropna(),
                                                     success_df['action_entropy_kde'].dropna())
        print(f"   Spearman correlation: ρ = {spearman_corr:.4f}, p = {spearman_p:.6f}")
        print(f"   Interpretation: {'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

        if corr > 0:
            print("   → Higher task success rate → Higher entropy (less negative)")
            print("   → Easier tasks have higher entropy in successful trajectories")
        else:
            print("   → Higher task success rate → Lower entropy (more negative)")
            print("   → Easier tasks have lower entropy in successful trajectories")

    print("\n2. FAILURE CASES:")
    print("-" * 70)
    if len(failure_df) > 2:
        corr, p_val = stats.pearsonr(failure_df['success_rate'].dropna(),
                                     failure_df['action_entropy_kde'].dropna())
        print(f"   Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
        spearman_corr, spearman_p = stats.spearmanr(failure_df['success_rate'].dropna(),
                                                     failure_df['action_entropy_kde'].dropna())
        print(f"   Spearman correlation: ρ = {spearman_corr:.4f}, p = {spearman_p:.6f}")
        print(f"   Interpretation: {'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

        if corr > 0:
            print("   → Higher task success rate → Higher entropy (less negative)")
            print("   → Failures on easier tasks have higher entropy")
        else:
            print("   → Higher task success rate → Lower entropy (more negative)")
            print("   → Failures on easier tasks have lower entropy")
    else:
        print(f"   Too few failure cases (n={len(failure_df)}) for reliable correlation")

    print("\n3. TASK-LEVEL (all trajectories aggregated by task):")
    print("-" * 70)
    corr, p_val = stats.pearsonr(task_entropy['success_rate'].dropna(),
                                task_entropy['action_entropy_kde'].dropna())
    print(f"   Pearson correlation: r = {corr:.4f}, p = {p_val:.6f}")
    spearman_corr, spearman_p = stats.spearmanr(task_entropy['success_rate'].dropna(),
                                                task_entropy['action_entropy_kde'].dropna())
    print(f"   Spearman correlation: ρ = {spearman_corr:.4f}, p = {spearman_p:.6f}")
    print(f"   Number of tasks: {len(task_entropy)}")

    # Show top and bottom tasks by success rate
    print("\n4. EXTREME CASES:")
    print("-" * 70)
    print("\nTasks with HIGHEST success rates:")
    top_tasks = task_entropy.nlargest(5, 'success_rate')[['task_description', 'success_rate', 'action_entropy_kde']]
    for idx, row in top_tasks.iterrows():
        print(f"   {row['task_description'][:60]:60s} | SR: {row['success_rate']:.2f} | Entropy: {row['action_entropy_kde']:.3f}")

    print("\nTasks with LOWEST success rates:")
    bottom_tasks = task_entropy.nsmallest(5, 'success_rate')[['task_description', 'success_rate', 'action_entropy_kde']]
    for idx, row in bottom_tasks.iterrows():
        print(f"   {row['task_description'][:60]:60s} | SR: {row['success_rate']:.2f} | Entropy: {row['action_entropy_kde']:.3f}")

    print("\n" + "="*70)

    # Save detailed results to CSV
    task_entropy.to_csv('task_level_entropy_analysis.csv', index=False)
    print(f"\n✓ Saved task-level analysis to task_level_entropy_analysis.csv")

    # Create summary table
    summary_data = {
        'Analysis': ['Success Cases', 'Failure Cases', 'Task-Level Average'],
        'N': [len(success_df), len(failure_df), len(task_entropy)],
        'Pearson r': [
            stats.pearsonr(success_df['success_rate'].dropna(), success_df['action_entropy_kde'].dropna())[0] if len(success_df) > 1 else np.nan,
            stats.pearsonr(failure_df['success_rate'].dropna(), failure_df['action_entropy_kde'].dropna())[0] if len(failure_df) > 2 else np.nan,
            corr
        ],
        'p-value': [
            stats.pearsonr(success_df['success_rate'].dropna(), success_df['action_entropy_kde'].dropna())[1] if len(success_df) > 1 else np.nan,
            stats.pearsonr(failure_df['success_rate'].dropna(), failure_df['action_entropy_kde'].dropna())[1] if len(failure_df) > 2 else np.nan,
            p_val
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('entropy_task_correlation_summary.csv', index=False)
    print(f"✓ Saved correlation summary to entropy_task_correlation_summary.csv")

    return task_entropy


if __name__ == "__main__":
    analyze_entropy_task_correlation()
