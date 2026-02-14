#!/usr/bin/env python3
"""
Visualize entropy results from calculate_entropy_by_suite.py

Usage:
    python visualize_entropy_results.py --results entropy_results.json [--output_dir plots/]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_entropy_comparison(results: dict, output_dir: Path):
    """
    Create bar plots comparing success vs failure entropy for each suite.
    """
    suites = list(results.keys())
    success_means = []
    success_stds = []
    failure_means = []
    failure_stds = []

    for suite in suites:
        success_means.append(results[suite]['success']['entropy_mean'] or 0)
        success_stds.append(results[suite]['success']['entropy_std'] or 0)
        failure_means.append(results[suite]['failure']['entropy_mean'] or 0)
        failure_stds.append(results[suite]['failure']['entropy_std'] or 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(suites))
    width = 0.35

    bars1 = ax.bar(x - width/2, success_means, width, yerr=success_stds,
                   label='Success', alpha=0.8, capsize=5, color='green')
    bars2 = ax.bar(x + width/2, failure_means, width, yerr=failure_stds,
                   label='Failure', alpha=0.8, capsize=5, color='red')

    ax.set_xlabel('Suite', fontsize=12)
    ax.set_ylabel('Action Entropy (KDE)', fontsize=12)
    ax.set_title('Action Entropy by Suite and Success/Failure Status', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_by_suite_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_by_suite_comparison.png'}")
    plt.close()


def plot_entropy_difference(results: dict, output_dir: Path):
    """
    Plot the difference in entropy between failure and success for each suite.
    """
    suites = []
    differences = []
    num_success = []
    num_failure = []

    for suite, data in results.items():
        success_mean = data['success']['entropy_mean']
        failure_mean = data['failure']['entropy_mean']

        if success_mean is not None and failure_mean is not None:
            suites.append(suite)
            differences.append(failure_mean - success_mean)
            num_success.append(data['success']['num_trials'])
            num_failure.append(data['failure']['num_trials'])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if d > 0 else 'green' for d in differences]
    bars = ax.bar(suites, differences, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Suite', fontsize=12)
    ax.set_ylabel('Entropy Difference (Failure - Success)', fontsize=12)
    ax.set_title('Entropy Difference: Higher values = Failed trials are more random',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add annotations for number of trials
    for i, (suite, diff, n_s, n_f) in enumerate(zip(suites, differences, num_success, num_failure)):
        ax.text(i, diff + (0.05 if diff > 0 else -0.05),
                f'S:{n_s}\nF:{n_f}',
                ha='center', va='bottom' if diff > 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_difference.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_difference.png'}")
    plt.close()


def plot_task_level_comparison(results: dict, output_dir: Path, suite_name: str):
    """
    Create detailed task-level comparison for a specific suite.
    """
    if suite_name not in results:
        print(f"Warning: Suite {suite_name} not found in results")
        return

    task_details = results[suite_name]['task_details']
    tasks = list(task_details.keys())

    success_means = []
    failure_means = []
    task_labels = []

    for task, stats in task_details.items():
        s_mean = stats['success_entropy_mean']
        f_mean = stats['failure_entropy_mean']

        # Only include tasks with data for at least one status
        if s_mean is not None or f_mean is not None:
            task_labels.append(task[:50] + '...' if len(task) > 50 else task)
            success_means.append(s_mean if s_mean is not None else 0)
            failure_means.append(f_mean if f_mean is not None else 0)

    fig, ax = plt.subplots(figsize=(12, max(8, len(tasks) * 0.4)))
    y = np.arange(len(task_labels))
    height = 0.35

    bars1 = ax.barh(y - height/2, success_means, height,
                    label='Success', alpha=0.8, color='green')
    bars2 = ax.barh(y + height/2, failure_means, height,
                    label='Failure', alpha=0.8, color='red')

    ax.set_yticks(y)
    ax.set_yticklabels(task_labels, fontsize=8)
    ax.set_xlabel('Action Entropy (KDE)', fontsize=12)
    ax.set_title(f'Task-Level Entropy Comparison: {suite_name}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    safe_name = suite_name.replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f'task_level_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'task_level_{safe_name}.png'}")
    plt.close()


def plot_distribution_violin(results_raw: dict, output_dir: Path):
    """
    Create violin plots showing the distribution of entropy values.
    """
    # Prepare data for violin plot
    data_for_plot = []

    for suite_id, tasks in results_raw.items():
        suite_name = f"LIBERO-{suite_id}"

        for task_desc, status_data in tasks.items():
            for status in ['success', 'failure']:
                for trial in status_data[status]:
                    entropy = trial['entropy_stats'].get('action_entropy_kde')
                    if entropy is not None:
                        data_for_plot.append({
                            'Suite': suite_name,
                            'Status': status.capitalize(),
                            'Entropy': entropy,
                        })

    if not data_for_plot:
        print("No data for violin plot")
        return

    # Create DataFrame for seaborn
    import pandas as pd
    df = pd.DataFrame(data_for_plot)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df, x='Suite', y='Entropy', hue='Status',
                   split=True, ax=ax, palette={'Success': 'green', 'Failure': 'red'})

    ax.set_xlabel('Suite', fontsize=12)
    ax.set_ylabel('Action Entropy (KDE)', fontsize=12)
    ax.set_title('Distribution of Action Entropy by Suite and Status',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_distribution_violin.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_distribution_violin.png'}")
    plt.close()


def generate_summary_report(results: dict, output_path: Path):
    """
    Generate a markdown summary report.
    """
    lines = []
    lines.append("# Action Entropy Analysis by Suite\n")
    lines.append("Analysis of action entropy separated by success/failure status.\n")
    lines.append("Lower (more negative) entropy = more deterministic actions\n")
    lines.append("Higher (less negative) entropy = more random/exploratory actions\n\n")

    lines.append("## Summary by Suite\n")

    for suite_name, stats in results.items():
        lines.append(f"\n### {suite_name}\n")
        lines.append(f"- **Total tasks**: {stats['num_tasks']}\n")

        lines.append(f"\n**Success Trials** ({stats['success']['num_trials']} total):\n")
        if stats['success']['entropy_mean'] is not None:
            lines.append(f"- Mean entropy: {stats['success']['entropy_mean']:.4f} ± {stats['success']['entropy_std']:.4f}\n")
            lines.append(f"- Range: [{stats['success']['entropy_min']:.4f}, {stats['success']['entropy_max']:.4f}]\n")
        else:
            lines.append("- No successful trials\n")

        lines.append(f"\n**Failure Trials** ({stats['failure']['num_trials']} total):\n")
        if stats['failure']['entropy_mean'] is not None:
            lines.append(f"- Mean entropy: {stats['failure']['entropy_mean']:.4f} ± {stats['failure']['entropy_std']:.4f}\n")
            lines.append(f"- Range: [{stats['failure']['entropy_min']:.4f}, {stats['failure']['entropy_max']:.4f}]\n")
        else:
            lines.append("- No failed trials\n")

        if stats['success']['entropy_mean'] is not None and stats['failure']['entropy_mean'] is not None:
            diff = stats['failure']['entropy_mean'] - stats['success']['entropy_mean']
            lines.append(f"\n**Entropy Difference** (Failure - Success): {diff:.4f}\n")
            if diff > 0:
                lines.append(f"- Failed trials have **higher entropy** (more random/exploratory)\n")
                lines.append(f"- Interpretation: Policy is uncertain/exploring during failures\n")
            else:
                lines.append(f"- Failed trials have **lower entropy** (more deterministic)\n")
                lines.append(f"- Interpretation: Policy is confident but wrong during failures\n")

    lines.append("\n## Key Findings\n\n")

    # Calculate overall stats
    all_success_entropy = []
    all_failure_entropy = []
    for suite_name, stats in results.items():
        if stats['success']['entropy_mean'] is not None:
            all_success_entropy.append(stats['success']['entropy_mean'])
        if stats['failure']['entropy_mean'] is not None:
            all_failure_entropy.append(stats['failure']['entropy_mean'])

    if all_success_entropy and all_failure_entropy:
        overall_success = np.mean(all_success_entropy)
        overall_failure = np.mean(all_failure_entropy)
        lines.append(f"- Overall success entropy: {overall_success:.4f}\n")
        lines.append(f"- Overall failure entropy: {overall_failure:.4f}\n")
        lines.append(f"- Overall difference: {overall_failure - overall_success:.4f}\n\n")

    lines.append("## Interpretation Guide\n\n")
    lines.append("**Entropy (KDE-based)**:\n")
    lines.append("- Measures the randomness/unpredictability of actions\n")
    lines.append("- Calculated using Kernel Density Estimation on action sequences\n")
    lines.append("- Negative values (closer to 0) = higher entropy = more random\n")
    lines.append("- More negative values = lower entropy = more deterministic\n\n")

    lines.append("**Success vs Failure Patterns**:\n")
    lines.append("1. **Higher entropy in failures**: Policy is uncertain, exploring randomly\n")
    lines.append("2. **Lower entropy in failures**: Policy is deterministic but making consistent mistakes\n")
    lines.append("3. **Similar entropy**: Success/failure may not be related to action randomness\n")

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize entropy results from calculate_entropy_by_suite.py"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="entropy_results.json",
        help="Path to entropy results JSON file"
    )
    parser.add_argument(
        "--results_raw",
        type=str,
        default="entropy_results_raw.json",
        help="Path to raw entropy results JSON file (for distributions)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="entropy_plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Load raw results if available
    results_raw = None
    results_raw_path = Path(args.results_raw)
    if results_raw_path.exists():
        with open(results_raw_path, 'r') as f:
            results_raw = json.load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating visualizations in: {output_dir}")

    # Generate plots
    plot_entropy_comparison(results, output_dir)
    plot_entropy_difference(results, output_dir)

    # Task-level plots for each suite
    for suite_name in results.keys():
        plot_task_level_comparison(results, output_dir, suite_name)

    # Violin plot (requires raw data)
    if results_raw is not None:
        try:
            plot_distribution_violin(results_raw, output_dir)
        except Exception as e:
            print(f"Warning: Could not create violin plot: {e}")

    # Generate summary report
    summary_path = output_dir / 'entropy_analysis_summary.md'
    generate_summary_report(results, summary_path)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
