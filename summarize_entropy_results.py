#!/usr/bin/env python3
"""
Generate a comprehensive summary report of entropy results.

Usage:
    python summarize_entropy_results.py --results entropy_results.json
"""

import json
import numpy as np
from pathlib import Path
import argparse


def generate_markdown_report(results: dict, output_path: Path):
    """
    Generate a detailed markdown summary report.
    """
    lines = []
    lines.append("# Action Entropy Analysis by LIBERO Suite\n\n")
    lines.append("This report analyzes action entropy across different LIBERO task suites, ")
    lines.append("comparing successful and failed trial runs.\n\n")

    lines.append("## Understanding Entropy\n\n")
    lines.append("**Action Entropy** measures the randomness/unpredictability of robot actions:\n\n")
    lines.append("- **Calculated using**: Kernel Density Estimation (KDE) on action sequences\n")
    lines.append("- **Values**: Negative numbers (entropy = -mean(log_density))\n")
    lines.append("- **Interpretation**:\n")
    lines.append("  - More negative (e.g., -10.0) = **Lower entropy** = More deterministic/consistent actions\n")
    lines.append("  - Less negative (e.g., -6.0) = **Higher entropy** = More random/exploratory actions\n\n")

    lines.append("---\n\n")
    lines.append("## Results by Suite\n\n")

    # Collect data for overall comparison
    suite_comparisons = []

    for suite_name, stats in sorted(results.items()):
        lines.append(f"### {suite_name}\n\n")

        # Basic stats
        lines.append(f"**Tasks**: {stats['num_tasks']}  \n")
        lines.append(f"**Success trials**: {stats['success']['num_trials']}  \n")
        lines.append(f"**Failure trials**: {stats['failure']['num_trials']}  \n")

        success_rate = 100 * stats['success']['num_trials'] / (
            stats['success']['num_trials'] + stats['failure']['num_trials']
        ) if (stats['success']['num_trials'] + stats['failure']['num_trials']) > 0 else 0
        lines.append(f"**Success rate**: {success_rate:.1f}%\n\n")

        # Success stats
        lines.append("#### Success Trials\n\n")
        if stats['success']['entropy_mean'] is not None:
            lines.append(f"| Metric | Value |\n")
            lines.append(f"|--------|-------|\n")
            lines.append(f"| Mean entropy | {stats['success']['entropy_mean']:.4f} |\n")
            lines.append(f"| Std deviation | {stats['success']['entropy_std']:.4f} |\n")
            lines.append(f"| Min entropy | {stats['success']['entropy_min']:.4f} |\n")
            lines.append(f"| Max entropy | {stats['success']['entropy_max']:.4f} |\n\n")
        else:
            lines.append("*No successful trials*\n\n")

        # Failure stats
        lines.append("#### Failure Trials\n\n")
        if stats['failure']['entropy_mean'] is not None:
            lines.append(f"| Metric | Value |\n")
            lines.append(f"|--------|-------|\n")
            lines.append(f"| Mean entropy | {stats['failure']['entropy_mean']:.4f} |\n")
            lines.append(f"| Std deviation | {stats['failure']['entropy_std']:.4f} |\n")
            lines.append(f"| Min entropy | {stats['failure']['entropy_min']:.4f} |\n")
            lines.append(f"| Max entropy | {stats['failure']['entropy_max']:.4f} |\n\n")
        else:
            lines.append("*No failed trials*\n\n")

        # Comparison
        if stats['success']['entropy_mean'] is not None and stats['failure']['entropy_mean'] is not None:
            diff = stats['failure']['entropy_mean'] - stats['success']['entropy_mean']
            lines.append("#### Comparison\n\n")
            lines.append(f"**Entropy difference** (Failure - Success): `{diff:.4f}`\n\n")

            if abs(diff) < 0.1:
                interpretation = "**Minimal difference**: Success and failure show similar action patterns"
            elif diff > 0:
                interpretation = (
                    f"**Failed trials have {diff:.4f} HIGHER entropy** (more random):\n"
                    "- Policy shows more exploratory/uncertain behavior during failures\n"
                    "- May indicate the policy is \"searching\" for solutions when struggling"
                )
            else:
                interpretation = (
                    f"**Failed trials have {abs(diff):.4f} LOWER entropy** (more deterministic):\n"
                    "- Policy is confident but systematically wrong\n"
                    "- Suggests learned incorrect behavior patterns"
                )

            lines.append(f"{interpretation}\n\n")

            suite_comparisons.append({
                'suite': suite_name,
                'diff': diff,
                'success_trials': stats['success']['num_trials'],
                'failure_trials': stats['failure']['num_trials'],
                'success_mean': stats['success']['entropy_mean'],
                'failure_mean': stats['failure']['entropy_mean'],
            })

        # Top tasks by success/failure
        lines.append("#### Task Details\n\n")

        task_details = stats['task_details']
        # Sort by number of trials
        sorted_tasks = sorted(
            task_details.items(),
            key=lambda x: x[1]['num_success_trials'] + x[1]['num_failure_trials'],
            reverse=True
        )

        lines.append("| Task | Success | Failure | S. Entropy | F. Entropy | Diff |\n")
        lines.append("|------|---------|---------|------------|------------|------|\n")

        for task, task_stats in sorted_tasks[:10]:  # Show top 10 tasks
            task_name = task[:40] + "..." if len(task) > 40 else task
            n_success = task_stats['num_success_trials']
            n_failure = task_stats['num_failure_trials']
            s_ent = f"{task_stats['success_entropy_mean']:.3f}" if task_stats['success_entropy_mean'] else "N/A"
            f_ent = f"{task_stats['failure_entropy_mean']:.3f}" if task_stats['failure_entropy_mean'] else "N/A"

            if task_stats['success_entropy_mean'] and task_stats['failure_entropy_mean']:
                diff = task_stats['failure_entropy_mean'] - task_stats['success_entropy_mean']
                diff_str = f"{diff:+.3f}"
            else:
                diff_str = "N/A"

            lines.append(f"| {task_name} | {n_success} | {n_failure} | {s_ent} | {f_ent} | {diff_str} |\n")

        if len(sorted_tasks) > 10:
            lines.append(f"\n*Showing top 10 of {len(sorted_tasks)} tasks*\n")

        lines.append("\n---\n\n")

    # Overall summary
    lines.append("## Overall Summary\n\n")

    if suite_comparisons:
        lines.append("### Suite Comparison Table\n\n")
        lines.append("| Suite | Success Rate | Success Entropy | Failure Entropy | Difference | Interpretation |\n")
        lines.append("|-------|--------------|-----------------|-----------------|------------|----------------|\n")

        for comp in suite_comparisons:
            suite = comp['suite']
            n_success = comp['success_trials']
            n_failure = comp['failure_trials']
            success_rate = 100 * n_success / (n_success + n_failure)
            s_ent = comp['success_mean']
            f_ent = comp['failure_mean']
            diff = comp['diff']

            if diff > 0.1:
                interp = "More random in failure"
            elif diff < -0.1:
                interp = "More deterministic in failure"
            else:
                interp = "Similar"

            lines.append(
                f"| {suite} | {success_rate:.1f}% | {s_ent:.3f} | {f_ent:.3f} | {diff:+.3f} | {interp} |\n"
            )

        lines.append("\n")

        # Calculate correlations
        lines.append("### Key Insights\n\n")

        avg_success_ent = np.mean([c['success_mean'] for c in suite_comparisons])
        avg_failure_ent = np.mean([c['failure_mean'] for c in suite_comparisons])
        avg_diff = np.mean([c['diff'] for c in suite_comparisons])

        lines.append(f"1. **Average success entropy**: {avg_success_ent:.4f}\n")
        lines.append(f"2. **Average failure entropy**: {avg_failure_ent:.4f}\n")
        lines.append(f"3. **Average difference**: {avg_diff:.4f}\n\n")

        if avg_diff > 0.1:
            lines.append("**Overall pattern**: Failed trials tend to have more random/exploratory actions. ")
            lines.append("This suggests the policy is uncertain when it fails.\n\n")
        elif avg_diff < -0.1:
            lines.append("**Overall pattern**: Failed trials tend to have more deterministic actions. ")
            lines.append("This suggests the policy is confident but systematically wrong when it fails.\n\n")
        else:
            lines.append("**Overall pattern**: Success and failure show similar entropy levels. ")
            lines.append("Action randomness may not be the primary factor distinguishing success from failure.\n\n")

    lines.append("## Recommendations\n\n")
    lines.append("Based on the entropy analysis:\n\n")

    if suite_comparisons:
        for comp in suite_comparisons:
            if comp['diff'] > 0.5:
                lines.append(f"- **{comp['suite']}**: High entropy in failures suggests adding more training data ")
                lines.append("or exploration strategies for difficult scenarios\n")
            elif comp['diff'] < -0.5:
                lines.append(f"- **{comp['suite']}**: Low entropy in failures suggests the policy has learned ")
                lines.append("incorrect patterns - consider analyzing failure modes and retraining\n")

    lines.append("\n")
    lines.append("---\n\n")
    lines.append("*Report generated using KDE-based action entropy analysis*\n")

    # Write report
    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Markdown report saved: {output_path}")


def generate_text_summary(results: dict):
    """
    Print a concise text summary to console.
    """
    print("\n" + "="*80)
    print("ENTROPY ANALYSIS SUMMARY")
    print("="*80 + "\n")

    for suite_name, stats in sorted(results.items()):
        print(f"{suite_name}:")
        print(f"  Tasks: {stats['num_tasks']}")
        print(f"  Success: {stats['success']['num_trials']} trials, "
              f"entropy = {stats['success']['entropy_mean']:.4f} ± {stats['success']['entropy_std']:.4f}"
              if stats['success']['entropy_mean'] else "  Success: No trials")
        print(f"  Failure: {stats['failure']['num_trials']} trials, "
              f"entropy = {stats['failure']['entropy_mean']:.4f} ± {stats['failure']['entropy_std']:.4f}"
              if stats['failure']['entropy_mean'] else "  Failure: No trials")

        if stats['success']['entropy_mean'] and stats['failure']['entropy_mean']:
            diff = stats['failure']['entropy_mean'] - stats['success']['entropy_mean']
            direction = "higher (more random)" if diff > 0 else "lower (more deterministic)"
            print(f"  Difference: {diff:+.4f} - failures are {direction}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize entropy results from calculate_entropy_by_suite.py"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="entropy_results.json",
        help="Path to entropy results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_analysis_report.md",
        help="Output markdown report path"
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Generate text summary
    generate_text_summary(results)

    # Generate markdown report
    output_path = Path(args.output)
    generate_markdown_report(results, output_path)

    print(f"\n✓ Analysis complete! Check '{output_path}' for detailed report.")


if __name__ == "__main__":
    main()
