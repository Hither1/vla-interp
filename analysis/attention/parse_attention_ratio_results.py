#!/usr/bin/env python3
"""
Parse and analyze visual/linguistic attention ratio results.

Usage:
  python parse_attention_ratio_results.py \
    --results results/attention_ratio/attention_ratio_results_libero_10.json \
    --output analysis_summary.txt

  # Generate comparison plots
  python parse_attention_ratio_results.py \
    --results results/attention_ratio/attention_ratio_results_*.json \
    --plot-comparison --output-dir results/attention_ratio_analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_results(results_path: str) -> Dict[str, List[Dict]]:
    """Load results, returning a dict mapping subset name -> list of episodes.

    When given a directory, each JSON file becomes one subset (name derived
    from the filename, e.g. attention_ratio_results_libero_goal.json -> libero_goal).
    When given a single file, returns a single-entry dict.
    """
    p = Path(results_path)
    if p.is_dir():
        subsets = {}
        for json_file in sorted(p.glob("*.json")):
            name = json_file.stem
            for prefix in ("attention_ratio_results_",):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            with open(json_file, 'r') as f:
                data = json.load(f)
            subsets[name] = data if isinstance(data, list) else [data]
        return subsets
    with open(results_path, 'r') as f:
        data = json.load(f)
    name = p.stem
    for prefix in ("attention_ratio_results_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return {name: data if isinstance(data, list) else [data]}


def compute_aggregate_stats(all_results: List[Dict]) -> Dict:
    """Compute aggregate statistics across all episodes."""
    stats = {
        "num_episodes": len(all_results),
        "num_tasks": len(set(r["task_id"] for r in all_results)),
        "success_rate": np.mean([r.get("success", False) for r in all_results]),
    }

    # Collect all per-step ratios
    all_ratios = []
    all_visual_fracs = []
    all_linguistic_fracs = []

    for result in all_results:
        per_step = result.get("per_step_ratios", {})

        # Use layers_avg if available, otherwise first layer
        key = "layers_avg" if "layers_avg" in per_step else list(per_step.keys())[0] if per_step else None

        if key and key in per_step:
            for step_data in per_step[key]:
                ratio = step_data.get("visual_linguistic_ratio", 0)
                if np.isfinite(ratio):
                    all_ratios.append(ratio)
                all_visual_fracs.append(step_data.get("visual_fraction", 0))
                all_linguistic_fracs.append(step_data.get("linguistic_fraction", 0))

    if all_ratios:
        stats["visual_linguistic_ratio"] = {
            "mean": float(np.mean(all_ratios)),
            "std": float(np.std(all_ratios)),
            "median": float(np.median(all_ratios)),
            "min": float(np.min(all_ratios)),
            "max": float(np.max(all_ratios)),
            "q25": float(np.percentile(all_ratios, 25)),
            "q75": float(np.percentile(all_ratios, 75)),
        }

    if all_visual_fracs:
        stats["visual_fraction"] = {
            "mean": float(np.mean(all_visual_fracs)),
            "std": float(np.std(all_visual_fracs)),
        }

    if all_linguistic_fracs:
        stats["linguistic_fraction"] = {
            "mean": float(np.mean(all_linguistic_fracs)),
            "std": float(np.std(all_linguistic_fracs)),
        }

    return stats


def plot_ratio_distribution(all_results: List[Dict], output_path: str):
    """Plot distribution of visual/linguistic ratios."""
    all_ratios = []

    for result in all_results:
        per_step = result.get("per_step_ratios", {})
        key = "layers_avg" if "layers_avg" in per_step else list(per_step.keys())[0] if per_step else None

        if key and key in per_step:
            for step_data in per_step[key]:
                ratio = step_data.get("visual_linguistic_ratio", 0)
                if np.isfinite(ratio) and ratio < 100:  # Filter outliers
                    all_ratios.append(ratio)

    if not all_ratios:
        print("No valid ratios to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(all_ratios, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Equal (ratio=1.0)')
    ax.axvline(x=np.mean(all_ratios), color='g', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_ratios):.2f}')
    ax.set_xlabel('Visual / Linguistic Ratio', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Visual/Linguistic Attention Ratios', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([all_ratios], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Equal (ratio=1.0)')
    ax.set_ylabel('Visual / Linguistic Ratio', fontsize=12)
    ax.set_title('Ratio Distribution Summary', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()


def plot_per_task_comparison(all_results: List[Dict], output_path: str):
    """Plot comparison of ratios across different tasks."""
    task_ratios = {}

    for result in all_results:
        task_desc = result.get("task_description", f"Task {result['task_id']}")
        per_step = result.get("per_step_ratios", {})
        key = "layers_avg" if "layers_avg" in per_step else list(per_step.keys())[0] if per_step else None

        if key and key in per_step:
            ratios = [
                step_data.get("visual_linguistic_ratio", 0)
                for step_data in per_step[key]
                if np.isfinite(step_data.get("visual_linguistic_ratio", 0))
            ]
            if ratios:
                if task_desc not in task_ratios:
                    task_ratios[task_desc] = []
                task_ratios[task_desc].extend(ratios)

    if not task_ratios:
        print("No task data to plot")
        return

    # Compute mean ratio per task
    task_names = []
    task_means = []
    task_stds = []

    for task_desc, ratios in sorted(task_ratios.items()):
        task_names.append(task_desc[:40])  # Truncate long task names
        task_means.append(np.mean(ratios))
        task_stds.append(np.std(ratios))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(task_names))
    ax.bar(x, task_means, yerr=task_stds, alpha=0.7, capsize=5, edgecolor='black')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Equal (ratio=1.0)')

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Mean Visual / Linguistic Ratio', fontsize=12)
    ax.set_title('Attention Ratio by Task', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-task comparison to {output_path}")
    plt.close()


def plot_attention_fractions(all_results: List[Dict], output_path: str):
    """Plot average attention fractions (visual vs linguistic)."""
    visual_fracs = []
    linguistic_fracs = []

    for result in all_results:
        per_step = result.get("per_step_ratios", {})
        key = "layers_avg" if "layers_avg" in per_step else list(per_step.keys())[0] if per_step else None

        if key and key in per_step:
            for step_data in per_step[key]:
                visual_fracs.append(step_data.get("visual_fraction", 0))
                linguistic_fracs.append(step_data.get("linguistic_fraction", 0))

    if not visual_fracs:
        print("No fraction data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Visual', 'Linguistic']
    means = [np.mean(visual_fracs), np.mean(linguistic_fracs)]
    stds = [np.std(visual_fracs), np.std(linguistic_fracs)]

    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, alpha=0.7, capsize=10, edgecolor='black',
                   color=['#3498db', '#e74c3c'])

    ax.set_ylabel('Mean Attention Fraction', fontsize=12)
    ax.set_title('Average Attention Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.1%}\n±{std:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention fractions plot to {output_path}")
    plt.close()


def print_summary(stats: Dict, label: str):
    """Print formatted summary of results."""
    print("\n" + "="*70)
    print(f"ATTENTION RATIO ANALYSIS: {label}")
    print("="*70)

    print(f"\nDataset:")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Tasks: {stats['num_tasks']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    if "visual_linguistic_ratio" in stats:
        ratio = stats["visual_linguistic_ratio"]
        print(f"\nVisual/Linguistic Ratio:")
        print(f"  Mean:   {ratio['mean']:.3f} ± {ratio['std']:.3f}")
        print(f"  Median: {ratio['median']:.3f}")
        print(f"  Range:  [{ratio['min']:.3f}, {ratio['max']:.3f}]")
        print(f"  IQR:    [{ratio['q25']:.3f}, {ratio['q75']:.3f}]")

        if ratio['mean'] > 1.5:
            print(f"  → Model is VISUAL-dominant (prefers visual features)")
        elif ratio['mean'] < 0.67:
            print(f"  → Model is LINGUISTIC-dominant (prefers text instructions)")
        else:
            print(f"  → Model shows BALANCED attention between visual and linguistic")

    if "visual_fraction" in stats:
        print(f"\nAttention Fractions:")
        print(f"  Visual:     {stats['visual_fraction']['mean']:.1%} ± {stats['visual_fraction']['std']:.1%}")
        print(f"  Linguistic: {stats['linguistic_fraction']['mean']:.1%} ± {stats['linguistic_fraction']['std']:.1%}")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse attention ratio results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file(s)")
    parser.add_argument("--output", type=str, help="Output summary text file")
    parser.add_argument("--output-dir", type=str, default="results/attention_ratio_analysis", help="Output directory for plots")
    parser.add_argument("--plot-distribution", action="store_true", help="Plot ratio distribution")
    parser.add_argument("--plot-per-task", action="store_true", help="Plot per-task comparison")
    parser.add_argument("--plot-fractions", action="store_true", help="Plot attention fractions")
    parser.add_argument("--plot-all", action="store_true", help="Generate all plots")

    args = parser.parse_args()

    # Load results split by subset
    subsets = load_results(args.results)

    output_lines = []

    for subset_name, results in subsets.items():
        stats = compute_aggregate_stats(results)

        # Print summary to stdout
        print_summary(stats, subset_name)

        # Accumulate lines for optional file output
        if args.output:
            output_lines.append(f"Attention Ratio Analysis: {subset_name}")
            output_lines.append("=" * 70)
            output_lines.append(f"Episodes: {stats['num_episodes']}")
            output_lines.append(f"Tasks: {stats['num_tasks']}")
            output_lines.append(f"Success Rate: {stats['success_rate']:.1%}")
            if "visual_linguistic_ratio" in stats:
                ratio = stats["visual_linguistic_ratio"]
                output_lines.append(f"Visual/Linguistic Ratio:")
                output_lines.append(f"  Mean:   {ratio['mean']:.3f} ± {ratio['std']:.3f}")
                output_lines.append(f"  Median: {ratio['median']:.3f}")
                output_lines.append(f"  Range:  [{ratio['min']:.3f}, {ratio['max']:.3f}]")
            if "visual_fraction" in stats:
                output_lines.append(f"Attention Fractions:")
                output_lines.append(f"  Visual:     {stats['visual_fraction']['mean']:.1%}")
                output_lines.append(f"  Linguistic: {stats['linguistic_fraction']['mean']:.1%}")
            output_lines.append("")

        # Generate per-subset plots with subset name as suffix
        if args.plot_all or args.plot_distribution:
            os.makedirs(args.output_dir, exist_ok=True)
            plot_ratio_distribution(results, os.path.join(args.output_dir, f"ratio_distribution_{subset_name}.png"))

        if args.plot_all or args.plot_per_task:
            os.makedirs(args.output_dir, exist_ok=True)
            plot_per_task_comparison(results, os.path.join(args.output_dir, f"ratio_per_task_{subset_name}.png"))

        if args.plot_all or args.plot_fractions:
            os.makedirs(args.output_dir, exist_ok=True)
            plot_attention_fractions(results, os.path.join(args.output_dir, f"attention_fractions_{subset_name}.png"))

    # Also print combined stats when multiple subsets are present
    if len(subsets) > 1:
        all_results = [r for results in subsets.values() for r in results]
        combined_stats = compute_aggregate_stats(all_results)
        print_summary(combined_stats, "COMBINED")

        if args.output:
            output_lines.append("Attention Ratio Analysis: COMBINED")
            output_lines.append("=" * 70)
            output_lines.append(f"Episodes: {combined_stats['num_episodes']}")
            output_lines.append(f"Tasks: {combined_stats['num_tasks']}")
            output_lines.append(f"Success Rate: {combined_stats['success_rate']:.1%}")
            if "visual_linguistic_ratio" in combined_stats:
                ratio = combined_stats["visual_linguistic_ratio"]
                output_lines.append(f"Visual/Linguistic Ratio:")
                output_lines.append(f"  Mean:   {ratio['mean']:.3f} ± {ratio['std']:.3f}")
                output_lines.append(f"  Median: {ratio['median']:.3f}")
                output_lines.append(f"  Range:  [{ratio['min']:.3f}, {ratio['max']:.3f}]")

    if args.output:
        with open(args.output, 'w') as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
