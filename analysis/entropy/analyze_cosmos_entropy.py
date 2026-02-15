#!/usr/bin/env python3
"""Analyze action entropy for Cosmos policy from logs/action_data.

Combines:
1. Pre-computed task-level entropy from action_entropy_*.json files (all suites)
2. Per-trial entropy from actions_*.json files (LIBERO-90, success vs failure)

Produces:
- Cross-suite entropy comparison bar chart
- Per-task entropy distribution (box/violin plots)
- Success vs failure entropy comparison for LIBERO-90
- Summary report

Usage:
    python analysis/entropy/analyze_cosmos_entropy.py
    python analysis/entropy/analyze_cosmos_entropy.py --data-dir <path> --output-dir results/entropy/cosmos
"""

import argparse
import collections
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITE_ORDER = ["spatial", "object", "goal", "10", "90"]
SUITE_LABELS = {
    "spatial": "LIBERO-Spatial",
    "object": "LIBERO-Object",
    "goal": "LIBERO-Goal",
    "10": "LIBERO-10",
    "90": "LIBERO-90",
}


def load_precomputed_entropy(data_dir: str) -> Dict[str, list]:
    """Load action_entropy_*.json files, grouped by suite."""
    pattern = os.path.join(data_dir, "action_entropy_*.json")
    files = sorted(glob.glob(pattern))

    suite_data = collections.defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        # Extract suite from filename: action_entropy_libero_{suite}_...
        basename = os.path.basename(fpath)
        after = basename.split("action_entropy_libero_")[1]
        suite = after.split("_")[0]
        suite_data[suite].append(d)

    return dict(suite_data)


def load_per_trial_data(data_dir: str) -> Dict[str, list]:
    """Load actions_*.json per-trial files, grouped by suite."""
    pattern = os.path.join(data_dir, "actions_*.json")
    files = sorted(glob.glob(pattern))

    suite_trials = collections.defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        basename = os.path.basename(fpath)
        after = basename.split("actions_libero_")[1]
        suite = after.split("_")[0]
        suite_trials[suite].append(d)

    return dict(suite_trials)


def compute_trial_entropy(actions_list: List[dict], action_dim: int = 7) -> float:
    """Compute KDE entropy for a single trial's actions."""
    from scipy.stats import gaussian_kde

    actions = np.array([a["action"] for a in actions_list])[:, :action_dim]
    N, D = actions.shape
    if N < D + 1:
        return None
    try:
        kde = gaussian_kde(actions.T)
        log_densities = kde.logpdf(actions.T)
        return -float(np.mean(log_densities))
    except np.linalg.LinAlgError:
        return None


def plot_cross_suite_entropy(suite_stats: dict, output_dir: Path):
    """Bar chart of mean task-level entropy across suites."""
    suites = [s for s in SUITE_ORDER if s in suite_stats]
    means = [suite_stats[s]["mean"] for s in suites]
    stds = [suite_stats[s]["std"] for s in suites]
    labels = [SUITE_LABELS.get(s, s) for s in suites]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(suites)))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Action Entropy (KDE)", fontsize=12)
    ax.set_title("Cosmos Policy: Action Entropy by LIBERO Suite", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add task count annotations
    for i, s in enumerate(suites):
        n = suite_stats[s]["n_tasks"]
        sr = suite_stats[s].get("success_rate", None)
        label = f"n={n} tasks"
        if sr is not None:
            label += f"\nSR={sr:.0f}%"
        ax.text(i, ax.get_ylim()[0] + 0.1, label, ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / "cross_suite_entropy.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'cross_suite_entropy.png'}")
    plt.close()


def plot_per_task_entropy(suite_data: dict, output_dir: Path):
    """Box plot of per-task entropy for each suite."""
    suites = [s for s in SUITE_ORDER if s in suite_data]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = []
    labels = []
    for s in suites:
        entropies = [d["action_entropy"]["action_entropy_kde"]
                     for d in suite_data[s] if "action_entropy" in d]
        data_to_plot.append(entropies)
        labels.append(SUITE_LABELS.get(s, s))

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(suites)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Action Entropy (KDE)", fontsize=12)
    ax.set_title("Cosmos Policy: Per-Task Entropy Distribution by Suite", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_task_entropy_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'per_task_entropy_boxplot.png'}")
    plt.close()


def plot_success_vs_failure(trial_data: dict, output_dir: Path):
    """Compare entropy of success vs failure trials (from per-trial data)."""
    # Compute per-trial entropy
    success_entropies = collections.defaultdict(list)
    failure_entropies = collections.defaultdict(list)

    for suite, trials in trial_data.items():
        print(f"Computing per-trial entropy for {suite} ({len(trials)} trials)...")
        for trial in trials:
            entropy = compute_trial_entropy(trial["actions"])
            if entropy is None:
                continue
            if trial.get("success", False):
                success_entropies[suite].append(entropy)
            else:
                failure_entropies[suite].append(entropy)

    suites = sorted(set(list(success_entropies.keys()) + list(failure_entropies.keys())),
                    key=lambda s: SUITE_ORDER.index(s) if s in SUITE_ORDER else 99)

    if not suites:
        print("No per-trial data to plot success vs failure")
        return success_entropies, failure_entropies

    fig, axes = plt.subplots(1, len(suites), figsize=(6 * len(suites), 6), squeeze=False)
    axes = axes[0]

    for ax, suite in zip(axes, suites):
        s_ent = success_entropies.get(suite, [])
        f_ent = failure_entropies.get(suite, [])

        data = []
        labels_list = []
        colors_list = []
        if s_ent:
            data.append(s_ent)
            labels_list.append(f"Success\n(n={len(s_ent)})")
            colors_list.append("green")
        if f_ent:
            data.append(f_ent)
            labels_list.append(f"Failure\n(n={len(f_ent)})")
            colors_list.append("red")

        if data:
            bp = ax.boxplot(data, labels=labels_list, patch_artist=True, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            # Add means as text
            for i, d in enumerate(data):
                ax.text(i + 1, np.mean(d), f"  {np.mean(d):.2f}", va="center", fontsize=9, fontweight="bold")

        ax.set_ylabel("Action Entropy (KDE)", fontsize=11)
        ax.set_title(SUITE_LABELS.get(suite, suite), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # t-test if both groups exist
        if len(s_ent) > 1 and len(f_ent) > 1:
            from scipy.stats import ttest_ind, mannwhitneyu
            t, p = ttest_ind(s_ent, f_ent)
            u, up = mannwhitneyu(s_ent, f_ent, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_xlabel(f"t={t:.2f}, p={p:.4f} ({sig})\nU={u:.0f}, p_U={up:.4f}", fontsize=9)

    plt.suptitle("Cosmos Policy: Success vs Failure Action Entropy", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "success_vs_failure_entropy.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'success_vs_failure_entropy.png'}")
    plt.close()

    return success_entropies, failure_entropies


def plot_entropy_vs_success_rate(suite_data: dict, output_dir: Path):
    """Scatter plot: per-task entropy vs success rate."""
    tasks = []
    for suite, items in suite_data.items():
        for d in items:
            if "action_entropy" not in d:
                continue
            ent = d["action_entropy"]["action_entropy_kde"]
            sr = d.get("num_successes", 0) / max(d.get("num_trials", 1), 1) * 100
            tasks.append({"suite": suite, "entropy": ent, "success_rate": sr,
                          "task": d.get("task_description", "")})

    if not tasks:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for suite in SUITE_ORDER:
        pts = [t for t in tasks if t["suite"] == suite]
        if not pts:
            continue
        ax.scatter([t["entropy"] for t in pts], [t["success_rate"] for t in pts],
                   label=SUITE_LABELS.get(suite, suite), alpha=0.7, s=50)

    ax.set_xlabel("Action Entropy (KDE)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cosmos Policy: Task Entropy vs Success Rate", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Add correlation
    all_ent = [t["entropy"] for t in tasks]
    all_sr = [t["success_rate"] for t in tasks]
    corr = np.corrcoef(all_ent, all_sr)[0, 1]
    ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_vs_success_rate.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'entropy_vs_success_rate.png'}")
    plt.close()


def generate_summary(suite_data: dict, suite_stats: dict, sf_success: dict, sf_failure: dict, output_dir: Path):
    """Generate a text summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Cosmos Policy: Action Entropy Analysis Summary")
    lines.append("=" * 70)
    lines.append("")

    # Cross-suite comparison
    lines.append("CROSS-SUITE COMPARISON (task-level entropy)")
    lines.append("-" * 70)
    lines.append(f"{'Suite':<20} {'Tasks':>5} {'Trials':>7} {'SR%':>6} {'Mean H':>10} {'Std H':>10} {'Min H':>10} {'Max H':>10}")
    lines.append("-" * 70)

    for suite in SUITE_ORDER:
        if suite not in suite_stats:
            continue
        s = suite_stats[suite]
        lines.append(
            f"{SUITE_LABELS.get(suite, suite):<20} {s['n_tasks']:>5} {s['n_trials']:>7} "
            f"{s.get('success_rate', 0):>5.1f} {s['mean']:>10.4f} {s['std']:>10.4f} "
            f"{s['min']:>10.4f} {s['max']:>10.4f}"
        )

    lines.append("")

    # Success vs failure for per-trial data
    if sf_success or sf_failure:
        lines.append("")
        lines.append("SUCCESS VS FAILURE (per-trial entropy)")
        lines.append("-" * 70)
        for suite in SUITE_ORDER:
            s_ent = sf_success.get(suite, [])
            f_ent = sf_failure.get(suite, [])
            if not s_ent and not f_ent:
                continue
            lines.append(f"\n{SUITE_LABELS.get(suite, suite)}:")
            if s_ent:
                lines.append(f"  Success (n={len(s_ent)}): mean={np.mean(s_ent):.4f} +/- {np.std(s_ent):.4f}")
            if f_ent:
                lines.append(f"  Failure (n={len(f_ent)}): mean={np.mean(f_ent):.4f} +/- {np.std(f_ent):.4f}")
            if s_ent and f_ent:
                diff = np.mean(f_ent) - np.mean(s_ent)
                direction = "higher" if diff > 0 else "lower"
                lines.append(f"  Difference (F-S): {diff:.4f} (failures have {direction} entropy)")

                if len(s_ent) > 1 and len(f_ent) > 1:
                    from scipy.stats import ttest_ind
                    t, p = ttest_ind(s_ent, f_ent)
                    lines.append(f"  t-test: t={t:.3f}, p={p:.6f}")

    lines.append("")

    # Key findings
    lines.append("")
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)

    # Sort suites by entropy
    sorted_suites = sorted(
        [(s, suite_stats[s]["mean"]) for s in SUITE_ORDER if s in suite_stats],
        key=lambda x: x[1]
    )
    lines.append(f"Entropy ranking (lowest to highest):")
    for s, m in sorted_suites:
        lines.append(f"  {SUITE_LABELS.get(s, s)}: {m:.4f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Lower (more negative) entropy = more concentrated/deterministic action distribution")
    lines.append("  Higher (less negative) entropy = more spread out/diverse action distribution")

    report = "\n".join(lines)
    print(report)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(report)
    print(f"\nSaved: {output_dir / 'summary.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Cosmos policy action entropy")
    parser.add_argument(
        "--data-dir", type=str,
        default="third_party/cosmos-policy/cosmos_policy/experiments/robot/libero/logs/action_data",
        help="Directory with action JSON files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/entropy/cosmos",
        help="Output directory for plots and summary"
    )
    parser.add_argument(
        "--skip-per-trial", action="store_true",
        help="Skip per-trial entropy computation (faster)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-computed task-level entropy
    print("Loading pre-computed task-level entropy...")
    suite_data = load_precomputed_entropy(args.data_dir)
    print(f"Found suites: {list(suite_data.keys())}")

    # Compute suite-level stats
    suite_stats = {}
    for suite, items in suite_data.items():
        entropies = [d["action_entropy"]["action_entropy_kde"]
                     for d in items if "action_entropy" in d]
        n_trials = sum(d.get("num_trials", 0) for d in items)
        n_success = sum(d.get("num_successes", 0) for d in items)
        suite_stats[suite] = {
            "mean": np.mean(entropies),
            "std": np.std(entropies),
            "min": np.min(entropies),
            "max": np.max(entropies),
            "n_tasks": len(items),
            "n_trials": n_trials,
            "success_rate": n_success / max(n_trials, 1) * 100,
        }

    # 2. Plots
    print("\nGenerating cross-suite entropy plot...")
    plot_cross_suite_entropy(suite_stats, output_dir)

    print("Generating per-task entropy box plot...")
    plot_per_task_entropy(suite_data, output_dir)

    print("Generating entropy vs success rate scatter...")
    plot_entropy_vs_success_rate(suite_data, output_dir)

    # 3. Per-trial success vs failure analysis
    sf_success, sf_failure = {}, {}
    if not args.skip_per_trial:
        print("\nLoading per-trial data...")
        trial_data = load_per_trial_data(args.data_dir)
        print(f"Per-trial suites: {list(trial_data.keys())} ({sum(len(v) for v in trial_data.values())} trials)")

        print("Computing success vs failure entropy...")
        sf_success, sf_failure = plot_success_vs_failure(trial_data, output_dir)

    # 4. Summary
    print("\n")
    generate_summary(suite_data, suite_stats, sf_success, sf_failure, output_dir)

    # 5. Save combined results as JSON
    results = {
        "suite_stats": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                             for kk, vv in v.items()}
                        for k, v in suite_stats.items()},
    }
    with open(output_dir / "cosmos_entropy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'cosmos_entropy_results.json'}")


if __name__ == "__main__":
    main()
