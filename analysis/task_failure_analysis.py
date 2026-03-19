"""
Find tasks that each model consistently fails on, broken down by LIBERO-90 subset.

Usage:
    python analysis/task_failure_analysis.py
    python analysis/task_failure_analysis.py --threshold 0.1   # show tasks with <10% success
    python analysis/task_failure_analysis.py --perturb none     # which perturbation condition to use

Reads per-episode JSON files from data/libero/{model}/{...}/{suite}/*.json
Each JSON must have: task_description (str), success (bool)
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data roots ────────────────────────────────────────────────────────────────
# Each entry: model_name → list of (suite, path_glob) pairs
# The glob should match all per-episode JSON files for that model/suite.

BASE = Path("data/libero")

# Models and where their baseline (no-perturbation) per-episode JSONs live.
# Suites we care about for "libero_90" subsets.
LIBERO_90_SUITES = ["libero_90_obj", "libero_90_spa", "libero_90_act", "libero_90_com"]
SUITE_LABELS = {
    "libero_90_obj": "Object",
    "libero_90_spa": "Spatial",
    "libero_90_act": "Action",
    "libero_90_com": "Composite",
}
SUITE_COLORS = {
    "libero_90_obj": "#ff7f0e",
    "libero_90_spa": "#2ca02c",
    "libero_90_act": "#d62728",
    "libero_90_com": "#9467bd",
}

MODEL_ROOTS = {
    "pi0.5": BASE / "pi05" / "policy_perturb" / "none",
    "dp":    BASE / "dp" / "videos",
}
# cosmos and dreamzero don't have libero_90 baselines in the same structure;
# you can add them here if you have a "none" baseline dir:
# "cosmos":    BASE / "cosmos" / "policy_perturb" / "none",  # uncomment if available

MODEL_DISPLAY = {
    "pi0.5": "π0.5 (pi0-fast)",
    "dp":    "Diffusion Policy",
    "cosmos": "Cosmos",
    "dreamzero": "DreamZero",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_filename(fpath: Path) -> tuple[str | None, bool | None]:
    """
    Fast extraction from filename like:
    actions_{task_underscored}_trial{N}_{success|failure}.json
    Returns (task_description, success) or (None, None) on failure.
    """
    stem = fpath.stem  # strip .json
    if stem.endswith("_success"):
        success = True
        stem = stem[: -len("_success")]
    elif stem.endswith("_failure"):
        success = False
        stem = stem[: -len("_failure")]
    else:
        return None, None
    # strip _trial{N}
    import re
    stem = re.sub(r"_trial\d+$", "", stem)
    # strip leading "actions_"
    if stem.startswith("actions_"):
        stem = stem[len("actions_"):]
    task = stem.replace("_", " ")
    return task, success


def load_suite_results(root: Path, suite: str) -> dict[str, list[bool]]:
    """
    Load all episode JSONs under root/suite/ and return
    {task_description: [success, success, ...]}

    Uses filename parsing for speed (avoids loading large JSON bodies).
    Falls back to JSON for files without the standard naming convention.
    """
    suite_dir = root / suite
    if not suite_dir.exists():
        return {}
    results: dict[str, list[bool]] = defaultdict(list)
    for fpath in suite_dir.glob("*.json"):
        task, success = _parse_filename(fpath)
        if task is not None and success is not None:
            results[task].append(success)
            continue
        # Fallback: read JSON
        try:
            with open(fpath) as f:
                ep = json.load(f)
        except Exception:
            continue
        task = ep.get("task_description")
        success = ep.get("success")
        if task is None or success is None:
            continue
        results[task].append(bool(success))
    return dict(results)


def compute_task_stats(
    results: dict[str, list[bool]]
) -> dict[str, dict]:
    """Return {task: {n_trials, n_success, success_rate}} sorted by success_rate."""
    stats = {}
    for task, outcomes in results.items():
        n = len(outcomes)
        s = sum(outcomes)
        stats[task] = {"n_trials": n, "n_success": s, "success_rate": s / n if n > 0 else 0.0}
    return dict(sorted(stats.items(), key=lambda x: x[1]["success_rate"]))


# ── Main analysis ─────────────────────────────────────────────────────────────

def collect_all(model_roots: dict[str, Path], suites: list[str]) -> dict:
    """
    Returns nested dict: data[model][suite][task] = {n_trials, n_success, success_rate}
    """
    data = {}
    for model, root in model_roots.items():
        data[model] = {}
        for suite in suites:
            raw = load_suite_results(root, suite)
            if raw:
                data[model][suite] = compute_task_stats(raw)
            else:
                print(f"  [warn] no data for {model}/{suite} at {root/suite}")
    return data


def find_failing_tasks(data: dict, threshold: float = 0.0) -> dict:
    """
    Returns {model: {suite: [tasks where success_rate <= threshold]}}
    """
    failing = {}
    for model, suites in data.items():
        failing[model] = {}
        for suite, tasks in suites.items():
            bad = [t for t, s in tasks.items() if s["success_rate"] <= threshold]
            failing[model][suite] = bad
    return failing


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(data: dict, threshold: float, out_path: str):
    """
    Heatmap: rows = tasks (all tasks with success_rate <= threshold in ANY model/suite),
             columns = models, coloured by suite.
    """
    # Gather union of failing tasks per (model, suite)
    models = list(data.keys())
    suites = LIBERO_90_SUITES

    # Build a set of (suite, task) pairs that fail in at least one model
    failing_pairs: set[tuple[str, str]] = set()
    for model in models:
        for suite in suites:
            if suite not in data[model]:
                continue
            for task, stats in data[model][suite].items():
                if stats["success_rate"] <= threshold:
                    failing_pairs.add((suite, task))

    if not failing_pairs:
        print(f"No tasks with success_rate <= {threshold:.0%} found.")
        return

    # Sort: by suite order, then by task name
    ordered = sorted(failing_pairs, key=lambda x: (suites.index(x[0]), x[1]))
    n_tasks = len(ordered)
    n_models = len(models)

    # Build matrix: rows=tasks, cols=models; value = success_rate (NaN if no data)
    mat = np.full((n_tasks, n_models), np.nan)
    for i, (suite, task) in enumerate(ordered):
        for j, model in enumerate(models):
            if suite in data[model] and task in data[model][suite]:
                mat[i, j] = data[model][suite][task]["success_rate"]

    # ── Figure ────────────────────────────────────────────────────────────────
    row_height = 0.35
    col_width = 1.4
    label_width = 5.5
    fig_w = label_width + n_models * col_width + 1.5
    fig_h = max(4, n_tasks * row_height + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Heatmap
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#cccccc")  # NaN → grey
    im = ax.imshow(
        mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in models],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_yticks(range(n_tasks))

    # Task labels coloured by suite
    ylabels = []
    for suite, task in ordered:
        short = task[:60] + "…" if len(task) > 60 else task
        ylabels.append(short)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    for tick, (suite, _) in zip(ax.get_yticklabels(), ordered):
        tick.set_color(SUITE_COLORS[suite])

    # Suite colour bars on the left
    for i, (suite, _) in enumerate(ordered):
        ax.add_patch(plt.Rectangle(
            (-0.5 - 0.15, i - 0.5), 0.12, 1.0,
            color=SUITE_COLORS[suite], transform=ax.transData,
            clip_on=False,
        ))

    # Annotate cells with success rate (skip NaN)
    for i in range(n_tasks):
        for j in range(n_models):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color="black" if 0.2 < v < 0.8 else "white",
                        fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Success rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Suite legend
    legend_handles = [
        mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_LABELS[s])
        for s in suites if any(suite == s for suite, _ in ordered)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, -0.08), ncol=len(legend_handles),
              fontsize=8, title="Subset", title_fontsize=8,
              framealpha=0.8)

    threshold_pct = int(threshold * 100)
    ax.set_title(
        f"Tasks with ≤{threshold_pct}% success rate — by model & LIBERO-90 subset\n"
        f"(grey = no data for that model/suite combination)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Model", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_per_model_bars(data: dict, threshold: float, out_path: str):
    """
    For each model, a grouped bar chart: x = suite, y = # failing tasks,
    with a stacked view showing tasks below threshold.
    """
    models = list(data.keys())
    suites = LIBERO_90_SUITES
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        counts_total = []
        counts_fail = []
        for suite in suites:
            if suite not in data[model]:
                counts_total.append(0)
                counts_fail.append(0)
                continue
            tasks = data[model][suite]
            counts_total.append(len(tasks))
            counts_fail.append(sum(1 for s in tasks.values() if s["success_rate"] <= threshold))

        x = np.arange(len(suites))
        bar_w = 0.6
        ax.bar(x, counts_total, bar_w, color=[SUITE_COLORS[s] for s in suites],
               alpha=0.3, label="Total tasks")
        ax.bar(x, counts_fail, bar_w, color=[SUITE_COLORS[s] for s in suites],
               alpha=0.9, label=f"Always fail (≤{int(threshold*100)}%)")

        # Annotate counts
        for xi, (ct, cf) in enumerate(zip(counts_total, counts_fail)):
            if cf > 0:
                ax.text(xi, cf + 0.3, str(cf), ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([SUITE_LABELS[s] for s in suites], rotation=20, ha="right")
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11, fontweight="bold")
        ax.set_ylabel("# tasks" if ax == axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(max(counts_total, default=1) + 2, 5))

    fig.suptitle(
        f"Number of consistently failing tasks (≤{int(threshold*100)}% success) per model & subset",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def print_summary(data: dict, threshold: float):
    """Print a text summary of failing tasks."""
    for model in data:
        print(f"\n{'='*70}")
        print(f"  Model: {MODEL_DISPLAY.get(model, model)}")
        print(f"{'='*70}")
        for suite in LIBERO_90_SUITES:
            if suite not in data[model]:
                print(f"  [{SUITE_LABELS[suite]}]  — no data")
                continue
            tasks = data[model][suite]
            n_total = len(tasks)
            failing = [(t, s) for t, s in tasks.items() if s["success_rate"] <= threshold]
            print(f"\n  [{SUITE_LABELS[suite]}]  {len(failing)}/{n_total} tasks fail ≤{int(threshold*100)}%:")
            for task, stats in sorted(failing, key=lambda x: x[1]["success_rate"]):
                print(f"    {stats['success_rate']:5.0%}  ({stats['n_success']}/{stats['n_trials']})  {task}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Tasks with success_rate <= threshold are 'failing' (default: 0.0 = always fail)")
    parser.add_argument("--out-dir", default="results/task_failures",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    threshold = args.threshold

    print(f"Loading data (threshold={threshold:.0%}) ...")
    data = collect_all(MODEL_ROOTS, LIBERO_90_SUITES)

    # Print text summary
    print_summary(data, threshold)

    # Plot heatmap of failing tasks
    thr_tag = f"leq{int(threshold*100)}pct"
    plot_heatmap(data, threshold,
                 out_path=f"{args.out_dir}/failing_tasks_heatmap_{thr_tag}.png")

    # Plot bar chart summary
    plot_per_model_bars(data, threshold,
                        out_path=f"{args.out_dir}/failing_tasks_bars_{thr_tag}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
