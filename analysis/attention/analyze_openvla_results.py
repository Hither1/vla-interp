#!/usr/bin/env python3
"""
Analyze all OpenVLA attention results in results/attention_openvla/.

Handles the two-level nested structure:
  results/attention_openvla/{ratio,iou}_openvla_*/<condition_name>/*.json

For ratio folders: calls parse_attention_ratio_results logic
For iou folders:   calls parse_iou_results logic

Usage:
  cd /path/to/vla-interp
  python analysis/attention/analyze_openvla_results.py \
      --results-dir results/attention_openvla \
      --output-dir results/attention_openvla_analysis \
      --plot-all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── import sibling modules ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from parse_attention_ratio_results import (
    compute_aggregate_stats,
    print_summary,
    plot_ratio_distribution,
    plot_per_task_comparison,
    plot_attention_fractions,
)
from parse_iou_results import parse_iou_results


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _filter_steps(entries: list, step_fraction: tuple[float, float] | None) -> list:
    """Return entries with per_step_ratios filtered to [lo, hi] fraction of each episode's step range.

    Uses the observed step indices within each episode (robust to sparse steps).
    Episodes with no steps in the window are kept with empty per_step_ratios.
    If step_fraction is None, returns entries unchanged.
    """
    if step_fraction is None:
        return entries

    lo_frac, hi_frac = step_fraction
    filtered_entries = []
    for ep in entries:
        per_step = ep.get("per_step_ratios", {})
        if not per_step:
            filtered_entries.append(ep)
            continue

        new_per_step = {}
        for layer, steps in per_step.items():
            if not steps:
                new_per_step[layer] = []
                continue
            step_ids = [s["step"] for s in steps if "step" in s]
            if not step_ids:
                new_per_step[layer] = []
                continue
            mn, mx = min(step_ids), max(step_ids)
            span = mx - mn if mx != mn else 1
            lo_step = mn + lo_frac * span
            hi_step = mn + hi_frac * span
            new_per_step[layer] = [s for s in steps if lo_step <= s.get("step", -1) <= hi_step]

        new_ep = dict(ep)
        new_ep["per_step_ratios"] = new_per_step
        filtered_entries.append(new_ep)

    return filtered_entries


def _condition_name(subdir: Path) -> str:
    """Extract condition label from the subdir name."""
    name = subdir.name
    # e.g. libero_10_seed7_prompt_shuffle_perturb_prompt_shuffle
    # strip leading suite+seed prefix up to first 'prompt_' occurrence
    idx = name.find("prompt_")
    return name[idx:] if idx != -1 else name


# ── ratio analysis ────────────────────────────────────────────────────────────

def _collect_ratio_conditions(top_folders: list[Path]) -> dict[str, list]:
    """Collect all ratio episodes across all top-level folders, grouped by condition."""
    conditions: dict[str, list] = {}
    for top_folder in top_folders:
        for jf in sorted(top_folder.rglob("attention_ratio_results_*.json")):
            cond = _condition_name(jf.parent)
            data = _load_json(jf)
            if isinstance(data, dict) and "results" in data:
                entries = data["results"]
            elif isinstance(data, list):
                entries = data
            else:
                entries = [data]
            conditions.setdefault(cond, []).extend(entries)
    return conditions


def analyze_ratio_folders(
    top_folders: list[Path],
    output_dir: Path,
    plot_all: bool,
    step_fraction: tuple[float, float] | None = None,
):
    """Analyze all ratio folders together, grouped by condition across all suites/tasks."""
    conditions = _collect_ratio_conditions(top_folders)
    if not conditions:
        print("  [skip] no ratio JSONs found")
        return

    # Apply step-fraction filter to each episode's per_step_ratios
    filtered_conditions = {
        cond: _filter_steps(entries, step_fraction)
        for cond, entries in conditions.items()
    }

    all_entries = [e for entries in filtered_conditions.values() for e in entries]

    sf_label = f" [steps {step_fraction[0]:.0%}–{step_fraction[1]:.0%}]" if step_fraction else ""
    print(f"\n{'#'*80}")
    print(f"# RATIO ANALYSIS: all suites combined ({len(all_entries)} episodes){sf_label}")
    print(f"{'#'*80}")

    # Combined summary across everything
    combined = compute_aggregate_stats(all_entries)
    print_summary(combined, f"ALL SUITES COMBINED{sf_label}")

    # Per-condition table (each row = all episodes+tasks for that condition)
    _print_ratio_condition_table(filtered_conditions, f"all suites{sf_label}")

    if plot_all and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_ratio_distribution(all_entries, str(output_dir / "ratio_distribution_all.png"))
        plot_per_task_comparison(all_entries, str(output_dir / "ratio_per_task_all.png"))
        plot_attention_fractions(all_entries, str(output_dir / "attention_fractions_all.png"))


def _print_ratio_condition_table(conditions: dict, label: str):
    print(f"\n{'─'*80}")
    print(f"CONDITION COMPARISON ({label})")
    print(f"{'─'*80}")
    header = f"{'Condition':<55} {'Episodes':>8} {'Tasks':>6} {'Success%':>9} {'VL_Ratio':>10} {'Vis%':>7} {'Ling%':>7}"
    print(header)
    print("─" * len(header))
    for cond, entries in sorted(conditions.items()):
        stats = compute_aggregate_stats(entries)
        sr = f"{stats['success_rate']:.0%}"
        ep = stats['num_episodes']
        ntasks = stats['num_tasks']
        ratio_mean = stats.get("visual_linguistic_ratio", {}).get("mean", float("nan"))
        vis = stats.get("visual_fraction", {}).get("mean", float("nan"))
        ling = stats.get("linguistic_fraction", {}).get("mean", float("nan"))
        print(f"{cond:<55} {ep:>8} {ntasks:>6} {sr:>9} {ratio_mean:>10.3f} {vis:>6.1%} {ling:>6.1%}")
    print()


# ── iou analysis ──────────────────────────────────────────────────────────────

def analyze_iou_folders(
    top_folders: list[Path],
    output_dir: Path,
    plot_all: bool,
    step_fraction: tuple[float, float] | None = None,
):
    """Analyze all iou folders together, grouped by condition across all suites/tasks."""
    cond_layer_ious: dict[str, dict[str, list]] = {}
    found = False
    for top_folder in top_folders:
        for jf in sorted(top_folder.rglob("iou_results_*.json")):
            found = True
            cond = _condition_name(jf.parent)
            layer_stats = parse_iou_results(str(jf), step_fraction=step_fraction)
            for layer, stats in layer_stats.items():
                cond_layer_ious.setdefault(cond, {}).setdefault(layer, []).append(
                    stats["mean_combined_iou"]
                )

    if not found:
        print("  [skip] no iou JSONs found")
        return

    sf_label = f" [steps {step_fraction[0]:.0%}–{step_fraction[1]:.0%}]" if step_fraction else ""
    print(f"\n{'#'*80}")
    print(f"# IOU ANALYSIS: all suites combined{sf_label}")
    print(f"{'#'*80}")

    _print_iou_condition_table(cond_layer_ious, f"all suites{sf_label}")

    if plot_all and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_iou_conditions(cond_layer_ious, "all_suites", output_dir)


def _print_iou_condition_table(
    cond_layer_ious: dict[str, dict[str, list]], folder_name: str
):
    print(f"\n{'─'*80}")
    print(f"CONDITION COMPARISON: {folder_name}")
    print(f"{'─'*80}")

    # Collect all layers across conditions for aligned columns
    all_layers = sorted({l for cv in cond_layer_ious.values() for l in cv})
    col_w = 12
    header = f"{'Condition':<55}" + "".join(f"{l[-8:]:>{col_w}}" for l in all_layers)
    print(header)
    print("─" * len(header))

    for cond, layer_ious in sorted(cond_layer_ious.items()):
        row = f"{cond:<55}"
        for l in all_layers:
            vals = layer_ious.get(l, [])
            if vals:
                row += f"{np.mean(vals):>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)
    print()


def _plot_iou_conditions(
    cond_layer_ious: dict[str, dict[str, list]], folder_name: str, output_dir: Path
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot: one bar per condition, averaged across layers
    conds = sorted(cond_layer_ious)
    means = []
    stds = []
    for cond in conds:
        all_vals = [v for vals in cond_layer_ious[cond].values() for v in vals]
        means.append(np.mean(all_vals) if all_vals else 0)
        stds.append(np.std(all_vals) if all_vals else 0)

    fig, ax = plt.subplots(figsize=(max(10, len(conds) * 1.2), 5))
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([c[7:] if len(c) > 7 else c for c in conds], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Combined IoU")
    ax.set_title(f"IoU by Condition: {folder_name}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = str(output_dir / f"{folder_name}_iou_by_condition.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze all OpenVLA attention results")
    parser.add_argument(
        "--results-dir",
        default="results/attention_openvla",
        help="Top-level results directory (default: results/attention_openvla)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/attention_openvla_analysis",
        help="Directory for output plots/summaries",
    )
    parser.add_argument("--plot-all", action="store_true", help="Generate all plots")
    parser.add_argument(
        "--only",
        choices=["ratio", "iou"],
        default=None,
        help="Analyze only ratio or iou results",
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Only use steps in [LO, HI] fraction of each episode's observed step range. "
             "Example: --step-fraction 0.5 1.0 for the second half of each episode.",
    )
    args = parser.parse_args()
    step_fraction = tuple(args.step_fraction) if args.step_fraction else None

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.plot_all else None

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    top_folders = sorted(results_dir.iterdir())
    ratio_folders = [f for f in top_folders if f.is_dir() and f.name.startswith("ratio_")]
    iou_folders = [f for f in top_folders if f.is_dir() and f.name.startswith("iou_")]

    if args.only != "iou":
        analyze_ratio_folders(ratio_folders, output_dir, args.plot_all, step_fraction)

    if args.only != "ratio":
        analyze_iou_folders(iou_folders, output_dir, args.plot_all, step_fraction)


if __name__ == "__main__":
    main()
