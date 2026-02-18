#!/usr/bin/env python3
"""
Parse attention IoU results JSON files and compute stats over combined_iou.

Supports JSON formats:
  A) Old: [ {trajectory}, {trajectory}, ... ]
  B) New: { "metric": "...", "results": [ {trajectory}, ... ], ... }

Features:
- Optional layer filtering (--layer)
- Optional step-fraction filtering using per_step_iou (--step-fraction LO HI)
- Per-episode aggregation over filtered steps: mean or max (--step-agg)
- Processes one file or all files in a directory (--all / --directory)
- Optionally plots mean IoU vs success rate across suites (--plot)

Notes on step-fraction:
- Uses the *observed step indices* within each per_step_iou list to define the
  fraction window (robust to sparse / non-0..T step ids).
- If per_step_iou missing or empty, falls back to summary combined_iou mean.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# JSON normalization utilities
# -----------------------------
def _get_trajectories(raw: Any) -> list:
    """Normalize supported JSON layouts into a list of trajectory dicts."""
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        # New format: {"metric": "...", "results": [...]}
        if "results" in raw and isinstance(raw["results"], list):
            return raw["results"]

    raise ValueError(
        f"Unrecognized JSON format. Top-level type={type(raw)} "
        f"keys={list(raw)[:10] if isinstance(raw, dict) else None}"
    )


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _fraction_window_from_observed_steps(
    step_ids: list, frac: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Compute [lo, hi] window using observed step indices.
    Example: steps are [10, 26, 40, 70], frac (0.6, 1.0) => window near late steps.
    """
    lo_frac, hi_frac = frac
    if not step_ids:
        return (float("inf"), float("-inf"))

    mn, mx = min(step_ids), max(step_ids)
    if mx == mn:
        # Degenerate: only one step observed
        return (mn, mx)

    lo = mn + lo_frac * (mx - mn)
    hi = mn + hi_frac * (mx - mn)
    return (lo, hi)


# -----------------------------
# Core parsing
# -----------------------------
def parse_iou_results(
    json_path: str,
    layer: Optional[str] = None,
    step_fraction: Optional[Tuple[float, float]] = None,
    step_agg: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """
    Parse IOU results and compute per-layer statistics.

    Returns dict:
      {layer_name: {
          num_trajectories, mean_combined_iou, std_combined_iou,
          min_combined_iou, max_combined_iou
      }}
    """
    if step_agg not in ("mean", "max"):
        raise ValueError(f"step_agg must be 'mean' or 'max', got {step_agg}")
    agg_fn = np.max if step_agg == "max" else np.mean

    with open(json_path, "r") as f:
        raw = json.load(f)

    trajectories = _get_trajectories(raw)

    layer_ious = defaultdict(list)

    for traj in trajectories:
        if not isinstance(traj, dict):
            continue

        per_step = traj.get("per_step_iou", {}) or {}
        summary = traj.get("summary", {}) or {}

        # Prefer per_step_iou when step_fraction requested AND per_step exists
        if step_fraction is not None and isinstance(per_step, dict) and per_step:
            lo_frac, hi_frac = step_fraction

            for layer_name, steps in per_step.items():
                if layer is not None and layer_name != layer:
                    continue
                if not isinstance(steps, list) or not steps:
                    continue

                step_ids = [s.get("step") for s in steps if isinstance(s, dict) and "step" in s]
                step_ids = [sid for sid in step_ids if isinstance(sid, (int, float))]
                lo_step, hi_step = _fraction_window_from_observed_steps(step_ids, (lo_frac, hi_frac))

                filtered = []
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    sid = s.get("step", None)
                    ciou = s.get("combined_iou", None)
                    if sid is None or ciou is None:
                        continue
                    if not isinstance(sid, (int, float)):
                        continue
                    if lo_step <= sid <= hi_step:
                        fciou = _safe_float(ciou, default=None)
                        if fciou is not None:
                            filtered.append(fciou)

                if filtered:
                    layer_ious[layer_name].append(float(agg_fn(filtered)))

        # Fallback: summary combined_iou mean (no step filtering possible)
        else:
            if isinstance(summary, dict) and summary:
                for layer_name, layer_data in summary.items():
                    if layer is not None and layer_name != layer:
                        continue
                    if not isinstance(layer_data, dict):
                        continue
                    ciou = layer_data.get("combined_iou", {})
                    if isinstance(ciou, dict) and "mean" in ciou:
                        m = _safe_float(ciou["mean"], default=None)
                        if m is not None:
                            layer_ious[layer_name].append(m)

    # Compute stats
    results = {}
    for layer_name, ious in layer_ious.items():
        if not ious:
            continue
        arr = np.asarray(ious, dtype=np.float64)
        results[layer_name] = {
            "num_trajectories": int(arr.size),
            "mean_combined_iou": float(arr.mean()),
            "std_combined_iou": float(arr.std()),
            "min_combined_iou": float(arr.min()),
            "max_combined_iou": float(arr.max()),
        }

    return results


def parse_all_files(
    directory: str,
    step_fraction: Optional[Tuple[float, float]] = None,
    step_agg: str = "mean",
) -> Dict[str, Dict]:
    """Parse all iou_results*.json files in a directory."""
    results_dir = Path(directory)
    json_files = sorted(results_dir.glob("iou_results*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return {}

    all_results = {}
    for jf in json_files:
        try:
            all_results[jf.name] = parse_iou_results(
                str(jf), step_fraction=step_fraction, step_agg=step_agg
            )
        except Exception as e:
            print(f"[WARN] Failed parsing {jf.name}: {e}")
            all_results[jf.name] = {}

    return all_results


def get_success_rate(json_path: str) -> float:
    """Compute success rate from an iou_results JSON file."""
    with open(json_path, "r") as f:
        raw = json.load(f)
    trajectories = _get_trajectories(raw)
    if not trajectories:
        return 0.0
    successes = sum(1 for t in trajectories if isinstance(t, dict) and t.get("success"))
    return successes / len(trajectories)


def _suite_name_from_filename(filename: str) -> str:
    """Extract suite name like 'libero_spatial' from 'iou_results_libero_spatial.json'."""
    stem = Path(filename).stem
    return stem.replace("iou_results_", "")


# -----------------------------
# Plotting
# -----------------------------
def plot_iou_vs_success(
    directory: str,
    layer: Optional[str],
    step_fraction: Optional[Tuple[float, float]] = None,
    step_agg: str = "mean",
    output_path: str = "iou_vs_success.png",
):
    """Scatter plot of mean IoU vs success rate across task suites."""
    results_dir = Path(directory)
    json_files = sorted(results_dir.glob("iou_results*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    suites, ious, stds, success_rates = [], [], [], []

    for jf in json_files:
        stats = parse_iou_results(
            str(jf), layer=layer, step_fraction=step_fraction, step_agg=step_agg
        )
        sr = get_success_rate(str(jf))
        suite = _suite_name_from_filename(jf.name)

        if not stats:
            continue

        if layer and layer in stats:
            s = stats[layer]
        else:
            # pick best layer by mean IoU
            s = max(stats.values(), key=lambda d: d["mean_combined_iou"])

        suites.append(suite)
        ious.append(s["mean_combined_iou"])
        stds.append(s["std_combined_iou"])
        success_rates.append(sr)

    if not suites:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        success_rates,
        ious,
        yerr=stds,
        fmt="o",
        capsize=4,
        markersize=8,
        color="steelblue",
    )

    for i, suite in enumerate(suites):
        ax.annotate(
            suite,
            (success_rates[i], ious[i]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xlabel("Success Rate", fontsize=12)
    ax.set_ylabel(f"Mean Combined IoU ({step_agg}/ep)", fontsize=12)

    title_parts = ["Attention IoU vs Success Rate (Cosmos)"]
    if layer:
        title_parts.append(f"[{layer}]")
    if step_fraction:
        title_parts.append(f"steps [{step_fraction[0]:.0%}-{step_fraction[1]:.0%}]")
    ax.set_title("  ".join(title_parts), fontsize=11)

    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Parse IOU results and compute average combined_iou"
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        help="Path to iou_results JSON file (if not provided, processes all files in --directory)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="outputs_iou_cosmos/test",
        help="Directory containing JSON files (default: outputs_iou_cosmos/test)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer to analyze (default: all layers)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all iou_results*.json files in the directory",
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Filter to steps in [LO, HI] fraction of observed step-index range (per episode). "
             "Example: 0.6 1.0 for late steps.",
    )
    parser.add_argument(
        "--step-agg",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to aggregate per-step IoU within each episode (default: mean)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="OUTPUT_PATH",
        help="Plot IoU vs success rate across suites and save to this path (e.g. iou_vs_success.png)",
    )

    args = parser.parse_args()
    step_fraction = tuple(args.step_fraction) if args.step_fraction else None

    # Process all files (default when json_file omitted)
    if args.json_file is None or args.all:
        all_results = parse_all_files(
            args.directory, step_fraction=step_fraction, step_agg=args.step_agg
        )
        if not all_results:
            return 1

        if step_fraction:
            print(
                f"\nStep fraction filter: [{step_fraction[0]:.0%}, {step_fraction[1]:.0%}] "
                f"of observed step-index range, per-episode agg: {args.step_agg}"
            )
            print("(Falls back to full-episode summary when per_step_iou unavailable)")

        for filename, layer_results in all_results.items():
            print(f"\n{'='*80}")
            print(f"File: {filename}")
            print(f"{'='*80}")

            if not layer_results:
                print("  No IoU data found (or failed to parse).")
                continue

            for layer_name in sorted(layer_results.keys()):
                s = layer_results[layer_name]
                print(f"\n{layer_name}:")
                print(f"  Number of trajectories: {s['num_trajectories']}")
                print(f"  Mean combined_iou:      {s['mean_combined_iou']:.6f}")
                print(f"  Std combined_iou:       {s['std_combined_iou']:.6f}")
                print(f"  Min combined_iou:       {s['min_combined_iou']:.6f}")
                print(f"  Max combined_iou:       {s['max_combined_iou']:.6f}")

            # Overall average across layers (unweighted mean of layer means)
            all_means = [s["mean_combined_iou"] for s in layer_results.values()]
            if all_means:
                print(f"\n{'─'*80}")
                print("OVERALL (average across all layers' means):")
                print(f"  Average combined_iou:   {np.mean(all_means):.6f}")
                print(f"  Std dev:                {np.std(all_means):.6f}")
                print(f"  Min:                    {np.min(all_means):.6f}")
                print(f"  Max:                    {np.max(all_means):.6f}")

        if args.plot:
            plot_iou_vs_success(
                directory=args.directory,
                layer=args.layer,
                step_fraction=step_fraction,
                step_agg=args.step_agg,
                output_path=args.plot,
            )

    # Process single file
    else:
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"Error: File not found: {args.json_file}")
            return 1

        results = parse_iou_results(
            str(json_path),
            layer=args.layer,
            step_fraction=step_fraction,
            step_agg=args.step_agg,
        )

        if not results:
            print("No IoU data found in this file.")
            return 1

        if args.layer and args.layer in results:
            s = results[args.layer]
            print(f"File: {args.json_file}")
            print(f"Layer: {args.layer}")
            print("-" * 50)
            print(f"Number of trajectories: {s['num_trajectories']}")
            print(f"Mean combined_iou: {s['mean_combined_iou']:.6f}")
            print(f"Std combined_iou:  {s['std_combined_iou']:.6f}")
            print(f"Min combined_iou:  {s['min_combined_iou']:.6f}")
            print(f"Max combined_iou:  {s['max_combined_iou']:.6f}")
        else:
            print(f"File: {args.json_file}")
            print("-" * 80)
            for layer_name in sorted(results.keys()):
                s = results[layer_name]
                print(f"\n{layer_name}:")
                print(f"  Number of trajectories: {s['num_trajectories']}")
                print(f"  Mean combined_iou:      {s['mean_combined_iou']:.6f}")
                print(f"  Std combined_iou:       {s['std_combined_iou']:.6f}")
                print(f"  Min combined_iou:       {s['min_combined_iou']:.6f}")
                print(f"  Max combined_iou:       {s['max_combined_iou']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())