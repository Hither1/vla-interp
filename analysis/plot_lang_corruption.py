"""Plot language corruption experiment results.

Reads JSON output from evaluate_lang_corruption_ratio.py and/or
evaluate_lang_corruption_ratio_dreamzero.py and produces:

  Figure 1: VCI bar chart (main result)
  Figure 2: Success rate bar chart
  Figure 3: Temporal analysis (visual_fraction over episode phases)
  Figure 4: Scatter VCI vs delta-success

Usage:
  python analysis/plot_lang_corruption.py \\
    --pi05-results results/lang_corruption_ratio/lang_corruption_ratio_libero_10.json \\
    --dreamzero-results results/lang_corruption_ratio_dreamzero/lang_corruption_ratio_libero_10.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Style constants (consistent with plot_generalization.py)
CORRUPTION_MODES = ["empty", "opposite", "random", "shuffle", "synonym"]
ALL_MODES = ["empty", "opposite", "random", "shuffle", "synonym"]

# Colors for models
MODEL_COLORS = {
    "pi0.5": "#4C72B0",
    "DreamZero": "#17becf",
}

# Colors for corruption modes
MODE_COLORS = {
    "empty":    "#e41a1c",
    "shuffle":  "#377eb8",
    "random":   "#ff7f00",
    "synonym":  "#4daf4a",
    "opposite": "#984ea3",
}

MODE_LINESTYLES = {
    "empty":    "-",
    "shuffle":  "--",
    "random":   "-.",
    "synonym":  ":",
    "opposite": (0, (3, 1, 1, 1)),
}

MODEL_MARKERS = {"pi0.5": "o", "DreamZero": "s"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load_results(path: str) -> List[Dict]:
    """Load JSON results from a lang corruption evaluation run."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def aggregate_results(results: List[Dict], modes: List[str] = None) -> Dict:
    """Aggregate per-episode results across episodes and tasks.

    Returns dict:
      mode -> {
        vci: list of floats (one per episode),
        delta_success: list of ints,
        success: list of bool (absolute success for this mode),
        temporal_slope: list of floats,
        temporal_early: list, temporal_mid: list, temporal_late: list,
        visual_fraction_mean: list,
      }
    """
    if modes is None:
        modes = ALL_MODES
    out: Dict = {m: {
        "vci": [], "delta_success": [], "success": [],
        "temporal_slope": [], "temporal_early": [], "temporal_mid": [], "temporal_late": [],
        "visual_fraction_mean": [], "linguistic_fraction_mean": [],
    } for m in modes}
    out["original"] = {
        "vci": [], "delta_success": [], "success": [],
        "temporal_slope": [], "temporal_early": [], "temporal_mid": [], "temporal_late": [],
        "visual_fraction_mean": [], "linguistic_fraction_mean": [],
    }

    for ep in results:
        modes_data = ep.get("modes", {})
        deltas = ep.get("deltas", {})

        # original mode data
        orig = modes_data.get("original", {})
        if orig:
            out["original"]["success"].append(bool(orig.get("success", False)))
            out["original"]["visual_fraction_mean"].append(float(orig.get("visual_fraction_mean", 0.0)))
            out["original"]["linguistic_fraction_mean"].append(float(orig.get("linguistic_fraction_mean", 0.0)))

        for m in modes:
            mode_data = modes_data.get(m, {})
            delta = deltas.get(m, {})

            if mode_data:
                out[m]["success"].append(bool(mode_data.get("success", False)))
                out[m]["visual_fraction_mean"].append(float(mode_data.get("visual_fraction_mean", 0.0)))
                out[m]["linguistic_fraction_mean"].append(float(mode_data.get("linguistic_fraction_mean", 0.0)))

            if delta:
                out[m]["vci"].append(float(delta.get("vci", 0.0)))
                out[m]["delta_success"].append(int(delta.get("delta_success", 0)))
                ta = delta.get("temporal_analysis", {})
                out[m]["temporal_slope"].append(float(ta.get("temporal_slope", 0.0)))
                out[m]["temporal_early"].append(float(ta.get("mode_early", 0.0)))
                out[m]["temporal_mid"].append(float(ta.get("mode_mid", 0.0)))
                out[m]["temporal_late"].append(float(ta.get("mode_late", 0.0)))

    return out


def _mean_sem(vals):
    """Return (mean, sem, n) for a list of values."""
    if not vals:
        return float("nan"), 0.0, 0
    a = np.array(vals, dtype=float)
    n = len(a)
    return float(np.mean(a)), float(np.std(a) / max(1, np.sqrt(n))), n


def _save_fig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_vci_bars(
    aggregated: Dict[str, Dict],  # model_name -> aggregate_results()
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Figure 1: VCI bar chart per corruption mode, grouped by model."""
    n_modes = len(modes)
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(n_modes)

    fig, ax = plt.subplots(figsize=(max(7, n_modes * 1.6), 4.5))

    for i, model_name in enumerate(model_names):
        agg = aggregated.get(model_name, {})
        if not agg:
            continue
        means, sems = [], []
        for m in modes:
            mu, sem, _ = _mean_sem(agg.get(m, {}).get("vci", []))
            means.append(mu)
            sems.append(sem)
        color = MODEL_COLORS.get(model_name, f"C{i}")
        offset = (i - n_models / 2.0 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, color=color,
               label=model_name, zorder=3,
               yerr=sems, capsize=4, error_kw={"zorder": 4})

    ax.axhline(0, color="black", lw=1.0, ls="-", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_xlabel("Corruption mode")
    ax.set_ylabel("VCI = Δ visual_fraction")
    ax.set_title("Visual Compensation Index under Language Corruption")
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig1_vci_bars.png")


def plot_success_bars(
    aggregated: Dict[str, Dict],
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Figure 2: Success rate bar chart (original + per-mode)."""
    # Include original as the first group
    all_conds = ["original"] + modes
    n_conds = len(all_conds)
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(n_conds)

    fig, ax = plt.subplots(figsize=(max(8, n_conds * 1.4), 4.5))

    for i, model_name in enumerate(model_names):
        agg = aggregated.get(model_name, {})
        if not agg: continue
        means, sems = [], []
        for cond in all_conds:
            vals = [100.0 * float(v) for v in agg.get(cond, {}).get("success", [])]
            mu, sem, _ = _mean_sem(vals)
            means.append(mu)
            sems.append(sem)
        color = MODEL_COLORS.get(model_name, f"C{i}")
        offset = (i - n_models / 2.0 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, color=color,
               label=model_name, zorder=3,
               yerr=sems, capsize=4, error_kw={"zorder": 4})

    ax.set_xticks(x)
    ax.set_xticklabels(all_conds, rotation=20, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate under Language Corruption")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig2_success_bars.png")


def plot_temporal_analysis(
    aggregated: Dict[str, Dict],
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Figure 3: Temporal analysis - visual_fraction over early/mid/late phases."""
    phases = ["early", "mid", "late"]
    x = np.arange(len(phases))

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), sharey=False)
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Visual Fraction over Episode Phases under Language Corruption",
                 fontsize=11, fontweight="bold")

    for ax, model_name in zip(axes, model_names):
        agg = aggregated.get(model_name, {})
        # Plot original first as a thick dashed reference
        orig = agg.get("original", {})
        orig_phases = []
        for phase in phases:
            mu, _, _ = _mean_sem(orig.get(f"temporal_{phase}", []))
            orig_phases.append(mu)
        if any(np.isfinite(v) for v in orig_phases):
            ax.plot(x, orig_phases, color="black", lw=2, ls="--",
                    marker="D", ms=6, label="original", zorder=5)

        for mode in modes:
            mode_agg = agg.get(mode, {})
            ys = []
            for phase in phases:
                mu, _, _ = _mean_sem(mode_agg.get(f"temporal_{phase}", []))
                ys.append(mu)
            color = MODE_COLORS.get(mode, "grey")
            ls = MODE_LINESTYLES.get(mode, "-")
            ax.plot(x, ys, color=color, lw=1.8, ls=ls,
                    marker="o", ms=5, label=mode, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_xlabel("Episode phase")
        ax.set_ylabel("visual_fraction (mean)")
        ax.set_title(model_name)
        ax.yaxis.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    _save_fig(fig, out_dir, "fig3_temporal_analysis.png")


def plot_vci_vs_delta_success(
    aggregated: Dict[str, Dict],
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Figure 4: Scatter VCI vs delta-success; color by mode, shape by model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in model_names:
        agg = aggregated.get(model_name, {})
        marker = MODEL_MARKERS.get(model_name, "o")
        for mode in modes:
            mode_agg = agg.get(mode, {})
            vcis = mode_agg.get("vci", [])
            delta_succ = mode_agg.get("delta_success", [])
            if not vcis: continue
            color = MODE_COLORS.get(mode, "grey")
            label = f"{model_name}/{mode}" if model_name == model_names[0] else None
            ax.scatter(
                vcis, [float(d) for d in delta_succ],
                c=color, marker=marker, alpha=0.6, s=50,
                zorder=4, label=label,
            )
            # Plot mean point larger
            if vcis:
                ax.scatter(
                    [np.mean(vcis)], [np.mean(delta_succ)],
                    c=color, marker=marker, s=180, zorder=5,
                    edgecolors="black", linewidths=0.8,
                )

    # Reference lines
    ax.axhline(0, color="grey", lw=1.0, ls=":", zorder=1)
    ax.axvline(0, color="grey", lw=1.0, ls=":", zorder=1)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = (xlim[0] + xlim[1]) / 2.0
    cy = (ylim[0] + ylim[1]) / 2.0
    quad_labels = [
        ((xlim[1] + cx) / 2, (ylim[1] + cy) / 2, "False\ncompensation", "#ff7f00"),
        ((xlim[0] + cx) / 2, (ylim[1] + cy) / 2, "Language\ndependent", "#1f77b4"),
        ((xlim[1] + cx) / 2, (ylim[0] + cy) / 2, "Rerouting", "#4daf4a"),
        ((xlim[0] + cx) / 2, (ylim[0] + cy) / 2, "Fails &\nnot compensating", "#e41a1c"),
    ]
    for qx, qy, qlbl, qc in quad_labels:
        ax.text(qx, qy, qlbl, ha="center", va="center", fontsize=8,
                color=qc, alpha=0.55, fontweight="bold", zorder=2)

    # Legends: colors = modes, shapes = models
    mode_handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=MODE_COLORS.get(m, "grey"), ms=8, label=m)
        for m in modes
    ]
    model_handles = [
        plt.Line2D([0],[0], marker=MODEL_MARKERS.get(mn, "o"), color="black",
                   ms=8, label=mn, lw=0)
        for mn in model_names
    ]
    ax.legend(handles=mode_handles + model_handles, loc="lower right",
              fontsize=7, framealpha=0.85, ncol=2)

    ax.set_xlabel("VCI = Δ visual_fraction (positive = more visual attention)")
    ax.set_ylabel("Δ success (positive = better)")
    ax.set_title("VCI vs Δ Success (color=mode, shape=model)")
    ax.grid(True, ls="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig4_vci_vs_delta_success.png")


def print_summary_table(
    aggregated: Dict[str, Dict],
    modes: List[str],
    model_names: List[str],
) -> None:
    """Print a text summary table of VCI, delta-success, and success rate."""
    col_w = 14
    hdr = f"{"Model":<12} | {"Mode":<10} | {"VCI (mean±sem)":<18} | {"Δsuccess":<10} | {"N_ep":<8}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for model_name in model_names:
        agg = aggregated.get(model_name, {})
        for mode in modes:
            mode_agg = agg.get(mode, {})
            mu_vci, sem_vci, n = _mean_sem(mode_agg.get("vci", []))
            mu_ds, _, _ = _mean_sem([float(v) for v in mode_agg.get("delta_success", [])])
            if n == 0: continue
            vci_str = f"{mu_vci:+.3f} ± {sem_vci:.3f}"
            ds_str = f"{mu_ds:+.3f}"
            print(f"{model_name:<12} | {mode:<10} | {vci_str:<18} | {ds_str:<10} | {n:<8}")
    print("=" * len(hdr) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot language corruption experiment results (VCI + success)"
    )
    parser.add_argument("--pi05-results", type=str, default="",
                        help="Path to pi0.5 JSON results")
    parser.add_argument("--dreamzero-results", type=str, default="",
                        help="Path to DreamZero JSON results")
    parser.add_argument("--output-dir", type=str, default="results/lang_corruption_plots")
    parser.add_argument("--model-names", nargs="+", default=["pi0.5", "DreamZero"],
                        help="Display names for pi0.5 and DreamZero results")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        choices=ALL_MODES + ["original"],
                        help="Corruption modes to include (sorted on x-axis)")
    args = parser.parse_args()

    model_names = args.model_names
    modes = sorted(args.modes)  # alphabetical = empty, opposite, random, shuffle, synonym

    # Load and aggregate results for each model
    result_paths = [args.pi05_results, args.dreamzero_results]
    aggregated: Dict = {}
    for model_name, rpath in zip(model_names, result_paths):
        if not rpath:
            print(f"Skipping {model_name}: no results path provided")
            continue
        raw = load_results(rpath)
        if not raw:
            print(f"Warning: no results loaded for {model_name} from {rpath}")
            continue
        agg = aggregate_results(raw, modes=modes)
        aggregated[model_name] = agg
        print(f"Loaded {len(raw)} episode records for {model_name} from {rpath}")

    if not aggregated:
        print("No results loaded. Provide at least one of --pi05-results or --dreamzero-results")
        return

    active_models = [m for m in model_names if m in aggregated]

    print_summary_table(aggregated, modes=modes, model_names=active_models)

    print("Generating Figure 1: VCI bar chart...")
    plot_vci_bars(aggregated, modes=modes, out_dir=args.output_dir, model_names=active_models)

    print("Generating Figure 2: Success rate bar chart...")
    plot_success_bars(aggregated, modes=modes, out_dir=args.output_dir, model_names=active_models)

    print("Generating Figure 3: Temporal analysis...")
    plot_temporal_analysis(aggregated, modes=modes, out_dir=args.output_dir, model_names=active_models)

    print("Generating Figure 4: VCI vs delta-success scatter...")
    plot_vci_vs_delta_success(aggregated, modes=modes, out_dir=args.output_dir, model_names=active_models)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
