"""Plot language corruption experiment results.

Validates the hypothesis:
  pi0.5 behaves like an immediate cue-conditioned controller:
  under language corruption it increases visual engagement (delta_ratio > 0)
  but loses selectivity (delta_iou < 0).

Reads per-perturbation JSON files from:
  results/attention/ratio/{model}/perturb/{perturb_type}/...
  results/attention/iou/{model}/perturb/{perturb_type}/...

Produces:
  fig1_engagement_selectivity.png  -- delta_ratio (top) + delta_iou (bottom) per benchmark
  fig2_hypothesis_scatter.png      -- scatter delta_ratio vs delta_iou (the key hypothesis figure)

Usage:
  python analysis/plot_lang_corruption.py \\
    --pi05-ratio-dir   results/attention/ratio/pi05 \\
    --pi05-iou-dir     results/attention/iou/pi05 \\
    --dreamzero-ratio-dir results/attention/ratio/dreamzero \\
    --dreamzero-iou-dir   results/attention/iou/dreamzero \\
    --output-dir results/lang_corruption_plots
"""

import argparse
import glob as glob_module
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── Constants ────────────────────────────────────────────────────────────────

ALL_MODES = ["empty", "random", "shuffle", "synonym"]

BENCHMARK_DISPLAY = {
    "libero_10":     "LIBERO-In domain",
    "libero_90_obj": "LIBERO-90-Object",
    "libero_90_spa": "LIBERO-90-Spatial",
    "libero_90_act": "LIBERO-90-Act",
    "libero_90_com": "LIBERO-90-Com",
}
BENCHMARK_ORDER = ["libero_90_obj", "libero_90_spa", "libero_90_act", "libero_90_com"]

MODEL_COLORS = {
    "pi0.5":     "#4C72B0",
    "DreamZero": "#17becf",
}
MODEL_MARKERS = {"pi0.5": "o", "DreamZero": "s"}

MODE_COLORS = {
    "empty":   "#e41a1c",
    "shuffle": "#377eb8",
    "random":  "#ff7f00",
    "synonym": "#4daf4a",
}
MODE_LINESTYLES = {
    "empty":   "-",
    "shuffle": "--",
    "random":  "-.",
    "synonym": ":",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

# Maps perturb directory name -> corruption mode name
_PERTURB_DIR_TO_MODE = {
    "none":           "original",
    "prompt_empty":   "empty",
    "prompt_random":  "random",
    "prompt_shuffle": "shuffle",
    "prompt_synonym": "synonym",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_iou_from_dir(
    model_dir: str,
    modes: List[str],
    iou_layer: str = "layer_25",
) -> Dict[Tuple, float]:
    """Load IoU values from per-perturbation IOU directory.

    Returns: {(benchmark, task_id, episode_idx, mode_name): iou_value}
    """
    perturb_root = os.path.join(model_dir, "perturb")
    result: Dict[Tuple, float] = {}
    if not os.path.isdir(perturb_root):
        return result

    wanted = set(modes) | {"original"}
    for perturb_name, mode_name in _PERTURB_DIR_TO_MODE.items():
        if mode_name not in wanted:
            continue
        perturb_path = os.path.join(perturb_root, perturb_name)
        if not os.path.isdir(perturb_path):
            continue
        for json_path in glob_module.glob(os.path.join(perturb_path, "*", "iou_results_*.json")):
            with open(json_path) as f:
                data = json.load(f)
            ep_list = data["results"] if isinstance(data, dict) and "results" in data else data
            parent = os.path.basename(os.path.dirname(json_path))
            benchmark = parent.rsplit("_seed", 1)[0]
            for ep in ep_list:
                summary = ep.get("summary", {})
                layer_data = summary.get(iou_layer) or next(iter(summary.values()), {})
                iou_val = layer_data.get("combined_iou", {}).get("mean")
                if iou_val is None:
                    continue
                key = (benchmark, int(ep["task_id"]), int(ep["episode_idx"]), mode_name)
                result[key] = float(iou_val)
    return result


def load_results_from_ratio_dir(
    model_dir: str,
    modes: List[str],
    iou_data: Optional[Dict[Tuple, float]] = None,
) -> List[Dict]:
    """Load per-perturbation ratio results, optionally joined with IoU data.

    Returns a list of per-episode records with keys:
      benchmark, modes {mode: {success, visual_fraction, ...}},
      deltas {mode: {delta_ratio, delta_iou, delta_success, temporal_analysis}}
    """
    perturb_root = os.path.join(model_dir, "perturb")
    if not os.path.isdir(perturb_root):
        return []

    episodes_by_mode: Dict[str, Dict] = {}
    wanted = set(modes) | {"original"}
    for perturb_name, mode_name in _PERTURB_DIR_TO_MODE.items():
        if mode_name not in wanted:
            continue
        perturb_path = os.path.join(perturb_root, perturb_name)
        if not os.path.isdir(perturb_path):
            continue
        episodes_by_mode[mode_name] = {}
        for json_path in glob_module.glob(
            os.path.join(perturb_path, "*", "attention_ratio_results_*.json")
        ):
            with open(json_path) as f:
                data = json.load(f)
            ep_list = data["results"] if isinstance(data, dict) and "results" in data else data
            parent = os.path.basename(os.path.dirname(json_path))
            benchmark = parent.rsplit("_seed", 1)[0]
            for ep in ep_list:
                key = (benchmark, int(ep["task_id"]), int(ep["episode_idx"]))
                episodes_by_mode[mode_name][key] = ep

    if "original" not in episodes_by_mode:
        return []

    def _vf(ep):
        return float(ep["summary"]["layers_avg"]["visual_fraction"]["mean"])

    def _temporal_phases(ep):
        steps = ep.get("per_step_ratios", {}).get("layers_avg", [])
        if not steps:
            return None, None, None
        vfs = [s["visual_fraction"] for s in steps]
        n = len(vfs)
        t1, t2 = n // 3, 2 * n // 3
        early = float(np.mean(vfs[:t1]))  if t1 > 0  else float("nan")
        mid   = float(np.mean(vfs[t1:t2])) if t2 > t1 else float("nan")
        late  = float(np.mean(vfs[t2:]))  if n > t2  else float("nan")
        return early, mid, late

    def _mode_record(ep):
        vf = _vf(ep)
        early, mid, late = _temporal_phases(ep)
        d = {"success": bool(ep["success"]), "visual_fraction": vf}
        if early is not None:
            d["temporal_early"] = early
            d["temporal_mid"]   = mid
            d["temporal_late"]  = late
            d["temporal_slope"] = (late - early) if (late and early) else 0.0
        return d

    combined = []
    for key, orig_ep in episodes_by_mode["original"].items():
        benchmark = key[0]
        vf_orig = _vf(orig_ep)
        iou_orig = iou_data.get((*key, "original")) if iou_data else None

        record: Dict = {
            "benchmark": benchmark,
            "modes": {"original": _mode_record(orig_ep)},
            "deltas": {},
        }
        for mode in modes:
            mode_ep = episodes_by_mode.get(mode, {}).get(key)
            if mode_ep is None:
                continue
            vf_mode = _vf(mode_ep)
            mode_e, mode_m, mode_l = _temporal_phases(mode_ep)
            record["modes"][mode] = _mode_record(mode_ep)

            delta: Dict = {
                "delta_ratio":   vf_mode - vf_orig,
                "delta_success": int(mode_ep["success"]) - int(orig_ep["success"]),
                "temporal_analysis": {
                    "mode_early": mode_e or 0.0,
                    "mode_mid":   mode_m or 0.0,
                    "mode_late":  mode_l or 0.0,
                    "temporal_slope": (mode_l - mode_e) if (mode_l and mode_e) else 0.0,
                },
            }
            if iou_data is not None:
                iou_mode = iou_data.get((*key, mode))
                if iou_orig is not None and iou_mode is not None:
                    delta["delta_iou"] = iou_mode - iou_orig
                    delta["iou_orig"]  = iou_orig
                    delta["iou_mode"]  = iou_mode
            record["deltas"][mode] = delta
        combined.append(record)

    return combined


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(
    results: List[Dict],
    modes: List[str] = None,
    benchmark: str = None,
) -> Dict:
    """Aggregate per-episode records.

    Returns: mode -> {delta_ratio, delta_iou, delta_success, success,
                      visual_fraction, temporal_early/mid/late}
    """
    if modes is None:
        modes = ALL_MODES
    if benchmark is not None:
        results = [ep for ep in results if ep.get("benchmark") == benchmark]

    empty_lists = lambda: {
        "delta_ratio": [], "delta_iou": [], "delta_success": [],
        "success": [], "visual_fraction": [],
        "temporal_early": [], "temporal_mid": [], "temporal_late": [],
    }
    out = {m: empty_lists() for m in modes}
    out["original"] = empty_lists()

    for ep in results:
        modes_data = ep.get("modes", {})
        deltas = ep.get("deltas", {})

        orig = modes_data.get("original", {})
        if orig:
            out["original"]["success"].append(bool(orig.get("success", False)))
            out["original"]["visual_fraction"].append(float(orig.get("visual_fraction", 0.0)))
            for p in ("early", "mid", "late"):
                v = orig.get(f"temporal_{p}")
                if v is not None:
                    out["original"][f"temporal_{p}"].append(float(v))

        for m in modes:
            mode_data = modes_data.get(m, {})
            delta = deltas.get(m, {})

            if mode_data:
                out[m]["success"].append(bool(mode_data.get("success", False)))
                out[m]["visual_fraction"].append(float(mode_data.get("visual_fraction", 0.0)))

            if delta:
                out[m]["delta_ratio"].append(float(delta.get("delta_ratio", 0.0)))
                out[m]["delta_success"].append(int(delta.get("delta_success", 0)))
                if "delta_iou" in delta:
                    out[m]["delta_iou"].append(float(delta["delta_iou"]))
                ta = delta.get("temporal_analysis", {})
                for p in ("early", "mid", "late"):
                    v = ta.get(f"mode_{p}")
                    if v is not None:
                        out[m][f"temporal_{p}"].append(float(v))

    return out


# ── Utilities ─────────────────────────────────────────────────────────────────

def _mean_sem(vals):
    if not vals:
        return float("nan"), 0.0, 0
    a = np.array(vals, dtype=float)
    n = len(a)
    return float(np.nanmean(a)), float(np.nanstd(a) / max(1, np.sqrt(n))), n


def _save_fig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def _get_benchmark_panels(raw_by_model):
    benchmarks = [b for b in BENCHMARK_ORDER if any(
        any(ep.get("benchmark") == b for ep in raw)
        for raw in raw_by_model.values()
    )]
    panels = benchmarks + ["_all_"]
    labels = [BENCHMARK_DISPLAY.get(b, b) for b in benchmarks] + ["All (aggregated)"]
    return panels, labels


# ── Panel draw helpers ────────────────────────────────────────────────────────

def _draw_delta_panel(ax, raw_by_model, modes, model_names, panel, metric_key, ylabel, first):
    """Generic bar panel for any delta metric (delta_ratio or delta_iou)."""
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(len(modes))
    has_data = False
    for i, model_name in enumerate(model_names):
        raw = raw_by_model.get(model_name, [])
        agg = aggregate_results(raw, modes=modes,
                                benchmark=None if panel == "_all_" else panel)
        means = [_mean_sem(agg.get(m, {}).get(metric_key, []))[0] for m in modes]
        sems  = [_mean_sem(agg.get(m, {}).get(metric_key, []))[1] for m in modes]
        if any(np.isfinite(v) for v in means):
            has_data = True
        color = MODEL_COLORS.get(model_name, f"C{i}")
        offset = (i - n_models / 2.0 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, color=color,
               label=model_name, zorder=3, yerr=sems, capsize=3,
               error_kw={"zorder": 4})
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if first:
        ax.legend(fontsize=7, framealpha=0.85)
    if not has_data:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="grey")


# ── Figure 1: engagement + selectivity grid ───────────────────────────────────

def plot_engagement_selectivity_grid(
    raw_by_model: Dict[str, List[Dict]],
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Fig 1: 4-row grid.
    Rows 0-1: delta_ratio (visual engagement) per benchmark + All.
    Rows 2-3: delta_iou  (selectivity)        per benchmark + All.
    """
    panels, panel_labels = _get_benchmark_panels(raw_by_model)
    ncols = 3
    nrows_half = (len(panels) + ncols - 1) // ncols  # 2 for 6 panels
    nrows = nrows_half * 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.5), squeeze=False)
    fig.suptitle(
        r"Language Corruption: $\Delta$ratio (visual engagement, top) "
        r"and $\Delta$IoU (selectivity, bottom)",
        fontsize=11, fontweight="bold",
    )

    for idx, (panel, label) in enumerate(zip(panels, panel_labels)):
        ax_r = axes[idx // ncols][idx % ncols]
        _draw_delta_panel(ax_r, raw_by_model, modes, model_names, panel,
                          "delta_ratio", r"$\Delta$ratio", first=(idx == 0))
        ax_r.set_title(label, fontsize=9)

        ax_i = axes[nrows_half + idx // ncols][idx % ncols]
        _draw_delta_panel(ax_i, raw_by_model, modes, model_names, panel,
                          "delta_iou", r"$\Delta$IoU", first=(idx == 0))
        ax_i.set_title(label, fontsize=9)

    for idx in range(len(panels), nrows_half * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
        axes[nrows_half + idx // ncols][idx % ncols].set_visible(False)

    fig.text(0.005, 0.76, r"$\Delta$ratio", fontsize=10, fontweight="bold",
             va="center", rotation="vertical")
    fig.text(0.005, 0.27, r"$\Delta$IoU", fontsize=10, fontweight="bold",
             va="center", rotation="vertical")

    fig.tight_layout(rect=[0.015, 0, 1, 0.97])
    _save_fig(fig, out_dir, "fig1_engagement_selectivity.png")


# ── Figure 2: hypothesis scatter ─────────────────────────────────────────────

def plot_hypothesis_scatter(
    raw_by_model: Dict[str, List[Dict]],
    modes: List[str],
    out_dir: str,
    model_names: List[str],
) -> None:
    """Fig 2: Scatter delta_ratio vs delta_iou.

    Each point = mean (delta_ratio, delta_iou) for one (model, mode, benchmark).
    Large marker = mean across all benchmarks.
    Hypothesis: pi0.5 clusters in the bottom-right quadrant
                (delta_ratio > 0, delta_iou < 0) = cue-conditioned controller.
    """
    panels, panel_labels = _get_benchmark_panels(raw_by_model)
    # per-benchmark panels only (no _all_)
    bench_panels = [(p, l) for p, l in zip(panels, panel_labels) if p != "_all_"]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    for model_name in model_names:
        raw = raw_by_model.get(model_name, [])
        marker = MODEL_MARKERS.get(model_name, "o")
        model_color = MODEL_COLORS.get(model_name, "grey")

        for mode in modes:
            color = MODE_COLORS.get(mode, "grey")
            per_bench_ratios, per_bench_ious = [], []

            for panel, _ in bench_panels:
                agg = aggregate_results(raw, modes=modes, benchmark=panel)
                dr = agg.get(mode, {}).get("delta_ratio", [])
                di = agg.get(mode, {}).get("delta_iou", [])
                if dr and di:
                    per_bench_ratios.append(float(np.mean(dr)))
                    per_bench_ious.append(float(np.mean(di)))

            if not per_bench_ratios:
                continue

            # Small points: one per benchmark
            ax.scatter(per_bench_ratios, per_bench_ious,
                       c=color, marker=marker, alpha=0.45, s=55, zorder=4,
                       edgecolors=model_color, linewidths=0.5)

            # Large point: mean across benchmarks
            ax.scatter([np.mean(per_bench_ratios)], [np.mean(per_bench_ious)],
                       c=color, marker=marker, s=220, zorder=6,
                       edgecolors="black", linewidths=1.2,
                       label=f"{model_name} / {mode}")

    ax.axhline(0, color="black", lw=1.0, ls="-", zorder=2)
    ax.axvline(0, color="black", lw=1.0, ls="-", zorder=2)

    # Quadrant annotations
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    cx, cy = sum(xlim) / 2, sum(ylim) / 2
    quadrants = [
        ((xlim[1] + cx) / 2, (ylim[1] + cy) / 2,
         "more visual\nmore selective",     "grey"),
        ((xlim[0] + cx) / 2, (ylim[1] + cy) / 2,
         "less visual\nmore selective",     "grey"),
        ((xlim[1] + cx) / 2, (ylim[0] + cy) / 2,
         "cue-conditioned\n(more visual, less selective)", "#d62728"),
        ((xlim[0] + cx) / 2, (ylim[0] + cy) / 2,
         "less visual\nless selective",     "grey"),
    ]
    for qx, qy, qlbl, qc in quadrants:
        weight = "bold" if "cue-conditioned" in qlbl else "normal"
        ax.text(qx, qy, qlbl, ha="center", va="center", fontsize=8,
                color=qc, alpha=0.65, fontweight=weight, zorder=1)

    # Legend: mode colors + model shapes
    mode_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=MODE_COLORS.get(m, "grey"), ms=9, label=m)
        for m in modes
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS.get(mn, "o"), color="black",
                   ms=9, label=mn, lw=0)
        for mn in model_names
    ]
    ax.legend(handles=mode_handles + model_handles,
              loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    ax.set_xlabel(r"$\Delta$ratio (visual engagement; positive = more visual attention)",
                  fontsize=9)
    ax.set_ylabel(r"$\Delta$IoU (selectivity; positive = more on task objects)", fontsize=9)
    ax.set_title(
        r"Hypothesis: $\pi_{0.5}$ is cue-conditioned"
        "\n"
        r"($\Delta$ratio > 0, $\Delta$IoU < 0 under language corruption)",
        fontsize=10,
    )
    ax.grid(True, ls="--", alpha=0.25, zorder=0)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig2_hypothesis_scatter.png")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(
    aggregated: Dict[str, Dict],
    modes: List[str],
    model_names: List[str],
) -> None:
    hdr = (f"{'Model':<12} | {'Mode':<10} | "
           f"{'d_ratio (mean+/-sem)':<22} | "
           f"{'d_iou (mean+/-sem)':<22} | "
           f"{'N_iou':<6} | {'N_ratio':<8}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for model_name in model_names:
        agg = aggregated.get(model_name, {})
        for mode in modes:
            m = agg.get(mode, {})
            mu_r, sem_r, n_r = _mean_sem(m.get("delta_ratio", []))
            mu_i, sem_i, n_i = _mean_sem(m.get("delta_iou", []))
            if n_r == 0:
                continue
            r_str = f"{mu_r:+.4f}+/-{sem_r:.4f}"
            i_str = f"{mu_i:+.4f}+/-{sem_i:.4f}" if n_i > 0 else "N/A"
            print(f"{model_name:<12} | {mode:<10} | {r_str:<22} | {i_str:<22} | "
                  f"{n_i:<6} | {n_r:<8}")
    print("=" * len(hdr) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate cue-conditioned controller hypothesis via delta_ratio and delta_iou"
    )
    parser.add_argument("--pi05-ratio-dir", type=str, default="",
                        help="results/attention/ratio/pi05")
    parser.add_argument("--pi05-iou-dir", type=str, default="",
                        help="results/attention/iou/pi05")
    parser.add_argument("--dreamzero-ratio-dir", type=str, default="",
                        help="results/attention/ratio/dreamzero")
    parser.add_argument("--dreamzero-iou-dir", type=str, default="",
                        help="results/attention/iou/dreamzero")
    parser.add_argument("--iou-layer", type=str, default="layer_25",
                        help="Layer key to read from IOU summary (default: layer_25)")
    parser.add_argument("--output-dir", type=str, default="results/lang_corruption_plots")
    parser.add_argument("--model-names", nargs="+", default=["pi0.5", "DreamZero"],
                        help="Display names (order: pi0.5, DreamZero)")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        choices=ALL_MODES + ["original"])
    args = parser.parse_args()

    model_names = args.model_names
    modes = sorted(args.modes)

    ratio_dirs = [args.pi05_ratio_dir, args.dreamzero_ratio_dir]
    iou_dirs   = [args.pi05_iou_dir,   args.dreamzero_iou_dir]

    raw_by_model: Dict[str, List[Dict]] = {}
    aggregated:   Dict[str, Dict]       = {}

    for model_name, rdir, idir in zip(model_names, ratio_dirs, iou_dirs):
        if not rdir:
            print(f"Skipping {model_name}: no --ratio-dir provided")
            continue

        iou_data = None
        if idir:
            iou_data = load_iou_from_dir(idir, modes=modes, iou_layer=args.iou_layer)
            print(f"  {model_name}: loaded {len(iou_data)} IoU entries from {idir}")
        else:
            print(f"  {model_name}: no IoU dir provided, delta_iou will be absent")

        raw = load_results_from_ratio_dir(rdir, modes=modes, iou_data=iou_data)
        if not raw:
            print(f"Warning: no ratio results loaded for {model_name} from {rdir}")
            continue

        raw_by_model[model_name] = raw
        aggregated[model_name]   = aggregate_results(raw, modes=modes)
        n_iou = sum(
            1 for ep in raw
            for m in modes
            if "delta_iou" in ep.get("deltas", {}).get(m, {})
        )
        print(f"  {model_name}: {len(raw)} episodes, {n_iou} with IoU deltas")

    if not aggregated:
        print("No results loaded.")
        return

    active_models = [m for m in model_names if m in aggregated]
    print_summary_table(aggregated, modes=modes, model_names=active_models)

    print("Generating Figure 1: delta_ratio + delta_iou by benchmark...")
    plot_engagement_selectivity_grid(
        raw_by_model, modes=modes, out_dir=args.output_dir, model_names=active_models
    )

    print("Generating Figure 2: hypothesis scatter (delta_ratio vs delta_iou)...")
    plot_hypothesis_scatter(
        raw_by_model, modes=modes, out_dir=args.output_dir, model_names=active_models
    )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
