#!/usr/bin/env python3
"""
Temporal reliance analysis: instantaneous vs. temporally-integrated grounding.

Core question: Is episode success better predicted by current-frame IoU (IoU_t)
or by temporally accumulated IoU (rolling mean over past k steps)?

If a model relies on immediate perceptual grounding (pi0.5-style), then IoU_t
should be a strong predictor and rolling averages add little.
If a model integrates temporal context (DreamZero-style), then rolling/aggregated
IoU should outperform instantaneous IoU.

Three analyses:
  1. Correlation sweep: corr(success, IoU) for instantaneous vs. rolling k=3,5,10
     → Temporal Integration Score (TIS) = corr_rolling / corr_instant

  2. Episode-position predictiveness: split episode into early / mid / late thirds,
     compute mean IoU in each third, measure correlation with success.
     → DreamZero should have stronger late-episode dependence (context accumulated);
       pi0.5 should show flatter profile (each step independently predictive).

  3. Lag analysis: for each lag d=0,1,2,...,D, compute corr(success, IoU_{t-d})
     pooled across (episode, step) pairs.
     → A model with temporal memory should show a flatter lag curve (past IoU
        still informative); an instantaneous model should peak at lag 0.

Usage:
  python analyze_temporal_reliance.py \\
    --inputs pi0.5:/path/to/iou_results.json DreamZero:/path/to/dreamzero_iou.json \\
    --layer layer_25 \\
    --rolling-windows 3 5 10 \\
    --max-lag 10 \\
    --output-dir analysis/temporal_reliance

  # Multiple files per model (pooled):
  python analyze_temporal_reliance.py \\
    --inputs pi0.5:/path/A.json pi0.5:/path/B.json DreamZero:/path/C.json \\
    --output-dir analysis/temporal_reliance
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model colours (consistent with plot_generalization.py)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "pi0.5":     "#4C72B0",
    "DreamZero": "#17becf",
    "Cosmos":    "#55A868",
    "OpenVLA":   "#DD8452",
    "DP":        "#C44E52",
}

def _model_color(name: str) -> str:
    for k, v in MODEL_COLORS.items():
        if k.lower() in name.lower():
            return v
    cycle = list(MODEL_COLORS.values())
    return cycle[hash(name) % len(cycle)]


# ──────────────────────────────────────────────────────────────────────────────
# JSON loading helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_trajectories(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    raise ValueError(f"Unknown JSON format in {path}")


def _pick_layer(per_step_iou: dict, preferred: Optional[str]) -> Optional[List[dict]]:
    """Return step list for the requested layer, or best-available layer."""
    if not per_step_iou:
        return None
    if preferred and preferred in per_step_iou:
        return per_step_iou[preferred]
    # Pick layer with most steps (likely middle layer)
    return max(per_step_iou.values(), key=len)


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode IoU time series extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_iou_series(
    trajectories: list,
    layer: Optional[str],
    iou_key: str = "combined_iou",
) -> List[Dict]:
    """
    For each trajectory, extract sorted IoU time series and success label.

    Returns list of:
      {
        "success": bool,
        "steps": [int, ...],       # env step indices (sorted)
        "ious": [float, ...],      # IoU at each step
        "num_env_steps": int,
      }
    """
    episodes = []
    for traj in trajectories:
        if not isinstance(traj, dict):
            continue
        success = bool(traj.get("success", False))
        per_step = traj.get("per_step_iou", {}) or {}
        steps_list = _pick_layer(per_step, layer)
        if not steps_list:
            continue

        pairs = []
        for s in steps_list:
            if not isinstance(s, dict):
                continue
            step = s.get("step")
            iou = s.get(iou_key)
            if step is None or iou is None:
                continue
            try:
                pairs.append((int(step), float(iou)))
            except (TypeError, ValueError):
                continue

        if not pairs:
            continue

        pairs.sort(key=lambda x: x[0])
        steps, ious = zip(*pairs)
        episodes.append({
            "success": success,
            "steps": list(steps),
            "ious": list(ious),
            "num_env_steps": int(traj.get("num_steps", steps[-1])),
        })
    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# Temporal feature computation
# ──────────────────────────────────────────────────────────────────────────────
def rolling_mean(ious: List[float], k: int) -> List[float]:
    """Left-bounded causal rolling mean: mean(ious[max(0,t-k+1):t+1])."""
    result = []
    for i, v in enumerate(ious):
        window = ious[max(0, i - k + 1): i + 1]
        result.append(float(np.mean(window)))
    return result


def episode_position_ious(
    ious: List[float],
    n_bins: int = 3,
) -> List[float]:
    """
    Split IoU series into n_bins equal-width position windows.
    Returns mean IoU in each bin.
    """
    n = len(ious)
    bins = []
    for b in range(n_bins):
        lo = int(b * n / n_bins)
        hi = int((b + 1) * n / n_bins)
        chunk = ious[lo:hi] if hi > lo else ious[lo:lo+1]
        bins.append(float(np.mean(chunk)) if chunk else float("nan"))
    return bins


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1: Correlation sweep over rolling windows
# ──────────────────────────────────────────────────────────────────────────────
def correlation_sweep(
    episodes: List[Dict],
    windows: List[int],
    agg: str = "mean",  # how to reduce per-episode step series → scalar
) -> Dict[str, float]:
    """
    For each window k (0 = instantaneous), compute Pearson r between
    per-episode aggregated IoU and success.

    agg="mean"  : mean of rolling_k(IoU) over all steps in episode
    agg="last"  : last value of rolling_k(IoU) in episode
    agg="late"  : mean of rolling_k(IoU) over last third of steps

    Returns dict: {window_label: pearson_r}
    """
    successes = np.array([e["success"] for e in episodes], dtype=float)

    results: Dict[str, float] = {}

    all_windows = [1] + [w for w in windows if w != 1]  # 1 = instantaneous

    for k in all_windows:
        values = []
        for ep in episodes:
            ious = ep["ious"]
            if k == 1:
                rolled = ious
            else:
                rolled = rolling_mean(ious, k)

            n = len(rolled)
            if agg == "mean":
                val = float(np.mean(rolled))
            elif agg == "last":
                val = rolled[-1]
            elif agg == "late":
                lo = max(0, 2 * n // 3)
                chunk = rolled[lo:]
                val = float(np.mean(chunk)) if chunk else float("nan")
            else:
                val = float(np.mean(rolled))
            values.append(val)

        values = np.array(values, dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 5 or successes[mask].std() < 1e-9:
            results[f"k={k}"] = float("nan")
            continue
        r, p = stats.pearsonr(values[mask], successes[mask])
        results[f"k={k}"] = float(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2: Episode-position predictiveness
# ──────────────────────────────────────────────────────────────────────────────
def position_predictiveness(
    episodes: List[Dict],
    n_bins: int = 3,
    bin_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    For each position bin (early / mid / late), compute correlation of
    mean IoU in that bin with episode success.

    Returns dict: {bin_label: pearson_r}
    """
    if bin_labels is None:
        bin_labels = ["early", "mid", "late"][:n_bins]

    successes = np.array([e["success"] for e in episodes], dtype=float)
    bin_ious = [[] for _ in range(n_bins)]

    for ep in episodes:
        bins = episode_position_ious(ep["ious"], n_bins=n_bins)
        for b, val in enumerate(bins):
            bin_ious[b].append(val)

    results: Dict[str, float] = {}
    for b, label in enumerate(bin_labels):
        vals = np.array(bin_ious[b], dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() < 5 or successes[mask].std() < 1e-9:
            results[label] = float("nan")
            continue
        r, _ = stats.pearsonr(vals[mask], successes[mask])
        results[label] = float(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 3: Lag analysis (step-level)
# ──────────────────────────────────────────────────────────────────────────────
def lag_analysis(
    episodes: List[Dict],
    max_lag: int = 10,
) -> Dict[int, float]:
    """
    For each lag d=0,1,...,max_lag, collect all (IoU_{t-d}, success) pairs
    pooled across episodes and compute Pearson correlation.

    For lag d: IoU_{t-d} = ious[t-d], paired with episode success.
    Pairs are only included where t-d >= 0.
    """
    lag_pairs: Dict[int, Tuple[List[float], List[float]]] = {
        d: ([], []) for d in range(max_lag + 1)
    }

    for ep in episodes:
        ious = ep["ious"]
        s = float(ep["success"])
        n = len(ious)
        for t in range(n):
            for d in range(max_lag + 1):
                if t - d >= 0:
                    lag_pairs[d][0].append(ious[t - d])
                    lag_pairs[d][1].append(s)

    results: Dict[int, float] = {}
    for d, (iou_vals, suc_vals) in lag_pairs.items():
        iou_arr = np.array(iou_vals, dtype=float)
        suc_arr = np.array(suc_vals, dtype=float)
        mask = np.isfinite(iou_arr)
        if mask.sum() < 10 or suc_arr[mask].std() < 1e-9:
            results[d] = float("nan")
            continue
        r, _ = stats.pearsonr(iou_arr[mask], suc_arr[mask])
        results[d] = float(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Integration Score
# ──────────────────────────────────────────────────────────────────────────────
def temporal_integration_score(corr_sweep: Dict[str, float]) -> float:
    """
    TIS = mean(corr_rolling_k) / corr_instant
    where corr_instant = corr_sweep["k=1"] and we average over k > 1.
    """
    instant = corr_sweep.get("k=1", float("nan"))
    if not np.isfinite(instant) or abs(instant) < 1e-9:
        return float("nan")
    rolling_vals = [v for key, v in corr_sweep.items()
                    if key != "k=1" and np.isfinite(v)]
    if not rolling_vals:
        return float("nan")
    return float(np.mean(rolling_vals) / abs(instant))


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", path)
    plt.close(fig)


def plot_correlation_sweep(
    model_sweeps: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Temporal Integration: rolling IoU vs. success correlation",
):
    """Bar chart of corr(success, rolling_k IoU) for each model at each k."""
    models = list(model_sweeps.keys())
    if not models:
        return

    # Collect all window labels across models
    all_keys = []
    for sw in model_sweeps.values():
        for k in sw:
            if k not in all_keys:
                all_keys.append(k)
    # Sort by numeric k
    all_keys.sort(key=lambda x: int(x.split("=")[1]) if "=" in x else 0)

    n_models = len(models)
    n_keys = len(all_keys)
    x = np.arange(n_keys)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(max(6, n_keys * 1.2), 4))
    for i, model in enumerate(models):
        sw = model_sweeps[model]
        heights = [sw.get(k, float("nan")) for k in all_keys]
        offsets = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            offsets, heights, width=width * 0.9,
            color=_model_color(model), label=model, alpha=0.85,
        )
        # Annotate bars with TIS score
        tis = temporal_integration_score(sw)
        if np.isfinite(tis):
            ax.annotate(
                f"TIS={tis:.2f}",
                xy=(offsets[-1], max(h for h in heights if np.isfinite(h)) + 0.01),
                ha="center", fontsize=7, color=_model_color(model),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("k=1", "instant\n(k=1)") for k in all_keys], fontsize=8
    )
    ax.set_ylabel("Pearson r  (IoU → success)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_position_predictiveness(
    model_positions: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Episode-position predictiveness of IoU",
):
    """Line plot of corr(success, mean_IoU_in_bin) vs episode position."""
    models = list(model_positions.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    for model, pos_dict in model_positions.items():
        labels = list(pos_dict.keys())
        vals = [pos_dict[k] for k in labels]
        x = np.arange(len(labels))
        ax.plot(x, vals, "-o", color=_model_color(model), label=model, linewidth=2, markersize=6)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson r  (IoU → success)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_lag_analysis(
    model_lags: Dict[str, Dict[int, float]],
    output_path: str,
    title: str = "Lag analysis: IoU_{t−d} → success",
):
    """Line plot of corr(success, IoU_{t-d}) vs lag d."""
    models = list(model_lags.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, lag_dict in model_lags.items():
        lags = sorted(lag_dict.keys())
        vals = [lag_dict[d] for d in lags]
        ax.plot(lags, vals, "-o", color=_model_color(model), label=model, linewidth=2, markersize=5)

    ax.set_xlabel("Lag d (steps back)", fontsize=9)
    ax.set_ylabel("Pearson r  (IoU_{t−d} → success)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_tis_comparison(
    model_sweeps: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Temporal Integration Score (TIS)",
):
    """
    Horizontal bar chart comparing TIS across models.
    TIS = mean(corr_rolling_k) / |corr_instant|
    """
    models = list(model_sweeps.keys())
    tis_vals = [temporal_integration_score(model_sweeps[m]) for m in models]

    fig, ax = plt.subplots(figsize=(5, max(2, 0.5 * len(models) + 1.5)))
    colors = [_model_color(m) for m in models]
    y = np.arange(len(models))
    bars = ax.barh(y, tis_vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Temporal Integration Score", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="TIS=1 (neutral)")
    for bar, val in zip(bars, tis_vals):
        if np.isfinite(val):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=8,
            )
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_combined_summary(
    model_sweeps: Dict[str, Dict[str, float]],
    model_positions: Dict[str, Dict[str, float]],
    model_lags: Dict[str, Dict[int, float]],
    output_path: str,
):
    """Single-figure summary with all three analyses side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    models = list(model_sweeps.keys())

    # ── Panel 1: Correlation sweep ─────────────────────────────────────────────
    ax = axes[0]
    all_keys = []
    for sw in model_sweeps.values():
        for k in sw:
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort(key=lambda x: int(x.split("=")[1]) if "=" in x else 0)
    n_models = len(models)
    n_keys = len(all_keys)
    x = np.arange(n_keys)
    width = 0.7 / max(n_models, 1)
    for i, model in enumerate(models):
        sw = model_sweeps[model]
        heights = [sw.get(k, float("nan")) for k in all_keys]
        offsets = x + (i - n_models / 2 + 0.5) * width
        ax.bar(offsets, heights, width=width * 0.9,
               color=_model_color(model), label=model, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("k=1", "k=1\n(instant)") for k in all_keys], fontsize=7
    )
    ax.set_ylabel("Pearson r (IoU → success)", fontsize=8)
    ax.set_title("Rolling-window correlation", fontsize=9)
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Episode position ──────────────────────────────────────────────
    ax = axes[1]
    for model, pos_dict in model_positions.items():
        labels = list(pos_dict.keys())
        vals = [pos_dict[k] for k in labels]
        ax.plot(np.arange(len(labels)), vals, "-o",
                color=_model_color(model), label=model, linewidth=2, markersize=6)
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson r (IoU → success)", fontsize=8)
    ax.set_title("Episode-position predictiveness", fontsize=9)
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 3: Lag analysis ──────────────────────────────────────────────────
    ax = axes[2]
    for model, lag_dict in model_lags.items():
        lags = sorted(lag_dict.keys())
        vals = [lag_dict[d] for d in lags]
        ax.plot(lags, vals, "-o",
                color=_model_color(model), label=model, linewidth=2, markersize=5)
    ax.set_xlabel("Lag d (replan steps)", fontsize=8)
    ax.set_ylabel("Pearson r (IoU_{t−d} → success)", fontsize=8)
    ax.set_title("Lag analysis", fontsize=9)
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Temporal reliance: instantaneous vs. integrated grounding",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(
    model_name: str,
    episodes: List[Dict],
    sweep: Dict[str, float],
    positions: Dict[str, float],
    lags: Dict[int, float],
):
    n_eps = len(episodes)
    sr = np.mean([e["success"] for e in episodes])
    tis = temporal_integration_score(sweep)

    print(f"\n{'═'*60}")
    print(f"  Model : {model_name}")
    print(f"  N episodes : {n_eps}   success rate : {sr:.1%}")
    print(f"{'─'*60}")

    print("  Rolling-window correlation (episode-level):")
    for k, r in sweep.items():
        marker = " ← instant" if k == "k=1" else ""
        print(f"    {k:8s}  r = {r:+.3f}{marker}")
    print(f"  Temporal Integration Score (TIS) = {tis:+.3f}")

    print("  Episode-position predictiveness:")
    for pos, r in positions.items():
        print(f"    {pos:8s}  r = {r:+.3f}")

    print("  Lag analysis:")
    for d, r in sorted(lags.items()):
        print(f"    lag={d:2d}  r = {r:+.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Temporal reliance analysis: instantaneous vs. rolling IoU → success"
    )
    parser.add_argument(
        "--inputs", nargs="+", metavar="MODEL:PATH", required=True,
        help="One or more model:path pairs, e.g. pi0.5:/path/to/iou.json. "
             "Multiple entries with the same model name are pooled.",
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        help="Layer to use from per_step_iou (e.g. layer_25). "
             "If omitted, uses the layer with most steps.",
    )
    parser.add_argument(
        "--iou-key", type=str, default="combined_iou",
        help="Key in each step dict to use as IoU signal (default: combined_iou).",
    )
    parser.add_argument(
        "--rolling-windows", type=int, nargs="+", default=[3, 5, 10],
        help="Rolling window sizes k to include in sweep (default: 3 5 10).",
    )
    parser.add_argument(
        "--max-lag", type=int, default=10,
        help="Maximum lag d for lag analysis (default: 10).",
    )
    parser.add_argument(
        "--n-position-bins", type=int, default=3,
        help="Number of equal-width episode-position bins (default: 3 → early/mid/late).",
    )
    parser.add_argument(
        "--episode-agg", type=str, default="mean", choices=["mean", "last", "late"],
        help="How to aggregate per-episode step series for correlation sweep (default: mean).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="analysis/temporal_reliance",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ── Parse inputs ────────────────────────────────────────────────────────────
    model_paths: Dict[str, List[str]] = defaultdict(list)
    for item in args.inputs:
        if ":" not in item:
            parser.error(f"--inputs entries must be MODEL:PATH, got: {item!r}")
        colon = item.index(":")
        model, path = item[:colon], item[colon + 1:]
        model_paths[model].append(path)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and analyse each model ─────────────────────────────────────────────
    model_episodes: Dict[str, List[Dict]] = {}
    model_sweeps: Dict[str, Dict[str, float]] = {}
    model_positions: Dict[str, Dict[str, float]] = {}
    model_lags: Dict[str, Dict[int, float]] = {}

    bin_labels = {3: ["early", "mid", "late"], 4: ["q1", "q2", "q3", "q4"]}.get(
        args.n_position_bins,
        [f"bin{b+1}" for b in range(args.n_position_bins)],
    )

    for model, paths in model_paths.items():
        episodes = []
        for path in paths:
            try:
                trajs = _load_trajectories(path)
                eps = extract_iou_series(trajs, layer=args.layer, iou_key=args.iou_key)
                log.info("%s: %s → %d episodes (%d with IoU)", model, path, len(trajs), len(eps))
                episodes.extend(eps)
            except Exception as e:
                log.warning("Failed loading %s: %s", path, e)

        if not episodes:
            log.warning("No valid episodes for model %s, skipping", model)
            continue

        model_episodes[model] = episodes

        sweep = correlation_sweep(episodes, windows=args.rolling_windows, agg=args.episode_agg)
        model_sweeps[model] = sweep

        positions = position_predictiveness(
            episodes, n_bins=args.n_position_bins, bin_labels=bin_labels
        )
        model_positions[model] = positions

        lags = lag_analysis(episodes, max_lag=args.max_lag)
        model_lags[model] = lags

        print_summary(model, episodes, sweep, positions, lags)

    if not model_sweeps:
        log.error("No data loaded. Check --inputs paths.")
        return 1

    # ── Plots ────────────────────────────────────────────────────────────────────
    plot_correlation_sweep(
        model_sweeps,
        output_path=str(out_dir / "correlation_sweep.png"),
        title="Rolling-window IoU → success correlation",
    )
    plot_position_predictiveness(
        model_positions,
        output_path=str(out_dir / "position_predictiveness.png"),
    )
    plot_lag_analysis(
        model_lags,
        output_path=str(out_dir / "lag_analysis.png"),
    )
    plot_tis_comparison(
        model_sweeps,
        output_path=str(out_dir / "tis_comparison.png"),
    )
    plot_combined_summary(
        model_sweeps,
        model_positions,
        model_lags,
        output_path=str(out_dir / "temporal_reliance_summary.png"),
    )

    # ── Save numeric results ─────────────────────────────────────────────────────
    results_out = {}
    for model in model_sweeps:
        tis = temporal_integration_score(model_sweeps[model])
        results_out[model] = {
            "n_episodes": len(model_episodes[model]),
            "success_rate": float(np.mean([e["success"] for e in model_episodes[model]])),
            "temporal_integration_score": tis,
            "correlation_sweep": model_sweeps[model],
            "position_predictiveness": model_positions.get(model, {}),
            "lag_analysis": {str(d): r for d, r in model_lags.get(model, {}).items()},
        }
    out_json = out_dir / "temporal_reliance_results.json"
    with open(out_json, "w") as f:
        json.dump(results_out, f, indent=2)
    log.info("Results saved to %s", out_json)

    # ── TIS comparison table ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  TEMPORAL INTEGRATION SCORE COMPARISON")
    print(f"{'─'*60}")
    print(f"  {'Model':<15}  {'TIS':>8}  {'Instant r':>10}  {'Best roll r':>12}")
    print(f"{'─'*60}")
    for model, sw in model_sweeps.items():
        tis = temporal_integration_score(sw)
        inst = sw.get("k=1", float("nan"))
        roll_vals = [v for k, v in sw.items() if k != "k=1" and np.isfinite(v)]
        best_roll = max(roll_vals, key=abs) if roll_vals else float("nan")
        print(
            f"  {model:<15}  {tis:>8.3f}  {inst:>10.3f}  {best_roll:>12.3f}"
        )
    print(
        "\n  Interpretation: TIS > 1 → rolling IoU more predictive than instantaneous"
        "\n                  (expected for DreamZero; ~1 or < 1 for pi0.5)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
