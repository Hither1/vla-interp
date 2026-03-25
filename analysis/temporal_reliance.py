#!/usr/bin/env python3
"""Temporal reliance analysis: instantaneous vs temporally integrated grounding.

This script is the non-language grounding analysis entry point. It compares
how strongly eventual success is predicted by:

1. instantaneous grounding: current per-step signal x_t
2. temporally integrated grounding: rolling mean over x_[t-k:t]

Supported signals are pulled from existing evaluation JSONs:
- IoU-like signals from ``per_step_iou``
- ratio / fraction signals from ``per_step_ratios``

Typical use:

python analysis/temporal_reliance.py \
  --inputs pi0.5:results/.../iou_results_libero_10.json \
           DreamZero:results/.../attention_results_libero_10.json \
  --feature iou \
  --rolling-windows 3 5 10 \
  --output-dir results/temporal_reliance
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

SUITE_LABELS: Dict[str, str] = {
    "libero_10":      "LIBERO-10",
    "libero_90_spa":  "LIBERO-Spatial",
    "libero_90_obj":  "LIBERO-Object",
    "libero_90_act":  "LIBERO-Goal",
    "libero_90_com":  "LIBERO-Long",
}
_SUITE_RE = re.compile(r"libero_(10|90_(?:act|com|obj|spa))")


def _detect_suite(path: str) -> Optional[str]:
    m = _SUITE_RE.search(path)
    if not m:
        return None
    key = "libero_" + m.group(1)
    return SUITE_LABELS.get(key, key)

DEFAULT_INPUTS = [
    "pi0.5:results/attention/iou/pi05/perturb/none/libero_10",
    "DreamZero:data/libero/dreamzero/perturb/none/libero_10",
]

MODEL_COLORS = {
    "pi0.5":    "#4C72B0",
    "OpenVLA":  "#DD8452",
    "Cosmos":   "#55A868",
    "DP":       "#C44E52",
    "DreamZero":"#17becf",
}

FEATURE_SPECS = {
    "iou": {
        "containers": ["per_step_iou", "attention_steps"],
        "candidates": ["combined_iou", "iou", "iou_iou"],
        "label": "IoU",
    },
    "attention_mass": {
        "containers": ["per_step_iou", "attention_steps"],
        "candidates": ["attention_mass", "iou_attention_mass"],
        "label": "Attention Mass",
    },
    "ratio": {
        "containers": ["per_step_ratios", "attention_steps"],
        "candidates": ["visual_linguistic_ratio"],
        "label": "Visual/Linguistic Ratio",
    },
    "visual_fraction": {
        "containers": ["per_step_ratios", "attention_steps"],
        "candidates": ["visual_fraction"],
        "label": "Visual Fraction",
    },
    "linguistic_fraction": {
        "containers": ["per_step_ratios", "attention_steps"],
        "candidates": ["linguistic_fraction"],
        "label": "Linguistic Fraction",
    },
    "action_fraction": {
        "containers": ["per_step_ratios"],
        "candidates": ["action_fraction"],
        "label": "Action Fraction",
    },
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def _model_color(name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if key.lower() in name.lower():
            return color
    palette = list(MODEL_COLORS.values())
    return palette[hash(name) % len(palette)]


def _load_trajectories(path: str) -> List[dict]:
    path_obj = Path(path)
    if path_obj.is_dir():
        trajectories = []
        for json_path in sorted(path_obj.rglob("*.json")):
            if json_path.name == "summary.json":
                continue
            with open(json_path, "r") as f:
                raw = json.load(f)
            suite = _detect_suite(str(json_path))
            items: List[dict] = []
            if isinstance(raw, dict) and ("attention_steps" in raw or "task_id" in raw):
                items = [raw]
            elif isinstance(raw, list):
                items = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict) and isinstance(raw.get("results"), list):
                items = [item for item in raw["results"] if isinstance(item, dict)]
            for traj in items:
                traj.setdefault("_suite", suite)
            trajectories.extend(items)
        if trajectories:
            return trajectories
        raise ValueError(f"No supported trajectory JSONs found in directory {path}")

    with open(path, "r") as f:
        raw = json.load(f)
    suite = _detect_suite(path)
    if isinstance(raw, list):
        trajs = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("results"), list):
        trajs = [item for item in raw["results"] if isinstance(item, dict)]
    elif isinstance(raw, dict) and ("attention_steps" in raw or "task_id" in raw):
        trajs = [raw]
    else:
        raise ValueError(f"Unsupported JSON format in {path}")
    for traj in trajs:
        traj.setdefault("_suite", suite)
    return trajs


def _episode_key(traj: dict) -> Tuple[object, ...]:
    vis = traj.get("visual_perturbation") or {}
    pol = traj.get("policy_perturbation") or {}
    return (
        traj.get("_suite"),
        traj.get("task_id"),
        traj.get("episode_idx"),
        traj.get("task_description"),
        traj.get("prompt_mode"),
        vis.get("mode"),
        pol.get("mode"),
    )


def _pick_layer(series_dict: dict, preferred: Optional[str]) -> Tuple[Optional[str], Optional[list]]:
    if not isinstance(series_dict, dict) or not series_dict:
        return None, None
    if preferred and preferred in series_dict:
        return preferred, series_dict[preferred]
    if "layers_avg" in series_dict:
        return "layers_avg", series_dict["layers_avg"]
    best_key = max(
        series_dict,
        key=lambda key: len(series_dict.get(key) or []),
    )
    return best_key, series_dict[best_key]


def _safe_float(value) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_feature_series(
    traj: dict,
    feature: str,
    layer: Optional[str],
) -> Tuple[Optional[str], List[Tuple[int, float]]]:
    spec = FEATURE_SPECS[feature]
    for container_name in spec["containers"]:
        container = traj.get(container_name, {}) or {}
        if isinstance(container, list):
            layer_name = container_name
            steps = container
        else:
            layer_name, steps = _pick_layer(container, layer)
        if not steps:
            continue

        pairs: List[Tuple[int, float]] = []
        for step_info in steps:
            if not isinstance(step_info, dict):
                continue
            step = step_info.get("step", step_info.get("t"))
            if step is None:
                continue
            value = None
            for key in spec["candidates"]:
                if key in step_info:
                    value = _safe_float(step_info[key])
                    if value is not None:
                        break
            if value is None:
                continue
            try:
                pairs.append((int(step), value))
            except Exception:
                continue

        pairs.sort(key=lambda item: item[0])
        deduped: List[Tuple[int, float]] = []
        for step, value in pairs:
            if deduped and deduped[-1][0] == step:
                deduped[-1] = (step, float(np.mean([deduped[-1][1], value])))
            else:
                deduped.append((step, value))
        if deduped:
            return layer_name, deduped

    return None, []


def load_model_episodes(
    paths: Sequence[str],
    features: Sequence[str],
    layer: Optional[str],
) -> List[dict]:
    merged: Dict[Tuple[object, object, object], dict] = {}

    for path in paths:
        trajectories = _load_trajectories(path)
        log.info("Loading %s (%d trajectories)", path, len(trajectories))
        for traj in trajectories:
            if not isinstance(traj, dict):
                continue
            key = _episode_key(traj)
            ep = merged.setdefault(
                key,
                {
                    "task_id": traj.get("task_id"),
                    "episode_idx": traj.get("episode_idx"),
                    "task_description": traj.get("task_description"),
                    "success": bool(traj.get("success", False)),
                    "num_steps": traj.get("num_steps"),
                    "_suite": traj.get("_suite"),
                    "signals": {},
                    "layers": {},
                    "source_paths": [],
                },
            )

            ep["success"] = bool(traj.get("success", ep["success"]))
            if ep["num_steps"] is None and traj.get("num_steps") is not None:
                ep["num_steps"] = traj.get("num_steps")
            if path not in ep["source_paths"]:
                ep["source_paths"].append(path)

            for feature in features:
                layer_name, series = _extract_feature_series(traj, feature, layer)
                if not series:
                    continue
                if feature not in ep["signals"] or len(series) > len(ep["signals"][feature]):
                    ep["signals"][feature] = series
                    ep["layers"][feature] = layer_name

    episodes = [ep for ep in merged.values() if ep["signals"]]
    episodes.sort(key=lambda ep: (ep["task_id"], ep["episode_idx"], ep["task_description"] or ""))
    return episodes


def normalized_episode_curves(
    episodes: Sequence[dict],
    feature: str,
    n_bins: int = 50,
) -> Dict[str, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins)
    groups = {"success": [], "failure": []}
    for ep in episodes:
        series = ep["signals"].get(feature)
        if not series or len(series) < 2:
            continue
        vals = np.asarray([value for _, value in series], dtype=float)
        src = np.linspace(0.0, 1.0, len(vals))
        interp = np.interp(bins, src, vals)
        groups["success" if ep["success"] else "failure"].append(interp)

    output = {}
    for key, curves in groups.items():
        if curves:
            output[key] = np.mean(np.vstack(curves), axis=0)
        else:
            output[key] = np.full_like(bins, np.nan)
    output["positions"] = bins
    return output


def _run_lengths(mask: Sequence[bool]) -> List[int]:
    runs: List[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def burst_tolerance_stats(
    episodes: Sequence[dict],
    feature: str,
    low_quantile: float,
    transient_max_len: int,
    persistent_min_len: int,
) -> Dict[str, dict]:
    pooled = []
    for ep in episodes:
        series = ep["signals"].get(feature)
        if series:
            pooled.extend(value for _, value in series)
    if not pooled:
        return {}

    threshold = float(np.quantile(np.asarray(pooled, dtype=float), low_quantile))
    groups = {
        "none": {"successes": 0, "count": 0},
        "transient": {"successes": 0, "count": 0},
        "persistent": {"successes": 0, "count": 0},
        "mixed": {"successes": 0, "count": 0},
    }

    for ep in episodes:
        series = ep["signals"].get(feature)
        if not series:
            continue
        mask = [value <= threshold for _, value in series]
        runs = _run_lengths(mask)
        longest = max(runs) if runs else 0
        has_transient = any(run <= transient_max_len for run in runs)
        has_persistent = any(run >= persistent_min_len for run in runs)

        if longest == 0:
            bucket = "none"
        elif has_persistent and has_transient:
            bucket = "mixed"
        elif has_persistent:
            bucket = "persistent"
        else:
            bucket = "transient"

        groups[bucket]["count"] += 1
        groups[bucket]["successes"] += int(ep["success"])

    for bucket, stats_dict in groups.items():
        count = stats_dict["count"]
        stats_dict["success_rate"] = float(stats_dict["successes"] / count) if count else float("nan")
    groups["_threshold"] = {
        "count": len(pooled),
        "low_quantile": low_quantile,
        "threshold": threshold,
    }
    return groups


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_episode_trajectories(
    curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    feature: str,
    output_path: Path,
) -> None:
    models = list(curves)
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        pos = curves[model][feature]["positions"]
        suc = curves[model][feature]["success"]
        fail = curves[model][feature]["failure"]
        color = _model_color(model)
        ax.plot(pos, suc, color=color, linewidth=2.5, label="success")
        ax.plot(pos, fail, color=color, linewidth=2.5, linestyle="--", label="failure")
        ax.set_title(model)
        ax.set_xlabel("Episode progress")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(FEATURE_SPECS[feature]["label"])
    axes[0].legend()
    fig.suptitle(f"{FEATURE_SPECS[feature]['label']}: mean trajectory in successful vs failed episodes")
    _save(fig, output_path)


def plot_burst_tolerance(
    burst_stats: Dict[str, Dict[str, dict]],
    feature: str,
    output_path: Path,
) -> None:
    models = list(burst_stats)
    if not models:
        return
    buckets = ["none", "transient", "persistent", "mixed"]
    x = np.arange(len(buckets))
    width = 0.75 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, model in enumerate(models):
        stats_dict = burst_stats[model][feature]
        vals = [stats_dict[bucket]["success_rate"] for bucket in buckets]
        offsets = x + (idx - len(models) / 2 + 0.5) * width
        ax.bar(offsets, vals, width=width * 0.9, color=_model_color(model), alpha=0.85, label=model)
        for xpos, bucket in zip(offsets, buckets):
            count = stats_dict[bucket]["count"]
            ax.text(xpos, 0.02, f"n={count}", rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Episode success rate")
    ax.set_title(f"{FEATURE_SPECS[feature]['label']}: tolerance to low-grounding dips")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def plot_summary(
    trajectory_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    burst_stats: Dict[str, Dict[str, dict]],
    feature: str,
    output_path: Path,
) -> None:
    models = list(trajectory_curves)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for model in models:
        pos = trajectory_curves[model][feature]["positions"]
        ax.plot(pos, trajectory_curves[model][feature]["success"], color=_model_color(model), linewidth=2, label=f"{model} success")
        ax.plot(pos, trajectory_curves[model][feature]["failure"], color=_model_color(model), linewidth=2, linestyle="--", label=f"{model} failure")
    ax.set_title("Mean trajectories")
    ax.set_xlabel("Episode progress")
    ax.set_ylabel(FEATURE_SPECS[feature]["label"])
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    ax = axes[1]
    buckets = ["none", "transient", "persistent", "mixed"]
    x = np.arange(len(buckets))
    width = 0.75 / max(len(models), 1)
    for idx, model in enumerate(models):
        vals = [burst_stats[model][feature][bucket]["success_rate"] for bucket in buckets]
        offsets = x + (idx - len(models) / 2 + 0.5) * width
        ax.bar(offsets, vals, width=width * 0.9, color=_model_color(model), alpha=0.85, label=model)
        for xpos, bucket in zip(offsets, buckets):
            count = burst_stats[model][feature][bucket]["count"]
            ax.text(xpos, 0.02, f"n={count}", rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Dip tolerance")
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Temporal reliance: {FEATURE_SPECS[feature]['label']}",
        fontsize=11,
        fontweight="bold",
    )
    _save(fig, output_path)


def _print_model_summary(
    model: str,
    episodes: Sequence[dict],
    feature: str,
    burst_stats: Dict[str, dict],
) -> None:
    success_rate = float(np.mean([ep["success"] for ep in episodes])) if episodes else float("nan")
    print(f"\n{'=' * 72}")
    print(f"Model: {model} | feature: {feature}")
    print(f"Episodes: {len(episodes)} | success rate: {success_rate:.1%}")
    print("Dip buckets:")
    for bucket in ("none", "transient", "persistent", "mixed"):
        bucket_stats = burst_stats[bucket]
        print(
            f"  {bucket:>10}: n={bucket_stats['count']:>3} "
            f"success={bucket_stats['success_rate']:.3f}"
        )
    meta = burst_stats["_threshold"]
    print(
        f"Low-signal threshold: q={meta['low_quantile']:.2f} "
        f"-> {meta['threshold']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        metavar="MODEL:PATH",
        default=None,
        help="Model/path pairs. Multiple files for the same model are merged by (task_id, episode_idx, task_description).",
    )
    parser.add_argument(
        "--feature",
        action="append",
        choices=sorted(FEATURE_SPECS),
        help="Feature(s) to analyze. Defaults to IoU only.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Requested layer key (for example layer_25). Defaults to layers_avg when present.",
    )
    parser.add_argument(
        "--burst-quantile",
        type=float,
        default=0.2,
        help="Quantile used to define low-grounding dips.",
    )
    parser.add_argument(
        "--transient-max-len",
        type=int,
        default=2,
        help="Max consecutive low-signal steps counted as transient.",
    )
    parser.add_argument(
        "--persistent-min-len",
        type=int,
        default=5,
        help="Min consecutive low-signal steps counted as persistent.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/temporal_reliance",
        help="Directory for plots and JSON summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.inputs = args.inputs or list(DEFAULT_INPUTS)
    features = args.feature or ["iou"]

    model_paths: Dict[str, List[str]] = defaultdict(list)
    for item in args.inputs:
        if ":" not in item:
            parser.error(f"Each --inputs entry must be MODEL:PATH, got {item!r}")
        model, path = item.split(":", 1)
        model_paths[model].append(path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_episodes: Dict[str, List[dict]] = {}
    for model, paths in model_paths.items():
        episodes = load_model_episodes(paths, features=features, layer=args.layer)
        if not episodes:
            log.warning("No usable episodes for %s", model)
            continue
        suite_groups: Dict[str, List[dict]] = defaultdict(list)
        for ep in episodes:
            suite_groups[ep.get("_suite") or "unknown"].append(ep)
        suite_order = list(SUITE_LABELS.values())
        for suite in sorted(suite_groups, key=lambda s: suite_order.index(s) if s in suite_order else 99):
            key = f"{model} ({suite})"
            eps = suite_groups[suite]
            model_episodes[key] = eps
            for feature in features:
                n_with_feature = sum(1 for ep in eps if feature in ep["signals"])
                log.info("%s: %d episodes loaded, %d with feature=%s", key, len(eps), n_with_feature, feature)

    if not model_episodes:
        log.error("No data loaded.")
        return 1

    trajectory_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
    burst_results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    summary_json: Dict[str, dict] = {}

    for model, episodes in model_episodes.items():
        summary_json[model] = {
            "n_episodes": len(episodes),
            "success_rate": float(np.mean([ep["success"] for ep in episodes])),
            "features": {},
        }
        for feature in features:
            usable = [ep for ep in episodes if feature in ep["signals"]]
            if not usable:
                continue
            traj_curves = normalized_episode_curves(usable, feature=feature)
            burst_stats = burst_tolerance_stats(
                usable,
                feature=feature,
                low_quantile=args.burst_quantile,
                transient_max_len=args.transient_max_len,
                persistent_min_len=args.persistent_min_len,
            )
            if not burst_stats:
                continue

            trajectory_curves[model][feature] = traj_curves
            burst_results[model][feature] = burst_stats

            summary_json[model]["features"][feature] = {
                "n_episodes_with_feature": len(usable),
                "burst_tolerance": burst_stats,
                "selected_layers": sorted({ep["layers"].get(feature) for ep in usable if ep["layers"].get(feature)}),
            }
            _print_model_summary(model, usable, feature, burst_stats)

    for feature in features:
        models = [model for model in model_episodes if feature in trajectory_curves.get(model, {})]
        if not models:
            continue
        plot_episode_trajectories(
            {model: trajectory_curves[model] for model in models},
            feature,
            out_dir / f"{feature}_trajectory_success_vs_failure.png",
        )
        plot_burst_tolerance(
            {model: burst_results[model] for model in models},
            feature,
            out_dir / f"{feature}_dip_tolerance.png",
        )
        plot_summary(
            {model: trajectory_curves[model] for model in models},
            {model: burst_results[model] for model in models},
            feature,
            out_dir / f"{feature}_summary.png",
        )

    out_json = out_dir / "temporal_reliance_results.json"
    with open(out_json, "w") as f:
        json.dump(summary_json, f, indent=2)
    log.info("Saved %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
