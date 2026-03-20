"""Prepare paired lang-corruption JSON from per-condition ratio results.

Reads results/attention/ratio/{model}/perturb/{condition}/**/*.json and builds
the format expected by plot_lang_corruption.py:

  [
    {
      "modes": {
        "original": {"success": bool, "visual_fraction_mean": float,
                     "linguistic_fraction_mean": float,
                     "temporal_early": float, "temporal_mid": float, "temporal_late": float},
        "empty":    { ... },
        ...
      },
      "deltas": {
        "empty": {
          "vci": float,           # visual_fraction_mean(mode) - visual_fraction_mean(original)
          "delta_success": int,   # success(mode) - success(original)
          "temporal_analysis": {
            "temporal_slope": float,
            "mode_early": float, "mode_mid": float, "mode_late": float,
          },
        },
        ...
      },
    },
    ...
  ]

For DreamZero: full paired records (baseline exists under perturb/none/).
For pi05:      modes-only records (no baseline → no deltas/VCI).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

PROMPT_COND_MAP = {
    "prompt_empty":    "empty",
    "prompt_opposite": "opposite",
    "prompt_random":   "random",
    "prompt_shuffle":  "shuffle",
    "prompt_synonym":  "synonym",
}

RATIO_BASE = "results/attention/ratio"


def _ep_key(ep: dict) -> Tuple:
    return (ep.get("task_id"), ep.get("episode_idx"), ep.get("task_description"))


def load_condition(base_dir: str, condition: str) -> Dict[Tuple, dict]:
    """Load all ratio JSONs for one condition; key by (task_id, ep_idx, task_desc)."""
    pattern = os.path.join(base_dir, condition, "**", "*.json")
    eps: Dict[Tuple, dict] = {}
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        with open(fpath) as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else raw.get("results", [])
        for ep in items:
            if not isinstance(ep, dict):
                continue
            key = _ep_key(ep)
            eps[key] = ep
    return eps


def _get_steps(ep: dict) -> List[dict]:
    psr = ep.get("per_step_ratios", {})
    if isinstance(psr, dict):
        return psr.get("layers_avg", [])
    if isinstance(psr, list):
        return psr
    return []


def episode_stats(ep: dict) -> dict:
    """Compute per-episode summary stats from per_step_ratios."""
    steps = _get_steps(ep)
    vf = [s.get("visual_fraction", float("nan")) for s in steps]
    lf = [s.get("linguistic_fraction", float("nan")) for s in steps]
    vf = [v for v in vf if np.isfinite(v)]
    lf = [v for v in lf if np.isfinite(v)]

    vf_mean = float(np.mean(vf)) if vf else float("nan")
    lf_mean = float(np.mean(lf)) if lf else float("nan")

    n = len(vf)
    if n >= 3:
        t1, t2 = n // 3, 2 * n // 3
        early_mean = float(np.mean(vf[:t1]))
        mid_mean   = float(np.mean(vf[t1:t2]))
        late_mean  = float(np.mean(vf[t2:]))
    elif n > 0:
        early_mean = mid_mean = late_mean = vf_mean
    else:
        early_mean = mid_mean = late_mean = float("nan")

    # Temporal slope: linear regression over normalized step positions
    if n >= 5:
        xs = np.linspace(0.0, 1.0, n)
        slope = float(np.polyfit(xs, vf, 1)[0])
    else:
        slope = float("nan")

    return {
        "success": bool(ep.get("success", False)),
        "visual_fraction_mean": vf_mean,
        "linguistic_fraction_mean": lf_mean,
        "temporal_early": early_mean,
        "temporal_mid": mid_mean,
        "temporal_late": late_mean,
        "temporal_slope": slope,
    }


def build_dreamzero_records(base_dir: str) -> List[dict]:
    """Build fully paired records for DreamZero (has none baseline)."""
    baseline = load_condition(base_dir, "none")
    print(f"  DreamZero baseline (none): {len(baseline)} episodes")

    cond_eps: Dict[str, Dict[Tuple, dict]] = {}
    for dir_name, mode_name in PROMPT_COND_MAP.items():
        eps = load_condition(base_dir, dir_name)
        cond_eps[mode_name] = eps
        overlap = len(set(baseline) & set(eps))
        print(f"  DreamZero {dir_name}: {len(eps)} episodes, {overlap} overlap with none")

    records: List[dict] = []
    for key, base_ep in baseline.items():
        base_stats = episode_stats(base_ep)
        modes: dict = {"original": base_stats}
        deltas: dict = {}

        for mode_name, eps in cond_eps.items():
            if key not in eps:
                continue
            mode_stats = episode_stats(eps[key])
            modes[mode_name] = mode_stats

            vci = (mode_stats["visual_fraction_mean"] - base_stats["visual_fraction_mean"])
            delta_success = int(mode_stats["success"]) - int(base_stats["success"])
            deltas[mode_name] = {
                "vci": vci if np.isfinite(vci) else 0.0,
                "delta_success": delta_success,
                "temporal_analysis": {
                    "temporal_slope": mode_stats["temporal_slope"],
                    "mode_early":     mode_stats["temporal_early"],
                    "mode_mid":       mode_stats["temporal_mid"],
                    "mode_late":      mode_stats["temporal_late"],
                },
            }

        if len(modes) > 1:  # at least original + one mode
            records.append({"modes": modes, "deltas": deltas})

    print(f"  DreamZero: {len(records)} paired records built")
    return records


def build_pi05_records(base_dir: str) -> List[dict]:
    """Build modes-only records for pi05 (no none baseline → no VCI/deltas)."""
    records: List[dict] = []
    for dir_name, mode_name in PROMPT_COND_MAP.items():
        eps = load_condition(base_dir, dir_name)
        if not eps:
            continue
        print(f"  pi0.5 {dir_name}: {len(eps)} episodes")
        for ep in eps.values():
            stats = episode_stats(ep)
            records.append({"modes": {mode_name: stats}, "deltas": {}})

    print(f"  pi0.5: {len(records)} mode-only records built")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio-base", default=RATIO_BASE,
                        help="Root of results/attention/ratio")
    parser.add_argument("--output-dir", default="results/lang_corruption_data",
                        help="Where to write the prepared JSONs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building DreamZero records...")
    dz_records = build_dreamzero_records(os.path.join(args.ratio_base, "dreamzero", "perturb"))
    dz_out = os.path.join(args.output_dir, "lang_corruption_dreamzero.json")
    with open(dz_out, "w") as f:
        json.dump(dz_records, f, indent=2)
    print(f"  Saved {len(dz_records)} records → {dz_out}\n")

    print("Building pi0.5 records...")
    pi_records = build_pi05_records(os.path.join(args.ratio_base, "pi05", "perturb"))
    pi_out = os.path.join(args.output_dir, "lang_corruption_pi05.json")
    with open(pi_out, "w") as f:
        json.dump(pi_records, f, indent=2)
    print(f"  Saved {len(pi_records)} records → {pi_out}\n")

    print(f"Done. Run plot_lang_corruption.py with:")
    print(f"  --dreamzero-results {dz_out}")
    print(f"  --pi05-results {pi_out}")


if __name__ == "__main__":
    main()
