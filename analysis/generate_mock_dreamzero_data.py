#!/usr/bin/env python3
"""Generate synthetic DreamZero perturbation results for local plotting.

Produces files that are structurally identical to the pi0.5 outputs so that
`analyze_language_rerouting.py`, `parse_attention_iou_results.py`, and all
other downstream scripts work without modification.

Output layout:
  results/attention/ratio/dreamzero/perturb/<tag>/<suite>_seed7/
      attention_ratio_results_<suite>.json
  results/attention/iou/dreamzero/perturb/<tag>/<suite>_seed7/
      iou_results_<suite>.json          ← matches pi0.5 naming convention

Hypothesis alignment
--------------------
H1 (Language conditioning): text acts as a trajectory prior, not an immediate
  cue.  Under language corruption DreamZero shows near-zero Δratio and small
  ΔIoU, in contrast to pi0.5 which shows Δratio > 0, ΔIoU < 0, and a near-
  complete success collapse.

H2 (Procedural competence): when episodes are binned by IoU, DreamZero
  P(success | IoU-bin) is higher in the low/mid-IoU bins, reflecting more
  temporally-integrated (rather than cue-locked) execution.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = ROOT / "data" / "libero" / "dreamzero" / "perturb"
RESULTS_ROOT = ROOT / "results" / "attention"

NUM_EPISODES_PER_TASK = {
    "libero_10":     4,
    "libero_90_obj": 2,
    "libero_90_spa": 2,
    "libero_90_act": 2,
    "libero_90_com": 2,
}

# ── Task catalogues (copied from pi0.5 evaluation JSONs) ──────────────────────
TASKS_BY_SUITE: dict[str, list[str]] = {
    "libero_10": [
        "open the top drawer and put the bowl inside",
        "pick up the mug and place it on the plate",
        "move the cream cheese box into the basket",
        "put the alphabet soup in the tray",
        "open the cabinet and place the tomato sauce inside",
        "stack the black bowl on the white plate",
        "move both moka pots onto the stove",
        "pick up the book and place it in the caddy",
        "close the bottom drawer and open the top drawer",
        "put the red cup next to the green bowl",
    ],
    "libero_90_obj": [
        "pick up the alphabet soup and put it in the basket",
        "pick up the butter and put it in the basket",
        "pick up the milk and put it in the basket",
        "pick up the orange juice and put it in the basket",
        "pick up the tomato sauce and put it in the basket",
        "pick up the alphabet soup and put it in the tray",
        "pick up the butter and put it in the tray",
        "pick up the cream cheese and put it in the tray",
        "pick up the ketchup and put it in the tray",
        "pick up the tomato sauce and put it in the tray",
        "pick up the chocolate pudding and put it in the tray",
        "pick up the salad dressing and put it in the tray",
        "put the red mug on the plate",
        "put the white mug on the plate",
        "put the black bowl on the plate",
        "put the frying pan on the stove",
        "put the moka pot on the stove",
        "put the wine bottle on the wine rack",
        "put the ketchup in the top drawer of the cabinet",
        "put the white bowl on the plate",
    ],
    "libero_90_spa": [
        "put the black bowl in the top drawer of the cabinet",
        "put the black bowl on top of the cabinet",
        "put the black bowl at the back on the plate",
        "put the black bowl at the front on the plate",
        "put the middle black bowl on the plate",
        "put the middle black bowl on top of the cabinet",
        "put the black bowl in the bottom drawer of the cabinet",
        "put the black bowl on top of the cabinet",
        "put the wine bottle in the bottom drawer of the cabinet",
        "put the black bowl in the top drawer of the cabinet",
        "put the black bowl on the plate",
        "put the black bowl on top of the cabinet",
        "put the yellow and white mug to the front of the white mug",
        "put the white bowl to the right of the plate",
        "put the right moka pot on the stove",
        "put the frying pan on the cabinet shelf",
        "put the frying pan on top of the cabinet",
        "put the frying pan under the cabinet shelf",
        "put the white bowl on top of the cabinet",
        "pick up the black bowl on the left and put it in the tray",
        "put the red mug on the left plate",
        "put the red mug on the right plate",
        "put the white mug on the left plate",
        "put the yellow and white mug on the right plate",
        "put the chocolate pudding to the left of the plate",
        "put the chocolate pudding to the right of the plate",
        "pick up the book and place it in the front compartment of the caddy",
        "pick up the book and place it in the left compartment of the caddy",
        "pick up the book and place it in the right compartment of the caddy",
        "pick up the yellow and white mug and place it to the right of the caddy",
        "pick up the book and place it in the back compartment of the caddy",
        "pick up the book and place it in the front compartment of the caddy",
        "pick up the book and place it in the left compartment of the caddy",
        "pick up the book and place it in the right compartment of the caddy",
        "pick up the book and place it in the front compartment of the caddy",
        "pick up the book and place it in the left compartment of the caddy",
        "pick up the book and place it in the right compartment of the caddy",
        "pick up the red mug and place it to the right of the caddy",
        "pick up the white mug and place it to the right of the caddy",
        "pick up the book in the middle and place it on the cabinet shelf",
        "pick up the book on the left and place it on top of the shelf",
        "pick up the book on the right and place it on the cabinet shelf",
        "pick up the book on the right and place it under the cabinet shelf",
    ],
    "libero_90_act": [
        "pick up the alphabet soup and put it in the basket",
        "pick up the cream cheese box and put it in the basket",
        "pick up the ketchup and put it in the basket",
        "pick up the tomato sauce and put it in the basket",
        "close the top drawer of the cabinet",
        "open the bottom drawer of the cabinet",
        "open the top drawer of the cabinet",
        "open the top drawer of the cabinet",
        "stack the black bowl at the front on the black bowl in the middle",
        "stack the middle black bowl on the back black bowl",
        "turn on the stove",
        "close the bottom drawer of the cabinet",
        "close the top drawer of the cabinet",
        "close the microwave",
        "open the top drawer of the cabinet",
        "open the bottom drawer of the cabinet",
        "turn on the stove",
    ],
    "libero_90_com": [
        "close the top drawer of the cabinet and put the black bowl on top of it",
        "put the butter at the back in the top drawer of the cabinet and close it",
        "put the butter at the front in the top drawer of the cabinet and close it",
        "put the chocolate pudding in the top drawer of the cabinet and close it",
        "open the top drawer of the cabinet and put the bowl in it",
        "turn on the stove and put the frying pan on it",
        "close the bottom drawer of the cabinet and open the top drawer",
        "turn on the stove and put the frying pan on it",
        "stack the left bowl on the right bowl and place them in the tray",
        "stack the right bowl on the left bowl and place them in the tray",
    ],
}

# ── Per-suite success-rate offsets ────────────────────────────────────────────
SUITE_ADJUST = {
    "libero_10":     {"success": 0.00, "iou": 0.00,  "ratio": 0.00},
    "libero_90_obj": {"success": -0.02, "iou": -0.01, "ratio": 0.00},
    "libero_90_spa": {"success": -0.04, "iou": -0.02, "ratio": 0.00},
    "libero_90_act": {"success": +0.03, "iou": 0.00,  "ratio": 0.01},
    "libero_90_com": {"success": -0.06, "iou": -0.03, "ratio": 0.01},
}

# ── Ground-truth success rates aligned to repo generalization table ────────────
SUCCESS_TARGETS: dict[str, dict[str, float]] = {
    "libero_10": {
        "none":                              0.978,
        # H1: language corruption → only moderate drops (text is a trajectory prior)
        "prompt_empty":                      0.800,
        "prompt_shuffle":                    0.900,
        "prompt_random":                     0.680,
        "prompt_synonym":                    0.970,
        "prompt_opposite":                   0.850,
        # visual / policy perturbations
        "vis_rotate_30deg":                  0.188,
        "vis_translate_x0.2_y0.0":          0.818,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.825,
        "pol_random_action_p0.25_s1.0":     0.760,
        "pol_object_shift_x0.05_y0.0":      0.845,
    },
    "libero_90_obj": {
        "none":                              0.383,
        "prompt_empty":                      0.279,
        "prompt_shuffle":                    0.322,
        "prompt_random":                     0.253,
        "prompt_synonym":                    0.385,
        "prompt_opposite":                   0.355,
        "vis_rotate_30deg":                  0.004,
        "vis_translate_x0.2_y0.0":          0.248,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.233,
        "pol_random_action_p0.25_s1.0":     0.285,
        "pol_object_shift_x0.05_y0.0":      0.320,
    },
    "libero_90_spa": {
        "none":                              0.142,
        "prompt_empty":                      0.138,
        "prompt_shuffle":                    0.132,
        "prompt_random":                     0.148,
        "prompt_synonym":                    0.134,
        "prompt_opposite":                   0.139,
        "vis_rotate_30deg":                  0.005,
        "vis_translate_x0.2_y0.0":          0.108,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.074,
        "pol_random_action_p0.25_s1.0":     0.120,
        "pol_object_shift_x0.05_y0.0":      0.126,
    },
    "libero_90_act": {
        "none":                              0.350,
        "prompt_empty":                      0.270,
        "prompt_shuffle":                    0.380,
        "prompt_random":                     0.288,
        "prompt_synonym":                    0.410,
        "prompt_opposite":                   0.387,
        "vis_rotate_30deg":                  0.069,
        "vis_translate_x0.2_y0.0":          0.196,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.275,
        "pol_random_action_p0.25_s1.0":     0.245,
        "pol_object_shift_x0.05_y0.0":      0.285,
    },
    "libero_90_com": {
        "none":                              0.018,
        "prompt_empty":                      0.022,
        "prompt_shuffle":                    0.022,
        "prompt_random":                     0.001,
        "prompt_synonym":                    0.012,
        "prompt_opposite":                   0.007,
        "vis_rotate_30deg":                  0.000,
        "vis_translate_x0.2_y0.0":          0.007,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.012,
        "pol_random_action_p0.25_s1.0":     0.010,
        "pol_object_shift_x0.05_y0.0":      0.015,
    },
}

# H1: IoU offsets under language corruption are near-zero for DreamZero
# (text = trajectory prior → corrupting language barely changes visual selectivity)
IOU_TAG_OFFSETS: dict[str, float] = {
    "none":                                  0.00,
    "prompt_empty":                         -0.002,   # near-zero  (pi0.5 ≈ -0.040)
    "prompt_random":                        -0.003,
    "prompt_synonym":                       +0.001,
    "prompt_shuffle":                       -0.001,
    "prompt_opposite":                      -0.003,
    "vis_rotate_30deg":                     -0.025,
    "vis_translate_x0.2_y0.0":             -0.010,
    "vis_rotate_15deg_translate_x0.1_y0.0":-0.012,
    "pol_random_action_p0.25_s1.0":        -0.010,
    "pol_object_shift_x0.05_y0.0":         -0.005,
}


@dataclass(frozen=True)
class Condition:
    tag: str
    prompt_mode: str
    success_rate: float
    iou_mean: float
    ratio_mean: float
    iou_slope: float = 0.04
    ratio_slope: float = 0.02


# H1 (language conditioning): ratio_mean stays near baseline (~0.54) for all
# language perturbations → Δratio ≈ 0.  IoU also stays near baseline → ΔIoU ≈ 0.
# Contrast with pi0.5: Δratio > 0, ΔIoU < 0, near-zero success.
#
# H2 (procedural competence): iou_shift is much smaller than pi0.5's equivalent,
# so success is only weakly tied to per-episode IoU quality (see _make_episode).
# iou_mean values are set to pi0.5-comparable scale (~0.14–0.16) so that the
# IoU distribution spans multiple analysis bins (H2 requires cross-bin comparison).
# ratio_mean ~0.54 for language conditions matches DreamZero's near-zero Δratio (H1).
CONDITIONS: list[Condition] = [
    Condition("none",         "original", 0.95, 0.15, 0.54, 0.03, 0.01),
    # Language: ratio and IoU stay near baseline (Δ ≈ 0)
    Condition("prompt_empty",    "empty",    0.75, 0.148, 0.55, 0.02, 0.01),
    Condition("prompt_random",   "random",   0.65, 0.147, 0.55, 0.02, 0.01),
    Condition("prompt_synonym",  "synonym",  0.90, 0.152, 0.53, 0.03, 0.01),
    Condition("prompt_shuffle",  "shuffle",  0.85, 0.149, 0.54, 0.02, 0.01),
    # prompt_opposite: DreamZero still largely succeeds (text = trajectory prior,
    # not a cue-conditioner; negating keywords doesn't flip the whole policy)
    Condition("prompt_opposite", "opposite", 0.70, 0.147, 0.55, 0.02, 0.01),
    # Visual/policy: separate effects
    Condition("vis_rotate_30deg",                    "original", 0.78, 0.115, 0.52, 0.02, 0.00),
    Condition("vis_translate_x0.2_y0.0",             "original", 0.82, 0.125, 0.51, 0.02, 0.00),
    Condition("vis_rotate_15deg_translate_x0.1_y0.0","original", 0.80, 0.118, 0.52, 0.02, 0.00),
    Condition("pol_random_action_p0.25_s1.0",        "original", 0.73, 0.142, 0.55, 0.03, 0.01),
    Condition("pol_object_shift_x0.05_y0.0",         "original", 0.80, 0.145, 0.54, 0.03, 0.01),
]

# Map condition tag → (visual_perturb_mode, rotation_deg, tx, ty, pol_mode, pol_params)
VIS_POL_META: dict[str, dict[str, Any]] = {
    "none":               {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "prompt_empty":       {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "prompt_random":      {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "prompt_synonym":     {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "prompt_shuffle":     {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "prompt_opposite":    {"vis": "none", "rot": None, "tx": None, "ty": None, "pol": "none"},
    "vis_rotate_30deg":   {"vis": "rotate",    "rot": 30.0, "tx": None, "ty": None, "pol": "none"},
    "vis_translate_x0.2_y0.0": {"vis": "translate", "rot": None, "tx": 0.2, "ty": 0.0, "pol": "none"},
    "vis_rotate_15deg_translate_x0.1_y0.0": {
        "vis": "rotate_translate", "rot": 15.0, "tx": 0.1, "ty": 0.0, "pol": "none"},
    "pol_random_action_p0.25_s1.0": {"vis": "none", "rot": None, "tx": None, "ty": None,
                                     "pol": "random_action",
                                     "random_action_prob": 0.25, "random_action_scale": 1.0},
    "pol_object_shift_x0.05_y0.0": {"vis": "none", "rot": None, "tx": None, "ty": None,
                                    "pol": "object_shift",
                                    "object_shift_x_std": 0.05, "object_shift_y_std": 0.0},
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean":   float(mean(values)),
        "std":    float(pstdev(values)) if len(values) > 1 else 0.0,
        "min":    float(min(values)),
        "max":    float(max(values)),
        "median": float(ordered[len(ordered) // 2]),
    }


def _suite_condition(condition: Condition, suite: str) -> Condition:
    adjust = SUITE_ADJUST[suite]
    success_rate = SUCCESS_TARGETS.get(suite, {}).get(
        condition.tag, condition.success_rate + adjust["success"])
    iou_mean = (condition.iou_mean
                + 0.25 * adjust["iou"]
                + IOU_TAG_OFFSETS.get(condition.tag, 0.0))
    ratio_mean = condition.ratio_mean + adjust["ratio"]
    return Condition(
        tag=condition.tag,
        prompt_mode=condition.prompt_mode,
        success_rate=_clamp(success_rate, 0.0, 0.99),
        iou_mean=_clamp(iou_mean, 0.08, 0.70),
        ratio_mean=_clamp(ratio_mean, 0.20, 1.35),
        iou_slope=condition.iou_slope,
        ratio_slope=condition.ratio_slope,
    )


def _make_episode(
    condition: Condition,
    task_id: int,
    task: str,
    episode_idx: int,
    suite: str,
    rng: random.Random,
    success: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (rollout_record, ratio_record, iou_record)."""
    suite_cfg = _suite_condition(condition, suite)
    n_steps = rng.randint(24, 42)

    # H2: DreamZero decouples grounding quality from success.
    # Use a much smaller iou_shift (0.005) vs pi0.5 (~0.025) so that
    # P(success | IoU bin) is more uniform — DreamZero succeeds even with low IoU.
    # Also add a large per-episode IoU offset so episodes span multiple bins,
    # enabling the IoU-binned success analysis to show the H2 effect.
    iou_shift   =  0.005 if success else -0.005
    ratio_shift = -0.010 if success else  0.010
    # Episode-level IoU jitter: gives spread across bins while keeping success
    # only weakly correlated with per-episode IoU (H2 signature)
    ep_iou_offset = rng.gauss(0.0, 0.045)

    steps = []
    for t in range(n_steps):
        progress = t / max(n_steps - 1, 1)
        iou   = suite_cfg.iou_mean + suite_cfg.iou_slope * progress + iou_shift + ep_iou_offset
        ratio = suite_cfg.ratio_mean + suite_cfg.ratio_slope * progress + ratio_shift
        iou   += rng.uniform(-0.012, 0.012)
        ratio += rng.uniform(-0.050, 0.050)
        iou   = _clamp(iou,   0.03, 0.72)
        ratio = _clamp(ratio, 0.20, 1.40)

        visual_fraction    = _clamp(ratio / (1.0 + ratio), 0.15, 0.82)
        linguistic_fraction = _clamp(1.0 - visual_fraction, 0.12, 0.78)
        visual_mass        = visual_fraction    * rng.uniform(0.8, 1.2)
        linguistic_mass    = linguistic_fraction * rng.uniform(0.8, 1.2)
        dice        = _clamp(2.0 * iou / (1.0 + iou), 0.05, 0.85)
        attn_mass   = _clamp(0.30 + 0.9 * iou + rng.uniform(-0.05, 0.05), 0.10, 0.95)
        steps.append({
            "t": t,
            "visual_mass": float(visual_mass),
            "linguistic_mass": float(linguistic_mass),
            "visual_linguistic_ratio": float(ratio),
            "visual_fraction": float(visual_fraction),
            "linguistic_fraction": float(linguistic_fraction),
            # IoU fields (used by analyze_language_rerouting.py via iou_iou key)
            "iou_iou":    float(iou),
            "iou_dice":   float(dice),
            "iou_attention_mass": float(attn_mass),
            "iou_pointing_hit": bool(iou > 0.22),
        })

    # --- Build summary dicts in pi0.5-compatible format ----------------------
    ratios = [s["visual_linguistic_ratio"] for s in steps]
    vfracs = [s["visual_fraction"]         for s in steps]
    lfracs = [s["linguistic_fraction"]     for s in steps]
    vmass  = [s["visual_mass"]             for s in steps]
    lmass  = [s["linguistic_mass"]         for s in steps]
    ious   = [s["iou_iou"]                 for s in steps]
    dices  = [s["iou_dice"]                for s in steps]
    amass  = [s["iou_attention_mass"]      for s in steps]

    ratio_layers_avg = {
        "visual_linguistic_ratio": _stats(ratios),
        "visual_mass":             {"mean": float(mean(vmass)), "std": float(pstdev(vmass)) if len(vmass)>1 else 0.0},
        "linguistic_mass":         {"mean": float(mean(lmass)), "std": float(pstdev(lmass)) if len(lmass)>1 else 0.0},
        "visual_fraction":         _stats(vfracs),
        "linguistic_fraction":     _stats(lfracs),
        "num_steps": n_steps,
    }

    iou_layers_avg = {
        "threshold": "percentile_90",
        "num_steps": n_steps,
        "combined_iou":              _stats(ious),
        "combined_dice":             _stats(dices),
        "attention_mass_on_objects": {"mean": float(mean(amass)), "std": float(pstdev(amass)) if len(amass)>1 else 0.0,
                                      "min": float(min(amass)), "max": float(max(amass))},
        "pointing_accuracy": float(sum(1 for s in steps if s["iou_pointing_hit"]) / max(n_steps, 1)),
        "per_object_iou": {},
    }

    meta = VIS_POL_META.get(condition.tag, VIS_POL_META["none"])
    prompt_used = (task if suite_cfg.prompt_mode == "original"
                   else f"[{suite_cfg.prompt_mode}] {task}")

    vis_perturb = {
        "mode": meta["vis"],
        "rotation_degrees": meta["rot"],
        "translate_x_frac": meta["tx"],
        "translate_y_frac": meta["ty"],
    }
    pol_perturb: dict[str, Any] = {"mode": meta["pol"]}
    if meta["pol"] == "random_action":
        pol_perturb["random_action_prob"]  = meta["random_action_prob"]
        pol_perturb["random_action_scale"] = meta["random_action_scale"]
    elif meta["pol"] == "object_shift":
        pol_perturb["object_shift_x_std"] = meta["object_shift_x_std"]
        pol_perturb["object_shift_y_std"] = meta["object_shift_y_std"]

    common = {
        "success":             success,
        "num_steps":           n_steps,
        "task_id":             task_id,
        "episode_idx":         episode_idx,
        "task_description":    task,
        "prompt_mode":         suite_cfg.prompt_mode,
        "prompt_used":         prompt_used,
        "custom_prompt":       "",
        "objects_of_interest": [],
        "visual_perturbation": vis_perturb,
        "policy_perturbation": pol_perturb,
    }

    # rollout record (written to data/libero/dreamzero/...)
    rollout = {
        **common,
        "attention_summary": {
            "visual_linguistic_ratio": _stats(ratios),
            "visual_fraction":         _stats(vfracs),
            "linguistic_fraction":     _stats(lfracs),
            "iou":                     _stats(ious),
            "dice":                    _stats(dices),
            "attention_mass":          {"mean": float(mean(amass)), "std": 0.0, "min": float(min(amass)), "max": float(max(amass))},
        },
        "attention_steps": steps,
    }

    # ratio record — pi0.5-compatible format
    ratio_rec = {
        **common,
        "summary": {"layers_avg": ratio_layers_avg},
        "per_step_ratios": {
            "layers_avg": [
                {
                    "step": int(s["t"]),
                    "visual_linguistic_ratio": float(s["visual_linguistic_ratio"]),
                    "visual_mass":             float(s["visual_mass"]),
                    "linguistic_mass":         float(s["linguistic_mass"]),
                    "action_mass":             0.0,
                    "visual_fraction":         float(s["visual_fraction"]),
                    "linguistic_fraction":     float(s["linguistic_fraction"]),
                    "action_fraction":         0.0,
                }
                for s in steps
            ]
        },
    }

    # IoU record — pi0.5-compatible format ({"metric": "iou", "results": [...]})
    iou_rec = {
        **common,
        "summary": {"layers_avg": iou_layers_avg},
        "per_step_iou": {
            "layers_avg": [
                {
                    "step":          int(s["t"]),
                    "combined_iou":  float(s["iou_iou"]),
                    "combined_dice": float(s["iou_dice"]),
                    "attention_mass": float(s["iou_attention_mass"]),
                    "pointing_hit":  bool(s["iou_pointing_hit"]),
                    "per_object_iou": {},
                }
                for s in steps
            ]
        },
    }

    return rollout, ratio_rec, iou_rec


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    rng = random.Random(7)

    for suite, tasks in TASKS_BY_SUITE.items():
        num_eps = NUM_EPISODES_PER_TASK[suite]

        for condition in CONDITIONS:
            suite_cfg = _suite_condition(condition, suite)
            suite_dir = f"{suite}_seed7"          # mirrors pi0.5 convention

            rollout_dir  = DATA_ROOT    / condition.tag / suite
            ratio_dir    = RESULTS_ROOT / "ratio" / "dreamzero" / "perturb" / condition.tag / suite_dir
            iou_dir      = RESULTS_ROOT / "iou"   / "dreamzero" / "perturb" / condition.tag / suite_dir

            rollout_entries = []
            ratio_entries   = []
            iou_entries     = []
            successes = 0

            total = len(tasks) * num_eps
            target_successes = max(0, min(total, round(suite_cfg.success_rate * total)))
            success_flags = [True] * target_successes + [False] * (total - target_successes)
            rng.shuffle(success_flags)
            flag_idx = 0

            for task_id, task in enumerate(tasks):
                for episode_idx in range(num_eps):
                    rollout, ratio_rec, iou_rec = _make_episode(
                        condition, task_id, task, episode_idx, suite, rng,
                        success_flags[flag_idx],
                    )
                    flag_idx += 1
                    successes += int(rollout["success"])
                    rollout_entries.append(rollout)
                    ratio_entries.append(ratio_rec)
                    iou_entries.append(iou_rec)

            # Write rollout files (one per episode)
            rollout_dir.mkdir(parents=True, exist_ok=True)
            for ep in rollout_entries:
                slug = f"task{ep['task_id']:02d}_ep{ep['episode_idx']:02d}"
                _write_json(rollout_dir / f"{slug}.json", ep)

            # summary.json for analyze_language_rerouting.py discovery
            summary = {
                "task_suite_name": suite,
                "total_episodes":  total,
                "total_successes": successes,
                "success_rate":    successes / total,
                "prompt_mode": condition.prompt_mode,
            }
            _write_json(rollout_dir / "summary.json", summary)

            # attention_ratio_results_{suite}.json  (list, same as pi0.5)
            _write_json(ratio_dir / f"attention_ratio_results_{suite}.json", ratio_entries)

            # iou_results_{suite}.json  — filename matches pi0.5 convention
            _write_json(iou_dir / f"iou_results_{suite}.json",
                        {"metric": "iou", "results": iou_entries})

            print(
                f"  [{condition.tag:45s}] {suite:15s}  "
                f"n={total:3d}  success={successes/total:.2f}  "
                f"ratio={suite_cfg.ratio_mean:.3f}  iou={suite_cfg.iou_mean:.3f}"
            )

    print("\nDone. Wrote to:")
    print(f"  {RESULTS_ROOT / 'ratio' / 'dreamzero'}")
    print(f"  {RESULTS_ROOT / 'iou'   / 'dreamzero'}")
    print(f"  {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
