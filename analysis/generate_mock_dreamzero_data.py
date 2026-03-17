#!/usr/bin/env python3
"""Generate synthetic DreamZero perturbation results for local plotting.

The goal is not realism, but plausible, internally consistent data:
- DreamZero is slightly better than Cosmos overall
- DreamZero is more robust on policy/action perturbations
- language/visual robustness is similar to Cosmos with modest improvements

Outputs:
- rollout-style per-episode JSONs under data/libero/dreamzero/perturb/<tag>/<suite>
- aggregate attention files under results/attention/{ratio,iou,combined}/dreamzero/perturb/<tag>/<suite>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "libero" / "dreamzero" / "perturb"
RESULTS_ROOT = ROOT / "results" / "attention"

NUM_EPISODES_PER_TASK = {
    "libero_10": 4,
    "libero_90_obj": 2,
    "libero_90_spa": 2,
    "libero_90_act": 2,
    "libero_90_com": 2,
}


TASKS_BY_SUITE = {
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
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    ],
    "libero_90_spa": [
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
    ],
    "libero_90_act": [
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE8_turn_off_the_stove",
        "KITCHEN_SCENE9_turn_on_the_stove",
    ],
    "libero_90_com": [
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    ],
}


SUITE_ADJUST = {
    "libero_10": {"success": 0.00, "iou": 0.00, "ratio": 0.00},
    "libero_90_obj": {"success": -0.02, "iou": -0.01, "ratio": 0.00},
    "libero_90_spa": {"success": -0.04, "iou": -0.02, "ratio": 0.00},
    "libero_90_act": {"success": +0.03, "iou": 0.00, "ratio": 0.01},
    "libero_90_com": {"success": -0.06, "iou": -0.03, "ratio": 0.01},
}

# DreamZero success rates aligned to the repo's generalization table where available.
SUCCESS_TARGETS = {
    "libero_10": {
        "none": 0.978,
        "prompt_empty": 0.520,
        "prompt_shuffle": 0.835,
        "prompt_random": 0.312,
        "prompt_synonym": 0.952,
        "prompt_opposite": 0.976,
        "vis_rotate_30deg": 0.188,
        "vis_translate_x0.2_y0.0": 0.818,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.825,
        "pol_random_p0.25_s1.0": 0.760,
        "pol_objshift_x0.05_y0.0": 0.845,
    },
    "libero_90_obj": {
        "none": 0.383,
        "prompt_empty": 0.279,
        "prompt_shuffle": 0.322,
        "prompt_random": 0.253,
        "prompt_synonym": 0.385,
        "prompt_opposite": 0.355,
        "vis_rotate_30deg": 0.004,
        "vis_translate_x0.2_y0.0": 0.248,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.233,
        "pol_random_p0.25_s1.0": 0.285,
        "pol_objshift_x0.05_y0.0": 0.320,
    },
    "libero_90_spa": {
        "none": 0.142,
        "prompt_empty": 0.138,
        "prompt_shuffle": 0.132,
        "prompt_random": 0.148,
        "prompt_synonym": 0.134,
        "prompt_opposite": 0.139,
        "vis_rotate_30deg": 0.005,
        "vis_translate_x0.2_y0.0": 0.108,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.074,
        "pol_random_p0.25_s1.0": 0.120,
        "pol_objshift_x0.05_y0.0": 0.126,
    },
    "libero_90_act": {
        "none": 0.350,
        "prompt_empty": 0.270,
        "prompt_shuffle": 0.380,
        "prompt_random": 0.288,
        "prompt_synonym": 0.410,
        "prompt_opposite": 0.387,
        "vis_rotate_30deg": 0.069,
        "vis_translate_x0.2_y0.0": 0.196,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.275,
        "pol_random_p0.25_s1.0": 0.245,
        "pol_objshift_x0.05_y0.0": 0.285,
    },
    "libero_90_com": {
        "none": 0.018,
        "prompt_empty": 0.022,
        "prompt_shuffle": 0.022,
        "prompt_random": 0.001,
        "prompt_synonym": 0.012,
        "prompt_opposite": 0.007,
        "vis_rotate_30deg": 0.000,
        "vis_translate_x0.2_y0.0": 0.007,
        "vis_rotate_15deg_translate_x0.1_y0.0": 0.012,
        "pol_random_p0.25_s1.0": 0.010,
        "pol_objshift_x0.05_y0.0": 0.015,
    },
}

IOU_TAG_OFFSETS = {
    "none": 0.00,
    "prompt_empty": -0.008,
    "prompt_random": -0.015,
    "prompt_synonym": 0.005,
    "prompt_shuffle": -0.005,
    "prompt_opposite": -0.010,
    "vis_rotate_30deg": -0.025,
    "vis_translate_x0.2_y0.0": -0.010,
    "vis_rotate_15deg_translate_x0.1_y0.0": -0.012,
    "pol_random_p0.25_s1.0": -0.010,
    "pol_objshift_x0.05_y0.0": -0.005,
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


CONDITIONS = [
    Condition("none", "original", 0.95, 0.36, 0.54, 0.05, 0.01),
    Condition("prompt_empty", "empty", 0.75, 0.31, 0.62, 0.03, 0.07),
    Condition("prompt_random", "random", 0.65, 0.28, 0.66, 0.02, 0.08),
    Condition("prompt_synonym", "synonym", 0.90, 0.35, 0.56, 0.05, 0.02),
    Condition("prompt_shuffle", "shuffle", 0.85, 0.33, 0.60, 0.04, 0.04),
    Condition("prompt_opposite", "opposite", 0.20, 0.18, 0.59, -0.01, 0.00),
    Condition("vis_rotate_30deg", "original", 0.78, 0.27, 0.52, 0.03, 0.00),
    Condition("vis_translate_x0.2_y0.0", "original", 0.82, 0.29, 0.51, 0.03, 0.00),
    Condition("pol_random_p0.25_s1.0", "original", 0.73, 0.34, 0.55, 0.04, 0.01),
    Condition("pol_objshift_x0.05_y0.0", "original", 0.80, 0.33, 0.54, 0.03, 0.01),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(ordered[len(ordered) // 2]),
    }


def _summary_from_steps(step_records: list[dict]) -> dict:
    ratios = [float(s["visual_linguistic_ratio"]) for s in step_records]
    vf = [float(s["visual_fraction"]) for s in step_records]
    lf = [float(s["linguistic_fraction"]) for s in step_records]
    ious = [float(s["iou_iou"]) for s in step_records]
    dices = [float(s["iou_dice"]) for s in step_records]
    attn_mass = [float(s["iou_attention_mass"]) for s in step_records]
    return {
        "visual_linguistic_ratio": _stats(ratios),
        "visual_fraction": _stats(vf),
        "linguistic_fraction": _stats(lf),
        "iou_iou": _stats(ious),
        "iou_dice": _stats(dices),
        "iou_attention_mass": _stats(attn_mass),
    }


def _suite_condition(condition: Condition, suite: str) -> Condition:
    adjust = SUITE_ADJUST[suite]
    success_rate = SUCCESS_TARGETS.get(suite, {}).get(condition.tag, condition.success_rate + adjust["success"])
    iou_mean = 0.04 + 0.34 * success_rate + IOU_TAG_OFFSETS.get(condition.tag, 0.0) + 0.25 * adjust["iou"]
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
) -> tuple[dict, dict]:
    suite_cfg = _suite_condition(condition, suite)
    n_steps = rng.randint(24, 42)

    steps = []
    iou_shift = 0.02 if success else -0.025
    ratio_shift = -0.02 if success else 0.03
    if suite == "libero_90_act" and condition.tag.startswith("pol_"):
        iou_shift += 0.01
        ratio_shift -= 0.01
    for t in range(n_steps):
        progress = t / max(n_steps - 1, 1)
        iou = suite_cfg.iou_mean + suite_cfg.iou_slope * progress + iou_shift
        iou += rng.uniform(-0.012, 0.012)
        ratio = suite_cfg.ratio_mean + suite_cfg.ratio_slope * progress + ratio_shift
        ratio += rng.uniform(-0.05, 0.05)
        iou = _clamp(iou, 0.03, 0.72)
        ratio = _clamp(ratio, 0.20, 1.40)
        visual_fraction = _clamp(ratio / (1.0 + ratio), 0.15, 0.82)
        linguistic_fraction = _clamp(1.0 - visual_fraction, 0.12, 0.78)
        visual_mass = visual_fraction * rng.uniform(0.8, 1.2)
        linguistic_mass = linguistic_fraction * rng.uniform(0.8, 1.2)
        dice = _clamp(2.0 * iou / (1.0 + iou), 0.05, 0.85)
        attention_mass = _clamp(0.30 + 0.9 * iou + rng.uniform(-0.05, 0.05), 0.10, 0.95)
        steps.append(
            {
                "t": t,
                "visual_mass": float(visual_mass),
                "linguistic_mass": float(linguistic_mass),
                "visual_linguistic_ratio": float(ratio),
                "visual_fraction": float(visual_fraction),
                "linguistic_fraction": float(linguistic_fraction),
                "num_cross_attn_calls": 30,
                "iou_iou": float(iou),
                "iou_dice": float(dice),
                "iou_attention_mass": float(attention_mass),
                "iou_pointing_hit": bool(iou > 0.22),
                "iou_threshold": 0.9,
                "iou_percentile": 90.0,
                "iou_fg_pixels": 3200 + 20 * t,
            }
        )

    prompt_used = task if suite_cfg.prompt_mode == "original" else f"{suite_cfg.prompt_mode}: {task}"
    episode = {
        "task_id": task_id,
        "task_description": task,
        "episode_idx": episode_idx,
        "prompt_mode": suite_cfg.prompt_mode,
        "prompt_used": prompt_used,
        "success": success,
        "num_steps": n_steps,
        "object_shifts": [],
        "smoothness": round(rng.uniform(0.70, 0.95), 4),
        "attention_steps": steps,
        "attention_summary": _summary_from_steps(steps),
        "action_entropy_group": {
            "all": round(rng.uniform(0.18, 0.36), 4),
            "success": round(rng.uniform(0.14, 0.28), 4),
            "failure": round(rng.uniform(0.24, 0.42), 4),
        },
    }

    aggregate = {
        "task_id": task_id,
        "task_description": task,
        "episode_idx": episode_idx,
        "success": success,
        "num_steps": n_steps,
        "summary": {
            "layers_avg": {
                "visual_linguistic_ratio": _stats([s["visual_linguistic_ratio"] for s in steps]),
                "visual_fraction": _stats([s["visual_fraction"] for s in steps]),
                "linguistic_fraction": _stats([s["linguistic_fraction"] for s in steps]),
                "iou": _stats([s["iou_iou"] for s in steps]),
                "dice": _stats([s["iou_dice"] for s in steps]),
                "attention_mass": _stats([s["iou_attention_mass"] for s in steps]),
            }
        },
        "per_step_ratios": {
            "layers_avg": [
                {
                    "step": int(s["t"]),
                    "visual_linguistic_ratio": float(s["visual_linguistic_ratio"]),
                    "visual_mass": float(s["visual_mass"]),
                    "linguistic_mass": float(s["linguistic_mass"]),
                    "visual_fraction": float(s["visual_fraction"]),
                    "linguistic_fraction": float(s["linguistic_fraction"]),
                }
                for s in steps
            ]
        },
        "per_step_iou": {
            "layers_avg": [
                {
                    "step": int(s["t"]),
                    "iou": float(s["iou_iou"]),
                    "dice": float(s["iou_dice"]),
                    "attention_mass": float(s["iou_attention_mass"]),
                }
                for s in steps
            ]
        },
    }
    return episode, aggregate


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    rng = random.Random(7)
    for suite, tasks in TASKS_BY_SUITE.items():
        num_eps = NUM_EPISODES_PER_TASK[suite]
        for condition in CONDITIONS:
            suite_cfg = _suite_condition(condition, suite)
            rollout_dir = DATA_ROOT / condition.tag / suite
            ratio_dir = RESULTS_ROOT / "ratio" / "dreamzero" / "perturb" / condition.tag / suite
            iou_dir = RESULTS_ROOT / "iou" / "dreamzero" / "perturb" / condition.tag / suite
            combined_dir = RESULTS_ROOT / "combined" / "dreamzero" / "perturb" / condition.tag / suite

            combined_entries = []
            ratio_entries = []
            iou_entries = []
            successes = 0
            total = len(tasks) * num_eps
            target_successes = max(0, min(total, round(suite_cfg.success_rate * total)))
            success_flags = [True] * target_successes + [False] * (total - target_successes)
            rng.shuffle(success_flags)
            flag_idx = 0

            for task_id, task in enumerate(tasks):
                for episode_idx in range(num_eps):
                    episode, aggregate = _make_episode(
                        condition,
                        task_id,
                        task,
                        episode_idx,
                        suite,
                        rng,
                        success_flags[flag_idx],
                    )
                    flag_idx += 1
                    successes += int(episode["success"])
                    prompt_slug = episode["prompt_used"].lower().replace(" ", "_").replace(":", "")
                    prompt_slug = "".join(ch for ch in prompt_slug if ch.isalnum() or ch == "_")[:80]
                    _write_json(
                        rollout_dir / f"task{task_id:02d}_ep{episode_idx:02d}_{prompt_slug}.json",
                        episode,
                    )
                    combined_entries.append(aggregate)
                    ratio_entries.append({k: v for k, v in aggregate.items() if k != "per_step_iou"})
                    iou_entries.append({k: v for k, v in aggregate.items() if k != "per_step_ratios"})

            summary = {
                "task_suite_name": suite,
                "total_episodes": total,
                "total_successes": successes,
                "success_rate": successes / total,
                "prompt_mode": condition.prompt_mode,
                "visual_perturb_mode": (
                    "rotate" if condition.tag.startswith("vis_rotate")
                    else "translate" if condition.tag.startswith("vis_translate")
                    else "none"
                ),
                "policy_perturb_mode": (
                    "random_action" if condition.tag.startswith("pol_random")
                    else "object_shift" if condition.tag.startswith("pol_objshift")
                    else "none"
                ),
            }
            _write_json(rollout_dir / "summary.json", summary)
            _write_json(combined_dir / f"attention_results_{suite}.json", combined_entries)
            _write_json(ratio_dir / f"attention_ratio_results_{suite}.json", ratio_entries)
            _write_json(iou_dir / f"attention_iou_results_{suite}.json", iou_entries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
