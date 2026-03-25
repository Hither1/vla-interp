"""Language corruption experiment: attention ratio analysis for pi0.5 (JAX).

For each episode, run the same initial state under multiple prompt conditions
and measure delta visual_fraction (VCI), delta success, temporal slope.

Usage:
  python evaluate_lang_corruption_ratio.py \\
    --checkpoint /path/to/pi05_libero \\
    --task-suite libero_10 --num-episodes 5 \\
    --layers 15 16 17 --output-dir results/lang_corruption_ratio
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
_LIBERO_EVAL_DIR = str(_PROJECT_ROOT / "examples" / "libero")
if _LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EVAL_DIR)

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from PIL import Image  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from example_attention_viz import load_model_from_checkpoint, get_paligemma_tokenizer  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import gemma  # noqa: F401, E402
from visualize_attention import (  # noqa: E402
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
)
from visual_perturbations import VisualPerturbConfig, perturb_image  # noqa: E402
from policy_perturbations import (  # noqa: E402
    PolicyPerturbConfig,
    apply_object_shift,
    maybe_perturb_action,
)
from prompt_perturbations import perturb_prompt  # noqa: E402
from evaluate_attention_ratio import (  # noqa: E402
    compute_attention_ratio,
    summarize_episode_ratios,
    create_libero_observation,
    _get_robot_state,
    _quat2axisangle,
    _json_default,
    _build_avg_step_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MODEL_INPUT_RESOLUTION = 224
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "libero_90_obj": 400,
    "libero_90_spa": 400,
    "libero_90_act": 400,
    "libero_90_com": 400,
}
ALL_PROMPT_MODES = ["original", "empty", "shuffle", "random", "synonym", "opposite"]


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def run_episode_single_mode(
    env,
    model,
    model_config,
    prompt: str,
    initial_state,
    layers: List[int],
    max_steps: int,
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    max_token_len: int = 256,
    state_dim: int = 7,
    pi05: bool = True,
    discrete_state_input: bool = False,
    num_image_tokens: int = 256,
    vis_cfg=None,
    policy_cfg=None,
    policy_rng=None,
) -> Dict:
    """Run one episode with a single (possibly perturbed) prompt.

    Returns dict with keys: success, num_steps, visual_fractions,
    linguistic_fractions, per_step_data, visual_fraction_mean,
    linguistic_fraction_mean.
    """
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    if policy_cfg is not None and policy_rng is not None:
        apply_object_shift(env, policy_cfg, policy_rng)

    action_plan = collections.deque()
    step_ratio_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Rotate 180 to match training convention
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        agentview_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(agentview, MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION)
        )
        wrist_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION)
        )

        if vis_cfg is not None and vis_cfg.mode != "none":
            agentview_resized = perturb_image(agentview_resized, vis_cfg)
            wrist_resized = perturb_image(wrist_resized, vis_cfg)

        is_replan = not action_plan
        if is_replan:
            robot_state = _get_robot_state(obs)

            observation = create_libero_observation(
                agentview_img=agentview_resized,
                wrist_img=wrist_resized,
                state=robot_state,
                prompt=prompt,
                max_token_len=max_token_len,
                state_dim=state_dim,
                pi05=pi05,
                discrete_state_input=discrete_state_input,
            )

            num_text_tokens = int(observation.tokenized_prompt.shape[1])

            enable_attention_recording()
            rng = jax.random.PRNGKey(t)
            actions = model.sample_actions(rng, observation, num_steps=10)
            attention_dict = get_recorded_attention_weights()
            disable_attention_recording()

            action_chunk = np.array(actions[0])
            action_chunk = action_chunk[:, :7]
            action_plan.extend(action_chunk[:replan_steps])

            if attention_dict:
                per_layer_metrics = []
                for layer_idx in layers:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key not in attention_dict:
                        continue
                    attn = attention_dict[layer_key][0]
                    ratio_result = compute_attention_ratio(
                        attention_weights=attn,
                        num_image_tokens=num_image_tokens,
                        num_text_tokens=num_text_tokens,
                        query_token_type="action",
                    )
                    ratio_result["layer"] = layer_idx
                    ratio_result["step"] = t
                    ratio_result["num_text_tokens"] = num_text_tokens
                    step_ratio_results.append(ratio_result)
                    per_layer_metrics.append({
                        "layer": layer_idx,
                        "ratio": ratio_result["visual_linguistic_ratio"],
                        "visual_fraction": ratio_result["visual_fraction"],
                        "linguistic_fraction": ratio_result["linguistic_fraction"],
                    })

                if per_layer_metrics:
                    avg_vis = float(np.mean([m["visual_fraction"] for m in per_layer_metrics]))
                    avg_ling = float(np.mean([m["linguistic_fraction"] for m in per_layer_metrics]))
                    fin_ratios = [m["ratio"] for m in per_layer_metrics if np.isfinite(m["ratio"])]
                    avg_ratio = float(np.mean(fin_ratios)) if fin_ratios else 0.0
                    log.info(
                        "  t=%d layers_avg: visual_frac=%.3f linguistic_frac=%.3f ratio=%.3f",
                        t, avg_vis, avg_ling, avg_ratio,
                    )

        action = action_plan.popleft()
        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(
                np.asarray(action, dtype=np.float32), policy_cfg, policy_rng
            )
        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    # Build per-step averaged results across layers
    avg_step_results = _build_avg_step_results(step_ratio_results) if step_ratio_results else []

    visual_fractions = [r["visual_fraction"] for r in avg_step_results]
    linguistic_fractions = [r["linguistic_fraction"] for r in avg_step_results]
    per_step_data = [
        {
            "step": int(r["step"]),
            "visual_fraction": float(r["visual_fraction"]),
            "linguistic_fraction": float(r["linguistic_fraction"]),
            "visual_linguistic_ratio": float(r["visual_linguistic_ratio"]),
        }
        for r in avg_step_results
    ]

    return {
        "success": bool(done),
        "num_steps": t,
        "visual_fractions": visual_fractions,
        "linguistic_fractions": linguistic_fractions,
        "per_step_data": per_step_data,
        "visual_fraction_mean": float(np.mean(visual_fractions)) if visual_fractions else 0.0,
        "linguistic_fraction_mean": float(np.mean(linguistic_fractions)) if linguistic_fractions else 0.0,
    }


def _temporal_slope(per_step_data: List[Dict]) -> Dict:
    """Split into early/mid/late thirds; compute slope via linear regression."""
    if not per_step_data:
        return {"early": 0.0, "mid": 0.0, "late": 0.0, "slope": 0.0}

    n = len(per_step_data)
    early = per_step_data[: max(1, n // 3)]
    mid = per_step_data[n // 3 : max(n // 3 + 1, 2 * n // 3)]
    late = per_step_data[2 * n // 3 :]

    early_mean = float(np.mean([d["visual_fraction"] for d in early])) if early else 0.0
    mid_mean = float(np.mean([d["visual_fraction"] for d in mid])) if mid else 0.0
    late_mean = float(np.mean([d["visual_fraction"] for d in late])) if late else 0.0

    steps = np.array([d["step"] for d in per_step_data], dtype=float)
    vfracs = np.array([d["visual_fraction"] for d in per_step_data], dtype=float)
    if len(steps) >= 2 and steps.std() > 1e-8:
        slope = float(np.polyfit(steps, vfracs, 1)[0])
    else:
        slope = 0.0

    return {"early": early_mean, "mid": mid_mean, "late": late_mean, "slope": slope}


def compute_deltas(mode_results: Dict, baseline_mode: str = "original") -> Dict:
    """Compute per-mode deltas relative to the baseline mode.

    Returns dict keyed by non-baseline mode name, each containing:
      delta_visual_fraction, delta_linguistic_fraction, delta_success,
      vci (Visual Compensation Index), temporal_analysis.
    """
    baseline = mode_results.get(baseline_mode)
    if baseline is None:
        log.warning("Baseline mode %r not found; returning empty deltas.", baseline_mode)
        return {}

    baseline_vfrac = baseline["visual_fraction_mean"]
    baseline_lfrac = baseline["linguistic_fraction_mean"]
    baseline_success = int(baseline["success"])
    baseline_temporal = _temporal_slope(baseline["per_step_data"])

    deltas = {}
    for mode, res in mode_results.items():
        if mode == baseline_mode:
            continue
        delta_vfrac = res["visual_fraction_mean"] - baseline_vfrac
        delta_lfrac = res["linguistic_fraction_mean"] - baseline_lfrac
        delta_success = int(res["success"]) - baseline_success
        mode_temporal = _temporal_slope(res["per_step_data"])

        deltas[mode] = {
            "delta_visual_fraction": float(delta_vfrac),
            "delta_linguistic_fraction": float(delta_lfrac),
            "delta_success": int(delta_success),
            "vci": float(delta_vfrac),
            "temporal_analysis": {
                "baseline_early": float(baseline_temporal["early"]),
                "baseline_mid": float(baseline_temporal["mid"]),
                "baseline_late": float(baseline_temporal["late"]),
                "mode_early": float(mode_temporal["early"]),
                "mode_mid": float(mode_temporal["mid"]),
                "mode_late": float(mode_temporal["late"]),
                "delta_early": float(mode_temporal["early"] - baseline_temporal["early"]),
                "delta_mid": float(mode_temporal["mid"] - baseline_temporal["mid"]),
                "delta_late": float(mode_temporal["late"] - baseline_temporal["late"]),
                "temporal_slope": float(mode_temporal["slope"]),
            },
        }
    return deltas


def run_episode_all_modes(
    env,
    model,
    model_config,
    task_description: str,
    initial_state,
    layers: List[int],
    max_steps: int,
    prompt_modes: List[str],
    all_tasks: List[str],
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    max_token_len: int = 256,
    state_dim: int = 7,
    pi05: bool = True,
    discrete_state_input: bool = False,
    num_image_tokens: int = 256,
    vis_cfg=None,
    policy_cfg=None,
    policy_rng=None,
) -> Dict:
    """Run the episode under all prompt modes; compute deltas vs original.

    Returns dict with mode_results and deltas.
    """
    mode_results: Dict = {}

    for mode in prompt_modes:
        prompt = perturb_prompt(task_description, mode=mode, all_tasks=all_tasks)
        log.info("  Mode=%r  prompt=%r", mode, prompt[:80])

        env.reset()
        env.set_init_state(initial_state)

        result = run_episode_single_mode(
            env=env, model=model, model_config=model_config,
            prompt=prompt, initial_state=initial_state, layers=layers,
            max_steps=max_steps, num_steps_wait=num_steps_wait,
            replan_steps=replan_steps, max_token_len=max_token_len,
            state_dim=state_dim, pi05=pi05, discrete_state_input=discrete_state_input,
            num_image_tokens=num_image_tokens,
            vis_cfg=vis_cfg, policy_cfg=policy_cfg, policy_rng=policy_rng,
        )

        result["prompt"] = prompt
        result["mode"] = mode
        mode_results[mode] = result

        log.info(
            "    -> success=%s steps=%d vfrac=%.3f lfrac=%.3f",
            result["success"], result["num_steps"],
            result["visual_fraction_mean"], result["linguistic_fraction_mean"],
        )

    deltas = compute_deltas(mode_results, baseline_mode="original")
    return {"mode_results": mode_results, "deltas": deltas}


def main():
    parser = argparse.ArgumentParser(
        description="Language corruption experiment: visual compensation index for pi0.5"
    )

    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--no-pi05", action="store_true")

    # LIBERO
    parser.add_argument(
        "--task-suite", type=str, default="libero_10",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
            "libero_90_obj",
            "libero_90_spa",
            "libero_90_act",
            "libero_90_com",
        ],
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 16, 17])
    parser.add_argument("--num-image-tokens", type=int, default=256)

    # Prompt perturbation
    parser.add_argument("--prompt-modes", nargs="+", default=ALL_PROMPT_MODES, choices=ALL_PROMPT_MODES)

    # Visual perturbation
    parser.add_argument("--visual-perturb-mode", type=str, default="none",
                        choices=["none", "rotate", "translate", "rotate_translate"])
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--translate-x-frac", type=float, default=0.0)
    parser.add_argument("--translate-y-frac", type=float, default=0.0)

    # Policy perturbation
    parser.add_argument("--policy-perturb-mode", type=str, default="none",
                        choices=["none", "random_action", "object_shift"])
    parser.add_argument("--random-action-prob", type=float, default=0.0)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--object-shift-x-std", type=float, default=0.0)
    parser.add_argument("--object-shift-y-std", type=float, default=0.0)

    # Output
    parser.add_argument("--output-dir", type=str, default="results/lang_corruption_ratio")

    args = parser.parse_args()
    pi05 = not args.no_pi05
    max_steps = TASK_MAX_STEPS[args.task_suite]

    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    policy_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    policy_rng = np.random.default_rng(args.seed + 9999)

    log.info("Visual perturbation: %s", vis_cfg.as_dict())
    log.info("Policy perturbation: %s", policy_cfg.as_dict())
    log.info("Prompt modes: %s", args.prompt_modes)

    log.info("Loading model from checkpoint...")
    model, cfg, state_dim = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        pi05=pi05,
        discrete_state_input=False,
        dtype="bfloat16",
    )
    log.info("Model loaded. action_dim=%d state_dim=%d", cfg.action_dim, state_dim)

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[args.task_suite]()
    num_tasks = task_suite_obj.n_tasks
    task_ids = [args.task_id] if args.task_id is not None else list(range(num_tasks))
    all_task_descriptions = [task_suite_obj.get_task(i).language for i in range(num_tasks)]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for task_id in task_ids:
        task = task_suite_obj.get_task(task_id)
        initial_states = task_suite_obj.get_task_init_states(task_id)
        task_description = task.language

        log.info("\n%s\nTask %d: %s\n%s", "=" * 70, task_id, task_description, "=" * 70)

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info("--- Episode %d/%d ---", ep_idx + 1, args.num_episodes)

            episode_result = run_episode_all_modes(
                env=env, model=model, model_config=cfg,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                layers=args.layers, max_steps=max_steps,
                prompt_modes=args.prompt_modes, all_tasks=all_task_descriptions,
                num_steps_wait=10, replan_steps=args.replan_steps,
                max_token_len=args.max_token_len, state_dim=state_dim,
                pi05=pi05, discrete_state_input=cfg.discrete_state_input,
                num_image_tokens=args.num_image_tokens,
                vis_cfg=vis_cfg, policy_cfg=policy_cfg, policy_rng=policy_rng,
            )

            modes_serializable = {}
            for mode, res in episode_result["mode_results"].items():
                modes_serializable[mode] = {
                    "success": bool(res["success"]),
                    "num_steps": int(res["num_steps"]),
                    "prompt": res["prompt"],
                    "visual_fraction_mean": float(res["visual_fraction_mean"]),
                    "linguistic_fraction_mean": float(res["linguistic_fraction_mean"]),
                    "per_step_data": res["per_step_data"],
                }

            entry = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": ep_idx,
                "modes": modes_serializable,
                "deltas": episode_result["deltas"],
                "visual_perturbation": vis_cfg.as_dict(),
                "policy_perturbation": policy_cfg.as_dict(),
            }
            all_results.append(entry)

            orig_res = episode_result["mode_results"].get("original", {})
            log.info("  original: success=%s vfrac=%.3f",
                orig_res.get("success"), orig_res.get("visual_fraction_mean", 0.0))
            for mode, delta in episode_result["deltas"].items():
                log.info("  %s: delta_vfrac=%.4f (VCI) delta_success=%+d",
                    mode, delta["delta_visual_fraction"], delta["delta_success"])

        env.close()

    results_path = os.path.join(
        args.output_dir, f"lang_corruption_ratio_{args.task_suite}.json"
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    log.info("\n%s", "=" * 70)
    log.info("AGGREGATE SUMMARY")
    log.info("%s", "=" * 70)

    orig_successes = [
        int(r["modes"]["original"]["success"])
        for r in all_results
        if "original" in r.get("modes", {})
    ]
    if orig_successes:
        log.info("  original: success=%.1f%% (%d/%d)",
            100.0 * np.mean(orig_successes), sum(orig_successes), len(orig_successes))

    non_baseline_modes = [m for m in args.prompt_modes if m != "original"]
    log.info("\n  %-12s  %12s  %12s  %10s", "Mode", "VCI (mean)", "VCI (sem)", "Success%")
    for mode in non_baseline_modes:
        vcis = [
            r["deltas"][mode]["vci"]
            for r in all_results
            if mode in r.get("deltas", {})
        ]
        successes = [
            int(r["modes"][mode]["success"])
            for r in all_results
            if mode in r.get("modes", {})
        ]
        if not vcis:
            continue
        vci_mean = float(np.mean(vcis))
        vci_sem = float(np.std(vcis) / max(1, math.sqrt(len(vcis))))
        succ_pct = 100.0 * float(np.mean(successes)) if successes else float("nan")
        log.info("  %-12s  %12.4f  %12.4f  %9.1f%%", mode, vci_mean, vci_sem, succ_pct)

    log.info("\nSaved %d episode records to %s", len(all_results), results_path)


if __name__ == "__main__":
    main()
