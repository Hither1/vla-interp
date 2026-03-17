"""LIBERO evaluation with joint attention ratio and IoU analysis.

Runs a single LIBERO rollout loop in a segmentation-capable environment and
computes both:
  - visual / linguistic attention ratio
  - attention / segmentation IoU

The script preserves the legacy output filenames so downstream analysis scripts
can continue reading:
  - attention_ratio_results_<suite>.json
  - iou_results_<suite>.json
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from evaluate_attention_ratio import (  # noqa: E402
    TASK_MAX_STEPS,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    MODEL_INPUT_RESOLUTION,
    _json_default,
    create_libero_observation,
    _get_robot_state,
    compute_attention_ratio,
    summarize_episode_ratios,
    visualize_ratio_over_episode,
    _build_avg_step_results as _build_avg_ratio_step_results,
)
from evaluate_attention_iou import (  # noqa: E402
    _get_segmentation_env,
    _build_avg_step_results as _build_avg_iou_step_results,
)
from attention_iou import (  # noqa: E402
    compute_attention_object_iou,
    summarize_episode_iou,
    visualize_attention_vs_segmentation,
    visualize_iou_over_episode,
    find_segmentation_key,
    DEFAULT_THRESHOLD_METHODS,
)
from evaluate_attention_ratio import benchmark, load_model_from_checkpoint, jax, plt  # noqa: E402
from evaluate_attention_ratio import image_tools  # noqa: E402
from visualize_attention import (  # noqa: E402
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
    extract_image_attention,
    create_attention_heatmap,
)
from visual_perturbations import VisualPerturbConfig, perturb_image  # noqa: E402
from prompt_perturbations import perturb_prompt  # noqa: E402
from policy_perturbations import (  # noqa: E402
    PolicyPerturbConfig,
    apply_object_shift,
    maybe_perturb_action,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_object_ids(env) -> Dict[str, int]:
    object_ids = {}
    for obj_name in env.obj_of_interest:
        if obj_name in env.instance_to_id:
            object_ids[obj_name] = env.instance_to_id[obj_name]

    if object_ids:
        return object_ids

    log.warning("No exact matches found. Attempting fuzzy matching...")
    log.warning(f"  obj_of_interest: {list(env.obj_of_interest)}")
    log.warning(f"  instance_to_id keys: {list(env.instance_to_id.keys())}")
    for obj_name in env.obj_of_interest:
        matches = [k for k in env.instance_to_id.keys() if obj_name.lower() in k.lower()]
        if matches:
            for match in matches:
                object_ids[match] = env.instance_to_id[match]
            log.info(f"  Fuzzy matched '{obj_name}' -> {matches}")
    return object_ids


def run_episode(
    env,
    model,
    model_config,
    task_description: str,
    initial_state: np.ndarray,
    layers: List[int],
    max_steps: int,
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    max_token_len: int = 256,
    state_dim: int = 7,
    pi05: bool = True,
    save_viz: bool = False,
    ratio_output_dir: str = "results/attention_ratio",
    iou_output_dir: str = "results/attention/iou",
    episode_prefix: str = "ep",
    num_image_tokens: int = 256,
    threshold_methods=None,
    prompt_used: Optional[str] = None,
    vis_cfg=None,
    policy_cfg=None,
    policy_rng=None,
) -> Dict:
    if threshold_methods is None:
        threshold_methods = DEFAULT_THRESHOLD_METHODS
    effective_prompt = prompt_used if prompt_used is not None else task_description

    obs = env.reset()
    obs = env.set_init_state(initial_state)

    if policy_cfg is not None and policy_rng is not None:
        apply_object_shift(env, policy_cfg, policy_rng)

    object_ids = _build_object_ids(env)
    log.info(f"Objects of interest: {object_ids}")
    if not object_ids:
        log.warning("No objects of interest found in segmentation mapping even after fuzzy matching!")
        log.warning(f"  Available instance_to_id: {env.instance_to_id}")

    seg_key = find_segmentation_key(obs)
    if seg_key is None:
        for k in obs:
            if "seg" in k.lower():
                seg_key = k
                break
    if seg_key is None:
        log.error(f"No segmentation key found in obs. Keys: {list(obs.keys())}")
        return {"error": "no_segmentation_key", "obs_keys": list(obs.keys())}
    log.info(f"Using segmentation key: {seg_key}")

    action_plan = []
    step_ratio_results = []
    step_iou_results = []

    t = 0
    done = False

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        seg_rotated = np.ascontiguousarray(obs[seg_key][::-1, ::-1])

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
                prompt=effective_prompt,
                max_token_len=max_token_len,
                state_dim=state_dim,
                pi05=pi05,
            )

            num_text_tokens = int(observation.tokenized_prompt.shape[1])

            enable_attention_recording()
            rng = jax.random.PRNGKey(t)
            actions = model.sample_actions(rng, observation, num_steps=10)
            attention_dict = get_recorded_attention_weights()
            disable_attention_recording()

            action_chunk = np.array(actions[0])[:, :7]
            action_plan.extend(action_chunk[:replan_steps])

            if attention_dict:
                per_layer_ratio_metrics = []
                per_layer_iou_metrics = []

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
                    per_layer_ratio_metrics.append(
                        {
                            "layer": layer_idx,
                            "ratio": ratio_result["visual_linguistic_ratio"],
                            "visual_mass": ratio_result["visual_mass"],
                            "linguistic_mass": ratio_result["linguistic_mass"],
                        }
                    )

                    if object_ids:
                        query_token_idx = num_image_tokens + num_text_tokens
                        image_attn = extract_image_attention(
                            attn,
                            image_token_start=0,
                            image_token_end=num_image_tokens,
                            query_token_idx=query_token_idx,
                            head_idx=None,
                        )
                        heatmap = create_attention_heatmap(
                            np.array(image_attn),
                            (MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION),
                            patch_size=14,
                        )

                        iou_result = compute_attention_object_iou(
                            attention_heatmap=heatmap,
                            segmentation_mask=seg_rotated,
                            object_ids=object_ids,
                            threshold_methods=threshold_methods,
                        )
                        iou_result["layer"] = layer_idx
                        iou_result["step"] = t
                        step_iou_results.append(iou_result)

                        combined_iou = float(iou_result["combined"].get("percentile_90", {}).get("iou", 0.0))
                        mass = float(iou_result["attention_mass"].get("_all_objects", 0.0))
                        per_layer_iou_metrics.append(
                            {
                                "layer": layer_idx,
                                "iou": combined_iou,
                                "mass": mass,
                                "pointing_hit": bool(iou_result.get("pointing_hit", False)),
                            }
                        )

                        if save_viz:
                            viz_path = os.path.join(
                                iou_output_dir,
                                f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou.png",
                            )
                            fig = visualize_attention_vs_segmentation(
                                frame_rgb=agentview,
                                attention_heatmap=heatmap,
                                segmentation_mask=seg_rotated,
                                object_ids=object_ids,
                                iou_results=iou_result,
                                layer_idx=layer_idx,
                                output_path=viz_path,
                            )
                            plt.close(fig)

                if per_layer_ratio_metrics:
                    if len(layers) > 1:
                        avg_ratio = float(np.mean([m["ratio"] for m in per_layer_ratio_metrics if np.isfinite(m["ratio"])]))
                        avg_visual = float(np.mean([m["visual_mass"] for m in per_layer_ratio_metrics]))
                        avg_linguistic = float(np.mean([m["linguistic_mass"] for m in per_layer_ratio_metrics]))
                        used_layers = [m["layer"] for m in per_layer_ratio_metrics]
                        log.info(
                            f"  t={t} layers={used_layers}: "
                            f"AVG ratio={avg_ratio:.3f}, visual_mass={avg_visual:.3f}, "
                            f"linguistic_mass={avg_linguistic:.3f}"
                        )
                    else:
                        m = per_layer_ratio_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']}: ratio={m['ratio']:.3f}, "
                            f"visual_mass={m['visual_mass']:.3f}, linguistic_mass={m['linguistic_mass']:.3f}"
                        )

                if per_layer_iou_metrics:
                    if len(layers) > 1:
                        avg_iou = float(np.mean([m["iou"] for m in per_layer_iou_metrics]))
                        avg_mass = float(np.mean([m["mass"] for m in per_layer_iou_metrics]))
                        hit_rate = float(
                            np.mean([1.0 if m["pointing_hit"] else 0.0 for m in per_layer_iou_metrics])
                        )
                        used_layers = [m["layer"] for m in per_layer_iou_metrics]
                        log.info(
                            f"  t={t} layers={used_layers}: "
                            f"AVG IoU={avg_iou:.3f}, AVG attn_mass={avg_mass:.1%}, "
                            f"pointing_hit_rate={hit_rate:.1%}"
                        )
                    else:
                        m = per_layer_iou_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']}: IoU={m['iou']:.3f}, "
                            f"attn_mass={m['mass']:.1%}, pointing={'hit' if m['pointing_hit'] else 'miss'}"
                        )

        action = action_plan.pop(0)
        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(np.asarray(action, dtype=np.float32), policy_cfg, policy_rng)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    ratio_summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
        if layer_results:
            ratio_summary[f"layer_{layer_idx}"] = summarize_episode_ratios(layer_results)
    if len(layers) > 1 and step_ratio_results:
        avg_step_results = _build_avg_ratio_step_results(step_ratio_results)
        if avg_step_results:
            ratio_summary["layers_avg"] = summarize_episode_ratios(avg_step_results)

    iou_summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
        if layer_results:
            iou_summary[f"layer_{layer_idx}"] = summarize_episode_iou(layer_results)
    if len(layers) > 1 and step_iou_results:
        avg_step_results = _build_avg_iou_step_results(step_iou_results)
        if avg_step_results:
            iou_summary["layers_avg"] = summarize_episode_iou(avg_step_results)

    if step_ratio_results and save_viz:
        if len(layers) > 1:
            avg_step_results = _build_avg_ratio_step_results(step_ratio_results)
            if avg_step_results:
                avg_steps = [r["step"] for r in avg_step_results]
                fig = visualize_ratio_over_episode(
                    step_results=avg_step_results,
                    step_indices=avg_steps,
                    prompt=task_description,
                    output_path=os.path.join(
                        ratio_output_dir, f"{episode_prefix}_layers_avg_ratio_evolution.png"
                    ),
                )
                plt.close(fig)
        for layer_idx in layers:
            layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
            if layer_results:
                fig = visualize_ratio_over_episode(
                    step_results=layer_results,
                    step_indices=[r["step"] for r in layer_results],
                    prompt=effective_prompt,
                    output_path=os.path.join(
                        ratio_output_dir, f"{episode_prefix}_layer{layer_idx}_ratio_evolution.png"
                    ),
                )
                plt.close(fig)

    if step_iou_results and save_viz:
        if len(layers) > 1:
            avg_step_results = _build_avg_iou_step_results(step_iou_results)
            if avg_step_results:
                fig = visualize_iou_over_episode(
                    step_results=avg_step_results,
                    step_indices=[r["step"] for r in avg_step_results],
                    prompt=effective_prompt,
                    output_path=os.path.join(
                        iou_output_dir, f"{episode_prefix}_layers_avg_iou_evolution.png"
                    ),
                )
                plt.close(fig)
        for layer_idx in layers:
            layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
            if layer_results:
                fig = visualize_iou_over_episode(
                    step_results=layer_results,
                    step_indices=[r["step"] for r in layer_results],
                    prompt=effective_prompt,
                    output_path=os.path.join(
                        iou_output_dir, f"{episode_prefix}_layer{layer_idx}_iou_evolution.png"
                    ),
                )
                plt.close(fig)

    return {
        "success": bool(done),
        "num_steps": t,
        "step_ratio_results": step_ratio_results,
        "step_iou_results": step_iou_results,
        "ratio_summary": ratio_summary,
        "iou_summary": iou_summary,
        "objects_of_interest": list(object_ids.keys()),
        "prompt_used": prompt_used if prompt_used is not None else task_description,
    }


def _serialize_ratio_results(all_results: List[Dict], layers: List[int]) -> List[Dict]:
    serializable_results = []
    for r in all_results:
        entry = {
            k: v
            for k, v in r.items()
            if k not in {"step_ratio_results", "step_iou_results", "ratio_summary", "iou_summary"}
        }
        entry["summary"] = r.get("ratio_summary", {})

        if "step_ratio_results" in r:
            per_step_ratios = {}
            for layer_idx in layers:
                layer_key = f"layer_{layer_idx}"
                layer_results = [res for res in r["step_ratio_results"] if res.get("layer") == layer_idx]
                if layer_results:
                    per_step_ratios[layer_key] = [
                        {
                            "step": int(res["step"]),
                            "visual_linguistic_ratio": float(res["visual_linguistic_ratio"]),
                            "visual_mass": float(res["visual_mass"]),
                            "linguistic_mass": float(res["linguistic_mass"]),
                            "action_mass": float(res["action_mass"]),
                            "visual_fraction": float(res["visual_fraction"]),
                            "linguistic_fraction": float(res["linguistic_fraction"]),
                            "action_fraction": float(res["action_fraction"]),
                        }
                        for res in layer_results
                    ]

            if len(layers) > 1:
                avg_step_results = _build_avg_ratio_step_results(r["step_ratio_results"])
                if avg_step_results:
                    per_step_ratios["layers_avg"] = [
                        {
                            "step": int(res["step"]),
                            "visual_linguistic_ratio": float(res["visual_linguistic_ratio"]),
                            "visual_mass": float(res["visual_mass"]),
                            "linguistic_mass": float(res["linguistic_mass"]),
                            "action_mass": float(res["action_mass"]),
                            "visual_fraction": float(res["visual_fraction"]),
                            "linguistic_fraction": float(res["linguistic_fraction"]),
                            "action_fraction": float(res["action_fraction"]),
                        }
                        for res in avg_step_results
                    ]
            entry["per_step_ratios"] = per_step_ratios

        serializable_results.append(entry)
    return serializable_results


def _serialize_iou_results(all_results: List[Dict], layers: List[int]) -> Dict:
    serializable_results = []
    for r in all_results:
        entry = {
            k: v
            for k, v in r.items()
            if k not in {"step_ratio_results", "step_iou_results", "ratio_summary", "iou_summary"}
        }
        entry["summary"] = r.get("iou_summary", {})

        if "step_iou_results" in r:
            per_step_iou = {}
            for layer_idx in layers:
                layer_key = f"layer_{layer_idx}"
                layer_results = [res for res in r["step_iou_results"] if res.get("layer") == layer_idx]
                if layer_results:
                    per_step_iou[layer_key] = [
                        {
                            "step": int(res["step"]),
                            "combined_iou": float(res["combined"].get("percentile_90", {}).get("iou", 0.0)),
                            "combined_dice": float(res["combined"].get("percentile_90", {}).get("dice", 0.0)),
                            "attention_mass": float(res["attention_mass"].get("_all_objects", 0.0)),
                            "pointing_hit": bool(res.get("pointing_hit", False)),
                            "per_object_iou": {
                                obj_name: float(obj_metrics.get("percentile_90", {}).get("iou", 0.0))
                                for obj_name, obj_metrics in res.get("per_object", {}).items()
                            },
                        }
                        for res in layer_results
                    ]

            if len(layers) > 1:
                avg_step_results = _build_avg_iou_step_results(r["step_iou_results"])
                if avg_step_results:
                    per_step_iou["layers_avg"] = [
                        {
                            "step": int(res["step"]),
                            "combined_iou": float(res["combined"].get("percentile_90", {}).get("iou", 0.0)),
                            "combined_dice": float(res["combined"].get("percentile_90", {}).get("dice", 0.0)),
                            "attention_mass": float(res["attention_mass"].get("_all_objects", 0.0)),
                            "pointing_hit": bool(res.get("pointing_hit", False)),
                        }
                        for res in avg_step_results
                    ]
            entry["per_step_iou"] = per_step_iou

        serializable_results.append(entry)
    return {"metric": "iou", "results": serializable_results}


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation with joint attention ratio and IoU analysis"
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--no-pi05", action="store_true")

    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--layers", type=int, nargs="+", default=[15, 16, 17])
    parser.add_argument("--num-image-tokens", type=int, default=256)
    parser.add_argument(
        "--threshold-methods",
        type=str,
        nargs="+",
        default=["percentile_90", "percentile_75", "otsu_0"],
        help="Threshold methods as method_value strings",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="original",
        choices=["original", "empty", "shuffle", "random", "synonym", "opposite", "custom"],
    )
    parser.add_argument("--custom-prompt", type=str, default="")

    parser.add_argument(
        "--visual-perturb-mode",
        type=str,
        default="none",
        choices=["none", "rotate", "translate", "rotate_translate"],
    )
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--translate-x-frac", type=float, default=0.0)
    parser.add_argument("--translate-y-frac", type=float, default=0.0)

    parser.add_argument(
        "--policy-perturb-mode",
        type=str,
        default="none",
        choices=["none", "random_action", "object_shift"],
    )
    parser.add_argument("--random-action-prob", type=float, default=0.0)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--object-shift-x-std", type=float, default=0.0)
    parser.add_argument("--object-shift-y-std", type=float, default=0.0)

    parser.add_argument("--ratio-output-dir", type=str, required=True)
    parser.add_argument("--iou-output-dir", type=str, required=True)
    parser.add_argument("--save-viz", action="store_true")

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

    threshold_methods = []
    for tm in args.threshold_methods:
        method, value = tm.rsplit("_", 1)
        threshold_methods.append((method, float(value)))

    os.makedirs(args.ratio_output_dir, exist_ok=True)
    os.makedirs(args.iou_output_dir, exist_ok=True)

    log.info(f"Visual perturbation: {vis_cfg.as_dict()}")
    log.info(f"Policy perturbation: {policy_cfg.as_dict()}")
    log.info("Loading model from checkpoint...")
    model, cfg, state_dim = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        pi05=pi05,
        dtype="bfloat16",
    )
    log.info(f"Model loaded. action_dim={cfg.action_dim}, state_dim={state_dim}")

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    task_ids = [args.task_id] if args.task_id is not None else list(range(num_tasks))

    all_task_descriptions = []
    for task_idx in range(num_tasks):
        try:
            desc = task_suite.get_task(task_idx).language
            all_task_descriptions.append(str(desc))
        except Exception:
            all_task_descriptions.append(f"task_{task_idx}")
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))

    all_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        log.info(f"\n{'=' * 70}")
        log.info(f"Task {task_id}: {task_description}")
        log.info(f"{'=' * 70}")

        env, _ = _get_segmentation_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

            task_slug = task_description.replace(" ", "_")[:60]
            episode_prefix = f"task{task_id}_{task_slug}_ep{ep_idx}"
            episode_prompt = perturb_prompt(
                str(task_description),
                args.prompt_mode,
                all_task_descriptions,
                custom=args.custom_prompt,
            )
            log.info(f"  prompt_mode={args.prompt_mode!r}  prompt_used={episode_prompt!r}")

            result = run_episode(
                env=env,
                model=model,
                model_config=cfg,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                layers=args.layers,
                max_steps=max_steps,
                num_steps_wait=10,
                replan_steps=args.replan_steps,
                max_token_len=args.max_token_len,
                state_dim=state_dim,
                pi05=pi05,
                save_viz=args.save_viz,
                ratio_output_dir=args.ratio_output_dir,
                iou_output_dir=args.iou_output_dir,
                episode_prefix=episode_prefix,
                num_image_tokens=args.num_image_tokens,
                threshold_methods=threshold_methods,
                prompt_used=episode_prompt,
                vis_cfg=vis_cfg,
                policy_cfg=policy_cfg,
                policy_rng=policy_rng,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            result["prompt_mode"] = args.prompt_mode
            result["custom_prompt"] = args.custom_prompt if args.prompt_mode == "custom" else ""
            result["visual_perturbation"] = vis_cfg.as_dict()
            result["policy_perturbation"] = policy_cfg.as_dict()
            all_results.append(result)

            log.info(f"  Success: {result.get('success')}")
            log.info(f"  Steps: {result.get('num_steps')}")

            ratio_summary = result.get("ratio_summary", {})
            if "layers_avg" in ratio_summary:
                summ = ratio_summary["layers_avg"]
                log.info(
                    f"  ratio layers_avg: ratio={summ.get('visual_linguistic_ratio', {}).get('mean', 0):.3f}, "
                    f"visual_frac={summ.get('visual_fraction', {}).get('mean', 0):.1%}, "
                    f"linguistic_frac={summ.get('linguistic_fraction', {}).get('mean', 0):.1%}"
                )

            iou_summary = result.get("iou_summary", {})
            if "layers_avg" in iou_summary:
                summ = iou_summary["layers_avg"]
                log.info(
                    f"  iou layers_avg: IoU={summ.get('combined_iou', {}).get('mean', 0):.3f}, "
                    f"attn_mass={summ.get('attention_mass_on_objects', {}).get('mean', 0):.1%}, "
                    f"pointing={summ.get('pointing_accuracy', 0):.1%}"
                )

        env.close()

    ratio_results = _serialize_ratio_results(all_results, args.layers)
    ratio_results_path = os.path.join(
        args.ratio_output_dir, f"attention_ratio_results_{args.task_suite}.json"
    )
    with open(ratio_results_path, "w") as f:
        json.dump(ratio_results, f, indent=2, default=_json_default)
    log.info(f"\nRatio results saved to {ratio_results_path}")

    iou_results = _serialize_iou_results(all_results, args.layers)
    iou_results_path = os.path.join(args.iou_output_dir, f"iou_results_{args.task_suite}.json")
    with open(iou_results_path, "w") as f:
        json.dump(iou_results, f, indent=2, default=_json_default)
    log.info(f"IoU results saved to {iou_results_path}")


if __name__ == "__main__":
    main()
