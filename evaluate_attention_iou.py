"""LIBERO evaluation with attention-segmentation IoU analysis.

Runs live LIBERO rollouts using SegmentationRenderEnv for ground-truth object
masks and a locally-loaded Pi0 model with attention recording.  At every
re-plan step the script computes IoU between the model's visual attention
heatmap and the segmentation masks of task-relevant objects.

Usage examples:
  # Single task, 3 episodes
  python evaluate_attention_iou.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --task-id 0 --num-episodes 3 \
    --output-dir outputs_iou

  # All tasks in libero_10, 5 episodes each, specific layers
  python evaluate_attention_iou.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --num-episodes 5 \
    --layers 0 8 17 --output-dir outputs_iou

  # Save per-step visualizations
  python evaluate_attention_iou.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --task-id 0 --num-episodes 1 \
    --save-viz --output-dir outputs_iou
"""

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

# JAX must be imported before other modules that depend on it
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import SegmentationRenderEnv
from openpi_client import image_tools
from PIL import Image

from example_attention_viz import (
    load_model_from_checkpoint,
    get_paligemma_tokenizer,
)
from openpi.models import model as _model
from openpi.models import gemma

from visualize_attention import (
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
    extract_image_attention,
    create_attention_heatmap,
)
from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    visualize_attention_vs_segmentation,
    visualize_iou_over_episode,
    find_segmentation_key,
    DEFAULT_THRESHOLD_METHODS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MODEL_INPUT_RESOLUTION = 224


# ── Helpers ──────────────────────────────────────────────────────────────────


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (copied from robosuite)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _json_default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def create_libero_observation(
    agentview_img: np.ndarray,
    wrist_img: np.ndarray,
    state: np.ndarray,
    prompt: str,
    max_token_len: int = 256,
    state_dim: int = 7,
    pi05: bool = True,
) -> _model.Observation:
    """Create a model Observation from LIBERO environment data.

    Args:
        agentview_img: Agent-view RGB image, uint8 (H, W, 3). Already rotated
            180 degrees if needed.
        wrist_img: Wrist camera RGB image, uint8 (H, W, 3). Already rotated.
        state: Robot state vector (typically 8D for LIBERO: eef_pos + axisangle + gripper_qpos).
        prompt: Natural language task instruction.
        max_token_len: Maximum prompt token length.
        state_dim: State dimension the model expects.
        pi05: Whether to use Pi0-FAST tokenization.
    """
    # Resize images to model input resolution
    def _resize(img):
        pil = Image.fromarray(img)
        pil = pil.resize((MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION), Image.LANCZOS)
        return np.array(pil)

    base_img = _resize(agentview_img)
    wrist_resized = _resize(wrist_img)

    # Normalize to float [0, 1] and add batch dimension
    base_batch = base_img[None].astype(np.float32) / 255.0
    wrist_batch = wrist_resized[None].astype(np.float32) / 255.0
    dummy_batch = np.zeros_like(base_batch)

    # Tokenize prompt
    text_tok = get_paligemma_tokenizer(max_token_len)
    state_f32 = np.asarray(state, dtype=np.float32)
    tokens, mask = text_tok.tokenize(prompt, state=state_f32 if pi05 else None)
    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([mask], dtype=jnp.bool_)

    state_batch = jnp.array(state_f32[None])

    observation = _model.Observation(
        images={
            "base_0_rgb": jnp.array(base_batch),
            "left_wrist_0_rgb": jnp.array(wrist_batch),
            "right_wrist_0_rgb": jnp.array(dummy_batch),
        },
        image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.array([pi05], dtype=jnp.bool_),  # masked for Pi0, not for Pi0-FAST
        },
        state=state_batch,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return observation


def _get_segmentation_env(task, resolution, seed):
    """Create a SegmentationRenderEnv for a LIBERO task."""
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = SegmentationRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_segmentations="instance",
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _get_robot_state(obs):
    """Extract robot state from LIBERO observation dict."""
    return np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))


# ── Main Evaluation Loop ────────────────────────────────────────────────────


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
    output_dir: str = "outputs_iou",
    episode_prefix: str = "ep",
    num_image_tokens: int = 256,
    threshold_methods=None,
) -> Dict:
    """Run one episode, recording attention + segmentation IoU at each replan step.

    Returns dict with per-step IoU results, episode summary, and success status.
    """
    if threshold_methods is None:
        threshold_methods = DEFAULT_THRESHOLD_METHODS

    # Reset environment
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    # Build object ID map for objects of interest
    obj_of_interest = env.obj_of_interest
    object_ids = {}
    for obj_name in obj_of_interest:
        if obj_name in env.instance_to_id:
            object_ids[obj_name] = env.instance_to_id[obj_name]
    log.info(f"Objects of interest: {object_ids}")

    if not object_ids:
        log.warning("No objects of interest found in segmentation mapping!")

    # Find segmentation key in observation
    seg_key = find_segmentation_key(obs)
    if seg_key is None:
        # Try common patterns
        for k in obs:
            if "seg" in k.lower():
                seg_key = k
                break
    if seg_key is None:
        log.error(f"No segmentation key found in obs. Keys: {list(obs.keys())}")
        return {"error": "no_segmentation_key", "obs_keys": list(obs.keys())}
    log.info(f"Using segmentation key: {seg_key}")

    action_plan = collections.deque()
    step_iou_results = []
    step_indices = []
    replay_images = []

    t = 0
    done = False

    while t < max_steps + num_steps_wait:
        # Wait for objects to settle
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Preprocess images (rotate 180 to match training)
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        # Get segmentation (also rotate to match model's view)
        seg_raw = obs[seg_key]
        seg_rotated = np.ascontiguousarray(seg_raw[::-1, ::-1])

        # Resize images for policy input
        agentview_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(agentview, MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION)
        )
        wrist_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION)
        )

        replay_images.append(agentview_resized)

        is_replan = not action_plan
        if is_replan:
            # Get robot state
            robot_state = _get_robot_state(obs)

            # Create model observation
            observation = create_libero_observation(
                agentview_img=agentview_resized,
                wrist_img=wrist_resized,
                state=robot_state,
                prompt=task_description,
                max_token_len=max_token_len,
                state_dim=state_dim,
                pi05=pi05,
            )

            # Record attention
            enable_attention_recording()
            rng = jax.random.PRNGKey(t)
            actions = model.sample_actions(rng, observation, num_steps=10)
            attention_dict = get_recorded_attention_weights()
            disable_attention_recording()

            # Extract action chunk (first 7 dims for LIBERO)
            action_chunk = np.array(actions[0])  # remove batch dim
            action_chunk = action_chunk[:, :7]    # LIBERO action dim
            action_plan.extend(action_chunk[:replan_steps])

            # Compute attention heatmap and IoU for each layer
            if attention_dict and object_ids:
                for layer_idx in layers:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key not in attention_dict:
                        continue

                    attn = attention_dict[layer_key][0]  # first batch

                    # Extract image attention (query = first action token)
                    text_token_end = num_image_tokens + observation.tokenized_prompt.shape[1]
                    query_token_idx = text_token_end  # first action token

                    image_attn = extract_image_attention(
                        attn,
                        image_token_start=0,
                        image_token_end=num_image_tokens,
                        query_token_idx=query_token_idx,
                        head_idx=None,
                    )

                    # Create heatmap at model input resolution (224x224)
                    # so patch grid (16x16=256 tokens) matches correctly
                    heatmap = create_attention_heatmap(
                        np.array(image_attn),
                        (MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION),
                        patch_size=14,
                    )

                    # Compute IoU
                    iou_result = compute_attention_object_iou(
                        attention_heatmap=heatmap,
                        segmentation_mask=seg_rotated,
                        object_ids=object_ids,
                        threshold_methods=threshold_methods,
                    )
                    iou_result["layer"] = layer_idx
                    iou_result["step"] = t
                    step_iou_results.append(iou_result)
                    step_indices.append(t)

                    # Log headline metric
                    combined_iou = iou_result["combined"].get("percentile_90", {}).get("iou", 0)
                    mass = iou_result["attention_mass"].get("_all_objects", 0)
                    log.info(
                        f"  t={t} layer={layer_idx}: IoU={combined_iou:.3f}, "
                        f"attn_mass={mass:.1%}, pointing={'hit' if iou_result['pointing_hit'] else 'miss'}"
                    )

                    # Optional per-step visualization
                    if save_viz:
                        viz_path = os.path.join(
                            output_dir,
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

        # Step environment
        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        if done:
            break
        t += 1

    # Episode summary
    summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_iou(layer_results)

    # Episode-level IoU evolution plot
    if step_iou_results and save_viz:
        # Plot for each layer
        for layer_idx in layers:
            layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
            layer_steps = [r["step"] for r in layer_results]
            if layer_results:
                fig = visualize_iou_over_episode(
                    step_results=layer_results,
                    step_indices=layer_steps,
                    prompt=task_description,
                    output_path=os.path.join(
                        output_dir, f"{episode_prefix}_layer{layer_idx}_iou_evolution.png"
                    ),
                )
                plt.close(fig)

    return {
        "success": bool(done),
        "num_steps": t,
        "step_iou_results": step_iou_results,
        "summary": summary,
        "objects_of_interest": list(object_ids.keys()),
    }


# ── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation with attention-segmentation IoU analysis"
    )

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--no-pi05", action="store_true")

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=None,
                        help="Specific task ID (default: all tasks)")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes per task")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention / IoU
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 17],
                        help="Transformer layers to analyze")
    parser.add_argument("--num-image-tokens", type=int, default=256)
    parser.add_argument("--threshold-methods", type=str, nargs="+",
                        default=["percentile_90", "percentile_75", "otsu_0"],
                        help="Threshold methods as method_value strings")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs_iou")
    parser.add_argument("--save-viz", action="store_true",
                        help="Save per-step IoU visualizations (slow, disk-heavy)")

    args = parser.parse_args()

    pi05 = not args.no_pi05

    # Parse threshold methods
    threshold_methods = []
    for tm in args.threshold_methods:
        method, value = tm.rsplit("_", 1)
        threshold_methods.append((method, float(value)))

    # Determine max steps per task suite
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map[args.task_suite]

    # Load model
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

    # Initialize LIBERO benchmark
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks

    # Determine which tasks to evaluate
    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = list(range(num_tasks))

    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate results
    all_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        log.info(f"\n{'=' * 70}")
        log.info(f"Task {task_id}: {task_description}")
        log.info(f"{'=' * 70}")

        # Create segmentation env
        env, _ = _get_segmentation_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

            task_slug = task_description.replace(" ", "_")[:60]
            episode_prefix = f"task{task_id}_{task_slug}_ep{ep_idx}"

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
                output_dir=args.output_dir,
                episode_prefix=episode_prefix,
                num_image_tokens=args.num_image_tokens,
                threshold_methods=threshold_methods,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

            # Log episode summary
            log.info(f"  Success: {result.get('success')}")
            log.info(f"  Steps: {result.get('num_steps')}")
            for layer_key, summ in result.get("summary", {}).items():
                iou_mean = summ.get("combined_iou", {}).get("mean", 0)
                mass_mean = summ.get("attention_mass_on_objects", {}).get("mean", 0)
                pointing = summ.get("pointing_accuracy", 0)
                log.info(
                    f"  {layer_key}: IoU={iou_mean:.3f}, "
                    f"attn_mass={mass_mean:.1%}, pointing={pointing:.1%}"
                )

        env.close()

    # Save aggregate results
    # Strip non-serializable step_iou_results for the summary JSON
    serializable_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_iou_results"}
        serializable_results.append(entry)

    results_path = os.path.join(args.output_dir, "iou_results.json")
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {results_path}")

    # Print aggregate summary
    log.info(f"\n{'=' * 70}")
    log.info("AGGREGATE SUMMARY")
    log.info(f"{'=' * 70}")
    successes = sum(1 for r in all_results if r.get("success"))
    log.info(f"Success rate: {successes}/{len(all_results)} ({successes/len(all_results)*100:.1f}%)")

    for layer_idx in args.layers:
        layer_key = f"layer_{layer_idx}"
        ious = [
            r["summary"][layer_key]["combined_iou"]["mean"]
            for r in all_results
            if layer_key in r.get("summary", {})
        ]
        masses = [
            r["summary"][layer_key]["attention_mass_on_objects"]["mean"]
            for r in all_results
            if layer_key in r.get("summary", {})
        ]
        pointings = [
            r["summary"][layer_key]["pointing_accuracy"]
            for r in all_results
            if layer_key in r.get("summary", {})
        ]
        if ious:
            log.info(
                f"  {layer_key}: mean_IoU={np.mean(ious):.3f} +/- {np.std(ious):.3f}, "
                f"mean_mass={np.mean(masses):.1%}, mean_pointing={np.mean(pointings):.1%}"
            )


if __name__ == "__main__":
    main()
