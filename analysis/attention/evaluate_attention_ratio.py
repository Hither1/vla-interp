"""LIBERO evaluation with visual/linguistic attention ratio analysis.

Runs live LIBERO rollouts and computes the ratio of visual attention to linguistic
attention at every re-plan step. This helps understand whether the model is attending
more to visual features or textual instructions when predicting actions.

Attention ratio = (attention mass on image tokens) / (attention mass on text tokens)

Usage examples:
  # Single task, 3 episodes
  python evaluate_attention_ratio.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --task-id 0 --num-episodes 3 \
    --output-dir results/attention_ratio

  # All tasks in libero_10, 5 episodes each, specific layers
  python evaluate_attention_ratio.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --num-episodes 5 \
    --layers 0 8 17 --output-dir results/attention_ratio

  # Save per-step visualizations
  python evaluate_attention_ratio.py \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --task-suite libero_10 --task-id 0 --num-episodes 1 \
    --save-viz --output-dir results/attention_ratio
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
import matplotlib.pyplot as plt  # noqa: E402

# Add src and examples/libero to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
_LIBERO_EVAL_DIR = os.path.join(_PROJECT_ROOT, "examples", "libero")
if _LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EVAL_DIR)

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from PIL import Image  # noqa: E402
from openpi_client import image_tools  # noqa: E402

from example_attention_viz import (  # noqa: E402
    load_model_from_checkpoint,
    get_paligemma_tokenizer,
)
from openpi.models import model as _model  # noqa: E402
from openpi.models import gemma  # noqa: F401, E402

from visualize_attention import (  # noqa: E402
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
    extract_image_attention,
)
from visual_perturbations import VisualPerturbConfig, perturb_image  # noqa: E402
from policy_perturbations import (  # noqa: E402
    PolicyPerturbConfig,
    apply_object_shift,
    maybe_perturb_action,
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
}


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
    """Create a model Observation from LIBERO environment data."""

    def _resize(img):
        pil = Image.fromarray(img)
        pil = pil.resize((MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION), Image.LANCZOS)
        return np.array(pil)

    base_img = _resize(agentview_img)
    wrist_resized = _resize(wrist_img)

    base_batch = base_img[None].astype(np.float32) / 255.0
    wrist_batch = wrist_resized[None].astype(np.float32) / 255.0
    dummy_batch = np.zeros_like(base_batch)

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
            "right_wrist_0_rgb": jnp.array([pi05], dtype=jnp.bool_),
        },
        state=state_batch,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return observation


def _get_libero_env(task, resolution, seed):
    """Create a LIBERO env for a task."""
    from libero.libero.envs import OffScreenRenderEnv

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


def _get_robot_state(obs):
    """Extract robot state from LIBERO observation dict."""
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def compute_attention_ratio(
    attention_weights: np.ndarray,
    num_image_tokens: int,
    num_text_tokens: int,
    query_token_type: str = "action",
) -> Dict:
    """
    Compute the ratio of visual to linguistic attention.

    Args:
        attention_weights: (B, K, G, T, S) attention tensor
        num_image_tokens: Number of image tokens in the sequence
        num_text_tokens: Number of text tokens in the sequence
        query_token_type: Which token to use as query ("action" for first action token)

    Returns:
        Dictionary containing attention masses and ratios
    """
    batch_idx = 0

    # Determine query token index
    if query_token_type == "action":
        # First action token (comes after image and text tokens)
        query_token_idx = num_image_tokens + num_text_tokens
    else:  # "last"
        query_token_idx = -1

    # Average over all heads: (T, S)
    attn = attention_weights[batch_idx].reshape(-1, attention_weights.shape[3], attention_weights.shape[4])
    attn = attn[:, query_token_idx, :].mean(axis=0)  # (S,)

    # Extract attention to image tokens
    image_attn = attn[0:num_image_tokens]
    visual_mass = float(np.sum(image_attn))

    # Extract attention to text tokens
    text_start = num_image_tokens
    text_end = num_image_tokens + num_text_tokens
    text_attn = attn[text_start:text_end]
    linguistic_mass = float(np.sum(text_attn))

    # Extract attention to action tokens (everything after text)
    action_attn = attn[text_end:]
    action_mass = float(np.sum(action_attn))

    # Compute ratio (avoid division by zero)
    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float('inf') if visual_mass > 1e-8 else 0.0

    # Total attention (should be ~1.0 due to softmax)
    total_mass = visual_mass + linguistic_mass + action_mass

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "action_mass": action_mass,
        "total_mass": total_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass / max(total_mass, 1e-8),
        "linguistic_fraction": linguistic_mass / max(total_mass, 1e-8),
        "action_fraction": action_mass / max(total_mass, 1e-8),
    }


def summarize_episode_ratios(step_results: List[Dict]) -> Dict:
    """Summarize attention ratios over an episode."""
    if not step_results:
        return {}

    ratios = [r["visual_linguistic_ratio"] for r in step_results if np.isfinite(r["visual_linguistic_ratio"])]
    visual_masses = [r["visual_mass"] for r in step_results]
    linguistic_masses = [r["linguistic_mass"] for r in step_results]
    visual_fractions = [r["visual_fraction"] for r in step_results]
    linguistic_fractions = [r["linguistic_fraction"] for r in step_results]

    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
            "min": float(np.min(ratios)) if ratios else 0.0,
            "max": float(np.max(ratios)) if ratios else 0.0,
        },
        "visual_mass": {
            "mean": float(np.mean(visual_masses)),
            "std": float(np.std(visual_masses)),
        },
        "linguistic_mass": {
            "mean": float(np.mean(linguistic_masses)),
            "std": float(np.std(linguistic_masses)),
        },
        "visual_fraction": {
            "mean": float(np.mean(visual_fractions)),
            "std": float(np.std(visual_fractions)),
        },
        "linguistic_fraction": {
            "mean": float(np.mean(linguistic_fractions)),
            "std": float(np.std(linguistic_fractions)),
        },
        "num_steps": len(step_results),
    }


def visualize_ratio_over_episode(
    step_results: List[Dict],
    step_indices: List[int],
    prompt: str,
    output_path: str,
) -> plt.Figure:
    """Create a visualization of attention ratios over an episode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Attention Evolution: {prompt[:60]}", fontsize=14, fontweight='bold')

    # Plot 1: Visual/Linguistic Ratio over time
    ax = axes[0, 0]
    ratios = [r["visual_linguistic_ratio"] for r in step_results]
    ax.plot(step_indices, ratios, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal (ratio=1.0)')
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Visual / Linguistic Ratio", fontsize=11)
    ax.set_title("Visual vs Linguistic Attention Ratio", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Attention masses over time
    ax = axes[0, 1]
    visual_masses = [r["visual_mass"] for r in step_results]
    linguistic_masses = [r["linguistic_mass"] for r in step_results]
    action_masses = [r["action_mass"] for r in step_results]
    ax.plot(step_indices, visual_masses, 'o-', label='Visual', linewidth=2, markersize=6)
    ax.plot(step_indices, linguistic_masses, 's-', label='Linguistic', linewidth=2, markersize=6)
    ax.plot(step_indices, action_masses, '^-', label='Action', linewidth=2, markersize=6)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Attention Mass", fontsize=11)
    ax.set_title("Attention Mass by Token Type", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Attention fractions (stacked area)
    ax = axes[1, 0]
    visual_fracs = [r["visual_fraction"] for r in step_results]
    linguistic_fracs = [r["linguistic_fraction"] for r in step_results]
    action_fracs = [r["action_fraction"] for r in step_results]

    ax.fill_between(step_indices, 0, visual_fracs, label='Visual', alpha=0.7)
    ax.fill_between(step_indices, visual_fracs,
                     [v + l for v, l in zip(visual_fracs, linguistic_fracs)],
                     label='Linguistic', alpha=0.7)
    ax.fill_between(step_indices,
                     [v + l for v, l in zip(visual_fracs, linguistic_fracs)],
                     [v + l + a for v, l, a in zip(visual_fracs, linguistic_fracs, action_fracs)],
                     label='Action', alpha=0.7)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Attention Fraction", fontsize=11)
    ax.set_title("Attention Distribution (Stacked)", fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = summarize_episode_ratios(step_results)
    ratio_mean = summary["visual_linguistic_ratio"]["mean"]
    ratio_std = summary["visual_linguistic_ratio"]["std"]
    visual_frac_mean = summary["visual_fraction"]["mean"]
    linguistic_frac_mean = summary["linguistic_fraction"]["mean"]

    stats_text = f"""
    Episode Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Visual/Linguistic Ratio:
      Mean: {ratio_mean:.3f} ± {ratio_std:.3f}
      Median: {summary["visual_linguistic_ratio"]["median"]:.3f}
      Range: [{summary["visual_linguistic_ratio"]["min"]:.3f}, {summary["visual_linguistic_ratio"]["max"]:.3f}]

    Average Attention Fractions:
      Visual: {visual_frac_mean:.1%}
      Linguistic: {linguistic_frac_mean:.1%}
      Action: {summary["action_fraction"]["mean"]:.1%}

    Steps: {summary["num_steps"]}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def _build_avg_step_results(step_ratio_results: List[Dict]) -> List[Dict]:
    """Build per-step averaged results across layers."""
    by_step: Dict[int, List[Dict]] = {}
    for r in step_ratio_results:
        by_step.setdefault(int(r["step"]), []).append(r)

    avg_step_results: List[Dict] = []
    for step, rs in sorted(by_step.items(), key=lambda x: x[0]):
        visual_masses = [r["visual_mass"] for r in rs]
        linguistic_masses = [r["linguistic_mass"] for r in rs]
        action_masses = [r["action_mass"] for r in rs]
        ratios = [r["visual_linguistic_ratio"] for r in rs if np.isfinite(r["visual_linguistic_ratio"])]

        base = dict(rs[0])  # copy
        base["layer"] = "avg"
        base["step"] = step
        base["visual_mass"] = float(np.mean(visual_masses))
        base["linguistic_mass"] = float(np.mean(linguistic_masses))
        base["action_mass"] = float(np.mean(action_masses))
        base["visual_linguistic_ratio"] = float(np.mean(ratios)) if ratios else 0.0

        total = base["visual_mass"] + base["linguistic_mass"] + base["action_mass"]
        base["total_mass"] = total
        base["visual_fraction"] = base["visual_mass"] / max(total, 1e-8)
        base["linguistic_fraction"] = base["linguistic_mass"] / max(total, 1e-8)
        base["action_fraction"] = base["action_mass"] / max(total, 1e-8)

        avg_step_results.append(base)

    return avg_step_results


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
    output_dir: str = "results/attention_ratio",
    episode_prefix: str = "ep",
    num_image_tokens: int = 256,
    vis_cfg: Optional["VisualPerturbConfig"] = None,
    policy_cfg: Optional["PolicyPerturbConfig"] = None,
    policy_rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Run one episode, recording visual/linguistic attention ratios at each replan step."""

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

        # Rotate 180 to match training
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
                prompt=task_description,
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

            action_chunk = np.array(actions[0])
            action_chunk = action_chunk[:, :7]
            action_plan.extend(action_chunk[:replan_steps])

            # Compute attention ratios for each layer
            if attention_dict:
                per_layer_metrics = []

                for layer_idx in layers:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key not in attention_dict:
                        continue

                    attn = attention_dict[layer_key][0]  # first batch

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
                        "visual_mass": ratio_result["visual_mass"],
                        "linguistic_mass": ratio_result["linguistic_mass"],
                    })

                # Log averaged metrics across layers
                if per_layer_metrics:
                    if len(layers) > 1:
                        avg_ratio = float(np.mean([m["ratio"] for m in per_layer_metrics if np.isfinite(m["ratio"])]))
                        avg_visual = float(np.mean([m["visual_mass"] for m in per_layer_metrics]))
                        avg_linguistic = float(np.mean([m["linguistic_mass"] for m in per_layer_metrics]))
                        used_layers = [m["layer"] for m in per_layer_metrics]
                        log.info(
                            f"  t={t} layers={used_layers}: "
                            f"AVG ratio={avg_ratio:.3f}, visual_mass={avg_visual:.3f}, "
                            f"linguistic_mass={avg_linguistic:.3f}"
                        )
                    else:
                        m = per_layer_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']}: ratio={m['ratio']:.3f}, "
                            f"visual_mass={m['visual_mass']:.3f}, linguistic_mass={m['linguistic_mass']:.3f}"
                        )

        action = action_plan.popleft()
        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(np.asarray(action, dtype=np.float32), policy_cfg, policy_rng)
        obs, reward, done, info = env.step(action.tolist())

        if done:
            break
        t += 1

    # Episode summary (per-layer, plus layers_avg when multiple layers)
    summary: Dict[str, Dict] = {}
    for layer_idx in layers:
        layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_ratios(layer_results)

    if len(layers) > 1 and step_ratio_results:
        avg_step_results = _build_avg_step_results(step_ratio_results)
        if avg_step_results:
            summary["layers_avg"] = summarize_episode_ratios(avg_step_results)

    # Episode-level ratio evolution plot
    if step_ratio_results and save_viz:
        # If multiple layers, also plot averaged curve
        if len(layers) > 1:
            avg_step_results = _build_avg_step_results(step_ratio_results)
            if avg_step_results:
                avg_steps = [r["step"] for r in avg_step_results]
                fig = visualize_ratio_over_episode(
                    step_results=avg_step_results,
                    step_indices=avg_steps,
                    prompt=task_description,
                    output_path=os.path.join(output_dir, f"{episode_prefix}_layers_avg_ratio_evolution.png"),
                )
                plt.close(fig)

        # Plot for each layer
        for layer_idx in layers:
            layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
            layer_steps = [r["step"] for r in layer_results]
            if layer_results:
                fig = visualize_ratio_over_episode(
                    step_results=layer_results,
                    step_indices=layer_steps,
                    prompt=task_description,
                    output_path=os.path.join(
                        output_dir, f"{episode_prefix}_layer{layer_idx}_ratio_evolution.png"
                    ),
                )
                plt.close(fig)

    return {
        "success": bool(done),
        "num_steps": t,
        "step_ratio_results": step_ratio_results,
        "summary": summary,
    }


# ── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation with visual/linguistic attention ratio analysis"
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
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task-id", type=int, default=None, help="Specific task ID (default: all tasks)")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes per task")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 16, 17], help="Transformer layers to analyze")
    parser.add_argument("--num-image-tokens", type=int, default=256)

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
    parser.add_argument("--output-dir", type=str, default="results/attention_ratio")
    parser.add_argument("--save-viz", action="store_true", help="Save per-step ratio visualizations")

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

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        log.info(f"\n{'=' * 70}")
        log.info(f"Task {task_id}: {task_description}")
        log.info(f"{'=' * 70}")

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

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
                vis_cfg=vis_cfg,
                policy_cfg=policy_cfg,
                policy_rng=policy_rng,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            result["visual_perturbation"] = vis_cfg.as_dict()
            result["policy_perturbation"] = policy_cfg.as_dict()
            all_results.append(result)

            # Log episode summary
            log.info(f"  Success: {result.get('success')}")
            log.info(f"  Steps: {result.get('num_steps')}")

            summ_dict = result.get("summary", {})
            if "layers_avg" in summ_dict:
                summ = summ_dict["layers_avg"]
                ratio_mean = summ.get("visual_linguistic_ratio", {}).get("mean", 0)
                visual_frac = summ.get("visual_fraction", {}).get("mean", 0)
                linguistic_frac = summ.get("linguistic_fraction", {}).get("mean", 0)
                log.info(
                    f"  layers_avg: ratio={ratio_mean:.3f}, "
                    f"visual_frac={visual_frac:.1%}, linguistic_frac={linguistic_frac:.1%}"
                )
            else:
                for layer_key, summ in summ_dict.items():
                    ratio_mean = summ.get("visual_linguistic_ratio", {}).get("mean", 0)
                    visual_frac = summ.get("visual_fraction", {}).get("mean", 0)
                    linguistic_frac = summ.get("linguistic_fraction", {}).get("mean", 0)
                    log.info(
                        f"  {layer_key}: ratio={ratio_mean:.3f}, "
                        f"visual_frac={visual_frac:.1%}, linguistic_frac={linguistic_frac:.1%}"
                    )

        env.close()

    # Save aggregate results
    serializable_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_ratio_results"}

        # Extract per-step ratio arrays for each layer
        if "step_ratio_results" in r:
            per_step_ratios = {}
            for layer_idx in args.layers:
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

            # Add averaged per-step ratios if multiple layers
            if len(args.layers) > 1:
                avg_step_results = _build_avg_step_results(r["step_ratio_results"])
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

    results_path = os.path.join(args.output_dir, f"attention_ratio_results_{args.task_suite}.json")
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {results_path}")

    # Print aggregate summary
    log.info(f"\n{'=' * 70}")
    log.info("AGGREGATE SUMMARY")
    log.info(f"{'=' * 70}")

    successes = sum(1 for r in all_results if r.get("success"))
    total = max(1, len(all_results))
    log.info(f"Success rate: {successes}/{len(all_results)} ({successes/total*100:.1f}%)")

    if len(args.layers) > 1:
        ratios = [
            r["summary"]["layers_avg"]["visual_linguistic_ratio"]["mean"]
            for r in all_results
            if "layers_avg" in r.get("summary", {})
        ]
        visual_fracs = [
            r["summary"]["layers_avg"]["visual_fraction"]["mean"]
            for r in all_results
            if "layers_avg" in r.get("summary", {})
        ]
        linguistic_fracs = [
            r["summary"]["layers_avg"]["linguistic_fraction"]["mean"]
            for r in all_results
            if "layers_avg" in r.get("summary", {})
        ]
        if ratios:
            log.info(
                f"  layers_avg: mean_ratio={np.mean(ratios):.3f} +/- {np.std(ratios):.3f}, "
                f"mean_visual_frac={np.mean(visual_fracs):.1%}, "
                f"mean_linguistic_frac={np.mean(linguistic_fracs):.1%}"
            )
    else:
        for layer_idx in args.layers:
            layer_key = f"layer_{layer_idx}"
            ratios = [
                r["summary"][layer_key]["visual_linguistic_ratio"]["mean"]
                for r in all_results
                if layer_key in r.get("summary", {})
            ]
            visual_fracs = [
                r["summary"][layer_key]["visual_fraction"]["mean"]
                for r in all_results
                if layer_key in r.get("summary", {})
            ]
            linguistic_fracs = [
                r["summary"][layer_key]["linguistic_fraction"]["mean"]
                for r in all_results
                if layer_key in r.get("summary", {})
            ]
            if ratios:
                log.info(
                    f"  {layer_key}: mean_ratio={np.mean(ratios):.3f} +/- {np.std(ratios):.3f}, "
                    f"mean_visual_frac={np.mean(visual_fracs):.1%}, "
                    f"mean_linguistic_frac={np.mean(linguistic_fracs):.1%}"
                )


if __name__ == "__main__":
    main()
