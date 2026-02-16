#!/usr/bin/env python3
"""
LIBERO evaluation with visual/linguistic attention ratio analysis for Cosmos Policy.

Computes the ratio of attention from action tokens to visual tokens vs text tokens.
This helps understand whether the model relies more on visual features or linguistic
instructions when predicting actions.

Key differences from Pi0 version:
  - Uses Cosmos DiT architecture with state_t latent frames
  - Hooks into DiT transformer blocks for attention recording
  - Handles multi-frame token sequences properly

Usage:
  python evaluate_attention_ratio_cosmos.py \
    --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --task-suite libero_10 --num-episodes 5 \
    --layers 25 26 27 --output-dir results/attention_ratio_cosmos
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import MagicMock

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "cosmos-policy")
if _COSMOS_POLICY_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_POLICY_DIR)

# Mock transformer_engine if not available
try:
    import transformer_engine.pytorch  # noqa: F401
except Exception:
    import importlib.abc
    import types

    class _TEMockModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, attr):
            val = MagicMock()
            object.__setattr__(self, attr, val)
            return val

    class _TEFinder(importlib.abc.MetaPathFinder):
        def find_module(self, fullname, path=None):
            if fullname == "transformer_engine" or fullname.startswith("transformer_engine."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _TEMockModule(fullname)
            sys.modules[fullname] = mod
            parts = fullname.rsplit(".", 1)
            if len(parts) == 2:
                parent = sys.modules.get(parts[0])
                if parent is not None:
                    setattr(parent, parts[1], mod)
            return mod

    sys.meta_path.insert(0, _TEFinder())
    _TEFinder().load_module("transformer_engine")

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SubprocVectorEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ==============================================================================
# Hook-based attention recording for Cosmos DiT models
# ==============================================================================

_RECORDED: Dict[Any, Any] = {}
_HOOKS: List[torch.utils.hooks.RemovableHandle] = []


def _find_dit(model) -> Optional[torch.nn.Module]:
    """Find the DiT backbone in the model."""
    for attr_path in ["net", "model.net", "model"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "blocks") and hasattr(obj, "final_layer"):
                return obj
        except AttributeError:
            continue

    for _, mod in model.named_modules():
        if hasattr(mod, "blocks") and hasattr(mod, "final_layer") and hasattr(mod, "t_embedder"):
            return mod
    return None


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, Any]]:
    """Returns recorded attention and clears buffer."""
    out = dict(_RECORDED)
    _RECORDED.clear()
    return out


def remove_attention_hooks(handles: Optional[List] = None):
    global _HOOKS
    for h in (handles or _HOOKS):
        try:
            h.remove()
        except Exception:
            pass
    _HOOKS = []


def _is_i2v_cross_attention(module: torch.nn.Module) -> bool:
    """Check if a module is I2VCrossAttention (separate text/image K/V)."""
    return hasattr(module, "k_img") and hasattr(module, "v_img")


def _cross_attn_hook_fn(layer_idx: int):
    """Records cross-attention weights for text and image context separately.

    For I2VCrossAttention: computes separate text and image attention matrices.
    For regular cross-Attention: records a single attention matrix over the
    concatenated context (text || image).
    """
    def hook(module, args, output):
        if not args:
            return
        x = args[0]  # query: (B, T*H*W, D)
        if not torch.is_tensor(x):
            return

        # context is the second positional arg
        context = args[1] if len(args) > 1 else None
        if context is None:
            return

        with torch.no_grad():
            is_i2v = _is_i2v_cross_attention(module)

            if is_i2v:
                # I2VCrossAttention: context is (text_ctx, img_ctx)
                if not isinstance(context, (tuple, list)) or len(context) != 2:
                    return
                qkv = module.compute_qkv(x, context)
                if len(qkv) != 5:
                    return
                q, k_text, v_text, k_img, v_img = qkv

                # q: (B, Sq, H, D), k_text: (B, Sk_text, H, D), k_img: (B, Sk_img, H, D)
                # Rearrange to (B, H, S, D) for matmul
                q_f = q[0].float().permute(1, 0, 2)       # (H, Sq, D)
                k_text_f = k_text[0].float().permute(1, 0, 2)  # (H, Sk_text, D)
                k_img_f = k_img[0].float().permute(1, 0, 2)    # (H, Sk_img, D)

                scale = q_f.shape[-1] ** -0.5

                # Text attention: (H, Sq, Sk_text)
                attn_text = torch.matmul(q_f * scale, k_text_f.transpose(-2, -1))
                # Image attention: (H, Sq, Sk_img)
                attn_img = torch.matmul(q_f * scale, k_img_f.transpose(-2, -1))

                # Softmax over the JOINT key dimension (text + image together)
                # to get comparable attention masses
                attn_joint = torch.cat([attn_text, attn_img], dim=-1)  # (H, Sq, Sk_text+Sk_img)
                attn_joint = F.softmax(attn_joint, dim=-1)

                n_text = attn_text.shape[-1]
                attn_text_normed = attn_joint[..., :n_text]    # (H, Sq, Sk_text)
                attn_img_normed = attn_joint[..., n_text:]     # (H, Sq, Sk_img)

                _RECORDED.setdefault(layer_idx, {})
                _RECORDED[layer_idx].setdefault("calls", [])
                _RECORDED[layer_idx]["calls"].append({
                    "is_i2v": True,
                    "attn_text": attn_text_normed.cpu().numpy(),   # (H, Sq, Sk_text)
                    "attn_img": attn_img_normed.cpu().numpy(),     # (H, Sq, Sk_img)
                    "num_queries": int(q_f.shape[1]),
                    "num_text_keys": int(k_text_f.shape[1]),
                    "num_img_keys": int(k_img_f.shape[1]),
                })
            else:
                # Regular cross-attention: context is a single tensor (text || image concatenated)
                q, k, v = module.compute_qkv(x, context)
                q_f = q[0].float().permute(1, 0, 2)   # (H, Sq, D)
                k_f = k[0].float().permute(1, 0, 2)   # (H, Sk, D)

                scale = q_f.shape[-1] ** -0.5
                attn = torch.matmul(q_f * scale, k_f.transpose(-2, -1))
                attn = F.softmax(attn, dim=-1)  # (H, Sq, Sk)

                _RECORDED.setdefault(layer_idx, {})
                _RECORDED[layer_idx].setdefault("calls", [])
                _RECORDED[layer_idx]["calls"].append({
                    "is_i2v": False,
                    "cross_attn": attn.cpu().numpy(),  # (H, Sq, Sk)
                    "num_queries": int(q_f.shape[1]),
                    "num_keys": int(k_f.shape[1]),
                })

    return hook


def install_attention_hooks(
    model: torch.nn.Module,
    layers: Optional[List[int]] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Installs forward hooks on DiT blocks' cross_attn modules."""
    global _HOOKS
    _RECORDED.clear()

    net = _find_dit(model)
    if net is None:
        raise RuntimeError("Could not find DiT backbone in model")

    blocks = net.blocks
    if layers is None:
        layers = list(range(len(blocks)))

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for idx in layers:
        if idx >= len(blocks):
            continue
        block = blocks[idx]
        handles.append(block.cross_attn.register_forward_hook(_cross_attn_hook_fn(idx)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Attention ratio computation
# ==============================================================================

def compute_cosmos_attention_ratio(recorded_call: Dict[str, Any]) -> Dict:
    """
    Compute visual/linguistic attention ratio from cross-attention.

    For I2VCrossAttention: text and image have separate K/V projections,
    so we compare attention on text keys vs image keys directly.

    For regular cross-attention: the context is concatenated [text || image],
    and we need `num_text_keys` to split them.

    Uses max over heads to capture the strongest attention any head assigns.
    Normalizes by token count for fair comparison.

    Returns:
        Dict with visual_mass, linguistic_mass, ratio, etc.
    """
    is_i2v = recorded_call.get("is_i2v", False)

    if is_i2v:
        attn_text = recorded_call["attn_text"]  # (H, Sq, Sk_text)
        attn_img = recorded_call["attn_img"]    # (H, Sq, Sk_img)
        num_queries = recorded_call["num_queries"]
        num_text_keys = recorded_call["num_text_keys"]
        num_img_keys = recorded_call["num_img_keys"]

        # Max over heads: (Sq, Sk_text) and (Sq, Sk_img)
        text_max = attn_text.max(axis=0).astype(np.float32)
        img_max = attn_img.max(axis=0).astype(np.float32)

        # Per-token attention mass (sum over queries, then normalize by key count)
        linguistic_mass_raw = float(np.sum(text_max))
        visual_mass_raw = float(np.sum(img_max))
        linguistic_mass = linguistic_mass_raw / max(num_text_keys, 1)
        visual_mass = visual_mass_raw / max(num_img_keys, 1)

    else:
        # Regular cross-attention with concatenated context
        attn = recorded_call["cross_attn"]  # (H, Sq, Sk)
        num_queries = recorded_call["num_queries"]
        num_keys = recorded_call["num_keys"]
        # We don't know the text/image split without extra info
        # For now, return the full attention and warn
        log.warning("Regular (non-I2V) cross-attention: cannot split text vs image without split info")
        return {
            "visual_mass": 0.0,
            "linguistic_mass": 0.0,
            "visual_mass_raw": 0.0,
            "linguistic_mass_raw": 0.0,
            "total_mass": 0.0,
            "visual_linguistic_ratio": 0.0,
            "visual_fraction": 0.0,
            "linguistic_fraction": 0.0,
            "num_queries": num_queries,
            "num_visual_tokens": 0,
            "num_text_tokens": num_keys,
            "is_i2v": False,
        }

    # Ratio (per-token normalized, so fair regardless of token counts)
    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float('inf') if visual_mass > 1e-8 else 0.0

    total_mass = visual_mass + linguistic_mass

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "visual_mass_raw": visual_mass_raw,
        "linguistic_mass_raw": linguistic_mass_raw,
        "total_mass": total_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass / max(total_mass, 1e-8),
        "linguistic_fraction": linguistic_mass / max(total_mass, 1e-8),
        "num_queries": num_queries,
        "num_visual_tokens": num_img_keys if is_i2v else 0,
        "num_text_tokens": num_text_keys if is_i2v else 0,
        "is_i2v": is_i2v,
    }


def select_layer_call(
    layer_rec: Dict[str, Any],
    prefer_call: str = "last",
) -> Optional[Dict[str, Any]]:
    """Select which recorded call to use for a layer."""
    calls = layer_rec.get("calls", [])
    if not calls:
        return None
    return calls[0] if prefer_call == "first" else calls[-1]


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


# ==============================================================================
# LIBERO plumbing
# ==============================================================================

def _json_default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _get_libero_env(task, resolution, seed):
    """Create a LIBERO env for a task."""
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _prepare_observation(obs, flip_images: bool = True):
    """Prepare observation for Cosmos policy."""
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    return {
        "primary_image": np.ascontiguousarray(img),
        "wrist_image": np.ascontiguousarray(wrist_img),
        "proprio": np.concatenate(
            (obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        ),
    }


def _build_cosmos_cfg(args) -> PolicyEvalConfig:
    """Build Cosmos policy config."""
    return PolicyEvalConfig(
        config=args.config_name,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        seed=args.seed,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_episodes,
    )


# ==============================================================================
# Episode loop
# ==============================================================================

def run_episode(
    env,
    model,
    cfg: PolicyEvalConfig,
    dataset_stats,
    task_description: str,
    initial_state: np.ndarray,
    layers: List[int],
    max_steps: int,
    num_open_loop_steps: int,
    output_dir: str,
    episode_prefix: str,
    prefer_call: str,
) -> Dict:
    """Run one episode and track visual/linguistic cross-attention ratios."""
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    step_ratio_results = []

    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        observation = _prepare_observation(obs, flip_images=True)

        is_replan = not action_plan
        if is_replan:
            clear_recorded_attention()
            hooks = install_attention_hooks(model, layers=layers)

            try:
                action_return_dict = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=t,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
            finally:
                remove_attention_hooks(hooks)

            action_chunk = action_return_dict["actions"]
            action_plan.extend(action_chunk[:num_open_loop_steps])

            attention_dict = get_recorded_attention()

            if attention_dict:
                per_layer_metrics = []

                for layer_idx in layers:
                    if layer_idx not in attention_dict:
                        continue
                    layer_rec = attention_dict[layer_idx]

                    call = select_layer_call(layer_rec, prefer_call=prefer_call)
                    if call is None:
                        continue

                    try:
                        ratio_result = compute_cosmos_attention_ratio(recorded_call=call)
                    except Exception as e:
                        log.warning(f"[ratio] t={t} layer={layer_idx} failed: {e}")
                        continue

                    ratio_result["layer"] = layer_idx
                    ratio_result["step"] = t
                    step_ratio_results.append(ratio_result)

                    per_layer_metrics.append({
                        "layer": layer_idx,
                        "ratio": ratio_result["visual_linguistic_ratio"],
                        "visual_mass": ratio_result["visual_mass"],
                        "linguistic_mass": ratio_result["linguistic_mass"],
                    })

                if per_layer_metrics:
                    if len(layers) > 1:
                        avg_ratio = float(np.mean([m["ratio"] for m in per_layer_metrics if np.isfinite(m["ratio"])]))
                        avg_visual = float(np.mean([m["visual_mass"] for m in per_layer_metrics]))
                        avg_linguistic = float(np.mean([m["linguistic_mass"] for m in per_layer_metrics]))

                        log.info(
                            f"  t={t} layers={[m['layer'] for m in per_layer_metrics]} "
                            f"AVG ratio={avg_ratio:.3f}, visual={avg_visual:.6f}, linguistic={avg_linguistic:.6f}"
                        )
                    else:
                        m = per_layer_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']} "
                            f"ratio={m['ratio']:.3f}, visual={m['visual_mass']:.6f}, linguistic={m['linguistic_mass']:.6f}"
                        )

        action = action_plan.popleft()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        obs, reward, done, info = env.step(action)

        if done:
            break
        t += 1

    summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_ratios(layer_results)

    return {
        "success": bool(done),
        "num_steps": t,
        "summary": summary,
        "step_ratio_results": step_ratio_results,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="LIBERO eval with visual/linguistic ratio (Cosmos)")

    # Cosmos model
    parser.add_argument("--ckpt-path", type=str, default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    parser.add_argument("--config-name", type=str, default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument("--config-file", type=str, default="cosmos_policy/config/config.py")
    parser.add_argument("--dataset-stats-path", type=str,
                        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    parser.add_argument("--t5-text-embeddings-path", type=str,
                        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[25, 26, 27])

    # Call selection
    parser.add_argument("--prefer-call", type=str, default="last", choices=["last", "first"])

    # Output
    parser.add_argument("--output-dir", type=str, default="results/attention_ratio_cosmos")

    args = parser.parse_args()

    max_steps = TASK_MAX_STEPS[args.task_suite]

    log.info("Loading Cosmos model...")
    cfg = _build_cosmos_cfg(args)
    model, cosmos_config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log.info("Cosmos model loaded.")

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
                cfg=cfg,
                dataset_stats=dataset_stats,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                layers=args.layers,
                max_steps=max_steps,
                num_open_loop_steps=args.num_open_loop_steps,
                output_dir=args.output_dir,
                episode_prefix=episode_prefix,
                prefer_call=args.prefer_call,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

        env.close()

    # Save results
    serializable_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_ratio_results"}

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
                            "visual_fraction": float(res["visual_fraction"]),
                            "linguistic_fraction": float(res["linguistic_fraction"]),
                            "num_queries": int(res.get("num_queries", 0)),
                            "num_visual_tokens": int(res.get("num_visual_tokens", 0)),
                            "num_text_tokens": int(res.get("num_text_tokens", 0)),
                            "is_i2v": bool(res.get("is_i2v", False)),
                        }
                        for res in layer_results
                    ]

            entry["per_step_ratios"] = per_step_ratios

        serializable_results.append(entry)

    results_path = os.path.join(args.output_dir, f"attention_ratio_results_{args.task_suite}.json")
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
