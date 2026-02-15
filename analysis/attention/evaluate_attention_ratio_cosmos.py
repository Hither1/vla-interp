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
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "third_party" / "cosmos-policy")
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


def _self_attn_hook_fn(layer_idx: int):
    """Records q,k attention weights."""
    def hook(module, args, output):
        if not args:
            return
        x = args[0]
        if not torch.is_tensor(x):
            return

        rope_emb = None
        for a in args[1:]:
            if torch.is_tensor(a) and a.dtype in (torch.float16, torch.bfloat16, torch.float32):
                rope_emb = a
                break

        with torch.no_grad():
            try:
                q, k, v = module.compute_qkv(x)
            except TypeError:
                try:
                    q, k, v = module.compute_qkv(x, rope_emb=rope_emb)
                except Exception:
                    return

            q_float = q.float()
            if q_float.ndim != 4:
                return

            qn = torch.linalg.norm(q_float, dim=-1)
            if qn.shape[1] == x.shape[1]:
                qnorm = qn.mean(dim=-1)
            else:
                qnorm = qn.mean(dim=1)

            if q.ndim != 4 or k.ndim != 4:
                return

            if q.shape[1] == qnorm.shape[1]:
                q_f = q.float().permute(0, 2, 1, 3)
                k_f = k.float().permute(0, 2, 1, 3)
            else:
                q_f = q.float()
                k_f = k.float()

            S = int(q_f.shape[-2])
            scale = q_f.shape[-1] ** -0.5
            attn = torch.matmul(q_f * scale, k_f.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)

            _RECORDED.setdefault(layer_idx, {})
            _RECORDED[layer_idx].setdefault("calls", [])
            _RECORDED[layer_idx]["calls"].append({
                "self_attn": attn[0].cpu().numpy(),
                "qnorm": qnorm[0].cpu().numpy(),
                "S": S,
            })

    return hook


def install_attention_hooks(
    model: torch.nn.Module,
    layers: Optional[List[int]] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Installs forward hooks on DiT blocks' self_attn modules."""
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
        handles.append(block.self_attn.register_forward_hook(_self_attn_hook_fn(idx)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Attention ratio computation
# ==============================================================================

def _frame_token_layout(model: torch.nn.Module, S: int) -> Dict[str, Any]:
    """Interpret token axis as [state_t frames] x [tokens_per_frame]."""
    if not hasattr(model, "config") or not hasattr(model.config, "state_t"):
        raise ValueError("Model missing config.state_t")

    state_t = int(model.config.state_t)

    if not hasattr(model.config, "min_num_conditional_frames"):
        raise ValueError("Model missing config.min_num_conditional_frames")
    min_cond = int(model.config.min_num_conditional_frames)

    if S % state_t != 0:
        raise ValueError(f"S={S} not divisible by state_t={state_t}")
    tpf = S // state_t

    action_frame = min_cond
    curr_last_frame = min_cond - 1

    frame_slices: Dict[str, Tuple[int, int]] = {}
    for f in range(state_t):
        frame_slices[f"frame{f}"] = (f * tpf, (f + 1) * tpf)

    frame_slices["curr_last"] = frame_slices[f"frame{curr_last_frame}"]
    frame_slices["action"] = frame_slices[f"frame{action_frame}"]

    return {
        "state_t": state_t,
        "tokens_per_frame": tpf,
        "frame_slices": frame_slices,
        "action_frame": action_frame,
        "curr_last_frame": curr_last_frame,
    }


def compute_cosmos_attention_ratio(
    recorded_call: Dict[str, Any],
    model: torch.nn.Module,
    query_frame: str = "action",
    visual_frame: str = "curr_last",
    text_frame: str = "frame0",  # Assuming text embeddings are in frame 0
) -> Dict:
    """
    Compute visual/linguistic attention ratio for Cosmos DiT.

    Args:
        recorded_call: Dict with "self_attn" (H,S,S) and "qnorm" (S,)
        model: Cosmos model with config
        query_frame: Which frame provides queries (default: "action")
        visual_frame: Which frame contains visual tokens (default: "curr_last")
        text_frame: Which frame contains text tokens (default: "frame0")

    Returns:
        Dict with visual_mass, linguistic_mass, ratio, etc.
    """
    attn = recorded_call.get("self_attn", None)
    if attn is None or attn.ndim != 3:
        raise ValueError("recorded_call missing valid 'self_attn'")

    qnorm = recorded_call.get("qnorm", None)
    S = int(attn.shape[-1])

    layout = _frame_token_layout(model, S)
    fs = layout["frame_slices"]

    if query_frame not in fs:
        raise ValueError(f"Unknown query_frame='{query_frame}'")
    if visual_frame not in fs:
        raise ValueError(f"Unknown visual_frame='{visual_frame}'")
    if text_frame not in fs:
        raise ValueError(f"Unknown text_frame='{text_frame}'")

    q_idx = np.arange(fs[query_frame][0], fs[query_frame][1], dtype=np.int64)
    v_idx = np.arange(fs[visual_frame][0], fs[visual_frame][1], dtype=np.int64)
    t_idx = np.arange(fs[text_frame][0], fs[text_frame][1], dtype=np.int64)

    # Average over heads and queries
    attn_avg = attn.mean(axis=0).astype(np.float32)  # (S,S)
    attn_from_queries = attn_avg[q_idx, :]  # (Q,S)

    # Attention mass on visual tokens
    visual_attn = attn_from_queries[:, v_idx]  # (Q, V)
    visual_mass = float(np.sum(visual_attn))

    # Attention mass on text tokens
    text_attn = attn_from_queries[:, t_idx]  # (Q, T)
    linguistic_mass = float(np.sum(text_attn))

    # Ratio
    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float('inf') if visual_mass > 1e-8 else 0.0

    # Total for normalization
    total_mass = visual_mass + linguistic_mass

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "total_mass": total_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass / max(total_mass, 1e-8),
        "linguistic_fraction": linguistic_mass / max(total_mass, 1e-8),
        "num_queries": len(q_idx),
        "num_visual_tokens": len(v_idx),
        "num_text_tokens": len(t_idx),
    }


def select_layer_call(
    layer_rec: Dict[str, Any],
    prefer_S: Optional[int] = None,
    prefer_call: str = "last",
) -> Optional[Dict[str, Any]]:
    """Select which recorded call to use for a layer."""
    calls = layer_rec.get("calls", [])
    if not calls:
        return None
    if prefer_S is not None:
        exact = [c for c in calls if int(c.get("S", -1)) == int(prefer_S)]
        if exact:
            return exact[0] if prefer_call == "first" else exact[-1]
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
    prefer_S: Optional[int],
    prefer_call: str,
    query_frame: str,
    visual_frame: str,
    text_frame: str,
) -> Dict:
    """Run one episode and track visual/linguistic attention ratios."""
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

                    call = select_layer_call(layer_rec, prefer_S=prefer_S, prefer_call=prefer_call)
                    if call is None:
                        continue

                    try:
                        ratio_result = compute_cosmos_attention_ratio(
                            recorded_call=call,
                            model=model,
                            query_frame=query_frame,
                            visual_frame=visual_frame,
                            text_frame=text_frame,
                        )
                    except Exception as e:
                        log.warning(f"[ratio] t={t} layer={layer_idx} failed: {e}")
                        continue

                    ratio_result["layer"] = layer_idx
                    ratio_result["step"] = t
                    ratio_result["selected_call_S"] = int(call.get("S", -1))
                    ratio_result["query_frame"] = query_frame
                    ratio_result["visual_frame"] = visual_frame
                    ratio_result["text_frame"] = text_frame
                    step_ratio_results.append(ratio_result)

                    per_layer_metrics.append({
                        "layer": layer_idx,
                        "S": int(call.get("S", -1)),
                        "ratio": ratio_result["visual_linguistic_ratio"],
                        "visual_mass": ratio_result["visual_mass"],
                        "linguistic_mass": ratio_result["linguistic_mass"],
                    })

                if per_layer_metrics:
                    if len(layers) > 1:
                        avg_ratio = float(np.mean([m["ratio"] for m in per_layer_metrics if np.isfinite(m["ratio"])]))
                        avg_visual = float(np.mean([m["visual_mass"] for m in per_layer_metrics]))
                        avg_linguistic = float(np.mean([m["linguistic_mass"] for m in per_layer_metrics]))
                        Ss = [m["S"] for m in per_layer_metrics]

                        log.info(
                            f"  t={t} layers={[m['layer'] for m in per_layer_metrics]} "
                            f"S={Ss} [Q={query_frame} V={visual_frame} T={text_frame}]: "
                            f"AVG ratio={avg_ratio:.3f}, visual={avg_visual:.3f}, linguistic={avg_linguistic:.3f}"
                        )
                    else:
                        m = per_layer_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']} S={m['S']} "
                            f"[Q={query_frame} V={visual_frame} T={text_frame}]: "
                            f"ratio={m['ratio']:.3f}, visual={m['visual_mass']:.3f}, linguistic={m['linguistic_mass']:.3f}"
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
    parser.add_argument("--prefer-S", type=int, default=-1)
    parser.add_argument("--prefer-call", type=str, default="last", choices=["last", "first"])

    # Frame selection
    parser.add_argument("--query-frame", type=str, default="action")
    parser.add_argument("--visual-frame", type=str, default="curr_last")
    parser.add_argument("--text-frame", type=str, default="frame0")

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
    prefer_S = None if (args.prefer_S is None or args.prefer_S < 0) else int(args.prefer_S)

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
                prefer_S=prefer_S,
                prefer_call=args.prefer_call,
                query_frame=args.query_frame,
                visual_frame=args.visual_frame,
                text_frame=args.text_frame,
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
                            "selected_call_S": int(res.get("selected_call_S", -1)),
                            "query_frame": str(res.get("query_frame", "")),
                            "visual_frame": str(res.get("visual_frame", "")),
                            "text_frame": str(res.get("text_frame", "")),
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
