#!/usr/bin/env python3
"""
LIBERO evaluation with attention-segmentation IoU analysis for Cosmos Policy.

This version FIXES "weird/stripy" heatmaps when the transformer sequence contains
multiple token types (video/state/action/value tokens).

Key change:
  - We DO NOT average over all queries anymore.
  - We select ONLY queries from the ACTION latent-frame tokens.
  - We produce a heatmap over KEY tokens from a chosen visual frame (default: current third-person frame),
    then reshape ONLY that key-block into (sqrt(K), sqrt(K)).

Assumptions:
  - Cosmos-policy DiT uses state_t latent frames and concatenates them in the token axis.
  - S = state_t * tokens_per_frame.
  - The per-frame tokens for visual frames are a perfect square (so we can reshape to 2D).
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

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "cosmos-policy")
if _COSMOS_POLICY_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_POLICY_DIR)

_LIBERO_EXAMPLES_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "examples" / "libero")
if _LIBERO_EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EXAMPLES_DIR)

# Mock transformer_engine if not available
from unittest.mock import MagicMock

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
from libero.libero.envs.env_wrapper import SegmentationRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image

from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    visualize_attention_vs_segmentation,
    visualize_iou_over_episode,
    find_segmentation_key,
)

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
    # Try common attribute paths
    for attr_path in ["net", "model.net", "model"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "blocks") and hasattr(obj, "final_layer"):
                return obj
        except AttributeError:
            continue

    # Fallback search
    for _, mod in model.named_modules():
        if hasattr(mod, "blocks") and hasattr(mod, "final_layer") and hasattr(mod, "t_embedder"):
            return mod
    return None


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, Any]]:
    """
    Returns a copy of recorded attention and clears internal buffer.
    Format per layer:
      { layer_idx: { "calls": [ { "self_attn": (H,S,S) np, "qnorm": (S,) np, "S": int }, ... ] } }
    Also may include "_debug": list[dict] when debug enabled.
    """
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


def _self_attn_hook_fn(layer_idx: int, debug_shapes: bool = False):
    """
    Records q,k attention weights computed from compute_qkv(x,...).
    Avoids relying on forward() argument ordering.
    """
    def hook(module, args, output):
        if not args:
            return
        x = args[0]
        if not torch.is_tensor(x):
            return

        # Best-effort rope_emb guess (optional)
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

            # qnorm -> (B,S)
            qn = torch.linalg.norm(q_float, dim=-1)  # (B,S,H) or (B,H,S)
            if qn.shape[1] == x.shape[1]:  # likely (B,S,H)
                qnorm = qn.mean(dim=-1)      # (B,S)
            else:
                qnorm = qn.mean(dim=1)       # (B,S)

            if q.ndim != 4 or k.ndim != 4:
                return

            # Build (B,H,S,D)
            if q.shape[1] == qnorm.shape[1]:  # (B,S,H,D)
                q_f = q.float().permute(0, 2, 1, 3)
                k_f = k.float().permute(0, 2, 1, 3)
            else:  # (B,H,S,D)
                q_f = q.float()
                k_f = k.float()

            S = int(q_f.shape[-2])
            scale = q_f.shape[-1] ** -0.5
            attn = torch.matmul(q_f * scale, k_f.transpose(-2, -1))  # (B,H,S,S)
            attn = F.softmax(attn, dim=-1)

            _RECORDED.setdefault(layer_idx, {})
            _RECORDED[layer_idx].setdefault("calls", [])
            _RECORDED[layer_idx]["calls"].append({
                "self_attn": attn[0].cpu().numpy(),   # (H,S,S)
                "qnorm": qnorm[0].cpu().numpy(),      # (S,)
                "S": S,
            })

            if debug_shapes:
                _RECORDED.setdefault("_debug", [])
                _RECORDED["_debug"].append({
                    "layer": layer_idx,
                    "x": tuple(x.shape),
                    "q": tuple(q.shape),
                    "k": tuple(k.shape),
                    "rope": None if rope_emb is None else tuple(rope_emb.shape),
                    "S": S,
                    "num_calls_layer": len(_RECORDED[layer_idx]["calls"]),
                })

    return hook


def install_attention_hooks(
    model: torch.nn.Module,
    layers: Optional[List[int]] = None,
    debug_shapes: bool = False,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Installs forward hooks on DiT blocks' self_attn modules.
    """
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
        handles.append(block.self_attn.register_forward_hook(_self_attn_hook_fn(idx, debug_shapes=debug_shapes)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Heatmap construction (FRAME-SLICED)
# ==============================================================================

def _infer_square_side(S: int) -> int:
    side = int(round(S ** 0.5))
    if side * side != S:
        raise ValueError(f"S={S} is not a perfect square; cannot reshape to 2D grid.")
    return side


def _frame_token_layout(model: torch.nn.Module, S: int) -> Dict[str, Any]:
    """
    Interpret token axis as [state_t frames] x [tokens_per_frame].
    Provides useful frame aliases for Cosmos-policy style ordering.

    Returns:
      {
        "state_t": int,
        "tokens_per_frame": int,
        "frame_slices": {name: (start, end), ...},
        "action_frame": int,
        "curr_last_frame": int,
        "value_frame": int,
      }
    """
    if not hasattr(model, "config") or not hasattr(model.config, "state_t"):
        raise ValueError("Model missing config.state_t; cannot slice frames.")

    state_t = int(model.config.state_t)

    # Cosmos-policy uses min_num_conditional_frames; action is the next frame.
    if not hasattr(model.config, "min_num_conditional_frames"):
        raise ValueError("Model missing config.min_num_conditional_frames; cannot locate action frame.")
    min_cond = int(model.config.min_num_conditional_frames)

    if S % state_t != 0:
        raise ValueError(f"S={S} not divisible by state_t={state_t}; cannot frame-slice tokens.")
    tpf = S // state_t

    action_frame = min_cond
    curr_last_frame = min_cond - 1
    value_frame = state_t - 1

    frame_slices: Dict[str, Tuple[int, int]] = {}
    for f in range(state_t):
        frame_slices[f"frame{f}"] = (f * tpf, (f + 1) * tpf)

    # Useful aliases
    frame_slices["blank"] = frame_slices["frame0"]
    frame_slices["curr_last"] = frame_slices[f"frame{curr_last_frame}"]
    frame_slices["action"] = frame_slices[f"frame{action_frame}"]
    frame_slices["value"] = frame_slices[f"frame{value_frame}"]

    return {
        "state_t": state_t,
        "tokens_per_frame": tpf,
        "frame_slices": frame_slices,
        "action_frame": action_frame,
        "curr_last_frame": curr_last_frame,
        "value_frame": value_frame,
    }


def _slice_to_indices(slc: Tuple[int, int]) -> np.ndarray:
    a, b = slc
    return np.arange(a, b, dtype=np.int64)


def _subset_saliency(
    attn_hss: np.ndarray,
    qnorm_s: Optional[np.ndarray],
    q_idx: np.ndarray,
    k_idx: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Build a saliency vector over selected KEYS only, aggregating over selected QUERIES only.

    attn_hss: (H,S,S)
    q_idx: query indices
    k_idx: key indices
    returns: (len(k_idx),)
    """
    attn = attn_hss.mean(axis=0).astype(np.float32)  # (S,S)
    A = attn[np.ix_(q_idx, k_idx)]                   # (Q,K)

    if mode == "incoming_argmaxq":
        if qnorm_s is None:
            return A[0]
        qn = qnorm_s.astype(np.float32)
        qn_sub = qn[q_idx]
        q_star = int(np.argmax(qn_sub))
        return A[q_star]

    if mode == "incoming_qweighted":
        if qnorm_s is None:
            return A.mean(axis=0)
        qn = qnorm_s.astype(np.float32)
        w = np.maximum(qn[q_idx], 0.0)
        s = float(w.sum())
        if s < 1e-8:
            return A.mean(axis=0)
        w = w / s
        return (w[:, None] * A).sum(axis=0)

    if mode == "outgoing_mean":
        return A.mean(axis=0)

    raise ValueError(f"Unknown saliency mode: {mode}")


def select_layer_call(
    layer_rec: Dict[str, Any],
    prefer_S: Optional[int] = None,
    prefer_call: str = "last",
) -> Optional[Dict[str, Any]]:
    """
    Select which recorded call to use for a layer.

    prefer_call:
      - "last": last call regardless of S
      - "first": first call regardless of S
    """
    calls = layer_rec.get("calls", [])
    if not calls:
        return None
    if prefer_S is not None:
        exact = [c for c in calls if int(c.get("S", -1)) == int(prefer_S)]
        if exact:
            return exact[0] if prefer_call == "first" else exact[-1]
    return calls[0] if prefer_call == "first" else calls[-1]


def create_cosmos_attention_heatmap(
    recorded_call: Dict[str, Any],
    target_shape: Tuple[int, int],
    model: torch.nn.Module,
    query_frame: str = "action",
    key_frame: str = "curr_last",
    transpose_hw: bool = False,
    flip_ud: bool = False,
    flip_lr: bool = False,
    clip_percentiles: Tuple[float, float] = (5.0, 95.0),
    saliency_mode: str = "incoming_qweighted",
    debug: bool = False,
) -> np.ndarray:
    """
    Build normalized [0,1] heatmap at target_shape from one call's recorded data,
    but ONLY using:
      - QUERY tokens from query_frame (default: action)
      - KEY tokens from key_frame (default: curr_last visual frame)
    """
    import cv2

    attn = recorded_call.get("self_attn", None)
    if attn is None:
        raise ValueError("recorded_call missing 'self_attn'")
    qnorm = recorded_call.get("qnorm", None)

    if attn.ndim != 3:
        raise ValueError(f"self_attn must be (H,S,S), got {attn.shape}")

    S = int(attn.shape[-1])

    layout = _frame_token_layout(model, S)
    fs = layout["frame_slices"]

    if query_frame not in fs:
        raise ValueError(f"Unknown query_frame='{query_frame}'. Keys: {list(fs.keys())}")
    if key_frame not in fs:
        raise ValueError(f"Unknown key_frame='{key_frame}'. Keys: {list(fs.keys())}")

    q_idx = _slice_to_indices(fs[query_frame])
    k_idx = _slice_to_indices(fs[key_frame])

    K = int(len(k_idx))
    side = _infer_square_side(K)

    sal_k = _subset_saliency(attn, qnorm, q_idx=q_idx, k_idx=k_idx, mode=saliency_mode)  # (K,)
    hm = sal_k.reshape(side, side).astype(np.float32)

    if transpose_hw:
        hm = hm.T
    if flip_ud:
        hm = hm[::-1, :]
    if flip_lr:
        hm = hm[:, ::-1]

    if debug:
        log.info(
            f"[heatmap] S={S} state_t={layout['state_t']} tpf={layout['tokens_per_frame']} "
            f"query_frame={query_frame} key_frame={key_frame} K={K} side={side} "
            f"transpose={transpose_hw} flip_ud={flip_ud} flip_lr={flip_lr} mode={saliency_mode}"
        )

    hm_up = cv2.resize(hm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    lo_p, hi_p = clip_percentiles
    lo = float(np.percentile(hm_up, lo_p))
    hi = float(np.percentile(hm_up, hi_p))
    hm_up = np.clip(hm_up, lo, hi)
    hm_up = (hm_up - lo) / (hi - lo + 1e-8)
    return hm_up


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


def _get_segmentation_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_segmentations="instance",
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _prepare_observation(obs, flip_images: bool = True, vis_cfg: Optional[VisualPerturbConfig] = None):
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    img = np.ascontiguousarray(img)
    wrist_img = np.ascontiguousarray(wrist_img)
    if vis_cfg is not None:
        img = perturb_image(img, vis_cfg)
        wrist_img = perturb_image(wrist_img, vis_cfg)
    return {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate(
            (obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        ),
    }


def _build_cosmos_cfg(args) -> PolicyEvalConfig:
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
    save_viz: bool,
    output_dir: str,
    episode_prefix: str,
    threshold_methods,
    # heatmap / selection knobs
    transpose_hw: bool,
    flip_ud: bool,
    flip_lr: bool,
    clip_lo: float,
    clip_hi: float,
    saliency_mode: str,
    prefer_S: Optional[int],
    prefer_call: str,
    query_frame: str,
    key_frame: str,
    # debug knobs
    attn_debug: bool,
    debug_shapes: bool,
    debug_dump_first_replan: bool,
    # metric
    metric: str = "iou",
    # perturbations
    vis_cfg: Optional[VisualPerturbConfig] = None,
    policy_cfg: Optional[PolicyPerturbConfig] = None,
    policy_rng: Optional[np.random.Generator] = None,
) -> Dict:
    obs = env.reset()
    obs = env.set_init_state(initial_state)
    if policy_cfg is not None and policy_rng is not None:
        apply_object_shift(env, policy_cfg, policy_rng)

    obj_of_interest = env.obj_of_interest
    object_ids = {}

    # First try exact matching
    for obj_name in obj_of_interest:
        if obj_name in env.instance_to_id:
            object_ids[obj_name] = env.instance_to_id[obj_name]

    # If no exact matches (common in libero_goal), try fuzzy matching
    if not object_ids:
        log.warning(f"No exact matches found. Attempting fuzzy matching...")
        log.warning(f"  obj_of_interest: {list(obj_of_interest)}")
        log.warning(f"  instance_to_id keys: {list(env.instance_to_id.keys())}")

        for obj_name in obj_of_interest:
            # Try to find instance names that contain the obj_name as substring
            matches = [k for k in env.instance_to_id.keys()
                      if obj_name.lower() in k.lower()]

            if matches:
                # Use all matches
                for match in matches:
                    object_ids[match] = env.instance_to_id[match]
                log.info(f"  Fuzzy matched '{obj_name}' -> {matches}")

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
        log.error(f"No segmentation key found. obs keys: {list(obs.keys())}")
        return {"error": "no_segmentation_key", "obs_keys": list(obs.keys())}
    log.info(f"Using segmentation key: {seg_key}")

    action_plan = collections.deque()
    step_iou_results = []

    t = 0
    done = False
    did_dump = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        observation = _prepare_observation(obs, flip_images=True, vis_cfg=vis_cfg)
        agentview_for_viz = observation["primary_image"]
        wrist_for_viz = observation["wrist_image"]

        seg_raw = obs[seg_key]
        seg_rotated = np.ascontiguousarray(np.flipud(seg_raw))

        is_replan = not action_plan
        if is_replan:
            clear_recorded_attention()
            hooks = install_attention_hooks(model, layers=layers, debug_shapes=debug_shapes)

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

            if debug_dump_first_replan and (not did_dump):
                did_dump = True
                dump_path = os.path.join(output_dir, f"{episode_prefix}_attn_debug_dump.json")
                dbg = attention_dict.get("_debug", [])
                summary = {}
                for layer_idx in layers:
                    rec = attention_dict.get(layer_idx, {})
                    calls = rec.get("calls", [])
                    Ss = [int(c.get("S", -1)) for c in calls]
                    summary[str(layer_idx)] = {"num_calls": len(calls), "S_values": Ss[:50]}
                with open(dump_path, "w") as f:
                    json.dump({"debug_shapes": dbg[:200], "call_summary": summary}, f, indent=2, default=_json_default)
                log.info(f"[debug] wrote attention debug dump: {dump_path}")
            
            if attention_dict and object_ids:
                per_layer_metrics = []  # list of dicts with IoU/mass/etc for this step t

                for layer_idx in layers:
                    if layer_idx not in attention_dict:
                        continue
                    layer_rec = attention_dict[layer_idx]

                    call = select_layer_call(layer_rec, prefer_S=prefer_S, prefer_call=prefer_call)
                    if call is None:
                        continue

                    try:
                        heatmap = create_cosmos_attention_heatmap(
                            recorded_call=call,
                            target_shape=(LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
                            model=model,
                            query_frame=query_frame,
                            key_frame=key_frame,
                            transpose_hw=transpose_hw,
                            flip_ud=flip_ud,
                            flip_lr=flip_lr,
                            clip_percentiles=(clip_lo, clip_hi),
                            saliency_mode=saliency_mode,
                            debug=(attn_debug and (t % 50 == 0)),
                        )
                    except Exception as e:
                        log.warning(f"[attn->heatmap] t={t} layer={layer_idx} failed: {e}")
                        continue

                    iou_result = compute_attention_object_iou(
                        attention_heatmap=heatmap,
                        segmentation_mask=seg_rotated,
                        object_ids=object_ids,
                        threshold_methods=threshold_methods,
                    )
                    iou_result["layer"] = layer_idx
                    iou_result["step"] = t
                    iou_result["selected_call_S"] = int(call.get("S", -1))
                    iou_result["query_frame"] = query_frame
                    iou_result["key_frame"] = key_frame
                    step_iou_results.append(iou_result)

                    combined_iou = float(iou_result["combined"].get("percentile_90", {}).get("iou", 0.0))
                    mass = float(iou_result["attention_mass"].get("_all_objects", 0.0))
                    primary_val = mass if metric == "attention_ratio" else combined_iou
                    per_layer_metrics.append(
                        {
                            "layer": layer_idx,
                            "S": int(call.get("S", -1)),
                            "iou": combined_iou,
                            "mass": mass,
                            "primary": primary_val,
                            "pointing_hit": bool(iou_result.get("pointing_hit", False)),
                        }
                    )

                    # Keep per-layer visualizations if you want
                    if save_viz:
                        viz_path = os.path.join(
                            output_dir, f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou.png"
                        )
                        fig = visualize_attention_vs_segmentation(
                            frame_rgb=agentview_for_viz,
                            attention_heatmap=heatmap,
                            segmentation_mask=seg_rotated,
                            object_ids=object_ids,
                            iou_results=iou_result,
                            layer_idx=layer_idx,
                            output_path=viz_path,
                        )
                        plt.close(fig)

                        viz_wrist = os.path.join(
                            output_dir, f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou_WRIST.png"
                        )
                        fig2 = visualize_attention_vs_segmentation(
                            frame_rgb=wrist_for_viz,
                            attention_heatmap=heatmap,
                            segmentation_mask=seg_rotated,
                            object_ids=object_ids,
                            iou_results=iou_result,
                            layer_idx=layer_idx,
                            output_path=viz_wrist,
                        )
                        plt.close(fig2)

                # ---- NEW: aggregate reporting across layers ----
                if per_layer_metrics:
                    metric_label = "attn_ratio" if metric == "attention_ratio" else "IoU"
                    if len(layers) > 1:
                        avg_primary = float(np.mean([m["primary"] for m in per_layer_metrics]))
                        avg_mass = float(np.mean([m["mass"] for m in per_layer_metrics]))
                        hit_rate = float(np.mean([1.0 if m["pointing_hit"] else 0.0 for m in per_layer_metrics]))
                        Ss = [m["S"] for m in per_layer_metrics]

                        log.info(
                            f"  t={t} layers={ [m['layer'] for m in per_layer_metrics] } "
                            f"S={Ss} [Q={query_frame} -> K={key_frame}]: "
                            f"AVG {metric_label}={avg_primary:.3f}, AVG attn_mass={avg_mass:.1%}, "
                            f"pointing_hit_rate={hit_rate:.1%}"
                        )
                    else:
                        # single-layer case: print the one layer as before (but using the collected metrics)
                        m = per_layer_metrics[0]
                        log.info(
                            f"  t={t} layer={m['layer']} S={m['S']} "
                            f"[Q={query_frame} -> K={key_frame}]: "
                            f"{metric_label}={m['primary']:.3f}, attn_mass={m['mass']:.1%}, "
                            f"pointing={'hit' if m['pointing_hit'] else 'miss'}"
                        )

            # if attention_dict and object_ids:
            #     for layer_idx in layers:
            #         if layer_idx not in attention_dict:
            #             continue
            #         layer_rec = attention_dict[layer_idx]

            #         call = select_layer_call(layer_rec, prefer_S=prefer_S, prefer_call=prefer_call)
            #         if call is None:
            #             continue

            #         try:
            #             heatmap = create_cosmos_attention_heatmap(
            #                 recorded_call=call,
            #                 target_shape=(LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
            #                 model=model,
            #                 query_frame=query_frame,
            #                 key_frame=key_frame,
            #                 transpose_hw=transpose_hw,
            #                 flip_ud=flip_ud,
            #                 flip_lr=flip_lr,
            #                 clip_percentiles=(clip_lo, clip_hi),
            #                 saliency_mode=saliency_mode,
            #                 debug=(attn_debug and (t % 50 == 0)),
            #             )
            #         except Exception as e:
            #             log.warning(f"[attn->heatmap] t={t} layer={layer_idx} failed: {e}")
            #             continue

            #         iou_result = compute_attention_object_iou(
            #             attention_heatmap=heatmap,
            #             segmentation_mask=seg_rotated,
            #             object_ids=object_ids,
            #             threshold_methods=threshold_methods,
            #         )
            #         iou_result["layer"] = layer_idx
            #         iou_result["step"] = t
            #         iou_result["selected_call_S"] = int(call.get("S", -1))
            #         iou_result["query_frame"] = query_frame
            #         iou_result["key_frame"] = key_frame
            #         step_iou_results.append(iou_result)

            #         combined_iou = iou_result["combined"].get("percentile_90", {}).get("iou", 0)
            #         mass = iou_result["attention_mass"].get("_all_objects", 0)
            #         log.info(
            #             f"  t={t} layer={layer_idx} S={int(call.get('S', -1))} "
            #             f"[Q={query_frame} -> K={key_frame}]: "
            #             f"IoU={combined_iou:.3f}, attn_mass={mass:.1%}, "
            #             f"pointing={'hit' if iou_result['pointing_hit'] else 'miss'}"
            #         )

            #         if save_viz:
            #             viz_path = os.path.join(
            #                 output_dir, f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou.png"
            #             )
            #             fig = visualize_attention_vs_segmentation(
            #                 frame_rgb=agentview_for_viz,
            #                 attention_heatmap=heatmap,
            #                 segmentation_mask=seg_rotated,
            #                 object_ids=object_ids,
            #                 iou_results=iou_result,
            #                 layer_idx=layer_idx,
            #                 output_path=viz_path,
            #             )
            #             plt.close(fig)

            #             viz_wrist = os.path.join(
            #                 output_dir, f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou_WRIST.png"
            #             )
            #             fig2 = visualize_attention_vs_segmentation(
            #                 frame_rgb=wrist_for_viz,
            #                 attention_heatmap=heatmap,
            #                 segmentation_mask=seg_rotated,
            #                 object_ids=object_ids,
            #                 iou_results=iou_result,
            #                 layer_idx=layer_idx,
            #                 output_path=viz_wrist,
            #             )
            #             plt.close(fig2)

        action = action_plan.popleft()
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(action, policy_cfg, policy_rng)
        obs, reward, done, info = env.step(action.tolist())

        if done:
            break
        t += 1

    summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_iou(layer_results)

    if step_iou_results and save_viz:
        for layer_idx in layers:
            layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
            if layer_results:
                layer_steps = [r["step"] for r in layer_results]
                fig = visualize_iou_over_episode(
                    step_results=layer_results,
                    step_indices=layer_steps,
                    prompt=task_description,
                    output_path=os.path.join(output_dir, f"{episode_prefix}_layer{layer_idx}_iou_evolution.png"),
                )
                plt.close(fig)

    return {
        "success": bool(done),
        "num_steps": t,
        "summary": summary,
        "objects_of_interest": list(object_ids.keys()),
        "step_iou_results": step_iou_results,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="LIBERO eval with attention IoU (Cosmos DiT hooks, action-token queries)")

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

    # Attention / IoU
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[25, 26, 27])
    parser.add_argument("--threshold-methods", type=str, nargs="+",
                        default=["percentile_90", "percentile_75", "otsu_0"])

    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "attention_ratio"],
        help="Primary metric: 'iou' (thresholded binary IoU) or 'attention_ratio' "
             "(attention mass on GT region / total attention, no thresholding).",
    )

    # Saliency
    parser.add_argument("--saliency-mode", type=str, default="incoming_qweighted",
                        choices=["incoming_qweighted", "incoming_argmaxq", "outgoing_mean"])

    # Call selection
    parser.add_argument("--prefer-S", type=int, default=-1,
                        help="Prefer recorded calls with this S (set -1 to disable).")
    parser.add_argument("--prefer-call", type=str, default="last",
                        choices=["last", "first"])

    # FRAME selection (IMPORTANT)
    parser.add_argument("--query-frame", type=str, default="action",
                        help="Which latent frame provides QUERIES (default: action). Examples: action, value, curr_last, frame4.")
    parser.add_argument("--key-frame", type=str, default="curr_last",
                        help="Which latent frame provides KEYS to visualize (must be square tokens). Examples: curr_last, frame3.")

    # Alignment knobs
    parser.add_argument("--transpose-hw", action="store_true",
                        help="Transpose latent grid before upsampling.")
    parser.add_argument("--flip-ud", action="store_true",
                        help="Flip attention grid vertically before upsampling.")
    parser.add_argument("--flip-lr", action="store_true",
                        help="Flip attention grid horizontally before upsampling.")

    parser.add_argument("--clip-lo", type=float, default=5.0,
                        help="Low percentile for clipping normalization.")
    parser.add_argument("--clip-hi", type=float, default=95.0,
                        help="High percentile for clipping normalization.")

    # Debug
    parser.add_argument("--attn-debug", action="store_true",
                        help="Print attention/heatmap debug occasionally.")
    parser.add_argument("--debug-shapes", action="store_true",
                        help="Record q/k/x/rope shapes for hooked forwards.")
    parser.add_argument("--debug-dump-first-replan", action="store_true",
                        help="Dump attention call summary + shape debug to JSON on first replan.")

    # Visual perturbation
    parser.add_argument("--visual-perturb-mode", type=str, default="none",
                        choices=["none", "rotate", "translate", "rotate_translate"],
                        help="Image-level perturbation mode.")
    parser.add_argument("--rotation-degrees", type=float, default=0.0,
                        help="CCW rotation in degrees (rotate / rotate_translate modes).")
    parser.add_argument("--translate-x-frac", type=float, default=0.0,
                        help="Horizontal shift as fraction of image width (positive = right).")
    parser.add_argument("--translate-y-frac", type=float, default=0.0,
                        help="Vertical shift as fraction of image height (positive = down).")

    # Policy perturbation
    parser.add_argument("--policy-perturb-mode", type=str, default="none",
                        choices=["none", "random_action", "object_shift"],
                        help="Policy-level perturbation mode.")
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="Probability of replacing policy action with random noise per step.")
    parser.add_argument("--random-action-scale", type=float, default=1.0,
                        help="Scale of uniform random action noise: Uniform(-scale, scale).")
    parser.add_argument("--object-shift-x-std", type=float, default=0.0,
                        help="Std (metres) of Gaussian object shift along x at episode start.")
    parser.add_argument("--object-shift-y-std", type=float, default=0.0,
                        help="Std (metres) of Gaussian object shift along y at episode start.")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs_iou_cosmos")
    parser.add_argument("--save-viz", action="store_true")

    args = parser.parse_args()

    threshold_methods = []
    for tm in args.threshold_methods:
        method, value = tm.rsplit("_", 1)
        threshold_methods.append((method, float(value)))

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

        env, _ = _get_segmentation_env(task, LIBERO_ENV_RESOLUTION, args.seed)

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
                save_viz=args.save_viz,
                output_dir=args.output_dir,
                episode_prefix=episode_prefix,
                threshold_methods=threshold_methods,
                transpose_hw=args.transpose_hw,
                flip_ud=args.flip_ud,
                flip_lr=args.flip_lr,
                clip_lo=args.clip_lo,
                clip_hi=args.clip_hi,
                saliency_mode=args.saliency_mode,
                prefer_S=prefer_S,
                prefer_call=args.prefer_call,
                query_frame=args.query_frame,
                key_frame=args.key_frame,
                attn_debug=args.attn_debug,
                debug_shapes=args.debug_shapes,
                debug_dump_first_replan=args.debug_dump_first_replan,
                metric=args.metric,
                vis_cfg=vis_cfg,
                policy_cfg=policy_cfg,
                policy_rng=policy_rng,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

        env.close()

    # Save aggregate results (strip verbose step_iou_results but keep per_step_iou arrays)
    serializable_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_iou_results"}

        # Extract per-step IOU arrays for each layer
        if "step_iou_results" in r:
            per_step_iou = {}
            for layer_idx in args.layers:
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
                            "selected_call_S": int(res.get("selected_call_S", -1)),
                            "query_frame": str(res.get("query_frame", "")),
                            "key_frame": str(res.get("key_frame", "")),
                        }
                        for res in layer_results
                    ]

            entry["per_step_iou"] = per_step_iou

        serializable_results.append(entry)

    results_path = os.path.join(args.output_dir, f"iou_results_{args.task_suite}.json")
    output_data = {
        "metric": args.metric,
        "visual_perturbation": vis_cfg.as_dict(),
        "policy_perturbation": policy_cfg.as_dict(),
        "results": serializable_results,
    }
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()