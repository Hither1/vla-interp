#!/usr/bin/env python3
"""
LIBERO evaluation with visual/linguistic attention ratio analysis for DreamZero.

DreamZero is a WAN-based video diffusion model fine-tuned for action prediction.
Its token layout (inference, is_tf=False) is:
  [first_image (frame_seqlen)] [image_blocks (N*frame_seqlen)] [action_tokens (N*num_action)] [state_tokens (N*num_state)]

Text is conditioned via cross-attention (NOT in the self-attention sequence).

Metrics:
  - Visual mass:      self-attention from action tokens to image tokens (first_image + image_blocks)
  - Linguistic mass:  cross-attention from action tokens to text tokens (normalized by text_len)
  - Ratio:           visual_mass / linguistic_mass

Must be launched with torchrun (same as dreamzero_eval.py):
  torchrun --standalone --nproc_per_node=4 \
    analysis/attention/evaluate_attention_ratio_dreamzero.py \
    --model-path /path/to/dreamzero_libero_lora \
    --task-suite-name libero_10 --num-episodes 5 \
    --layers 10 20 30 --output-dir results/attention_ratio_dreamzero
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import math
import os
import pathlib
import pickle
import sys
import traceback
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DREAMZERO_DIR = str(_PROJECT_ROOT / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)
_LIBERO_EVAL_DIR = str(_PROJECT_ROOT / "examples" / "libero")
if _LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EVAL_DIR)

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanAttentionBlock, CausalWanSelfAttention,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image
from tianshou.data import Batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 128

_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


# ==============================================================================
# Global attention recording state
# ==============================================================================

_RECORDED: Dict[int, Dict[str, Any]] = {}   # layer_idx -> {self_attn: [], cross_attn: [], action_register_length: int}
_HOOKS: List[torch.utils.hooks.RemovableHandle] = []


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, Any]]:
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


def install_attention_hooks(
    policy: GrootSimPolicy,
    layers: List[int],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Install hooks on CausalWanAttentionBlock's self_attn.attn and cross_attn."""
    global _HOOKS
    _RECORDED.clear()

    # Navigate to the DiT model's blocks
    try:
        blocks = policy.trained_model.action_head.model.blocks
    except AttributeError as e:
        raise RuntimeError(f"Cannot find DiT blocks at policy.trained_model.action_head.model.blocks: {e}")

    handles: List[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in layers:
        if layer_idx >= len(blocks):
            log.warning("Layer %d out of range (model has %d blocks)", layer_idx, len(blocks))
            continue

        block = blocks[layer_idx]
        if not isinstance(block, CausalWanAttentionBlock):
            log.warning("Block %d is not CausalWanAttentionBlock, skipping", layer_idx)
            continue

        self_attn: CausalWanSelfAttention = block.self_attn
        _RECORDED[layer_idx] = {"self_attn": [], "cross_attn": [], "action_register_length": None}

        # ── Pre-hook on self_attn to capture action_register_length ──────────
        def _make_pre_hook(idx):
            def pre_hook(module, args):
                # args = (x, freqs, freqs_action, freqs_state, action_register_length, ...)
                if len(args) >= 5 and args[4] is not None:
                    _RECORDED[idx]["action_register_length"] = args[4]
            return pre_hook

        handles.append(self_attn.register_forward_pre_hook(_make_pre_hook(layer_idx)))

        # ── Post-hook on self_attn.attn (AttentionModule) ────────────────────
        # q, k, v are rope-applied and passed to flash attention.
        # Action blocks are identified by q.shape[1] == num_action_per_block.
        def _make_self_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block

            def hook(module, args, output):
                if len(args) < 3:
                    return
                q, k, v = args[0], args[1], args[2]
                # q: (B, Lq, num_heads, head_dim)
                # k: (B, Lk, num_heads, head_dim)
                if q.shape[1] != n_act:
                    return  # Not an action block call

                Lk = k.shape[1]
                # k_context layout: [first_image | image_blocks | action | state]
                visual_end = Lk - n_act - n_state  # end of image tokens in k_context

                with torch.no_grad():
                    q_f = q.float().permute(0, 2, 1, 3)   # (B, H, Lq, d)
                    k_f = k.float().permute(0, 2, 1, 3)   # (B, H, Lk, d)
                    scale = q.shape[-1] ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, Lq, Lk)
                    # Average over batch (dim 0) and heads (dim 1) → (Lq, Lk)
                    attn_np = attn[0].mean(0).cpu().float().numpy()

                _RECORDED[idx]["self_attn"].append({
                    "attn": attn_np,    # (num_action_per_block, Lk)
                    "Lk": Lk,
                    "visual_end": visual_end,
                    "n_act": n_act,
                    "n_state": n_state,
                })
            return hook

        handles.append(self_attn.attn.register_forward_hook(_make_self_attn_hook(layer_idx, self_attn)))

        # ── Post-hook on cross_attn (WanT2VCrossAttention or WanI2VCrossAttention) ──
        # args = (normed_x, context, ...) where normed_x is (B, S, C), context is (B, T, C)
        def _make_cross_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block

            def hook(module, args, output):
                if len(args) < 2:
                    return
                normed_x = args[0]   # (B, S, C) – normed full sequence
                context = args[1]    # (B, T, C) – text embeddings

                action_register_length = _RECORDED[idx].get("action_register_length")
                if action_register_length is None or action_register_length == 0:
                    return

                B, S, C = normed_x.shape
                n = module.num_heads
                d = module.head_dim

                # Compute chunk_size from action_register_length
                chunk_size = action_register_length // (n_act + n_state)
                action_horizon = chunk_size * n_act
                state_horizon = chunk_size * n_state

                # Action tokens are at positions [S - action_horizon - state_horizon : S - state_horizon]
                action_start = S - action_horizon - state_horizon
                action_end = S - state_horizon
                if action_start < 0 or action_end > S:
                    return

                with torch.no_grad():
                    x_act = normed_x[:, action_start:action_end]  # (B, action_horizon, C)
                    q = module.norm_q(module.q(x_act)).view(B, -1, n, d)   # (B, AH, H, d)
                    # Use context (text) for k — handle I2V cross-attn which splits context
                    if hasattr(module, 'k_img'):
                        # WanI2VCrossAttention: context[:, :257] is image, rest is text
                        text_context = context[:, 257:]
                    else:
                        text_context = context
                    if text_context.shape[1] == 0:
                        return
                    k = module.norm_k(module.k(text_context)).view(B, -1, n, d)  # (B, T, H, d)

                    q_f = q.float().permute(0, 2, 1, 3)   # (B, H, AH, d)
                    k_f = k.float().permute(0, 2, 1, 3)   # (B, H, T, d)
                    scale = d ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, AH, T)
                    attn_np = attn[0].mean(0).cpu().float().numpy()  # (AH, T)

                _RECORDED[idx]["cross_attn"].append({
                    "attn": attn_np,     # (action_horizon, T)
                    "action_horizon": action_horizon,
                    "T": text_context.shape[1],
                })
            return hook

        handles.append(block.cross_attn.register_forward_hook(_make_cross_attn_hook(layer_idx, self_attn)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Ratio computation
# ==============================================================================

def compute_ratio_from_recorded(
    layer_rec: Dict[str, Any],
) -> Optional[Dict]:
    """Compute visual/linguistic ratio from one layer's recorded calls."""
    self_attn_calls = layer_rec.get("self_attn", [])
    cross_attn_calls = layer_rec.get("cross_attn", [])

    if not self_attn_calls:
        return None

    # Visual mass: mean fraction of self-attention going to image tokens
    visual_fractions = []
    for call in self_attn_calls:
        attn = call["attn"]             # (num_action, Lk)
        visual_end = call["visual_end"]
        # Row sums are ≈ 1.0 (softmax). visual_fraction per row = sum over visual columns.
        vis_frac = float(attn[:, :visual_end].sum()) / max(float(attn.sum()), 1e-8)
        visual_fractions.append(vis_frac)
    visual_mass = float(np.mean(visual_fractions))

    # Linguistic mass: mean cross-attention weight × text_len (≈ expected attention per text token × T)
    if cross_attn_calls:
        ling_values = []
        for call in cross_attn_calls:
            attn = call["attn"]         # (action_horizon, T), rows sum to 1.0
            T = call["T"]
            # Mean attention per (query, text_token) pair, normalized so uniform = 1.0
            ling_values.append(float(attn.mean()) * T)
        linguistic_mass = float(np.mean(ling_values))
    else:
        linguistic_mass = 0.0

    # Ratio
    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float("inf") if visual_mass > 1e-8 else 0.0

    # Additional breakdown of self-attention
    state_fractions, self_fractions = [], []
    for call in self_attn_calls:
        attn = call["attn"]
        Lk = call["Lk"]
        n_act = call["n_act"]
        n_state = call["n_state"]
        vis_end = call["visual_end"]
        action_cols = attn[:, vis_end : vis_end + n_act]
        state_cols = attn[:, vis_end + n_act :]
        total = max(float(attn.sum()), 1e-8)
        state_fractions.append(float(state_cols.sum()) / total)
        self_fractions.append(float(action_cols.sum()) / total)

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass,
        "linguistic_fraction": linguistic_mass,
        "state_fraction": float(np.mean(state_fractions)),
        "self_fraction": float(np.mean(self_fractions)),
        "num_self_attn_calls": len(self_attn_calls),
        "num_cross_attn_calls": len(cross_attn_calls),
    }


def summarize_episode_ratios(step_results: List[Dict]) -> Dict:
    if not step_results:
        return {}
    ratios = [r["visual_linguistic_ratio"] for r in step_results
              if np.isfinite(r.get("visual_linguistic_ratio", float("nan")))]
    vis_fracs = [r["visual_fraction"] for r in step_results]
    ling_fracs = [r["linguistic_fraction"] for r in step_results]
    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
            "min": float(np.min(ratios)) if ratios else 0.0,
            "max": float(np.max(ratios)) if ratios else 0.0,
        },
        "visual_fraction": {
            "mean": float(np.mean(vis_fracs)),
            "std": float(np.std(vis_fracs)),
        },
        "linguistic_fraction": {
            "mean": float(np.mean(ling_fracs)),
            "std": float(np.std(ling_fracs)),
        },
        "num_steps": len(step_results),
    }


# ==============================================================================
# LIBERO helpers
# ==============================================================================

def _json_default(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)


def extract_actions(result_batch):
    act = result_batch.act
    items = list(act.items()) if isinstance(act, dict) else [
        (a, getattr(act, a)) for a in dir(act)
        if "joint_position" in a and not a.startswith("_")
    ]
    for k, v in items:
        if "joint_position" in k:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            v = np.array(v, dtype=np.float32)
            return v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v.reshape(1, -1)
    raise ValueError("action.joint_position not found in result_batch.act: %s" % act)


def _broadcast_obs(obs):
    data = pickle.dumps(obs)
    size = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    dist.broadcast(buf, src=0)


def _receive_obs():
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.zeros(int(size.item()), dtype=torch.uint8, device="cuda")
    dist.broadcast(buf, src=0)
    return pickle.loads(buf.cpu().numpy().tobytes())


def worker_loop(policy, signal_group):
    rank = dist.get_rank()
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    while True:
        try:
            dist.broadcast(signal, src=0, group=signal_group)
            if signal.item() == 1:
                break
            obs = _receive_obs()
            dist.barrier()
            with torch.no_grad():
                policy.lazy_joint_forward_causal(Batch(obs=obs))
            dist.barrier()
        except Exception:
            log.error("Rank %d error:\n%s", rank, traceback.format_exc())
            break


# ==============================================================================
# Episode loop
# ==============================================================================

def run_episode(
    env,
    policy: GrootSimPolicy,
    signal_group,
    task_description: str,
    initial_state: np.ndarray,
    layers: List[int],
    max_steps: int,
    replan_steps: int,
    num_steps_wait: int,
    vis_cfg: VisualPerturbConfig,
    policy_cfg: PolicyPerturbConfig,
    policy_rng: np.random.Generator,
) -> Dict:
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")

    env.reset()
    obs = env.set_init_state(initial_state)
    apply_object_shift(env, policy_cfg, policy_rng)

    action_plan: collections.deque = collections.deque()
    step_ratio_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + num_steps_wait:
        try:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = perturb_image(np.ascontiguousarray(obs["agentview_image"]), vis_cfg)
            wrist = perturb_image(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]), vis_cfg)

            is_new_chunk = not action_plan
            if is_new_chunk:
                dz_obs = {
                    "video.agentview_rgb":   img[None].astype(np.uint8),
                    "video.eye_in_hand_rgb": wrist[None].astype(np.uint8),
                    "state.joint_position":
                        np.array(obs["robot0_joint_pos"], dtype=np.float64).reshape(1, -1),
                    "state.gripper_position":
                        np.array(obs["robot0_gripper_qpos"], dtype=np.float64)[:1].reshape(1, -1),
                    "annotation.language.language_instruction": task_description,
                }

                clear_recorded_attention()
                hooks = install_attention_hooks(policy, layers)

                signal.fill_(0)
                dist.broadcast(signal, src=0, group=signal_group)
                _broadcast_obs(dz_obs)
                dist.barrier()
                with torch.no_grad():
                    result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
                dist.barrier()

                remove_attention_hooks(hooks)

                action_chunk = extract_actions(result_batch)
                action_plan.extend(action_chunk[:min(replan_steps, len(action_chunk))])

                # Process attention
                recorded = get_recorded_attention()
                for layer_idx in layers:
                    if layer_idx not in recorded:
                        continue
                    layer_rec = recorded[layer_idx]
                    ratio_result = compute_ratio_from_recorded(layer_rec)
                    if ratio_result is None:
                        continue
                    ratio_result["layer"] = layer_idx
                    ratio_result["step"] = t
                    step_ratio_results.append(ratio_result)

                    log.info(
                        "  t=%d layer=%d visual=%.3f linguistic=%.3f ratio=%.3f "
                        "(n_self=%d n_cross=%d)",
                        t, layer_idx,
                        ratio_result["visual_fraction"],
                        ratio_result["linguistic_fraction"],
                        ratio_result.get("visual_linguistic_ratio", 0.0),
                        ratio_result["num_self_attn_calls"],
                        ratio_result["num_cross_attn_calls"],
                    )

            pol_action = action_plan.popleft()
            action, _ = maybe_perturb_action(pol_action, policy_cfg, policy_rng)
            obs, reward, done, _ = env.step(action.tolist())
            t += 1
            if done:
                break
        except Exception:
            log.error("Step error:\n%s", traceback.format_exc())
            break

    # Summarize per layer
    summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_ratios(layer_results)

    return {
        "success": bool(done and reward > 0) if "reward" in dir() else False,
        "num_steps": t,
        "summary": summary,
        "step_ratio_results": step_ratio_results,
    }


# ==============================================================================
# Rank-0 evaluation loop
# ==============================================================================

def eval_rank0(args, policy, signal_group):
    np.random.seed(args.seed)
    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    pol_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    pol_rng = np.random.default_rng(args.seed + 9999)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]
    task_ids = [args.task_id] if args.task_id is not None else list(range(n_tasks))

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    all_results = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_description = task.language

        log.info("\n%s\nTask %d: %s\n%s", "=" * 70, task_id, task_description, "=" * 70)

        for ep_idx in range(min(args.num_episodes, len(init_states))):
            log.info("--- Episode %d/%d ---", ep_idx + 1, args.num_episodes)

            result = run_episode(
                env=env,
                policy=policy,
                signal_group=signal_group,
                task_description=task_description,
                initial_state=init_states[ep_idx],
                layers=args.layers,
                max_steps=max_steps,
                replan_steps=args.replan_steps,
                num_steps_wait=args.num_steps_wait,
                vis_cfg=vis_cfg,
                policy_cfg=pol_cfg,
                policy_rng=pol_rng,
            )
            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            result["visual_perturbation"] = vis_cfg.as_dict()
            result["policy_perturbation"] = pol_cfg.as_dict()
            all_results.append(result)

        env.close()

    # Shutdown workers
    signal.fill_(1)
    dist.broadcast(signal, src=0, group=signal_group)

    # Save results
    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_ratio_results"}
        per_step = {}
        for layer_idx in args.layers:
            layer_key = f"layer_{layer_idx}"
            lr = [s for s in r.get("step_ratio_results", []) if s.get("layer") == layer_idx]
            if lr:
                per_step[layer_key] = [
                    {
                        "step": int(s["step"]),
                        "visual_fraction": float(s["visual_fraction"]),
                        "linguistic_fraction": float(s["linguistic_fraction"]),
                        "visual_linguistic_ratio": float(s.get("visual_linguistic_ratio", 0.0)),
                        "state_fraction": float(s.get("state_fraction", 0.0)),
                        "self_fraction": float(s.get("self_fraction", 0.0)),
                        "num_self_attn_calls": int(s.get("num_self_attn_calls", 0)),
                        "num_cross_attn_calls": int(s.get("num_cross_attn_calls", 0)),
                    }
                    for s in lr
                ]
        entry["per_step_ratios"] = per_step
        serializable.append(entry)

    results_path = out_dir / f"attention_ratio_results_{args.task_suite_name}.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="LIBERO eval with visual/linguistic ratio (DreamZero)")

    # Model
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--no-dit-cache", action="store_true",
                        help="Disable DIT cache (simpler forward passes, better for analysis)")

    # LIBERO
    parser.add_argument("--task-suite-name", type=str, default="libero_10",
                        choices=list(_MAX_STEPS.keys()))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30])

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
    parser.add_argument("--output-dir", type=str, default="results/attention_ratio_dreamzero")

    args = parser.parse_args()

    # DIT cache setting (disable for cleaner analysis)
    os.environ["ENABLE_DIT_CACHE"] = "false" if args.no_dit_cache else "true"
    os.environ["ATTENTION_BACKEND"] = "FA2"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank{rank}] %(asctime)s %(message)s",
        force=True,
    )

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
    signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=10))

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
    )

    if rank == 0:
        # Print model info
        try:
            blocks = policy.trained_model.action_head.model.blocks
            block0 = blocks[0]
            sa = block0.self_attn
            log.info(
                "Model info: %d blocks, frame_seqlen=%d, num_action_per_block=%d, "
                "num_state_per_block=%d, num_heads=%d",
                len(blocks), sa.frame_seqlen, sa.num_action_per_block,
                sa.num_state_per_block, sa.num_heads,
            )
        except Exception as e:
            log.warning("Could not read model info: %s", e)

        eval_rank0(args, policy, signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
