#!/usr/bin/env python3
"""
LIBERO evaluation with attention-segmentation IoU analysis for DreamZero.

DreamZero token layout (inference, is_tf=False):
  [first_image (frame_seqlen)] [image_blocks (N*frame_seqlen)] [action_tokens (N*num_action)] [state_tokens (N*num_state)]

For each action block i, the action tokens attend to:
  k_context = [first_image | image_blocks_0..i | action_block_i | state_block_i]

We extract the action-to-first_image attention (the conditioning observation frame),
reshape from (frame_seqlen,) to (H_feat, W_feat), upsample to image resolution,
then compute IoU against object segmentation masks.

Must be launched with torchrun (same as dreamzero_eval.py):
  torchrun --standalone --nproc_per_node=4 \\
    analysis/attention/evaluate_attention_iou_dreamzero.py \\
    --model-path /path/to/dreamzero_libero_lora \\
    --task-suite-name libero_10 --num-episodes 5 \\
    --layers 10 20 30 --output-dir results/attention_iou_dreamzero
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
from typing import Dict, List, Optional, Any, Tuple

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
from libero.libero.envs.env_wrapper import SegmentationRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image
from tianshou.data import Batch

from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    find_segmentation_key,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

_RECORDED: Dict[int, Dict[str, Any]] = {}
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
    """
    Install hooks on CausalWanSelfAttention.attn (AttentionModule) to capture
    action-block attention calls.  Action blocks are identified by:
        q.shape[1] == num_action_per_block
    The k_context layout for action block i is:
        [first_image (frame_seqlen)] [image_blocks (variable)] [action (n_act)] [state (n_state)]
    We record the attention from action queries to the first_image portion,
    which maps to the current observation frame.
    """
    global _HOOKS
    _RECORDED.clear()

    try:
        blocks = policy.trained_model.action_head.model.blocks
    except AttributeError as e:
        raise RuntimeError(
            f"Cannot find DiT blocks at policy.trained_model.action_head.model.blocks: {e}"
        )

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
        _RECORDED[layer_idx] = {"action_attn_calls": []}

        def _make_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block
            frame_seqlen = parent.frame_seqlen

            def hook(module, args, output):
                if len(args) < 3:
                    return
                q, k, v = args[0], args[1], args[2]
                # q: (B, Lq, num_heads, head_dim)
                # k: (B, Lk, num_heads, head_dim)
                if q.shape[1] != n_act:
                    return  # Not an action block call

                Lk = k.shape[1]
                # k_context: [first_image (frame_seqlen)] [image_blocks] [action (n_act)] [state (n_state)]
                visual_end = Lk - n_act - n_state  # total image tokens in k_context
                first_image_end = frame_seqlen      # first_image tokens

                if first_image_end > visual_end:
                    return  # Degenerate layout

                with torch.no_grad():
                    q_f = q.float().permute(0, 2, 1, 3)   # (B, H, n_act, d)
                    k_f = k.float().permute(0, 2, 1, 3)   # (B, H, Lk, d)
                    scale = q.shape[-1] ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, n_act, Lk)

                    # Spatial heatmap over first_image tokens (current observation)
                    # Average over batch, heads, and action tokens → (frame_seqlen,)
                    attn_to_first = attn[0, :, :, :first_image_end]  # (H, n_act, frame_seqlen)
                    spatial_attn = attn_to_first.mean(dim=(0, 1)).cpu().float().numpy()  # (frame_seqlen,)

                    # Also record attention to all image tokens (first_image + image_blocks)
                    attn_to_all_img = attn[0, :, :, :visual_end]   # (H, n_act, visual_end)
                    all_img_attn = attn_to_all_img.mean(dim=(0, 1)).cpu().float().numpy()  # (visual_end,)

                _RECORDED[idx]["action_attn_calls"].append({
                    "spatial_attn_first_image": spatial_attn,    # (frame_seqlen,)
                    "attn_to_all_img": all_img_attn,             # (visual_end,)
                    "frame_seqlen": frame_seqlen,
                    "Lk": Lk,
                    "visual_end": visual_end,
                    "n_act": n_act,
                    "n_state": n_state,
                })
            return hook

        handles.append(self_attn.attn.register_forward_hook(_make_hook(layer_idx, self_attn)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Heatmap construction
# ==============================================================================

def _infer_grid(frame_seqlen: int) -> Tuple[int, int]:
    """Infer (H_feat, W_feat) from frame_seqlen. Assumes square grid."""
    side = int(round(frame_seqlen ** 0.5))
    if side * side == frame_seqlen:
        return side, side
    # Try common non-square sizes
    for h in range(1, frame_seqlen + 1):
        if frame_seqlen % h == 0:
            w = frame_seqlen // h
            if abs(h - w) <= 4:  # Roughly square
                return h, w
    raise ValueError(f"Cannot infer 2D grid from frame_seqlen={frame_seqlen}")


def build_heatmap(
    attn_call: Dict[str, Any],
    image_resolution: int,
    use_full_image_attn: bool = False,
) -> Optional[np.ndarray]:
    """
    Build a spatial attention heatmap at image_resolution.

    If use_full_image_attn=False: uses attention to first_image only (current obs frame).
    If use_full_image_attn=True:  uses attention to all image tokens in k_context.

    Returns normalized heatmap in [0, 1] of shape (image_resolution, image_resolution),
    or None if spatial reshape fails.
    """
    if use_full_image_attn:
        spatial_attn = attn_call["attn_to_all_img"]
        visual_end = attn_call["visual_end"]
        frame_seqlen = attn_call["frame_seqlen"]
        # Sum over all image blocks, taking the last frame_seqlen as the "latest" frame
        if len(spatial_attn) >= frame_seqlen:
            spatial_attn = spatial_attn[-frame_seqlen:]  # Latest image block
        else:
            return None
        feat_len = frame_seqlen
    else:
        spatial_attn = attn_call["spatial_attn_first_image"]
        feat_len = attn_call["frame_seqlen"]

    try:
        h_feat, w_feat = _infer_grid(feat_len)
    except ValueError as e:
        log.warning("Cannot build heatmap: %s", e)
        return None

    heatmap = spatial_attn.reshape(h_feat, w_feat).astype(np.float32)

    # Upsample to image resolution using bilinear interpolation
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_t, size=(image_resolution, image_resolution), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # Normalize to [0, 1]
    hmin, hmax = heatmap_up.min(), heatmap_up.max()
    if hmax > hmin:
        heatmap_up = (heatmap_up - hmin) / (hmax - hmin)
    else:
        heatmap_up = np.zeros_like(heatmap_up)

    return heatmap_up


# ==============================================================================
# LIBERO helpers
# ==============================================================================

def _json_default(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, bool): return bool(o)
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
    env: SegmentationRenderEnv,
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
    threshold_method: str,
    threshold_value: float,
    output_dir: str,
    episode_prefix: str,
    save_heatmaps: bool,
    image_resolution: int,
) -> Dict:
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")

    env.reset()
    obs = env.set_init_state(initial_state)
    apply_object_shift(env, policy_cfg, policy_rng)

    action_plan: collections.deque = collections.deque()
    # per-layer results: list of step dicts
    step_iou_results: Dict[int, List[Dict]] = {layer_idx: [] for layer_idx in layers}
    t = 0
    done = False
    reward = 0.0

    # Find segmentation key on first obs
    seg_key = None
    try:
        seg_key = find_segmentation_key(obs)
    except Exception:
        pass

    while t < max_steps + num_steps_wait:
        try:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = perturb_image(np.ascontiguousarray(obs["agentview_image"]), vis_cfg)
            wrist = perturb_image(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]), vis_cfg)

            # Get segmentation mask
            seg_mask = None
            if seg_key is not None and seg_key in obs:
                seg_mask = obs[seg_key]
                if seg_mask.ndim == 3:
                    seg_mask = seg_mask[:, :, 0]  # take first channel
                seg_mask = (seg_mask > 0).astype(np.float32)

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

                # Process attention and compute IoU per layer
                recorded = get_recorded_attention()
                for layer_idx in layers:
                    if layer_idx not in recorded:
                        continue
                    calls = recorded[layer_idx].get("action_attn_calls", [])
                    if not calls:
                        continue

                    # Use the first action block (block_idx=0) for IoU analysis
                    call = calls[0]
                    heatmap = build_heatmap(call, image_resolution=image_resolution)
                    if heatmap is None:
                        continue

                    step_result = {
                        "step": t,
                        "layer": layer_idx,
                        "frame_seqlen": int(call["frame_seqlen"]),
                        "num_action_blocks": len(calls),
                    }

                    if seg_mask is not None and seg_mask.shape[0] == image_resolution:
                        iou_metrics = compute_attention_object_iou(
                            heatmap=heatmap,
                            seg_mask=seg_mask,
                            threshold_method=threshold_method,
                            threshold_value=threshold_value,
                        )
                        step_result.update(iou_metrics)
                        log.info(
                            "  t=%d layer=%d IoU=%.3f Dice=%.3f (method=%s)",
                            t, layer_idx,
                            iou_metrics.get("iou", 0.0),
                            iou_metrics.get("dice", 0.0),
                            threshold_method,
                        )
                    else:
                        log.info("  t=%d layer=%d: no seg_mask, skipping IoU", t, layer_idx)

                    step_iou_results[layer_idx].append(step_result)

                    # Optionally save heatmap visualization
                    if save_heatmaps:
                        _save_heatmap(
                            heatmap=heatmap,
                            img=img,
                            seg_mask=seg_mask,
                            output_dir=output_dir,
                            prefix=f"{episode_prefix}_layer{layer_idx}_t{t:04d}",
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
        layer_results = step_iou_results.get(layer_idx, [])
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_iou(layer_results)

    return {
        "success": bool(done and reward > 0),
        "num_steps": t,
        "summary": summary,
        "step_iou_results": {f"layer_{li}": step_iou_results[li] for li in layers},
    }


def _save_heatmap(
    heatmap: np.ndarray,
    img: np.ndarray,
    seg_mask: Optional[np.ndarray],
    output_dir: str,
    prefix: str,
):
    """Save a 2-panel visualization: original image + attention heatmap overlay."""
    fig, axes = plt.subplots(1, 2 if seg_mask is None else 3, figsize=(12, 4))

    # Panel 1: original image
    axes[0].imshow(img)
    axes[0].set_title("Observation")
    axes[0].axis("off")

    # Panel 2: attention heatmap overlay
    axes[1].imshow(img)
    axes[1].imshow(heatmap, alpha=0.5, cmap="hot")
    axes[1].set_title("Action→Image Attention")
    axes[1].axis("off")

    # Panel 3: segmentation mask (if available)
    if seg_mask is not None and len(axes) > 2:
        axes[2].imshow(img)
        axes[2].imshow(seg_mask, alpha=0.4, cmap="Blues")
        axes[2].set_title("Segmentation Mask")
        axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


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
    if args.save_heatmaps:
        (out_dir / "heatmaps").mkdir(exist_ok=True)

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    all_results = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        try:
            env = SegmentationRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
        except Exception as e:
            log.warning("SegmentationRenderEnv failed (%s), falling back to OffScreenRenderEnv", e)
            from libero.libero.envs import OffScreenRenderEnv
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

            task_slug = task_description.replace(" ", "_")[:50]
            episode_prefix = f"task{task_id}_{task_slug}_ep{ep_idx}"
            heatmap_dir = str(out_dir / "heatmaps") if args.save_heatmaps else str(out_dir)

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
                threshold_method=args.threshold_method,
                threshold_value=args.threshold_value,
                output_dir=heatmap_dir,
                episode_prefix=episode_prefix,
                save_heatmaps=args.save_heatmaps,
                image_resolution=LIBERO_ENV_RESOLUTION,
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
        entry = {k: v for k, v in r.items() if k != "step_iou_results"}
        per_step = {}
        for layer_idx in args.layers:
            layer_key = f"layer_{layer_idx}"
            lr = r.get("step_iou_results", {}).get(layer_key, [])
            if lr:
                per_step[layer_key] = [
                    {
                        "step": int(s.get("step", -1)),
                        "iou": float(s.get("iou", 0.0)),
                        "dice": float(s.get("dice", 0.0)),
                        "attention_mass": float(s.get("attention_mass", 0.0)),
                        "frame_seqlen": int(s.get("frame_seqlen", -1)),
                        "num_action_blocks": int(s.get("num_action_blocks", 0)),
                    }
                    for s in lr
                ]
        entry["per_step_iou"] = per_step
        serializable.append(entry)

    results_path = out_dir / f"attention_iou_results_{args.task_suite_name}.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LIBERO eval with attention IoU vs segmentation (DreamZero)"
    )

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

    # IoU thresholding
    parser.add_argument("--threshold-method", type=str, default="percentile",
                        choices=["percentile", "otsu", "fixed", "top_k"],
                        help="Method to binarize the attention heatmap")
    parser.add_argument("--threshold-value", type=float, default=90.0,
                        help="Threshold value (percentile for 'percentile', threshold for 'fixed', etc.)")

    # Visualization
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save attention heatmap visualizations")

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
    parser.add_argument("--output-dir", type=str, default="results/attention_iou_dreamzero")

    args = parser.parse_args()

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
        try:
            blocks = policy.trained_model.action_head.model.blocks
            block0 = blocks[0]
            sa = block0.self_attn
            frame_seqlen = sa.frame_seqlen
            try:
                h_feat, w_feat = _infer_grid(frame_seqlen)
                spatial_str = f"{h_feat}x{w_feat}={frame_seqlen}"
            except ValueError:
                spatial_str = f"?x?={frame_seqlen}"
            log.info(
                "Model info: %d blocks, frame_seqlen=%d (spatial=%s), "
                "num_action_per_block=%d, num_state_per_block=%d, num_heads=%d",
                len(blocks), frame_seqlen, spatial_str,
                sa.num_action_per_block, sa.num_state_per_block, sa.num_heads,
            )
        except Exception as e:
            log.warning("Could not read model info: %s", e)

        eval_rank0(args, policy, signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
