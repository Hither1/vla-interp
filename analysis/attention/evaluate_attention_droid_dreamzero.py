#!/usr/bin/env python3
"""Offline attention analysis for DreamZero-DROID on real robot ZED camera frames.

Takes video frames from a directory (exported from Google Drive videos) and runs
the DreamZero-DROID checkpoint to compute attention IoU and attention ratio.

DreamZero-DROID token layout (self-attention):
  [visual_tokens (current+history frames)] [action_tokens] [state_tokens]
Text is conditioned via cross-attention to UMT5 text embeddings (not in the self-attn sequence).

Metrics:
  - Attention ratio: action→visual_tokens / cross-attn action→text_tokens
  - Attention IoU:   spatial heatmap (action→first_image_tokens) vs object seg mask

Must be launched with torchrun (same as dreamzero DROID eval):
  torchrun --standalone --nproc_per_node=<N_GPUS> \\
    analysis/attention/evaluate_attention_droid_dreamzero.py \\
    --model-path /path/to/dreamzero_droid_checkpoint \\
    --data-dir /path/to/zed_frames \\
    --prompt "pick up the red cup" \\
    --layers 10 20 30 --output-dir results/droid_dreamzero

Input layout expected (--data-dir):
  <data_dir>/
    frame_0000.png            # exterior camera 1 (required)
    frame_0000_ext2.png       # exterior camera 2 (optional, zeros if missing)
    frame_0000_wrist.png      # wrist camera (optional, zeros if missing)
    metadata.json             # optional: {"prompt": "...", "joint_positions": [...],
                              #             "gripper_positions": [...]}

OR supply --video (exterior cam 1), --video-ext2, --video-wrist for video files.

For IoU:
  Option A: --mask-dir with frame_XXXX_mask.npy binary masks
  Option B: --use-sam3 --object-desc "red cup" (SAM3, auto-downloads from HuggingFace)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import pickle
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DREAMZERO_DIR = str(_PROJECT_ROOT / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)
sys.path.insert(0, str(_SCRIPT_DIR))

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanAttentionBlock, CausalWanSelfAttention,
)
from tianshou.data import Batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# DROID images: the server/client uses 180×320 but the policy internally resizes.
# We pass 224×224 to match the policy server config.
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ==============================================================================
# Data loading
# ==============================================================================

def _load_img(path: pathlib.Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _zeros_frame() -> np.ndarray:
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def _resize(img: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS))


def load_frames_from_dir(
    data_dir: pathlib.Path,
    frame_step: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """Load ext1, ext2, wrist frames from a frame directory.

    Returns (ext1_frames, ext2_frames, wrist_frames, frame_indices).
    Missing ext2/wrist frames are filled with zeros.
    """
    candidates = sorted(data_dir.glob("frame_*.png")) + sorted(data_dir.glob("*.png"))
    ext1_paths = sorted(set(
        p for p in candidates
        if "_wrist" not in p.stem and "_ext2" not in p.stem and "_mask" not in p.stem
    ))
    ext1_paths = ext1_paths[::frame_step]

    ext1_frames, ext2_frames, wrist_frames, indices = [], [], [], []
    for p in ext1_paths:
        idx = int("".join(filter(str.isdigit, p.stem)) or 0)
        ext1_frames.append(_resize(_load_img(p)))

        ext2_p = p.parent / f"{p.stem}_ext2{p.suffix}"
        ext2_frames.append(_resize(_load_img(ext2_p)) if ext2_p.exists() else _zeros_frame())

        wrist_p = p.parent / f"{p.stem}_wrist{p.suffix}"
        wrist_frames.append(_resize(_load_img(wrist_p)) if wrist_p.exists() else _zeros_frame())

        indices.append(idx)

    return ext1_frames, ext2_frames, wrist_frames, indices


def load_frames_from_videos(
    video_ext1: str,
    video_ext2: Optional[str],
    video_wrist: Optional[str],
    frame_step: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    def _read(path: Optional[str]) -> List[np.ndarray]:
        if path is None:
            return []
        cap = cv2.VideoCapture(path)
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_step == 0:
                frames.append(_resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            i += 1
        cap.release()
        return frames

    ext1 = _read(video_ext1)
    ext2 = _read(video_ext2)
    wrist = _read(video_wrist)
    n = len(ext1)
    indices = list(range(0, n * frame_step, frame_step))

    if not ext2:
        ext2 = [_zeros_frame()] * n
    if not wrist:
        wrist = [_zeros_frame()] * n

    # Align lengths
    n = min(len(ext1), len(ext2), len(wrist))
    return ext1[:n], ext2[:n], wrist[:n], indices[:n]


def load_masks_from_dir(
    data_dir: pathlib.Path,
    indices: List[int],
) -> List[Optional[np.ndarray]]:
    masks = []
    for idx in indices:
        p = data_dir / f"frame_{idx:04d}_mask.npy"
        if not p.exists():
            p = data_dir / f"{idx:04d}_mask.npy"
        masks.append((np.load(str(p)) > 0).astype(np.float32) if p.exists() else None)
    return masks


def load_metadata(data_dir: pathlib.Path, n: int) -> Dict:
    p = data_dir / "metadata.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "prompt": "",
        "joint_positions": [[0.0] * 7] * n,
        "gripper_positions": [[0.0]] * n,
    }


# ==============================================================================
# SAM3 segmentation (optional)
# ==============================================================================

def _try_import_sam3():
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        return build_sam3_image_model, Sam3Processor
    except ImportError as e:
        raise ImportError(
            f"SAM3 not available: {e}\n"
            "Install with: cd /path/to/sam3 && pip install -e ."
        ) from e


def load_sam3_model(checkpoint_path: str = "", device: str = "cuda"):
    """Load SAM3 image model. Auto-downloads from HuggingFace if checkpoint_path is empty."""
    build_sam3_image_model, Sam3Processor = _try_import_sam3()
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path if checkpoint_path else None,
        device=device,
        eval_mode=True,
    )
    return Sam3Processor(model, device=device)


def segment_with_sam3(
    image: np.ndarray,
    object_desc: str,
    processor,
    confidence_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """Run SAM3 text-prompted segmentation. Returns best binary mask (H, W) or None."""
    img_pil = Image.fromarray(image)
    state = processor.set_image(img_pil)
    processor.set_confidence_threshold(confidence_threshold, state)
    state = processor.set_text_prompt(prompt=object_desc, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or len(masks) == 0:
        log.warning("SAM3: no detection for '%s'", object_desc)
        return None

    return masks[int(scores.argmax())].cpu().float().numpy()


# ==============================================================================
# Attention hook machinery (mirrors evaluate_attention_ratio_dreamzero.py)
# ==============================================================================

_RECORDED: Dict[int, Dict[str, Any]] = {}
_HOOKS: List = []


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, Any]]:
    out = dict(_RECORDED)
    _RECORDED.clear()
    return out


def remove_attention_hooks(handles=None):
    global _HOOKS
    for h in (handles or _HOOKS):
        try:
            h.remove()
        except Exception:
            pass
    _HOOKS = []


def install_attention_hooks(policy: GrootSimPolicy, layers: List[int]) -> List:
    global _HOOKS
    _RECORDED.clear()

    try:
        blocks = policy.trained_model.action_head.model.blocks
    except AttributeError as e:
        raise RuntimeError(f"Cannot find DiT blocks: {e}")

    handles = []

    for layer_idx in layers:
        if layer_idx >= len(blocks):
            log.warning("Layer %d out of range (%d blocks)", layer_idx, len(blocks))
            continue
        block = blocks[layer_idx]
        if not isinstance(block, CausalWanAttentionBlock):
            log.warning("Block %d is not CausalWanAttentionBlock", layer_idx)
            continue

        self_attn: CausalWanSelfAttention = block.self_attn
        _RECORDED[layer_idx] = {"self_attn": [], "cross_attn": [], "action_register_length": None}

        def _make_pre_hook(idx):
            def pre_hook(module, args):
                if len(args) >= 5 and args[4] is not None:
                    _RECORDED[idx]["action_register_length"] = args[4]
            return pre_hook

        handles.append(self_attn.register_forward_pre_hook(_make_pre_hook(layer_idx)))

        def _make_self_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block
            frame_seqlen = parent.frame_seqlen

            def hook(module, args, output):
                if len(args) < 3:
                    return
                q, k, v = args[0], args[1], args[2]
                # q: (B, Lq, H, d);  action block identified by q.shape[1] == n_act
                if q.shape[1] != n_act:
                    return

                Lk = k.shape[1]
                visual_end = Lk - n_act - n_state

                with torch.no_grad():
                    q_f = q.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = q.shape[-1] ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, n_act, Lk)
                    attn_np = attn[0].mean(0).cpu().float().numpy()  # (n_act, Lk)

                _RECORDED[idx]["self_attn"].append({
                    "attn": attn_np,
                    "Lk": Lk,
                    "visual_end": visual_end,
                    "n_act": n_act,
                    "n_state": n_state,
                    "frame_seqlen": frame_seqlen,
                })
            return hook

        handles.append(self_attn.attn.register_forward_hook(_make_self_attn_hook(layer_idx, self_attn)))

        def _make_cross_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block

            def hook(module, args, output):
                if len(args) < 2:
                    return
                normed_x = args[0]   # (B, S, C)
                context = args[1]    # (B, T, C) text embeddings

                action_register_length = _RECORDED[idx].get("action_register_length")
                if action_register_length is None or action_register_length == 0:
                    return

                B, S, C = normed_x.shape
                n = module.num_heads
                d = module.head_dim

                chunk_size = action_register_length // (n_act + n_state)
                action_horizon = chunk_size * n_act
                state_horizon = chunk_size * n_state
                action_start = S - action_horizon - state_horizon
                action_end = S - state_horizon
                if action_start < 0 or action_end > S:
                    return

                with torch.no_grad():
                    x_act = normed_x[:, action_start:action_end]
                    q = module.norm_q(module.q(x_act)).view(B, -1, n, d)
                    if hasattr(module, 'k_img'):
                        text_context = context[:, 257:]
                    else:
                        text_context = context
                    if text_context.shape[1] == 0:
                        return
                    k = module.norm_k(module.k(text_context)).view(B, -1, n, d)

                    q_f = q.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = d ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, AH, T)
                    attn_np = attn[0].mean(0).cpu().float().numpy()  # (AH, T)

                _RECORDED[idx]["cross_attn"].append({
                    "attn": attn_np,
                    "action_horizon": action_horizon,
                    "T": text_context.shape[1],
                })
            return hook

        handles.append(block.cross_attn.register_forward_hook(_make_cross_attn_hook(layer_idx, self_attn)))

    _HOOKS = handles
    return handles


# ==============================================================================
# Metric computation
# ==============================================================================

def compute_ratio_from_recorded(layer_rec: Dict) -> Optional[Dict]:
    self_calls = layer_rec.get("self_attn", [])
    cross_calls = layer_rec.get("cross_attn", [])
    if not self_calls:
        return None

    visual_fracs = []
    for c in self_calls:
        attn = c["attn"]  # (n_act, Lk)
        vis_end = c["visual_end"]
        total = max(float(attn.sum()), 1e-8)
        visual_fracs.append(float(attn[:, :vis_end].sum()) / total)
    visual_mass = float(np.mean(visual_fracs))

    if cross_calls:
        ling_vals = [float(c["attn"].mean()) * c["T"] for c in cross_calls]
        linguistic_mass = float(np.mean(ling_vals))
    else:
        linguistic_mass = 0.0

    ratio = visual_mass / linguistic_mass if linguistic_mass > 1e-8 else (float("inf") if visual_mass > 1e-8 else 0.0)

    state_fracs, self_fracs = [], []
    for c in self_calls:
        attn = c["attn"]
        vis_end = c["visual_end"]
        n_act = c["n_act"]
        n_state = c["n_state"]
        total = max(float(attn.sum()), 1e-8)
        state_fracs.append(float(attn[:, vis_end + n_act:].sum()) / total)
        self_fracs.append(float(attn[:, vis_end : vis_end + n_act].sum()) / total)

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass,
        "linguistic_fraction": linguistic_mass,
        "state_fraction": float(np.mean(state_fracs)),
        "self_fraction": float(np.mean(self_fracs)),
        "num_self_attn_calls": len(self_calls),
        "num_cross_attn_calls": len(cross_calls),
    }


def _infer_grid(frame_seqlen: int) -> Tuple[int, int]:
    side = int(round(frame_seqlen ** 0.5))
    if side * side == frame_seqlen:
        return side, side
    for h in range(1, frame_seqlen + 1):
        if frame_seqlen % h == 0:
            w = frame_seqlen // h
            if abs(h - w) <= 4:
                return h, w
    raise ValueError(f"Cannot infer grid from frame_seqlen={frame_seqlen}")


def build_heatmap_from_self_attn(
    calls: List[Dict],
    image_resolution: int,
) -> Optional[np.ndarray]:
    """Build spatial heatmap from the first action-block self-attention call (→ first image frame)."""
    if not calls:
        return None
    c = calls[0]
    frame_seqlen = c["frame_seqlen"]
    attn = c["attn"]  # (n_act, Lk)
    spatial_attn = attn[:, :frame_seqlen].mean(axis=0)  # (frame_seqlen,)

    try:
        h_feat, w_feat = _infer_grid(frame_seqlen)
    except ValueError:
        return None

    heatmap = spatial_attn.reshape(h_feat, w_feat).astype(np.float32)
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_t, size=(image_resolution, image_resolution), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    mn, mx = heatmap_up.min(), heatmap_up.max()
    if mx > mn:
        heatmap_up = (heatmap_up - mn) / (mx - mn)
    else:
        heatmap_up = np.zeros_like(heatmap_up)
    return heatmap_up


def compute_iou_from_heatmap(heatmap, seg_mask, threshold_method="percentile", threshold_value=90.0):
    from attention_iou import threshold_attention, compute_iou, compute_dice
    binary = threshold_attention(heatmap, threshold_method, threshold_value)
    seg_bool = (seg_mask > 0)
    return {
        "iou": compute_iou(binary, seg_bool),
        "dice": compute_dice(binary, seg_bool),
        "attention_mass": float(heatmap[seg_bool].sum() / max(heatmap.sum(), 1e-8)),
    }


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def summarize_ratios(step_results: List[Dict]) -> Dict:
    if not step_results:
        return {}
    ratios = [r["visual_linguistic_ratio"] for r in step_results if np.isfinite(r.get("visual_linguistic_ratio", float("nan")))]
    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
        },
        "visual_fraction": {"mean": float(np.mean([r["visual_fraction"] for r in step_results]))},
        "linguistic_fraction": {"mean": float(np.mean([r["linguistic_fraction"] for r in step_results]))},
        "num_steps": len(step_results),
    }


# ==============================================================================
# Distributed helpers
# ==============================================================================

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


def extract_action(result_batch) -> np.ndarray:
    act = result_batch.act
    items = list(act.items()) if isinstance(act, dict) else [
        (a, getattr(act, a)) for a in dir(act) if "joint_position" in a and not a.startswith("_")
    ]
    for k, v in items:
        if "joint_position" in k:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            v = np.array(v, dtype=np.float32)
            return v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v.reshape(1, -1)
    raise ValueError("action.joint_position not found in result_batch.act")


# ==============================================================================
# Rank-0 analysis loop
# ==============================================================================

def run_analysis_rank0(args, policy, signal_group):
    # ── Load frames ─────────────────────────────────────────────────────────
    if args.video:
        ext1_frames, ext2_frames, wrist_frames, frame_indices = load_frames_from_videos(
            args.video, args.video_ext2, args.video_wrist, args.frame_step,
        )
        data_dir = pathlib.Path(args.video).parent
    else:
        data_dir = pathlib.Path(args.data_dir)
        ext1_frames, ext2_frames, wrist_frames, frame_indices = load_frames_from_dir(
            data_dir, args.frame_step,
        )

    if not ext1_frames:
        log.error("No frames found.")
        return

    log.info("Loaded %d frames", len(ext1_frames))

    meta = load_metadata(data_dir, len(ext1_frames))
    prompt = args.prompt or meta.get("prompt", "")
    joint_positions = meta.get("joint_positions", [[0.0] * 7] * len(ext1_frames))
    gripper_positions = meta.get("gripper_positions", [[0.0]] * len(ext1_frames))
    n = len(ext1_frames)
    while len(joint_positions) < n:
        joint_positions.append([0.0] * 7)
    while len(gripper_positions) < n:
        gripper_positions.append([0.0])

    # ── Segmentation (IoU) ─────────────────────────────────────────────────
    sam3_processor = None
    if args.use_sam3:
        log.info("Loading SAM3 (version=%s) …", args.sam3_version)
        sam3_processor = load_sam3_model(
            checkpoint_path=args.sam3_checkpoint,
            device="cuda",
        )

    provided_masks = None
    if args.mask_dir:
        provided_masks = load_masks_from_dir(pathlib.Path(args.mask_dir), frame_indices)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        (out_dir / "heatmaps").mkdir(exist_ok=True)

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    all_step_results: List[Dict] = []

    # Frame buffer for DreamZero (accumulates T frames to simulate temporal context)
    frame_buffer_ext1: List[np.ndarray] = []
    frame_buffer_ext2: List[np.ndarray] = []
    frame_buffer_wrist: List[np.ndarray] = []

    for i, (ext1, ext2, wrist, frame_idx) in enumerate(
        zip(ext1_frames, ext2_frames, wrist_frames, frame_indices)
    ):
        log.info("Frame %d/%d (idx=%d)", i + 1, n, frame_idx)

        frame_buffer_ext1.append(ext1)
        frame_buffer_ext2.append(ext2)
        frame_buffer_wrist.append(wrist)

        # Use up to args.num_context_frames frames of history
        T = min(len(frame_buffer_ext1), args.num_context_frames)
        if i == 0:
            T = 1  # First call: single frame

        ext1_stack = np.stack(frame_buffer_ext1[-T:]).astype(np.uint8)  # (T, H, W, 3)
        ext2_stack = np.stack(frame_buffer_ext2[-T:]).astype(np.uint8)
        wrist_stack = np.stack(frame_buffer_wrist[-T:]).astype(np.uint8)

        joint_pos = np.array(joint_positions[i], dtype=np.float64).reshape(1, -1)
        gripper_pos = np.array(gripper_positions[i], dtype=np.float64).reshape(1, -1)

        dz_obs = {
            # DreamZero DROID uses 1-indexed cameras:
            # ext1 → exterior_image_1_left (left external), ext2 → exterior_image_2_left (right external)
            "video.exterior_image_1_left": ext1_stack,
            "video.exterior_image_2_left": ext2_stack,
            "video.wrist_image_left": wrist_stack,
            "state.joint_position": joint_pos,
            "state.gripper_position": gripper_pos,
            # DreamZero DROID uses 'annotation.language.action_text' key
            "annotation.language.action_text": prompt,
        }

        clear_recorded_attention()
        hooks = install_attention_hooks(policy, args.layers)

        signal.fill_(0)
        dist.broadcast(signal, src=0, group=signal_group)
        _broadcast_obs(dz_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
        dist.barrier()

        remove_attention_hooks(hooks)
        recorded = get_recorded_attention()

        # Get segmentation mask
        seg_mask = None
        if provided_masks is not None:
            seg_mask = provided_masks[i]
        elif sam3_processor is not None and args.object_desc:
            seg_mask = segment_with_sam3(
                image=ext1,
                object_desc=args.object_desc,
                processor=sam3_processor,
                confidence_threshold=args.sam3_confidence,
            )

        # Process per layer
        for layer_idx in args.layers:
            if layer_idx not in recorded:
                continue
            layer_rec = recorded[layer_idx]

            ratio_result = compute_ratio_from_recorded(layer_rec)
            if ratio_result is None:
                continue

            frame_result = dict(ratio_result)
            frame_result["layer"] = layer_idx
            frame_result["frame_idx"] = frame_idx

            # Heatmap & IoU
            heatmap = build_heatmap_from_self_attn(
                layer_rec.get("self_attn", []), IMAGE_HEIGHT,
            )
            if heatmap is not None and seg_mask is not None:
                seg_resized = cv2.resize(
                    seg_mask.astype(np.float32),
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                iou_metrics = compute_iou_from_heatmap(
                    heatmap, seg_resized,
                    threshold_method=args.threshold_method,
                    threshold_value=args.threshold_value,
                )
                frame_result.update(iou_metrics)
                log.info(
                    "  layer=%d ratio=%.3f iou=%.3f mass=%.3f",
                    layer_idx,
                    ratio_result["visual_linguistic_ratio"],
                    iou_metrics["iou"],
                    iou_metrics["attention_mass"],
                )
            else:
                log.info(
                    "  layer=%d ratio=%.3f visual=%.3f linguistic=%.3f",
                    layer_idx,
                    ratio_result["visual_linguistic_ratio"],
                    ratio_result["visual_fraction"],
                    ratio_result["linguistic_fraction"],
                )

            all_step_results.append(frame_result)

            if args.save_heatmaps and heatmap is not None:
                from attention_iou import overlay_heatmap
                viz = overlay_heatmap(ext1, heatmap)
                save_p = out_dir / "heatmaps" / f"frame{frame_idx:04d}_layer{layer_idx}_heatmap.png"
                Image.fromarray(viz).save(str(save_p))

    # Shutdown workers
    signal.fill_(1)
    dist.broadcast(signal, src=0, group=signal_group)

    # ── Save results ──────────────────────────────────────────────────────
    per_layer_summary = {}
    for layer_idx in args.layers:
        lr = [r for r in all_step_results if r.get("layer") == layer_idx]
        if lr:
            per_layer_summary[f"layer_{layer_idx}"] = summarize_ratios(lr)
            iou_vals = [r["iou"] for r in lr if "iou" in r]
            if iou_vals:
                per_layer_summary[f"layer_{layer_idx}"]["iou_mean"] = float(np.mean(iou_vals))
                per_layer_summary[f"layer_{layer_idx}"]["iou_std"] = float(np.std(iou_vals))

    output = {
        "prompt": prompt,
        "model_path": args.model_path,
        "num_frames": len(ext1_frames),
        "num_context_frames": args.num_context_frames,
        "layers": args.layers,
        "per_layer_summary": per_layer_summary,
        "per_frame_results": all_step_results,
    }

    results_path = out_dir / "attention_results_dreamzero_droid.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    if all_step_results:
        _save_summary_plot(all_step_results, args.layers, prompt, out_dir)


def _save_summary_plot(step_results, layers, prompt, out_dir):
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4), squeeze=False)
    for col, layer_idx in enumerate(layers):
        ax = axes[0][col]
        lr = [r for r in step_results if r.get("layer") == layer_idx]
        if not lr:
            continue
        xs = [r["frame_idx"] for r in lr]
        ratios = [r["visual_linguistic_ratio"] for r in lr]
        ax.plot(xs, ratios, "o-", linewidth=2)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"Layer {layer_idx} V/L Ratio", fontweight="bold")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Visual / Linguistic Ratio")
        ax.grid(alpha=0.3)

        iou_vals = [r.get("iou") for r in lr if "iou" in r]
        if iou_vals:
            ax2 = ax.twinx()
            iou_xs = [r["frame_idx"] for r in lr if "iou" in r]
            ax2.plot(iou_xs, iou_vals, "s--", color="green", alpha=0.7, linewidth=1.5)
            ax2.set_ylabel("IoU (green)")
            ax2.set_ylim(0, 1)

    fig.suptitle(f"DreamZero-DROID Attention: {prompt[:60]}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_dir / "summary_plot.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Offline attention analysis for DreamZero-DROID on real ZED camera frames"
    )

    # Model
    parser.add_argument("--model-path", required=True,
                        help="Path to DreamZero-DROID checkpoint directory")
    parser.add_argument("--embodiment-tag", default="OXE_DROID",
                        help="EmbodimentTag name (default: OXE_DROID)")
    parser.add_argument("--num-context-frames", type=int, default=4,
                        help="Number of historical frames to feed as video context (default: 4)")

    # Data input
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--data-dir",
                     help="Directory of PNG frames")
    grp.add_argument("--video",
                     help="Path to exterior-camera-1 MP4 video")
    parser.add_argument("--video-ext2",
                        help="Path to exterior-camera-2 MP4 (used with --video)")
    parser.add_argument("--video-wrist",
                        help="Path to wrist-camera MP4 (used with --video)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Use every Nth frame")

    # Prompt
    parser.add_argument("--prompt", default="",
                        help="Language instruction (overrides metadata.json)")

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30])

    # Segmentation / IoU
    seg_grp = parser.add_mutually_exclusive_group()
    seg_grp.add_argument("--mask-dir",
                         help="Directory with frame_XXXX_mask.npy binary masks")
    seg_grp.add_argument("--use-sam3", action="store_true",
                         help="Auto-segment using SAM3 text-prompted segmentation")
    parser.add_argument("--object-desc", default="",
                        help="Text description of target object for SAM3 (e.g. 'red cup')")
    parser.add_argument("--sam3-checkpoint", default="",
                        help="Path to local SAM3 checkpoint (.pt); auto-downloads from HF if empty")
    parser.add_argument("--sam3-version", default="sam3.1",
                        choices=["sam3", "sam3.1"],
                        help="SAM3 model version (default: sam3.1)")
    parser.add_argument("--sam3-confidence", type=float, default=0.5,
                        help="SAM3 confidence threshold (default: 0.5)")
    parser.add_argument("--threshold-method", default="percentile",
                        choices=["percentile", "otsu", "fixed", "top_k"])
    parser.add_argument("--threshold-value", type=float, default=90.0)

    # Output
    parser.add_argument("--output-dir", default="results/attention_droid_dreamzero")
    parser.add_argument("--save-heatmaps", action="store_true")

    # DIT cache
    parser.add_argument("--no-dit-cache", action="store_true")

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

    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    policy = GrootSimPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
    )

    if rank == 0:
        try:
            blocks = policy.trained_model.action_head.model.blocks
            sa = blocks[0].self_attn
            log.info(
                "Model: %d blocks, frame_seqlen=%d, num_action_per_block=%d, "
                "num_state_per_block=%d, num_heads=%d",
                len(blocks), sa.frame_seqlen, sa.num_action_per_block,
                sa.num_state_per_block, sa.num_heads,
            )
        except Exception as e:
            log.warning("Could not read model info: %s", e)

        run_analysis_rank0(args, policy, signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
