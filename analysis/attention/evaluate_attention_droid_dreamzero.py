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
            Pass multiple descriptions to compute IoU against both objects simultaneously:
              --object-desc "green cup" "blue bowl"
            Combined IoU is computed against the union mask; per-object metrics are also saved.
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

# DROID dataset native resolution. GrootSimPolicy.lazy_joint_forward_causal runs the
# full transform pipeline (VideoResize/VideoCrop) internally, and VideoToTensor checks
# that the input resolution matches the original dataset resolution from metadata.
# Pass images at the native 320×180 so the check passes; the policy resizes internally.
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320


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

_SAM3_HF_CHECKPOINT = (
    "/n/netscratch/sham_lab/Lab/chloe00/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7"
)


def load_sam3_hf(checkpoint: str, device: str = "cuda"):
    from transformers import Sam3Model, Sam3Processor
    log.info("Loading SAM3 from %s …", checkpoint)
    processor = Sam3Processor.from_pretrained(checkpoint)
    model = Sam3Model.from_pretrained(
        checkpoint, torch_dtype=torch.float16
    ).eval().to(device)
    log.info("SAM3 loaded on %s", device)
    return processor, model


def _run_sam3_inference(inputs, processor, model, device, h, w, confidence_threshold):
    """Shared SAM3 forward pass. Returns union mask (H, W) float32, or None."""
    inputs = {
        k: v.to(device=device, dtype=torch.float16) if isinstance(v, torch.Tensor) and v.is_floating_point() else (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[[h, w]], threshold=confidence_threshold,
    )
    masks = results[0].get("masks")
    if masks is None or len(masks) == 0:
        return None
    return torch.stack(list(masks)).any(dim=0).float().cpu().numpy()


def _segment_text_sam3(image_rgb, object_desc, processor, model, device, confidence_threshold):
    h, w = image_rgb.shape[:2]
    inputs = processor(images=Image.fromarray(image_rgb), text=object_desc, return_tensors="pt")
    return _run_sam3_inference(inputs, processor, model, device, h, w, confidence_threshold)


def _segment_mask_guided_sam3(image_rgb, prev_mask, processor, model, device, confidence_threshold):
    """Mask-guided propagation via bounding-box prompt derived from previous mask.
    Sam3Processor only accepts input_boxes/input_boxes_labels (not input_points/input_labels)."""
    h, w = image_rgb.shape[:2]
    ys, xs = np.where(prev_mask > 0.5)
    if len(xs) == 0:
        return None  # caller will fall back to text re-init
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    inputs = processor(images=Image.fromarray(image_rgb),
                       input_boxes=[[[x_min, y_min, x_max, y_max]]],
                       input_boxes_labels=[[1]],
                       return_tensors="pt")
    return _run_sam3_inference(inputs, processor, model, device, h, w, confidence_threshold)


def compute_tracking_masks(
    frames_rgb: List[np.ndarray],
    object_descs: List[str],
    processor,
    model,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """Track one or more objects through frames using SAM3.

    Each object is tracked independently: frame 0 (and any frame where tracking
    is lost) uses a text-prompted init; subsequent frames use mask-guided
    propagation from the previous frame's mask for that object.

    Returns a list (one entry per frame) of dicts mapping object description to
    its binary mask.  Entries are None when no object was found in the frame.
    """
    # Per-object state: last known mask for propagation
    prev_masks: Dict[str, Optional[np.ndarray]] = {desc: None for desc in object_descs}

    all_masks: List[Optional[Dict[str, np.ndarray]]] = []
    for i, frame_rgb in enumerate(frames_rgb):
        frame_obj_masks: Dict[str, np.ndarray] = {}
        for desc in object_descs:
            prev = prev_masks[desc]
            if prev is None:
                mask = _segment_text_sam3(frame_rgb, desc, processor, model,
                                          device, confidence_threshold)
                log.info("  SAM3 frame %d '%s': text init → %s", i, desc,
                         "found" if mask is not None else "not found")
            else:
                mask = _segment_mask_guided_sam3(frame_rgb, prev, processor, model,
                                                 device, confidence_threshold)
                if mask is None:
                    log.info("  SAM3 frame %d '%s': tracking lost, re-init with text", i, desc)
                    mask = _segment_text_sam3(frame_rgb, desc, processor, model,
                                              device, confidence_threshold)
            if mask is not None:
                frame_obj_masks[desc] = mask
                prev_masks[desc] = mask
        all_masks.append(frame_obj_masks if frame_obj_masks else None)

    for desc in object_descs:
        n_found = sum(1 for m in all_masks if m is not None and desc in m)
        log.info("SAM3 tracking '%s': %d / %d frames with mask",
                 desc, n_found, len(all_masks))
    return all_masks


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
                rec = _RECORDED.setdefault(idx, {"self_attn": [], "cross_attn": [], "action_register_length": None})
                if len(args) >= 5 and args[4] is not None:
                    rec["action_register_length"] = args[4]
            return pre_hook

        handles.append(self_attn.register_forward_pre_hook(_make_pre_hook(layer_idx)))

        def _make_self_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block
            frame_seqlen = parent.frame_seqlen

            def hook(module, args, output):
                rec = _RECORDED.setdefault(idx, {"self_attn": [], "cross_attn": [], "action_register_length": None})
                if len(args) < 3:
                    return
                q, k, v = args[0], args[1], args[2]
                # q: (B, Lq, H, d)
                # Training blockwise path: q.shape[1] == n_act  (action tokens only)
                # Inference path: q.shape[1] == num_frame_per_block*frame_seqlen + n_act + n_state
                #   (e.g. 2*880 + 24 + 1 = 1785 for DROID)
                if q.shape[1] == n_act:
                    q_act = q
                    visual_q_len = max(0, k.shape[1] - n_act - n_state)
                elif q.shape[1] > n_act + n_state:
                    # Extract action queries: they are the n_act tokens just before final n_state tokens
                    visual_q_len = q.shape[1] - n_act - n_state
                    q_act = q[:, visual_q_len:visual_q_len + n_act]
                else:
                    return  # unexpected shape

                Lk = k.shape[1]
                visual_end = Lk - n_act - n_state
                # n_hist: historical visual tokens in K before the current visual block
                n_hist = max(0, visual_end - visual_q_len)
                # current_frame_start: offset in K of the last frame_seqlen tokens in current visual
                current_frame_start = n_hist + max(0, visual_q_len - frame_seqlen)

                with torch.no_grad():
                    q_f = q_act.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = q_act.shape[-1] ** -0.5
                    attn = torch.softmax(
                        torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1
                    )  # (B, H, n_act, Lk)
                    attn_np = attn[0].mean(0).cpu().float().numpy()  # (n_act, Lk)

                rec["self_attn"].append({
                    "attn": attn_np,
                    "Lk": Lk,
                    "visual_end": visual_end,
                    "n_act": n_act,
                    "n_state": n_state,
                    "frame_seqlen": frame_seqlen,
                    "current_frame_start": current_frame_start,
                })
            return hook

        handles.append(self_attn.attn.register_forward_hook(_make_self_attn_hook(layer_idx, self_attn)))

        def _make_cross_attn_hook(idx, parent: CausalWanSelfAttention):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block

            def hook(module, args, output):
                rec = _RECORDED.setdefault(idx, {"self_attn": [], "cross_attn": [], "action_register_length": None})
                if len(args) < 2:
                    return
                normed_x = args[0]   # (B, S, C)
                context = args[1]    # (B, T, C) text embeddings

                # action_register_length from pre_hook is unreliable (call uses kwargs, so
                # args is empty in the pre_hook). For inference there is always exactly one
                # chunk, so derive it directly from n_act + n_state.
                B, S, C = normed_x.shape
                n = module.num_heads
                d = module.head_dim

                action_horizon = n_act
                state_horizon = n_state
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

                rec["cross_attn"].append({
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
    # Find the most square-ish factorization (largest h ≤ sqrt)
    best: Optional[Tuple[int, int]] = None
    for h in range(1, int(frame_seqlen ** 0.5) + 1):
        if frame_seqlen % h == 0:
            best = (h, frame_seqlen // h)
    if best is None or best[1] / best[0] > 10:
        raise ValueError(f"Cannot infer grid from frame_seqlen={frame_seqlen}")
    return best


def build_heatmap_from_self_attn(
    calls: List[Dict],
    image_h: int,
    image_w: int,
) -> Optional[np.ndarray]:
    """Build spatial heatmap from the first action-block self-attention call (→ current image frame)."""
    if not calls:
        return None
    c = calls[0]
    frame_seqlen = c["frame_seqlen"]
    attn = c["attn"]  # (n_act, Lk)
    # Use current_frame_start to point at the last visual frame in K
    start = c.get("current_frame_start", 0)
    slice_ = attn[:, start:start + frame_seqlen]
    if slice_.shape[1] == 0:
        return None
    spatial_attn = slice_.mean(axis=0)  # (frame_seqlen,)

    try:
        h_feat, w_feat = _infer_grid(frame_seqlen)
    except ValueError:
        return None

    heatmap = spatial_attn.reshape(h_feat, w_feat).astype(np.float32)
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_t, size=(image_h, image_w), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    mn, mx = heatmap_up.min(), heatmap_up.max()
    if mx > mn:
        heatmap_up = (heatmap_up - mn) / (mx - mn)
    else:
        heatmap_up = np.zeros_like(heatmap_up)
    return heatmap_up


def compute_iou_from_heatmap(
    heatmap: np.ndarray,
    seg_mask: np.ndarray,
    threshold_method: str = "percentile",
    threshold_value: float = 90.0,
    obj_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Compute IoU metrics for a (union) seg mask plus optional per-object breakdown.

    Args:
        heatmap:          (H, W) float in [0, 1].
        seg_mask:         (H, W) float/int union mask (> 0 = any object).
        threshold_method: thresholding method.
        threshold_value:  threshold parameter.
        obj_masks:        optional dict of {object_desc: binary (H, W) mask} for
                          per-object IoU.  Masks are assumed to already be resized
                          to match heatmap resolution.
    Returns:
        dict with keys: iou, dice, attention_mass (combined),
        and iou_per_object / attention_mass_per_object when obj_masks is given.
    """
    from attention_iou import threshold_attention, compute_iou, compute_dice
    binary = threshold_attention(heatmap, threshold_method, threshold_value)
    total_attn = max(float(heatmap.sum()), 1e-8)
    seg_bool = (seg_mask > 0)
    result = {
        "iou": compute_iou(binary, seg_bool),
        "dice": compute_dice(binary, seg_bool),
        "attention_mass": float(heatmap[seg_bool].sum() / total_attn),
    }
    if obj_masks:
        result["iou_per_object"] = {
            desc: compute_iou(binary, (m > 0.5)) for desc, m in obj_masks.items()
        }
        result["attention_mass_per_object"] = {
            desc: float(heatmap[m > 0.5].sum() / total_attn) for desc, m in obj_masks.items()
        }
    return result


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Add a black header bar with white label text above an image panel."""
    h, w = img.shape[:2]
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _write_attention_video(frames: List[np.ndarray], path: pathlib.Path, fps: float):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    log.info("Attention video saved: %s (%d frames @ %.1f fps)", path.name, len(frames), fps)


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

    if args.max_frames > 0:
        cap = min(args.max_frames, len(ext1_frames))
        ext1_frames = ext1_frames[:cap]
        ext2_frames = ext2_frames[:cap]
        wrist_frames = wrist_frames[:cap]
        frame_indices = frame_indices[:cap]
        log.info("Applying --max-frames=%d -> using %d frames", args.max_frames, cap)

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
    sam3_model = None
    if args.use_sam3:
        ckpt = args.sam3_checkpoint or _SAM3_HF_CHECKPOINT
        sam3_processor, sam3_model = load_sam3_hf(ckpt, device="cuda")

    provided_masks = None
    if args.mask_dir:
        provided_masks = load_masks_from_dir(pathlib.Path(args.mask_dir), frame_indices)

    # Pre-compute SAM3 tracking masks (sequential — must run before main loop)
    tracking_masks: Optional[List[Optional[Dict[str, np.ndarray]]]] = None
    if sam3_processor is not None and args.object_desc:
        log.info("Pre-computing SAM3 tracking masks for %s over %d frames …",
                 args.object_desc, len(ext1_frames))
        tracking_masks = compute_tracking_masks(
            ext1_frames,
            args.object_desc,  # now a list
            sam3_processor,
            sam3_model,
            device="cuda",
            confidence_threshold=args.sam3_confidence,
        )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        (out_dir / "heatmaps").mkdir(exist_ok=True)

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    all_step_results: List[Dict] = []
    avg_video_buf: List[np.ndarray] = []
    video_fps = 10.0

    hooks = install_attention_hooks(policy, args.layers)

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

        frame_t0 = datetime.datetime.now()
        clear_recorded_attention()

        signal.fill_(0)
        dist.broadcast(signal, src=0, group=signal_group)
        _broadcast_obs(dz_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
        dist.barrier()
        recorded = get_recorded_attention()

        # Get segmentation mask (union of all objects) and per-object masks.
        # For pre-computed masks: single binary mask, no per-object breakdown.
        # For SAM3 multi-object: union binary mask + per-object dict.
        seg_mask = None
        seg_obj_masks: Optional[Dict[str, np.ndarray]] = None
        if provided_masks is not None:
            seg_mask = provided_masks[i]
        elif tracking_masks is not None and tracking_masks[i] is not None:
            frame_obj_masks = tracking_masks[i]  # Dict[str, np.ndarray]
            # Union mask for combined IoU / visualization
            union = np.zeros_like(next(iter(frame_obj_masks.values())), dtype=np.float32)
            for m in frame_obj_masks.values():
                union = np.maximum(union, m.astype(np.float32))
            seg_mask = union
            seg_obj_masks = frame_obj_masks

        # Average attention across requested layers, then compute once
        valid_layers = [l for l in args.layers if l in recorded]
        if not valid_layers:
            continue
        layer_label_str = f"layers {', '.join(str(l) for l in valid_layers)}"

        # Merge self_attn/cross_attn calls from all layers (equivalent to averaging ratios)
        merged_rec: Dict = {"self_attn": [], "cross_attn": []}
        for l in valid_layers:
            merged_rec["self_attn"].extend(recorded[l].get("self_attn", []))
            merged_rec["cross_attn"].extend(recorded[l].get("cross_attn", []))

        ratio_result = compute_ratio_from_recorded(merged_rec)
        if ratio_result is None:
            continue

        frame_result = dict(ratio_result)
        frame_result["layer"] = "avg"
        frame_result["frame_idx"] = frame_idx

        # Heatmap: average spatial attention across layers
        layer_heatmaps = [
            build_heatmap_from_self_attn(recorded[l].get("self_attn", []), IMAGE_HEIGHT, IMAGE_WIDTH)
            for l in valid_layers
        ]
        layer_heatmaps = [h for h in layer_heatmaps if h is not None]
        heatmap = np.mean(layer_heatmaps, axis=0) if layer_heatmaps else None

        if heatmap is not None and seg_mask is not None:
            seg_resized = cv2.resize(
                seg_mask.astype(np.float32),
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
            # Resize per-object masks to match heatmap resolution
            obj_masks_resized: Optional[Dict[str, np.ndarray]] = None
            if seg_obj_masks is not None:
                obj_masks_resized = {
                    desc: cv2.resize(
                        m.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for desc, m in seg_obj_masks.items()
                }
            iou_metrics = compute_iou_from_heatmap(
                heatmap, seg_resized,
                threshold_method=args.threshold_method,
                threshold_value=args.threshold_value,
                obj_masks=obj_masks_resized,
            )
            frame_result.update(iou_metrics)
            per_obj_iou_str = ""
            if "iou_per_object" in iou_metrics:
                per_obj_iou_str = " [" + " | ".join(
                    f"{obj}={v:.3f}" for obj, v in iou_metrics["iou_per_object"].items()
                ) + "]"
            log.info(
                "  avg(%s) ratio=%.3f iou(combined)=%.3f%s mass=%.3f",
                layer_label_str,
                ratio_result["visual_linguistic_ratio"],
                iou_metrics["iou"],
                per_obj_iou_str,
                iou_metrics["attention_mass"],
            )
        else:
            log.info(
                "  avg(%s) ratio=%.3f visual=%.3f linguistic=%.3f",
                layer_label_str,
                ratio_result["visual_linguistic_ratio"],
                ratio_result["visual_fraction"],
                ratio_result["linguistic_fraction"],
            )

        all_step_results.append(frame_result)

        frame_dt = (datetime.datetime.now() - frame_t0).total_seconds()
        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = sum(r.get("_runtime_sec", 0.0) for r in all_step_results) + frame_dt
            done = i + 1
            eta = (elapsed / max(done, 1)) * max(n - done, 0)
            log.info("Progress: %d/%d frames | last=%.2fs | ETA=%.1f min", done, n, frame_dt, eta / 60.0)
        frame_result["_runtime_sec"] = frame_dt

        if args.save_heatmaps and heatmap is not None:
            from attention_iou import overlay_heatmap
            mn, mx = heatmap.min(), heatmap.max()
            heatmap_norm = (heatmap - mn) / (mx - mn) if mx > mn else heatmap
            viz = overlay_heatmap(ext1, heatmap_norm)
            save_p = out_dir / "heatmaps" / f"frame{frame_idx:04d}_avg_heatmap.png"
            Image.fromarray(viz).save(str(save_p))

        # Build video frame
        if args.save_video and heatmap is not None:
            from attention_iou import overlay_heatmap
            mn, mx = heatmap.min(), heatmap.max()
            heatmap_norm = (heatmap - mn) / (mx - mn) if mx > mn else heatmap
            overlay = overlay_heatmap(ext1, heatmap_norm)
            panels = [
                _label_panel(ext1, "Original"),
                _label_panel(overlay, f"Avg attention ({layer_label_str})"),
            ]
            if seg_mask is not None:
                seg_resized = cv2.resize(
                    seg_mask.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                seg_disp = (seg_resized[:, :, None] * np.array([0, 0, 255], dtype=np.uint8))
                seg_panel = np.clip(
                    ext1.astype(np.int32) + seg_disp.astype(np.int32) // 2, 0, 255
                ).astype(np.uint8)
                panels.append(_label_panel(seg_panel, "Segmentation mask"))
            avg_video_buf.append(np.concatenate(panels, axis=1))

    # Shutdown workers
    remove_attention_hooks(hooks)

    signal.fill_(1)
    dist.broadcast(signal, src=0, group=signal_group)

    # ── Save results ──────────────────────────────────────────────────────
    avg_summary = summarize_ratios(all_step_results)
    iou_vals = [r["iou"] for r in all_step_results if "iou" in r]
    if iou_vals:
        avg_summary["iou_mean"] = float(np.mean(iou_vals))
        avg_summary["iou_std"] = float(np.std(iou_vals))

    output = {
        "prompt": prompt,
        "model_path": args.model_path,
        "num_frames": len(ext1_frames),
        "num_context_frames": args.num_context_frames,
        "layers": args.layers,
        "avg_layer_summary": avg_summary,
        "per_frame_results": all_step_results,
    }

    results_path = out_dir / "attention_results_dreamzero_droid.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    if args.save_video and avg_video_buf:
        _write_attention_video(avg_video_buf, out_dir / "attention_avg.mp4", video_fps)

    if all_step_results:
        _save_summary_plot(all_step_results, args.layers, prompt, out_dir)


def _save_summary_plot(step_results, layers, prompt, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    xs = [r["frame_idx"] for r in step_results]
    ratios = [r["visual_linguistic_ratio"] for r in step_results]
    ax.plot(xs, ratios, "o-", linewidth=2)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    layer_label = f"layers {', '.join(str(l) for l in layers)} (avg)"
    ax.set_title(f"V/L Ratio ({layer_label})", fontweight="bold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Visual / Linguistic Ratio")
    ax.grid(alpha=0.3)

    iou_vals = [r.get("iou") for r in step_results if "iou" in r]
    if iou_vals:
        ax2 = ax.twinx()
        iou_xs = [r["frame_idx"] for r in step_results if "iou" in r]
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
    parser.add_argument("--max-frames", type=int, default=0,
                        help="If >0, analyze only the first N sampled frames")

    # Prompt
    parser.add_argument("--prompt", default="",
                        help="Language instruction (overrides metadata.json)")

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[37, 38, 39])

    # Segmentation / IoU
    seg_grp = parser.add_mutually_exclusive_group()
    seg_grp.add_argument("--mask-dir",
                         help="Directory with frame_XXXX_mask.npy binary masks")
    seg_grp.add_argument("--use-sam3", action="store_true",
                         help="Auto-segment using SAM3 text-prompted segmentation")
    parser.add_argument("--object-desc", nargs="+", default=[],
                        help="Text description(s) of target object(s) for SAM3. "
                             "Pass one or more quoted strings, e.g. --object-desc 'green cup' 'blue bowl'. "
                             "IoU is computed against the union; per-object metrics are also reported.")
    parser.add_argument("--sam3-checkpoint", default="",
                        help="Path to local SAM3 HuggingFace checkpoint directory")
    parser.add_argument("--sam3-confidence", type=float, default=0.5,
                        help="SAM3 confidence threshold (default: 0.5)")
    parser.add_argument("--threshold-method", default="percentile",
                        choices=["percentile", "otsu", "fixed", "top_k"])
    parser.add_argument("--threshold-value", type=float, default=90.0)

    # Output
    parser.add_argument("--output-dir", default="results/attention_droid_dreamzero")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save per-frame per-layer attention heatmap PNGs")
    parser.add_argument("--save-video", action="store_true",
                        help="Save averaged attention heatmap as attention_avg.mp4")

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
