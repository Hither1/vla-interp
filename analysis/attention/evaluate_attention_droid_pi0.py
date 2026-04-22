#!/usr/bin/env python3
"""Offline attention analysis for pi0.5-DROID on real robot ZED camera frames.

This version fixes a few important issues that can make the attention maps look
pathological:

1. Preserves the batch dimension when averaging attention across layers.
2. Uses the prompt mask (not padded sequence length) for counting valid text tokens.
3. Makes attention shape handling explicit and robust.
4. Adds debug logging for attention mass accounting.
5. Supports visualizing either:
   - average over all action query tokens, or
   - a specific action query token via --query-index
6. Avoids misleading percentile stretching unless explicitly requested.

Input layout expected (--data-dir):
  <data_dir>/
    frame_0000.png
    frame_0000_wrist.png
    frame_0001.png
    ...
    metadata.json

Optional metadata.json:
{
  "prompt": "...",
  "joint_positions": [[...], ...],
  "gripper_positions": [[...], ...]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# JAX must come before openpi imports
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn.functional as F

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_SCRIPT_DIR))

from example_attention_viz import load_model_from_checkpoint, get_paligemma_tokenizer
from openpi.models import model as _model
from openpi.shared import image_tools
from visualize_attention import (
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
)
from attention_iou import (
    compute_attention_object_iou,
    overlay_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_INPUT_RESOLUTION = 224
NUM_PATCHES_PER_IMAGE = 256
PATCH_GRID = 16

NUM_IMAGE_SLOTS_DROID = 3
NUM_IMAGE_TOKENS_TOTAL = NUM_IMAGE_SLOTS_DROID * NUM_PATCHES_PER_IMAGE  # 768
NUM_MODEL_LAYERS = 18


# ==============================================================================
# Data loading
# ==============================================================================

def load_frames_from_dir(
    data_dir: pathlib.Path,
    frame_step: int = 1,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[int]]:
    """Load exterior and wrist frames from a directory of PNGs."""
    candidates = sorted(data_dir.glob("frame_*.png")) + sorted(data_dir.glob("*.png"))
    ext_paths = [p for p in sorted(set(candidates)) if "_wrist" not in p.stem and "_mask" not in p.stem]
    ext_paths = ext_paths[::frame_step]

    ext_frames: List[np.ndarray] = []
    wrist_frames: List[Optional[np.ndarray]] = []
    indices: List[int] = []

    for p in ext_paths:
        idx = int("".join(filter(str.isdigit, p.stem)) or 0)
        img = np.array(Image.open(p).convert("RGB"))
        ext_frames.append(img)

        wrist_path = p.parent / f"{p.stem}_wrist{p.suffix}"
        if wrist_path.exists():
            wrist_frames.append(np.array(Image.open(wrist_path).convert("RGB")))
        else:
            wrist_frames.append(None)
        indices.append(idx)

    return ext_frames, wrist_frames, indices


def load_frames_from_video(
    video_path: str,
    wrist_video_path: Optional[str],
    frame_step: int = 1,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[int], float]:
    """Load frames from MP4 video files."""

    def _read_video(path: str) -> Tuple[List[np.ndarray], float]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frames: List[np.ndarray] = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        return frames, fps

    ext_frames, source_fps = _read_video(video_path)
    indices = list(range(0, len(ext_frames) * frame_step, frame_step))

    if wrist_video_path:
        wrist_frames, _ = _read_video(wrist_video_path)
        n = min(len(ext_frames), len(wrist_frames))
        ext_frames = ext_frames[:n]
        wrist_frames = wrist_frames[:n]
        indices = indices[:n]
    else:
        wrist_frames = [None] * len(ext_frames)

    return ext_frames, wrist_frames, indices, source_fps


def load_masks_from_dir(
    data_dir: pathlib.Path,
    indices: List[int],
) -> List[Optional[np.ndarray]]:
    """Load binary segmentation masks (.npy files) corresponding to frame indices."""
    masks: List[Optional[np.ndarray]] = []
    for idx in indices:
        mask_path = data_dir / f"frame_{idx:04d}_mask.npy"
        if not mask_path.exists():
            mask_path = data_dir / f"{idx:04d}_mask.npy"
        if mask_path.exists():
            m = np.load(str(mask_path)).astype(np.float32)
            masks.append((m > 0).astype(np.float32))
        else:
            masks.append(None)
    return masks


def load_metadata(data_dir: pathlib.Path, n_frames: int) -> Dict:
    """Load metadata.json if it exists, else return defaults."""
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {
        "prompt": "",
        "joint_positions": [np.zeros(7, dtype=np.float32).tolist()] * n_frames,
        "gripper_positions": [np.zeros(1, dtype=np.float32).tolist()] * n_frames,
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

    log.info("Loading SAM3 from %s ...", checkpoint)
    processor = Sam3Processor.from_pretrained(checkpoint)
    model = Sam3Model.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
    ).eval().to(device)
    log.info("SAM3 loaded on %s", device)
    return processor, model


def _run_sam3_inference(inputs, processor, model, device, h, w, confidence_threshold):
    """Shared SAM3 forward pass. Returns union mask (H, W) float32, or None."""
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[[h, w]],
        threshold=confidence_threshold,
    )
    masks = results[0].get("masks")
    if masks is None or len(masks) == 0:
        return None
    return torch.stack(list(masks)).any(dim=0).float().cpu().numpy()


def _segment_text_sam3(image_rgb, object_desc, processor, model, device, confidence_threshold):
    h, w = image_rgb.shape[:2]
    inputs = processor(images=Image.fromarray(image_rgb), text=object_desc, return_tensors="pt")
    return _run_sam3_inference(inputs, processor, model, device, h, w, confidence_threshold)


def compute_tracking_masks(
    frames_rgb: List[np.ndarray],
    object_descs: List[str],
    processor,
    model,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """Segment one or more objects in each frame using SAM3 text prompts."""
    all_masks: List[Optional[Dict[str, np.ndarray]]] = []
    for i, frame_rgb in enumerate(frames_rgb):
        frame_obj_masks: Dict[str, np.ndarray] = {}
        for desc in object_descs:
            mask = _segment_text_sam3(
                frame_rgb,
                desc,
                processor,
                model,
                device,
                confidence_threshold,
            )
            log.info("  SAM3 frame %d '%s': %s", i, desc, "found" if mask is not None else "not found")
            if mask is not None:
                frame_obj_masks[desc] = mask
        all_masks.append(frame_obj_masks if frame_obj_masks else None)

    for desc in object_descs:
        n_found = sum(1 for m in all_masks if m is not None and desc in m)
        log.info("SAM3 tracking '%s': %d / %d frames with mask", desc, n_found, len(all_masks))
    return all_masks


# ==============================================================================
# Observation construction
# ==============================================================================

def create_droid_observation(
    ext_img: np.ndarray,
    wrist_img: Optional[np.ndarray],
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    prompt: str,
    tokenizer,
) -> _model.Observation:
    """Build a pi0.5 Observation from DROID-style inputs."""

    def _resize_with_pad_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return np.array(
            image_tools.resize_with_pad(
                img,
                MODEL_INPUT_RESOLUTION,
                MODEL_INPUT_RESOLUTION,
            )
        )

    def _to_model_range(img_uint8: np.ndarray) -> np.ndarray:
        # Match Observation.from_dict uint8 conversion path: [0,255] -> [-1,1].
        return img_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0

    base_img_u8 = _resize_with_pad_uint8(ext_img)
    if wrist_img is not None:
        wrist_img_u8 = _resize_with_pad_uint8(wrist_img)
    else:
        wrist_img_u8 = np.zeros_like(base_img_u8)

    base_img_f = _to_model_range(base_img_u8)
    wrist_img_f = _to_model_range(wrist_img_u8)

    base_b = base_img_f[None]
    wrist_b = wrist_img_f[None]
    dummy_b = np.zeros_like(base_b)

    state = np.concatenate([
        np.asarray(joint_pos, dtype=np.float32).flatten()[:7],
        np.asarray(gripper_pos, dtype=np.float32).flatten()[:1],
    ])

    tokens, mask = tokenizer.tokenize(prompt, state=None)
    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([mask], dtype=jnp.bool_)

    return _model.Observation(
        images={
            "base_0_rgb": jnp.array(base_b),
            "left_wrist_0_rgb": jnp.array(wrist_b),
            "right_wrist_0_rgb": jnp.array(dummy_b),
        },
        image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.zeros((1,), dtype=jnp.bool_),
        },
        state=jnp.array(state[None], dtype=jnp.float32),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def _preprocess_viz_image(img: np.ndarray) -> np.ndarray:
    """Resize with pad to the model canvas while keeping uint8 RGB for viz/SAM3."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.array(
        image_tools.resize_with_pad(
            img,
            MODEL_INPUT_RESOLUTION,
            MODEL_INPUT_RESOLUTION,
        )
    )


def _resize_mask_with_pad(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize segmentation mask without distortion and preserve label IDs."""
    h, w = mask.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((target_h, target_w), dtype=resized.dtype)

    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


# ==============================================================================
# Attention helpers
# ==============================================================================

def _to_numpy(x) -> np.ndarray:
    arr = np.array(x)
    # Guard against bfloat16 object arrays: astype(float64) on object arrays can
    # produce string elements on some numpy/ml_dtypes combos. arr.tolist() forces
    # each element through Python __float__, then rebuild as float64.
    if arr.dtype == object:
        arr = np.array(arr.tolist(), dtype=np.float64)
    elif not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr.astype(np.float32)


def _validate_attention_shape(x: np.ndarray, name: str) -> None:
    if x.ndim != 5:
        raise ValueError(
            f"{name} must have shape (B, K, G, T, S), got shape {tuple(x.shape)}"
        )


def _collapse_attention_heads(attention_weights: np.ndarray) -> np.ndarray:
    """Collapse (B, K, G, T, S) -> (num_heads, T, S) using batch 0.

    Shape: B=batch, K=num_kv_heads, G=num_q_heads_per_kv, T=query_len, S=key_len.
    num_heads = K * G.
    """
    x = _to_numpy(attention_weights)
    _validate_attention_shape(x, "attention_weights")
    B, K, G, T, S = x.shape
    log.debug("attention shape: B=%d K=%d G=%d T=%d S=%d (num_heads=%d)", B, K, G, T, S, K * G)
    # x[0] = (K, G, T, S) → reshape to (K*G, T, S)
    return x[0].reshape(K * G, T, S)


def _get_valid_text_token_count(observation: _model.Observation) -> int:
    """Count only valid prompt tokens, not padding."""
    mask = np.array(observation.tokenized_prompt_mask[0]).astype(bool)
    return int(mask.sum())


def _get_prefix_ranges(
    num_active_image_tokens: int,
    num_valid_text_tokens: int,
    num_total_image_tokens: int,
) -> Dict[str, Tuple[int, int]]:
    """Return key-sequence index ranges for active visual and valid language tokens."""
    if num_total_image_tokens < num_active_image_tokens:
        raise ValueError(
            f"num_total_image_tokens ({num_total_image_tokens}) < "
            f"num_active_image_tokens ({num_active_image_tokens})"
        )

    visual_start = 0
    visual_end = num_active_image_tokens

    language_start = num_total_image_tokens
    language_end = num_total_image_tokens + num_valid_text_tokens

    return {
        "visual": (visual_start, visual_end),
        "language": (language_start, language_end),
    }


def compute_attention_ratio(
    attention_weights: np.ndarray,
    num_image_tokens: int,
    num_text_tokens: int,
    num_image_tokens_total: int = 0,
) -> Dict:
    """Compute visual/linguistic ratio from recorded attention tensor.

    Expected shape: (B, K, G, T, S)
    """
    if num_image_tokens_total == 0:
        num_image_tokens_total = num_image_tokens

    heads_t_s = _collapse_attention_heads(attention_weights)  # (H, T, S)
    mean_t_s = heads_t_s.mean(axis=0)                         # (T, S)
    mean_s = mean_t_s.mean(axis=0)                            # (S,)

    ranges = _get_prefix_ranges(
        num_active_image_tokens=num_image_tokens,
        num_valid_text_tokens=num_text_tokens,
        num_total_image_tokens=num_image_tokens_total,
    )

    visual_start, visual_end = ranges["visual"]
    lang_start, lang_end = ranges["language"]

    visual_mass = float(mean_s[visual_start:visual_end].sum())
    linguistic_mass = float(mean_s[lang_start:lang_end].sum())
    action_mass = float(mean_s[lang_end:].sum())
    total = max(visual_mass + linguistic_mass + action_mass, 1e-8)

    ratio = visual_mass / linguistic_mass if linguistic_mass > 1e-8 else float("inf")

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "action_mass": action_mass,
        "total_mass": total,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass / total,
        "linguistic_fraction": linguistic_mass / total,
        "action_fraction": action_mass / total,
    }


def build_spatial_heatmap(
    attention_weights: np.ndarray,
    image_resolution: int,
    image_slot: int = 0,
    query_index: Optional[int] = None,
    normalize_mode: str = "max",
    debug_prefix: str = "",
) -> Optional[np.ndarray]:
    """Build a spatial attention heatmap for one image slot.

    attention_weights shape must be (B, K, G, T, S).

    query_index:
      None -> average over all action query tokens
      int  -> use only that action query token
    """
    heads_t_s = _collapse_attention_heads(attention_weights)  # (H, T, S)
    H, T, S = heads_t_s.shape

    if query_index is None:
        key_mass = heads_t_s.mean(axis=(0, 1))  # (S,)
    else:
        if query_index < 0 or query_index >= T:
            raise ValueError(
                f"query_index={query_index} out of range for T={T}"
            )
        key_mass = heads_t_s[:, query_index, :].mean(axis=0)  # (S,)

    # Per-query-token attention at position 0 (top-left corner = potential attention sink)
    slot_start_diag = image_slot * NUM_PATCHES_PER_IMAGE
    attn_at_pos0_per_query = heads_t_s.mean(axis=0)[:, slot_start_diag]  # (T,)
    log.info(
        "%s slot=%d attn@pos0 per-query-token: %s",
        debug_prefix,
        image_slot,
        " ".join(f"q{t}={attn_at_pos0_per_query[t]:.5f}" for t in range(T)),
    )
    # Per-head attention at position 0
    attn_at_pos0_per_head = heads_t_s[:, :, slot_start_diag].mean(axis=1)  # (H,)
    log.info(
        "%s slot=%d attn@pos0 per-head: %s",
        debug_prefix,
        image_slot,
        " ".join(f"h{h}={attn_at_pos0_per_head[h]:.5f}" for h in range(H)),
    )

    slot_start = image_slot * NUM_PATCHES_PER_IMAGE
    slot_end = slot_start + NUM_PATCHES_PER_IMAGE
    img_attn = key_mass[slot_start:slot_end]

    if img_attn.shape[0] != NUM_PATCHES_PER_IMAGE:
        log.warning(
            "%s image slot %d extracted %d tokens, expected %d",
            debug_prefix,
            image_slot,
            img_attn.shape[0],
            NUM_PATCHES_PER_IMAGE,
        )
        return None

    log.info(
        "%s slot=%d raw image mass: sum=%.6f min=%.6f max=%.6f",
        debug_prefix,
        image_slot,
        float(img_attn.sum()),
        float(img_attn.min()),
        float(img_attn.max()),
    )

    # Diagnostic: show where the attention is concentrated
    grid = img_attn.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    # Flatten and find top patches
    flat = grid.flatten()
    top5_idx = np.argsort(flat)[::-1][:5]
    top5_str = ", ".join(
        f"({idx // PATCH_GRID},{idx % PATCH_GRID})={flat[idx]:.5f}"
        for idx in top5_idx
    )
    log.info("%s slot=%d top-5 patches (row,col)=attn: %s", debug_prefix, image_slot, top5_str)

    # Edges vs centre breakdown
    edge_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mean = float(grid[edge_mask].mean())
    center_mean = float(grid[~edge_mask].mean())
    log.info(
        "%s slot=%d edge_mean=%.6f center_mean=%.6f ratio=%.2f",
        debug_prefix,
        image_slot,
        edge_mean,
        center_mean,
        edge_mean / (center_mean + 1e-8),
    )

    heatmap = grid
    heatmap_t = torch.from_numpy(heatmap)[None, None]
    heatmap_up = F.interpolate(
        heatmap_t,
        size=(image_resolution, image_resolution),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    if normalize_mode == "none":
        return heatmap_up

    if normalize_mode == "max":
        denom = float(heatmap_up.max())
        if denom > 1e-8:
            return heatmap_up / denom
        return np.zeros_like(heatmap_up)

    if normalize_mode == "percentile":
        p_lo = float(np.percentile(heatmap_up, 1))
        p_hi = float(np.percentile(heatmap_up, 99))
        if p_hi > p_lo:
            return np.clip((heatmap_up - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        return np.zeros_like(heatmap_up)

    raise ValueError(f"Unknown normalize_mode: {normalize_mode}")


def extract_denoising_step_layers(attention_dict_raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Keep only denoising-step layers and remap them back to layer_0..layer_17."""
    out: Dict[str, np.ndarray] = {}
    all_layer_nums = sorted(int(k.split("_")[1]) for k in attention_dict_raw)
    log.info(
        "Recorded %d attention layers total (range %s..%s)",
        len(all_layer_nums),
        all_layer_nums[0] if all_layer_nums else "?",
        all_layer_nums[-1] if all_layer_nums else "?",
    )
    for key, val in attention_dict_raw.items():
        layer_num = int(key.split("_")[1])
        arr = _to_numpy(val)
        if layer_num == all_layer_nums[0] or layer_num == all_layer_nums[-1]:
            # Log shape of first and last layer to confirm prefix vs suffix pass
            log.info("  layer_%d shape=%s (T=%s)", layer_num, tuple(arr.shape), arr.shape[-2] if arr.ndim >= 2 else "?")
        if NUM_MODEL_LAYERS <= layer_num < 2 * NUM_MODEL_LAYERS:
            out[f"layer_{layer_num - NUM_MODEL_LAYERS}"] = arr
    log.info("Extracted %d denoising-step layers (suffix pass)", len(out))
    return out


def average_selected_layers(
    attention_dict: Dict[str, np.ndarray],
    layers: List[int],
) -> Optional[np.ndarray]:
    """Average attention across selected layers while preserving (B, K, G, T, S)."""
    selected = []
    for l in layers:
        key = f"layer_{l}"
        if key not in attention_dict:
            log.warning("Requested %s not found in recorded attention", key)
            continue
        x = _to_numpy(attention_dict[key])
        _validate_attention_shape(x, key)
        selected.append(x)

    if not selected:
        return None

    avg = np.mean(selected, axis=0)
    _validate_attention_shape(avg, "averaged_attention")
    return avg


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def summarize_ratios(step_results: List[Dict]) -> Dict:
    if not step_results:
        return {}
    ratios = [
        r["visual_linguistic_ratio"]
        for r in step_results
        if np.isfinite(r.get("visual_linguistic_ratio", float("nan")))
    ]
    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
        },
        "visual_fraction": {
            "mean": float(np.mean([r["visual_fraction"] for r in step_results]))
        },
        "linguistic_fraction": {
            "mean": float(np.mean([r["linguistic_fraction"] for r in step_results]))
        },
        "num_steps": len(step_results),
    }


def compute_edge_focus_ratio(heatmap: np.ndarray) -> float:
    """Mean attention on border patches divided by center patches."""
    h, w = heatmap.shape[:2]
    patch_h = h // PATCH_GRID
    patch_w = w // PATCH_GRID
    if patch_h <= 0 or patch_w <= 0:
        return float("nan")

    heatmap_crop = heatmap[: PATCH_GRID * patch_h, : PATCH_GRID * patch_w]
    grid = heatmap_crop.reshape(PATCH_GRID, patch_h, PATCH_GRID, patch_w).mean(axis=(1, 3))

    edge_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    edge_mean = float(grid[edge_mask].mean())
    center_mean = float(grid[~edge_mask].mean())
    return edge_mean / max(center_mean, 1e-8)


# ==============================================================================
# Main evaluation loop
# ==============================================================================

def run_analysis(args):
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frames ────────────────────────────────────────────────────────
    if args.video:
        ext_frames, wrist_frames, frame_indices, source_fps = load_frames_from_video(
            args.video,
            args.video_wrist,
            frame_step=args.frame_step,
        )
        video_fps = source_fps / args.frame_step
        data_dir = pathlib.Path(args.video).parent
    else:
        data_dir = pathlib.Path(args.data_dir)
        ext_frames, wrist_frames, frame_indices = load_frames_from_dir(
            data_dir,
            args.frame_step,
        )
        video_fps = 10.0

    if not ext_frames:
        log.error("No frames found. Check --data-dir / --video arguments.")
        return

    log.info("Loaded %d frames", len(ext_frames))

    # ── Metadata ───────────────────────────────────────────────────────────
    meta = load_metadata(data_dir, len(ext_frames))
    prompt = args.prompt or meta.get("prompt", "")
    if not prompt:
        log.warning("No prompt provided; using empty string.")

    joint_positions = meta.get("joint_positions", [[0.0] * 7] * len(ext_frames))
    gripper_positions = meta.get("gripper_positions", [[0.0]] * len(ext_frames))

    n = len(ext_frames)
    while len(joint_positions) < n:
        joint_positions.append([0.0] * 7)
    while len(gripper_positions) < n:
        gripper_positions.append([0.0])

    # ── Load model ─────────────────────────────────────────────────────────
    log.info("Loading pi0.5 model from %s ...", args.checkpoint)
    model, model_cfg, state_dim = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_token_len=args.max_token_len,
        pi05=True,
    )
    tokenizer = get_paligemma_tokenizer(args.max_token_len)

    num_image_tokens = (
        args.num_image_tokens
        if args.num_image_tokens
        else NUM_PATCHES_PER_IMAGE * 2
    )

    log.info(
        "Model loaded. num_image_tokens=%d, num_total_image_tokens=%d, layers=%s",
        num_image_tokens,
        NUM_IMAGE_TOKENS_TOTAL,
        args.layers,
    )

    # ── Load segmentation model ────────────────────────────────────────────
    sam3_processor = None
    sam3_model = None
    if args.use_sam3:
        ckpt = args.sam3_checkpoint or _SAM3_HF_CHECKPOINT
        sam3_processor, sam3_model = load_sam3_hf(ckpt, device="cuda")

    provided_masks = None
    if args.mask_dir:
        mask_dir = pathlib.Path(args.mask_dir)
        provided_masks = load_masks_from_dir(mask_dir, frame_indices)
        log.info("Loaded %d pre-computed masks", sum(1 for m in provided_masks if m is not None))

    tracking_masks: Optional[List[Optional[Dict[str, np.ndarray]]]] = None
    if sam3_processor is not None and args.object_desc:
        log.info(
            "Pre-computing SAM3 tracking masks for %s over %d frames ...",
            args.object_desc,
            len(ext_frames),
        )
        frames_for_tracking = [_preprocess_viz_image(f) for f in ext_frames]
        tracking_masks = compute_tracking_masks(
            frames_for_tracking,
            args.object_desc,
            sam3_processor,
            sam3_model,
            device="cuda",
            confidence_threshold=args.sam3_confidence,
        )

    # ── Analysis loop ──────────────────────────────────────────────────────
    all_step_results: List[Dict] = []
    avg_video_buf: List[np.ndarray] = []

    for i, (ext_img, wrist_img, frame_idx) in enumerate(zip(ext_frames, wrist_frames, frame_indices)):
        log.info("Frame %d/%d (idx=%d)", i + 1, len(ext_frames), frame_idx)

        joint_pos = np.array(joint_positions[i], dtype=np.float32)
        gripper_pos = np.array(gripper_positions[i], dtype=np.float32)

        observation = create_droid_observation(
            ext_img=ext_img,
            wrist_img=wrist_img,
            joint_pos=joint_pos,
            gripper_pos=gripper_pos,
            prompt=prompt,
            tokenizer=tokenizer,
        )

        num_text_tokens = _get_valid_text_token_count(observation)
        prompt_seq_len = int(observation.tokenized_prompt.shape[1])

        if args.debug:
            log.info(
                "Prompt token stats: valid=%d padded_seq_len=%d",
                num_text_tokens,
                prompt_seq_len,
            )

        # Run forward pass with attention recording
        enable_attention_recording()
        rng = jax.random.PRNGKey(i)
        _ = model.sample_actions(rng, observation, num_steps=1)
        attention_dict_raw = get_recorded_attention_weights()
        disable_attention_recording()

        attention_dict = extract_denoising_step_layers(attention_dict_raw)

        if not attention_dict:
            log.warning("No denoising-step attention recorded at frame %d", frame_idx)
            continue

        attn = average_selected_layers(attention_dict, args.layers)
        if attn is None:
            log.warning("No selected layers available at frame %d", frame_idx)
            continue

        if args.debug:
            log.info("Averaged attention shape: %s", tuple(attn.shape))

        seg_mask = None
        seg_object_ids: Dict[str, int] = {"object": 1}

        if provided_masks is not None:
            seg_mask = provided_masks[i]
        elif tracking_masks is not None and tracking_masks[i] is not None:
            frame_obj_masks = tracking_masks[i]
            first_mask = next(iter(frame_obj_masks.values()))
            labeled = np.zeros(first_mask.shape, dtype=np.int32)
            seg_object_ids = {}
            for obj_id, (desc, bin_mask) in enumerate(frame_obj_masks.items(), start=1):
                labeled[bin_mask > 0] = obj_id
                seg_object_ids[desc] = obj_id
            seg_mask = labeled.astype(np.float32)

        ext_viz_img = _preprocess_viz_image(ext_img)
        wrist_viz_img = _preprocess_viz_image(wrist_img) if wrist_img is not None else np.zeros_like(ext_viz_img)

        frame_results = {"frame_idx": frame_idx, "layers": {}}
        seg_resized = None
        viz_img = None

        layer_label_str = f"layers {', '.join(str(l) for l in args.layers)}"

        ratio_result = compute_attention_ratio(
            attention_weights=attn,
            num_image_tokens=num_image_tokens,
            num_text_tokens=num_text_tokens,
            num_image_tokens_total=NUM_IMAGE_TOKENS_TOTAL,
        )

        heatmap = build_spatial_heatmap(
            attention_weights=attn,
            image_resolution=MODEL_INPUT_RESOLUTION,
            image_slot=0,
            query_index=args.query_index,
            normalize_mode=args.heatmap_normalize,
            debug_prefix=f"[frame {frame_idx} ext]",
        )

        layer_result = dict(ratio_result)
        layer_result["layer"] = "avg"
        layer_result["frame_idx"] = frame_idx
        layer_result["num_text_tokens"] = num_text_tokens
        layer_result["prompt_seq_len"] = prompt_seq_len
        layer_result["query_index"] = args.query_index
        layer_result["attention_shape"] = list(attn.shape)
        if heatmap is not None:
            layer_result["edge_focus_ratio"] = compute_edge_focus_ratio(heatmap)

        if heatmap is not None and seg_mask is not None:
            if seg_resized is None:
                seg_resized = _resize_mask_with_pad(
                    seg_mask.astype(np.int32),
                    MODEL_INPUT_RESOLUTION,
                    MODEL_INPUT_RESOLUTION,
                )

            thresh_val = args.threshold_value
            thresh_key = (
                f"{args.threshold_method}_{int(thresh_val)}"
                if thresh_val == int(thresh_val)
                else f"{args.threshold_method}_{thresh_val}"
            )

            iou_result = compute_attention_object_iou(
                attention_heatmap=heatmap,
                segmentation_mask=seg_resized.astype(np.int32),
                object_ids=seg_object_ids,
                threshold_methods=[(args.threshold_method, thresh_val)],
            )

            layer_result.update({
                "iou": iou_result["combined"].get(thresh_key, {}).get("iou", None),
                "dice": iou_result["combined"].get(thresh_key, {}).get("dice", None),
                "attention_mass_on_object": iou_result["attention_mass"].get("_all_objects", None),
                "pointing_hit": iou_result.get("pointing_hit", None),
                "iou_per_object": {
                    obj: iou_result["per_object"].get(obj, {}).get(thresh_key, {}).get("iou", None)
                    for obj in seg_object_ids
                },
                "attention_mass_per_object": {
                    obj: iou_result["attention_mass"].get(obj, None)
                    for obj in seg_object_ids
                },
            })

            per_obj_iou_str = " | ".join(
                f"{obj}={layer_result['iou_per_object'].get(obj, 0.0) or 0.0:.3f}"
                for obj in seg_object_ids
            )
            log.info(
                "  avg(%s) ratio=%.3f iou(combined)=%.3f [%s] mass=%.3f",
                layer_label_str,
                ratio_result["visual_linguistic_ratio"],
                layer_result.get("iou", 0.0) or 0.0,
                per_obj_iou_str,
                layer_result.get("attention_mass_on_object", 0.0) or 0.0,
            )
        else:
            log.info(
                "  avg(%s) ratio=%.3f visual=%.3f linguistic=%.3f action=%.3f",
                layer_label_str,
                ratio_result["visual_linguistic_ratio"],
                ratio_result["visual_fraction"],
                ratio_result["linguistic_fraction"],
                ratio_result["action_fraction"],
            )

        frame_results["layers"]["avg"] = layer_result
        all_step_results.append(layer_result)

        if args.save_heatmaps and heatmap is not None:
            if viz_img is None:
                viz_img = ext_viz_img
            overlay = overlay_heatmap(viz_img, heatmap)
            panels = [
                _label_panel(viz_img, "Exterior"),
                _label_panel(
                    overlay,
                    f"Ext Attn {layer_label_str}" + (
                        " (all q)" if args.query_index is None else f" (q={args.query_index})"
                    ),
                ),
            ]
            if seg_resized is not None:
                seg_disp = (seg_resized[:, :, None] * np.array([0, 0, 255], dtype=np.uint8))
                seg_panel = np.clip(
                    viz_img.astype(np.int32) + seg_disp.astype(np.int32) // 2,
                    0,
                    255,
                ).astype(np.uint8)
                panels.append(_label_panel(seg_panel, "Segmentation"))
            combined = np.concatenate(panels, axis=1)
            save_path = out_dir / f"frame{frame_idx:04d}_avg_heatmap.png"
            Image.fromarray(combined).save(str(save_path))

        if args.save_video:
            if viz_img is None:
                viz_img = ext_viz_img

            heatmap_ext = build_spatial_heatmap(
                attention_weights=attn,
                image_resolution=MODEL_INPUT_RESOLUTION,
                image_slot=0,
                query_index=args.query_index,
                normalize_mode=args.heatmap_normalize,
                debug_prefix=f"[frame {frame_idx} ext video]",
            )

            wrist_viz = wrist_viz_img

            heatmap_wrist = build_spatial_heatmap(
                attention_weights=attn,
                image_resolution=MODEL_INPUT_RESOLUTION,
                image_slot=1,
                query_index=args.query_index,
                normalize_mode=args.heatmap_normalize,
                debug_prefix=f"[frame {frame_idx} wrist video]",
            )

            layer_label = f"Attn L{','.join(str(l) for l in args.layers)}"
            if args.query_index is not None:
                layer_label += f" q={args.query_index}"

            panels = [_label_panel(viz_img, "Exterior")]
            if heatmap_ext is not None:
                panels.append(_label_panel(overlay_heatmap(viz_img, heatmap_ext), f"Ext {layer_label}"))
            panels.append(_label_panel(wrist_viz, "Wrist"))
            if heatmap_wrist is not None:
                panels.append(_label_panel(overlay_heatmap(wrist_viz, heatmap_wrist), f"Wrist {layer_label}"))
            if seg_resized is not None:
                seg_disp = (seg_resized[:, :, None] * np.array([0, 0, 255], dtype=np.uint8))
                seg_panel = np.clip(
                    viz_img.astype(np.int32) + seg_disp.astype(np.int32) // 2,
                    0,
                    255,
                ).astype(np.uint8)
                panels.append(_label_panel(seg_panel, "Segmentation"))
            avg_video_buf.append(np.concatenate(panels, axis=1))

    # ── Save results ───────────────────────────────────────────────────────
    avg_summary = summarize_ratios(all_step_results)
    iou_vals = [r["iou"] for r in all_step_results if r.get("iou") is not None]
    if iou_vals:
        avg_summary["iou_mean"] = float(np.mean(iou_vals))
        avg_summary["iou_std"] = float(np.std(iou_vals))

    output = {
        "prompt": prompt,
        "checkpoint": args.checkpoint,
        "num_frames": len(ext_frames),
        "num_image_tokens": num_image_tokens,
        "num_total_image_tokens": NUM_IMAGE_TOKENS_TOTAL,
        "layers": args.layers,
        "query_index": args.query_index,
        "heatmap_normalize": args.heatmap_normalize,
        "avg_layer_summary": avg_summary,
        "per_frame_results": all_step_results,
    }

    results_path = out_dir / "attention_results_pi0_droid.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    if args.save_video and avg_video_buf:
        _write_attention_video(avg_video_buf, out_dir / "attention_avg.mp4", video_fps)

    if all_step_results:
        _save_summary_plot(all_step_results, args.layers, prompt, out_dir)


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Add a black header bar with white label text above an image panel."""
    h, w = img.shape[:2]
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([bar, img], axis=0)


def _write_attention_video(frames: List[np.ndarray], path: pathlib.Path, fps: float):
    """Write a list of RGB frames to an MP4 video."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    log.info("Attention video saved: %s (%d frames @ %.1f fps)", path.name, len(frames), fps)


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

    iou_vals = [r.get("iou") for r in step_results if r.get("iou") is not None]
    if iou_vals:
        ax2 = ax.twinx()
        iou_xs = [r["frame_idx"] for r in step_results if r.get("iou") is not None]
        ax2.plot(iou_xs, iou_vals, "s--", color="green", alpha=0.7, linewidth=1.5)
        ax2.set_ylabel("IoU (green)")
        ax2.set_ylim(0, 1)

    fig.suptitle(f"pi0.5-DROID Attention: {prompt[:60]}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_dir / "summary_plot.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary plot saved.")


# ==============================================================================
# Argument parsing
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Offline attention analysis for pi0.5-DROID on real ZED camera frames"
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path or GCS URI to the pi0.5-DROID checkpoint directory",
    )
    parser.add_argument("--max-token-len", type=int, default=256)

    # Data input
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--data-dir",
        help="Directory of PNG frames (frame_XXXX.png / frame_XXXX_wrist.png)",
    )
    grp.add_argument(
        "--video",
        help="Path to exterior camera MP4 video",
    )
    parser.add_argument(
        "--video-wrist",
        help="Path to wrist camera MP4 video (used with --video)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame",
    )

    # Prompt / state
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Language instruction (overrides metadata.json)",
    )

    # Attention layers
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 8, 17],
        help="Gemma layer indices to analyse",
    )
    parser.add_argument(
        "--num-image-tokens",
        type=int,
        default=0,
        help="Override number of active image tokens. Default: 512 = 2 x 256 for DROID pi0/pi0.5.",
    )

    # New debugging / visualization controls
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="If set, visualize attention from a single action query token instead of averaging all action queries.",
    )
    parser.add_argument(
        "--heatmap-normalize",
        type=str,
        default="max",
        choices=["none", "max", "percentile"],
        help="Heatmap normalization mode. Default: max. Use percentile only if you really want contrast stretching.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra debugging logs for attention shapes and token counts.",
    )

    # Segmentation
    seg_grp = parser.add_mutually_exclusive_group()
    seg_grp.add_argument(
        "--mask-dir",
        help="Directory containing pre-computed binary masks (frame_XXXX_mask.npy)",
    )
    seg_grp.add_argument(
        "--use-sam3",
        action="store_true",
        help="Use SAM3 text-prompted segmentation to auto-generate masks",
    )
    parser.add_argument(
        "--object-desc",
        type=str,
        nargs="+",
        default=[],
        help="Text description(s) of target object(s) for SAM3.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default="",
        help="Path to HuggingFace SAM3 checkpoint directory",
    )
    parser.add_argument(
        "--sam3-confidence",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold for mask acceptance",
    )

    # IoU thresholding
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="percentile",
        help="Attention thresholding method: 'percentile' or 'otsu'",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=90.0,
        help="Threshold value for percentile mode",
    )

    # Output
    parser.add_argument("--output-dir", default="results/attention_droid_pi0")
    parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        help="Save per-frame attention heatmap PNGs",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save attention overlay video",
    )

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()