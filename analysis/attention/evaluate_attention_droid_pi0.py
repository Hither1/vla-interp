#!/usr/bin/env python3
"""Offline attention analysis for pi0.5-DROID on real robot ZED camera frames.

Takes video frames from a directory (exported from Google Drive videos) and runs
pi0.5-DROID to compute attention IoU and attention ratio (visual vs linguistic).

No live robot environment needed — pure offline inference on saved frames.

Input layout expected (--data-dir):
  <data_dir>/
    frame_0000.png        # exterior/base camera image (required)
    frame_0000_wrist.png  # wrist camera image (optional, zeros if missing)
    frame_0001.png
    ...
    metadata.json         # optional: {"prompt": "...", "joint_positions": [[...], ...],
                          #             "gripper_positions": [[...], ...]}

OR supply --video for an exterior-camera MP4 and --video-wrist for the wrist-cam MP4.

For IoU you also need segmentation masks:
  Option A: provide --mask-dir with <data_dir>/frame_XXXX_mask.npy binary masks (H×W float32)
  Option B: use --use-sam3 with --object-desc "red cup" to auto-segment via SAM3
            (requires: pip install -e /path/to/sam3)

Usage examples:
  # Attention ratio only (no segmentation)
  python evaluate_attention_droid_pi0.py \\
    --checkpoint /path/to/pi05_droid \\
    --data-dir /path/to/zed_frames \\
    --prompt "pick up the red cup" \\
    --layers 0 8 17 --output-dir results/droid_pi0_ratio

  # With SAM3 for IoU (auto-downloads checkpoint from HuggingFace)
  python evaluate_attention_droid_pi0.py \\
    --checkpoint /path/to/pi05_droid \\
    --data-dir /path/to/zed_frames \\
    --prompt "pick up the red cup" \\
    --use-sam3 --object-desc "red cup" \\
    --output-dir results/droid_pi0_iou

  # With SAM3 using a local checkpoint
  python evaluate_attention_droid_pi0.py \\
    --checkpoint /path/to/pi05_droid \\
    --data-dir /path/to/zed_frames \\
    --prompt "pick up the red cup" \\
    --use-sam3 --object-desc "red cup" --sam3-checkpoint /path/to/sam3.pt \\
    --output-dir results/droid_pi0_iou

  # From video files
  python evaluate_attention_droid_pi0.py \\
    --checkpoint /path/to/pi05_droid \\
    --video /path/to/exterior.mp4 --video-wrist /path/to/wrist.mp4 \\
    --prompt "pick up the red cup" \\
    --frame-step 5 --output-dir results/droid_pi0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_SCRIPT_DIR))

from example_attention_viz import load_model_from_checkpoint, get_paligemma_tokenizer
from openpi.models import model as _model
from openpi.models import gemma
from visualize_attention import (
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
)
from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    overlay_heatmap,
)
from PIL import Image
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_INPUT_RESOLUTION = 224
NUM_PATCHES_PER_IMAGE = 256   # SigLIP 224×224 / 14px → 16×16 = 256
PATCH_GRID = 16               # 16×16 spatial grid


# ==============================================================================
# Data loading
# ==============================================================================

def load_frames_from_dir(
    data_dir: pathlib.Path,
    frame_step: int = 1,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[int]]:
    """Load exterior and wrist frames from a directory of PNGs.

    Returns:
        ext_frames:   list of (H, W, 3) uint8 exterior camera frames
        wrist_frames: list of (H, W, 3) uint8 or None wrist frames
        indices:      global frame indices (for logging/IoU matching)
    """
    # Find all exterior frames (named frame_XXXX.png or XXXX.png)
    candidates = sorted(data_dir.glob("frame_*.png")) + sorted(data_dir.glob("*.png"))
    # Filter out wrist/mask files
    ext_paths = [p for p in sorted(set(candidates)) if "_wrist" not in p.stem and "_mask" not in p.stem]
    ext_paths = ext_paths[::frame_step]

    ext_frames, wrist_frames, indices = [], [], []
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
    """Load frames from MP4 video files. Returns (ext_frames, wrist_frames, indices, fps)."""
    def _read_video(path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frames = []
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
        # Align lengths
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
    frame_step: int = 1,
) -> List[Optional[np.ndarray]]:
    """Load binary segmentation masks (.npy files) corresponding to frame indices."""
    masks = []
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
    log.info("Loading SAM3 from %s …", checkpoint)
    processor = Sam3Processor.from_pretrained(checkpoint)
    model = Sam3Model.from_pretrained(
        checkpoint, torch_dtype=torch.float16
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


def compute_tracking_masks(
    frames_rgb: List[np.ndarray],
    object_desc: str,
    processor,
    model,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
) -> List[Optional[np.ndarray]]:
    """Segment object in each frame using SAM3 text prompts.

    SAM3 requires text conditioning on every call, so each frame is
    independently prompted (no mask-guided propagation).
    Returns one mask (or None) per frame.
    """
    masks: List[Optional[np.ndarray]] = []
    for i, frame_rgb in enumerate(frames_rgb):
        mask = _segment_text_sam3(frame_rgb, object_desc, processor, model,
                                  device, confidence_threshold)
        log.info("  SAM3 frame %d: %s", i, "found" if mask is not None else "not found")
        masks.append(mask)
    n_found = sum(1 for m in masks if m is not None)
    log.info("SAM3 tracking: %d / %d frames with mask", n_found, len(masks))
    return masks


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
    """Build a pi0.5 Observation from DROID-style inputs.

    ext_img, wrist_img: (H, W, 3) uint8.  Resized to MODEL_INPUT_RESOLUTION.
    joint_pos:   (7,) float32 — arm joint angles (rad).
    gripper_pos: (1,) float32 — gripper [0, 1].
    """
    def _resize(img):
        pil = Image.fromarray(img)
        pil = pil.resize((MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION), Image.LANCZOS)
        return np.array(pil, dtype=np.float32) / 255.0

    base_img_f = _resize(ext_img)
    wrist_img_f = _resize(wrist_img) if wrist_img is not None else np.zeros_like(base_img_f)

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

    # For DROID pi0/pi0.5: only base + left_wrist are active (right_wrist masked out).
    # This matches DroidInputs transform: image_masks = (True, True, False).
    # If your checkpoint was trained differently, pass --num-image-tokens to override.
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


# ==============================================================================
# Attention analysis
# ==============================================================================

def compute_attention_ratio(
    attention_weights: np.ndarray,
    num_image_tokens: int,
    num_text_tokens: int,
) -> Dict:
    """Compute visual/linguistic ratio from recorded attention tensor.

    attention_weights shape: (B, K, G, T, S)
      K = num kv-head groups, G = heads per group, T = query len, S = key len.
    We use the first batch, average over all heads, and look at the first action
    query token (index = num_image_tokens + num_text_tokens).
    """
    b = 0
    # Average over K×G heads → (T, S)
    attn = attention_weights[b].reshape(-1, attention_weights.shape[3], attention_weights.shape[4])
    attn_mean = attn.mean(axis=0)  # (T, S)

    query_idx = num_image_tokens + num_text_tokens  # first action token
    if query_idx >= attn_mean.shape[0]:
        # Causal masking may truncate — fall back to last token
        query_idx = -1

    a = attn_mean[query_idx]  # (S,)

    visual_mass = float(a[:num_image_tokens].sum())
    linguistic_mass = float(a[num_image_tokens : num_image_tokens + num_text_tokens].sum())
    action_mass = float(a[num_image_tokens + num_text_tokens :].sum())
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
    num_image_tokens: int,
    image_resolution: int,
    image_slot: int = 0,           # 0 = exterior/base, 1 = wrist
) -> Optional[np.ndarray]:
    """Build a (image_resolution, image_resolution) spatial attention heatmap.

    Extracts the base-camera spatial attention from the first action token,
    reshapes from PATCH_GRID×PATCH_GRID → upsample to image_resolution.
    """
    b = 0
    attn = attention_weights[b].reshape(-1, attention_weights.shape[3], attention_weights.shape[4])
    attn_mean = attn.mean(axis=0)  # (T, S)

    num_text_tokens = int(attention_weights.shape[-1]) - num_image_tokens
    query_idx = num_image_tokens + num_text_tokens
    if query_idx >= attn_mean.shape[0]:
        query_idx = -1

    a = attn_mean[query_idx]  # (S,)

    slot_start = image_slot * NUM_PATCHES_PER_IMAGE
    slot_end = slot_start + NUM_PATCHES_PER_IMAGE
    img_attn = a[slot_start:slot_end]  # (256,)

    if len(img_attn) != NUM_PATCHES_PER_IMAGE:
        return None

    heatmap = img_attn.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_t, size=(image_resolution, image_resolution),
        mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    mn, mx = heatmap_up.min(), heatmap_up.max()
    if mx > mn:
        heatmap_up = (heatmap_up - mn) / (mx - mn)
    else:
        heatmap_up = np.zeros_like(heatmap_up)

    return heatmap_up


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
# Main evaluation loop
# ==============================================================================

def run_analysis(args):
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frames ────────────────────────────────────────────────────────
    if args.video:
        ext_frames, wrist_frames, frame_indices, source_fps = load_frames_from_video(
            args.video, args.video_wrist, frame_step=args.frame_step,
        )
        video_fps = source_fps / args.frame_step
        data_dir = pathlib.Path(args.video).parent
    else:
        data_dir = pathlib.Path(args.data_dir)
        ext_frames, wrist_frames, frame_indices = load_frames_from_dir(data_dir, args.frame_step)
        video_fps = 10.0

    if not ext_frames:
        log.error("No frames found. Check --data-dir / --video arguments.")
        return

    log.info("Loaded %d frames", len(ext_frames))

    # ── Metadata (joint / gripper states) ─────────────────────────────────
    meta = load_metadata(data_dir, len(ext_frames))
    prompt = args.prompt or meta.get("prompt", "")
    if not prompt:
        log.warning("No prompt provided; using empty string.")

    joint_positions = meta.get("joint_positions", [[0.0] * 7] * len(ext_frames))
    gripper_positions = meta.get("gripper_positions", [[0.0]] * len(ext_frames))

    # Pad/truncate to match frame count
    n = len(ext_frames)
    while len(joint_positions) < n:
        joint_positions.append([0.0] * 7)
    while len(gripper_positions) < n:
        gripper_positions.append([0.0])

    # ── Load pi0.5-DROID model ─────────────────────────────────────────────
    log.info("Loading pi0.5 model from %s …", args.checkpoint)
    model, model_cfg, state_dim = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_token_len=args.max_token_len,
        pi05=True,
    )
    tokenizer = get_paligemma_tokenizer(args.max_token_len)

    # For DROID with pi0/pi0.5: 2 active image slots (base + left_wrist).
    # Right wrist is masked out (matches DroidInputs transform).
    # Override with --num-image-tokens if your checkpoint uses 3 slots.
    num_image_tokens = args.num_image_tokens if args.num_image_tokens else NUM_PATCHES_PER_IMAGE * 2  # 512

    log.info("Model loaded. num_image_tokens=%d, using layers=%s", num_image_tokens, args.layers)

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

    # Pre-compute SAM3 tracking masks (sequential — must run before main loop)
    tracking_masks: Optional[List[Optional[np.ndarray]]] = None
    if sam3_processor is not None and args.object_desc:
        log.info("Pre-computing SAM3 tracking masks for '%s' over %d frames …",
                 args.object_desc, len(ext_frames))
        frames_for_tracking = [
            np.array(Image.fromarray(f).resize(
                (MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION), Image.LANCZOS
            )) for f in ext_frames
        ]
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
    video_frame_buf: Dict[int, List[np.ndarray]] = {l: [] for l in args.layers}

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

        num_text_tokens = int(observation.tokenized_prompt.shape[1])

        # Run forward pass with attention recording
        enable_attention_recording()
        rng = jax.random.PRNGKey(i)
        _ = model.sample_actions(rng, observation, num_steps=10)
        attention_dict = get_recorded_attention_weights()
        disable_attention_recording()

        if not attention_dict:
            log.warning("No attention weights recorded at frame %d", frame_idx)
            continue

        # Determine segmentation mask for IoU
        seg_mask = None
        if provided_masks is not None:
            seg_mask = provided_masks[i]
        elif tracking_masks is not None:
            seg_mask = tracking_masks[i]

        # Process per layer
        frame_results = {"frame_idx": frame_idx, "layers": {}}
        for layer_idx in args.layers:
            layer_key = f"layer_{layer_idx}"
            if layer_key not in attention_dict:
                continue

            attn = np.array(attention_dict[layer_key][0])  # (B, K, G, T, S)

            # Attention ratio
            ratio_result = compute_attention_ratio(
                attention_weights=attn,
                num_image_tokens=num_image_tokens,
                num_text_tokens=num_text_tokens,
            )

            # Spatial heatmap (from exterior/base camera, slot 0)
            heatmap = build_spatial_heatmap(
                attention_weights=attn,
                num_image_tokens=num_image_tokens,
                image_resolution=MODEL_INPUT_RESOLUTION,
                image_slot=0,
            )

            layer_result = dict(ratio_result)
            layer_result["layer"] = layer_idx
            layer_result["frame_idx"] = frame_idx
            layer_result["num_text_tokens"] = num_text_tokens

            # IoU vs segmentation mask
            if heatmap is not None and seg_mask is not None:
                seg_resized = cv2.resize(
                    seg_mask.astype(np.float32),
                    (MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION),
                    interpolation=cv2.INTER_NEAREST,
                )
                _thresh_val = args.threshold_value
                _thresh_key = f"{args.threshold_method}_{int(_thresh_val) if _thresh_val == int(_thresh_val) else _thresh_val}"
                iou_result = compute_attention_object_iou(
                    attention_heatmap=heatmap,
                    segmentation_mask=seg_resized.astype(np.int32),
                    object_ids={"object": 1},
                    threshold_methods=[(args.threshold_method, _thresh_val)],
                )
                layer_result.update({
                    "iou": iou_result["combined"].get(_thresh_key, {}).get("iou", None),
                    "dice": iou_result["combined"].get(_thresh_key, {}).get("dice", None),
                    "attention_mass_on_object": iou_result["attention_mass"].get("_all_objects", None),
                    "pointing_hit": iou_result.get("pointing_hit", None),
                })
                log.info(
                    "  layer=%d ratio=%.3f iou=%.3f mass=%.3f",
                    layer_idx,
                    ratio_result["visual_linguistic_ratio"],
                    layer_result.get("iou", 0.0) or 0.0,
                    layer_result.get("attention_mass_on_object", 0.0) or 0.0,
                )
            else:
                log.info(
                    "  layer=%d ratio=%.3f visual=%.3f linguistic=%.3f",
                    layer_idx,
                    ratio_result["visual_linguistic_ratio"],
                    ratio_result["visual_fraction"],
                    ratio_result["linguistic_fraction"],
                )

            frame_results["layers"][layer_key] = layer_result
            all_step_results.append(layer_result)

            # Build overlay frame (used for both --save-heatmaps and --save-video)
            if (args.save_heatmaps or args.save_video) and heatmap is not None:
                viz_img = np.array(Image.fromarray(ext_img).resize(
                    (MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION), Image.LANCZOS
                ))
                overlay = overlay_heatmap(viz_img, heatmap)
                panels = [viz_img, overlay]
                if seg_mask is not None:
                    seg_disp = (seg_resized[:, :, None] * np.array([0, 0, 255], dtype=np.uint8))
                    panels.append(np.clip(viz_img.astype(np.int32) + seg_disp.astype(np.int32) // 2, 0, 255).astype(np.uint8))
                combined = np.concatenate(panels, axis=1)
                if args.save_heatmaps:
                    save_path = out_dir / f"frame{frame_idx:04d}_layer{layer_idx}_heatmap.png"
                    Image.fromarray(combined).save(str(save_path))
                if args.save_video:
                    video_frame_buf[layer_idx].append(combined)

    # ── Save results ───────────────────────────────────────────────────────
    per_layer_summary = {}
    for layer_idx in args.layers:
        lr = [r for r in all_step_results if r.get("layer") == layer_idx]
        if lr:
            per_layer_summary[f"layer_{layer_idx}"] = summarize_ratios(lr)
            iou_vals = [r["iou"] for r in lr if r.get("iou") is not None]
            if iou_vals:
                per_layer_summary[f"layer_{layer_idx}"]["iou_mean"] = float(np.mean(iou_vals))
                per_layer_summary[f"layer_{layer_idx}"]["iou_std"] = float(np.std(iou_vals))

    output = {
        "prompt": prompt,
        "checkpoint": args.checkpoint,
        "num_frames": len(ext_frames),
        "num_image_tokens": num_image_tokens,
        "layers": args.layers,
        "per_layer_summary": per_layer_summary,
        "per_frame_results": all_step_results,
    }

    results_path = out_dir / "attention_results_pi0_droid.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    # ── Attention video ────────────────────────────────────────────────────
    if args.save_video:
        for layer_idx, frames in video_frame_buf.items():
            if frames:
                _write_attention_video(frames, out_dir / f"attention_layer_{layer_idx}.mp4", video_fps)

    # ── Summary plot ───────────────────────────────────────────────────────
    if all_step_results and len(args.layers) > 0:
        _save_summary_plot(all_step_results, args.layers, prompt, out_dir)


def _write_attention_video(frames: List[np.ndarray], path: pathlib.Path, fps: float):
    """Write a list of RGB numpy frames to an MP4 video."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    log.info("Attention video saved: %s (%d frames @ %.1f fps)", path.name, len(frames), fps)


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

        iou_vals = [r.get("iou") for r in lr if r.get("iou") is not None]
        if iou_vals:
            ax2 = ax.twinx()
            iou_xs = [r["frame_idx"] for r in lr if r.get("iou") is not None]
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
    parser.add_argument("--checkpoint", required=True,
                        help="Path or GCS URI to the pi0.5-DROID checkpoint directory")
    parser.add_argument("--max-token-len", type=int, default=256)

    # Data input
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--data-dir",
                     help="Directory of PNG frames (frame_XXXX.png / frame_XXXX_wrist.png)")
    grp.add_argument("--video",
                     help="Path to exterior camera MP4 video")
    parser.add_argument("--video-wrist",
                        help="Path to wrist camera MP4 video (used with --video)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Use every Nth frame (default: every frame)")

    # Prompt / state
    parser.add_argument("--prompt", type=str, default="",
                        help="Language instruction (overrides metadata.json)")

    # Attention layers
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 17],
                        help="Gemma layer indices to analyse (default: 0 8 17)")
    parser.add_argument("--num-image-tokens", type=int, default=0,
                        help="Override number of image tokens (default: 512 = 2×256 for DROID pi0/pi0.5; "
                             "use 768 for 3-image-slot checkpoints)")

    # Segmentation (IoU) — mutually exclusive sources
    seg_grp = parser.add_mutually_exclusive_group()
    seg_grp.add_argument("--mask-dir",
                         help="Directory containing pre-computed binary masks (frame_XXXX_mask.npy)")
    seg_grp.add_argument("--use-sam3", action="store_true",
                         help="Use SAM3 text-prompted segmentation to auto-generate masks")
    parser.add_argument("--object-desc", type=str, default="",
                        help="Text description of target object for SAM3 (e.g. 'red cup')")
    parser.add_argument("--sam3-checkpoint", type=str, default="",
                        help="Path to HuggingFace SAM3 checkpoint directory; "
                             "defaults to /n/netscratch/sham_lab/Lab/chloe00/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7")
    parser.add_argument("--sam3-confidence", type=float, default=0.5,
                        help="SAM3 confidence threshold for mask acceptance (default: 0.5)")

    # IoU thresholding
    parser.add_argument("--threshold-method", type=str, default="percentile",
                        help="Attention thresholding method: 'percentile' or 'otsu' (default: percentile)")
    parser.add_argument("--threshold-value", type=float, default=90.0,
                        help="Threshold value (percentile 0–100 for 'percentile' mode; unused for 'otsu'). Default: 90.0")

    # Output
    parser.add_argument("--output-dir", default="results/attention_droid_pi0")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save per-frame attention heatmap PNGs")
    parser.add_argument("--save-video", action="store_true",
                        help="Save per-layer attention overlay video (attention_layer_N.mp4)")

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
