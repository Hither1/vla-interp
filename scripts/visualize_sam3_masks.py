#!/usr/bin/env python3
"""
Visualize SAM3 mask tracking on a video.

Uses text-prompted segmentation on the first frame, then propagates the mask
forward using SAM3's spatial-prompt (mask-guided) mode for subsequent frames.
If tracking is lost, falls back to text prompting to re-initialize.

Usage:
  python scripts/visualize_sam3_masks.py \
      --video /path/to/input.mp4 \
      --object-desc "the book" \
      --output /path/to/output.mp4

  # With local SAM3 checkpoint:
  python scripts/visualize_sam3_masks.py \
      --video /path/to/input.mp4 \
      --object-desc "the book" \
      --sam3-checkpoint /path/to/sam3_hf_dir \
      --output /path/to/output.mp4
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_SAM3_CKPT = (
    "/n/netscratch/sham_lab/Lab/chloe00/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7"
)

# Mask overlay color (R, G, B) and alpha
MASK_COLOR = (0, 255, 0)   # green
MASK_ALPHA = 0.45


def load_sam3(checkpoint: str, device: str):
    from transformers import Sam3Model, Sam3Processor

    log.info("Loading SAM3 from %s …", checkpoint)
    processor = Sam3Processor.from_pretrained(checkpoint)
    model = Sam3Model.from_pretrained(
        checkpoint, torch_dtype=torch.float16
    ).eval().to(device)
    log.info("SAM3 loaded on %s", device)
    return processor, model


def _run_model(inputs: dict, processor, model, device: str, h: int, w: int,
               confidence_threshold: float) -> np.ndarray | None:
    """Shared inference + post-processing. Returns union mask or None."""
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
    result = results[0]
    masks = result.get("masks")

    if masks is None or len(masks) == 0:
        return None

    return torch.stack(list(masks)).any(dim=0).float().cpu().numpy()


def segment_text(image_rgb: np.ndarray, object_desc: str, processor, model,
                 device: str, confidence_threshold: float) -> np.ndarray | None:
    """Text-prompted segmentation on a single frame."""
    h, w = image_rgb.shape[:2]
    inputs = processor(images=Image.fromarray(image_rgb), text=object_desc,
                       return_tensors="pt")
    return _run_model(inputs, processor, model, device, h, w, confidence_threshold)


def segment_mask_guided(image_rgb: np.ndarray, prev_mask: np.ndarray,
                        processor, model, device: str,
                        confidence_threshold: float) -> np.ndarray | None:
    """Mask-guided propagation: derive centroid of previous mask as a foreground
    point prompt.  SAM3FastImageProcessor does not accept 'input_masks', so we
    use the supported input_points / input_labels interface instead."""
    h, w = image_rgb.shape[:2]
    ys, xs = np.where(prev_mask > 0.5)
    if len(xs) == 0:
        return None  # previous mask is empty — caller will fall back to text init
    cx, cy = int(xs.mean()), int(ys.mean())
    # processor expects lists: [[[x, y]]] → (batch=1, point_batch=1, n_points=1, 2)
    inputs = processor(images=Image.fromarray(image_rgb),
                       input_points=[[[cx, cy]]],
                       input_labels=[[1]],   # 1 = foreground point
                       return_tensors="pt")
    return _run_model(inputs, processor, model, device, h, w, confidence_threshold)


def track_video(
    frames_rgb: list[np.ndarray],
    object_desc: str,
    processor,
    model,
    device: str,
    confidence_threshold: float = 0.3,
    frame_step: int = 1,
) -> list[np.ndarray | None]:
    """
    Track an object through a video using SAM3.
    - Frame 0 (and any frame where tracking is lost): text-prompted init
    - Subsequent frames: mask-guided propagation from previous frame
    Returns one mask (or None) per frame in the original frame list.
    """
    masks: list[np.ndarray | None] = [None] * len(frames_rgb)
    prev_mask: np.ndarray | None = None

    for i, frame_rgb in enumerate(frames_rgb):
        if i % frame_step != 0:
            masks[i] = prev_mask  # carry forward for skipped frames
            continue

        if prev_mask is None:
            # Initialize with text prompt
            log.info("  Frame %d: text init ('%s')", i, object_desc)
            mask = segment_text(frame_rgb, object_desc, processor, model,
                                device, confidence_threshold)
        else:
            # Propagate from previous mask
            mask = segment_mask_guided(frame_rgb, prev_mask, processor, model,
                                       device, confidence_threshold)
            if mask is None:
                # Tracking lost — re-initialize with text
                log.info("  Frame %d: tracking lost, re-init with text", i)
                mask = segment_text(frame_rgb, object_desc, processor, model,
                                    device, confidence_threshold)

        masks[i] = mask
        if mask is not None:
            prev_mask = mask

        if (i + 1) % 10 == 0:
            log.info("  Processed %d / %d frames", i + 1, len(frames_rgb))

    return masks


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Blend a colored mask over a BGR frame. Returns a new BGR frame."""
    vis = frame_bgr.copy()
    if mask is not None and mask.max() > 0:
        color_overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_overlay[:] = MASK_COLOR[::-1]  # RGB → BGR
        binary = (mask > 0.5).astype(np.uint8)[:, :, None]  # (H, W, 1)
        vis = (
            vis.astype(np.float32) * (1 - MASK_ALPHA * binary)
            + color_overlay.astype(np.float32) * (MASK_ALPHA * binary)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, MASK_COLOR[::-1], 2)
    return vis


def frames_from_video(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return frames, fps


def write_video(frames: list[np.ndarray], output_path: str, fps: float):
    if not frames:
        log.warning("No frames to write")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    log.info("Wrote %d frames → %s", len(frames), output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAM3 mask tracking on a video"
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument(
        "--object-desc", required=True,
        help='Text description of object to track (e.g. "the book")'
    )
    parser.add_argument(
        "--output", default="",
        help="Output video path (default: <input>_sam3_masks.mp4)"
    )
    parser.add_argument(
        "--sam3-checkpoint", default=_DEFAULT_SAM3_CKPT,
        help="Path to local SAM3 HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Minimum confidence threshold for mask acceptance (default: 0.3)"
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda)"
    )
    args = parser.parse_args()

    output_path = args.output or (
        str(Path(args.video).with_suffix("")) + "_sam3_masks.mp4"
    )

    log.info("Reading video: %s", args.video)
    all_frames, fps = frames_from_video(args.video)
    log.info("Loaded %d frames at %.1f fps", len(all_frames), fps)

    processor, model = load_sam3(args.sam3_checkpoint, args.device)

    log.info("Tracking '%s' through %d frames …", args.object_desc, len(all_frames))
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames]
    masks = track_video(
        frames_rgb,
        args.object_desc,
        processor,
        model,
        args.device,
        confidence_threshold=args.confidence,
        frame_step=args.frame_step,
    )

    n_detected = sum(1 for m in masks if m is not None)
    log.info("Detection rate: %d / %d frames", n_detected, len(all_frames))

    vis_frames = []
    for i, (frame_bgr, mask) in enumerate(zip(all_frames, masks)):
        vis = overlay_mask(frame_bgr, mask)
        label = f"#{i}  '{args.object_desc}'  {'TRACKED' if mask is not None else 'lost'}"
        cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        vis_frames.append(vis)

    write_video(vis_frames, output_path, fps)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
