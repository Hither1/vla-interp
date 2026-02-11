"""Compute IoU between attention heatmaps and ground-truth object segmentation masks.

Provides:
  - Attention thresholding (percentile, Otsu, fixed, top-k)
  - IoU, Dice, precision/recall between binary masks
  - Attention mass analysis (soft overlap, no thresholding)
  - Per-object and combined IoU computation
  - Episode-level aggregation
  - Multi-panel visualization of attention vs segmentation
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple


# ── Binary Mask Metrics ──────────────────────────────────────────────────────


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice coefficient (F1 score of overlap)."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return float(2 * intersection) / float(total)


def compute_precision_recall(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> Tuple[float, float]:
    """Precision and recall of predicted mask vs ground truth."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    precision = float(tp) / float(pred.sum()) if pred.sum() > 0 else 0.0
    recall = float(tp) / float(gt.sum()) if gt.sum() > 0 else 0.0
    return precision, recall


# ── Attention Thresholding ───────────────────────────────────────────────────


def threshold_attention(
    heatmap: np.ndarray,
    method: str = "percentile",
    value: float = 90.0,
) -> np.ndarray:
    """Convert continuous attention heatmap [0,1] to binary mask.

    Methods:
        "percentile": Pixels above the `value`-th percentile (e.g. 90 = top 10%).
        "otsu":       Otsu's automatic threshold (`value` is ignored).
        "fixed":      Fixed threshold at `value` (should be in [0, 1]).
        "top_k":      Top `value` pixels by attention weight.
    """
    if method == "percentile":
        threshold = np.percentile(heatmap, value)
        return (heatmap >= threshold).astype(np.float32)
    elif method == "otsu":
        hm_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(
            hm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return (binary > 0).astype(np.float32)
    elif method == "fixed":
        return (heatmap >= value).astype(np.float32)
    elif method == "top_k":
        k = int(value)
        flat = heatmap.flatten()
        if k >= len(flat):
            return np.ones_like(heatmap, dtype=np.float32)
        threshold = np.partition(flat, -k)[-k]
        return (heatmap >= threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")


# ── Core IoU Analysis ────────────────────────────────────────────────────────


DEFAULT_THRESHOLD_METHODS: List[Tuple[str, float]] = [
    ("percentile", 90),
    ("percentile", 75),
    ("percentile", 50),
    ("otsu", 0),
]


def compute_attention_object_iou(
    attention_heatmap: np.ndarray,
    segmentation_mask: np.ndarray,
    object_ids: Dict[str, int],
    threshold_methods: Optional[List[Tuple[str, float]]] = None,
) -> Dict:
    """Compute IoU between thresholded attention and ground-truth object masks.

    Args:
        attention_heatmap: 2D float array in [0, 1], shape (H, W).
        segmentation_mask: 2D int array from env (instance IDs), shape (H, W).
        object_ids: {object_name: segmentation_id} for objects of interest.
        threshold_methods: List of (method, value) pairs for thresholding.

    Returns:
        Dict with keys:
          per_object:     {obj_name: {threshold_key: {iou, dice, precision, recall}}}
          combined:       {threshold_key: {iou, dice, precision, recall}}
          attention_mass: {obj_name: fraction, _all_objects: fraction}
          pointing_hit:   bool (does max-attention pixel overlap any object?)
    """
    if threshold_methods is None:
        threshold_methods = DEFAULT_THRESHOLD_METHODS

    # Ensure matching resolution
    if attention_heatmap.shape != segmentation_mask.shape[:2]:
        attention_heatmap = cv2.resize(
            attention_heatmap.astype(np.float32),
            (segmentation_mask.shape[1], segmentation_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    seg_2d = segmentation_mask.squeeze() if segmentation_mask.ndim > 2 else segmentation_mask

    results = {
        "per_object": {},
        "combined": {},
        "attention_mass": {},
    }

    # Build combined mask of all objects of interest
    combined_obj_mask = np.zeros(seg_2d.shape, dtype=bool)
    total_attn = attention_heatmap.sum()

    for obj_name, seg_id in object_ids.items():
        obj_mask = seg_2d == seg_id
        combined_obj_mask |= obj_mask

        # Attention mass (soft, no thresholding)
        mass = float(attention_heatmap[obj_mask].sum() / total_attn) if total_attn > 0 else 0.0
        results["attention_mass"][obj_name] = mass

        # IoU for each threshold method
        results["per_object"][obj_name] = {}
        for method, value in threshold_methods:
            key = f"{method}_{value}"
            binary_attn = threshold_attention(attention_heatmap, method, value)
            iou = compute_iou(binary_attn, obj_mask)
            dice = compute_dice(binary_attn, obj_mask)
            prec, rec = compute_precision_recall(binary_attn, obj_mask)
            results["per_object"][obj_name][key] = {
                "iou": iou,
                "dice": dice,
                "precision": prec,
                "recall": rec,
            }

    # Combined (all objects of interest)
    if total_attn > 0:
        results["attention_mass"]["_all_objects"] = float(
            attention_heatmap[combined_obj_mask].sum() / total_attn
        )
    else:
        results["attention_mass"]["_all_objects"] = 0.0

    for method, value in threshold_methods:
        key = f"{method}_{value}"
        binary_attn = threshold_attention(attention_heatmap, method, value)
        iou = compute_iou(binary_attn, combined_obj_mask)
        dice = compute_dice(binary_attn, combined_obj_mask)
        prec, rec = compute_precision_recall(binary_attn, combined_obj_mask)
        results["combined"][key] = {
            "iou": iou,
            "dice": dice,
            "precision": prec,
            "recall": rec,
        }

    # Pointing accuracy
    max_idx = np.unravel_index(np.argmax(attention_heatmap), attention_heatmap.shape)
    results["pointing_hit"] = bool(combined_obj_mask[max_idx])

    return results


# ── Episode-Level Aggregation ────────────────────────────────────────────────


def summarize_episode_iou(
    step_results: List[Dict],
    threshold_key: str = "percentile_90",
) -> Dict:
    """Aggregate per-step IoU results into episode-level summary.

    Args:
        step_results: List of dicts from compute_attention_object_iou().
        threshold_key: Which threshold to use for summary stats.

    Returns:
        Dict with mean/std/min/max IoU, attention mass, pointing accuracy.
    """
    if not step_results:
        return {}

    combined_ious = []
    combined_dices = []
    attention_masses = []
    pointing_hits = []

    # Collect all object names across steps
    all_objects = set()
    for r in step_results:
        all_objects.update(r.get("per_object", {}).keys())

    per_object_ious = {obj: [] for obj in all_objects}

    for r in step_results:
        # Combined metrics
        combined = r.get("combined", {}).get(threshold_key, {})
        if combined:
            combined_ious.append(combined["iou"])
            combined_dices.append(combined["dice"])

        # Attention mass
        mass = r.get("attention_mass", {}).get("_all_objects", 0.0)
        attention_masses.append(mass)

        # Pointing
        pointing_hits.append(r.get("pointing_hit", False))

        # Per-object
        for obj in all_objects:
            obj_data = r.get("per_object", {}).get(obj, {}).get(threshold_key, {})
            if obj_data:
                per_object_ious[obj].append(obj_data["iou"])

    def _stats(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "threshold": threshold_key,
        "num_steps": len(step_results),
        "combined_iou": _stats(combined_ious),
        "combined_dice": _stats(combined_dices),
        "attention_mass_on_objects": _stats(attention_masses),
        "pointing_accuracy": float(np.mean(pointing_hits)) if pointing_hits else 0.0,
        "per_object_iou": {obj: _stats(vals) for obj, vals in per_object_ious.items()},
    }

    return summary


# ── Visualization ────────────────────────────────────────────────────────────


def _colorize_segmentation(seg_mask: np.ndarray, object_ids: Dict[str, int]) -> np.ndarray:
    """Create an RGB visualization of the segmentation mask."""
    h, w = seg_mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Use distinct colors for each object
    color_cycle = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 128, 0),  # orange
        (128, 0, 255),  # purple
    ]
    for i, (obj_name, seg_id) in enumerate(object_ids.items()):
        color = color_cycle[i % len(color_cycle)]
        rgb[seg_mask == seg_id] = color
    return rgb


def visualize_attention_vs_segmentation(
    frame_rgb: np.ndarray,
    attention_heatmap: np.ndarray,
    segmentation_mask: np.ndarray,
    object_ids: Dict[str, int],
    iou_results: Dict,
    layer_idx: int = 0,
    threshold_key: str = "percentile_90",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel visualization comparing attention heatmap to object segmentation.

    Layout (2x3):
      [Original frame]  [Attention heatmap overlay]  [Segmentation overlay]
      [Thresholded attn] [Overlap visualization]      [IoU metrics text]
    """
    # Ensure matching resolution
    h, w = segmentation_mask.shape[:2]
    if attention_heatmap.shape != (h, w):
        attention_heatmap = cv2.resize(
            attention_heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
    if frame_rgb.shape[:2] != (h, w):
        frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    seg_2d = segmentation_mask.squeeze() if segmentation_mask.ndim > 2 else segmentation_mask

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)

    # 1. Original frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame_rgb)
    ax1.set_title("Original Frame", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # 2. Attention heatmap overlay
    ax2 = fig.add_subplot(gs[0, 1])
    attn_overlay = overlay_heatmap(frame_rgb, attention_heatmap)
    ax2.imshow(attn_overlay)
    ax2.set_title(f"Attention Heatmap (Layer {layer_idx})", fontsize=13, fontweight="bold")
    ax2.axis("off")

    # 3. Segmentation overlay
    ax3 = fig.add_subplot(gs[0, 2])
    seg_rgb = _colorize_segmentation(seg_2d, object_ids)
    seg_overlay = (0.5 * frame_rgb.astype(np.float32) + 0.5 * seg_rgb.astype(np.float32)).astype(np.uint8)
    ax3.imshow(seg_overlay)
    obj_names = list(object_ids.keys())
    color_cycle = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]
    for i, name in enumerate(obj_names):
        ax3.plot([], [], "s", color=color_cycle[i % len(color_cycle)], markersize=10, label=name)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax3.set_title("Ground Truth Objects", fontsize=13, fontweight="bold")
    ax3.axis("off")

    # 4. Thresholded attention
    ax4 = fig.add_subplot(gs[1, 0])
    method, value = threshold_key.rsplit("_", 1)
    binary_attn = threshold_attention(attention_heatmap, method, float(value))
    ax4.imshow(binary_attn, cmap="gray")
    ax4.set_title(f"Thresholded Attention ({threshold_key})", fontsize=13, fontweight="bold")
    ax4.axis("off")

    # 5. Overlap visualization
    ax5 = fig.add_subplot(gs[1, 1])
    combined_obj = np.zeros(seg_2d.shape, dtype=bool)
    for seg_id in object_ids.values():
        combined_obj |= (seg_2d == seg_id)
    overlap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    attn_only = binary_attn.astype(bool) & ~combined_obj
    obj_only = combined_obj & ~binary_attn.astype(bool)
    both = binary_attn.astype(bool) & combined_obj
    overlap_rgb[attn_only] = [255, 100, 100]   # red = attention only
    overlap_rgb[obj_only] = [100, 100, 255]     # blue = object only
    overlap_rgb[both] = [100, 255, 100]         # green = intersection
    ax5.imshow(overlap_rgb)
    ax5.plot([], [], "s", color="#ff6464", markersize=8, label="Attention only")
    ax5.plot([], [], "s", color="#6464ff", markersize=8, label="Object only")
    ax5.plot([], [], "s", color="#64ff64", markersize=8, label="Intersection")
    ax5.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax5.set_title("Overlap Visualization", fontsize=13, fontweight="bold")
    ax5.axis("off")

    # 6. Metrics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    lines = [f"IoU Metrics (Layer {layer_idx})", "=" * 40, ""]

    # Combined metrics
    combined = iou_results.get("combined", {}).get(threshold_key, {})
    lines.append(f"Combined (all objects):")
    lines.append(f"  IoU:       {combined.get('iou', 0):.4f}")
    lines.append(f"  Dice:      {combined.get('dice', 0):.4f}")
    lines.append(f"  Precision: {combined.get('precision', 0):.4f}")
    lines.append(f"  Recall:    {combined.get('recall', 0):.4f}")
    lines.append("")

    # Per-object metrics
    for obj_name in object_ids:
        obj_metrics = iou_results.get("per_object", {}).get(obj_name, {}).get(threshold_key, {})
        mass = iou_results.get("attention_mass", {}).get(obj_name, 0)
        lines.append(f"{obj_name}:")
        lines.append(f"  IoU:  {obj_metrics.get('iou', 0):.4f}  |  Mass: {mass:.1%}")

    lines.append("")
    total_mass = iou_results.get("attention_mass", {}).get("_all_objects", 0)
    lines.append(f"Total attn on objects: {total_mass:.1%}")
    lines.append(f"Pointing hit: {'Yes' if iou_results.get('pointing_hit') else 'No'}")

    ax6.text(
        0.05, 0.95, "\n".join(lines), fontsize=10, verticalalignment="top",
        family="monospace", transform=ax6.transAxes,
    )

    plt.suptitle(
        "Attention vs Ground Truth Segmentation",
        fontsize=15, fontweight="bold", y=0.98,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, colormap: str = "jet", alpha: float = 0.5
) -> np.ndarray:
    """Overlay a [0,1] heatmap on an image. Returns uint8 RGB."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]
    blended = (1 - alpha) * image + alpha * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def visualize_iou_over_episode(
    step_results: List[Dict],
    step_indices: List[int],
    prompt: str,
    threshold_key: str = "percentile_90",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot IoU and attention mass evolution across an episode."""
    combined_ious = []
    attention_masses = []
    pointing_hits = []

    # Collect per-object IoUs
    all_objects = set()
    for r in step_results:
        all_objects.update(r.get("per_object", {}).keys())
    per_obj_ious = {obj: [] for obj in all_objects}

    for r in step_results:
        c = r.get("combined", {}).get(threshold_key, {})
        combined_ious.append(c.get("iou", 0.0))
        attention_masses.append(r.get("attention_mass", {}).get("_all_objects", 0.0))
        pointing_hits.append(1.0 if r.get("pointing_hit") else 0.0)
        for obj in all_objects:
            obj_data = r.get("per_object", {}).get(obj, {}).get(threshold_key, {})
            per_obj_ious[obj].append(obj_data.get("iou", 0.0))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Combined IoU over time
    ax = axes[0]
    ax.plot(step_indices, combined_ious, "o-", color="steelblue", linewidth=2, label="Combined IoU")
    for obj, vals in per_obj_ious.items():
        ax.plot(step_indices, vals, "x--", alpha=0.6, label=f"{obj} IoU")
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title(f"IoU Over Episode ({threshold_key})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(-0.05, 1.05)

    # 2. Attention mass on objects
    ax = axes[1]
    ax.fill_between(step_indices, attention_masses, alpha=0.4, color="coral")
    ax.plot(step_indices, attention_masses, "o-", color="coral", linewidth=2)
    ax.set_ylabel("Attention Mass", fontsize=12)
    ax.set_title("Fraction of Attention on Objects of Interest", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(-0.05, 1.05)

    # 3. Pointing accuracy
    ax = axes[2]
    ax.bar(step_indices, pointing_hits, width=max(1, (step_indices[-1] - step_indices[0]) / len(step_indices) * 0.8),
           color="seagreen", alpha=0.7)
    ax.set_ylabel("Pointing Hit", fontsize=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_title("Pointing Accuracy (max-attention on object?)", fontsize=14, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Miss", "Hit"])
    ax.grid(alpha=0.3, linestyle="--")

    plt.suptitle(f'Attention-Object IoU Analysis\nPrompt: "{prompt}"', fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── Segmentation Utilities ───────────────────────────────────────────────────


def find_segmentation_key(obs: dict, camera_name: str = "agentview") -> Optional[str]:
    """Find the segmentation observation key in the obs dict."""
    candidates = [
        f"{camera_name}_segmentation_instance",
        f"{camera_name}_segmentation_class",
        f"{camera_name}_segmentation_element",
        f"{camera_name}_seg",
        f"{camera_name}_segmentation",
    ]
    for key in candidates:
        if key in obs:
            return key
    # Fallback: search for any key containing 'seg' and the camera name
    for key in obs:
        if "seg" in key.lower() and camera_name in key.lower():
            return key
    return None


