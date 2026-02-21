"""Visual-level perturbations for LIBERO policy evaluation.

Perturbations are applied to uint8 (H, W, 3) RGB images *after* the standard
upside-down correction but *before* passing to the policy model.  This lets
you measure how robust a policy is to changes in camera orientation and
framing (simulating a tilted or shifted camera).

Modes
-----
none
    Identity – no perturbation.
rotate
    Rotate the image by ``rotation_degrees`` degrees (CCW).  Black pixels fill
    the regions exposed by rotation.
translate
    Shift the image by ``(translate_x_frac, translate_y_frac)`` of the image
    dimensions (positive x = right, positive y = down).  Black pixels fill the
    exposed border.
rotate_translate
    Apply rotation first, then translation.

Usage
-----
    from visual_perturbations import VisualPerturbConfig, perturb_image

    cfg = VisualPerturbConfig(mode="rotate", rotation_degrees=30.0)
    perturbed = perturb_image(img, cfg)   # (H, W, 3) uint8
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class VisualPerturbConfig:
    """Parameters for a single visual perturbation.

    All eval scripts accept these as flat CLI arguments and construct this
    dataclass in ``eval_libero()``.
    """

    mode: str = "none"
    """Perturbation mode.  One of: none | rotate | translate | rotate_translate."""

    rotation_degrees: float = 30.0
    """Rotation angle in degrees (counter-clockwise).
    Used by: rotate, rotate_translate."""

    translate_x_frac: float = 0.2
    """Horizontal shift as a fraction of image width (positive = right).
    Used by: translate, rotate_translate."""

    translate_y_frac: float = 0.0
    """Vertical shift as a fraction of image height (positive = down).
    Used by: translate, rotate_translate."""

    def as_dict(self) -> dict:
        """Serialisable summary for JSON logging."""
        return {
            "mode": self.mode,
            "rotation_degrees": self.rotation_degrees if self.mode in ("rotate", "rotate_translate") else None,
            "translate_x_frac": self.translate_x_frac if self.mode in ("translate", "rotate_translate") else None,
            "translate_y_frac": self.translate_y_frac if self.mode in ("translate", "rotate_translate") else None,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate *img* by *degrees* CCW around its centre.

    Uses bilinear interpolation (scipy) when available; falls back to a
    nearest-neighbour NumPy implementation.

    Parameters
    ----------
    img:
        (H, W, 3) uint8 array.
    degrees:
        Counter-clockwise rotation in degrees.

    Returns
    -------
    (H, W, 3) uint8 array.  Exposed corners are filled with zeros (black).
    """
    try:
        from scipy.ndimage import rotate as _sp_rotate  # type: ignore
        # axes=(0, 1) → rotate in the (H, W) plane; cval=0 fills exposed regions.
        rotated = _sp_rotate(img.astype(np.float32), degrees, axes=(0, 1), reshape=False, cval=0.0)
        return np.clip(rotated, 0, 255).astype(np.uint8)
    except ImportError:
        pass

    # Pure-NumPy fallback (nearest-neighbour)
    H, W = img.shape[:2]
    angle_rad = np.deg2rad(-degrees)  # negate for CCW in image coords
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    ys, xs = np.mgrid[0:H, 0:W]
    # Offset from centre
    ys_c = ys - cy
    xs_c = xs - cx
    # Inverse rotation to find source pixel
    src_y = cos_a * ys_c - sin_a * xs_c + cy
    src_x = sin_a * ys_c + cos_a * xs_c + cx

    src_y = np.round(src_y).astype(int)
    src_x = np.round(src_x).astype(int)

    valid = (src_y >= 0) & (src_y < H) & (src_x >= 0) & (src_x < W)
    result = np.zeros_like(img)
    result[valid] = img[src_y[valid], src_x[valid]]
    return result


def _translate_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift *img* by *(dy, dx)* pixels; exposed borders are filled with zeros.

    Parameters
    ----------
    img:
        (H, W, 3) uint8 array.
    dy:
        Vertical shift in pixels (positive = down).
    dx:
        Horizontal shift in pixels (positive = right).

    Returns
    -------
    (H, W, 3) uint8 array.
    """
    H, W = img.shape[:2]
    result = np.zeros_like(img)

    # Source region in the original image
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)

    # Destination region in the output image
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perturb_image(img: np.ndarray, cfg: VisualPerturbConfig) -> np.ndarray:
    """Apply a visual perturbation to a single (H, W, 3) uint8 image.

    Parameters
    ----------
    img:
        Input image of shape (H, W, 3), dtype uint8.
    cfg:
        Perturbation configuration produced by :class:`VisualPerturbConfig`.

    Returns
    -------
    Perturbed image of shape (H, W, 3), dtype uint8.

    Notes
    -----
    * ``mode="none"`` returns *img* unchanged (zero-copy).
    * Rotation is applied before translation in ``rotate_translate`` mode.
    * Both operations preserve the image resolution; exposed pixels are black.
    """
    if cfg.mode == "none":
        return img

    if cfg.mode not in ("rotate", "translate", "rotate_translate"):
        raise ValueError(
            f"Unknown visual_perturb_mode: {cfg.mode!r}. "
            "Choose one of: none | rotate | translate | rotate_translate"
        )

    H, W = img.shape[:2]
    result = img

    if cfg.mode in ("rotate", "rotate_translate"):
        result = _rotate_image(result, cfg.rotation_degrees)

    if cfg.mode in ("translate", "rotate_translate"):
        dy = int(round(cfg.translate_y_frac * H))
        dx = int(round(cfg.translate_x_frac * W))
        result = _translate_image(result, dy, dx)

    return result


def perturb_images(images: dict[str, np.ndarray], cfg: VisualPerturbConfig) -> dict[str, np.ndarray]:
    """Apply the same perturbation to a dict of named images.

    Parameters
    ----------
    images:
        Mapping from image name to (H, W, 3) uint8 array.
    cfg:
        Perturbation configuration.

    Returns
    -------
    New dict with perturbed images (original dict is unchanged).
    """
    return {k: perturb_image(v, cfg) for k, v in images.items()}
