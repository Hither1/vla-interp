"""Post-process visual_perturb videos by applying the transformation encoded in
the top-level perturbation folder name.

Folder name format (same as run_dreamzero_perturb.sh VIS_TAG):
  rotate_{deg}deg
  translate_x{xfrac}_y{yfrac}
  rotate_{deg}deg_translate_x{xfrac}_y{yfrac}

The script reads each .mp4, applies the matching per-frame transform in-place,
and overwrites the original file.

Usage:
  python postprocess_visual_perturb_videos.py [--root PATH] [--dry-run] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import rotate as _sp_rotate


# ---------------------------------------------------------------------------
# Perturbation helpers (mirrors visual_perturbations.py)
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, degrees: float) -> np.ndarray:
    rotated = _sp_rotate(img.astype(np.float32), degrees, axes=(0, 1), reshape=False, cval=0.0)
    return np.clip(rotated, 0, 255).astype(np.uint8)


def _translate_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = img.shape[:2]
    result = np.zeros_like(img)
    src_y0 = max(0, -dy); src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx); src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy);  dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx);  dst_x1 = min(W, W + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return result


def apply_perturb(img: np.ndarray, mode: str, rotation_degrees: float,
                  translate_x_frac: float, translate_y_frac: float) -> np.ndarray:
    result = img
    if mode in ("rotate", "rotate_translate"):
        result = _rotate_image(result, rotation_degrees)
    if mode in ("translate", "rotate_translate"):
        H, W = result.shape[:2]
        dy = int(round(translate_y_frac * H))
        dx = int(round(translate_x_frac * W))
        result = _translate_image(result, dy, dx)
    return result


# ---------------------------------------------------------------------------
# Folder name parser
# ---------------------------------------------------------------------------

_ROTATE_RE = re.compile(r"rotate_([\d.]+)deg")
_TRANSL_X_RE = re.compile(r"translate_x(-?[\d.]+)_y(-?[\d.]+)")


def parse_perturb_from_name(folder_name: str):
    """Return (mode, rotation_degrees, translate_x_frac, translate_y_frac)."""
    has_rot = bool(_ROTATE_RE.search(folder_name))
    has_tr  = bool(_TRANSL_X_RE.search(folder_name))

    rotation_degrees  = 0.0
    translate_x_frac  = 0.0
    translate_y_frac  = 0.0

    if has_rot:
        rotation_degrees = float(_ROTATE_RE.search(folder_name).group(1))
    if has_tr:
        m = _TRANSL_X_RE.search(folder_name)
        translate_x_frac = float(m.group(1))
        translate_y_frac = float(m.group(2))

    if has_rot and has_tr:
        mode = "rotate_translate"
    elif has_rot:
        mode = "rotate"
    elif has_tr:
        mode = "translate"
    else:
        mode = "none"

    return mode, rotation_degrees, translate_x_frac, translate_y_frac


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_path: str, mode: str, rotation_degrees: float,
                  translate_x_frac: float, translate_y_frac: float,
                  dry_run: bool = False) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"SKIP (cannot open): {video_path}"

    fps    = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmp_path = video_path + ".tmp.mp4"
    if not dry_run:
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    n_frames = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        perturbed_rgb = apply_perturb(frame_rgb, mode, rotation_degrees,
                                      translate_x_frac, translate_y_frac)
        perturbed_bgr = cv2.cvtColor(perturbed_rgb, cv2.COLOR_RGB2BGR)
        if not dry_run:
            writer.write(perturbed_bgr)
        n_frames += 1

    cap.release()
    if not dry_run:
        writer.release()
        os.replace(tmp_path, video_path)

    return f"OK ({n_frames} frames): {video_path}"


# ---------------------------------------------------------------------------
# Worker shim for multiprocessing
# ---------------------------------------------------------------------------

def _worker(args):
    return process_video(*args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_jobs(root: Path, dry_run: bool):
    jobs = []
    for perturb_dir in root.iterdir():
        if not perturb_dir.is_dir():
            continue
        mode, rot, tx, ty = parse_perturb_from_name(perturb_dir.name)
        if mode == "none":
            print(f"WARNING: could not parse perturbation from '{perturb_dir.name}', skipping.")
            continue
        print(f"Folder '{perturb_dir.name}' → mode={mode}, rot={rot}°, tx={tx}, ty={ty}")
        for video_path in perturb_dir.rglob("*.mp4"):
            jobs.append((str(video_path), mode, rot, tx, ty, dry_run))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero/dp/visual_perturb",
        help="Root directory containing the perturbation sub-folders.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and iterate but do not write any files.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes.")
    args = parser.parse_args()

    root = Path(args.root)
    jobs = collect_jobs(root, args.dry_run)
    print(f"\nTotal videos to process: {len(jobs)}")
    if args.dry_run:
        print("(dry-run — no files will be modified)")
        return

    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futs):
            result = fut.result()
            done += 1
            if not result.startswith("OK"):
                failed.append(result)
                print(f"[{done}/{len(jobs)}] {result}", flush=True)
            elif done % 100 == 0:
                print(f"[{done}/{len(jobs)}] {result}", flush=True)

    print(f"\nDone: {done - len(failed)} succeeded, {len(failed)} failed.")
    for f in failed:
        print("  FAIL:", f)


if __name__ == "__main__":
    main()
