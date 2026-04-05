#!/usr/bin/env bash
#SBATCH --job-name=sam3-mask-vis
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/sam3_mask_vis_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=0:30:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────────────────────
# Run SAM3 mask visualization on a single video.
#
# Usage:
#   # From a local video file:
#   VIDEO=/path/to/input.mp4 \
#   OBJECT_DESC="the book" \
#   sbatch scripts/run_visualize_sam3_masks.sh
#
#   # From Google Drive (requires rclone "gdrive" remote):
#   GDRIVE_VIDEO="MyFolder/recording.mp4" \
#   OBJECT_DESC="the book" \
#   sbatch scripts/run_visualize_sam3_masks.sh
#
#   # Override the rclone remote or root path:
#   GDRIVE_REMOTE=gdrive  GDRIVE_ROOT="MyDrive/videos" \
#   GDRIVE_VIDEO="recording.mp4" \
#   sbatch scripts/run_visualize_sam3_masks.sh
#
#   # With custom output path:
#   VIDEO=... OBJECT_DESC="..." OUTPUT=/path/to/output.mp4 \
#   sbatch scripts/run_visualize_sam3_masks.sh
#
#   # Process every Nth frame (faster):
#   VIDEO=... OBJECT_DESC="..." FRAME_STEP=2 \
#   sbatch scripts/run_visualize_sam3_masks.sh
#
#   # Loop over multiple objects (space-separated):
#   VIDEO=... OBJECT_DESC="banana plate" \
#   sbatch scripts/run_visualize_sam3_masks.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate || true
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}/src:${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/n/netscratch/sham_lab/Lab/chloe00/huggingface}"
mkdir -p "${WORKDIR}/logs"

# ── Google Drive / rclone (optional) ─────────────────────────────────────────
# Set GDRIVE_ROOT to a folder path on the rclone remote; the script picks the
# first .MOV or .mp4 it finds there.  Override GDRIVE_VIDEO to pin a specific
# file instead of auto-selecting.
#
#   GDRIVE_ROOT="MyDrive/videos" OBJECT_DESC="banana" sbatch ...
#
# Override the remote if needed:
#   GDRIVE_REMOTE=gdrive  GDRIVE_ROOT="MyDrive/videos"
GDRIVE_VIDEO="${GDRIVE_VIDEO:-}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_ROOT="${GDRIVE_ROOT:-DROID/pi05/pi05_image_logs}"
GDRIVE_TMPDIR=""

VIDEO="${VIDEO:-}"

if [[ -n "${GDRIVE_ROOT}" || -n "${GDRIVE_VIDEO}" ]]; then
    GDRIVE_TMPDIR="$(mktemp -d)"
    if [[ -n "${GDRIVE_VIDEO}" ]]; then
        # Specific file requested
        if [[ -n "${GDRIVE_ROOT}" ]]; then
            GDRIVE_SRC="${GDRIVE_REMOTE}:${GDRIVE_ROOT}/${GDRIVE_VIDEO}"
        else
            GDRIVE_SRC="${GDRIVE_REMOTE}:${GDRIVE_VIDEO}"
        fi
        echo "Downloading from Drive: ${GDRIVE_SRC}"
        rclone copy "${GDRIVE_SRC}" "${GDRIVE_TMPDIR}/" --progress
    else
        # Auto-select: find the first video in the folder
        GDRIVE_SRC="${GDRIVE_REMOTE}:${GDRIVE_ROOT}"
        echo "Browsing Drive folder: ${GDRIVE_SRC}"
        GDRIVE_VIDEO="$(rclone lsf "${GDRIVE_SRC}" --include "*.mp4" --include "*.MOV" --include "*.mov" | head -1)"
        if [[ -z "${GDRIVE_VIDEO}" ]]; then
            echo "ERROR: No .MOV or .mp4 found in Drive folder: ${GDRIVE_SRC}"
            exit 1
        fi
        echo "Auto-selected: ${GDRIVE_VIDEO}"
        rclone copy "${GDRIVE_SRC}/${GDRIVE_VIDEO}" "${GDRIVE_TMPDIR}/" --progress
    fi
    COMBINED_VIDEO="$(find "${GDRIVE_TMPDIR}" -maxdepth 2 \( -iname "*.mov" -o -iname "*.mp4" \) | head -1)"
    if [[ -z "${COMBINED_VIDEO}" ]]; then
        echo "ERROR: No .MOV or .mp4 found after downloading from: ${GDRIVE_SRC}"
        exit 1
    fi
    # Split side-by-side video: left=exterior, right=wrist
    echo "Splitting side-by-side video: ${COMBINED_VIDEO}"
    export _COMBINED_VIDEO="${COMBINED_VIDEO}" _GDRIVE_TMPDIR="${GDRIVE_TMPDIR}"
    python - <<'PYEOF'
import cv2, os
src = os.environ["_COMBINED_VIDEO"]
tmpdir = os.environ["_GDRIVE_TMPDIR"]
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half = w // 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
ext_w = cv2.VideoWriter(f"{tmpdir}/exterior.mp4", fourcc, fps, (half, h))
wst_w = cv2.VideoWriter(f"{tmpdir}/wrist.mp4",    fourcc, fps, (half, h))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ext_w.write(frame[:, :half])
    wst_w.write(frame[:, half:])
cap.release(); ext_w.release(); wst_w.release()
print(f"Split into exterior.mp4 and wrist.mp4 ({half}x{h} each)")
PYEOF
    VIDEO="${GDRIVE_TMPDIR}/exterior.mp4"
    VIDEO_WRIST="${GDRIVE_TMPDIR}/wrist.mp4"
fi

VIDEO_WRIST="${VIDEO_WRIST:-}"

# Cleanup temp dir on exit (only if we created one)
if [[ -n "${GDRIVE_TMPDIR}" ]]; then
    trap 'echo "Cleaning up ${GDRIVE_TMPDIR}"; rm -rf "${GDRIVE_TMPDIR}"' EXIT
fi

# ── Required: video ───────────────────────────────────────────────────────────
if [[ -z "${VIDEO}" ]]; then
    echo "ERROR: Provide VIDEO=/path/to/input.mp4 or GDRIVE_VIDEO=<drive-path>."
    echo "  VIDEO=/path/to/input.mp4 OBJECT_DESC='the book' sbatch $0"
    exit 1
fi

# ── Optional ──────────────────────────────────────────────────────────────────
# OBJECT_DESC can be a space-separated list of objects, e.g. "banana plate".
# Defaults to "banana plate" if not set.
# Space-separated list of quoted object descriptions, delimited by '|'.
# e.g. OBJECT_DESCS="green cup|blue bowl|the book"
OBJECT_DESCS="${OBJECT_DESCS:-green cup|blue bowl}"
OUTPUT="${OUTPUT:-}"
FRAME_STEP="${FRAME_STEP:-1}"
CONFIDENCE="${CONFIDENCE:-0.3}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/n/netscratch/sham_lab/Lab/chloe00/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7}"

echo "============================================================"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Exterior:    ${VIDEO}"
echo "Wrist:       ${VIDEO_WRIST:-none}"
echo "Objects:     ${OBJECT_DESCS}"
echo "Frame step:  ${FRAME_STEP}"
echo "Confidence:  ${CONFIDENCE}"
echo "Checkpoint:  ${SAM3_CHECKPOINT}"
echo "============================================================"

# ── Run SAM3 on one video, return output path ─────────────────────────────────
run_sam3() {
    local vid="$1" obj="$2" out="$3"
    python "${WORKDIR}/scripts/visualize_sam3_masks.py" \
        --video "${vid}" \
        --object-desc "${obj}" \
        --output "${out}" \
        --sam3-checkpoint "${SAM3_CHECKPOINT}" \
        --frame-step "${FRAME_STEP}" \
        --confidence "${CONFIDENCE}"
}

# ── Run for each object description ──────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}}"
# Use the original Drive filename (e.g. "combined") as the basename when available
if [[ -n "${GDRIVE_VIDEO}" ]]; then
    VIDEO_BASENAME="$(basename "${GDRIVE_VIDEO%.*}")"
else
    VIDEO_BASENAME="$(basename "${VIDEO%.*}")"
fi
IFS='|' read -r -a OBJ_ARR <<< "${OBJECT_DESCS}"
for OBJ in "${OBJ_ARR[@]}"; do
    SLUG="${OBJ// /_}"

    if [[ -n "${VIDEO_WRIST}" ]]; then
        # Process each view separately then hstack them
        EXT_OUT="${GDRIVE_TMPDIR:-${OUTPUT_DIR}}/exterior_sam3_${SLUG}.mp4"
        WST_OUT="${GDRIVE_TMPDIR:-${OUTPUT_DIR}}/wrist_sam3_${SLUG}.mp4"
        FINAL_OUT="${OUTPUT_DIR}/${VIDEO_BASENAME}_sam3_${SLUG}.mp4"

        echo ""
        echo ">>> Object: '${OBJ}'  (exterior)"
        run_sam3 "${VIDEO}" "${OBJ}" "${EXT_OUT}"

        echo ""
        echo ">>> Object: '${OBJ}'  (wrist)"
        run_sam3 "${VIDEO_WRIST}" "${OBJ}" "${WST_OUT}"

        echo ""
        echo ">>> Combining views → ${FINAL_OUT}"
        export _EXT_OUT="${EXT_OUT}" _WST_OUT="${WST_OUT}" _FINAL_OUT="${FINAL_OUT}"
        python - <<'PYEOF'
import cv2, os
import numpy as np
def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps
ext_frames, fps = read_frames(os.environ["_EXT_OUT"])
wst_frames, _   = read_frames(os.environ["_WST_OUT"])
out_path = os.environ["_FINAL_OUT"]
n = min(len(ext_frames), len(wst_frames))
h = max(ext_frames[0].shape[0], wst_frames[0].shape[0])
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w_total = ext_frames[0].shape[1] + wst_frames[0].shape[1]
writer = cv2.VideoWriter(out_path, fourcc, fps, (w_total, h))
for e, w in zip(ext_frames[:n], wst_frames[:n]):
    writer.write(np.hstack([e, w]))
writer.release()
print(f"Wrote {n} frames → {out_path}")
PYEOF
        echo "Done. Output: ${FINAL_OUT}"
    else
        OUT="${OUTPUT_DIR}/${VIDEO_BASENAME}_sam3_${SLUG}.mp4"
        echo ""
        echo ">>> Object: '${OBJ}'  →  ${OUT}"
        run_sam3 "${VIDEO}" "${OBJ}" "${OUT}"
        echo "Done. Output: ${OUT}"
    fi
done
