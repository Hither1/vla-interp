#!/usr/bin/env bash
#SBATCH --job-name=attn-droid-dz
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/attn_droid_dreamzero_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────────────────────
# Offline attention analysis for DreamZero-DROID on real ZED camera frames.
# Uses torchrun for multi-GPU distributed inference.
#
# Usage (all via env vars):
#
#   # Ratio only, from a directory of frames:
#   DATA_DIR=/path/to/zed_frames \
#   CKPT=/path/to/dreamzero_droid_ckpt \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # From video files (3 cameras):
#   VIDEO=/path/to/ext1.mp4 \
#   VIDEO_EXT2=/path/to/ext2.mp4 \
#   VIDEO_WRIST=/path/to/wrist.mp4 \
#   CKPT=/path/to/dreamzero_droid_ckpt \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # From Google Drive by perturbation name (requires rclone "gdrive" remote):
#   PERTURBATION=shuffle \
#   CKPT=/path/to/dreamzero_droid_ckpt \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # Video is a 2x2 grid: top=wrist (full width), bottom=left_ext|right_ext.
#   # Splitting is automatic (GDRIVE_SPLIT_COMBINED=1 by default).
#
#   # With pre-computed masks (IoU):
#   DATA_DIR=/path/to/zed_frames \
#   MASK_DIR=/path/to/masks \
#   CKPT=/path/to/dreamzero_droid_ckpt \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # With SAM3 (auto-segmentation, auto-downloads from HuggingFace):
#   DATA_DIR=/path/to/zed_frames \
#   USE_SAM3=1 \
#   OBJECT_DESC="red cup" \
#   CKPT=/path/to/dreamzero_droid_ckpt \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # With SAM3 using a local checkpoint:
#   DATA_DIR=/path/to/zed_frames \
#   USE_SAM3=1 OBJECT_DESC="red cup" SAM3_CHECKPOINT=/path/to/sam3.pt \
#   CKPT=/path/to/dreamzero_droid_ckpt PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_dreamzero.sh
#
#   # Fewer GPUs (e.g. 2):
#   NUM_GPUS=2 CKPT=... DATA_DIR=... sbatch scripts/run_attention_droid_dreamzero.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate || true
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export DISABLE_TORCH_COMPILE=true
export TORCHDYNAMO_DISABLE=1
export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
DREAMZERO_DIR="${WORKDIR}/dreamzero"
export PYTHONPATH="${WORKDIR}/src:${WORKDIR}:${DREAMZERO_DIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/n/netscratch/sham_lab/Lab/chloe00/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"
mkdir -p "${WORKDIR}/logs" "${NUMBA_CACHE_DIR}"

# ── Required: checkpoint ───────────────────────────────────────────────────────
CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: CKPT is required."
    echo "  CKPT=/path/to/dreamzero_droid_ckpt sbatch $0"
    exit 1
fi

# ── Google Drive / rclone (optional) ──────────────────────────────────────────
# Set PERTURBATION to download directly from Google Drive via rclone.
#
#   PERTURBATION=shuffle CKPT=... PROMPT=... sbatch scripts/run_attention_droid_dreamzero.sh
#
# Folder layout expected on Drive (rclone remote "gdrive"):
#   gdrive:DROID/dreamzero/<perturbation>/
#
# Videos are 2x2 grid .MOV (top=wrist full-width, bottom=left_ext|right_ext).
# Set GDRIVE_SPLIT_COMBINED=0 to skip splitting (if files are already separate).
#
# Override the remote or root path if needed:
#   GDRIVE_REMOTE=gdrive  GDRIVE_DREAMZERO_ROOT="DROID/dreamzero"
PERTURBATION="${PERTURBATION:-}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_DREAMZERO_ROOT="${GDRIVE_DREAMZERO_ROOT:-DROID/dreamzero}"
GDRIVE_SPLIT_COMBINED="${GDRIVE_SPLIT_COMBINED:-1}"
GDRIVE_TMPDIR=""

if [[ -n "${PERTURBATION}" ]]; then
    GDRIVE_TMPDIR="$(mktemp -d)"
    GDRIVE_SRC="${GDRIVE_REMOTE}:${GDRIVE_DREAMZERO_ROOT}/${PERTURBATION}"
    echo "Downloading from Drive: ${GDRIVE_SRC}"
    rclone copy "${GDRIVE_SRC}" "${GDRIVE_TMPDIR}/" --progress

    if [[ "${GDRIVE_SPLIT_COMBINED}" == "1" || "${GDRIVE_SPLIT_COMBINED}" == "true" ]]; then
        # 2x2 grid layout:
        #   [wrist       | wrist      ]   ← top half, full width
        #   [left_ext    | right_ext  ]   ← bottom half, split L/R
        COMBINED_VIDEO="$(find "${GDRIVE_TMPDIR}" -maxdepth 1 \( -iname "*.mov" -o -iname "*.mp4" \) | head -1)"
        if [[ -z "${COMBINED_VIDEO}" ]]; then
            echo "ERROR: No .MOV or .mp4 found in Drive folder: ${GDRIVE_SRC}"
            exit 1
        fi
        echo "Splitting 2x2 grid video: ${COMBINED_VIDEO}"
        # wrist: top half, full width
        ffmpeg -loglevel warning -i "${COMBINED_VIDEO}" \
            -vf "crop=iw:ih/2:0:0" -c:v libx264 -crf 18 -an \
            "${GDRIVE_TMPDIR}/wrist.mp4"
        # left exterior: bottom-left quadrant
        ffmpeg -loglevel warning -i "${COMBINED_VIDEO}" \
            -vf "crop=iw/2:ih/2:0:ih/2" -c:v libx264 -crf 18 -an \
            "${GDRIVE_TMPDIR}/exterior.mp4"
        # right exterior: bottom-right quadrant
        ffmpeg -loglevel warning -i "${COMBINED_VIDEO}" \
            -vf "crop=iw/2:ih/2:iw/2:ih/2" -c:v libx264 -crf 18 -an \
            "${GDRIVE_TMPDIR}/exterior2.mp4"
        VIDEO="${GDRIVE_TMPDIR}/exterior.mp4"
        VIDEO_EXT2="${GDRIVE_TMPDIR}/exterior2.mp4"
        VIDEO_WRIST="${GDRIVE_TMPDIR}/wrist.mp4"
    else
        # Separate files — match by name pattern
        VIDEO="$(find "${GDRIVE_TMPDIR}" -maxdepth 1 \( -iname "*ext*1*" -o -iname "*exterior*" -o -iname "*left*" \) \( -iname "*.mov" -o -iname "*.mp4" \) | head -1)"
        VIDEO_EXT2="$(find "${GDRIVE_TMPDIR}" -maxdepth 1 \( -iname "*ext*2*" \) \( -iname "*.mov" -o -iname "*.mp4" \) | head -1)"
        VIDEO_WRIST="$(find "${GDRIVE_TMPDIR}" -maxdepth 1 \( -iname "*wrist*" \) \( -iname "*.mov" -o -iname "*.mp4" \) | head -1)"
        # Fallback: first video found if name patterns don't match
        if [[ -z "${VIDEO}" ]]; then
            VIDEO="$(find "${GDRIVE_TMPDIR}" -maxdepth 1 \( -iname "*.mov" -o -iname "*.mp4" \) | sort | head -1)"
        fi
        if [[ -z "${VIDEO}" ]]; then
            echo "ERROR: No video files found in Drive folder: ${GDRIVE_SRC}"
            exit 1
        fi
    fi
fi

# Cleanup temp dir on exit (only if we created one)
if [[ -n "${GDRIVE_TMPDIR}" ]]; then
    trap 'echo "Cleaning up ${GDRIVE_TMPDIR}"; rm -rf "${GDRIVE_TMPDIR}"' EXIT
fi

# ── Required: data source ─────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-}"
VIDEO="${VIDEO:-}"
VIDEO_EXT2="${VIDEO_EXT2:-}"
VIDEO_WRIST="${VIDEO_WRIST:-}"

if [[ -z "${DATA_DIR}" && -z "${VIDEO}" ]]; then
    echo "ERROR: Set DATA_DIR (frame directory), VIDEO (MP4), or PERTURBATION (Drive folder)."
    exit 1
fi

# ── Language prompt ────────────────────────────────────────────────────────────
PROMPT="${PROMPT:-}"

# ── Model / inference settings ────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-OXE_DROID}"
NUM_CONTEXT_FRAMES="${NUM_CONTEXT_FRAMES:-4}"   # frames of video history per inference call
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-true}"

# ── Attention layers ───────────────────────────────────────────────────────────
LAYERS="${LAYERS:-10 20 30}"

# ── Frame sampling ─────────────────────────────────────────────────────────────
FRAME_STEP="${FRAME_STEP:-1}"

# ── Segmentation / IoU ────────────────────────────────────────────────────────
MASK_DIR="${MASK_DIR:-}"
USE_SAM3="${USE_SAM3:-0}"
OBJECT_DESC="${OBJECT_DESC:-}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/n/netscratch/sham_lab/Lab/chloe00/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}"
SAM3_VERSION="${SAM3_VERSION:-sam3.1}"   # "sam3" or "sam3.1"
SAM3_CONFIDENCE="${SAM3_CONFIDENCE:-0.5}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-percentile}"
THRESHOLD_VALUE="${THRESHOLD_VALUE:-90.0}"

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_HEATMAPS="${SAVE_HEATMAPS:-0}"

if [[ -n "${DATA_DIR}" ]]; then
    DATA_SLUG="$(basename "${DATA_DIR}")"
else
    DATA_SLUG="$(basename "${VIDEO%.mp4}")"
fi
PROMPT_SLUG="${PROMPT:0:40}"
PROMPT_SLUG="${PROMPT_SLUG// /_}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/results/attention/droid_dreamzero/${DATA_SLUG}/${PROMPT_SLUG}}"
mkdir -p "${OUTPUT_DIR}"

# ── Build argument arrays ──────────────────────────────────────────────────────
read -r -a LAYERS_ARR <<< "${LAYERS}"

DATA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
    DATA_ARGS+=(--data-dir "${DATA_DIR}")
else
    DATA_ARGS+=(--video "${VIDEO}")
    [[ -n "${VIDEO_EXT2}" ]]   && DATA_ARGS+=(--video-ext2 "${VIDEO_EXT2}")
    [[ -n "${VIDEO_WRIST}" ]]  && DATA_ARGS+=(--video-wrist "${VIDEO_WRIST}")
fi

SEG_ARGS=()
if [[ -n "${MASK_DIR}" ]]; then
    SEG_ARGS+=(--mask-dir "${MASK_DIR}")
elif [[ "${USE_SAM3}" == "1" || "${USE_SAM3}" == "true" ]]; then
    SEG_ARGS+=(--use-sam3)
    [[ -n "${OBJECT_DESC}" ]]     && SEG_ARGS+=(--object-desc "${OBJECT_DESC}")
    [[ -n "${SAM3_CHECKPOINT}" ]] && SEG_ARGS+=(--sam3-checkpoint "${SAM3_CHECKPOINT}")
    SEG_ARGS+=(--sam3-version "${SAM3_VERSION}")
    SEG_ARGS+=(--sam3-confidence "${SAM3_CONFIDENCE}")
fi

DIT_CACHE_ARG=()
[[ "${ENABLE_DIT_CACHE}" == "false" ]] && DIT_CACHE_ARG+=(--no-dit-cache)

SAVE_HEATMAPS_ARG=()
[[ "${SAVE_HEATMAPS}" == "1" || "${SAVE_HEATMAPS}" == "true" ]] && SAVE_HEATMAPS_ARG+=(--save-heatmaps)

# ── Print config ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job:               ${SLURM_JOB_ID:-local}"
echo "Model:             DreamZero-DROID"
echo "Checkpoint:        ${CKPT}"
echo "Embodiment tag:    ${EMBODIMENT_TAG}"
echo "Num GPUs:          ${NUM_GPUS}"
echo "Context frames:    ${NUM_CONTEXT_FRAMES}"
echo "DIT cache:         ${ENABLE_DIT_CACHE}"
if [[ -n "${PERTURBATION}" ]]; then
    echo "Perturbation:      ${PERTURBATION} (Drive: ${GDRIVE_REMOTE}:${GDRIVE_DREAMZERO_ROOT}/${PERTURBATION})"
    echo "Data source:       ${VIDEO}${VIDEO_WRIST:+ + ${VIDEO_WRIST}} (from Drive)"
else
    echo "Data source:       ${DATA_DIR:-${VIDEO}}"
fi
echo "Prompt:            ${PROMPT}"
echo "Layers:            ${LAYERS}"
echo "Frame step:        ${FRAME_STEP}"
if [[ -n "${MASK_DIR}" ]]; then
    echo "Segmentation:      pre-computed masks from ${MASK_DIR}"
elif [[ "${USE_SAM3}" == "1" ]]; then
    echo "Segmentation:      SAM3 ${SAM3_VERSION} (object: '${OBJECT_DESC}')"
else
    echo "Segmentation:      none (ratio only)"
fi
echo "Output:            ${OUTPUT_DIR}"
echo "============================================================"

# ── Run ────────────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    "${WORKDIR}/analysis/attention/evaluate_attention_droid_dreamzero.py" \
        --model-path "${CKPT}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --num-context-frames "${NUM_CONTEXT_FRAMES}" \
        "${DATA_ARGS[@]}" \
        --frame-step "${FRAME_STEP}" \
        --prompt "${PROMPT}" \
        --layers "${LAYERS_ARR[@]}" \
        --threshold-method "${THRESHOLD_METHOD}" \
        --threshold-value "${THRESHOLD_VALUE}" \
        --output-dir "${OUTPUT_DIR}" \
        "${DIT_CACHE_ARG[@]}" \
        "${SEG_ARGS[@]}" \
        "${SAVE_HEATMAPS_ARG[@]}"

echo ""
echo "Results: ${OUTPUT_DIR}/attention_results_dreamzero_droid.json"
echo "Done."
