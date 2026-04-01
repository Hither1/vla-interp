#!/usr/bin/env bash
#SBATCH --job-name=attn-droid-pi0
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/attn_droid_pi0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=6:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# ──────────────────────────────────────────────────────────────────────────────
# Offline attention analysis for pi0.5-DROID on real ZED camera frames.
#
# Usage (all via env vars):
#
#   # Ratio only, from a directory of frames:
#   DATA_DIR=/path/to/zed_frames \
#   CHECKPOINT=/path/to/pi05_droid \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_pi0.sh
#
#   # From video files:
#   VIDEO=/path/to/exterior.mp4 \
#   VIDEO_WRIST=/path/to/wrist.mp4 \
#   CHECKPOINT=/path/to/pi05_droid \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_pi0.sh
#
#   # With pre-computed masks (IoU):
#   DATA_DIR=/path/to/zed_frames \
#   MASK_DIR=/path/to/masks \
#   CHECKPOINT=/path/to/pi05_droid \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_pi0.sh
#
#   # With SAM3 (auto-segmentation, auto-downloads from HuggingFace):
#   DATA_DIR=/path/to/zed_frames \
#   USE_SAM3=1 \
#   OBJECT_DESC="red cup" \
#   CHECKPOINT=/path/to/pi05_droid \
#   PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_pi0.sh
#
#   # With SAM3 using a local checkpoint:
#   DATA_DIR=/path/to/zed_frames \
#   USE_SAM3=1 OBJECT_DESC="red cup" SAM3_CHECKPOINT=/path/to/sam3.pt \
#   CHECKPOINT=/path/to/pi05_droid PROMPT="pick up the red cup" \
#   sbatch scripts/run_attention_droid_pi0.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate || true
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}/src:${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/n/netscratch/sham_lab/Lab/chloe00/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"
mkdir -p "${WORKDIR}/logs" "${NUMBA_CACHE_DIR}"

# ── Required: checkpoint ───────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: CHECKPOINT is required."
    echo "  CHECKPOINT=/path/to/pi05_droid sbatch $0"
    exit 1
fi

# ── Required: data source (one of DATA_DIR or VIDEO) ──────────────────────────
DATA_DIR="${DATA_DIR:-}"
VIDEO="${VIDEO:-}"
VIDEO_WRIST="${VIDEO_WRIST:-}"

if [[ -z "${DATA_DIR}" && -z "${VIDEO}" ]]; then
    echo "ERROR: Set DATA_DIR (frame directory) or VIDEO (MP4 path)."
    exit 1
fi

# ── Language prompt ────────────────────────────────────────────────────────────
PROMPT="${PROMPT:-}"

# ── Attention layers ───────────────────────────────────────────────────────────
LAYERS="${LAYERS:-0 8 17}"

# ── Image token count override (0 = auto: 512 for DROID pi0/pi0.5) ────────────
NUM_IMAGE_TOKENS="${NUM_IMAGE_TOKENS:-0}"

# ── Frame sampling ─────────────────────────────────────────────────────────────
FRAME_STEP="${FRAME_STEP:-1}"

# ── Segmentation / IoU ────────────────────────────────────────────────────────
MASK_DIR="${MASK_DIR:-}"           # pre-computed .npy masks
USE_SAM3="${USE_SAM3:-0}"          # 1 = use SAM3 text-prompted segmentation
OBJECT_DESC="${OBJECT_DESC:-}"     # text description for SAM3
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}"   # empty = auto-download from HuggingFace
SAM3_VERSION="${SAM3_VERSION:-sam3.1}"   # "sam3" or "sam3.1"
SAM3_CONFIDENCE="${SAM3_CONFIDENCE:-0.5}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-percentile}"
THRESHOLD_VALUE="${THRESHOLD_VALUE:-90.0}"

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_HEATMAPS="${SAVE_HEATMAPS:-0}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-256}"

# Derive output tag from data source
if [[ -n "${DATA_DIR}" ]]; then
    DATA_SLUG="$(basename "${DATA_DIR}")"
else
    DATA_SLUG="$(basename "${VIDEO%.mp4}")"
fi
PROMPT_SLUG="${PROMPT:0:40}"
PROMPT_SLUG="${PROMPT_SLUG// /_}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/results/attention/droid_pi0/${DATA_SLUG}/${PROMPT_SLUG}}"
mkdir -p "${OUTPUT_DIR}"

# ── Build argument arrays ──────────────────────────────────────────────────────
read -r -a LAYERS_ARR <<< "${LAYERS}"

DATA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
    DATA_ARGS+=(--data-dir "${DATA_DIR}")
else
    DATA_ARGS+=(--video "${VIDEO}")
    [[ -n "${VIDEO_WRIST}" ]] && DATA_ARGS+=(--video-wrist "${VIDEO_WRIST}")
fi

SEG_ARGS=()
if [[ -n "${MASK_DIR}" ]]; then
    SEG_ARGS+=(--mask-dir "${MASK_DIR}")
elif [[ "${USE_SAM3}" == "1" || "${USE_SAM3}" == "true" ]]; then
    SEG_ARGS+=(--use-sam3)
    [[ -n "${OBJECT_DESC}" ]]      && SEG_ARGS+=(--object-desc "${OBJECT_DESC}")
    [[ -n "${SAM3_CHECKPOINT}" ]]  && SEG_ARGS+=(--sam3-checkpoint "${SAM3_CHECKPOINT}")
    SEG_ARGS+=(--sam3-version "${SAM3_VERSION}")
    SEG_ARGS+=(--sam3-confidence "${SAM3_CONFIDENCE}")
fi

SAVE_HEATMAPS_ARG=()
[[ "${SAVE_HEATMAPS}" == "1" || "${SAVE_HEATMAPS}" == "true" ]] && SAVE_HEATMAPS_ARG+=(--save-heatmaps)

NUM_IMG_TOK_ARG=()
[[ "${NUM_IMAGE_TOKENS}" -gt 0 ]] && NUM_IMG_TOK_ARG+=(--num-image-tokens "${NUM_IMAGE_TOKENS}")

# ── Print config ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Model:        pi0.5-DROID"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Data source:  ${DATA_DIR:-${VIDEO}}"
echo "Prompt:       ${PROMPT}"
echo "Layers:       ${LAYERS}"
echo "Frame step:   ${FRAME_STEP}"
if [[ -n "${MASK_DIR}" ]]; then
    echo "Segmentation: pre-computed masks from ${MASK_DIR}"
elif [[ "${USE_SAM3}" == "1" ]]; then
    echo "Segmentation: SAM3 ${SAM3_VERSION} (object: '${OBJECT_DESC}')"
else
    echo "Segmentation: none (ratio only)"
fi
echo "Output:       ${OUTPUT_DIR}"
echo "============================================================"

# ── Run ────────────────────────────────────────────────────────────────────────
python "${WORKDIR}/analysis/attention/evaluate_attention_droid_pi0.py" \
    --checkpoint "${CHECKPOINT}" \
    "${DATA_ARGS[@]}" \
    --frame-step "${FRAME_STEP}" \
    --prompt "${PROMPT}" \
    --layers "${LAYERS_ARR[@]}" \
    --max-token-len "${MAX_TOKEN_LEN}" \
    --threshold-method "${THRESHOLD_METHOD}" \
    --threshold-value "${THRESHOLD_VALUE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${NUM_IMG_TOK_ARG[@]}" \
    "${SEG_ARGS[@]}" \
    "${SAVE_HEATMAPS_ARG[@]}"

echo ""
echo "Results: ${OUTPUT_DIR}/attention_results_pi0_droid.json"
echo "Done."
