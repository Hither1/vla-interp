#!/usr/bin/env bash
#SBATCH --job-name=cosmos-vis-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/cosmos_visual_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=16:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Cosmos Policy — Visual Perturbation Evaluation ───────────────────────────
#
# Evaluates Cosmos-Policy-LIBERO under image-level perturbations
# (orientation / position) to measure robustness.
#
# Quick-start examples
# --------------------
# Baseline (no perturbation):
#   sbatch scripts/run_cosmos_visual_perturb.sh
#
# 30° rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_cosmos_visual_perturb.sh
#
# 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 \
#       sbatch scripts/run_cosmos_visual_perturb.sh
#
# Rotate then translate:
#   VISUAL_PERTURB_MODE=rotate_translate ROTATION_DEGREES=15 TRANSLATE_X_FRAC=0.1 \
#       sbatch scripts/run_cosmos_visual_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_cosmos_visual_perturb.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

# Offscreen rendering for MuJoCo
export MUJOCO_GL=egl

# ── Paths ────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
# cosmos_policy package must be importable
export PYTHONPATH="${WORKDIR}/third_party/cosmos-policy:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Cosmos model config ───────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl}"

# Inference settings
CHUNK_SIZE="${CHUNK_SIZE:-16}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS_ACTION="${NUM_DENOISING_STEPS_ACTION:-5}"

# Visual perturbation
# mode: none | rotate | translate | rotate_translate
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-30.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.2}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-195}"
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Derived output tag ────────────────────────────────────────────────────────
if [[ "$VISUAL_PERTURB_MODE" == "none" ]]; then
    PERTURB_TAG="none"
elif [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
    PERTURB_TAG="rotate_${ROTATION_DEGREES}deg"
elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
    PERTURB_TAG="translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
else
    PERTURB_TAG="rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
fi

# ── Task suite list ────────────────────────────────────────────────────────────
if [[ "$TASK_SUITE" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Cosmos Policy"
echo "Checkpoint:          ${CKPT_PATH}"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (${PERTURB_TAG})"
echo "  rotation_degrees:  ${ROTATION_DEGREES}"
echo "  translate_x_frac:  ${TRANSLATE_X_FRAC}"
echo "  translate_y_frac:  ${TRANSLATE_Y_FRAC}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "Prompt mode:         ${PROMPT_MODE}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/cosmos/visual_perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/cosmos_eval.py" \
        --ckpt-path "${CKPT_PATH}" \
        --config-name "${CONFIG_NAME}" \
        --config-file "${CONFIG_FILE}" \
        --dataset-stats-path "${DATASET_STATS_PATH}" \
        --t5-text-embeddings-path "${T5_EMBEDDINGS_PATH}" \
        --chunk-size "${CHUNK_SIZE}" \
        --num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}" \
        --num-denoising-steps-action "${NUM_DENOISING_STEPS_ACTION}" \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --prompt-mode "${PROMPT_MODE}" \
        --custom-prompt "${CUSTOM_PROMPT}" \
        --visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
        --rotation-degrees "${ROTATION_DEGREES}" \
        --translate-x-frac "${TRANSLATE_X_FRAC}" \
        --translate-y-frac "${TRANSLATE_Y_FRAC}" \
        --video-out-path "${VIDEO_OUT}"

    echo "Finished: ${SUITE}"
done

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation: ${PERTURB_TAG}"
echo "============================================================"
