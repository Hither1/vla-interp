#!/usr/bin/env bash
#SBATCH --job-name=dp-vis-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dp_visual_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=08:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

# ── Diffusion Policy — Visual Perturbation Evaluation ────────────────────────
#
# Evaluates the diffusion policy checkpoint under image-level perturbations
# (orientation / position) to measure robustness.
#
# Quick-start examples
# --------------------
# Baseline (no perturbation):
#   sbatch scripts/run_dp_visual_perturb.sh
#
# 30° rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_dp_visual_perturb.sh
#
# 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 \
#       sbatch scripts/run_dp_visual_perturb.sh
#
# Rotate then translate:
#   VISUAL_PERTURB_MODE=rotate_translate ROTATION_DEGREES=15 TRANSLATE_X_FRAC=0.1 \
#       sbatch scripts/run_dp_visual_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_dp_visual_perturb.sh
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
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT="${CKPT:-dp_scratch/ckpt_300.pt}"

# Visual perturbation
# mode: none | rotate | translate | rotate_translate
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-30.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.2}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Derived output tag ────────────────────────────────────────────────────────
# Build a human-readable tag to distinguish perturbation runs.
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
elif [[ "$TASK_SUITE" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Diffusion Policy"
echo "Checkpoint:          ${CKPT}"
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

    VIDEO_OUT="${WORKDIR}/data/libero/dp/visual_perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/dp_eval.py" \
        --ckpt "${WORKDIR}/${CKPT}" \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
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
