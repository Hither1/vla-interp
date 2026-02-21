#!/usr/bin/env bash
#SBATCH --job-name=dp-policy-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dp_policy_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=12:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Diffusion Policy — Policy Perturbation Evaluation ────────────────────────
#
# Evaluates Diffusion Policy under policy-level perturbations.
#
# Perturbation modes
# ------------------
# Baseline (no perturbation):
#   sbatch scripts/run_dp_policy_perturb.sh
#
# 25% random action replacement:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_dp_policy_perturb.sh
#
# Random object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_dp_policy_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_dp_policy_perturb.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export MUJOCO_GL=egl

# ── Paths ────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── DP checkpoint ─────────────────────────────────────────────────────────────
CKPT="${CKPT:-}"  # Required: set via env var, e.g. CKPT=checkpoints/dp/ckpt_300.pt

# Policy perturbation
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.25}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.05}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

REPLAN_STEPS="${REPLAN_STEPS:-8}"
DEVICE="${DEVICE:-cuda}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-0}"
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Derived output tag ────────────────────────────────────────────────────────
if [[ "$POLICY_PERTURB_MODE" == "none" ]]; then
    PERTURB_TAG="none"
elif [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
    PERTURB_TAG="random_action_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
    PERTURB_TAG="object_shift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
else
    PERTURB_TAG="${POLICY_PERTURB_MODE}"
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
echo "Policy perturbation: ${POLICY_PERTURB_MODE} (${PERTURB_TAG})"
echo "  random_action_prob:   ${RANDOM_ACTION_PROB}"
echo "  random_action_scale:  ${RANDOM_ACTION_SCALE}"
echo "  object_shift_x_std:   ${OBJECT_SHIFT_X_STD}"
echo "  object_shift_y_std:   ${OBJECT_SHIFT_Y_STD}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "Replan steps:        ${REPLAN_STEPS}"
echo "Prompt mode:         ${PROMPT_MODE}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/dp/policy_perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/dp_eval.py" \
        --ckpt "${CKPT}" \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
        --device "${DEVICE}" \
        --prompt-mode "${PROMPT_MODE}" \
        --custom-prompt "${CUSTOM_PROMPT}" \
        --policy-perturb-mode "${POLICY_PERTURB_MODE}" \
        --random-action-prob "${RANDOM_ACTION_PROB}" \
        --random-action-scale "${RANDOM_ACTION_SCALE}" \
        --object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
        --object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
        --video-out-path "${VIDEO_OUT}"

    echo "Finished: ${SUITE}"
done

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation: ${PERTURB_TAG}"
echo "============================================================"
