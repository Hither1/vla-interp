#!/usr/bin/env bash
#SBATCH --job-name=cosmos-policy-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/cosmos_policy_perturb_%j.log
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

# ── Cosmos Policy — Policy Perturbation Evaluation ───────────────────────────
#
# Evaluates Cosmos-Policy-LIBERO under policy-level perturbations.
#
# Perturbation modes
# ------------------
# Baseline (no perturbation):
#   sbatch scripts/run_cosmos_policy_perturb.sh
#
# 25% random action replacement:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_cosmos_policy_perturb.sh
#
# Random object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_cosmos_policy_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_cosmos_policy_perturb.sh
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

CHUNK_SIZE="${CHUNK_SIZE:-16}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS_ACTION="${NUM_DENOISING_STEPS_ACTION:-5}"

# Policy perturbation
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.25}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.05}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-195}"
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
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Cosmos Policy"
echo "Checkpoint:          ${CKPT_PATH}"
echo "Policy perturbation: ${POLICY_PERTURB_MODE} (${PERTURB_TAG})"
echo "  random_action_prob:   ${RANDOM_ACTION_PROB}"
echo "  random_action_scale:  ${RANDOM_ACTION_SCALE}"
echo "  object_shift_x_std:   ${OBJECT_SHIFT_X_STD}"
echo "  object_shift_y_std:   ${OBJECT_SHIFT_Y_STD}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "Prompt mode:         ${PROMPT_MODE}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/cosmos/policy_perturb/${PERTURB_TAG}/${SUITE}"
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
