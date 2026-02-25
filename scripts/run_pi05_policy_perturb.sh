#!/usr/bin/env bash
#SBATCH --job-name=pi05-policy-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi05_policy_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100 
#SBATCH --time=18:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Pi0.5 — Policy Perturbation Evaluation ───────────────────────────────────
#
# Starts the OpenPI WebSocket policy server, then runs the LIBERO client with
# policy-level perturbations to measure robustness.
#
# Perturbation modes
# ------------------
# Baseline (no perturbation):
#   sbatch scripts/run_pi05_policy_perturb.sh
#
# 25% random action replacement:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_pi05_policy_perturb.sh
#
# Random object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_pi05_policy_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_pi05_policy_perturb.sh
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

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
ENV_NAME="LIBERO"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"

# Policy perturbation
# mode: none | random_action | object_shift
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"
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

# ── Server lifecycle helpers ──────────────────────────────────────────────────
# wait_for_server() {
#     echo "Waiting for policy server on port ${PORT}..."
#     for i in {1..300}; do
#         if nc -z localhost "${PORT}" 2>/dev/null; then
#             echo "Server is up ($((i * 2))s)."
#             return 0
#         fi
#         sleep 2
#     done
#     echo "ERROR: Policy server failed to start within 600s."
#     exit 1
# }

# kill_server() {
#     if [[ -n "${SERVER_PID:-}" ]]; then
#         echo "Stopping policy server (PID=${SERVER_PID})..."
#         kill "${SERVER_PID}" 2>/dev/null || true
#         wait "${SERVER_PID}" 2>/dev/null || true
#         unset SERVER_PID
#     fi
# }

# trap kill_server EXIT

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Pi0.5"
echo "Policy server port:  ${PORT}"
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

# ── Start policy server ───────────────────────────────────────────────────────
# echo ""
# echo "Starting policy server..."
# SERVER_LOG="${WORKDIR}/logs/pi05_server_${SLURM_JOB_ID:-0}.log"
# PYTHONUNBUFFERED=1 python "${WORKDIR}/scripts/serve_policy.py" --env "${ENV_NAME}" \
#     > "${SERVER_LOG}" 2>&1 &
# SERVER_PID=$!
# wait_for_server

# ── Evaluate each suite ───────────────────────────────────────────────────────
for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/pi05/policy_perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/main.py" \
        --args.port "${PORT}" \
        --args.task-suite-name "${SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.seed "${SEED}" \
        --args.replan-steps "${REPLAN_STEPS}" \
        --args.resize-size "${RESIZE_SIZE}" \
        --args.prompt-mode "${PROMPT_MODE}" \
        --args.custom-prompt "${CUSTOM_PROMPT}" \
        --args.policy-perturb-mode "${POLICY_PERTURB_MODE}" \
        --args.random-action-prob "${RANDOM_ACTION_PROB}" \
        --args.random-action-scale "${RANDOM_ACTION_SCALE}" \
        --args.object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
        --args.object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
        --args.video-out-path "${VIDEO_OUT}" \
        2>&1 | tee "${WORKDIR}/logs/pi05_${SUITE}_${PERTURB_TAG}_${SLURM_JOB_ID:-0}.log"

    echo "Finished: ${SUITE}"
done

# kill_server

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation: ${PERTURB_TAG}"
echo "============================================================"
