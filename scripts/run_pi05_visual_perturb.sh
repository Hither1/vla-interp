#!/usr/bin/env bash
#SBATCH --job-name=pi05-vis-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi05_visual_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=18:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Pi0.5 — Visual Perturbation Evaluation ───────────────────────────────────
#
# Starts the OpenPI WebSocket policy server, then runs the LIBERO client with
# image-level perturbations (orientation / position) to measure robustness.
#
# Quick-start examples
# --------------------
# Baseline (no perturbation):
#   sbatch scripts/run_pi05_visual_perturb.sh
#
# 30° rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_pi05_visual_perturb.sh
#
# 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 \
#       sbatch scripts/run_pi05_visual_perturb.sh
#
# Rotate then translate:
#   VISUAL_PERTURB_MODE=rotate_translate ROTATION_DEGREES=15 TRANSLATE_X_FRAC=0.1 \
#       sbatch scripts/run_pi05_visual_perturb.sh
#
# Run all four main suites:
#   TASK_SUITE=all sbatch scripts/run_pi05_visual_perturb.sh
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
PORT="${PORT:-8000}"
ENV_NAME="LIBERO"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"

# Visual perturbation
# mode: none | rotate | translate | rotate_translate
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# LIBERO settings
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"
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
elif [[ "$TASK_SUITE" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

# ── Server lifecycle helpers ──────────────────────────────────────────────────
# wait_for_server() {
#     echo "Waiting for policy server on port ${PORT}..."
#     for i in {1..60}; do
#         if nc -z localhost "${PORT}" 2>/dev/null; then
#             echo "Server is up (${i}s)."
#             return 0
#         fi
#         sleep 2
#     done
#     echo "ERROR: Policy server failed to start within 120s."
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
echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (${PERTURB_TAG})"
echo "  rotation_degrees:  ${ROTATION_DEGREES}"
echo "  translate_x_frac:  ${TRANSLATE_X_FRAC}"
echo "  translate_y_frac:  ${TRANSLATE_Y_FRAC}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "Replan steps:        ${REPLAN_STEPS}"
echo "Prompt mode:         ${PROMPT_MODE}"
echo "============================================================"

# ── Start policy server (shared across all suites) ───────────────────────────
# echo ""
# echo "Starting policy server..."
# SERVER_LOG="${WORKDIR}/logs/pi05_server_${SLURM_JOB_ID:-0}.log"
# python "${WORKDIR}/scripts/serve_policy.py" --env "${ENV_NAME}" \
#     > "${SERVER_LOG}" 2>&1 &
# SERVER_PID=$!
# wait_for_server

# ── Evaluate each suite ───────────────────────────────────────────────────────
for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/pi05/visual_perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/main.py" \
        --args.host "0.0.0.0" \
        --args.port "${PORT}" \
        --args.task-suite-name "${SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.seed "${SEED}" \
        --args.replan-steps "${REPLAN_STEPS}" \
        --args.resize-size "${RESIZE_SIZE}" \
        --args.prompt-mode "${PROMPT_MODE}" \
        --args.custom-prompt "${CUSTOM_PROMPT}" \
        --args.visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
        --args.rotation-degrees "${ROTATION_DEGREES}" \
        --args.translate-x-frac "${TRANSLATE_X_FRAC}" \
        --args.translate-y-frac "${TRANSLATE_Y_FRAC}" \
        --args.video-out-path "${VIDEO_OUT}" \
        2>&1 | tee "${WORKDIR}/logs/pi05_${SUITE}_${PERTURB_TAG}_${SLURM_JOB_ID:-0}.log"

    echo "Finished: ${SUITE}"
done

# ── Shutdown ──────────────────────────────────────────────────────────────────
# kill_server

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation: ${PERTURB_TAG}"
echo "============================================================"
