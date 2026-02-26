#!/usr/bin/env bash
#SBATCH --job-name=pi05-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi05_perturb_%j.log
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

# ── Pi0.5 — Perturbation Evaluation ─────────────────────────────────────────
#
# Runs the LIBERO client (main.py) under any combination of language, visual,
# and policy perturbations.  All three types can be set independently; the
# output path encodes every active perturbation.
#
# Prompt modes (PROMPT_MODE)
# --------------------------
#   original  (default) use ground-truth task description
#   empty               pass an empty string
#   shuffle             randomly shuffle instruction words
#   random              replace with a random other task's instruction
#   synonym             substitute key verbs with synonyms
#   opposite            substitute key phrases with antonyms
#   custom              use CUSTOM_PROMPT verbatim
#
# Visual modes (VISUAL_PERTURB_MODE)
# ------------------------------------
#   none          (default) no perturbation
#   rotate        CCW rotation by ROTATION_DEGREES
#   translate     shift by TRANSLATE_{X,Y}_FRAC * image size
#   rotate_translate  rotation then translation
#
# Policy modes (POLICY_PERTURB_MODE)
# ------------------------------------
#   none          (default) no perturbation
#   random_action replace action with probability RANDOM_ACTION_PROB
#   object_shift  displace objects at episode start by OBJECT_SHIFT_{X,Y}_STD
#
# Quick-start examples
# --------------------
# Baseline:
#   sbatch scripts/run_pi05_perturb.sh
#
# Prompt — shuffled words:
#   PROMPT_MODE=shuffle sbatch scripts/run_pi05_perturb.sh
#
# Visual — 30° rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_pi05_perturb.sh
#
# Visual — 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 \
#       sbatch scripts/run_pi05_perturb.sh
#
# Policy — 25% random action:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_pi05_perturb.sh
#
# Policy — object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_pi05_perturb.sh
#
# Combined (shuffle prompt + rotate image), all suites:
#   PROMPT_MODE=shuffle VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=15 \
#   TASK_SUITE=all sbatch scripts/run_pi05_perturb.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export MUJOCO_GL=egl

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Pi0.5 server config ───────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ENV_NAME="LIBERO"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"

# ── Prompt perturbation ───────────────────────────────────────────────────────
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Visual perturbation ───────────────────────────────────────────────────────
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# ── Policy perturbation ───────────────────────────────────────────────────────
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

# ── LIBERO settings ───────────────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"

# ── Derived output tag (one segment per active perturbation) ──────────────────
PROMPT_TAG=""
if [[ "$PROMPT_MODE" != "original" ]]; then
    if [[ "$PROMPT_MODE" == "custom" && -n "$CUSTOM_PROMPT" ]]; then
        SLUG="${CUSTOM_PROMPT:0:30}"
        SLUG="${SLUG// /_}"
        PROMPT_TAG="prompt_custom_${SLUG}"
    else
        PROMPT_TAG="prompt_${PROMPT_MODE}"
    fi
fi

VIS_TAG=""
if [[ "$VISUAL_PERTURB_MODE" != "none" ]]; then
    if [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
        VIS_TAG="vis_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
fi

POL_TAG=""
if [[ "$POLICY_PERTURB_MODE" != "none" ]]; then
    if [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
        POL_TAG="pol_random_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
        POL_TAG="pol_objshift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else
        POL_TAG="pol_${POLICY_PERTURB_MODE}"
    fi
fi

PERTURB_TAG=""
for _tag in "$PROMPT_TAG" "$VIS_TAG" "$POL_TAG"; do
    if [[ -n "$_tag" ]]; then
        PERTURB_TAG="${PERTURB_TAG:+${PERTURB_TAG}__}${_tag}"
    fi
done
PERTURB_TAG="${PERTURB_TAG:-none}"

# ── Task suite list ───────────────────────────────────────────────────────────
if [[ "$TASK_SUITE" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [[ "$TASK_SUITE" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

# ── Server lifecycle helpers (uncomment when running with a live server) ──────
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
#
# kill_server() {
#     if [[ -n "${SERVER_PID:-}" ]]; then
#         echo "Stopping policy server (PID=${SERVER_PID})..."
#         kill "${SERVER_PID}" 2>/dev/null || true
#         wait "${SERVER_PID}" 2>/dev/null || true
#         unset SERVER_PID
#     fi
# }
#
# trap kill_server EXIT

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Pi0.5"
echo "Policy server:       ${HOST}:${PORT}"
echo "Prompt perturbation: ${PROMPT_MODE}"
[[ "$PROMPT_MODE" == "custom" ]] && echo "  custom_prompt:       '${CUSTOM_PROMPT}'"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  rotation_degrees:    ${ROTATION_DEGREES}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  translate_x_frac:    ${TRANSLATE_X_FRAC}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  translate_y_frac:    ${TRANSLATE_Y_FRAC}"
echo "Policy perturbation: ${POLICY_PERTURB_MODE}"
[[ "$POLICY_PERTURB_MODE" == "random_action" ]] && echo "  random_action_prob:  ${RANDOM_ACTION_PROB}"
[[ "$POLICY_PERTURB_MODE" == "random_action" ]] && echo "  random_action_scale: ${RANDOM_ACTION_SCALE}"
[[ "$POLICY_PERTURB_MODE" == "object_shift"  ]] && echo "  object_shift_x_std:  ${OBJECT_SHIFT_X_STD}"
[[ "$POLICY_PERTURB_MODE" == "object_shift"  ]] && echo "  object_shift_y_std:  ${OBJECT_SHIFT_Y_STD}"
echo "Perturbation tag:    ${PERTURB_TAG}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "Replan steps:        ${REPLAN_STEPS}"
echo "============================================================"

# ── Start policy server (uncomment when running with a live server) ───────────
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

    VIDEO_OUT="${WORKDIR}/data/libero/pi05/perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    python "${WORKDIR}/examples/libero/main.py" \
        --args.host "${HOST}" \
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
echo "All suites complete.  Perturbation tag: ${PERTURB_TAG}"
echo "============================================================"
