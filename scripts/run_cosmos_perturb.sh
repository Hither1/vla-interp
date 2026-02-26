#!/usr/bin/env bash
#SBATCH --job-name=cosmos-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/cosmos_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=16:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Cosmos Policy — Perturbation Evaluation ──────────────────────────────────
#
# Evaluates Cosmos-Policy-LIBERO under any combination of language, visual,
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
#   sbatch scripts/run_cosmos_perturb.sh
#
# Prompt — empty:
#   PROMPT_MODE=empty sbatch scripts/run_cosmos_perturb.sh
#
# Prompt — opposite, all suites:
#   PROMPT_MODE=opposite TASK_SUITE=all sbatch scripts/run_cosmos_perturb.sh
#
# Visual — 30° rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_cosmos_perturb.sh
#
# Visual — 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 \
#       sbatch scripts/run_cosmos_perturb.sh
#
# Policy — 25% random action:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_cosmos_perturb.sh
#
# Policy — object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_cosmos_perturb.sh
#
# Combined (shuffle prompt + rotate image):
#   PROMPT_MODE=shuffle VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=15 \
#       sbatch scripts/run_cosmos_perturb.sh
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
SEED="${SEED:-195}"

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

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               Cosmos Policy"
echo "Checkpoint:          ${CKPT_PATH}"
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
echo "Chunk size:          ${CHUNK_SIZE}"
echo "Open-loop steps:     ${NUM_OPEN_LOOP_STEPS}"
echo "Denoising steps:     ${NUM_DENOISING_STEPS_ACTION}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/cosmos/perturb/${PERTURB_TAG}/${SUITE}"
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
echo "All suites complete.  Perturbation tag: ${PERTURB_TAG}"
echo "============================================================"
