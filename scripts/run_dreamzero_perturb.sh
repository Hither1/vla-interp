#!/usr/bin/env bash
#SBATCH --job-name=dreamzero-libero
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dreamzero_libero_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=24:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── DreamZero (finetuned on LIBERO) — Evaluation on all LIBERO suites ─────────
#
# Runs dreamzero_eval.py with torchrun across all LIBERO suites.
# The 14B DreamZero model requires multi-GPU tensor parallelism; use
# NUM_GPUS=2 minimum (4 recommended for speed).
#
# Quick-start
# -----------
#   # Baseline, all 4 standard suites:
#   CKPT=/path/to/dreamzero_libero_lora TASK_SUITE=all \
#       sbatch scripts/run_dreamzero_perturb.sh
#
#   # Single suite:
#   CKPT=/path/to/dreamzero_libero_lora TASK_SUITE=libero_10 \
#       sbatch scripts/run_dreamzero_perturb.sh
#
# Prompt modes (PROMPT_MODE): original | empty | shuffle | random | synonym | opposite | custom
# Visual modes (VISUAL_PERTURB_MODE): none | rotate | translate | rotate_translate
# Policy modes (POLICY_PERTURB_MODE): none | random_action | object_shift
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate
# Activate the conda environment that has DreamZero + LIBERO + PyTorch installed
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
DREAMZERO_DIR="${WORKDIR}/dreamzero"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${DREAMZERO_DIR}:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Model ──────────────────────────────────────────────────────────────────────
CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
    echo "ERROR: CKPT is required.  Example:"
    echo "  CKPT=/path/to/dreamzero_libero_lora sbatch $0"
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-4}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-true}"

# ── Prompt perturbation ────────────────────────────────────────────────────────
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Visual perturbation ────────────────────────────────────────────────────────
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# ── Policy perturbation ────────────────────────────────────────────────────────
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

# ── LIBERO settings ────────────────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-4}"   # max_chunk_size from libero_training.sh

# ── Derived output tag ─────────────────────────────────────────────────────────
PROMPT_TAG=""
if [[ "$PROMPT_MODE" != "original" ]]; then
    if [[ "$PROMPT_MODE" == "custom" && -n "$CUSTOM_PROMPT" ]]; then
        SLUG="${CUSTOM_PROMPT:0:30}"; SLUG="${SLUG// /_}"
        PROMPT_TAG="prompt_custom_${SLUG}"
    else
        PROMPT_TAG="prompt_${PROMPT_MODE}"
    fi
fi

VIS_TAG=""
if [[ "$VISUAL_PERTURB_MODE" != "none" ]]; then
    if   [[ "$VISUAL_PERTURB_MODE" == "rotate" ]];    then VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then VIS_TAG="vis_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
fi

POL_TAG=""
if [[ "$POLICY_PERTURB_MODE" != "none" ]]; then
    if   [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then POL_TAG="pol_random_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]];  then POL_TAG="pol_objshift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else POL_TAG="pol_${POLICY_PERTURB_MODE}"
    fi
fi

PERTURB_TAG=""
for _tag in "$PROMPT_TAG" "$VIS_TAG" "$POL_TAG"; do
    [[ -n "$_tag" ]] && PERTURB_TAG="${PERTURB_TAG:+${PERTURB_TAG}__}${_tag}"
done
PERTURB_TAG="${PERTURB_TAG:-none}"

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
echo "Model:               DreamZero (LIBERO finetune)"
echo "Checkpoint:          ${CKPT}"
echo "Num GPUs:            ${NUM_GPUS}"
echo "Prompt perturbation: ${PROMPT_MODE}"
[[ "$PROMPT_MODE" == "custom" ]] && echo "  custom_prompt:       '${CUSTOM_PROMPT}'"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE}"
echo "Policy perturbation: ${POLICY_PERTURB_MODE}"
echo "Perturbation tag:    ${PERTURB_TAG}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Replan steps:        ${REPLAN_STEPS}"
echo "Seed:                ${SEED}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/dreamzero/perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    ENABLE_DIT_CACHE_ARG=""
    [[ "$ENABLE_DIT_CACHE" == "false" ]] && ENABLE_DIT_CACHE_ARG="--no-enable-dit-cache"

    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        "${WORKDIR}/examples/libero/dreamzero_eval.py" \
            --model-path "${CKPT}" \
            --task-suite-name "${SUITE}" \
            --num-trials-per-task "${NUM_TRIALS}" \
            --replan-steps "${REPLAN_STEPS}" \
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
            --video-out-path "${VIDEO_OUT}" \
            ${ENABLE_DIT_CACHE_ARG}

    echo "Finished: ${SUITE}"
done

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation tag: ${PERTURB_TAG}"
echo "============================================================"
