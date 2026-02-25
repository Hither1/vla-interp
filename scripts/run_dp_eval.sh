#!/usr/bin/env bash
#SBATCH --job-name=eval-dp
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dp_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

# ── Diffusion Policy — Prompt Perturbation Evaluation ────────────────────────
#
# Single mode (one SLURM job):
#   PROMPT_MODE=opposite sbatch scripts/run_dp_eval.sh
#
# Full sweep (submits one job per mode, then exits):
#   PROMPT_MODE=all bash scripts/run_dp_eval.sh
#   PROMPT_MODE=all TASK_SUITE=90_all bash scripts/run_dp_eval.sh
#
# Prompt modes: original | empty | shuffle | random | synonym | opposite | all
# Task suites:  libero_spatial | libero_object | libero_goal | libero_10
#               | libero_90 | all | 90_all
#
# Outputs (one directory per mode):
#   data/libero/dp/prompt_<mode>/<suite>/
# ─────────────────────────────────────────────────────────────────────────────

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

# ── Configuration ─────────────────────────────────────────────────────────────

# Checkpoint path (absolute so it works regardless of cwd at submission time).
# Override: CKPT=/path/to/ckpt.pt sbatch scripts/run_dp_eval.sh
CKPT="${CKPT:-/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/dp_scratch/ckpt_300.pt}"

# Prompt perturbation mode.
# Options: original, empty, shuffle, random, synonym, opposite
# Use "all" to submit a separate SLURM job for every mode.
PROMPT_MODE="${PROMPT_MODE:-original}"

# Custom prompt text (only used when PROMPT_MODE=custom)
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# Task suite to evaluate.
# Options: libero_spatial, libero_object, libero_goal, libero_10, all, 90_all
TASK_SUITE="${TASK_SUITE:-libero_10}"

# Number of trials per task
NUM_TRIALS="${NUM_TRIALS:-20}"

# Seed
SEED="${SEED:-0}"

# Replan steps (actions executed before replanning)
REPLAN_STEPS="${REPLAN_STEPS:-8}"

WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"

# ── Sweep mode: re-submit one job per prompt mode then exit ───────────────────
if [[ "${PROMPT_MODE}" == "all" ]]; then
    ALL_MODES=(original empty shuffle random synonym opposite)
    echo "========================================================"
    echo "Submitting DP prompt-perturbation sweep"
    echo "  Prompt modes : ${ALL_MODES[*]}"
    echo "  Task suite   : ${TASK_SUITE}"
    echo "  Checkpoint   : ${CKPT}"
    echo "  Trials/task  : ${NUM_TRIALS}"
    echo "  Seed         : ${SEED}"
    echo "  Replan steps : ${REPLAN_STEPS}"
    echo "========================================================"
    for MODE in "${ALL_MODES[@]}"; do
        JOB_ID=$(sbatch \
            --export=ALL,PROMPT_MODE="${MODE}",TASK_SUITE="${TASK_SUITE}",CKPT="${CKPT}",NUM_TRIALS="${NUM_TRIALS}",SEED="${SEED}",REPLAN_STEPS="${REPLAN_STEPS}" \
            "${WORKDIR}/scripts/run_dp_eval.sh" \
            | awk '{print $NF}')
        echo "  Submitted PROMPT_MODE=${MODE}  →  job ${JOB_ID}"
    done
    echo "========================================================"
    echo "All jobs submitted.  Monitor with: squeue -u \$USER"
    echo "========================================================"
    exit 0
fi

# ── Single-mode evaluation ────────────────────────────────────────────────────

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

export MUJOCO_GL=egl

if [[ "${TASK_SUITE}" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [[ "${TASK_SUITE}" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Model:           Diffusion Policy"
echo "Checkpoint:      ${CKPT}"
echo "Prompt mode:     ${PROMPT_MODE}"
echo "Task suites:     ${SUITES[*]}"
echo "Trials per task: ${NUM_TRIALS}"
echo "Seed:            ${SEED}"
echo "Replan steps:    ${REPLAN_STEPS}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo "========================================"
    echo "Evaluating suite: ${SUITE}  prompt_mode: ${PROMPT_MODE}"
    echo "========================================"

    python "${WORKDIR}/examples/libero/dp_eval.py" \
        --ckpt "${CKPT}" \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
        --prompt-mode "${PROMPT_MODE}" \
        --custom-prompt "${CUSTOM_PROMPT}" \
        --video-out-path "${WORKDIR}/data/libero/dp/prompt_${PROMPT_MODE}/${SUITE}"
done
