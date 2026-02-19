#!/usr/bin/env bash
#SBATCH --job-name=eval-dp
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dp_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=08:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

# Checkpoint path.
# Override at submission time: sbatch --export=ALL,CKPT=checkpoints/dp/ckpt_300.pt run_dp_eval.sh
CKPT="${CKPT:-dp_scratch/ckpt_300.pt}"

# Prompt perturbation mode.
# Options: original, empty, shuffle, random, synonym, opposite
PROMPT_MODE="${PROMPT_MODE:-original}"

# Custom prompt text (only used when PROMPT_MODE=custom)
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# Task suite to evaluate.
# Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all
TASK_SUITE="${TASK_SUITE:-libero_10}"

# Number of trials per task (default 20)
NUM_TRIALS="${NUM_TRIALS:-20}"

# Seed
SEED="${SEED:-0}"

# Replan steps (actions executed before replanning)
REPLAN_STEPS="${REPLAN_STEPS:-8}"

WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

if [ "${TASK_SUITE}" = "all" ]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
else
    SUITES=("${TASK_SUITE}")
fi

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
        --video-out-path "${WORKDIR}/data/libero/dp/videos/${SUITE}"
done
