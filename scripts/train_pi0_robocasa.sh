#!/usr/bin/env bash
#SBATCH --job-name=pi0-robocasa
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi0_robocasa_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=480G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=48:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

set -e

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_OFFLINE=0
export PYTHONPATH=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/src:$PYTHONPATH
# Prevent accidental float64 tensors, which can double memory usage.
export JAX_ENABLE_X64=0
# JAX defaults to preallocating ~75% of GPU memory, which can OOM large init graphs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Allow JAX to use most of the GPU memory budget during large initialization steps.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

cd /n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp

EXP_NAME="${1:-composite_seen_v1}"

CONFIG_NAME="${2:-pi0_fast_robocasa_target_composite_seen}"

# Use both allocated GPUs for FSDP sharding to reduce per-GPU model/optimizer memory.
FSDP_DEVICES="${3:-2}"

# Disable EMA by default to avoid storing a second full copy of model params at init.
EMA_DECAY="${4:-None}"

# Lower default global batch size to reduce first-step activation/gradient memory.
BATCH_SIZE="${5:-16}"

echo "Starting pi0 RoboCasa training: config=${CONFIG_NAME} exp_name=${EXP_NAME} fsdp_devices=${FSDP_DEVICES} ema_decay=${EMA_DECAY} batch_size=${BATCH_SIZE}"

/n/home13/chloe00/miniforge3/envs/vla/bin/python scripts/train.py \
    "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --fsdp-devices "${FSDP_DEVICES}" \
    --ema-decay "${EMA_DECAY}" \
    --batch-size "${BATCH_SIZE}" \
    --overwrite
