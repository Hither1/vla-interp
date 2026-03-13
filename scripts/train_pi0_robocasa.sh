#!/usr/bin/env bash
#SBATCH --job-name=pi0-robocasa
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi0_robocasa_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
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

cd /n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp

EXP_NAME="${1:-composite_seen_v1}"

CONFIG_NAME="${2:-pi0_fast_robocasa_target_composite_seen}"

echo "Starting pi0 RoboCasa training: config=${CONFIG_NAME} exp_name=${EXP_NAME}"

/n/home13/chloe00/miniforge3/envs/vla/bin/python scripts/train.py \
    "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}"
