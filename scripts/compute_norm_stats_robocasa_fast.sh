#!/usr/bin/env bash
#SBATCH --job-name=norm-stats-robocasa-fast
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/norm_stats_robocasa_fast_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=15:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

set -e

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/src:$PYTHONPATH

cd /n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp

echo "Computing norm stats for pi0_fast_robocasa_target_composite_seen"

/n/home13/chloe00/miniforge3/envs/vla/bin/python scripts/compute_norm_stats.py \
    --config-name pi0_fast_robocasa_target_composite_seen
