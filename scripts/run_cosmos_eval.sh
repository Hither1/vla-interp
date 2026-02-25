#!/usr/bin/env bash
#SBATCH --job-name=eval-cosmos
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/action_entropy_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=16:30:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

# Prompt perturbation mode.
# Options: original, empty, shuffle, random, synonym, opposite, custom
# Override at submission time: sbatch --export=ALL,PROMPT_MODE=opposite run_cosmos_eval.sh
# Or set inline:              PROMPT_MODE=synonym sbatch run_cosmos_eval.sh
PROMPT_MODE="${PROMPT_MODE:-original}"

# Custom prompt text (only used when PROMPT_MODE=custom)
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# Task suite to evaluate.
# Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all
# "all" runs all five suites sequentially in a single job.
TASK_SUITE="${TASK_SUITE:-libero_10}"

# Ensure cosmos_policy package is importable
export PYTHONPATH="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/third_party/cosmos-policy${PYTHONPATH:+:${PYTHONPATH}}"


export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

export PYTHONPATH="${PYTHONPATH:-}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero


if [ "${TASK_SUITE}" = "all" ]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [[ "$TASK_SUITE" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

for SUITE in "${SUITES[@]}"; do
    echo "========================================"
    echo "Evaluating suite: ${SUITE}  prompt_mode: ${PROMPT_MODE}"
    echo "========================================"

    RUN_ID_NOTE="chkpt45000--5stepAct--seed195--deterministic--prompt_${PROMPT_MODE}--suite_${SUITE}"

    python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name "${SUITE}" \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note "${RUN_ID_NOTE}" \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --prompt_mode "${PROMPT_MODE}" \
        --custom_prompt "${CUSTOM_PROMPT}"
done