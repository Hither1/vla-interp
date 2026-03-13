# Finetuning pi0-FAST on RoboCasa

This guide covers finetuning [pi0-FAST](https://www.physicalintelligence.company/blog/pi0) on the RoboCasa Composite-Seen dataset using local pre-converted LeRobot data.

## Dataset

**Composite-Seen** consists of 16 kitchen manipulation tasks x ~500 demos each, collected with a PandaOmron mobile manipulator.

Action space (12D):
- Dims 0–2: EEF position delta
- Dims 3–5: EEF orientation delta
- Dim 6: gripper
- Dims 7–11: mobile base

The pre-converted LeRobot dataset lives at:
```
/n/netscratch/sham_lab/Lab/chloe00/robocasa_data/composite_seen/lerobot_v30
```

### Downloading / converting from scratch

If the pre-converted data is unavailable, you can rebuild it:

**Option A — Download pre-converted tarballs (no robosuite required):**
```bash
python examples/robocasa/download_composite_seen.py \
    --download_dir /path/to/robocasa_data
```

**Option B — Convert from raw RoboCasa HDF5:**
```bash
python examples/robocasa/convert_robocasa_data_to_lerobot.py \
    --data_dir /path/to/robocasa/composite_seen \
    --repo_id your_username/robocasa_composite_seen
```

## Training config

The config `pi0_fast_robocasa_target_composite_seen` is defined in
[`src/openpi/training/config.py`](../src/openpi/training/config.py):

- **Model:** `Pi0FASTConfig(action_dim=12, action_horizon=10, max_token_len=180)`
- **Weights:** initialized from `gs://openpi-assets/checkpoints/pi0_fast_base/params`
- **Data:** `LeRobotRobocasaDataConfig` pointing to local `lerobot_v30` directory
- **Steps:** 50,000 (checkpoints every 5,000; kept every 10,000)
- **Batch size:** 32 (4x H100s)

## Step-by-step

### 1. Compute norm stats

Must be run before training. Saves stats to `./assets/pi0_fast_robocasa_target_composite_seen/robocasa_local/`.

```bash
cd /n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp
PYTHONPATH=src python scripts/compute_norm_stats.py \
    --config-name pi0_fast_robocasa_target_composite_seen
```

### 2. Submit training job

```bash
sbatch scripts/train_pi0_robocasa.sh <exp_name>
# e.g.:
sbatch scripts/train_pi0_robocasa.sh composite_seen_pi0fast_v1
```

The second argument selects the config (defaults to `pi0_fast_robocasa_target_composite_seen`):
```bash
# Explicit config:
sbatch scripts/train_pi0_robocasa.sh <exp_name> pi0_fast_robocasa_target_composite_seen

# Use the original pi0 (non-FAST) config instead:
sbatch scripts/train_pi0_robocasa.sh <exp_name> pi0_robocasa_target_composite_seen
```

Logs are written to:
```
/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi0_robocasa_<JOBID>.log
```

Checkpoints are written to:
```
./checkpoints/pi0_fast_robocasa_target_composite_seen/<exp_name>/
```

## SLURM job settings

See [`scripts/train_pi0_robocasa.sh`](../scripts/train_pi0_robocasa.sh):

| Setting | Value |
|---|---|
| Partition | `kempner_h100` |
| Account | `kempner_grads` |
| GPUs | 4x H100 |
| CPUs | 32 |
| Memory | 480 GB |
| Wall time | 48 hours |
