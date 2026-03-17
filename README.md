# vla-interp

Interpretability toolkit for Vision-Language-Action (VLA) models. Built on top of [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence), this repo provides tools for understanding the internal representations and attention patterns of the Pi0 flow-matching VLA model, evaluated on the [LIBERO](https://libero-project.github.io/) benchmark.

## Installation

```bash
conda create -n vla python=3.11
conda activate vla
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set up the LIBERO environment:

```bash
# export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# export LIBERO_CONFIG_PATH=$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:/n/netscratch/sham_lab/Lab/chloe00/libero
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

export NUMBA_CACHE_DIR="$TMPDIR/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

python examples/libero/main.py --args.port 8000
```

## Repository Structure

```
vla-interp/
├── analysis/                    # Analysis and visualization scripts
│   ├── attention/              # Attention visualization & IoU analysis
│   ├── entropy/                # Action entropy analysis
│   ├── probing/                # Linear probes, PCA, geometry analysis
│   └── utils/                  # Shared analysis utilities
│
├── experiments/                 # Shell scripts for running experiments
│   ├── run_attention_iou.sh
│   ├── run_action_entropy.sh
│   └── run_cosmos_eval.sh
│
├── results/                     # Experiment outputs and results
│   ├── entropy/                # Entropy analysis results
│   ├── attention/              # Attention visualizations
│   ├── iou/                    # IoU evaluation results
│   └── probes/                 # Probing results
│
├── docs/                        # Documentation and analysis reports
│   ├── ATTENTION_VIZ_README.md
│   ├── ENTROPY_ANALYSIS_README.md
│   └── entropy_analysis_report.md
│
├── examples/                    # Example evaluation scripts
│   ├── libero/                 # LIBERO benchmark evaluation
│   ├── aloha_sim/              # ALOHA simulation
│   ├── aloha_real/             # ALOHA real robot
│   └── droid/                  # DROID dataset
│
├── src/openpi/                  # Core openpi library
│   ├── models/                 # Pi0 model implementation
│   ├── policies/               # Policy inference & transforms
│   ├── training/               # Training utilities
│   └── serving/                # WebSocket policy server
│
├── scripts/                     # Training and serving scripts
│   ├── serve_policy.py         # Start policy server
│   └── train.py                # Training script
│
├── third_party/                 # External dependencies
│   ├── cosmos-policy/          # Cosmos policy implementation
│   └── aloha/                  # ALOHA utilities
│
├── checkpoints/                 # Model checkpoints
└── sae/                        # Sparse autoencoder utilities
```

## Generalization Experiments

Results tracking language, visual, and policy perturbation experiments across all LIBERO suites and all four models (pi0.5, OpenVLA, Cosmos, DP): [generalization.md](generalization.md)

## Analysis Tools

### Attention Visualization

Tools for visualizing visual, linguistic, and multimodal attention patterns in Pi0. See [docs/ATTENTION_VIZ_README.md](docs/ATTENTION_VIZ_README.md) for full documentation.

| Script | Description |
| --- | --- |
| `analysis/attention/example_attention_viz.py` | Visualize attention heatmaps for single frames or full episodes |
| `analysis/attention/visualize_attention.py` | Visual attention heatmap library |
| `analysis/attention/visualize_text_attention.py` | Linguistic attention analysis (per-token attention weights) |
| `analysis/attention/visualize_combined_attention.py` | Combined multimodal (vision + language) attention visualization |
| `analysis/attention/attention_iou.py` | IoU metrics between attention heatmaps and object segmentation masks |
| `analysis/attention/evaluate_attention_iou.py` | Evaluate attention-segmentation alignment across episodes |

### Sparse Autoencoders (SAEs)

Train sparse autoencoders on Pi0 activations to discover interpretable features, then mine concepts by analyzing what maximally activates each learned feature.

| Script | Description |
| --- | --- |
| `analysis/probing/train_save.py` | Train TopK / BatchTopK SAEs on Pi0 activations |
| `sae/top_activating_frames.py` | Find video frames that maximally activate each SAE feature |
| `sae/top_activating_actions.py` | Analyze action distributions associated with each SAE feature |
| `sae/top_activating_prompts.py` | Identify text prompts that maximally activate each SAE feature |
| `analysis/probing/sweep_sae_score_actions.py` | Evaluate SAE features for action prediction (MCC scoring) |

### Linear Probing and Representation Geometry

| Script | Description |
| --- | --- |
| `analysis/probing/linear_probe.py` | Ridge regression probes and INLP for measuring linear information about actions |
| `analysis/probing/geometry.py` | Activation geometry analysis (stepwise distances, temporal curvature across layers) |
| `analysis/probing/action_pca.py` | PCA visualization of the action space |

### Action Entropy Analysis

Analyze the entropy of predicted action distributions to understand model confidence and decision-making. See [docs/ENTROPY_ANALYSIS_README.md](docs/ENTROPY_ANALYSIS_README.md) for details.

| Script | Description |
| --- | --- |
| `analysis/entropy/compute_action_entropy.py` | Compute entropy of action distributions across episodes |
| `analysis/entropy/calculate_entropy_by_suite.py` | Calculate entropy statistics grouped by LIBERO task suite |
| `analysis/entropy/visualize_entropy_results.py` | Generate entropy visualization plots |
| `analysis/entropy/analyze_entropy_task_correlation.py` | Analyze correlation between entropy and task success |

To compute mean entropy for a data split (e.g. `90_b`, `90_d`), run each split separately since the suite name is inferred from the directory path. Each run automatically computes entropy for all episodes, success-only, and failure-only:

```bash
python analysis/entropy/compute_action_entropy.py \
    --data-dir data/libero/90_b/videos \
    --output results/entropy/action_entropy_90_b.json

python analysis/entropy/compute_action_entropy.py \
    --data-dir data/libero/90_d/videos \
    --output results/entropy/action_entropy_90_d.json
```

Results are nested by outcome (`all`, `success`, `failure`). To read mean task-level entropy:

```python
import json
for fname in ["results/entropy/action_entropy_90_b.json",
              "results/entropy/action_entropy_90_d.json"]:
    d = json.load(open(fname))
    for suite, suite_result in d.items():
        for outcome, res in suite_result.items():
            stats = res.get("task_entropy_stats", {})
            print(f"{fname}  [{outcome}]  mean: {stats.get('mean')}  std: {stats.get('std')}")
```

### Temporal Reliance Analysis

Compare instantaneous grounding (`x_t`) against temporally integrated grounding (`mean(x_{t-k:t})`) using per-step IoU or other analysis signals already saved in evaluation JSONs. The script pools episodes per model, computes rolling-window predictiveness for success, and produces summary plots plus a JSON report.

```bash
python analysis/temporal_reliance.py \
  --inputs \
    pi0.5:results/attention/iou/pi05/perturb/none/libero_10_seed7/iou_results_libero_10.json \
    DreamZero:data/libero/dreamzero/perturb/none/libero_10 \
  --feature iou \
  --rolling-windows 1 3 5 10 \
  --output-dir results/temporal_reliance
```

Useful options:

- `--feature iou` for the non-language version
- `--feature ratio` or `--feature visual_fraction` if you also want attention-ratio comparisons
- `--layer layer_25` to force a specific layer instead of auto-preferring `layers_avg`

Outputs in `--output-dir` include:

- `*_window_sweep.png`: correlation with success as a function of rolling window size
- `*_position_curve.png`: instantaneous vs rolling predictiveness over episode progress
- `*_trajectory_success_vs_failure.png`: normalized successful vs failed trajectory averages
- `*_dip_tolerance.png`: transient vs persistent low-grounding sensitivity
- `temporal_integration_score.png` and `temporal_reliance_results.json`

### Language Rerouting Analysis

Compare clean instructions (`prompt_mode=original`) against corrupted language conditions (`empty`, `random`, `synonym`, `shuffle`, `opposite`) and measure:

- `Δ visual ratio`
- `Δ IoU`
- `Δ success`
- `VCI = Δ visual ratio`
- `GRI = Δ ratio - λ|ΔIoU|`

The analyzer recursively scans model output directories, matches clean and corrupted episodes by `(suite, task_id, episode_id)`, and uses any discovered rollout JSONs and/or saved `attention_ratio_results_*.json` / `iou_results_*.json` files.

```bash
python analysis/attention/analyze_language_rerouting.py \
  --model-run pi0.5=/path/to/pi05_outputs \
  --model-run DreamZero=data/libero/dreamzero/perturb \
  --output-dir results/language_rerouting
```

Useful options:

- `--perturbations empty random synonym shuffle opposite` to restrict the comparison set
- `--gri-lambda 1.0` to control the penalty term in `GRI`
- `--skip-plots` to write CSV/Markdown summaries only

Outputs in `--output-dir` include:

- `discovered_episode_metrics.csv`: all discovered per-episode metrics before pairing
- `episode_pairs.csv`: matched clean vs corrupted episode-level deltas
- `summary_by_model_and_perturbation.csv`: grouped means/stds for each model and perturbation
- `summary.md`: short text summary
- `delta_visual_ratio.png`, `delta_iou.png`, `delta_success.png`
- `delta_ratio_vs_iou.png`
- `temporal_delta_ratio.png`, `temporal_delta_iou.png`

### LIBERO Evaluation

```bash
# Start the policy server
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=<checkpoint_dir>

# Run evaluation
python examples/libero/main.py --args.port 8000
```

### Cosmos Policy (Cluster Setup)

```bash
module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01
```

If `nccl.h` is missing, locate it:

```bash
python - <<'PY'
import glob
candidates = []
for root in ["/n/sw", "/usr", "/usr/local", "/opt", "/n"]:
    candidates += glob.glob(root + "/**/nccl.h", recursive=True)
print("\n".join(candidates[:50]))
PY
```

Force build transformer engine from source:

```bash
pip install -v --no-build-isolation --no-cache-dir "transformer_engine_torch==2.11.0"
```


flash_attn_2_cuda

ask for `-c 24` when requesting nodes

Recommended MAX_JOBS: 8 (larger will make the build slower)

`pip install --no-build-isolation --no-binary=:all: --no-cache-dir flash-attn==2.8.3 -v`






### Diffusion Policy



---

# openpi
 [Physical Intelligence team](https://www.physicalintelligence.company/).

- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

[ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Updates
- We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID. 


## Requirements

NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
