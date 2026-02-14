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

## Analysis Tools

### Attention Visualization

Tools for visualizing visual, linguistic, and multimodal attention patterns in Pi0. See [ATTENTION_VIZ_README.md](ATTENTION_VIZ_README.md) for full documentation.

| Script | Description |
| --- | --- |
| `example_attention_viz.py` | Visualize attention heatmaps for single frames or full episodes |
| `visualize_attention.py` | Visual attention heatmap library |
| `visualize_text_attention.py` | Linguistic attention analysis (per-token attention weights) |
| `visualize_combined_attention.py` | Combined multimodal (vision + language) attention visualization |
| `attention_iou.py` | IoU metrics between attention heatmaps and object segmentation masks |
| `evaluate_attention_iou.py` | Evaluate attention-segmentation alignment across episodes |

### Sparse Autoencoders (SAEs)

Train sparse autoencoders on Pi0 activations to discover interpretable features, then mine concepts by analyzing what maximally activates each learned feature.

| Script | Description |
| --- | --- |
| `train_save.py` | Train TopK / BatchTopK SAEs on Pi0 activations |
| `sae/top_activating_frames.py` | Find video frames that maximally activate each SAE feature |
| `sae/top_activating_actions.py` | Analyze action distributions associated with each SAE feature |
| `sae/top_activating_prompts.py` | Identify text prompts that maximally activate each SAE feature |
| `sweep_sae_score_actions.py` | Evaluate SAE features for action prediction (MCC scoring) |

### Linear Probing and Representation Geometry

| Script | Description |
| --- | --- |
| `linear_probe.py` | Ridge regression probes and INLP for measuring linear information about actions |
| `geometry.py` | Activation geometry analysis (stepwise distances, temporal curvature across layers) |
| `action_pca.py` | PCA visualization of the action space |

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






