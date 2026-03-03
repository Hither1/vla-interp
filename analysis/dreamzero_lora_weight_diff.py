"""Analyze weight differences between finetuned DreamZero checkpoint-3400 and base Wan2.1.

For each LoRA-adapted layer, computes:
  delta_W = lora_B @ lora_A * (lora_alpha / lora_rank)
and then measures:
  - ||delta_W||_F (absolute change)
  - ||W_base||_F (base weight norm)
  - relative_change = ||delta_W||_F / ||W_base||_F

Also reports the newly trained weights (action_encoder, action_decoder, state_encoder)
which had no counterpart in the base model.

Usage:
    python analysis/dreamzero_lora_weight_diff.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

# ── Paths ─────────────────────────────────────────────────────────────────────
LORA_CKPT = (
    "/n/netscratch/sham_lab/Lab/chloe00/libero/"
    "dreamzero_libero_all_lora/DreamZero_libero/checkpoint-3400/model.safetensors"
)
BASE_MODEL_DIR = "/n/netscratch/sham_lab/Lab/chloe00/Wan2.1-I2V-14B-480P"

# LoRA hyper-parameters (from experiment_cfg/conf.yaml)
LORA_RANK  = 4
LORA_ALPHA = 4
LORA_SCALE = LORA_ALPHA / LORA_RANK  # = 1.0

# Module types that received LoRA
LORA_MODULES = [
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "self_attn.q",  "self_attn.k",  "self_attn.v",  "self_attn.o",
    "ffn.0",        "ffn.2",
]
NUM_LAYERS = 40  # DiT blocks 0..39


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_lora_weights(lora_path: str) -> dict:
    """Load all tensors from the LoRA safetensors into a dict."""
    weights = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def build_base_shard_map(base_dir: str) -> dict:
    """Return {key: shard_path} for all base model weights."""
    index_path = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
    index = json.load(open(index_path))
    shard_map = {}
    for key, shard_file in index["weight_map"].items():
        shard_map[key] = os.path.join(base_dir, shard_file)
    return shard_map


def load_base_tensor(key: str, shard_map: dict, shard_cache: dict) -> torch.Tensor:
    """Load a single tensor from the (cached) sharded base model."""
    shard_path = shard_map[key]
    if shard_path not in shard_cache:
        print(f"  Loading shard: {os.path.basename(shard_path)}")
        tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        shard_cache[shard_path] = tensors
    return shard_cache[shard_path][key]


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(output_dir: str):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LoRA checkpoint …")
    lora_weights = load_lora_weights(LORA_CKPT)

    print("Building base model shard map …")
    shard_map   = build_base_shard_map(BASE_MODEL_DIR)
    shard_cache = {}  # shard_path → {key: tensor}

    # ── 1. LoRA delta analysis ─────────────────────────────────────────────────
    # relative_change[layer][module] = ||delta_W||_F / ||W_base||_F
    rel_change   = np.zeros((NUM_LAYERS, len(LORA_MODULES)))
    abs_delta    = np.zeros((NUM_LAYERS, len(LORA_MODULES)))
    base_norms   = np.zeros((NUM_LAYERS, len(LORA_MODULES)))

    lora_prefix = "action_head.model.base_model.model."

    print(f"\nAnalyzing {NUM_LAYERS} blocks × {len(LORA_MODULES)} module types …")
    for li in range(NUM_LAYERS):
        for mi, mod in enumerate(LORA_MODULES):
            # Keys in the LoRA safetensors
            key_A = f"{lora_prefix}blocks.{li}.{mod}.lora_A.default.weight"
            key_B = f"{lora_prefix}blocks.{li}.{mod}.lora_B.default.weight"

            # Key in the base Wan2.1 shards
            base_key = f"blocks.{li}.{mod}.weight"

            A = lora_weights[key_A].float()   # (r, in)
            B = lora_weights[key_B].float()   # (out, r)

            delta_W = (B @ A) * LORA_SCALE    # (out, in)

            W_base = load_base_tensor(base_key, shard_map, shard_cache).float()

            d_norm = delta_W.norm().item()
            w_norm = W_base.norm().item()

            abs_delta[li, mi]  = d_norm
            base_norms[li, mi] = w_norm
            rel_change[li, mi] = d_norm / (w_norm + 1e-12)

        if (li + 1) % 10 == 0:
            print(f"  Done layer {li + 1}/{NUM_LAYERS}")

    # ── 2. Newly trained action/state head weights ─────────────────────────────
    head_keys = [k for k in lora_weights if "lora" not in k]
    head_stats = {}
    for k in head_keys:
        t = lora_weights[k].float()
        head_stats[k] = {
            "shape":    list(t.shape),
            "norm":     t.norm().item(),
            "num_params": t.numel(),
        }

    # ── 3. Summary statistics ──────────────────────────────────────────────────
    print("\n── Layer-averaged relative change (||ΔW||/||W||) ──")
    layer_avg = rel_change.mean(axis=1)
    for li, v in enumerate(layer_avg):
        bar = "█" * int(v * 400)
        print(f"  Block {li:2d}: {v:.4f}  {bar}")

    print("\n── Module-type-averaged relative change ──")
    mod_avg = rel_change.mean(axis=0)
    for mi, mod in enumerate(LORA_MODULES):
        bar = "█" * int(mod_avg[mi] * 400)
        print(f"  {mod:20s}: {mod_avg[mi]:.4f}  {bar}")

    print("\n── Action/State head weights (newly trained, no base comparison) ──")
    for k, s in head_stats.items():
        short = k.replace("action_head.model.base_model.model.", "")
        print(f"  {short:40s}  shape={s['shape']}  norm={s['norm']:.4f}  params={s['num_params']}")

    # ── 4. Save numerical results ──────────────────────────────────────────────
    results = {
        "relative_change": rel_change.tolist(),   # [40, 10]
        "abs_delta_norm":  abs_delta.tolist(),
        "base_norms":      base_norms.tolist(),
        "modules":         LORA_MODULES,
        "num_layers":      NUM_LAYERS,
        "lora_scale":      LORA_SCALE,
        "head_stats":      head_stats,
    }
    out_json = output_dir / "lora_weight_diff.json"
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\nNumerical results → {out_json}")

    # ── 5. Heatmap: relative change per layer × module ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))

    # 5a. Relative change heatmap
    ax = axes[0]
    im = ax.imshow(rel_change, aspect="auto", cmap="hot", origin="upper")
    ax.set_xticks(range(len(LORA_MODULES)))
    ax.set_xticklabels(LORA_MODULES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, NUM_LAYERS, 5))
    ax.set_yticklabels(range(0, NUM_LAYERS, 5))
    ax.set_xlabel("Module type")
    ax.set_ylabel("DiT Block")
    ax.set_title("Relative change\n||ΔW||_F / ||W_base||_F")
    plt.colorbar(im, ax=ax)

    # 5b. Absolute delta norm heatmap
    ax = axes[1]
    im = ax.imshow(abs_delta, aspect="auto", cmap="Blues", origin="upper")
    ax.set_xticks(range(len(LORA_MODULES)))
    ax.set_xticklabels(LORA_MODULES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, NUM_LAYERS, 5))
    ax.set_yticklabels(range(0, NUM_LAYERS, 5))
    ax.set_xlabel("Module type")
    ax.set_ylabel("DiT Block")
    ax.set_title("Absolute delta norm\n||ΔW||_F")
    plt.colorbar(im, ax=ax)

    # 5c. Layer-averaged relative change bar chart
    ax = axes[2]
    ax.barh(range(NUM_LAYERS), layer_avg, color="steelblue")
    ax.set_yticks(range(0, NUM_LAYERS, 5))
    ax.set_yticklabels(range(0, NUM_LAYERS, 5))
    ax.invert_yaxis()
    ax.set_xlabel("Mean relative change ||ΔW||_F / ||W||_F")
    ax.set_ylabel("DiT Block")
    ax.set_title("Layer-averaged relative\nweight change")
    ax.axvline(layer_avg.mean(), color="red", linestyle="--", label=f"mean={layer_avg.mean():.4f}")
    ax.legend(fontsize=8)

    plt.suptitle(
        "DreamZero: checkpoint-3400 vs. Wan2.1-I2V-14B-480P base\n"
        f"LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, scale={LORA_SCALE:.1f}",
        fontsize=12,
    )
    plt.tight_layout()
    out_fig = output_dir / "lora_weight_diff_heatmap.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap → {out_fig}")

    # ── 6. Per-module bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LORA_MODULES))
    ax.bar(x, mod_avg, color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(LORA_MODULES, rotation=30, ha="right")
    ax.set_ylabel("Mean relative change ||ΔW||_F / ||W||_F")
    ax.set_title("Module-type-averaged relative weight change\n(checkpoint-3400 vs. Wan2.1 base)")
    ax.axhline(mod_avg.mean(), color="red", linestyle="--", label=f"mean={mod_avg.mean():.4f}")
    ax.legend()
    plt.tight_layout()
    out_mod = output_dir / "lora_weight_diff_by_module.png"
    plt.savefig(out_mod, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Module bar chart → {out_mod}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/dreamzero_analysis/lora_weight_diff",
        help="Directory to write results and figures",
    )
    args = parser.parse_args()
    analyze(args.output_dir)
