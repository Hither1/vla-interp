#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import jax
import jax.numpy as jnp

import openpi.models.gemma as gemma_mod

from openpi.models.pi0 import Pi0  
from openpi.models import pi0_config
import flax.nnx as nnx


def load_sae_decoder_vector(ckpt_path: str, concept_id: int) -> np.ndarray:
    """
    Robustly pulls a single decoder direction for concept_id from a TopKSAE checkpoint.
    This assumes the checkpoint has 'model_state_dict' and that the decoder weights exist.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Heuristic key search: look for a (nb_concepts, d) or (d, nb_concepts) weight matrix
    nb_concepts = ckpt.get("nb_concepts", None)
    d = ckpt.get("d", None)
    if nb_concepts is None or d is None:
        raise ValueError("SAE ckpt missing nb_concepts or d")

    candidates = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim != 2:
            continue
        sh = tuple(v.shape)
        if sh == (nb_concepts, d) or sh == (d, nb_concepts):
            candidates.append((k, v))

    if len(candidates) == 0:
        raise RuntimeError(
            f"Could not find a 2D weight with shape ({nb_concepts},{d}) or ({d},{nb_concepts}) in SAE state_dict."
        )

    # Prefer keys that look like decoder/dictionary weights
    def score_key(name: str) -> int:
        name_l = name.lower()
        score = 0
        if "decoder" in name_l or "dict" in name_l or "w_dec" in name_l:
            score += 10
        if "weight" in name_l:
            score += 2
        return score

    candidates.sort(key=lambda kv: score_key(kv[0]), reverse=True)
    key, W = candidates[0]

    W = W.detach().cpu()
    if W.shape == (d, nb_concepts):
        vec = W[:, concept_id]
    else:
        vec = W[concept_id, :]
    vec = vec.numpy().astype(np.float32)

    # normalize (optional but usually stabilizes)
    norm = np.linalg.norm(vec) + 1e-8
    vec = vec / norm
    print(f"[SAE] using decoder key='{key}', vec_norm(before_norm)≈{norm:.4f}")
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_ckpt", type=str, required=True)
    ap.add_argument("--concept", type=int, required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="scale for delta injection")
    ap.add_argument("--layer", type=int, default=11, help="Gemma block layer index to intervene on")
    ap.add_argument("--token_start", type=int, default=None, help="token slice start (allow negative)")
    ap.add_argument("--token_end", type=int, default=None, help="token slice end (allow negative)")
    ap.add_argument("--mode", type=str, default="add", choices=["add", "zero", "clamp"])
    ap.add_argument("--clamp_value", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_steps", type=int, default=10)

    # You need to construct an observation; provide whatever your codebase uses.
    ap.add_argument("--dummy", action="store_true", help="Run a dummy example; replace with real obs loading.")
    args = ap.parse_args()

    # 1) Build model
    cfg = pi0_config.Pi0Config()  # if needs args, fill in here
    rngs = nnx.Rngs(args.seed)
    model = Pi0(cfg, rngs=rngs)
    model.eval()

    # 2) Create / load an observation (YOU MUST REPLACE THIS)
    if args.dummy:
        obs = cfg.fake_obs()  # often exists in your config
    else:
        raise RuntimeError(
            "Please replace observation creation with your real env/data loader. "
            "E.g., load an episode frame and build openpi.models.model.Observation."
        )

    # 3) Baseline actions
    rng = jax.random.PRNGKey(args.seed)
    baseline = model.sample_actions(rng, obs, num_steps=args.num_steps)

    # 4) Prepare intervention delta from SAE concept decoder
    if args.mode == "add":
        vec = load_sae_decoder_vector(args.sae_ckpt, args.concept)  # (D,)
        delta = jnp.asarray(args.scale * vec)  # jnp (D,)
    else:
        delta = None

    # 5) Enable intervention in gemma
    gemma_mod.set_intervention(
        enabled=True,
        layer=args.layer,
        expert=1,               # action expert stream index
        mode=args.mode,
        token_start=args.token_start,
        token_end=args.token_end,
        delta=delta,
        clamp_value=args.clamp_value,
    )

    # 6) Intervened actions
    rng2 = jax.random.PRNGKey(args.seed + 1)
    intervened = model.sample_actions(rng2, obs, num_steps=args.num_steps)

    # 7) Disable intervention (important for future calls)
    gemma_mod.set_intervention(enabled=False, layer=-1)

    # 8) Report differences
    base_np = np.array(baseline)
    int_np = np.array(intervened)

    diff = int_np - base_np  # (B, H, action_dim)
    print("Mean |Δaction| per dim:", np.mean(np.abs(diff), axis=(0, 1)))
    print("Mean Δaction per dim:", np.mean(diff, axis=(0, 1)))


if __name__ == "__main__":
    main()