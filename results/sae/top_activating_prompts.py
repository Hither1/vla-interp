#!/usr/bin/env python3
import os, glob, json, re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch

from overcomplete.sae import TopKSAE
from utils import *


nb_concepts = 512

def rank_prompts_for_concept(
    hits: List[Dict[str, Any]],
    prompt_top_k_frames: int = 10,
    prompt_score: str = "mean_topk",  # {"max", "mean_topk", "sum_topk"}
) -> List[Tuple[str, float, List[Dict[str, Any]]]]:
    """
    Returns a list of (prompt, prompt_score_value, prompt_hits_sorted),
    sorted by prompt_score_value desc.

    prompt_score options:
      - "max": max frame score for that prompt
      - "mean_topk": mean of top-k frame scores for that prompt
      - "sum_topk": sum of top-k frame scores for that prompt

    prompt_hits_sorted is sorted by frame score desc.
    """
    by_prompt = defaultdict(list)
    for h in hits:
        by_prompt[h["prompt"]].append(h)

    ranked: List[Tuple[str, float, List[Dict[str, Any]]]] = []
    for p, phits in by_prompt.items():
        phits.sort(key=lambda x: x["score"], reverse=True)
        top = phits[:prompt_top_k_frames]

        scores = np.array([x["score"] for x in top], dtype=np.float32)
        if scores.size == 0:
            continue

        if prompt_score == "max":
            s = float(scores.max())
        elif prompt_score == "sum_topk":
            s = float(scores.sum())
        elif prompt_score == "mean_topk":
            s = float(scores.mean())
        else:
            raise ValueError(f"Unknown prompt_score={prompt_score}")

        ranked.append((p, s, phits))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =========================
# 6) Concept mining (global hits, then top prompts per concept)
# =========================

def mine_concepts_global(
    ckpt_path: str,
    data_root: str,
    activations_root: str,
    out_dir: str,
    layer_idx: int,
    device: str = "cuda",
    encode_batch: int = 8192,

    # Frame-level cap for building prompt rankings (set large or None to be thorough)
    top_m_frames_per_concept: Optional[int] = 5000,

    # Prompt selection controls
    top_prompts_per_concept: int = 10,
    prompt_top_k_frames: int = 10,
    frames_to_save_per_prompt: int = 8,
    prompt_score: str = "mean_topk",  # {"max","mean_topk","sum_topk"}

    # ---- NEW (Option A): concept-level filtering ----
    # Only write concept outputs if total number of frame hits for that concept >= this threshold
    min_hits_per_concept: int = 0,
    # Optional: also require the concept's max score (top hit after sorting) >= this
    min_max_score_per_concept: float = 0.0,

    # If True, drop episodes that are missing video or activations
    strict: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- load SAE checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(
        ckpt["d"],
        nb_concepts=ckpt["nb_concepts"],
        top_k=ckpt["top_k"],
        device="cpu",
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval().to(device)

    nb_concepts = int(ckpt["nb_concepts"])
    d_expected = int(ckpt["d"])
    topk = min(int(ckpt["top_k"]), nb_concepts)
    print(f"Loaded SAE: nb_concepts={nb_concepts}, d={d_expected}, top_k={ckpt['top_k']}")

    episodes = index_libero_dataset(data_root=data_root, activations_root=activations_root)

    # ============================================================
    # PASS 1: load & align (actions, activations[layer]) across episodes
    # ============================================================
    cached = []
    total_T = 0
    dropped = 0

    for ep in episodes:
        if strict and (ep.act_path is None):
            dropped += 1
            continue

        acts = load_actions(ep.actions_path)  # (T_a, action_dim)
        A = np.load(ep.act_path).astype(np.float32)

        if A.ndim == 4:
            # your old comment: squeeze(-2) — keep same behavior
            A = A.squeeze(-2)

        if A.ndim != 3:
            print(f"[WARN] Unexpected activation shape {A.shape} in {ep.act_path}")
            dropped += 1
            continue

        T, num_layers, d = A.shape
        if d != d_expected:
            print(f"[WARN] d mismatch: got {d} in {ep.act_path}, expected {d_expected}")
            dropped += 1
            continue
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers-1}]")

        T_use = min(T, acts.shape[0])
        if T_use <= 0:
            dropped += 1
            continue

        A_layer = A[:T_use, layer_idx, :]  # (T_use, d)
        cached.append((ep, A_layer, acts[:T_use], T_use))
        total_T += T_use

    print(f"Usable episodes: {len(cached)} (dropped {dropped}). Total frames: {total_T}")

    if total_T == 0:
        raise RuntimeError("No usable frames found. Check indexing, activation paths, and strict mode.")

    # Allocate global arrays
    X_all = np.empty((total_T, d_expected), dtype=np.float32)

    # metadata aligned to global frame index i
    ep_group = [None] * total_T
    ep_id    = [None] * total_T
    t_in_ep  = np.empty((total_T,), dtype=np.int32)
    prompt   = [None] * total_T
    video    = [None] * total_T
    actions_all = None  # allocate once we know action_dim

    # Fill
    offset = 0
    for (ep, A_layer, acts_use, T_use) in cached:
        if actions_all is None:
            action_dim = acts_use.shape[1]
            actions_all = np.empty((total_T, action_dim), dtype=np.float32)

        X_all[offset:offset+T_use] = A_layer
        actions_all[offset:offset+T_use] = acts_use.astype(np.float32)

        ep_group[offset:offset+T_use] = [ep.group] * T_use
        ep_id[offset:offset+T_use]    = [ep.episode_id] * T_use
        t_in_ep[offset:offset+T_use]  = np.arange(T_use, dtype=np.int32)
        prompt[offset:offset+T_use]   = [ep.prompt] * T_use
        video[offset:offset+T_use]    = [ep.video_path] * T_use

        offset += T_use

    assert offset == total_T
    assert actions_all is not None

    # ============================================================
    # PASS 2: encode globally in batches and collect frame-level hits
    # ============================================================
    concept_hits: List[List[Dict[str, Any]]] = [[] for _ in range(nb_concepts)]

    with torch.no_grad():
        for start in range(0, total_T, encode_batch):
            end = min(total_T, start + encode_batch)
            x = torch.from_numpy(X_all[start:end]).to(device)  # (B, d)

            pre_codes, codes = sae.encode(x)  # codes: (B, nb_concepts)
            codes = codes.detach().float()
            scores = codes.abs()

            per_t_top = torch.topk(scores, k=topk, dim=1)
            top_vals = per_t_top.values.cpu().numpy()
            top_ids  = per_t_top.indices.cpu().numpy()

            for i_local in range(end - start):
                i_global = start + i_local
                for j in range(top_ids.shape[1]):
                    c = int(top_ids[i_local, j])
                    s = float(top_vals[i_local, j])
                    if s <= 0:
                        continue

                    concept_hits[c].append({
                        "score": s,
                        "group": ep_group[i_global],
                        "episode_id": ep_id[i_global],
                        "t": int(t_in_ep[i_global]),
                        "prompt": prompt[i_global],
                        "action": actions_all[i_global].copy(),
                        "video_path": video[i_global],
                    })

    # ============================================================
    # PASS 3: for each concept, rank prompts by activation, then save examples
    # ============================================================
    summaries: Dict[int, Any] = {}
    skipped_by_hit_threshold = 0
    skipped_by_score_threshold = 0

    for c in range(nb_concepts):
        hits = concept_hits[c]
        if len(hits) == 0:
            continue

        # Sort by frame score desc
        hits.sort(key=lambda x: x["score"], reverse=True)

        # ---- NEW (Option A): concept-level thresholding BEFORE writing anything ----
        if min_hits_per_concept > 0 and len(hits) < min_hits_per_concept:
            skipped_by_hit_threshold += 1
            continue

        if min_max_score_per_concept > 0.0 and hits[0]["score"] < min_max_score_per_concept:
            skipped_by_score_threshold += 1
            continue

        # Optionally cap frames considered for prompt ranking (for speed/memory)
        if top_m_frames_per_concept is not None and top_m_frames_per_concept > 0:
            hits_considered = hits[:top_m_frames_per_concept]
        else:
            hits_considered = hits

        ranked_prompts = rank_prompts_for_concept(
            hits_considered,
            prompt_top_k_frames=prompt_top_k_frames,
            prompt_score=prompt_score,
        )
        ranked_prompts = ranked_prompts[:top_prompts_per_concept]

        concept_dir = os.path.join(out_dir, f"nb_{nb_concepts}_concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

        prompts_txt_path = os.path.join(concept_dir, f"prompts_{c:04d}.txt")
        with open(prompts_txt_path, "w") as f:
            for p_rank, (p, p_score, phits_sorted) in enumerate(ranked_prompts):
                f.write(f"[{p_rank:02d}] score={p_score:.6f} hits={len(phits_sorted)}\n")
                f.write(p.strip() + "\n\n")

        # Basic distribution info
        prompt_counts_all = Counter([h["prompt"] for h in hits_considered])

        top_prompts_summary = []
        all_actions_for_summary = []

        for p_rank, (p, p_score, phits_sorted) in enumerate(ranked_prompts):
            prompt_dir = os.path.join(concept_dir, f"prompt_{p_rank:02d}")
            os.makedirs(prompt_dir, exist_ok=True)

            # Save the top frames for this prompt
            saved = 0
            top_examples = []

            # iterate more than needed in case some frames fail to read
            for h in phits_sorted[:max(frames_to_save_per_prompt * 3, frames_to_save_per_prompt)]:
                if h["video_path"] is None:
                    continue
                try:
                    rgb = get_frame_opencv(h["video_path"], h["t"])
                except Exception:
                    rgb = None
                if rgb is None:
                    continue

                png_path = os.path.join(
                    prompt_dir,
                    f"rank_{saved:02d}_score_{h['score']:.4f}_{h['episode_id']}_t{h['t']:05d}.png"
                )
                save_frame_png(rgb, png_path)

                top_examples.append({
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "video_path": h["video_path"],
                    "action": h["action"].tolist(),
                })
                all_actions_for_summary.append(h["action"])
                saved += 1
                if saved >= frames_to_save_per_prompt:
                    break

            top_prompts_summary.append({
                "prompt_rank": p_rank,
                "prompt_score": float(p_score),
                "prompt": p,
                "num_hits_for_prompt": len(phits_sorted),
                "num_hits_for_prompt_considered": int(prompt_counts_all[p]),
                "saved_examples": top_examples,
            })

        # Action stats over saved examples (top prompts only)
        if len(all_actions_for_summary) > 0:
            action_mat = np.stack(all_actions_for_summary, axis=0)
            action_mean = action_mat.mean(axis=0)
            action_median = np.median(action_mat, axis=0)
        else:
            action_mean = None
            action_median = None

        summary = {
            "concept": c,
            "num_frame_hits_total": len(hits),
            "num_frame_hits_considered": len(hits_considered),

            "prompt_score_method": prompt_score,
            "top_prompts_per_concept": top_prompts_per_concept,
            "prompt_top_k_frames": prompt_top_k_frames,
            "frames_to_save_per_prompt": frames_to_save_per_prompt,

            # NEW: record filtering thresholds used
            "min_hits_per_concept": int(min_hits_per_concept),
            "min_max_score_per_concept": float(min_max_score_per_concept),

            # quick view
            "top_prompts": [
                {
                    "prompt": x["prompt"],
                    "prompt_score": x["prompt_score"],
                    "num_hits_for_prompt": x["num_hits_for_prompt"],
                }
                for x in top_prompts_summary
            ],

            # detailed view
            "top_prompt_details": top_prompts_summary,

            "action_mean_over_saved": None if action_mean is None else action_mean.tolist(),
            "action_median_over_saved": None if action_median is None else action_median.tolist(),
        }

        with open(os.path.join(concept_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summaries[c] = summary

    with open(os.path.join(out_dir, "all_concepts_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Done. Wrote concept folders to: {out_dir}")
    if min_hits_per_concept > 0 or min_max_score_per_concept > 0.0:
        print(
            f"Filtering: skipped {skipped_by_hit_threshold} concepts by min_hits_per_concept={min_hits_per_concept}, "
            f"skipped {skipped_by_score_threshold} by min_max_score_per_concept={min_max_score_per_concept}"
        )


# =========================
# 7) Run
# =========================

if __name__ == "__main__":
    ckpt_path = f"./checkpoints/BatchTopKSAE/sae_libero_all_layer11_k16_c{nb_concepts}.pt"
    data_root = "/n/netscratch/sham_lab/Lab/chloe00/data/libero"
    activations_root = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"

    out_dir = "./concept_mining_out_prompts"
    layer_idx = 11

    mine_concepts_global(
        ckpt_path=ckpt_path,
        data_root=data_root,
        activations_root=activations_root,
        out_dir=out_dir,
        layer_idx=layer_idx,
        device="cuda",
        encode_batch=8192,

        # Tune these:
        top_m_frames_per_concept=500,   # increase (or None) for better prompt rankings
        top_prompts_per_concept=5,
        prompt_top_k_frames=10,
        frames_to_save_per_prompt=8,
        prompt_score="mean_topk",        # or "max" if you want “single strongest moment”

        # ---- NEW (Option A): concept-level filtering ----
        min_hits_per_concept=1000,        # example: only save concepts with >= 200 frame hits
        min_max_score_per_concept=0.0,   # optional: e.g. 0.5 to require strong max activation

        strict=True,
    )