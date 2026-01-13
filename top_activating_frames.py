#!/usr/bin/env python3
import os
import glob
import json
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
import torch

from overcomplete.sae import TopKSAE, BatchTopKSAE
from utils import *



# =========================
# selecting top activating frames per concept (diversity + NMS)
# =========================

def select_top_frames_with_nms(
    hits,
    k: int,
    time_window: int = 5,
    max_per_episode: int = 1,
):
    """
    Select up to k hits sorted by score, enforcing:
      - at most max_per_episode per episode_id
      - temporal NMS within episode: do not select frames within +/- time_window

    hits: list of dicts with keys ["score", "episode_id", "t", ...]
    """
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)

    selected = []
    chosen_times = defaultdict(list)  # episode_id -> list of chosen t
    chosen_counts = Counter()

    for h in hits_sorted:
        if len(selected) >= k:
            break

        eid = h["episode_id"]
        t = int(h["t"])

        if chosen_counts[eid] >= max_per_episode:
            continue

        too_close = any(abs(t - tt) <= time_window for tt in chosen_times[eid])
        if too_close:
            continue

        selected.append(h)
        chosen_times[eid].append(t)
        chosen_counts[eid] += 1

    return selected

# =========================
# 6) Main mining
# =========================

def mine_concepts_global(
    ckpt_path: str,
    data_root: str,
    activations_root: str,
    out_dir: str,
    layer_idx: int,
    top_m: int = 200,              # pool size per concept (larger = more choices for NMS)
    per_concept_save_k: int = 16,  # number of frames to actually save per concept
    device: str = "cuda",
    encode_batch: int = 8192,      # global batch size over all frames
    nms_time_window: int = 5,
    max_per_episode: int = 1,
    skip_missing: bool = True,
    include_groups: tuple[str, ...] | None = ("10", "goal", "spatial", "object"),
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- load SAE checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = BatchTopKSAE(
        ckpt["d"],
        nb_concepts=ckpt["nb_concepts"],
        top_k=ckpt["top_k"],
        device="cpu",
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval().to(device)

    nb_concepts = ckpt["nb_concepts"]
    d_expected = ckpt["d"]
    topk = min(ckpt["top_k"], nb_concepts)
    print(f"Loaded SAE: nb_concepts={nb_concepts}, d={d_expected}, top_k={ckpt['top_k']}")

    episodes = index_libero_dataset(data_root=data_root, activations_root=activations_root)

    # ============================================================
    # PASS 1: cache usable (ep, A_layer, acts) and compute total_T
    # ============================================================
    cached = []
    total_T = 0


    if include_groups is not None:
        include_set = set(include_groups)
        before = len(episodes)
        episodes = [ep for ep in episodes if getattr(ep, "group", None) in include_set]
        after = len(episodes)
        print(f"Filtered episodes by group {sorted(include_set)}: {before} -> {after}")


    for ep in episodes:
        if ep.act_path is None or ep.video_path is None:
            if skip_missing:
                continue
            else:
                raise FileNotFoundError(f"Missing paths for ep={ep.episode_id}: video={ep.video_path}, act={ep.act_path}")

        acts = load_actions(ep.actions_path)  # (T_a, action_dim)
        A = np.load(ep.act_path).astype(np.float32)

        if A.ndim == 4:
            A = A.squeeze(-2)
        if A.ndim != 3:
            raise ValueError(f"Unexpected activation shape {A.shape} in {ep.act_path}")

        T, num_layers, d = A.shape
        if d != d_expected:
            raise ValueError(f"d mismatch: got {d} in {ep.act_path}, expected {d_expected}")
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers-1}]")

        T_use = min(T, acts.shape[0])
        if T_use <= 0:
            continue

        A_layer = A[:T_use, layer_idx, :]  # (T_use, d)
        cached.append((ep, A_layer, acts[:T_use], T_use))
        total_T += T_use

    print(f"Usable frames total_T={total_T} across {len(cached)} episodes.")
    if total_T == 0:
        raise RuntimeError("No usable frames found. Check paths / parsing.")

    # Allocate global arrays
    X_all = np.empty((total_T, d_expected), dtype=np.float32)

    ep_group = [None] * total_T
    ep_id    = [None] * total_T
    t_in_ep  = np.empty((total_T,), dtype=np.int32)
    prompt   = [None] * total_T
    video    = [None] * total_T
    actions_all = None  # allocated once we know action_dim

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

    # ============================================================
    # PASS 2: encode globally in batches and collect hits
    # ============================================================
    concept_hits = [[] for _ in range(nb_concepts)]

    sae.train()
    with torch.no_grad():
        for start in range(0, total_T, encode_batch):
            end = min(total_T, start + encode_batch)
            x = torch.from_numpy(X_all[start:end]).to(device)  # (B, d)

            _, codes = sae.encode(x)  # codes: (B, nb_concepts)
            scores = codes.detach().float().abs()

            # only consider per-frame top-k concepts (fast; matches your original logic)
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
    # PASS 3: per concept, pick top activating frames (with NMS), save
    # ============================================================
    summaries = {}

    for c in range(nb_concepts):
        hits = concept_hits[c]
        if len(hits) == 0:
            summaries[c] = {
                "concept": c,
                "num_hits_considered": 0,
                "top_prompts": [],
                "action_mean": None,
                "action_median": None,
                "top_examples": [],
            }
            continue

        hits.sort(key=lambda x: x["score"], reverse=True)
        pool = hits[:top_m]  # big pool; NMS will pick diverse top frames

        selected = select_top_frames_with_nms(
            pool,
            k=per_concept_save_k,
            time_window=nms_time_window,
            max_per_episode=max_per_episode,
        )

        prompt_counts = Counter([h["prompt"] for h in pool])

        action_mat = np.stack([h["action"] for h in pool], axis=0)
        action_mean = action_mat.mean(axis=0)
        action_median = np.median(action_mat, axis=0)

        concept_dir = os.path.join(out_dir, f"concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

        for rank, h in enumerate(selected):
            if h["video_path"] is None:
                continue
            rgb = get_frame_opencv(h["video_path"], h["t"])
            if rgb is None:
                continue
            png_path = os.path.join(
                concept_dir,
                f"rank_{rank:02d}_score_{h['score']:.4f}_{h['episode_id']}_t{h['t']:05d}.png"
            )
            save_frame_png(rgb, png_path)

        summary = {
            "concept": c,
            "num_hits_total": len(hits),
            "num_hits_pool": len(pool),
            "num_selected_saved": len(selected),
            "selection": {
                "top_m_pool": top_m,
                "per_concept_save_k": per_concept_save_k,
                "nms_time_window": nms_time_window,
                "max_per_episode": max_per_episode,
            },
            "top_prompts_in_pool": prompt_counts.most_common(10),
            "action_mean_in_pool": action_mean.tolist(),
            "action_median_in_pool": action_median.tolist(),
            "top_examples_saved": [
                {
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "prompt": h["prompt"],
                    "action": h["action"].tolist(),
                    "video_path": h["video_path"],
                }
                for h in selected[:10]
            ],
            "top_examples_by_score_in_pool": [
                {
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "prompt": h["prompt"],
                    "action": h["action"].tolist(),
                    "video_path": h["video_path"],
                }
                for h in pool[:10]
            ],
        }

        with open(os.path.join(concept_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summaries[c] = summary

    with open(os.path.join(out_dir, "all_concepts_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Done. Wrote concept folders to: {out_dir}")

# =========================
# 7) Run
# =========================

if __name__ == "__main__":
    # ckpt_path = "./checkpoints/TopKSAE/sae_layer11_k10_c16000.pt"
    ckpt_path = "./checkpoints/BatchTopKSAE/sae_libero_10_layer11_k16_c1024.pt"
    data_root = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero"
    activations_root = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"

    out_dir = "./concept_mining_out"
    layer_idx = 11

    mine_concepts_global(
        ckpt_path=ckpt_path,
        data_root=data_root,
        activations_root=activations_root,
        out_dir=out_dir,
        layer_idx=layer_idx,
        top_m=200,               # increase if you want more diversity options
        per_concept_save_k=16,
        device="cuda",
        encode_batch=8192,
        nms_time_window=5,       # try 0, 3, 5, 10
        max_per_episode=1,       # try 1 or 2
        skip_missing=True,
    )