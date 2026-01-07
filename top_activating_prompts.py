#!/usr/bin/env python3
import os, glob, json, re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch

from overcomplete.sae import TopKSAE  
from utils import *


# =========================
# 1) Customize these parsers
# =========================

def prompt_for_group_and_episode(group_name: str, episode_id: str) -> str:
    """
    If your episode id encodes which task index it is, parse it here.
    Otherwise, fall back to 'unknown' or store the group only.

    Common Libero setup: episodes are grouped by task (10 tasks per group).
    If you have metadata elsewhere, swap this to use that.
    """
    # import pdb; pdb.set_trace()
    m = re.search(r"task(\d+)", episode_id)
    if m:
        idx = int(m.group(1))
        key = f"libero_{group_name}" if not group_name.startswith("libero_") else group_name
        if key in libero_task_map and 0 <= idx < len(libero_task_map[key]):
            return libero_task_map[key][idx]
    return f"{group_name}:{episode_id}"


# =========================
# 2) Data indexing
# =========================


def index_libero_dataset(
    data_root: str,
    activations_root: str,
    groups=("10", "goal", "object", "spatial"),
) -> List[Episode]:
    # Build maps from episode_id -> path for actions/videos
    actions_map: Dict[Tuple[str, str], str] = {}
    video_map: Dict[Tuple[str, str], str] = {}

    for g in groups:
        actions_dir = os.path.join(data_root, g, "actions")
        videos_dir = os.path.join(data_root, g, "videos")

        for p in sorted(glob.glob(os.path.join(actions_dir, "*.json"))):
            eid = parse_episode_id_from_actions_json(p)
            actions_map[(g, eid)] = p

        for p in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
            eid = parse_episode_id_from_video(p)
            video_map[(g, eid)] = p

    act_paths = sorted(glob.glob(os.path.join(activations_root, "*.npy")))
    act_map: Dict[str, str] = {}
    for p in act_paths:
        eid = parse_episode_id_from_activation_npy(p)
        act_map[eid] = p

    episodes: List[Episode] = []

    for (g, raw_eid), a_path in actions_map.items():
        # Your original code had a custom mapping from actions->videos and actions->activations.
        # Keep your logic but make it a bit more defensive.

        # Find video: in some datasets actions file name differs from video stem.
        # If your stems match exactly, this will work:
        v_path = video_map.get((g, raw_eid), None)

        # Your downstream logic expects to parse a number from raw_eid
        mnum = re.search(r"\d+", raw_eid)
        if mnum is None:
            # just won't map activations via your naming scheme
            num = None
        else:
            num = int(mnum.group())

        # Your normalization:
        #   raw: "actions_..._trial..." -> eid = "...", and then find task index by substring search.
        eid = raw_eid
        eid = eid.replace("actions_", "")
        eid = eid.split("_trial")[0]

        # Determine task index based on eid contained in the prompt strings
        task = -1
        key = f"libero_{g}"
        if key in libero_task_map:
            task = next((i for i, s in enumerate(libero_task_map[key]) if eid in s), -1)

        # Activation path convention you used:
        #   f"task{task}_ep{num}_post_ffn_last_step"
        act_path = None
        if (task is not None) and (task >= 0) and (num is not None):
            act_key = f"task{task}_ep{num}_post_ffn_last_step"
            act_path = act_map.get(act_key, None)

        prompt = prompt_for_group_and_episode(g, eid)
        episodes.append(Episode(g, eid, a_path, v_path, prompt, act_path))

    print(f"Indexed {len(episodes)} episodes (may include missing video/activation).")
    return episodes


# =========================
# 3) Actions loader (customize to your json schema)
# =========================

def _is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def _as_float_vec(x):
    """Try convert x into a 1D float32 vector; return None if impossible."""
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and np.issubdtype(x.dtype, np.number):
            return x.astype(np.float32)
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_num(v) for v in x):
        return np.asarray(x, dtype=np.float32)
    return None

def _find_action_in_dict(d):
    """
    Heuristics: try common keys, else find first list-of-numbers value (possibly nested one level).
    Returns np.float32 vector or None.
    """
    candidate_keys = [
        "action", "actions",
        "robot_action", "robot_actions",
        "ctrl", "control", "command",
        "ee_action", "ee_delta", "delta",
    ]

    for k in candidate_keys:
        if k in d:
            v = d[k]
            vec = _as_float_vec(v)
            if vec is not None:
                return vec
            if isinstance(v, dict):
                for vv in v.values():
                    vec2 = _as_float_vec(vv)
                    if vec2 is not None:
                        return vec2

    for v in d.values():
        vec = _as_float_vec(v)
        if vec is not None:
            return vec
        if isinstance(v, dict):
            for vv in v.values():
                vec2 = _as_float_vec(vv)
                if vec2 is not None:
                    return vec2

    return None

def load_actions(actions_json_path: str) -> np.ndarray:
    """
    Returns actions as float array (T, action_dim).
    Supports:
      - {"actions": [[...], ...]}
      - {"actions": [{...}, {...}, ...]}
      - [[...], ...]
      - [{...}, {...}, ...]   (list of dict per timestep)
    """
    with open(actions_json_path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        if "actions" in obj:
            obj = obj["actions"]
        else:
            raise ValueError(f"Dict JSON without 'actions' key in {actions_json_path}")

    if isinstance(obj, list):
        if len(obj) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if isinstance(obj[0], (list, tuple, np.ndarray)):
            acts = np.asarray(obj, dtype=np.float32)
            if acts.ndim != 2:
                raise ValueError(f"Expected (T, action_dim) from list-of-vectors; got {acts.shape} in {actions_json_path}")
            return acts

        if isinstance(obj[0], dict):
            rows = []
            for i, step in enumerate(obj):
                vec = _find_action_in_dict(step)
                if vec is None:
                    raise ValueError(
                        f"Could not find numeric action vector at step {i} in {actions_json_path}. "
                        f"Keys were: {list(step.keys())[:30]}"
                    )
                rows.append(vec)

            dim0 = rows[0].shape[0]
            for i, v in enumerate(rows):
                if v.shape[0] != dim0:
                    raise ValueError(
                        f"Inconsistent action_dim in {actions_json_path}: step0={dim0}, step{i}={v.shape[0]}"
                    )
            return np.stack(rows, axis=0).astype(np.float32)

    raise ValueError(f"Unrecognized action json schema in {actions_json_path}: type={type(obj)}")




# =========================
# 5) Prompt ranking helpers
# =========================

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
        # import pdb; pdb.set_trace()
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

    for c in range(nb_concepts):
        hits = concept_hits[c]
        if len(hits) == 0:
            continue

        # Sort by frame score desc
        hits.sort(key=lambda x: x["score"], reverse=True)

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

        concept_dir = os.path.join(out_dir, f"concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

        prompts_txt_path = os.path.join(concept_dir, "prompts.txt")
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


# =========================
# 7) Run
# =========================

if __name__ == "__main__":
    ckpt_path = "./checkpoints/TopKSAE/sae_layer11_k10_c50.pt"  # TODO
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
        device="cuda",
        encode_batch=8192,

        # Tune these:
        top_m_frames_per_concept=5000,   # increase (or None) for better prompt rankings
        top_prompts_per_concept=10,
        prompt_top_k_frames=10,
        frames_to_save_per_prompt=8,
        prompt_score="mean_topk",        # or "max" if you want “single strongest moment”

        strict=True,
    )