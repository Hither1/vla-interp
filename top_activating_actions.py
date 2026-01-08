import os, glob, json, re
from dataclasses import dataclass
from collections import Counter
import numpy as np
import torch

from overcomplete.sae import TopKSAE  
from utils import *



# def _find_action_in_dict(d):
#     candidate_keys = [
#         "action", "actions",
#         "robot_action", "robot_actions",
#         "ctrl", "control", "command",
#         "ee_action", "ee_delta", "delta",
#     ]

#     for k in candidate_keys:
#         if k in d:
#             v = d[k]
#             vec = _as_float_vec(v)
#             if vec is not None:
#                 return vec
#             if isinstance(v, dict):
#                 for vv in v.values():
#                     vec2 = _as_float_vec(vv)
#                     if vec2 is not None:
#                         return vec2

#     for v in d.values():
#         vec = _as_float_vec(v)
#         if vec is not None:
#             return vec
#         if isinstance(v, dict):
#             for vv in v.values():
#                 vec2 = _as_float_vec(vv)
#                 if vec2 is not None:
#                     return vec2

#     return None


# def load_actions(actions_json_path: str) -> np.ndarray:
#     with open(actions_json_path, "r") as f:
#         obj = json.load(f)

#     if isinstance(obj, dict):
#         if "actions" in obj:
#             obj = obj["actions"]
#         else:
#             raise ValueError(f"Dict JSON without 'actions' key in {actions_json_path}")

#     if isinstance(obj, list):
#         if len(obj) == 0:
#             return np.zeros((0, 0), dtype=np.float32)

#         if isinstance(obj[0], (list, tuple, np.ndarray)):
#             acts = np.asarray(obj, dtype=np.float32)
#             if acts.ndim != 2:
#                 raise ValueError(f"Expected (T, action_dim); got {acts.shape} in {actions_json_path}")
#             return acts

#         if isinstance(obj[0], dict):
#             rows = []
#             for i, step in enumerate(obj):
#                 vec = _find_action_in_dict(step)
#                 if vec is None:
#                     raise ValueError(
#                         f"Could not find numeric action vector at step {i} in {actions_json_path}. "
#                         f"Keys: {list(step.keys())[:30]}"
#                     )
#                 rows.append(vec)

#             dim0 = rows[0].shape[0]
#             for i, v in enumerate(rows):
#                 if v.shape[0] != dim0:
#                     raise ValueError(f"Inconsistent action_dim in {actions_json_path}: step0={dim0}, step{i}={v.shape[0]}")
#             return np.stack(rows, axis=0).astype(np.float32)

#     raise ValueError(f"Unrecognized action json schema in {actions_json_path}: type={type(obj)}")



# def get_frame_opencv(video_path: str, frame_idx: int):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {video_path}")
#     cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
#     ok, frame_bgr = cap.read()
#     cap.release()
#     if not ok:
#         return None
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     return frame_rgb

# def save_frame_png(rgb: np.ndarray, out_path: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(out_path, bgr)


# =========================
# 5) Top-activating actions utilities
# =========================


def top_action_clusters_from_hits(
    hits,
    action_quant: float = 0.02,
    top_k: int = 10,
    min_count: int = 1,
    rank_by: str = "score_sum",   # {"score_sum","score_mean","score_max","count"}
):
    """
    Cluster continuous action vectors by per-dimension quantization, then rank clusters.
    This tends to produce interpretable "action modes" for each concept.
    """
    if len(hits) == 0:
        return []

    groups = {}

    for h in hits:
        a = np.asarray(h["action"], dtype=np.float32)
        s = float(h["score"])

        key = tuple(np.round(a / action_quant).astype(np.int32).tolist())
        if key not in groups:
            groups[key] = {
                "count": 0,
                "score_sum": 0.0,
                "score_max": -1e9,
                "actions_sum": np.zeros_like(a, dtype=np.float64),
            }

        g = groups[key]
        g["count"] += 1
        g["score_sum"] += s
        g["score_max"] = max(g["score_max"], s)
        g["actions_sum"] += a.astype(np.float64)

    rows = []
    for key, g in groups.items():
        if g["count"] < min_count:
            continue
        mean_action = (g["actions_sum"] / g["count"]).astype(np.float32)
        row = {
            "count": int(g["count"]),
            "score_sum": float(g["score_sum"]),
            "score_mean": float(g["score_sum"] / g["count"]),
            "score_max": float(g["score_max"]),
            "action_mean": mean_action.tolist(),
            "action_mean_named": {ACTION_NAMES[i]: float(mean_action[i]) for i in range(min(len(ACTION_NAMES), mean_action.shape[0]))},
            "action_key": list(key),
        }
        rows.append(row)

    if len(rows) == 0:
        return []

    if rank_by not in rows[0]:
        raise ValueError(f"rank_by must be one of {list(rows[0].keys())}, got {rank_by}")

    rows.sort(key=lambda r: r[rank_by], reverse=True)
    return rows[:top_k]

def action_dimension_association(hits, method: str = "corr"):
    """
    For a concept's top hits, compute association between score and each action dimension.
    - method="corr": Pearson correlation corr(score, action_dim)
    - method="abs_corr": abs(Pearson corr)
    Returns sorted list of per-dim stats.
    """
    if len(hits) == 0:
        return []

    scores = np.asarray([h["score"] for h in hits], dtype=np.float32)
    A = np.stack([np.asarray(h["action"], dtype=np.float32) for h in hits], axis=0)  # (N, 7)

    if A.ndim != 2:
        return []
    if A.shape[1] != len(ACTION_NAMES):
        # still handle, but name what we can
        names = [f"a{i}" for i in range(A.shape[1])]
    else:
        names = ACTION_NAMES

    out = []
    s_std = float(np.std(scores))
    for j in range(A.shape[1]):
        aj = A[:, j]
        a_std = float(np.std(aj))
        if s_std < 1e-8 or a_std < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(scores, aj)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0

        out.append({
            "dim": int(j),
            "name": names[j],
            "corr": corr,
            "abs_corr": abs(corr),
            "mean": float(np.mean(aj)),
            "std": float(np.std(aj)),
        })

    key = "corr" if method == "corr" else "abs_corr"
    out.sort(key=lambda r: r[key], reverse=True)
    return out




def mine_concepts_global(
    ckpt_path: str,
    data_root: str,
    activations_root: str,
    out_dir: str,
    layer_idx: int,
    top_m: int = 50,
    per_concept_save_k: int = 16,
    device: str = "cuda",
    encode_batch: int = 8192,

    # action summarization knobs
    top_action_clusters_k: int = 10,
    action_quant: float = 0.02,
    action_cluster_min_count: int = 1,
    action_cluster_rank_by: str = "score_sum",

    # per-dimension view
    action_assoc_method: str = "abs_corr",  # {"corr","abs_corr"}
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

    nb_concepts = ckpt["nb_concepts"]
    d_expected = ckpt["d"]
    topk = min(ckpt["top_k"], nb_concepts)
    print(f"Loaded SAE: nb_concepts={nb_concepts}, d={d_expected}, top_k={ckpt['top_k']}")

    episodes = index_libero_dataset(data_root=data_root, activations_root=activations_root)

    # ============================================================
    # PASS 1: gather all usable frames
    # ============================================================
    cached = []
    total_T = 0

    for ep in episodes:
        if ep.act_path is None or (not os.path.exists(ep.act_path)):
            continue
        if ep.video_path is None or (not os.path.exists(ep.video_path)):
            continue
        if ep.actions_path is None or (not os.path.exists(ep.actions_path)):
            continue

        acts = load_actions(ep.actions_path)  # (T_a, 7)
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

    if total_T == 0:
        raise RuntimeError("No usable frames found (check paths / matching logic).")

    X_all = np.empty((total_T, d_expected), dtype=np.float32)

    ep_group = [None] * total_T
    ep_id    = [None] * total_T
    t_in_ep  = np.empty((total_T,), dtype=np.int32)
    prompt   = [None] * total_T
    video    = [None] * total_T
    actions_all = np.empty((total_T, 7), dtype=np.float32)

    offset = 0
    for (ep, A_layer, acts_use, T_use) in cached:
        if acts_use.shape[1] != 7:
            raise ValueError(f"Expected action_dim=7 for {ep.actions_path}, got {acts_use.shape[1]}")

        X_all[offset:offset+T_use] = A_layer
        actions_all[offset:offset+T_use] = acts_use.astype(np.float32)

        ep_group[offset:offset+T_use] = [ep.group] * T_use
        ep_id[offset:offset+T_use]    = [ep.episode_id] * T_use
        t_in_ep[offset:offset+T_use]  = np.arange(T_use, dtype=np.int32)
        prompt[offset:offset+T_use]   = [ep.prompt] * T_use
        video[offset:offset+T_use]    = [ep.video_path] * T_use

        offset += T_use

    assert offset == total_T
    print(f"Total usable frames: {total_T}")

    # ============================================================
    # PASS 2: encode globally and collect concept hits
    # ============================================================
    concept_hits = [[] for _ in range(nb_concepts)]

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
                    print('s', s)
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
    # PASS 3: per-concept summaries (+ top activating actions)
    # ============================================================
    summaries = {}

    for c in range(nb_concepts):
        hits = concept_hits[c]
        hits.sort(key=lambda x: x["score"], reverse=True)
        hits = hits[:top_m]

        prompt_counts = Counter([h["prompt"] for h in hits])

        if len(hits) > 0:
            action_mat = np.stack([h["action"] for h in hits], axis=0)  # (N,7)
            action_mean = action_mat.mean(axis=0)
            action_median = np.median(action_mat, axis=0)
        else:
            action_mean = np.zeros((7,), dtype=np.float32)
            action_median = np.zeros((7,), dtype=np.float32)

        # NEW 1) clustered top actions (continuous -> discrete-ish modes)
        top_action_clusters = top_action_clusters_from_hits(
            hits,
            action_quant=action_quant,
            top_k=top_action_clusters_k,
            min_count=action_cluster_min_count,
            rank_by=action_cluster_rank_by,
        )

        # NEW 2) per-dimension association with concept score
        action_assoc = action_dimension_association(hits, method=action_assoc_method)

        concept_dir = os.path.join(out_dir, f"concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

        # Save frames for top examples
        for rank, h in enumerate(hits[:per_concept_save_k]):
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
            "num_hits_considered": len(hits),
            "top_prompts": prompt_counts.most_common(10),

            "action_semantics": {
                "a_t": ["dx","dy","dz","droll","dpitch","dyaw","gripper"]
            },

            "action_mean": action_mean.tolist(),
            "action_mean_named": {ACTION_NAMES[i]: float(action_mean[i]) for i in range(7)},
            "action_median": action_median.tolist(),
            "action_median_named": {ACTION_NAMES[i]: float(action_median[i]) for i in range(7)},

            # top activating "action modes"
            "top_action_clusters": top_action_clusters,
            "action_cluster_params": {
                "action_quant": action_quant,
                "top_action_clusters_k": top_action_clusters_k,
                "min_count": action_cluster_min_count,
                "rank_by": action_cluster_rank_by,
            },

            # dimension-level view: which action dims co-vary with activation
            "action_dimension_association": action_assoc,
            "action_assoc_method": action_assoc_method,

            "top_examples": [
                {
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "prompt": h["prompt"],
                    "action": h["action"].tolist(),
                    "action_named": {ACTION_NAMES[i]: float(h["action"][i]) for i in range(7)},
                    "video_path": h["video_path"],
                }
                for h in hits[:10]
            ],
        }

        with open(os.path.join(concept_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summaries[c] = summary

    with open(os.path.join(out_dir, "all_concepts_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Done. Wrote concept folders to: {out_dir}")




if __name__ == "__main__":
    ckpt_path = "./checkpoints/TopKSAE/sae_layer11_k10_c16000.pt"  
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
        top_m=50,
        per_concept_save_k=16,
        device="cuda",

        # action modes
        top_action_clusters_k=12,
        action_quant=0.01,              # tune: 0.005–0.05 depending on action scale
        action_cluster_min_count=2,     # ignore singletons (optional)
        action_cluster_rank_by="score_sum",

        # dimension association
        action_assoc_method="abs_corr", # shows strongest dims regardless of sign
    )