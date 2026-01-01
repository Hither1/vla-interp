#!/usr/bin/env python3
import os
import glob
import json
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
import torch

# ---- video frame extraction ----
import cv2

from overcomplete.sae import TopKSAE  

# =========================
# 0) Libero task map
# =========================

libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
    # libero_90 omitted here for brevity; keep yours
}

# =========================
# 1) Parsers / prompt helpers
# =========================

def parse_episode_id_from_actions_json(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_video(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_activation_npy(path: str) -> str:
    """
    IMPORTANT: customize if your activation filenames include extra suffixes.
    Example:
      pi0_activations/libero_goal_ep000123.npy -> "libero_goal_ep000123"
      or libero_goal/ep000123.npy -> "ep000123"
    """
    return os.path.splitext(os.path.basename(path))[0]

def prompt_for_group_and_episode(group_name: str, episode_id: str) -> str:
    m = re.search(r"task(\d+)", episode_id)
    if m:
        idx = int(m.group(1))
        key = f"libero_{group_name}" if not group_name.startswith("libero_") else group_name
        if key in libero_task_map and 0 <= idx < len(libero_task_map[key]):
            return libero_task_map[key][idx]
    return f"{group_name}:unknown_prompt"

# =========================
# 2) Data indexing
# =========================

@dataclass
class Episode:
    group: str
    episode_id: str
    actions_path: str
    video_path: str
    prompt: str
    act_path: str  # activation npy path

def index_libero_dataset(
    data_root: str,
    activations_root: str,
    groups=("10", "goal", "object", "spatial"),
):
    # Build maps from (group, episode_id) -> path for actions/videos
    actions_map = {}
    video_map = {}

    for g in groups:
        actions_dir = os.path.join(data_root, g, "actions")
        videos_dir = os.path.join(data_root, g, "videos")

        for p in sorted(glob.glob(os.path.join(actions_dir, "*.json"))):
            eid = parse_episode_id_from_actions_json(p)
            actions_map[(g, eid)] = p

        for p in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
            eid = parse_episode_id_from_video(p)
            video_map[(g, eid)] = p

    # Activation files (separate folder). Map by activation "eid" stem.
    act_paths = sorted(glob.glob(os.path.join(activations_root, "*.npy")))
    act_map = {}
    for p in act_paths:
        eid = parse_episode_id_from_activation_npy(p)
        act_map[eid] = p

    episodes = []

    for (g, eid_raw), a_path in actions_map.items():
        # If your video naming differs, fix this mapping
        v_path = video_map.get((g, eid_raw.replace("actions", "rollout")), None)

        # Your custom parsing logic
        mnum = re.search(r"\d+", eid_raw)
        num = int(mnum.group()) if mnum else -1

        eid = eid_raw.replace("actions_", "").split("_trial")[0]

        # Map prompt/task index by substring match
        task = next(
            (i for i, s in enumerate(libero_task_map[f"libero_{g}"]) if eid in s),
            -1,
        )
        act_path = act_map.get(f"task{task}_ep{num}_post_ffn_last_step", None)

        prompt = prompt_for_group_and_episode(g, eid)
        episodes.append(Episode(g, eid, a_path, v_path, prompt, act_path))

    print(f"Indexed {len(episodes)} episodes (may include missing video/act paths).")
    return episodes

# =========================
# 3) Actions loader
# =========================

def _is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def _as_float_vec(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and np.issubdtype(x.dtype, np.number):
            return x.astype(np.float32)
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_num(v) for v in x):
        return np.asarray(x, dtype=np.float32)
    return None

def _find_action_in_dict(d):
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
                for _, vv in v.items():
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
                raise ValueError(f"Expected (T, action_dim); got {acts.shape} in {actions_json_path}")
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
# 4) Video frame extraction
# =========================

def get_frame_opencv(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def save_frame_png(rgb: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)

# =========================
# 5) NEW: selecting top activating frames per concept (diversity + NMS)
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
    # NEW knobs:
    nms_time_window: int = 5,
    max_per_episode: int = 1,
    skip_missing: bool = True,
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
    # PASS 1: cache usable (ep, A_layer, acts) and compute total_T
    # ============================================================
    cached = []
    total_T = 0

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
        top_m=200,               # increase if you want more diversity options
        per_concept_save_k=16,
        device="cuda",
        encode_batch=8192,
        nms_time_window=5,       # try 0, 3, 5, 10
        max_per_episode=1,       # try 1 or 2
        skip_missing=True,
    )