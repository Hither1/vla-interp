import os, glob, json, re
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
import torch
import os, json


# ---- video frame extraction (choose one) ----
# OpenCV
import cv2

# Option B: decord (often faster / cleaner indexing)
# from decord import VideoReader, cpu

from overcomplete.sae import TopKSAE  # assumes same as your training

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
# 1) Customize these parsers
# =========================

def parse_episode_id_from_actions_json(path: str) -> str:
    """
    Return a stable episode id that also matches video filename and activation npy filename.
    Common pattern: {stem}.json and {stem}.mp4 and {stem}.npy
    """
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
    """
    If your episode id encodes which task index it is, parse it here.
    Otherwise, fall back to 'unknown' or store the group only.

    Common Libero setup: episodes are grouped by task (10 tasks per group).
    If you have metadata elsewhere, swap this to use that.
    """
    # Minimal default: just return group name (you can improve later)
    # Better: if episode_id ends with something like "_task03", map to index 3.
    m = re.search(r"task(\d+)", episode_id)
    if m:
        idx = int(m.group(1))
        key = f"libero_{group_name}" if not group_name.startswith("libero_") else group_name
        if key in libero_task_map and 0 <= idx < len(libero_task_map[key]):
            return libero_task_map[key][idx]
    # fallback: unknown prompt
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
    # Build maps from episode_id -> path for actions/videos
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

    # Activation files (likely separate folder). Build by episode_id only,
    # or if your filename includes group, parse that and include it.
    act_paths = sorted(glob.glob(os.path.join(activations_root, "*.npy")))
    act_map = {}
    for p in act_paths:
        eid = parse_episode_id_from_activation_npy(p)
        act_map[eid] = p

    episodes = []
    
    for (g, eid), a_path in actions_map.items():
        v_path = video_map.get((g, eid.replace('actions', 'rollout')), None)
        
        num = int(re.search(r'\d+', eid).group())
        eid = eid.replace('actions_', '').split("_trial")[0]

        task = next((i for i, s in enumerate(libero_task_map[f'libero_{g}']) if eid in s), -1) 
        act_path = act_map.get(f'task{task}_ep{num}_post_ffn_last_step', None)
        # if v_path is None or act_path is None:
        #     continue
        prompt = prompt_for_group_and_episode(g, eid)
        episodes.append(Episode(g, eid, a_path, v_path, prompt, act_path))

    print(f"Indexed {len(episodes)} matched episodes.")
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
    # common keys in robotics logs
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
            # sometimes nested: {"action": {"arm": [...], "gripper": ...}}
            if isinstance(v, dict):
                # try flatten common nested patterns
                for kk, vv in v.items():
                    vec2 = _as_float_vec(vv)
                    if vec2 is not None:
                        return vec2

    # fallback: search any value that looks like a numeric vector (including one-level nested dicts)
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

    # Case 1: dict container
    if isinstance(obj, dict):
        if "actions" in obj:
            obj = obj["actions"]
        else:
            # sometimes a single dict per episode with timesteps elsewhere; add your key here if needed
            raise ValueError(f"Dict JSON without 'actions' key in {actions_json_path}")

    # Case 2: list container
    if isinstance(obj, list):
        if len(obj) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # list of vectors
        if isinstance(obj[0], (list, tuple, np.ndarray)):
            acts = np.asarray(obj, dtype=np.float32)
            if acts.ndim != 2:
                raise ValueError(f"Expected (T, action_dim) from list-of-vectors; got {acts.shape} in {actions_json_path}")
            return acts

        # list of dicts (your case)
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

            # ensure consistent action_dim
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





def mine_concepts(
    ckpt_path: str,
    data_root: str,
    activations_root: str,
    out_dir: str,
    layer_idx: int,
    top_m: int = 50,          # top frames per concept (global)
    per_concept_save_k: int = 16,  # save top K frame images per concept
    device: str = "cuda",
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
    print(f"Loaded SAE: nb_concepts={nb_concepts}, d={ckpt['d']}, top_k={ckpt['top_k']}")

    episodes = index_libero_dataset(data_root=data_root, activations_root=activations_root)

    # For each concept: maintain a list of (score, meta...)
    # For simplicity: store all candidates then take top at end.
    concept_hits = [[] for _ in range(nb_concepts)]

    # Also prompt counts among top hits (computed after selecting top)
    # and action stats among top hits.
    for ep in episodes:
        acts = load_actions(ep.actions_path)  # (T_a, action_dim)
        A = np.load(ep.act_path).astype(np.float32)

        # Expect A shape something like (T, num_layers, d) or (T, num_layers, 1, d)
        if A.ndim == 4:
            # (T, num_layers, 1, d) -> squeeze
            A = A.squeeze(-2)
        if A.ndim != 3:
            raise ValueError(f"Unexpected activation shape {A.shape} in {ep.act_path}")

        T, num_layers, d = A.shape
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers-1}]")

        # align lengths (very common mismatch: actions length vs frames length)
        T_use = min(T, acts.shape[0])
        if T_use <= 0:
            continue

        x = torch.from_numpy(A[:T_use, layer_idx, :]).to(device)  # (T_use, d)
        
        with torch.no_grad():
            pre_codes, codes = sae.encode(x)
        codes = codes.detach().float()
        import pdb; pdb.set_trace()

        # score: activation magnitude (TopK usually nonnegative, but be safe)
        scores = codes.abs()

        # For each t, for the few nonzero concepts, record hits.
        # If codes is dense, this is heavy; but TopK should be sparse-ish.
        # We’ll just take top-k concepts at each timestep as candidates.
        per_t_top = torch.topk(scores, k=min(ckpt["top_k"], nb_concepts), dim=1)
        top_vals = per_t_top.values.cpu().numpy()
        top_ids = per_t_top.indices.cpu().numpy()

        for t in range(T_use):
            for j in range(top_ids.shape[1]):
                c = int(top_ids[t, j])
                s = float(top_vals[t, j])
                if s <= 0:
                    continue
                concept_hits[c].append({
                    "score": s,
                    "group": ep.group,
                    "episode_id": ep.episode_id,
                    "t": t,
                    "prompt": ep.prompt,
                    "action": acts[t].astype(np.float32),
                    "video_path": ep.video_path,
                })

    # ---- post-process each concept ----
    summaries = {}

    for c in range(nb_concepts):
        hits = concept_hits[c]
        if len(hits) == 0:
            continue

        # keep global top_m
        hits.sort(key=lambda x: x["score"], reverse=True)
        hits = hits[:top_m]

        # 1) prompts
        prompt_counts = Counter([h["prompt"] for h in hits])

        # 2) actions: compute mean/median over top hits
        action_mat = np.stack([h["action"] for h in hits], axis=0)  # (top_m, action_dim)
        action_mean = action_mat.mean(axis=0)
        action_median = np.median(action_mat, axis=0)

        # 3) save top frames
        concept_dir = os.path.join(out_dir, f"concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

        for rank, h in enumerate(hits[:per_concept_save_k]):
            rgb = get_frame_opencv(h["video_path"], h["t"])
            if rgb is None:
                continue
            png_path = os.path.join(concept_dir, f"rank_{rank:02d}_score_{h['score']:.4f}_{h['episode_id']}_t{h['t']:05d}.png")
            save_frame_png(rgb, png_path)

        # write a small json summary per concept
        summary = {
            "concept": c,
            "num_hits_considered": len(hits),
            "top_prompts": prompt_counts.most_common(10),
            "action_mean": action_mean.tolist(),
            "action_median": action_median.tolist(),
            "top_examples": [
                {
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "prompt": h["prompt"],
                    "action": h["action"].tolist(),
                    "video_path": h["video_path"],
                }
                for h in hits[:10]
            ],
        }
        with open(os.path.join(concept_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summaries[c] = summary

    # global index file
    with open(os.path.join(out_dir, "all_concepts_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Done. Wrote concept folders to: {out_dir}")




def mine_concepts_global(
    ckpt_path: str,
    data_root: str,
    activations_root: str,
    out_dir: str,
    layer_idx: int,
    top_m: int = 50,
    per_concept_save_k: int = 16,
    device: str = "cuda",
    encode_batch: int = 8192,   # global batch size over all frames
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
    # PASS 1: figure out total number of usable frames and allocate
    # ============================================================
    lengths = []
    cached = []  # store per-episode (ep, A_layer, acts, T_use) to avoid re-loading if you want
    total_T = 0

    for ep in episodes:
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

        lengths.append(T_use)
        total_T += T_use

    if total_T == 0:
        print("No usable frames found.")
        return

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

    # ============================================================
    # PASS 2: encode globally in batches and collect hits
    # ============================================================
    concept_hits = [[] for _ in range(nb_concepts)]

    with torch.no_grad():
        for start in range(0, total_T, encode_batch):
            end = min(total_T, start + encode_batch)
            x = torch.from_numpy(X_all[start:end]).to(device)  # (B, d)

            pre_codes, codes = sae.encode(x)  # codes: (B, nb_concepts) (TopK sparse-ish)
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

    # ---- post-process each concept ----
    summaries = {}

    for c in range(nb_concepts):
        hits = concept_hits[c]
        if not hits:
            continue

        hits.sort(key=lambda x: x["score"], reverse=True)
        hits = hits[:top_m]

        prompt_counts = Counter([h["prompt"] for h in hits])

        action_mat = np.stack([h["action"] for h in hits], axis=0)
        action_mean = action_mat.mean(axis=0)
        action_median = np.median(action_mat, axis=0)

        concept_dir = os.path.join(out_dir, f"concept_{c:04d}")
        os.makedirs(concept_dir, exist_ok=True)

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
            "action_mean": action_mean.tolist(),
            "action_median": action_median.tolist(),
            "top_examples": [
                {
                    "score": h["score"],
                    "group": h["group"],
                    "episode_id": h["episode_id"],
                    "t": h["t"],
                    "prompt": h["prompt"],
                    "action": h["action"].tolist(),
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

# =========================
# 7) Run
# =========================

if __name__ == "__main__":
    ckpt_path = "./checkpoints/sae_layer11_k10_c16000.pt"  # TODO
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
    )