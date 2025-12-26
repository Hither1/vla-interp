#!/usr/bin/env python3
import os, glob, json, re, argparse, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import torch

from overcomplete.sae import TopKSAE  # same as your training

# -------------------------
# Task map (unchanged)
# -------------------------
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
}

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

# -------------------------
# Parsers (unchanged)
# -------------------------
def parse_episode_id_from_actions_json(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_video(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_activation_npy(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def prompt_for_group_and_episode(group_name: str, episode_id: str) -> str:
    m = re.search(r"task(\d+)", episode_id)
    if m:
        idx = int(m.group(1))
        key = f"libero_{group_name}" if not group_name.startswith("libero_") else group_name
        if key in libero_task_map and 0 <= idx < len(libero_task_map[key]):
            return libero_task_map[key][idx]
    return f"{group_name}:unknown_prompt"

# -------------------------
# Dataset indexing (unchanged)
# -------------------------
@dataclass
class Episode:
    group: str
    episode_id: str
    actions_path: str
    video_path: Optional[str]
    prompt: str
    act_path: Optional[str]  # activation npy path

def index_libero_dataset(
    data_root: str,
    activations_root: str,
    groups=("10", "goal", "object", "spatial"),
):
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

    act_paths = sorted(glob.glob(os.path.join(activations_root, "*.npy")))
    act_map = {parse_episode_id_from_activation_npy(p): p for p in act_paths}

    episodes = []
    for (g, eid_raw), a_path in actions_map.items():
        v_path = video_map.get((g, eid_raw.replace("actions", "rollout")), None)

        mnum = re.search(r"\d+", eid_raw)
        if mnum is None:
            continue
        num = int(mnum.group())

        eid = eid_raw.replace("actions_", "").split("_trial")[0]
        task = next((i for i, s in enumerate(libero_task_map[f"libero_{g}"]) if eid in s), -1)

        # your activation naming convention
        act_path = act_map.get(f"task{task}_ep{num}_post_ffn_last_step", None)

        prompt = prompt_for_group_and_episode(g, eid)
        episodes.append(Episode(g, eid, a_path, v_path, prompt, act_path))

    print(f"Indexed {len(episodes)} episodes (some may have missing video/activation paths).")
    return episodes

# -------------------------
# Actions loader (unchanged)
# -------------------------
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
                        f"Keys: {list(step.keys())[:30]}"
                    )
                rows.append(vec)

            dim0 = rows[0].shape[0]
            for i, v in enumerate(rows):
                if v.shape[0] != dim0:
                    raise ValueError(f"Inconsistent action_dim in {actions_json_path}: step0={dim0}, step{i}={v.shape[0]}")
            return np.stack(rows, axis=0).astype(np.float32)

    raise ValueError(f"Unrecognized action json schema in {actions_json_path}: type={type(obj)}")

# -------------------------
# Utilities: splitting, metrics, standardization, ridge
# -------------------------
def split_episodes(episodes: List[Episode], seed: int, train_frac=0.8, val_frac=0.1):
    """Split by episode (not by frame) to avoid leakage."""
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(episodes))
    rng.shuffle(idxs)

    n = len(episodes)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    tr = [episodes[i] for i in idxs[:n_train]]
    va = [episodes[i] for i in idxs[n_train:n_train+n_val]]
    te = [episodes[i] for i in idxs[n_train+n_val:]]
    return tr, va, te

def standardize_fit(X: torch.Tensor, eps=1e-6):
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(eps)
    return mu, std

def standardize_apply(X: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    return (X - mu) / std

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps=1e-12) -> torch.Tensor:
    # per-dim R2
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0).clamp_min(eps)
    return 1.0 - ss_res / ss_tot

def ridge_closed_form(X: torch.Tensor, Y: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve: min_W ||XW - Y||^2 + lam ||W||^2 with bias.
    Returns (W, b).
    X: (N, D), Y: (N, K)
    """
    device = X.device
    N, D = X.shape
    K = Y.shape[1]

    # augment with bias column
    ones = torch.ones((N, 1), device=device, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)  # (N, D+1)

    # ridge only on weights, not bias
    I = torch.eye(D + 1, device=device, dtype=X.dtype)
    I[-1, -1] = 0.0  # don't penalize bias

    # (Xb^T Xb + lam I)^{-1} Xb^T Y
    XtX = Xb.T @ Xb
    A = XtX + lam * I
    XtY = Xb.T @ Y

    Wb = torch.linalg.solve(A, XtY)  # (D+1, K)
    W = Wb[:D, :]
    b = Wb[D:, :].squeeze(0)  # (K,)
    return W, b

# -------------------------
# Feature extraction
# -------------------------
def load_layer_acts(ep: Episode, layer_idx: int, d_expected: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_layer, A_actions) aligned in time:
      X_layer: (T_use, d)
      A_actions: (T_use, 7)
    """
    acts = load_actions(ep.actions_path).astype(np.float32)  # (T_a, 7)
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
        return np.zeros((0, d), np.float32), np.zeros((0, 7), np.float32)

    X_layer = A[:T_use, layer_idx, :]  # (T_use, d)
    Y = acts[:T_use, :]
    if Y.shape[1] != 7:
        raise ValueError(f"Expected action_dim=7, got {Y.shape[1]} in {ep.actions_path}")
    return X_layer, Y

@torch.no_grad()
def extract_features_and_targets(
    episodes: List[Episode],
    layer_idx: int,
    d_expected: int,
    feature_mode: str,  # "raw" or "sae"
    sae: Optional[TopKSAE],
    device: str,
    encode_batch: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collects frame-level dataset from a list of episodes.
    Returns:
      X: (N, Dfeat)
      Y: (N, 7)
    """
    # First pass: load everything into CPU arrays (keeps code simple).
    X_list = []
    Y_list = []

    for ep in episodes:
        if ep.act_path is None or (not os.path.exists(ep.act_path)):
            continue
        if ep.actions_path is None or (not os.path.exists(ep.actions_path)):
            continue
        X_layer, Y = load_layer_acts(ep, layer_idx=layer_idx, d_expected=d_expected)
        if X_layer.shape[0] == 0:
            continue
        X_list.append(X_layer)
        Y_list.append(Y)

    if len(X_list) == 0:
        raise RuntimeError("No usable episodes found in this split (check matching logic / paths).")

    X_np = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_np = np.concatenate(Y_list, axis=0).astype(np.float32)

    X = torch.from_numpy(X_np).to(device)
    Y = torch.from_numpy(Y_np).to(device)

    if feature_mode == "raw":
        return X, Y

    if feature_mode == "sae":
        if sae is None:
            raise ValueError("feature_mode='sae' but sae is None")

        # Encode in batches -> use sae.encode(x) -> codes (B, nb_concepts)
        codes_all = []
        for start in range(0, X.shape[0], encode_batch):
            end = min(X.shape[0], start + encode_batch)
            x = X[start:end]
            _, codes = sae.encode(x)
            codes_all.append(codes.float())
        Xs = torch.cat(codes_all, dim=0)
        return Xs, Y

    raise ValueError(f"Unknown feature_mode: {feature_mode}")

# -------------------------
# Main training / eval
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, default="./checkpoints/sae_layer11_k10_c16000.pt", help="SAE checkpoint path")
    ap.add_argument("--data_root", type=str, default="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero")
    ap.add_argument("--activations_root", type=str, default="/n/netscratch/sham_lab/Lab/chloe00/pi0_activations")
    ap.add_argument("--layer_idx", type=int, default=11)

    ap.add_argument("--feature_mode", type=str, default="sae", choices=["sae", "raw"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--encode_batch", type=int, default=8192)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--ridge_lambda", type=float, default=1.0, help="Ridge penalty (try 0.1, 1, 10, 100)")
    ap.add_argument("--standardize_x", action="store_true", help="Z-score features using train split stats")
    ap.add_argument("--standardize_y", action="store_true", help="Z-score targets using train split stats")

    ap.add_argument("--out_path", type=str, default="probe_results.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    # ---- load SAE checkpoint if needed ----
    sae = None
    d_expected = None
    nb_concepts = None

    if args.feature_mode == "sae":
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        d_expected = ckpt["d"]
        nb_concepts = ckpt["nb_concepts"]

        sae = TopKSAE(
            ckpt["d"],
            nb_concepts=ckpt["nb_concepts"],
            top_k=ckpt["top_k"],
            device="cpu",
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval().to(args.device)
        print(f"Loaded SAE: d={d_expected}, nb_concepts={nb_concepts}, top_k={ckpt['top_k']}")
    else:
        # if raw, we still need d_expected; easiest: read from ckpt if provided
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        d_expected = ckpt["d"]
        print(f"Raw probe mode. Using d_expected={d_expected} from ckpt.")

    # ---- index episodes ----
    episodes = index_libero_dataset(
        data_root=args.data_root,
        activations_root=args.activations_root,
        groups=("10", "goal", "object", "spatial"),
    )

    # filter to those with act + actions
    usable = [ep for ep in episodes
              if ep.act_path is not None and os.path.exists(ep.act_path)
              and ep.actions_path is not None and os.path.exists(ep.actions_path)]
    if len(usable) == 0:
        raise RuntimeError("No usable episodes with both activation and actions found.")

    train_eps, val_eps, test_eps = split_episodes(
        usable, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac
    )
    print(f"Episodes split: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")

    # ---- extract features ----
    Xtr, Ytr = extract_features_and_targets(
        train_eps, layer_idx=args.layer_idx, d_expected=d_expected,
        feature_mode=args.feature_mode, sae=sae, device=args.device,
        encode_batch=args.encode_batch,
    )
    Xva, Yva = extract_features_and_targets(
        val_eps, layer_idx=args.layer_idx, d_expected=d_expected,
        feature_mode=args.feature_mode, sae=sae, device=args.device,
        encode_batch=args.encode_batch,
    ) if len(val_eps) > 0 else (None, None)
    Xte, Yte = extract_features_and_targets(
        test_eps, layer_idx=args.layer_idx, d_expected=d_expected,
        feature_mode=args.feature_mode, sae=sae, device=args.device,
        encode_batch=args.encode_batch,
    )

    print(f"Frames: train={Xtr.shape[0]}, val={(0 if Xva is None else Xva.shape[0])}, test={Xte.shape[0]}")
    print(f"Feature dim: {Xtr.shape[1]}, target dim: {Ytr.shape[1]}")

    # ---- standardization (fit on train only) ----
    x_mu = x_std = None
    y_mu = y_std = None
    if args.standardize_x:
        x_mu, x_std = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, x_mu, x_std)
        if Xva is not None:
            Xva = standardize_apply(Xva, x_mu, x_std)
        Xte = standardize_apply(Xte, x_mu, x_std)

    if args.standardize_y:
        y_mu, y_std = standardize_fit(Ytr)
        Ytr = standardize_apply(Ytr, y_mu, y_std)
        if Yva is not None:
            Yva = standardize_apply(Yva, y_mu, y_std)
        Yte = standardize_apply(Yte, y_mu, y_std)

    # ---- train ridge probe ----
    W, b = ridge_closed_form(Xtr, Ytr, lam=args.ridge_lambda)

    def predict(X: torch.Tensor) -> torch.Tensor:
        return X @ W + b

    # ---- eval ----
    def eval_split(name: str, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        Yhat = predict(X)
        mse = torch.mean((Y - Yhat) ** 2).item()
        mse_dim = torch.mean((Y - Yhat) ** 2, dim=0).detach().cpu().numpy().tolist()
        r2_dim = r2_score(Y, Yhat).detach().cpu().numpy().tolist()
        r2_mean = float(np.mean(r2_dim))
        return {
            "name": name,
            "n": int(X.shape[0]),
            "mse": float(mse),
            "mse_per_dim": {ACTION_NAMES[i]: float(mse_dim[i]) for i in range(7)},
            "r2_mean": float(r2_mean),
            "r2_per_dim": {ACTION_NAMES[i]: float(r2_dim[i]) for i in range(7)},
        }

    results = {
        "config": {
            "ckpt_path": args.ckpt_path,
            "data_root": args.data_root,
            "activations_root": args.activations_root,
            "layer_idx": args.layer_idx,
            "feature_mode": args.feature_mode,
            "ridge_lambda": args.ridge_lambda,
            "standardize_x": bool(args.standardize_x),
            "standardize_y": bool(args.standardize_y),
            "seed": args.seed,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
        },
        "splits": {},
        "weights": {
            # save weights for later analysis if you want
            "W_shape": list(W.shape),
            "b_shape": list(b.shape),
        }
    }

    results["splits"]["train"] = eval_split("train", Xtr, Ytr)
    if Xva is not None and Xva.numel() > 0:
        results["splits"]["val"] = eval_split("val", Xva, Yva)
    results["splits"]["test"] = eval_split("test", Xte, Yte)

    # ---- optionally unstandardize weight interpretation ----
    # If you standardize Y, reported metrics are in standardized Y units.
    # For absolute-action-space metrics, leave --standardize_y off.

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote results to: {args.out_path}")
    print("Test R2 per dim:")
    for k, v in results["splits"]["test"]["r2_per_dim"].items():
        print(f"  {k:>7s}: {v: .4f}")
    print(f"Test R2 mean: {results['splits']['test']['r2_mean']:.4f}")
    print(f"Test MSE: {results['splits']['test']['mse']:.6f}")

if __name__ == "__main__":
    main()