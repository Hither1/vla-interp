#!/usr/bin/env python3
import os, json, argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from overcomplete.sae import TopKSAE
from utils import * 



# =========================
# Probe weight utilities
# =========================

def unstandardize_probe_weights(
    W: torch.Tensor,
    b: torch.Tensor,
    x_mu: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If X_std = (X - mu)/std and Yhat = X_std @ W + b,
    then in raw-X space: Yhat = X @ W_raw + b_raw, where:
      W_raw = W / std
      b_raw = b - (mu/std) @ W
    Shapes:
      W: (D, K), b: (K,), mu/std: (1, D)
    """
    if x_mu is None or x_std is None:
        return W, b

    std = x_std.squeeze(0)          # (D,)
    mu = x_mu.squeeze(0)            # (D,)
    W_raw = W / std.unsqueeze(1)    # (D, K)
    b_raw = b - (mu / std) @ W      # (K,)
    return W_raw, b_raw


def participation_ratio(w: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Effective number of coordinates used by vector w.
    d_eff = (sum w^2)^2 / sum w^4
    """
    w2 = w.pow(2)
    num = w2.sum().pow(2)
    den = w2.pow(2).sum().clamp_min(eps)
    return float((num / den).item())


def topk_energy_count(w: torch.Tensor, frac: float = 0.9, eps: float = 1e-12) -> int:
    """
    Smallest k such that top-k squared weights capture `frac` of total squared weight energy.
    """
    w2 = w.pow(2)
    total = w2.sum().clamp_min(eps)
    vals, _ = torch.sort(w2, descending=True)
    csum = torch.cumsum(vals, dim=0)
    k = int(torch.searchsorted(csum, frac * total).item()) + 1
    return min(k, w.numel())


def stable_rank_matrix(W: torch.Tensor, eps: float = 1e-12) -> float:
    """
    stable_rank(W) = ||W||_F^2 / sigma_max(W)^2
    """
    if W.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(W)
    fro2 = W.pow(2).sum()
    smax2 = s.max().pow(2).clamp_min(eps)
    return float((fro2 / smax2).item())


def effective_rank_entropy(W: torch.Tensor, eps: float = 1e-12) -> float:
    """
    effective_rank(W) = exp(H(p)), p_i = s_i / sum s_i
    """
    if W.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(W)
    ssum = s.sum().clamp_min(eps)
    p = (s / ssum).clamp_min(eps)
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())


# =========================
# INLP-style routines (regression)
# =========================

def ridge_fit_predictor(Xtr: torch.Tensor, Ytr: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Use your existing closed-form ridge solver from utils.py
    return ridge_closed_form(Xtr, Ytr, lam=lam)


@torch.no_grad()
def baseline_predictor_mse(Ytr: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Baseline for regression: predict train mean per target dim.
    If Y is standardized on train, baseline ~ 0-vector and baseline R2 ~ 0.
    """
    mu = Ytr.mean(dim=0, keepdim=True)
    return float(torch.mean((Y - mu) ** 2).item())


@torch.no_grad()
def eval_probe_regression(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    Yhat = X @ W + b
    mse = torch.mean((Y - Yhat) ** 2).item()
    r2_dim = r2_score(Y, Yhat).detach().cpu().numpy()
    r2_mean = float(np.mean(r2_dim))
    return {"mse": float(mse), "r2_mean": float(r2_mean)}


def orthonormal_basis_of_columns(W: torch.Tensor, rcond: float = 1e-6) -> torch.Tensor:
    """
    Returns U where columns of U form an orthonormal basis for col(W).
    Uses SVD and keeps singular vectors with s > rcond * s_max.
    W: (D, K)
    U: (D, r)
    """
    if W.numel() == 0:
        return W.new_zeros((W.shape[0], 0))
    U, s, _ = torch.linalg.svd(W, full_matrices=False)
    if s.numel() == 0:
        return W.new_zeros((W.shape[0], 0))
    thresh = rcond * s.max()
    r = int((s > thresh).sum().item())
    return U[:, :r]


def project_onto_nullspace(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Project features X onto nullspace of span(U):
      P = I - U U^T
      X_new = X P = X - (XU)U^T
    X: (N, D), U: (D, r)
    """
    if U.numel() == 0:
        return X
    return X - (X @ U) @ U.T


@torch.no_grad()
def stacked_spectrum_stats(W_stack: torch.Tensor) -> Dict:
    """
    Save spectrum + summary stats for later plotting/inspection.
    Uses spectral energy based on s^2.
    """
    if W_stack.numel() == 0:
        return {
            "shape": [int(W_stack.shape[0]), int(W_stack.shape[1])],
            "singular_values": [],
            "stable_rank": 0.0,
            "effective_rank_entropy": 0.0,
            "k_90pct_spectral_energy": 0,
        }

    s = torch.linalg.svdvals(W_stack).detach().cpu()
    s_np = s.numpy().tolist()

    e = s**2
    e_sum = float(e.sum().item())
    if e_sum <= 0:
        k90 = 0
    else:
        c = torch.cumsum(e, dim=0) / e_sum
        k90 = int(torch.searchsorted(c, torch.tensor(0.90)).item()) + 1

    return {
        "shape": [int(W_stack.shape[0]), int(W_stack.shape[1])],
        "singular_values": s_np,
        "stable_rank": stable_rank_matrix(W_stack),
        "effective_rank_entropy": effective_rank_entropy(W_stack),
        "k_90pct_spectral_energy": int(k90),
    }


def inlp_ridge_regression(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xstop: torch.Tensor,
    Ystop: torch.Tensor,
    lam: float,
    max_iters: int = 50,
    stop_on: str = "r2",         # "r2" or "mse"
    r2_tol: float = 0.0,         # stop if r2_mean <= r2_tol for `patience` iters
    mse_ratio_tol: float = 1.0,  # stop if mse >= mse_ratio_tol * baseline_mse for `patience` iters
    patience: int = 3,
    rcond: float = 1e-6,
) -> Dict:
    Xtr_res = Xtr
    Xstop_res = Xstop

    baseline_mse = baseline_predictor_mse(Ytr, Ystop)

    probes = []
    W_list = []
    U_list = []

    bad_count = 0

    for t in range(max_iters):
        W, b = ridge_fit_predictor(Xtr_res, Ytr, lam=lam)
        metrics = eval_probe_regression(Xstop_res, Ystop, W, b)

        # decide whether we're at "chance"/baseline
        at_baseline = False
        if stop_on == "r2":
            at_baseline = (metrics["r2_mean"] <= r2_tol)
        elif stop_on == "mse":
            at_baseline = (metrics["mse"] >= mse_ratio_tol * baseline_mse)
        else:
            raise ValueError(f"stop_on must be 'r2' or 'mse', got {stop_on}")

        bad_count = bad_count + 1 if at_baseline else 0

        U = orthonormal_basis_of_columns(W, rcond=rcond)

        probes.append({
            "iter": int(t),
            "stop_metrics": metrics,
            "W_shape": [int(W.shape[0]), int(W.shape[1])],
            "U_rank": int(U.shape[1]),
            "at_baseline": bool(at_baseline),
            "bad_count": int(bad_count),
        })
        W_list.append(W.detach())
        U_list.append(U.detach())

        # if we've hit baseline for `patience` consecutive iterations, stop
        if bad_count >= patience:
            break

        # project residual features
        Xtr_res = project_onto_nullspace(Xtr_res, U)
        Xstop_res = project_onto_nullspace(Xstop_res, U)

    W_stack = torch.cat(W_list, dim=1) if len(W_list) else Xtr.new_zeros((Xtr.shape[1], 0))
    U_cat = torch.cat(U_list, dim=1) if len(U_list) else Xtr.new_zeros((Xtr.shape[1], 0))

    return {
        "probes": probes,
        "W_list": W_list,
        "U_list": U_list,
        "W_stack": W_stack,
        "U_cat": U_cat,
        "baseline_mse_stop": float(baseline_mse),
        "n_iters": int(len(W_list)),
        "Xtr_res": Xtr_res,
        "Xstop_res": Xstop_res,
    }




# =========================
# Feature extraction
# =========================

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
    feature_mode: str,
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

        codes_all = []
        for start in range(0, X.shape[0], encode_batch):
            end = min(X.shape[0], start + encode_batch)
            x = X[start:end]
            _, codes = sae.encode(x)
            codes_all.append(codes.float())
        Xs = torch.cat(codes_all, dim=0)
        return Xs, Y

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def plot_inlp_metrics(inlp_results, out_path_prefix: Optional[str] = None):
    """
    Plot r2_mean and mse across INLP iterations.
    Saves figures if out_path_prefix is provided, otherwise just shows them.
    """
    import matplotlib.pyplot as plt

    iters = [p["iter"] for p in inlp_results["probes"]]
    r2 = [p["stop_metrics"]["r2_mean"] for p in inlp_results["probes"]]
    mse = [p["stop_metrics"]["mse"] for p in inlp_results["probes"]]

    # ---- R2 plot ----
    plt.figure()
    plt.plot(iters, r2, marker="o")
    plt.axhline(0.0, linestyle="--", label="R2 = 0 (chance)")
    plt.xlabel("INLP iteration")
    plt.ylabel("R2 (mean over action dims)")
    plt.title("INLP: R2 decay on stop set")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"inlp_r2.png", bbox_inches="tight") # {out_path_prefix}_
    plt.close()

    # ---- MSE plot ----
    baseline_mse = inlp_results["baseline_mse_stop"]

    plt.figure()
    plt.plot(iters, mse, marker="o")
    plt.axhline(baseline_mse, linestyle="--", label="Baseline MSE (mean predictor)")
    plt.xlabel("INLP iteration")
    plt.ylabel("MSE")
    plt.title("INLP: MSE increase on stop set")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"inlp_mse.png", bbox_inches="tight") # {out_path_prefix}_
    plt.close()


# =========================
# Main training / eval
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt_path", type=str, default="./checkpoints/sae_layer11_k10_c16000.pt",
                    help="SAE checkpoint path")
    ap.add_argument("--data_root", type=str, default="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero")
    ap.add_argument("--activations_root", type=str, default="/n/netscratch/sham_lab/Lab/chloe00/pi0_activations")
    ap.add_argument("--layer_idx", type=int, default=11)

    ap.add_argument("--feature_mode", type=str, default="raw", choices=["sae", "raw"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--encode_batch", type=int, default=8192)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--ridge_lambda", type=float, default=1.0, help="Ridge penalty (try 0.1, 1, 10, 100)")
    ap.add_argument("--standardize_x", action="store_true", help="Z-score features using train split stats")
    ap.add_argument("--standardize_y", action="store_true", help="Z-score targets using train split stats")

    # INLP / iterative nullspace projection
    ap.add_argument("--use_inlp", action="store_true",
                    help="Run INLP-style iterative nullspace projection (regression) before final probe training/eval")
    ap.add_argument("--inlp_max_iters", type=int, default=50)
    ap.add_argument("--inlp_stop_on", type=str, default="r2", choices=["r2", "mse"])
    ap.add_argument("--inlp_r2_tol", type=float, default=0.0, help="Stop when stop-set r2_mean <= this (baseline ~ 0)")
    ap.add_argument("--inlp_mse_ratio_tol", type=float, default=1.0,
                    help="Stop when stop-set MSE >= ratio * baseline_mean_predictor_MSE")
    ap.add_argument("--inlp_min_delta", type=float, default=1e-4, help="Stop when progress < this")
    ap.add_argument("--inlp_rcond", type=float, default=1e-6, help="SVD cutoff for basis extraction")
    ap.add_argument("--inlp_patience", type=int, default=3)

    ap.add_argument("--out_path", type=str, default="probe_results.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    # ---- load SAE checkpoint if needed ----
    sae = None
    d_expected = None

    if args.feature_mode == "sae":
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        d_expected = int(ckpt["d"])
        nb_concepts = int(ckpt["nb_concepts"])

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
        # If raw, set expected dim.
        d_expected = 1024
        print(f"Raw probe mode. Using d_expected={d_expected}.")

    # ---- index episodes ----
    episodes = index_libero_dataset(
        data_root=args.data_root,
        activations_root=args.activations_root,
        groups=("spatial",),
    )

    usable = [
        ep for ep in episodes
        if ep.act_path is not None and os.path.exists(ep.act_path)
        and ep.actions_path is not None and os.path.exists(ep.actions_path)
    ]
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
    if len(val_eps) > 0:
        Xva, Yva = extract_features_and_targets(
            val_eps, layer_idx=args.layer_idx, d_expected=d_expected,
            feature_mode=args.feature_mode, sae=sae, device=args.device,
            encode_batch=args.encode_batch,
        )
    else:
        Xva, Yva = None, None

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

    # ---- choose stopping set for INLP ----
    if Xva is not None and Xva.numel() > 0:
        Xstop, Ystop = Xva, Yva
        stop_name = "val"
    else:
        Xstop, Ystop = Xte, Yte
        stop_name = "test"

    # ---- run INLP (optional) ----
    inlp_results = None
    if args.use_inlp:
        inlp_results = inlp_ridge_regression(
            Xtr=Xtr, Ytr=Ytr,
            Xstop=Xstop, Ystop=Ystop,
            lam=args.ridge_lambda,
            max_iters=args.inlp_max_iters,
            stop_on=args.inlp_stop_on,
            r2_tol=args.inlp_r2_tol,
            mse_ratio_tol=args.inlp_mse_ratio_tol,
            # min_delta=args.inlp_min_delta,
            rcond=args.inlp_rcond,
            patience=args.inlp_patience,
        )

        # Replace features with residualized ones for downstream probe training/eval
        Xtr = inlp_results["Xtr_res"]
        if stop_name == "val":
            Xva = inlp_results["Xstop_res"]
        else:
            Xte = inlp_results["Xstop_res"]

        print(
            f"[INLP] iters={inlp_results['n_iters']} stop_set={stop_name} "
            f"baseline_mse={inlp_results['baseline_mse_stop']:.6f}"
        )

        plot_inlp_metrics(inlp_results, out_path_prefix=None)

    # ---- train ridge probe on (possibly residualized) features ----
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

    # ---- probe dimensionality diagnostics (single-shot final probe) ----
    W_for_dim, b_for_dim = unstandardize_probe_weights(W, b, x_mu, x_std)

    per_action_dim = {}
    for i, name in enumerate(ACTION_NAMES):
        wi = W_for_dim[:, i]
        per_action_dim[name] = {
            "participation_ratio": participation_ratio(wi),
            "topk_90pct_energy": topk_energy_count(wi, frac=0.90),
            "topk_95pct_energy": topk_energy_count(wi, frac=0.95),
            "l2_norm": float(torch.linalg.vector_norm(wi).item()),
        }

    probe_dim_summary = {
        "W_space": "raw_X" if args.standardize_x else "model_space",
        "stable_rank": stable_rank_matrix(W_for_dim),
        "effective_rank_entropy": effective_rank_entropy(W_for_dim),
        "spectrum": torch.linalg.svdvals(W_for_dim).detach().cpu().numpy().tolist(),
        "k_90pct_spectral_energy": stacked_spectrum_stats(W_for_dim).get("k_90pct_spectral_energy", 0),
    }

    # ---- INLP compactness metrics (if used) ----
    inlp_metrics = None
    if inlp_results is not None:
        # Two views:
        #   (1) W_stack: concatenation of ridge probes (D x 7*T)
        #   (2) U_cat: concatenation of removed bases (D x sum r_t)  [often most "INLP-faithful"]
        W_stack = inlp_results["W_stack"]
        U_cat = inlp_results["U_cat"]

        if args.standardize_x:
            # unstandardize each W^(t), restack
            W_list_raw = []
            for Wt in inlp_results["W_list"]:
                Wt_raw, _ = unstandardize_probe_weights(Wt, torch.zeros(7, device=Wt.device), x_mu, x_std)
                W_list_raw.append(Wt_raw)
            W_stack_raw = torch.cat(W_list_raw, dim=1) if len(W_list_raw) else W_stack

            inlp_metrics = {
                "stop_set": stop_name,
                "baseline_mse_stop": inlp_results["baseline_mse_stop"],
                "n_iters": inlp_results["n_iters"],
                "per_iter": inlp_results["probes"],
                "W_stack_space": "raw_X",
                "W_stack_stats": stacked_spectrum_stats(W_stack_raw),
                "U_removed_space": "model_space",  # U comes from standardized space if you standardized X
                "U_removed_stats": stacked_spectrum_stats(U_cat),
            }
        else:
            inlp_metrics = {
                "stop_set": stop_name,
                "baseline_mse_stop": inlp_results["baseline_mse_stop"],
                "n_iters": inlp_results["n_iters"],
                "per_iter": inlp_results["probes"],
                "W_stack_space": "model_space",
                "W_stack_stats": stacked_spectrum_stats(W_stack),
                "U_removed_space": "model_space",
                "U_removed_stats": stacked_spectrum_stats(U_cat),
            }

    # ---- assemble results ----
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
            "use_inlp": bool(args.use_inlp),
            "inlp_max_iters": args.inlp_max_iters,
            "inlp_stop_on": args.inlp_stop_on,
            "inlp_r2_tol": args.inlp_r2_tol,
            "inlp_mse_ratio_tol": args.inlp_mse_ratio_tol,
            "inlp_min_delta": args.inlp_min_delta,
            "inlp_rcond": args.inlp_rcond,
        },
        "splits": {},
        "weights": {
            "W_shape": list(W.shape),
            "b_shape": list(b.shape),
        },
        "probe_dimensionality": {
            "summary": probe_dim_summary,
            "per_action": per_action_dim,
        },
    }

    results["splits"]["train"] = eval_split("train", Xtr, Ytr)
    if Xva is not None and Xva.numel() > 0:
        results["splits"]["val"] = eval_split("val", Xva, Yva)
    results["splits"]["test"] = eval_split("test", Xte, Yte)

    if inlp_metrics is not None:
        results["inlp"] = inlp_metrics

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- prints ----
    print(f"\nWrote results to: {args.out_path}")
    print("Test R2 per dim:")
    for k, v in results["splits"]["test"]["r2_per_dim"].items():
        print(f"  {k:>7s}: {v: .4f}")
    print(f"Test R2 mean: {results['splits']['test']['r2_mean']:.4f}")
    print(f"Test MSE: {results['splits']['test']['mse']:.6f}")

    print("\n=== Final Probe Dimensionality Diagnostics ===")
    summary = results["probe_dimensionality"]["summary"]
    print(f"Global probe: W ∈ R^{W.shape[0]}×{W.shape[1]}")
    print(f"  Stable rank        : {summary['stable_rank']:.2f}")
    print(f"  Effective rank (H) : {summary['effective_rank_entropy']:.2f}")
    print(f"  k@90% spec energy  : {summary['k_90pct_spectral_energy']}")
    print(f"  Weight space       : {summary['W_space']}")

    print("\nPer-action effective dimensionality:")
    for name in ACTION_NAMES:
        d = results["probe_dimensionality"]["per_action"][name]
        print(
            f"  {name:>7s} | "
            f"d_eff(PR) ≈ {d['participation_ratio']:.2f}, "
            f"k_90% = {d['topk_90pct_energy']:4d}, "
            f"k_95% = {d['topk_95pct_energy']:4d}"
        )

    if "inlp" in results:
        print("\n=== INLP (Iterative Nullspace Projection) Diagnostics ===")
        print(f"Stop set: {results['inlp']['stop_set']}")
        print(f"INLP iters: {results['inlp']['n_iters']}")
        print(f"Baseline MSE (stop): {results['inlp']['baseline_mse_stop']:.6f}")

        ws = results["inlp"]["W_stack_stats"]
        us = results["inlp"]["U_removed_stats"]

        print("\n[W_stack] (concatenated probes) spectrum summary:")
        print(f"  shape              : {ws['shape']}")
        print(f"  stable rank        : {ws['stable_rank']:.2f}")
        print(f"  effective rank (H) : {ws['effective_rank_entropy']:.2f}")
        print(f"  k@90% spec energy  : {ws['k_90pct_spectral_energy']}")

        print("\n[U_removed] (concatenated removed bases) summary:")
        print(f"  shape              : {us['shape']}")
        print(f"  stable rank        : {us['stable_rank']:.2f}")
        print(f"  effective rank (H) : {us['effective_rank_entropy']:.2f}")
        print(f"  k@90% spec energy  : {us['k_90pct_spectral_energy']}")

        # Optional: show per-iter quick table
        print("\nPer-iteration stop metrics:")
        for it in results["inlp"]["per_iter"]:
            r2m = it["stop_metrics"]["r2_mean"]
            mse = it["stop_metrics"]["mse"]
            ur = it["U_rank"]
            print(f"  iter={it['iter']:02d}  U_rank={ur:4d}  r2_mean={r2m:+.4f}  mse={mse:.6f}")


if __name__ == "__main__":
    main()