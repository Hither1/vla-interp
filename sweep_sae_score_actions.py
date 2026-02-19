#!/usr/bin/env python3
import os, glob, json, re, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from overcomplete.sae import TopKSAE  
from utils import index_libero_dataset, ACTION_NAMES, split_episodes, load_actions, ridge_closed_form, r2_score, standardize_fit, standardize_apply


# ---------- MCC ----------
def matthews_corrcoef_binary(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    """
    y_true, y_pred: (N,) bool/int {0,1}
    """
    y_true = y_true.to(torch.int64)
    y_pred = y_pred.to(torch.int64)

    tp = ((y_true == 1) & (y_pred == 1)).sum().float()
    tn = ((y_true == 0) & (y_pred == 0)).sum().float()
    fp = ((y_true == 0) & (y_pred == 1)).sum().float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().float()

    num = tp * tn - fp * fn
    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp_min(eps)
    return float((num / den).item())

def mcc_per_dim(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mode: str,
    train_thresholds: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    y_true, y_pred: (N, 7) continuous
    mode:
      - "sign": threshold at 0
      - "median": threshold at per-dim median from train_thresholds
    train_thresholds: (7,) if mode=="median"
    """
    out = {}
    for i, name in enumerate(ACTION_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        if mode == "sign":
            yt_b = (yt > 0)
            yp_b = (yp > 0)
        elif mode == "median":
            assert train_thresholds is not None
            thr = train_thresholds[i]
            yt_b = (yt > thr)
            yp_b = (yp > thr)
        else:
            raise ValueError(f"Unknown MCC mode: {mode}")

        out[name] = matthews_corrcoef_binary(yt_b, yp_b)
    out["mean"] = float(np.mean([out[n] for n in ACTION_NAMES]))
    return out

# ---------- replan downsampling ----------
def downsample_replan(X: np.ndarray, Y: np.ndarray, replan_freq: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (T, d), Y: (T, 7)
    Keep frames 0, replan_freq, 2*replan_freq, ...
    """
    T = min(X.shape[0], Y.shape[0])
    if T <= 0:
        return X[:0], Y[:0]
    idxs = np.arange(0, T, replan_freq, dtype=np.int64)
    return X[idxs], Y[idxs]

# ---------- EDIT your load_layer_acts to apply replan downsampling ----------
def load_layer_acts(ep, layer_idx: int, d_expected: int, replan_freq: int) -> Tuple[np.ndarray, np.ndarray]:
    acts = load_actions(ep.actions_path).astype(np.float32)  # (T_a, 7)
    A = np.load(ep.act_path).astype(np.float32)

    if A.ndim == 4:
        A = A.squeeze(-2)
    if A.ndim != 3:
        raise ValueError(f"Unexpected activation shape {A.shape} in {ep.act_path}")

    T, num_layers, d = A.shape
    if d != d_expected:
        raise ValueError(f"d mismatch: got {d} in {ep.act_path}, expected {d_expected}")
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers-1}]")

    T_use = min(T, acts.shape[0])
    if T_use <= 0:
        return np.zeros((0, d), np.float32), np.zeros((0, 7), np.float32)

    X_layer = A[:T_use, layer_idx, :]  # (T_use, d)
    Y = acts[:T_use, :]               # (T_use, 7)

    # --- replan downsample ---
    X_layer, Y = downsample_replan(X_layer, Y, replan_freq=replan_freq)

    return X_layer, Y

# ---------- EDIT extract_features_and_targets to pass replan_freq ----------
@torch.no_grad()
def extract_features_and_targets(
    episodes,
    layer_idx: int,
    d_expected: int,
    feature_mode: str,
    sae: Optional[TopKSAE],
    device: str,
    replan_freq: int,
    encode_batch: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_list, Y_list = [], []

    for ep in episodes:
        if ep.act_path is None or (not os.path.exists(ep.act_path)):
            continue
        if ep.actions_path is None or (not os.path.exists(ep.actions_path)):
            continue

        X_layer, Y = load_layer_acts(ep, layer_idx=layer_idx, d_expected=d_expected, replan_freq=replan_freq)
        if X_layer.shape[0] == 0:
            continue

        X_list.append(X_layer)
        Y_list.append(Y)

    if len(X_list) == 0:
        raise RuntimeError("No usable episodes found in this split (check matching logic / paths).")

    X = torch.from_numpy(np.concatenate(X_list, axis=0).astype(np.float32)).to(device)
    Y = torch.from_numpy(np.concatenate(Y_list, axis=0).astype(np.float32)).to(device)

    if feature_mode == "raw":
        return X, Y

    if feature_mode == "sae":
        if sae is None:
            raise ValueError("feature_mode='sae' but sae is None")

        codes_all = []
        for start in range(0, X.shape[0], encode_batch):
            end = min(X.shape[0], start + encode_batch)
            x = X[start:end]
            _, codes = sae.encode(x)   # codes: (B, nb_concepts)
            codes_all.append(codes.float())
        Z = torch.cat(codes_all, dim=0)
        return Z, Y

    raise ValueError(f"Unknown feature_mode: {feature_mode}")

# ---------- scoring ----------
def fit_linear_readout(Ztr: torch.Tensor, Ytr: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # lam=0.0 -> OLS (still stable if Z is full rank; else consider lam small)
    W, b = ridge_closed_form(Ztr, Ytr, lam=lam)
    return W, b

def predict(Z: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return Z @ W + b

def eval_regression(Z: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> Dict:
    Yhat = predict(Z, W, b)
    mse = torch.mean((Y - Yhat) ** 2).item()
    r2_dim = r2_score(Y, Yhat).detach().cpu().numpy().tolist()
    return {
        "mse": float(mse),
        "r2_per_dim": {ACTION_NAMES[i]: float(r2_dim[i]) for i in range(7)},
        "r2_mean": float(np.mean(r2_dim)),
    }

# -------------------------
# Main sweep
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", type=str, default="./checkpoints/BatchTopKSAE/sae_libero_object_layer11_k*_c*.pt",
                    help="Glob for SAE ckpts, e.g. ./checkpoints/BatchTopKSAE/sae_libero_object_layer11_k*_c*.pt")
    ap.add_argument("--data_root", type=str, default="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero")
    ap.add_argument("--activations_root", type=str, default="/n/netscratch/sham_lab/Lab/chloe00/pi0_activations")
    ap.add_argument("--group", type=str, default="object", choices=["10", "goal", "object", "spatial"])
    ap.add_argument("--layer_idx", type=int, default=11)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--replan_freq", type=int, default=5)
    ap.add_argument("--encode_batch", type=int, default=8192)

    ap.add_argument("--readout_lambda", type=float, default=1e-3,
                    help="0.0 = OLS, >0 ridge")
    ap.add_argument("--standardize_z", action="store_true",
                    help="Z-score codes/features using train stats")
    ap.add_argument("--standardize_y", action="store_true",
                    help="Z-score targets using train stats")

    ap.add_argument("--mcc_mode", type=str, default="sign", choices=["sign","median"])
    ap.add_argument("--out_json", type=str, default="sweep_results.json")
    ap.add_argument("--out_csv", type=str, default="sweep_results.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # ---- index episodes ----
    episodes = index_libero_dataset(
        data_root=args.data_root,
        activations_root=args.activations_root,
        groups=(args.group,),
    )

    usable = [ep for ep in episodes
              if ep.act_path is not None and os.path.exists(ep.act_path)
              and ep.actions_path is not None and os.path.exists(ep.actions_path)]
    if len(usable) == 0:
        raise RuntimeError("No usable episodes with both activation and actions found.")

    train_eps, val_eps, test_eps = split_episodes(usable, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    print(f"Episodes split: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if len(ckpts) == 0:
        raise RuntimeError(f"No ckpts matched: {args.ckpt_glob}")

    all_rows = []
    all_results = {"config": vars(args), "runs": []}

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        d_expected = ckpt["d"]
        nb_concepts = ckpt["nb_concepts"]
        top_k = ckpt["top_k"]

        # load SAE
        sae = TopKSAE(d_expected, nb_concepts=nb_concepts, top_k=top_k, device="cpu")
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval().to(args.device)

        # extract codes and y
        Ztr, Ytr = extract_features_and_targets(
            train_eps, layer_idx=args.layer_idx, d_expected=d_expected,
            feature_mode="sae", sae=sae, device=args.device,
            replan_freq=args.replan_freq, encode_batch=args.encode_batch,
        )
        Zva, Yva = extract_features_and_targets(
            val_eps, layer_idx=args.layer_idx, d_expected=d_expected,
            feature_mode="sae", sae=sae, device=args.device,
            replan_freq=args.replan_freq, encode_batch=args.encode_batch,
        ) if len(val_eps) else (None, None)
        Zte, Yte = extract_features_and_targets(
            test_eps, layer_idx=args.layer_idx, d_expected=d_expected,
            feature_mode="sae", sae=sae, device=args.device,
            replan_freq=args.replan_freq, encode_batch=args.encode_batch,
        )

        # standardize Z (train only)
        z_mu = z_std = None
        if args.standardize_z:
            z_mu, z_std = standardize_fit(Ztr)
            Ztr = standardize_apply(Ztr, z_mu, z_std)
            if Zva is not None:
                Zva = standardize_apply(Zva, z_mu, z_std)
            Zte = standardize_apply(Zte, z_mu, z_std)

        # standardize Y (train only) [optional; usually leave off if you care about action scales]
        y_mu = y_std = None
        if args.standardize_y:
            y_mu, y_std = standardize_fit(Ytr)
            Ytr = standardize_apply(Ytr, y_mu, y_std)
            if Yva is not None:
                Yva = standardize_apply(Yva, y_mu, y_std)
            Yte = standardize_apply(Yte, y_mu, y_std)

        # train linear readout on concepts
        W, b = fit_linear_readout(Ztr, Ytr, lam=args.readout_lambda)

        # regression metrics
        tr_reg = eval_regression(Ztr, Ytr, W, b)
        va_reg = eval_regression(Zva, Yva, W, b) if Zva is not None else None
        te_reg = eval_regression(Zte, Yte, W, b)

        # MCC thresholds from train (if median)
        train_thresholds = None
        if args.mcc_mode == "median":
            train_thresholds = torch.median(Ytr, dim=0).values  # (7,)

        # MCC on predictions in ORIGINAL scale of Ytr (if you standardized Y, this is in standardized units)
        Ytr_hat = predict(Ztr, W, b)
        Yte_hat = predict(Zte, W, b)
        tr_mcc = mcc_per_dim(Ytr, Ytr_hat, mode=args.mcc_mode, train_thresholds=train_thresholds)
        te_mcc = mcc_per_dim(Yte, Yte_hat, mode=args.mcc_mode, train_thresholds=train_thresholds)
        va_mcc = None
        if Zva is not None:
            Yva_hat = predict(Zva, W, b)
            va_mcc = mcc_per_dim(Yva, Yva_hat, mode=args.mcc_mode, train_thresholds=train_thresholds)

        run = {
            "ckpt_path": ckpt_path,
            "nb_concepts": int(nb_concepts),
            "top_k": int(top_k),
            "n_train": int(Ztr.shape[0]),
            "n_val": int(0 if Zva is None else Zva.shape[0]),
            "n_test": int(Zte.shape[0]),
            "train": {"reg": tr_reg, "mcc": tr_mcc},
            "val": None if va_reg is None else {"reg": va_reg, "mcc": va_mcc},
            "test": {"reg": te_reg, "mcc": te_mcc},
        }
        all_results["runs"].append(run)

        # flatten a CSV row (rank by val_r2_mean if available else test)
        val_r2 = None if va_reg is None else va_reg["r2_mean"]
        row = {
            "ckpt_path": ckpt_path,
            "nb_concepts": int(nb_concepts),
            "top_k": int(top_k),
            "train_r2_mean": tr_reg["r2_mean"],
            "val_r2_mean": (float("nan") if val_r2 is None else val_r2),
            "test_r2_mean": te_reg["r2_mean"],
            "train_mcc_mean": tr_mcc["mean"],
            "val_mcc_mean": (float("nan") if va_mcc is None else va_mcc["mean"]),
            "test_mcc_mean": te_mcc["mean"],
        }
        all_rows.append(row)

        print(f"[DONE] n={nb_concepts:5d} k={top_k:2d} | "
              f"val R2={row['val_r2_mean']:.4f} test R2={row['test_r2_mean']:.4f} | "
              f"val MCC={row['val_mcc_mean']:.4f} test MCC={row['test_mcc_mean']:.4f}")

    # pick best by val_r2_mean (fallback to test_r2_mean if no val)
    def key_fn(r):
        return r["val_r2_mean"] if np.isfinite(r["val_r2_mean"]) else r["test_r2_mean"]
    best = max(all_rows, key=key_fn)
    all_results["best_by_r2"] = best

    # write json
    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote JSON: {args.out_json}")

    # write csv
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote CSV: {args.out_csv}")

    print("\nBest (by val R2 if present):")
    for k, v in best.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()