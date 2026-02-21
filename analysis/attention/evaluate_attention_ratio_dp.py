#!/usr/bin/env python3
"""
LIBERO evaluation with visual/linguistic gradient attribution ratio for Diffusion Policy.

Since Diffusion Policy has no explicit attention mechanism (uses FiLM conditioning),
this script computes gradient-based attribution as a proxy for "how much the model
relies on visual vs. linguistic inputs":

  visual attribution   = ||∂L/∂img_features||²
  linguistic attribution = ||∂L/∂text_emb||²

where L = ||noise_pred||²_F at a single mid-diffusion timestep.

This is analogous to the visual/linguistic attention ratio computed for Cosmos and Pi0.5,
but uses input-gradient attribution instead of attention weights.

Usage:
  python evaluate_attention_ratio_dp.py \\
    --ckpt dp_scratch/ckpt_300.pt \\
    --task-suite libero_10 --num-episodes 5 \\
    --output-dir results/attention_ratio_dp
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "examples" / "libero")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from dp_scratch.model import DiffusionPolicy, IMAGENET_MEAN, IMAGENET_STD, _to_tensor_img

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ── Observation helpers ───────────────────────────────────────────────────────


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


def _build_dp_tensors(obs: dict, model: DiffusionPolicy, device: torch.device):
    """Convert raw LIBERO env obs to model input tensors (no flip, matches dp_eval.py)."""
    img = np.ascontiguousarray(obs["agentview_image"].copy())
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"].copy())

    img_t = _to_tensor_img(img).to(device)          # (3, 224, 224)
    wrist_t = _to_tensor_img(wrist).to(device)       # (3, 224, 224)

    # (1, n_obs_steps, 2, 3, 224, 224)
    imgs_single = torch.stack([img_t, wrist_t], dim=0)  # (2, 3, 224, 224)
    images = imgs_single.unsqueeze(0).unsqueeze(0).expand(
        1, model.n_obs_steps, -1, -1, -1, -1
    )

    state_np = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    state = state.unsqueeze(0).unsqueeze(0).expand(1, model.n_obs_steps, -1)  # (1, T, state_dim)

    return images, state


# ── Gradient attribution ──────────────────────────────────────────────────────


def compute_gradient_attribution_ratio(
    model: DiffusionPolicy,
    images: torch.Tensor,
    state: torch.Tensor,
    task_idx: torch.Tensor,
    t_frac: float = 0.5,
) -> Dict:
    """Compute visual / linguistic gradient attribution ratio.

    Runs a single forward+backward pass of the noise prediction network at a
    fixed diffusion timestep (default: middle of the schedule).  The gradient
    of ||noise_pred||²_F w.r.t. the (detached) image-feature and text-embedding
    vectors is used as a proxy for how much each modality influences the prediction.

    Args:
        model:     DiffusionPolicy in eval mode.
        images:    (1, n_obs_steps, n_cameras, 3, H, W) float [0, 1].
        state:     (1, n_obs_steps, state_dim) float.
        task_idx:  (1,) long tensor.
        t_frac:    Diffusion timestep as a fraction of num_train_timesteps.

    Returns:
        Dict with visual_fraction, linguistic_fraction, visual_linguistic_ratio, etc.
    """
    device = images.device
    B = images.shape[0]

    # Forward through vision encoder outside grad context first,
    # then detach and re-enable grad to isolate modality contributions.
    imgs_flat = images.reshape(
        B * model.n_obs_steps * model.n_cameras, 3, images.shape[-2], images.shape[-1]
    )
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    imgs_flat_norm = (imgs_flat - mean) / std

    with torch.enable_grad():
        # Encode image features, detach from vision encoder graph, track grads
        img_feat_raw = model.vision_enc(imgs_flat_norm)        # (B*T*C, 512)
        img_features = img_feat_raw.reshape(B, -1).detach().requires_grad_(True)

        # Encode text, detach, track grads
        text_raw = model.text_proj(model.text_embeddings[task_idx])  # (B, task_embed_dim)
        text_emb = text_raw.detach().requires_grad_(True)

        # State doesn't need grad tracking (not a modality of interest)
        state_norm = model.normalize_state(state).reshape(B, -1).detach()

        global_cond = torch.cat([img_features, state_norm, text_emb], dim=-1)

        # Single denoising step at t = T * t_frac
        x = torch.randn(B, model.horizon, model.action_dim, device=device)
        t_val = int(model.num_train_timesteps * t_frac)
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, t_batch.float(), global_cond)

        loss = noise_pred.pow(2).mean()
        loss.backward()

    visual_grad_sq = float(img_features.grad.norm() ** 2) if img_features.grad is not None else 0.0
    text_grad_sq = float(text_emb.grad.norm() ** 2) if text_emb.grad is not None else 0.0
    total = visual_grad_sq + text_grad_sq

    return {
        "visual_grad_sq": visual_grad_sq,
        "text_grad_sq": text_grad_sq,
        "visual_fraction": visual_grad_sq / max(total, 1e-10),
        "linguistic_fraction": text_grad_sq / max(total, 1e-10),
        "visual_linguistic_ratio": visual_grad_sq / max(text_grad_sq, 1e-10),
        "diffusion_t_frac": t_frac,
    }


# ── Episode / aggregation ─────────────────────────────────────────────────────


def summarize_episode_ratios(step_results: List[Dict]) -> Dict:
    if not step_results:
        return {}
    ratios = [r["visual_linguistic_ratio"] for r in step_results
              if np.isfinite(r["visual_linguistic_ratio"])]
    visual_fracs = [r["visual_fraction"] for r in step_results]
    ling_fracs = [r["linguistic_fraction"] for r in step_results]
    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
            "min": float(np.min(ratios)) if ratios else 0.0,
            "max": float(np.max(ratios)) if ratios else 0.0,
        },
        "visual_fraction": {
            "mean": float(np.mean(visual_fracs)),
            "std": float(np.std(visual_fracs)),
        },
        "linguistic_fraction": {
            "mean": float(np.mean(ling_fracs)),
            "std": float(np.std(ling_fracs)),
        },
        "num_steps": len(step_results),
    }


def run_episode(
    env,
    model: DiffusionPolicy,
    task_description: str,
    initial_state: np.ndarray,
    max_steps: int,
    device: torch.device,
    replan_steps: int,
    t_frac: float,
) -> Dict:
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    action_plan: collections.deque = collections.deque()
    step_ratio_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        is_replan = not action_plan
        if is_replan:
            images, state_t = _build_dp_tensors(obs, model, device)
            task_idx = torch.tensor(
                [model._prompt_to_idx(task_description)], dtype=torch.long, device=device
            )

            # Gradient attribution (requires grad-enabled context)
            try:
                ratio_result = compute_gradient_attribution_ratio(
                    model, images, state_t, task_idx, t_frac=t_frac
                )
                ratio_result["step"] = t
                step_ratio_results.append(ratio_result)
                log.info(
                    f"  t={t}: ratio={ratio_result['visual_linguistic_ratio']:.3f}  "
                    f"visual={ratio_result['visual_fraction']:.3f}  "
                    f"linguistic={ratio_result['linguistic_fraction']:.3f}"
                )
            except Exception as e:
                log.warning(f"  t={t}: gradient attribution failed: {e}")

            # Policy action (no grad needed)
            with torch.no_grad():
                actions_t = model.predict_action(images, state_t, task_idx)
            action_chunk = actions_t[0].cpu().numpy()
            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        if done:
            break
        t += 1

    return {
        "success": bool(done),
        "num_steps": t,
        "summary": summarize_episode_ratios(step_ratio_results),
        "step_ratio_results": step_ratio_results,
    }


# ── LIBERO helpers ────────────────────────────────────────────────────────────


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO eval with gradient attribution ratio (Diffusion Policy)"
    )

    # Model
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to DP checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--replan-steps", type=int, default=16,
                        help="Action chunk steps before replanning")
    parser.add_argument("--t-frac", type=float, default=0.5,
                        help="Diffusion timestep fraction for gradient computation (0=clean, 1=noise)")

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=list(TASK_MAX_STEPS.keys()))
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run only this task ID (default: all tasks)")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Output
    parser.add_argument("--output-dir", type=str, default="results/attention_ratio_dp")

    args = parser.parse_args()

    device = torch.device(args.device)
    max_steps = TASK_MAX_STEPS[args.task_suite]

    log.info("Loading Diffusion Policy checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiffusionPolicy(task_descs=ckpt["task_descs"]).to(device)
    model.set_norm_stats(**ckpt["stats"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Model loaded.")

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks

    task_ids = [args.task_id] if args.task_id is not None else list(range(num_tasks))
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        log.info(f"\n{'=' * 70}")
        log.info(f"Task {task_id}: {task_description}")
        log.info(f"{'=' * 70}")

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

            result = run_episode(
                env=env,
                model=model,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                max_steps=max_steps,
                device=device,
                replan_steps=args.replan_steps,
                t_frac=args.t_frac,
            )
            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

            log.info(
                f"  Episode done: success={result['success']}  steps={result['num_steps']}"
            )
            if result["summary"]:
                r = result["summary"]["visual_linguistic_ratio"]
                log.info(
                    f"  Ratio summary: mean={r['mean']:.3f}  std={r['std']:.3f}  "
                    f"median={r['median']:.3f}"
                )

        env.close()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = pathlib.Path(args.output_dir) / f"ratio_results_{args.task_suite}.json"
    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_ratio_results"}
        entry["step_ratio_results"] = [
            {
                "step": int(s["step"]),
                "visual_linguistic_ratio": float(s["visual_linguistic_ratio"]),
                "visual_fraction": float(s["visual_fraction"]),
                "linguistic_fraction": float(s["linguistic_fraction"]),
                "visual_grad_sq": float(s["visual_grad_sq"]),
                "text_grad_sq": float(s["text_grad_sq"]),
            }
            for s in r.get("step_ratio_results", [])
        ]
        serializable.append(entry)

    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {out_file}")

    # ── Task-level summary ────────────────────────────────────────────────────
    for task_id in task_ids:
        task_res = [r for r in all_results if r.get("task_id") == task_id]
        if not task_res:
            continue
        success_rate = float(np.mean([r["success"] for r in task_res]))
        all_step_ratios = [
            s
            for r in task_res
            for s in r.get("step_ratio_results", [])
            if np.isfinite(s["visual_linguistic_ratio"])
        ]
        if all_step_ratios:
            mean_ratio = float(np.mean([s["visual_linguistic_ratio"] for s in all_step_ratios]))
            mean_vis = float(np.mean([s["visual_fraction"] for s in all_step_ratios]))
            mean_ling = float(np.mean([s["linguistic_fraction"] for s in all_step_ratios]))
            task_desc = task_res[0]["task_description"]
            log.info(
                f"Task {task_id} ({task_desc[:50]}): "
                f"success={success_rate:.1%}  "
                f"ratio={mean_ratio:.3f}  "
                f"visual={mean_vis:.3f}  linguistic={mean_ling:.3f}"
            )


if __name__ == "__main__":
    main()
