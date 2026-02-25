#!/usr/bin/env python3
"""
LIBERO evaluation with GradCAM saliency IoU for Diffusion Policy.

Since Diffusion Policy has no explicit attention mechanism, this script uses
GradCAM over multiple ResNet-18 visual encoder layers as a proxy for spatial
visual attention.  Heatmaps from each layer are averaged and compared against
ground-truth object segmentation masks from the LIBERO environment.

GradCAM computation (at each replan step, per layer):
  1. Register a forward hook on model.vision_enc.net.<layer_name>
  2. Single forward pass of noise_pred_net at t = T * t_frac
  3. Backward of ||noise_pred||²_F to populate gradients at that layer
  4. GradCAM = ReLU( channel_mean(grad * activation) )
  5. Upsample to 256×256 and normalise to [0, 1]
  6. Average heatmaps across all layers, then compute IoU with segmentation mask

Usage:
  python evaluate_attention_iou_dp.py \\
    --ckpt dp_scratch/ckpt_300.pt \\
    --task-suite libero_10 --num-episodes 5 \\
    --layers layer2 layer3 layer4 \\
    --output-dir results/attention_iou_dp
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_ATTN_DIR = pathlib.Path(__file__).resolve().parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "examples" / "libero"), str(_ATTN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import SegmentationRenderEnv

from dp_scratch.model import DiffusionPolicy, IMAGENET_MEAN, IMAGENET_STD, _to_tensor_img
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image
from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    find_segmentation_key,
    visualize_attention_vs_segmentation,
    overlay_heatmap,
    DEFAULT_THRESHOLD_METHODS,
)

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


def _build_dp_tensors(obs: dict, model: DiffusionPolicy, device: torch.device, vis_cfg=None):
    """Convert raw LIBERO env obs to model input tensors (no flip, matches dp_eval.py)."""
    img = np.ascontiguousarray(obs["agentview_image"].copy())
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"].copy())
    if vis_cfg is not None and vis_cfg.mode != "none":
        img = perturb_image(img, vis_cfg)
        wrist = perturb_image(wrist, vis_cfg)

    img_t = _to_tensor_img(img).to(device)    # (3, 224, 224)
    wrist_t = _to_tensor_img(wrist).to(device) # (3, 224, 224)

    imgs_single = torch.stack([img_t, wrist_t], dim=0)  # (2, 3, 224, 224)
    images = imgs_single.unsqueeze(0).unsqueeze(0).expand(
        1, model.n_obs_steps, -1, -1, -1, -1
    )  # (1, n_obs_steps, 2, 3, 224, 224)

    state_np = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    state = state.unsqueeze(0).unsqueeze(0).expand(1, model.n_obs_steps, -1)

    return images, state


# ── GradCAM ──────────────────────────────────────────────────────────────────


def compute_gradcam_heatmap(
    model: DiffusionPolicy,
    images: torch.Tensor,
    state: torch.Tensor,
    task_idx: torch.Tensor,
    camera_idx: int = 0,
    t_frac: float = 0.5,
    target_size: int = LIBERO_ENV_RESOLUTION,
    layer_name: str = "layer4",
) -> np.ndarray:
    """Compute GradCAM saliency map for the agentview image from one ResNet layer.

    Args:
        model:       DiffusionPolicy in eval mode.
        images:      (1, n_obs_steps, n_cameras, 3, H, W) float [0, 1].
        state:       (1, n_obs_steps, state_dim) float.
        task_idx:    (1,) long tensor.
        camera_idx:  Which camera to visualise (0=agentview, 1=wrist).
        t_frac:      Diffusion timestep fraction for grad computation.
        target_size: Output heatmap size (square).
        layer_name:  ResNet-18 layer to hook, e.g. "layer2", "layer3", "layer4".

    Returns:
        Normalised heatmap in [0, 1], shape (target_size, target_size).
    """
    device = images.device
    B = images.shape[0]

    # Flat camera index into (B*n_obs_steps*n_cameras) batch:
    # layout: [t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...] for B=1
    # Use the most recent observation step.
    flat_cam_idx = (model.n_obs_steps - 1) * model.n_cameras + camera_idx

    resnet_layer = getattr(model.vision_enc.net, layer_name)

    # Storage for hook
    saved_act: List[Optional[torch.Tensor]] = [None]
    saved_grad: List[Optional[torch.Tensor]] = [None]

    def _fwd_hook(module, inp, out):
        saved_act[0] = out
        out.register_hook(lambda g: saved_grad.__setitem__(0, g))

    handle = resnet_layer.register_forward_hook(_fwd_hook)

    try:
        model.zero_grad()
        with torch.enable_grad():
            imgs_flat = images.reshape(
                B * model.n_obs_steps * model.n_cameras, 3, images.shape[-2], images.shape[-1]
            )
            mean = IMAGENET_MEAN.to(device)
            std = IMAGENET_STD.to(device)
            imgs_flat_norm = (imgs_flat - mean) / std

            img_feat_all = model.vision_enc(imgs_flat_norm)  # (B*T*C, 512)
            img_feat_batch = img_feat_all.reshape(B, -1)

            state_norm = model.normalize_state(state).reshape(B, -1)
            text_emb = model.text_proj(model.text_embeddings[task_idx])
            global_cond = torch.cat([img_feat_batch, state_norm, text_emb], dim=-1)

            x = torch.randn(B, model.horizon, model.action_dim, device=device)
            t_val = int(model.num_train_timesteps * t_frac)
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise_pred = model.noise_pred_net(x, t_batch.float(), global_cond)

            loss = noise_pred.pow(2).mean()
            loss.backward()
    finally:
        handle.remove()

    if saved_act[0] is None or saved_grad[0] is None:
        raise RuntimeError(f"GradCAM hooks did not fire — check {layer_name} hook registration")

    act = saved_act[0].detach()   # (B*T*C, C_layer, H_layer, W_layer)
    grad = saved_grad[0].detach() # same shape

    act_cam = act[flat_cam_idx]   # (C_layer, H_layer, W_layer)
    grad_cam = grad[flat_cam_idx] # same shape

    # Global-average-pooled gradients as channel weights
    weights = grad_cam.mean(dim=(1, 2))                      # (C_layer,)
    cam = (weights[:, None, None] * act_cam).sum(dim=0)      # (H_layer, W_layer)
    cam = torch.relu(cam).cpu().numpy()

    # Normalise and upsample to target_size
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    cam = cv2.resize(cam.astype(np.float32), (target_size, target_size),
                     interpolation=cv2.INTER_LINEAR)
    # Clip-normalise to spread the dynamic range
    lo, hi = float(np.percentile(cam, 5)), float(np.percentile(cam, 95))
    cam = np.clip(cam, lo, hi)
    cam = (cam - lo) / (hi - lo + 1e-8)
    return cam.astype(np.float32)


def compute_multilayer_gradcam(
    model: DiffusionPolicy,
    images: torch.Tensor,
    state: torch.Tensor,
    task_idx: torch.Tensor,
    layers: List[str],
    camera_idx: int = 0,
    t_frac: float = 0.5,
    target_size: int = LIBERO_ENV_RESOLUTION,
) -> np.ndarray:
    """Average GradCAM heatmaps over multiple ResNet layers.

    Args:
        layers: List of ResNet-18 layer names, e.g. ["layer2", "layer3", "layer4"].

    Returns:
        Averaged, normalised heatmap in [0, 1], shape (target_size, target_size).
    """
    heatmaps = []
    for layer_name in layers:
        try:
            h = compute_gradcam_heatmap(
                model, images, state, task_idx,
                camera_idx=camera_idx, t_frac=t_frac,
                target_size=target_size, layer_name=layer_name,
            )
            heatmaps.append(h)
        except Exception as e:
            log.warning(f"    GradCAM layer {layer_name} failed: {e}")

    if not heatmaps:
        raise RuntimeError("All GradCAM layers failed")

    avg = np.mean(np.stack(heatmaps, axis=0), axis=0)
    # Re-normalise after averaging
    lo, hi = float(np.percentile(avg, 5)), float(np.percentile(avg, 95))
    avg = np.clip(avg, lo, hi)
    avg = (avg - lo) / (hi - lo + 1e-8)
    return avg.astype(np.float32)


# ── Segmentation helpers ──────────────────────────────────────────────────────


def _get_object_ids(env) -> Dict[str, int]:
    """Build {object_name: segmentation_id} for objects of interest."""
    obj_of_interest = getattr(env, "obj_of_interest", [])
    instance_to_id = getattr(env, "instance_to_id", {})

    object_ids: Dict[str, int] = {}
    # Exact match first
    for obj in obj_of_interest:
        if obj in instance_to_id:
            object_ids[obj] = instance_to_id[obj]

    # Fuzzy fallback (common in libero_goal)
    if not object_ids:
        for obj in obj_of_interest:
            matches = [k for k in instance_to_id if obj.lower() in k.lower()]
            if matches:
                object_ids[obj] = instance_to_id[matches[0]]

    return object_ids


def _get_seg_mask(obs: dict, camera_name: str = "agentview") -> Optional[np.ndarray]:
    """Extract instance segmentation mask from obs dict."""
    key = find_segmentation_key(obs, camera_name)
    if key is None:
        return None
    mask = obs[key]
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


# ── Episode loop ──────────────────────────────────────────────────────────────


def run_episode(
    env,
    model: DiffusionPolicy,
    task_description: str,
    initial_state: np.ndarray,
    max_steps: int,
    device: torch.device,
    replan_steps: int,
    t_frac: float,
    layers: List[str],
    threshold_methods,
    save_viz: bool,
    output_dir: str,
    episode_prefix: str,
    vis_cfg=None,
    policy_cfg=None,
    policy_rng=None,
) -> Dict:
    obs = env.reset()
    obs = env.set_init_state(initial_state)
    if policy_cfg is not None and policy_rng is not None:
        apply_object_shift(env, policy_cfg, policy_rng)

    object_ids = _get_object_ids(env)
    if not object_ids:
        log.warning("  No objects of interest found in environment — IoU will be skipped")

    action_plan: collections.deque = collections.deque()
    step_iou_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        is_replan = not action_plan
        if is_replan:
            images, state_t = _build_dp_tensors(obs, model, device, vis_cfg=vis_cfg)
            task_idx = torch.tensor(
                [model._prompt_to_idx(task_description)], dtype=torch.long, device=device
            )

            # GradCAM heatmap averaged over all requested layers
            heatmap = None
            try:
                heatmap = compute_multilayer_gradcam(
                    model, images, state_t, task_idx,
                    layers=layers,
                    camera_idx=0, t_frac=t_frac, target_size=LIBERO_ENV_RESOLUTION,
                )
            except Exception as e:
                log.warning(f"  t={t}: GradCAM failed: {e}")

            # IoU against segmentation mask
            if heatmap is not None and object_ids:
                seg_mask = _get_seg_mask(obs, camera_name="agentview")
                if seg_mask is not None:
                    iou_result = compute_attention_object_iou(
                        attention_heatmap=heatmap,
                        segmentation_mask=seg_mask,
                        object_ids=object_ids,
                        threshold_methods=threshold_methods,
                    )
                    iou_result["step"] = t
                    step_iou_results.append(iou_result)

                    combined_iou = float(
                        iou_result["combined"].get("percentile_90", {}).get("iou", 0.0)
                    )
                    mass = float(iou_result["attention_mass"].get("_all_objects", 0.0))
                    pointing = bool(iou_result.get("pointing_hit", False))
                    log.info(
                        f"  t={t}: IoU={combined_iou:.3f}  "
                        f"attn_mass={mass:.1%}  pointing={'hit' if pointing else 'miss'}"
                    )

                    if save_viz:
                        # Flip both axes (180° rotation) to match the display orientation
                        # used in dp_eval.py's video recording — the raw env render is
                        # upside-down relative to how humans expect to view the scene.
                        frame_rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1].copy())
                        heatmap_viz = heatmap[::-1, ::-1]
                        seg_mask_viz = seg_mask[::-1, ::-1]
                        viz_path = str(
                            pathlib.Path(output_dir) /
                            f"{episode_prefix}_step{t:04d}_gradcam_iou.png"
                        )
                        layer_label = "+".join(layers)
                        fig = visualize_attention_vs_segmentation(
                            frame_rgb=frame_rgb,
                            attention_heatmap=heatmap_viz,
                            segmentation_mask=seg_mask_viz,
                            object_ids=object_ids,
                            iou_results=iou_result,
                            layer_idx=layer_label,
                            output_path=viz_path,
                        )
                        plt.close(fig)

            # Policy action (no grad)
            with torch.no_grad():
                actions_t = model.predict_action(images, state_t, task_idx)
            action_chunk = actions_t[0].cpu().numpy()
            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(action, policy_cfg, policy_rng)
        obs, reward, done, info = env.step(action.tolist())

        if done:
            break
        t += 1

    episode_summary = summarize_episode_iou(step_iou_results)

    return {
        "success": bool(done),
        "num_steps": t,
        "object_ids": {k: int(v) for k, v in object_ids.items()},
        "summary": episode_summary,
        "step_iou_results": step_iou_results,
    }


# ── LIBERO helpers ────────────────────────────────────────────────────────────


def _get_segmentation_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = SegmentationRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_segmentations="instance",
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
        description="LIBERO eval with GradCAM IoU (Diffusion Policy)"
    )

    # Model
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to DP checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--replan-steps", type=int, default=16,
                        help="Action chunk steps before replanning")
    parser.add_argument("--t-frac", type=float, default=0.5,
                        help="Diffusion timestep fraction for GradCAM (0=clean, 1=noise)")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3", "layer4"],
                        help="ResNet-18 layers to use for GradCAM (averaged)")

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=list(TASK_MAX_STEPS.keys()))
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run only this task ID (default: all tasks)")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Visualisation
    parser.add_argument("--save-viz", action="store_true",
                        help="Save GradCAM overlay images per replan step")

    # Output
    parser.add_argument("--output-dir", type=str, default="results/attention_iou_dp")

    # Visual perturbation
    parser.add_argument("--visual-perturb-mode", type=str, default="none",
                        choices=["none", "rotate", "translate", "rotate_translate"])
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--translate-x-frac", type=float, default=0.0)
    parser.add_argument("--translate-y-frac", type=float, default=0.0)

    # Policy perturbation
    parser.add_argument("--policy-perturb-mode", type=str, default="none",
                        choices=["none", "random_action", "object_shift"])
    parser.add_argument("--random-action-prob", type=float, default=0.25)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--object-shift-x-std", type=float, default=0.0)
    parser.add_argument("--object-shift-y-std", type=float, default=0.0)

    args = parser.parse_args()

    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    policy_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    policy_rng = np.random.default_rng(args.seed + 9999)
    log.info(f"Visual perturbation: {vis_cfg.as_dict()}")
    log.info(f"Policy perturbation: {policy_cfg.as_dict()}")

    device = torch.device(args.device)
    max_steps = TASK_MAX_STEPS[args.task_suite]

    log.info(f"GradCAM layers: {args.layers}")
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

    threshold_methods = DEFAULT_THRESHOLD_METHODS

    all_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        log.info(f"\n{'=' * 70}")
        log.info(f"Task {task_id}: {task_description}")
        log.info(f"{'=' * 70}")

        env, _ = _get_segmentation_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

            task_slug = task_description.replace(" ", "_")[:60]
            episode_prefix = f"task{task_id}_{task_slug}_ep{ep_idx}"

            result = run_episode(
                env=env,
                model=model,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                max_steps=max_steps,
                device=device,
                replan_steps=args.replan_steps,
                t_frac=args.t_frac,
                layers=args.layers,
                threshold_methods=threshold_methods,
                save_viz=args.save_viz,
                output_dir=args.output_dir,
                episode_prefix=episode_prefix,
                vis_cfg=vis_cfg,
                policy_cfg=policy_cfg,
                policy_rng=policy_rng,
            )
            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

            log.info(
                f"  Episode done: success={result['success']}  steps={result['num_steps']}"
            )
            if result["summary"]:
                s = result["summary"]
                iou_stats = s.get("combined_iou", {})
                mass_stats = s.get("attention_mass_on_objects", {})
                log.info(
                    f"  IoU summary: mean={iou_stats.get('mean', 0):.3f}  "
                    f"std={iou_stats.get('std', 0):.3f}  "
                    f"attn_mass={mass_stats.get('mean', 0):.3f}  "
                    f"pointing_acc={s.get('pointing_accuracy', 0):.1%}"
                )

        env.close()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = pathlib.Path(args.output_dir) / f"iou_results_{args.task_suite}.json"
    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_iou_results"}
        # Store compact per-step metrics (skip full heatmaps)
        entry["step_iou_results"] = [
            {
                "step": int(s.get("step", -1)),
                "combined_iou_p90": float(
                    s.get("combined", {}).get("percentile_90", {}).get("iou", 0.0)
                ),
                "attention_mass_all": float(
                    s.get("attention_mass", {}).get("_all_objects", 0.0)
                ),
                "pointing_hit": bool(s.get("pointing_hit", False)),
            }
            for s in r.get("step_iou_results", [])
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
        all_step = [
            s
            for r in task_res
            for s in r.get("step_iou_results", [])
        ]
        if all_step:
            mean_iou = float(np.mean([s.get("combined_iou_p90", 0) for s in all_step]))
            mean_mass = float(np.mean([s.get("attention_mass_all", 0) for s in all_step]))
            point_acc = float(np.mean([s.get("pointing_hit", False) for s in all_step]))
            task_desc = task_res[0]["task_description"]
            log.info(
                f"Task {task_id} ({task_desc[:50]}): "
                f"success={success_rate:.1%}  "
                f"IoU(p90)={mean_iou:.3f}  "
                f"attn_mass={mean_mass:.3f}  "
                f"pointing={point_acc:.1%}"
            )


if __name__ == "__main__":
    main()
