"""
LIBERO evaluation with attention-segmentation IoU analysis for Cosmos Policy.

Uses Q-norm weighted attention saliency for pure-spatial self-attn (S is square).
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parent / "third_party" / "cosmos-policy")
if _COSMOS_POLICY_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_POLICY_DIR)

# Mock transformer_engine if not available
from unittest.mock import MagicMock

try:
    import transformer_engine.pytorch  # noqa: F401
except Exception:
    import importlib.abc
    import types

    class _TEMockModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
        def __getattr__(self, attr):
            val = MagicMock()
            object.__setattr__(self, attr, val)
            return val

    class _TEFinder(importlib.abc.MetaPathFinder):
        def find_module(self, fullname, path=None):
            if fullname == "transformer_engine" or fullname.startswith("transformer_engine."):
                return self
            return None
        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _TEMockModule(fullname)
            sys.modules[fullname] = mod
            parts = fullname.rsplit(".", 1)
            if len(parts) == 2:
                parent = sys.modules.get(parts[0])
                if parent is not None:
                    setattr(parent, parts[1], mod)
            return mod

    sys.meta_path.insert(0, _TEFinder())
    _TEFinder().load_module("transformer_engine")

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import SegmentationRenderEnv

from attention_iou import (
    compute_attention_object_iou,
    summarize_episode_iou,
    visualize_attention_vs_segmentation,
    visualize_iou_over_episode,
    find_segmentation_key,
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





"""
Hook-based attention recording for Cosmos DiT models.

This version addresses "stripy / weird" heatmaps when self-attn is pure spatial
(S is a perfect square, e.g. 1764=42*42) by:
  1) recording per-query weights from Q (qnorm)
  2) building a query-weighted saliency map:
        sal[k] = sum_q w[q] * attn(q->k),  where w[q] ~ ||Q_q||
  3) using percentile clipping normalization (more stable than min-max)

It supports ONLY the pure-spatial case (S is a perfect square) since your logs
show S=1764 (42x42).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_RECORDED: Dict[int, Dict[str, np.ndarray]] = {}
_HOOKS: List[torch.utils.hooks.RemovableHandle] = []


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

def _self_attn_hook_fn(layer_idx: int):
    def hook(module, args, output):
        x = args[0]
        context = args[1] if len(args) > 1 else None
        rope_emb = args[2] if len(args) > 2 else None

        with torch.no_grad():
            q, k, v = module.compute_qkv(x, context, rope_emb=rope_emb)
            # Expected q,k: (B, S, H, D) in Cosmos DiT blocks

            # Record qnorm weights per query token (B, S)
            qnorm = torch.linalg.norm(q.float(), dim=-1)  # (B, S, H)
            qnorm = qnorm.mean(dim=-1)                    # (B, S)

            # Attention weights
            scale = q.shape[-1] ** -0.5
            q_f = (q.float() * scale).permute(0, 2, 1, 3)  # (B, H, S, D)
            k_f = k.float().permute(0, 2, 1, 3)           # (B, H, S, D)
            attn = torch.matmul(q_f, k_f.transpose(-2, -1))  # (B, H, S, S)
            attn = F.softmax(attn, dim=-1)

            _RECORDED.setdefault(layer_idx, {})
            _RECORDED[layer_idx]["self_attn"] = attn[0].cpu().numpy()   # (H, S, S)
            _RECORDED[layer_idx]["qnorm"] = qnorm[0].cpu().numpy()      # (S,)

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def install_attention_hooks(
    model: torch.nn.Module,
    layers: Optional[List[int]] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    global _HOOKS
    _RECORDED.clear()

    net = _find_dit(model)
    if net is None:
        raise RuntimeError("Could not find DiT backbone in model")

    blocks = net.blocks
    if layers is None:
        layers = list(range(len(blocks)))

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for idx in layers:
        if idx >= len(blocks):
            continue
        block = blocks[idx]
        handles.append(block.self_attn.register_forward_hook(_self_attn_hook_fn(idx)))

    _HOOKS = handles
    return handles


def remove_attention_hooks(handles: Optional[List] = None):
    global _HOOKS
    for h in (handles or _HOOKS):
        try:
            h.remove()
        except Exception:
            pass
    _HOOKS = []


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, np.ndarray]]:
    out = dict(_RECORDED)
    _RECORDED.clear()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap construction
# ─────────────────────────────────────────────────────────────────────────────

def _infer_square_side(S: int) -> int:
    side = int(round(S ** 0.5))
    if side * side != S:
        raise ValueError(f"S={S} is not a perfect square; expected pure-spatial self-attn.")
    return side


def _query_weighted_saliency(attn_hss: np.ndarray, qnorm_s: Optional[np.ndarray]) -> np.ndarray:
    """
    attn_hss: (H, S, S)
    qnorm_s: (S,) or None

    Returns saliency over keys: (S,)
        sal[k] = sum_q w[q] * mean_h attn[h,q,k]
    """
    attn = attn_hss.mean(axis=0)  # (S, S)

    if qnorm_s is None:
        # fallback: incoming mean (often stripy)
        return attn.mean(axis=0)

    w = qnorm_s.astype(np.float32)
    # robust normalize
    w = np.maximum(w, 0.0)
    w_sum = float(w.sum())
    if w_sum < 1e-8:
        return attn.mean(axis=0)
    w = w / w_sum
    sal = (w[:, None] * attn).sum(axis=0)  # (S,)
    return sal


def create_cosmos_attention_heatmap(
    recorded_layer: Dict[str, np.ndarray],
    target_shape: Tuple[int, int],
    transpose_hw: bool = False,
    clip_percentiles: Tuple[float, float] = (5.0, 95.0),
    debug: bool = False,
) -> np.ndarray:
    """
    Build normalized [0,1] heatmap at target_shape from one layer's recorded data.

    recorded_layer should contain:
      - "self_attn": (H, S, S)
      - "qnorm": (S,)   (optional but recommended)
    """
    import cv2

    attn = recorded_layer.get("self_attn", None)
    if attn is None:
        raise ValueError("recorded_layer missing 'self_attn'")
    qnorm = recorded_layer.get("qnorm", None)

    if attn.ndim != 3:
        raise ValueError(f"self_attn must be (H,S,S), got {attn.shape}")

    S = attn.shape[-1]
    side = _infer_square_side(S)  # e.g. 42

    sal = _query_weighted_saliency(attn, qnorm)  # (S,)
    if sal.size != S:
        raise ValueError(f"saliency size mismatch: {sal.size} vs S={S}")

    hm = sal.reshape(side, side).astype(np.float32)
    if transpose_hw:
        hm = hm.T

    if debug:
        print(f"[heatmap] S={S} side={side} transpose={transpose_hw} "
              f"qnorm={'yes' if qnorm is not None else 'no'}")

    # Upsample
    hm_up = cv2.resize(hm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Percentile clip normalization (more stable than min-max)
    lo_p, hi_p = clip_percentiles
    lo = float(np.percentile(hm_up, lo_p))
    hi = float(np.percentile(hm_up, hi_p))
    hm_up = np.clip(hm_up, lo, hi)
    hm_up = (hm_up - lo) / (hi - lo + 1e-8)

    return hm_up


# ─────────────────────────────────────────────────────────────────────────────
# Internal: find DiT backbone
# ─────────────────────────────────────────────────────────────────────────────

def _find_dit(model) -> Optional[torch.nn.Module]:
    for attr_path in ["net", "model.net", "model"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "blocks") and hasattr(obj, "final_layer"):
                return obj
        except AttributeError:
            continue

    for _, mod in model.named_modules():
        if hasattr(mod, "blocks") and hasattr(mod, "final_layer") and hasattr(mod, "t_embedder"):
            return mod
    return None



def _json_default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _get_segmentation_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_segmentations="instance",
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _prepare_observation(obs, flip_images: bool = True):
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    return {
        "primary_image": np.ascontiguousarray(img),
        "wrist_image": np.ascontiguousarray(wrist_img),
        "proprio": np.concatenate(
            (obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        ),
    }


def _build_cosmos_cfg(args) -> PolicyEvalConfig:
    return PolicyEvalConfig(
        config=args.config_name,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        seed=args.seed,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_episodes,
    )


def run_episode(
    env,
    model,
    cfg: PolicyEvalConfig,
    dataset_stats,
    task_description: str,
    initial_state: np.ndarray,
    layers: List[int],
    max_steps: int,
    num_open_loop_steps: int,
    save_viz: bool,
    output_dir: str,
    episode_prefix: str,
    threshold_methods,
    transpose_hw: bool,
    clip_lo: float,
    clip_hi: float,
    attn_debug: bool,
) -> Dict:
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    obj_of_interest = env.obj_of_interest
    object_ids = {name: env.instance_to_id[name] for name in obj_of_interest if name in env.instance_to_id}
    log.info(f"Objects of interest: {object_ids}")
    if not object_ids:
        log.warning("No objects of interest found in segmentation mapping!")

    seg_key = find_segmentation_key(obs)
    if seg_key is None:
        for k in obs:
            if "seg" in k.lower():
                seg_key = k
                break
    if seg_key is None:
        log.error(f"No segmentation key found. obs keys: {list(obs.keys())}")
        return {"error": "no_segmentation_key", "obs_keys": list(obs.keys())}
    log.info(f"Using segmentation key: {seg_key}")

    action_plan = collections.deque()
    step_iou_results = []

    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        observation = _prepare_observation(obs, flip_images=True)
        agentview_for_viz = observation["primary_image"]

        seg_raw = obs[seg_key]
        seg_rotated = np.ascontiguousarray(np.flipud(seg_raw))

        is_replan = not action_plan
        if is_replan:
            clear_recorded_attention()
            hooks = install_attention_hooks(model, layers=layers)

            try:
                action_return_dict = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=t,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
            finally:
                remove_attention_hooks(hooks)

            action_chunk = action_return_dict["actions"]
            action_plan.extend(action_chunk[:num_open_loop_steps])

            attention_dict = get_recorded_attention()

            if attention_dict and object_ids:
                for layer_idx in layers:
                    if layer_idx not in attention_dict:
                        continue
                    layer_rec = attention_dict[layer_idx]

                    try:
                        heatmap = create_cosmos_attention_heatmap(
                            recorded_layer=layer_rec,
                            target_shape=(LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
                            transpose_hw=transpose_hw,
                            clip_percentiles=(clip_lo, clip_hi),
                            debug=(attn_debug and (t % 50 == 0)),
                        )
                    except Exception as e:
                        log.warning(f"[attn->heatmap] t={t} layer={layer_idx} failed: {e}")
                        continue

                    iou_result = compute_attention_object_iou(
                        attention_heatmap=heatmap,
                        segmentation_mask=seg_rotated,
                        object_ids=object_ids,
                        threshold_methods=threshold_methods,
                    )
                    iou_result["layer"] = layer_idx
                    iou_result["step"] = t
                    step_iou_results.append(iou_result)

                    combined_iou = iou_result["combined"].get("percentile_90", {}).get("iou", 0)
                    mass = iou_result["attention_mass"].get("_all_objects", 0)
                    log.info(
                        f"  t={t} layer={layer_idx}: IoU={combined_iou:.3f}, "
                        f"attn_mass={mass:.1%}, pointing={'hit' if iou_result['pointing_hit'] else 'miss'}"
                    )

                    if save_viz:
                        viz_path = os.path.join(
                            output_dir, f"{episode_prefix}_step{t:04d}_layer{layer_idx}_iou.png"
                        )
                        fig = visualize_attention_vs_segmentation(
                            frame_rgb=agentview_for_viz,
                            attention_heatmap=heatmap,
                            segmentation_mask=seg_rotated,
                            object_ids=object_ids,
                            iou_results=iou_result,
                            layer_idx=layer_idx,
                            output_path=viz_path,
                        )
                        plt.close(fig)

        action = action_plan.popleft()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        obs, reward, done, info = env.step(action)

        if done:
            break
        t += 1

    summary = {}
    for layer_idx in layers:
        layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = summarize_episode_iou(layer_results)

    if step_iou_results and save_viz:
        for layer_idx in layers:
            layer_results = [r for r in step_iou_results if r.get("layer") == layer_idx]
            if layer_results:
                layer_steps = [r["step"] for r in layer_results]
                fig = visualize_iou_over_episode(
                    step_results=layer_results,
                    step_indices=layer_steps,
                    prompt=task_description,
                    output_path=os.path.join(output_dir, f"{episode_prefix}_layer{layer_idx}_iou_evolution.png"),
                )
                plt.close(fig)

    return {
        "success": bool(done),
        "num_steps": t,
        "step_iou_results": step_iou_results,
        "summary": summary,
        "objects_of_interest": list(object_ids.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO eval with attention IoU (Q-weighted saliency)")

    # Cosmos model
    parser.add_argument("--ckpt-path", type=str, default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    parser.add_argument("--config-name", type=str, default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument("--config-file", type=str, default="cosmos_policy/config/config.py")
    parser.add_argument("--dataset-stats-path", type=str,
                        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    parser.add_argument("--t5-text-embeddings-path", type=str,
                        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention / IoU
    parser.add_argument("--layers", type=int, nargs="+", default=[27])
    parser.add_argument("--threshold-methods", type=str, nargs="+",
                        default=["percentile_90", "percentile_75", "otsu_0"])

    # Heatmap debugging knobs
    parser.add_argument("--transpose-hw", action="store_true",
                        help="Transpose latent grid before upsampling (tests flatten ordering).")
    parser.add_argument("--clip-lo", type=float, default=5.0,
                        help="Low percentile for clipping normalization (default 5).")
    parser.add_argument("--clip-hi", type=float, default=95.0,
                        help="High percentile for clipping normalization (default 95).")
    parser.add_argument("--attn-debug", action="store_true",
                        help="Print attention/heatmap debug info occasionally.")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs_iou_cosmos")
    parser.add_argument("--save-viz", action="store_true")

    args = parser.parse_args()

    threshold_methods = []
    for tm in args.threshold_methods:
        method, value = tm.rsplit("_", 1)
        threshold_methods.append((method, float(value)))

    max_steps = TASK_MAX_STEPS[args.task_suite]

    log.info("Loading Cosmos model...")
    cfg = _build_cosmos_cfg(args)
    model, cosmos_config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log.info("Cosmos model loaded.")

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks

    task_ids = [args.task_id] if args.task_id is not None else list(range(num_tasks))
    os.makedirs(args.output_dir, exist_ok=True)

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
                cfg=cfg,
                dataset_stats=dataset_stats,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                layers=args.layers,
                max_steps=max_steps,
                num_open_loop_steps=args.num_open_loop_steps,
                save_viz=args.save_viz,
                output_dir=args.output_dir,
                episode_prefix=episode_prefix,
                threshold_methods=threshold_methods,
                transpose_hw=args.transpose_hw,
                clip_lo=args.clip_lo,
                clip_hi=args.clip_hi,
                attn_debug=args.attn_debug,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            all_results.append(result)

        env.close()

    # Save summary (excluding huge per-step arrays)
    serializable_results = [{k: v for k, v in r.items() if k != "step_iou_results"} for r in all_results]
    results_path = os.path.join(args.output_dir, "iou_results.json")
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()