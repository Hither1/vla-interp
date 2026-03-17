"""Language corruption experiment: attention ratio analysis for DreamZero.

Distributed inference via torchrun. Measures Visual Compensation Index (VCI)
and success deltas under prompt corruption.

Usage:
  torchrun --standalone --nproc_per_node=4 \\
    analysis/attention/evaluate_lang_corruption_ratio_dreamzero.py \\
    --model-path /path/to/dreamzero_libero_lora \\
    --task-suite-name libero_10 --num-episodes 5 \\
    --layers 10 20 30 --output-dir results/lang_corruption_ratio_dreamzero
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import math
import os
import pathlib
import pickle
import sys
import traceback
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

# Path setup
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DREAMZERO_DIR = str(_PROJECT_ROOT / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)
_LIBERO_EVAL_DIR = str(_PROJECT_ROOT / "examples" / "libero")
if _LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EVAL_DIR)

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanAttentionBlock, CausalWanSelfAttention,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image
from prompt_perturbations import perturb_prompt
from tianshou.data import Batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 128

_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}

ALL_PROMPT_MODES = ["original", "empty", "shuffle", "random", "synonym", "opposite"]


# Global attention recording state (copied from evaluate_attention_ratio_dreamzero.py)
_RECORDED: Dict[int, Dict[str, Any]] = {}
_HOOKS: List[torch.utils.hooks.RemovableHandle] = []


def clear_recorded_attention():
    _RECORDED.clear()


def get_recorded_attention() -> Dict[int, Dict[str, Any]]:
    out = dict(_RECORDED)
    _RECORDED.clear()
    return out


def remove_attention_hooks(handles=None):
    global _HOOKS
    for h in (handles or _HOOKS):
        try:
            h.remove()
        except Exception:
            pass
    _HOOKS = []


def install_attention_hooks(policy: GrootSimPolicy, layers: List[int]):
    """Install hooks on CausalWanAttentionBlock self_attn and cross_attn."""
    global _HOOKS
    _RECORDED.clear()
    try:
        blocks = policy.trained_model.action_head.model.blocks
    except AttributeError as e:
        raise RuntimeError(f"Cannot find DiT blocks: {e}")

    handles = []
    for layer_idx in layers:
        if layer_idx >= len(blocks):
            log.warning("Layer %d out of range (%d blocks)", layer_idx, len(blocks))
            continue
        block = blocks[layer_idx]
        if not isinstance(block, CausalWanAttentionBlock):
            log.warning("Block %d is not CausalWanAttentionBlock", layer_idx)
            continue

        self_attn: CausalWanSelfAttention = block.self_attn
        _RECORDED[layer_idx] = {"self_attn": [], "cross_attn": [], "action_register_length": None}

        def _make_pre_hook(idx):
            def pre_hook(module, args):
                if len(args) >= 5 and args[4] is not None:
                    _RECORDED[idx]["action_register_length"] = args[4]
            return pre_hook
        handles.append(self_attn.register_forward_pre_hook(_make_pre_hook(layer_idx)))

        def _make_self_attn_hook(idx, parent):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block
            def hook(module, args, output):
                if len(args) < 3: return
                q, k, v = args[0], args[1], args[2]
                if q.shape[1] != n_act: return
                Lk = k.shape[1]
                visual_end = Lk - n_act - n_state
                with torch.no_grad():
                    q_f = q.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = q.shape[-1] ** -0.5
                    attn = torch.softmax(torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1)
                    attn_np = attn[0].mean(0).cpu().float().numpy()
                _RECORDED[idx]["self_attn"].append({
                    "attn": attn_np, "Lk": Lk, "visual_end": visual_end,
                    "n_act": n_act, "n_state": n_state,
                })
            return hook
        handles.append(self_attn.attn.register_forward_hook(_make_self_attn_hook(layer_idx, self_attn)))

        def _make_cross_attn_hook(idx, parent):
            n_act = parent.num_action_per_block
            n_state = parent.num_state_per_block
            def hook(module, args, output):
                if len(args) < 2: return
                normed_x = args[0]
                context = args[1]
                action_register_length = _RECORDED[idx].get("action_register_length")
                if action_register_length is None or action_register_length == 0: return
                B, S, C = normed_x.shape
                n = module.num_heads
                d = module.head_dim
                chunk_size = action_register_length // (n_act + n_state)
                action_horizon = chunk_size * n_act
                state_horizon = chunk_size * n_state
                action_start = S - action_horizon - state_horizon
                action_end = S - state_horizon
                if action_start < 0 or action_end > S: return
                with torch.no_grad():
                    x_act = normed_x[:, action_start:action_end]
                    q = module.norm_q(module.q(x_act)).view(B, -1, n, d)
                    if hasattr(module, "k_img"):
                        text_context = context[:, 257:]
                    else:
                        text_context = context
                    if text_context.shape[1] == 0: return
                    k = module.norm_k(module.k(text_context)).view(B, -1, n, d)
                    q_f = q.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = d ** -0.5
                    attn = torch.softmax(torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1)
                    attn_np = attn[0].mean(0).cpu().float().numpy()
                _RECORDED[idx]["cross_attn"].append({
                    "attn": attn_np, "action_horizon": action_horizon,
                    "T": text_context.shape[1],
                })
            return hook
        handles.append(block.cross_attn.register_forward_hook(_make_cross_attn_hook(layer_idx, self_attn)))

    _HOOKS = handles
    return handles


def compute_ratio_from_recorded(layer_rec: Dict[str, Any]) -> Optional[Dict]:
    """Compute visual/linguistic ratio from one layer recording."""
    self_attn_calls = layer_rec.get("self_attn", [])
    cross_attn_calls = layer_rec.get("cross_attn", [])
    if not self_attn_calls:
        return None

    visual_fractions = []
    for call in self_attn_calls:
        attn = call["attn"]
        visual_end = call["visual_end"]
        vis_frac = float(attn[:, :visual_end].sum()) / max(float(attn.sum()), 1e-8)
        visual_fractions.append(vis_frac)
    visual_mass = float(np.mean(visual_fractions))

    if cross_attn_calls:
        ling_values = []
        for call in cross_attn_calls:
            attn = call["attn"]
            T = call["T"]
            ling_values.append(float(attn.mean()) * T)
        linguistic_mass = float(np.mean(ling_values))
    else:
        linguistic_mass = 0.0

    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float("inf") if visual_mass > 1e-8 else 0.0

    state_fractions, self_fractions = [], []
    for call in self_attn_calls:
        attn = call["attn"]
        vis_end = call["visual_end"]
        n_act = call["n_act"]
        n_state = call["n_state"]
        action_cols = attn[:, vis_end: vis_end + n_act]
        state_cols = attn[:, vis_end + n_act:]
        total = max(float(attn.sum()), 1e-8)
        state_fractions.append(float(state_cols.sum()) / total)
        self_fractions.append(float(action_cols.sum()) / total)

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass,
        "linguistic_fraction": linguistic_mass,
        "state_fraction": float(np.mean(state_fractions)),
        "self_fraction": float(np.mean(self_fractions)),
        "num_self_attn_calls": len(self_attn_calls),
        "num_cross_attn_calls": len(cross_attn_calls),
    }


def _json_default(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)


def extract_actions(result_batch):
    act = result_batch.act
    items = list(act.items()) if isinstance(act, dict) else [
        (a, getattr(act, a)) for a in dir(act)
        if "joint_position" in a and not a.startswith("_")
    ]
    for k, v in items:
        if "joint_position" in k:
            if isinstance(v, torch.Tensor): v = v.cpu().numpy()
            v = np.array(v, dtype=np.float32)
            return v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v.reshape(1, -1)
    raise ValueError("action.joint_position not found in result_batch.act: %s" % act)


def _broadcast_obs(obs):
    data = pickle.dumps(obs)
    size = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    dist.broadcast(buf, src=0)


def _receive_obs():
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    buf = torch.zeros(int(size.item()), dtype=torch.uint8, device="cuda")
    dist.broadcast(buf, src=0)
    return pickle.loads(buf.cpu().numpy().tobytes())


def worker_loop(policy, signal_group):
    rank = dist.get_rank()
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    while True:
        try:
            dist.broadcast(signal, src=0, group=signal_group)
            if signal.item() == 1:
                break
            obs = _receive_obs()
            dist.barrier()
            with torch.no_grad():
                policy.lazy_joint_forward_causal(Batch(obs=obs))
            dist.barrier()
        except Exception:
            log.error("Rank %d error:\n%s", rank, traceback.format_exc())
            break


def _temporal_slope(per_step_data: List[Dict]) -> Dict:
    """Split into early/mid/late thirds; compute slope via linear regression."""
    if not per_step_data:
        return {"early": 0.0, "mid": 0.0, "late": 0.0, "slope": 0.0}
    n = len(per_step_data)
    early = per_step_data[: max(1, n // 3)]
    mid = per_step_data[n // 3 : max(n // 3 + 1, 2 * n // 3)]
    late = per_step_data[2 * n // 3 :]
    early_mean = float(np.mean([d["visual_fraction"] for d in early])) if early else 0.0
    mid_mean = float(np.mean([d["visual_fraction"] for d in mid])) if mid else 0.0
    late_mean = float(np.mean([d["visual_fraction"] for d in late])) if late else 0.0
    steps = np.array([d["step"] for d in per_step_data], dtype=float)
    vfracs = np.array([d["visual_fraction"] for d in per_step_data], dtype=float)
    slope = float(np.polyfit(steps, vfracs, 1)[0]) if (len(steps) >= 2 and steps.std() > 1e-8) else 0.0
    return {"early": early_mean, "mid": mid_mean, "late": late_mean, "slope": slope}


def compute_deltas(mode_results: Dict, baseline_mode: str = "original") -> Dict:
    """Compute per-mode deltas relative to the baseline mode."""
    baseline = mode_results.get(baseline_mode)
    if baseline is None:
        log.warning("Baseline mode %r not found.", baseline_mode)
        return {}
    baseline_vfrac = baseline["visual_fraction_mean"]
    baseline_lfrac = baseline["linguistic_fraction_mean"]
    baseline_success = int(baseline["success"])
    baseline_temporal = _temporal_slope(baseline["per_step_data"])
    deltas = {}
    for mode, res in mode_results.items():
        if mode == baseline_mode: continue
        delta_vfrac = res["visual_fraction_mean"] - baseline_vfrac
        delta_lfrac = res["linguistic_fraction_mean"] - baseline_lfrac
        mode_temporal = _temporal_slope(res["per_step_data"])
        deltas[mode] = {
            "delta_visual_fraction": float(delta_vfrac),
            "delta_linguistic_fraction": float(delta_lfrac),
            "delta_success": int(res["success"]) - baseline_success,
            "vci": float(delta_vfrac),
            "temporal_analysis": {
                "baseline_early": float(baseline_temporal["early"]),
                "baseline_mid": float(baseline_temporal["mid"]),
                "baseline_late": float(baseline_temporal["late"]),
                "mode_early": float(mode_temporal["early"]),
                "mode_mid": float(mode_temporal["mid"]),
                "mode_late": float(mode_temporal["late"]),
                "delta_early": float(mode_temporal["early"] - baseline_temporal["early"]),
                "delta_mid": float(mode_temporal["mid"] - baseline_temporal["mid"]),
                "delta_late": float(mode_temporal["late"] - baseline_temporal["late"]),
                "temporal_slope": float(mode_temporal["slope"]),
            },
        }
    return deltas


def run_episode_single_mode(
    env,
    policy: GrootSimPolicy,
    signal_group,
    prompt: str,
    initial_state,
    layers: List[int],
    max_steps: int,
    replan_steps: int,
    num_steps_wait: int,
    vis_cfg: VisualPerturbConfig,
    policy_cfg: PolicyPerturbConfig,
    policy_rng: np.random.Generator,
) -> Dict:
    """Run one DreamZero episode with a single (possibly perturbed) prompt.

    Returns dict with: success, num_steps, visual_fractions_per_step,
    linguistic_fractions_per_step, per_step_data, visual_fraction_mean,
    linguistic_fraction_mean.
    """
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")

    env.reset()
    obs = env.set_init_state(initial_state)
    apply_object_shift(env, policy_cfg, policy_rng)

    action_plan: collections.deque = collections.deque()
    step_ratio_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + num_steps_wait:
        try:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = perturb_image(np.ascontiguousarray(obs["agentview_image"]), vis_cfg)
            wrist = perturb_image(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]), vis_cfg)

            is_new_chunk = not action_plan
            if is_new_chunk:
                dz_obs = {
                    "video.agentview_rgb": img[None].astype(np.uint8),
                    "video.eye_in_hand_rgb": wrist[None].astype(np.uint8),
                    "state.joint_position":
                        np.array(obs["robot0_joint_pos"], dtype=np.float64).reshape(1, -1),
                    "state.gripper_position":
                        np.array(obs["robot0_gripper_qpos"], dtype=np.float64)[:1].reshape(1, -1),
                    "annotation.language.language_instruction": prompt,
                }

                clear_recorded_attention()
                hooks = install_attention_hooks(policy, layers)

                signal.fill_(0)
                dist.broadcast(signal, src=0, group=signal_group)
                _broadcast_obs(dz_obs)
                dist.barrier()
                with torch.no_grad():
                    result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
                dist.barrier()

                remove_attention_hooks(hooks)
                action_chunk = extract_actions(result_batch)
                action_plan.extend(action_chunk[:min(replan_steps, len(action_chunk))])

                recorded = get_recorded_attention()
                for layer_idx in layers:
                    if layer_idx not in recorded: continue
                    ratio_result = compute_ratio_from_recorded(recorded[layer_idx])
                    if ratio_result is None: continue
                    ratio_result["layer"] = layer_idx
                    ratio_result["step"] = t
                    step_ratio_results.append(ratio_result)
                    log.info(
                        "  t=%d layer=%d visual=%.3f linguistic=%.3f ratio=%.3f",
                        t, layer_idx, ratio_result["visual_fraction"],
                        ratio_result["linguistic_fraction"],
                        ratio_result.get("visual_linguistic_ratio", 0.0),
                    )

            pol_action = action_plan.popleft()
            action, _ = maybe_perturb_action(pol_action, policy_cfg, policy_rng)
            obs, reward, done, _ = env.step(action.tolist())
            t += 1
            if done: break
        except Exception:
            log.error("Step error:\n%s", traceback.format_exc())
            break

    # Average visual/linguistic fractions across layers per step
    by_step: Dict = {}
    for r in step_ratio_results:
        by_step.setdefault(int(r["step"]), []).append(r)

    per_step_data = []
    for step in sorted(by_step.keys()):
        rs = by_step[step]
        avg_vis = float(np.mean([r["visual_fraction"] for r in rs]))
        avg_ling = float(np.mean([r["linguistic_fraction"] for r in rs]))
        fin_ratios = [r["visual_linguistic_ratio"] for r in rs
                      if np.isfinite(r.get("visual_linguistic_ratio", float("nan")))]
        avg_ratio = float(np.mean(fin_ratios)) if fin_ratios else 0.0
        per_step_data.append({
            "step": step,
            "visual_fraction": avg_vis,
            "linguistic_fraction": avg_ling,
            "visual_linguistic_ratio": avg_ratio,
        })

    visual_fracs = [d["visual_fraction"] for d in per_step_data]
    linguistic_fracs = [d["linguistic_fraction"] for d in per_step_data]

    return {
        "success": bool(done and "reward" in dir() and reward > 0),
        "num_steps": t,
        "visual_fractions_per_step": visual_fracs,
        "linguistic_fractions_per_step": linguistic_fracs,
        "per_step_data": per_step_data,
        "visual_fraction_mean": float(np.mean(visual_fracs)) if visual_fracs else 0.0,
        "linguistic_fraction_mean": float(np.mean(linguistic_fracs)) if linguistic_fracs else 0.0,
    }


def run_episode_all_modes(
    env,
    policy: GrootSimPolicy,
    signal_group,
    task_description: str,
    initial_state,
    layers: List[int],
    max_steps: int,
    prompt_modes: List[str],
    all_tasks: List[str],
    replan_steps: int = 4,
    num_steps_wait: int = 10,
    vis_cfg: Optional[VisualPerturbConfig] = None,
    policy_cfg: Optional[PolicyPerturbConfig] = None,
    policy_rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Run the episode under all prompt modes; compute deltas vs original."""
    mode_results: Dict = {}

    for mode in prompt_modes:
        prompt = perturb_prompt(task_description, mode=mode, all_tasks=all_tasks)
        log.info("  Mode=%r  prompt=%r", mode, prompt[:80])

        env.reset()
        env.set_init_state(initial_state)

        result = run_episode_single_mode(
            env=env, policy=policy, signal_group=signal_group,
            prompt=prompt, initial_state=initial_state, layers=layers,
            max_steps=max_steps, replan_steps=replan_steps,
            num_steps_wait=num_steps_wait,
            vis_cfg=vis_cfg if vis_cfg is not None else VisualPerturbConfig(),
            policy_cfg=policy_cfg if policy_cfg is not None else PolicyPerturbConfig(),
            policy_rng=policy_rng if policy_rng is not None else np.random.default_rng(),
        )

        result["prompt"] = prompt
        result["mode"] = mode
        mode_results[mode] = result

        log.info(
            "    -> success=%s steps=%d vfrac=%.3f lfrac=%.3f",
            result["success"], result["num_steps"],
            result["visual_fraction_mean"], result["linguistic_fraction_mean"],
        )

    deltas = compute_deltas(mode_results, baseline_mode="original")
    return {"mode_results": mode_results, "deltas": deltas}


def eval_rank0(args, policy, signal_group):
    np.random.seed(args.seed)
    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    pol_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    pol_rng = np.random.default_rng(args.seed + 9999)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]
    task_ids = [args.task_id] if args.task_id is not None else list(range(n_tasks))
    all_task_descriptions = [suite.get_task(i).language for i in range(n_tasks)]

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    all_results = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_description = task.language

        log.info("\n%s\nTask %d: %s\n%s", "=" * 70, task_id, task_description, "=" * 70)

        for ep_idx in range(min(args.num_episodes, len(init_states))):
            log.info("--- Episode %d/%d ---", ep_idx + 1, args.num_episodes)

            episode_result = run_episode_all_modes(
                env=env, policy=policy, signal_group=signal_group,
                task_description=task_description,
                initial_state=init_states[ep_idx],
                layers=args.layers, max_steps=max_steps,
                prompt_modes=args.prompt_modes, all_tasks=all_task_descriptions,
                replan_steps=args.replan_steps, num_steps_wait=args.num_steps_wait,
                vis_cfg=vis_cfg, policy_cfg=pol_cfg, policy_rng=pol_rng,
            )

            modes_serializable = {}
            for mode, res in episode_result["mode_results"].items():
                modes_serializable[mode] = {
                    "success": bool(res["success"]),
                    "num_steps": int(res["num_steps"]),
                    "prompt": res["prompt"],
                    "visual_fraction_mean": float(res["visual_fraction_mean"]),
                    "linguistic_fraction_mean": float(res["linguistic_fraction_mean"]),
                    "per_step_data": res["per_step_data"],
                }

            entry = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": ep_idx,
                "modes": modes_serializable,
                "deltas": episode_result["deltas"],
                "visual_perturbation": vis_cfg.as_dict(),
                "policy_perturbation": pol_cfg.as_dict(),
            }
            all_results.append(entry)

            orig_res = episode_result["mode_results"].get("original", {})
            log.info("  original: success=%s vfrac=%.3f",
                orig_res.get("success"), orig_res.get("visual_fraction_mean", 0.0))
            for mode, delta in episode_result["deltas"].items():
                log.info("  %s: delta_vfrac=%.4f (VCI) delta_success=%+d",
                    mode, delta["delta_visual_fraction"], delta["delta_success"])

        env.close()

    # Shutdown workers
    signal.fill_(1)
    dist.broadcast(signal, src=0, group=signal_group)

    # Save results
    results_path = out_dir / f"lang_corruption_ratio_{args.task_suite_name}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    log.info("Results saved to %s", results_path)

    log.info("\n%s", "=" * 70)
    log.info("AGGREGATE SUMMARY")
    log.info("%s", "=" * 70)
    orig_successes = [
        int(r["modes"]["original"]["success"])
        for r in all_results if "original" in r.get("modes", {})
    ]
    if orig_successes:
        log.info("  original: success=%.1f%% (%d/%d)",
            100.0 * np.mean(orig_successes), sum(orig_successes), len(orig_successes))
    non_baseline_modes = [m for m in args.prompt_modes if m != "original"]
    log.info("\n  %-12s  %12s  %12s  %10s", "Mode", "VCI (mean)", "VCI (sem)", "Success%")
    for mode in non_baseline_modes:
        vcis = [r["deltas"][mode]["vci"] for r in all_results if mode in r.get("deltas", {})]
        successes = [int(r["modes"][mode]["success"]) for r in all_results if mode in r.get("modes", {})]
        if not vcis: continue
        vci_mean = float(np.mean(vcis))
        vci_sem = float(np.std(vcis) / max(1, math.sqrt(len(vcis))))
        succ_pct = 100.0 * float(np.mean(successes)) if successes else float("nan")
        log.info("  %-12s  %12.4f  %12.4f  %9.1f%%", mode, vci_mean, vci_sem, succ_pct)

    log.info("\nSaved %d episode records to %s", len(all_results), results_path)


def main():
    parser = argparse.ArgumentParser(description="Language corruption experiment for DreamZero")

    # Model
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--num-steps-wait", type=int, default=10)

    # LIBERO
    parser.add_argument("--task-suite-name", type=str, default="libero_10",
                        choices=list(_MAX_STEPS.keys()))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30])

    # Prompt perturbation
    parser.add_argument("--prompt-modes", nargs="+", default=ALL_PROMPT_MODES, choices=ALL_PROMPT_MODES)

    # Visual perturbation
    parser.add_argument("--visual-perturb-mode", type=str, default="none",
                        choices=["none", "rotate", "translate", "rotate_translate"])
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--translate-x-frac", type=float, default=0.0)
    parser.add_argument("--translate-y-frac", type=float, default=0.0)

    # Policy perturbation
    parser.add_argument("--policy-perturb-mode", type=str, default="none",
                        choices=["none", "random_action", "object_shift"])
    parser.add_argument("--random-action-prob", type=float, default=0.0)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--object-shift-x-std", type=float, default=0.0)
    parser.add_argument("--object-shift-y-std", type=float, default=0.0)

    # Output
    parser.add_argument("--output-dir", type=str, default="results/lang_corruption_ratio_dreamzero")

    args = parser.parse_args()

    os.environ["ENABLE_DIT_CACHE"] = "true"
    os.environ["ATTENTION_BACKEND"] = "FA2"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank{rank}] %(asctime)s %(message)s",
        force=True,
    )

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
    signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=10))

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
    )

    if rank == 0:
        try:
            blocks = policy.trained_model.action_head.model.blocks
            sa = blocks[0].self_attn
            log.info(
                "Model: %d blocks, frame_seqlen=%d, num_action=%d, num_state=%d, num_heads=%d",
                len(blocks), sa.frame_seqlen, sa.num_action_per_block,
                sa.num_state_per_block, sa.num_heads,
            )
        except Exception as e:
            log.warning("Could not read model info: %s", e)

        eval_rank0(args, policy, signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
