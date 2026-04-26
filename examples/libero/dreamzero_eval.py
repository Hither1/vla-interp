"""DreamZero LIBERO distributed evaluation.

Key safety changes vs older version:
- Only rank 0 imports and uses LIBERO / robosuite / MuJoCo envs.
- LOCAL_RANK is used for torch.cuda.set_device().
- Env close is always attempted in finally.
- Worker failures raise instead of silently breaking.
- Rank 0 always broadcasts shutdown in finally.
- EGL-related env pinning is done before any rank-0 env creation.

Usage:
    torchrun --standalone --nproc_per_node=2 examples/libero/dreamzero_eval.py \
        --model-path /path/to/dreamzero_libero_lora \
        --task-suite-name libero_10 \
        --num-trials-per-task 20
"""

import collections
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import pickle
import sys
import traceback
from typing import Any

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import tyro
from safetensors.torch import load_file
from scipy.stats import gaussian_kde
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

_DREAMZERO_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)

_ANALYSIS_ATTN_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "analysis" / "attention")
if _ANALYSIS_ATTN_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_ATTN_DIR)

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanAttentionBlock,
    clear_attn_buffer,
    get_attn_buffer,
    set_attn_capture,
)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224

# Per-block action_register_length captured from forward pre-hook
_BLOCK_ARL: dict[int, int] = {}
# Cross-attention (action→text) calls
_CROSS_ATTN_BUFFER: list[dict[str, Any]] = []


@dataclasses.dataclass
class Args:
    model_path: str

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    replan_steps: int = 4
    seed: int = 7
    video_out_path: str = "data/libero/dreamzero/videos"

    prompt_mode: str = "original"
    custom_prompt: str = ""

    visual_perturb_mode: str = "none"
    rotation_degrees: float = 30.0
    translate_x_frac: float = 0.2
    translate_y_frac: float = 0.0

    policy_perturb_mode: str = "none"
    random_action_prob: float = 0.25
    random_action_scale: float = 1.0
    object_shift_x_std: float = 0.05
    object_shift_y_std: float = 0.0

    enable_dit_cache: bool = True

    visualize_attention: bool = False
    attn_alpha: float = 0.5

    compute_attention_ratio: bool = False
    compute_attention_iou: bool = False
    iou_threshold_percentile: float = 90.0

    task_id: int = -1  # -1 = run all tasks; otherwise run only this task index


SYNONYM_MAP = {
    "pick": ["grab", "grasp", "take", "lift"],
    "place": ["put", "set", "position", "lay"],
    "open": ["unlock", "unfasten", "unseal"],
    "close": ["shut", "seal", "fasten"],
    "push": ["shove", "press", "move"],
    "pull": ["drag", "draw", "tug"],
    "turn": ["rotate", "twist", "spin"],
    "put": ["place", "set", "position"],
    "move": ["shift", "transfer", "relocate"],
    "take": ["grab", "pick", "grasp"],
    "lift": ["raise", "pick up", "elevate"],
}

OPPOSITE_MAP = {
    "open": "close",
    "close": "open",
    "pick": "place",
    "place": "pick",
    "pick up": "put down",
    "put": "pick up",
    "push": "pull",
    "pull": "push",
    "turn on": "turn off",
    "turn off": "turn on",
    "lift": "lower",
    "lower": "lift",
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "front": "back",
    "back": "front",
    "into": "out of",
    "out of": "into",
    "on": "off",
    "off": "on",
}

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "libero_90_obj": 400,
    "libero_90_spa": 400,
    "libero_90_act": 400,
    "libero_90_com": 400,
}


def _configure_rank0_render_env(local_rank: int) -> None:
    """Best-effort EGL-related configuration before any LIBERO/MuJoCo env import."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Some systems honor this for EGL device selection.
    os.environ.setdefault("EGL_DEVICE_ID", str(local_rank))


def load_model(checkpoint_dir: pathlib.Path) -> VLA:
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    config = VLAConfig(**config_dict)
    if isinstance(config.action_head_cfg.get("config"), dict):
        config.action_head_cfg["config"]["defer_lora_injection"] = False

    model = VLA(config)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"
    state_dict = {}

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard_file in set(index["weight_map"].values()):
            state_dict.update(load_file(str(checkpoint_dir / shard_file)))
    elif single_path.exists():
        state_dict = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"No safetensors found in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def perturb_prompt(original: str, mode: str, all_tasks=None, custom: str = "") -> str:
    if mode == "original":
        return original
    if mode == "custom":
        return custom
    if mode == "empty":
        return ""
    if mode == "shuffle":
        words = original.split()
        np.random.shuffle(words)
        return " ".join(words)
    if mode == "random":
        others = [t for t in (all_tasks or []) if t != original]
        return str(np.random.choice(others)) if others else original
    if mode == "synonym":
        r = original.lower()
        for w, syns in SYNONYM_MAP.items():
            if w in r:
                r = r.replace(w, str(np.random.choice(syns)), 1)
                break
        return r
    if mode == "opposite":
        r = original.lower()
        for phrase in sorted(OPPOSITE_MAP, key=len, reverse=True):
            if phrase in r:
                r = r.replace(phrase, OPPOSITE_MAP[phrase], 1)
                break
        return r
    return original


def _compute_attn_map() -> np.ndarray | None:
    buf = get_attn_buffer()
    if not buf:
        return None

    frame_seqlen = buf[0]["frame_seqlen"]

    best_h, best_w = 1, frame_seqlen
    for h in range(1, int(math.isqrt(frame_seqlen)) + 1):
        if frame_seqlen % h == 0:
            w = frame_seqlen // h
            if abs(h - w) < abs(best_h - best_w):
                best_h, best_w = h, w
    h_grid, w_grid = best_h, best_w

    maps = []
    for entry in buf:
        attn = entry["attn"]
        fsl = entry["frame_seqlen"]
        n_vis = attn.shape[1]
        if n_vis < fsl:
            continue
        n_frames = n_vis // fsl
        a = attn[0, : n_frames * fsl]
        a = a.reshape(n_frames, fsl).mean(0)
        maps.append(a)

    if not maps:
        return None

    mean_map = torch.stack(maps).mean(0)
    return mean_map.numpy().reshape(h_grid, w_grid)


def _overlay_attn(img_uint8: np.ndarray, attn_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = img_uint8.shape[:2]
    am = cv2.resize(attn_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    heat = cv2.applyColorMap((am * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(img_uint8 * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


def _install_ratio_hooks(policy) -> list:
    global _BLOCK_ARL, _CROSS_ATTN_BUFFER
    _BLOCK_ARL.clear()
    _CROSS_ATTN_BUFFER.clear()

    try:
        blocks = policy.trained_model.action_head.model.blocks
    except AttributeError:
        logging.warning("Cannot find DiT blocks for ratio hooks")
        return []

    handles = []

    for block_idx, block in enumerate(blocks):
        if not isinstance(block, CausalWanAttentionBlock):
            continue

        n_act = block.self_attn.num_action_per_block
        n_state = block.self_attn.num_state_per_block

        def _make_block_pre_hook(idx):
            def pre_hook(module, args, kwargs):
                arl = kwargs.get("action_register_length")
                if arl is not None:
                    _BLOCK_ARL[idx] = arl
            return pre_hook

        handles.append(block.register_forward_pre_hook(_make_block_pre_hook(block_idx), with_kwargs=True))

        def _make_cross_hook(idx, na, ns):
            def hook(module, args, output):
                if len(args) < 2:
                    return
                normed_x = args[0]
                context = args[1]

                arl = _BLOCK_ARL.get(idx)
                if not arl:
                    return

                B, S, C = normed_x.shape
                T = context.shape[1]
                if T == 0:
                    return

                n = module.num_heads
                d = module.head_dim

                chunk_size = arl // (na + ns)
                action_horizon = chunk_size * na
                state_horizon = chunk_size * ns

                action_start = S - action_horizon - state_horizon
                action_end = S - state_horizon
                if action_start < 0 or action_end > S or action_start >= action_end:
                    return

                with torch.no_grad():
                    x_act = normed_x[:, action_start:action_end]
                    q = module.norm_q(module.q(x_act)).view(B, -1, n, d)

                    text_ctx = context[:, 257:] if hasattr(module, "k_img") else context
                    if text_ctx.shape[1] == 0:
                        return

                    k = module.norm_k(module.k(text_ctx)).view(B, -1, n, d)

                    q_f = q.float().permute(0, 2, 1, 3)
                    k_f = k.float().permute(0, 2, 1, 3)
                    scale = d ** -0.5
                    attn = torch.softmax(torch.matmul(q_f * scale, k_f.transpose(-2, -1)), dim=-1)
                    attn_np = attn[0].mean(0).cpu().float().numpy()

                _CROSS_ATTN_BUFFER.append({"attn": attn_np, "T": text_ctx.shape[1]})

            return hook

        handles.append(block.cross_attn.register_forward_hook(_make_cross_hook(block_idx, n_act, n_state)))

    return handles


def _remove_hooks(handles: list) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _compute_ratio(attn_map: np.ndarray | None) -> dict:
    visual_mass = float(attn_map.sum()) if attn_map is not None else 0.0

    ling_values = []
    for call in _CROSS_ATTN_BUFFER:
        attn = call["attn"]
        T = int(call["T"])
        ling_values.append(float(attn.mean()) * T)

    linguistic_mass = float(np.mean(ling_values)) if ling_values else 0.0

    if linguistic_mass > 1e-8:
        ratio = visual_mass / linguistic_mass
    else:
        ratio = float("inf") if visual_mass > 1e-8 else 0.0

    total = visual_mass + linguistic_mass
    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "visual_linguistic_ratio": ratio,
        "visual_fraction": visual_mass / max(total, 1e-8),
        "linguistic_fraction": linguistic_mass / max(total, 1e-8),
        "num_cross_attn_calls": len(ling_values),
    }


def _compute_iou(attn_map: np.ndarray, seg_mask: np.ndarray, percentile: float = 90.0) -> dict:
    h, w = seg_mask.shape[:2]
    hm = cv2.resize(attn_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    hmin, hmax = hm.min(), hm.max()
    if hmax > hmin:
        hm = (hm - hmin) / (hmax - hmin)

    fg = (seg_mask > 0).astype(bool)
    total_attn = float(hm.sum())
    attention_mass = float(hm[fg].sum() / total_attn) if total_attn > 0 else 0.0

    max_idx = np.unravel_index(np.argmax(hm), hm.shape)
    pointing_hit = bool(fg[max_idx])

    thresh = float(np.percentile(hm, percentile))
    pred = hm >= thresh
    intersection = float(np.logical_and(pred, fg).sum())
    union = float(np.logical_or(pred, fg).sum())
    iou = intersection / union if union > 0 else 0.0

    denom = float(pred.sum() + fg.sum())
    dice = float(2 * intersection / denom) if denom > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "attention_mass": attention_mass,
        "pointing_hit": pointing_hit,
        "threshold": float(thresh),
        "percentile": float(percentile),
        "fg_pixels": int(fg.sum()),
    }


def _summarize_attn_steps(step_results: list) -> dict:
    if not step_results:
        return {}
    keys = [k for k in step_results[0] if isinstance(step_results[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in step_results if k in r]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    out["num_steps"] = len(step_results)
    return out


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    denom = np.sqrt(max(1.0 - quat[3] ** 2, 0.0))
    if denom < 1e-8:
        return np.zeros(3)
    return (quat[:3] / denom) * 2.0 * np.arccos(np.clip(quat[3], -1.0, 1.0))


def _to_numpy_2d(v):
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    v = np.array(v, dtype=np.float32)
    return v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v.reshape(1, -1)


def extract_actions(result_batch):
    act = result_batch.act
    items = dict(act.items()) if isinstance(act, dict) else {
        a: getattr(act, a) for a in dir(act) if not a.startswith("_")
    }

    joint, gripper = None, None
    for k, v in items.items():
        if "joint_position" in k and "gripper" not in k:
            joint = _to_numpy_2d(v)
        elif "gripper_position" in k or "gripper" in k:
            g = _to_numpy_2d(v)
            gripper = g.reshape(g.shape[0], -1) if g.ndim == 2 else g.reshape(-1, 1)

    if joint is None:
        raise ValueError(f"action.joint_position not found in result_batch.act: {act}")
    if gripper is None:
        raise ValueError(f"action.gripper_position not found in result_batch.act: {act}")
    if gripper.shape[0] != joint.shape[0]:
        gripper = np.broadcast_to(gripper, (joint.shape[0], gripper.shape[-1])).copy()
    return np.concatenate([joint, gripper], axis=-1)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def compute_smoothness(action_log):
    a = np.array([e["action"] for e in action_log], dtype=np.float32)
    if len(a) < 2:
        return {}
    d = np.diff(a, axis=0)
    return {
        "mean_delta_norm": float(np.mean(np.linalg.norm(d[:, :6], axis=1))),
        "mean_gripper_delta": float(np.mean(np.abs(d[:, 6]))),
    }


def kde_entropy(logs, action_dim=7):
    all_a = []
    for log in logs:
        a = np.array([e["action"] for e in log], dtype=np.float32)
        if a.ndim == 2 and a.shape[1] >= action_dim:
            all_a.append(a[:, :action_dim])

    if not all_a:
        return {}
    A = np.concatenate(all_a, axis=0)
    if A.shape[0] < A.shape[1] + 1:
        return {}

    try:
        kde = gaussian_kde(A.T)
        return {
            "action_entropy_kde": float(-np.mean(kde.logpdf(A.T))),
            "num_samples": int(A.shape[0]),
        }
    except np.linalg.LinAlgError:
        return {}


def entropy_triplet(logs, successes, action_dim=7):
    return {
        "all": kde_entropy(logs, action_dim),
        "success": kde_entropy([l for l, ok in zip(logs, successes) if ok], action_dim),
        "failure": kde_entropy([l for l, ok in zip(logs, successes) if not ok], action_dim),
        "counts": {
            "all": len(logs),
            "success": sum(successes),
            "failure": sum(not s for s in successes),
        },
    }


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


def _broadcast_shutdown(signal_group) -> None:
    if not dist.is_initialized():
        return
    if dist.get_rank() != 0:
        return
    try:
        signal = torch.tensor([1], dtype=torch.int32, device="cpu")
        dist.broadcast(signal, src=0, group=signal_group)
    except Exception:
        logging.exception("Failed to broadcast shutdown signal")


def worker_loop(policy, signal_group):
    rank = dist.get_rank()
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    logging.info("Rank %d: worker loop started", rank)

    while True:
        dist.broadcast(signal, src=0, group=signal_group)
        if signal.item() == 1:
            logging.info("Rank %d: received shutdown", rank)
            break

        obs = _receive_obs()
        dist.barrier()
        with torch.no_grad():
            policy.lazy_joint_forward_causal(Batch(obs=obs))
        dist.barrier()


def _import_benchmark():
    from libero.libero import benchmark
    return benchmark


def _get_libero_path_fn():
    from libero.libero import get_libero_path
    return get_libero_path


def get_libero_env(task, resolution, seed, compute_attention_iou=False):
    get_libero_path = _get_libero_path_fn()
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    if compute_attention_iou:
        try:
            from libero.libero.envs.env_wrapper import SegmentationRenderEnv
            env = SegmentationRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=resolution,
                camera_widths=resolution,
            )
            np.random.seed(seed)
            return env, task.language
        except Exception as e:
            logging.warning("SegmentationRenderEnv unavailable (%s). Falling back to OffScreenRenderEnv.", e)

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    np.random.seed(seed)
    return env, task.language


def eval_rank0(args, policy, signal_group, local_rank: int):
    _configure_rank0_render_env(local_rank)

    benchmark = _import_benchmark()

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

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_descs = []
    for i in range(n_tasks):
        try:
            all_descs.append(str(suite.get_task(i).language))
        except Exception:
            all_descs.append(f"task_{i}")
    all_descs = list(dict.fromkeys(all_descs))

    total_eps = 0
    total_succ = 0
    per_task_results = []

    need_attn_capture = (
        args.visualize_attention or args.compute_attention_ratio or args.compute_attention_iou
    )

    try:
        for task_id in tqdm.tqdm(range(n_tasks), desc="Tasks"):
            if args.task_id >= 0 and task_id != args.task_id:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            desc = task.language

            task_eps = 0
            task_succ = 0
            groups = collections.defaultdict(lambda: {"logs": [], "successes": []})
            ep_jsons = []

            for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"T{task_id}", leave=False):
                env = None
                ratio_hooks = []
                try:
                    env, _ = get_libero_env(
                        task=task,
                        resolution=LIBERO_ENV_RESOLUTION,
                        seed=args.seed,
                        compute_attention_iou=args.compute_attention_iou,
                    )

                    env.reset()
                    obs = env.set_init_state(init_states[ep_idx])

                    obj_shifts = apply_object_shift(env, pol_cfg, pol_rng)
                    ep_prompt = perturb_prompt(str(desc), args.prompt_mode, all_descs, custom=args.custom_prompt)

                    action_plan = collections.deque()
                    action_log = []
                    replay_imgs = []
                    action_chunk = None
                    current_attn_map = None
                    attn_step_results = []
                    seg_key = None
                    t = 0
                    done = False
                    reward = 0.0
                    is_new_chunk = False

                    while t < max_steps + args.num_steps_wait:
                        try:
                            if t < args.num_steps_wait:
                                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                                if args.compute_attention_iou and seg_key is None:
                                    for cand in (
                                        "agentview_segmentation_instance",
                                        "agentview_segmentation_class",
                                        "agentview_seg",
                                        "agentview_segmentation",
                                    ):
                                        if cand in obs:
                                            seg_key = cand
                                            break
                                    if seg_key is None:
                                        for k in obs:
                                            if "seg" in k.lower() and "agentview" in k.lower():
                                                seg_key = k
                                                break
                                t += 1
                                continue

                            img = perturb_image(np.ascontiguousarray(obs["agentview_image"]), vis_cfg)
                            wrist = perturb_image(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]), vis_cfg)

                            frame_vis = img[::-1, ::-1].copy()
                            if args.visualize_attention and action_plan and current_attn_map is not None:
                                frame_vis = _overlay_attn(frame_vis, current_attn_map, alpha=args.attn_alpha)
                            replay_imgs.append(frame_vis)

                            is_new_chunk = not action_plan
                            if is_new_chunk:
                                dz_obs = {
                                    "video.agentview_rgb": img[None].astype(np.uint8),
                                    "video.eye_in_hand_rgb": wrist[None].astype(np.uint8),
                                    "state.joint_position": np.concatenate([
                                        np.array(obs["robot0_eef_pos"], dtype=np.float64),
                                        _quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)),
                                    ]).reshape(1, -1),
                                    "state.gripper_position": np.array(
                                        obs["robot0_gripper_qpos"], dtype=np.float64
                                    )[:2].reshape(1, -1),
                                    "annotation.language.language_instruction": ep_prompt,
                                }

                                if task_id == 0 and ep_idx == 0 and t == args.num_steps_wait:
                                    logging.info(
                                        "[Visual input check] agentview_rgb: shape=%s dtype=%s range=[%d,%d] "
                                        "eye_in_hand_rgb: shape=%s range=[%d,%d]",
                                        dz_obs["video.agentview_rgb"].shape,
                                        dz_obs["video.agentview_rgb"].dtype,
                                        dz_obs["video.agentview_rgb"].min(),
                                        dz_obs["video.agentview_rgb"].max(),
                                        dz_obs["video.eye_in_hand_rgb"].shape,
                                        dz_obs["video.eye_in_hand_rgb"].min(),
                                        dz_obs["video.eye_in_hand_rgb"].max(),
                                    )

                                signal = torch.tensor([0], dtype=torch.int32, device="cpu")
                                dist.broadcast(signal, src=0, group=signal_group)
                                _broadcast_obs(dz_obs)

                                if need_attn_capture:
                                    clear_attn_buffer()
                                    set_attn_capture(True)

                                if args.compute_attention_ratio:
                                    _remove_hooks(ratio_hooks)
                                    ratio_hooks = _install_ratio_hooks(policy)

                                dist.barrier()
                                with torch.no_grad():
                                    result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
                                dist.barrier()

                                if need_attn_capture:
                                    set_attn_capture(False)
                                    current_attn_map = _compute_attn_map()
                                    if args.visualize_attention and current_attn_map is not None:
                                        replay_imgs[-1] = _overlay_attn(
                                            replay_imgs[-1], current_attn_map, alpha=args.attn_alpha
                                        )

                                if (args.compute_attention_ratio or args.compute_attention_iou) and current_attn_map is not None:
                                    step_metrics = {"t": int(t)}

                                    if args.compute_attention_ratio:
                                        ratio_result = _compute_ratio(current_attn_map)
                                        step_metrics.update(ratio_result)
                                        logging.info(
                                            "t=%d visual=%.3f linguistic=%.3f ratio=%.3f",
                                            t,
                                            ratio_result["visual_mass"],
                                            ratio_result["linguistic_mass"],
                                            ratio_result["visual_linguistic_ratio"]
                                            if np.isfinite(ratio_result["visual_linguistic_ratio"])
                                            else float("nan"),
                                        )

                                    if args.compute_attention_iou and seg_key is not None and seg_key in obs:
                                        seg_raw = obs[seg_key]
                                        if seg_raw.ndim == 3:
                                            seg_raw = seg_raw[:, :, 0]
                                        iou_result = _compute_iou(
                                            current_attn_map,
                                            seg_raw,
                                            percentile=args.iou_threshold_percentile,
                                        )
                                        step_metrics.update({f"iou_{k}": v for k, v in iou_result.items()})
                                        logging.info(
                                            "t=%d IoU=%.3f mass=%.3f pointing=%s",
                                            t,
                                            iou_result["iou"],
                                            iou_result["attention_mass"],
                                            "hit" if iou_result["pointing_hit"] else "miss",
                                        )

                                    attn_step_results.append(step_metrics)

                                action_chunk = extract_actions(result_batch)
                                n_exec = min(args.replan_steps, len(action_chunk))
                                action_plan.extend(action_chunk[:n_exec])

                            pol_action = action_plan.popleft()
                            action, perturbed = maybe_perturb_action(pol_action, pol_cfg, pol_rng)
                            obs, reward, done, info = env.step(action.tolist())

                            entry = {
                                "t": int(t),
                                "action": np.asarray(action, dtype=np.float32).tolist(),
                                "action_perturbed": bool(perturbed),
                                "reward": float(reward),
                                "done": bool(done),
                                "is_chunk_start": bool(is_new_chunk),
                            }
                            if is_new_chunk and action_chunk is not None:
                                entry["full_action_chunk"] = action_chunk.tolist()
                            action_log.append(entry)

                            t += 1
                            if done and reward > 0:
                                task_succ += 1
                                total_succ += 1
                                break

                        except Exception:
                            logging.error("Step error:\n%s", traceback.format_exc())
                            break

                    _remove_hooks(ratio_hooks)
                    ratio_hooks = []

                    task_eps += 1
                    total_eps += 1
                    success = bool(done and reward > 0)

                    logging.info(
                        "Ep %d/%d: %s (t=%d, task %d/%d so far)",
                        ep_idx + 1,
                        args.num_trials_per_task,
                        "SUCCESS" if success else "FAILURE",
                        t - args.num_steps_wait,
                        task_succ,
                        task_eps,
                    )

                    groups[ep_prompt]["logs"].append(action_log)
                    groups[ep_prompt]["successes"].append(success)

                    ep_entry = {
                        "task_id": task_id,
                        "task_description": str(desc),
                        "episode_idx": ep_idx,
                        "prompt_mode": args.prompt_mode,
                        "prompt_used": ep_prompt,
                        "success": success,
                        "num_steps": t - args.num_steps_wait,
                        "object_shifts": obj_shifts,
                        "smoothness": compute_smoothness(action_log),
                    }
                    if attn_step_results:
                        ep_entry["attention_steps"] = attn_step_results
                        ep_entry["attention_summary"] = _summarize_attn_steps(attn_step_results)
                    ep_jsons.append(ep_entry)

                    prompt_slug = ep_prompt.lower().replace(" ", "_")
                    prompt_slug = "".join(c if c.isalnum() or c == "_" else "" for c in prompt_slug)[:60]
                    vid_name = f"task{task_id:02d}_ep{ep_idx:02d}_{prompt_slug}.mp4"
                    if replay_imgs:
                        imageio.mimsave(str(out_dir / vid_name), replay_imgs, fps=10)

                except Exception:
                    logging.error("Episode error:\n%s", traceback.format_exc())
                finally:
                    _remove_hooks(ratio_hooks)
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            logging.exception("Failed to close env")

            for ej in ep_jsons:
                g = groups[ej["prompt_used"]]
                ej["action_entropy_group"] = entropy_triplet(g["logs"], g["successes"])
                p_slug = ej["prompt_used"].lower().replace(" ", "_")
                p_slug = "".join(c if c.isalnum() or c == "_" else "" for c in p_slug)[:60]
                jp = out_dir / f"task{ej['task_id']:02d}_ep{ej['episode_idx']:02d}_{p_slug}.json"
                with open(jp, "w") as f:
                    json.dump(ej, f, default=_json_default, indent=2)

            rate = task_succ / task_eps if task_eps else 0.0
            per_task_results.append({
                "task_id": task_id,
                "task_description": str(desc),
                "successes": task_succ,
                "trials": task_eps,
                "success_rate": rate,
            })
            running_rate = total_succ / total_eps if total_eps else 0.0
            logging.info(
                "Task %d: %d/%d = %.1f%% [%s] (running: %d/%d = %.1f%%)",
                task_id,
                task_succ,
                task_eps,
                rate * 100.0,
                desc,
                total_succ,
                total_eps,
                running_rate * 100.0,
            )

        total_rate = total_succ / total_eps if total_eps else 0.0
        logging.info(
            "\n%s\n%s: %d/%d = %.1f%%\n%s",
            "=" * 60,
            args.task_suite_name,
            total_succ,
            total_eps,
            total_rate * 100.0,
            "=" * 60,
        )

        summary = {
            "task_suite_name": args.task_suite_name,
            "model_path": args.model_path,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "total_episodes": total_eps,
            "total_successes": total_succ,
            "success_rate": total_rate,
            "prompt_mode": args.prompt_mode,
            "visual_perturb_mode": args.visual_perturb_mode,
            "policy_perturb_mode": args.policy_perturb_mode,
            "per_task": per_task_results,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logging.info("Summary -> %s", out_dir / "summary.json")

    finally:
        _broadcast_shutdown(signal_group)


def main(args: Args):
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank{rank}] %(asctime)s %(message)s",
        force=True,
    )

    logging.info(
        "Initialized dist: rank=%d world_size=%d local_rank=%d cuda_device=%d",
        rank,
        world_size,
        local_rank,
        torch.cuda.current_device(),
    )

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
    signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=10))

    if rank == 0:
        logging.info("Loading model from %s", args.model_path)
    model = load_model(pathlib.Path(args.model_path))

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
        pretrained_model=model,
    )

    try:
        if rank == 0:
            eval_rank0(args, policy, signal_group, local_rank=local_rank)
        else:
            worker_loop(policy, signal_group)
    except Exception:
        logging.error("Fatal error on rank %d:\n%s", rank, traceback.format_exc())
        raise
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main(tyro.cli(Args))