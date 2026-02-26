"""DreamZero LIBERO distributed evaluation.

Evaluates a DreamZero model finetuned on LIBERO.
Must be launched with torchrun (14B model requires multi-GPU tensor parallelism).

Usage:
    torchrun --standalone --nproc_per_node=2 examples/libero/dreamzero_eval.py \
        --model-path /path/to/dreamzero_libero_lora \
        --task-suite-name libero_10 \
        --num-trials-per-task 20

Key differences vs Pi0 main.py:
  - NO 180-deg image flip (DreamZero trained on raw LIBERO images)
  - State = joint positions, keys: state.joint_position / state.gripper_position
  - Actions: action.joint_position (N, 7) = [EEF delta x,y,z,rx,ry,rz, gripper]
  - Distributed inference: rank 0 runs eval, other ranks participate in forward pass
"""

import collections
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import pickle
import sys
import traceback

import imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from scipy.stats import gaussian_kde
import tqdm
import tyro
from tianshou.data import Batch

_DREAMZERO_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    model_path: str  # Path to DreamZero LIBERO checkpoint dir

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    replan_steps: int = 4       # max_chunk_size used in libero_training.sh
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


# -- Prompt helpers (mirrored from main.py) --

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
    "open": "close", "close": "open", "pick": "place", "place": "pick",
    "pick up": "put down", "put": "pick up", "push": "pull", "pull": "push",
    "turn on": "turn off", "turn off": "turn on", "lift": "lower", "lower": "lift",
    "left": "right", "right": "left", "top": "bottom", "bottom": "top",
    "front": "back", "back": "front", "into": "out of", "out of": "into",
    "on": "off", "off": "on",
}


def perturb_prompt(original, mode, all_tasks=None, custom=""):
    if mode == "original": return original
    if mode == "custom":   return custom
    if mode == "empty":    return ""
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


# -- Distributed helpers (adapted from socket_test_optimized_AR.py) --

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
    """Non-rank-0 processes participate in distributed forward passes."""
    rank = dist.get_rank()
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    logging.info("Rank %d: worker loop started", rank)
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
            logging.error("Rank %d error:\n%s", rank, traceback.format_exc())
            break


# -- Observation / action helpers --

def extract_actions(result_batch):
    """Extract (N, 7) float32 action array from DreamZero result_batch."""
    act = result_batch.act
    items = list(act.items()) if isinstance(act, dict) else [
        (a, getattr(act, a)) for a in dir(act)
        if "joint_position" in a and not a.startswith("_")
    ]
    for k, v in items:
        if "joint_position" in k:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            v = np.array(v, dtype=np.float32)
            return v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v.reshape(1, -1)
    raise ValueError("action.joint_position not found in result_batch.act: %s" % act)


# -- Metrics (condensed from main.py) --

def _json_default(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)


def compute_smoothness(action_log):
    a = np.array([e["action"] for e in action_log], dtype=np.float32)
    if len(a) < 2: return {}
    d = np.diff(a, axis=0)
    return {
        "mean_delta_norm":    float(np.mean(np.linalg.norm(d[:, :6], axis=1))),
        "mean_gripper_delta": float(np.mean(np.abs(d[:, 6]))),
    }


def kde_entropy(logs, action_dim=7):
    all_a = []
    for log in logs:
        a = np.array([e["action"] for e in log], dtype=np.float32)
        if a.ndim == 2 and a.shape[1] >= action_dim:
            all_a.append(a[:, :action_dim])
    if not all_a: return {}
    A = np.concatenate(all_a, axis=0)
    if A.shape[0] < A.shape[1] + 1: return {}
    try:
        kde = gaussian_kde(A.T)
        return {"action_entropy_kde": float(-np.mean(kde.logpdf(A.T))),
                "num_samples": int(A.shape[0])}
    except np.linalg.LinAlgError:
        return {}


def entropy_triplet(logs, successes, action_dim=7):
    return {
        "all":     kde_entropy(logs, action_dim),
        "success": kde_entropy([l for l, ok in zip(logs, successes) if ok],     action_dim),
        "failure": kde_entropy([l for l, ok in zip(logs, successes) if not ok], action_dim),
        "counts": {"all": len(logs), "success": sum(successes),
                   "failure": sum(not s for s in successes)},
    }


# -- LIBERO helpers --

def get_libero_env(task, resolution, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution
    )
    env.seed(seed)
    return env, task.language


_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
    "libero_90_obj": 400, "libero_90_spa": 400, "libero_90_act": 400, "libero_90_com": 400,
}


# -- Rank-0 evaluation loop --

def eval_rank0(args, policy, signal_group):
    np.random.seed(args.seed)
    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode, rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac, translate_y_frac=args.translate_y_frac,
    )
    pol_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode, random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std, object_shift_y_std=args.object_shift_y_std,
    )
    pol_rng   = np.random.default_rng(args.seed + 9999)
    suite     = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks   = suite.n_tasks
    max_steps = _MAX_STEPS[args.task_suite_name]
    out_dir   = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_descs = []
    for i in range(n_tasks):
        try:    all_descs.append(str(suite.get_task(i).language))
        except: all_descs.append(f"task_{i}")
    all_descs = list(dict.fromkeys(all_descs))

    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    total_eps = total_succ = 0

    for task_id in tqdm.tqdm(range(n_tasks), desc="Tasks"):
        task        = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, desc   = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_eps = task_succ = 0
        groups   = collections.defaultdict(lambda: {"logs": [], "successes": []})
        ep_jsons = []

        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"T{task_id}", leave=False):
            env.reset()
            obs        = env.set_init_state(init_states[ep_idx])
            obj_shifts = apply_object_shift(env, pol_cfg, pol_rng)
            ep_prompt  = perturb_prompt(
                str(desc), args.prompt_mode, all_descs, custom=args.custom_prompt
            )

            action_plan  = collections.deque()
            action_log   = []
            replay_imgs  = []
            action_chunk = None
            t = 0; done = False; reward = 0.0; is_new_chunk = False

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # DreamZero: raw images, NO flip
                    img   = perturb_image(np.ascontiguousarray(obs["agentview_image"]),          vis_cfg)
                    wrist = perturb_image(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]), vis_cfg)
                    replay_imgs.append(img)

                    is_new_chunk = not action_plan
                    if is_new_chunk:
                        dz_obs = {
                            "video.agentview_rgb":    img[None].astype(np.uint8),
                            "video.eye_in_hand_rgb":  wrist[None].astype(np.uint8),
                            "state.joint_position":
                                np.array(obs["robot0_joint_pos"],    dtype=np.float64).reshape(1, -1),
                            "state.gripper_position":
                                np.array(obs["robot0_gripper_qpos"], dtype=np.float64)[:1].reshape(1, -1),
                            "annotation.language.language_instruction": ep_prompt,
                        }
                        # Signal workers to run inference (signal=0 means "infer")
                        signal.fill_(0)
                        dist.broadcast(signal, src=0, group=signal_group)
                        _broadcast_obs(dz_obs)

                        dist.barrier()
                        with torch.no_grad():
                            result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=dz_obs))
                        dist.barrier()

                        action_chunk = extract_actions(result_batch)    # (N, 7)
                        n_exec = min(args.replan_steps, len(action_chunk))
                        action_plan.extend(action_chunk[:n_exec])

                    pol_action = action_plan.popleft()
                    action, perturbed = maybe_perturb_action(pol_action, pol_cfg, pol_rng)
                    obs, reward, done, info = env.step(action.tolist())

                    entry = {
                        "t": int(t),
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "action_perturbed": bool(perturbed),
                        "reward": float(reward), "done": bool(done),
                        "is_chunk_start": bool(is_new_chunk),
                    }
                    if is_new_chunk and action_chunk is not None:
                        entry["full_action_chunk"] = action_chunk.tolist()
                    action_log.append(entry)
                    t += 1
                    if done:
                        task_succ += 1; total_succ += 1
                        logging.info("  SUCCESS t=%d", t)
                        break
                except Exception:
                    logging.error("Step error:\n%s", traceback.format_exc())
                    break

            task_eps += 1; total_eps += 1
            success = bool(done and reward > 0)
            groups[ep_prompt]["logs"].append(action_log)
            groups[ep_prompt]["successes"].append(success)
            ep_jsons.append({
                "task_id": task_id, "task_description": str(desc),
                "episode_idx": ep_idx, "prompt_mode": args.prompt_mode,
                "prompt_used": ep_prompt, "success": success,
                "num_steps": t - args.num_steps_wait, "object_shifts": obj_shifts,
                "smoothness": compute_smoothness(action_log),
            })
            if replay_imgs:
                imageio.mimsave(
                    str(out_dir / f"task{task_id:02d}_ep{ep_idx:02d}.mp4"), replay_imgs, fps=10
                )

        env.close()
        for ej in ep_jsons:
            g = groups[ej["prompt_used"]]
            ej["action_entropy_group"] = entropy_triplet(g["logs"], g["successes"])
            jp = out_dir / f"task{ej['task_id']:02d}_ep{ej['episode_idx']:02d}.json"
            with open(jp, "w") as f:
                json.dump(ej, f, default=_json_default, indent=2)

        rate = task_succ / task_eps if task_eps else 0.0
        logging.info("Task %d: %d/%d = %.1f%%  [%s]", task_id, task_succ, task_eps, rate * 100, desc)

    # Shutdown workers
    signal.fill_(1)
    dist.broadcast(signal, src=0, group=signal_group)

    total_rate = total_succ / total_eps if total_eps else 0.0
    logging.info("\n%s\n%s: %d/%d = %.1f%%\n%s",
                 "=" * 60, args.task_suite_name, total_succ, total_eps, total_rate * 100, "=" * 60)
    summary = {
        "task_suite_name": args.task_suite_name, "model_path": args.model_path,
        "num_trials_per_task": args.num_trials_per_task, "seed": args.seed,
        "total_episodes": total_eps, "total_successes": total_succ, "success_rate": total_rate,
        "prompt_mode": args.prompt_mode, "visual_perturb_mode": args.visual_perturb_mode,
        "policy_perturb_mode": args.policy_perturb_mode,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Summary -> %s", out_dir / "summary.json")


# -- Entry point --

def main(args):
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank{rank}] %(asctime)s %(message)s",
        force=True,
    )

    mesh         = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
    signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=10))

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
    )

    if rank == 0:
        eval_rank0(args, policy, signal_group)
    else:
        worker_loop(policy, signal_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
