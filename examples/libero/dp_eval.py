"""Evaluate Diffusion Policy (dp_scratch) on LIBERO simulation benchmarks.

Runs the diffusion policy locally (no WebSocket server) on a Linux cluster.
Mirrors the structure of cosmos_eval.py for consistent metrics and JSON output.

Usage:
    # Full evaluation on one suite
    python examples/libero/dp_eval.py \
        --ckpt checkpoints/dp/ckpt_300.pt \
        --task-suite-name libero_spatial \
        --num-trials-per-task 20 \
        --seed 0

    # Quick smoke test (1 trial)
    python examples/libero/dp_eval.py \
        --ckpt checkpoints/dp/ckpt_300.pt \
        --task-suite-name libero_10 \
        --num-trials-per-task 1

    # Prompt perturbation
    python examples/libero/dp_eval.py \
        --ckpt checkpoints/dp/ckpt_300.pt \
        --task-suite-name libero_spatial \
        --prompt-mode synonym

Requires LIBERO environment:
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
"""

import collections
import dataclasses
import json
import logging
import pathlib

import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde
import torch
import tqdm
import tyro

from dp_scratch.model import DiffusionPolicy


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

# ── Prompt perturbation helpers ───────────────────────────────────────────────

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
    "open": "close", "close": "open",
    "pick": "place", "place": "pick",
    "pick up": "put down", "put": "pick up",
    "push": "pull", "pull": "push",
    "turn on": "turn off", "turn off": "turn on",
    "lift": "lower", "lower": "lift",
    "left": "right", "right": "left",
    "top": "bottom", "bottom": "top",
    "front": "back", "back": "front",
    "into": "out of", "out of": "into",
    "on": "off", "off": "on",
}


def perturb_prompt(original: str, mode: str = "original", all_tasks: list = None) -> str:
    if mode == "original":
        return original
    elif mode == "empty":
        return ""
    elif mode == "shuffle":
        words = original.split()
        np.random.shuffle(words)
        return " ".join(words)
    elif mode == "random":
        others = [t for t in all_tasks if t != original]
        return np.random.choice(others) if others else original
    elif mode == "synonym":
        result = original.lower()
        for word, synonyms in SYNONYM_MAP.items():
            if word in result:
                result = result.replace(word, np.random.choice(synonyms), 1)
                break
        return result
    elif mode == "opposite":
        result = original.lower()
        for phrase in sorted(OPPOSITE_MAP.keys(), key=len, reverse=True):
            if phrase in result:
                result = result.replace(phrase, OPPOSITE_MAP[phrase], 1)
                break
        return result
    return original


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_smoothness_metrics(action_log):
    actions = np.array([entry["action"] for entry in action_log])
    if len(actions) < 2:
        return {}

    deltas = np.diff(actions, axis=0)
    delta_norms = np.linalg.norm(deltas[:, :6], axis=1)
    pos_delta_norms = np.linalg.norm(deltas[:, :3], axis=1)
    rot_delta_norms = np.linalg.norm(deltas[:, 3:6], axis=1)
    gripper_deltas = np.abs(deltas[:, 6])

    accels = np.diff(deltas, axis=0)
    accel_norms = np.linalg.norm(accels[:, :6], axis=1)

    cosine_sims = []
    for i in range(len(deltas) - 1):
        d1, d2 = deltas[i, :6], deltas[i + 1, :6]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        cosine_sims.append(float(np.dot(d1, d2) / (n1 * n2)) if n1 > 1e-8 and n2 > 1e-8 else 1.0)

    gripper_actions = actions[:, 6]
    gripper_signs = np.sign(gripper_actions)
    gripper_transitions = []
    for i in range(1, len(gripper_signs)):
        if gripper_signs[i] != gripper_signs[i - 1] and gripper_signs[i] != 0:
            gripper_transitions.append({
                "t": action_log[i]["t"], "step_idx": i,
                "from": "open" if gripper_signs[i - 1] > 0 else "close",
                "to": "open" if gripper_signs[i] > 0 else "close",
            })

    chunk_start_set = {i for i, entry in enumerate(action_log) if entry.get("is_chunk_start", False)}
    boundary_deltas, within_deltas = [], []
    for i in range(len(delta_norms)):
        (boundary_deltas if (i + 1) in chunk_start_set else within_deltas).append(delta_norms[i])

    chunk_boundary_stats = {}
    if boundary_deltas:
        mean_within = float(np.mean(within_deltas)) if within_deltas else 0.0
        mean_boundary = float(np.mean(boundary_deltas))
        chunk_boundary_stats = {
            "mean_boundary_delta": mean_boundary,
            "mean_within_chunk_delta": mean_within,
            "boundary_ratio": mean_boundary / mean_within if mean_within > 1e-8 else float("inf"),
            "num_chunk_boundaries": len(boundary_deltas),
        }

    for i, entry in enumerate(action_log):
        if i < len(delta_norms):
            entry["action_delta_norm"] = float(delta_norms[i])
            entry["pos_delta_norm"] = float(pos_delta_norms[i])
            entry["rot_delta_norm"] = float(rot_delta_norms[i])
            entry["gripper_delta"] = float(gripper_deltas[i])
        if i < len(accel_norms):
            entry["action_accel_norm"] = float(accel_norms[i])
        if i < len(cosine_sims):
            entry["direction_consistency"] = float(cosine_sims[i])

    return {
        "mean_action_delta": float(np.mean(delta_norms)),
        "max_action_delta": float(np.max(delta_norms)),
        "std_action_delta": float(np.std(delta_norms)),
        "mean_action_accel": float(np.mean(accel_norms)) if len(accel_norms) > 0 else 0.0,
        "max_action_accel": float(np.max(accel_norms)) if len(accel_norms) > 0 else 0.0,
        "mean_pos_delta": float(np.mean(pos_delta_norms)),
        "mean_rot_delta": float(np.mean(rot_delta_norms)),
        "mean_direction_consistency": float(np.mean(cosine_sims)) if cosine_sims else 1.0,
        "num_gripper_transitions": len(gripper_transitions),
        "gripper_transitions": gripper_transitions,
        **chunk_boundary_stats,
    }


def compute_replan_consistency(action_log, replan_steps):
    chunk_entries = [e for e in action_log if e.get("is_chunk_start") and "full_action_chunk" in e]
    if len(chunk_entries) < 2:
        return {}
    replan_l2s = []
    for i in range(len(chunk_entries) - 1):
        old_chunk = np.array(chunk_entries[i]["full_action_chunk"])
        new_chunk = np.array(chunk_entries[i + 1]["full_action_chunk"])
        overlap_old = old_chunk[replan_steps:]
        overlap_new = new_chunk[: len(overlap_old)]
        n = min(len(overlap_old), len(overlap_new))
        if n > 0:
            replan_l2s.append(float(np.linalg.norm(overlap_old[:n] - overlap_new[:n])))
    if not replan_l2s:
        return {}
    return {
        "mean_replan_l2": float(np.mean(replan_l2s)),
        "max_replan_l2": float(np.max(replan_l2s)),
        "std_replan_l2": float(np.std(replan_l2s)),
        "num_replans": len(replan_l2s),
    }


def compute_action_entropy_kde(action_log, action_dim=7):
    actions = np.array([entry["action"] for entry in action_log])[:, :action_dim]
    H, D = actions.shape
    if H < D + 1:
        return {}
    try:
        kde = gaussian_kde(actions.T)
        log_densities = kde.logpdf(actions.T)
        return {
            "action_entropy_kde": -float(np.mean(log_densities)),
            "mean_log_density": float(np.mean(log_densities)),
            "std_log_density": float(np.std(log_densities)),
            "kde_bandwidth_factor": float(kde.factor),
            "num_samples": H,
            "action_dim": D,
        }
    except np.linalg.LinAlgError:
        return {}


# ── Config & helpers ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    # ── Checkpoint ───────────────────────────────────────────────────────
    ckpt: str = ""

    # ── Action prediction ────────────────────────────────────────────────
    replan_steps: int = 8
    """Number of actions to execute before replanning (open-loop steps)."""

    # ── LIBERO environment ───────────────────────────────────────────────
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 20
    seed: int = 0

    # ── Compute ──────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Output ───────────────────────────────────────────────────────────
    video_out_path: str = "data/libero/dp/videos"

    # ── Prompt perturbation ──────────────────────────────────────────────
    prompt_mode: str = "original"
    custom_prompt: str = ""


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _quat2axisangle(quat):
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


def _build_obs_dict(obs, prompt: str) -> dict:
    """Convert raw LIBERO env observation to DiffusionPolicy input dict.

    Images are flipped 180° (both axes) to correct the upside-down
    rendering from OffScreenRenderEnv — matching dp_scratch/eval.py.
    """
    return {
        "observation/image": obs["agentview_image"][::-1, ::-1].copy(),
        "observation/wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        "observation/state": np.concatenate([
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32),
        "prompt": prompt,
    }


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


# ── Main evaluation ───────────────────────────────────────────────────────────

def eval_libero(args: Args) -> None:
    from libero.libero import benchmark

    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load DP model ────────────────────────────────────────────────────
    logging.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiffusionPolicy(task_descs=ckpt["task_descs"]).to(device)
    model.set_norm_stats(**ckpt["stats"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    logging.info("Model loaded.")

    # ── LIBERO benchmark setup ───────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS.get(args.task_suite_name, 300)

    all_task_descriptions = []
    for i in range(num_tasks):
        try:
            t = task_suite.get_task(i)
            desc = getattr(t, "language", None) or str(t)
            all_task_descriptions.append(str(desc))
        except Exception:
            all_task_descriptions.append(f"task_{i}")
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))

    logging.info(f"Suite: {args.task_suite_name} ({num_tasks} tasks, max_steps={max_steps})")

    out_path = pathlib.Path(args.video_out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            prompt = perturb_prompt(str(task_description), args.prompt_mode, all_task_descriptions)
            if args.custom_prompt:
                prompt = args.custom_prompt
            logging.info(f"\nTask: {task_description} | Prompt: {prompt}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan = collections.deque()
            action_log = []
            replay_images = []
            t = 0
            done = False

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Record frame for video
                    replay_images.append(obs["agentview_image"][::-1, ::-1].copy())

                    is_new_chunk = not action_plan
                    if is_new_chunk:
                        obs_dict = _build_obs_dict(obs, prompt)
                        result = model.infer(obs_dict)
                        action_chunk = result["actions"]  # (horizon, 7)
                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    log_entry = {
                        "t": t,
                        "kind": "policy",
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "is_chunk_start": is_new_chunk,
                    }
                    if is_new_chunk:
                        log_entry["full_action_chunk"] = np.asarray(action_chunk, dtype=np.float32).tolist()
                    action_log.append(log_entry)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception at t={t}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # ── Save replay video ────────────────────────────────────────
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:60]
            if replay_images:
                imageio.mimwrite(
                    out_path / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # ── Compute metrics ──────────────────────────────────────────
            smoothness_summary = compute_smoothness_metrics(action_log)
            if smoothness_summary:
                logging.info(
                    f"Smoothness: mean_delta={smoothness_summary['mean_action_delta']:.4f}, "
                    f"max_delta={smoothness_summary['max_action_delta']:.4f}, "
                    f"mean_accel={smoothness_summary['mean_action_accel']:.4f}, "
                    f"gripper_transitions={smoothness_summary['num_gripper_transitions']}"
                )

            replan_summary = compute_replan_consistency(action_log, args.replan_steps)
            if replan_summary:
                logging.info(
                    f"Replan consistency: mean_L2={replan_summary['mean_replan_l2']:.4f}, "
                    f"max_L2={replan_summary['max_replan_l2']:.4f}"
                )

            entropy_summary = compute_action_entropy_kde(action_log)
            if entropy_summary:
                logging.info(f"Action entropy (KDE): {entropy_summary['action_entropy_kde']:.4f}")

            actions_path = out_path / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"
            with open(actions_path, "w") as f:
                json.dump(
                    {
                        "task_id": int(task_id),
                        "trial_id": int(episode_idx),
                        "seed": int(args.seed),
                        "task_description": str(task_description),
                        "prompt_mode": args.prompt_mode,
                        "prompt_used": prompt,
                        "success": bool(done),
                        "model": "dp",
                        "replan_steps": args.replan_steps,
                        "actions": action_log,
                        "smoothness": smoothness_summary,
                        "replan_consistency": replan_summary,
                        "action_entropy": entropy_summary,
                    },
                    f,
                    indent=2,
                    default=_json_default,
                )

            logging.info(f"Success: {done}")
            logging.info(
                f"Episodes: {total_episodes} | Successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Task {task_id} success rate: {task_successes / max(task_episodes, 1):.3f}")
        env.close()

    logging.info(
        f"Total success rate: {total_successes / max(total_episodes, 1):.3f} "
        f"({total_successes}/{total_episodes})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
