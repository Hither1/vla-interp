#!/usr/bin/env python3
"""
LIBERO evaluation script with:
  - prompt perturbation modes (original/empty/shuffle/random/synonym/opposite/custom)
  - per-episode smoothness + boundary metrics
  - replan consistency metric
  - optional MMD tracking from the policy server
  - action entropy computed once per (task_id, prompt_used) group by pooling
         executed actions across ALL trajectories in that group.

NEW (this version):
  - For each (task_id, prompt_used) group, compute KDE action entropy for:
        (1) success trajectories only
        (2) failure trajectories only
        (3) all trajectories
    and attach these to every per-episode JSON under "action_entropy_group".
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.stats import gaussian_kde
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_90"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"data/libero/{task_suite_name.split('_')[1]}/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # Prompt perturbation options
    prompt_mode: str = "original"  # original, empty, shuffle, random, synonym, opposite, custom
    custom_prompt: str = ""  # Used when prompt_mode="custom"

    # MMD (Maximum Mean Discrepancy) tracking
    track_mmd: bool = False  # Enable multi-sample MMD (policy uncertainty)
    mmd_num_samples: int = 8  # Number of diffusion samples for MMD estimation

    # Action entropy options
    entropy_action_dim: int = 7  # LIBERO action dims (x,y,z,rx,ry,rz,gripper)

    # ── Visual perturbation ──────────────────────────────────────────────
    visual_perturb_mode: str = "none"
    """Image-level perturbation mode: none | rotate | translate | rotate_translate."""
    rotation_degrees: float = 30.0
    """Rotation angle in degrees (CCW). Used with rotate / rotate_translate modes."""
    translate_x_frac: float = 0.2
    """Horizontal shift as fraction of image width (positive = right).
    Used with translate / rotate_translate modes."""
    translate_y_frac: float = 0.0
    """Vertical shift as fraction of image height (positive = down).
    Used with translate / rotate_translate modes."""

    # ── Policy perturbation ───────────────────────────────────────────────────────────────────
    policy_perturb_mode: str = "none"
    """Policy-level perturbation mode: none | random_action | object_shift."""
    random_action_prob: float = 0.25
    """Probability of replacing policy action with random noise per step."""
    random_action_scale: float = 1.0
    """Scale of uniform random action noise: Uniform(-scale, scale)."""
    object_shift_x_std: float = 0.05
    """Std (metres) of Gaussian object shift along x-axis at episode start."""
    object_shift_y_std: float = 0.0
    """Std (metres) of Gaussian object shift along y-axis at episode start."""


# Synonym mappings for common LIBERO action verbs
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

# Opposite action mappings
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


def perturb_prompt(original: str, mode: str = "original", all_tasks: list[str] | None = None, custom: str = "") -> str:
    """Return a perturbed prompt string."""
    if mode == "original":
        return original
    if mode == "custom":
        return custom

    if mode == "empty":
        return ""

    if mode == "shuffle":
        words = original.split()
        np.random.shuffle(words)
        result = " ".join(words)
        logging.info(f"Modified prompt (shuffle): {result}")
        return result

    if mode == "random":
        if not all_tasks:
            return original
        others = [t for t in all_tasks if t != original]
        result = str(np.random.choice(others)) if others else original
        logging.info(f"Modified prompt (random): {result}")
        return result

    if mode == "synonym":
        result = original.lower()
        for word, synonyms in SYNONYM_MAP.items():
            if word in result:
                replacement = str(np.random.choice(synonyms))
                result = result.replace(word, replacement, 1)
                break
        logging.info(f"Modified prompt (synonym): {result}")
        return result

    if mode == "opposite":
        result = original.lower()
        for phrase in sorted(OPPOSITE_MAP.keys(), key=len, reverse=True):
            if phrase in result:
                result = result.replace(phrase, OPPOSITE_MAP[phrase], 1)
                break
        logging.info(f"Modified prompt (opposite): {result}")
        return result

    return original


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def compute_smoothness_metrics(action_log: list[dict]):
    """Compute action smoothness and boundary metrics from an episode's action log.

    Annotates each action_log entry in-place with per-step smoothness fields, and
    returns an episode-level summary dict.

    Per-step fields added to action_log entries:
        action_delta_norm   - L2 norm of action change (position+rotation, excludes gripper)
        pos_delta_norm      - L2 norm of position (xyz) change
        rot_delta_norm      - L2 norm of rotation (axis-angle) change
        gripper_delta       - absolute change in gripper action
        action_accel_norm   - L2 norm of acceleration (2nd derivative)
        direction_consistency - cosine similarity between consecutive action deltas

    LIBERO actions are [x, y, z, rx, ry, rz, gripper] (7D).
    """
    actions = np.array([entry["action"] for entry in action_log], dtype=np.float32)
    if len(actions) < 2:
        return {}

    deltas = np.diff(actions, axis=0)  # (T-1, 7)
    delta_norms = np.linalg.norm(deltas[:, :6], axis=1)
    pos_delta_norms = np.linalg.norm(deltas[:, :3], axis=1)
    rot_delta_norms = np.linalg.norm(deltas[:, 3:6], axis=1)
    gripper_deltas = np.abs(deltas[:, 6])

    accels = np.diff(deltas, axis=0)  # (T-2, 7)
    accel_norms = np.linalg.norm(accels[:, :6], axis=1)

    cosine_sims = []
    for i in range(len(deltas) - 1):
        d1, d2 = deltas[i, :6], deltas[i + 1, :6]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 > 1e-8 and n2 > 1e-8:
            cosine_sims.append(float(np.dot(d1, d2) / (n1 * n2)))
        else:
            cosine_sims.append(1.0)

    gripper_actions = actions[:, 6]
    gripper_signs = np.sign(gripper_actions)
    gripper_transitions = []
    for i in range(1, len(gripper_signs)):
        if gripper_signs[i] != gripper_signs[i - 1] and gripper_signs[i] != 0:
            gripper_transitions.append(
                {
                    "t": action_log[i]["t"],
                    "step_idx": i,
                    "from": "open" if gripper_signs[i - 1] > 0 else "close",
                    "to": "open" if gripper_signs[i] > 0 else "close",
                }
            )

    chunk_start_set = {i for i, entry in enumerate(action_log) if entry.get("is_chunk_start", False)}
    boundary_deltas = []
    within_deltas = []
    for i in range(len(delta_norms)):
        if (i + 1) in chunk_start_set:
            boundary_deltas.append(delta_norms[i])
        else:
            within_deltas.append(delta_norms[i])

    chunk_boundary_stats = {}
    if boundary_deltas:
        mean_within = float(np.mean(within_deltas)) if within_deltas else 0.0
        mean_boundary = float(np.mean(boundary_deltas))
        chunk_boundary_stats = {
            "mean_boundary_delta": mean_boundary,
            "mean_within_chunk_delta": mean_within,
            "boundary_ratio": (mean_boundary / mean_within if mean_within > 1e-8 else float("inf")),
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

    summary = {
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
    return summary


def compute_replan_consistency(action_log: list[dict], replan_steps: int):
    """Measure how much the policy's predicted future changes between re-plans.

    Overlap: old_chunk[replan_steps:] vs new_chunk[:H-replan_steps]
    """
    chunk_entries = [e for e in action_log if e.get("is_chunk_start") and "full_action_chunk" in e]
    if len(chunk_entries) < 2:
        return {}

    replan_l2s = []
    for i in range(len(chunk_entries) - 1):
        old_chunk = np.array(chunk_entries[i]["full_action_chunk"], dtype=np.float32)
        new_chunk = np.array(chunk_entries[i + 1]["full_action_chunk"], dtype=np.float32)
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


def compute_action_entropy_kde(action_logs: list[list[dict]], action_dim: int = 7):
    """Estimate differential entropy of pooled executed actions across many episodes using Gaussian KDE.

    H_hat = - (1/N) * sum_i log p(x_i)
    """
    all_actions = []
    for action_log in action_logs:
        if not action_log:
            continue
        actions = np.array([entry["action"] for entry in action_log], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < action_dim:
            continue
        all_actions.append(actions[:, :action_dim])

    if not all_actions:
        return {}

    actions = np.concatenate(all_actions, axis=0)  # (N_total, D)
    N, D = actions.shape
    if N < D + 1:
        return {}

    try:
        kde = gaussian_kde(actions.T)  # expects (D, N)
        log_densities = kde.logpdf(actions.T)
        entropy = -float(np.mean(log_densities))
        return {
            "action_entropy_kde": entropy,
            "mean_log_density": float(np.mean(log_densities)),
            "std_log_density": float(np.std(log_densities)),
            "kde_bandwidth_factor": float(kde.factor),
            "num_samples": int(N),
            "num_episodes": int(len(action_logs)),
            "action_dim": int(D),
        }
    except np.linalg.LinAlgError:
        return {}


def compute_entropy_triplet(action_logs: list[list[dict]], successes: list[bool], action_dim: int = 7) -> dict:
    """Compute KDE entropy for all / success-only / failure-only trajectories within a group."""
    if len(action_logs) != len(successes):
        raise ValueError("action_logs and successes must have the same length")

    all_logs = action_logs
    succ_logs = [log for log, ok in zip(action_logs, successes) if ok]
    fail_logs = [log for log, ok in zip(action_logs, successes) if not ok]

    return {
        "all": compute_action_entropy_kde(all_logs, action_dim=action_dim),
        "success": compute_action_entropy_kde(succ_logs, action_dim=action_dim),
        "failure": compute_action_entropy_kde(fail_logs, action_dim=action_dim),
        "counts": {
            "num_episodes_all": int(len(all_logs)),
            "num_episodes_success": int(len(succ_logs)),
            "num_episodes_failure": int(len(fail_logs)),
        },
    }


def _get_libero_env(task, resolution: int, seed: int):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # affects object positions even when using fixed init state
    return env, task_description


def _quat2axisangle(quat: np.ndarray):
    """robosuite quat -> axis-angle."""
    quat = np.array(quat, dtype=np.float32)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    if task_suite_name in ("libero_90_obj", "libero_90_spa", "libero_90_act", "libero_90_com"):
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    logging.info(f"Visual perturbation: {vis_cfg.as_dict()}")

    policy_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    policy_rng = np.random.default_rng(args.seed + 9999)
    logging.info(f"Policy perturbation: {policy_cfg.as_dict()}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (n_tasks={num_tasks_in_suite})")

    # Prepare all_task_descriptions (used by prompt_mode="random")
    all_task_descriptions = []
    for i in range(num_tasks_in_suite):
        try:
            t = task_suite.get_task(i)
            desc = getattr(t, "language", None) or getattr(t, "task_description", None) or str(t)
            all_task_descriptions.append(str(desc))
        except Exception as e:
            logging.warning(f"Failed to get task description for task {i}: {e}")
            all_task_descriptions.append(f"task_{i}")
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = _get_max_steps(args.task_suite_name)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        # Group episodes by prompt used, but keep logs + success flags:
        # (task_id, prompt_used) -> {"logs": [...], "successes": [...]}
        action_groups_by_prompt: dict[str, dict] = collections.defaultdict(lambda: {"logs": [], "successes": []})

        # deferred JSON writing so we can attach group entropy later
        deferred_episode_jsons: list[dict] = []

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id} eps", leave=False):
            logging.info(f"\nTask {task_id}: {task_description}")

            env.reset()
            action_plan = collections.deque()
            action_log: list[dict] = []

            obs = env.set_init_state(initial_states[episode_idx])
            episode_object_shifts = apply_object_shift(env, policy_cfg, policy_rng)

            # Freeze the prompt ONCE per episode (so group entropy is well-defined).
            if args.prompt_mode == "custom":
                episode_prompt = perturb_prompt(
                    str(task_description), "custom", all_task_descriptions, custom=args.custom_prompt
                )
            else:
                episode_prompt = perturb_prompt(
                    str(task_description), args.prompt_mode, all_task_descriptions, custom=args.custom_prompt
                )

            t = 0
            replay_images = []
            chunk_mmd_score = None
            chunk_action_sample_std = None
            chunk_action_std_per_ts = None
            is_new_chunk = False
            done = False

            logging.info(
                f"Starting episode {task_episodes + 1}/{args.num_trials_per_task}... prompt_mode={args.prompt_mode}"
            )

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Apply visual perturbation (on full-res image before resize)
                    img = perturb_image(img, vis_cfg)
                    wrist_img = perturb_image(wrist_img, vis_cfg)

                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    is_new_chunk = not action_plan
                    if is_new_chunk:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                ),
                                axis=0,
                            ),
                            "prompt": episode_prompt,  # frozen per episode
                            "episode_id": f"task{task_id}_ep{episode_idx}",
                        }
                        if args.track_mmd:
                            element["compute_mmd"] = True
                            element["mmd_num_samples"] = args.mmd_num_samples

                        result = client.infer(element)
                        action_chunk = result["actions"]

                        chunk_mmd_score = result.get("mmd_score")
                        chunk_action_sample_std = result.get("action_sample_std")
                        chunk_action_std_per_ts = result.get("action_std_per_timestep")

                        if chunk_mmd_score is not None:
                            logging.info(f"  MMD: {chunk_mmd_score:.6f}, sample_std: {chunk_action_sample_std:.4f}")

                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Need replan_steps={args.replan_steps} but policy predicted only {len(action_chunk)} steps."
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    policy_action = action_plan.popleft()
                    action, action_was_perturbed = maybe_perturb_action(policy_action, policy_cfg, policy_rng)
                    obs, reward, done, info = env.step(action.tolist())

                    log_entry = {
                        "t": int(t),
                        "kind": "random" if action_was_perturbed else "policy",
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "action_perturbed": bool(action_was_perturbed),
                        "reward": float(reward),
                        "done": bool(done),
                        "is_chunk_start": bool(is_new_chunk),
                    }
                    if action_was_perturbed:
                        log_entry["policy_action"] = np.asarray(policy_action, dtype=np.float32).tolist()

                    if is_new_chunk:
                        log_entry["full_action_chunk"] = np.asarray(action_chunk, dtype=np.float32).tolist()
                        if chunk_mmd_score is not None:
                            log_entry["mmd_score"] = float(chunk_mmd_score)
                            if chunk_action_sample_std is not None:
                                log_entry["action_sample_std"] = float(chunk_action_sample_std)
                            if chunk_action_std_per_ts is not None:
                                log_entry["action_std_per_timestep"] = chunk_action_std_per_ts

                    action_log.append(log_entry)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception in episode {episode_idx}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = str(task_description).replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            actions_path = pathlib.Path(args.video_out_path) / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"

            # Smoothness & boundary metrics (annotates log entries)
            smoothness_summary = compute_smoothness_metrics(action_log)
            if smoothness_summary:
                logging.info(
                    f"Smoothness: mean_delta={smoothness_summary['mean_action_delta']:.4f}, "
                    f"max_delta={smoothness_summary['max_action_delta']:.4f}, "
                    f"mean_accel={smoothness_summary['mean_action_accel']:.4f}, "
                    f"gripper_transitions={smoothness_summary['num_gripper_transitions']}"
                )
                if "boundary_ratio" in smoothness_summary:
                    logging.info(
                        f"Chunk boundaries: ratio={smoothness_summary['boundary_ratio']:.2f} "
                        f"(boundary={smoothness_summary['mean_boundary_delta']:.4f} vs "
                        f"within={smoothness_summary['mean_within_chunk_delta']:.4f})"
                    )

            # Per-episode MMD summary
            episode_mmds = [entry["mmd_score"] for entry in action_log if "mmd_score" in entry]
            mmd_summary = {}
            if episode_mmds:
                mmd_summary = {
                    "mean_mmd": float(np.mean(episode_mmds)),
                    "max_mmd": float(np.max(episode_mmds)),
                    "std_mmd": float(np.std(episode_mmds)),
                }
                episode_stds = [entry.get("action_sample_std") for entry in action_log if "action_sample_std" in entry]
                episode_stds = [x for x in episode_stds if x is not None]
                if episode_stds:
                    mmd_summary["mean_action_sample_std"] = float(np.mean(episode_stds))
                logging.info(f"Episode MMD: mean={mmd_summary['mean_mmd']:.6f}, max={mmd_summary['max_mmd']:.6f}")

            # Replan consistency
            replan_summary = compute_replan_consistency(action_log, args.replan_steps)
            if replan_summary:
                logging.info(
                    f"Replan consistency: mean_L2={replan_summary['mean_replan_l2']:.4f}, "
                    f"max_L2={replan_summary['max_replan_l2']:.4f}"
                )

            # Collect logs + success labels by prompt for group entropy
            action_groups_by_prompt[episode_prompt]["logs"].append(action_log)
            action_groups_by_prompt[episode_prompt]["successes"].append(bool(done))

            # Defer JSON until group entropy computed
            deferred_episode_jsons.append(
                {
                    "actions_path": actions_path,
                    "prompt_used": episode_prompt,
                    "json_data": {
                        "task_id": int(task_id),
                        "trial_id": int(episode_idx),
                        "seed": int(args.seed),
                        "task_description": str(task_description),
                        "prompt_mode": str(args.prompt_mode),
                        "prompt_used": str(episode_prompt),
                        "custom_prompt": str(args.custom_prompt) if args.prompt_mode == "custom" else "",
                        "success": bool(done),
                        "visual_perturbation": vis_cfg.as_dict(),
                        "policy_perturbation": policy_cfg.as_dict(),
                        "object_shifts": episode_object_shifts,
                        "actions": action_log,
                        "smoothness": smoothness_summary,
                        "mmd": mmd_summary,
                        "replan_consistency": replan_summary,
                        "video_path": str(video_path),
                    },
                }
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({(total_successes / max(1,total_episodes))*100:.1f}%)")

        # Compute entropy ONCE per prompt group for this task_id (all/success/failure)
        entropy_by_prompt: dict[str, dict] = {}
        for prompt_str, group in action_groups_by_prompt.items():
            logs = group["logs"]
            successes = group["successes"]
            ent_triplet = compute_entropy_triplet(logs, successes, action_dim=args.entropy_action_dim)
            entropy_by_prompt[prompt_str] = ent_triplet

        def _fmt_ent(ent: dict) -> str:
            if not ent:
                return "H=NA"
            return (
                f"H={ent.get('action_entropy_kde', float('nan')):.4f} | "
                f"eps={ent.get('num_episodes', 0)} | "
                f"N={ent.get('num_samples', 0)} | "
                f"bw={ent.get('kde_bandwidth_factor', float('nan')):.4f}"
            )

        # Log group entropies
        for prompt_str, ent3 in entropy_by_prompt.items():
            c = ent3.get("counts", {})
            logging.info(
                f"Task {task_id} group entropy (task_id+prompt_used): "
                f"[all: {_fmt_ent(ent3.get('all', {}))}] "
                f"[succ: {_fmt_ent(ent3.get('success', {}))}] "
                f"[fail: {_fmt_ent(ent3.get('failure', {}))}] "
                f"| counts={c} | prompt='{prompt_str[:120]}'"
            )

        # Write deferred per-episode JSONs (attach the correct group entropy triplet)
        for ep in deferred_episode_jsons:
            p = ep["prompt_used"]
            ep["json_data"]["action_entropy_group"] = entropy_by_prompt.get(p, {})
            with open(ep["actions_path"], "w") as f:
                json.dump(ep["json_data"], f, indent=2, default=_json_default)

        logging.info(f"Task {task_id} success rate: {float(task_successes) / float(max(1, task_episodes)):.3f}")
        logging.info(f"Total success rate so far: {float(total_successes) / float(max(1, total_episodes)):.3f}")

    logging.info(f"Total success rate: {float(total_successes) / float(max(1, total_episodes)):.3f}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)