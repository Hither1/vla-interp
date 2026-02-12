import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from scipy.stats import gaussian_kde
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json

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
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
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
    track_mmd: bool = True  # Enable multi-sample MMD (policy uncertainty)
    mmd_num_samples: int = 8  # Number of diffusion samples for MMD estimation


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


def perturb_prompt(original: str, mode: str = "original", all_tasks: list = None) -> str:
    if mode == "original":
        return original

    elif mode == "empty":
        return ""

    elif mode == "shuffle":
        words = original.split()
        np.random.shuffle(words)

        result = " ".join(words)
        print("Modified prompt:", result)
        return result

    elif mode == "random":
        # Use a random (different) task's prompt
        # return np.random.choice(all_tasks)
        others = [t for t in all_tasks if t != original]
        result = np.random.choice(others) if others else original

        print("Modified prompt:", result)
        return result

    elif mode == "synonym":
        # Replace action verbs with synonyms
        result = original.lower()
        for word, synonyms in SYNONYM_MAP.items():
            if word in result:
                replacement = np.random.choice(synonyms)
                result = result.replace(word, replacement, 1)
                break  # Only replace one word to keep prompt mostly intact
        print("Modified prompt:", result)
        return result

    elif mode == "opposite":
        # Replace actions/directions with opposites
        result = original.lower()
        # Sort by length (descending) to match longer phrases first
        for phrase in sorted(OPPOSITE_MAP.keys(), key=len, reverse=True):
            if phrase in result:
                result = result.replace(phrase, OPPOSITE_MAP[phrase], 1)
                break  # Only replace one phrase
        print("Modified prompt:", result)
        return result

    return original


def _json_default(o):
    # numpy scalars -> python scalars
    if isinstance(o, np.generic):
        return o.item()
    # numpy arrays -> lists
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def compute_smoothness_metrics(action_log):
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
    actions = np.array([entry["action"] for entry in action_log])
    if len(actions) < 2:
        return {}

    # --- 1st derivative: action velocity ---
    deltas = np.diff(actions, axis=0)  # (T-1, 7)
    delta_norms = np.linalg.norm(deltas[:, :6], axis=1)  # exclude gripper for L2
    pos_delta_norms = np.linalg.norm(deltas[:, :3], axis=1)
    rot_delta_norms = np.linalg.norm(deltas[:, 3:6], axis=1)
    gripper_deltas = np.abs(deltas[:, 6])

    # --- 2nd derivative: action acceleration ---
    accels = np.diff(deltas, axis=0)  # (T-2, 7)
    accel_norms = np.linalg.norm(accels[:, :6], axis=1)

    # --- Direction consistency (cosine similarity between consecutive deltas) ---
    cosine_sims = []
    for i in range(len(deltas) - 1):
        d1, d2 = deltas[i, :6], deltas[i + 1, :6]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 > 1e-8 and n2 > 1e-8:
            cosine_sims.append(float(np.dot(d1, d2) / (n1 * n2)))
        else:
            cosine_sims.append(1.0)  # no motion = perfectly consistent

    # --- Gripper state transitions (task-phase boundaries) ---
    gripper_actions = actions[:, 6]
    gripper_signs = np.sign(gripper_actions)
    gripper_transitions = []
    for i in range(1, len(gripper_signs)):
        if gripper_signs[i] != gripper_signs[i - 1] and gripper_signs[i] != 0:
            gripper_transitions.append({
                "t": action_log[i]["t"],
                "step_idx": i,
                "from": "open" if gripper_signs[i - 1] > 0 else "close",
                "to": "open" if gripper_signs[i] > 0 else "close",
            })

    # --- Chunk-boundary discontinuity analysis ---
    chunk_start_set = {
        i for i, entry in enumerate(action_log) if entry.get("is_chunk_start", False)
    }
    boundary_deltas = []
    within_deltas = []
    for i in range(len(delta_norms)):
        # delta[i] = action_log[i+1] - action_log[i]
        # If action_log[i+1] is a chunk start, this delta crosses a re-plan boundary.
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
            "boundary_ratio": (
                mean_boundary / mean_within if mean_within > 1e-8 else float("inf")
            ),
            "num_chunk_boundaries": len(boundary_deltas),
        }

    # --- Annotate per-step metrics into action_log entries ---
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

    # --- Episode-level summary ---
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


def compute_replan_consistency(action_log, replan_steps):
    """Measure how much the policy's predicted future changes between re-plans.

    Consecutive action chunks overlap: chunk_i predicts [t..t+H-1], chunk_{i+1}
    predicts [t+R..t+R+H-1], where R = replan_steps, H = action_horizon.  The
    overlap is chunk_i[R:] vs chunk_{i+1}[:H-R].  A large L2 in this overlap
    means the policy substantially revised its plan.

    Requires ``full_action_chunk`` to have been stored in action_log entries at
    chunk starts.
    """
    chunk_entries = [
        e for e in action_log
        if e.get("is_chunk_start") and "full_action_chunk" in e
    ]
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


def compute_action_entropy_kde(action_logs, action_dim=7):
    """Estimate entropy of the action distribution across multiple episodes via Gaussian KDE.

    Pools executed actions from all episodes for a given task/scenario and fits
    a single Gaussian KDE.  Differential entropy is estimated as:
        H_hat = -1/N * sum_i log p(x_i)

    Args:
        action_logs: List of action logs (one per episode). Each action log is
            a list of dicts with an "action" key.
        action_dim: Number of action dimensions to use (default 7, the real
            LIBERO action dims before padding).

    Returns:
        Dict with entropy estimate and diagnostics, or {} if estimation fails.
    """
    all_actions = []
    for action_log in action_logs:
        actions = np.array([entry["action"] for entry in action_log])  # (H_i, D_full)
        all_actions.append(actions[:, :action_dim])
    actions = np.concatenate(all_actions, axis=0)  # (N_total, D)
    N, D = actions.shape

    if N < D + 1:
        # Not enough samples for reliable KDE
        return {}

    try:
        # gaussian_kde expects shape (D, N) — each column is one observation
        kde = gaussian_kde(actions.T)  # bandwidth via Scott's rule
        log_densities = kde.logpdf(actions.T)  # (N,)
        entropy = -float(np.mean(log_densities))

        return {
            "action_entropy_kde": entropy,
            "mean_log_density": float(np.mean(log_densities)),
            "std_log_density": float(np.std(log_densities)),
            "kde_bandwidth_factor": float(kde.factor),
            "num_samples": N,
            "num_episodes": len(action_logs),
            "action_dim": D,
        }
    except np.linalg.LinAlgError:
        # Degenerate covariance (e.g., all actions identical)
        return {}


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    logging.info(f"Task suite: {args.task_suite_name}")

    # -------------------------------------------------------------------------
    # Prepare all_task_descriptions (used by prompt_mode="random")
    # -------------------------------------------------------------------------
    all_task_descriptions = []
    for i in range(num_tasks_in_suite):
        try:
            t = task_suite.get_task(i)
            # Usually the natural language instruction is stored here
            desc = getattr(t, "language", None)
            if desc is None:
                desc = getattr(t, "task_description", None)
            if desc is None:
                desc = str(t)
            all_task_descriptions.append(str(desc))
        except Exception as e:
            logging.warning(f"Failed to get task description for task {i}: {e}")
            all_task_descriptions.append(f"task_{i}")

    # Optional: ensure uniqueness to avoid choosing the same string a lot
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))


    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_episode_data = []  # deferred JSON data for all episodes in this task
        task_action_logs = []   # action logs across episodes for task-level entropy
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
             
            # Reset environment
            env.reset()
            action_plan = collections.deque()
            action_log = []

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            chunk_mmd_score = None
            chunk_action_sample_std = None
            chunk_action_std_per_ts = None
            is_new_chunk = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    is_new_chunk = not action_plan
                    if is_new_chunk:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            # "prompt": str(task_description),
                            "prompt": perturb_prompt(str(task_description), args.prompt_mode, all_task_descriptions),
                            "episode_id": f"task{task_id}_ep{episode_idx}",
                        }
                        if args.track_mmd:
                            element["compute_mmd"] = True
                            element["mmd_num_samples"] = args.mmd_num_samples

                        # Query model to get action
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        chunk_mmd_score = result.get("mmd_score")
                        chunk_action_sample_std = result.get("action_sample_std")
                        chunk_action_std_per_ts = result.get("action_std_per_timestep")
                        if chunk_mmd_score is not None:
                            logging.info(f"  MMD: {chunk_mmd_score:.6f}, sample_std: {chunk_action_sample_std:.4f}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    log_entry = {
                        "t": t,
                        "kind": "policy",
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "reward": reward,
                        "done": done,
                        "is_chunk_start": is_new_chunk,
                    }
                    # Only attach per-chunk data on the first step of each new chunk
                    if is_new_chunk:
                        # Save full predicted chunk for replan consistency analysis
                        log_entry["full_action_chunk"] = np.asarray(
                            action_chunk, dtype=np.float32
                        ).tolist()
                        if chunk_mmd_score is not None:
                            log_entry["mmd_score"] = chunk_mmd_score
                            log_entry["action_sample_std"] = chunk_action_sample_std
                            log_entry["action_std_per_timestep"] = chunk_action_std_per_ts
                    action_log.append(log_entry)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            actions_path = (
                pathlib.Path(args.video_out_path)
                / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"
            )
            # Compute smoothness & boundary metrics (also annotates action_log in-place)
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
            episode_mmds = [
                entry["mmd_score"]
                for entry in action_log
                if "mmd_score" in entry
            ]
            mmd_summary = {}
            if episode_mmds:
                mmd_summary = {
                    "mean_mmd": float(np.mean(episode_mmds)),
                    "max_mmd": float(np.max(episode_mmds)),
                    "std_mmd": float(np.std(episode_mmds)),
                }
                episode_stds = [
                    entry["action_sample_std"]
                    for entry in action_log
                    if "action_sample_std" in entry
                ]
                if episode_stds:
                    mmd_summary["mean_action_sample_std"] = float(np.mean(episode_stds))
                logging.info(
                    f"Episode MMD: mean={mmd_summary['mean_mmd']:.6f}, "
                    f"max={mmd_summary['max_mmd']:.6f}"
                )

            # Replan consistency: how much predicted future changes between chunks
            replan_summary = compute_replan_consistency(action_log, args.replan_steps)
            if replan_summary:
                logging.info(
                    f"Replan consistency: mean_L2={replan_summary['mean_replan_l2']:.4f}, "
                    f"max_L2={replan_summary['max_replan_l2']:.4f}"
                )

            # Collect action log for task-level entropy (computed after all episodes)
            task_action_logs.append(action_log)

            # Defer JSON writing until task-level entropy is computed
            task_episode_data.append({
                "actions_path": actions_path,
                "json_data": {
                    "task_id": int(task_id),
                    "trial_id": int(episode_idx),
                    "seed": int(args.seed),
                    "task_description": str(task_description),
                    "success": bool(done),
                    "actions": action_log,
                    "smoothness": smoothness_summary,
                    "mmd": mmd_summary,
                    "replan_consistency": replan_summary,
                },
            })

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Task-level action entropy across all episodes
        task_entropy = compute_action_entropy_kde(task_action_logs)
        if task_entropy:
            logging.info(
                f"Task {task_id} action entropy (KDE, {task_entropy['num_episodes']} episodes): "
                f"{task_entropy['action_entropy_kde']:.4f} "
                f"(bw_factor={task_entropy['kde_bandwidth_factor']:.4f}, "
                f"N={task_entropy['num_samples']}, D={task_entropy['action_dim']})"
            )

        # Write deferred per-episode JSONs (now including task-level entropy)
        for ep_data in task_episode_data:
            ep_data["json_data"]["action_entropy"] = task_entropy
            with open(ep_data["actions_path"], "w") as f:
                json.dump(ep_data["json_data"], f, indent=2, default=_json_default)

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
