"""Evaluate Cosmos Policy directly on LIBERO simulation benchmarks.

Runs the Cosmos diffusion VLA locally (no WebSocket server) on a Linux cluster.

Usage:
    python examples/libero/cosmos_eval.py \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --task_suite_name libero_10 \
        --num_trials_per_task 20 \
        --seed 195

    # Minimal test run:
    python examples/libero/cosmos_eval.py \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --task_suite_name libero_10 \
        --num_trials_per_task 1
"""

import collections
import dataclasses
import json
import logging
import pathlib
import sys

import imageio
import numpy as np
from scipy.stats import gaussian_kde
import tqdm
import tyro

# Add cosmos-policy to path so we can import cosmos_policy
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "cosmos-policy")
if _COSMOS_POLICY_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_POLICY_DIR)

from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Reuse metric helpers from sibling Pi0 evaluation script
from main import (
    compute_smoothness_metrics,
    compute_replan_consistency,
    compute_action_entropy_kde,
    perturb_prompt,
    SYNONYM_MAP,
    OPPOSITE_MAP,
)


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


@dataclasses.dataclass
class Args:
    # ── Cosmos model paths ───────────────────────────────────────────────
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    config_name: str = "cosmos_predict2_2b_480p_libero__inference_only"
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"

    # ── Action prediction ────────────────────────────────────────────────
    chunk_size: int = 16
    num_open_loop_steps: int = 16  # How many actions to execute before replanning
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1

    # ── Input preprocessing ──────────────────────────────────────────────
    use_wrist_image: bool = True
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    flip_images: bool = True  # LIBERO images render upside-down

    # ── LIBERO environment ───────────────────────────────────────────────
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 20
    seed: int = 7

    # ── Output ───────────────────────────────────────────────────────────
    video_out_path: str = "data/libero/cosmos/videos"

    # ── Prompt perturbation ──────────────────────────────────────────────
    prompt_mode: str = "original"  # original, empty, shuffle, random, synonym, opposite
    custom_prompt: str = ""


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _build_cosmos_cfg(args: Args) -> PolicyEvalConfig:
    """Build a PolicyEvalConfig from our simplified Args."""
    return PolicyEvalConfig(
        config=args.config_name,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        use_wrist_image=args.use_wrist_image,
        use_proprio=args.use_proprio,
        normalize_proprio=args.normalize_proprio,
        unnormalize_actions=args.unnormalize_actions,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        trained_with_image_aug=args.trained_with_image_aug,
        use_jpeg_compression=args.use_jpeg_compression,
        flip_images=args.flip_images,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value,
        seed=args.seed,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
    )


def _get_libero_env(task, resolution, seed):
    """Initialize a LIBERO environment and return (env, task_description)."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs, flip_images: bool = True):
    """Prepare an observation dict for Cosmos get_action().

    Cosmos expects:
        primary_image: (H, W, 3) uint8 — third-person camera
        wrist_image:   (H, W, 3) uint8 — wrist camera
        proprio:       (9,) float — [gripper_qpos(2), eef_pos(3), eef_quat(4)]
    """
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    return {
        "primary_image": np.ascontiguousarray(img),
        "wrist_image": np.ascontiguousarray(wrist_img),
        "proprio": np.concatenate((
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
        )),
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    # ── Build Cosmos config and load model ───────────────────────────────
    cfg = _build_cosmos_cfg(args)
    logging.info("Loading Cosmos model...")
    model, cosmos_config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    logging.info("Model loaded.")

    # ── LIBERO benchmark setup ───────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS[args.task_suite_name]

    # Collect all task descriptions (for prompt_mode="random")
    all_task_descriptions = []
    for i in range(num_tasks):
        try:
            t = task_suite.get_task(i)
            desc = getattr(t, "language", None) or getattr(t, "task_description", None) or str(t)
            all_task_descriptions.append(str(desc))
        except Exception:
            all_task_descriptions.append(f"task_{i}")
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))

    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks, max_steps={max_steps})")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ──────────────────────────────────────────────────
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            prompt = perturb_prompt(str(task_description), args.prompt_mode, all_task_descriptions)
            logging.info(f"\nTask: {task_description} | Prompt: {prompt}")

            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan = collections.deque()
            action_log = []
            replay_images = []
            t = 0
            done = False
            is_new_chunk = False

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # Wait for objects to settle
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Prepare observation for Cosmos
                    observation = _prepare_observation(obs, flip_images=args.flip_images)
                    replay_images.append(observation["primary_image"])

                    is_new_chunk = not action_plan
                    if is_new_chunk:
                        # Query Cosmos model for a new action chunk
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            prompt,
                            seed=args.seed,
                            num_denoising_steps_action=args.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True,
                        )
                        action_chunk = action_return_dict["actions"]
                        assert len(action_chunk) >= args.num_open_loop_steps, (
                            f"Expected >= {args.num_open_loop_steps} actions, got {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[: args.num_open_loop_steps])

                    action = action_plan.popleft()

                    # Step environment
                    obs, reward, done, info = env.step(action.tolist())

                    log_entry = {
                        "t": t,
                        "kind": "policy",
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "reward": reward,
                        "done": done,
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
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # ── Save replay video ────────────────────────────────────────
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # ── Compute & save metrics ───────────────────────────────────
            smoothness_summary = compute_smoothness_metrics(action_log)
            if smoothness_summary:
                logging.info(
                    f"Smoothness: mean_delta={smoothness_summary['mean_action_delta']:.4f}, "
                    f"max_delta={smoothness_summary['max_action_delta']:.4f}, "
                    f"mean_accel={smoothness_summary['mean_action_accel']:.4f}, "
                    f"gripper_transitions={smoothness_summary['num_gripper_transitions']}"
                )

            replan_summary = compute_replan_consistency(action_log, args.num_open_loop_steps)
            if replan_summary:
                logging.info(
                    f"Replan consistency: mean_L2={replan_summary['mean_replan_l2']:.4f}, "
                    f"max_L2={replan_summary['max_replan_l2']:.4f}"
                )

            entropy_summary = compute_action_entropy_kde(action_log)
            if entropy_summary:
                logging.info(
                    f"Action entropy (KDE): {entropy_summary['action_entropy_kde']:.4f}"
                )

            actions_path = (
                pathlib.Path(args.video_out_path)
                / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"
            )
            with open(actions_path, "w") as f:
                json.dump(
                    {
                        "task_id": int(task_id),
                        "trial_id": int(episode_idx),
                        "seed": int(args.seed),
                        "task_description": str(task_description),
                        "prompt": prompt,
                        "success": bool(done),
                        "model": "cosmos",
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
            logging.info(f"Episodes: {total_episodes} | Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Task {task_id} success rate: {task_successes / max(task_episodes, 1):.3f}")

    logging.info(f"Total success rate: {total_successes / max(total_episodes, 1):.3f} ({total_successes}/{total_episodes})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
