"""Evaluate Cosmos Policy directly on LIBERO simulation benchmarks.

Runs the Cosmos diffusion VLA locally (no WebSocket server) on a Linux cluster.
Requires a conda env with Cosmos dependencies (torch, megatron-core,
transformer-engine, etc.).

Usage:
    conda activate <cosmos-env>

    # Full evaluation
    python examples/libero/cosmos_eval.py \
        --task-suite-name libero_10 \
        --num-trials-per-task 20 \
        --seed 195

    # Quick smoke test (1 trial)
    python examples/libero/cosmos_eval.py \
        --task-suite-name libero_10 \
        --num-trials-per-task 1
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys

import imageio
import numpy as np
from scipy.stats import gaussian_kde
import tqdm
import tyro
from unittest.mock import MagicMock

# Add cosmos-policy to path so we can import cosmos_policy
_COSMOS_POLICY_DIR = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "cosmos-policy")
if _COSMOS_POLICY_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_POLICY_DIR)

# Mock broken transformer_engine if needed — it's only pulled in transitively
# by megatron.core for training utilities, not used for Cosmos inference.
try:
    import transformer_engine.pytorch  # noqa: F401
except Exception:
    import importlib.abc
    import importlib.machinery
    import types

    class _TEMockModule(types.ModuleType):
        """A fake module/package that satisfies Python's import machinery and
        returns MagicMock for every attribute (classes, functions, constants)."""

        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []       # make it a package
            self.__file__ = f"<te-mock {name}>"
            self.__loader__ = None
            self.__spec__ = None
            self.__package__ = name

        def __getattr__(self, attr):
            # Return a MagicMock for any attribute not already set.
            # Cache it so the same object is returned on repeat access
            # (important for `te.pytorch` == `sys.modules[...]`).
            val = MagicMock()
            object.__setattr__(self, attr, val)
            return val

    class _TEFinder(importlib.abc.MetaPathFinder):
        """Auto-create mock modules for any `transformer_engine.*` import."""

        def find_module(self, fullname, path=None):
            if fullname == "transformer_engine" or fullname.startswith("transformer_engine."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = _TEMockModule(fullname)
            sys.modules[fullname] = mod
            # Wire child onto parent so attribute access matches sys.modules
            parts = fullname.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, child_name, mod)
            return mod

    # Install the finder so ANY transformer_engine.* import is handled
    sys.meta_path.insert(0, _TEFinder())
    # Pre-create the root so `import transformer_engine` works immediately
    _TEFinder().load_module("transformer_engine")

from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

TASK_MAX_STEPS = {
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

# ── Prompt perturbation helpers (inlined from main.py) ───────────────────────

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


# ── Metric helpers (inlined from main.py) ────────────────────────────────────

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


# ── Config & helpers ─────────────────────────────────────────────────────────

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
    num_open_loop_steps: int = 16
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
    prompt_mode: str = "original"
    custom_prompt: str = ""

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


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _build_cosmos_cfg(args: Args) -> PolicyEvalConfig:
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
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs, flip_images: bool = True, vis_cfg: VisualPerturbConfig | None = None):
    """Prepare an observation dict for Cosmos get_action().

    Cosmos expects:
        primary_image: (H, W, 3) uint8 — third-person camera
        wrist_image:   (H, W, 3) uint8 — wrist camera
        proprio:       (9,) float — [gripper_qpos(2), eef_pos(3), eef_quat(4)]

    The flip correction is applied first, then the optional visual perturbation.
    """
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    img = np.ascontiguousarray(img)
    wrist_img = np.ascontiguousarray(wrist_img)
    if vis_cfg is not None:
        img = perturb_image(img, vis_cfg)
        wrist_img = perturb_image(wrist_img, vis_cfg)
    return {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
        )),
    }


# ── Main evaluation ─────────────────────────────────────────────────────────

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

    # Use absolute path for output so it works regardless of cwd
    out_path = pathlib.Path(args.video_out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

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

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            episode_object_shifts = apply_object_shift(env, policy_cfg, policy_rng)

            action_plan = collections.deque()
            action_log = []
            replay_images = []
            t = 0
            done = False
            is_new_chunk = False

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    observation = _prepare_observation(obs, flip_images=args.flip_images, vis_cfg=vis_cfg)
                    replay_images.append(observation["primary_image"])

                    is_new_chunk = not action_plan
                    if is_new_chunk:
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

                    policy_action = action_plan.popleft()
                    action, action_was_perturbed = maybe_perturb_action(policy_action, policy_cfg, policy_rng)
                    obs, reward, done, info = env.step(action.tolist())

                    log_entry = {
                        "t": t,
                        "kind": "random" if action_was_perturbed else "policy",
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "action_perturbed": bool(action_was_perturbed),
                        "reward": reward,
                        "done": done,
                        "is_chunk_start": is_new_chunk,
                    }
                    if action_was_perturbed:
                        log_entry["policy_action"] = np.asarray(policy_action, dtype=np.float32).tolist()
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
                out_path / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4",
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

            actions_path = out_path / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"
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
                        "visual_perturbation": vis_cfg.as_dict(),
                        "policy_perturbation": policy_cfg.as_dict(),
                        "object_shifts": episode_object_shifts,
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
    eval_libero(tyro.cli(Args))
