"""Evaluate OpenVLA directly on LIBERO simulation benchmarks.

Runs OpenVLA locally (no WebSocket server) on a Linux cluster.
Requires a conda env with the openvla dependencies (torch, transformers, tensorflow, etc.).

Usage:
    conda activate vla

    # Full evaluation
    python examples/libero/openvla_eval.py \
        --checkpoint /path/to/openvla-7b \
        --task-suite-name libero_10 \
        --num-trials-per-task 20 \
        --seed 7

    # Quick smoke test (1 trial)
    python examples/libero/openvla_eval.py \
        --checkpoint /path/to/openvla-7b \
        --task-suite-name libero_10 \
        --num-trials-per-task 1

    # Prompt perturbation
    python examples/libero/openvla_eval.py \
        --checkpoint /path/to/openvla-7b \
        --prompt-mode shuffle

    # Visual perturbation
    python examples/libero/openvla_eval.py \
        --checkpoint /path/to/openvla-7b \
        --visual-perturb-mode rotate --rotation-degrees 30

    # Policy perturbation
    python examples/libero/openvla_eval.py \
        --checkpoint /path/to/openvla-7b \
        --policy-perturb-mode random_action --random-action-prob 0.25
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

import imageio
import numpy as np
from scipy.stats import gaussian_kde
import tqdm
import tyro

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives at examples/libero/ → repo root is 2 levels up
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_OPENVLA_DIR = str(_REPO_ROOT / "openvla")
if _OPENVLA_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_DIR)

import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action
from visual_perturbations import VisualPerturbConfig, perturb_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
ACTION_DIM = 7

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

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


# ── Prompt perturbation ───────────────────────────────────────────────────────

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
    elif mode == "custom":
        return original  # caller replaces with custom_prompt
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
    }


def compute_action_entropy_kde(action_log, action_dim=ACTION_DIM):
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


# ── OpenVLA inference ─────────────────────────────────────────────────────────

def _load_openvla(
    checkpoint_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load OpenVLA from a HF checkpoint directory (no pdb.set_trace)."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not load_in_8bit and not load_in_4bit:
        model = model.to(DEVICE)

    stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            model.norm_stats = json.load(f)
    else:
        log.warning(
            "No dataset_statistics.json found. Action un-normalization may fail if "
            "unnorm_key is set."
        )

    return model


def _build_openvla_action(
    obs_img: np.ndarray,
    task_label: str,
    model,
    processor,
    unnorm_key: str,
    base_vla_name: str,
    center_crop: bool = True,
) -> np.ndarray:
    """Run OpenVLA inference on one observation image; return raw 7-D action."""
    image = Image.fromarray(obs_img).convert("RGB")

    if center_crop:
        try:
            import tensorflow as tf
            batch_size, crop_scale = 1, 0.9
            img_tf = tf.convert_to_tensor(np.array(image))
            orig_dtype = img_tf.dtype
            img_tf = tf.image.convert_image_dtype(img_tf, tf.float32)
            new_side = tf.clip_by_value(tf.sqrt(crop_scale), 0.0, 1.0)
            new_h = tf.reshape(new_side, (batch_size,))
            new_w = tf.reshape(new_side, (batch_size,))
            h_off = (1.0 - new_h) / 2.0
            w_off = (1.0 - new_w) / 2.0
            boxes = tf.stack([h_off, w_off, h_off + new_h, w_off + new_w], axis=1)
            img_tf = tf.image.crop_and_resize(
                tf.expand_dims(img_tf, 0), boxes, tf.range(batch_size), (224, 224)
            )
            img_tf = tf.squeeze(img_tf, 0)
            img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
            img_tf = tf.image.convert_image_dtype(img_tf, orig_dtype, saturate=True)
            image = Image.fromarray(img_tf.numpy()).convert("RGB")
        except ImportError:
            log.warning("TensorFlow not available; skipping center crop.")

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take "
            f"to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def _normalize_gripper_action(action: np.ndarray) -> np.ndarray:
    """Map gripper from [0,1] → [-1,+1] and binarize."""
    action = action.copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign for LIBERO convention."""
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


# ── Environment helpers ───────────────────────────────────────────────────────

def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    # ── OpenVLA model ────────────────────────────────────────────────────
    checkpoint: str = "openvla/openvla-7b"
    """Path or HF hub ID for the OpenVLA checkpoint directory."""
    unnorm_key: str = ""
    """Action un-normalization key (default: derived from task_suite_name)."""
    center_crop: bool = True
    """Apply center-crop preprocessing (mimics training augmentation)."""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ── LIBERO environment ───────────────────────────────────────────────
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 20
    seed: int = 7

    # ── Output ───────────────────────────────────────────────────────────
    video_out_path: str = "data/libero/openvla/videos"

    # ── Prompt perturbation ──────────────────────────────────────────────
    prompt_mode: str = "original"
    """Prompt perturbation: original | empty | shuffle | random | synonym | opposite | custom."""
    custom_prompt: str = ""
    """Fixed prompt string used when prompt_mode='custom'."""

    # ── Visual perturbation ──────────────────────────────────────────────
    visual_perturb_mode: str = "none"
    """Image-level perturbation: none | rotate | translate | rotate_translate."""
    rotation_degrees: float = 30.0
    """Rotation angle in degrees (CCW). Used with rotate / rotate_translate modes."""
    translate_x_frac: float = 0.2
    """Horizontal shift as fraction of image width (positive = right)."""
    translate_y_frac: float = 0.0
    """Vertical shift as fraction of image height (positive = down)."""

    # ── Policy perturbation ──────────────────────────────────────────────
    policy_perturb_mode: str = "none"
    """Policy-level perturbation: none | random_action | object_shift."""
    random_action_prob: float = 0.25
    """Probability of replacing policy action with random noise per step."""
    random_action_scale: float = 1.0
    """Scale of uniform random action noise: Uniform(-scale, scale)."""
    object_shift_x_std: float = 0.05
    """Std (metres) of Gaussian object shift along x-axis at episode start."""
    object_shift_y_std: float = 0.0
    """Std (metres) of Gaussian object shift along y-axis at episode start."""


# ── Main evaluation ───────────────────────────────────────────────────────────

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    log.info(f"Visual perturbation:  {vis_cfg.as_dict()}")

    policy_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    policy_rng = np.random.default_rng(args.seed + 9999)
    log.info(f"Policy perturbation:  {policy_cfg.as_dict()}")
    log.info(f"Prompt perturbation:  {args.prompt_mode}")

    # ── Load model ────────────────────────────────────────────────────────
    log.info("Loading OpenVLA model...")
    model = _load_openvla(args.checkpoint, args.load_in_8bit, args.load_in_4bit)
    model.eval()

    # Resolve unnorm_key
    unnorm_key = args.unnorm_key or args.task_suite_name
    if hasattr(model, "norm_stats"):
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key not in model.norm_stats:
            log.warning(
                f"unnorm_key '{unnorm_key}' not in model.norm_stats. "
                f"Available: {list(model.norm_stats.keys())}"
            )

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    log.info("OpenVLA model loaded.")

    base_vla_name = args.checkpoint  # used to select prompt template

    # ── LIBERO benchmark setup ────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS[args.task_suite_name]

    # Collect all task descriptions for random prompt perturbation
    all_task_descriptions = []
    for i in range(num_tasks):
        try:
            t = task_suite.get_task(i)
            desc = getattr(t, "language", None) or str(t)
            all_task_descriptions.append(str(desc))
        except Exception:
            all_task_descriptions.append(f"task_{i}")
    all_task_descriptions = list(dict.fromkeys(all_task_descriptions))

    log.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks, max_steps={max_steps})")

    out_path = pathlib.Path(args.video_out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            # Select prompt
            if args.prompt_mode == "custom" and args.custom_prompt:
                prompt = args.custom_prompt
            else:
                prompt = perturb_prompt(str(task_description), args.prompt_mode, all_task_descriptions)
            log.info(f"\nTask: {task_description} | Prompt: {prompt}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            episode_object_shifts = apply_object_shift(env, policy_cfg, policy_rng)

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

                    img = obs["agentview_image"]
                    if vis_cfg.mode != "none":
                        img = perturb_image(img, vis_cfg)

                    replay_images.append(img.copy())

                    policy_action = _build_openvla_action(
                        img, prompt, model, processor, unnorm_key, base_vla_name, args.center_crop
                    )
                    # Post-process action for LIBERO
                    policy_action = _normalize_gripper_action(np.asarray(policy_action, dtype=np.float32))
                    policy_action = _invert_gripper_action(policy_action)

                    action, action_was_perturbed = maybe_perturb_action(policy_action, policy_cfg, policy_rng)
                    obs, reward, done, info = env.step(action.tolist())

                    log_entry = {
                        "t": t,
                        "kind": "random" if action_was_perturbed else "policy",
                        "action": action.tolist(),
                        "action_perturbed": bool(action_was_perturbed),
                        "reward": float(reward),
                        "done": bool(done),
                    }
                    if action_was_perturbed:
                        log_entry["policy_action"] = policy_action.tolist()
                    action_log.append(log_entry)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    log.error(f"Caught exception at t={t}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # ── Save replay video ─────────────────────────────────────────
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if replay_images:
                imageio.mimwrite(
                    out_path / f"rollout_{task_segment}_trial{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # ── Compute & save metrics ────────────────────────────────────
            smoothness_summary = compute_smoothness_metrics(action_log)
            if smoothness_summary:
                log.info(
                    f"Smoothness: mean_delta={smoothness_summary['mean_action_delta']:.4f}, "
                    f"max_delta={smoothness_summary['max_action_delta']:.4f}, "
                    f"gripper_transitions={smoothness_summary['num_gripper_transitions']}"
                )

            entropy_summary = compute_action_entropy_kde(action_log)
            if entropy_summary:
                log.info(f"Action entropy (KDE): {entropy_summary['action_entropy_kde']:.4f}")

            actions_path = out_path / f"actions_{task_segment}_trial{episode_idx}_{suffix}.json"
            with open(actions_path, "w") as f:
                json.dump(
                    {
                        "task_id": int(task_id),
                        "trial_id": int(episode_idx),
                        "seed": int(args.seed),
                        "task_description": str(task_description),
                        "prompt": prompt,
                        "prompt_mode": args.prompt_mode,
                        "success": bool(done),
                        "model": "openvla",
                        "checkpoint": args.checkpoint,
                        "visual_perturbation": vis_cfg.as_dict(),
                        "policy_perturbation": policy_cfg.as_dict(),
                        "object_shifts": episode_object_shifts,
                        "actions": action_log,
                        "smoothness": smoothness_summary,
                        "action_entropy": entropy_summary,
                    },
                    f,
                    indent=2,
                    default=_json_default,
                )

            log.info(f"Success: {done}")
            log.info(
                f"Episodes: {total_episodes} | Successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        log.info(f"Task {task_id} success rate: {task_successes / max(task_episodes, 1):.3f}")
        env.close()

    log.info(
        f"Total success rate: {total_successes / max(total_episodes, 1):.3f} "
        f"({total_successes}/{total_episodes})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
