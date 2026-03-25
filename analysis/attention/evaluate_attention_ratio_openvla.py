#!/usr/bin/env python3
"""LIBERO evaluation with visual/linguistic attention ratio analysis for OpenVLA.

Hooks into q_proj and k_proj during the LLM prefill to compute attention from
the last text token (the "action-deciding" token) to visual (image patch) tokens
vs linguistic (text) tokens.

Token layout in the LLM: [BOS(0)] [image_patches(1..N)] [text_tokens(N+1..S-1)]
Query: last text token at index S-1 (proxy for action prediction context).

Visual fraction = visual_mass / (visual_mass + text_mass)
  where visual_mass = sum of attention weights to image tokens 1..N
        text_mass   = sum of attention weights to text tokens N+1..S-1

NOTE: Flash attention 2 does not materialise attention weights internally, so we
compute them manually from q_proj / k_proj hook outputs (prefill only, where the
full key matrix is visible to the query).

Usage:
  python evaluate_attention_ratio_openvla.py \\
    --checkpoint /path/to/openvla-7b \\
    --task-suite libero_10 --num-episodes 5 \\
    --layers 15 16 17 \\
    --output-dir results/attention/ratio/openvla/perturb/none/libero_10_seed7

  # Visual perturbation: rotate 30°
  python evaluate_attention_ratio_openvla.py \\
    --checkpoint /path/to/openvla-7b \\
    --task-suite libero_10 \\
    --visual-perturb-mode rotate --rotation-degrees 30 \\
    --output-dir results/attention/ratio/openvla/perturb/rotate30/libero_10_seed7

  # Visual perturbation: translate 20% right
  python evaluate_attention_ratio_openvla.py \\
    --checkpoint /path/to/openvla-7b \\
    --task-suite libero_10 \\
    --visual-perturb-mode translate --translate-x-frac 0.2 \\
    --output-dir results/attention/ratio/openvla/perturb/translate20/libero_10_seed7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

_OPENVLA_DIR = str(_PROJECT_ROOT / "openvla")
if _OPENVLA_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_DIR)

_LIBERO_EVAL_DIR = str(_PROJECT_ROOT / "examples" / "libero")
if _LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_EVAL_DIR)

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from visual_perturbations import VisualPerturbConfig, perturb_image
from policy_perturbations import PolicyPerturbConfig, apply_object_shift, maybe_perturb_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

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


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_openvla(
    checkpoint_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
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
        log.warning("No dataset_statistics.json found. Action un-normalization may fail.")

    return model


# ── Attention hooking ─────────────────────────────────────────────────────────


class AttentionCapturer:
    """Captures q_proj and k_proj outputs during LLM prefill for specified layers.

    Flash attention 2 does not materialise attention weights, so we hook the
    linear projections and recompute attention manually for the query token of
    interest (last text token in the prefill sequence).
    """

    def __init__(self, model: torch.nn.Module, layer_indices: List[int]):
        self.layer_indices = layer_indices
        self._q_buf: Dict[int, torch.Tensor] = {}
        self._k_buf: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register(model)

    def _register(self, model: torch.nn.Module) -> None:
        for layer_idx in self.layer_indices:
            layer = model.language_model.model.layers[layer_idx]
            self._hooks.append(
                layer.self_attn.q_proj.register_forward_hook(self._make_q_hook(layer_idx))
            )
            self._hooks.append(
                layer.self_attn.k_proj.register_forward_hook(self._make_k_hook(layer_idx))
            )

    def _make_q_hook(self, layer_idx: int):
        def hook(module, input, output):
            # Only capture prefill (seq_len > 1); during autoregressive decode seq_len == 1
            if output.shape[1] > 1:
                self._q_buf[layer_idx] = output.detach().float()
        return hook

    def _make_k_hook(self, layer_idx: int):
        def hook(module, input, output):
            if output.shape[1] > 1:
                self._k_buf[layer_idx] = output.detach().float()
        return hook

    def pop(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return captured q/k tensors keyed by layer index, then clear buffers."""
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx in self.layer_indices:
            if idx in self._q_buf and idx in self._k_buf:
                out[idx] = {"q": self._q_buf[idx], "k": self._k_buf[idx]}
        self._q_buf.clear()
        self._k_buf.clear()
        return out

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Attention ratio computation ───────────────────────────────────────────────


def compute_attention_ratio_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    num_image_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Dict:
    """Compute visual/linguistic attention ratio from q_proj and k_proj outputs.

    Token layout: [BOS(0)] [image_patches(1..N)] [text_tokens(N+1..S-1)]
    Query: last text token at position S-1.

    Args:
        q: [B, S, num_heads * head_dim]       float32 - q_proj output
        k: [B, S, num_kv_heads * head_dim]    float32 - k_proj output
        num_image_tokens: N, number of image patch tokens (excluding BOS)
        num_heads: number of query attention heads
        num_kv_heads: number of key/value heads (handles GQA)
        head_dim: dimension per head

    Returns:
        Dict with visual_mass, linguistic_mass, action_mass, visual_fraction, etc.
    """
    B, S, _ = q.shape
    q = q.view(B, S, num_heads, head_dim)             # [B, S, H, D]
    k = k.view(B, S, num_kv_heads, head_dim)          # [B, S, Hkv, D]

    # Expand kv heads to match query heads (grouped query attention)
    if num_kv_heads != num_heads:
        groups = num_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=2)        # [B, S, H, D]

    # Query is the last token in the prefill (= last text token)
    query_idx = S - 1
    q_query = q[:, query_idx, :, :]                   # [B, H, D]
    k_past = k[:, : query_idx + 1, :, :]              # [B, S, H, D]

    # Scaled dot-product: [B, H, S]
    scale = math.sqrt(head_dim)
    attn_logits = torch.einsum("bhd,bshd->bhs", q_query, k_past) / scale
    attn_weights = F.softmax(attn_logits, dim=-1)     # [B, H, S]

    # Average over batch and heads → [S]
    attn = attn_weights.mean(dim=(0, 1)).cpu().numpy()

    # BOS: index 0 (excluded from ratio denominator)
    # Image patches: indices 1..N
    # Text tokens: indices N+1..S-1
    visual_mass = float(attn[1 : num_image_tokens + 1].sum())
    linguistic_mass = float(attn[num_image_tokens + 1 :].sum())
    # No action tokens in prefill for OpenVLA (actions are decoded autoregressively)
    action_mass = 0.0
    total_mass = float(attn.sum())  # ≈ 1.0 (includes BOS)

    # Fraction denominator: visual + linguistic (exclude BOS for comparability with pi0)
    total_vl = visual_mass + linguistic_mass

    return {
        "visual_mass": visual_mass,
        "linguistic_mass": linguistic_mass,
        "action_mass": action_mass,
        "total_mass": total_mass,
        "visual_linguistic_ratio": visual_mass / max(linguistic_mass, 1e-8),
        "visual_fraction": visual_mass / max(total_vl, 1e-8),
        "linguistic_fraction": linguistic_mass / max(total_vl, 1e-8),
        "action_fraction": 0.0,
    }


def _summarize_episode_ratios(step_results: List[Dict]) -> Dict:
    if not step_results:
        return {}
    ratios = [r["visual_linguistic_ratio"] for r in step_results
              if np.isfinite(r["visual_linguistic_ratio"])]
    visual_fracs = [r["visual_fraction"] for r in step_results]
    ling_fracs = [r["linguistic_fraction"] for r in step_results]
    vis_masses = [r["visual_mass"] for r in step_results]
    ling_masses = [r["linguistic_mass"] for r in step_results]
    return {
        "visual_linguistic_ratio": {
            "mean": float(np.mean(ratios)) if ratios else 0.0,
            "std": float(np.std(ratios)) if ratios else 0.0,
            "median": float(np.median(ratios)) if ratios else 0.0,
            "min": float(np.min(ratios)) if ratios else 0.0,
            "max": float(np.max(ratios)) if ratios else 0.0,
        },
        "visual_mass": {"mean": float(np.mean(vis_masses)), "std": float(np.std(vis_masses))},
        "linguistic_mass": {"mean": float(np.mean(ling_masses)), "std": float(np.std(ling_masses))},
        "visual_fraction": {"mean": float(np.mean(visual_fracs)), "std": float(np.std(visual_fracs))},
        "linguistic_fraction": {"mean": float(np.mean(ling_fracs)), "std": float(np.std(ling_fracs))},
        "num_steps": len(step_results),
    }


def _build_avg_step_results(step_ratio_results: List[Dict]) -> List[Dict]:
    """Average per-step results across layers."""
    by_step: Dict[int, List[Dict]] = {}
    for r in step_ratio_results:
        by_step.setdefault(int(r["step"]), []).append(r)
    avg: List[Dict] = []
    for step, rs in sorted(by_step.items()):
        vis = [r["visual_mass"] for r in rs]
        ling = [r["linguistic_mass"] for r in rs]
        ratios = [r["visual_linguistic_ratio"] for r in rs
                  if np.isfinite(r["visual_linguistic_ratio"])]
        base = dict(rs[0])
        base["layer"] = "avg"
        base["step"] = step
        base["visual_mass"] = float(np.mean(vis))
        base["linguistic_mass"] = float(np.mean(ling))
        base["action_mass"] = 0.0
        base["visual_linguistic_ratio"] = float(np.mean(ratios)) if ratios else 0.0
        total_vl = base["visual_mass"] + base["linguistic_mass"]
        base["total_mass"] = total_vl
        base["visual_fraction"] = base["visual_mass"] / max(total_vl, 1e-8)
        base["linguistic_fraction"] = base["linguistic_mass"] / max(total_vl, 1e-8)
        base["action_fraction"] = 0.0
        avg.append(base)
    return avg


# ── Environment helpers ───────────────────────────────────────────────────────


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


# ── Prompt / action helpers ───────────────────────────────────────────────────


def _build_prompt(task_label: str, base_vla_name: str) -> str:
    if "openvla-v01" in base_vla_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take "
            f"to {task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def _normalize_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] *= -1.0
    return action


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ── Per-step inference with attention capture ─────────────────────────────────


def _get_action_with_attention(
    obs_img: np.ndarray,
    task_label: str,
    model,
    processor,
    unnorm_key: str,
    base_vla_name: str,
    capturer: AttentionCapturer,
    num_image_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_indices: List[int],
    center_crop: bool = True,
) -> tuple[np.ndarray, Dict[int, Dict]]:
    """Run one OpenVLA forward pass and return (action, per_layer_ratios)."""
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
            pass

    prompt = _build_prompt(task_label, base_vla_name)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # Retrieve prefill q/k captures (hooks fire during predict_action's prefill pass)
    captured = capturer.pop()

    layer_ratios: Dict[int, Dict] = {}
    for layer_idx in layer_indices:
        if layer_idx not in captured:
            continue
        ratio = compute_attention_ratio_from_qk(
            q=captured[layer_idx]["q"],
            k=captured[layer_idx]["k"],
            num_image_tokens=num_image_tokens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        layer_ratios[layer_idx] = ratio

    return np.asarray(action, dtype=np.float32), layer_ratios


# ── Episode loop ──────────────────────────────────────────────────────────────


def run_episode(
    env,
    model,
    processor,
    unnorm_key: str,
    base_vla_name: str,
    task_description: str,
    initial_state: np.ndarray,
    capturer: AttentionCapturer,
    layer_indices: List[int],
    num_image_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_steps: int,
    vis_cfg: Optional[VisualPerturbConfig] = None,
    policy_cfg: Optional[PolicyPerturbConfig] = None,
    policy_rng: Optional[np.random.Generator] = None,
    center_crop: bool = True,
) -> Dict:
    obs = env.reset()
    obs = env.set_init_state(initial_state)

    if policy_cfg is not None and policy_rng is not None:
        apply_object_shift(env, policy_cfg, policy_rng)

    step_ratio_results: List[Dict] = []
    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = obs["agentview_image"]
        if vis_cfg is not None and vis_cfg.mode != "none":
            img = perturb_image(img, vis_cfg)

        try:
            action, layer_ratios = _get_action_with_attention(
                obs_img=img,
                task_label=task_description,
                model=model,
                processor=processor,
                unnorm_key=unnorm_key,
                base_vla_name=base_vla_name,
                capturer=capturer,
                num_image_tokens=num_image_tokens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_indices=layer_indices,
                center_crop=center_crop,
            )
        except Exception as e:
            log.error(f"Inference error at t={t}: {e}")
            break

        action = _normalize_gripper_action(action)
        action = _invert_gripper_action(action)

        if policy_cfg is not None and policy_rng is not None:
            action, _ = maybe_perturb_action(action, policy_cfg, policy_rng)

        # Record per-layer attention ratios for this step
        for layer_idx, ratio in layer_ratios.items():
            entry = dict(ratio)
            entry["layer"] = layer_idx
            entry["step"] = t
            step_ratio_results.append(entry)

        if layer_ratios:
            avg_vis = float(np.mean([r["visual_fraction"] for r in layer_ratios.values()]))
            avg_ling = float(np.mean([r["linguistic_fraction"] for r in layer_ratios.values()]))
            log.info(f"  t={t}: vis_frac={avg_vis:.3f} ling_frac={avg_ling:.3f}")

        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    # Summarize per layer
    summary: Dict = {}
    for layer_idx in layer_indices:
        layer_results = [r for r in step_ratio_results if r.get("layer") == layer_idx]
        if layer_results:
            summary[f"layer_{layer_idx}"] = _summarize_episode_ratios(layer_results)

    if len(layer_indices) > 1 and step_ratio_results:
        avg_steps = _build_avg_step_results(step_ratio_results)
        if avg_steps:
            summary["layers_avg"] = _summarize_episode_ratios(avg_steps)

    return {
        "success": bool(done),
        "num_steps": t,
        "step_ratio_results": step_ratio_results,
        "summary": summary,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation with visual/linguistic attention ratio analysis for OpenVLA"
    )
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path or HF hub ID for OpenVLA checkpoint")
    parser.add_argument("--unnorm-key", type=str, default="",
                        help="Action un-normalization key (default: task_suite name)")
    parser.add_argument("--center-crop", action="store_true", default=True)
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    # LIBERO
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=list(TASK_MAX_STEPS.keys()))
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run a single task ID (default: all tasks)")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Episodes per task")
    parser.add_argument("--seed", type=int, default=7)

    # Attention
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 16, 17],
                        help="LLM layer indices to analyse (0-indexed)")
    parser.add_argument("--num-image-tokens", type=int, default=256,
                        help="Image patch tokens N (auto-detected from model if possible)")

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
    parser.add_argument("--output-dir", type=str,
                        default="results/attention_ratio_openvla")

    args = parser.parse_args()

    center_crop = not args.no_center_crop
    max_steps = TASK_MAX_STEPS[args.task_suite]

    vis_cfg = VisualPerturbConfig(
        mode=args.visual_perturb_mode,
        rotation_degrees=args.rotation_degrees,
        translate_x_frac=args.translate_x_frac,
        translate_y_frac=args.translate_y_frac,
    )
    policy_cfg = PolicyPerturbConfig(
        mode=args.policy_perturb_mode,
        random_action_prob=args.random_action_prob,
        random_action_scale=args.random_action_scale,
        object_shift_x_std=args.object_shift_x_std,
        object_shift_y_std=args.object_shift_y_std,
    )
    policy_rng = np.random.default_rng(args.seed + 9999)

    log.info(f"Visual perturbation:  {vis_cfg.as_dict()}")
    log.info(f"Policy perturbation:  {policy_cfg.as_dict()}")

    # ── Load model ────────────────────────────────────────────────────────
    log.info("Loading OpenVLA model...")
    model = _load_openvla(args.checkpoint, args.load_in_8bit, args.load_in_4bit)
    model.eval()

    unnorm_key = args.unnorm_key or args.task_suite
    if hasattr(model, "norm_stats"):
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key not in model.norm_stats:
            log.warning(
                f"unnorm_key '{unnorm_key}' not in model.norm_stats. "
                f"Available: {list(model.norm_stats.keys())}"
            )

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    # ── Determine LLM attention configuration ────────────────────────────
    llm_config = model.language_model.config
    num_heads = llm_config.num_attention_heads
    num_kv_heads = getattr(llm_config, "num_key_value_heads", num_heads)
    head_dim = llm_config.hidden_size // num_heads
    log.info(f"LLM: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Auto-detect num_image_tokens from model if possible
    num_image_tokens = args.num_image_tokens
    try:
        n = model.vision_backbone.featurizer.patch_embed.num_patches
        log.info(f"Auto-detected num_image_tokens={n} from vision backbone")
        num_image_tokens = n
    except AttributeError:
        log.info(f"Using --num-image-tokens={num_image_tokens}")

    # ── Register attention hooks ──────────────────────────────────────────
    capturer = AttentionCapturer(model, args.layers)
    log.info(f"Registered attention hooks on layers {args.layers}")

    # ── LIBERO setup ──────────────────────────────────────────────────────
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

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep_idx in range(min(args.num_episodes, len(initial_states))):
            log.info(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

            result = run_episode(
                env=env,
                model=model,
                processor=processor,
                unnorm_key=unnorm_key,
                base_vla_name=args.checkpoint,
                task_description=task_description,
                initial_state=initial_states[ep_idx],
                capturer=capturer,
                layer_indices=args.layers,
                num_image_tokens=num_image_tokens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_steps=max_steps,
                vis_cfg=vis_cfg,
                policy_cfg=policy_cfg,
                policy_rng=policy_rng,
                center_crop=center_crop,
            )

            result["task_id"] = task_id
            result["episode_idx"] = ep_idx
            result["task_description"] = task_description
            result["visual_perturbation"] = vis_cfg.as_dict()
            result["policy_perturbation"] = policy_cfg.as_dict()
            all_results.append(result)

            log.info(f"  Success: {result['success']}  Steps: {result['num_steps']}")
            summ = result.get("summary", {})
            log_key = "layers_avg" if "layers_avg" in summ else (
                f"layer_{args.layers[0]}" if args.layers else None
            )
            if log_key and log_key in summ:
                vf = summ[log_key].get("visual_fraction", {}).get("mean", 0.0)
                lf = summ[log_key].get("linguistic_fraction", {}).get("mean", 0.0)
                log.info(f"  {log_key}: vis_frac={vf:.3f} ling_frac={lf:.3f}")

        env.close()

    capturer.remove()

    # ── Serialize results ─────────────────────────────────────────────────
    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "step_ratio_results"}
        per_step_ratios: Dict = {}

        for layer_idx in args.layers:
            layer_key = f"layer_{layer_idx}"
            layer_results = [s for s in r.get("step_ratio_results", [])
                             if s.get("layer") == layer_idx]
            if layer_results:
                per_step_ratios[layer_key] = [
                    {
                        "step": int(s["step"]),
                        "visual_linguistic_ratio": float(s["visual_linguistic_ratio"]),
                        "visual_mass": float(s["visual_mass"]),
                        "linguistic_mass": float(s["linguistic_mass"]),
                        "action_mass": float(s["action_mass"]),
                        "visual_fraction": float(s["visual_fraction"]),
                        "linguistic_fraction": float(s["linguistic_fraction"]),
                        "action_fraction": float(s["action_fraction"]),
                    }
                    for s in layer_results
                ]

        if len(args.layers) > 1:
            avg_steps = _build_avg_step_results(r.get("step_ratio_results", []))
            if avg_steps:
                per_step_ratios["layers_avg"] = [
                    {
                        "step": int(s["step"]),
                        "visual_linguistic_ratio": float(s["visual_linguistic_ratio"]),
                        "visual_mass": float(s["visual_mass"]),
                        "linguistic_mass": float(s["linguistic_mass"]),
                        "action_mass": float(s["action_mass"]),
                        "visual_fraction": float(s["visual_fraction"]),
                        "linguistic_fraction": float(s["linguistic_fraction"]),
                        "action_fraction": float(s["action_fraction"]),
                    }
                    for s in avg_steps
                ]

        entry["per_step_ratios"] = per_step_ratios
        serializable.append(entry)

    out_path = os.path.join(args.output_dir, f"attention_ratio_results_{args.task_suite}.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {out_path}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    successes = sum(1 for r in all_results if r.get("success"))
    total = max(1, len(all_results))
    log.info(f"Success rate: {successes}/{len(all_results)} ({successes / total * 100:.1f}%)")

    log_key = "layers_avg" if len(args.layers) > 1 else f"layer_{args.layers[0]}"
    vis_fracs = [
        r["summary"][log_key]["visual_fraction"]["mean"]
        for r in all_results
        if log_key in r.get("summary", {})
    ]
    ling_fracs = [
        r["summary"][log_key]["linguistic_fraction"]["mean"]
        for r in all_results
        if log_key in r.get("summary", {})
    ]
    if vis_fracs:
        log.info(
            f"  {log_key}: mean_vis_frac={np.mean(vis_fracs):.3f} ± {np.std(vis_fracs):.3f}, "
            f"mean_ling_frac={np.mean(ling_fracs):.3f} ± {np.std(ling_fracs):.3f}"
        )


if __name__ == "__main__":
    main()
