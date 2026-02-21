"""Policy-level perturbations for LIBERO evaluation.

Two perturbation modes:

random_action
    At each execution step, with probability p replace the policy action
    with uniform random noise in [-scale, scale]^D.

object_shift
    At episode start (after env.set_init_state()), randomly displace every
    non-robot free object along x (and optionally y) by a Gaussian amount.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np


@dataclasses.dataclass
class PolicyPerturbConfig:
    mode: str = "none"
    """Perturbation mode. One of: none | random_action | object_shift."""

    random_action_prob: float = 0.25
    """Probability of replacing the policy action at each step (random_action mode)."""

    random_action_scale: float = 1.0
    """Action noise sampled from Uniform(-scale, scale)."""

    object_shift_x_std: float = 0.05
    """Std (metres) of Gaussian shift along x applied to each free object at episode start."""

    object_shift_y_std: float = 0.0
    """Std (metres) of Gaussian shift along y. Set to 0 for x-only shift."""

    def as_dict(self) -> dict:
        d: dict = {"mode": self.mode}
        if self.mode == "random_action":
            d["random_action_prob"] = self.random_action_prob
            d["random_action_scale"] = self.random_action_scale
        elif self.mode == "object_shift":
            d["object_shift_x_std"] = self.object_shift_x_std
            d["object_shift_y_std"] = self.object_shift_y_std
        return d


def maybe_perturb_action(
    action: np.ndarray,
    cfg: PolicyPerturbConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Optionally replace action with uniform random noise.

    Returns (output_action, was_perturbed).
    """
    if cfg.mode != "random_action":
        return action, False
    if rng.random() < cfg.random_action_prob:
        noisy = rng.uniform(
            -cfg.random_action_scale,
            cfg.random_action_scale,
            size=action.shape,
        ).astype(np.float32)
        return noisy, True
    return action, False


_ROBOT_KEYWORDS = ("robot", "gripper", "arm", "base", "torso", "shoulder", "elbow", "wrist", "finger")


def _joint_name(sim, j: int) -> str:
    try:
        return sim.model.joint_id2name(j)
    except AttributeError:
        pass
    try:
        return sim.model.joint_names[j]
    except Exception:
        return f"joint_{j}"


def apply_object_shift(
    env,
    cfg: PolicyPerturbConfig,
    rng: np.random.Generator,
) -> dict:
    """Randomly shift non-robot free objects in the MuJoCo sim.

    Call after env.set_init_state() and before the first env.step().
    Calls env.sim.forward() to propagate changes.

    Returns dict mapping joint_name -> {"dx": float, "dy": float}.
    Returns {} when cfg.mode != 'object_shift'.
    """
    if cfg.mode != "object_shift":
        return {}

    sim = env.sim
    shifts: dict[str, dict] = {}

    for j in range(sim.model.njnt):
        # mjJNT_FREE == 0 marks floating free joints (movable objects).
        if sim.model.jnt_type[j] != 0:
            continue

        jname = _joint_name(sim, j)
        if any(kw in jname.lower() for kw in _ROBOT_KEYWORDS):
            continue

        dx = float(rng.normal(0.0, cfg.object_shift_x_std)) if cfg.object_shift_x_std > 0 else 0.0
        dy = float(rng.normal(0.0, cfg.object_shift_y_std)) if cfg.object_shift_y_std > 0 else 0.0

        addr = sim.model.jnt_qposadr[j]
        sim.data.qpos[addr] += dx
        sim.data.qpos[addr + 1] += dy

        shifts[jname] = {"dx": dx, "dy": dy}
        logging.debug("Object shift: %r  dx=%.4f  dy=%.4f", jname, dx, dy)

    sim.forward()
    logging.info("Object shift applied to %d free object(s): %s", len(shifts), list(shifts.keys()))
    return shifts
