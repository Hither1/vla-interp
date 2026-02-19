from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

import openpi.models.gemma as gemma
from overcomplete.sae import TopKSAE, BatchTopKSAE

# ckpt = torch.load("checkpoints/BatchTopKSAE/sae_libero_all_layer11_k16_c512.pt", map_location="cpu", weights_only=False)
# sae = BatchTopKSAE(
#     ckpt["d"],
#     nb_concepts=ckpt["nb_concepts"],
#     top_k=ckpt["top_k"],
#     device="cpu",
#     )
# sae.load_state_dict(ckpt["model_state_dict"])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sae.eval().to(device)
# gemma.SAE_MODEL = sae   # torch SAE
# gemma.SAE_INTERVENTION.update({
#     "enabled": True,
#     "mode": "ablate",          # or "steer"
#     "feature_idx": 42,         # e.g. grasp-readiness
#     "strength": 2.0,           # steering only
#     "layer_idx": 11,           # planning layer
# })


BasePolicy: TypeAlias = _base_policy.BasePolicy


def _compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None) -> float:
    """Compute MMD² between two sample sets using a Gaussian RBF kernel.

    Args:
        X: (n, d) samples from distribution P.
        Y: (m, d) samples from distribution Q.
        bandwidth: RBF kernel bandwidth σ.  If None, uses median heuristic.

    Returns:
        Non-negative MMD² estimate.
    """
    def _sq_dists(A, B):
        # ||a-b||² = ||a||² + ||b||² - 2 a·b
        return np.maximum(
            np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1, keepdims=True).T - 2 * A @ B.T,
            0.0,
        )

    if bandwidth is None:
        all_pts = np.vstack([X, Y])
        dists = _sq_dists(all_pts, all_pts)
        med = np.median(dists[dists > 0])
        bandwidth = float(np.sqrt(med)) if med > 0 else 1.0

    gamma = 1.0 / (2.0 * bandwidth**2)
    mmd2 = (
        np.exp(-gamma * _sq_dists(X, X)).mean()
        + np.exp(-gamma * _sq_dists(Y, Y)).mean()
        - 2.0 * np.exp(-gamma * _sq_dists(X, Y)).mean()
    )
    return float(max(0.0, mmd2))


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # Extract analysis flags before transforms strip them.
        compute_mmd = bool(inputs.pop("compute_mmd", False))
        mmd_num_samples = int(inputs.pop("mmd_num_samples", 8))

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        # Multi-sample MMD: draw additional diffusion samples from the same
        # observation, then compute split-sample kernel MMD to measure how
        # spread / multimodal the action distribution is.
        mmd_data = {}
        if compute_mmd and not self._is_pytorch_model:
            mmd_start = time.monotonic()
            all_samples = [np.asarray(actions[0])]  # first sample, shape (ah, ad)
            for _ in range(mmd_num_samples - 1):
                self._rng, s_rng = jax.random.split(self._rng)
                s = self._sample_actions(s_rng, observation, **sample_kwargs)
                all_samples.append(np.asarray(s[0]))
            samples_np = np.stack(all_samples)  # (N, ah, ad)

            # Flatten action chunks for kernel MMD: (N, ah*ad)
            flat = samples_np.reshape(mmd_num_samples, -1)
            half = mmd_num_samples // 2
            mmd_score = _compute_mmd_rbf(flat[:half], flat[half : 2 * half])

            # Per-timestep std across samples (first 7 dims = real LIBERO actions)
            per_step_std = np.std(samples_np[:, :, :7], axis=0)  # (ah, 7)

            mmd_data["mmd_score"] = mmd_score
            mmd_data["action_sample_std"] = float(np.mean(per_step_std))
            mmd_data["action_std_per_timestep"] = np.mean(per_step_std, axis=-1).tolist()
            mmd_data["mmd_timing_ms"] = (time.monotonic() - mmd_start) * 1000

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs.update(mmd_data)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
