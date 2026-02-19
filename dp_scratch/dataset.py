"""LIBERO HDF5 dataset loader for Diffusion Policy training."""

import os
import glob

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def quat2axisangle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle (3,)."""
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


ALL_SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]


class LiberoDataset(Dataset):
    """
    Loads all demos from one or more LIBERO task suites (HDF5 files).

    Each sample returns:
        images:   (n_obs_steps, n_cameras, 3, H, W) float32 [0,1]
        state:    (n_obs_steps, 8) float32
        actions:  (horizon, 7) float32
        task_idx: long scalar

    States and actions are preloaded; images are read lazily from HDF5.
    """

    def __init__(self, data_dir, suite_names, n_obs_steps=2, horizon=16, img_size=224, verbose=True):
        if isinstance(suite_names, str):
            suite_names = [suite_names]
        self.suite_names = list(suite_names)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.img_size = img_size
        self.verbose = verbose

        # Build episode list and task descriptions across all suites
        self.episodes = []
        self.task_descs = []
        self._task_map = {}

        for suite_name in self.suite_names:
            suite_dir = os.path.join(data_dir, suite_name)
            hdf5_files = sorted(glob.glob(os.path.join(suite_dir, "*.hdf5")))
            assert len(hdf5_files) > 0, f"No HDF5 files found in {suite_dir}"

            for fpath in hdf5_files:
                fname = os.path.splitext(os.path.basename(fpath))[0]
                task_name = fname.replace("_demo", "").replace("_", " ")
                if task_name not in self._task_map:
                    self._task_map[task_name] = len(self.task_descs)
                    self.task_descs.append(task_name)
                task_idx = self._task_map[task_name]

                with h5py.File(fpath, "r") as f:
                    demo_keys = sorted(f["data"].keys())
                    for dk in demo_keys:
                        T = f["data"][dk]["actions"].shape[0]
                        self.episodes.append({
                            "path": fpath,
                            "demo_key": dk,
                            "task_idx": task_idx,
                            "length": T,
                        })

        # Preload states and actions (small memory footprint)
        if self.verbose:
            print(f"Preloading states and actions for {len(self.episodes)} episodes...")
        self._states = []
        self._actions = []
        for i, ep in enumerate(self.episodes):
            if self.verbose and i % 50 == 0:
                print(f"  Loading episode: {i}/{len(self.episodes)}")
            with h5py.File(ep["path"], "r") as f:
                demo = f["data"][ep["demo_key"]]
                eef_pos = demo["obs"]["robot0_eef_pos"][:].astype(np.float32)
                eef_quat = demo["obs"]["robot0_eef_quat"][:].astype(np.float64)
                grip = demo["obs"]["robot0_gripper_qpos"][:].astype(np.float32)
                aa = np.stack([quat2axisangle(q) for q in eef_quat])
                state = np.concatenate([eef_pos, aa, grip], axis=-1)  # (T, 8)
                self._states.append(state)
                self._actions.append(demo["actions"][:].astype(np.float32))

        # Build sample index: (episode_idx, start_step)
        if self.verbose:
            print("Building sample index...")
        self.samples = []
        for ep_idx, ep in enumerate(self.episodes):
            T = ep["length"]
            for t in range(T - horizon + 1):
                self.samples.append((ep_idx, t))

        if self.verbose:
            suites_str = "+".join(self.suite_names)
            print(f"LiberoDataset [{suites_str}]: {len(self.task_descs)} tasks, "
                  f"{len(self.episodes)} episodes, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]

        # Observation timestep indices (with padding at episode start)
        obs_indices = [max(0, t - i) for i in reversed(range(self.n_obs_steps))]

        # State and actions from preloaded arrays
        states = np.stack([self._states[ep_idx][oi] for oi in obs_indices])
        actions = self._actions[ep_idx][t: t + self.horizon].copy()

        # Images from HDF5 (lazy read)
        with h5py.File(ep["path"], "r") as f:
            demo = f["data"][ep["demo_key"]]
            images = []
            for oi in obs_indices:
                base_img = self._load_img(demo["obs"]["agentview_image"], oi)
                wrist_img = self._load_img(demo["obs"]["robot0_eye_in_hand_image"], oi)
                images.append(np.stack([base_img, wrist_img], axis=0))
            images = np.stack(images, axis=0)  # (n_obs, 2, H, W, 3)

        # To tensors
        images = torch.from_numpy(images).float().permute(0, 1, 4, 2, 3) / 255.0  # (T, C, 3, H, W)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)

        return {
            "images": images,
            "state": states,
            "actions": actions,
            "task_idx": torch.tensor(ep["task_idx"], dtype=torch.long),
        }

    def _load_img(self, dataset, idx):
        """Load, flip 180°, and resize a single image."""
        img = dataset[idx][::-1, ::-1].copy()  # flip to match eval convention
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img


def compute_stats(dataset: LiberoDataset, verbose: bool = True) -> dict:
    """Compute mean/std of actions and states from preloaded arrays."""
    if verbose:
        print(f"Computing stats for {len(dataset._actions)} episodes...")
        print("  Concatenating actions arrays...")
    all_actions = np.concatenate(dataset._actions, axis=0)
    if verbose:
        print(f"    actions shape: {all_actions.shape}")
        print("  Concatenating states arrays...")
    all_states = np.concatenate(dataset._states, axis=0)
    if verbose:
        print(f"    states shape: {all_states.shape}")
        print("  Computing mean and std...")
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = all_actions.std(axis=0).astype(np.float32)
    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = all_states.std(axis=0).astype(np.float32)
    
    if verbose:
        print("Stats computation done.")
    
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "state_mean": state_mean,
        "state_std": state_std,
    }
