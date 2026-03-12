"""Convert RoboCasa HDF5 datasets to LeRobot format for pi0-FAST finetuning.

RoboCasa HDF5 structure per demo:
    data/{demo_key}/obs/robot0_agentview_left_rgb   (T, H, W, 3) uint8
    data/{demo_key}/obs/robot0_eye_in_hand_rgb      (T, H, W, 3) uint8
    data/{demo_key}/actions                          (T, 12) float32
    data/{demo_key}/robot_states                     (T, 9)  float32
    data/{demo_key}.attrs["task_description"]        str

Usage:
    python examples/robocasa/convert_robocasa_data_to_lerobot.py \
        --data_dir /path/to/robocasa/composite_seen \
        --repo_id your_username/robocasa_composite_seen

Download the Composite-Seen dataset (16 tasks x 500 demos) from:
    https://robocasa.ai/docs/build/html/datasets/datasets_overview.html#target-datasets
"""

import io
import pathlib
import shutil

import h5py
import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from PIL import Image

# RoboCasa action space is 12D; first 7 = EEF pos/orn delta + gripper.
# Dims 7-11 control the mobile base and are not used for arm-only finetuning.
ACTION_DIM = 7
STATE_DIM = 9   # robot_states: EEF pos(3) + EEF quat(4) + gripper(2) or similar
FPS = 20


def _decode_jpeg(data) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(bytes(data))))


def _load_images(obs_group, key_raw: str, key_jpeg: str) -> np.ndarray:
    if key_raw in obs_group:
        return obs_group[key_raw][:]
    if key_jpeg in obs_group:
        frames = obs_group[key_jpeg]
        return np.stack([_decode_jpeg(frames[i]) for i in range(len(frames))])
    raise KeyError(f"Neither '{key_raw}' nor '{key_jpeg}' in obs group")


def _detect_image_shape(obs_group) -> tuple:
    if "robot0_agentview_left_rgb" in obs_group:
        return tuple(obs_group["robot0_agentview_left_rgb"].shape[1:])  # (H, W, 3)
    if "robot0_agentview_left_rgb_jpeg" in obs_group:
        img = Image.open(io.BytesIO(bytes(obs_group["robot0_agentview_left_rgb_jpeg"][0])))
        return (img.height, img.width, 3)
    raise KeyError("No agentview image in obs group")


def main(
    data_dir: str,
    repo_id: str = "your_username/robocasa_composite_seen",
    push_to_hub: bool = False,
):
    data_dir = pathlib.Path(data_dir)
    hdf5_files = sorted(data_dir.rglob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_dir}")
    print(f"Found {len(hdf5_files)} HDF5 file(s)")

    # Detect image shape from first available demo
    with h5py.File(hdf5_files[0], "r") as f:
        first_key = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[0]
        img_shape = _detect_image_shape(f[f"data/{first_key}/obs"])
    print(f"Image shape: {img_shape}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": img_shape, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": img_shape, "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_eps, n_frames = 0, 0
    for hdf5_path in hdf5_files:
        print(f"Processing {hdf5_path.name} ...")
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
                grp = f[f"data/{demo_key}"]
                obs = grp["obs"]
                task = grp.attrs["task_description"]

                imgs = _load_images(obs, "robot0_agentview_left_rgb", "robot0_agentview_left_rgb_jpeg")
                wrist = _load_images(obs, "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_rgb_jpeg")
                actions = grp["actions"][:, :ACTION_DIM].astype(np.float32)
                states = grp["robot_states"][:].astype(np.float32)

                for t in range(len(actions)):
                    dataset.add_frame({
                        "image": imgs[t],
                        "wrist_image": wrist[t],
                        "state": states[t],
                        "actions": actions[t],
                        "task": task,
                    })
                dataset.save_episode()
                n_eps += 1
                n_frames += len(actions)

        print(f"  {n_eps} episodes | {n_frames} frames so far")

    print(f"Converted {n_eps} episodes ({n_frames} frames) -> {output_path}")
    print("Next steps:")
    print(f"  1. Compute norm stats: python scripts/compute_norm_stats.py --repo-id {repo_id}")
    print(f"  2. Train: python scripts/train.py pi0_fast_robocasa --exp-name my_run")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["robocasa", "panda", "composite-seen"],
            private=False,
            push_videos=True,
            license="mit",
        )


if __name__ == "__main__":
    tyro.cli(main)
