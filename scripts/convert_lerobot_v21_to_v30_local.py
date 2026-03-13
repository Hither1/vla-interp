"""Convert a local LeRobot v2.1 dataset to v3.0 format without HuggingFace access.

Usage:
    python scripts/convert_lerobot_v21_to_v30_local.py \
        --src /path/to/dataset_v21 \
        --dst /path/to/dataset_v30
"""

import argparse
import json
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset


def load_jsonlines(path):
    with jsonlines.open(path) as reader:
        return list(reader)


def cast_stats_to_numpy(stats: dict) -> dict:
    return {k: {sk: np.array(sv) for sk, sv in v.items()} for k, v in stats.items()}


def aggregate_stats(ep_stats_list: list[dict]) -> dict:
    """Aggregate per-episode stats into global stats."""
    all_keys = ep_stats_list[0].keys()
    global_stats = {}
    for key in all_keys:
        means = np.array([s[key]["mean"] for s in ep_stats_list])
        stds = np.array([s[key]["std"] for s in ep_stats_list])
        mins = np.array([s[key]["min"] for s in ep_stats_list])
        maxs = np.array([s[key]["max"] for s in ep_stats_list])
        counts = np.array([s[key].get("count", 1) for s in ep_stats_list], dtype=float)
        total = counts.sum()
        global_mean = (means * counts).sum(axis=0) / total
        global_var = ((stds**2 + (means - global_mean) ** 2) * counts).sum(axis=0) / total
        global_stats[key] = {
            "mean": global_mean.tolist(),
            "std": np.sqrt(global_var).tolist(),
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
        }
    return global_stats


def main(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # --- Load v2.1 metadata ---
    info = json.loads((src / "meta" / "info.json").read_text())
    episodes_raw = load_jsonlines(src / "meta" / "episodes.jsonl")
    ep_stats_raw = load_jsonlines(src / "meta" / "episodes_stats.jsonl")
    tasks_raw = load_jsonlines(src / "meta" / "tasks.jsonl")

    episodes_by_idx = {e["episode_index"]: e for e in episodes_raw}
    ep_stats_by_idx = {e["episode_index"]: cast_stats_to_numpy(e["stats"]) for e in ep_stats_raw}
    valid_indices = sorted(episodes_by_idx.keys())

    print(f"Valid episodes: {len(valid_indices)}")

    # --- 1. Update info.json ---
    new_info = dict(info)
    new_info["codebase_version"] = "v3.0"
    new_info.pop("total_chunks", None)
    new_info.pop("total_videos", None)
    new_info["data_files_size_in_mb"] = 100
    new_info["video_files_size_in_mb"] = 500
    new_info["fps"] = int(info["fps"])
    # Add fps to non-video features
    for key in new_info["features"]:
        if new_info["features"][key].get("dtype") != "video":
            new_info["features"][key]["fps"] = new_info["fps"]
    (dst / "meta").mkdir(exist_ok=True)
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print("Wrote info.json")

    # --- 2. Write tasks.parquet ---
    tasks_df = pd.DataFrame([{"task_index": t["task_index"], "task": t["task"]} for t in tasks_raw])
    tasks_df.to_parquet(dst / "meta" / "tasks.parquet", index=False)
    print("Wrote tasks.parquet")

    # --- 3. Copy/repack data parquets (only valid episodes) ---
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    data_src = src / "data" / "chunk-000"
    data_dst = dst / "data" / "chunk-000"

    ep_metadata_rows = []
    frame_offset = 0
    for ep_idx in valid_indices:
        src_file = data_src / f"episode_{ep_idx:06d}.parquet"
        if not src_file.exists():
            print(f"WARNING: missing data file for episode {ep_idx}, skipping")
            continue
        dst_file = data_dst / f"episode_{ep_idx:06d}.parquet"
        shutil.copy2(src_file, dst_file)
        table = pq.read_table(src_file)
        n_frames = len(table)
        ep_metadata_rows.append({
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": frame_offset,
            "dataset_to_index": frame_offset + n_frames,
            "length": n_frames,
        })
        frame_offset += n_frames
    print(f"Copied {len(ep_metadata_rows)} data files")

    # --- 4. Copy video files ---
    videos_src = src / "videos"
    videos_dst = dst / "videos"
    if videos_src.exists():
        if videos_dst.exists():
            shutil.rmtree(videos_dst)
        shutil.copytree(videos_src, videos_dst)
        print("Copied videos")

    # --- 5. Write meta/episodes parquet ---
    (dst / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    # Add per-episode stats as flattened columns
    for row in ep_metadata_rows:
        ep_idx = row["episode_index"]
        if ep_idx in ep_stats_by_idx:
            for feat_key, feat_stats in ep_stats_by_idx[ep_idx].items():
                for stat_key, stat_val in feat_stats.items():
                    col = f"stats/{feat_key}/{stat_key}"
                    row[col] = stat_val.tolist() if hasattr(stat_val, "tolist") else stat_val
    ep_ds = Dataset.from_list(ep_metadata_rows)
    ep_ds.to_parquet(dst / "meta" / "episodes" / "chunk-000" / "file_000.parquet")
    print("Wrote episodes parquet")

    # --- 6. Write global stats.json ---
    ep_stats_list = [ep_stats_by_idx[i] for i in valid_indices if i in ep_stats_by_idx]
    global_stats = aggregate_stats(ep_stats_list)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    (dst / "meta" / "stats.json").write_text(json.dumps(to_serializable(global_stats), indent=2))
    print("Wrote stats.json")

    print(f"\nDone! v3.0 dataset at: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    main(args.src, args.dst)
