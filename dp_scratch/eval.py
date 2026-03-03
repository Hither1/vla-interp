"""Evaluate Diffusion Policy on LIBERO benchmark.

Usage:
    python -m dp_scratch.eval --ckpt checkpoints/dp/.../ckpt_300.pt
    python -m dp_scratch.eval --ckpt ... --libero_path /path/to/LIBERO
"""

import argparse
import os
import sys

import numpy as np
import torch

_orig_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_patch
from scipy.spatial.transform import Rotation

from dp_scratch.model import DiffusionPolicy
from dp_scratch.dataset import ALL_SUITES

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def get_suite_max_steps(suite_name):
    return SUITE_MAX_STEPS.get(suite_name, 400)


def quat2axisangle(quat):
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


def build_obs_dict(obs, task_language):
    return {
        "observation/image": obs["agentview_image"].copy(),
        "observation/wrist_image": obs["robot0_eye_in_hand_image"].copy(),
        "observation/state": np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32),
        "prompt": task_language,
    }


def evaluate_task(model, env, init_states, task_language, n_episodes, max_steps,
                   replan_steps=8, verbose=False, record_video=False):
    """Run episodes for a single task. Returns (successes, videos).
    videos: list of (frames_list, done_bool) if record_video else empty list.
    """
    from tqdm import tqdm
    total_eps = min(n_episodes, len(init_states))
    successes = []
    videos = []
    for ep in range(total_eps):
        env.reset()
        env.set_init_state(init_states[ep])
        obs, _, _, _ = env.step([0.0] * 7)

        done = False
        step = 0
        action_queue = []
        n_chunks = 0
        frames = []

        pbar = tqdm(total=max_steps, desc=f"    ep{ep+1}/{total_eps}", leave=False)
        while not done and step < max_steps:
            if record_video:
                frames.append(obs["agentview_image"].copy())

            if len(action_queue) == 0:
                obs_dict = build_obs_dict(obs, task_language)
                result = model.infer(obs_dict)
                action_queue = list(result["actions"][:replan_steps])
                if verbose and ep == 0 and n_chunks < 3:
                    chunk = result["actions"]
                    print(f"    [diag] chunk {n_chunks} | action[0]={chunk[0].tolist()} | range=[{chunk.min():.3f}, {chunk.max():.3f}]")
                n_chunks += 1

            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action.tolist())
            step += 1
            pbar.update(1)
        pbar.close()

        sr_so_far = (sum(successes) + float(done)) / (ep + 1)
        successes.append(float(done))
        if record_video and frames:
            videos.append((frames, done))
        print(f"    ep{ep+1}/{total_eps} | steps={step}/{max_steps} | {'OK' if done else 'FAIL'} | sr={sr_so_far:.2f}")
    return successes, videos


# ── Reusable helpers for training-time eval ──────────────────────────────────


def setup_suite_envs(suite_name, seed, max_tasks=None):
    """Pre-create envs for tasks in a suite.
    max_tasks: if set, only create envs for first max_tasks tasks (e.g. 1 for quick eval).
    Returns: list of (env, init_states, task_language)
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    n_tasks = task_suite.n_tasks if max_tasks is None else min(max_tasks, task_suite.n_tasks)
    envs_data = []
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        task_bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(seed)
        init_states = task_suite.get_task_init_states(task_id)
        envs_data.append((env, init_states, task.language))

    return envs_data


def evaluate_with_envs(model, envs_data, n_episodes, max_steps, replan_steps=8, record_video=False):
    """Evaluate using pre-created envs.
    Returns (per_task_dict, avg_success_rate, all_videos).
    all_videos: list of (task_language, frames, done) if record_video.
    """
    results = {}
    all_videos = []
    for env, init_states, task_language in envs_data:
        successes, videos = evaluate_task(
            model, env, init_states, task_language,
            n_episodes, max_steps, replan_steps, record_video=record_video,
        )
        results[task_language] = np.mean(successes)
        for frames, done in videos:
            all_videos.append((task_language, frames, done))
    avg = np.mean(list(results.values()))
    return results, avg, all_videos


def close_envs(envs_data):
    for env, _, _ in envs_data:
        env.close()


# ── Standalone evaluation ────────────────────────────────────────────────────


def evaluate_suite(model, suite_name, n_episodes, max_steps, seed, verbose=False):
    """Evaluate model on all tasks of a single suite. Returns per-task dict."""
    envs_data = setup_suite_envs(suite_name, seed)

    if verbose:
        print(f"\n  [diag] Model task_descs ({len(model.task_descs)} total):")
        for i, d in enumerate(model.task_descs):
            print(f"    [{i:2d}] {d}")

    results = {}
    for task_id, (env, init_states, task_language) in enumerate(envs_data):
        matched_idx = model._prompt_to_idx(task_language)
        matched_desc = model.task_descs[matched_idx] if matched_idx < len(model.task_descs) else "???"
        match_type = "exact" if task_language.strip().lower() == matched_desc.strip().lower() else ("substr" if matched_idx > 0 or task_language.strip().lower() in matched_desc.lower() else "FALLBACK")
        if verbose:
            print(f"\n  Task {task_id:2d} | prompt=\"{task_language}\"")
            print(f"           matched=[{matched_idx}] \"{matched_desc}\" ({match_type})")

        successes, _ = evaluate_task(
            model, env, init_states, task_language,
            n_episodes=n_episodes,
            max_steps=max_steps,
            verbose=verbose and task_id < 2,
        )
        avg = np.mean(successes)
        results[task_language] = avg
        print(f"  Task {task_id:2d} | success={avg:.2f} | {task_language}")
    close_envs(envs_data)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="+", default=None,
                        choices=ALL_SUITES,
                        help="Suites to evaluate. Default: all suites.")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path. Omit to eval random init (requires --data_dir)")
    parser.add_argument("--data_dir", default=None,
                        help="Data dir for task_descs/stats when --ckpt not given")
    parser.add_argument("--libero_path", default=None,
                        help="Path to LIBERO repo (dir containing 'libero' pkg). Or set env LIBERO_PATH.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps. Default: per-suite setting.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print diagnostic info: task matching, action ranges, etc.")
    args = parser.parse_args()

    libero_dir = args.libero_path or os.environ.get("LIBERO_PATH")
    if not libero_dir:
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent = os.path.dirname(proj_root)
        for name in ["LIBERO", "LIBERO-master"]:
            p = os.path.join(parent, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "libero")):
                libero_dir = p
                break
    if libero_dir:
        libero_dir = os.path.abspath(os.path.expanduser(libero_dir))
        if libero_dir not in sys.path:
            sys.path.insert(0, libero_dir)
    import importlib.util
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise RuntimeError(
            "libero not found. Set --libero_path /path/to/LIBERO or env LIBERO_PATH"
        )
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not libero_dir and spec.origin:
        libero_dir = os.path.dirname(os.path.dirname(spec.origin))
    if libero_dir:
        lb = os.path.join(libero_dir, "libero", "libero")
        cfg_dir = os.path.join(proj_root, ".libero_eval_config")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
            f.write(f"benchmark_root: {lb}\nbddl_files: {lb}/bddl_files\ninit_states: {lb}/init_files\ndatasets: {os.path.join(libero_dir, 'libero', 'datasets')}\nassets: {lb}/assets\n")
        os.environ["LIBERO_CONFIG_PATH"] = cfg_dir

    device = torch.device(args.device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = DiffusionPolicy(task_descs=ckpt["task_descs"]).to(device)
        model.set_norm_stats(**ckpt["stats"])
        model.load_state_dict(ckpt["model"])
        suites = args.suite or ckpt.get("suites", ALL_SUITES)
    else:
        if not args.data_dir:
            raise RuntimeError("--data_dir required when --ckpt not specified (for task_descs and stats)")
        from dp_scratch.dataset import LiberoDataset, compute_stats, ALL_SUITES as _ALL
        dataset = LiberoDataset(args.data_dir, _ALL, verbose=False)
        stats = compute_stats(dataset, verbose=False)
        model = DiffusionPolicy(task_descs=dataset.task_descs).to(device)
        model.set_norm_stats(**stats)
        suites = args.suite or _ALL
        print("Eval random init (no checkpoint)")
    model.eval()

    all_suite_results = {}
    for suite_name in suites:
        max_steps = args.max_steps or get_suite_max_steps(suite_name)
        print(f"\n{'='*60}")
        print(f"Evaluating: {suite_name}  (max_steps={max_steps})")
        print(f"{'='*60}")
        results = evaluate_suite(
            model, suite_name,
            n_episodes=args.n_episodes,
            max_steps=max_steps,
            seed=args.seed,
            verbose=args.verbose,
        )
        suite_avg = np.mean(list(results.values()))
        all_suite_results[suite_name] = suite_avg
        print(f"  => {suite_name} average: {suite_avg:.3f}")

    print(f"\n{'='*60}")
    print("Summary:")
    for s, v in all_suite_results.items():
        print(f"  {s:20s} : {v:.3f}")
    overall = np.mean(list(all_suite_results.values()))
    print(f"  {'Overall':20s} : {overall:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
