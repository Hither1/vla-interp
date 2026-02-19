"""Evaluate Diffusion Policy on LIBERO benchmark.

Usage:
    # Eval on all four suites (auto-detected from checkpoint):
    python -m dp_scratch.eval --ckpt checkpoints/dp/.../ckpt_300.pt

    # Eval on specific suites:
    python -m dp_scratch.eval --suite libero_spatial libero_goal --ckpt checkpoints/dp/.../ckpt_300.pt

Requires LIBERO environment:
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
"""

import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dp_scratch.model import DiffusionPolicy
from dp_scratch.dataset import ALL_SUITES


def quat2axisangle(quat):
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


def build_obs_dict(obs, task_language):
    """Convert raw LIBERO env observation to policy input dict."""
    return {
        "observation/image": obs["agentview_image"][::-1, ::-1].copy(),
        "observation/wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        "observation/state": np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32),
        "prompt": task_language,
    }


def evaluate_task(model, env, init_states, task_language, n_episodes, max_steps, replan_steps=8):
    """Run episodes for a single task and return per-episode success."""
    successes = []
    for ep in range(min(n_episodes, len(init_states))):
        env.reset()
        env.set_init_state(init_states[ep])
        obs, _, _, _ = env.step([0.0] * 7)

        done = False
        step = 0
        action_queue = []

        while not done and step < max_steps:
            if len(action_queue) == 0:
                obs_dict = build_obs_dict(obs, task_language)
                result = model.infer(obs_dict)
                action_queue = list(result["actions"][:replan_steps])

            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action.tolist())
            step += 1

        successes.append(float(done))
    return successes


def evaluate_suite(model, suite_name, n_episodes, max_steps, seed):
    """Evaluate model on all tasks of a single suite. Returns per-task dict."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    results = {}
    for task_id in range(task_suite.n_tasks):
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
        successes = evaluate_task(
            model, env, init_states, task.language,
            n_episodes=n_episodes,
            max_steps=max_steps,
        )
        avg = np.mean(successes)
        results[task.language] = avg
        print(f"  Task {task_id:2d} | success={avg:.2f} | {task.language}")
        env.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="+", default=None,
                        choices=ALL_SUITES,
                        help="Suites to evaluate. Default: all suites from checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiffusionPolicy(task_descs=ckpt["task_descs"]).to(device)
    model.set_norm_stats(**ckpt["stats"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Determine which suites to evaluate
    suites = args.suite or ckpt.get("suites", ALL_SUITES)

    # Evaluate each suite
    all_suite_results = {}
    for suite_name in suites:
        print(f"\n{'='*60}")
        print(f"Evaluating: {suite_name}")
        print(f"{'='*60}")
        results = evaluate_suite(
            model, suite_name,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        suite_avg = np.mean(list(results.values()))
        all_suite_results[suite_name] = suite_avg
        print(f"  => {suite_name} average: {suite_avg:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for s, v in all_suite_results.items():
        print(f"  {s:20s} : {v:.3f}")
    overall = np.mean(list(all_suite_results.values()))
    print(f"  {'Overall':20s} : {overall:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()