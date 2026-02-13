#!/usr/bin/env python3
"""Debug script to inspect object naming in LIBERO environments."""

import pathlib
import sys
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import SegmentationRenderEnv

def inspect_task_objects(task_suite_name, task_id=0):
    """Inspect object naming for a specific task."""
    print(f"\n{'='*80}")
    print(f"Inspecting {task_suite_name}, task {task_id}")
    print(f"{'='*80}\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    print(f"Task description: {task.language}")
    print(f"Problem folder: {task.problem_folder}")
    print(f"BDDL file: {task.bddl_file}\n")

    # Create environment
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_segmentations="instance",
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)

    # Reset and get initial state
    initial_states = task_suite.get_task_init_states(task_id)
    obs = env.reset()
    if initial_states:
        obs = env.set_init_state(initial_states[0])

    # Inspect objects
    print(f"obj_of_interest: {env.obj_of_interest}")
    print(f"  Type: {type(env.obj_of_interest)}")
    print(f"  Length: {len(env.obj_of_interest) if hasattr(env.obj_of_interest, '__len__') else 'N/A'}\n")

    print(f"instance_to_id keys: {list(env.instance_to_id.keys())}")
    print(f"  Type: {type(env.instance_to_id)}")
    print(f"  Length: {len(env.instance_to_id)}\n")

    # Check for matches
    matching = []
    missing = []
    for obj_name in env.obj_of_interest:
        if obj_name in env.instance_to_id:
            matching.append(obj_name)
        else:
            missing.append(obj_name)

    print(f"Matching objects (in both obj_of_interest and instance_to_id):")
    for obj in matching:
        print(f"  - {obj} -> ID {env.instance_to_id[obj]}")

    print(f"\nMissing objects (in obj_of_interest but NOT in instance_to_id):")
    for obj in missing:
        print(f"  - {obj}")

    # Try to find potential matches using fuzzy matching
    if missing:
        print(f"\nPotential matches (checking if obj_of_interest names are substrings):")
        for missing_obj in missing:
            potential = [k for k in env.instance_to_id.keys() if missing_obj.lower() in k.lower() or k.lower() in missing_obj.lower()]
            if potential:
                print(f"  {missing_obj} might match:")
                for p in potential:
                    print(f"    - {p} -> ID {env.instance_to_id[p]}")

    # Show all segmentation IDs present in the environment
    seg_key = None
    for key in obs.keys():
        if "segmentation" in key.lower() and "agentview" in key.lower():
            seg_key = key
            break

    if seg_key:
        seg_mask = obs[seg_key]
        unique_ids = set(seg_mask.flatten())
        print(f"\nUnique segmentation IDs in {seg_key}: {sorted(unique_ids)}")

    env.close()
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Compare different task suites
    for suite in ["libero_10", "libero_goal"]:
        try:
            inspect_task_objects(suite, task_id=0)
        except Exception as e:
            print(f"Error inspecting {suite}: {e}")
            import traceback
            traceback.print_exc()
