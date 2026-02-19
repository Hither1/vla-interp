"""Compute action entropy (KDE) from saved rollout JSON files.

Reads per-episode action JSON files produced by examples/libero/main.py,
groups them by task, and computes per-task and suite-level action entropy
using Gaussian KDE.

For each suite, computes entropy for all episodes, success-only, and
failure-only in a single run. Results are nested under "all", "success",
and "failure" keys in the output JSON.

Can run post-hoc on existing data (no GPU or policy server needed).

Usage:
    # Single suite
    python compute_action_entropy.py --data-dir data/libero/spatial/videos

    # All suites at once
    python compute_action_entropy.py \
        --data-dir data/libero/spatial/videos \
                   data/libero/object/videos \
                   data/libero/goal/videos \
                   data/libero/10/videos

    # Auto-discover all suites under data/libero/
    python compute_action_entropy.py --auto-discover

    # Custom splits (e.g. 90_b, 90_d)
    python compute_action_entropy.py \
        --data-dir data/libero/90_b/videos \
        --output results/entropy/action_entropy_90_b.json
"""

import argparse
import collections
import glob
import json
import logging
import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Map from task suite name to data subdirectory (as created by main.py)
SUITE_TO_DIR = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "10",
    "libero_90": "90",
}


def compute_action_entropy_kde(action_logs: List[List[dict]], action_dim: int = 7) -> dict:
    """Estimate entropy of the action distribution via Gaussian KDE.

    Mirrors the implementation in examples/libero/main.py.

    Args:
        action_logs: List of action logs (one per episode). Each action log is
            a list of dicts with an "action" key.
        action_dim: Number of action dimensions to use (default 7).

    Returns:
        Dict with entropy estimate and diagnostics, or {} if estimation fails.
    """
    all_actions = []
    for action_log in action_logs:
        actions = np.array([entry["action"] for entry in action_log])
        all_actions.append(actions[:, :action_dim])
    actions = np.concatenate(all_actions, axis=0)
    N, D = actions.shape

    if N < D + 1:
        return {}

    try:
        kde = gaussian_kde(actions.T)
        log_densities = kde.logpdf(actions.T)
        entropy = -float(np.mean(log_densities))

        return {
            "action_entropy_kde": entropy,
            "mean_log_density": float(np.mean(log_densities)),
            "std_log_density": float(np.std(log_densities)),
            "kde_bandwidth_factor": float(kde.factor),
            "num_samples": N,
            "num_episodes": len(action_logs),
            "action_dim": D,
        }
    except np.linalg.LinAlgError:
        return {}


def load_action_logs_from_dir(
    data_dir: str, filter_outcome: Optional[str] = None
) -> Dict[str, List[List[dict]]]:
    """Load action logs from a directory of rollout JSONs.

    Groups by task_description. When duplicate files exist for the same trial
    (e.g. both success and failure), keeps the later-modified one unless
    filter_outcome restricts to one kind.

    Args:
        data_dir: Directory containing actions_*.json files.
        filter_outcome: If "success" or "failure", only load matching files.

    Returns:
        Dict mapping task_description -> list of action logs (one per episode).
    """
    pattern = os.path.join(data_dir, "actions_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        log.warning(f"No action JSON files found in {data_dir}")
        return {}

    # Deduplicate: group by (task_description, trial_id), keep latest
    # file for each pair (unless filter_outcome is set)
    file_key_map: Dict[tuple, str] = {}
    for fpath in json_files:
        basename = os.path.basename(fpath)
        # Determine outcome from filename suffix
        if basename.endswith("_success.json"):
            outcome = "success"
        elif basename.endswith("_failure.json"):
            outcome = "failure"
        else:
            outcome = "unknown"

        if filter_outcome and outcome != filter_outcome:
            continue

        # Read just the task_description and trial_id for grouping
        try:
            with open(fpath) as f:
                data = json.load(f)
            key = (data["task_description"], data["trial_id"])
            # If we already have an entry, prefer success over failure
            if key in file_key_map:
                existing = file_key_map[key]
                existing_is_success = existing.endswith("_success.json")
                current_is_success = fpath.endswith("_success.json")
                if current_is_success and not existing_is_success:
                    file_key_map[key] = fpath
                # Otherwise keep existing
            else:
                file_key_map[key] = fpath
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {fpath}: {e}")
            continue

    # Group by task_description
    task_logs: Dict[str, List[List[dict]]] = collections.defaultdict(list)
    for (task_desc, _trial_id), fpath in sorted(file_key_map.items()):
        try:
            with open(fpath) as f:
                data = json.load(f)
            task_logs[task_desc].append(data["actions"])
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {fpath}: {e}")

    return dict(task_logs)


def compute_suite_entropy(
    data_dir: str,
    suite_name: str = "",
    action_dim: int = 7,
    filter_outcome: Optional[str] = None,
) -> dict:
    """Compute per-task and suite-level entropy for one data directory.

    Returns:
        Dict with per-task entropy, suite-level entropy, and summary stats.
    """
    task_logs = load_action_logs_from_dir(data_dir, filter_outcome=filter_outcome)

    if not task_logs:
        log.warning(f"No action data loaded from {data_dir}")
        return {"suite": suite_name, "error": "no_data", "data_dir": data_dir}

    log.info(f"Suite '{suite_name}': {len(task_logs)} tasks, "
             f"{sum(len(v) for v in task_logs.values())} total episodes")

    # Per-task entropy
    per_task = {}
    all_task_entropies = []
    all_action_logs = []  # for suite-level entropy

    for task_desc, action_logs in sorted(task_logs.items()):
        entropy_result = compute_action_entropy_kde(action_logs, action_dim=action_dim)
        per_task[task_desc] = {
            "entropy": entropy_result,
            "num_episodes": len(action_logs),
            "num_action_steps": sum(len(al) for al in action_logs),
        }
        if entropy_result:
            all_task_entropies.append(entropy_result["action_entropy_kde"])
            log.info(
                f"  Task '{task_desc[:60]}': entropy={entropy_result['action_entropy_kde']:.4f} "
                f"(N={entropy_result['num_samples']}, eps={entropy_result['num_episodes']})"
            )
        else:
            log.warning(f"  Task '{task_desc[:60]}': entropy computation failed")

        all_action_logs.extend(action_logs)

    # Suite-level entropy (pooling all actions across all tasks)
    suite_entropy = compute_action_entropy_kde(all_action_logs, action_dim=action_dim)

    # Summary statistics
    summary = {
        "suite": suite_name,
        "data_dir": data_dir,
        "num_tasks": len(task_logs),
        "num_total_episodes": sum(len(v) for v in task_logs.values()),
        "per_task": per_task,
        "suite_level_entropy": suite_entropy,
    }

    if all_task_entropies:
        summary["task_entropy_stats"] = {
            "mean": float(np.mean(all_task_entropies)),
            "std": float(np.std(all_task_entropies)),
            "min": float(np.min(all_task_entropies)),
            "max": float(np.max(all_task_entropies)),
            "median": float(np.median(all_task_entropies)),
        }
        log.info(
            f"  Suite '{suite_name}' task-entropy stats: "
            f"mean={np.mean(all_task_entropies):.4f}, "
            f"std={np.std(all_task_entropies):.4f}, "
            f"range=[{np.min(all_task_entropies):.4f}, {np.max(all_task_entropies):.4f}]"
        )
    if suite_entropy:
        log.info(
            f"  Suite '{suite_name}' pooled entropy: "
            f"{suite_entropy['action_entropy_kde']:.4f} "
            f"(N={suite_entropy['num_samples']})"
        )

    return summary


def load_action_logs_by_outcome(
    data_dir: str,
) -> Dict[str, Dict[str, List[List[dict]]]]:
    """Load action logs grouped by task and outcome in a single pass.

    For each (task_description, trial_id), keeps one file per outcome
    (success / failure). If the same trial has both, success is preferred
    for the 'all' view; each outcome bucket is kept separately.

    Returns:
        Dict mapping task_description -> {"success": [...], "failure": [...]}
        where each value is a list of action logs (one per episode).
    """
    pattern = os.path.join(data_dir, "actions_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        log.warning(f"No action JSON files found in {data_dir}")
        return {}

    # Deduplicate by (task, trial_id): prefer success over failure for the
    # same trial, but track outcome so we can split later.
    file_key_map: Dict[tuple, str] = {}
    for fpath in json_files:
        basename = os.path.basename(fpath)
        if basename.endswith("_success.json"):
            outcome = "success"
        elif basename.endswith("_failure.json"):
            outcome = "failure"
        else:
            outcome = "unknown"
        try:
            with open(fpath) as f:
                data = json.load(f)
            key = (data["task_description"], data["trial_id"], outcome)
            if key not in file_key_map:
                file_key_map[key] = fpath
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {fpath}: {e}")
            continue

    # Group by task and outcome
    task_outcome_logs: Dict[str, Dict[str, List]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for (task_desc, _trial_id, outcome), fpath in sorted(file_key_map.items()):
        try:
            with open(fpath) as f:
                data = json.load(f)
            task_outcome_logs[task_desc][outcome].append(data["actions"])
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {fpath}: {e}")

    return {task: dict(outcomes) for task, outcomes in task_outcome_logs.items()}


def compute_suite_entropy_by_outcome(
    data_dir: str,
    suite_name: str = "",
    action_dim: int = 7,
) -> Dict[str, dict]:
    """Compute per-task entropy for all, success, and failure outcomes.

    Iterates over tasks once, computing KDE entropy per task for each outcome
    simultaneously, then averages per-task entropies over tasks.

    Returns:
        Dict with keys "all", "success", "failure", each containing
        task_entropy_stats (mean/std/...), per_task, suite_level_entropy.
    """
    task_outcome_logs = load_action_logs_by_outcome(data_dir)

    if not task_outcome_logs:
        empty = {"suite": suite_name, "error": "no_data", "data_dir": data_dir}
        return {"all": empty, "success": empty, "failure": empty}

    log.info(f"Suite '{suite_name}': {len(task_outcome_logs)} tasks")

    # outcome label -> which outcome buckets to pool for that split
    outcome_sources = {
        "all":     ["success", "failure", "unknown"],
        "success": ["success"],
        "failure": ["failure"],
    }
    outcome_labels = ["all", "success", "failure"]

    # Accumulators
    per_task:      Dict[str, Dict[str, dict]] = {o: {} for o in outcome_labels}
    task_entropies: Dict[str, list]           = {o: [] for o in outcome_labels}
    all_logs:       Dict[str, list]           = {o: [] for o in outcome_labels}

    for task_desc, outcome_logs in sorted(task_outcome_logs.items()):
        task_row = {}  # outcome -> entropy value, for compact logging
        for outcome_label, include in outcome_sources.items():
            logs = []
            for o in include:
                logs.extend(outcome_logs.get(o, []))
            if not logs:
                continue

            result = compute_action_entropy_kde(logs, action_dim=action_dim)
            per_task[outcome_label][task_desc] = {
                "entropy": result,
                "num_episodes": len(logs),
                "num_action_steps": sum(len(al) for al in logs),
            }
            if result:
                task_entropies[outcome_label].append(result["action_entropy_kde"])
                task_row[outcome_label] = result["action_entropy_kde"]
            all_logs[outcome_label].extend(logs)

        if task_row:
            parts = "  ".join(
                f"{o}={task_row[o]:.4f}" for o in outcome_labels if o in task_row
            )
            log.info(f"  {task_desc[:55]}: {parts}")

    # Compile per-outcome summaries
    results = {}
    for outcome_label in outcome_labels:
        task_ents = task_entropies[outcome_label]
        suite_entropy = (
            compute_action_entropy_kde(all_logs[outcome_label], action_dim=action_dim)
            if all_logs[outcome_label] else {}
        )
        summary: dict = {
            "suite": suite_name,
            "data_dir": data_dir,
            "num_tasks": len(per_task[outcome_label]),
            "num_total_episodes": sum(
                v["num_episodes"] for v in per_task[outcome_label].values()
            ),
            "per_task": per_task[outcome_label],
            "suite_level_entropy": suite_entropy,
        }
        if task_ents:
            summary["task_entropy_stats"] = {
                "mean":   float(np.mean(task_ents)),
                "std":    float(np.std(task_ents)),
                "min":    float(np.min(task_ents)),
                "max":    float(np.max(task_ents)),
                "median": float(np.median(task_ents)),
            }
        results[outcome_label] = summary

    # Summary log
    log.info(f"\nSuite '{suite_name}' task-entropy stats (mean ± std):")
    for outcome_label in outcome_labels:
        stats = results[outcome_label].get("task_entropy_stats", {})
        n = results[outcome_label].get("num_tasks", 0)
        if stats:
            log.info(
                f"  [{outcome_label:<8}] mean={stats['mean']:.4f} ± {stats['std']:.4f}  "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]  n_tasks={n}"
            )

    return results


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="Compute action entropy (KDE) from saved rollout JSON files"
    )
    parser.add_argument(
        "--data-dir", type=str, nargs="+", default=None,
        help="One or more data directories containing actions_*.json files"
    )
    parser.add_argument(
        "--auto-discover", action="store_true",
        help="Auto-discover all suites under data/libero/"
    )
    parser.add_argument(
        "--data-root", type=str, default="data/libero",
        help="Root directory for auto-discovery (default: data/libero)"
    )
    parser.add_argument(
        "--action-dim", type=int, default=7,
        help="Number of action dimensions (default: 7 for LIBERO)"
    )
    parser.add_argument(
        "--output", type=str, default="data/libero/action_entropy_results.json",
        help="Path to save aggregated results"
    )

    args = parser.parse_args()

    # Build list of (suite_name, data_dir) pairs
    suite_dirs = []
    if args.auto_discover:
        for suite_name, subdir in SUITE_TO_DIR.items():
            candidate = os.path.join(args.data_root, subdir, "videos")
            if os.path.isdir(candidate):
                suite_dirs.append((suite_name, candidate))
            else:
                log.info(f"Skipping {suite_name}: {candidate} not found")
    elif args.data_dir:
        for d in args.data_dir:
            # Infer suite name from path
            parts = pathlib.Path(d).parts
            suite_name = "unknown"
            for sn, sd in SUITE_TO_DIR.items():
                if sd in parts:
                    suite_name = sn
                    break
            suite_dirs.append((suite_name, d))
    else:
        parser.error("Provide --data-dir or --auto-discover")

    if not suite_dirs:
        log.error("No data directories found")
        return

    # Compute entropy for each suite (all / success / failure) in one pass
    all_results = {}
    for suite_name, data_dir in suite_dirs:
        log.info(f"\n{'=' * 70}")
        log.info(f"Processing {suite_name}: {data_dir}")
        log.info(f"{'=' * 70}")
        all_results[suite_name] = compute_suite_entropy_by_outcome(
            data_dir=data_dir,
            suite_name=suite_name,
            action_dim=args.action_dim,
        )

    # Print cross-suite comparison
    log.info(f"\n{'=' * 70}")
    log.info("CROSS-SUITE COMPARISON")
    log.info(f"{'=' * 70}")
    log.info(f"{'Suite':<20} {'Outcome':<10} {'Tasks':>5} {'Episodes':>8} {'Mean Task H':>12} {'Pooled H':>10}")
    log.info("-" * 70)
    for suite_name, suite_result in all_results.items():
        for outcome_label, result in suite_result.items():
            n_tasks = result.get("num_tasks", 0)
            n_eps = result.get("num_total_episodes", 0)
            mean_h = result.get("task_entropy_stats", {}).get("mean", float("nan"))
            pooled_h = result.get("suite_level_entropy", {}).get("action_entropy_kde", float("nan"))
            log.info(f"{suite_name:<20} {outcome_label:<10} {n_tasks:>5} {n_eps:>8} {mean_h:>12.4f} {pooled_h:>10.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
