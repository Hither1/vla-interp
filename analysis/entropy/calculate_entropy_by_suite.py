#!/usr/bin/env python3
"""
Calculate average action entropy for each LIBERO task suite,
separated by success/failure status.

Usage:
    python calculate_entropy_by_suite.py --action_data_dir <path_to_action_data> [--output output.json]
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import gaussian_kde
import argparse
from typing import Dict, List, Tuple


# Map suite numbers to suite names (adjust based on your setup)
SUITE_MAPPING = {
    "10": "LIBERO-10",  # Generic small suite or mixed Spatial/Object/Goal/Long
    "90": "LIBERO-90",
    "spatial": "LIBERO-Spatial",
    "object": "LIBERO-Object",
    "goal": "LIBERO-Goal",
    "long": "LIBERO-Long",
}


def extract_actions_from_file(file_path: Path) -> Tuple[np.ndarray, bool, str, str, int]:
    """
    Extract actions, success status, task description, and suite from a JSON file.

    Returns:
        actions: numpy array of shape (num_timesteps, action_dim)
        success: boolean indicating success/failure
        task_description: task description string
        suite_id: suite identifier (e.g., "90", "spatial", etc.)
        task_id: task ID from the JSON
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract actions
    actions_list = []
    for action_dict in data['actions']:
        actions_list.append(action_dict['action'])
    actions = np.array(actions_list)

    # Extract success status
    success = data.get('success', False)

    # Extract task description
    task_description = data.get('task_description', '')

    # Extract task_id
    task_id = data.get('task_id', -1)

    # Extract suite ID from filename or infer from task_id
    filename = file_path.stem
    suite_id = None

    # First, try to extract from filename: actions_libero_90_...
    if 'libero_' in filename:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == 'libero' and i + 1 < len(parts):
                suite_id = parts[i + 1]
                break

    # If not found in filename, infer from task_id patterns
    # LIBERO-90 typically has task_ids 10-89 or higher
    # LIBERO-Spatial/Object/Goal have task_ids 0-9
    # LIBERO-Long has task_ids 0-3
    if suite_id is None:
        if task_id >= 10:
            suite_id = "90"  # Likely LIBERO-90
        else:
            # For task_id 0-9, we can't distinguish between suites without more info
            # Group them together or use task characteristics
            suite_id = "10"  # Group as LIBERO-10 (generic small suite)

    return actions, success, task_description, suite_id, task_id


def calculate_action_entropy_kde(actions: np.ndarray, bandwidth_factor: float = None) -> Dict:
    """
    Calculate action entropy using KDE (Kernel Density Estimation).

    Args:
        actions: numpy array of shape (num_timesteps, action_dim)
        bandwidth_factor: optional bandwidth factor for KDE (if None, use Scott's rule)

    Returns:
        Dictionary with entropy statistics
    """
    if len(actions) == 0:
        return {
            'action_entropy_kde': None,
            'mean_log_density': None,
            'std_log_density': None,
            'num_samples': 0,
        }

    # Transpose to (action_dim, num_timesteps) for KDE
    actions_T = actions.T

    try:
        # Create KDE
        if bandwidth_factor is not None:
            kde = gaussian_kde(actions_T, bw_method=bandwidth_factor)
        else:
            kde = gaussian_kde(actions_T)  # Uses Scott's rule by default

        # Evaluate log density at each action
        log_densities = kde.logpdf(actions_T)

        # Entropy = -E[log p(x)] = -mean(log_densities)
        mean_log_density = np.mean(log_densities)
        action_entropy = -mean_log_density

        return {
            'action_entropy_kde': float(action_entropy),
            'mean_log_density': float(mean_log_density),
            'std_log_density': float(np.std(log_densities)),
            'kde_bandwidth_factor': float(kde.factor) if hasattr(kde, 'factor') else None,
            'num_samples': len(actions),
        }
    except Exception as e:
        print(f"Warning: KDE calculation failed: {e}")
        return {
            'action_entropy_kde': None,
            'mean_log_density': None,
            'std_log_density': None,
            'num_samples': len(actions),
            'error': str(e),
        }


def calculate_action_entropy_simple(actions: np.ndarray) -> Dict:
    """
    Calculate simple action entropy using discrete binning.
    Alternative to KDE for comparison.

    Args:
        actions: numpy array of shape (num_timesteps, action_dim)

    Returns:
        Dictionary with entropy statistics
    """
    if len(actions) == 0:
        return {'action_entropy_bins': None, 'num_samples': 0}

    # Simple approach: calculate entropy per action dimension, then average
    entropies = []
    for dim in range(actions.shape[1]):
        # Use histogram to estimate density
        counts, _ = np.histogram(actions[:, dim], bins=50, density=False)
        # Normalize to get probabilities
        probs = counts / counts.sum()
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        # Calculate entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

    return {
        'action_entropy_bins': float(np.mean(entropies)),
        'action_entropy_per_dim': [float(e) for e in entropies],
        'num_samples': len(actions),
    }


def process_action_data_directory(action_data_dir: Path, use_simple_entropy: bool = False) -> Dict:
    """
    Process all action JSON files in the directory and calculate entropy statistics.

    Args:
        action_data_dir: Path to directory containing action JSON files
        use_simple_entropy: If True, use simple binning entropy instead of KDE

    Returns:
        Dictionary with results organized by suite, task, and success/failure
    """
    # Structure: {suite_id: {task_description: {'success': [], 'failure': []}}}
    results = defaultdict(lambda: defaultdict(lambda: {'success': [], 'failure': []}))

    # Find all actions_*.json files
    action_files = list(action_data_dir.glob("actions_*.json"))
    print(f"Found {len(action_files)} action files")

    for i, file_path in enumerate(action_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(action_files)}: {file_path.name}")

        try:
            actions, success, task_description, suite_id, task_id = extract_actions_from_file(file_path)

            # Calculate entropy
            if use_simple_entropy:
                entropy_stats = calculate_action_entropy_simple(actions)
            else:
                entropy_stats = calculate_action_entropy_kde(actions)

            # Store result
            status = 'success' if success else 'failure'
            results[suite_id][task_description][status].append({
                'file': file_path.name,
                'task_id': task_id,
                'entropy_stats': entropy_stats,
                'num_actions': len(actions),
            })

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    return results


def aggregate_results(results: Dict) -> Dict:
    """
    Aggregate entropy results by suite and success/failure status.

    Args:
        results: Raw results from process_action_data_directory

    Returns:
        Aggregated statistics
    """
    aggregated = {}

    for suite_id, tasks in results.items():
        suite_name = SUITE_MAPPING.get(suite_id, f"LIBERO-{suite_id}")

        # Collect entropy values for success and failure across all tasks
        success_entropies = []
        failure_entropies = []

        task_stats = {}

        for task_description, status_data in tasks.items():
            # Calculate average entropy for this task
            task_success_entropies = [
                trial['entropy_stats']['action_entropy_kde']
                for trial in status_data['success']
                if trial['entropy_stats']['action_entropy_kde'] is not None
            ]
            task_failure_entropies = [
                trial['entropy_stats']['action_entropy_kde']
                for trial in status_data['failure']
                if trial['entropy_stats']['action_entropy_kde'] is not None
            ]

            success_entropies.extend(task_success_entropies)
            failure_entropies.extend(task_failure_entropies)

            task_stats[task_description] = {
                'num_success_trials': len(status_data['success']),
                'num_failure_trials': len(status_data['failure']),
                'success_entropy_mean': float(np.mean(task_success_entropies)) if task_success_entropies else None,
                'success_entropy_std': float(np.std(task_success_entropies)) if task_success_entropies else None,
                'failure_entropy_mean': float(np.mean(task_failure_entropies)) if task_failure_entropies else None,
                'failure_entropy_std': float(np.std(task_failure_entropies)) if task_failure_entropies else None,
            }

        # Calculate suite-level statistics
        aggregated[suite_name] = {
            'num_tasks': len(tasks),
            'success': {
                'num_trials': len(success_entropies),
                'entropy_mean': float(np.mean(success_entropies)) if success_entropies else None,
                'entropy_std': float(np.std(success_entropies)) if success_entropies else None,
                'entropy_min': float(np.min(success_entropies)) if success_entropies else None,
                'entropy_max': float(np.max(success_entropies)) if success_entropies else None,
            },
            'failure': {
                'num_trials': len(failure_entropies),
                'entropy_mean': float(np.mean(failure_entropies)) if failure_entropies else None,
                'entropy_std': float(np.std(failure_entropies)) if failure_entropies else None,
                'entropy_min': float(np.min(failure_entropies)) if failure_entropies else None,
                'entropy_max': float(np.max(failure_entropies)) if failure_entropies else None,
            },
            'task_details': task_stats,
        }

    return aggregated


def print_summary(aggregated: Dict):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("ENTROPY SUMMARY BY SUITE AND SUCCESS/FAILURE STATUS")
    print("="*80)

    for suite_name, stats in aggregated.items():
        print(f"\n{suite_name}:")
        print(f"  Total tasks: {stats['num_tasks']}")

        print(f"\n  SUCCESS trials ({stats['success']['num_trials']} total):")
        if stats['success']['entropy_mean'] is not None:
            print(f"    Mean entropy: {stats['success']['entropy_mean']:.4f} ± {stats['success']['entropy_std']:.4f}")
            print(f"    Range: [{stats['success']['entropy_min']:.4f}, {stats['success']['entropy_max']:.4f}]")
        else:
            print("    No successful trials")

        print(f"\n  FAILURE trials ({stats['failure']['num_trials']} total):")
        if stats['failure']['entropy_mean'] is not None:
            print(f"    Mean entropy: {stats['failure']['entropy_mean']:.4f} ± {stats['failure']['entropy_std']:.4f}")
            print(f"    Range: [{stats['failure']['entropy_min']:.4f}, {stats['failure']['entropy_max']:.4f}]")
        else:
            print("    No failed trials")

        # Print difference if both exist
        if stats['success']['entropy_mean'] is not None and stats['failure']['entropy_mean'] is not None:
            diff = stats['failure']['entropy_mean'] - stats['success']['entropy_mean']
            print(f"\n  Entropy difference (failure - success): {diff:.4f}")
            if diff > 0:
                print(f"    → Failed trials have {diff:.4f} higher entropy (more random)")
            else:
                print(f"    → Failed trials have {abs(diff):.4f} lower entropy (less random)")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average action entropy for LIBERO tasks by suite and success/failure"
    )
    parser.add_argument(
        "--action_data_dir",
        type=str,
        default="third_party/cosmos-policy/cosmos_policy/experiments/robot/libero/logs/action_data",
        help="Path to directory containing action JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_by_suite_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--use_simple_entropy",
        action="store_true",
        help="Use simple binning entropy instead of KDE"
    )

    args = parser.parse_args()

    action_data_dir = Path(args.action_data_dir)
    if not action_data_dir.exists():
        print(f"Error: Directory not found: {action_data_dir}")
        return

    print(f"Processing action data from: {action_data_dir}")
    print(f"Using {'simple binning' if args.use_simple_entropy else 'KDE'} for entropy calculation")

    # Process all files
    results = process_action_data_directory(action_data_dir, args.use_simple_entropy)

    # Aggregate results
    aggregated = aggregate_results(results)

    # Print summary
    print_summary(aggregated)

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    # Also save raw results for detailed analysis
    raw_output_path = output_path.parent / f"{output_path.stem}_raw.json"
    # Convert defaultdict to regular dict for JSON serialization
    results_serializable = {
        suite_id: {
            task: {
                status: trials
                for status, trials in status_data.items()
            }
            for task, status_data in tasks.items()
        }
        for suite_id, tasks in results.items()
    }
    with open(raw_output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Raw results saved to: {raw_output_path}")


if __name__ == "__main__":
    main()
