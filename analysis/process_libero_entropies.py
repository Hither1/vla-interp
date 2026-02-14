#!/usr/bin/env python3
"""
Process LIBERO data to extract action entropies for success and failure cases.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def process_task_suite(suite_path):
    """Process all JSON files in a task suite."""
    results = []

    videos_dir = suite_path / "videos"
    if not videos_dir.exists():
        print(f"No videos directory in {suite_path}")
        return results

    json_files = list(videos_dir.glob("*.json"))
    print(f"Processing {len(json_files)} files in {suite_path.name}...")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract metadata
            task_id = data.get('task_id')
            trial_id = data.get('trial_id')
            task_description = data.get('task_description', '')
            success = data.get('success', False)

            # Extract action entropy group
            action_entropy_group = data.get('action_entropy_group', {})

            # Extract the relevant entropy based on success/failure
            if success:
                entropy_data = action_entropy_group.get('success', {})
            else:
                entropy_data = action_entropy_group.get('failure', {})

            # Get action_entropy_kde value
            action_entropy_kde = entropy_data.get('action_entropy_kde', None)
            mean_log_density = entropy_data.get('mean_log_density', None)
            std_log_density = entropy_data.get('std_log_density', None)

            # Also get the 'all' group stats for reference
            all_entropy_data = action_entropy_group.get('all', {})
            all_action_entropy_kde = all_entropy_data.get('action_entropy_kde', None)

            # Extract per-action entropies if available
            actions = data.get('actions', [])
            num_actions = len(actions)

            # Calculate trajectory-level metrics
            action_delta_norms = [a.get('action_delta_norm', 0) for a in actions if 'action_delta_norm' in a]
            pos_delta_norms = [a.get('pos_delta_norm', 0) for a in actions if 'pos_delta_norm' in a]
            rot_delta_norms = [a.get('rot_delta_norm', 0) for a in actions if 'rot_delta_norm' in a]

            result = {
                'suite': suite_path.name,
                'task_id': task_id,
                'trial_id': trial_id,
                'task_description': task_description,
                'success': success,
                'action_entropy_kde': action_entropy_kde,
                'mean_log_density': mean_log_density,
                'std_log_density': std_log_density,
                'all_action_entropy_kde': all_action_entropy_kde,
                'num_actions': num_actions,
                'mean_action_delta_norm': np.mean(action_delta_norms) if action_delta_norms else None,
                'std_action_delta_norm': np.std(action_delta_norms) if action_delta_norms else None,
                'mean_pos_delta_norm': np.mean(pos_delta_norms) if pos_delta_norms else None,
                'mean_rot_delta_norm': np.mean(rot_delta_norms) if rot_delta_norms else None,
                'smoothness': data.get('smoothness'),
                'mmd': data.get('mmd'),
                'replan_consistency': data.get('replan_consistency'),
                'file': json_file.name
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def main():
    # Base path
    data_dir = Path("data/libero")

    # Task suites to process
    task_suites = ['10', 'goal', 'libero_90', 'object', 'spatial']

    all_results = []

    for suite_name in task_suites:
        suite_path = data_dir / suite_name
        if suite_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing suite: {suite_name}")
            print('='*60)
            results = process_task_suite(suite_path)
            all_results.extend(results)
        else:
            print(f"Suite {suite_name} not found at {suite_path}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    output_file = "libero_entropy_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved raw results to {output_file}")

    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Overall stats
    print(f"\nTotal trajectories: {len(df)}")
    print(f"Success cases: {df['success'].sum()} ({df['success'].mean()*100:.1f}%)")
    print(f"Failure cases: {(~df['success']).sum()} ({(~df['success']).mean()*100:.1f}%)")

    # Stats by suite
    print("\n" + "-"*60)
    print("By Task Suite:")
    print("-"*60)
    suite_stats = df.groupby('suite').agg({
        'success': ['count', 'sum', 'mean'],
        'action_entropy_kde': ['mean', 'std'],
        'num_actions': ['mean', 'std']
    }).round(3)
    print(suite_stats)

    # Entropy comparison: Success vs Failure
    print("\n" + "-"*60)
    print("Action Entropy KDE: Success vs Failure")
    print("-"*60)

    for suite in task_suites:
        suite_df = df[df['suite'] == suite]
        if len(suite_df) == 0:
            continue

        success_df = suite_df[suite_df['success'] == True]
        failure_df = suite_df[suite_df['success'] == False]

        print(f"\n{suite}:")
        if len(success_df) > 0:
            success_entropy = success_df['action_entropy_kde'].dropna()
            print(f"  Success (n={len(success_df)}): "
                  f"mean={success_entropy.mean():.4f}, "
                  f"std={success_entropy.std():.4f}, "
                  f"median={success_entropy.median():.4f}")

        if len(failure_df) > 0:
            failure_entropy = failure_df['action_entropy_kde'].dropna()
            print(f"  Failure (n={len(failure_df)}): "
                  f"mean={failure_entropy.mean():.4f}, "
                  f"std={failure_entropy.std():.4f}, "
                  f"median={failure_entropy.median():.4f}")

        # Statistical test if both groups exist
        if len(success_df) > 0 and len(failure_df) > 0:
            from scipy import stats
            success_vals = success_df['action_entropy_kde'].dropna()
            failure_vals = failure_df['action_entropy_kde'].dropna()
            if len(success_vals) > 0 and len(failure_vals) > 0:
                t_stat, p_val = stats.ttest_ind(success_vals, failure_vals)
                print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Overall entropy comparison
    print("\n" + "-"*60)
    print("Overall Action Entropy KDE Comparison")
    print("-"*60)

    success_all = df[df['success'] == True]['action_entropy_kde'].dropna()
    failure_all = df[df['success'] == False]['action_entropy_kde'].dropna()

    print(f"\nAll Success Cases (n={len(success_all)}):")
    print(f"  Mean: {success_all.mean():.4f}")
    print(f"  Std:  {success_all.std():.4f}")
    print(f"  Median: {success_all.median():.4f}")
    print(f"  Min: {success_all.min():.4f}")
    print(f"  Max: {success_all.max():.4f}")

    print(f"\nAll Failure Cases (n={len(failure_all)}):")
    if len(failure_all) > 0:
        print(f"  Mean: {failure_all.mean():.4f}")
        print(f"  Std:  {failure_all.std():.4f}")
        print(f"  Median: {failure_all.median():.4f}")
        print(f"  Min: {failure_all.min():.4f}")
        print(f"  Max: {failure_all.max():.4f}")
    else:
        print("  No failure cases with entropy data available")

    if len(success_all) > 0 and len(failure_all) > 1:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(success_all, failure_all)
        print(f"\nt-test: t={t_stat:.3f}, p={p_val:.6f}")
        u_stat, u_p_val = stats.mannwhitneyu(success_all, failure_all, alternative='two-sided')
        print(f"Mann-Whitney U test: U={u_stat:.1f}, p={u_p_val:.6f}")

    # Save summary report
    summary_file = "libero_entropy_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LIBERO Action Entropy KDE Analysis - Summary Report\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total trajectories: {len(df)}\n")
        f.write(f"Success cases: {df['success'].sum()} ({df['success'].mean()*100:.1f}%)\n")
        f.write(f"Failure cases: {(~df['success']).sum()} ({(~df['success']).mean()*100:.1f}%)\n\n")

        f.write("\nSuccess Cases - Action Entropy KDE:\n")
        f.write(f"  Mean ± Std: {success_all.mean():.4f} ± {success_all.std():.4f}\n")
        f.write(f"  Median: {success_all.median():.4f}\n")
        f.write(f"  Range: [{success_all.min():.4f}, {success_all.max():.4f}]\n\n")

        f.write("Failure Cases - Action Entropy KDE:\n")
        if len(failure_all) > 0:
            f.write(f"  Mean ± Std: {failure_all.mean():.4f} ± {failure_all.std():.4f}\n")
            f.write(f"  Median: {failure_all.median():.4f}\n")
            f.write(f"  Range: [{failure_all.min():.4f}, {failure_all.max():.4f}]\n\n")
        else:
            f.write("  No failure cases with entropy data available\n\n")

        if len(success_all) > 0 and len(failure_all) > 1:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(success_all, failure_all)
            f.write(f"Statistical Test (t-test):\n")
            f.write(f"  t-statistic: {t_stat:.3f}\n")
            f.write(f"  p-value: {p_val:.6f}\n")
            f.write(f"  Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}\n")

    print(f"\n✓ Saved summary report to {summary_file}")

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
