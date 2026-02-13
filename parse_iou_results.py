#!/usr/bin/env python3
"""Parse IOU results JSON files and compute average combined_iou across trajectories."""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np


def parse_iou_results(json_path: str, layer: str = None) -> Dict[str, Dict[str, float]]:
    """Parse IOU results and compute statistics.

    Args:
        json_path: Path to the iou_results JSON file
        layer: Which layer to analyze (default: None, analyzes all layers)

    Returns:
        Dictionary with layer names as keys and statistics dictionaries as values
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Collect combined_iou means for each layer
    layer_ious = defaultdict(list)

    for trajectory in data:
        if "summary" in trajectory:
            for layer_name, layer_data in trajectory["summary"].items():
                # If specific layer requested, only process that one
                if layer is not None and layer_name != layer:
                    continue

                if "combined_iou" in layer_data:
                    # Get the mean combined_iou for this trajectory
                    layer_ious[layer_name].append(layer_data["combined_iou"]["mean"])

    # Calculate statistics for each layer
    results = {}
    for layer_name, ious in layer_ious.items():
        if ious:
            results[layer_name] = {
                "num_trajectories": len(ious),
                "mean_combined_iou": float(np.mean(ious)),
                "std_combined_iou": float(np.std(ious)),
                "min_combined_iou": float(np.min(ious)),
                "max_combined_iou": float(np.max(ious)),
            }

    return results


def parse_all_files(directory: str = "outputs_iou_cosmos/test") -> Dict[str, Dict]:
    """Parse all iou_results JSON files in a directory.

    Args:
        directory: Directory containing JSON files

    Returns:
        Dictionary mapping filenames to their layer statistics
    """
    results_dir = Path(directory)
    json_files = sorted(results_dir.glob("iou_results*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return {}

    all_results = {}
    for json_file in json_files:
        all_results[json_file.name] = parse_iou_results(str(json_file))

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Parse IOU results and compute average combined_iou"
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        help="Path to iou_results JSON file (if not provided, processes all files in outputs_iou_cosmos/test)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="outputs_iou_cosmos/test",
        help="Directory containing JSON files (default: outputs_iou_cosmos/test)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer to analyze (default: all layers)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSON files in the directory",
    )

    args = parser.parse_args()

    # If no specific file provided or --all flag, process all files
    if args.json_file is None or args.all:
        all_results = parse_all_files(args.directory)

        if not all_results:
            return 1

        # Print results for all files
        for filename, layer_results in all_results.items():
            print(f"\n{'='*80}")
            print(f"File: {filename}")
            print(f"{'='*80}")

            if not layer_results:
                print("  No IoU data found in this file.")
                continue

            # Print per-layer statistics
            for layer_name in sorted(layer_results.keys()):
                stats = layer_results[layer_name]
                print(f"\n{layer_name}:")
                print(f"  Number of trajectories: {stats['num_trajectories']}")
                print(f"  Mean combined_iou:      {stats['mean_combined_iou']:.6f}")
                print(f"  Std combined_iou:       {stats['std_combined_iou']:.6f}")
                print(f"  Min combined_iou:       {stats['min_combined_iou']:.6f}")
                print(f"  Max combined_iou:       {stats['max_combined_iou']:.6f}")

            # Calculate overall average across all layers
            all_means = [stats['mean_combined_iou'] for stats in layer_results.values()]
            if all_means:
                print(f"\n{'─'*80}")
                print(f"OVERALL (average across all layers):")
                print(f"  Average combined_iou:   {np.mean(all_means):.6f}")
                print(f"  Std dev:                {np.std(all_means):.6f}")
                print(f"  Min:                    {np.min(all_means):.6f}")
                print(f"  Max:                    {np.max(all_means):.6f}")

    else:
        # Process single file
        if not Path(args.json_file).exists():
            print(f"Error: File not found: {args.json_file}")
            return 1

        results = parse_iou_results(args.json_file, args.layer)

        if not results:
            print("No IoU data found in this file.")
            return 1

        # If specific layer requested, show only that layer
        if args.layer and args.layer in results:
            stats = results[args.layer]
            print(f"File: {args.json_file}")
            print(f"Layer: {args.layer}")
            print("-" * 50)
            print(f"Number of trajectories: {stats['num_trajectories']}")
            print(f"Mean combined_iou: {stats['mean_combined_iou']:.6f}")
            print(f"Std combined_iou: {stats['std_combined_iou']:.6f}")
            print(f"Min combined_iou: {stats['min_combined_iou']:.6f}")
            print(f"Max combined_iou: {stats['max_combined_iou']:.6f}")
        else:
            # Show all layers
            print(f"File: {args.json_file}")
            print("-" * 80)
            for layer_name in sorted(results.keys()):
                stats = results[layer_name]
                print(f"\n{layer_name}:")
                print(f"  Number of trajectories: {stats['num_trajectories']}")
                print(f"  Mean combined_iou:      {stats['mean_combined_iou']:.6f}")
                print(f"  Std combined_iou:       {stats['std_combined_iou']:.6f}")
                print(f"  Min combined_iou:       {stats['min_combined_iou']:.6f}")
                print(f"  Max combined_iou:       {stats['max_combined_iou']:.6f}")

    return 0


if __name__ == "__main__":
    exit(main())
