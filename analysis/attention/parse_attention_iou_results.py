#!/usr/bin/env python3
"""
Parse and summarize attention IoU results.

Usage:
  python parse_attention_iou_results.py \
    --results results/attention/iou/pi05/.../iou_results_libero_10.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_path: str) -> Dict[str, List[Dict]]:
    p = Path(results_path)
    if p.is_dir():
        subsets = {}
        for json_file in sorted(p.glob("*.json")):
            name = json_file.stem
            if name.startswith("iou_results_"):
                name = name[len("iou_results_"):]
            with open(json_file) as f:
                data = json.load(f)
            episodes = data.get("results", data) if isinstance(data, dict) else data
            subsets[name] = episodes if isinstance(episodes, list) else [episodes]
        return subsets
    with open(results_path) as f:
        data = json.load(f)
    name = p.stem
    if name.startswith("iou_results_"):
        name = name[len("iou_results_"):]
    episodes = data.get("results", data) if isinstance(data, dict) else data
    return {name: episodes if isinstance(episodes, list) else [episodes]}


def compute_aggregate_stats(episodes: List[Dict]) -> Dict:
    stats = {
        "num_episodes": len(episodes),
        "num_tasks": len(set(e["task_id"] for e in episodes)),
        "success_rate": float(np.mean([e.get("success", False) for e in episodes])),
    }

    all_iou, all_dice, all_mass, all_pointing = [], [], [], []

    for ep in episodes:
        summary = ep.get("summary", {})
        # average across layers
        layer_ious, layer_dice, layer_mass, layer_pointing = [], [], [], []
        for layer_data in summary.values():
            if isinstance(layer_data, dict):
                if "combined_iou" in layer_data:
                    layer_ious.append(layer_data["combined_iou"]["mean"])
                if "combined_dice" in layer_data:
                    layer_dice.append(layer_data["combined_dice"]["mean"])
                if "attention_mass_on_objects" in layer_data:
                    layer_mass.append(layer_data["attention_mass_on_objects"]["mean"])
                if "pointing_accuracy" in layer_data:
                    layer_pointing.append(layer_data["pointing_accuracy"])
        if layer_ious:
            all_iou.append(np.mean(layer_ious))
        if layer_dice:
            all_dice.append(np.mean(layer_dice))
        if layer_mass:
            all_mass.append(np.mean(layer_mass))
        if layer_pointing:
            all_pointing.append(np.mean(layer_pointing))

    def _s(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    if all_iou:
        stats["combined_iou"] = _s(all_iou)
    if all_dice:
        stats["combined_dice"] = _s(all_dice)
    if all_mass:
        stats["attention_mass_on_objects"] = _s(all_mass)
    if all_pointing:
        stats["pointing_accuracy"] = float(np.mean(all_pointing))

    return stats


def print_summary(stats: Dict, label: str):
    print("\n" + "=" * 70)
    print(f"ATTENTION IoU ANALYSIS: {label}")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Tasks:    {stats['num_tasks']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    if "combined_iou" in stats:
        s = stats["combined_iou"]
        print(f"\nCombined IoU (attn mask vs object mask):")
        print(f"  Mean:   {s['mean']:.3f} ± {s['std']:.3f}")
        print(f"  Range:  [{s['min']:.3f}, {s['max']:.3f}]")

    if "combined_dice" in stats:
        s = stats["combined_dice"]
        print(f"\nCombined Dice:")
        print(f"  Mean:   {s['mean']:.3f} ± {s['std']:.3f}")
        print(f"  Range:  [{s['min']:.3f}, {s['max']:.3f}]")

    if "attention_mass_on_objects" in stats:
        s = stats["attention_mass_on_objects"]
        print(f"\nAttention Mass on Objects:")
        print(f"  Mean:   {s['mean']:.3f} ± {s['std']:.3f}")
        print(f"  Range:  [{s['min']:.3f}, {s['max']:.3f}]")

    if "pointing_accuracy" in stats:
        print(f"\nPointing Accuracy: {stats['pointing_accuracy']:.3f}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse attention IoU results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to IoU results JSON file or directory")
    parser.add_argument("--output", type=str, help="Optional output summary text file")
    args = parser.parse_args()

    subsets = load_results(args.results)
    output_lines = []

    for name, episodes in subsets.items():
        stats = compute_aggregate_stats(episodes)
        print_summary(stats, name)

        if args.output:
            output_lines.append(f"IoU Analysis: {name}")
            output_lines.append("=" * 70)
            output_lines.append(f"Episodes: {stats['num_episodes']}")
            output_lines.append(f"Success Rate: {stats['success_rate']:.1%}")
            if "combined_iou" in stats:
                output_lines.append(f"Combined IoU: {stats['combined_iou']['mean']:.3f} ± {stats['combined_iou']['std']:.3f}")
            if "combined_dice" in stats:
                output_lines.append(f"Combined Dice: {stats['combined_dice']['mean']:.3f} ± {stats['combined_dice']['std']:.3f}")
            if "attention_mass_on_objects" in stats:
                output_lines.append(f"Attn Mass on Objects: {stats['attention_mass_on_objects']['mean']:.3f} ± {stats['attention_mass_on_objects']['std']:.3f}")
            if "pointing_accuracy" in stats:
                output_lines.append(f"Pointing Accuracy: {stats['pointing_accuracy']:.3f}")
            output_lines.append("")

    if len(subsets) > 1:
        all_episodes = [e for eps in subsets.values() for e in eps]
        print_summary(compute_aggregate_stats(all_episodes), "COMBINED")

    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
