#!/usr/bin/env python3
"""Analyze language-corruption rerouting across models.

This script joins:
- per-episode rollout/eval JSONs that contain success and prompt metadata
- optional attention ratio result JSONs
- optional attention IoU result JSONs

It then compares clean instructions (`prompt_mode=original`) against corrupted
language conditions such as `empty`, `random`, `synonym`, `shuffle`, and
`opposite`, producing:
- episode-level delta table
- grouped summary table
- plots for delta ratio / delta IoU / delta success
- a temporal early-mid-late delta view

Typical usage:
  python analysis/attention/analyze_language_rerouting.py \
    --model-run pi0.5=/path/to/pi05_outputs \
    --model-run DreamZero=/path/to/dreamzero_outputs \
    --output-dir results/language_rerouting
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

DEFAULT_PERTURBATIONS = ["empty", "random", "synonym", "shuffle", "opposite"]
PROMPT_RE = re.compile(r"prompt_([a-z0-9_]+)")


@dataclass(frozen=True)
class EpisodeKey:
    model: str
    suite: str
    prompt_mode: str
    task_id: int
    episode_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Model name and root directory to scan recursively.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=DEFAULT_PERTURBATIONS,
        help="Prompt perturbations to compare against prompt_mode=original.",
    )
    parser.add_argument(
        "--gri-lambda",
        type=float,
        default=1.0,
        help="Lambda used in GRI = delta_ratio - lambda * abs(delta_iou).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/markdown summaries only.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return float(np.mean(clean))


def _std(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return float(np.std(clean))


def _sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def infer_prompt_mode_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = PROMPT_RE.search(part.lower())
        if match:
            return match.group(1)
    return None


def infer_suite_from_filename(path: Path) -> str | None:
    stem = path.stem
    for prefix in ("attention_ratio_results_", "iou_results_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return None


def collect_summary_suites(root: Path) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    for summary_path in root.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "task_suite_name" in data:
            mapping[summary_path.parent] = str(data["task_suite_name"])
    return mapping


def find_suite_for_path(path: Path, summary_suites: dict[Path, str]) -> str | None:
    for parent in [path.parent, *path.parents]:
        if parent in summary_suites:
            return summary_suites[parent]
    return None


def choose_layer_dict(data: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(data, dict) or not data:
        return None
    if "layers_avg" in data and isinstance(data["layers_avg"], dict):
        return data["layers_avg"]
    first_key = sorted(data.keys())[0]
    return data.get(first_key) if isinstance(data.get(first_key), dict) else None


def choose_step_series(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not data:
        return []
    if "layers_avg" in data and isinstance(data["layers_avg"], list):
        return [x for x in data["layers_avg"] if isinstance(x, dict)]
    first_key = sorted(data.keys())[0]
    value = data.get(first_key)
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def summarize_series_by_phase(values: list[float | None]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"early": None, "mid": None, "late": None, "slope": None}
    arr = np.asarray(clean, dtype=np.float64)
    chunks = np.array_split(arr, 3)
    early = float(np.mean(chunks[0])) if len(chunks[0]) else None
    mid = float(np.mean(chunks[1])) if len(chunks[1]) else None
    late = float(np.mean(chunks[2])) if len(chunks[2]) else None
    slope = None
    if early is not None and late is not None:
        slope = late - early
    return {"early": early, "mid": mid, "late": late, "slope": slope}


def extract_ratio_from_summary(summary: dict[str, Any]) -> float | None:
    layer = choose_layer_dict(summary)
    if not layer:
        return None
    ratio_stats = layer.get("visual_linguistic_ratio", {})
    if isinstance(ratio_stats, dict):
        return _safe_float(ratio_stats.get("mean"))
    return None


def extract_iou_from_summary(summary: dict[str, Any]) -> float | None:
    layer = choose_layer_dict(summary)
    if not layer:
        return None

    for key in ("combined_iou", "iou", "iou_iou"):
        value = layer.get(key)
        if isinstance(value, dict):
            mean = _safe_float(value.get("mean"))
            if mean is not None:
                return mean

    combined = layer.get("combined", {})
    if isinstance(combined, dict):
        pct90 = combined.get("percentile_90", {})
        if isinstance(pct90, dict):
            value = _safe_float(pct90.get("iou"))
            if value is not None:
                return value
    return None


def extract_ratio_series(step_container: dict[str, Any]) -> dict[str, float | None]:
    steps = choose_step_series(step_container)
    values = [_safe_float(step.get("visual_linguistic_ratio")) for step in steps]
    return summarize_series_by_phase(values)


def extract_iou_series(step_container: dict[str, Any]) -> dict[str, float | None]:
    steps = choose_step_series(step_container)
    values = []
    for step in steps:
        value = _safe_float(step.get("combined_iou"))
        if value is None:
            value = _safe_float(step.get("iou"))
        if value is None:
            value = _safe_float(step.get("iou_iou"))
        values.append(value)
    return summarize_series_by_phase(values)


def extract_ratio_series_from_attention_steps(attention_steps: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [_safe_float(step.get("visual_linguistic_ratio")) for step in attention_steps if isinstance(step, dict)]
    return summarize_series_by_phase(values)


def extract_iou_series_from_attention_steps(attention_steps: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [_safe_float(step.get("iou_iou")) for step in attention_steps if isinstance(step, dict)]
    return summarize_series_by_phase(values)


def init_episode_record(model: str, suite: str, prompt_mode: str, prompt_used: str | None) -> dict[str, Any]:
    return {
        "model": model,
        "suite": suite,
        "prompt_mode": prompt_mode,
        "prompt_used": prompt_used,
        "task_description": None,
        "success": None,
        "ratio_mean": None,
        "iou_mean": None,
        "ratio_early": None,
        "ratio_mid": None,
        "ratio_late": None,
        "ratio_slope": None,
        "iou_early": None,
        "iou_mid": None,
        "iou_late": None,
        "iou_slope": None,
        "sources": [],
    }


def merge_metric(record: dict[str, Any], field: str, value: Any) -> None:
    if value is None:
        return
    if record.get(field) is None:
        record[field] = value


def merge_record_dicts(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for field, value in incoming.items():
        if field == "sources":
            existing.setdefault("sources", [])
            existing["sources"].extend(value)
            continue
        if existing.get(field) is None and value is not None:
            existing[field] = value
    return existing


def load_episode_eval_json(
    path: Path,
    model: str,
    suite: str,
    records: dict[EpisodeKey, dict[str, Any]],
) -> None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    if "task_id" not in data:
        return
    if "prompt_mode" not in data and "prompt_used" not in data and "success" not in data:
        return

    episode_id = data.get("episode_idx", data.get("trial_id"))
    if episode_id is None:
        return

    prompt_mode = str(data.get("prompt_mode") or infer_prompt_mode_from_path(path) or "original")
    prompt_used = data.get("prompt_used")
    key = EpisodeKey(
        model=model,
        suite=suite,
        prompt_mode=prompt_mode,
        task_id=int(data["task_id"]),
        episode_id=int(episode_id),
    )
    record = records.setdefault(key, init_episode_record(model, suite, prompt_mode, prompt_used))

    record["prompt_mode"] = prompt_mode
    record["prompt_used"] = prompt_used or record.get("prompt_used")
    record["task_description"] = data.get("task_description") or record.get("task_description")
    merge_metric(record, "success", 1.0 if bool(data.get("success")) else 0.0 if "success" in data else None)

    attention_summary = data.get("attention_summary")
    if isinstance(attention_summary, dict):
        merge_metric(record, "ratio_mean", extract_ratio_from_summary({"layers_avg": attention_summary}))
        merge_metric(record, "iou_mean", extract_iou_from_summary({"layers_avg": attention_summary}))

    attention_steps = data.get("attention_steps")
    if isinstance(attention_steps, list) and attention_steps:
        ratio_series = extract_ratio_series_from_attention_steps(attention_steps)
        iou_series = extract_iou_series_from_attention_steps(attention_steps)
        for name, value in ratio_series.items():
            merge_metric(record, f"ratio_{name}", value)
        for name, value in iou_series.items():
            merge_metric(record, f"iou_{name}", value)

    record["sources"].append(str(path))


def normalize_results_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [x for x in data["results"] if isinstance(x, dict)]
    return []


def load_ratio_results_json(
    path: Path,
    model: str,
    suite: str,
    prompt_mode: str,
    records: dict[EpisodeKey, dict[str, Any]],
) -> None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return

    for result in normalize_results_list(data):
        episode_id = result.get("episode_idx", result.get("trial_id"))
        if episode_id is None or "task_id" not in result:
            continue
        key = EpisodeKey(
            model=model,
            suite=suite,
            prompt_mode=prompt_mode,
            task_id=int(result["task_id"]),
            episode_id=int(episode_id),
        )
        record = records.setdefault(key, init_episode_record(model, suite, prompt_mode, None))
        record["prompt_mode"] = prompt_mode
        record["task_description"] = result.get("task_description") or record.get("task_description")
        merge_metric(record, "success", 1.0 if bool(result.get("success")) else 0.0 if "success" in result else None)
        merge_metric(record, "ratio_mean", extract_ratio_from_summary(result.get("summary", {})))
        ratio_series = extract_ratio_series(result.get("per_step_ratios", {}))
        for name, value in ratio_series.items():
            merge_metric(record, f"ratio_{name}", value)
        record["sources"].append(str(path))


def load_iou_results_json(
    path: Path,
    model: str,
    suite: str,
    prompt_mode: str,
    records: dict[EpisodeKey, dict[str, Any]],
) -> None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return

    for result in normalize_results_list(data):
        episode_id = result.get("episode_idx", result.get("trial_id"))
        if episode_id is None or "task_id" not in result:
            continue
        key = EpisodeKey(
            model=model,
            suite=suite,
            prompt_mode=prompt_mode,
            task_id=int(result["task_id"]),
            episode_id=int(episode_id),
        )
        record = records.setdefault(key, init_episode_record(model, suite, prompt_mode, None))
        record["prompt_mode"] = prompt_mode
        record["task_description"] = result.get("task_description") or record.get("task_description")
        merge_metric(record, "success", 1.0 if bool(result.get("success")) else 0.0 if "success" in result else None)
        merge_metric(record, "iou_mean", extract_iou_from_summary(result.get("summary", {})))
        iou_series = extract_iou_series(result.get("per_step_iou", {}))
        for name, value in iou_series.items():
            merge_metric(record, f"iou_{name}", value)
        record["sources"].append(str(path))


def discover_model_records(model: str, root: Path) -> dict[EpisodeKey, dict[str, Any]]:
    summary_suites = collect_summary_suites(root)
    records: dict[EpisodeKey, dict[str, Any]] = {}

    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue

        name = path.name
        prompt_mode = infer_prompt_mode_from_path(path) or "original"
        suite = infer_suite_from_filename(path) or find_suite_for_path(path, summary_suites)

        if name.startswith("attention_ratio_results_") and suite is not None:
            load_ratio_results_json(path, model, suite, prompt_mode, records)
            continue

        if name.startswith("iou_results_") and suite is not None:
            load_iou_results_json(path, model, suite, prompt_mode, records)
            continue

        if name == "summary.json":
            continue

        if suite is not None:
            load_episode_eval_json(path, model, suite, records)

    return records


def compute_deltas(
    records: dict[EpisodeKey, dict[str, Any]],
    perturbations: list[str],
    gri_lambda: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, record in records.items():
        grouped[(key.model, key.suite, key.task_id, key.episode_id)][key.prompt_mode] = record

    rows: list[dict[str, Any]] = []
    for (model, suite, task_id, episode_id), variants in sorted(grouped.items()):
        clean = variants.get("original")
        if clean is None:
            continue
        for perturbation in perturbations:
            corrupt = variants.get(perturbation)
            if corrupt is None:
                continue

            row = {
                "model": model,
                "suite": suite,
                "task_id": task_id,
                "episode_id": episode_id,
                "task_description": clean.get("task_description") or corrupt.get("task_description"),
                "perturbation": perturbation,
                "clean_success": clean.get("success"),
                "corrupt_success": corrupt.get("success"),
                "clean_ratio": clean.get("ratio_mean"),
                "corrupt_ratio": corrupt.get("ratio_mean"),
                "clean_iou": clean.get("iou_mean"),
                "corrupt_iou": corrupt.get("iou_mean"),
            }

            row["delta_success"] = (
                row["corrupt_success"] - row["clean_success"]
                if row["clean_success"] is not None and row["corrupt_success"] is not None
                else None
            )
            row["delta_visual_ratio"] = (
                row["corrupt_ratio"] - row["clean_ratio"]
                if row["clean_ratio"] is not None and row["corrupt_ratio"] is not None
                else None
            )
            row["delta_iou"] = (
                row["corrupt_iou"] - row["clean_iou"]
                if row["clean_iou"] is not None and row["corrupt_iou"] is not None
                else None
            )
            row["vci"] = row["delta_visual_ratio"]
            row["gri"] = (
                row["delta_visual_ratio"] - gri_lambda * abs(row["delta_iou"])
                if row["delta_visual_ratio"] is not None and row["delta_iou"] is not None
                else None
            )

            for phase in ("early", "mid", "late", "slope"):
                clean_ratio = clean.get(f"ratio_{phase}")
                corrupt_ratio = corrupt.get(f"ratio_{phase}")
                clean_iou = clean.get(f"iou_{phase}")
                corrupt_iou = corrupt.get(f"iou_{phase}")

                row[f"delta_ratio_{phase}"] = (
                    corrupt_ratio - clean_ratio
                    if clean_ratio is not None and corrupt_ratio is not None
                    else None
                )
                row[f"delta_iou_{phase}"] = (
                    corrupt_iou - clean_iou
                    if clean_iou is not None and corrupt_iou is not None
                    else None
                )

            rows.append(row)

    return rows


def summarize_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["perturbation"])].append(row)

    summary_rows = []
    for (model, perturbation), items in sorted(grouped.items()):
        summary_rows.append(
            {
                "model": model,
                "perturbation": perturbation,
                "num_pairs": len(items),
                "delta_visual_ratio_mean": _mean([x.get("delta_visual_ratio") for x in items]),
                "delta_visual_ratio_std": _std([x.get("delta_visual_ratio") for x in items]),
                "delta_iou_mean": _mean([x.get("delta_iou") for x in items]),
                "delta_iou_std": _std([x.get("delta_iou") for x in items]),
                "delta_success_mean": _mean([x.get("delta_success") for x in items]),
                "delta_success_std": _std([x.get("delta_success") for x in items]),
                "vci_mean": _mean([x.get("vci") for x in items]),
                "gri_mean": _mean([x.get("gri") for x in items]),
                "delta_ratio_early_mean": _mean([x.get("delta_ratio_early") for x in items]),
                "delta_ratio_mid_mean": _mean([x.get("delta_ratio_mid") for x in items]),
                "delta_ratio_late_mean": _mean([x.get("delta_ratio_late") for x in items]),
                "delta_iou_early_mean": _mean([x.get("delta_iou_early") for x in items]),
                "delta_iou_mid_mean": _mean([x.get("delta_iou_mid") for x in items]),
                "delta_iou_late_mean": _mean([x.get("delta_iou_late") for x in items]),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_metric(
    summary_rows: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if plt is None:
        return
    if not summary_rows:
        return

    perturbations = sorted({row["perturbation"] for row in summary_rows})
    models = sorted({row["model"] for row in summary_rows})
    x = np.arange(len(perturbations))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model in enumerate(models):
        means = []
        errs = []
        for perturbation in perturbations:
            match = next(
                (r for r in summary_rows if r["model"] == model and r["perturbation"] == perturbation),
                None,
            )
            means.append(match.get(metric_key) if match else None)
            std_key = metric_key.replace("_mean", "_std")
            errs.append(match.get(std_key) if match else None)

        offsets = x + (idx - (len(models) - 1) / 2) * width
        valid_means = [0.0 if m is None else m for m in means]
        valid_errs = [0.0 if e is None else e for e in errs]
        ax.bar(offsets, valid_means, width=width, label=model, yerr=valid_errs, capsize=4, alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prompt perturbation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_delta_scatter(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return
    if not summary_rows:
        return

    valid_rows = [
        row
        for row in summary_rows
        if row.get("delta_visual_ratio_mean") is not None and row.get("delta_iou_mean") is not None
    ]
    if not valid_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = None
    for row in valid_rows:
        x = row.get("delta_visual_ratio_mean")
        y = row.get("delta_iou_mean")
        color = row.get("delta_success_mean")
        scatter = ax.scatter(
            x,
            y,
            c=[0.0 if color is None else color],
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.text(x, y, f"{row['model']}:{row['perturbation']}", fontsize=8, ha="left", va="bottom")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Mean delta visual ratio")
    ax.set_ylabel("Mean delta IoU")
    ax.grid(True, alpha=0.25)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Mean delta success")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_deltas(summary_rows: list[dict[str, Any]], metric_prefix: str, output_path: Path) -> None:
    if plt is None:
        return
    if not summary_rows:
        return

    phases = ["early", "mid", "late"]
    perturbations = sorted({row["perturbation"] for row in summary_rows})
    models = sorted({row["model"] for row in summary_rows})

    fig, axes = plt.subplots(1, len(perturbations), figsize=(5 * max(len(perturbations), 1), 4), sharey=True)
    if len(perturbations) == 1:
        axes = [axes]

    for ax, perturbation in zip(axes, perturbations, strict=True):
        for model in models:
            row = next(
                (r for r in summary_rows if r["model"] == model and r["perturbation"] == perturbation),
                None,
            )
            if row is None:
                continue
            ys = [row.get(f"{metric_prefix}_{phase}_mean") for phase in phases]
            if all(v is None for v in ys):
                continue
            ax.plot(phases, [0.0 if v is None else v for v in ys], marker="o", linewidth=2, label=model)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(perturbation)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(metric_prefix.replace("_", " "))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    path: Path,
    summary_rows: list[dict[str, Any]],
    records: dict[EpisodeKey, dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    by_model = defaultdict(int)
    for record in records.values():
        by_model[record["model"]] += 1

    lines = [
        "# Language-Conditioning Comparison",
        "",
        "This summary compares `prompt_mode=original` against language-corrupted prompts.",
        "",
        "## Coverage",
        "",
    ]
    for model, count in sorted(by_model.items()):
        lines.append(f"- {model}: {count} discovered episode records")
    lines.append(f"- paired clean/corrupt comparisons: {len(delta_rows)}")
    lines.append("")
    lines.append("## Group Means")
    lines.append("")
    if not summary_rows:
        lines.append("No matched clean/corrupt pairs were found.")
    else:
        for row in summary_rows:
            lines.append(
                f"- {row['model']} / {row['perturbation']}: "
                f"delta_ratio={row.get('delta_visual_ratio_mean')}, "
                f"delta_iou={row.get('delta_iou_mean')}, "
                f"delta_success={row.get('delta_success_mean')}, "
                f"VCI={row.get('vci_mean')}, GRI={row.get('gri_mean')}"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_model_runs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got: {spec}")
        name, raw_path = spec.split("=", 1)
        parsed.append((name, Path(raw_path).expanduser().resolve()))
    return parsed


def main() -> None:
    args = parse_args()
    model_runs = parse_model_runs(args.model_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[EpisodeKey, dict[str, Any]] = {}
    for model, root in model_runs:
        records = discover_model_records(model, root)
        for key, record in records.items():
            if key in all_records:
                merge_record_dicts(all_records[key], record)
            else:
                all_records[key] = record

    delta_rows = compute_deltas(all_records, args.perturbations, args.gri_lambda)
    summary_rows = summarize_deltas(delta_rows)

    write_csv(args.output_dir / "episode_pairs.csv", delta_rows)
    write_csv(args.output_dir / "summary_by_model_and_perturbation.csv", summary_rows)
    write_markdown_summary(args.output_dir / "summary.md", summary_rows, all_records, delta_rows)

    if not args.skip_plots and plt is None:
        raise SystemExit("matplotlib is not installed; rerun with --skip-plots or install matplotlib.")

    if not args.skip_plots:
        plot_grouped_metric(
            summary_rows,
            metric_key="delta_visual_ratio_mean",
            ylabel="Mean delta visual ratio",
            output_path=args.output_dir / "delta_visual_ratio.png",
        )
        plot_grouped_metric(
            summary_rows,
            metric_key="delta_iou_mean",
            ylabel="Mean delta IoU",
            output_path=args.output_dir / "delta_iou.png",
        )
        plot_grouped_metric(
            summary_rows,
            metric_key="delta_success_mean",
            ylabel="Mean delta success",
            output_path=args.output_dir / "delta_success.png",
        )
        plot_delta_scatter(summary_rows, args.output_dir / "delta_ratio_vs_iou.png")
        plot_temporal_deltas(summary_rows, "delta_ratio", args.output_dir / "temporal_delta_ratio.png")
        plot_temporal_deltas(summary_rows, "delta_iou", args.output_dir / "temporal_delta_iou.png")

    manifest_rows = []
    for key, record in sorted(
        all_records.items(),
        key=lambda item: (item[0].model, item[0].suite, item[0].task_id, item[0].episode_id, item[0].prompt_mode),
    ):
        manifest_rows.append(
            {
                "model": key.model,
                "suite": key.suite,
                "prompt_mode": key.prompt_mode,
                "task_id": key.task_id,
                "episode_id": key.episode_id,
                "prompt_used": record.get("prompt_used"),
                "success": record.get("success"),
                "ratio_mean": record.get("ratio_mean"),
                "iou_mean": record.get("iou_mean"),
                "sources": " | ".join(record.get("sources", [])),
            }
        )
    write_csv(args.output_dir / "discovered_episode_metrics.csv", manifest_rows)


if __name__ == "__main__":
    main()
