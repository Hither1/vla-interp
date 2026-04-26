"""Plot DROID generalization experiment success rates from generalization_droid.md.

Produces three radar plot figures:
  fig1_droid_language.png  -- Language perturbation success rates
  fig2_droid_visual.png    -- Visual perturbation success rates
  fig3_droid_policy.png    -- Policy perturbation success rates

Usage:
  python analysis/plot_droid_generalization.py --output-dir analysis/droid_generalization_plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Color / style constants (consistent with plot_generalization.py) ──────────

MODEL_COLORS = {
    "pi0.5":     "#4C72B0",
    "DreamZero": "#17becf",
}
MODELS = ["pi0.5", "DreamZero"]

SUITE_DISPLAY = {
    "DROID-Object":  "DROID-Object",
    "DROID-Spatial": "DROID-Spatial",
    "DROID-Act":     "DROID-Act",
    "DROID-Com":     "DROID-Com",
}
SUITES = [s for s in SUITE_DISPLAY if s != "DROID-Com"]  # DROID-Com commented out for now

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ── Hard-coded success rate data from generalization_droid.md ─────────────────
# Format: data[suite][condition][model] = success_rate (0–100)

LANGUAGE_DATA = {
    "DROID-Object": {
        "original": {"pi0.5": 52, "DreamZero": 85},
        "empty":    {"pi0.5":  3, "DreamZero":  3},
        "shuffle":  {"pi0.5": 28, "DreamZero": 78},
        "random":   {"pi0.5":  2, "DreamZero":  2},
        "synonym":  {"pi0.5": 47, "DreamZero": 82},
    },
    "DROID-Spatial": {
        "original": {"pi0.5": 22, "DreamZero": 73},
        "empty":    {"pi0.5":  0, "DreamZero":  2},
        "shuffle":  {"pi0.5":  8, "DreamZero": 65},
        "random":   {"pi0.5":  0, "DreamZero":  2},
        "synonym":  {"pi0.5": 18, "DreamZero": 70},
    },
    "DROID-Act": {
        "original": {"pi0.5": 48, "DreamZero": 80},
        "empty":    {"pi0.5":  2, "DreamZero":  2},
        "shuffle":  {"pi0.5": 33, "DreamZero": 72},
        "random":   {"pi0.5":  0, "DreamZero":  2},
        "synonym":  {"pi0.5": 52, "DreamZero": 78},
    },
    "DROID-Com": {
        "original": {"pi0.5": 0, "DreamZero": 0},
        "empty":    {"pi0.5": 0, "DreamZero": 0},
        "shuffle":  {"pi0.5": 0, "DreamZero": 0},
        "random":   {"pi0.5": 0, "DreamZero": 0},
        "synonym":  {"pi0.5": 0, "DreamZero": 0},
    },
}
LANGUAGE_CONDITIONS = ["original", "empty", "shuffle", "random", "synonym"]

VISUAL_DATA = {
    "DROID-Object": {
        "original":        {"pi0.5": 52, "DreamZero": 85},
        "rotate 30°":      {"pi0.5":  2, "DreamZero":  8},
        "translate 20%":   {"pi0.5": 45, "DreamZero": 48},
        "rotate+translate":{"pi0.5":  0, "DreamZero":  5},
    },
    "DROID-Spatial": {
        "original":        {"pi0.5": 22, "DreamZero": 73},
        "rotate 30°":      {"pi0.5":  0, "DreamZero":  5},
        "translate 20%":   {"pi0.5": 18, "DreamZero": 38},
        "rotate+translate":{"pi0.5":  0, "DreamZero":  3},
    },
    "DROID-Act": {
        "original":        {"pi0.5": 48, "DreamZero": 80},
        "rotate 30°":      {"pi0.5":  3, "DreamZero":  8},
        "translate 20%":   {"pi0.5": 42, "DreamZero": 45},
        "rotate+translate":{"pi0.5":  2, "DreamZero":  5},
    },
    "DROID-Com": {
        "original":        {"pi0.5": 0, "DreamZero": 0},
        "rotate 30°":      {"pi0.5": 0, "DreamZero": 0},
        "translate 20%":   {"pi0.5": 0, "DreamZero": 0},
        "rotate+translate":{"pi0.5": 0, "DreamZero": 0},
    },
}
VISUAL_CONDITIONS = ["original", "rotate 30°", "translate 20%", "rotate+translate"]

POLICY_DATA = {
    "DROID-Object": {
        "original":          {"pi0.5": 52, "DreamZero": 85},
        "random action 25%": {"pi0.5":  3, "DreamZero": 22},
        "object shift x":    {"pi0.5": 45, "DreamZero": 72},
    },
    "DROID-Spatial": {
        "original":          {"pi0.5": 22, "DreamZero": 73},
        "random action 25%": {"pi0.5":  0, "DreamZero": 18},
        "object shift x":    {"pi0.5": 17, "DreamZero": 62},
    },
    "DROID-Act": {
        "original":          {"pi0.5": 48, "DreamZero": 80},
        "random action 25%": {"pi0.5":  2, "DreamZero": 20},
        "object shift x":    {"pi0.5": 48, "DreamZero": 68},
    },
    "DROID-Com": {
        "original":          {"pi0.5": 0, "DreamZero": 0},
        "random action 25%": {"pi0.5": 0, "DreamZero": 0},
        "object shift x":    {"pi0.5": 0, "DreamZero": 0},
    },
}
POLICY_CONDITIONS = ["original", "random action 25%", "object shift x"]


# ── Plot helper ───────────────────────────────────────────────────────────────

def _plot_radar(ax, suite_data, conditions, title):
    angles = np.linspace(0, 2 * np.pi, len(conditions), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    for model in MODELS:
        values = [suite_data[cond][model] for cond in conditions]
        closed_values = np.concatenate([values, [values[0]]])
        ax.plot(closed_angles, closed_values, color=MODEL_COLORS[model], lw=2.4, label=model)
        ax.fill(closed_angles, closed_values, color=MODEL_COLORS[model], alpha=0.10)

    ax.set_title(title, fontsize=18, pad=18)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(conditions, fontsize=16)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in [20, 40, 60, 80, 100]], fontsize=15)
    ax.set_rlabel_position(0)
    ax.grid(True, ls="--", alpha=0.4)


def _plot_section(data, conditions, title, out_path):
    """Radar plots: one subplot per suite, conditions around the circle, models as traces."""
    n_suites = len(SUITES)
    fig, axes = plt.subplots(
        1, n_suites, figsize=(5.0 * n_suites, 5.0),
        subplot_kw={"projection": "polar"}
    )
    fig.suptitle(title, fontsize=19, fontweight="bold", y=1.01)

    for ax, suite in zip(axes, SUITES):
        _plot_radar(ax, data[suite], conditions, SUITE_DISPLAY[suite])

    handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=3, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="upper right", fontsize=16, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {out_path}, {pdf_path}")
    plt.close(fig)


def _plot_combined(out_dir):
    """Single figure with 3 rows (one per perturbation type), 4 radar plots per suite."""
    sections = [
        (LANGUAGE_DATA, LANGUAGE_CONDITIONS, "Language Perturbation"),
        (VISUAL_DATA,   VISUAL_CONDITIONS,   "Visual Perturbation"),
        (POLICY_DATA,   POLICY_CONDITIONS,   "Policy Perturbation"),
    ]

    n_rows = len(sections)
    n_cols = len(SUITES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.7 * n_cols, 4.7 * n_rows),
        subplot_kw={"projection": "polar"},
    )
    fig.suptitle("DROID Generalization – Success Rates", fontsize=21, fontweight="bold", y=1.01)

    for row, (data, conditions, section_title) in enumerate(sections):
        for col, suite in enumerate(SUITES):
            ax = axes[row][col]
            _plot_radar(ax, data[suite], conditions, SUITE_DISPLAY[suite] if row == 0 else "")
            if col == 0:
                ax.text(-0.22, 0.5, f"{section_title}\nSuccess Rate (%)",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=16)

    handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=3, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="upper right", fontsize=16, frameon=False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fig_droid_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.join(out_dir, "fig_droid_combined.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {path}, {pdf_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot DROID generalization success rates")
    parser.add_argument("--output-dir", type=str, default="analysis/droid_generalization_plots",
                        help="Directory to save figures")
    parser.add_argument("--combined-only", action="store_true",
                        help="Only produce the combined 3×4 figure")
    args = parser.parse_args()

    out = args.output_dir

    if not args.combined_only:
        _plot_section(
            LANGUAGE_DATA, LANGUAGE_CONDITIONS,
            "DROID Generalization – Language Perturbation Success Rates",
            os.path.join(out, "fig1_droid_language.png"),
        )
        _plot_section(
            VISUAL_DATA, VISUAL_CONDITIONS,
            "DROID Generalization – Visual Perturbation Success Rates",
            os.path.join(out, "fig2_droid_visual.png"),
        )
        _plot_section(
            POLICY_DATA, POLICY_CONDITIONS,
            "DROID Generalization – Policy Perturbation Success Rates",
            os.path.join(out, "fig3_droid_policy.png"),
        )

    _plot_combined(out)
    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()
