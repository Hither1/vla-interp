"""Plot DROID generalization experiment success rates from generalization_droid.md.

Produces three grouped bar chart figures:
  fig1_droid_language.png  -- Language perturbation success rates
  fig2_droid_visual.png    -- Visual perturbation success rates
  fig3_droid_policy.png    -- Policy perturbation success rates

Usage:
  python analysis/plot_droid_generalization.py --output-dir results/droid_generalization_plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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

def _plot_section(data, conditions, title, out_path):
    """Grouped bar chart: one subplot per suite, conditions on x-axis, models as bar groups."""
    n_suites = len(SUITES)
    fig, axes = plt.subplots(1, n_suites, figsize=(4.5 * n_suites, 4.5), sharey=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    n_models = len(MODELS)
    bar_width = 0.75 / n_models
    x = np.arange(len(conditions))

    for ax, suite in zip(axes, SUITES):
        suite_data = data[suite]
        for i, model in enumerate(MODELS):
            heights = [suite_data[cond][model] for cond in conditions]
            offset = (i - n_models / 2.0 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, heights, width=bar_width * 0.92,
                color=MODEL_COLORS[model], label=model, zorder=3,
            )
            # Label bars with non-zero values
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{h}%",
                        ha="center", va="bottom", fontsize=12, color="black",
                    )

        ax.set_title(SUITE_DISPLAY[suite], fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=12)
        ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Success Rate (%)", fontsize=13)

    # Single legend on the first axis
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], label=m)
        for m in MODELS
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.9)

    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.08)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {out_path}, {pdf_path}")
    plt.close(fig)


def _plot_combined(out_dir):
    """Single figure with 3 rows (one per perturbation type), 4 cols (one per suite)."""
    sections = [
        (LANGUAGE_DATA, LANGUAGE_CONDITIONS, "Language Perturbation"),
        (VISUAL_DATA,   VISUAL_CONDITIONS,   "Visual Perturbation"),
        (POLICY_DATA,   POLICY_CONDITIONS,   "Policy Perturbation"),
    ]

    n_rows = len(sections)
    n_cols = len(SUITES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        sharey="row",
    )
    fig.suptitle("DROID Generalization – Success Rates", fontsize=17, fontweight="bold", y=1.01)

    n_models = len(MODELS)
    bar_width = 0.75 / n_models

    for row, (data, conditions, section_title) in enumerate(sections):
        x = np.arange(len(conditions))
        for col, suite in enumerate(SUITES):
            ax = axes[row][col]
            suite_data = data[suite]
            for i, model in enumerate(MODELS):
                heights = [suite_data[cond][model] for cond in conditions]
                offset = (i - n_models / 2.0 + 0.5) * bar_width
                bars = ax.bar(
                    x + offset, heights, width=bar_width * 0.92,
                    color=MODEL_COLORS[model], label=model, zorder=3,
                )
                for bar, h in zip(bars, heights):
                    if h > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f"{h}%",
                            ha="center", va="bottom", fontsize=11, color="black",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=11)
            ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 108)

            if col == 0:
                ax.set_ylabel(f"{section_title}\nSuccess Rate (%)", fontsize=12)
            if row == 0:
                ax.set_title(SUITE_DISPLAY[suite], fontsize=14, fontweight="bold")

    # Shared legend (bottom right panel)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], label=m)
        for m in MODELS
    ]
    axes[-1][-1].legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.9)

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
    parser.add_argument("--output-dir", type=str, default="results/droid_generalization_plots",
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
