import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_generalization import *
NaN = np.nan

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
MODEL_COLORS = {
    "pi0.5":     "#4C72B0",
    "OpenVLA":   "#DD8452",
    "Cosmos":    "#55A868",
    "DreamZero": "#17becf",
    "DP":        "#C44E52",
}

MODELS_ALL  = ["pi0.5", "OpenVLA", "Cosmos", "DreamZero", "DP"]
MODELS_ATTN = ["pi0.5", "OpenVLA", "Cosmos", "DreamZero", "DP"]

ALL_SUITES = [
    "LIBERO-In domain",
    "LIBERO-90-Object",
    "LIBERO-90-Spatial",
    "LIBERO-90-Act",
    "LIBERO-90-Com",
]

SUITE_SHORT_TITLES = {
    "LIBERO-In domain":  "In domain",
    "LIBERO-90-Object":  "90-Object",
    "LIBERO-90-Spatial": "90-Spatial",
    "LIBERO-90-Act":     "90-Act",
    "LIBERO-90-Com":     "90-Com",
}

LANG_CONDS = ["original", "empty", "shuffle", "random", "synonym"]
VIS_CONDS  = ["original", "rotate 30°", "translate 20%", "rotate+translate"]
POL_CONDS  = ["original", "random action 25%", "object shift x"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_generalization")
os.makedirs(OUT_DIR, exist_ok=True)

RADAR_RC = {
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 14,
}

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def save_fig(fig, name, out_dir=OUT_DIR):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def _closed(vals):
    vals = np.asarray(vals, dtype=float)
    return np.concatenate([vals, [vals[0]]])


def _max_value_for_section(data_dict, suites, models):
    max_val = 0.0
    for suite in suites:
        suite_data = data_dict.get(suite, {})
        for model in models:
            for value in suite_data.get(model, []):
                if value is None:
                    continue
                value = float(value)
                if np.isnan(value):
                    continue
                max_val = max(max_val, value)
    return max_val


def _default_condition_text_offsets(conditions):
    """
    Small manual nudges for angle labels placed with ax.text.
    Values are (theta_delta, radius_scale_delta).
    radius is applied as base_radius * (1 + radius_scale_delta)
    """
    if conditions == VIS_CONDS:
        return {
            "original":            (0.00, -0.1),
            "rotate 30°":         (-0.40,  -0.1),
            "translate 20%":      ( 0.10,  -0.05),
            "rotate+translate":   ( 0.1,  0.00),
        }
    if conditions == POL_CONDS:
        return {
            "original":            (0.00, -0.1),
            "random action 25%":  (-0.5,  0.00),
            "object shift x":      ( 0.,  0.),
        }
    if conditions == LANG_CONDS:
        return {
            "original": (0.00, -0.1),
            "empty":    (-0.06, 0.02),
            "shuffle":  (0.00, 0.02),
            "random":   (0.04, 0.02),
            "synonym":  (0.05, 0.03),
        }
    return {c: (0.0, 0.0) for c in conditions}


def radar_ax(
    ax,
    data,
    conditions,
    models,
    colors,
    title="",
    ylim=(0, 100),
    pct_fmt=True,
    r_ticks=None,
    rlabel_angle=30,
    title_pad=34,
    show_condition_labels=True,
    condition_fs=21,
    radial_fs=16,
    suite_title_fs=26,
):
    """
    Cleaner radar plot:
      - radial labels moved away from top
      - angle labels drawn manually outside circle
      - suite title padded upward
    """
    n_cond = len(conditions)
    angles = np.linspace(0, 2 * np.pi, n_cond, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for m in models:
        vals = data.get(m, [NaN] * n_cond)
        vals = [NaN if v is None else v for v in vals]
        ax.plot(
            closed_angles,
            _closed(vals),
            color=colors[m],
            lw=2.6,
            label=m,
            zorder=3,
        )
        ax.fill(
            closed_angles,
            _closed(vals),
            color=colors[m],
            alpha=0.08,
            zorder=2,
        )

    ax.set_title(title, pad=title_pad, fontsize=suite_title_fs, fontweight="normal")

    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n_cond)

    lo, hi = ylim
    ax.set_ylim(lo, hi)

    if r_ticks is None:
        if pct_fmt:
            r_ticks = [25, 50, 75, 100] if hi == 100 else np.linspace(lo, hi, 4)
        else:
            r_ticks = np.linspace(lo, hi, 4)

    ax.set_yticks(r_ticks)
    if pct_fmt:
        ax.set_yticklabels([f"{int(round(t))}%" for t in r_ticks], fontsize=radial_fs)
    else:
        decimals = 2 if hi <= 1.0 else 1
        ax.set_yticklabels([f"{t:.{decimals}f}" for t in r_ticks], fontsize=radial_fs)

    ax.set_rlabel_position(rlabel_angle)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.tick_params(axis="y", pad=6)

    if show_condition_labels:
        offsets = _default_condition_text_offsets(conditions)

        # push labels farther outside than before
        base_r = hi * 1.18 if hi > 1 else hi * 1.24

        for theta, label in zip(angles, conditions):
            dtheta, dr_scale = offsets.get(label, (0.0, 0.0))
            ax.text(
                theta + dtheta,
                base_r * (1.0 + dr_scale),
                label,
                ha="center",
                va="center",
                fontsize=condition_fs,
                clip_on=False,
            )


def add_shared_legend(fig, models, colors, y=0.92, fontsize=18):
    handles = [Line2D([0], [0], color=colors[m], lw=3) for m in models]
    fig.legend(
        handles,
        models,
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, y),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.5,
        fontsize=fontsize,
    )


def plot_radar_section(
    prefix,
    section_title,
    suites,
    conditions,
    data_dict,
    models,
    colors,
    ylim,
    pct_fmt,
    filename_suffix,
    figure_title_fs=38,
    legend_y=0.90,
):
    with plt.rc_context(RADAR_RC):
        n = len(suites)
        if ylim is None:
            section_max = _max_value_for_section(data_dict, suites, models)
            ylim = (0, section_max if section_max > 0 else 1.0)

        fig, axes = plt.subplots(
            1,
            n,
            figsize=(5.4 * n, 7.0),
            subplot_kw={"projection": "polar"},
        )

        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"LIBERO {section_title} – {filename_suffix}",
            fontsize=figure_title_fs,
            fontweight="bold",
            y=0.975,
        )

        for ax, suite in zip(axes, suites):
            radar_ax(
                ax=ax,
                data=data_dict.get(suite, {}),
                conditions=conditions,
                models=models,
                colors=colors,
                title=SUITE_SHORT_TITLES.get(suite, suite),
                ylim=ylim,
                pct_fmt=pct_fmt,
                rlabel_angle=22.5,
                title_pad=26,
                condition_fs=19 if len(conditions) <= 4 else 18,
                radial_fs=14,
                suite_title_fs=22,
            )

        add_shared_legend(fig, models, colors, y=legend_y, fontsize=18)

        fig.subplots_adjust(
            left=0.03,
            right=0.995,
            bottom=0.08,
            top=0.82,
            wspace=0.42,
        )

        safe_suffix = filename_suffix.lower().replace(" ", "_")
        save_fig(fig, f"{prefix}_{safe_suffix}.png")


# -----------------------------------------------------------------------------
# ENTRY POINTS FOR THE 9 RADAR FIGURES
# -----------------------------------------------------------------------------
def make_all_radar_plots():
    # 1a: Language success rate
    plot_radar_section(
        prefix="1",
        section_title="Language Perturbation",
        suites=ALL_SUITES,
        conditions=LANG_CONDS,
        data_dict=LANG_SR,
        models=MODELS_ALL,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=True,
        filename_suffix="Success Rate",
        figure_title_fs=34,
        legend_y=0.90,
    )

    # 1b: Language attention IoU
    plot_radar_section(
        prefix="1",
        section_title="Language Perturbation",
        suites=ALL_SUITES,
        conditions=LANG_CONDS,
        data_dict=LANG_IOU,
        models=MODELS_ATTN,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention IoU",
        figure_title_fs=30,
        legend_y=0.90,
    )

    # 1c: Language attention ratio
    lang_ratio_suites = [
        "LIBERO-In domain",
        "LIBERO-90-Spatial",
        "LIBERO-90-Object",
        "LIBERO-90-Com",
    ]
    lang_ratio_models = [m for m in MODELS_ATTN if any(LANG_RATIO.get(s, {}).get(m) for s in lang_ratio_suites)]
    plot_radar_section(
        prefix="1",
        section_title="Language Perturbation",
        suites=lang_ratio_suites,
        conditions=LANG_CONDS,
        data_dict=LANG_RATIO,
        models=lang_ratio_models,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention Ratio",
        figure_title_fs=30,
        legend_y=0.90,
    )

    # 2a: Visual success rate
    plot_radar_section(
        prefix="2",
        section_title="Visual Perturbation",
        suites=ALL_SUITES,
        conditions=VIS_CONDS,
        data_dict=VIS_SR,
        models=MODELS_ALL,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=True,
        filename_suffix="Success Rate",
        figure_title_fs=34,
        legend_y=0.90,
    )

    # 2b: Visual attention IoU
    plot_radar_section(
        prefix="2",
        section_title="Visual Perturbation",
        suites=ALL_SUITES,
        conditions=VIS_CONDS,
        data_dict=VIS_IOU,
        models=MODELS_ATTN,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention IoU",
        figure_title_fs=30,
        legend_y=0.90,
    )

    # 2c: Visual attention ratio
    vis_ratio_models = [m for m in MODELS_ATTN if any(VIS_RATIO.get(s, {}).get(m) for s in ALL_SUITES)]
    plot_radar_section(
        prefix="2",
        section_title="Visual Perturbation",
        suites=ALL_SUITES,
        conditions=VIS_CONDS,
        data_dict=VIS_RATIO,
        models=vis_ratio_models,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention Ratio",
        figure_title_fs=30,
        legend_y=0.90,
    )

    # 3a: Policy success rate
    plot_radar_section(
        prefix="3",
        section_title="Policy Perturbation",
        suites=ALL_SUITES,
        conditions=POL_CONDS,
        data_dict=POL_SR,
        models=MODELS_ALL,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=True,
        filename_suffix="Success Rate",
        figure_title_fs=34,
        legend_y=0.90,
    )

    # 3b: Policy attention IoU
    plot_radar_section(
        prefix="3",
        section_title="Policy Perturbation",
        suites=ALL_SUITES,
        conditions=POL_CONDS,
        data_dict=POL_IOU,
        models=MODELS_ATTN,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention IoU",
        figure_title_fs=30,
        legend_y=0.90,
    )

    # 3c: Policy attention ratio
    pol_ratio_models = [m for m in MODELS_ATTN if any(POL_RATIO.get(s, {}).get(m) for s in ALL_SUITES)]
    plot_radar_section(
        prefix="3",
        section_title="Policy Perturbation",
        suites=ALL_SUITES,
        conditions=POL_CONDS,
        data_dict=POL_RATIO,
        models=pol_ratio_models,
        colors=MODEL_COLORS,
        ylim=None,
        pct_fmt=False,
        filename_suffix="Attention Ratio",
        figure_title_fs=30,
        legend_y=0.90,
    )


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    make_all_radar_plots()
    print(f"Done. Radar plots saved to: {OUT_DIR}")
