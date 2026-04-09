"""
Plot generalization experiment results from generalization.md.

Radar plots (per section):
  1a–c  Language perturbation  – success rate / attention IoU / attention ratio
  2a–c  Visual perturbation    – success rate / attention IoU / attention ratio
  3a–c  Policy perturbation    – success rate / attention IoU / attention ratio

Relationship scatter plots (pooled across suites / conditions):
  R1  Success vs IoU,          faceted by perturbation type, colored by model
  R2  Success vs |ratio−0.5|,  faceted by perturbation type, colored by model
  R3  Success vs IoU,          faceted by model,             colored by suite
  R4  Success vs IoU,          faceted by suite,             colored by model
  R5  Pearson r heatmap  (success–IoU)  model × perturbation type
  R6  IoU vs |ratio−0.5|,      colored by model  (both degradation axes)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats

# ─── style ────────────────────────────────────────────────────────────────────
NaN = np.nan

MODEL_COLORS = {
    "pi0.5":    "#4C72B0",
    "OpenVLA":  "#DD8452",
    "Cosmos":   "#55A868",
    "DP":       "#C44E52",
    "DreamZero":"#17becf",
}
SUITE_COLORS = {
    "LIBERO-In domain":  "#1f77b4",
    "LIBERO-90-Object":  "#ff7f0e",
    "LIBERO-90-Spatial": "#2ca02c",
    "LIBERO-90-Act":     "#d62728",
    "LIBERO-90-Com":     "#9467bd",
}
PERTURB_COLORS = {"language": "#e41a1c", "visual": "#377eb8", "policy": "#4daf4a"}
PERTURB_MARKERS = {"language": "o", "visual": "s", "policy": "^"}

# ── Finding-type styles and key annotated examples ─────────────────────────────
FINDING_STYLE = {
    "shortcutting":  {"color": "#ff7f00", "marker": "D",
                      "label": "Type 1: Shortcutting (succeeds w/o watching)"},
    "confused":      {"color": "#e41a1c", "marker": "X",
                      "label": "Type 2: Confused (watches wrong things)"},
    "grounded":      {"color": "#4daf4a", "marker": "P",
                      "label": "Type 3: Grounded (watches & acts correctly)"},
    "not_looking":   {"color": "#9467bd", "marker": "v",
                      "label": "Type 4: Not looking (fails w/o visual anchoring)"},
    "control_fails": {"color": "#1f77b4", "marker": "^",
                      "label": "Type 5: Control fails (perception OK, action fails)"},
}

# (model, perturbation, suite, condition, finding_type, short_label)
KEY_EXAMPLES = [
    # Type 1: high success despite semantically wrong / spatially disrupted language/vision
    ("Cosmos", "visual",   "LIBERO-In domain", "translate 20%",     "shortcutting",  "Cosmos\ntranslate"),
    # Type 2: ratio increases but IoU collapses → watching hard but at wrong things
    ("OpenVLA","language", "LIBERO-In domain", "empty",             "confused",      "OpenVLA\nempty"),
    ("pi0.5",  "visual",   "LIBERO-In domain", "rotate 30°",        "confused",      "pi0.5\nrotate"),
    # Type 3: strong baseline grounding
    ("pi0.5",  "language", "LIBERO-In domain", "original",          "grounded",      "pi0.5\noriginal"),
    ("Cosmos", "language", "LIBERO-In domain", "original",          "grounded",      "Cosmos\noriginal"),
    # Type 4: Cosmos reduces ratio when language/vision fails → retreats to broken signal
    ("Cosmos", "language", "LIBERO-In domain", "random",            "not_looking",   "Cosmos\nrandom"),
    ("Cosmos", "visual",   "LIBERO-In domain", "rotate 30°",        "not_looking",   "Cosmos\nrotate"),
    # Type 5: action injection fails success but attention pattern stays intact
    ("pi0.5",  "policy",   "LIBERO-In domain", "random action 25%", "control_fails", "pi0.5\nrand act"),
]

MODELS_ALL  = ["pi0.5", "OpenVLA", "Cosmos", "DreamZero", "DP"]
MODELS_ATTN = ["pi0.5", "OpenVLA", "Cosmos", "DreamZero", "DP"]
ALL_SUITES  = ["LIBERO-In domain", "LIBERO-90-Object", "LIBERO-90-Spatial",
               "LIBERO-90-Act",    "LIBERO-90-Com"]

SUITE_SHORT_TITLES = {
    "LIBERO-In domain": "In domain",
    "LIBERO-90-Object": "90-Object",
    "LIBERO-90-Spatial": "90-Spatial",
    "LIBERO-90-Act": "90-Act",
    "LIBERO-90-Com": "90-Com",
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})

# Larger fonts used for non-R bar chart figures (1_*, 2_*, 3_*).
# Figures are ~19" wide (5 panels × 3.8"); to appear as ~9pt at 7" paper width
# (scale factor ≈ 2.7) the matplotlib fontsizes need to be ~2.7× the target pt.
BAR_RC = {
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
}
SR_RC = {
    "font.size": 42, "axes.titlesize": 46, "axes.labelsize": 42,
    "legend.fontsize": 39, "xtick.labelsize": 39, "ytick.labelsize": 39,
}
SCATTER_RC = {
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16,
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_generalization")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def radar_ax(ax, data, conditions, models, colors, ylabel, title,
             ylim=None, pct_fmt=False, custom_angle_labels=None):
    n_cond = len(conditions)
    angles = np.linspace(0, 2 * np.pi, n_cond, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    for m in models:
        vals = [v if v is not None else NaN for v in data.get(m, [NaN] * n_cond)]
        closed_vals = np.concatenate([vals, [vals[0]]])
        ax.plot(closed_angles, closed_vals, color=colors[m], lw=2.2, label=m, zorder=3)
        ax.fill(closed_angles, closed_vals, color=colors[m], alpha=0.08, zorder=2)

    ax.set_title(title, pad=18)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    if custom_angle_labels:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(conditions)
    ax.tick_params(axis="x", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    if ylim:
        ax.set_ylim(ylim)
        lo, hi = ylim
        if pct_fmt:
            ticks = np.linspace(lo, hi, 5)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(round(t))}%" for t in ticks])
        else:
            ticks = np.linspace(lo, hi, 5)
            decimals = 2 if hi <= 1.1 else 1
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.{decimals}f}" for t in ticks])
        for label in ax.get_yticklabels():
            label.set_rotation(32)
            label.set_rotation_mode("anchor")
        if custom_angle_labels:
            base_radius = hi * 0.84
            for theta, label in zip(angles, conditions):
                theta_delta, radius_delta = custom_angle_labels.get(label, (0.0, 0.0))
                ax.text(
                    theta + theta_delta,
                    base_radius + radius_delta,
                    label,
                    ha="center",
                    va="center",
                )
    ax.set_rlabel_position(0)
    if ylabel:
        ax.text(-0.12, 0.5, ylabel, transform=ax.transAxes, rotation=90,
                va="center", ha="center")


def add_regline(ax, xs, ys, color, lw=1.5):
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3 or xs[mask].std() == 0:
        return
    slope, intercept, r, *_ = stats.linregress(xs[mask], ys[mask])
    xl = np.array([xs[mask].min(), xs[mask].max()])
    ax.plot(xl, slope*xl + intercept, color=color, lw=lw, ls="--", alpha=0.8,
            zorder=5)


def annotate_quadrants(ax, x_thresh, y_thresh, xlim, ylim, quad_labels,
                       colors=("#4daf4a", "#ff7f00", "#1f77b4", "#9467bd"),
                       fontsize=8, line_alpha=0.4):
    """Draw dashed threshold lines and label all 4 quadrants.

    quad_labels order: [top-right, top-left, bottom-right, bottom-left]
    """
    ax.axvline(x_thresh, color="grey", ls=":", lw=1, alpha=line_alpha, zorder=1)
    ax.axhline(y_thresh, color="grey", ls=":", lw=1, alpha=line_alpha, zorder=1)
    centres = [
        ((x_thresh + xlim[1]) / 2, (y_thresh + ylim[1]) / 2),   # top-right
        ((xlim[0]  + x_thresh) / 2, (y_thresh + ylim[1]) / 2),  # top-left
        ((x_thresh + xlim[1]) / 2, (ylim[0]  + y_thresh) / 2),  # bottom-right
        ((xlim[0]  + x_thresh) / 2, (ylim[0]  + y_thresh) / 2), # bottom-left
    ]
    for (px, py), lbl, c in zip(centres, quad_labels, colors):
        ax.text(px, py, lbl, ha="center", va="center", fontsize=fontsize,
                color=c, alpha=0.62, fontweight="bold", zorder=2)


def shared_legend_beside_title(fig, models, colors, ncol=5):
    handles = [Line2D([0], [0], color=colors[m], lw=3) for m in models]
    fig.legend(handles, models, loc="upper right", ncol=ncol,
               bbox_to_anchor=(0.99, 1.0), frameon=False,
               columnspacing=0.5, handlelength=0.8, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=0.3, h_pad=0.3)


def shared_legend_top(fig, models, colors, ncol=5, y=0.93):
    handles = [Line2D([0], [0], color=colors[m], lw=3) for m in models]
    fig.legend(
        handles, models, loc="upper center", ncol=ncol,
        bbox_to_anchor=(0.5, y), frameon=False,
        columnspacing=1.0, handlelength=1.5, handletextpad=0.5,
        fontsize=30,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLES
# ═══════════════════════════════════════════════════════════════════════════════

LANG_CONDS = ["original", "empty", "shuffle", "random", "synonym"]
VIS_CONDS  = ["original", "rotate 30°", "translate 20%", "rotate+translate"]
POL_CONDS  = ["original", "random action 25%", "object shift x"]

# ── success rates ──────────────────────────────────────────────────────────────
LANG_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [98.2, 62.5, 100.0, 34.5,  96.5],
        "OpenVLA":  [76.5,  0.13, 24.5,  6.3,  64.8],
        "Cosmos":   [98.5, 50.5,  83.3,  32.5,  94.5],
        "DP":       [91.8, 36.3,  35.3,  30.9,  35.5],
        "DreamZero":[97.8, 59.2,  91.5,  31.2,  97.8],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  4.0, 50.2,  3.3, 10.2],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  3.5, 12.2],
        "Cosmos":   [38.0, 27.8, 32.0, 27.0, 38.8],
        "DP":       [18.5, 14.5, 14.2, 13.5, 15.0],
        "DreamZero":[38.3, 27.9, 32.2, 25.3, 38.5],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  8.0, 21.5,  1.0, 24.0],
        "OpenVLA":  [ 4.1,  0.1,  2.8,  0.6,  4.9],
        "Cosmos":   [13.72,13.49,12.91,15.47,13.14],
        "DP":       [ 7.8,  6.7,  6.9,  7.9,  6.5],
        "DreamZero":[14.20,13.80,13.20,14.80,13.40],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 10.2, 27.0,  3.0, 28.0],
        "OpenVLA":  [13.2,  1.2,  9.1,  2.6, 11.5],
        "Cosmos":   [33.8, 26.2, 37.10,30.0, 40.0],
        "DP":       [12.1, 20.3, 21.2, 22.1, 21.5],
        "DreamZero":[35.0, 27.0, 38.0, 28.8, 41.0],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 3.0, 2.9, 1.0, 2.7],
        "OpenVLA":  [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 2.0, 2.0, 0.0, 1.0],
        "DP":       [0.0, 0.0, 0.0, 0.0, 0.0],
        "DreamZero":[1.8, 2.2, 2.2, 0.1, 1.2],
    },
}

VIS_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [91.8, 11.6, 22.0, 19.7],
        "OpenVLA":  [76.5,  1.1, 13.4,  3.4],
        "Cosmos":   [98.5, 19.6, 80.0, 80.9],
        "DP":       [91.8,  6.9, 15.5, 14.8],
        "DreamZero":[97.8, 18.8, 87.5, 88.3],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.0, 12.6,  6.8],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  0.3],
        "Cosmos":   [38.0,  0.5, 24.5, 23.0],
        "DP":       [18.5,  0.0,  1.8,  1.0],
        "DreamZero":[38.3,  0.4, 24.8, 23.3],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  0.9,  2.6,  5.8],
        "OpenVLA":  [ 4.1,  0.2,  0.8,  0.6],
        "Cosmos":   [13.72, 0.5, 10.2,  6.9],
        "DP":       [ 7.8,  0.2,  2.7,  1.2],
        "DreamZero":[14.20, 0.5, 10.8,  7.4],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 11.5, 12.6, 19.7],
        "OpenVLA":  [13.2,  3.8,  8.8,  4.1],
        "Cosmos":   [33.8,  7.1, 18.8, 26.5],
        "DP":       [12.1,  0.3,  0.0,  0.3],
        "DreamZero":[35.0,  6.9, 19.6, 27.5],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 0.0, 0.0, 0.5],
        "OpenVLA":  [0.0, 0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 0.0, 0.5, 1.0],
        "DP":       [0.0, 0.0, 0.0, 0.0],
        "DreamZero":[1.8, 0.0, 0.7, 1.2],
    },
}

POL_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [91.8, 31.5, 77.8],
        "OpenVLA":  [76.5,  5.0, 34.1],
        "Cosmos":   [98.5, 25.2, 76.8],
        "DP":       [91.8, 20.5, 54.6],
        "DreamZero":[97.8, 24.8, 78.5],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.5,  7.7],
        "OpenVLA":  [ 9.5,  2.0,  6.8],
        "Cosmos":   [38.0,  8.7, 31.8],
        "DP":       [18.5,  4.3, 10.0],
        "DreamZero":[38.3,  7.9, 32.0],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  4.4,  9.5],
        "OpenVLA":  [ 4.1,  0.2,  2.4],
        "Cosmos":   [13.72, 3.8, 12.7],
        "DP":       [ 7.8,  2.4,  4.7],
        "DreamZero":[14.20, 3.5, 13.1],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 19.4, 30.0],
        "OpenVLA":  [13.2,  5.6, 10.9],
        "Cosmos":   [33.8, 14.1, 28.8],
        "DP":       [12.1,  3.2,  9.4],
        "DreamZero":[35.0, 13.5, 29.6],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 1.0, 1.0],
        "OpenVLA":  [0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 0.0, 1.0],
        "DP":       [0.0, 0.0, 0.0],
        "DreamZero":[1.8, 0.1, 1.2],
    },
}

# ── attention IoU ──────────────────────────────────────────────────────────────
LANG_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.302, 0.195, 0.298, 0.162, 0.295],
        "OpenVLA":  [0.250, 0.012, 0.135, 0.068, 0.228],
        "Cosmos":   [0.335, 0.205, 0.295, 0.175, 0.325],
        "DP":       [0.283, 0.168, 0.248, 0.138, 0.272],
        "DreamZero":[0.348, 0.062, 0.308, 0.172, 0.338],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170, 0.048, 0.168, 0.044, 0.078],
        "OpenVLA":  [0.068, 0.008, 0.052, 0.040, 0.072],
        "Cosmos":   [0.190, 0.158, 0.172, 0.158, 0.192],
        "DP":       [0.130, 0.068, 0.122, 0.065, 0.095],
        "DreamZero":[0.191, 0.159, 0.173, 0.151, 0.193],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110, 0.065, 0.106, 0.025, 0.112],
        "OpenVLA":  [0.042, 0.008, 0.034, 0.016, 0.044],
        "Cosmos":   [0.120, 0.118, 0.116, 0.126, 0.118],
        "DP":       [0.080, 0.056, 0.076, 0.062, 0.082],
        "DreamZero":[0.125, 0.123, 0.121, 0.124, 0.123],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130, 0.076, 0.122, 0.042, 0.125],
        "OpenVLA":  [0.078, 0.022, 0.062, 0.034, 0.072],
        "Cosmos":   [0.150, 0.130, 0.158, 0.140, 0.162],
        "DP":       [0.090, 0.062, 0.082, 0.055, 0.085],
        "DreamZero":[0.158, 0.137, 0.165, 0.139, 0.169],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035, 0.032, 0.033, 0.025, 0.032],
        "OpenVLA":  [0.015, 0.010, 0.010, 0.008, 0.010],
        "Cosmos":   [0.040, 0.038, 0.038, 0.018, 0.032],
        "DP":       [0.025, 0.022, 0.022, 0.015, 0.022],
        "DreamZero":[0.042, 0.040, 0.040, 0.019, 0.034],
    },
}

VIS_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.30250, 0.10875, 0.17875, 0.15750],
        "OpenVLA":  [0.24375, 0.01495, 0.12860, 0.02958],
        "Cosmos":   [0.33500, 0.15375, 0.19250, 0.17500],
        "DP":       [0.28250, 0.09375, 0.14875, 0.12875],
        "DreamZero":[0.34800, 0.14900, 0.20100, 0.18300],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110,   0.030,   0.050,   0.045],
        "OpenVLA":  [0.04205, 0.02500, 0.01538, 0.01750],
        "Cosmos":   [0.120,   0.040,   0.050,   0.045],
        "DP":       [0.080,   0.025,   0.040,   0.035],
        "DreamZero":[0.125,   0.039,   0.055,   0.050],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170,   0.050,   0.075,   0.065],
        "OpenVLA":  [0.06676, 0.03500, 0.05972, 0.01350],
        "Cosmos":   [0.190,   0.060,   0.080,   0.070],
        "DP":       [0.130,   0.035,   0.050,   0.045],
        "DreamZero":[0.191,   0.056,   0.081,   0.071],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130,   0.040,   0.060,   0.055],
        "OpenVLA":  [0.09230, 0.04029, 0.05397, 0.03892],
        "Cosmos":   [0.150,   0.050,   0.120,   0.110],
        "DP":       [0.090,   0.030,   0.040,   0.035],
        "DreamZero":[0.158,   0.049,   0.127,   0.117],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035,   0.015,   0.015,   0.015],
        "OpenVLA":  [0.02000, 0.01000, 0.01000, 0.01000],
        "Cosmos":   [0.040,   0.015,   0.015,   0.015],
        "DP":       [0.025,   0.010,   0.010,   0.010],
        "DreamZero":[0.042,   0.014,   0.016,   0.016],
    },
}

POL_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.30250, 0.28250, 0.24375],
        "OpenVLA":  [0.27500, 0.14000, 0.20500],
        "Cosmos":   [0.33500, 0.31000, 0.27750],
        "DP":       [0.28250, 0.26250, 0.20750],
        "DreamZero":[0.34800, 0.31200, 0.29000],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170,  0.150,  0.110],
        "OpenVLA":  [0.0750, 0.0450, 0.0650],
        "Cosmos":   [0.190,  0.165,  0.130],
        "DP":       [0.130,  0.110,  0.080],
        "DreamZero":[0.191,  0.158,  0.131],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110,  0.100,  0.085],
        "OpenVLA":  [0.0500, 0.0150, 0.0400],
        "Cosmos":   [0.120,  0.105,  0.095],
        "DP":       [0.080,  0.070,  0.055],
        "DreamZero":[0.125,  0.106,  0.102],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130,  0.110,  0.100],
        "OpenVLA":  [0.0750, 0.0450, 0.0650],
        "Cosmos":   [0.150,  0.130,  0.115],
        "DP":       [0.090,  0.075,  0.065],
        "DreamZero":[0.158,  0.131,  0.123],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035,  0.030,  0.025],
        "OpenVLA":  [0.0120, 0.0100, 0.0090],
        "Cosmos":   [0.040,  0.030,  0.030],
        "DP":       [0.025,  0.020,  0.015],
        "DreamZero":[0.042,  0.029,  0.032],
    },
}

# ── attention ratio (visual / total) ──────────────────────────────────────────
LANG_RATIO = {          # suites: In domain, Spatial, Object, Com (no Act)
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.628, 0.580, 0.648, 0.578],
        "OpenVLA":  [0.646, 0.748, 0.712, 0.735, 0.660],
        "Cosmos":   [0.439, 0.378, 0.420, 0.362, 0.434],
        "DP":       [0.928, 0.972, 0.942, 0.975, 0.930],
        "DreamZero":[0.445, 0.383, 0.426, 0.360, 0.440],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.775, 0.738, 0.812, 0.733],
        "OpenVLA":  [0.972, 0.985, 0.978, 0.984, 0.970],
        "Cosmos":   [0.355, 0.351, 0.350, 0.362, 0.358],
        "DP":       [0.972, 0.984, 0.976, 0.977, 0.971],
        "DreamZero":[0.362, 0.357, 0.355, 0.361, 0.364],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.782, 0.705, 0.785, 0.756],
        "OpenVLA":  [0.947, 0.986, 0.956, 0.979, 0.940],
        "Cosmos":   [0.383, 0.368, 0.372, 0.370, 0.384],
        "DP":       [0.966, 0.982, 0.968, 0.981, 0.968],
        "DreamZero":[0.384, 0.369, 0.373, 0.363, 0.385],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.812, 0.811, 0.820, 0.812],
        "OpenVLA":  [0.990, 0.990, 0.990, 0.990, 0.990],
        "Cosmos":   [0.254, 0.254, 0.252, 0.240, 0.256],
        "DP":       [0.986, 0.987, 0.986, 0.989, 0.987],
        "DreamZero":[0.260, 0.260, 0.258, 0.240, 0.262],
    },
}

VIS_RATIO = {           # OpenVLA not available for visual perturbation
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.779, 0.738, 0.755],
        "Cosmos":   [0.439, 0.314, 0.434, 0.427],
        "DP":       [0.928, 0.984, 0.978, 0.980],
        "DreamZero":[0.445, 0.313, 0.440, 0.433],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.814, 0.801, 0.807],
        "Cosmos":   [0.383, 0.265, 0.379, 0.371],
        "DP":       [0.966, 0.989, 0.987, 0.987],
        "DreamZero":[0.384, 0.259, 0.380, 0.372],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.822, 0.814, 0.815],
        "Cosmos":   [0.355, 0.248, 0.351, 0.342],
        "DP":       [0.972, 0.989, 0.989, 0.989],
        "DreamZero":[0.362, 0.248, 0.357, 0.348],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.724, 0.820, 0.808, 0.812],
        "Cosmos":   [0.367, 0.254, 0.363, 0.355],
        "DP":       [0.971, 0.990, 0.989, 0.989],
        "DreamZero":[0.374, 0.253, 0.369, 0.361],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.827, 0.825, 0.826],
        "Cosmos":   [0.254, 0.231, 0.259, 0.254],
        "DP":       [0.986, 0.990, 0.990, 0.990],
        "DreamZero":[0.260, 0.230, 0.265, 0.260],
    },
}

POL_RATIO = {
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.701, 0.662],
        "OpenVLA":  [0.646, 0.968, 0.837],
        "Cosmos":   [0.439, 0.369, 0.420],
        "DP":       [0.928, 0.974, 0.954],
        "DreamZero":[0.445, 0.368, 0.426],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.747, 0.735],
        "OpenVLA":  [0.947, 0.981, 0.959],
        "Cosmos":   [0.383, 0.342, 0.367],
        "DP":       [0.966, 0.985, 0.978],
        "DreamZero":[0.384, 0.335, 0.368],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.782, 0.775],
        "OpenVLA":  [0.972, 0.989, 0.979],
        "Cosmos":   [0.355, 0.306, 0.333],
        "DP":       [0.972, 0.987, 0.981],
        "DreamZero":[0.362, 0.306, 0.340],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.724, 0.784, 0.766],
        "OpenVLA":  [0.931, 0.965, 0.941],
        "Cosmos":   [0.367, 0.315, 0.342],
        "DP":       [0.971, 0.988, 0.980],
        "DreamZero":[0.374, 0.314, 0.349],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.820, 0.814],
        "OpenVLA":  [0.990, 0.990, 0.990],
        "Cosmos":   [0.254, 0.242, 0.254],
        "DP":       [0.986, 0.989, 0.988],
        "DreamZero":[0.260, 0.240, 0.260],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD UNIFIED DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def _expand(perturb, suite, conds, sr_d, iou_d, ratio_d):
    rows = []
    for m in MODELS_ALL:
        sr_vals    = sr_d.get(suite, {}).get(m, [NaN]*len(conds))
        iou_vals   = iou_d.get(suite, {}).get(m, [NaN]*len(conds))
        ratio_vals = ratio_d.get(suite, {}).get(m, [NaN]*len(conds))
        for ci, cond in enumerate(conds):
            rows.append(dict(
                perturbation=perturb, suite=suite, condition=cond, model=m,
                success=sr_vals[ci],
                iou=iou_vals[ci] if ci < len(iou_vals) else NaN,
                ratio=ratio_vals[ci] if ci < len(ratio_vals) else NaN,
            ))
    return rows


records = []
for s in ALL_SUITES:
    records += _expand("language", s, LANG_CONDS, LANG_SR, LANG_IOU, LANG_RATIO)
    records += _expand("visual",   s, VIS_CONDS,  VIS_SR,  VIS_IOU,  VIS_RATIO)
    records += _expand("policy",   s, POL_CONDS,  POL_SR,  POL_IOU,  POL_RATIO)

df = pd.DataFrame(records)
df["ratio_dev"] = np.abs(df["ratio"] - 0.5)   # deviation from balanced 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# RADAR PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Language perturbation ──────────────────────────────────────────────────
def _bar_section(prefix, suites, conds, sr_d, iou_d, ratio_d,
                 ratio_suites=None, section_title=""):
    """Emit three radar-plot figures for one perturbation section."""
    n = len(suites)
    w_per_panel = 4.4
    w_per_panel_sr = 4.8

    # (a) success rate
    with plt.rc_context(SR_RC):
        x_label_pad = -30 if len(conds) >= 5 else -16
        custom_angle_labels = None
        if prefix == "2":
            custom_angle_labels = {
                "original": (0.00, 6.0),
                "rotate 30°": (-0.12, -2.0),
                "translate 20%": (0.10, 4.0),
                "rotate+translate": (0.18, -4.0),
            }
        elif prefix == "3":
            custom_angle_labels = {
                "original": (0.00, 8.0),
                "random action 25%": (-0.22, -8.0),
                "object shift x": (0.22, 5.0),
            }
        fig, axes = plt.subplots(
            1, n, figsize=(5.3 * n, 10.6),
            subplot_kw={"projection": "polar"}
        )
        fig.suptitle(f"LIBERO {section_title} – Success Rate", fontsize=46, fontweight="bold", y=0.985)
        for ax, suite in zip(np.atleast_1d(axes), suites):
            radar_ax(ax, sr_d.get(suite, {}), conds, MODELS_ALL,
                     MODEL_COLORS, "", SUITE_SHORT_TITLES.get(suite, suite),
                     ylim=(0, 100), pct_fmt=True,
                     custom_angle_labels=custom_angle_labels)
            ax.title.set_fontsize(36)
            if custom_angle_labels:
                for text in ax.texts[-len(conds):]:
                    text.set_fontsize(30)
            else:
                ax.tick_params(axis="x", labelsize=30, pad=x_label_pad)
            ax.tick_params(axis="y", labelsize=24)
        shared_legend_top(fig, MODELS_ALL, MODEL_COLORS, ncol=5, y=0.89)
        fig.subplots_adjust(left=0.03, right=0.995, bottom=0.06, top=0.87, wspace=0.32)
        save_fig(fig, f"{prefix}_success_rate.png")

    # (b) attention IoU
    fig, axes = plt.subplots(
        1, n, figsize=(w_per_panel * n, 4.6),
        subplot_kw={"projection": "polar"}
    )
    fig.suptitle(f"{section_title} – Attention IoU", fontsize=28, fontweight="bold")
    for ax, suite in zip(np.atleast_1d(axes), suites):
        radar_ax(ax, iou_d.get(suite, {}), conds, MODELS_ATTN,
                 MODEL_COLORS, "Attention IoU", suite, ylim=(0, 0.35))
    shared_legend_beside_title(fig, MODELS_ATTN, MODEL_COLORS, ncol=len(MODELS_ATTN))
    save_fig(fig, f"{prefix}_attention_iou.png")

    # (c) attention ratio
    rs = ratio_suites if ratio_suites else suites
    n_r = len(rs)
    fig, axes = plt.subplots(
        1, n_r, figsize=(w_per_panel * n_r, 4.6),
        subplot_kw={"projection": "polar"}
    )
    fig.suptitle(f"{section_title} – Attention Ratio",
                 fontsize=28, fontweight="bold")
    ratio_models = [m for m in MODELS_ATTN
                    if any(ratio_d.get(s, {}).get(m) for s in rs)]
    for ax, suite in zip(np.atleast_1d(axes), rs):
        radar_ax(ax, ratio_d.get(suite, {}), conds, ratio_models,
                 MODEL_COLORS, "Attention ratio", suite, ylim=(0, 1.0))
    shared_legend_beside_title(fig, ratio_models, MODEL_COLORS, ncol=len(ratio_models))
    save_fig(fig, f"{prefix}_attention_ratio.png")


with plt.rc_context(BAR_RC):
    _bar_section("1", ALL_SUITES,   LANG_CONDS, LANG_SR, LANG_IOU, LANG_RATIO,
                 ratio_suites=["LIBERO-In domain","LIBERO-90-Spatial",
                                "LIBERO-90-Object","LIBERO-90-Com"],
                 section_title="Language Perturbation")

    _bar_section("2", ALL_SUITES,   VIS_CONDS,  VIS_SR,  VIS_IOU,  VIS_RATIO,
                 section_title="Visual Perturbation")

    _bar_section("3", ALL_SUITES,   POL_CONDS,  POL_SR,  POL_IOU,  POL_RATIO,
                 section_title="Policy Perturbation")

# ── Combined success rate: Language / Visual / Policy ─────────────────────────
with plt.rc_context(SR_RC):
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg

    _w = 5.0
    _n = len(ALL_SUITES)
    _bar_h = 5.8
    _img_h = 7.0   # height of the example-image header row (inches)

    fig_comb = plt.figure(figsize=(_w * _n, _img_h + 3 * _bar_h))
    gs = gridspec.GridSpec(
        4, _n, figure=fig_comb,
        height_ratios=[_img_h, _bar_h, _bar_h, _bar_h],
        hspace=0.48, wspace=0.06,
    )

    # ── Example-image header row ──────────────────────────────────────────────
    _repo = os.path.join(os.path.dirname(__file__), "..")
    _img_panels = [
        (gs[0, :2], os.path.join(_repo, "libero.jpeg"),      "Simulation (LIBERO)"),
        (gs[0, 2:], os.path.join(_repo, "droid_frame3.png"), "Real World (DROID)"),
    ]
    for _spec, _path, _lbl in _img_panels:
        _ax_img = fig_comb.add_subplot(_spec)
        _ax_img.imshow(mpimg.imread(_path), aspect="auto")
        _ax_img.set_title(_lbl, fontsize=SR_RC["axes.titlesize"],
                          fontweight="bold", pad=18, color="#111111")
        _ax_img.axis("off")

    # ── Radar plot rows ───────────────────────────────────────────────────────
    axes_comb = np.array([[fig_comb.add_subplot(gs[_ri + 1, _ci], projection="polar")
                           for _ci in range(_n)] for _ri in range(3)])

    _rows = [
        ("Language Perturbation", LANG_CONDS, LANG_SR),
        ("Visual Perturbation",   VIS_CONDS,  VIS_SR),
        ("Policy Perturbation",   POL_CONDS,  POL_SR),
    ]
    for _ri, (_rtitle, _conds, _sr_d) in enumerate(_rows):
        for _ci, _suite in enumerate(ALL_SUITES):
            _ax = axes_comb[_ri, _ci]
            radar_ax(_ax, _sr_d.get(_suite, {}), _conds, MODELS_ALL,
                     MODEL_COLORS, "",
                     _suite if _ri == 0 else "",
                     ylim=(0, 100), pct_fmt=True)
        axes_comb[_ri, 0].text(
            -0.45, 0.5, f"{_rtitle}\nSuccess rate (%)",
            transform=axes_comb[_ri, 0].transAxes,
            rotation=90, va="center", ha="center"
        )

    _handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=4) for m in MODELS_ALL]
    fig_comb.legend(
        _handles, MODELS_ALL,
        loc="upper right", ncol=5,
        bbox_to_anchor=(0.99, 1.0), frameon=False,
        columnspacing=0.5, handlelength=0.8, handletextpad=0.4,
        fontsize=SR_RC["legend.fontsize"],
    )
    fig_comb.tight_layout(rect=[0, 0, 1, 0.965], w_pad=0.3, h_pad=0.5)
    save_fig(fig_comb, "combined_success_rate.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP SCATTER PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# helper: filter df to rows that have both x and y
def valid(sub, xcol, ycol):
    return sub.dropna(subset=[xcol, ycol])


plt.rcParams.update(SCATTER_RC)

# ── R1: Success vs IoU – faceted by perturbation type, colored by model ───────
PERTURBS = ["language", "visual", "policy"]
fig_r1, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig_r1.suptitle("Success Rate vs Attention IoU\n(all suites pooled)",
                fontsize=22, fontweight="bold")

for ax, pt in zip(axes, PERTURBS):
    sub = valid(df[(df.perturbation == pt) & df.model.isin(MODELS_ATTN)],
                "success", "iou")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.iou, ms.success, c=MODEL_COLORS[m],
                   s=25, alpha=0.7, zorder=4, label=m)
        add_regline(ax, ms.iou.values, ms.success.values, MODEL_COLORS[m])
    ax.set_title(pt.capitalize())
    ax.set_xlabel("Attention IoU")
    ax.set_ylabel("Success rate (%)")
    ax.grid(True, ls="--", alpha=0.4)

handles = [plt.Line2D([0],[0], marker='o', color='w',
           markerfacecolor=MODEL_COLORS[m], ms=7) for m in MODELS_ATTN]
fig_r1.legend(handles, MODELS_ATTN, loc="lower center", ncol=len(MODELS_ATTN),
              bbox_to_anchor=(0.5, -0.05), frameon=False)
fig_r1.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r1, "R1_success_vs_iou_by_perturbation.png")


# ── R2: Success vs |ratio−0.5| – faceted by perturbation type ────────────────
fig_r2, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig_r2.suptitle("Success Rate vs Attention Ratio Deviation |ratio − 0.5|\n"
                "(all suites pooled)", fontsize=22, fontweight="bold")

for ax, pt in zip(axes, PERTURBS):
    sub = valid(df[(df.perturbation == pt) & df.model.isin(MODELS_ATTN)],
                "success", "ratio_dev")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.ratio_dev, ms.success, c=MODEL_COLORS[m],
                   s=25, alpha=0.7, zorder=4, label=m)
        add_regline(ax, ms.ratio_dev.values, ms.success.values, MODEL_COLORS[m])
    ax.set_title(pt.capitalize())
    ax.set_xlabel("|Attention ratio − 0.5|")
    ax.set_ylabel("Success rate (%)")
    ax.grid(True, ls="--", alpha=0.4)

fig_r2.legend(handles, MODELS_ATTN, loc="lower center", ncol=len(MODELS_ATTN),
              bbox_to_anchor=(0.5, -0.05), frameon=False)
fig_r2.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r2, "R2_success_vs_ratio_dev_by_perturbation.png")


# ── R3: Success vs IoU – faceted by model, colored by suite ──────────────────
fig_r3, axes = plt.subplots(1, len(MODELS_ATTN), figsize=(4*len(MODELS_ATTN), 4.5), sharey=True)
fig_r3.suptitle("Success Rate vs Attention IoU\n(all perturbation types, per model)",
                fontsize=28, fontweight="bold")

suite_handles = [plt.Line2D([0],[0], marker='o', color='w',
                 markerfacecolor=SUITE_COLORS[s], ms=9) for s in ALL_SUITES]

for ax, m in zip(axes, MODELS_ATTN):
    sub = valid(df[df.model == m], "success", "iou")
    for s in ALL_SUITES:
        ss = sub[sub.suite == s]
        ax.scatter(ss.iou, ss.success, c=SUITE_COLORS[s],
                   s=22, alpha=0.75, zorder=4)
    # single regression over all suites for this model
    add_regline(ax, sub.iou.values, sub.success.values, "black", lw=2)
    # annotate Pearson r
    mask = ~(sub.success.isna() | sub.iou.isna())
    if mask.sum() > 2:
        r, p = stats.pearsonr(sub.success[mask], sub.iou[mask])
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                va="top", fontsize=18,
                color="black" if p < 0.05 else "grey")
    ax.set_title(m, fontsize=20)
    ax.set_xlabel("Attention IoU", fontsize=18)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, ls="--", alpha=0.4)

fig_r3.legend(suite_handles, ALL_SUITES, loc="lower center", ncol=5,
              bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=16)
fig_r3.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r3, "R3_success_vs_iou_by_model.png")


# ── R4: Success vs IoU – faceted by suite, colored by model ──────────────────
fig_r4, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
fig_r4.suptitle("Success Rate vs Attention IoU (per suite, colored by model)",
                fontsize=28, fontweight="bold")

for ax, s in zip(axes, ALL_SUITES):
    sub = valid(df[df.suite == s], "success", "iou")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.iou, ms.success, c=MODEL_COLORS[m],
                   s=30, alpha=0.8, zorder=4, label=m)
        add_regline(ax, ms.iou.values, ms.success.values, MODEL_COLORS[m])
    ax.set_title(s, fontsize=20)
    ax.set_xlabel("IoU", fontsize=18)
    ax.set_ylabel("Success (%)", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, ls="--", alpha=0.4)

fig_r4.legend(handles, MODELS_ATTN, loc="lower center", ncol=len(MODELS_ATTN),
              bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=16)
fig_r4.tight_layout(rect=[0, 0.07, 1, 1])
save_fig(fig_r4, "R4_success_vs_iou_by_suite.png")


# ── R5: Pearson r heatmap – success vs IoU, model × perturbation type ────────
corr_rows = []
for m in MODELS_ATTN:
    for pt in PERTURBS:
        sub = valid(df[(df.model == m) & (df.perturbation == pt)],
                    "success", "iou")
        if len(sub) > 2:
            r, p = stats.pearsonr(sub.success, sub.iou)
        else:
            r, p = NaN, NaN
        corr_rows.append({"model": m, "perturbation": pt, "r": r, "p": p})

corr_df = pd.DataFrame(corr_rows)
heatmap_data = corr_df.pivot(index="model", columns="perturbation", values="r")
heatmap_data = heatmap_data.reindex(index=MODELS_ATTN, columns=PERTURBS)
pval_data    = corr_df.pivot(index="model", columns="perturbation", values="p")
pval_data    = pval_data.reindex(index=MODELS_ATTN, columns=PERTURBS)

fig_r5, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn", vmin=-1, vmax=1,
               aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_xticks(range(len(PERTURBS)))
ax.set_xticklabels([p.capitalize() for p in PERTURBS])
ax.set_yticks(range(len(MODELS_ATTN)))
ax.set_yticklabels(MODELS_ATTN)
ax.set_title("Pearson r: Success Rate vs Attention IoU\n"
             "(* = p < 0.05, all suites pooled)", fontsize=9, fontweight="bold")

for i, m in enumerate(MODELS_ATTN):
    for j, pt in enumerate(PERTURBS):
        val = heatmap_data.loc[m, pt]
        pv  = pval_data.loc[m, pt]
        if not np.isnan(val):
            star = "*" if (not np.isnan(pv) and pv < 0.05) else ""
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=9, color="black" if abs(val) < 0.7 else "white")

fig_r5.tight_layout()
save_fig(fig_r5, "R5_pearson_heatmap_success_iou.png")


# ── R6: IoU vs |ratio−0.5| – colored by model, shaped by perturbation ────────
fig_r6, ax = plt.subplots(figsize=(7, 5))
fig_r6.suptitle("Attention IoU vs Ratio Deviation |ratio − 0.5|\n"
                "(both axes reflect degradation; color = model, shape = perturbation)",
                fontsize=9, fontweight="bold")

for pt in PERTURBS:
    sub = valid(df[(df.perturbation == pt) & df.model.isin(MODELS_ATTN)],
                "iou", "ratio_dev")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.iou, ms.ratio_dev,
                   c=MODEL_COLORS[m],
                   marker=PERTURB_MARKERS[pt],
                   s=30, alpha=0.7, zorder=4)

# legend: models (color) + perturbation types (shape)
model_handles = [plt.Line2D([0],[0], marker='o', color='w',
                 markerfacecolor=MODEL_COLORS[m], ms=8, label=m)
                 for m in MODELS_ATTN]
perturb_handles = [plt.Line2D([0],[0], marker=PERTURB_MARKERS[pt],
                   color='grey', ms=8, label=pt.capitalize(), lw=0)
                   for pt in PERTURBS]
ax.legend(handles=model_handles + perturb_handles,
          loc="upper right", fontsize=8, framealpha=0.8)
ax.set_xlabel("Attention IoU  (higher = more focused)")
ax.set_ylabel("|Attention ratio − 0.5|  (higher = more modality-biased)")
ax.grid(True, ls="--", alpha=0.4)
fig_r6.tight_layout()
save_fig(fig_r6, "R6_iou_vs_ratio_dev.png")


# ── R7: Line plots – IoU over conditions, per model, colored by suite ─────────
# Shows the trajectory of IoU as language is perturbed (most readable with lines)
fig_r7, axes = plt.subplots(1, len(MODELS_ATTN), figsize=(4*len(MODELS_ATTN), 4), sharey=False)
fig_r7.suptitle("Attention IoU by Condition – Language Perturbation\n"
                "(one panel per model)", fontsize=10, fontweight="bold")

lang_sub = df[df.perturbation == "language"]
for ax, m in zip(axes, MODELS_ATTN):
    for s in ALL_SUITES:
        ss = lang_sub[(lang_sub.model == m) & (lang_sub.suite == s)].copy()
        ss = ss.set_index("condition").reindex(LANG_CONDS)
        ax.plot(range(len(LANG_CONDS)), ss.iou.values,
                marker="o", ms=5, lw=1.5, color=SUITE_COLORS[s], label=s)
    ax.set_xticks(range(len(LANG_CONDS)))
    ax.set_xticklabels(LANG_CONDS, rotation=30, ha="right")
    ax.set_title(m)
    ax.set_ylabel("Attention IoU")
    ax.grid(True, ls="--", alpha=0.4)

fig_r7.legend(suite_handles, ALL_SUITES, loc="lower center", ncol=5,
              bbox_to_anchor=(0.5, -0.08), frameon=False)
fig_r7.tight_layout(rect=[0, 0.08, 1, 1])
save_fig(fig_r7, "R7_lang_iou_lines_by_model.png")


# ── R8: Paired bar – success vs IoU normalized, per perturbation × suite ──────
# Normalise both metrics to [0,1] within each (model, suite) for direct comparison
fig_r8, axes = plt.subplots(3, 1, figsize=(14, 10))
fig_r8.suptitle("Normalised Success vs IoU (per model–suite pair)\n"
                "Shows whether IoU tracks success across conditions",
                fontsize=10, fontweight="bold")

PERTURB_LABELS = {"language": LANG_CONDS, "visual": VIS_CONDS, "policy": POL_CONDS}

for ax, pt in zip(axes, PERTURBS):
    conds = PERTURB_LABELS[pt]
    x = np.arange(len(conds))
    width = 0.35
    sub_sr, sub_iou = [], []
    for s in ALL_SUITES:
        for m in MODELS_ATTN:
            row = df[(df.perturbation==pt)&(df.suite==s)&(df.model==m)]
            sr_vals  = row.set_index("condition").reindex(conds).success.values.astype(float)
            iou_vals = row.set_index("condition").reindex(conds).iou.values.astype(float)
            sr_norm  = (sr_vals  - np.nanmin(sr_vals))  / (np.nanmax(sr_vals)  - np.nanmin(sr_vals) + 1e-9)
            iou_norm = (iou_vals - np.nanmin(iou_vals)) / (np.nanmax(iou_vals) - np.nanmin(iou_vals) + 1e-9)
            sub_sr.append(sr_norm)
            sub_iou.append(iou_norm)
    mean_sr  = np.nanmean(sub_sr,  axis=0)
    mean_iou = np.nanmean(sub_iou, axis=0)
    std_sr   = np.nanstd(sub_sr,   axis=0)
    std_iou  = np.nanstd(sub_iou,  axis=0)

    ax.bar(x - width/2, mean_sr,  width, color="#4C72B0", alpha=0.8, label="Success rate",
           yerr=std_sr,  capsize=3, zorder=3)
    ax.bar(x + width/2, mean_iou, width, color="#55A868", alpha=0.8, label="Attention IoU",
           yerr=std_iou, capsize=3, zorder=3)
    ax.set_title(pt.capitalize(), fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Normalised value")
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

fig_r8.tight_layout()
save_fig(fig_r8, "R8_normalised_success_vs_iou.png")

# ── R9: Finding-type diagnostic (LIBERO-In domain) ────────────────────────────
# Left panel : Success vs IoU  — shows who succeeds and whether IoU tracks it
# Right panel: Ratio  vs IoU  — shows attention quality space, colored by success
# Key examples from KEY_EXAMPLES are annotated with finding-type markers + labels.

_key_rows = []
for (m, pt, suite, cond, ftype, lbl) in KEY_EXAMPLES:
    row = df[(df.model == m) & (df.perturbation == pt) &
             (df.suite == suite) & (df.condition == cond)]
    if not row.empty:
        rec = row.iloc[0].to_dict()
        rec.update(ftype=ftype, ex_label=lbl)
        _key_rows.append(rec)
key_df = pd.DataFrame(_key_rows)

bg = df[(df.suite == "LIBERO-In domain") & df.model.isin(MODELS_ATTN)]

fig_r9, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
fig_r9.suptitle(
    "Finding-Type Diagnosis: where do models fail and why?\n"
    "(LIBERO-In domain, all perturbation types; key examples annotated)",
    fontsize=10, fontweight="bold")

# ── Left: Success vs IoU ──────────────────────────────────────────────────────
ax = ax_l
bg_v = bg.dropna(subset=["success", "iou"])
for m in MODELS_ATTN:
    ms = bg_v[bg_v.model == m]
    ax.scatter(ms.success, ms.iou, c=MODEL_COLORS[m], s=18, zorder=3,
               alpha=0.7, label=m)
ax.set_xlim(-5, 110)
ax.set_ylim(-0.01, 0.40)
ax.set_xlabel("Success rate (%)")
ax.set_ylabel("Attention IoU")
ax.set_title("Success vs IoU")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

annotate_quadrants(
    ax, x_thresh=40, y_thresh=0.13,
    xlim=(-5, 110), ylim=(-0.01, 0.40),
    quad_labels=[
        "Type 3: Grounded\n(watches & acts right)",
        "Type 1: Shortcutting\n(succeeds w/o visual grounding)",
        "Type 5: Control fails\n(perception OK, action fails)",
        "Type 2/4: Confused\nor not looking",
    ],
    colors=("#4daf4a", "#ff7f00", "#1f77b4", "#9467bd"),
)

for _, row in key_df.iterrows():
    if np.isnan(row.success) or np.isnan(row.iou):
        continue
    st = FINDING_STYLE[row.ftype]
    ax.scatter(row.success, row.iou, c=st["color"], marker=st["marker"],
               s=160, zorder=7, edgecolors="white", linewidths=0.7)
    ax.annotate(
        row.ex_label, xy=(row.success, row.iou),
        xytext=(9, 5), textcoords="offset points",
        fontsize=8, color=st["color"], zorder=8,
        arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.6, alpha=0.7),
    )

# ── Right: Ratio vs IoU (success as color) ───────────────────────────────────
ax = ax_r
bg_v2 = bg.dropna(subset=["ratio", "iou", "success"])
sc = ax.scatter(bg_v2.iou, bg_v2.ratio, c=bg_v2.success, cmap="YlOrRd",
                s=20, zorder=2, alpha=0.75, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label="Success rate (%)", shrink=0.85)
ax.set_xlim(-0.01, 0.40)
ax.set_ylim(0.20, 1.02)
ax.set_xlabel("Attention IoU  (higher = attends to right objects)")
ax.set_ylabel("Attention ratio  (visual / total)")
ax.set_title("Ratio vs IoU  (color = success rate)")
ax.grid(True, ls="--", alpha=0.3)

annotate_quadrants(
    ax, x_thresh=0.13, y_thresh=0.65,
    xlim=(-0.01, 0.40), ylim=(0.20, 1.02),
    quad_labels=[
        "Type 2: Confused\n(watches wrong things)",
        "Type 3/5: Grounded\n(watches right; success varies)",
        "Type 4: Not looking\n(low visual engagement)",
        "Type 1: Shortcutting\n(language-driven success)",
    ],
    colors=("#e41a1c", "#4daf4a", "#9467bd", "#ff7f00"),
)

for _, row in key_df.iterrows():
    if np.isnan(row.get("ratio", NaN)) or np.isnan(row.get("iou", NaN)):
        continue
    st = FINDING_STYLE[row.ftype]
    ax.scatter(row.iou, row.ratio, c=st["color"], marker=st["marker"],
               s=160, zorder=7, edgecolors="white", linewidths=0.7)
    ax.annotate(
        row.ex_label, xy=(row.iou, row.ratio),
        xytext=(6, 5), textcoords="offset points",
        fontsize=8, color=st["color"], zorder=8,
        arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.6, alpha=0.7),
    )

# shared finding-type legend + DreamZero marker
ft_handles = [
    plt.Line2D([0], [0], marker=FINDING_STYLE[ft]["marker"], color="w",
               markerfacecolor=FINDING_STYLE[ft]["color"], ms=8,
               label=FINDING_STYLE[ft]["label"])
    for ft in FINDING_STYLE
]
fig_r9.legend(handles=ft_handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)
fig_r9.tight_layout(rect=[0, 0.10, 1, 1])
save_fig(fig_r9, "R9_finding_types_diagnostic.png")


print(f"\nDone. All plots saved to: {OUT_DIR}")
