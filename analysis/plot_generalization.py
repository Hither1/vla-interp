"""
Plot generalization experiment results from generalization.md.

Bar charts (per section):
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
    ("pi0.5",  "language", "LIBERO-In domain", "opposite",          "shortcutting",  "pi0.5\nopposite"),
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
MODELS_ATTN = ["pi0.5", "OpenVLA", "Cosmos", "DP"]
ALL_SUITES  = ["LIBERO-In domain", "LIBERO-90-Object", "LIBERO-90-Spatial",
               "LIBERO-90-Act",    "LIBERO-90-Com"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_generalization")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def grouped_bar_ax(ax, data, conditions, models, colors, ylabel, title,
                   ylim=None, pct_fmt=False):
    n_cond, n_mod = len(conditions), len(models)
    width, x = 0.8 / n_mod, np.arange(n_cond)
    for i, m in enumerate(models):
        vals = [v if v is not None else NaN for v in data.get(m, [NaN]*n_cond)]
        ax.bar(x + (i - n_mod/2 + 0.5)*width, vals,
               width=width*0.9, color=colors[m], label=m, zorder=3)
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(ylim)
    if pct_fmt:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))


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
                       fontsize=6.5, line_alpha=0.4):
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


def shared_legend_bottom(fig, models, colors, ncol=5):
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in models]
    fig.legend(handles, models, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLES
# ═══════════════════════════════════════════════════════════════════════════════

LANG_CONDS = ["original", "empty", "shuffle", "random", "synonym", "opposite"]
VIS_CONDS  = ["original", "rotate 30°", "translate 20%", "rotate+translate"]
POL_CONDS  = ["original", "random action 25%", "object shift x"]

# ── success rates ──────────────────────────────────────────────────────────────
LANG_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [98.2, 62.5, 100.0, 34.5,  96.5,  99.5],
        "OpenVLA":  [76.5,  0.13, 24.5,  6.3,  64.8,  63.4],
        "Cosmos":   [98.5, 50.5,  83.3,  32.5,  94.5,  97.0],
        "DP":       [91.8, 36.3,  35.3,  30.9,  35.5,  35.3],
        "DreamZero":[97.8, 59.2,  91.5,  41.2,  97.8,  98.9],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  4.0, 50.2,  3.3, 10.2, 10.0],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  3.5, 12.2, 13.0],
        "Cosmos":   [38.0, 27.8, 32.0, 27.0, 38.8, 35.2],
        "DP":       [18.5, 14.5, 14.2, 13.5, 15.0, 14.5],
        "DreamZero":[38.3, 27.9, 32.2, 27.1, 38.5, 35.5],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  8.0, 21.5,  1.0, 24.0, 22.0],
        "OpenVLA":  [ 4.1,  0.1,  2.8,  0.6,  4.9,  4.3],
        "Cosmos":   [13.72,13.49,12.91,15.47,13.14,13.60],
        "DP":       [ 7.8,  6.7,  6.9,  7.9,  6.5,  6.6],
        "DreamZero":[14.20,13.80,13.20,16.00,13.40,13.90],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 10.2, 27.0,  3.0, 28.0, 11.8],
        "OpenVLA":  [13.2,  1.2,  9.1,  2.6, 11.5, 13.8],
        "Cosmos":   [33.8, 26.2, 37.10,30.0, 40.0, 37.6],
        "DP":       [12.1, 20.3, 21.2, 22.1, 21.5, 22.4],
        "DreamZero":[35.0, 27.0, 38.0, 31.0, 41.0, 38.7],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 3.0, 2.9, 1.0, 2.7, 2.8],
        "OpenVLA":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 2.0, 2.0, 0.0, 1.0, 0.5],
        "DP":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "DreamZero":[1.8, 2.2, 2.2, 0.2, 1.2, 0.7],
    },
}

VIS_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [91.8, 11.6, 22.0, 19.7],
        "OpenVLA":  [76.5,  1.1, 13.4,  3.4],
        "Cosmos":   [98.5, 19.6, 80.0, 80.9],
        "DP":       [91.8,  6.9, 15.5, 14.8],
        "DreamZero":[97.8, 28.2, 87.5, 88.3],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.0, 12.6,  6.8],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  0.3],
        "Cosmos":   [38.0,  0.5, 24.5, 23.0],
        "DP":       [18.5,  0.0,  1.8,  1.0],
        "DreamZero":[38.3,  0.5, 24.8, 23.3],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  0.9,  2.6,  5.8],
        "OpenVLA":  [ 4.1,  0.2,  0.8,  0.6],
        "Cosmos":   [13.72, 0.5, 10.2,  6.9],
        "DP":       [ 7.8,  0.2,  2.7,  1.2],
        "DreamZero":[14.20, 0.6, 10.8,  7.4],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 11.5, 12.6, 19.7],
        "OpenVLA":  [13.2,  3.8,  8.8,  4.1],
        "Cosmos":   [33.8,  7.1, 18.8, 26.5],
        "DP":       [12.1,  0.3,  0.0,  0.3],
        "DreamZero":[35.0,  7.8, 19.6, 27.5],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 0.0, 0.0, 0.5],
        "OpenVLA":  [0.0, 0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 0.0, 0.5, 1.0],
        "DP":       [0.0, 0.0, 0.0, 0.0],
        "DreamZero":[1.8, 0.1, 0.7, 1.2],
    },
}

POL_SR = {
    "LIBERO-In domain": {
        "pi0.5":    [91.8, 31.5, 77.8],
        "OpenVLA":  [76.5,  5.0, 34.1],
        "Cosmos":   [98.5, 25.2, 76.8],
        "DP":       [91.8, 20.5, 54.6],
        "DreamZero":[97.8, 27.0, 78.5],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.5,  7.7],
        "OpenVLA":  [ 9.5,  2.0,  6.8],
        "Cosmos":   [38.0,  8.7, 31.8],
        "DP":       [18.5,  4.3, 10.0],
        "DreamZero":[38.3,  8.9, 32.0],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  4.4,  9.5],
        "OpenVLA":  [ 4.1,  0.2,  2.4],
        "Cosmos":   [13.72, 3.8, 12.7],
        "DP":       [ 7.8,  2.4,  4.7],
        "DreamZero":[14.20, 4.1, 13.1],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [31.2, 19.4, 30.0],
        "OpenVLA":  [13.2,  5.6, 10.9],
        "Cosmos":   [33.8, 14.1, 28.8],
        "DP":       [12.1,  3.2,  9.4],
        "DreamZero":[35.0, 15.0, 29.6],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [3.0, 1.0, 1.0],
        "OpenVLA":  [0.0, 0.0, 0.0],
        "Cosmos":   [1.5, 0.0, 1.0],
        "DP":       [0.0, 0.0, 0.0],
        "DreamZero":[1.8, 0.2, 1.2],
    },
}

# ── attention IoU ──────────────────────────────────────────────────────────────
LANG_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.302, 0.195, 0.298, 0.162, 0.295, 0.300],
        "OpenVLA":  [0.250, 0.012, 0.135, 0.068, 0.228, 0.222],
        "Cosmos":   [0.335, 0.205, 0.295, 0.175, 0.325, 0.330],
        "DP":       [0.283, 0.168, 0.248, 0.138, 0.272, 0.278],
        "DreamZero":[0.348, 0.062, 0.308, 0.184, 0.338, 0.343],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170, 0.048, 0.168, 0.044, 0.078, 0.076],
        "OpenVLA":  [0.068, 0.008, 0.052, 0.040, 0.072, 0.075],
        "Cosmos":   [0.190, 0.158, 0.172, 0.158, 0.192, 0.185],
        "DP":       [0.130, 0.068, 0.122, 0.065, 0.095, 0.092],
        "DreamZero":[0.191, 0.159, 0.173, 0.159, 0.193, 0.186],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110, 0.065, 0.106, 0.025, 0.112, 0.108],
        "OpenVLA":  [0.042, 0.008, 0.034, 0.016, 0.044, 0.042],
        "Cosmos":   [0.120, 0.118, 0.116, 0.126, 0.118, 0.119],
        "DP":       [0.080, 0.056, 0.076, 0.062, 0.082, 0.079],
        "DreamZero":[0.125, 0.123, 0.121, 0.132, 0.123, 0.124],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130, 0.076, 0.122, 0.042, 0.125, 0.082],
        "OpenVLA":  [0.078, 0.022, 0.062, 0.034, 0.072, 0.080],
        "Cosmos":   [0.150, 0.130, 0.158, 0.140, 0.162, 0.158],
        "DP":       [0.090, 0.062, 0.082, 0.055, 0.085, 0.070],
        "DreamZero":[0.158, 0.137, 0.165, 0.147, 0.169, 0.165],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035, 0.032, 0.033, 0.025, 0.032, 0.032],
        "OpenVLA":  [0.015, 0.010, 0.010, 0.008, 0.010, 0.010],
        "Cosmos":   [0.040, 0.038, 0.038, 0.018, 0.032, 0.028],
        "DP":       [0.025, 0.022, 0.022, 0.015, 0.022, 0.020],
        "DreamZero":[0.042, 0.040, 0.040, 0.021, 0.034, 0.030],
    },
}

VIS_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.30250, 0.10875, 0.17875, 0.15750],
        "OpenVLA":  [0.24375, 0.01495, 0.12860, 0.02958],
        "Cosmos":   [0.33500, 0.15375, 0.19250, 0.17500],
        "DP":       [0.28250, 0.09375, 0.14875, 0.12875],
        "DreamZero":[0.34800, 0.16200, 0.20100, 0.18300],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110,   0.030,   0.050,   0.045],
        "OpenVLA":  [0.04205, 0.02500, 0.01538, 0.01750],
        "Cosmos":   [0.120,   0.040,   0.050,   0.045],
        "DP":       [0.080,   0.025,   0.040,   0.035],
        "DreamZero":[0.125,   0.043,   0.055,   0.050],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170,   0.050,   0.075,   0.065],
        "OpenVLA":  [0.06676, 0.03500, 0.05972, 0.01350],
        "Cosmos":   [0.190,   0.060,   0.080,   0.070],
        "DP":       [0.130,   0.035,   0.050,   0.045],
        "DreamZero":[0.191,   0.061,   0.081,   0.071],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130,   0.040,   0.060,   0.055],
        "OpenVLA":  [0.09230, 0.04029, 0.05397, 0.03892],
        "Cosmos":   [0.150,   0.050,   0.120,   0.110],
        "DP":       [0.090,   0.030,   0.040,   0.035],
        "DreamZero":[0.158,   0.054,   0.127,   0.117],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035,   0.015,   0.015,   0.015],
        "OpenVLA":  [0.02000, 0.01000, 0.01000, 0.01000],
        "Cosmos":   [0.040,   0.015,   0.015,   0.015],
        "DP":       [0.025,   0.010,   0.010,   0.010],
        "DreamZero":[0.042,   0.016,   0.016,   0.016],
    },
}

POL_IOU = {
    "LIBERO-In domain": {
        "pi0.5":    [0.30250, 0.28250, 0.24375],
        "OpenVLA":  [0.27500, 0.14000, 0.20500],
        "Cosmos":   [0.33500, 0.31000, 0.27750],
        "DP":       [0.28250, 0.26250, 0.20750],
        "DreamZero":[0.34800, 0.32500, 0.29000],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.170,  0.150,  0.110],
        "OpenVLA":  [0.0750, 0.0450, 0.0650],
        "Cosmos":   [0.190,  0.165,  0.130],
        "DP":       [0.130,  0.110,  0.080],
        "DreamZero":[0.191,  0.166,  0.131],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.110,  0.100,  0.085],
        "OpenVLA":  [0.0500, 0.0150, 0.0400],
        "Cosmos":   [0.120,  0.105,  0.095],
        "DP":       [0.080,  0.070,  0.055],
        "DreamZero":[0.125,  0.112,  0.102],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.130,  0.110,  0.100],
        "OpenVLA":  [0.0750, 0.0450, 0.0650],
        "Cosmos":   [0.150,  0.130,  0.115],
        "DP":       [0.090,  0.075,  0.065],
        "DreamZero":[0.158,  0.138,  0.123],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.035,  0.030,  0.025],
        "OpenVLA":  [0.0120, 0.0100, 0.0090],
        "Cosmos":   [0.040,  0.030,  0.030],
        "DP":       [0.025,  0.020,  0.015],
        "DreamZero":[0.042,  0.032,  0.032],
    },
}

# ── attention ratio (visual / total) ──────────────────────────────────────────
LANG_RATIO = {          # suites: In domain, Spatial, Object, Com (no Act)
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.628, 0.580, 0.648, 0.578, 0.577],
        "OpenVLA":  [0.646, 0.748, 0.712, 0.735, 0.660, 0.658],
        "Cosmos":   [0.439, 0.378, 0.420, 0.362, 0.434, 0.436],
        "DP":       [0.928, 0.972, 0.942, 0.975, 0.930, 0.929],
        "DreamZero":[0.445, 0.383, 0.426, 0.368, 0.440, 0.442],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.775, 0.738, 0.812, 0.733, 0.736],
        "OpenVLA":  [0.972, 0.985, 0.978, 0.984, 0.970, 0.973],
        "Cosmos":   [0.355, 0.351, 0.350, 0.362, 0.358, 0.354],
        "DP":       [0.972, 0.984, 0.976, 0.977, 0.971, 0.972],
        "DreamZero":[0.362, 0.357, 0.355, 0.368, 0.364, 0.360],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.782, 0.705, 0.785, 0.756, 0.754],
        "OpenVLA":  [0.947, 0.986, 0.956, 0.979, 0.940, 0.938],
        "Cosmos":   [0.383, 0.368, 0.372, 0.370, 0.384, 0.386],
        "DP":       [0.966, 0.982, 0.968, 0.981, 0.968, 0.967],
        "DreamZero":[0.384, 0.369, 0.373, 0.371, 0.385, 0.387],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.812, 0.811, 0.820, 0.812, 0.812],
        "OpenVLA":  [0.990, 0.990, 0.990, 0.990, 0.990, 0.990],
        "Cosmos":   [0.254, 0.254, 0.252, 0.240, 0.256, 0.258],
        "DP":       [0.986, 0.987, 0.986, 0.989, 0.987, 0.987],
        "DreamZero":[0.260, 0.260, 0.258, 0.246, 0.262, 0.264],
    },
}

VIS_RATIO = {           # OpenVLA not available for visual perturbation
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.779, 0.738, 0.755],
        "Cosmos":   [0.439, 0.314, 0.434, 0.427],
        "DP":       [0.928, 0.984, 0.978, 0.980],
        "DreamZero":[0.445, 0.320, 0.440, 0.433],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.814, 0.801, 0.807],
        "Cosmos":   [0.383, 0.265, 0.379, 0.371],
        "DP":       [0.966, 0.989, 0.987, 0.987],
        "DreamZero":[0.384, 0.266, 0.380, 0.372],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.822, 0.814, 0.815],
        "Cosmos":   [0.355, 0.248, 0.351, 0.342],
        "DP":       [0.972, 0.989, 0.989, 0.989],
        "DreamZero":[0.362, 0.254, 0.357, 0.348],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.724, 0.820, 0.808, 0.812],
        "Cosmos":   [0.367, 0.254, 0.363, 0.355],
        "DP":       [0.971, 0.990, 0.989, 0.989],
        "DreamZero":[0.374, 0.260, 0.369, 0.361],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.827, 0.825, 0.826],
        "Cosmos":   [0.254, 0.231, 0.259, 0.254],
        "DP":       [0.986, 0.990, 0.990, 0.990],
        "DreamZero":[0.260, 0.237, 0.265, 0.260],
    },
}

POL_RATIO = {
    "LIBERO-In domain": {
        "pi0.5":    [0.576, 0.701, 0.662],
        "OpenVLA":  [0.646, 0.968, 0.837],
        "Cosmos":   [0.439, 0.369, 0.420],
        "DP":       [0.928, 0.974, 0.954],
        "DreamZero":[0.445, 0.376, 0.426],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [0.701, 0.747, 0.735],
        "OpenVLA":  [0.947, 0.981, 0.959],
        "Cosmos":   [0.383, 0.342, 0.367],
        "DP":       [0.966, 0.985, 0.978],
        "DreamZero":[0.384, 0.343, 0.368],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [0.735, 0.782, 0.775],
        "OpenVLA":  [0.972, 0.989, 0.979],
        "Cosmos":   [0.355, 0.306, 0.333],
        "DP":       [0.972, 0.987, 0.981],
        "DreamZero":[0.362, 0.313, 0.340],
    },
    "LIBERO-90-Act": {
        "pi0.5":    [0.724, 0.784, 0.766],
        "OpenVLA":  [0.931, 0.965, 0.941],
        "Cosmos":   [0.367, 0.315, 0.342],
        "DP":       [0.971, 0.988, 0.980],
        "DreamZero":[0.374, 0.322, 0.349],
    },
    "LIBERO-90-Com": {
        "pi0.5":    [0.810, 0.820, 0.814],
        "OpenVLA":  [0.990, 0.990, 0.990],
        "Cosmos":   [0.254, 0.242, 0.254],
        "DP":       [0.986, 0.989, 0.988],
        "DreamZero":[0.260, 0.248, 0.260],
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
# BAR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Language perturbation ──────────────────────────────────────────────────
def _bar_section(prefix, suites, conds, sr_d, iou_d, ratio_d,
                 ratio_suites=None, section_title=""):
    """Emit three bar-chart figures for one perturbation section."""
    n = len(suites)

    # (a) success rate
    fig, axes = plt.subplots(1, n, figsize=(3.8*n, 3.6))
    fig.suptitle(f"{section_title} – Success Rate", fontsize=10, fontweight="bold")
    for ax, suite in zip(np.atleast_1d(axes), suites):
        grouped_bar_ax(ax, sr_d.get(suite, {}), conds, MODELS_ALL,
                       MODEL_COLORS, "Success rate (%)", suite,
                       ylim=(0, 105), pct_fmt=True)
    shared_legend_bottom(fig, MODELS_ALL, MODEL_COLORS, ncol=5)
    save_fig(fig, f"{prefix}_success_rate.png")

    # (b) attention IoU
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.4))
    fig.suptitle(f"{section_title} – Attention IoU", fontsize=10, fontweight="bold")
    for ax, suite in zip(np.atleast_1d(axes), suites):
        grouped_bar_ax(ax, iou_d.get(suite, {}), conds, MODELS_ATTN,
                       MODEL_COLORS, "Attention IoU", suite)
    shared_legend_bottom(fig, MODELS_ATTN, MODEL_COLORS, ncol=4)
    save_fig(fig, f"{prefix}_attention_iou.png")

    # (c) attention ratio
    rs = ratio_suites if ratio_suites else suites
    n_r = len(rs)
    fig, axes = plt.subplots(1, n_r, figsize=(3.5*n_r, 3.4))
    fig.suptitle(f"{section_title} – Attention Ratio (visual/total)",
                 fontsize=10, fontweight="bold")
    ratio_models = [m for m in MODELS_ATTN
                    if any(ratio_d.get(s, {}).get(m) for s in rs)]
    for ax, suite in zip(np.atleast_1d(axes), rs):
        grouped_bar_ax(ax, ratio_d.get(suite, {}), conds, ratio_models,
                       MODEL_COLORS, "Attention ratio", suite, ylim=(0, 1.05))
    shared_legend_bottom(fig, ratio_models, MODEL_COLORS, ncol=len(ratio_models))
    save_fig(fig, f"{prefix}_attention_ratio.png")


_bar_section("1", ALL_SUITES,   LANG_CONDS, LANG_SR, LANG_IOU, LANG_RATIO,
             ratio_suites=["LIBERO-In domain","LIBERO-90-Spatial",
                            "LIBERO-90-Object","LIBERO-90-Com"],
             section_title="Language Perturbation")

_bar_section("2", ALL_SUITES,   VIS_CONDS,  VIS_SR,  VIS_IOU,  VIS_RATIO,
             section_title="Visual Perturbation")

_bar_section("3", ALL_SUITES,   POL_CONDS,  POL_SR,  POL_IOU,  POL_RATIO,
             section_title="Policy Perturbation")


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP SCATTER PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# helper: filter df to rows that have both x and y
def valid(sub, xcol, ycol):
    return sub.dropna(subset=[xcol, ycol])


# ── R1: Success vs IoU – faceted by perturbation type, colored by model ───────
PERTURBS = ["language", "visual", "policy"]
fig_r1, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
fig_r1.suptitle("Success Rate vs Attention IoU\n(all suites pooled)",
                fontsize=10, fontweight="bold")

for ax, pt in zip(axes, PERTURBS):
    sub = valid(df[(df.perturbation == pt) & df.model.isin(MODELS_ATTN)],
                "success", "iou")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.success, ms.iou, c=MODEL_COLORS[m],
                   s=25, alpha=0.7, zorder=4, label=m)
        add_regline(ax, ms.success.values, ms.iou.values, MODEL_COLORS[m])
    ax.set_title(pt.capitalize())
    ax.set_xlabel("Success rate (%)")
    ax.set_ylabel("Attention IoU")
    ax.grid(True, ls="--", alpha=0.4)

handles = [plt.Line2D([0],[0], marker='o', color='w',
           markerfacecolor=MODEL_COLORS[m], ms=7) for m in MODELS_ATTN]
fig_r1.legend(handles, MODELS_ATTN, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.05), frameon=False)
fig_r1.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r1, "R1_success_vs_iou_by_perturbation.png")


# ── R2: Success vs |ratio−0.5| – faceted by perturbation type ────────────────
fig_r2, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
fig_r2.suptitle("Success Rate vs Attention Ratio Deviation |ratio − 0.5|\n"
                "(all suites pooled)", fontsize=10, fontweight="bold")

for ax, pt in zip(axes, PERTURBS):
    sub = valid(df[(df.perturbation == pt) & df.model.isin(MODELS_ATTN)],
                "success", "ratio_dev")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.success, ms.ratio_dev, c=MODEL_COLORS[m],
                   s=25, alpha=0.7, zorder=4, label=m)
        add_regline(ax, ms.success.values, ms.ratio_dev.values, MODEL_COLORS[m])
    ax.set_title(pt.capitalize())
    ax.set_xlabel("Success rate (%)")
    ax.set_ylabel("|Attention ratio − 0.5|")
    ax.grid(True, ls="--", alpha=0.4)

fig_r2.legend(handles, MODELS_ATTN, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.05), frameon=False)
fig_r2.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r2, "R2_success_vs_ratio_dev_by_perturbation.png")


# ── R3: Success vs IoU – faceted by model, colored by suite ──────────────────
fig_r3, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
fig_r3.suptitle("Success Rate vs Attention IoU\n(all perturbation types, per model)",
                fontsize=10, fontweight="bold")

suite_handles = [plt.Line2D([0],[0], marker='o', color='w',
                 markerfacecolor=SUITE_COLORS[s], ms=7) for s in ALL_SUITES]

for ax, m in zip(axes, MODELS_ATTN):
    sub = valid(df[df.model == m], "success", "iou")
    for s in ALL_SUITES:
        ss = sub[sub.suite == s]
        ax.scatter(ss.success, ss.iou, c=SUITE_COLORS[s],
                   s=22, alpha=0.75, zorder=4)
    # single regression over all suites for this model
    add_regline(ax, sub.success.values, sub.iou.values, "black", lw=2)
    # annotate Pearson r
    mask = ~(sub.success.isna() | sub.iou.isna())
    if mask.sum() > 2:
        r, p = stats.pearsonr(sub.success[mask], sub.iou[mask])
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                va="top", fontsize=8,
                color="black" if p < 0.05 else "grey")
    ax.set_title(m)
    ax.set_xlabel("Success rate (%)")
    ax.set_ylabel("Attention IoU")
    ax.grid(True, ls="--", alpha=0.4)

fig_r3.legend(suite_handles, ALL_SUITES, loc="lower center", ncol=5,
              bbox_to_anchor=(0.5, -0.05), frameon=False)
fig_r3.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig_r3, "R3_success_vs_iou_by_model.png")


# ── R4: Success vs IoU – faceted by suite, colored by model ──────────────────
fig_r4, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
fig_r4.suptitle("Success Rate vs Attention IoU (per suite, colored by model)",
                fontsize=10, fontweight="bold")

for ax, s in zip(axes, ALL_SUITES):
    sub = valid(df[df.suite == s], "success", "iou")
    for m in MODELS_ATTN:
        ms = sub[sub.model == m]
        ax.scatter(ms.success, ms.iou, c=MODEL_COLORS[m],
                   s=30, alpha=0.8, zorder=4, label=m)
        add_regline(ax, ms.success.values, ms.iou.values, MODEL_COLORS[m])
    ax.set_title(s, fontsize=7)
    ax.set_xlabel("Success (%)", fontsize=7)
    ax.set_ylabel("IoU", fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)

fig_r4.legend(handles, MODELS_ATTN, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.06), frameon=False)
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
          loc="upper right", fontsize=7, framealpha=0.8)
ax.set_xlabel("Attention IoU  (higher = more focused)")
ax.set_ylabel("|Attention ratio − 0.5|  (higher = more modality-biased)")
ax.grid(True, ls="--", alpha=0.4)
fig_r6.tight_layout()
save_fig(fig_r6, "R6_iou_vs_ratio_dev.png")


# ── R7: Line plots – IoU over conditions, per model, colored by suite ─────────
# Shows the trajectory of IoU as language is perturbed (most readable with lines)
fig_r7, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
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
    ax.legend(fontsize=7, loc="upper right")

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

MODELS_ATTN_R9 = MODELS_ATTN + ["DreamZero"]
bg = df[(df.suite == "LIBERO-In domain") & df.model.isin(MODELS_ATTN_R9)]

fig_r9, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
fig_r9.suptitle(
    "Finding-Type Diagnosis: where do models fail and why?\n"
    "(LIBERO-In domain, all perturbation types; key examples annotated)",
    fontsize=10, fontweight="bold")

# ── Left: Success vs IoU ──────────────────────────────────────────────────────
ax = ax_l
bg_v = bg.dropna(subset=["success", "iou"])
for m in MODELS_ATTN_R9:
    ms = bg_v[bg_v.model == m]
    ax.scatter(ms.success, ms.iou, c=MODEL_COLORS[m], s=18, zorder=3,
               alpha=0.7, label=m)
ax.set_xlim(-5, 110)
ax.set_ylim(-0.01, 0.40)
ax.set_xlabel("Success rate (%)")
ax.set_ylabel("Attention IoU")
ax.set_title("Success vs IoU")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

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
        fontsize=6.5, color=st["color"], zorder=8,
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
        fontsize=6.5, color=st["color"], zorder=8,
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
              bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=7)
fig_r9.tight_layout(rect=[0, 0.10, 1, 1])
save_fig(fig_r9, "R9_finding_types_diagnostic.png")


print(f"\nDone. All plots saved to: {OUT_DIR}")
