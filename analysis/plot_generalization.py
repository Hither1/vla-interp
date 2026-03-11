"""
Plot generalization experiment results from generalization.md.

Generates three figures:
  1. Language perturbation – success rate
  2. Visual perturbation   – success rate, attention IoU, attention ratio
  3. Policy perturbation   – success rate, attention IoU, attention ratio
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── colour / style ──────────────────────────────────────────────────────────
MODEL_COLORS = {
    "pi0.5":    "#4C72B0",
    "OpenVLA":  "#DD8452",
    "Cosmos":   "#55A868",
    "DP":       "#C44E52",
    "DreamZero":"#8172B2",
}
MODELS_ALL   = ["pi0.5", "OpenVLA", "Cosmos", "DreamZero", "DP"]
MODELS_ATTN  = ["pi0.5", "Cosmos", "DP"]   # OpenVLA/DreamZero missing in most attn tables

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_generalization")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

def pct(x):
    """Strip % and convert to float 0-100."""
    if isinstance(x, str):
        return float(x.strip("%"))
    return float(x)


def grouped_bar_ax(ax, data, conditions, models, colors, ylabel, title,
                   ylim=None, pct_fmt=False):
    """
    data   : dict[model] -> list of values (one per condition); None = missing
    """
    n_cond = len(conditions)
    n_mod  = len(models)
    width  = 0.8 / n_mod
    x      = np.arange(n_cond)

    for i, model in enumerate(models):
        vals = data.get(model, [None] * n_cond)
        ys   = [v if v is not None else np.nan for v in vals]
        bars = ax.bar(x + (i - n_mod / 2 + 0.5) * width, ys,
                      width=width * 0.9, color=colors[model],
                      label=model, zorder=3)

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


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LANGUAGE PERTURBATION – success rate
# ═══════════════════════════════════════════════════════════════════════════════

LANG_CONDITIONS = ["original", "empty", "shuffle", "random", "synonym", "opposite"]
LANG_SUITES = [
    "LIBERO-10",
    "LIBERO-90-Object",
    "LIBERO-90-Spatial",
    "LIBERO-90-Act",
    "LIBERO-90-Com",
]

# success rate (%) – rows: [original, empty, shuffle, random, synonym, opposite]
LANG_SR = {
    "LIBERO-10": {
        "pi0.5":    [98.2, 62.5, 100.0, 34.5,  96.5,  99.5],
        "OpenVLA":  [76.5,  0.13, 24.5,   6.3,  64.8,  63.4],
        "Cosmos":   [98.5, 50.5,  83.3,  32.5,  94.5,  97.0],
        "DP":       [91.8, 36.3,  35.3,  30.9,  35.5,  35.3],
        "DreamZero":[99.0, 52.0,  84.5,  34.0,  95.2,  97.6],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  4.0, 50.2,  3.3, 10.2, 10.0],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  3.5, 12.2, 13.0],
        "Cosmos":   [38.0, 27.8, 32.0, 27.0, 38.8, 35.2],
        "DP":       [18.5, 14.5, 14.2, 13.5, 15.0, 14.5],
        "DreamZero":[39.5, 28.5, 33.0, 27.6, 39.2, 36.0],
    },
    "LIBERO-90-Spatial": {
        "pi0.5":    [23.0,  8.0, 21.5,  1.0, 24.0, 22.0],
        "OpenVLA":  [ 4.1,  0.1,  2.8,  0.6,  4.9,  4.3],
        "Cosmos":   [13.72,13.49,12.91,15.47,13.14,13.6],
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

n_suites = len(LANG_SUITES)
fig1, axes = plt.subplots(1, n_suites, figsize=(4 * n_suites, 3.6), sharey=False)
fig1.suptitle("Language Perturbation – Success Rate", fontsize=10, fontweight="bold")

for ax, suite in zip(axes, LANG_SUITES):
    grouped_bar_ax(
        ax, LANG_SR[suite], LANG_CONDITIONS, MODELS_ALL,
        MODEL_COLORS, "Success rate (%)", suite,
        ylim=(0, 105), pct_fmt=True,
    )

# single shared legend
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ALL]
fig1.legend(handles, MODELS_ALL, loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.04), frameon=False)
fig1.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig1, "1_lang_success_rate.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  VISUAL PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════════

VIS_CONDITIONS = ["original", "rotate 30°", "translate 20%", "rotate+translate"]
VIS_SUITES_SR  = ["LIBERO-10", "LIBERO-90-Object", "LIBERO-90-Spatial",
                  "LIBERO-90-Act", "LIBERO-90-Com"]

VIS_SR = {
    "LIBERO-10": {
        "pi0.5":    [91.8, 11.6, 22.0, 19.7],
        "OpenVLA":  [76.5,  1.1, 13.4,  3.4],
        "Cosmos":   [98.5, 19.6, 80.0, 80.9],
        "DP":       [91.8,  6.9, 15.5, 14.8],
        "DreamZero":[99.0, 21.0, 81.8, 82.5],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.0, 12.6,  6.8],
        "OpenVLA":  [ 9.5,  0.0,  6.0,  0.3],
        "Cosmos":   [38.0,  0.5, 24.5, 23.0],
        "DP":       [18.5,  0.0,  1.8,  1.0],
        "DreamZero":[39.5,  0.7, 25.6, 24.2],
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

# Attention IoU – visual perturbation
VIS_SUITES_IOU = ["LIBERO-90-Spatial", "LIBERO-90-Object",
                  "LIBERO-90-Act", "LIBERO-90-Com"]

VIS_IOU = {
    "LIBERO-90-Spatial": {
        "pi0.5": [0.11, 0.03, 0.05, 0.045],
        "Cosmos":[0.12, 0.04, 0.05, 0.045],
        "DP":    [0.08, 0.025,0.04, 0.035],
    },
    "LIBERO-90-Object": {
        "pi0.5": [0.17, 0.05, 0.075,0.065],
        "Cosmos":[0.19, 0.06, 0.08, 0.07],
        "DP":    [0.13, 0.035,0.05, 0.045],
    },
    "LIBERO-90-Act": {
        "pi0.5": [0.13, 0.04, 0.06, 0.055],
        "Cosmos":[0.15, 0.05, 0.12, 0.11],
        "DP":    [0.09, 0.03, 0.04, 0.035],
    },
    "LIBERO-90-Com": {
        "pi0.5": [0.035, 0.015, 0.015, 0.015],
        "Cosmos":[0.04,  0.015, 0.015, 0.015],
        "DP":    [0.025, 0.01,  0.01,  0.01],
    },
}

# Attention Ratio – visual perturbation
VIS_SUITES_AR = ["LIBERO-10", "LIBERO-90-Object", "LIBERO-90-Spatial",
                 "LIBERO-90-Act", "LIBERO-90-Com"]

VIS_AR = {
    "LIBERO-10": {
        "pi0.5": [0.576, 0.779, 0.738, 0.755],
        "Cosmos":[0.439, 0.314, 0.434, 0.427],
        "DP":    [0.928, 0.984, 0.978, 0.980],
    },
    "LIBERO-90-Object": {
        "pi0.5": [0.701, 0.814, 0.801, 0.807],
        "Cosmos":[0.383, 0.265, 0.379, 0.371],
        "DP":    [0.966, 0.989, 0.987, 0.987],
    },
    "LIBERO-90-Spatial": {
        "pi0.5": [0.735, 0.822, 0.814, 0.815],
        "Cosmos":[0.355, 0.248, 0.351, 0.342],
        "DP":    [0.972, 0.989, 0.989, 0.989],
    },
    "LIBERO-90-Act": {
        "pi0.5": [0.724, 0.820, 0.808, 0.812],
        "Cosmos":[0.367, 0.254, 0.363, 0.355],
        "DP":    [0.971, 0.990, 0.989, 0.989],
    },
    "LIBERO-90-Com": {
        "pi0.5": [0.810, 0.827, 0.825, 0.826],
        "Cosmos":[0.254, 0.231, 0.259, 0.254],
        "DP":    [0.986, 0.990, 0.990, 0.990],
    },
}

# --- figure 2a: success rate ---
n = len(VIS_SUITES_SR)
fig2a, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
fig2a.suptitle("Visual Perturbation – Success Rate", fontsize=10, fontweight="bold")
for ax, suite in zip(axes, VIS_SUITES_SR):
    grouped_bar_ax(ax, VIS_SR[suite], VIS_CONDITIONS, MODELS_ALL,
                   MODEL_COLORS, "Success rate (%)", suite,
                   ylim=(0, 105), pct_fmt=True)
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ALL]
fig2a.legend(handles, MODELS_ALL, loc="lower center", ncol=5,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig2a.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig2a, "2a_vis_success_rate.png")

# --- figure 2b: attention IoU ---
n = len(VIS_SUITES_IOU)
fig2b, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
fig2b.suptitle("Visual Perturbation – Attention IoU", fontsize=10, fontweight="bold")
for ax, suite in zip(axes, VIS_SUITES_IOU):
    grouped_bar_ax(ax, VIS_IOU[suite], VIS_CONDITIONS, MODELS_ATTN,
                   MODEL_COLORS, "Attention IoU", suite)
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ATTN]
fig2b.legend(handles, MODELS_ATTN, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig2b.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig2b, "2b_vis_attention_iou.png")

# --- figure 2c: attention ratio ---
n = len(VIS_SUITES_AR)
fig2c, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
fig2c.suptitle("Visual Perturbation – Attention Ratio (visual/total)",
               fontsize=10, fontweight="bold")
for ax, suite in zip(axes, VIS_SUITES_AR):
    grouped_bar_ax(ax, VIS_AR[suite], VIS_CONDITIONS, MODELS_ATTN,
                   MODEL_COLORS, "Attention ratio", suite, ylim=(0, 1.05))
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ATTN]
fig2c.legend(handles, MODELS_ATTN, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig2c.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig2c, "2c_vis_attention_ratio.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  POLICY PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════════

POL_CONDITIONS = ["original", "random action 25%", "object shift x"]
POL_SUITES = ["LIBERO-10", "LIBERO-90-Object", "LIBERO-90-Spatial",
              "LIBERO-90-Act", "LIBERO-90-Com"]

POL_SR = {
    "LIBERO-10": {
        "pi0.5":    [91.8, 31.5, 77.8],
        "OpenVLA":  [76.5,  5.0, 34.1],
        "Cosmos":   [98.5, 25.2, 76.8],
        "DP":       [91.8, 20.5, 54.6],
        "DreamZero":[99.0, 27.0, 78.5],
    },
    "LIBERO-90-Object": {
        "pi0.5":    [49.5,  5.5,  7.7],
        "OpenVLA":  [ 9.5,  2.0,  6.8],
        "Cosmos":   [38.0,  8.7, 31.8],
        "DP":       [18.5,  4.3, 10.0],
        "DreamZero":[39.5,  9.4, 32.8],
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

# Attention IoU – policy perturbation
POL_SUITES_IOU = ["LIBERO-In domain", "LIBERO-90-Object", "LIBERO-90-Spatial",
                  "LIBERO-90-Act", "LIBERO-90-Com"]

POL_IOU = {
    "LIBERO-In domain": {
        "pi0.5": [0.31, 0.29, 0.26],
    },
    "LIBERO-90-Object": {
        "pi0.5": [0.17, 0.15, 0.11],
        "Cosmos":[0.19, 0.165,0.13],
        "DP":    [0.13, 0.11, 0.08],
    },
    "LIBERO-90-Spatial": {
        "pi0.5": [0.11, 0.10, 0.085],
        "Cosmos":[0.12, 0.105,0.095],
        "DP":    [0.08, 0.07, 0.055],
    },
    "LIBERO-90-Act": {
        "pi0.5": [0.13, 0.11, 0.10],
        "Cosmos":[0.15, 0.13, 0.115],
        "DP":    [0.09, 0.075,0.065],
    },
    "LIBERO-90-Com": {
        "pi0.5": [0.035, 0.03, 0.025],
        "Cosmos":[0.04,  0.03, 0.03],
        "DP":    [0.025, 0.02, 0.015],
    },
}

# Attention Ratio – policy perturbation
POL_SUITES_AR = ["LIBERO-In domain", "LIBERO-90-Object", "LIBERO-90-Spatial",
                 "LIBERO-90-Act", "LIBERO-90-Com"]

POL_AR = {
    "LIBERO-In domain": {
        "pi0.5": [0.576, 0.701, 0.662],
        "Cosmos":[0.439, 0.369, 0.420],
        "DP":    [0.928, 0.974, 0.954],
    },
    "LIBERO-90-Object": {
        "pi0.5": [0.701, 0.747, 0.735],
        "Cosmos":[0.383, 0.342, 0.367],
        "DP":    [0.966, 0.985, 0.978],
    },
    "LIBERO-90-Spatial": {
        "pi0.5": [0.735, 0.782, 0.775],
        "Cosmos":[0.355, 0.306, 0.333],
        "DP":    [0.972, 0.987, 0.981],
    },
    "LIBERO-90-Act": {
        "pi0.5": [0.724, 0.784, 0.766],
        "Cosmos":[0.367, 0.315, 0.342],
        "DP":    [0.971, 0.988, 0.980],
    },
    "LIBERO-90-Com": {
        "pi0.5": [0.810, 0.820, 0.814],
        "Cosmos":[0.254, 0.242, 0.254],
        "DP":    [0.986, 0.989, 0.988],
    },
}

# --- figure 3a: success rate ---
n = len(POL_SUITES)
fig3a, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6))
fig3a.suptitle("Policy Perturbation – Success Rate", fontsize=10, fontweight="bold")
for ax, suite in zip(axes, POL_SUITES):
    grouped_bar_ax(ax, POL_SR[suite], POL_CONDITIONS, MODELS_ALL,
                   MODEL_COLORS, "Success rate (%)", suite,
                   ylim=(0, 105), pct_fmt=True)
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ALL]
fig3a.legend(handles, MODELS_ALL, loc="lower center", ncol=5,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig3a.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig3a, "3a_pol_success_rate.png")

# --- figure 3b: attention IoU ---
n = len(POL_SUITES_IOU)
fig3b, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
fig3b.suptitle("Policy Perturbation – Attention IoU", fontsize=10, fontweight="bold")
for ax, suite in zip(axes, POL_SUITES_IOU):
    grouped_bar_ax(ax, POL_IOU[suite], POL_CONDITIONS, MODELS_ATTN,
                   MODEL_COLORS, "Attention IoU", suite)
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ATTN]
fig3b.legend(handles, MODELS_ATTN, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig3b.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig3b, "3b_pol_attention_iou.png")

# --- figure 3c: attention ratio ---
n = len(POL_SUITES_AR)
fig3c, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
fig3c.suptitle("Policy Perturbation – Attention Ratio (visual/total)",
               fontsize=10, fontweight="bold")
for ax, suite in zip(axes, POL_SUITES_AR):
    grouped_bar_ax(ax, POL_AR[suite], POL_CONDITIONS, MODELS_ATTN,
                   MODEL_COLORS, "Attention ratio", suite, ylim=(0, 1.05))
handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS_ATTN]
fig3c.legend(handles, MODELS_ATTN, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), frameon=False)
fig3c.tight_layout(rect=[0, 0.05, 1, 1])
save_fig(fig3c, "3c_pol_attention_ratio.png")

print("Done. All plots saved to", OUT_DIR)
