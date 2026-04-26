"""
Teaser radar: axes = perturbation types (Language, Visual, Policy).
Value on each axis = mean success rate (%) across all non-original conditions
of that perturbation type.

Left panel  : LIBERO (Simulation) — 5 models, averaged across all 5 suites
Right panel : DROID  (Real World) — 2 models, averaged across Object/Spatial/Act
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── colour / style palette ─────────────────────────────────────────────────────
MODEL_COLORS = {
    "pi0.5":     "#4C72B0",
    "OpenVLA":   "#DD8452",
    "Cosmos":    "#55A868",
    "DP":        "#C44E52",
    "DreamZero": "#17becf",
}
MODEL_LS = {
    "pi0.5":     "-",
    "OpenVLA":   "--",
    "Cosmos":    "-.",
    "DP":        ":",
    "DreamZero": "-",
}
MODEL_LW = {
    "pi0.5":     2.0,
    "OpenVLA":   2.0,
    "Cosmos":    2.0,
    "DP":        2.0,
    "DreamZero": 2.5,
}

LIBERO_MODELS = ["pi0.5", "OpenVLA", "Cosmos", "DP", "DreamZero"]
DROID_MODELS  = ["pi0.5", "DreamZero"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots_generalization")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LIBERO success-rate tables (non-original conditions only)
# LANG_CONDS = ["original", "empty", "shuffle", "random", "synonym"]  → idx 1-4
# VIS_CONDS  = ["original", "rotate 30°", "translate 20%", "rotate+translate"] → idx 1-3
# POL_CONDS  = ["original", "random action 25%", "object shift x"]    → idx 1-2
# ═══════════════════════════════════════════════════════════════════════════════

LIBERO_LANG_SR = {
    "in-domain": {
        "pi0.5":     [62.5, 100.0, 34.5, 96.5],
        "OpenVLA":   [ 0.13,  24.5,  6.3, 64.8],
        "Cosmos":    [50.5,   83.3, 32.5, 94.5],
        "DP":        [36.3,   35.3, 30.9, 35.5],
        "DreamZero": [59.2,   91.5, 31.2, 97.8],
    },
    "object": {
        "pi0.5":     [ 4.0, 50.2,  3.3, 10.2],
        "OpenVLA":   [ 0.0,  6.0,  3.5, 12.2],
        "Cosmos":    [27.8, 32.0, 27.0, 38.8],
        "DP":        [14.5, 14.2, 13.5, 15.0],
        "DreamZero": [27.9, 32.2, 25.3, 38.5],
    },
    "spatial": {
        "pi0.5":     [ 8.0, 21.5,  1.0, 24.0],
        "OpenVLA":   [ 0.1,  2.8,  0.6,  4.9],
        "Cosmos":    [13.49,12.91,15.47,13.14],
        "DP":        [ 6.7,  6.9,  7.9,  6.5],
        "DreamZero": [13.8, 13.2, 14.8, 13.4],
    },
    "act": {
        "pi0.5":     [10.2, 27.0,  3.0, 28.0],
        "OpenVLA":   [ 1.2,  9.1,  2.6, 11.5],
        "Cosmos":    [26.2, 37.1, 30.0, 40.0],
        "DP":        [20.3, 21.2, 22.1, 21.5],
        "DreamZero": [27.0, 38.0, 28.8, 41.0],
    },
    "com": {
        "pi0.5":     [3.0, 2.9, 1.0, 2.7],
        "OpenVLA":   [0.0, 0.0, 0.0, 0.0],
        "Cosmos":    [2.0, 2.0, 0.0, 1.0],
        "DP":        [0.0, 0.0, 0.0, 0.0],
        "DreamZero": [2.2, 2.2, 0.1, 1.2],
    },
}

LIBERO_VIS_SR = {
    "in-domain": {
        "pi0.5":     [11.6, 22.0, 19.7],
        "OpenVLA":   [ 1.1, 13.4,  3.4],
        "Cosmos":    [19.6, 80.0, 80.9],
        "DP":        [ 6.9, 15.5, 14.8],
        "DreamZero": [18.8, 87.5, 88.3],
    },
    "object": {
        "pi0.5":     [ 5.0, 12.6,  6.8],
        "OpenVLA":   [ 0.0,  6.0,  0.3],
        "Cosmos":    [ 0.5, 24.5, 23.0],
        "DP":        [ 0.0,  1.8,  1.0],
        "DreamZero": [ 0.4, 24.8, 23.3],
    },
    "spatial": {
        "pi0.5":     [0.9,  2.6,  5.8],
        "OpenVLA":   [0.2,  0.8,  0.6],
        "Cosmos":    [0.5, 10.2,  6.9],
        "DP":        [0.2,  2.7,  1.2],
        "DreamZero": [0.5, 10.8,  7.4],
    },
    "act": {
        "pi0.5":     [11.5, 12.6, 19.7],
        "OpenVLA":   [ 3.8,  8.8,  4.1],
        "Cosmos":    [ 7.1, 18.8, 26.5],
        "DP":        [ 0.3,  0.0,  0.3],
        "DreamZero": [ 6.9, 19.6, 27.5],
    },
    "com": {
        "pi0.5":     [0.0, 0.0, 0.5],
        "OpenVLA":   [0.0, 0.0, 0.0],
        "Cosmos":    [0.0, 0.5, 1.0],
        "DP":        [0.0, 0.0, 0.0],
        "DreamZero": [0.0, 0.7, 1.2],
    },
}

LIBERO_POL_SR = {
    "in-domain": {
        "pi0.5":     [31.5, 77.8],
        "OpenVLA":   [ 5.0, 34.1],
        "Cosmos":    [25.2, 76.8],
        "DP":        [20.5, 54.6],
        "DreamZero": [24.8, 78.5],
    },
    "object": {
        "pi0.5":     [ 5.5,  7.7],
        "OpenVLA":   [ 2.0,  6.8],
        "Cosmos":    [ 8.7, 31.8],
        "DP":        [ 4.3, 10.0],
        "DreamZero": [ 7.9, 32.0],
    },
    "spatial": {
        "pi0.5":     [4.4,  9.5],
        "OpenVLA":   [0.2,  2.4],
        "Cosmos":    [3.8, 12.7],
        "DP":        [2.4,  4.7],
        "DreamZero": [3.5, 13.1],
    },
    "act": {
        "pi0.5":     [19.4, 30.0],
        "OpenVLA":   [ 5.6, 10.9],
        "Cosmos":    [14.1, 28.8],
        "DP":        [ 3.2,  9.4],
        "DreamZero": [13.5, 29.6],
    },
    "com": {
        "pi0.5":     [1.0, 1.0],
        "OpenVLA":   [0.0, 0.0],
        "Cosmos":    [0.0, 1.0],
        "DP":        [0.0, 0.0],
        "DreamZero": [0.1, 1.2],
    },
}

LIBERO_SUITES = ["in-domain", "object", "spatial", "act", "com"]

# ═══════════════════════════════════════════════════════════════════════════════
# DROID success-rate tables (non-original conditions only)
# same perturbation types; DROID-Com excluded (all-zero)
# ═══════════════════════════════════════════════════════════════════════════════

DROID_LANG_SR = {
    "object":  {"pi0.5": [ 3, 28,  2, 47], "DreamZero": [ 3, 78,  2, 82]},
    "spatial": {"pi0.5": [ 0,  8,  0, 18], "DreamZero": [ 2, 65,  2, 70]},
    "act":     {"pi0.5": [ 2, 33,  0, 52], "DreamZero": [ 2, 72,  2, 78]},
}

DROID_VIS_SR = {
    "object":  {"pi0.5": [ 2, 45,  0], "DreamZero": [ 8, 48,  5]},
    "spatial": {"pi0.5": [ 0, 18,  0], "DreamZero": [ 5, 38,  3]},
    "act":     {"pi0.5": [ 3, 42,  2], "DreamZero": [ 8, 45,  5]},
}

DROID_POL_SR = {
    "object":  {"pi0.5": [ 3, 45], "DreamZero": [22, 72]},
    "spatial": {"pi0.5": [ 0, 17], "DreamZero": [18, 62]},
    "act":     {"pi0.5": [ 2, 48], "DreamZero": [20, 68]},
}

DROID_SUITES = ["object", "spatial", "act"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mean_sr(sr_dict, suites, model):
    """Mean success rate across given suites and all listed conditions."""
    vals = []
    for s in suites:
        vals.extend(sr_dict.get(s, {}).get(model, []))
    return float(np.mean(vals)) if vals else 0.0


def build_panel_data(lang_d, vis_d, pol_d, suites, models):
    return {
        "Language": {m: mean_sr(lang_d, suites, m) for m in models},
        "Visual":   {m: mean_sr(vis_d,  suites, m) for m in models},
        "Policy":   {m: mean_sr(pol_d,  suites, m) for m in models},
    }


def normalize_panel(data, axes_keys):
    """Normalize each axis to [0, 1] by dividing by the max across all models.
    Returns (normed_data, axis_maxes)."""
    normed = {}
    maxes = {}
    for k in axes_keys:
        mx = max(data[k].values()) if data[k] else 1.0
        mx = mx if mx > 0 else 1.0
        maxes[k] = mx
        normed[k] = {m: v / mx for m, v in data[k].items()}
    return normed, maxes


# ═══════════════════════════════════════════════════════════════════════════════
# Radar chart
# ═══════════════════════════════════════════════════════════════════════════════

def radar_chart(ax, data, axes_keys, models,
                colors, linestyles, linewidths,
                title, axis_maxes=None,
                r_max=1.0, r_ticks=(0.25, 0.5, 0.75, 1.0),
                label_pad=1.22):
    N = len(axes_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, r_max)

    ax.set_yticks(list(r_ticks))
    ax.set_yticklabels(
        [f"{int(t*100)}%" for t in r_ticks],
        fontsize=11.5, color="#777",
    )
    ax.set_rlabel_position(12)

    ax.set_xticks(angles)
    if axis_maxes:
        xticklabels = [f"{k}\n({axis_maxes[k]:.1f}%)" for k in axes_keys]
    else:
        xticklabels = axes_keys
    ax.set_xticklabels(xticklabels, fontsize=15, color="#222")
    for label, angle in zip(ax.get_xticklabels(), angles):
        deg = np.degrees(angle)
        if 10 < deg < 170:
            label.set_horizontalalignment("left")
        elif 190 < deg < 350:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")
        x, y = label.get_position()
        label.set_position((x, y * label_pad))

    ax.grid(color="#ccc", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_color("#bbb")

    for m in models:
        vals = [data[k].get(m, 0.0) for k in axes_keys]
        closed = vals + [vals[0]]
        ax.plot(
            angles_closed, closed,
            color=colors[m],
            linestyle=linestyles[m],
            linewidth=linewidths[m],
            zorder=4,
        )
        ax.fill(
            angles_closed, closed,
            color=colors[m],
            alpha=0.09,
            zorder=3,
        )

    ax.set_title(title, fontsize=17, fontweight="bold", pad=26, color="#111")


# ═══════════════════════════════════════════════════════════════════════════════
# Draw
# ═══════════════════════════════════════════════════════════════════════════════

AXES_KEYS = ["Language", "Visual", "Policy"]

libero_data, libero_maxes = normalize_panel(
    build_panel_data(LIBERO_LANG_SR, LIBERO_VIS_SR, LIBERO_POL_SR,
                     LIBERO_SUITES, LIBERO_MODELS),
    AXES_KEYS,
)
droid_data, droid_maxes = normalize_panel(
    build_panel_data(DROID_LANG_SR, DROID_VIS_SR, DROID_POL_SR,
                     DROID_SUITES, DROID_MODELS),
    AXES_KEYS,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   14,
})

fig = plt.figure(figsize=(13, 6.2))
fig.patch.set_facecolor("white")

ax_libero = fig.add_subplot(121, polar=True)
ax_droid  = fig.add_subplot(122, polar=True)
ax_libero.set_position([0.02, 0.05, 0.48, 0.85])
ax_droid.set_position( [0.46, 0.05, 0.48, 0.85])

radar_chart(
    ax_libero, libero_data,
    axes_keys=AXES_KEYS,
    models=LIBERO_MODELS,
    colors=MODEL_COLORS, linestyles=MODEL_LS, linewidths=MODEL_LW,
    title="LIBERO (Simulation)",
    axis_maxes=libero_maxes,
)

radar_chart(
    ax_droid, droid_data,
    axes_keys=AXES_KEYS,
    models=DROID_MODELS,
    colors=MODEL_COLORS, linestyles=MODEL_LS, linewidths=MODEL_LW,
    title="DROID (Real World)",
    axis_maxes=droid_maxes,
)

# ── Shared legend ──────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [],
                  color=MODEL_COLORS[m],
                  linestyle=MODEL_LS[m],
                  linewidth=MODEL_LW[m],
                  label=m)
    for m in LIBERO_MODELS
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=len(LIBERO_MODELS),
    bbox_to_anchor=(0.5, -0.04),
    frameon=False,
    fontsize=15,
    handlelength=2.2,
    columnspacing=1.4,
)

fig.suptitle(
    "Perturbation Robustness Across Models\n"
    "(normalized by best model per axis; mean success rate on non-original conditions)",
    fontsize=16, fontweight="bold", y=1.03,
)
fig.tight_layout(rect=[0, 0.06, 1, 1])

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"teaser_radar_perturbation.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
