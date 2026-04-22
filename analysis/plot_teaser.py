"""
Teaser plot: radar (spider) charts showing generalization across task suites.

Two radar plots side by side:
  Left:  LIBERO (Simulation) — 5 models, 5 generalization suites
  Right: DROID  (Real World) — 2 models, 3 generalization suites

Each radar axis = one generalization suite (original, unperturbed success rate).
Value = success rate (%) on that suite under standard evaluation.

LIBERO axes: In-domain | Object | Spatial | Act | Composite
DROID  axes: Object | Spatial | Act   (Composite excluded — all models score 0)
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
# Original (unperturbed) success rates — from language-perturbation "original"
# rows, which are the most complete.
# ═══════════════════════════════════════════════════════════════════════════════

# LIBERO: suite → model → success %
LIBERO_ORIG = {
    "In-domain": {
        "pi0.5":     98.2,
        "OpenVLA":   76.5,
        "Cosmos":    98.5,
        "DP":        91.8,
        "DreamZero": 97.8,
    },
    "Object": {
        "pi0.5":     49.5,
        "OpenVLA":    9.5,
        "Cosmos":    38.0,
        "DP":        18.5,
        "DreamZero": 47.2,
    },
    "Spatial": {
        "pi0.5":     23.0,
        "OpenVLA":    4.1,
        "Cosmos":    13.72,
        "DP":         7.8,
        "DreamZero": 21.5,
    },
    "Action": {
        "pi0.5":     31.2,
        "OpenVLA":   13.2,
        "Cosmos":    33.8,
        "DP":        12.1,
        "DreamZero": 35.0,
    },
    "Composite": {
        "pi0.5":     3.0,
        "OpenVLA":   0.0,
        "Cosmos":    1.5,
        "DP":        0.0,
        "DreamZero": 1.8,
    },
}

LIBERO_AXES = ["In-domain", "Object", "Spatial", "Action", "Composite"]

# DROID: suite → model → success %
DROID_ORIG = {
    "Object": {
        "pi0.5":     52.0,
        "DreamZero": 85.0,
    },
    "Spatial": {
        "pi0.5":     22.0,
        "DreamZero": 73.0,
    },
    "Action": {
        "pi0.5":     48.0,
        "DreamZero": 80.0,
    },
}

DROID_AXES = ["Object", "Spatial", "Action"]


# ═══════════════════════════════════════════════════════════════════════════════
# Radar chart
# ═══════════════════════════════════════════════════════════════════════════════

def radar_chart(ax, data, axes_keys, axes_labels, models,
                colors, linestyles, linewidths,
                title, r_max=100, r_ticks=(25, 50, 75, 100),
                label_pad=1.22):
    N = len(axes_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, r_max)

    # ring labels
    ax.set_yticks(list(r_ticks))
    ax.set_yticklabels(
        [f"{t}%" for t in r_ticks],
        fontsize=11.5, color="#777",
    )
    ax.set_rlabel_position(12)

    # spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=15, color="#222")
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

    # model polygons
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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   14,
})

fig = plt.figure(figsize=(13, 6.2))
fig.patch.set_facecolor("white")

ax_libero = fig.add_subplot(121, polar=True)
ax_droid  = fig.add_subplot(122, polar=True)
ax_libero.set_position([0.02, 0.05, 0.48, 0.85])
ax_droid.set_position([0.46, 0.05, 0.48, 0.85])

radar_chart(
    ax_libero, LIBERO_ORIG,
    axes_keys=LIBERO_AXES,
    axes_labels=LIBERO_AXES,
    models=LIBERO_MODELS,
    colors=MODEL_COLORS, linestyles=MODEL_LS, linewidths=MODEL_LW,
    title="LIBERO (Simulation)",
    r_max=100, r_ticks=(25, 50, 75, 100),
)

radar_chart(
    ax_droid, DROID_ORIG,
    axes_keys=DROID_AXES,
    axes_labels=DROID_AXES,
    models=DROID_MODELS,
    colors=MODEL_COLORS, linestyles=MODEL_LS, linewidths=MODEL_LW,
    title="DROID (Real World)",
    r_max=100, r_ticks=(25, 50, 75, 100),
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
    "Generalization Across Task Suites",
    fontsize=18, fontweight="bold", y=1.01,
)
fig.tight_layout(rect=[0, 0.06, 1, 1])

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"teaser_radar.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
