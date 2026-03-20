"""
Combined figure: generalisation donut (left) + perturbation types (right).

Run:
    python analysis/plot_combined.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
SUBSETS = [
    {"label": "LIBERO-90\nObject",    "color": "#ff7f0e"},
    {"label": "LIBERO-90\nSpatial",   "color": "#2ca02c"},
    {"label": "LIBERO-90\nAction",    "color": "#d62728"},
    {"label": "LIBERO-90\nComposite", "color": "#9467bd"},
]

PERTURBATIONS = [
    {
        "title":    "Language-level",
        "subtitle": "Same task, different text",
        "color":    "#4C72B0",
        "light":    "#e8eef8",
        "items": [
            ("Empty\nPrompt",         "Instruction\nremoved"),
            ("Opposite\nInstruction", "Key terms negated\n(left → right)"),
            ("Synonym\nSwap",         "Content words\nreplaced"),
            ("Random\nPrompt",        "Replaced with\nunrelated text"),
            ("Shuffled\nWords",       "Words randomly\npermuted"),
        ],
    },
    {
        "title":    "Vision-level",
        "subtitle": "Same task, different pose",
        "color":    "#2ca02c",
        "light":    "#e8f5e8",
        "items": [
            ("Orientation",           "Scene rotated\nby 30°"),
            ("Position",              "Scene translated\n20% rightward"),
            ("Rotate +\nTranslate",   "15° rotation +\n10% translation"),
        ],
    },
    {
        "title":    "Policy-level",
        "subtitle": "Execution noise",
        "color":    "#d62728",
        "light":    "#f8e8e8",
        "items": [
            ("Random\nAction",        "Replace action w/\nrandom (p=0.25)"),
            ("Object\nShift",         "Random x-axis\nshift, σ=5 cm"),
        ],
    },
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6.8))
sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[1, 1.72], wspace=0.01)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — donut
# ══════════════════════════════════════════════════════════════════════════════
sf_left.text(0.5, 0.98, "LIBERO-90 (Simulation) & DROID (Real Robot)",
             ha="center", va="top", fontsize=13, color="#444444")

ax_pie = sf_left.add_axes([0.05, 0.08, 0.90, 0.84])

wedges, _ = ax_pie.pie(
    [1] * 4,
    colors=[s["color"] for s in SUBSETS],
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
)

for wedge, subset in zip(wedges, SUBSETS):
    angle = (wedge.theta1 + wedge.theta2) / 2
    rad   = np.deg2rad(angle)
    r     = 0.78
    ax_pie.text(r * np.cos(rad), r * np.sin(rad), subset["label"],
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="white",
                multialignment="center")

ax_pie.text(0,  0.12, "Sim: LIBERO-90",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#333333")
ax_pie.text(0, -0.12, "Real: DROID\n(3 tasks each)",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#555555",
            multialignment="center")

ax_pie.set_title("Four Generalisation Types",
                 fontsize=16, fontweight="bold", pad=2, y=0.97)
ax_pie.axis("equal")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — perturbation bands
# ══════════════════════════════════════════════════════════════════════════════
sf_right.text(0.5, 0.98,
              "Perturbation Types — Language, Vision, and Policy Levels",
              ha="center", va="top",
              fontsize=15, fontweight="bold", color="#222222")

LM, RM        = 0.015, 0.015
TM, BM        = 0.10,  0.04
BAND_GAP      = 0.025
LABEL_W       = 0.155
LABEL_PAD     = 0.006
BAR_W         = 0.004
ITEM_MARGIN   = 0.016
ITEM_GAP      = 0.010
ITEM_V_MARGIN = 0.13

n_bands = len(PERTURBATIONS)
avail_h = 1.0 - TM - BM
band_h  = (avail_h - (n_bands - 1) * BAND_GAP) / n_bands

for i, ptype in enumerate(PERTURBATIONS):
    y_bot = 1.0 - TM - (i + 1) * band_h - i * BAND_GAP

    bg = sf_right.add_axes([LM, y_bot, 1.0 - LM - RM, band_h])
    bg.set_facecolor(ptype["light"])
    bg.set_xticks([]); bg.set_yticks([])
    for sp in bg.spines.values():
        sp.set_visible(False)

    bar = sf_right.add_axes([LM, y_bot, BAR_W, band_h])
    bar.set_facecolor(ptype["color"])
    bar.set_xticks([]); bar.set_yticks([])
    for sp in bar.spines.values():
        sp.set_visible(False)

    lx  = LM + BAR_W + LABEL_PAD
    lbl = sf_right.add_axes([lx, y_bot, LABEL_W, band_h])
    lbl.set_facecolor(ptype["light"])
    lbl.set_xticks([]); lbl.set_yticks([])
    for sp in lbl.spines.values():
        sp.set_visible(False)
    lbl.text(0.06, 0.66, ptype["title"],
             transform=lbl.transAxes,
             fontsize=15, fontweight="bold", color=ptype["color"], va="center")
    lbl.text(0.06, 0.30, ptype["subtitle"],
             transform=lbl.transAxes,
             fontsize=11, color="#666666", va="center", style="italic")

    n_items      = len(ptype["items"])
    ix_start     = lx + LABEL_W + ITEM_MARGIN
    avail_item_w = 1.0 - ix_start - RM
    item_w       = (avail_item_w - (n_items - 1) * ITEM_GAP) / n_items
    item_h       = band_h * (1 - 2 * ITEM_V_MARGIN)
    item_y       = y_bot + band_h * ITEM_V_MARGIN

    for j, (name, desc) in enumerate(ptype["items"]):
        card = sf_right.add_axes([ix_start + j * (item_w + ITEM_GAP), item_y, item_w, item_h])
        card.set_facecolor("white")
        card.set_xticks([]); card.set_yticks([])
        for sp in card.spines.values():
            sp.set_linewidth(1.4)
            sp.set_edgecolor(ptype["color"])
        card.text(0.5, 0.72, name,
                  transform=card.transAxes,
                  fontsize=14.5, fontweight="bold", color=ptype["color"],
                  ha="center", va="center", multialignment="center")
        card.text(0.5, 0.26, desc,
                  transform=card.transAxes,
                  fontsize=13, color="#444444",
                  ha="center", va="center", multialignment="center")

# ── Divider ───────────────────────────────────────────────────────────────────
divider_x = 1.0 / (1 + 1.72)
fig.add_artist(plt.Line2D(
    [divider_x, divider_x], [0.04, 0.96],
    transform=fig.transFigure,
    color="#cccccc", linewidth=1.2, linestyle="--",
))

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig("analysis/combined.pdf", bbox_inches="tight", dpi=150)
plt.savefig("analysis/combined.png", bbox_inches="tight", dpi=150)
print("Saved → analysis/combined.{pdf,png}")
plt.show()
