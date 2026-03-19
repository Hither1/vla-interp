"""
Circle / donut plot illustrating the four LIBERO-90 task subsets.

Each wedge represents one subset:
  libero_90_obj  → LIBERO-90-Object
  libero_90_spa  → LIBERO-90-Spatial
  libero_90_act  → LIBERO-90-Act(ion)
  libero_90_com  → LIBERO-90-Com(posite)

Run:
    python analysis/plot_libero90_subsets.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
SUBSETS = [
    {
        "key":   "libero_90_obj",
        "label": "LIBERO-90\nObject",
        "short": "Object",
        "color": "#ff7f0e",
        "size":  1,   # equal wedges; change to reflect actual task counts if desired
        "title": "Object Generalisation",
        "desc":  (
            "Same actions & scenes,\n"
            "different manipulated objects.\n"
            "Tests: object recognition,\n"
            "robust grasping."
        ),
    },
    {
        "key":   "libero_90_spa",
        "label": "LIBERO-90\nSpatial",
        "short": "Spatial",
        "color": "#2ca02c",
        "size":  1,
        "title": "Spatial Generalisation",
        "desc":  (
            "Same objects & actions,\n"
            "varied spatial layouts.\n"
            "Tests: spatial reasoning,\n"
            "relative positioning."
        ),
    },
    {
        "key":   "libero_90_act",
        "label": "LIBERO-90\nAction",
        "short": "Action",
        "color": "#d62728",
        "size":  1,
        "title": "Action Generalisation",
        "desc":  (
            "Same objects & scenes,\n"
            "different action primitives.\n"
            "Tests: action semantics,\n"
            "verb-to-motion grounding."
        ),
    },
    {
        "key":   "libero_90_com",
        "label": "LIBERO-90\nComposite",
        "short": "Composite",
        "color": "#9467bd",
        "size":  1,
        "title": "Composite Generalisation",
        "desc":  (
            "Simultaneously varied objects,\n"
            "spatial configs, and actions.\n"
            "Tests: joint generalisation,\n"
            "hardest transfer setting."
        ),
    },
]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
# Left column: donut; right column: legend / description cards
ax_pie = fig.add_axes([0.03, 0.08, 0.45, 0.84])   # [left, bottom, width, height]

sizes  = [s["size"]  for s in SUBSETS]
colors = [s["color"] for s in SUBSETS]
labels = [s["label"] for s in SUBSETS]

wedge_props = dict(width=0.45, edgecolor="white", linewidth=2.5)

wedges, texts, autotexts = ax_pie.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct="%1.0f%%",
    startangle=90,
    counterclock=False,
    wedgeprops=wedge_props,
    pctdistance=0.78,
    textprops={"fontsize": 11},
)

for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(10)

# ── Wedge labels (placed along the mid-radius of each wedge) ──────────────────
for wedge, subset in zip(wedges, SUBSETS):
    angle = (wedge.theta1 + wedge.theta2) / 2          # mid-angle in degrees
    rad   = np.deg2rad(angle)
    r     = 0.62                                        # label radius (inside ring)
    x, y  = r * np.cos(rad), r * np.sin(rad)
    ax_pie.text(
        x, y, subset["label"],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="white",
        multialignment="center",
    )

# Centre label
ax_pie.text(
    0, 0, "LIBERO-90\n(90 tasks)",
    ha="center", va="center",
    fontsize=11, fontweight="bold", color="#333333",
    multialignment="center",
)

ax_pie.set_title("LIBERO-90 Task Subsets", fontsize=14, fontweight="bold", pad=12)
ax_pie.axis("equal")

# ── Description cards (right side) ───────────────────────────────────────────
n = len(SUBSETS)
card_h = 0.20          # height of each card in figure-fraction units
card_w = 0.46
card_x = 0.51          # left edge of cards
gap    = 0.025

total_used = n * card_h + (n - 1) * gap
start_y    = 0.5 + total_used / 2   # top of first card (centred vertically)

for i, subset in enumerate(SUBSETS):
    y_top = start_y - i * (card_h + gap)
    y_bot = y_top - card_h

    # Coloured left bar
    bar_ax = fig.add_axes([card_x, y_bot, 0.008, card_h])
    bar_ax.set_facecolor(subset["color"])
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    # Card background
    card_ax = fig.add_axes([card_x + 0.010, y_bot, card_w - 0.010, card_h])
    card_ax.set_facecolor("#f7f7f7")
    card_ax.set_xticks([])
    card_ax.set_yticks([])
    for spine in card_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#cccccc")

    # Title
    card_ax.text(
        0.04, 0.82, subset["title"],
        transform=card_ax.transAxes,
        fontsize=11, fontweight="bold",
        color=subset["color"], va="top",
    )
    # Key label
    card_ax.text(
        0.96, 0.82, f"({subset['key']})",
        transform=card_ax.transAxes,
        fontsize=8, color="#888888",
        va="top", ha="right", style="italic",
    )
    # Description
    card_ax.text(
        0.04, 0.52, subset["desc"],
        transform=card_ax.transAxes,
        fontsize=9, color="#333333",
        va="top", multialignment="left",
    )

# ── Overall title ─────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.97,
    "LIBERO-90 Generalisation Benchmark — Task Subset Overview",
    ha="center", va="top",
    fontsize=13, fontweight="bold", color="#222222",
)

plt.savefig("analysis/libero90_subsets.pdf", bbox_inches="tight", dpi=150)
plt.savefig("analysis/libero90_subsets.png", bbox_inches="tight", dpi=150)
print("Saved → analysis/libero90_subsets.{pdf,png}")
plt.show()
