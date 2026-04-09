"""
Combined figure: experiment examples + generalisation summary (left)
+ perturbation taxonomy (right).

Run:
    python analysis/plot_combined.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Data ──────────────────────────────────────────────────────────────────────
SUBSETS = [
    {"label": "Object", "color": "#c98a2e"},
    {"label": "Spatial", "color": "#5a8f63"},
    {"label": "Action", "color": "#b85c5c"},
    {"label": "Composite", "color": "#597da6"},
]

PERTURBATIONS = [
    {
        "title": "Language-level",
        "subtitle": "Same task, different text",
        "color": "#3f6ea5",
        "light": "#eff4fa",
        "items": [
            ("Empty\nPrompt", "Instruction\nremoved"),
            ("Synonym\nSwap", "Content words\nreplaced"),
            ("Random\nPrompt", "Replaced with\nunrelated text"),
            ("Shuffled\nWords", "Words randomly\npermuted"),
        ],
    },
    {
        "title": "Vision-level",
        "subtitle": "Same task, different pose",
        "color": "#4f8a73",
        "light": "#edf5f1",
        "items": [
            ("Orientation", "Scene rotated\nby 30°"),
            ("Position", "Scene translated\n20% rightward"),
            ("Rotate +\nTranslate", "15° rotation +\n10% translation"),
        ],
    },
    {
        "title": "Policy-level",
        "subtitle": "Execution noise",
        "color": "#b46a52",
        "light": "#fbf0ec",
        "items": [
            ("Random\nAction", "Replace action with\nrandom (p = 0.25)"),
            ("Object\nShift", "Random x-axis\nshift, σ = 5 cm"),
        ],
    },
]

ROOT = Path(__file__).resolve().parent.parent
SIM_IMAGE = ROOT / "libero.jpeg"
REAL_IMAGE = ROOT / "droid_frame3.png"

FIG_BG = "#fbfaf7"
CARD_BG = "#ffffff"
CARD_EDGE = "#ddd8cf"
TEXT_MAIN = "#202020"
TEXT_MID = "#626262"
TEXT_SOFT = "#7a7a7a"
EXPORT_TRANSPARENT = True


def add_rounded_panel(ax, facecolor, edgecolor, linewidth=1.3, radius=18):
    """Draw a rounded card background in axes coordinates."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    patch = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=-10,
    )
    ax.add_patch(patch)


def add_image_card(subfig, bounds, image_path, eyebrow, title, accent):
    """Add an example image card with a small caption header."""
    ax = subfig.add_axes(bounds)
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(0.05, 0.90, eyebrow.upper(), fontsize=9.5, fontweight="bold", color=accent, va="center")
    ax.text(0.05, 0.82, title, fontsize=13.2, fontweight="bold", color=TEXT_MAIN, va="center")

    image = plt.imread(image_path)
    image_h, image_w = image.shape[:2]
    image_aspect = image_w / image_h

    max_w = 0.90
    max_h = 0.66
    box_aspect = max_w / max_h
    if image_aspect >= box_aspect:
        inset_w = max_w
        inset_h = max_w / image_aspect
    else:
        inset_h = max_h
        inset_w = max_h * image_aspect

    inset_x = 0.05 + (max_w - inset_w) / 2
    inset_y = 0.08 + (max_h - inset_h) / 2

    img_ax = ax.inset_axes([inset_x, inset_y, inset_w, inset_h])
    img_ax.set_facecolor("#f6f4ef")
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for sp in img_ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_edgecolor("#d8d2c8")

    img_ax.imshow(image)
    img_ax.set_aspect("equal")


def main():
    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )

    canvas_bg = "none" if EXPORT_TRANSPARENT else FIG_BG
    fig = plt.figure(figsize=(21, 8.6), facecolor=canvas_bg)
    sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[1.08, 1.62], wspace=0.03)
    sf_left.set_facecolor(canvas_bg)
    sf_right.set_facecolor(canvas_bg)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT — examples + donut
    # ══════════════════════════════════════════════════════════════════════════
    sf_left.text(
        0.50,
        0.965,
        "Experiment Settings",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TEXT_MAIN,
    )
    sf_left.text(
        0.50,
        0.925,
        "Representative scenes from simulation and real-robot evaluation.",
        ha="center",
        va="top",
        fontsize=10.8,
        color=TEXT_MID,
    )



    donut_panel = sf_left.add_axes([0.05, 0.08, 0.90, 0.40])
    add_rounded_panel(donut_panel, facecolor=CARD_BG, edgecolor=CARD_EDGE, linewidth=1.0, radius=16)
    donut_panel.text(
        0.05,
        0.93,
        "Four Generalisation Types",
        transform=donut_panel.transAxes,
        ha="left",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=TEXT_MAIN,
    )
    donut_panel.text(
        0.05,
        0.84,
        "Task families shared between the simulation benchmark and the matched DROID subset.",
        transform=donut_panel.transAxes,
        ha="left",
        va="center",
        fontsize=10.2,
        color=TEXT_MID,
    )

    ax_pie = donut_panel.inset_axes([0.05, 0.08, 0.48, 0.72])
    wedges, _ = ax_pie.pie(
        [1] * 4,
        colors=[s["color"] for s in SUBSETS],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor=CARD_BG, linewidth=1.8),
    )

    ax_pie.text(
        0,
        0.08,
        "Simulation",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT_MAIN,
    )
    ax_pie.text(
        0,
        -0.10,
        "Real Robot",
        ha="center",
        va="center",
        fontsize=11.5,
        color=TEXT_MID,
        multialignment="center",
    )
    ax_pie.axis("equal")

    legend_ax = donut_panel.inset_axes([0.57, 0.16, 0.35, 0.56])
    legend_ax.set_axis_off()
    legend_ax.text(0.0, 1.02, "Task Families", fontsize=12.5, fontweight="bold", color=TEXT_MAIN)
    for idx, subset in enumerate(SUBSETS):
        y = 0.82 - idx * 0.22
        legend_ax.scatter([0.08], [y], s=170, color=subset["color"], marker="o")
        legend_ax.text(
            0.20,
            y,
            subset["label"].replace("\n", " "),
            va="center",
            fontsize=11.1,
            color=TEXT_MID,
        )
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1.08)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT — perturbation taxonomy
    # ══════════════════════════════════════════════════════════════════════════
    sf_right.text(
        0.5,
        0.98,
        "Perturbation Taxonomy",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TEXT_MAIN,
    )
    sf_right.text(
        0.5,
        0.94,
        "Three controlled shift types used to stress language, vision, and policy robustness.",
        ha="center",
        va="top",
        fontsize=10.8,
        color=TEXT_MID,
    )

    left_margin = 0.015
    right_margin = 0.015
    top_margin = 0.13
    bottom_margin = 0.04
    band_gap = 0.03
    label_width = 0.175
    label_pad = 0.008
    bar_width = 0.004
    item_margin = 0.018
    item_gap = 0.012
    item_vertical_margin = 0.17

    band_count = len(PERTURBATIONS)
    available_height = 1.0 - top_margin - bottom_margin
    band_height = (available_height - (band_count - 1) * band_gap) / band_count

    for i, perturbation in enumerate(PERTURBATIONS):
        y_bottom = 1.0 - top_margin - (i + 1) * band_height - i * band_gap

        band_bg = sf_right.add_axes([left_margin, y_bottom, 1.0 - left_margin - right_margin, band_height])
        add_rounded_panel(
            band_bg,
            facecolor=perturbation["light"],
            edgecolor=CARD_EDGE,
            linewidth=0.9,
            radius=16,
        )

        band_bar = sf_right.add_axes([left_margin, y_bottom, bar_width, band_height])
        band_bar.set_facecolor(perturbation["color"])
        band_bar.set_xticks([])
        band_bar.set_yticks([])
        for sp in band_bar.spines.values():
            sp.set_visible(False)

        label_left = left_margin + bar_width + label_pad
        label_ax = sf_right.add_axes([label_left, y_bottom, label_width, band_height])
        label_ax.set_facecolor("none")
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for sp in label_ax.spines.values():
            sp.set_visible(False)
        label_ax.text(
            0.06,
            0.63,
            perturbation["title"],
            transform=label_ax.transAxes,
            fontsize=15,
            fontweight="bold",
            color=perturbation["color"],
            va="center",
        )
        label_ax.text(
            0.06,
            0.32,
            perturbation["subtitle"],
            transform=label_ax.transAxes,
            fontsize=10.5,
            color=TEXT_SOFT,
            va="center",
            style="italic",
        )

        item_count = len(perturbation["items"])
        item_start = label_left + label_width + item_margin
        available_item_width = 1.0 - item_start - right_margin
        item_width = (available_item_width - (item_count - 1) * item_gap) / item_count
        item_height = band_height * (1 - 2 * item_vertical_margin)
        item_y = y_bottom + band_height * item_vertical_margin

        for j, (name, description) in enumerate(perturbation["items"]):
            card = sf_right.add_axes([item_start + j * (item_width + item_gap), item_y, item_width, item_height])
            add_rounded_panel(
                card,
                facecolor=CARD_BG,
                edgecolor="#e5e1d8",
                linewidth=0.9,
                radius=12,
            )
            card.plot([0.08, 0.92], [0.96, 0.96], transform=card.transAxes, color=perturbation["color"], lw=2.0)
            card.text(
                0.5,
                0.63,
                name,
                transform=card.transAxes,
                fontsize=13.8,
                fontweight="bold",
                color=TEXT_MAIN,
                ha="center",
                va="center",
                multialignment="center",
            )
            card.text(
                0.5,
                0.24,
                description,
                transform=card.transAxes,
                fontsize=11.6,
                color=TEXT_MID,
                ha="center",
                va="center",
                multialignment="center",
            )

    # ── Divider ───────────────────────────────────────────────────────────────
    divider_x = 1.08 / (1.08 + 1.62)
    fig.add_artist(
        plt.Line2D(
            [divider_x, divider_x],
            [0.05, 0.95],
            transform=fig.transFigure,
            color="#dfdbd2",
            linewidth=0.9,
            linestyle=(0, (2, 4)),
        )
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(
        "analysis/combined.pdf",
        bbox_inches="tight",
        dpi=200,
        transparent=EXPORT_TRANSPARENT,
        facecolor=canvas_bg,
    )
    plt.savefig(
        "analysis/combined.png",
        bbox_inches="tight",
        dpi=200,
        transparent=EXPORT_TRANSPARENT,
        facecolor=canvas_bg,
    )
    print("Saved -> analysis/combined.{pdf,png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
