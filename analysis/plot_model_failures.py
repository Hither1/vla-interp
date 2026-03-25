"""
Plot tasks that each model fails at (0% success), grouped by failure pattern.
Shows PI0.5-only, Cosmos-only, DreamZero-only, and shared failures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/analysis/model_failures.pdf"

# ---------------------------------------------------------------------------
# Data: (task_label, suite, [pi05_fails, cosmos_fails, dz_fails])
# ---------------------------------------------------------------------------

SUITE_COLOR = {
    "90_spa": "#377eb8",
    "90_act": "#e41a1c",
    "90_obj": "#4daf4a",
}
SUITE_LABEL = {
    "90_spa": "L90-Spa",
    "90_act": "L90-Act",
    "90_obj": "L90-Obj",
}

MODELS = ["PI0.5", "Cosmos", "DreamZero"]
MODEL_COLOR = {
    "PI0.5":     "#4477AA",
    "Cosmos":    "#EE6677",
    "DreamZero": "#44AA99",
}

# Each entry: (display_name, suite, (pi05_fails, cosmos_fails, dz_fails))
# Failures are ordered by group: PI0.5-only, Cosmos-only, Cosmos+DZ shared
TASKS = [
    # --- PI0.5-only failures ---
    ("Put chocolate pudding right of plate",                    "90_spa", (True,  False, False)),
    ("Put white mug on left plate",                             "90_spa", (True,  False, False)),
    ("Put yellow+white mug on right plate",                     "90_spa", (True,  False, False)),
    ("Put white bowl on top of cabinet",                        "90_spa", (True,  False, False)),
    ("Pick up book, place in caddy compartment",                "90_spa", (True,  False, False)),

    # --- Cosmos-only failures (DreamZero succeeds) ---
    ("Open the microwave",                                      "90_act", (False, True,  False)),
    ("Open top drawer of cabinet",                              "90_act", (False, True,  False)),
    ("Put white bowl on the plate",                             "90_obj", (False, True,  False)),
    ("Pick up orange juice, put in basket",                     "90_obj", (False, True,  False)),

    # --- Cosmos + DreamZero failures (PI0.5 succeeds) ---
    ("Put middle black bowl on the plate",                      "90_spa", (False, True,  True)),
    ("Put middle black bowl on top of cabinet",                 "90_spa", (False, True,  True)),
    ("Put wine bottle in bottom drawer of cabinet",             "90_spa", (False, True,  True)),
    ("Pick up chocolate pudding, put in tray",                  "90_obj", (False, True,  True)),
]

# Group separators: (after_index, label)
GROUPS = [
    (4,  "PI0.5-only failures"),
    (8,  "Cosmos-only failures"),
    (12, "Cosmos + DreamZero failures"),
]
GROUP_LABELS = {
    4:  "PI0.5 fails\n(Cosmos & DZ succeed)",
    8:  "Cosmos fails\n(PI0.5 & DZ succeed)",
    12: "Cosmos & DZ fail\n(PI0.5 succeeds)",
}
GROUP_RANGES = [
    (0, 5,  "PI0.5 fails\n(Cosmos & DZ succeed)"),
    (5, 9,  "Cosmos fails\n(PI0.5 & DZ succeed)"),
    (9, 13, "Cosmos & DZ fail\n(PI0.5 succeeds)"),
]
GROUP_BGCOLORS = ["#eef3fa", "#fdf0f0", "#f0f8f5"]

n_tasks  = len(TASKS)
n_models = len(MODELS)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.subplots_adjust(left=0.42, right=0.78, top=0.88, bottom=0.18)
fig.suptitle("Model-Specific Task Failures\n(tasks at 0% success for each model)",
             fontsize=13, fontweight="bold")

ax.set_xlim(-0.5, n_models - 0.5)
ax.set_ylim(-0.7, n_tasks - 0.3)
ax.invert_yaxis()  # top = first task

# ---------------------------------------------------------------------------
# Group background shading
# ---------------------------------------------------------------------------

for (start, end, _), bg in zip(GROUP_RANGES, GROUP_BGCOLORS):
    ax.axhspan(start - 0.5, end - 0.5, color=bg, zorder=0)

# Separator lines between groups
for start, end, _ in GROUP_RANGES[1:]:
    ax.axhline(start - 0.5, color="#aaaaaa", lw=1.2, ls="--", zorder=1)

# ---------------------------------------------------------------------------
# Dots: hollow = succeeds, filled = fails
# ---------------------------------------------------------------------------

DOT_RADIUS_FAIL = 220   # marker size for failure
DOT_RADIUS_PASS = 40

for row, (task, suite, fails) in enumerate(TASKS):
    for col, (model, did_fail) in enumerate(zip(MODELS, fails)):
        if did_fail:
            color = MODEL_COLOR[model]
            ax.scatter(col, row, s=DOT_RADIUS_FAIL, color=color,
                       zorder=3, linewidths=0.5, edgecolors="white")
        else:
            ax.scatter(col, row, s=DOT_RADIUS_PASS, color="#cccccc",
                       zorder=3, linewidths=0.8, edgecolors="#999999", marker="o")

# ---------------------------------------------------------------------------
# Task labels on the left (y-axis), colored by suite
# ---------------------------------------------------------------------------

ax.set_yticks(range(n_tasks))
ax.set_yticklabels(
    [f"{task}  [{SUITE_LABEL[suite]}]" for task, suite, _ in TASKS],
    fontsize=9,
)
for ytick, (task, suite, _) in zip(ax.get_yticklabels(), TASKS):
    ytick.set_color(SUITE_COLOR[suite])

# ---------------------------------------------------------------------------
# Model labels on top (x-axis)
# ---------------------------------------------------------------------------

ax.set_xticks(range(n_models))
ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
for xtick, model in zip(ax.get_xticklabels(), MODELS):
    xtick.set_color(MODEL_COLOR[model])

# ---------------------------------------------------------------------------
# Group labels on the right margin (figure-level text, outside axes)
# ---------------------------------------------------------------------------

for (start, end, label), bg in zip(GROUP_RANGES, GROUP_BGCOLORS):
    mid = (start + end - 1) / 2
    # Use axes transform: x in axes fraction (>1 = right of axes), y in data coords
    ax.annotate(label,
                xy=(1.03, mid), xycoords=("axes fraction", "data"),
                fontsize=8, color="#444444", va="center", ha="left",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec="#cccccc", lw=0.7))

# ---------------------------------------------------------------------------
# Count annotations below model columns (total failures per model)
# ---------------------------------------------------------------------------

model_fail_counts = [sum(f[i] for _, _, f in TASKS) for i in range(n_models)]
for col, (model, count) in enumerate(zip(MODELS, model_fail_counts)):
    ax.text(col, n_tasks - 0.15, f"{count} tasks",
            ha="center", va="top", fontsize=9, color=MODEL_COLOR[model],
            fontweight="bold")


ax.text(0.5, -0.04, "● filled = 0% success   ○ hollow = >0% success",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555555")

ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)
ax.grid(axis="x", alpha=0.2, lw=0.5, zorder=0)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")
