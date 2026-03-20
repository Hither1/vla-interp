"""
Visualize per-task success rates for pi0.5 vs Cosmos across LIBERO tasks.
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero"
OUT = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/analysis/task_failures.pdf"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_task_success_rates(model):
    results = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for perturb_type in ["policy_perturb", "visual_perturb"]:
        model_dir = os.path.join(BASE, model, perturb_type)
        if not os.path.exists(model_dir):
            continue
        for perturb in os.listdir(model_dir):
            perturb_dir = os.path.join(model_dir, perturb)
            for suite in os.listdir(perturb_dir):
                suite_dir = os.path.join(perturb_dir, suite)
                if not os.path.isdir(suite_dir):
                    continue
                for fname in os.listdir(suite_dir):
                    if not fname.startswith("rollout_"):
                        continue
                    m = re.match(r"rollout_(.+)_trial\d+_(success|failure)\.mp4", fname)
                    if not m:
                        continue
                    task, outcome = m.group(1), m.group(2)
                    key = f"{perturb_type}/{perturb}"
                    results[key][task][1] += 1
                    if outcome == "success":
                        results[key][task][0] += 1
    return results


def aggregate_by_task(results):
    task_totals = defaultdict(lambda: [0, 0])
    for tasks in results.values():
        for task, (s, t) in tasks.items():
            task_totals[task][0] += s
            task_totals[task][1] += t
    return task_totals


def aggregate_by_perturb(results, task):
    """Return dict: perturbation -> (success_rate or None)"""
    out = {}
    for perturb, tasks in results.items():
        if task in tasks:
            s, t = tasks[task]
            out[perturb] = s / t if t else None
    return out


pi05_results = get_task_success_rates("pi05")
cosmos_results = get_task_success_rates("cosmos")
pi05_totals = aggregate_by_task(pi05_results)
cosmos_totals = aggregate_by_task(cosmos_results)

all_tasks = sorted(set(pi05_totals) | set(cosmos_totals))


def get_task_suite_mapping():
    """Return dict: task -> suite name (from directory structure)."""
    mapping = {}
    for model in ["pi05", "cosmos"]:
        for pt in ["policy_perturb", "visual_perturb"]:
            model_dir = os.path.join(BASE, model, pt)
            if not os.path.exists(model_dir):
                continue
            for perturb in os.listdir(model_dir):
                for suite in os.listdir(os.path.join(model_dir, perturb)):
                    suite_dir = os.path.join(model_dir, perturb, suite)
                    if not os.path.isdir(suite_dir):
                        continue
                    for fname in os.listdir(suite_dir):
                        if not fname.startswith("rollout_"):
                            continue
                        m = re.match(r"rollout_(.+)_trial\d+_(success|failure)\.mp4", fname)
                        if m:
                            mapping[m.group(1)] = suite
    return mapping

TASK_SUITE = get_task_suite_mapping()

SUITE_ABBREV = {
    "libero_90_act": "L90-Act",
    "libero_90_com": "L90-Com",
    "libero_90_obj": "L90-Obj",
    "libero_90_spa": "L90-Spa",
    "libero_10":     "L10",
    "libero_goal":   "LGoal",
    "libero_object": "LObj",
    "libero_spatial":"LSpa",
}
SUITE_COLORS = {
    "libero_90_act": "#e41a1c",
    "libero_90_com": "#ff7f00",
    "libero_90_obj": "#4daf4a",
    "libero_90_spa": "#377eb8",
    "libero_10":     "#984ea3",
    "libero_goal":   "#a65628",
    "libero_object": "#f781bf",
    "libero_spatial":"#17becf",
}


def categorize(task):
    t = task.lower()
    if "stove" in t:             return "Stove"
    if "drawer" in t or "cabinet" in t: return "Drawer/Cabinet"
    if "stack" in t:             return "Stacking"
    if "frying_pan" in t:        return "Frying Pan"
    if "wine" in t:              return "Wine Bottle"
    if "book" in t:              return "Book/Caddy"
    if "mug" in t or "plate" in t: return "Mug/Plate"
    if "bowl" in t:              return "Bowl"
    if ("butter" in t or "pudding" in t or "ketchup" in t or "salad" in t
            or "orange" in t or "basket" in t or "tray" in t
            or "soup" in t or "milk" in t or "cream" in t
            or "bbq" in t or "sauce" in t):
        return "Grocery Pick&Place"
    if "microwave" in t:         return "Microwave"
    return "Other"


CATEGORIES = [
    "Stacking", "Book/Caddy", "Wine Bottle", "Grocery Pick&Place",
    "Frying Pan", "Bowl", "Mug/Plate", "Drawer/Cabinet", "Microwave", "Stove",
]
CAT_COLORS = {
    "Stacking":          "#e41a1c",
    "Book/Caddy":        "#ff7f00",
    "Wine Bottle":       "#984ea3",
    "Grocery Pick&Place":"#a65628",
    "Frying Pan":        "#f781bf",
    "Bowl":              "#4daf4a",
    "Mug/Plate":         "#377eb8",
    "Drawer/Cabinet":    "#17becf",
    "Microwave":         "#999999",
    "Stove":             "#bcbd22",
    "Other":             "#7f7f7f",
}

# Build per-task arrays
task_data = []
for task in all_tasks:
    ps, pt = pi05_totals.get(task, [0, 0])
    cs, ct = cosmos_totals.get(task, [0, 0])
    if pt == 0 and ct == 0:
        continue
    pi_rate  = ps / pt if pt else None
    co_rate  = cs / ct if ct else None
    cat = categorize(task)
    suite = TASK_SUITE.get(task, "unknown")
    task_data.append(dict(task=task, cat=cat, suite=suite,
                          pi_rate=pi_rate, co_rate=co_rate,
                          pi_s=ps, pi_t=pt, co_s=cs, co_t=ct))

# Category summary
cat_pi  = defaultdict(lambda: [0, 0])
cat_cos = defaultdict(lambda: [0, 0])
for d in task_data:
    cat_pi[d["cat"]][0]  += d["pi_s"];  cat_pi[d["cat"]][1]  += d["pi_t"]
    cat_cos[d["cat"]][0] += d["co_s"];  cat_cos[d["cat"]][1] += d["co_t"]


# ---------------------------------------------------------------------------
# Per-perturbation data for hard tasks heatmap
# ---------------------------------------------------------------------------

PERTURB_LABELS = {
    "policy_perturb/none":                              "PP: none",
    "policy_perturb/object_shift_x0.05_y0.0":          "PP: obj shift",
    "policy_perturb/random_action_p0.25_s1.0":         "PP: rand act",
    "visual_perturb/none":                              "VP: none",
    "visual_perturb/rotate_15deg_translate_x0.1_y0.0": "VP: rot15+tr",
    "visual_perturb/rotate_30deg":                      "VP: rot30",
    "visual_perturb/translate_x0.2_y0.0":              "VP: translate",
}
ALL_PERTURBS = list(PERTURB_LABELS.keys())

# Focus on tasks with <20% overall for both models (the hardest)
HARD_TASKS = [d["task"] for d in task_data
              if (d["pi_rate"] is not None and d["pi_rate"] < 0.20)
              or (d["co_rate"] is not None and d["co_rate"] < 0.20)]
HARD_TASKS = sorted(set(HARD_TASKS))

def build_heatmap(results, tasks, perturbs):
    mat = np.full((len(tasks), len(perturbs)), np.nan)
    for j, p in enumerate(perturbs):
        for i, t in enumerate(tasks):
            if p in results and t in results[p]:
                s, n = results[p][t]
                if n > 0:
                    mat[i, j] = s / n
    return mat

pi05_mat   = build_heatmap(pi05_results,   HARD_TASKS, ALL_PERTURBS)
cosmos_mat = build_heatmap(cosmos_results, HARD_TASKS, ALL_PERTURBS)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(22, 26))
fig.suptitle("PI0.5 vs Cosmos: Task Success Rates on LIBERO-90", fontsize=16, fontweight="bold", y=0.995)

# Grid: 3 rows
# Row 0: scatter (left) + category bars (right)
# Row 1-2: heatmaps (pi05 | cosmos), side by side
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.3, 1.3], hspace=0.45, wspace=0.35)

ax_scatter = fig.add_subplot(gs[0, 0])
ax_cat     = fig.add_subplot(gs[0, 1])
ax_heat_pi = fig.add_subplot(gs[1, :])
ax_heat_co = fig.add_subplot(gs[2, :])


# ---------------------------------------------------------------------------
# Panel A: Scatter — pi0.5 vs Cosmos success rate
# ---------------------------------------------------------------------------

ax = ax_scatter
for d in task_data:
    if d["pi_rate"] is None or d["co_rate"] is None:
        continue
    ax.scatter(d["pi_rate"], d["co_rate"],
               color=SUITE_COLORS.get(d["suite"], "#7f7f7f"),
               s=30, alpha=0.75, linewidths=0.3, edgecolors="white", zorder=3)

# Diagonal
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, zorder=2)
ax.axhline(0, color="gray", lw=0.4, alpha=0.3)
ax.axvline(0, color="gray", lw=0.4, alpha=0.3)

# Shade quadrants
ax.fill_between([0, 0.2], 0, 0.2, color="red", alpha=0.06, zorder=1)
ax.text(0.01, 0.01, "Both\nalways fail", fontsize=6.5, color="red", alpha=0.7,
        ha="left", va="bottom", transform=ax.transAxes)
ax.text(0.97, 0.01, "PI0.5 only fails", fontsize=6.5, color="#cc4400", alpha=0.8,
        ha="right", va="bottom", transform=ax.transAxes)
ax.text(0.01, 0.97, "Cosmos only fails", fontsize=6.5, color="#0055cc", alpha=0.8,
        ha="left", va="top", transform=ax.transAxes)

ax.set_xlabel("PI0.5 success rate", fontsize=10)
ax.set_ylabel("Cosmos success rate", fontsize=10)
ax.set_title("A. Per-task success rates", fontsize=11, fontweight="bold")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.tick_params(labelsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25, lw=0.5)

legend_handles = [mpatches.Patch(color=SUITE_COLORS[s], label=SUITE_ABBREV.get(s, s))
                  for s in SUITE_COLORS]
ax.legend(handles=legend_handles, fontsize=7, loc="upper left",
          framealpha=0.9, ncol=2, columnspacing=0.5, handlelength=0.8, title="Suite", title_fontsize=7)


# ---------------------------------------------------------------------------
# Panel B: Category-level bars
# ---------------------------------------------------------------------------

ax = ax_cat
cats_ordered = sorted(CATEGORIES,
                      key=lambda c: (cat_pi[c][0]/cat_pi[c][1] if cat_pi[c][1] else 0))
pi_rates  = [cat_pi[c][0]  / cat_pi[c][1]  if cat_pi[c][1]  else 0 for c in cats_ordered]
cos_rates = [cat_cos[c][0] / cat_cos[c][1] if cat_cos[c][1] else 0 for c in cats_ordered]

y = np.arange(len(cats_ordered))
h = 0.35
bars1 = ax.barh(y + h/2, pi_rates,  height=h, label="PI0.5",  color="#4477AA", alpha=0.85)
bars2 = ax.barh(y - h/2, cos_rates, height=h, label="Cosmos", color="#EE6677", alpha=0.85)

for bar, val in zip(bars1, pi_rates):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.0%}", va="center", ha="left", fontsize=7)
for bar, val in zip(bars2, cos_rates):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.0%}", va="center", ha="left", fontsize=7)

ax.set_yticks(y)
ax.set_yticklabels(cats_ordered, fontsize=8.5)
ax.set_xlabel("Success rate", fontsize=10)
ax.set_title("B. Success rate by task category", fontsize=11, fontweight="bold")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, 1.12)
ax.tick_params(labelsize=8)
ax.axvline(0, color="black", lw=0.5)
ax.grid(axis="x", alpha=0.3, lw=0.5)
ax.legend(fontsize=9, loc="lower right")


# ---------------------------------------------------------------------------
# Panels C & D: Heatmaps over perturbations for hard tasks
# ---------------------------------------------------------------------------

def draw_heatmap(ax, mat, tasks, perturbs, title, cmap="RdYlGn"):
    # Separate tasks with any data vs all-NaN
    has_data = [i for i in range(len(tasks)) if not np.all(np.isnan(mat[i]))]
    mat2 = mat[has_data]
    tasks2 = [tasks[i] for i in has_data]

    im = ax.imshow(mat2, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(perturbs)))
    ax.set_xticklabels([PERTURB_LABELS.get(p, p) for p in perturbs],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks2)))
    ylabels = [
        f"[{SUITE_ABBREV.get(TASK_SUITE.get(t, '?'), '?')}] {t.replace('_', ' ')}"
        for t in tasks2
    ]
    ax.set_yticklabels(ylabels, fontsize=6.5)
    for tick, t in zip(ax.get_yticklabels(), tasks2):
        suite = TASK_SUITE.get(t, "unknown")
        tick.set_color(SUITE_COLORS.get(suite, "#333333"))
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            v = mat2[i, j]
            if not np.isnan(v):
                txt = f"{v:.0%}"
                color = "white" if v < 0.3 or v > 0.7 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5.5, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, format="%.0%%")
    return im

draw_heatmap(ax_heat_pi, pi05_mat,   HARD_TASKS, ALL_PERTURBS,
             "C. PI0.5 — success rate per task × perturbation (hard tasks ≤ 20%)")
draw_heatmap(ax_heat_co, cosmos_mat, HARD_TASKS, ALL_PERTURBS,
             "D. Cosmos — success rate per task × perturbation (hard tasks ≤ 20%)")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")
