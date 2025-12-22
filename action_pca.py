import json, glob, os
import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from collections import defaultdict



ROOT = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero"
SPLITS = ["10", "goal", "object", "spatial"]   # folders under ROOT

def load_actions(root, splits, only_policy=True, only_success=False):
    X = []
    meta = []  # keep labels for plotting: split, task_id, trial_id, t, etc.

    for split in splits:
        action_dir = os.path.join(root, split, "actions")
        for fp in glob.glob(os.path.join(action_dir, "**", "*.json"), recursive=True):
            with open(fp, "r") as f:
                ex = json.load(f)

            if only_success and (not ex.get("success", False)):
                continue

            for step in ex.get("actions", []):
                if only_policy and step.get("kind") != "policy":
                    continue
                a = step.get("action", None)
                if a is None or len(a) != 7:
                    continue

                X.append(a)
                meta.append({
                    "split": split,
                    "task_id": ex.get("task_id"),
                    "trial_id": ex.get("trial_id"),
                    "t": step.get("t"),
                    "path": fp,
                })

    X = np.asarray(X, dtype=np.float32)
    return X, meta

X, meta = load_actions(ROOT, SPLITS, only_policy=True, only_success=False)
print("X shape:", X.shape)   # (N, 7)



scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=0)  # use 2 for a 2D plot
Z = pca.fit_transform(Xz)

print("Explained variance ratio:", pca.explained_variance_ratio_)




# 1) fit PCA (3D)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=3, random_state=0)
Z = pca.fit_transform(Xz)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# 2) 3D scatter colored by split

groups = defaultdict(list)
for i, m in enumerate(meta):
    groups[m["split"]].append(i)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for split, idxs in groups.items():
    pts = Z[idxs]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.4, label=split)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA of 7D actions (standardized)")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig('action_pca_3d.png')

Xz = StandardScaler().fit_transform(X)
Z3 = PCA(n_components=3, random_state=0).fit_transform(Xz)

splits = [m["split"] for m in meta]

fig = px.scatter_3d(
    x=Z3[:, 0], y=Z3[:, 1], z=Z3[:, 2],
    color=splits,
    opacity=0.5,
    title="3D PCA of 7D actions (standardized)",
    labels={"x":"PC1", "y":"PC2", "z":"PC3"},
)

# Optional: richer hover info
fig.update_traces(
    hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<br>split=%{marker.color}<extra></extra>"
)

# Save interactive HTML (no display needed)
out_html = "actions_pca_3d.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print("Wrote:", out_html)


pca_full = PCA(n_components=7, random_state=0).fit(Xz)

plt.figure()
plt.plot(np.arange(1, 8), np.cumsum(pca_full.explained_variance_ratio_), marker="o")
plt.xticks(range(1, 8))
plt.xlabel("# principal components")
plt.ylabel("cumulative explained variance")
plt.title("Explained variance of action PCA")
plt.tight_layout()
plt.savefig('action_variance.png')

# group indices by (split, path)
by_ep = defaultdict(list)
for i, m in enumerate(meta):
    by_ep[(m["split"], m["path"])].append(i)

plt.figure()
for (split, path), idxs in list(by_ep.items())[:50]:  # cap to avoid huge plots
    idxs = sorted(idxs, key=lambda i: meta[i]["t"])
    pts = Z[idxs]
    plt.plot(pts[:,0], pts[:,1], alpha=0.2)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Action trajectories in PCA space (subset of episodes)")
plt.tight_layout()
plt.savefig('action_trajectories.png')
