"""EDA cells (Akhil). Run from anywhere: python docs/eda_cells.py
Each '# %%' block is one notebook cell. Test LABELS are used only for the class-count bar plot (cell a)."""
# %% Setup: imports, data, split (fixed seed)
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")  # delete this line inside a notebook to show plots inline
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, FIGURES_DIR, RESULTS_DIR, SEED
from src.data import (class_centroid_distances, load_dataset, pca_projection,
                      prepare_splits, stratified_subset)

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
X_full, y_full, X_test, y_test = load_dataset()
s = prepare_splits(X_full, y_full)
X_tr, y_tr = s["X_train"], s["y_train"]  # 54k TRAIN split: all statistics below use this only
rng = np.random.default_rng(SEED)

# %% a) Class distribution: train (official 60k) vs test (official 10k) label counts
fig, ax = plt.subplots(figsize=(10, 4))
x, w = np.arange(10), 0.4
ax.bar(x - w / 2, np.bincount(y_full, minlength=10), w, label="Train (official 60k)")
ax.bar(x + w / 2, np.bincount(y_test, minlength=10), w, label="Test (official 10k)")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_xlabel("Class")
ax.set_ylabel("Number of images")
ax.set_title("Fashion-MNIST class distribution")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
plt.close(fig)

# %% b) 5 random images per class (fixed seed, from the train split)
fig, axes = plt.subplots(10, 5, figsize=(6, 12))
for c in range(10):
    idx = rng.choice(np.where(y_tr == c)[0], 5, replace=False)
    for j, i in enumerate(idx):
        ax = axes[c, j]
        ax.imshow(X_tr[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=8)
fig.suptitle("Five random training images per class")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_samples_per_class.png", dpi=150)
plt.close(fig)

# %% c) Mean image per class (train split)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for c, ax in enumerate(axes.ravel()):
    ax.imshow(X_tr[y_tr == c].mean(axis=0), cmap="gray")
    ax.set_title(CLASS_NAMES[c], fontsize=9)
    ax.axis("off")
fig.suptitle("Mean image per class (54k train split)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_mean_images.png", dpi=150)
plt.close(fig)

# %% d) Confusable upper-body classes: 8 examples each
panel = [6, 0, 2, 4]  # Shirt, T-shirt/top, Pullover, Coat
fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
for r, c in enumerate(panel):
    idx = rng.choice(np.where(y_tr == c)[0], 8, replace=False)
    for j, i in enumerate(idx):
        ax = axes[r, j]
        ax.imshow(X_tr[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=9)
fig.suptitle("Shirt vs T-shirt/top vs Pullover vs Coat (8 random training images each)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_confusable_panel.png", dpi=150)
plt.close(fig)

# %% e) Pixel statistics on the TRAIN split only (scale 0-1)
pix = {
    "n_images": int(len(X_tr)),
    "mean": float(X_tr.mean(dtype=np.float64) / 255),
    "std": float(X_tr.std(dtype=np.float64) / 255),
    "fraction_zero_pixels": float((X_tr == 0).mean()),
    "per_class_mean_intensity": {CLASS_NAMES[c]: float(X_tr[y_tr == c].mean(dtype=np.float64) / 255)
                                 for c in range(10)},
}
print(json.dumps(pix, indent=2))
dq_path = RESULTS_DIR / "data_quality.json"
dq = json.loads(dq_path.read_text()) if dq_path.exists() else {}
dq["pixel_stats_train54k"] = pix

# %% f) PCA (fit on the 54k train split only) and class-centroid distances
Z, evr = pca_projection(X_tr, 50)
dq["pca_evr_first2"] = [float(v) for v in evr[:2]]
print("PCA variance, PC1 and PC2:", dq["pca_evr_first2"], "| first 10:", float(evr[:10].sum()), "| 50:", float(evr.sum()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(1, 51), np.cumsum(evr), marker="o", markersize=3)
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA explained variance (fit on 54k train split)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_pca_variance.png", dpi=150)
plt.close(fig)

Zs, ys, _ = stratified_subset(Z, y_tr, 5000)  # 5000 points for readability, fixed seed
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(10):
    ax.scatter(Zs[ys == c, 0], Zs[ys == c, 1], s=4, alpha=0.6, color=plt.cm.tab10(c), label=CLASS_NAMES[c])
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.set_title("PCA 2D projection (5000 stratified train images)")
ax.legend(markerscale=4, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_pca_scatter.png", dpi=150)
plt.close(fig)

D = class_centroid_distances(X_tr, y_tr)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(D.values, cmap="viridis_r")
ax.set_xticks(range(10))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticks(range(10))
ax.set_yticklabels(CLASS_NAMES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{D.values[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")
ax.set_title("Euclidean distance between class mean images (train split, pixels in 0-1)")
fig.colorbar(im, ax=ax, label="Distance")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_centroid_heatmap.png", dpi=150)
plt.close(fig)

dq_path.write_text(json.dumps(dq, indent=2))
print("saved figures:", sorted(p.name for p in FIGURES_DIR.glob("eda_*.png")))
