"""Robustness to simple image corruptions (owner: Akhil).

Fashion-MNIST images are clean, centred and evenly lit; photographs of real inventory are not. This module
measures how fast each model's accuracy falls when the input is perturbed in controlled ways. It is a stress
test of already-trained models: nothing is trained, tuned or selected here, and it runs only AFTER the test gate.

Every corruption maps uint8 (N,28,28) -> uint8 (N,28,28), so the same corrupted image feeds both the CNN
(normalised tensor) and the handcrafted pipeline (HOG). Level 0 is always the identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEED


def _check(X) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype != np.uint8 or X.ndim != 3:
        raise ValueError(f"expected uint8 (N,H,W) images, got {X.dtype} {X.shape}")
    return X


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def gaussian_noise(X, level: float, seed: int = SEED) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation `level` (pixel scale 0-1)."""
    X = _check(X)
    if level == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    return _to_uint8(X.astype(np.float32) / 255.0 + rng.normal(0.0, level, X.shape).astype(np.float32))


def brightness(X, level: float, seed: int = SEED) -> np.ndarray:
    """Add a constant `level` to every pixel (0-1 scale): a uniformly brighter photograph."""
    X = _check(X)
    return _to_uint8(X.astype(np.float32) / 255.0 + level)


def contrast(X, level: float, seed: int = SEED) -> np.ndarray:
    """Reduce contrast: pixel -> pixel * (1 - level). level 0.5 halves every intensity."""
    X = _check(X)
    return _to_uint8(X.astype(np.float32) / 255.0 * (1.0 - level))


def shift(X, level: int, seed: int = SEED) -> np.ndarray:
    """Translate every image by `level` pixels right and down; uncovered pixels become background (0)."""
    X = _check(X)
    k = int(level)
    if k == 0:
        return X.copy()
    out = np.zeros_like(X)
    out[:, k:, k:] = X[:, :-k, :-k]
    return out


def occlusion(X, level: int, seed: int = SEED) -> np.ndarray:
    """Cover one random `level` x `level` square per image with mid-gray (128)."""
    X = _check(X)
    k = int(level)
    out = X.copy()
    if k == 0:
        return out
    rng = np.random.default_rng(seed)
    n, h, w = X.shape
    ys, xs = rng.integers(0, h - k + 1, n), rng.integers(0, w - k + 1, n)
    for i in range(n):
        out[i, ys[i]:ys[i] + k, xs[i]:xs[i] + k] = 128
    return out


# name -> (function, levels, axis label). The first level is always 0 (clean images).
CORRUPTIONS = {
    "gaussian_noise": (gaussian_noise, (0.0, 0.05, 0.10, 0.20), "noise std (pixel scale 0-1)"),
    "brightness": (brightness, (0.0, 0.10, 0.20, 0.30), "added brightness"),
    "contrast": (contrast, (0.0, 0.25, 0.50, 0.75), "contrast reduction"),
    "shift": (shift, (0, 1, 2, 3), "shift (pixels)"),
    "occlusion": (occlusion, (0, 5, 8, 11), "occluding square side (pixels)"),
}


def evaluate_robustness(predict_fns: dict, X, y, corruptions: dict = CORRUPTIONS, seed: int = SEED) -> pd.DataFrame:
    """Accuracy of each model on each corrupted copy of X.

    predict_fns: {model name: callable(X_uint8) -> y_pred (N,)}.
    Returns DataFrame[model, corruption, level, accuracy, drop_vs_clean].
    """
    X, y = _check(X), np.asarray(y)
    rows = []
    for cname, (fn, levels, _) in corruptions.items():
        clean = {}
        for level in levels:
            Xc = fn(X, level, seed)
            for mname, predict in predict_fns.items():
                acc = float((np.asarray(predict(Xc)) == y).mean())
                if level == levels[0]:
                    clean[mname] = acc
                rows.append({"model": mname, "corruption": cname, "level": float(level), "accuracy": acc,
                             "drop_vs_clean": clean[mname] - acc})
    return pd.DataFrame(rows)


def plot_robustness(df: pd.DataFrame, corruptions: dict = CORRUPTIONS):
    """Small multiples: one panel per corruption, one line per model, shared accuracy axis."""
    import matplotlib.pyplot as plt

    from src.dl_evaluation import INK, PALETTE, style_axes

    names = [c for c in corruptions if c in set(df["corruption"])]
    models = list(dict.fromkeys(df["model"]))
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cname in zip(axes, names):
        sub = df[df["corruption"] == cname]
        for i, m in enumerate(models):
            s = sub[sub["model"] == m].sort_values("level")
            ax.plot(np.arange(len(s)), s["accuracy"], marker=markers[i % len(markers)], markersize=5, linewidth=2,
                    color=PALETTE[i % len(PALETTE)], label=m)
        levels = sorted(sub["level"].unique())
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.set_xlabel(corruptions[cname][2], fontsize=8.5)
        ax.set_title(cname.replace("_", " "), fontsize=10, color=INK)
        style_axes(ax)
    axes[0].set_ylabel("Accuracy on corrupted test images")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Robustness stress test (level 0 = clean images)", fontsize=11, color=INK)
    fig.tight_layout()
    return fig
