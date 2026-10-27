"""Input representations for Fashion-MNIST: flattened pixels and HOG (owner: Anant).

Contracts (see project spec): raw X is uint8 (N,28,28) and is never modified in place;
pixel features are float32 (N,784) in [0,1]; HOG features are float32 (N,HOG_DIM) computed
from the float32 X/255 image. Nothing here is fitted on data, so computing features on
train/val/test is leak-free (HOG is a fixed, per-image function).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import hog

from src.config import DATA_DIR, HOG_PARAMS, PIXEL_DIM, SEED

_SIDE = 28  # Fashion-MNIST image side length

# Fixed comparison baseline for the sensitivity study (kept literal on purpose: the study
# must compare against the ORIGINAL baseline even after HOG_PARAMS is changed at M2).
_BASELINE = dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2-Hys")

# Suggested study grid: baseline, coarser cells, finer cells, fewer orientation bins.
DEFAULT_STUDY_CONFIGS = [
    dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)),  # baseline
    dict(orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2)),
    dict(orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2)),
    dict(orientations=6, pixels_per_cell=(4, 4), cells_per_block=(2, 2)),
]


def _check_images(X) -> np.ndarray:
    """Validate the raw-image contract without copying."""
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != (_SIDE, _SIDE):
        raise ValueError(f"expected images of shape (N,{_SIDE},{_SIDE}), got {X.shape}")
    if X.dtype != np.uint8:
        raise ValueError(f"expected uint8 images, got {X.dtype}")
    return X


def flatten_pixels(X) -> np.ndarray:
    """uint8 (N,28,28) -> float32 (N,784) in [0,1]. Does not modify X."""
    X = _check_images(X)
    out = X.reshape(len(X), -1).astype(np.float32)  # astype always copies -> X untouched
    out /= 255.0  # in-place on the NEW array; python float keeps float32
    assert out.shape[1] == PIXEL_DIM
    return out


def hog_feature_dim(orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2), block_norm="L2-Hys") -> int:
    """Analytic HOG length for a 28x28 image (block_norm does not affect the length).

    skimage floors the number of cells (28 // ppc); blocks slide with stride 1 cell:
    blocks = cells - cells_per_block + 1; length = blocks_r*blocks_c*cpb_r*cpb_c*orientations.
    """
    ppc, cpb = tuple(pixels_per_cell), tuple(cells_per_block)
    cells_r, cells_c = _SIDE // ppc[0], _SIDE // ppc[1]
    blocks_r, blocks_c = cells_r - cpb[0] + 1, cells_c - cpb[1] + 1
    if orientations < 1 or min(blocks_r, blocks_c) < 1:
        raise ValueError("configuration yields no complete HOG block on a 28x28 image")
    return int(blocks_r * blocks_c * cpb[0] * cpb[1] * orientations)


def _hog_dim_from_skimage(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2-Hys") -> int:
    """Ground truth: length of a REAL skimage.feature.hog output on a dummy 28x28 image."""
    dummy = np.random.default_rng(0).random((_SIDE, _SIDE)).astype(np.float32)
    return int(hog(dummy, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, block_norm=block_norm,
                   feature_vector=True, channel_axis=None).size)


def extract_hog_features(X, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                         block_norm="L2-Hys") -> np.ndarray:
    """uint8 (N,28,28) -> float32 (N,dim) HOG. Serial, deterministic, X not modified.

    Defaults must equal config.HOG_PARAMS (checked in tests); change them together at M2.
    """
    X = _check_images(X)
    dim = hog_feature_dim(orientations, pixels_per_cell, cells_per_block, block_norm)
    out = np.empty((len(X), dim), dtype=np.float32)  # allocate once, no list appends
    for i in range(len(X)):
        img = X[i].astype(np.float32) / 255.0  # float32 copy; X[i] untouched
        # channel_axis=None = grayscale (replaces deprecated `multichannel`); visualize stays False
        out[i] = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, block_norm=block_norm,
                     feature_vector=True, channel_axis=None)  # shape mismatch would raise here
    return out


def plot_hog_example(X, index, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                     block_norm="L2-Hys", title=None):
    """Original image beside its HOG visualisation. visualize=True is used HERE ONLY."""
    import matplotlib.pyplot as plt
    from skimage import exposure

    X = _check_images(X)
    img = X[index].astype(np.float32) / 255.0
    _, hog_img = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, block_norm=block_norm,
                     visualize=True, feature_vector=True, channel_axis=None)
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, max(float(hog_img.max()), 1e-12)))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Original (28x28)")
    axes[1].imshow(hog_img, cmap="gray"); axes[1].set_title(f"HOG ppc={tuple(pixels_per_cell)}, o={orientations}")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.close(fig)  # avoid double display in notebooks; the returned Figure still renders
    return fig


# ----------------------------------------------------------------------------- caching
def _hog_tag(p) -> str:
    return (f"o{p['orientations']}_ppc{p['pixels_per_cell'][0]}x{p['pixels_per_cell'][1]}"
            f"_cpb{p['cells_per_block'][0]}x{p['cells_per_block'][1]}_{p['block_norm']}")


def _cache_key(kind: str, split_name: str, X: np.ndarray, hog_params: dict) -> str:
    """Key = kind + split_name + N + (HOG params for hog) + short content hash of X.
    The content hash is an addition to the required fields: it stops a stale file being
    loaded when a different array is passed under the same split_name and N."""
    digest = hashlib.sha1(np.ascontiguousarray(X)).hexdigest()[:10]
    if kind == "pixels":
        return f"pixels_{split_name}_N{len(X)}_{digest}"
    return f"hog_{split_name}_N{len(X)}_{_hog_tag(hog_params)}_{digest}"


def cached_features(X, kind, split_name, cache_dir=DATA_DIR / "cache") -> np.ndarray:
    """Return features for X, loading <cache_dir>/<key>.npy if present, else computing + saving.
    kind in {"pixels","hog"}. HOG parameters are read from config.HOG_PARAMS at call time."""
    if kind not in ("pixels", "hog"):
        raise ValueError(f"kind must be 'pixels' or 'hog', got {kind!r}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(split_name)):
        raise ValueError("split_name may only contain letters, digits, '_' and '-'")
    X = _check_images(X)
    params = dict(HOG_PARAMS)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (_cache_key(kind, split_name, X, params) + ".npy")
    expected_dim = PIXEL_DIM if kind == "pixels" else hog_feature_dim(**params)

    if path.exists():
        arr = np.load(path)
        if arr.shape == (len(X), expected_dim) and arr.dtype == np.float32:
            return arr
        warnings.warn(f"cache file {path.name} has wrong shape/dtype; recomputing")

    arr = flatten_pixels(X) if kind == "pixels" else extract_hog_features(X, **params)
    tmp = path.with_name(path.name + ".tmp")  # write-then-rename: no half-written cache files
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return arr


# ----------------------------------------------------------------------------- HOG study
def _mock_build_pipeline(model_name, params=None):
    """TEMPORARY stand-in for classical_models.build_classical_pipeline (logreg only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    assert model_name == "logreg"
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=(params or {}).get("C", 1.0), max_iter=200))])


def _mock_stratified_subset(X, y, n, seed=SEED):
    """TEMPORARY stand-in for data.stratified_subset."""
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    idx = np.sort(idx)
    return X[idx], y[idx], idx


def _resolve_dependencies():
    """Use teammates' real functions when they exist; fall back to mocks ONLY if the module
    itself is missing (any other import error is re-raised). The source is recorded."""
    try:
        from src.classical_models import build_classical_pipeline
        build, src_build = build_classical_pipeline, "src.classical_models"
    except ModuleNotFoundError as e:
        if e.name != "src.classical_models":
            raise
        warnings.warn("src.classical_models missing: using LOCAL MOCK pipeline (results not reportable)")
        build, src_build = _mock_build_pipeline, "LOCAL MOCK"
    try:
        from src.data import stratified_subset
        subset, src_subset = stratified_subset, "src.data"
    except ModuleNotFoundError as e:
        if e.name != "src.data":
            raise
        warnings.warn("src.data missing: using LOCAL MOCK stratified subset")
        subset, src_subset = _mock_stratified_subset, "LOCAL MOCK"
    return build, subset, src_build, src_subset


def _config_label(p) -> str:
    return _hog_tag(p)


def hog_sensitivity_study(X_tr, y_tr, X_val, y_val, configs, subset_n=10000, seed=SEED) -> pd.DataFrame:
    """VALIDATION-ONLY HOG study. Per config: HOG on a stratified train subset and on the
    full validation set, LogReg (C=1.0, fixed for all configs), scored on validation.

    extract_seconds = wall time to extract HOG for subset + validation images (serial).
    The StandardScaler sits inside the pipeline and is fitted on the train subset only.
    Test data must never be passed here. df.attrs records the pipeline/subset source and size.
    """
    from sklearn.metrics import accuracy_score, f1_score

    build, subset_fn, src_build, src_subset = _resolve_dependencies()
    X_tr, X_val = _check_images(X_tr), _check_images(X_val)
    n_used = min(int(subset_n), len(X_tr))
    if n_used < len(X_tr):
        X_sub, y_sub, _ = subset_fn(X_tr, y_tr, n_used, seed)
    else:
        X_sub, y_sub = X_tr, y_tr  # subset_n >= N: use everything (no subsampling)

    rows = []
    for cfg in configs:
        unknown = set(cfg) - set(_BASELINE)
        if unknown:
            raise ValueError(f"unknown HOG keys: {unknown}")
        p = {**_BASELINE, **cfg}
        t0 = time.perf_counter()
        F_tr = extract_hog_features(X_sub, **p)
        F_val = extract_hog_features(X_val, **p)
        secs = time.perf_counter() - t0
        pipe = build("logreg", {"C": 1.0})
        pipe.fit(F_tr, y_sub)
        pred = pipe.predict(F_val)
        rows.append(dict(config=_config_label(p), hog_dim=F_tr.shape[1], extract_seconds=round(secs, 3),
                         val_accuracy=float(accuracy_score(y_val, pred)),
                         val_macro_f1=float(f1_score(y_val, pred, average="macro"))))
    df = pd.DataFrame(rows, columns=["config", "hog_dim", "extract_seconds", "val_accuracy", "val_macro_f1"])
    df.attrs.update(pipeline_source=src_build, subset_source=src_subset, subset_n_used=n_used, n_val=len(X_val))
    return df


if __name__ == "__main__":  # python -m src.features : version, dimension check, timing (mock images)
    import platform
    import skimage
    print("numpy", np.__version__, "| scikit-image", skimage.__version__, "| python", platform.python_version())
    for c in DEFAULT_STUDY_CONFIGS:
        p = {**_BASELINE, **c}
        a, b = hog_feature_dim(**p), _hog_dim_from_skimage(**p)
        print(f"{_hog_tag(p):32s} analytic={a} skimage={b} match={a == b}")
    Xm = np.random.default_rng(0).integers(0, 256, (1000, 28, 28), dtype=np.uint8)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); extract_hog_features(Xm); ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f"HOG 1000 imgs: runs={[round(t, 3) for t in ts]} median={med:.3f}s -> {med:.3f} ms/img")
    print(f"ESTIMATE for 70,000 images (linear extrapolation, serial): {med * 70:.0f}s = {med * 70 / 60:.1f} min")
    t0 = time.perf_counter(); flatten_pixels(Xm); print(f"flatten 1000 imgs: {time.perf_counter() - t0:.5f}s")
