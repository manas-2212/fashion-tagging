"""Tests for src/features.py on MOCK data. Run from repo root: python -m pytest tests/test_features.py -q"""
import warnings

import numpy as np
import pytest

import src.features as F
from src.config import HOG_DIM, HOG_PARAMS, PIXEL_DIM


def mock(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 28, 28), dtype=np.uint8), rng.integers(0, 10, n).astype(np.int64)


def test_flatten_pixels_contract():
    X, _ = mock(50)
    X0 = X.copy()
    out = F.flatten_pixels(X)
    assert out.shape == (50, PIXEL_DIM) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    for i in (0, 17, 49):  # row i == X[i].reshape(-1)/255 (exact in float32, close to float64 version)
        assert np.array_equal(out[i], X[i].reshape(-1).astype(np.float32) / 255)
        assert np.allclose(out[i], X[i].reshape(-1) / 255, atol=1e-7)
    assert np.array_equal(X, X0)


def test_input_validation():
    X, _ = mock(3)
    for bad in (X.astype(np.float32), X[:, :27, :], X[0]):
        with pytest.raises(ValueError):
            F.flatten_pixels(bad)
        with pytest.raises(ValueError):
            F.extract_hog_features(bad)


def test_hog_dimension_analytic_matches_skimage():
    assert F.hog_feature_dim() == 1296 == HOG_DIM == F._hog_dim_from_skimage()
    for cfg in F.DEFAULT_STUDY_CONFIGS + [dict(orientations=8, pixels_per_cell=(5, 5), cells_per_block=(1, 1))]:
        p = {**F._BASELINE, **cfg}
        assert F.hog_feature_dim(**p) == F._hog_dim_from_skimage(**p), p


def test_defaults_match_config():  # guards against features.py defaults drifting from config.HOG_PARAMS at M2
    import inspect
    sig = inspect.signature(F.extract_hog_features).parameters
    for k, v in HOG_PARAMS.items():
        assert sig[k].default == v, k
    assert F.hog_feature_dim() == HOG_DIM


def test_hog_contract_finite_deterministic():
    X, _ = mock(40)
    a, b = F.extract_hog_features(X), F.extract_hog_features(X)
    assert a.shape == (40, HOG_DIM) and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_hog_finite_on_constant_images():  # zero-gradient images must not give NaN (division by zero in norm)
    X = np.stack([np.zeros((28, 28), np.uint8), np.full((28, 28), 255, np.uint8)])
    out = F.extract_hog_features(X)
    assert np.isfinite(out).all()


def test_same_dim_for_batch_sizes():
    X, _ = mock(100)
    for n in (1, 7, 100):
        out = F.extract_hog_features(X[:n])
        assert out.shape == (n, HOG_DIM) == (n, F.hog_feature_dim())


def test_vertical_vs_horizontal_edge_differ():
    v = np.zeros((28, 28), np.uint8); v[:, 14:] = 255   # vertical edge  -> horizontal gradient
    h = np.zeros((28, 28), np.uint8); h[14:, :] = 255   # horizontal edge -> vertical gradient
    fv, fh = F.extract_hog_features(np.stack([v, h]))
    assert not np.array_equal(fv, fh)
    bins_v, bins_h = fv.reshape(-1, 9).sum(0), fh.reshape(-1, 9).sum(0)
    assert bins_v.argmax() != bins_h.argmax()  # energy sits in different orientation bins
    print("dominant orientation bin: vertical edge", int(bins_v.argmax()), "| horizontal edge", int(bins_h.argmax()))


def test_batch_independence_and_no_inplace_modification():
    X, _ = mock(20)
    X.setflags(write=False)  # any in-place write would raise
    full = F.extract_hog_features(X)
    for i in (0, 9, 19):
        assert np.array_equal(full[i], F.extract_hog_features(X[i:i + 1])[0])
    rev = F.extract_hog_features(X[::-1])  # different batch composition/order
    assert np.array_equal(rev[::-1], full)
    F.flatten_pixels(X)  # read-only input also fine


def test_cache_roundtrip_and_key_changes(tmp_path):
    X, _ = mock(30)
    first = F.cached_features(X, "hog", "tr", cache_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.npy"))
    assert len(files) == 1 and "hog_tr_N30" in files[0] and "o9_ppc4x4_cpb2x2_L2-Hys" in files[0]
    original = F.extract_hog_features
    F.extract_hog_features = lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed instead of loading"))
    try:
        second = F.cached_features(X, "hog", "tr", cache_dir=tmp_path)  # must come from disk
    finally:
        F.extract_hog_features = original
    assert np.array_equal(first, second) and second.dtype == np.float32
    # different split_name / N / kind -> different files
    F.cached_features(X, "hog", "val", cache_dir=tmp_path)
    F.cached_features(X[:20], "hog", "tr", cache_dir=tmp_path)
    px = F.cached_features(X, "pixels", "tr", cache_dir=tmp_path)
    assert px.shape == (30, PIXEL_DIM) and len(list(tmp_path.glob("*.npy"))) == 4
    # changed HOG parameter -> different file AND correct (different) dimension
    saved = F.HOG_PARAMS
    F.HOG_PARAMS = {**saved, "pixels_per_cell": (7, 7)}
    try:
        other = F.cached_features(X, "hog", "tr", cache_dir=tmp_path)
    finally:
        F.HOG_PARAMS = saved
    assert other.shape == (30, 324) and len(list(tmp_path.glob("*.npy"))) == 5
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        F.cached_features(X, "bogus", "tr", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        F.cached_features(X, "hog", "../evil", cache_dir=tmp_path)


def test_cache_not_stale_for_different_content(tmp_path):
    A, _ = mock(15, seed=1)
    B, _ = mock(15, seed=2)  # same split_name and N, different images
    fa = F.cached_features(A, "hog", "tr", cache_dir=tmp_path)
    fb = F.cached_features(B, "hog", "tr", cache_dir=tmp_path)
    assert not np.array_equal(fa, fb) and np.array_equal(fb, F.extract_hog_features(B))


def test_plot_hog_example():
    X, _ = mock(3)
    fig = F.plot_hog_example(X, 1)
    assert len(fig.axes) == 2


def test_sensitivity_study_mock_end_to_end():
    Xtr, ytr = mock(300, seed=3)
    Xv, yv = mock(100, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mock-pipeline notice + ConvergenceWarning on noise data
        df = F.hog_sensitivity_study(Xtr, ytr, Xv, yv, F.DEFAULT_STUDY_CONFIGS, subset_n=150)
    assert list(df.columns) == ["config", "hog_dim", "extract_seconds", "val_accuracy", "val_macro_f1"]
    assert df["hog_dim"].tolist() == [1296, 324, 6084, 864]
    assert ((df[["val_accuracy", "val_macro_f1"]] >= 0) & (df[["val_accuracy", "val_macro_f1"]] <= 1)).all().all()
    assert df.attrs["subset_n_used"] == 150
    print("study plumbing OK; pipeline_source =", df.attrs["pipeline_source"], "(mock noise data: accuracies meaningless)")
