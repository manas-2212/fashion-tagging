"""Tests for src/hybrid.py (owner: Anant). Synthetic features: fast."""
import numpy as np
import pytest
import torch

from src.dl_models import FashionNet
from src.hybrid import (HybridFeatures, HybridModel, cascade_predict, complementarity, extract_embeddings,
                        fit_hog_pca, fit_hybrid, pca_transform, tune_cascade, tune_hybrid)


def _features(n=300, seed=0):
    """Two blocks: 'emb' separates classes 0-4 from 5-9, 'hog' separates even from odd classes."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(10), n // 10).astype(np.int64)
    emb = rng.normal(0, 1, (n, 16)).astype(np.float32)
    emb[:, 0] += 4.0 * (y >= 5)
    emb[:, 1:6] += 3.0 * np.eye(5)[y % 5]
    hog = rng.normal(0, 1, (n, 60)).astype(np.float32)
    hog[:, :10] += 3.0 * np.eye(10)[y]
    return emb, hog, y


def test_extract_embeddings_shape_and_eval_mode():
    model = FashionNet(widths=(8, 16), dropout=0.5)
    model.train()
    x = torch.randn(10, 1, 28, 28)
    a = extract_embeddings(model, x, "cpu", batch_size=4)
    b = extract_embeddings(model, x, "cpu", batch_size=10)
    assert a.shape == (10, 16) and a.dtype == np.float32
    assert np.allclose(a, b, atol=1e-5) and not model.training       # deterministic: eval mode, no dropout


def test_feature_dimensions_per_mode():
    emb, hog, y = _features()
    for mode, dim in (("emb", 16), ("hog", 8), ("fusion", 24)):
        f = HybridFeatures(mode, hog_dim=8).fit(emb, hog, y)
        assert f.transform(emb, hog).shape == (len(y), dim) and f.output_dim == dim
    with pytest.raises(ValueError):
        HybridFeatures("stack")
    with pytest.raises(ValueError):
        HybridFeatures("fusion").fit(None, hog, y)


def test_alpha_scales_only_the_hog_block():
    emb, hog, y = _features()
    pca = fit_hog_pca(hog, y, 8)
    a = HybridFeatures("fusion", hog_dim=8, alpha=1.0).fit(emb, hog, y, pca=pca).transform(emb, hog)
    b = HybridFeatures("fusion", hog_dim=8, alpha=0.5).fit(emb, hog, y, pca=pca).transform(emb, hog)
    assert np.allclose(a[:, :16], b[:, :16]) and np.allclose(0.5 * a[:, 16:], b[:, 16:], atol=1e-6)


def test_preprocessing_is_fitted_on_training_data_only():
    emb, hog, y = _features()
    f = HybridFeatures("fusion", hog_dim=8).fit(emb[:200], hog[:200], y[:200])
    before = (f.scaler_.mean_.copy(), f.pca_.components_.copy())
    f.transform(emb[200:] + 100.0, hog[200:] + 100.0)                 # transforming new data must not refit
    assert np.array_equal(before[0], f.scaler_.mean_) and np.array_equal(before[1], f.pca_.components_)
    assert np.allclose(f.scaler_.mean_, emb[:200].mean(axis=0), atol=1e-5)


def test_shared_pca_must_be_large_enough_and_whitened():
    emb, hog, y = _features()
    pca = fit_hog_pca(hog, y, 4)
    with pytest.raises(ValueError):
        HybridFeatures("hog", hog_dim=8).fit(None, hog, y, pca=pca)
    z = pca_transform(pca, hog, chunk=64)
    assert z.shape == (len(y), 4) and np.allclose(z.std(axis=0), 1.0, atol=0.15)


def test_fusion_beats_each_block_when_blocks_are_complementary():
    emb, hog, y = _features(600)
    tr, va = slice(0, 400), slice(400, 600)
    acc = {}
    for mode in ("emb", "hog", "fusion"):
        m = HybridModel(mode, C=10.0, hog_dim=10).fit(y[tr], emb[tr], hog[tr])
        acc[mode] = (m.predict(emb[va], hog[va]) == y[va]).mean()
        assert m.fit_info_["n_train_used"] == 400
    assert acc["fusion"] >= acc["hog"] and acc["fusion"] >= acc["emb"]


def test_tune_hybrid_returns_grid_and_valid_best():
    emb, hog, y = _features(400)
    tr, va = slice(0, 300), slice(300, 400)
    grid = dict(C=(1.0, 10.0), hog_dim=(4, 8), alpha=(0.5, 1.0))
    params, df = tune_hybrid("fusion", grid, y[tr], y[va], emb[tr], emb[va], hog[tr], hog[va], subset_n=200)
    assert len(df) == 8 and set(df["n_train_used"]) == {200}
    best = df.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).iloc[0]
    assert df.loc[(df["C"] == params["C"]) & (df["hog_dim"] == params["hog_dim"])
                  & (df["alpha"] == params["alpha"]), "val_macro_f1"].iloc[0] == best["val_macro_f1"]
    p_emb, df_emb = tune_hybrid("emb", dict(C=(1.0, 10.0)), y[tr], y[va], emb_tr=emb[tr], emb_val=emb[va])
    assert len(df_emb) == 2 and p_emb["mode"] == "emb"
    model = fit_hybrid(params, y[tr], emb[tr], hog[tr])
    assert model.predict(emb[va], hog[va]).shape == (100,)


def test_cascade_routes_only_low_confidence_images():
    probs = np.array([[0.9, 0.1] + [0] * 8, [0.55, 0.45] + [0] * 8, [0.2, 0.8] + [0] * 8])
    expert = np.array([5, 5, 5])
    pred, routed = cascade_predict(probs, expert, tau=0.7)
    assert routed.tolist() == [False, True, False] and pred.tolist() == [0, 5, 1]
    assert cascade_predict(probs, expert, 0.0)[0].tolist() == [0, 0, 1]          # tau 0 = pure CNN
    assert cascade_predict(probs, expert, 1.01)[0].tolist() == [5, 5, 5]         # tau > 1 = pure expert
    with pytest.raises(ValueError):
        cascade_predict(probs, expert[:2], 0.5)


def test_tune_cascade_prefers_fewest_routed_on_ties():
    y = np.array([0, 1, 2, 3])
    probs = np.zeros((4, 10))
    probs[np.arange(4), y] = [0.95, 0.9, 0.6, 0.55]
    probs[np.arange(4), (y + 1) % 10] = 1 - probs[np.arange(4), y]
    tau, df = tune_cascade(probs, y.copy(), y, (0.0, 0.5, 0.7, 0.99))            # CNN and expert are both right
    assert tau == 0.0 and list(df["routed_fraction"]) == [0.0, 0.0, 0.5, 1.0]
    wrong_cnn = probs.copy()
    wrong_cnn[3] = 0
    wrong_cnn[3, 9] = 0.55                                                        # low-confidence CNN mistake
    wrong_cnn[3, 0] = 0.45
    tau, _ = tune_cascade(wrong_cnn, y.copy(), y, (0.0, 0.5, 0.7, 0.99))
    assert tau == 0.7                                                             # smallest tau that fixes it


def test_complementarity_counts():
    y = np.array([0, 0, 0, 0, 0, 0])
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 0, 1, 0, 1, 1])
    c = complementarity(y, a, b)
    assert (c["both_correct"], c["only_a_correct"], c["only_b_correct"], c["both_wrong"]) == (2, 1, 1, 2)
    assert c["oracle_accuracy"] == pytest.approx(4 / 6) and c["share_of_a_errors_fixed_by_b"] == pytest.approx(1 / 3)
    assert c["oracle_accuracy"] >= max(c["accuracy_a"], c["accuracy_b"])
