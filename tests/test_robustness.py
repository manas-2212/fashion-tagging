"""Tests for src/robustness.py (owner: Akhil)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.robustness import (CORRUPTIONS, brightness, contrast, evaluate_robustness, gaussian_noise, occlusion,
                            plot_robustness, shift)


def _images(n=40, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 28, 28), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_contract_shape_dtype_and_level_zero_identity(name):
    fn, levels, _ = CORRUPTIONS[name]
    X = _images()
    before = X.copy()
    assert levels[0] == 0
    assert np.array_equal(fn(X, levels[0]), X)                       # level 0 = clean images
    for level in levels[1:]:
        out = fn(X, level)
        assert out.shape == X.shape and out.dtype == np.uint8
        assert not np.array_equal(out, X)
    assert np.array_equal(X, before)                                 # input never modified in place


def test_noise_is_seeded_and_grows_with_level():
    X = _images()
    assert np.array_equal(gaussian_noise(X, 0.1, seed=1), gaussian_noise(X, 0.1, seed=1))
    assert not np.array_equal(gaussian_noise(X, 0.1, seed=1), gaussian_noise(X, 0.1, seed=2))
    mid = np.full((10, 28, 28), 128, dtype=np.uint8)
    small = np.abs(gaussian_noise(mid, 0.05).astype(int) - 128).mean()
    large = np.abs(gaussian_noise(mid, 0.20).astype(int) - 128).mean()
    assert large > 2 * small


def test_brightness_and_contrast_values():
    X = np.full((2, 28, 28), 100, dtype=np.uint8)
    assert int(brightness(X, 0.2)[0, 0, 0]) == 151                   # 100 + 0.2 * 255
    assert int(brightness(np.full((1, 28, 28), 250, dtype=np.uint8), 0.3).max()) == 255   # clipped
    assert int(contrast(X, 0.5)[0, 0, 0]) == 50


def test_shift_moves_content_and_fills_background():
    X = _images()
    out = shift(X, 2)
    assert np.array_equal(out[:, 2:, 2:], X[:, :-2, :-2])
    assert out[:, :2, :].max() == 0 and out[:, :, :2].max() == 0


def test_occlusion_covers_one_square():
    X = np.zeros((20, 28, 28), dtype=np.uint8)
    out = occlusion(X, 8, seed=3)
    assert ((out == 128).reshape(20, -1).sum(axis=1) == 64).all()


def test_bad_input_raises():
    with pytest.raises(ValueError):
        gaussian_noise(np.zeros((2, 28, 28), dtype=np.float32), 0.1)


def test_evaluate_robustness_table_and_plot():
    X = _images(60)
    y = (X.reshape(60, -1).mean(axis=1) > 127.5).astype(int)
    fns = {"mean threshold": lambda Z: (Z.reshape(len(Z), -1).mean(axis=1) > 127.5).astype(int),
           "always zero": lambda Z: np.zeros(len(Z), dtype=int)}
    df = evaluate_robustness(fns, X, y)
    assert len(df) == 2 * sum(len(v[1]) for v in CORRUPTIONS.values())
    clean = df[df["level"] == 0]
    assert (clean[clean["model"] == "mean threshold"]["accuracy"] == 1.0).all()
    assert (clean["drop_vs_clean"] == 0).all()
    const = df[df["model"] == "always zero"]
    assert const["accuracy"].nunique() == 1                          # a constant model is unaffected
    bright = df[(df["model"] == "mean threshold") & (df["corruption"] == "brightness") & (df["level"] == 0.3)]
    assert bright["accuracy"].iloc[0] < 1.0
    fig = plot_robustness(df)
    assert len(fig.axes) == len(CORRUPTIONS)
    import matplotlib.pyplot as plt
    plt.close("all")
