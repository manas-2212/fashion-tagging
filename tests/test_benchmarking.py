"""Tests for src/benchmarking.py (python -m pytest tests/test_benchmarking.py -s)."""
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmarking import benchmark_classical_inference, benchmark_cnn_inference, count_trainable_params
from src.cnn import build_cnn

KEYS = {"device", "batch_size", "n_images", "total_s_median", "per_image_ms", "repeats_s", "includes_preprocessing"}


def test_cnn_benchmark_cpu():
    m = build_cnn(); m.train()
    X = torch.randn(600, 1, 28, 28)
    r = benchmark_cnn_inference(m, X, torch.device("cpu"), batch_size=256, warmup_batches=3, repeats=5)
    print({k: r[k] for k in ["device", "batch_size", "n_images", "total_s_median", "per_image_ms", "includes_preprocessing"]})
    assert KEYS <= set(r) and r["n_images"] == 600 and r["batch_size"] == 256 and len(r["repeats_s"]) == 5
    assert r["total_s_median"] > 0 and r["per_image_ms"] > 0
    assert r["per_image_ms"] == pytest.approx(r["total_s_median"] / 600 * 1000)       # units: s -> ms
    assert r["total_s_median"] == pytest.approx(float(np.median(r["repeats_s"])))
    assert r["includes_preprocessing"] is False and r["device"] == "cpu"
    assert next(m.parameters()).device.type == "cpu"


def test_cnn_benchmark_cuda_absent_raises_cleanly():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError):
        benchmark_cnn_inference(build_cnn(), torch.randn(4, 1, 28, 28), "cuda")


def test_classical_benchmark_both_modes():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (300, 28, 28), dtype=np.uint8); y = rng.integers(0, 10, 300)
    feat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=50))]).fit(feat(X), y)
    a = benchmark_classical_inference(pipe, feat(X), repeats=3)
    b = benchmark_classical_inference(pipe, X, repeats=3, feature_fn=feat)
    print(a["label"], a["includes_preprocessing"], "|", b["label"], b["includes_preprocessing"])
    for r in (a, b):
        assert KEYS <= set(r) and r["total_s_median"] > 0 and r["per_image_ms"] > 0 and len(r["repeats_s"]) == 3
        assert r["n_images"] == 300 and r["device"] == "cpu"
    assert a["includes_preprocessing"] is False and b["includes_preprocessing"] is True


def test_count_params_counts_only_trainable():
    m = build_cnn()
    for p in m.features[0].parameters():
        p.requires_grad_(False)
    assert count_trainable_params(m) == 241546 - 320
