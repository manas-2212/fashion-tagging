"""Leakage checks and the end-to-end mock run: features -> tune -> final fit -> freeze -> test gate once."""
import numpy as np
import pytest

from src.config import HOG_DIM, PIXEL_DIM
from src.data import prepare_splits
from test_experiments import (build_state, exp, fake_cnn_predict, freeze_for,  # also installs the stubs
                              make_mock, patch_test_cache)


@pytest.fixture(scope="module")
def mock():
    return make_mock()


@pytest.fixture(scope="module")
def state(mock, tmp_path_factory):
    return build_state(mock, tmp_path_factory.mktemp("leak"))


def test_idx_train_and_idx_val_are_disjoint_and_dims_match(mock, tmp_path):
    X_full = np.concatenate([mock["X_train"], mock["X_val"]])  # 800 mock "official train" images
    y_full = np.concatenate([mock["y_train"], mock["y_val"]])
    splits = prepare_splits(X_full, y_full)
    idx_tr, idx_va = np.asarray(splits["idx_train"]), np.asarray(splits["idx_val"])
    assert set(idx_tr).isdisjoint(set(idx_va))
    assert sorted(np.concatenate([idx_tr, idx_va])) == list(range(800))
    assert abs(len(idx_va) - 80) <= 1
    feats = exp.build_feature_sets(splits["X_train"], splits["X_val"], cache_dir=tmp_path)
    for rep, dim in (("pixels", PIXEL_DIM), ("hog", HOG_DIM)):
        assert feats[rep]["train"].shape[1] == feats[rep]["val"].shape[1] == dim


def test_scaler_mean_is_the_train_mean_not_train_plus_val(state):
    checked = 0
    for (model, rep), pipe in state["models"].items():
        if "scaler" not in pipe.named_steps:
            continue
        tr = state["feats"][rep]["train"].astype(np.float64)
        va = state["feats"][rep]["val"].astype(np.float64)
        mean = pipe.named_steps["scaler"].mean_
        assert np.allclose(mean, tr.mean(axis=0), rtol=1e-4, atol=1e-5), (model, rep)
        assert not np.allclose(mean, np.concatenate([tr, va]).mean(axis=0), rtol=1e-4, atol=1e-5), (model, rep)
        checked += 1
    assert checked >= 4  # logreg and svm on both representations


def test_gate_refuses_without_freeze_file(mock, state, tmp_path):
    flag = tmp_path / "flag"
    with pytest.raises(RuntimeError, match="freeze"):
        exp.run_test_gate(tmp_path / "missing.json", state["models"], mock["X_test"], mock["y_test"],
                          cnn_predict_fn=fake_cnn_predict, flag_path=flag)
    assert not flag.exists()


def test_end_to_end_gate_runs_exactly_once(mock, state, tmp_path, monkeypatch):
    patch_test_cache(monkeypatch, tmp_path / "cache")
    frozen, flag = freeze_for(tmp_path, state), tmp_path / "test_gate_done.flag"
    results = exp.run_test_gate(frozen, state["models"], mock["X_test"], mock["y_test"],
                                cnn_predict_fn=fake_cnn_predict, flag_path=flag)
    assert set(results) == set(exp.CLASSICAL_KEYS) | {"cnn"}  # 9 models
    for key, res in results.items():
        assert res["y_pred"].shape == (200,) and res["conf"].shape == (200,), key
        assert np.issubdtype(res["y_pred"].dtype, np.integer), key
    assert flag.is_file()
    with pytest.raises(RuntimeError, match="already"):  # second call must be refused
        exp.run_test_gate(frozen, state["models"], mock["X_test"], mock["y_test"],
                          cnn_predict_fn=fake_cnn_predict, flag_path=flag)
    results_dir = state["out"] / "results"  # tuning/fit files from the same mock run
    assert (results_dir / "fit_info.json").is_file()
    assert len(list(results_dir.glob("tuning_*_*.csv"))) == 8
