import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from src import classical_models as cm

rng = np.random.default_rng(0)
X784 = rng.random((600, 784), dtype=np.float32)
X1296 = rng.random((600, 1296), dtype=np.float32)
Y = rng.integers(0, 10, 600).astype(np.int64)
MODELS = ["logreg", "knn", "rf", "svm"]
DATA = {784: X784, 1296: X1296}


@pytest.mark.parametrize("dim", [784, 1296])
@pytest.mark.parametrize("name", MODELS)
def test_build_fit_predict(name, dim):
    X = DATA[dim]
    pipe = cm.build_classical_pipeline(name)
    pipe.fit(X[:500], Y[:500])
    pred = pipe.predict(X[500:])
    assert pred.shape == (100,)
    assert np.issubdtype(pred.dtype, np.integer)


@pytest.mark.parametrize("name", MODELS)
def test_step_names(name):
    names = [s for s, _ in cm.build_classical_pipeline(name).steps]
    assert names == (["scaler", "clf"] if name in ("logreg", "svm") else ["clf"])


def test_params_override_without_duplicate_keyword():
    clf = cm.build_classical_pipeline("logreg", {"C": 1.0, "max_iter": 50}).named_steps["clf"]
    assert clf.C == 1.0 and clf.max_iter == 50
    assert cm.build_classical_pipeline("logreg").named_steps["clf"].max_iter == 500


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        cm.build_classical_pipeline("xgboost")


@pytest.mark.parametrize("name", ["logreg", "svm"])
def test_scaler_leakage(name):
    X_tr, X_val = X784[:500], X784[500:]
    pipe = cm.build_classical_pipeline(name).fit(X_tr, Y[:500])
    mean_ = pipe.named_steps["scaler"].mean_
    assert np.allclose(mean_, X_tr.mean(axis=0), atol=1e-5)
    assert not np.allclose(mean_, np.vstack([X_tr, X_val]).mean(axis=0), atol=1e-4)


def test_selection_logic_hand_checkable():
    assert cm._select_best_index([(0.5, 0.6), (0.7, 0.7), (0.7, 0.8)]) == 2  # F1 tie -> accuracy wins
    assert cm._select_best_index([(0.7, 0.8), (0.7, 0.8)]) == 0              # full tie -> first
    assert cm._select_best_index([(0.4, 0.9), (0.6, 0.1)]) == 1              # F1 beats accuracy


def test_tune_classical():
    grid = {"n_neighbors": [3, 5]}
    grid_before = copy.deepcopy(grid)
    X_copy = X784.copy()
    best, df = cm.tune_classical("knn", X784[:500], Y[:500], X784[500:], Y[500:], grid)
    assert best in list(ParameterGrid(grid))
    assert grid == grid_before
    assert np.array_equal(X784, X_copy)
    assert list(df.columns) == ["model", "params", "val_accuracy", "val_macro_f1", "fit_seconds", "n_train_used"]
    assert isinstance(df, pd.DataFrame) and len(df) == 2


def test_subset_path():
    pipe, info = cm.fit_classical("knn", {"n_neighbors": 3}, X784, Y, subset_n=200)
    assert info["n_train_used"] == 200
    assert len(pipe.named_steps["clf"].classes_) == 10
    _, info_full = cm.fit_classical("knn", {}, X784, Y)
    assert info_full["n_train_used"] == 600


def test_stratified_subset_proportions():
    _, y_s = cm._subset(X784, Y, 100)
    expected = 100 * np.bincount(Y, minlength=10) / len(Y)
    assert len(y_s) == 100
    assert np.all(np.abs(np.bincount(y_s, minlength=10) - expected) <= 1.0)


@pytest.mark.parametrize("dim,expected", [(784, 7850), (1296, 12970)])
def test_logreg_param_count(dim, expected):
    pipe = cm.build_classical_pipeline("logreg").fit(DATA[dim][:300], Y[:300])
    count, _ = cm.classical_param_count(pipe)
    assert count == expected


def test_other_param_counts():
    for name in ("rf", "svm"):
        pipe = cm.build_classical_pipeline(name).fit(X784[:300], Y[:300])
        count, note = cm.classical_param_count(pipe)
        assert isinstance(count, int) and count > 0 and "not comparable" in note
    pipe = cm.build_classical_pipeline("knn").fit(X784[:300], Y[:300])
    assert cm.classical_param_count(pipe)[0] is None


@pytest.mark.parametrize("name", ["rf", "svm"])
def test_same_seed_same_predictions(name):
    p1 = cm.build_classical_pipeline(name).fit(X784[:500], Y[:500]).predict(X784[500:])
    p2 = cm.build_classical_pipeline(name).fit(X784[:500], Y[:500]).predict(X784[500:])
    assert np.array_equal(p1, p2)


def test_overfit_check():
    pipe = cm.build_classical_pipeline("rf").fit(X784[:500], Y[:500])
    res = cm.overfit_check(pipe, X784[:500], Y[:500], X784[500:], Y[500:], n_eval=100)
    assert set(res) == {"train_acc", "val_acc", "gap", "n_eval"}
    assert res["n_eval"] == 100
    assert res["gap"] == pytest.approx(res["train_acc"] - res["val_acc"])


@pytest.mark.parametrize("name", MODELS)
def test_estimate_fit_seconds(name):
    t = cm.estimate_fit_seconds(name, {}, X784, Y, probe_n=200)
    assert isinstance(t, float) and t > 0
