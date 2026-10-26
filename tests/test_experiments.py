"""Tests for src/experiments.py on tiny mock data (600 train / 200 val / 200 test)."""
import hashlib
import importlib
import inspect
import itertools
import json
import re
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import HOG_DIM, MAX_FIT_MINUTES, PIXEL_DIM, SEED


# =========================== STUB BLOCK START (delete at integration) ===========================
# Used ONLY for modules that do not exist yet (features, classical_models, evaluation).
# A real module that exists is always used instead of its stub.
def _subset(X, y, n, seed):
    if n is None or n >= len(y):
        return X, y
    idx = np.random.default_rng(seed).permutation(len(y))[:n]  # STUB: not stratified
    return X[idx], y[idx]


def _stub_features():
    mod = types.ModuleType("src.features")

    def cached_features(X, kind, split_name, cache_dir=None):
        flat = X.reshape(len(X), -1).astype(np.float32) / 255.0
        if kind == "pixels":
            return flat
        if kind == "hog":  # fixed random projection standing in for HOG, same dim as HOG_DIM
            proj = np.random.default_rng(1).standard_normal((PIXEL_DIM, HOG_DIM)).astype(np.float32)
            return (flat @ proj).astype(np.float32)
        raise ValueError(kind)

    mod.cached_features = cached_features
    return mod


def _stub_classical_models():
    mod = types.ModuleType("src.classical_models")
    mod.DEFAULT_GRIDS = {"logreg": {"C": [0.1, 1.0]}, "knn": {"n_neighbors": [3, 5, 7, 11]},
                         "rf": {"min_samples_leaf": [1, 3]}, "svm": {"C": [1, 10]}}

    def build_classical_pipeline(model_name, params=None):
        params = dict(params or {})
        if model_name == "logreg":
            steps = [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, **params))]
        elif model_name == "knn":
            steps = [("clf", KNeighborsClassifier(**params))]
        elif model_name == "rf":
            steps = [("clf", RandomForestClassifier(n_estimators=10, random_state=SEED, **params))]
        elif model_name == "svm":
            steps = [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", **params))]
        else:
            raise ValueError(model_name)
        return Pipeline(steps)

    def estimate_fit_seconds(model_name, params, X_tr, y_tr, probe_n=5000):
        return 1.0

    def tune_classical(model_name, X_tr, y_tr, X_val, y_val, param_grid, subset_n=None, seed=SEED):
        X_fit, y_fit = _subset(X_tr, y_tr, subset_n, seed)
        names = list(param_grid)
        rows, combos = [], []
        for values in itertools.product(*(param_grid[n] for n in names)):
            params = dict(zip(names, values))
            pred = build_classical_pipeline(model_name, params).fit(X_fit, y_fit).predict(X_val)
            rows.append({**params, "val_macro_f1": f1_score(y_val, pred, average="macro"),
                         "val_accuracy": accuracy_score(y_val, pred)})
            combos.append(params)
        df = pd.DataFrame(rows)
        best = df.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).index[0]
        return combos[best], df

    def fit_classical(model_name, params, X_tr, y_tr, subset_n=None, seed=SEED):
        X_fit, y_fit = _subset(X_tr, y_tr, subset_n, seed)
        start = time.perf_counter()
        pipe = build_classical_pipeline(model_name, params).fit(X_fit, y_fit)
        return pipe, {"fit_seconds": time.perf_counter() - start,
                      "n_train_used": len(y_fit), "params": params}

    mod.build_classical_pipeline = build_classical_pipeline
    mod.estimate_fit_seconds = estimate_fit_seconds
    mod.tune_classical = tune_classical
    mod.fit_classical = fit_classical
    return mod


def _stub_evaluation():
    mod = types.ModuleType("src.evaluation")

    def predict_with_confidence(pipeline, X):
        y_pred = pipeline.predict(X)
        if hasattr(pipeline, "predict_proba"):
            conf = pipeline.predict_proba(X).max(axis=1).astype(float)
        else:  # e.g. SVC without probability=True
            conf = np.full(len(X), np.nan)
        return y_pred, conf

    mod.predict_with_confidence = predict_with_confidence
    return mod


_STUBS = {"src.features": _stub_features, "src.classical_models": _stub_classical_models,
          "src.evaluation": _stub_evaluation}


def install_stubs():
    """Register a stub for every module that is genuinely missing; return the stubbed names."""
    import src
    stubbed = []
    for name, factory in _STUBS.items():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as err:
            if err.name != name:  # the module exists but one of ITS imports is missing: surface it
                raise
            mod = factory()
            sys.modules[name] = mod
            setattr(src, name.split(".")[1], mod)
            stubbed.append(name)
    return stubbed


STUBBED = install_stubs()
print(f"STUBBED MODULES: {STUBBED or 'none (all real modules found)'}")
# =========================== STUB BLOCK END ===========================

from src import experiments as exp  # noqa: E402  (must come after install_stubs)

GRIDS_1 = {"logreg": {"C": [1.0]}, "knn": {"n_neighbors": [3]},
           "rf": {"min_samples_leaf": [1]}, "svm": {"C": [1]}}  # 1-value grids keep tests fast


def make_mock(n_train=600, n_val=200, n_test=200):
    rng = np.random.default_rng(0)

    def make(n):
        return rng.integers(0, 256, (n, 28, 28), dtype=np.uint8), rng.integers(0, 10, n).astype(np.int64)

    X_train, y_train = make(n_train)
    X_val, y_val = make(n_val)
    X_test, y_test = make(n_test)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


def fake_cnn_predict(X_uint8):
    """Stands in for the closure Manas builds: returns (y_pred int64, probs float32 (N, 10))."""
    probs = np.random.default_rng(5).dirichlet(np.ones(10), size=len(X_uint8)).astype(np.float32)
    return probs.argmax(axis=1).astype(np.int64), probs


def build_state(mock, out):
    """features -> tuning (1-value grids) -> final fits, all written under `out`."""
    feats = exp.build_feature_sets(mock["X_train"], mock["X_val"], cache_dir=out / "cache")
    best = exp.run_classical_tuning(feats, mock["y_train"], mock["y_val"], grids=GRIDS_1,
                                    results_dir=out / "results")
    models, infos = exp.run_classical_final_fits(feats, mock["y_train"], best, results_dir=out / "results")
    return {"out": out, "feats": feats, "best": best, "models": models, "infos": infos}


def patch_test_cache(monkeypatch, cache_dir):
    """Keep mock test features out of the real data/cache directory."""
    real = exp.build_test_features
    monkeypatch.setattr(exp, "build_test_features", lambda X_test, **_: real(X_test, cache_dir=cache_dir))


def freeze_for(tmp_path, state):
    path = tmp_path / "selected_configs.json"
    exp.freeze_selection({"classical": state["best"], "cnn": {"best_epoch": np.int64(3)}}, path=path)
    return path


@pytest.fixture(scope="module")
def mock():
    return make_mock()


@pytest.fixture(scope="module")
def state(mock, tmp_path_factory):
    return build_state(mock, tmp_path_factory.mktemp("exp"))


# ----------------------------------------------------------------------------- tests
def test_key_roundtrip():
    assert exp.key_to_str(("logreg", "pixels")) == "logreg|pixels"
    assert exp.str_to_key("logreg|pixels") == ("logreg", "pixels")
    assert exp.key_to_str("cnn") == "cnn" and exp.str_to_key("cnn") == "cnn"


def test_build_feature_sets_has_no_test_inputs_or_outputs(state):
    assert list(inspect.signature(exp.build_feature_sets).parameters) == ["X_train", "X_val", "cache_dir"]
    feats = state["feats"]
    assert set(feats) == {"pixels", "hog"}
    for rep in feats:
        assert set(feats[rep]) == {"train", "val"}
    assert feats["pixels"]["train"].shape == (600, PIXEL_DIM)
    assert feats["hog"]["val"].shape == (200, HOG_DIM)


def test_feature_sets_only_request_train_and_val_splits(mock, tmp_path, monkeypatch):
    seen, real = [], exp.cached_features

    def spy(X, kind, split_name, cache_dir=None):
        seen.append(split_name)
        return real(X, kind, split_name, cache_dir)

    monkeypatch.setattr(exp, "cached_features", spy)
    exp.build_feature_sets(mock["X_train"], mock["X_val"], cache_dir=tmp_path)
    assert set(seen) == {"train", "val"}


def test_tuning_saves_csvs_with_expected_names(state):
    assert len(state["best"]) == 8
    for (model, rep), params in state["best"].items():
        csv = state["out"] / "results" / f"tuning_{model}_{rep}.csv"
        assert csv.is_file() and len(pd.read_csv(csv)) == 1
        assert params == {k: v[0] for k, v in GRIDS_1[model].items()}


def test_fit_info_json_written_and_serialisable(state):
    records = json.loads((state["out"] / "results" / "fit_info.json").read_text())
    assert len(records) == 8
    for rec in records:
        assert {"model", "rep", "params", "fit_seconds", "n_train_used", "gated"} <= set(rec)
        assert rec["gated"] is False and rec["n_train_used"] == 600
    assert set(state["infos"]) == set(exp.CLASSICAL_KEYS)


def test_runtime_gate_subsets_and_prints_when_estimate_too_large(mock, state, tmp_path, monkeypatch, capsys):
    limit = MAX_FIT_MINUTES * 60
    est = limit * 10.0
    monkeypatch.setattr(exp, "estimate_fit_seconds", lambda *a, **k: est)
    _, infos = exp.run_classical_final_fits(state["feats"], mock["y_train"], state["best"],
                                            results_dir=tmp_path)
    printed = capsys.readouterr().out
    expected = min(600, max(exp.MIN_GATED_SUBSET, int(600 * limit / est)))
    assert expected < 600
    # count only experiments.py lines "[model|rep] n_train_used = ..." (the real fit_classical prints its own too)
    assert len(re.findall(r"^\[\w+\|\w+\] n_train_used = ", printed, flags=re.M)) == 8
    assert f"n_train_used = {expected} of 600" in printed
    for info in infos.values():
        assert info["gated"] is True and info["n_train_used"] == expected
    assert all(r["gated"] for r in json.loads((tmp_path / "fit_info.json").read_text()))


def test_runtime_gate_not_triggered_prints_full_size(mock, state, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exp, "estimate_fit_seconds", lambda *a, **k: 1.0)
    exp.run_classical_final_fits(state["feats"], mock["y_train"], state["best"], results_dir=tmp_path)
    assert "n_train_used = 600 of 600" in capsys.readouterr().out


def test_silent_subsetting_is_rejected(mock, state, tmp_path, monkeypatch):
    real = exp.fit_classical

    def lying_fit(*args, **kwargs):
        pipe, info = real(*args, **kwargs)
        return pipe, dict(info, n_train_used=100)  # claims 100 rows although the gate did not trigger

    monkeypatch.setattr(exp, "estimate_fit_seconds", lambda *a, **k: 1.0)
    monkeypatch.setattr(exp, "fit_classical", lying_fit)
    with pytest.raises(RuntimeError, match="without the runtime gate"):
        exp.run_classical_final_fits(state["feats"], mock["y_train"], state["best"], results_dir=tmp_path)


def test_freeze_selection_writes_json_and_returns_sha256(tmp_path):
    selected = {"classical": {("logreg", "pixels"): {"C": np.float64(1.0)}},
                "cnn": {"best_epoch": np.int64(7), "best_val_macro_f1": 0.5}}
    path = tmp_path / "sub" / "selected_configs.json"  # directory is created if needed
    sha = exp.freeze_selection(selected, path=path)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    assert data["selected"]["classical"]["logreg|pixels"]["C"] == 1.0
    assert data["selected"]["cnn"]["best_epoch"] == 7
    assert "created_utc" in data and isinstance(data["git_commit"], str)
    assert ("logreg", "pixels") in exp.load_selection(path)["selected"]["classical"]


def test_git_commit_falls_back_to_unavailable(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(exp.subprocess, "run", boom)
    assert exp._git_commit() == "unavailable"


def test_test_features_are_built_only_inside_the_gate(mock, state, tmp_path, monkeypatch):
    calls, real = [], exp.build_test_features

    def spy(X_test, **_):
        calls.append(len(X_test))
        return real(X_test, cache_dir=tmp_path / "cache")

    monkeypatch.setattr(exp, "build_test_features", spy)
    frozen = freeze_for(tmp_path, state)
    assert calls == []
    exp.run_test_gate(frozen, state["models"], mock["X_test"], mock["y_test"],
                      cnn_predict_fn=fake_cnn_predict, flag_path=tmp_path / "flag")
    assert calls == [200]


def test_gate_validation_failures_do_not_burn_the_flag(mock, state, tmp_path, monkeypatch):
    patch_test_cache(monkeypatch, tmp_path / "cache")
    frozen, flag = freeze_for(tmp_path, state), tmp_path / "flag"
    with pytest.raises(ValueError):  # wrong number of test labels
        exp.run_test_gate(frozen, state["models"], mock["X_test"], mock["y_test"][:-1], flag_path=flag)
    partial = {k: v for k, v in state["models"].items() if k != ("svm", "hog")}
    with pytest.raises(RuntimeError, match="missing models"):
        exp.run_test_gate(frozen, partial, mock["X_test"], mock["y_test"], flag_path=flag)
    assert not flag.exists()
