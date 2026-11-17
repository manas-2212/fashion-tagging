"""Tests for src/phase2_experiments.py and src/phase2_reporting.py (owner: Puneet)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.phase2_experiments import (FLAG_NAME, FREEZE_NAME, RunStore, freeze_phase2, load_gate_outputs,
                                    load_phase2_selection, run_id, run_phase2_test_gate)
from src.phase2_reporting import df_to_latex, df_to_markdown, flatten_records, format_cell, mean_std, save_table

DATA = {"y_tr": np.zeros(50, dtype=np.int64)}


class FakeConfig:
    """Stand-in for dl_train.TrainConfig: the store only needs `.name` and `.to_dict()`."""

    def __init__(self, name="base", max_epochs=30):
        self.name, self.max_epochs = name, max_epochs

    def to_dict(self):
        return {"name": self.name, "arch": "fashionnet", "arch_kwargs": {"dropout": 0.3}, "optimizer": "adamw",
                "lr": 2e-3, "weight_decay": 1e-2, "schedule": "cosine", "label_smoothing": 0.1,
                "aug": "crop_flip_erase", "max_epochs": self.max_epochs, "patience": 8}


def base_config():
    return FakeConfig()


def with_budget(cfg, max_epochs):
    return FakeConfig(cfg.name, max_epochs)


def _fake_run(calls):
    def run_fn(cfg, seed, data, device, checkpoint_path=None, verbose=True):
        calls.append((cfg.name, seed))
        checkpoint_path.write_bytes(b"weights")
        record = dict(name=cfg.name, seed=seed, config=cfg.to_dict(), n_train=len(data["y_tr"]), n_params=10,
                      val_accuracy=0.9, val_macro_f1=0.89, val_loss=0.3, train_accuracy_eval=0.95,
                      generalisation_gap=0.05, best_epoch=2, epochs_run=3, stopped_early=False, diverged=False,
                      train_seconds=1.0, device="cpu", amp=False, val_per_class_recall=[0.9] * 10,
                      history={"epoch": [1, 2, 3]})
        return None, record, np.full((6, 10), 0.1, dtype=np.float32)
    return run_fn


def test_run_store_resumes_and_detects_changed_config(tmp_path):
    store, calls = RunStore(tmp_path), []
    cfg = base_config()
    assert not store.is_done(cfg, 42, 50)
    rec = store.run_or_load(cfg, 42, DATA, "cpu", _fake_run(calls), verbose=False)
    assert store.is_done(cfg, 42, 50) and calls == [("base", 42)]
    again = store.run_or_load(cfg, 42, DATA, "cpu", _fake_run(calls), verbose=False)
    assert calls == [("base", 42)] and again == json.loads(json.dumps(rec))      # not re-trained
    assert store.load_val_probs("base", 42).shape == (6, 10)
    assert not store.is_done(with_budget(cfg, max_epochs=5), 42, 50)             # other budget -> not the same run
    assert not store.is_done(cfg, 42, 51)                                        # other data size
    assert not store.is_done(cfg, 43, 50)                                        # other seed
    store.run_or_load(with_budget(cfg, max_epochs=5), 42, DATA, "cpu", _fake_run(calls), verbose=False)
    assert len(calls) == 2
    store.run_or_load(cfg, 43, DATA, "cpu", _fake_run(calls), verbose=False)
    assert len(store.records()) == 2 and len(store.records(seeds={43})) == 1
    assert run_id("base", 42) == "base__seed42"


def test_incomplete_run_is_not_counted(tmp_path):
    store, calls = RunStore(tmp_path), []
    cfg = base_config()
    store.run_or_load(cfg, 42, DATA, "cpu", _fake_run(calls), verbose=False)
    store.checkpoint_path("base", 42).unlink()                                   # e.g. lost with a Colab runtime
    assert not store.is_done(cfg, 42, 50)


def _selection(models=("final_dl@42", "h0_hog_svm")):
    return {"final_config": "base", "gate_models": list(models), "hybrid": {"tau": 0.9}}


def _predict_all(names, n_classes=10):
    def predict_all(X):
        n = len(X)
        out = {name: {"y_pred": np.zeros(n, dtype=np.int64), "probs": None} for name in names}
        first = list(names)[0]
        out[first]["probs"] = np.full((n, n_classes), 1.0 / n_classes, dtype=np.float32)
        return out
    return predict_all


def _load_test():
    X = np.zeros((100, 28, 28), dtype=np.uint8)
    return X, np.tile(np.arange(10), 10).astype(np.int64)


def test_freeze_requires_gate_models(tmp_path):
    with pytest.raises(ValueError):
        freeze_phase2({"final_config": "base"}, tmp_path)
    with pytest.raises(ValueError):
        freeze_phase2(_selection(("a", "a")), tmp_path)
    sha = freeze_phase2(_selection(), tmp_path)
    assert len(sha) == 64 and (tmp_path / FREEZE_NAME).is_file()
    assert load_phase2_selection(tmp_path)["gate_models"] == ["final_dl@42", "h0_hog_svm"]


def test_gate_refuses_without_freeze_file(tmp_path):
    with pytest.raises(RuntimeError, match="freeze file"):
        run_phase2_test_gate(_predict_all(["final_dl@42"]), _load_test, tmp_path)
    assert not (tmp_path / FLAG_NAME).exists()


def test_gate_refuses_model_list_mismatch_without_burning_the_flag(tmp_path):
    freeze_phase2(_selection(), tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        run_phase2_test_gate(_predict_all(["final_dl@42"]), _load_test, tmp_path)                 # one missing
    with pytest.raises(RuntimeError, match="differs"):
        run_phase2_test_gate(_predict_all(["final_dl@42", "h0_hog_svm", "extra"]), _load_test, tmp_path)
    assert not (tmp_path / FLAG_NAME).exists()


def test_gate_runs_once_and_saves_outputs(tmp_path):
    freeze_phase2(_selection(), tmp_path)
    loads = []

    def load():
        loads.append(1)
        return _load_test()

    out = run_phase2_test_gate(_predict_all(["final_dl@42", "h0_hog_svm"]), load, tmp_path)
    assert len(loads) == 1 and out["results"]["final_dl@42"]["probs"].shape == (100, 10)
    flag = json.loads((tmp_path / FLAG_NAME).read_text())
    assert flag["n_test"] == 100 and flag["full_test_set"] and flag["freeze_sha256"] == out["freeze_sha256"]
    with pytest.raises(RuntimeError, match="already evaluated"):
        run_phase2_test_gate(_predict_all(["final_dl@42", "h0_hog_svm"]), load, tmp_path)
    assert len(loads) == 1                                           # the second attempt never loaded test data
    saved = load_gate_outputs(tmp_path)
    assert np.array_equal(saved["y_test"], out["y_test"])
    assert saved["results"]["h0_hog_svm"]["probs"] is None and saved["results"]["h0_hog_svm"]["y_pred"].shape == (100,)


def test_gate_validates_prediction_shapes(tmp_path):
    freeze_phase2(_selection(("m",)), tmp_path)
    bad = lambda X: {"m": {"y_pred": np.zeros(len(X) - 1, dtype=np.int64), "probs": None}}
    with pytest.raises(ValueError):
        run_phase2_test_gate(bad, _load_test, tmp_path)
    assert not (tmp_path / FLAG_NAME).exists()


def test_gate_subset_is_stratified_and_flagged(tmp_path):
    freeze_phase2(_selection(("m",)), tmp_path)
    ok = lambda X: {"m": {"y_pred": np.zeros(len(X), dtype=np.int64), "probs": None}}
    out = run_phase2_test_gate(ok, _load_test, tmp_path, test_subset_n=50)
    assert len(out["y_test"]) == 50 and (np.bincount(out["y_test"]) == 5).all()
    assert not json.loads((tmp_path / FLAG_NAME).read_text())["full_test_set"]


def test_load_gate_outputs_needs_the_flag(tmp_path):
    with pytest.raises(RuntimeError):
        load_gate_outputs(tmp_path)


def test_reporting_helpers(tmp_path):
    assert format_cell(0.123456) == "0.1235" and format_cell(307690) == "307,690" and format_cell(float("nan")) == "--"
    assert format_cell(True) == "yes" and mean_std(0.94123, 0.00123) == "0.9412 +- 0.0012"
    df = pd.DataFrame({"model_name": ["a_b", "c"], "acc": [mean_std(0.9, 0.01), mean_std(0.8, 0.0)], "n": [10, 2000]})
    tex = df_to_latex(df, "Caption with 95% CI", "tab:x")
    assert "model\\_name" in tex and "a\\_b" in tex and "$\\pm$" in tex and "95\\% CI" in tex and "2,000" in tex
    assert tex.count("\\\\") == 3 and "\\begin{table}" in tex
    assert "\\begin{table*}" in df_to_latex(df, "c", "tab:x", wide=True)
    md = df_to_markdown(df)
    assert md.splitlines()[1].count("---") == 3 and len(md.splitlines()) == 4
    paths = save_table(df, "t", tmp_path, tmp_path / "tex", caption="c", label="tab:t")
    assert all(p.is_file() for p in paths.values()) and set(paths) == {"csv", "md", "tex"}
    assert pd.read_csv(paths["csv"]).shape == (2, 3)


def test_flatten_records_has_one_scalar_row_per_run(tmp_path):
    store, calls = RunStore(tmp_path), []
    for seed in (42, 43):
        store.run_or_load(base_config(), seed, DATA, "cpu", _fake_run(calls), verbose=False)
    flat = flatten_records(store.records())
    assert len(flat) == 2 and {"name", "seed", "val_macro_f1", "aug", "label_smoothing"} <= set(flat.columns)
    assert not any(isinstance(v, (dict, list)) for v in flat.iloc[0])
