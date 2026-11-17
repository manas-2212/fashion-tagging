"""Tests for src/dl_evaluation.py (owner: Rohan)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.dl_evaluation import (ablation_deltas, aggregate_seed_metrics, expected_calibration_error,
                               in_group_confusion_rate, per_class_recall_table, plot_ablation_deltas, plot_cascade,
                               plot_generalisation_gap, plot_learning_curves_compare, plot_per_class_recall,
                               plot_reliability, plot_roadmap, plot_test_summary, reliability_curve,
                               select_final_config, summarise_runs)


def _rec(name, seed, f1, acc=None, params=1000, diverged=False):
    acc = f1 if acc is None else acc
    return dict(name=name, seed=seed, val_macro_f1=f1, val_accuracy=acc, val_loss=0.3, train_accuracy_eval=acc + 0.02,
                generalisation_gap=0.02, val_per_class_recall=[0.9] * 10, n_params=params, best_epoch=5,
                epochs_run=8, train_seconds=10.0, diverged=diverged)


RECORDS = ([_rec("base", s, f) for s, f in ((42, 0.940), (43, 0.944), (44, 0.942))]
           + [_rec("aug_none", s, f) for s, f in ((42, 0.930), (43, 0.936), (44, 0.931))]
           + [_rec("opt_sgd", s, f) for s, f in ((42, 0.941), (43, 0.943), (44, 0.943))])


def test_summarise_runs_mean_and_sample_std():
    s = summarise_runs(RECORDS).set_index("name")
    assert s.loc["base", "n_seeds"] == 3
    assert s.loc["base", "val_macro_f1_mean"] == pytest.approx(0.942)
    assert s.loc["base", "val_macro_f1_std"] == pytest.approx(0.002)        # ddof = 1
    assert summarise_runs([_rec("x", 1, 0.9)])["val_macro_f1_std"].iloc[0] == 0.0
    with pytest.raises(ValueError):
        summarise_runs(RECORDS + [_rec("base", 42, 0.9)])                   # duplicate seed
    with pytest.raises(ValueError):
        summarise_runs([])


def test_ablation_deltas_are_paired_by_seed():
    d = ablation_deltas(RECORDS, "base").set_index("name")
    assert d.loc["aug_none", "delta_mean"] == pytest.approx(np.mean([-0.010, -0.008, -0.011]))
    assert d.loc["aug_none", "consistent_sign"] and d.loc["aug_none", "n_pairs"] == 3
    assert not d.loc["opt_sgd", "consistent_sign"]                          # +0.001, -0.001, +0.001
    assert "base" not in d.index
    partial = ablation_deltas(RECORDS[:3] + RECORDS[3:5], "base").set_index("name")
    assert partial.loc["aug_none", "n_pairs"] == 2                          # only shared seeds are compared
    with pytest.raises(ValueError):
        ablation_deltas(RECORDS, "missing")


def test_selection_rule():
    s = summarise_runs(RECORDS)
    assert select_final_config(s) == "opt_sgd"                              # 0.94233 > 0.942
    tie = summarise_runs([_rec("big", 1, 0.9, params=2000), _rec("small", 1, 0.9, params=1000)])
    assert select_final_config(tie) == "small"                              # tie -> fewer parameters
    bad = summarise_runs([_rec("div", 1, 0.99, diverged=True), _rec("ok", 1, 0.9)])
    assert select_final_config(bad) == "ok"                                 # diverged runs are not eligible
    short = summarise_runs(RECORDS + [_rec("one_seed", 42, 0.99)])
    assert select_final_config(short, required_seeds=3) == "opt_sgd"
    with pytest.raises(ValueError):
        select_final_config(summarise_runs([_rec("div", 1, 0.9, diverged=True)]))


def test_aggregate_seed_metrics():
    a = aggregate_seed_metrics({1: {"accuracy": 0.9, "macro_f1": 0.8}, 2: {"accuracy": 0.92, "macro_f1": 0.84}})
    assert a["accuracy_mean"] == pytest.approx(0.91) and a["macro_f1_mean"] == pytest.approx(0.82)
    assert a["accuracy_std"] == pytest.approx(np.std([0.9, 0.92], ddof=1)) and a["n_seeds"] == 2


def test_ece_of_calibrated_and_overconfident_predictions():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    y = np.zeros(n, dtype=int)
    pred_correct = rng.uniform(size=n) < conf
    probs = np.zeros((n, 10))
    probs[:, 0] = np.where(pred_correct, conf, 1 - conf)
    probs[:, 1] = 1 - probs[:, 0]
    assert expected_calibration_error(probs, y) < 0.02                      # confidence matches accuracy
    over = np.zeros((1000, 10))
    over[:, 0] = 0.99
    over[:, 1] = 0.01
    y_over = np.array([0] * 700 + [1] * 300)
    assert expected_calibration_error(over, y_over) == pytest.approx(0.29, abs=1e-6)
    curve = reliability_curve(over, y_over)
    assert len(curve) == 1 and curve["count"].iloc[0] == 1000 and curve["accuracy"].iloc[0] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        expected_calibration_error(over, y_over[:10])


def test_in_group_confusion_rate():
    cm = np.diag([100] * 10)
    assert in_group_confusion_rate(cm) == 0.0
    cm[6, 0] = 20        # Shirt -> T-shirt (inside the group)
    cm[6, 6] = 80
    cm[0, 9] = 10        # T-shirt -> Ankle boot (outside the group: not counted)
    cm[0, 0] = 90
    assert in_group_confusion_rate(cm) == pytest.approx(20 / 400)


def test_per_class_recall_table_shape():
    t = per_class_recall_table({"a": np.linspace(0.8, 1, 10), "b": np.ones(10)})
    assert t.shape == (10, 2) and t.index[6] == "Shirt"


def test_figures_are_created():
    s = summarise_runs(RECORDS)
    groups = {"aug_none": ("augmentation", "all augmentation", "aug='none'", ""),
              "opt_sgd": ("optimisation", "optimiser", "SGD", "")}
    cascade = pd.DataFrame({"tau": [0.0, 0.5, 0.9], "routed_fraction": [0.0, 0.1, 0.4],
                            "val_accuracy": [0.9, 0.91, 0.905], "val_macro_f1": [0.9, 0.91, 0.905]})
    hist = {"epoch": [1, 2], "train_loss": [1.0, 0.5], "val_loss": [0.9, 0.6], "val_acc": [0.7, 0.8]}
    over = np.zeros((100, 10))
    over[:, 0] = 0.9
    over[:, 1] = 0.1
    curve = reliability_curve(over, np.zeros(100, dtype=int))
    figs = [plot_roadmap(s, ["base", "aug_none"], {"base": "Base"}),
            plot_ablation_deltas(ablation_deltas(RECORDS, "base"), groups),
            plot_generalisation_gap(s, ["base", "aug_none"]),
            plot_learning_curves_compare({"a": hist, "b": hist}),
            plot_reliability({"m": curve}, {"m": 0.1}),
            plot_per_class_recall(per_class_recall_table({"a": np.linspace(0.8, 1, 10), "b": np.ones(10)})),
            plot_test_summary(pd.DataFrame({"label": ["x", "y"], "accuracy_mean": [0.9, 0.93],
                                            "accuracy_std": [0.0, 0.002], "kind": ["deep", "hybrid"]})),
            plot_cascade(cascade, 0.5)]
    assert all(len(f.axes) >= 1 for f in figs)
    import matplotlib.pyplot as plt
    plt.close("all")
