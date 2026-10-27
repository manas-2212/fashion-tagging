"""Tests for src/evaluation.py (python -m pytest tests/test_evaluation.py -q)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import src.evaluation as ev
from src.config import CLASS_NAMES


def make_preds(n=2000, acc=0.8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n).astype(np.int64)
    p = y.copy()
    flip = rng.random(n) > acc
    p[flip] = rng.integers(0, 10, flip.sum())
    return y, p


def test_evaluate_predictions_matches_sklearn():
    y, p = make_preds()
    r = ev.evaluate_predictions(y, p, "m")
    assert r["name"] == "m" and r["confusion"].dtype == np.int64 and r["confusion"].shape == (10, 10)
    assert np.array_equal(r["confusion"], confusion_matrix(y, p, labels=range(10)))
    assert r["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert r["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", labels=range(10), zero_division=0))
    assert r["per_class_recall"].shape == (10,)
    assert np.allclose(r["per_class_recall"], np.diag(r["confusion"]) / r["confusion"].sum(1))
    assert r["confusion"].sum() == len(y)


def test_macro_f1_counts_a_never_predicted_class_as_zero():
    y = np.repeat(np.arange(10), 5).astype(np.int64)
    p = y.copy(); p[p == 3] = 4                      # class 3 never predicted
    assert ev.evaluate_predictions(y, p, "x")["macro_f1"] == pytest.approx(
        f1_score(y, p, average="macro", labels=range(10), zero_division=0))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ev.evaluate_predictions([0, 1], [0], "x")
    with pytest.raises(ValueError):
        ev.evaluate_predictions([0, 10], [0, 1], "x")
    with pytest.raises(ValueError):
        ev.evaluate_predictions([], [], "x")


def test_top_confusions():
    cm = np.zeros((10, 10), dtype=np.int64)
    cm[np.arange(10), np.arange(10)] = 90
    cm[6, 0], cm[0, 6], cm[2, 4], cm[1, 3] = 9, 5, 7, 1
    df = ev.top_confusions(cm, k=3)
    assert list(df.columns) == ["true_class", "pred_class", "count", "rate_of_true_class"]
    assert df.iloc[0].tolist()[:3] == [CLASS_NAMES[6], CLASS_NAMES[0], 9]
    assert df["count"].tolist() == [9, 7, 5]
    assert df.iloc[0]["rate_of_true_class"] == pytest.approx(9 / (90 + 9))
    assert (df["true_class"] != df["pred_class"]).all()
    assert len(ev.top_confusions(cm, k=50)) == 4          # only non-zero off-diagonal cells


def test_bootstrap_ci_properties_and_reproducibility():
    y, p = make_preds(3000, 0.85)
    pt, lo, hi = ev.bootstrap_ci(y, p, "accuracy", n_boot=400, seed=1)
    assert pt == pytest.approx(accuracy_score(y, p)) and lo < pt < hi
    se = np.sqrt(pt * (1 - pt) / len(y))                   # binomial standard error
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.2)
    assert ev.bootstrap_ci(y, p, "accuracy", n_boot=400, seed=1) == (pt, lo, hi)       # same seed -> same CI
    pt2, lo2, hi2 = ev.bootstrap_ci(y, p, "macro_f1", n_boot=200, seed=1)
    assert pt2 == pytest.approx(f1_score(y, p, average="macro", labels=range(10), zero_division=0))
    assert lo2 < pt2 < hi2
    with pytest.raises(ValueError):
        ev.bootstrap_ci(y, p, "auc")


def test_mcnemar_exact_and_chi_square():
    # exact branch: 3 vs 9 discordant pairs
    y = np.zeros(100, dtype=np.int64)
    a = np.zeros(100, dtype=np.int64); b = np.zeros(100, dtype=np.int64)
    a[:3] = 1        # A wrong, B right  -> n01 = 3
    b[3:12] = 1      # B wrong, A right  -> n10 = 9
    r = ev.mcnemar_test(y, a, b)
    assert (r["n01"], r["n10"]) == (3, 9) and r["method"] == "exact binomial"
    assert r["p_value"] == pytest.approx(stats.binomtest(3, 12, 0.5).pvalue)
    # chi-square branch: 40 vs 80
    a = np.zeros(400, dtype=np.int64); b = np.zeros(400, dtype=np.int64); y = np.zeros(400, dtype=np.int64)
    a[:40] = 1; b[40:120] = 1
    r = ev.mcnemar_test(y, a, b)
    assert r["method"].startswith("chi-square")
    assert r["statistic"] == pytest.approx((abs(40 - 80) - 1) ** 2 / 120)
    assert r["p_value"] == pytest.approx(stats.chi2.sf(r["statistic"], 1)) and r["p_value"] < 0.001
    # identical models
    r0 = ev.mcnemar_test(y, a, a)
    assert (r0["n01"], r0["n10"], r0["p_value"]) == (0, 0, 1.0)
    # symmetry of the p-value
    assert ev.mcnemar_test(y, b, a)["p_value"] == pytest.approx(ev.mcnemar_test(y, a, b)["p_value"])


def test_predict_with_confidence():
    rng = np.random.default_rng(0)
    X = rng.random((120, 5)); y = rng.integers(0, 3, 120)
    lr = Pipeline([("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    pred, conf = ev.predict_with_confidence(lr, X)
    assert pred.dtype == np.int64 and pred.shape == (120,) and ((conf >= 1 / 3 - 1e-9) & (conf <= 1)).all()
    svm = Pipeline([("clf", SVC())]).fit(X, y)             # no predict_proba -> NaN confidence
    pred, conf = ev.predict_with_confidence(svm, X)
    assert pred.shape == (120,) and np.isnan(conf).all()


def test_compare_errors():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    a = y.copy(); a[[0, 1]] = 9          # wrong on 0,1
    b = y.copy(); b[[1, 2]] = 9          # wrong on 1,2
    df = ev.compare_errors(y, {"a": a, "b": b})
    assert df.set_index("model").loc["a", "n_errors"] == 2
    assert df.set_index("model").loc["a", "n_unique_errors"] == 1       # image 0 only
    assert (df["n_shared_by_all"] == 1).all()                            # image 1


def test_plots_return_figures():
    y, p = make_preds(500)
    r = ev.evaluate_predictions(y, p, "m")
    X = np.random.default_rng(0).integers(0, 256, (500, 28, 28), dtype=np.uint8)
    conf = np.random.default_rng(1).random(500)
    assert ev.plot_confusion_matrix(r["confusion"], "t").axes
    assert ev.plot_confusion_matrix(r["confusion"], "t", normalize=False)
    for mode in ("correct", "incorrect"):
        fig = ev.show_prediction_examples(X, y, p, conf, mode=mode, n=12, title=mode)
        assert sum(1 for ax in fig.axes if ax.images) == 12
        fig = ev.show_prediction_examples(X, y, p, np.full(500, np.nan), mode=mode, n=5)   # NaN confidence works
        assert sum(1 for ax in fig.axes if ax.images) == 5
    with pytest.raises(ValueError):
        ev.show_prediction_examples(X, y, p, conf, mode="both")
    df = pd.DataFrame({"model": ["a", "b"], "accuracy": [.8, .9], "macro_f1": [.79, .89],
                       "acc_lo": [.78, .88], "acc_hi": [.82, .92], "f1_lo": [.77, .87], "f1_hi": [.81, .91]})
    assert ev.plot_model_comparison(df).axes and ev.plot_model_comparison(df[["model", "accuracy", "macro_f1"]]).axes
    with pytest.raises(ValueError):
        ev.plot_model_comparison(df[["model", "accuracy"]])
    assert ev.plot_per_class_recall_heatmap({"m": r, "n": r["per_class_recall"]}).axes
    assert ev.plot_per_class_recall_heatmap(pd.DataFrame(np.random.rand(2, 10), index=["a", "b"])).axes


def test_show_incorrect_orders_by_confidence():
    y = np.array([0, 0, 0, 0], dtype=np.int64)
    p = np.array([1, 1, 1, 0], dtype=np.int64)
    X = np.zeros((4, 28, 28), dtype=np.uint8)
    fig = ev.show_prediction_examples(X, y, p, np.array([.5, .9, .7, 1.0]), mode="incorrect", n=2)
    titles = [ax.get_title() for ax in fig.axes if ax.images]
    assert "(0.90)" in titles[0] and "(0.70)" in titles[1]
