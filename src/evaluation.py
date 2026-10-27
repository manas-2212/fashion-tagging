"""Evaluation helpers: predictions, metrics, confusion analysis, plots, bootstrap CI, McNemar (owner: Rohan).

Conventions (project-wide):
  * labels are int64 class indices 0-9, names in config.CLASS_NAMES
  * confusion matrix: int64 (10, 10), rows = TRUE class, columns = PREDICTED class, labels=range(10)
  * nothing here fits or tunes anything; these functions only describe predictions that already exist
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, NUM_CLASSES, SEED

LABELS = list(range(NUM_CLASSES))


# --------------------------------------------------------------------------- predictions
def predict_with_confidence(pipeline, X):
    """Return (y_pred int64 (N,), conf float (N,)).

    conf = max predicted class probability when the model offers predict_proba, else NaN
    (e.g. the RBF-SVM is fitted without probability=True, so its confidence is NaN by design).
    """
    y_pred = np.asarray(pipeline.predict(X)).astype(np.int64)
    if hasattr(pipeline, "predict_proba"):  # a Pipeline exposes it only if its last step does
        conf = np.asarray(pipeline.predict_proba(X), dtype=float).max(axis=1)
    else:
        conf = np.full(len(y_pred), np.nan)
    return y_pred, conf


# --------------------------------------------------------------------------- metrics
def _check_labels(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must both be 1-D and equal length")
    if len(y_true) == 0:
        raise ValueError("empty prediction arrays")
    for name, a in (("y_true", y_true), ("y_pred", y_pred)):
        if a.min() < 0 or a.max() >= NUM_CLASSES:
            raise ValueError(f"{name} contains labels outside 0..{NUM_CLASSES - 1}")
    return y_true.astype(np.int64), y_pred.astype(np.int64)


def _confusion(y_true, y_pred):
    """int64 (10,10) confusion matrix via bincount (rows true, cols predicted)."""
    return np.bincount(y_true * NUM_CLASSES + y_pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).astype(np.int64)


def _macro_f1_from_cm(cm):
    """Macro-F1 over all 10 classes; a class with no true and no predicted samples scores 0 (sklearn zero_division=0)."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=1) + cm.sum(axis=0)  # F1 = 2TP / (2TP + FP + FN) = 2TP / (row sum + column sum)
    f1 = np.divide(2 * tp, denom, out=np.zeros(NUM_CLASSES), where=denom > 0)
    return float(f1.mean())


def evaluate_predictions(y_true, y_pred, name):
    """-> dict(name, accuracy, macro_f1, per_class_recall (10,), confusion (10,10) int64)."""
    y_true, y_pred = _check_labels(y_true, y_pred)
    cm = _confusion(y_true, y_pred)
    support = cm.sum(axis=1).astype(float)
    recall = np.divide(np.diag(cm), support, out=np.full(NUM_CLASSES, np.nan), where=support > 0)  # NaN if class absent
    return {"name": name, "accuracy": float((y_true == y_pred).mean()), "macro_f1": _macro_f1_from_cm(cm),
            "per_class_recall": recall, "confusion": cm}


def top_confusions(cm, k=10):
    """The k most frequent off-diagonal cells: DataFrame[true_class, pred_class, count, rate_of_true_class]."""
    cm = np.asarray(cm)
    if cm.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"confusion matrix must be {NUM_CLASSES}x{NUM_CLASSES}, got {cm.shape}")
    rows = []
    for t in range(NUM_CLASSES):
        total = cm[t].sum()
        for p in range(NUM_CLASSES):
            if t != p and cm[t, p] > 0:
                rows.append((CLASS_NAMES[t], CLASS_NAMES[p], int(cm[t, p]), float(cm[t, p] / total)))
    df = pd.DataFrame(rows, columns=["true_class", "pred_class", "count", "rate_of_true_class"])
    return df.sort_values(["count", "true_class", "pred_class"], ascending=[False, True, True],
                          kind="stable").head(k).reset_index(drop=True)


def compare_errors(y_true, preds):
    """Error overlap between models. preds: dict name -> y_pred.

    Columns: model, n_errors, error_rate, n_unique_errors (only this model is wrong),
    n_shared_by_all (images EVERY model gets wrong; identical for all rows).
    """
    y_true = np.asarray(y_true)
    wrong = {}
    for name, p in preds.items():
        _, p = _check_labels(y_true, p)
        wrong[name] = p != y_true
    if not wrong:
        raise ValueError("preds is empty")
    stack = np.stack(list(wrong.values()))
    all_wrong = int(stack.all(axis=0).sum())
    n_wrong = stack.sum(axis=0)
    rows = [{"model": name, "n_errors": int(w.sum()), "error_rate": float(w.mean()),
             "n_unique_errors": int((w & (n_wrong == 1)).sum()), "n_shared_by_all": all_wrong}
            for name, w in wrong.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def plot_confusion_matrix(cm, title, normalize=True, ax=None):
    """Heat map of a confusion matrix; normalize=True divides each row by its true-class total."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    shown = cm / np.where(cm.sum(axis=1, keepdims=True) > 0, cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    thresh = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if shown[i, j] > thresh else "black")
    ax.figure.colorbar(im, ax=ax, label="Row-normalised rate" if normalize else "Count")
    ax.figure.tight_layout()
    return ax.figure


def show_prediction_examples(X_img, y_true, y_pred, conf, mode, n=12, title=""):
    """Grid of example images. mode 'correct': the first n correct predictions in dataset order (not cherry-picked).
    mode 'incorrect': the n MOST CONFIDENT mistakes (NaN confidence sorts last, then dataset order)."""
    import matplotlib.pyplot as plt

    if mode not in ("correct", "incorrect"):
        raise ValueError("mode must be 'correct' or 'incorrect'")
    y_true, y_pred = _check_labels(y_true, y_pred)
    conf = np.asarray(conf, dtype=float)
    hit = np.flatnonzero(y_true == y_pred) if mode == "correct" else np.flatnonzero(y_true != y_pred)
    if mode == "incorrect":
        order = np.lexsort((hit, -np.nan_to_num(conf[hit], nan=-1.0)))  # high confidence first, ties by index
        hit = hit[order]
    hit = hit[:n]
    cols = 6
    rows = max(1, int(np.ceil(max(len(hit), 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), hit):
        ax.imshow(X_img[i], cmap="gray")
        c = "" if np.isnan(conf[i]) else f" ({conf[i]:.2f})"
        ax.set_title(f"T: {CLASS_NAMES[y_true[i]]}\nP: {CLASS_NAMES[y_pred[i]]}{c}", fontsize=6.5)
    if len(hit) == 0:
        axes[0, 0].text(0.5, 0.5, f"no {mode} examples", ha="center", va="center")
    fig.suptitle(title or f"{mode} predictions")
    fig.tight_layout()
    return fig


def plot_model_comparison(results_df):
    """Grouped horizontal bars of accuracy and macro-F1 per model.
    Needs columns model, accuracy, macro_f1; optional CI columns acc_lo, acc_hi, f1_lo, f1_hi draw error bars."""
    import matplotlib.pyplot as plt

    need = {"model", "accuracy", "macro_f1"}
    if not need <= set(results_df.columns):
        raise ValueError(f"results_df needs columns {sorted(need)}; has {list(results_df.columns)}")
    df = results_df.reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.8))
    for off, col, lo, hi, label in ((-0.2, "accuracy", "acc_lo", "acc_hi", "Accuracy"),
                                    (0.2, "macro_f1", "f1_lo", "f1_hi", "Macro-F1")):
        xerr = None
        if lo in df and hi in df:
            xerr = np.vstack([df[col] - df[lo], df[hi] - df[col]]).clip(min=0)
        ax.barh(y + off, df[col], height=0.38, xerr=xerr, capsize=2, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    lo_x = float(min(df["accuracy"].min(), df["macro_f1"].min()))
    ax.set_xlim(max(0.0, lo_x - 0.05), 1.0)
    ax.set_xlabel("Score (bars start above 0; error bars = 95% bootstrap CI where shown)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_class_recall_heatmap(results_df_or_dict):
    """Rows = models, columns = classes. Accepts a DataFrame (index models, 10 class columns), or a dict
    name -> per-class-recall array (10,), or name -> evaluate_predictions dict."""
    import matplotlib.pyplot as plt

    if isinstance(results_df_or_dict, pd.DataFrame):
        df = results_df_or_dict.copy()
        if df.shape[1] != NUM_CLASSES:
            raise ValueError(f"expected {NUM_CLASSES} class columns, got {df.shape[1]}")
        df.columns = CLASS_NAMES
    else:
        data = {}
        for name, v in results_df_or_dict.items():
            rec = v["per_class_recall"] if isinstance(v, dict) else v
            rec = np.asarray(rec, dtype=float)
            if rec.shape != (NUM_CLASSES,):
                raise ValueError(f"{name}: per-class recall must have shape ({NUM_CLASSES},), got {rec.shape}")
            data[str(name)] = rec
        df = pd.DataFrame.from_dict(data, orient="index", columns=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 2.5))
    im = ax.imshow(df.values.astype(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    for i in range(len(df)):
        for j in range(NUM_CLASSES):
            v = df.values[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (pd.isna(v) or v < 0.6) else "black")
    ax.set_title("Per-class recall")
    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- uncertainty and significance
def bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=1000, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI over test images (resample image indices with replacement).

    Returns (point_estimate_on_all_data, lower, upper). metric in {'accuracy','macro_f1'}.
    Quantifies sampling noise of THIS test set for ONE trained model; it does not include training randomness.
    """
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError("metric must be 'accuracy' or 'macro_f1'")
    y_true, y_pred = _check_labels(y_true, y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = float((y_true == y_pred).mean()) if metric == "accuracy" else _macro_f1_from_cm(_confusion(y_true, y_pred))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "accuracy":
            vals[b] = (y_true[idx] == y_pred[idx]).mean()
        else:
            vals[b] = _macro_f1_from_cm(_confusion(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar test on paired correctness of two models on the same images.

    n10 = A correct & B wrong, n01 = A wrong & B correct. Exact two-sided binomial test when n01+n10 < 25,
    otherwise the chi-square statistic with continuity correction. Returns dict(n01, n10, statistic, p_value, method).
    """
    from scipy import stats

    y_true, pred_a = _check_labels(y_true, pred_a)
    _, pred_b = _check_labels(y_true, pred_b)
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n10, n01 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    if n < 25:
        p = float(stats.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)), "p_value": p, "method": "exact binomial"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(stats.chi2.sf(stat, 1)),
            "method": "chi-square with continuity correction"}
