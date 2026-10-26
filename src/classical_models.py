"""Classical ML models for Fashion-MNIST (owner: Tanish).

Pipelines, validation-based tuning, runtime estimate, fitting, parameter counts, overfit check.
Every function is stateless and never mutates its inputs, so it is safe to call in a loop.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import N_TEST, N_VAL, SEED

DEFAULT_GRIDS = {
    "logreg": {"C": [0.1, 1.0]},
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "rf": {"min_samples_leaf": [1, 3]},
    "svm": {"C": [1, 10]},
}


def build_classical_pipeline(model_name, params=None):
    """Return a sklearn Pipeline. Steps: 'scaler' (logreg, svm only) and 'clf'."""
    params = dict(params or {})  # copy so the caller's dict is never mutated
    if model_name == "logreg":
        kw = dict(max_iter=500, random_state=42)
        kw.update(params)  # merge: caller may override defaults without duplicate-keyword error
        # multi_class is deprecated in recent sklearn; default is multinomial softmax, which we want
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**kw))])
    if model_name == "knn":
        kw = dict(algorithm="brute", n_jobs=-1)
        kw.update(params)
        return Pipeline([("clf", KNeighborsClassifier(**kw))])
    if model_name == "rf":
        kw = dict(n_estimators=200, n_jobs=-1, random_state=42)
        kw.update(params)
        return Pipeline([("clf", RandomForestClassifier(**kw))])
    if model_name == "svm":
        kw = dict(kernel="rbf", gamma="scale", random_state=42)  # no probability=True: too costly
        kw.update(params)
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(**kw))])
    raise ValueError(f"Unknown model_name {model_name!r}; expected logreg, knn, rf or svm")


def _stratified_idx(y, n, seed=SEED):
    """Sorted indices of a stratified subsample of size n (all indices if n is None or >= len(y))."""
    y = np.asarray(y)
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def _subset(X, y, n, seed=SEED):
    """Stratified subset (X_s, y_s). Returns the original arrays, no copy, when n covers everything."""
    if n is None or n >= len(y):
        return X, y
    idx = _stratified_idx(y, n, seed)
    return X[idx], y[idx]


def _select_best_index(scores):
    """scores = list of (val_macro_f1, val_accuracy). Highest F1, accuracy breaks ties, first wins full ties."""
    best_i, best_key = 0, scores[0]
    for i, key in enumerate(scores):
        if key > best_key:  # strict '>' keeps the earliest candidate on exact ties
            best_i, best_key = i, key
    return best_i


def estimate_fit_seconds(model_name, params, X_tr, y_tr, probe_n=5000):
    """Estimate full-size fit time from a stratified probe fit.

    ASSUMPTION: SVM time scales as (n/probe_n)**2 (probably optimistic: real SVC cost can grow faster);
    kNN, LogReg and RF scale linearly. For kNN the printed predict-time estimate matters more than fit time.
    """
    n = len(X_tr)
    X_p, y_p = _subset(X_tr, y_tr, probe_n)
    n_p = len(y_p)
    pipe = build_classical_pipeline(model_name, params)
    t0 = time.perf_counter()
    pipe.fit(X_p, y_p)
    probe_fit = time.perf_counter() - t0
    factor = (n / n_p) ** 2 if model_name == "svm" else n / n_p
    est = float(probe_fit * factor)
    print(f"[{model_name}] probe n={n_p}: {probe_fit:.2f}s -> estimated fit on {n}: {est:.1f}s "
          f"({'quadratic' if model_name == 'svm' else 'linear'} extrapolation)")
    if model_name == "knn":
        q = X_p[: min(500, n_p)]  # timing only, so using training rows as queries is fine
        t0 = time.perf_counter()
        pipe.predict(q)
        t_pred = time.perf_counter() - t0
        for label, n_query in (("val", N_VAL), ("test", N_TEST)):
            est_pred = t_pred * (n_query / len(q)) * (n / n_p)  # cost ~ queries x stored rows
            print(f"[knn] estimated predict time on {n_query} {label} images against {n} stored rows: {est_pred:.1f}s")
    return est


def tune_classical(model_name, X_tr, y_tr, X_val, y_val, param_grid, subset_n=None, seed=SEED):
    """Manual grid loop (no CV). Select max val macro-F1; val accuracy breaks ties."""
    X_fit, y_fit = _subset(X_tr, y_tr, subset_n, seed)
    candidates = list(ParameterGrid(param_grid))  # ParameterGrid does not modify param_grid
    rows, scores = [], []
    for params in candidates:
        pipe = build_classical_pipeline(model_name, params)
        t0 = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(X_val)
        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average="macro"))
        rows.append({"model": model_name, "params": str(params), "val_accuracy": acc,
                     "val_macro_f1": f1, "fit_seconds": fit_s, "n_train_used": len(y_fit)})
        scores.append((f1, acc))
        print(f"[{model_name}] {params} | val_acc={acc:.4f} val_macro_f1={f1:.4f} "
              f"| fit={fit_s:.1f}s | n_train_used={len(y_fit)}")
    best = _select_best_index(scores)
    print(f"[{model_name}] best params: {candidates[best]}")
    return dict(candidates[best]), pd.DataFrame(rows)


def fit_classical(model_name, params, X_tr, y_tr, subset_n=None, seed=SEED):
    """Fit on all of X_tr, or on a stratified subset of size subset_n. Always prints n_train_used."""
    X_fit, y_fit = _subset(X_tr, y_tr, subset_n, seed)
    pipe = build_classical_pipeline(model_name, params)
    t0 = time.perf_counter()
    pipe.fit(X_fit, y_fit)
    fit_s = time.perf_counter() - t0
    info = {"fit_seconds": fit_s, "n_train_used": int(len(y_fit)), "params": dict(params or {})}
    print(f"[{model_name}] n_train_used = {info['n_train_used']} of {len(y_tr)} | "
          f"fit_seconds = {fit_s:.1f} | params = {info['params']}")
    return pipe, info


def classical_param_count(fitted_pipeline):
    """(count, note). Counts are NOT comparable across model families."""
    clf = fitted_pipeline.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return int(clf.coef_.size + clf.intercept_.size), "weights + intercepts"
    if isinstance(clf, RandomForestClassifier):
        return int(sum(e.tree_.node_count for e in clf.estimators_)), \
            "total tree nodes; not comparable to weights"
    if isinstance(clf, SVC):
        return int(sum(clf.n_support_)), "support vectors; not comparable to weights"
    if isinstance(clf, KNeighborsClassifier):
        return None, "non-parametric; stores N x d training values"
    return None, "unknown model type"


def overfit_check(pipeline, X_tr, y_tr, X_val, y_val, n_eval=5000, seed=SEED):
    """Train accuracy on a stratified n_eval subsample of the fitted data vs full validation accuracy.

    k-NN caveat: each training point is its own nearest neighbour, so train accuracy is inflated
    (exactly 1.0 for k=1). See docs/classical_notes.md.
    """
    X_s, y_s = _subset(X_tr, y_tr, n_eval, seed)
    train_acc = float(accuracy_score(y_s, pipeline.predict(X_s)))
    val_acc = float(accuracy_score(y_val, pipeline.predict(X_val)))
    gap = train_acc - val_acc
    print(f"overfit_check: train_acc={train_acc:.4f} (stratified subsample n={len(y_s)}) "
          f"val_acc={val_acc:.4f} gap={gap:.4f}")
    return {"train_acc": train_acc, "val_acc": val_acc, "gap": gap, "n_eval": int(len(y_s))}
