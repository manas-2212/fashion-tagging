"""Experiment orchestration and protocol enforcement (tuning, final fits, freeze, test gate).

Protocol enforced in code:
  * feature sets for tuning contain train/val only; test features exist only inside run_test_gate
  * nothing is subsetted silently: n_train_used is printed and saved for every final fit
  * the test gate needs a freeze file and can run only once (flag file)
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.classical_models import (DEFAULT_GRIDS, estimate_fit_seconds,
                                  fit_classical, tune_classical)
from src.config import (DATA_DIR, MAX_FIT_MINUTES, RESULTS_DIR, ROOT, SEED,
                        SVM_TUNE_N)
from src.evaluation import predict_with_confidence
from src.features import cached_features

MODEL_NAMES = ("logreg", "knn", "rf", "svm")
REPRESENTATIONS = ("pixels", "hog")
CLASSICAL_KEYS = [(m, r) for r in REPRESENTATIONS for m in MODEL_NAMES]  # the 8 classical pairs
MIN_GATED_SUBSET = 100  # floor for a runtime-gated subset, so a bad estimate cannot give a 0-row fit


# ----------------------------------------------------------------------------- JSON helpers
def key_to_str(key):
    """('logreg','pixels') -> 'logreg|pixels'; the CNN key 'cnn' stays 'cnn'."""
    return "|".join(key) if isinstance(key, tuple) else str(key)


def str_to_key(text):
    """Inverse of key_to_str."""
    return tuple(text.split("|")) if "|" in text else text


def _stringify_keys(obj):
    """Recursively turn tuple dict keys into JSON-safe strings."""
    if isinstance(obj, dict):
        return {key_to_str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_keys(v) for v in obj]
    return obj


def _restore_keys(obj):
    """Recursively turn 'a|b' dict keys back into tuples."""
    if isinstance(obj, dict):
        return {str_to_key(k): _restore_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_keys(v) for v in obj]
    return obj


def _json_default(obj):
    """Make numpy scalars/arrays and Paths JSON-serialisable; anything else is an error."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _git_commit():
    """Current git commit hash, or 'unavailable' (no git, not a repo, ...)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- features
def build_feature_sets(X_train, X_val, cache_dir=DATA_DIR / "cache"):
    """Pixel and HOG features for train and val only (this function never receives test images)."""
    return {rep: {"train": cached_features(X_train, rep, "train", cache_dir),
                  "val": cached_features(X_val, rep, "val", cache_dir)}
            for rep in REPRESENTATIONS}


def build_test_features(X_test, cache_dir=DATA_DIR / "cache"):
    """Test features. Called ONLY from run_test_gate."""
    return {rep: cached_features(X_test, rep, "test", cache_dir) for rep in REPRESENTATIONS}


# ----------------------------------------------------------------------------- tuning
def run_classical_tuning(feats, y_tr, y_val, grids=DEFAULT_GRIDS, results_dir=RESULTS_DIR):
    """Tune all 8 (model, representation) pairs on validation; save results/tuning_<model>_<rep>.csv."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    best = {}
    for rep in REPRESENTATIONS:
        for model_name in MODEL_NAMES:
            X_tr, X_val = feats[rep]["train"], feats[rep]["val"]
            subset_n = SVM_TUNE_N if model_name == "svm" else None  # SVM is tuned on a stratified subset
            start = time.perf_counter()
            params, tuning_df = tune_classical(model_name, X_tr, y_tr, X_val, y_val,
                                               grids[model_name], subset_n=subset_n, seed=SEED)
            tuning_df.to_csv(results_dir / f"tuning_{model_name}_{rep}.csv", index=False)
            best[(model_name, rep)] = dict(params)
            note = (f" (tuned on a stratified subset of {min(subset_n, len(y_tr))} of {len(y_tr)})"
                    if subset_n else "")
            print(f"[{model_name}|{rep}] best = {best[(model_name, rep)]}  "
                  f"{time.perf_counter() - start:.1f}s{note}")
    return best


# ----------------------------------------------------------------------------- final fits
def run_classical_final_fits(feats, y_tr, best_params, results_dir=RESULTS_DIR):
    """Fit the 8 final classical models on the train split, with the runtime gate.

    If the estimated fit time exceeds MAX_FIT_MINUTES, fit on a stratified subset sized by a linear
    time ratio (safe for models whose cost grows at least linearly in n). Always prints and saves
    n_train_used. Saves results/fit_info.json (a list of records).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    limit_seconds = MAX_FIT_MINUTES * 60
    n_full = len(y_tr)
    models, infos, records = {}, {}, []
    for rep in REPRESENTATIONS:
        for model_name in MODEL_NAMES:
            key = (model_name, rep)
            params = best_params[key]
            X_tr = feats[rep]["train"]
            estimate = float(estimate_fit_seconds(model_name, params, X_tr, y_tr))
            if not np.isfinite(estimate):
                raise ValueError(f"{key}: fit-time estimate is not finite ({estimate})")
            gated = estimate > limit_seconds
            subset_n = None
            if gated:
                subset_n = min(n_full, max(MIN_GATED_SUBSET, int(n_full * limit_seconds / estimate)))
            pipeline, fit_info = fit_classical(model_name, params, X_tr, y_tr,
                                               subset_n=subset_n, seed=SEED)
            n_used = int(fit_info["n_train_used"])
            if not gated and n_used != n_full:  # nothing may subset silently
                raise RuntimeError(f"{key}: fit used {n_used} of {n_full} rows without the runtime gate")
            print(f"[{model_name}|{rep}] estimate {estimate:.0f}s (limit {limit_seconds}s), "
                  f"gated = {gated}, fit {fit_info['fit_seconds']:.1f}s")
            print(f"[{model_name}|{rep}] n_train_used = {n_used} of {n_full}")
            record = {"model": model_name, "rep": rep, "params": params,
                      "fit_seconds": fit_info["fit_seconds"], "n_train_used": n_used,
                      "n_train_available": n_full, "gated": bool(gated),
                      "estimated_fit_seconds": estimate}
            models[key], infos[key] = pipeline, record
            records.append(record)
    with open(results_dir / "fit_info.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_default)
    return models, infos


# ----------------------------------------------------------------------------- freeze + gate
def freeze_selection(selected, path=RESULTS_DIR / "selected_configs.json"):
    """Write the frozen selection (+ UTC timestamp, git commit). Returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"WARNING: overwriting existing freeze file {path}")
    payload = {"selected": _stringify_keys(selected), "created_utc": _now(),
               "git_commit": _git_commit()}
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    path.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selection(path):
    """Read a freeze file; 'a|b' keys come back as tuples."""
    with open(path, "r", encoding="utf-8") as f:
        return _restore_keys(json.load(f))


def _check_prediction(name, y_pred, conf, n):
    y_pred = np.asarray(y_pred).astype(np.int64)
    conf = np.asarray(conf, dtype=float)
    if y_pred.shape != (n,) or conf.shape != (n,):
        raise ValueError(f"{name}: expected predictions of shape ({n},), "
                         f"got y_pred {y_pred.shape}, conf {conf.shape}")
    return {"y_pred": y_pred, "conf": conf}


def run_test_gate(frozen_path, models, X_test, y_test, cnn_predict_fn=None,
                  flag_path=RESULTS_DIR / "test_gate_done.flag"):
    """Evaluate all models on the test images exactly once. Test labels are used only to check shapes."""
    frozen_path, flag_path = Path(frozen_path), Path(flag_path)
    if not frozen_path.is_file():
        raise RuntimeError(f"Test gate refused: freeze file {frozen_path} is missing. "
                           "Call freeze_selection before touching the test set.")
    if flag_path.exists():
        raise RuntimeError(f"Test gate refused: the test set was already evaluated ({flag_path} exists). "
                           "It may be used only once; delete the flag only if you intend to redo the protocol.")
    if "selected" not in load_selection(frozen_path):
        raise RuntimeError(f"Test gate refused: {frozen_path} is not a valid freeze file.")
    missing = [k for k in CLASSICAL_KEYS if k not in models]
    if missing:  # the gate is one-shot, so a partial model set must not burn it
        raise RuntimeError(f"Test gate refused: missing models {missing}.")
    n = len(X_test)
    if len(y_test) != n:
        raise ValueError(f"X_test has {n} rows but y_test has {len(y_test)}")

    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print(f"Test gate open: {n} test images, freeze sha256 = {frozen_sha}")
    feats = build_test_features(X_test)
    results = {}
    for key in CLASSICAL_KEYS:
        y_pred, conf = predict_with_confidence(models[key], feats[key[1]])
        results[key] = _check_prediction(key_to_str(key), y_pred, conf, n)
    if cnn_predict_fn is not None:
        y_pred, probs = cnn_predict_fn(X_test)
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (n, 10):
            raise ValueError(f"cnn: expected probabilities of shape ({n}, 10), got {probs.shape}")
        results["cnn"] = _check_prediction("cnn", y_pred, probs.max(axis=1), n)
    else:
        print("WARNING: no cnn_predict_fn given; the CNN is NOT evaluated in this gate run.")

    flag_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(flag_path, "x", encoding="utf-8") as f:  # "x" fails if the flag already exists
            json.dump({"created_utc": _now(), "freeze_sha256": frozen_sha, "n_test": n}, f, indent=2)
    except FileExistsError:
        raise RuntimeError("Test gate refused: flag file appeared during evaluation.") from None
    return results
