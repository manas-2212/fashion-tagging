"""Phase 2 experiment bookkeeping and protocol enforcement (owner: Puneet).

Three responsibilities:
  1. RunStore: one JSON record (+ checkpoint + validation probabilities) per (configuration, seed). A finished
     run is never repeated, so a Colab session that disconnects can simply be restarted.
  2. Freeze: the Phase 2 selection (final deep model, hybrid hyper-parameters, cascade threshold, the exact list
     of models allowed through the gate) is written to one file with a timestamp, git commit and sha256.
  3. Test gate: the official test set is loaded INSIDE the gate, only if the freeze file exists and the Phase 2
     flag file does not. The set of evaluated models must equal the frozen list exactly.

Honest note on test-set reuse: the same 10,000 test images were already used once at the Phase 1 gate, and the
Phase 1 error analysis motivated Phase 2 design choices. Phase 2 therefore selects everything on validation and
opens the test set once more, for the frozen Phase 2 list only. This is disclosed in the report as a limitation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import NUM_CLASSES
from src.data import stratified_subset
from src.dl_config import P2_RESULTS_DIR
from src.experiments import _json_default, _now, freeze_selection, load_selection

FREEZE_NAME = "selected_configs_phase2.json"
FLAG_NAME = "test_gate_phase2_done.flag"
GATE_OUTPUT_NAME = "test_gate_outputs_phase2.npz"


def run_id(name: str, seed: int) -> str:
    return f"{name}__seed{int(seed)}"


class RunStore:
    """Folder layout: <root>/runs/<id>.json, <root>/checkpoints/<id>.pt, <root>/val_probs/<id>.npy."""

    def __init__(self, root=P2_RESULTS_DIR):
        self.root = Path(root)
        for sub in ("runs", "checkpoints", "val_probs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def record_path(self, name, seed) -> Path:
        return self.root / "runs" / f"{run_id(name, seed)}.json"

    def checkpoint_path(self, name, seed) -> Path:
        return self.root / "checkpoints" / f"{run_id(name, seed)}.pt"

    def val_probs_path(self, name, seed) -> Path:
        return self.root / "val_probs" / f"{run_id(name, seed)}.npy"

    def load_record(self, name, seed) -> dict | None:
        path = self.record_path(name, seed)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def is_done(self, cfg, seed, n_train: int) -> bool:
        """A stored run counts only if its configuration AND training-set size match the request exactly."""
        rec = self.load_record(cfg.name, seed)
        return (rec is not None and rec.get("config") == cfg.to_dict() and rec.get("n_train") == int(n_train)
                and self.checkpoint_path(cfg.name, seed).is_file() and self.val_probs_path(cfg.name, seed).is_file())

    def save(self, record: dict, val_probs: np.ndarray) -> None:
        name, seed = record["name"], record["seed"]
        np.save(self.val_probs_path(name, seed), np.asarray(val_probs, dtype=np.float32))
        tmp = self.record_path(name, seed).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
        tmp.replace(self.record_path(name, seed))          # the record appears last: its presence means "complete"

    def run_or_load(self, cfg, seed, data: dict, device, run_fn, verbose: bool = True) -> dict:
        """Return the stored record, or train with run_fn(cfg, seed, data, device, checkpoint_path) and store it."""
        if self.is_done(cfg, seed, len(data["y_tr"])):
            if verbose:
                rec = self.load_record(cfg.name, seed)
                print(f"[{cfg.name} | seed {seed}] already done: val macro-F1 {rec['val_macro_f1']:.4f} (skipped)")
                return rec
            return self.load_record(cfg.name, seed)
        if self.record_path(cfg.name, seed).is_file():
            print(f"[{cfg.name} | seed {seed}] stored run has a different configuration or data size: re-training")
        _, record, val_probs = run_fn(cfg, seed, data, device, checkpoint_path=self.checkpoint_path(cfg.name, seed),
                                      verbose=verbose)
        self.save(record, val_probs)
        return record

    def load_val_probs(self, name, seed) -> np.ndarray:
        return np.load(self.val_probs_path(name, seed))

    def records(self, names=None, seeds=None) -> list[dict]:
        """All stored records, optionally restricted to given configuration names and seeds."""
        out = []
        for path in sorted((self.root / "runs").glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if (names is None or rec["name"] in names) and (seeds is None or rec["seed"] in seeds):
                out.append(rec)
        return out


# ----------------------------------------------------------------------------- freeze
def freeze_phase2(selection: dict, results_dir=P2_RESULTS_DIR) -> str:
    """Write the Phase 2 freeze file. `selection` must list the gate models. Returns the file's sha256."""
    if not selection.get("gate_models"):
        raise ValueError("selection['gate_models'] must list every model that will be evaluated on the test set")
    if len(set(selection["gate_models"])) != len(selection["gate_models"]):
        raise ValueError("gate_models contains duplicates")
    return freeze_selection(selection, Path(results_dir) / FREEZE_NAME)


def load_phase2_selection(results_dir=P2_RESULTS_DIR) -> dict:
    return load_selection(Path(results_dir) / FREEZE_NAME)["selected"]


# ----------------------------------------------------------------------------- test gate
def run_phase2_test_gate(predict_all, load_test_fn, results_dir=P2_RESULTS_DIR, test_subset_n: int | None = None,
                         seed: int = 42) -> dict:
    """Evaluate the frozen Phase 2 models on the official test set exactly once.

    predict_all(X_test_uint8) -> {model name: {"y_pred": (N,) int, "probs": (N,10) float or None}}
    load_test_fn() -> (X_test, y_test). It is called here and nowhere earlier in the pipeline.
    test_subset_n: ONLY for the smoke test (a stratified subset keeps the dry run fast); None = all 10,000 images.

    Refuses to run without a freeze file, with an existing flag, or when the predicted model names differ from
    the frozen list. Saves predictions, probabilities and labels to <results_dir>/test_gate_outputs_phase2.npz.
    """
    results_dir = Path(results_dir)
    frozen_path, flag_path = results_dir / FREEZE_NAME, results_dir / FLAG_NAME
    if not frozen_path.is_file():
        raise RuntimeError(f"Phase 2 test gate refused: freeze file {frozen_path} is missing.")
    if flag_path.exists():
        raise RuntimeError(f"Phase 2 test gate refused: the test set was already evaluated ({flag_path} exists).")
    frozen = load_selection(frozen_path)
    if "selected" not in frozen or "gate_models" not in frozen["selected"]:
        raise RuntimeError(f"Phase 2 test gate refused: {frozen_path} is not a valid Phase 2 freeze file.")
    expected = list(frozen["selected"]["gate_models"])
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()

    X_test, y_test = load_test_fn()                         # the only place where test images are loaded
    test_idx = np.arange(len(y_test))
    if test_subset_n is not None and test_subset_n < len(y_test):
        X_test, y_test, test_idx = stratified_subset(X_test, y_test, test_subset_n, seed=seed)
    n = len(y_test)
    print(f"Phase 2 test gate open: {n} test images, {len(expected)} frozen models, freeze sha256 = {frozen_sha}")

    outputs = predict_all(X_test)
    if set(outputs) != set(expected):                       # nothing extra, nothing missing
        missing, extra = sorted(set(expected) - set(outputs)), sorted(set(outputs) - set(expected))
        raise RuntimeError(f"Phase 2 test gate refused: model list differs from the freeze file. "
                           f"missing={missing} extra={extra}")
    arrays = {"y_test": np.asarray(y_test, dtype=np.int64), "test_idx": np.asarray(test_idx, dtype=np.int64)}
    results = {}
    for name in expected:
        y_pred = np.asarray(outputs[name]["y_pred"]).astype(np.int64)
        if y_pred.shape != (n,) or y_pred.min() < 0 or y_pred.max() >= NUM_CLASSES:
            raise ValueError(f"{name}: predictions must be class indices of shape ({n},)")
        results[name] = {"y_pred": y_pred, "probs": None}
        arrays[f"{name}::pred"] = y_pred
        probs = outputs[name].get("probs")
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float32)
            if probs.shape != (n, NUM_CLASSES):
                raise ValueError(f"{name}: probabilities must have shape ({n}, {NUM_CLASSES})")
            results[name]["probs"] = probs
            arrays[f"{name}::probs"] = probs

    try:
        with open(flag_path, "x", encoding="utf-8") as f:   # "x": fails if the flag already exists
            json.dump({"created_utc": _now(), "freeze_sha256": frozen_sha, "n_test": n,
                       "full_test_set": bool(test_subset_n is None), "models": expected}, f, indent=2)
    except FileExistsError:
        raise RuntimeError("Phase 2 test gate refused: flag file appeared during evaluation.") from None
    np.savez_compressed(results_dir / GATE_OUTPUT_NAME, **arrays)
    return {"y_test": arrays["y_test"], "test_idx": arrays["test_idx"], "results": results,
            "freeze_sha256": frozen_sha}


def load_gate_outputs(results_dir=P2_RESULTS_DIR) -> dict:
    """Read the saved gate outputs (only exists after the gate has run)."""
    results_dir = Path(results_dir)
    if not (results_dir / FLAG_NAME).exists():
        raise RuntimeError("the Phase 2 test gate has not run yet; there are no test outputs to analyse")
    z = np.load(results_dir / GATE_OUTPUT_NAME)
    results = {}
    for key in z.files:
        if "::" in key:
            name, kind = key.rsplit("::", 1)
            results.setdefault(name, {"y_pred": None, "probs": None})["y_pred" if kind == "pred" else "probs"] = z[key]
    return {"y_test": z["y_test"], "test_idx": z["test_idx"], "results": results}
