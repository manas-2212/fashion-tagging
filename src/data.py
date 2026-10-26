"""Data loading, verification and splitting (Milestone M1, owner: Akhil).

Only load_dataset / verify_dataset read the official test images (shape and label checks).
prepare_splits and stratified_subset only ever see the arrays they are given.
"""
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES, DATA_DIR, N_TEST, N_TRAIN_FULL, NUM_CLASSES,
    RESULTS_DIR, SEED, VAL_FRACTION,
)

IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad IDX image magic number {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).copy()


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad IDX label magic number {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_dataset(root=DATA_DIR):
    """Return (X_train_full, y_train_full, X_test, y_test) as uint8 (N,28,28) / int64 (N,)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision import datasets
        train = datasets.FashionMNIST(root=str(root), train=True, download=True)
        test = datasets.FashionMNIST(root=str(root), train=False, download=True)
        # .data is a uint8 tensor, .targets an int64 tensor; np.array copies, dtype is explicit
        X_train = np.array(train.data.numpy(), dtype=np.uint8)
        y_train = np.array(train.targets.numpy(), dtype=np.int64)
        X_test = np.array(test.data.numpy(), dtype=np.uint8)
        y_test = np.array(test.targets.numpy(), dtype=np.int64)
    except Exception as err:  # download/network/torchvision problem -> try local IDX files
        fb = root / "idx_fallback"  # put the four official .gz files here manually
        if not all((fb / name).exists() for name in IDX_FILES.values()):
            raise RuntimeError(
                f"torchvision load failed ({err!r}) and no IDX files found in {fb}. "
                "Download the four official Fashion-MNIST .gz files manually into that folder."
            ) from err
        X_train = _read_idx_images(fb / IDX_FILES["train_x"])
        y_train = _read_idx_labels(fb / IDX_FILES["train_y"])
        X_test = _read_idx_images(fb / IDX_FILES["test_x"])
        y_test = _read_idx_labels(fb / IDX_FILES["test_y"])
    return X_train, y_train, X_test, y_test


def verify_dataset(X_train_full, y_train_full, X_test, y_test):
    """Assert the data contract; return a dict of findings. Raises AssertionError on failure."""
    assert X_train_full.shape == (N_TRAIN_FULL, 28, 28), X_train_full.shape
    assert X_test.shape == (N_TEST, 28, 28), X_test.shape
    assert y_train_full.shape == (N_TRAIN_FULL,) and y_test.shape == (N_TEST,)
    assert X_train_full.dtype == np.uint8 and X_test.dtype == np.uint8
    assert y_train_full.dtype == np.int64 and y_test.dtype == np.int64
    for X in (X_train_full, X_test):
        assert int(X.min()) >= 0 and int(X.max()) <= 255
        assert not np.isnan(X.astype(np.float32)).any()  # trivially true for uint8; kept as required
    for y in (y_train_full, y_test):
        assert int(y.min()) >= 0 and int(y.max()) <= NUM_CLASSES - 1
    train_counts = np.bincount(y_train_full, minlength=NUM_CLASSES)
    test_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    # The official set is expected to be balanced; assert it and report what was found.
    assert (train_counts == N_TRAIN_FULL // NUM_CLASSES).all(), train_counts
    assert (test_counts == N_TEST // NUM_CLASSES).all(), test_counts
    # torchvision's class list is a class attribute, so no download is needed to check it
    try:
        from torchvision.datasets import FashionMNIST
        assert list(FashionMNIST.classes) == list(CLASS_NAMES), FashionMNIST.classes
        aligned = True
    except ImportError:
        aligned = None  # torchvision missing: not checked
    return {
        "train_shape": list(X_train_full.shape), "test_shape": list(X_test.shape),
        "pixel_min": int(min(X_train_full.min(), X_test.min())),
        "pixel_max": int(max(X_train_full.max(), X_test.max())),
        "train_class_counts": train_counts.tolist(), "test_class_counts": test_counts.tolist(),
        "classes_aligned_with_torchvision": aligned,
    }


def prepare_splits(X_train_full, y_train_full, val_fraction=VAL_FRACTION, seed=SEED):
    """Stratified train/val split of the official TRAIN images only (test data never touched)."""
    n_total = len(y_train_full)
    n_val = int(round(n_total * val_fraction))
    idx_train, idx_val = train_test_split(
        np.arange(n_total), test_size=n_val, stratify=y_train_full, random_state=seed
    )
    idx_train, idx_val = np.sort(idx_train), np.sort(idx_val)  # sorted for readability; still deterministic
    out = {
        "X_train": X_train_full[idx_train], "y_train": y_train_full[idx_train],
        "X_val": X_train_full[idx_val], "y_val": y_train_full[idx_val],
        "idx_train": idx_train, "idx_val": idx_val,
    }
    # Write the audit file only for the official configuration, so tests with other seeds/mock data don't overwrite it
    if seed == SEED and val_fraction == VAL_FRACTION and n_total == N_TRAIN_FULL:
        audit = {
            "seed": seed, "val_fraction": val_fraction,
            "n_train": int(len(idx_train)), "n_val": int(len(idx_val)),
            "val_per_class": np.bincount(out["y_val"], minlength=NUM_CLASSES).tolist(),
            "train_per_class": np.bincount(out["y_train"], minlength=NUM_CLASSES).tolist(),
            "idx_val_sha256": hashlib.sha256(idx_val.astype(np.int64).tobytes()).hexdigest(),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "split_audit.json").write_text(json.dumps(audit, indent=2))
    return out


def stratified_subset(X, y, n, seed=SEED):
    """Reproducible class-proportional subset. Returns (X_sub, y_sub, idx) with idx sorted."""
    if n >= len(y):
        idx = np.arange(len(y))
    else:
        idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
        idx = np.sort(idx)
    return X[idx], y[idx], idx
