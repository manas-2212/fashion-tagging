import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports work under plain pytest

import numpy as np
import pytest

from src.config import N_TRAIN, N_VAL, SEED
from src.data import load_dataset, prepare_splits, stratified_subset, verify_dataset


@pytest.fixture(scope="module")
def real():
    try:
        return load_dataset()
    except Exception as err:
        pytest.skip(f"real dataset unavailable: {err}")


@pytest.fixture(scope="module")
def mock():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(10), 600).astype(np.int64)  # balanced 6000-image mock
    X = rng.integers(0, 256, (len(y), 28, 28), dtype=np.uint8)
    return X, y


def test_load_and_verify(real):
    Xtr, ytr, Xte, yte = real
    print("shapes", Xtr.shape, Xte.shape, "dtypes", Xtr.dtype, ytr.dtype)
    info = verify_dataset(Xtr, ytr, Xte, yte)
    print("verify:", info)


def test_real_split(real):
    Xtr, ytr, _, _ = real
    Xtr_copy = Xtr.copy()
    s = prepare_splits(Xtr, ytr)
    assert len(s["idx_train"]) == N_TRAIN and len(s["idx_val"]) == N_VAL
    assert len(np.intersect1d(s["idx_train"], s["idx_val"])) == 0
    assert len(np.union1d(s["idx_train"], s["idx_val"])) == 60000
    val_counts = np.bincount(s["y_val"], minlength=10)
    print("val per class:", val_counts.tolist())
    assert (val_counts == 600).all()
    assert np.array_equal(Xtr, Xtr_copy)  # input not modified


def test_seed_behaviour(mock):
    X, y = mock
    a = prepare_splits(X, y, seed=42)["idx_val"]
    b = prepare_splits(X, y, seed=42)["idx_val"]
    c = prepare_splits(X, y, seed=43)["idx_val"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_subset(real):
    _, ytr, _, _ = real
    X = np.zeros((len(ytr), 28, 28), dtype=np.uint8)
    _, ys, idx = stratified_subset(X, ytr, 10000)
    counts = np.bincount(ys, minlength=10)
    print("subset per class:", counts.tolist())
    assert len(ys) == 10000 and (np.abs(counts - 1000) <= 1).all()
    assert np.array_equal(idx, stratified_subset(X, ytr, 10000)[2])
