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


# ======================= Second PR tests =======================
from src.config import CLASS_NAMES
from src.data import class_centroid_distances, find_duplicates, pca_projection, sanity_baselines


def test_find_duplicates_mock():
    rng = np.random.default_rng(1)
    A = rng.integers(0, 256, (100, 28, 28), dtype=np.uint8)
    A[10] = A[3]; A[50] = A[3]            # 2 injected copies inside A
    B = rng.integers(0, 256, (50, 28, 28), dtype=np.uint8)
    B[7] = A[20]                          # A[20] also appears in B
    B[9] = B[8]                           # 1 copy inside B
    A_copy = A.copy()
    r = find_duplicates(A, B)
    assert r["n_within_a"] == 2 and r["n_within_b"] == 1
    assert r["n_overlap"] == 1 and r["overlap_indices_a"] == [20]
    r0 = find_duplicates(A)               # X_b=None must work
    assert r0["n_within_a"] == 2 and r0["n_within_b"] == 0 and r0["n_overlap"] == 0
    assert np.array_equal(A, A_copy)      # input not modified


def test_pca_projection_mock(mock):
    X, _ = mock
    Z, evr = pca_projection(X, n_components=5)   # train-only array passed in
    assert Z.shape == (len(X), 5) and Z.dtype == np.float32 and evr.shape == (5,)
    assert 0 < evr.sum() <= 1 + 1e-6


def test_centroids_mock(mock):
    X, y = mock
    D = class_centroid_distances(X, y)
    assert list(D.columns) == CLASS_NAMES and list(D.index) == CLASS_NAMES
    assert np.allclose(D.values, D.values.T) and np.allclose(np.diag(D.values), 0)


def test_sanity_baselines_mock(mock):
    X, y = mock
    s = prepare_splits(X, y)              # 6000 mock images -> 600 val, 60 per class
    df = sanity_baselines(s["X_train"], s["y_train"], s["X_val"], s["y_val"])
    print(df)
    assert list(df.columns) == ["name", "val_accuracy", "val_macro_f1"] and len(df) == 2
    assert ((df.val_accuracy >= 0) & (df.val_accuracy <= 1)).all()
    maj = df.loc[df.name == "majority_class", "val_accuracy"].iloc[0]
    assert abs(maj - 0.1) < 1e-9          # balanced val -> majority accuracy = 1/10
