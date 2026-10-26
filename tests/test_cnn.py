"""Tests for src/cnn.py (run from repo root: python -m pytest tests/test_cnn.py -s)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

import src.cnn as cnn
from src.cnn import (build_cnn, compute_mean_std, make_cnn_tensor, plot_learning_curves, predict_cnn, train_cnn)
from src.benchmarking import count_trainable_params
from src.config import SEED, set_global_seed

CPU = torch.device("cpu")


def mock(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 28, 28), dtype=np.uint8), rng.integers(0, 10, n).astype(np.int64)


def test_param_count_and_forward_shape():
    set_global_seed(SEED)
    m = build_cnn()
    out = m(torch.randn(8, 1, 28, 28))
    n = count_trainable_params(m)
    print("output shape:", tuple(out.shape), "| trainable params:", n)
    assert tuple(out.shape) == (8, 10)
    assert n == 241546 == 320 + 18496 + 73856 + 147584 + 1290


def test_make_cnn_tensor_stats_and_purity():
    X, _ = mock(2000)
    X_before = X.copy()
    mean, std = compute_mean_std(X)
    t = make_cnn_tensor(X, mean, std)
    print("tensor:", tuple(t.shape), t.dtype, "| mean %.5f std %.5f" % (t.mean(), t.std(unbiased=False)))
    assert tuple(t.shape) == (2000, 1, 28, 28) and t.dtype == torch.float32
    assert abs(t.mean().item()) < 1e-4 and abs(t.std(unbiased=False).item() - 1) < 1e-3
    assert np.array_equal(X, X_before) and X.dtype == np.uint8          # input never modified
    # mean/std really come from X/255 of the given split
    assert abs(mean - (X / 255.0).mean()) < 1e-12


def test_overfit_200_images():
    X, y = mock(200)
    Xv, yv = mock(50, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    set_global_seed(SEED)
    m = build_cnn(dropout=0.0)    # dropout off: this tests the optimisation loop's capacity to memorise
    m, hist, info = train_cnn(m, Xt, y, Xvt, yv, CPU, lr=1e-3, batch_size=32, max_epochs=30, patience=30)
    pred, _ = predict_cnn(m, Xt, CPU)
    # NOTE: after train_cnn the weights are the BEST-VAL-LOSS epoch, not the last one, so evaluate the last-epoch
    # memorisation via the training-mode running accuracy in the history as well.
    print("max running train acc: %.3f | eval-mode acc with restored best weights: %.3f | best_epoch %d"
          % (hist.train_acc.max(), (pred == y).mean(), info["best_epoch"]))
    assert hist.train_acc.max() > 0.9


def test_early_stopping_real_rise_and_weights_restored():
    # random labels, disjoint val set -> val loss must eventually rise while the model memorises train
    X, y = mock(200)
    Xv, yv = mock(200, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    set_global_seed(SEED)
    m, hist, info = train_cnn(build_cnn(dropout=0.0), Xt, y, Xvt, yv, CPU, batch_size=32, max_epochs=80, patience=3)
    print("epochs_run", info["epochs_run"], "best_epoch", info["best_epoch"], "best_val_loss %.5f" % info["best_val_loss"])
    assert info["stopped_early"] and info["epochs_run"] == info["best_epoch"] + 3
    assert hist.val_loss.min() == pytest.approx(info["best_val_loss"])
    vl, va = cnn._evaluate(m, Xvt, torch.as_tensor(yv), CPU)           # restored weights reproduce best val loss
    print("re-evaluated val loss %.5f vs best %.5f" % (vl, info["best_val_loss"]))
    assert vl == pytest.approx(info["best_val_loss"], abs=1e-6) and va == pytest.approx(info["best_val_acc"])


def test_early_stopping_forced_rise_restores_best_epoch_state(monkeypatch):
    X, y = mock(64)
    Xv, yv = mock(32, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    scripted = [1.0, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4]               # best at epoch 2, then rises
    snaps, calls = [], {"i": 0}

    def fake_eval(model, *a, **k):
        snaps.append({kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()})
        loss = scripted[calls["i"]]; calls["i"] += 1
        return loss, 0.5

    monkeypatch.setattr(cnn, "_evaluate", fake_eval)
    set_global_seed(SEED)
    m, hist, info = train_cnn(build_cnn(), Xt, y, Xvt, yv, CPU, batch_size=32, max_epochs=8, patience=3)
    print("forced rise -> epochs_run", info["epochs_run"], "best_epoch", info["best_epoch"])
    assert info["best_epoch"] == 2 and info["epochs_run"] == 5 and info["best_val_loss"] == 0.9
    for k, v in m.state_dict().items():                                  # weights == epoch-2 snapshot
        assert torch.equal(v.cpu(), snaps[1][k])


def test_label_dtype_and_shape_passed_to_loss(monkeypatch):
    X, y = mock(70)
    Xv, yv = mock(20, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    seen = []
    orig = cnn.F.cross_entropy

    def spy(out, target, *a, **k):
        seen.append((target.dtype, tuple(target.shape))); return orig(out, target, *a, **k)

    monkeypatch.setattr(cnn.F, "cross_entropy", spy)
    train_cnn(build_cnn(), Xt, y.astype(np.int32), Xvt, yv, CPU, batch_size=32, max_epochs=1, patience=1)
    print("loss targets (dtype, shape), first 3:", seen[:3])
    assert all(d == torch.int64 and len(s) == 1 for d, s in seen)
    with pytest.raises(ValueError):                                      # one-hot labels are rejected
        train_cnn(build_cnn(), Xt, np.eye(10)[y[:70]], Xvt, yv, CPU, max_epochs=1)


def test_two_cpu_runs_identical_histories():
    X, y = mock(256)
    Xv, yv = mock(64, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    hs = []
    for _ in range(2):
        set_global_seed(SEED)
        _, h, _ = train_cnn(build_cnn(), Xt, y, Xvt, yv, CPU, batch_size=64, max_epochs=3, patience=3, seed=SEED)
        hs.append(h)
    pd.testing.assert_frame_equal(hs[0], hs[1], check_exact=True)
    print("identical histories:\n", hs[0])


def test_checkpoint_roundtrip_and_predict(tmp_path):
    X, y = mock(128)
    Xv, yv = mock(32, seed=1)
    mean, std = compute_mean_std(X)
    Xt, Xvt = make_cnn_tensor(X, mean, std), make_cnn_tensor(Xv, mean, std)
    ck = tmp_path / "sub" / "best.pt"
    m, _, info = train_cnn(build_cnn(), Xt, y, Xvt, yv, CPU, max_epochs=2, patience=2, checkpoint_path=ck)
    blob = torch.load(ck, weights_only=True)
    m2 = build_cnn(); m2.load_state_dict(blob["state_dict"])
    grads_before = [None if q.grad is None else q.grad.clone() for q in m.parameters()]
    p1, pr1 = predict_cnn(m, Xvt, CPU, batch_size=7)
    p2, pr2 = predict_cnn(m2, Xvt, CPU)
    assert blob["best_epoch"] == info["best_epoch"]
    assert p1.dtype == np.int64 and pr1.dtype == np.float32 and pr1.shape == (32, 10)
    assert np.array_equal(p1, p2) and np.allclose(pr1.sum(1), 1, atol=1e-5) and np.allclose(pr1, pr2, atol=1e-6)
    for g0, q in zip(grads_before, m.parameters()):                      # predict_cnn must not touch gradients
        assert (g0 is None and q.grad is None) or torch.equal(g0, q.grad)
    assert not m.training                                                # model left in eval mode


def test_plot_learning_curves():
    h = pd.DataFrame(dict(epoch=[1, 2, 3], train_loss=[1, .8, .7], train_acc=[.5, .6, .7], val_loss=[1, .9, .95], val_acc=[.5, .55, .54]))
    fig = plot_learning_curves(h, 2)
    assert len(fig.axes) == 2 and all(ax.get_legend() is not None for ax in fig.axes)
