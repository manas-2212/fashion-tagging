"""Tests for src/dl_train.py (owner: Manas). Tiny synthetic data: fast on a CPU."""
import math

import numpy as np
import pytest
import torch

from src.dl_models import FashionNet
from src.dl_train import (TrainConfig, load_checkpoint, lr_at_step, make_optimizer, predict_proba, run_single,
                          split_decay_groups, train_model)

SMALL_ARCH = dict(widths=(8, 16), blocks_per_stage=1, use_bn=True, residual=True, dropout=0.1)


def _data(n_tr=200, n_val=80, seed=0):
    """Two easy 'classes': bright top half vs bright bottom half (labels 0 and 1)."""
    rng = np.random.default_rng(seed)

    def make(n):
        y = rng.integers(0, 2, n)
        x = rng.normal(0, 0.3, (n, 1, 28, 28)).astype(np.float32)
        x[y == 0, :, :14] += 2.0
        x[y == 1, :, 14:] += 2.0
        return torch.from_numpy(x), y.astype(np.int64)

    X_tr, y_tr = make(n_tr)
    X_val, y_val = make(n_val)
    return dict(X_tr_t=X_tr, y_tr=y_tr, X_val_t=X_val, y_val=y_val, pad_value=-0.8)


def _cfg(**kw):
    base = dict(name="t", arch="fashionnet", arch_kwargs=dict(SMALL_ARCH), max_epochs=3, patience=3, batch_size=32,
                warmup_epochs=1, aug="crop_flip_erase")
    base.update(kw)
    return TrainConfig(**base)


def test_config_validation():
    for bad in (dict(optimizer="rmsprop"), dict(schedule="step"), dict(aug="mixup"), dict(max_epochs=0)):
        with pytest.raises(ValueError):
            _cfg(**bad)
    d = _cfg(arch_kwargs=dict(widths=(8, 16))).to_dict()
    assert d["arch_kwargs"]["widths"] == [8, 16]          # JSON-friendly


def test_lr_schedule_values():
    cfg = _cfg(lr=1.0, max_epochs=10, warmup_epochs=2, schedule="cosine")
    spe = 10
    assert lr_at_step(0, cfg, spe) == pytest.approx(1 / 20)
    assert lr_at_step(19, cfg, spe) == pytest.approx(1.0)                    # end of warm-up
    assert lr_at_step(20, cfg, spe) == pytest.approx(1.0)                    # cosine starts at the peak
    assert lr_at_step(60, cfg, spe) == pytest.approx(0.5)                    # half-way through the decay
    assert lr_at_step(99, cfg, spe) < 0.001
    lrs = [lr_at_step(s, cfg, spe) for s in range(20, 100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))                         # monotone decay
    assert lr_at_step(57, _cfg(lr=0.3, schedule="constant", warmup_epochs=0), spe) == 0.3


def test_weight_decay_skips_biases_and_batchnorm():
    model = FashionNet(**SMALL_ARCH)
    decay, no_decay = split_decay_groups(model)
    assert all(p.ndim > 1 for p in decay) and all(p.ndim <= 1 for p in no_decay)
    assert sum(p.numel() for p in decay + no_decay) == sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, _cfg(weight_decay=0.01))
    assert opt.param_groups[0]["weight_decay"] == 0.01 and opt.param_groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize("name,cls", [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW),
                                      ("sgd", torch.optim.SGD)])
def test_optimizer_types(name, cls):
    assert type(make_optimizer(FashionNet(**SMALL_ARCH), _cfg(optimizer=name))) is cls


def test_training_learns_and_history_is_complete():
    data = _data()
    model, record, val_probs = run_single(_cfg(), 0, data, "cpu", verbose=False)
    assert record["val_accuracy"] > 0.9
    assert val_probs.shape == (80, 10) and np.allclose(val_probs.sum(axis=1), 1, atol=1e-5)
    h = record["history"]
    assert set(h) == {"epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"}
    assert len(h["epoch"]) == record["epochs_run"] <= 3
    assert record["generalisation_gap"] == pytest.approx(record["train_accuracy_eval"] - record["val_accuracy"])
    assert not model.training and not record["diverged"]


def test_runs_are_reproducible_for_a_seed():
    data = _data()
    _, a, pa = run_single(_cfg(), 5, data, "cpu", verbose=False)
    _, b, pb = run_single(_cfg(), 5, data, "cpu", verbose=False)
    _, c, pc = run_single(_cfg(), 6, data, "cpu", verbose=False)
    assert a["history"]["train_loss"] == b["history"]["train_loss"] and np.array_equal(pa, pb)
    assert a["history"]["train_loss"] != c["history"]["train_loss"]


def test_early_stopping_restores_best_weights():
    data = _data()
    data["y_val"] = np.random.default_rng(0).integers(0, 2, len(data["y_val"]))    # unlearnable validation labels
    cfg = _cfg(max_epochs=12, patience=2, schedule="constant", warmup_epochs=0, lr=5e-3, aug="none")
    model = FashionNet(**SMALL_ARCH)
    model, hist, info = train_model(model, cfg, data["X_tr_t"], data["y_tr"], data["X_val_t"], data["y_val"],
                                    "cpu", seed=0, verbose=False)
    assert info["stopped_early"] and info["epochs_run"] < 12
    assert info["best_epoch"] == int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    probs = predict_proba(model, data["X_val_t"], "cpu")
    loss = -np.log(probs[np.arange(len(probs)), data["y_val"]]).mean()
    assert loss == pytest.approx(info["best_val_loss"], rel=1e-4)                # weights are the best ones


def test_label_validation_and_shapes():
    data = _data()
    with pytest.raises(ValueError):
        train_model(FashionNet(**SMALL_ARCH), _cfg(), data["X_tr_t"], np.eye(10)[data["y_tr"]], data["X_val_t"],
                    data["y_val"], "cpu", verbose=False)
    with pytest.raises(ValueError):
        train_model(FashionNet(**SMALL_ARCH), _cfg(), data["X_tr_t"][:, 0], data["y_tr"], data["X_val_t"],
                    data["y_val"], "cpu", verbose=False)


def test_checkpoint_round_trip(tmp_path):
    data = _data()
    path = tmp_path / "m.pt"
    model, _, val_probs = run_single(_cfg(), 1, data, "cpu", checkpoint_path=path, verbose=False)
    again = load_checkpoint(path)
    assert np.allclose(predict_proba(again, data["X_val_t"], "cpu"), val_probs, atol=1e-6)


def test_steps_per_epoch_uses_ceiling():
    assert math.ceil(54000 / 128) == 422
