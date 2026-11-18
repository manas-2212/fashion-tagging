"""Phase 2 training engine (owner: Manas).

One function trains every Phase 2 configuration, so that two runs differ ONLY in their TrainConfig:
  * optimiser: Adam / AdamW (decoupled weight decay) / SGD with Nesterov momentum
  * learning-rate schedule: constant, or linear warm-up followed by cosine decay to zero (updated every step)
  * regularisation: label smoothing, weight decay (never applied to biases and BatchNorm parameters)
  * augmentation policy from src/augment.py, applied to training batches only
  * early stopping on VALIDATION cross-entropy (no label smoothing in the validation loss), best weights restored
  * automatic mixed precision on CUDA (float16 forward pass, float32 weights) for speed

Protocol: this file never sees the test set. Model selection uses the validation split only.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.augment import AUG_POLICIES, BatchAugmenter
from src.config import NUM_CLASSES, SEED, set_global_seed
from src.dl_models import build_model, count_parameters
from src.evaluation import evaluate_predictions

OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULES = ("constant", "cosine")


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines one training run except the seed."""
    name: str
    arch: str = "fashionnet"
    arch_kwargs: dict = field(default_factory=dict)
    optimizer: str = "adamw"
    lr: float = 2e-3
    weight_decay: float = 1e-2
    schedule: str = "cosine"
    warmup_epochs: int = 1
    max_epochs: int = 30
    patience: int = 8
    batch_size: int = 128
    label_smoothing: float = 0.1
    aug: str = "crop_flip_erase"

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.aug not in AUG_POLICIES:
            raise ValueError(f"aug must be one of {sorted(AUG_POLICIES)}, got {self.aug!r}")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("max_epochs, patience and batch_size must be positive")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arch_kwargs"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.arch_kwargs.items()}
        return d


# ----------------------------------------------------------------------------- optimiser and schedule
def split_decay_groups(model: nn.Module):
    """(decay, no_decay) parameter lists. Biases and BatchNorm scale/shift (all 1-D tensors) get no weight decay:
    shrinking them does not reduce model complexity and it interferes with normalisation."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return decay, no_decay


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = split_decay_groups(model)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, nesterov=True)


def lr_at_step(step: int, cfg: TrainConfig, steps_per_epoch: int) -> float:
    """Learning rate for optimisation step `step` (0-indexed).

    constant: cfg.lr. cosine: linear warm-up from lr/warmup_steps to lr over the warm-up epochs,
    then lr * 0.5 * (1 + cos(pi * progress)) down to 0 at the end of max_epochs.
    """
    if cfg.schedule == "constant":
        return cfg.lr
    warm = cfg.warmup_epochs * steps_per_epoch
    total = cfg.max_epochs * steps_per_epoch
    if step < warm:
        return cfg.lr * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------- evaluation helpers
def predict_logits(model: nn.Module, X_t: torch.Tensor, device, batch_size: int = 1024) -> torch.Tensor:
    """Eval-mode logits as a float32 CPU tensor (N,10)."""
    device = torch.device(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out.append(model(X_t[i:i + batch_size].to(device)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, NUM_CLASSES)


def predict_proba(model: nn.Module, X_t: torch.Tensor, device, batch_size: int = 1024) -> np.ndarray:
    """Eval-mode softmax probabilities, float32 (N,10)."""
    return torch.softmax(predict_logits(model, X_t, device, batch_size), dim=1).numpy().astype(np.float32)


def _as_labels(y) -> torch.Tensor:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"labels must be class indices of shape (N,), got {y.shape}")
    return torch.as_tensor(y.astype(np.int64, copy=False), dtype=torch.long)


# ----------------------------------------------------------------------------- training
def train_model(model: nn.Module, cfg: TrainConfig, X_tr_t, y_tr, X_val_t, y_val, device, seed: int = SEED,
                pad_value: float = 0.0, verbose: bool = True):
    """Train `model` under `cfg`. Returns (model with best weights [eval mode], history_df, info).

    history_df columns: epoch, lr, train_loss, train_acc, val_loss, val_acc
      train_* are running averages over the epoch in train mode (dropout and augmentation ON, label smoothing in
      the loss), so they are pessimistic; val_* are computed after the epoch in eval mode with plain cross-entropy.
    info: best_epoch, best_val_loss, best_val_acc, epochs_run, stopped_early, diverged, train_seconds, amp.
    """
    device = torch.device(device)
    set_global_seed(seed)
    y_tr_t, y_val_t = _as_labels(y_tr), _as_labels(y_val)
    if X_tr_t.dtype != torch.float32 or X_tr_t.ndim != 4 or X_tr_t.shape[1:] != (1, 28, 28):
        raise ValueError(f"X_tr_t must be float32 (N,1,28,28), got {X_tr_t.dtype} {tuple(X_tr_t.shape)}")
    if len(X_tr_t) != len(y_tr_t) or len(X_val_t) != len(y_val_t):
        raise ValueError("images and labels differ in length")

    model.to(device)
    X_dev, y_dev = X_tr_t.to(device), y_tr_t.to(device)
    Xv_dev = X_val_t.to(device)
    n = len(y_dev)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    shuffle_gen = torch.Generator().manual_seed(seed)
    augment = BatchAugmenter(cfg.aug, pad_value, device=device, seed=seed)
    opt = make_optimizer(model, cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rows = []
    best = dict(val_loss=float("inf"), val_acc=float("nan"), epoch=0, state=None)
    n_bad, step, diverged = 0, 0, False
    t_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        perm = torch.randperm(n, generator=shuffle_gen).to(device)
        lr = cfg.lr
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_dev[idx], y_dev[idx]
            if not augment.is_identity:
                xb = augment(xb)
            lr = lr_at_step(step, cfg, steps_per_epoch)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = F.cross_entropy(out.float(), yb, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            loss_sum += loss.detach() * len(idx)
            correct += (out.detach().argmax(dim=1) == yb).sum()
        train_loss, train_acc = loss_sum.item() / n, correct.item() / n

        logits = predict_logits(model, Xv_dev, device)
        val_loss = F.cross_entropy(logits, y_val_t).item()
        val_acc = (logits.argmax(dim=1) == y_val_t).float().mean().item()
        rows.append(dict(epoch=epoch, lr=lr, train_loss=train_loss, train_acc=train_acc,
                         val_loss=val_loss, val_acc=val_acc))
        if verbose:
            print(f"  epoch {epoch:3d} | lr {lr:.5f} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.perf_counter() - t_ep:.1f}s")

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            diverged = True                                 # recorded, never hidden
            break
        if val_loss < best["val_loss"]:
            best.update(val_loss=val_loss, val_acc=val_acc, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            n_bad = 0
        else:
            n_bad += 1
            if n_bad >= cfg.patience:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - t_start
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    history = pd.DataFrame(rows, columns=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
    info = dict(best_epoch=best["epoch"], best_val_loss=best["val_loss"], best_val_acc=best["val_acc"],
                epochs_run=len(rows), stopped_early=len(rows) < cfg.max_epochs and not diverged,
                diverged=diverged, train_seconds=train_seconds, amp=use_amp)
    return model, history, info


def run_single(cfg: TrainConfig, seed: int, data: dict, device, checkpoint_path=None, verbose: bool = True):
    """Seed -> build -> train -> evaluate on train (eval mode) and validation. Returns (model, record, val_probs).

    data: dict with X_tr_t, y_tr, X_val_t, y_val, pad_value (see phase2_pipeline.load_phase2_data).
    record is JSON-serialisable: config, seed, validation metrics, generalisation gap, history, timing.
    """
    set_global_seed(seed)                                   # BEFORE building: fixes the initial weights
    model = build_model(cfg.arch, **cfg.arch_kwargs)
    n_params = count_parameters(model)
    if verbose:
        print(f"[{cfg.name} | seed {seed}] {cfg.arch} {cfg.arch_kwargs} | {n_params:,} parameters")
    model, history, info = train_model(model, cfg, data["X_tr_t"], data["y_tr"], data["X_val_t"], data["y_val"],
                                       device, seed=seed, pad_value=data["pad_value"], verbose=verbose)
    val_probs = predict_proba(model, data["X_val_t"], device)
    val = evaluate_predictions(data["y_val"], val_probs.argmax(axis=1), cfg.name)
    train_pred = predict_logits(model, data["X_tr_t"], device).argmax(dim=1).numpy()
    train_acc_eval = float((train_pred == np.asarray(data["y_tr"])).mean())
    record = dict(
        name=cfg.name, seed=int(seed), config=cfg.to_dict(), n_params=int(n_params),
        n_train=int(len(data["y_tr"])), n_val=int(len(data["y_val"])),
        val_accuracy=val["accuracy"], val_macro_f1=val["macro_f1"],
        val_per_class_recall=[float(r) for r in val["per_class_recall"]],
        val_loss=float(info["best_val_loss"]), train_accuracy_eval=train_acc_eval,
        generalisation_gap=train_acc_eval - val["accuracy"],
        best_epoch=int(info["best_epoch"]), epochs_run=int(info["epochs_run"]),
        stopped_early=bool(info["stopped_early"]), diverged=bool(info["diverged"]),
        train_seconds=float(info["train_seconds"]), amp=bool(info["amp"]), device=str(device),
        history=history.to_dict(orient="list"),
    )
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": cfg.to_dict(), "seed": int(seed)}, checkpoint_path)
    return model, record, val_probs


def load_checkpoint(path, device="cpu") -> nn.Module:
    """Rebuild a model from a checkpoint written by run_single (eval mode, on `device`)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["arch_kwargs"].items()}
    model = build_model(cfg["arch"], **kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(torch.device(device)).eval()
