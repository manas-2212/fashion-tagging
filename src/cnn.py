"""Small CNN for Fashion-MNIST (Manas).

Contracts (see project spec):
  * raw images: np.uint8 (N,28,28), never modified in place
  * CNN input : float32 torch tensor (N,1,28,28) = (X/255 - mean)/std, mean/std from the 54k TRAIN split only
  * CNN output: (N,10) raw logits (no softmax inside the model)
  * labels    : int64 class indices, shape (N,) -> (B,) per batch (no one-hot)
The test set never appears in this file.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.config import NUM_CLASSES, SEED, set_global_seed


# --------------------------------------------------------------------------- data prep
def compute_mean_std(X_train_uint8: np.ndarray) -> tuple[float, float]:
    """Global pixel mean/std of X/255 over the TRAIN split only (float64 accumulation, returned as Python floats)."""
    x = np.asarray(X_train_uint8, dtype=np.float64) / 255.0  # float64 copy; the input is not modified
    return float(x.mean()), float(x.std())


def make_cnn_tensor(X_uint8: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """(N,28,28) uint8 -> float32 tensor (N,1,28,28) = (X/255 - mean)/std."""
    X = np.asarray(X_uint8)
    if X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError(f"expected (N,28,28) images, got {X.shape}")
    x = (X.astype(np.float32) / np.float32(255.0) - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(1)  # add channel dim


# --------------------------------------------------------------------------- model
class FashionCNN(nn.Module):
    """3 conv blocks (28->14->7->3) + 2-layer head. Returns raw logits."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14 -> 7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7 -> 3 (floor; last row/col dropped)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128*3*3 = 1152
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),        # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_cnn(dropout: float = 0.5) -> nn.Module:
    """Create FashionCNN. Call set_global_seed() BEFORE this for reproducible weight initialisation."""
    return FashionCNN(dropout=dropout)


# --------------------------------------------------------------------------- training
def _evaluate(model: nn.Module, X_t: torch.Tensor, y_t: torch.Tensor, device: torch.device,
              batch_size: int = 1024) -> tuple[float, float]:
    """Mean cross-entropy and accuracy in eval mode under no_grad (used for validation each epoch)."""
    model.eval()
    n = len(y_t)
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X_t[i:i + batch_size].to(device)
            yb = y_t[i:i + batch_size].to(device)
            out = model(xb)
            loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
            correct += (out.argmax(dim=1) == yb).sum().item()
    return loss_sum / n, correct / n


def _as_labels(y) -> torch.Tensor:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"labels must be class indices of shape (N,), got {y.shape} (no one-hot)")
    return torch.as_tensor(y.astype(np.int64, copy=False), dtype=torch.long)


def train_cnn(model, X_tr_t, y_tr, X_val_t, y_val, device, lr=1e-3, batch_size=128, max_epochs=30,
              patience=5, seed=SEED, checkpoint_path=None):
    """Train with Adam + CrossEntropyLoss; early stopping on VALIDATION LOSS; restores the best weights.

    Returns (model_with_best_weights [eval mode, on `device`], history_df, info).
    history_df columns: epoch, train_loss, train_acc, val_loss, val_acc.
      * train_loss/train_acc are RUNNING AVERAGES over the mini-batches of that epoch, computed in train mode
        (dropout active, weights changing during the epoch) -> they are pessimistic vs. a clean eval-mode pass.
      * val_loss/val_acc are computed after the epoch in eval mode under torch.no_grad.
    info: best_epoch (1-indexed), best_val_loss, best_val_acc (at best_epoch), train_seconds,
          plus extras: epochs_run, stopped_early, epoch_seconds (list).
    Reproducibility: re-seeds all RNGs from `seed` at the start (dropout) and shuffles with a seeded CPU torch.Generator.
    """
    device = torch.device(device)
    set_global_seed(seed)
    y_tr_t, y_val_t = _as_labels(y_tr), _as_labels(y_val)
    if X_tr_t.dtype != torch.float32 or X_tr_t.ndim != 4 or X_tr_t.shape[1:] != (1, 28, 28):
        raise ValueError(f"X_tr_t must be float32 (N,1,28,28), got {X_tr_t.dtype} {tuple(X_tr_t.shape)}")
    if len(X_tr_t) != len(y_tr_t) or len(X_val_t) != len(y_val_t):
        raise ValueError("images and labels differ in length")

    model.to(device)
    X_dev, y_dev = X_tr_t.to(device), y_tr_t.to(device)   # no-op on CPU; one-off copy on GPU
    Xv_dev, yv_dev = X_val_t.to(device), y_val_t.to(device)
    n = len(y_dev)
    gen = torch.Generator().manual_seed(seed)             # CPU generator for the shuffling order
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rows, epoch_secs = [], []
    best = dict(val_loss=float("inf"), val_acc=float("nan"), epoch=0, state=None)
    n_bad = 0
    t_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_dev[idx], y_dev[idx]               # yb: int64, shape (B,)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)          # weight by batch size (last batch may be smaller)
            correct += (out.detach().argmax(dim=1) == yb).sum()
        train_loss, train_acc = loss_sum.item() / n, correct.item() / n
        val_loss, val_acc = _evaluate(model, Xv_dev, yv_dev, device)
        epoch_secs.append(time.perf_counter() - t_ep)
        rows.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_secs[-1]:.1f}s")

        if val_loss < best["val_loss"]:                    # strict improvement
            best.update(val_loss=val_loss, val_acc=val_acc, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            n_bad = 0
        else:
            n_bad += 1
            if n_bad >= patience:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - t_start

    model.load_state_dict(best["state"])                   # restore best weights (copy back onto model's device)
    model.eval()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best["state"], best_epoch=best["epoch"], best_val_loss=best["val_loss"]),
                   checkpoint_path)

    history_df = pd.DataFrame(rows, columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    info = dict(best_epoch=best["epoch"], best_val_loss=best["val_loss"], best_val_acc=best["val_acc"],
                train_seconds=train_seconds, epochs_run=len(rows), stopped_early=len(rows) < max_epochs,
                epoch_seconds=epoch_secs)
    return model, history_df, info


# --------------------------------------------------------------------------- inference / plotting
def predict_cnn(model, X_t, device, batch_size: int = 512):
    """Batched eval-mode prediction. Returns (y_pred int64 (N,), probs float32 (N,10) softmax)."""
    device = torch.device(device)
    model.to(device).eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i:i + batch_size].to(device))
            p = torch.softmax(logits, dim=1)
            probs.append(p.float().cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros((0, NUM_CLASSES), dtype=np.float32)
    return np.concatenate(preds).astype(np.int64), np.concatenate(probs).astype(np.float32)


def plot_learning_curves(history_df: pd.DataFrame, best_epoch: int):
    """Two panels (loss, accuracy): train vs validation, best epoch marked. Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tr, va, ylabel, title) in zip(axes, [("train_loss", "val_loss", "Cross-entropy loss", "Loss"),
                                                   ("train_acc", "val_acc", "Accuracy", "Accuracy")]):
        ax.plot(history_df["epoch"], history_df[tr], marker="o", ms=3, label="train (running avg, dropout on)")
        ax.plot(history_df["epoch"], history_df[va], marker="s", ms=3, label="validation (eval mode)")
        ax.axvline(best_epoch, color="gray", ls="--", label=f"best epoch = {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig
