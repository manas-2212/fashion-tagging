"""Batched data augmentation on normalised image tensors (owner: Akhil).

All operations work on a whole mini-batch that already lives on the training device (GPU or CPU), so there is
no per-image Python loop and no DataLoader worker. Augmentation is applied to TRAINING batches only; validation
and test images are never augmented.

Input / output: float32 tensor (B,1,28,28), normalised as (x/255 - mean)/std with the TRAIN mean/std.

Operations
  crop  : pad `pad` pixels on every side with the background value, then cut a random 28x28 window.
          This is a random shift of up to +-pad pixels; clothing items are roughly but not exactly centred.
  flip  : horizontal mirror with probability 0.5. Label-preserving for every class. Nearly a no-op for
          symmetric garments, and it creates right-facing footwear that the dataset does not contain,
          so its value is an empirical question (ablation `aug_no_flip`).
  erase : Random Erasing (Zhong et al., 2020): with probability p, overwrite a random rectangle with the
          dataset mean (0 after normalisation). Forces the network not to rely on one small region.

`pad_value` must be the normalised value of a black pixel, (0 - mean)/std, so that padding looks like background.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.config import SEED
from src.dl_config import AUG_PAD, ERASE_PROB, ERASE_RATIO, ERASE_SCALE, FLIP_PROB

AUG_POLICIES = {
    "none": (),
    "crop": ("crop",),
    "crop_flip": ("crop", "flip"),
    "crop_erase": ("crop", "erase"),
    "crop_flip_erase": ("crop", "flip", "erase"),
}


def background_value(mean: float, std: float) -> float:
    """Normalised value of a raw pixel equal to 0 (the Fashion-MNIST background)."""
    return (0.0 - float(mean)) / float(std)


def _check(x: torch.Tensor) -> None:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"expected a (B,1,H,W) batch, got {tuple(x.shape)}")


def random_crop(x: torch.Tensor, pad: int, pad_value: float, gen: torch.Generator) -> torch.Tensor:
    """Pad with `pad_value` and take one random HxW window per image (independent shifts in -pad..+pad)."""
    _check(x)
    if pad <= 0:
        return x
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), value=float(pad_value))
    dy = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device=x.device)
    rows = dy[:, None] + torch.arange(h, device=x.device)[None, :]            # (B,H)
    cols = dx[:, None] + torch.arange(w, device=x.device)[None, :]            # (B,W)
    batch = torch.arange(b, device=x.device)[:, None, None]
    return xp[batch, 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)      # (B,1,H,W)


def random_hflip(x: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """Mirror each image left-right with probability p."""
    _check(x)
    flip = torch.rand(x.shape[0], generator=gen, device=x.device) < p
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def random_erase(x: torch.Tensor, p: float, scale, ratio, gen: torch.Generator, value: float = 0.0) -> torch.Tensor:
    """With probability p per image, overwrite one random rectangle with `value` (0 = dataset mean)."""
    _check(x)
    b, _, h, w = x.shape
    dev = x.device
    apply = torch.rand(b, generator=gen, device=dev) < p
    area = (scale[0] + (scale[1] - scale[0]) * torch.rand(b, generator=gen, device=dev)) * h * w
    log_r = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * torch.rand(b, generator=gen, device=dev)
    r = torch.exp(log_r)                                                      # aspect ratio = height / width
    eh = torch.sqrt(area * r).round().clamp(1, h).long()
    ew = torch.sqrt(area / r).round().clamp(1, w).long()
    y0 = (torch.rand(b, generator=gen, device=dev) * (h - eh + 1).float()).long()
    x0 = (torch.rand(b, generator=gen, device=dev) * (w - ew + 1).float()).long()
    rr = torch.arange(h, device=dev)[None, :, None]
    cc = torch.arange(w, device=dev)[None, None, :]
    mask = ((rr >= y0[:, None, None]) & (rr < (y0 + eh)[:, None, None])
            & (cc >= x0[:, None, None]) & (cc < (x0 + ew)[:, None, None]) & apply[:, None, None])
    return torch.where(mask[:, None, :, :], torch.full_like(x, float(value)), x)


class BatchAugmenter:
    """Callable that applies one augmentation policy to a batch. Deterministic for a given (seed, call order)."""

    def __init__(self, policy: str, pad_value: float, device="cpu", seed: int = SEED, pad: int = AUG_PAD,
                 flip_prob: float = FLIP_PROB, erase_prob: float = ERASE_PROB, erase_scale=ERASE_SCALE,
                 erase_ratio=ERASE_RATIO):
        if policy not in AUG_POLICIES:
            raise ValueError(f"unknown augmentation policy {policy!r}; expected one of {sorted(AUG_POLICIES)}")
        self.policy, self.ops = policy, AUG_POLICIES[policy]
        self.pad_value, self.pad = float(pad_value), int(pad)
        self.flip_prob, self.erase_prob = float(flip_prob), float(erase_prob)
        self.erase_scale, self.erase_ratio = tuple(erase_scale), tuple(erase_ratio)
        self.gen = torch.Generator(device=torch.device(device)).manual_seed(int(seed))

    @property
    def is_identity(self) -> bool:
        return len(self.ops) == 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            if op == "crop":
                x = random_crop(x, self.pad, self.pad_value, self.gen)
            elif op == "flip":
                x = random_hflip(x, self.flip_prob, self.gen)
            elif op == "erase":
                x = random_erase(x, self.erase_prob, self.erase_scale, self.erase_ratio, self.gen)
        return x


def plot_augmentation_examples(X_uint8, mean: float, std: float, indices=(0, 1, 2, 3, 4, 5), n_variants: int = 5,
                               seed: int = SEED):
    """Figure: each row is one training image followed by `n_variants` augmented versions (full policy)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.cnn import make_cnn_tensor

    x = make_cnn_tensor(np.asarray(X_uint8)[list(indices)], mean, std)
    aug = BatchAugmenter("crop_flip_erase", background_value(mean, std), seed=seed, erase_prob=0.5)
    fig, axes = plt.subplots(len(indices), n_variants + 1, figsize=(1.5 * (n_variants + 1), 1.5 * len(indices)))
    cols = [x] + [aug(x) for _ in range(n_variants)]
    for j, batch in enumerate(cols):
        img = (batch * std + mean).clamp(0, 1).numpy()
        for i in range(len(indices)):
            ax = axes[i, j]
            ax.imshow(img[i, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("original" if j == 0 else f"augmented {j}", fontsize=9)
    fig.suptitle("Training-time augmentation: random shift, flip, random erasing", fontsize=11)
    fig.tight_layout()
    return fig
