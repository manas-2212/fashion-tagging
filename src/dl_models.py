"""Phase 2 architectures (owner: Manas).

One configurable family, FashionNet, so that every architectural claim can be tested by switching ONE flag:

    stem  : 3x3 conv (1 -> w0) + BN + ReLU                                     28x28
    stage : `blocks_per_stage` ConvBlocks at constant resolution; MaxPool2d(2) between stages
            widths (32, 64, 128) -> resolutions 28 -> 14 -> 7
    head  : global average pooling -> Dropout -> Linear(w_last, 10)            logits

    ConvBlock = conv3x3 - BN - ReLU - conv3x3 - BN (+ shortcut) - ReLU
      residual=True  : output = ReLU(F(x) + shortcut(x))   (identity, or 1x1 conv + BN when channels change)
      residual=False : output = ReLU(F(x))                 ("plain" VGG-style network, same layers otherwise)
      use_bn=False   : BatchNorm replaced by identity and the conv layers get a bias

Why this shape (details in docs/phase2/architecture_notes.md):
  * the Phase 1 CNN puts 61% of its 241,546 weights in ONE dense layer (1152 -> 128) and its last feature map
    sees only 22x22 pixels; FashionNet replaces the dense layer by global average pooling, spends the weights on
    convolutions and reaches a receptive field that covers the whole 28x28 image;
  * two conv layers per resolution give the network more non-linear steps where the confusable classes
    (T-shirt / Pullover / Coat / Shirt) differ: collars, plackets and sleeve shape are small local patterns.

Contracts are the same as in Phase 1: input float32 (N,1,28,28) normalised with TRAIN mean/std, output raw logits.
"""
from __future__ import annotations

import torch
from torch import nn

from src.cnn import FashionCNN
from src.config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with optional BatchNorm and optional residual shortcut."""

    def __init__(self, c_in: int, c_out: int, use_bn: bool = True, residual: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(c_out) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(c_out) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.residual = residual
        if residual and c_in != c_out:      # projection shortcut: match the channel count with a 1x1 conv
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=not use_bn),
                                          nn.BatchNorm2d(c_out) if use_bn else nn.Identity())
        elif residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return self.act(out)


class FashionNet(nn.Module):
    """Configurable CNN for 28x28 grayscale images. Returns raw logits; forward_features returns the embedding."""

    def __init__(self, widths=(32, 64, 128), blocks_per_stage: int = 1, use_bn: bool = True,
                 residual: bool = True, dropout: float = 0.3):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if len(widths) < 1 or blocks_per_stage < 1:
            raise ValueError("need at least one stage and one block per stage")
        if len(widths) > 3:
            raise ValueError("at most 3 stages: 28 -> 14 -> 7 is the last even resolution")
        self.widths, self.blocks_per_stage = widths, int(blocks_per_stage)
        self.use_bn, self.residual = bool(use_bn), bool(residual)
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, padding=1, bias=not use_bn),
                                  nn.BatchNorm2d(widths[0]) if use_bn else nn.Identity(),
                                  nn.ReLU(inplace=True))
        layers, c_in = [], widths[0]
        for i, w in enumerate(widths):
            for _ in range(self.blocks_per_stage):
                layers.append(ConvBlock(c_in, w, use_bn=use_bn, residual=residual))
                c_in = w
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(widths[-1], NUM_CLASSES)
        self.embed_dim = widths[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(N,1,28,28) -> (N, embed_dim): globally pooled feature vector (the input of the linear classifier)."""
        return torch.flatten(self.pool(self.stages(self.stem(x))), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))


# ----------------------------------------------------------------------------- registry and helpers
ARCHITECTURES = ("p1_cnn", "fashionnet")


def build_model(arch: str, **kwargs) -> nn.Module:
    """Create a model by name. Call set_global_seed() BEFORE this for reproducible initial weights."""
    if arch == "p1_cnn":
        return FashionCNN(**kwargs)
    if arch == "fashionnet":
        return FashionNet(**kwargs)
    raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def embed_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate-layer features of either architecture, shape (N, 128) for the default sizes.

    FashionNet : global-average-pooled feature vector.
    FashionCNN : output of the hidden dense layer after ReLU (classifier[0:4] = Flatten, Dropout, Linear, ReLU).
    Call under model.eval() so that dropout is off and BatchNorm uses its running statistics.
    """
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    if isinstance(model, FashionCNN):
        return model.classifier[:4](model.features(x))
    raise TypeError(f"no embedding rule for {type(model).__name__}")


def embedding_dim(model: nn.Module) -> int:
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    if isinstance(model, FashionCNN):
        return int(model.classifier[2].out_features)
    raise TypeError(f"no embedding rule for {type(model).__name__}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: nn.Module) -> dict:
    """Trainable parameters split into convolution / normalisation / dense layers (for the architecture table)."""
    out = {"conv": 0, "norm": 0, "dense": 0}
    for m in model.modules():
        n = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        if isinstance(m, nn.Conv2d):
            out["conv"] += n
        elif isinstance(m, nn.BatchNorm2d):
            out["norm"] += n
        elif isinstance(m, nn.Linear):
            out["dense"] += n
    out["total"] = count_parameters(model)
    assert out["conv"] + out["norm"] + out["dense"] == out["total"]
    return out


def receptive_field(model: nn.Module) -> int:
    """Side length (pixels) of the input region seen by one unit of the LAST spatial feature map.

    Walks the layers in order: a k x k conv / pool with stride s grows the field by (k-1)*jump and multiplies
    the jump by s. 1x1 shortcut convolutions are skipped (they never enlarge the field of the main path).
    """
    rf, jump = 1, 1
    for name, m in model.named_modules():
        if "shortcut" in name:
            continue
        if isinstance(m, nn.Conv2d):
            k, s = m.kernel_size[0], m.stride[0]
        elif isinstance(m, nn.MaxPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
        else:
            continue
        rf += (k - 1) * jump
        jump *= s
    return rf


def layer_table(model: nn.Module) -> list[dict]:
    """One row per top-level computation step: name, output shape for a 28x28 input, trainable parameters."""
    rows = []
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 1, 28, 28)
    with torch.no_grad():
        if isinstance(model, FashionNet):
            steps = [("stem", model.stem)] + [(f"stages.{i} ({type(m).__name__})", m)
                                              for i, m in enumerate(model.stages)]
            steps += [("global average pool", nn.Sequential(model.pool, nn.Flatten())),
                      ("dropout", model.dropout), ("fc (logits)", model.fc)]
        else:
            steps = [(f"features.{i} ({type(m).__name__})", m) for i, m in enumerate(model.features)]
            steps += [(f"classifier.{i} ({type(m).__name__})", m) for i, m in enumerate(model.classifier)]
        for name, m in steps:
            x = m(x)
            rows.append({"layer": name, "output_shape": "x".join(str(d) for d in x.shape[1:]),
                         "params": sum(p.numel() for p in m.parameters() if p.requires_grad)})
    model.train(was_training)
    return rows
