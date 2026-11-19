"""Ablation study design (owner: Tanish).

The Phase 2 base model ("base") is fixed before any run: FashionNet with residual blocks trained with the
Phase 2 recipe. Every ablation changes exactly ONE factor of that base and is trained with the same seeds, so a
difference in validation macro-F1 can be attributed to that factor (tests/test_ablation.py enforces this).

Two kinds of configuration are defined here:
  1. the architecture roadmap: Phase 1 CNN + Phase 1 recipe -> Phase 1 CNN + Phase 2 recipe -> plain BN network
     -> residual network (= base). It separates "better training recipe" from "better architecture".
  2. one-factor ablations of the base, in four groups: architecture, regularisation, augmentation, optimisation.

Nothing in this file trains anything; it only declares configurations and compares them.
"""
from __future__ import annotations

from src.dl_config import BASE_ARCH, P1_ARCH, P1_RECIPE, P2_RECIPE, SGD_LR, SGD_WEIGHT_DECAY
from src.dl_train import TrainConfig

BASE_NAME = "base"
P1_NAME = "p1_cnn_p1_recipe"


def base_config(**recipe_overrides) -> TrainConfig:
    """The Phase 2 base model: residual FashionNet + Phase 2 recipe."""
    recipe = dict(P2_RECIPE)
    recipe.update(recipe_overrides)
    return TrainConfig(name=BASE_NAME, arch="fashionnet", arch_kwargs=dict(BASE_ARCH), **recipe)


def _variant(name: str, arch_changes: dict | None = None, **recipe_changes) -> TrainConfig:
    """Copy of the base with the given changes. Keeps max_epochs/patience overrides out of the factor count."""
    arch = dict(BASE_ARCH)
    arch.update(arch_changes or {})
    recipe = dict(P2_RECIPE)
    recipe.update(recipe_changes)
    return TrainConfig(name=name, arch="fashionnet", arch_kwargs=arch, **recipe)


# name -> (group, factor, what changed relative to the base, question the ablation answers)
ABLATIONS = {
    "arch_plain": ("architecture", "residual shortcut", "residual=False",
                   "Do the shortcut connections help at this depth?"),
    "arch_no_bn": ("architecture", "batch normalisation", "use_bn=False",
                   "How much does BatchNorm contribute to optimisation and accuracy?"),
    "arch_deeper": ("architecture", "depth", "blocks_per_stage=2 (13 conv layers instead of 7)",
                    "Is the base network too shallow for this dataset?"),
    "arch_narrow": ("architecture", "width", "widths=(16, 32, 64) (about 4x fewer weights)",
                    "How much capacity does the task actually need?"),
    "reg_dropout0": ("regularisation", "dropout", "dropout=0.0", "Is dropout before the classifier needed?"),
    "reg_dropout05": ("regularisation", "dropout", "dropout=0.5", "Does stronger dropout help or under-fit?"),
    "reg_no_wd": ("regularisation", "weight decay", "weight_decay=0.0", "Does decoupled weight decay matter?"),
    "reg_no_ls": ("regularisation", "label smoothing", "label_smoothing=0.0",
                  "Does label smoothing help accuracy, and what does it do to calibration?"),
    "aug_none": ("augmentation", "all augmentation", "aug='none'",
                 "How much of the gain comes from data augmentation as a whole?"),
    "aug_no_flip": ("augmentation", "horizontal flip", "aug='crop_erase'",
                    "Is flipping useful when garments are symmetric and footwear always faces one way?"),
    "aug_no_erase": ("augmentation", "random erasing", "aug='crop_flip'",
                     "Does occluding a random patch regularise the network?"),
    "opt_sgd": ("optimisation", "optimiser", f"SGD + Nesterov momentum 0.9, lr={SGD_LR}, wd={SGD_WEIGHT_DECAY}",
                "Is the adaptive optimiser responsible for the result?"),
    "opt_constant_lr": ("optimisation", "learning-rate schedule", "schedule='constant', warmup_epochs=0",
                        "Does cosine decay with warm-up beat a constant learning rate?"),
}


def ablation_configs() -> list[TrainConfig]:
    """All one-factor ablations of the base, in the order of ABLATIONS."""
    cfgs = [
        _variant("arch_plain", {"residual": False}),
        _variant("arch_no_bn", {"use_bn": False}),
        _variant("arch_deeper", {"blocks_per_stage": 2}),
        _variant("arch_narrow", {"widths": (16, 32, 64)}),
        _variant("reg_dropout0", {"dropout": 0.0}),
        _variant("reg_dropout05", {"dropout": 0.5}),
        _variant("reg_no_wd", weight_decay=0.0),
        _variant("reg_no_ls", label_smoothing=0.0),
        _variant("aug_none", aug="none"),
        _variant("aug_no_flip", aug="crop_erase"),
        _variant("aug_no_erase", aug="crop_flip"),
        _variant("opt_sgd", optimizer="sgd", lr=SGD_LR, weight_decay=SGD_WEIGHT_DECAY),
        _variant("opt_constant_lr", schedule="constant", warmup_epochs=0),
    ]
    assert [c.name for c in cfgs] == list(ABLATIONS), "ABLATIONS and ablation_configs() are out of sync"
    return cfgs


def roadmap_configs() -> list[TrainConfig]:
    """Architecture roadmap, in order. The last two entries are shared with the ablation study."""
    return [
        TrainConfig(name=P1_NAME, arch="p1_cnn", arch_kwargs=dict(P1_ARCH), **P1_RECIPE),
        TrainConfig(name="p1_cnn_p2_recipe", arch="p1_cnn", arch_kwargs=dict(P1_ARCH), **P2_RECIPE),
        _variant("arch_plain", {"residual": False}),
        base_config(),
    ]


ROADMAP_LABELS = {
    P1_NAME: "Phase 1 CNN\nPhase 1 recipe",
    "p1_cnn_p2_recipe": "Phase 1 CNN\nPhase 2 recipe",
    "arch_plain": "FashionNet plain\n(BN, no shortcut)",
    BASE_NAME: "FashionNet residual\n(base)",
}


def all_configs() -> list[TrainConfig]:
    """Every configuration trained in Phase 2, each name once: roadmap first, then the remaining ablations."""
    seen, out = set(), []
    for cfg in roadmap_configs() + ablation_configs():
        if cfg.name not in seen:
            seen.add(cfg.name)
            out.append(cfg)
    return out


def config_diff(a: TrainConfig, b: TrainConfig) -> dict:
    """Fields in which two configurations differ: {field: (value_in_a, value_in_b)}. `name` is ignored;
    architecture keyword arguments are compared key by key as 'arch.<key>'."""
    da, db = a.to_dict(), b.to_dict()
    diff = {}
    for key in da:
        if key in ("name", "arch_kwargs"):
            continue
        if da[key] != db[key]:
            diff[key] = (da[key], db[key])
    for key in sorted(set(da["arch_kwargs"]) | set(db["arch_kwargs"])):
        va, vb = da["arch_kwargs"].get(key), db["arch_kwargs"].get(key)
        if va != vb:
            diff[f"arch.{key}"] = (va, vb)
    return diff


# Factors that legitimately move together: the SGD ablation needs its own learning rate and weight decay scale
# (a learning rate of 2e-3 is far too small for SGD), and a constant schedule has no warm-up.
COUPLED_FIELDS = {
    "opt_sgd": {"optimizer", "lr", "weight_decay"},
    "opt_constant_lr": {"schedule", "warmup_epochs"},
}


def check_one_factor(cfg: TrainConfig, base: TrainConfig | None = None) -> set:
    """Return the set of changed fields and raise if an ablation changes more than its declared factor."""
    base = base or base_config()
    changed = set(config_diff(base, cfg))
    allowed = COUPLED_FIELDS.get(cfg.name)
    if allowed is not None:
        if not changed or not changed <= allowed:
            raise ValueError(f"{cfg.name}: changed {sorted(changed)}, allowed {sorted(allowed)}")
    elif len(changed) != 1:
        raise ValueError(f"{cfg.name}: an ablation must change exactly one field, changed {sorted(changed)}")
    return changed


def with_budget(cfg: TrainConfig, max_epochs: int | None = None, patience: int | None = None) -> TrainConfig:
    """Same configuration with a different epoch budget (used by the smoke test and the --max-epochs option)."""
    d = cfg.to_dict()
    d["arch_kwargs"] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d["arch_kwargs"].items()}
    if max_epochs is not None:
        d["max_epochs"] = int(max_epochs)
        d["warmup_epochs"] = min(d["warmup_epochs"], max(0, int(max_epochs) - 1))
    if patience is not None:
        d["patience"] = int(patience)
    return TrainConfig(**d)
