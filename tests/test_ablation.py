"""Tests for src/ablation.py (owner: Tanish): the ablation design must be a true one-factor design."""
import pytest

from src import ablation
from src.dl_config import BASE_ARCH, P1_RECIPE, P2_RECIPE
from src.dl_models import build_model


def test_base_is_residual_fashionnet_with_phase2_recipe():
    base = ablation.base_config()
    assert base.name == "base" and base.arch == "fashionnet"
    assert base.arch_kwargs == BASE_ARCH
    for key, value in P2_RECIPE.items():
        assert getattr(base, key) == value


def test_names_are_unique_and_documented():
    cfgs = ablation.ablation_configs()
    names = [c.name for c in cfgs]
    assert len(set(names)) == len(names) == len(ablation.ABLATIONS)
    assert set(names) == set(ablation.ABLATIONS)
    assert {v[0] for v in ablation.ABLATIONS.values()} == {"architecture", "regularisation", "augmentation",
                                                           "optimisation"}


@pytest.mark.parametrize("cfg", ablation.ablation_configs(), ids=lambda c: c.name)
def test_each_ablation_changes_exactly_its_factor(cfg):
    changed = ablation.check_one_factor(cfg)
    assert changed                                     # an ablation identical to the base would be meaningless
    if cfg.name not in ablation.COUPLED_FIELDS:
        assert len(changed) == 1


def test_check_one_factor_rejects_two_changes():
    bad = ablation._variant("bad", {"dropout": 0.0}, label_smoothing=0.0)
    with pytest.raises(ValueError):
        ablation.check_one_factor(bad)


@pytest.mark.parametrize("cfg", ablation.all_configs(), ids=lambda c: c.name)
def test_every_configuration_builds(cfg):
    build_model(cfg.arch, **cfg.arch_kwargs)


def test_roadmap_separates_recipe_from_architecture():
    p1, p1_new, plain, base = ablation.roadmap_configs()
    assert p1.arch == p1_new.arch == "p1_cnn" and p1.arch_kwargs == p1_new.arch_kwargs
    for key, value in P1_RECIPE.items():
        assert getattr(p1, key) == value
    assert set(ablation.config_diff(p1_new, plain)) >= {"arch"}          # same recipe, new architecture
    assert all(k == "arch" or k.startswith("arch.") for k in ablation.config_diff(p1_new, plain))
    assert set(ablation.config_diff(plain, base)) == {"arch.residual"}   # the last step adds only the shortcuts
    assert set(ablation.ROADMAP_LABELS) == {c.name for c in ablation.roadmap_configs()}


def test_all_configs_has_no_duplicates_and_16_entries():
    names = [c.name for c in ablation.all_configs()]
    assert len(names) == len(set(names)) == 16
    assert names[:4] == [c.name for c in ablation.roadmap_configs()]


def test_with_budget_changes_only_the_budget():
    base = ablation.base_config()
    short = ablation.with_budget(base, max_epochs=2, patience=1)
    assert set(ablation.config_diff(base, short)) <= {"max_epochs", "patience", "warmup_epochs"}
    assert short.max_epochs == 2 and short.patience == 1 and short.warmup_epochs <= 1
    assert ablation.with_budget(base).to_dict() == base.to_dict()
