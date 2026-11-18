"""Tests for src/dl_models.py (owner: Manas)."""
import pytest
import torch

from src.cnn import FashionCNN
from src.config import set_global_seed
from src.dl_config import BASE_ARCH
from src.dl_models import (ConvBlock, FashionNet, build_model, count_parameters, embed_features, embedding_dim,
                           layer_table, parameter_breakdown, receptive_field)


def test_base_output_and_embedding_shapes():
    model = FashionNet(**BASE_ARCH).eval()
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
    assert model.forward_features(x).shape == (4, 128)
    assert embedding_dim(model) == 128


def test_base_parameter_count_is_documented_value():
    assert count_parameters(FashionNet(**BASE_ARCH)) == 308_074
    assert count_parameters(FashionCNN()) == 241_546


@pytest.mark.parametrize("kwargs", [dict(residual=False), dict(use_bn=False), dict(blocks_per_stage=2),
                                    dict(widths=(16, 32, 64)), dict(dropout=0.0)])
def test_every_ablation_variant_runs(kwargs):
    model = FashionNet(**{**BASE_ARCH, **kwargs}).eval()
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_residual_flag_only_removes_shortcut_parameters():
    res, plain = FashionNet(**BASE_ARCH), FashionNet(**{**BASE_ARCH, "residual": False})
    shortcut = sum(p.numel() for n, p in res.named_parameters() if "shortcut" in n)
    assert shortcut > 0
    assert count_parameters(res) - count_parameters(plain) == shortcut


def test_no_bn_variant_has_no_batchnorm_and_uses_bias():
    model = FashionNet(**{**BASE_ARCH, "use_bn": False})
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert all(m.bias is not None for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert parameter_breakdown(model)["norm"] == 0


def test_identity_shortcut_when_channels_match():
    assert isinstance(ConvBlock(32, 32).shortcut, torch.nn.Identity)
    assert not isinstance(ConvBlock(32, 64).shortcut, torch.nn.Identity)
    assert ConvBlock(32, 64, residual=False).shortcut is None


def test_receptive_fields():
    assert receptive_field(FashionCNN()) == 22          # Phase 1 CNN does not see the whole 28x28 image
    assert receptive_field(FashionNet(**BASE_ARCH)) == 34   # base covers the whole image
    assert receptive_field(FashionNet(**{**BASE_ARCH, "residual": False})) == 34


def test_parameter_breakdown_sums_and_dense_share():
    p1 = parameter_breakdown(FashionCNN())
    assert p1["conv"] + p1["norm"] + p1["dense"] == p1["total"]
    assert p1["dense"] / p1["total"] > 0.6               # most Phase 1 weights sit in dense layers
    base = parameter_breakdown(FashionNet(**BASE_ARCH))
    assert base["dense"] == 128 * 10 + 10


def test_embed_features_for_phase1_cnn():
    model = FashionCNN().eval()
    x = torch.randn(3, 1, 28, 28)
    emb = embed_features(model, x)
    assert emb.shape == (3, 128) and embedding_dim(model) == 128
    assert torch.allclose(model.classifier[4:](emb), model(x), atol=1e-6)   # embedding feeds the last layer


def test_logits_equal_fc_of_embedding():
    model = FashionNet(**BASE_ARCH).eval()
    x = torch.randn(3, 1, 28, 28)
    assert torch.allclose(model.fc(model.forward_features(x)), model(x), atol=1e-6)


def test_seeded_initialisation_is_reproducible():
    set_global_seed(7)
    a = build_model("fashionnet", **BASE_ARCH)
    set_global_seed(7)
    b = build_model("fashionnet", **BASE_ARCH)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_build_model_rejects_unknown_and_bad_sizes():
    with pytest.raises(ValueError):
        build_model("resnet50")
    with pytest.raises(ValueError):
        FashionNet(widths=(8, 16, 32, 64))


def test_layer_table_matches_parameter_count():
    for model in (FashionNet(**BASE_ARCH), FashionCNN()):
        rows = layer_table(model)
        assert sum(r["params"] for r in rows) == count_parameters(model)
        assert rows[-1]["output_shape"] == "10"
