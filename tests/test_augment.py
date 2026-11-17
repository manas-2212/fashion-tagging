"""Tests for src/augment.py (owner: Akhil)."""
import pytest
import torch

from src.augment import (AUG_POLICIES, BatchAugmenter, background_value, random_crop, random_erase, random_hflip)

PAD_VALUE = background_value(0.2860, 0.3530)


def _batch(n=64, seed=0):
    return torch.randn(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_background_value_is_normalised_zero_pixel():
    assert PAD_VALUE == pytest.approx(-0.2860 / 0.3530)


@pytest.mark.parametrize("policy", sorted(AUG_POLICIES))
def test_shape_dtype_preserved_and_input_untouched(policy):
    x = _batch()
    before = x.clone()
    out = BatchAugmenter(policy, PAD_VALUE, seed=1)(x)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert torch.equal(x, before)


def test_none_policy_is_identity():
    aug = BatchAugmenter("none", PAD_VALUE)
    x = _batch()
    assert aug.is_identity and torch.equal(aug(x), x)


def test_same_seed_same_result_different_seed_different_result():
    x = _batch()
    a = BatchAugmenter("crop_flip_erase", PAD_VALUE, seed=3)(x)
    b = BatchAugmenter("crop_flip_erase", PAD_VALUE, seed=3)(x)
    c = BatchAugmenter("crop_flip_erase", PAD_VALUE, seed=4)(x)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_crop_is_a_shift_of_at_most_pad_pixels():
    x = _batch(32)
    out = random_crop(x, 2, PAD_VALUE, _gen(5))
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2), value=PAD_VALUE)
    for i in range(len(x)):
        windows = [padded[i, 0, dy:dy + 28, dx:dx + 28] for dy in range(5) for dx in range(5)]
        assert any(torch.equal(out[i, 0], w) for w in windows)


def test_crop_with_zero_pad_returns_input():
    x = _batch(4)
    assert torch.equal(random_crop(x, 0, PAD_VALUE, _gen()), x)


def test_flip_gives_original_or_mirror_and_respects_probability():
    x = _batch(400)
    out = random_hflip(x, 0.5, _gen(2))
    same = (out == x).flatten(1).all(dim=1)
    mirrored = (out == x.flip(3)).flatten(1).all(dim=1)
    assert bool((same | mirrored).all())
    assert 0.35 < mirrored.float().mean().item() < 0.65
    assert torch.equal(random_hflip(x, 0.0, _gen()), x)
    assert torch.equal(random_hflip(x, 1.0, _gen()), x.flip(3))


def test_erase_changes_only_a_bounded_rectangle():
    x = _batch(200) + 10.0                                # no pixel equals the fill value 0
    out = random_erase(x, 1.0, (0.02, 0.2), (0.3, 3.3), _gen(1))
    erased = (out == 0.0)[:, 0]
    frac = erased.flatten(1).float().mean(dim=1)
    assert bool((frac > 0).all()) and frac.max().item() <= 0.30
    assert torch.equal(out[~(out == 0.0)], x[~(out == 0.0)])        # everything else is untouched
    for m in erased[:20]:                                 # erased area is one full rectangle
        rows, cols = m.any(dim=1).nonzero().flatten(), m.any(dim=0).nonzero().flatten()
        assert bool(m[rows.min():rows.max() + 1, cols.min():cols.max() + 1].all())


def test_erase_probability_zero_is_identity():
    x = _batch()
    assert torch.equal(random_erase(x, 0.0, (0.02, 0.2), (0.3, 3.3), _gen()), x)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        BatchAugmenter("mixup", PAD_VALUE)
    with pytest.raises(ValueError):
        random_hflip(torch.zeros(4, 28, 28), 0.5, _gen())
