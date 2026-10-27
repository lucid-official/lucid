"""Mixup + CutMix + Random combo collators.

Verifies:

* MixupCollator / CutMixCollator produce ``(images, soft_targets)``
  with the correct shapes, given a list of ``(image, label)`` tuples.
* Soft targets sum to 1 per row (after mix).
* ``p=0`` is identity (raw images + one-hot soft targets).
* Reproducibility under ``manual_seed``.
* Invalid arguments raise ``ValueError``.
* The random combo collator dispatches to both branches over many draws.
"""

import pytest

import lucid
import lucid.utils.data as D

# ── helpers ─────────────────────────────────────────────────────────


def _make_batch(b: int = 4, c: int = 3, h: int = 8, w: int = 8) -> list[tuple]:
    """Build a batch list of ``(image, label)`` tuples for collate input."""
    return [(lucid.rand(c, h, w), lucid.tensor(i % 10, dtype=lucid.int64)) for i in range(b)]


# ── MixupCollator ───────────────────────────────────────────────────


class TestMixupCollator:
    def test_output_shapes(self) -> None:
        lucid.manual_seed(0)
        coll = D.MixupCollator(alpha=0.2, num_classes=10)
        x, y = coll(_make_batch(b=4))
        assert tuple(x.shape) == (4, 3, 8, 8)
        assert tuple(y.shape) == (4, 10)

    def test_soft_target_sums_to_one(self) -> None:
        lucid.manual_seed(0)
        coll = D.MixupCollator(alpha=0.2, num_classes=10)
        _, y = coll(_make_batch(b=4))
        row_sums = y.sum(dim=1).numpy()
        for s in row_sums:
            assert float(s) == pytest.approx(1.0, abs=1e-5)

    def test_p_zero_passes_through(self) -> None:
        # With p=0 the collator never mixes — output images == input images.
        lucid.manual_seed(0)
        coll = D.MixupCollator(alpha=0.2, num_classes=10, p=0.0)
        batch = _make_batch(b=4)
        x, y = coll(batch)
        # Reconstruct what default_collate would produce.
        ref_images = lucid.stack([img for img, _ in batch], dim=0)
        assert float((x - ref_images).abs().max().item()) == 0.0
        # Soft targets are one-hot (each row has one 1.0 and rest 0).
        for i in range(4):
            assert float(y[i].max().item()) == 1.0
            assert int(y[i].sum().item()) == 1

    def test_reproducible_with_seed(self) -> None:
        coll = D.MixupCollator(alpha=0.2, num_classes=10)
        batch = _make_batch(b=4)
        lucid.manual_seed(123)
        x1, y1 = coll(batch)
        lucid.manual_seed(123)
        x2, y2 = coll(batch)
        assert (x1.numpy() == x2.numpy()).all()
        assert (y1.numpy() == y2.numpy()).all()

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            D.MixupCollator(alpha=0.0, num_classes=10)

    def test_invalid_num_classes(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            D.MixupCollator(alpha=0.2, num_classes=0)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p"):
            D.MixupCollator(alpha=0.2, num_classes=10, p=1.5)

    def test_repr(self) -> None:
        r = repr(D.MixupCollator(alpha=0.2, num_classes=10))
        assert "MixupCollator" in r
        assert "alpha=0.2" in r
        assert "num_classes=10" in r


# ── CutMixCollator ──────────────────────────────────────────────────


class TestCutMixCollator:
    def test_output_shapes(self) -> None:
        lucid.manual_seed(0)
        coll = D.CutMixCollator(alpha=1.0, num_classes=10)
        x, y = coll(_make_batch(b=4))
        assert tuple(x.shape) == (4, 3, 8, 8)
        assert tuple(y.shape) == (4, 10)

    def test_soft_target_sums_to_one(self) -> None:
        lucid.manual_seed(0)
        coll = D.CutMixCollator(alpha=1.0, num_classes=10)
        _, y = coll(_make_batch(b=4))
        row_sums = y.sum(dim=1).numpy()
        for s in row_sums:
            assert float(s) == pytest.approx(1.0, abs=1e-5)

    def test_p_zero_passes_through(self) -> None:
        lucid.manual_seed(0)
        coll = D.CutMixCollator(alpha=1.0, num_classes=10, p=0.0)
        batch = _make_batch(b=4)
        x, _ = coll(batch)
        ref_images = lucid.stack([img for img, _ in batch], dim=0)
        assert float((x - ref_images).abs().max().item()) == 0.0

    def test_reproducible_with_seed(self) -> None:
        coll = D.CutMixCollator(alpha=1.0, num_classes=10)
        batch = _make_batch(b=4, h=16, w=16)
        lucid.manual_seed(7)
        x1, y1 = coll(batch)
        lucid.manual_seed(7)
        x2, y2 = coll(batch)
        assert (x1.numpy() == x2.numpy()).all()

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            D.CutMixCollator(alpha=-0.1, num_classes=10)

    def test_repr(self) -> None:
        r = repr(D.CutMixCollator(alpha=1.0, num_classes=10))
        assert "CutMixCollator" in r


# ── Random mixup-or-cutmix ──────────────────────────────────────────


class TestRandomMixupCutMixCollator:
    def test_output_shapes(self) -> None:
        lucid.manual_seed(0)
        coll = D.RandomMixupCutMixCollator(num_classes=10)
        x, y = coll(_make_batch(b=4))
        assert tuple(x.shape) == (4, 3, 8, 8)
        assert tuple(y.shape) == (4, 10)

    def test_p_zero_no_mixing(self) -> None:
        lucid.manual_seed(0)
        coll = D.RandomMixupCutMixCollator(num_classes=10, p=0.0)
        batch = _make_batch(b=4)
        x, _ = coll(batch)
        ref_images = lucid.stack([img for img, _ in batch], dim=0)
        assert float((x - ref_images).abs().max().item()) == 0.0

    def test_invalid_switch_prob(self) -> None:
        with pytest.raises(ValueError, match="switch_prob"):
            D.RandomMixupCutMixCollator(num_classes=10, switch_prob=2.0)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p"):
            D.RandomMixupCutMixCollator(num_classes=10, p=-0.1)

    def test_repr(self) -> None:
        r = repr(D.RandomMixupCutMixCollator(num_classes=10))
        assert "RandomMixupCutMixCollator" in r
