"""Numerical parity for Lucid's new AutoAugment-family functional ops.

Opt-in tier — auto-skips when the reference framework and its vision
package are not installed.

Covers five new functions in :mod:`lucid.utils.transforms.functional`:

* ``adjust_sharpness(img, factor)`` vs ``TF.adjust_sharpness``
* ``autocontrast(img)`` vs ``TF.autocontrast``
* ``posterize(img, num_bits)`` vs ``TF.posterize`` (uint8 round-trip)
* ``solarize(img, threshold)`` vs ``TF.solarize``
* ``invert(img)`` vs ``TF.invert``

Run with::

    pytest -m parity lucid/test/parity/transforms/test_strong_aug_functional_parity.py
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.utils.transforms.functional as F
from lucid.test._fixtures.ref_framework import require_ref, require_ref_vision
from lucid.test._helpers.compare import assert_close

_ref = require_ref(module_level=True)
TF = require_ref_vision(module_level=True).transforms.functional


# ── helpers ────────────────────────────────────────────────────────


def _make_float_image(
    seed: int = 0, c: int = 3, h: int = 24, w: int = 32
) -> tuple[lucid.Tensor, Any]:
    """Build a matched (lucid (C,H,W), reference (C,H,W)) float pair in [0,1]."""
    rng = np.random.default_rng(seed)
    arr = rng.random((c, h, w), dtype=np.float32)
    return lucid.tensor(arr.tolist()), _ref.from_numpy(arr.copy())


def _make_float_batch(
    seed: int = 0, b: int = 2, c: int = 3, h: int = 24, w: int = 32
) -> tuple[lucid.Tensor, Any]:
    """Build a matched (lucid (B,C,H,W), reference (B,C,H,W)) float pair in [0,1]."""
    rng = np.random.default_rng(seed)
    arr = rng.random((b, c, h, w), dtype=np.float32)
    return lucid.tensor(arr.tolist()), _ref.from_numpy(arr.copy())


def _interior(arr: np.ndarray, border: int = 1) -> np.ndarray:
    """Strip ``border`` pixels off the last two axes — used to skip the
    1-pixel halo where reference and Lucid diverge in ``adjust_sharpness``
    (Lucid: zero-padded conv2d at the edges; reference: passes border
    pixels through untouched, matching the PIL ``ImageFilter.SMOOTH`` spec).

    For ``factor != 1.0`` the two impls disagree on the border but should
    agree exactly on the interior.
    """
    return arr[..., border:-border, border:-border]


# ── adjust_sharpness ───────────────────────────────────────────────


@pytest.mark.parity
class TestAdjustSharpnessParity:
    """Lucid's ``adjust_sharpness`` matches reference on the interior; the
    1-pixel border differs because Lucid's depthwise conv zero-pads while
    PIL leaves border pixels untouched.  Interior comparison pins the
    blend math itself.
    """

    @pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_chw_interior(self, factor: float) -> None:
        lx, tx = _make_float_image(seed=10 + int(factor * 10))
        got = F.adjust_sharpness(lx, factor).numpy()
        ref = TF.adjust_sharpness(tx, factor).numpy()
        assert got.shape == ref.shape
        if factor == 1.0:
            # identity path — agree everywhere bit-exact
            assert_close(got, ref, atol=1e-6, rtol=1e-5)
        else:
            assert_close(_interior(got), _interior(ref), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("factor", [0.0, 0.5, 1.5, 2.0])
    def test_bchw_interior(self, factor: float) -> None:
        lx, tx = _make_float_batch(seed=42)
        got = F.adjust_sharpness(lx, factor).numpy()
        ref = TF.adjust_sharpness(tx, factor).numpy()
        assert got.shape == ref.shape
        assert_close(_interior(got), _interior(ref), atol=1e-5, rtol=1e-4)

    def test_identity_factor_is_exact(self) -> None:
        # factor == 1.0 short-circuits in Lucid — must be identity, including
        # at the border.
        lx, tx = _make_float_image(seed=7)
        got = F.adjust_sharpness(lx, 1.0).numpy()
        ref = TF.adjust_sharpness(tx, 1.0).numpy()
        assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_zero_image(self) -> None:
        arr = np.zeros((3, 24, 32), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.adjust_sharpness(lx, 1.5).numpy()
        ref = TF.adjust_sharpness(tx, 1.5).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_all_ones_image(self) -> None:
        arr = np.ones((3, 24, 32), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.adjust_sharpness(lx, 1.5).numpy()
        ref = TF.adjust_sharpness(tx, 1.5).numpy()
        # Interior of a flat image is the same flat constant — and clipped
        # to [0, 1] in both impls.
        assert_close(_interior(got), _interior(ref), atol=1e-5, rtol=1e-4)


# ── autocontrast ───────────────────────────────────────────────────


@pytest.mark.parity
class TestAutocontrastParity:
    """Per-channel min-max stretch.  Both impls share the same spec; should
    match to float epsilon."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_chw(self, seed: int) -> None:
        lx, tx = _make_float_image(seed=seed)
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)

    def test_bchw(self) -> None:
        lx, tx = _make_float_batch(seed=11)
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)

    def test_flat_channel_passthrough(self) -> None:
        # A channel where max == min must pass through unchanged in both.
        rng = np.random.default_rng(99)
        arr = rng.random((3, 16, 20), dtype=np.float32)
        arr[1, :, :] = 0.42  # flat green channel
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)
        # And the flat channel really is unchanged.
        assert_close(got[1], arr[1], atol=0.0, rtol=0.0)

    def test_zero_image(self) -> None:
        arr = np.zeros((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)

    def test_all_ones_image(self) -> None:
        arr = np.ones((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)

    def test_gradient_image(self) -> None:
        # Linear ramp 0 → 1 along width: stretch should leave it unchanged.
        ramp = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        arr = np.broadcast_to(ramp, (3, 24, 32)).copy()
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.autocontrast(lx).numpy()
        ref = TF.autocontrast(tx).numpy()
        assert_close(got, ref, atol=1e-5, rtol=1e-4)


# ── posterize ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestPosterizeParity:
    """uint8 bit-mask quantisation.  The reference vision library requires uint8
    input, so we convert float→uint8 for it and back to float for the
    comparison.

    Inherent ``±1/255`` quantisation noise lives in the float→uint8 step,
    not in either op, so we use ``atol=1/255`` to absorb it.
    """

    @pytest.mark.parametrize("num_bits", [1, 2, 4, 6, 8])
    def test_chw(self, num_bits: int) -> None:
        rng = np.random.default_rng(100 + num_bits)
        arr = rng.random((3, 24, 32), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx_u8 = _ref.from_numpy((arr * 255.0).round().clip(0, 255).astype(np.uint8))
        got = F.posterize(lx, num_bits).numpy()
        ref_u8 = TF.posterize(tx_u8, num_bits).numpy()
        ref = ref_u8.astype(np.float32) / 255.0
        # Both ops should agree to within the uint8 quantisation noise on
        # the input side (Lucid does the round internally, reference
        # consumes a pre-rounded uint8 — so they share the same rounded
        # int and the bit-mask is bit-exact thereafter).
        assert_close(got, ref, atol=1.0 / 255.0, rtol=0.0)

    @pytest.mark.parametrize("num_bits", [1, 2, 4, 6])
    def test_bchw(self, num_bits: int) -> None:
        rng = np.random.default_rng(200 + num_bits)
        arr = rng.random((2, 3, 24, 32), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx_u8 = _ref.from_numpy((arr * 255.0).round().clip(0, 255).astype(np.uint8))
        got = F.posterize(lx, num_bits).numpy()
        ref_u8 = TF.posterize(tx_u8, num_bits).numpy()
        ref = ref_u8.astype(np.float32) / 255.0
        assert_close(got, ref, atol=1.0 / 255.0, rtol=0.0)

    def test_num_bits_8_is_identity(self) -> None:
        # Lucid short-circuits at 8 bits; reference is also a no-op there.
        lx, _ = _make_float_image(seed=3)
        got = F.posterize(lx, 8).numpy()
        ref = lx.numpy()
        assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_zero_image(self) -> None:
        arr = np.zeros((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx_u8 = _ref.from_numpy((arr * 255.0).astype(np.uint8))
        got = F.posterize(lx, 4).numpy()
        ref = TF.posterize(tx_u8, 4).numpy().astype(np.float32) / 255.0
        assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_all_ones_image(self) -> None:
        arr = np.ones((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx_u8 = _ref.from_numpy((arr * 255.0).astype(np.uint8))
        got = F.posterize(lx, 4).numpy()
        ref = TF.posterize(tx_u8, 4).numpy().astype(np.float32) / 255.0
        assert_close(got, ref, atol=1.0 / 255.0, rtol=0.0)


# ── solarize ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestSolarizeParity:
    """Threshold-and-invert.  The reference vision library's float solarize uses
    the same ``>= threshold`` spec as Lucid; should be bit-exact."""

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_chw(self, threshold: float) -> None:
        lx, tx = _make_float_image(seed=300 + int(threshold * 100))
        got = F.solarize(lx, threshold).numpy()
        ref = TF.solarize(tx, threshold).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
    def test_bchw(self, threshold: float) -> None:
        lx, tx = _make_float_batch(seed=400 + int(threshold * 100))
        got = F.solarize(lx, threshold).numpy()
        ref = TF.solarize(tx, threshold).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_zero_image(self) -> None:
        arr = np.zeros((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.solarize(lx, 0.5).numpy()
        ref = TF.solarize(tx, 0.5).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_all_ones_image(self) -> None:
        # Every pixel >= 0.5: all get inverted to 0.
        arr = np.ones((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.solarize(lx, 0.5).numpy()
        ref = TF.solarize(tx, 0.5).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_gradient_image(self) -> None:
        ramp = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        arr = np.broadcast_to(ramp, (3, 24, 32)).copy()
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.solarize(lx, 0.5).numpy()
        ref = TF.solarize(tx, 0.5).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)


# ── invert ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestInvertParity:
    """``1 - img``.  Bit-exact in both impls."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_chw(self, seed: int) -> None:
        lx, tx = _make_float_image(seed=seed)
        got = F.invert(lx).numpy()
        ref = TF.invert(tx).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_bchw(self) -> None:
        lx, tx = _make_float_batch(seed=21)
        got = F.invert(lx).numpy()
        ref = TF.invert(tx).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_zero_image(self) -> None:
        arr = np.zeros((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.invert(lx).numpy()
        ref = TF.invert(tx).numpy()
        assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_all_ones_image(self) -> None:
        arr = np.ones((3, 8, 10), dtype=np.float32)
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.invert(lx).numpy()
        ref = TF.invert(tx).numpy()
        assert_close(got, ref, atol=0.0, rtol=0.0)

    def test_gradient_image(self) -> None:
        ramp = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        arr = np.broadcast_to(ramp, (3, 24, 32)).copy()
        lx = lucid.tensor(arr.tolist())
        tx = _ref.from_numpy(arr.copy())
        got = F.invert(lx).numpy()
        ref = TF.invert(tx).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)

    def test_double_invert_is_identity(self) -> None:
        lx, tx = _make_float_image(seed=5)
        got = F.invert(F.invert(lx)).numpy()
        ref = TF.invert(TF.invert(tx)).numpy()
        assert_close(got, ref, atol=1e-6, rtol=1e-5)
        # And really equals the original input.
        assert_close(got, lx.numpy(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
