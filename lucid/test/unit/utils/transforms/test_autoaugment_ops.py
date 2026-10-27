"""Shared building blocks for AutoAugment / RandAugment / TrivialAugment.

Verifies:

* the four new ``functional`` ops behave correctly
  (``adjust_sharpness``, ``autocontrast``, ``posterize``, ``solarize``)
* the magnitude lookup table covers every op in the vocabulary
* ``apply_op`` dispatches all 14 ops without raising and preserves shape
* zero-magnitude on signed ops is (numerically) an identity
* ``sample_signed_magnitude`` produces both signs across many draws
"""

import math

import pytest

import lucid
import lucid.utils.transforms.functional as F
from lucid.utils.transforms._autoaugment import (
    _OP_NAMES,
    NO_MAGNITUDE_OPS,
    SIGNED_OPS,
    _magnitudes_for,
    apply_op,
    sample_signed_magnitude,
)
from lucid.utils.transforms._interpolation import Interpolation

# ── functional ops ──────────────────────────────────────────────────


class TestAdjustSharpness:
    def test_factor_one_is_identity(self) -> None:
        x = lucid.rand(3, 16, 16)
        out = F.adjust_sharpness(x, 1.0)
        assert float((out - x).abs().max().item()) == 0.0

    def test_factor_zero_blurs(self) -> None:
        # Use a 6x6 with markers in the *interior* — Lucid's
        # adjust_sharpness leaves the 1-pixel border untouched
        # (matches PIL ``ImageFilter.SMOOTH``), so border markers
        # would not blur.
        x = lucid.zeros(1, 6, 6)
        x[0, 2, 2] = 1.0
        x[0, 3, 3] = 1.0
        blurred = F.adjust_sharpness(x, 0.0)
        # Interior markers spread to their 3x3 neighbourhood → max < 1.
        assert float(blurred.max().item()) < 1.0

    def test_negative_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            F.adjust_sharpness(lucid.rand(3, 8, 8), -0.1)

    def test_shape_preserved_batched(self) -> None:
        x = lucid.rand(2, 3, 16, 16)
        out = F.adjust_sharpness(x, 0.5)
        assert tuple(out.shape) == (2, 3, 16, 16)


class TestAutoContrast:
    def test_full_range_image_is_identity(self) -> None:
        # Image already spans [0, 1] in each channel.
        x = lucid.tensor([[[0.0, 0.5], [1.0, 0.25]]])  # (1, 2, 2)
        out = F.autocontrast(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_stretches_narrow_range(self) -> None:
        x = lucid.full((1, 4, 4), 0.5)
        x[0, 0, 0] = 0.2
        x[0, -1, -1] = 0.6
        out = F.autocontrast(x)
        # min → 0, max → 1.
        assert float(out.min().item()) == pytest.approx(0.0, abs=1e-6)
        assert float(out.max().item()) == pytest.approx(1.0, abs=1e-6)

    def test_flat_channel_passthrough(self) -> None:
        x = lucid.full((1, 4, 4), 0.3)
        out = F.autocontrast(x)
        # Span is 0 → input passes through unchanged.
        assert float((out - x).abs().max().item()) < 1e-6


class TestPosterize:
    def test_num_bits_eight_is_identity(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((F.posterize(x, 8) - x).abs().max().item()) == 0.0

    def test_invalid_num_bits(self) -> None:
        with pytest.raises(ValueError, match="num_bits"):
            F.posterize(lucid.rand(3, 8, 8), 0)
        with pytest.raises(ValueError, match="num_bits"):
            F.posterize(lucid.rand(3, 8, 8), 9)

    def test_quantization_levels(self) -> None:
        # 1 bit → at most 2 distinct values per channel after rounding.
        x = lucid.rand(1, 32, 32)
        out = F.posterize(x, 1)
        unique = set(out.numpy().reshape(-1).tolist())
        assert len(unique) <= 2


class TestSolarize:
    def test_threshold_one_is_identity(self) -> None:
        x = lucid.rand(3, 8, 8)
        # Strictly < 1, so no pixel is inverted.
        out = F.solarize(x, 1.000001)
        assert float((out - x).abs().max().item()) == 0.0

    def test_threshold_zero_inverts_all(self) -> None:
        x = lucid.rand(3, 8, 8)
        # All pixels >= 0 → all inverted.
        out = F.solarize(x, 0.0)
        assert float((out - (1.0 - x)).abs().max().item()) < 1e-6


# ── lookup table coverage ───────────────────────────────────────────


class TestMagnitudeTable:
    @pytest.mark.parametrize("op", _OP_NAMES)
    def test_every_op_has_entry(self, op: str) -> None:
        magnitudes, signed = _magnitudes_for(op, num_bins=31)
        if op in NO_MAGNITUDE_OPS:
            assert magnitudes == []
            assert signed is False
        else:
            assert len(magnitudes) == 31
            assert all(isinstance(v, (int, float)) for v in magnitudes)

    def test_signed_flag_matches_set(self) -> None:
        for op in _OP_NAMES:
            _, signed = _magnitudes_for(op, num_bins=31)
            assert signed is (op in SIGNED_OPS)

    def test_invalid_num_bins(self) -> None:
        with pytest.raises(ValueError, match="num_bins"):
            _magnitudes_for("Brightness", num_bins=1)

    def test_unknown_op(self) -> None:
        with pytest.raises(KeyError, match="unknown op"):
            _magnitudes_for("NotAnOp", num_bins=31)

    def test_shear_max_matches_reference(self) -> None:
        mags, signed = _magnitudes_for("ShearX", num_bins=31)
        assert mags[0] == 0.0
        assert mags[-1] == pytest.approx(0.3)
        assert signed is True

    def test_rotate_max_matches_reference(self) -> None:
        mags, _ = _magnitudes_for("Rotate", num_bins=31)
        assert mags[-1] == pytest.approx(30.0)

    def test_posterize_steps_8_to_4(self) -> None:
        mags, signed = _magnitudes_for("Posterize", num_bins=31)
        # First bin is 8 bits, last bin is 4 bits.
        assert mags[0] == 8
        assert mags[-1] == 4
        assert signed is False

    def test_solarize_descends_1_to_0(self) -> None:
        mags, signed = _magnitudes_for("Solarize", num_bins=31)
        assert mags[0] == pytest.approx(1.0)
        assert mags[-1] == pytest.approx(0.0)
        assert signed is False


# ── dispatch ────────────────────────────────────────────────────────


class TestApplyOp:
    @pytest.mark.parametrize("op", _OP_NAMES)
    def test_every_op_runs(self, op: str) -> None:
        """Every op must dispatch without raising and preserve shape."""
        lucid.manual_seed(0)
        x = lucid.rand(3, 32, 32)
        # Pick a representative magnitude for ops that need one.
        if op in NO_MAGNITUDE_OPS:
            mag = 0.0
        elif op == "Posterize":
            mag = 5.0
        elif op == "Solarize":
            mag = 0.5
        elif op in {"ShearX", "ShearY", "TranslateX", "TranslateY"}:
            mag = 0.1
        elif op == "Rotate":
            mag = 10.0
        else:  # Brightness/Color/Contrast/Sharpness factor offset
            mag = 0.3
        out = apply_op(x, op, mag, interpolation=Interpolation.BILINEAR)
        assert tuple(out.shape) == (3, 32, 32)

    def test_identity_is_exact(self) -> None:
        x = lucid.rand(3, 16, 16)
        assert float((apply_op(x, "Identity", 0.0) - x).abs().max().item()) == 0.0

    def test_zero_magnitude_brightness_is_identity(self) -> None:
        # factor = 1 + 0 = 1 → identity for brightness/saturation/contrast/sharpness.
        x = lucid.rand(3, 16, 16)
        for op in ("Brightness", "Color", "Contrast", "Sharpness"):
            out = apply_op(x, op, 0.0)
            # Saturation/contrast involve grayscale conversion + blend.
            # _blend clips to [0, 1] but at factor=1 returns img unchanged.
            assert float((out - x).abs().max().item()) < 1e-5, op

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown op"):
            apply_op(lucid.rand(3, 8, 8), "NotAnOp", 0.5)

    def test_fill_value_passes_through_to_warp(self) -> None:
        # Rotate with a 90-degree corner-going translate → border fill exposed.
        # Just verify it runs with non-zero fill (correctness of border
        # colouring is the warp_affine path's responsibility, tested there).
        x = lucid.rand(3, 16, 16)
        out = apply_op(x, "Rotate", 30.0, fill=0.5)
        assert tuple(out.shape) == (3, 16, 16)

    def test_fill_value_actually_paints_exposed_pixels(self) -> None:
        """Non-zero ``fill`` should paint the entire out-of-bounds region
        with the requested constant value, not zero (codex P2 regression
        guard)."""
        # Uniform 1.0 input → after rotation, exposed corners must equal `fill`.
        x = lucid.full((3, 16, 16), 1.0)
        for fill in (0.25, 0.5, 0.75):
            out = apply_op(
                x, "Rotate", 30.0, fill=fill, interpolation=Interpolation.BILINEAR
            ).numpy()
            # Each corner is fully outside the rotated source for a 30°
            # rotation of a square — should equal `fill` to 1e-5.
            for cy, cx in [(0, 0), (0, 15), (15, 0), (15, 15)]:
                assert float(out[0, cy, cx]) == pytest.approx(
                    fill, abs=1e-5
                ), f"corner ({cy},{cx}) for fill={fill}: got {out[0, cy, cx]}"
            # Center is still inside the source — should be ~1.0.
            assert float(out[0, 8, 8]) == pytest.approx(1.0, abs=1e-5)

    def test_fill_value_works_for_shear(self) -> None:
        """ShearX with top-left anchor exposes a triangle in the top-right
        and bottom-left of the canvas (or vice versa for negative).
        Verify that triangle is filled with the requested constant."""
        x = lucid.full((3, 16, 16), 1.0)
        out = apply_op(x, "ShearX", 0.3, fill=0.7, interpolation=Interpolation.BILINEAR).numpy()
        # Far top-right pixel is fully outside the sheared source.
        assert float(out[0, 0, 15]) == pytest.approx(1.0)  # in-bounds (top edge)
        assert float(out[0, 15, 15]) == pytest.approx(0.7, abs=1e-5)


class TestSampleSignedMagnitude:
    def test_no_magnitude_returns_zero(self) -> None:
        assert sample_signed_magnitude([], 0) == 0.0

    def test_produces_both_signs(self) -> None:
        # Across many draws we should see both positive and negative signs.
        lucid.manual_seed(0)
        mags = [1.0, 2.0, 3.0]
        signs = {math.copysign(1.0, sample_signed_magnitude(mags, 1)) for _ in range(200)}
        assert signs == {-1.0, 1.0}

    def test_magnitude_magnitude_preserved(self) -> None:
        for _ in range(20):
            v = sample_signed_magnitude([0.7], 0)
            assert abs(v) == pytest.approx(0.7)


# ── TrivialAugmentWide policy class ─────────────────────────────────


import lucid.utils.transforms as T  # noqa: E402  (after the helpers section)


class TestTrivialAugmentWide:
    def test_shape_preserved(self) -> None:
        lucid.manual_seed(0)
        tf = T.TrivialAugmentWide()
        out = tf(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_batch_shape_preserved(self) -> None:
        lucid.manual_seed(0)
        tf = T.TrivialAugmentWide()
        out = tf(lucid.rand(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 3, 32, 32)

    def test_reproducible_with_seed(self) -> None:
        tf = T.TrivialAugmentWide()
        x = lucid.rand(3, 16, 16)
        lucid.manual_seed(42)
        out1 = tf(x).numpy()
        lucid.manual_seed(42)
        out2 = tf(x).numpy()
        assert (out1 == out2).all()

    def test_all_ops_eventually_sampled(self) -> None:
        # Across many draws, every op in _OP_NAMES should appear.
        lucid.manual_seed(0)
        tf = T.TrivialAugmentWide()
        seen: set[str] = set()
        for _ in range(500):
            params = tf.make_params(lucid.rand(3, 16, 16))
            seen.add(params.op_name)
        assert seen == set(_OP_NAMES)

    def test_p_zero_is_identity(self) -> None:
        tf = T.TrivialAugmentWide(p=0.0)
        x = lucid.rand(3, 16, 16)
        out = tf(x)
        assert float((out - x).abs().max().item()) == 0.0

    def test_invalid_num_bins(self) -> None:
        with pytest.raises(ValueError, match="num_magnitude_bins"):
            T.TrivialAugmentWide(num_magnitude_bins=1)

    def test_string_interpolation_normalised(self) -> None:
        tf = T.TrivialAugmentWide(interpolation="bilinear")
        assert tf.interpolation == Interpolation.BILINEAR

    def test_repr(self) -> None:
        tf = T.TrivialAugmentWide(num_magnitude_bins=31)
        r = repr(tf)
        assert "TrivialAugmentWide" in r
        assert "num_magnitude_bins=31" in r


class TestRandAugment:
    def test_shape_preserved(self) -> None:
        lucid.manual_seed(0)
        tf = T.RandAugment()
        out = tf(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_batch_shape_preserved(self) -> None:
        lucid.manual_seed(0)
        tf = T.RandAugment()
        out = tf(lucid.rand(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 3, 32, 32)

    def test_reproducible_with_seed(self) -> None:
        tf = T.RandAugment(num_ops=2, magnitude=9)
        x = lucid.rand(3, 16, 16)
        lucid.manual_seed(123)
        out1 = tf(x).numpy()
        lucid.manual_seed(123)
        out2 = tf(x).numpy()
        assert (out1 == out2).all()

    def test_num_ops_chains_ops(self) -> None:
        # With num_ops=3, the sampled params should contain exactly 3 ops.
        lucid.manual_seed(0)
        tf = T.RandAugment(num_ops=3)
        params = tf.make_params(lucid.rand(3, 16, 16))
        assert len(params.ops) == 3

    def test_num_ops_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_ops"):
            T.RandAugment(num_ops=0)

    def test_magnitude_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            T.RandAugment(magnitude=31, num_magnitude_bins=31)
        with pytest.raises(ValueError, match="magnitude"):
            T.RandAugment(magnitude=-1)

    def test_invalid_num_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="num_magnitude_bins"):
            T.RandAugment(num_magnitude_bins=1)

    def test_p_zero_is_identity(self) -> None:
        tf = T.RandAugment(p=0.0)
        x = lucid.rand(3, 16, 16)
        out = tf(x)
        assert float((out - x).abs().max().item()) == 0.0

    def test_string_interpolation_normalised(self) -> None:
        tf = T.RandAugment(interpolation="nearest")
        assert tf.interpolation == Interpolation.NEAREST

    def test_repr(self) -> None:
        tf = T.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31)
        r = repr(tf)
        assert "RandAugment" in r
        assert "num_ops=2" in r
        assert "magnitude=9" in r


class TestAutoAugment:
    @pytest.mark.parametrize("policy", ["imagenet", "cifar10", "svhn"])
    def test_shape_preserved(self, policy: str) -> None:
        lucid.manual_seed(0)
        tf = T.AutoAugment(policy=policy)
        out = tf(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    @pytest.mark.parametrize("policy", ["imagenet", "cifar10", "svhn"])
    def test_batch_shape_preserved(self, policy: str) -> None:
        lucid.manual_seed(0)
        tf = T.AutoAugment(policy=policy)
        out = tf(lucid.rand(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 3, 32, 32)

    @pytest.mark.parametrize("policy", ["imagenet", "cifar10", "svhn"])
    def test_policy_table_has_exactly_25_sub_policies(self, policy: str) -> None:
        # Policy tables are paper-faithful: exactly 25 sub-policies each.
        from lucid.utils.transforms._autoaugment import _POLICY_TABLES

        assert len(_POLICY_TABLES[policy]) == 25
        for sub in _POLICY_TABLES[policy]:
            # Each sub-policy is 2 ops.
            assert len(sub) == 2

    def test_reproducible_with_seed(self) -> None:
        tf = T.AutoAugment(policy="imagenet")
        x = lucid.rand(3, 16, 16)
        lucid.manual_seed(7)
        out1 = tf(x).numpy()
        lucid.manual_seed(7)
        out2 = tf(x).numpy()
        assert (out1 == out2).all()

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown policy"):
            T.AutoAugment(policy="cifar100")

    def test_invalid_num_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="num_magnitude_bins"):
            T.AutoAugment(num_magnitude_bins=1)

    def test_validates_against_num_bins(self) -> None:
        # The default ImageNet policy uses mag_idx up to 9 → needs >= 10.
        with pytest.raises(ValueError, match="magnitude_idx"):
            T.AutoAugment(policy="imagenet", num_magnitude_bins=3)

    def test_p_zero_is_identity(self) -> None:
        tf = T.AutoAugment(p=0.0)
        x = lucid.rand(3, 16, 16)
        out = tf(x)
        assert float((out - x).abs().max().item()) == 0.0

    def test_string_interpolation_normalised(self) -> None:
        tf = T.AutoAugment(interpolation="bilinear")
        assert tf.interpolation == Interpolation.BILINEAR

    def test_repr(self) -> None:
        tf = T.AutoAugment(policy="imagenet")
        r = repr(tf)
        assert "AutoAugment" in r
        assert "policy='imagenet'" in r

    @pytest.mark.parametrize("policy", ["imagenet", "cifar10", "svhn"])
    def test_all_table_ops_are_known(self, policy: str) -> None:
        """Smoke-check the policy tables — every referenced op must exist."""
        from lucid.utils.transforms._autoaugment import _POLICY_TABLES

        for sub in _POLICY_TABLES[policy]:
            for op_name, prob, mag_idx in sub:
                assert op_name in _OP_NAMES
                assert 0.0 <= prob <= 1.0
                assert 0 <= mag_idx < 10  # default num_magnitude_bins

    def test_invert_op_is_handled(self) -> None:
        # ImageNet sub-policy #14 uses Invert — make sure it dispatches.
        lucid.manual_seed(0)
        # Force the call to use Invert by sampling many calls until we
        # observe at least one Invert in the resolved ops sequence.
        tf = T.AutoAugment(policy="imagenet")
        found_invert = False
        for _ in range(200):
            params = tf.make_params(lucid.rand(3, 8, 8))
            if any(op == "Invert" for op, _ in params.ops):
                found_invert = True
                break
        assert found_invert, "Invert should appear in some sub-policy draw"
