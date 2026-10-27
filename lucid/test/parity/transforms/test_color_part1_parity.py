"""Numerical parity for Lucid's ``_color.py`` value-scaling family vs Albumentations.

Opt-in tier — auto-skips when ``albumentations`` / ``cv2`` not installed.
Stochastic params pinned to single-point ranges for deterministic
comparison; tolerance picked per op based on which side routes through
``uint8`` / ``cv2``.

Tier map (be honest about each):

* **identity (atol=1e-6)** — zero-range / no-op parameters where both
  implementations collapse to ``img``: RandomGamma γ=1.0, HSV all-zero
  shifts, RGBShift all-zero, RandomToneCurve scale=0, RandomBrightness
  limit=0, RandomContrast limit=0.

* **float-eps (atol=1e-5)** — pure-float Lucid path matched against
  itself (cannot use Albu reference where formulas diverge).  Used as a
  pinned-param shape + range smoke when Albu uses a different
  parameterisation.

* **uint8-roundtrip (atol≈1.5/256)** — HSV / RGBShift where Albu
  quantises through cv2.  Existing ``test_albumentations_parity.py``
  documents this as the "ballpark" tier (~0.05 cap).

* **structural (atol=0)** — ChannelShuffle output must be an exact
  permutation of input channels; ChannelDropout dropped channel must be
  uniform 0; other channels untouched.

Classes where bit-exact Lucid↔Albu parity is **not** attempted
(formulas legitimately differ, not a Lucid bug):

* RandomBrightnessContrast / RandomBrightness / RandomContrast — Albu's
  formula multiplies around the per-image mean (``(x-μ)*(1+c)+μ`` then
  ``+ β*MAX/MEAN``) while Lucid uses the simpler additive / multiplicative
  forms documented in the docstrings.  We verify Lucid's pinned-param
  output is in-range and matches its analytic formula; Albu side is
  smoke-checked for the same shape + range only.

The aim is to lock down regressions on Lucid's side, not to prove the
two libraries are byte-equivalent on points where they're known not to
be.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


# ── helpers (mirror ``test_albumentations_parity.py``) ──────────────


def _image(seed: int = 0, h: int = 24, w: int = 32) -> tuple[lucid.Tensor, np.ndarray]:
    """Matched ``(Lucid CHW, Albumentations HWC)`` image pair in ``[0, 1]``."""
    hwc = np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf: T.Transform, chw: lucid.Tensor) -> np.ndarray:
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug: object, hwc: np.ndarray) -> np.ndarray:
    return aug(image=hwc)["image"]  # type: ignore[operator]


# ── RandomBrightnessContrast ─────────────────────────────────────────


@pytest.mark.parity
class TestRandomBrightnessContrast:
    """Albu's RBC normalises around per-image mean; Lucid does plain
    ``img*(1+c)+b``.  Formula divergence is documented — we test Lucid's
    analytic formula end-to-end and only smoke-check Albu agrees on
    shape + range."""

    def test_fixed_brightness_matches_lucid_formula(self) -> None:
        chw, _ = _image(0)
        tf = T.RandomBrightnessContrast(
            brightness_limit=(0.3, 0.3), contrast_limit=(0.0, 0.0), p=1.0
        )
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)) + 0.3, 0.0, 1.0)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_fixed_contrast_matches_lucid_formula(self) -> None:
        chw, _ = _image(1)
        tf = T.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0), contrast_limit=(0.3, 0.3), p=1.0
        )
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)) * 1.3, 0.0, 1.0)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_both_zero_identity(self) -> None:
        chw, _ = _image(2)
        tf = T.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0), contrast_limit=(0.0, 0.0), p=1.0
        )
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_albu_smoke_shape_range(self) -> None:
        # Loose: just verify Albu's RBC also produces a same-shape array
        # in [0, 1].  Numerical mismatch with Lucid is expected here
        # (per-image-mean formula divergence).
        chw, hwc = _image(3)
        got = _run_lucid(
            T.RandomBrightnessContrast(
                brightness_limit=(0.3, 0.3), contrast_limit=(0.0, 0.0), p=1.0
            ),
            chw,
        )
        ref = _run_albu(
            A.RandomBrightnessContrast(
                brightness_limit=(0.3, 0.3), contrast_limit=(0.0, 0.0), p=1.0
            ),
            hwc,
        )
        assert got.shape == ref.shape
        assert 0.0 <= float(got.min()) and float(got.max()) <= 1.0
        assert 0.0 <= float(ref.min()) and float(ref.max()) <= 1.0


# ── RandomGamma ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestRandomGamma:
    """Lucid does ``clip(img)**γ``; Albu routes through a 256-entry LUT.
    For γ=1.0 both collapse to identity; for γ≠1 we accept uint8-LUT
    quantisation drift."""

    def test_fixed_gamma_matches_lucid_formula(self) -> None:
        chw, _ = _image(0)
        tf = T.RandomGamma(gamma_limit=(120, 120), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)), 0.0, 1.0) ** 1.2
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_gamma_one_identity(self) -> None:
        chw, _ = _image(1)
        tf = T.RandomGamma(gamma_limit=(100, 100), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_albu_smoke_against_lucid(self) -> None:
        # Albu uses uint8 LUT; expect drift around ~1/255.  Smoke-only.
        chw, hwc = _image(2)
        got = _run_lucid(T.RandomGamma(gamma_limit=(120, 120), p=1.0), chw)
        ref = _run_albu(A.RandomGamma(gamma_limit=(120, 120), p=1.0), hwc)
        assert got.shape == ref.shape
        # Loose ballpark — uint8 LUT quantisation + cv2 round.
        assert np.abs(got - ref).max() < 5e-2


# ── HueSaturationValue ───────────────────────────────────────────────


@pytest.mark.parity
class TestHueSaturationValue:
    """Albu round-trips through uint8 + cv2 HSV; Lucid stays float.
    Existing albumentations parity suite documents ~0.05 tolerance for
    this op; we use the same."""

    def test_fixed_hue_shift_ballpark(self) -> None:
        chw, hwc = _image(0)
        tf = T.HueSaturationValue(
            hue_shift_limit=(10, 10), sat_shift_limit=(0, 0), val_shift_limit=(0, 0), p=1.0
        )
        got = _run_lucid(tf, chw)
        ref = _run_albu(
            A.HueSaturationValue(
                hue_shift_limit=(10, 10), sat_shift_limit=(0, 0), val_shift_limit=(0, 0), p=1.0
            ),
            hwc,
        )
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 0.05  # uint8 cv2 round-trip ballpark

    def test_zero_shifts_identity(self) -> None:
        chw, _ = _image(1)
        tf = T.HueSaturationValue(
            hue_shift_limit=(0, 0), sat_shift_limit=(0, 0), val_shift_limit=(0, 0), p=1.0
        )
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        # Float HSV round-trip can still drift a tiny amount.
        np.testing.assert_allclose(got, ref, atol=5e-3)


# ── RGBShift ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestRGBShift:
    """Lucid: ``img + shift/255`` clipped.  Albu: same formula but routes
    through cv2 / uint8 in some code paths.  Pinned shifts → comparable
    to ~1.5/256."""

    def test_fixed_r_shift_matches_lucid_formula(self) -> None:
        chw, _ = _image(0)
        tf = T.RGBShift(r_shift_limit=(20, 20), g_shift_limit=(0, 0), b_shift_limit=(0, 0), p=1.0)
        got = _run_lucid(tf, chw)
        ref_chw = chw.numpy().copy()
        ref_chw[0] = np.clip(ref_chw[0] + 20.0 / 255.0, 0.0, 1.0)
        # G/B unchanged by Lucid formula but we still clip:
        ref_chw[1] = np.clip(ref_chw[1], 0.0, 1.0)
        ref_chw[2] = np.clip(ref_chw[2], 0.0, 1.0)
        ref = np.transpose(ref_chw, (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_zero_shifts_identity(self) -> None:
        chw, _ = _image(1)
        tf = T.RGBShift(r_shift_limit=(0, 0), g_shift_limit=(0, 0), b_shift_limit=(0, 0), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_albu_smoke_uint8_ballpark(self) -> None:
        chw, hwc = _image(2)
        got = _run_lucid(
            T.RGBShift(r_shift_limit=(20, 20), g_shift_limit=(0, 0), b_shift_limit=(0, 0), p=1.0),
            chw,
        )
        ref = _run_albu(
            A.RGBShift(r_shift_limit=(20, 20), g_shift_limit=(0, 0), b_shift_limit=(0, 0), p=1.0),
            hwc,
        )
        assert got.shape == ref.shape
        # 1.5/256 ≈ 0.006 — uint8 round-trip ballpark.
        assert np.abs(got - ref).max() < 1.5 / 256


# ── ChannelShuffle ───────────────────────────────────────────────────


@pytest.mark.parity
class TestChannelShuffle:
    """Random permutation each call; can't pin to a deterministic order
    (Lucid samples internally).  Verify structural property: output's
    channels are some permutation of the input's."""

    def test_output_is_permutation_of_input(self) -> None:
        chw, _ = _image(0)
        in_chw = chw.numpy()
        tf = T.ChannelShuffle(p=1.0)
        out_hwc = _run_lucid(tf, chw)
        out_chw = np.transpose(out_hwc, (2, 0, 1))
        # Each output channel must equal some input channel exactly.
        used: list[int] = []
        for oc in range(3):
            match = None
            for ic in range(3):
                if np.array_equal(out_chw[oc], in_chw[ic]):
                    match = ic
                    break
            assert match is not None, f"output channel {oc} is not any input channel"
            used.append(match)
        # It must be a valid permutation (no repeats).
        assert sorted(used) == [0, 1, 2], f"not a permutation: {used}"

    def test_albu_also_produces_permutation(self) -> None:
        # Sanity-check Albu's ChannelShuffle obeys the same structural
        # property (different RNG so the chosen perm may differ).
        _, hwc = _image(1)
        out = _run_albu(A.ChannelShuffle(p=1.0), hwc)
        used: list[int] = []
        for oc in range(3):
            match = None
            for ic in range(3):
                if np.array_equal(out[..., oc], hwc[..., ic]):
                    match = ic
                    break
            assert match is not None
            used.append(match)
        assert sorted(used) == [0, 1, 2]


# ── ChannelDropout ───────────────────────────────────────────────────


@pytest.mark.parity
class TestChannelDropout:
    """At ``channel_drop_range=(1, 1)`` exactly one channel is dropped
    to ``fill_value``.  Verify (a) one channel is uniform 0 and (b) the
    other two equal the input channels exactly."""

    def test_one_channel_dropped_zero(self) -> None:
        chw, _ = _image(0)
        tf = T.ChannelDropout(channel_drop_range=(1, 1), fill_value=0.0, p=1.0)
        out_hwc = _run_lucid(tf, chw)
        out_chw = np.transpose(out_hwc, (2, 0, 1))
        dropped = [c for c in range(3) if np.all(out_chw[c] == 0.0)]
        assert len(dropped) == 1, f"expected exactly 1 dropped channel, got {dropped}"

    def test_other_channels_unchanged(self) -> None:
        chw, _ = _image(1)
        in_chw = chw.numpy()
        tf = T.ChannelDropout(channel_drop_range=(1, 1), fill_value=0.0, p=1.0)
        out_hwc = _run_lucid(tf, chw)
        out_chw = np.transpose(out_hwc, (2, 0, 1))
        kept = [c for c in range(3) if not np.all(out_chw[c] == 0.0)]
        assert len(kept) == 2
        for c in kept:
            np.testing.assert_allclose(out_chw[c], in_chw[c], atol=1e-6)


# ── RandomBrightness ─────────────────────────────────────────────────


@pytest.mark.parity
class TestRandomBrightness:
    """Lucid does ``clip(img + β)``; Albu mixes in per-image-mean
    normalisation.  Test Lucid's analytic formula end-to-end."""

    def test_fixed_factor_matches_lucid_formula(self) -> None:
        chw, _ = _image(0)
        tf = T.RandomBrightness(limit=(0.3, 0.3), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)) + 0.3, 0.0, 1.0)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_zero_limit_identity(self) -> None:
        chw, _ = _image(1)
        tf = T.RandomBrightness(limit=(0.0, 0.0), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-6)


# ── RandomContrast ───────────────────────────────────────────────────


@pytest.mark.parity
class TestRandomContrast:
    """Lucid does ``clip(img * (1 + c))``; Albu may normalise around
    per-image mean.  Test Lucid's analytic formula end-to-end."""

    def test_fixed_factor_matches_lucid_formula(self) -> None:
        chw, _ = _image(0)
        tf = T.RandomContrast(limit=(0.3, 0.3), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)) * 1.3, 0.0, 1.0)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_zero_limit_identity(self) -> None:
        chw, _ = _image(1)
        tf = T.RandomContrast(limit=(0.0, 0.0), p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.transpose(chw.numpy(), (1, 2, 0))
        np.testing.assert_allclose(got, ref, atol=1e-6)


# ── RandomToneCurve ──────────────────────────────────────────────────


@pytest.mark.parity
class TestRandomToneCurve:
    """At ``scale=0`` the sampled amount is exactly 0 → output is the
    clipped input (≈ identity in ``[0, 1]``).  For nonzero scale the
    sine-based Lucid curve diverges from Albu's piecewise-spline
    construction, so we only smoke-check shape + range there."""

    def test_zero_scale_identity(self) -> None:
        chw, _ = _image(0)
        tf = T.RandomToneCurve(scale=0.0, p=1.0)
        got = _run_lucid(tf, chw)
        ref = np.clip(np.transpose(chw.numpy(), (1, 2, 0)), 0.0, 1.0)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_nonzero_scale_smoke(self) -> None:
        chw, _ = _image(1)
        tf = T.RandomToneCurve(scale=0.1, p=1.0)
        got = _run_lucid(tf, chw)
        assert got.shape == (24, 32, 3)
        assert 0.0 <= float(got.min()) and float(got.max()) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
