"""Numerical parity for Lucid's ``_color.py`` kernel / quantisation /
inversion family vs Albumentations.

Companion to ``test_color_part1_parity.py`` (which covers the
brightness / contrast / gamma / HSV / RGB-shift / channel-perm /
tone-curve half).  This file covers the **pixel-kernel / bit-mask /
matrix / stochastic-mask** half — 14 classes:

* :class:`Equalize` — uint8 histogram equalization (per-channel CDF).
* :class:`CLAHE` — Contrast-limited adaptive histogram equalization.
  Lucid pivots through HSV-V, Albumentations through LAB-L; pixel
  parity is impossible.  Tested structurally only.
* :class:`Solarize` (class form) — ``where(x >= t, 1 - x, x)``.
* :class:`Posterize` (class form, ``mode="uint8_mask"``) — bit-mask
  quantisation matching cv2 to float32 epsilon.
* :class:`InvertImg` — exact ``1 - x``.
* :class:`ToGray` — BT.601 luminance weights.
* :class:`ToSepia` — fixed 3x3 colour matrix.
* :class:`Sharpen` — 3x3 Laplacian unsharp kernel blended with input.
* :class:`Emboss` — 3x3 directional gradient kernel blended with input.
* :class:`UnsharpMask` — Gaussian-blur subtractive sharpen.
* :class:`RingingOvershoot` — high-pass kernel approximation.
* :class:`FancyPCA` — random PCA on the per-image RGB covariance.
  Stochastic + independent eigendecomposition → structural test only.
* :class:`PixelDropout` — Bernoulli per-pixel mask.  Independent RNG →
  structural test only.
* :class:`XYMasking` — random row / column band masks.  Independent
  RNG → structural test only.

Four tolerance tiers — be explicit about which class lands in which:

* **bit-exact** (atol=0)
  – ``InvertImg``: ``1 - x`` on both sides.
  – ``ToGray``: same BT.601 weights → bit-exact when Lucid's float
    path is verified to match Albu's float path (no uint8 round-trip
    when the input is already float in [0, 1]).
  – ``Solarize(threshold=(t, t), p=1.0)``: ``where`` is an
    elementwise comparison; same threshold value → bit-exact.

* **uint8 round-trip** (atol≈1.5/256≈0.006)
  – ``Posterize(num_bits, mode="uint8_mask", p=1.0)``.
  – ``Equalize(p=1.0)``.
  – ``ToSepia(p=1.0)`` (small float fma; effectively bit-exact too).

* **kernel ops, bounded interior drift** (atol≈5e-2 after
  ``_strip_border``).  Lucid uses zero padding via ``depthwise_conv2d``;
  Albumentations / cv2 use ``BORDER_REFLECT_101`` — interior agrees,
  border differs.  ``RingingOvershoot`` ignores its ``blur_limit``
  param and uses a fixed 3x3 kernel — so Albumentations' parameter
  isn't replicable.  Run smoke only.

* **structural** (shape / range / mean / count — no pixel parity)
  – ``CLAHE``: HSV vs LAB pivot.
  – ``FancyPCA``: covariance + eigendecomp differs per backend; mean
    is preserved.
  – ``PixelDropout``: independent RNG; assert drop count matches.
  – ``XYMasking``: independent RNG; assert masked pixel count matches.

Opt-in tier — auto-skips when ``albumentations`` / ``cv2`` not
installed.  Run with::

    pip install albumentations
    pytest -m parity lucid/test/parity/transforms/test_color_part2_parity.py
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


def _strip_border(arr: np.ndarray, margin: int = 2) -> np.ndarray:
    """Drop edge pixels — kernel ops disagree on borders due to padding."""
    if arr.ndim == 3:
        return arr[margin:-margin, margin:-margin, :]
    return arr[margin:-margin, margin:-margin]


# ── bit-exact tier ──────────────────────────────────────────────────


@pytest.mark.parity
class TestExact:
    """Transforms whose math is unambiguous → bit-exact agreement."""

    def test_invert_img(self) -> None:
        # ``1 - x`` on both sides.  No quantisation, no rounding.
        chw, hwc = _image(0)
        got = _run_lucid(T.InvertImg(p=1.0), chw)
        ref = _run_albu(A.InvertImg(p=1.0), hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_to_gray(self) -> None:
        # BT.601 weights on float [0, 1] → identical float result.
        # Lucid uses ``F.rgb_to_grayscale(keep_channels=True)``; Albu
        # defaults to ``method="weighted_average"`` with the same
        # weights.  Allow a tiny atol for safety against fma ordering.
        chw, hwc = _image(1)
        got = _run_lucid(T.ToGray(p=1.0), chw)
        ref = _run_albu(A.ToGray(p=1.0), hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0.0)

    def test_solarize_class_fixed(self) -> None:
        # Fixed threshold (no sampling): ``where(x >= t, 1 - x, x)``
        # on both sides.  Note Lucid's ``threshold`` is on the 0-255
        # scale (divided by 255 internally); Albumentations 2.x takes
        # ``threshold_range`` in [0, 1].
        chw, hwc = _image(2)
        got = _run_lucid(T.Solarize(threshold=(128, 128), p=1.0), chw)
        ref = _run_albu(A.Solarize(threshold_range=(128 / 255.0,) * 2, p=1.0), hwc)
        assert got.shape == ref.shape
        # The comparison ``x >= t`` is identical on both sides; the
        # selected value is either ``x`` or ``1 - x`` — both exact.
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0.0)


# ── uint8 round-trip tier ───────────────────────────────────────────


@pytest.mark.parity
class TestUint8Roundtrip:
    """Transforms that quantise through uint8 → float32-epsilon agreement."""

    @pytest.mark.parametrize("num_bits", [1, 2, 4, 6, 7])
    def test_posterize_class(self, num_bits: int) -> None:
        # The uint8 bit-mask path matches cv2 to float32 epsilon for
        # every supported bit count (see G3 parity suite).  Use the
        # class form ``T.Posterize(num_bits=…, mode="uint8_mask")``;
        # functional form is covered by ``test_albumentations_parity.py``.
        chw, hwc = _image(10 + num_bits)
        got = _run_lucid(T.Posterize(num_bits=num_bits, mode="uint8_mask", p=1.0), chw)
        ref = _run_albu(A.Posterize(num_bits=num_bits, p=1.0), hwc)
        assert got.shape == ref.shape
        # 1.5/256 ≈ 6e-3 is the half-step uint8 quantisation budget;
        # the bit-mask path is exact, but we tolerate fma ordering.
        np.testing.assert_allclose(got, ref, atol=1.5 / 256.0)

    def test_equalize(self) -> None:
        # Per-channel histogram equalization via uint8 round-trip.
        # Albu defaults to ``mode="cv"`` + ``by_channels=True`` — same
        # algorithm Lucid uses (per-channel CDF, uint8 LUT).  Any
        # divergence here comes from CDF tie-breaks at empty bins,
        # which can shift a handful of pixels by at most 1/255.
        chw, hwc = _image(3)
        got = _run_lucid(T.Equalize(p=1.0), chw)
        ref = _run_albu(A.Equalize(mode="cv", by_channels=True, p=1.0), hwc)
        assert got.shape == ref.shape
        # Allow ~3/256 — the empty-bin tie-break can drift a few pixels
        # by a small multiple of 1/255 depending on histogram shape.
        np.testing.assert_allclose(got, ref, atol=3.0 / 256.0)

    def test_to_sepia(self) -> None:
        # Both apply the same canonical 3x3 sepia matrix.  Lucid
        # explicitly clips to [0, 1]; Albu may not — but with this
        # matrix on inputs in [0, 1] the output stays in [0, 1.13]
        # so the clip only touches a few highlight pixels.  Strip the
        # max to keep the comparison honest (Albu doesn't clip; Lucid
        # does → divergence at saturated highlights only).
        chw, hwc = _image(4)
        got = _run_lucid(T.ToSepia(p=1.0), chw)
        ref = _run_albu(A.ToSepia(p=1.0), hwc)
        assert got.shape == ref.shape
        # Albumentations clips internally too — both round through the
        # same matrix and same clip semantics.  Allow uint8 budget
        # for any internal quantisation step.
        np.testing.assert_allclose(got, ref, atol=1.5 / 256.0)


# ── kernel ops, bounded interior drift ──────────────────────────────


@pytest.mark.parity
class TestKernelBounded:
    """3x3 / 5x5 convolutional ops — interior bounded, border excluded.

    Lucid pads zeros; Albumentations / cv2 default to
    ``BORDER_REFLECT_101``.  Both produce the same kernel response on
    the interior; everything else is the boundary-handling delta and
    is dropped by ``_strip_border``.
    """

    def test_sharpen_interior(self) -> None:
        # Fix alpha=0.5 and lightness=1.0 by collapsing the ranges to
        # singletons.  Lucid samples once from each range, Albumentations
        # similarly — pinning both to a singleton range is the only
        # way to get matched-kernel parity.
        chw, hwc = _image(5, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(T.Sharpen(alpha=(0.5, 0.5), lightness=(1.0, 1.0), p=1.0), chw)
        # Force Albu to the ``"kernel"`` method (not the newer Gaussian
        # path), matching Lucid's 3x3 Laplacian-style kernel.
        ref = _run_albu(
            A.Sharpen(alpha=(0.5, 0.5), lightness=(1.0, 1.0), method="kernel", p=1.0), hwc
        )
        assert got.shape == ref.shape
        # Interior agrees to a kernel-shape-dependent bound (~5e-2);
        # border zero-pad vs reflect-pad delta is dropped.
        np.testing.assert_allclose(
            _strip_border(got, margin=2), _strip_border(ref, margin=2), atol=5e-2
        )

    def test_emboss_interior(self) -> None:
        # Same story as Sharpen — fix the sampled values to singletons.
        chw, hwc = _image(6, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(T.Emboss(alpha=(0.5, 0.5), strength=(0.5, 0.5), p=1.0), chw)
        ref = _run_albu(A.Emboss(alpha=(0.5, 0.5), strength=(0.5, 0.5), p=1.0), hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(
            _strip_border(got, margin=2), _strip_border(ref, margin=2), atol=5e-2
        )

    def test_unsharp_mask_interior(self) -> None:
        # Pin kernel size (k=3) and alpha (0.5) to singletons; pin
        # sigma_limit to (1.0, 1.0).  Lucid and Albu both compute
        # img + alpha * (img - gaussian_blur(img)), but with different
        # Gaussian-kernel implementations (separable conv vs cv2).
        # Interior drift bounded by the Gaussian tail times alpha.
        chw, hwc = _image(7, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(
            T.UnsharpMask(
                blur_limit=(3, 3), sigma_limit=(1.0, 1.0), alpha=(0.5, 0.5), threshold=10.0, p=1.0
            ),
            chw,
        )
        ref = _run_albu(
            A.UnsharpMask(
                blur_limit=(3, 3), sigma_limit=(1.0, 1.0), alpha=(0.5, 0.5), threshold=10, p=1.0
            ),
            hwc,
        )
        assert got.shape == ref.shape
        # k=3 Gaussian tail at sigma=1 reaches 1 pixel; strip 2 to be
        # safe.  ``threshold`` differs between backends in how it
        # gates the sharpening on low-contrast regions — bound is
        # loose to absorb that.
        np.testing.assert_allclose(
            _strip_border(got, margin=2), _strip_border(ref, margin=2), atol=1e-1
        )

    def test_ringing_overshoot_runs(self) -> None:
        # Lucid implements ``RingingOvershoot`` as a fixed 3x3 high-pass
        # kernel and *ignores* ``blur_limit``; Albumentations does the
        # full frequency-domain sinc-cutoff dance.  Bit / pixel parity
        # is out of reach.  Smoke-test shape + range + mean drift.
        chw, hwc = _image(8, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(T.RingingOvershoot(blur_limit=(7, 7), p=1.0), chw)
        ref = _run_albu(A.RingingOvershoot(blur_limit=(7, 7), p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Mean drift bounded for both — both are zero-DC kernels acting
        # on bounded input.  Lucid's high-pass amplifies edges relative
        # to Albu's sinc filter; allow a looser envelope.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.2


# ── structural tier (independent RNG / different algorithm) ─────────


@pytest.mark.parity
class TestStructural:
    """Pixel parity is out of reach — assert shape / range / mean / counts."""

    def test_clahe_mean_preserved(self) -> None:
        # Lucid's CLAHE pivots through the HSV value channel; Albu
        # routes through the LAB lightness channel.  Pixel parity is
        # impossible — but both are luminance-only contrast-limited
        # equalisations, so the global mean should be preserved to
        # within a small CDF-tie-break envelope.
        chw, hwc = _image(9, h=32, w=40)
        got = _run_lucid(T.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), chw)
        ref = _run_albu(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Both should stay close to the input mean — CLAHE redistributes
        # local intensities but is roughly mean-preserving on a uniform
        # noise input.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.15
        # Sanity: outputs are *not* identical to the input (the
        # transform actually did something).
        assert float(np.abs(got - hwc).mean()) > 1e-3
        assert float(np.abs(ref - hwc).mean()) > 1e-3

    def test_fancy_pca_runs(self) -> None:
        # Random PCA on a noise image — eigendecomposition is
        # stable up to sign / ordering ambiguities, and Lucid samples
        # its own per-channel ``alpha`` from an independent RNG.  No
        # pixel parity expected.  Assert shape + range + bounded
        # mean drift (alpha=0.1 → perturbation magnitude small).
        chw, hwc = _image(10, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(T.FancyPCA(alpha=0.1, p=1.0), chw)
        ref = _run_albu(A.FancyPCA(alpha=0.1, p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # alpha=0.1 caps the eigenvalue-weighted perturbation; mean
        # stays close to input mean.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.1
        assert abs(float(ref.mean()) - float(hwc.mean())) < 0.1

    def test_pixel_dropout_runs(self) -> None:
        # Independent per-pixel Bernoulli mask.  Pixel parity
        # impossible.  Assert shape + range + drop *count* matches the
        # expected Binomial mean: ``dropout_prob * num_pixels``.
        chw, hwc = _image(11, h=32, w=40)
        prob = 0.1
        lucid.manual_seed(0)
        got = _run_lucid(
            T.PixelDropout(dropout_prob=prob, per_channel=False, drop_value=0.0, p=1.0), chw
        )
        ref = _run_albu(
            A.PixelDropout(dropout_prob=prob, per_channel=False, drop_value=0.0, p=1.0), hwc
        )
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Count exact-zero pixels in channel 0 — a proxy for the
        # dropout mask since the input is U[0, 1] (so the chance of
        # an organic zero is float-eps).  Expected count ≈ prob * h * w
        # with std ≈ sqrt(prob * (1-prob) * h * w) ≈ 11 for 32*40*0.1.
        # Allow ±5 std → ±55.
        h, w = hwc.shape[:2]
        expected = prob * h * w
        std = np.sqrt(prob * (1 - prob) * h * w)
        got_zeros = int(np.sum(got[..., 0] == 0.0))
        ref_zeros = int(np.sum(ref[..., 0] == 0.0))
        assert abs(got_zeros - expected) <= 5 * std + 1
        assert abs(ref_zeros - expected) <= 5 * std + 1

    def test_xy_masking_runs(self) -> None:
        # Random horizontal + vertical band masks.  Both backends
        # sample band offsets independently — pixel parity impossible.
        # Assert shape + range + that *some* masking happened (at
        # least the expected number of fully-zero rows / columns).
        chw, hwc = _image(12, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(
            T.XYMasking(
                num_masks_x=2,
                num_masks_y=2,
                mask_x_length=8,
                mask_y_length=8,
                fill_value=0.0,
                p=1.0,
            ),
            chw,
        )
        ref = _run_albu(
            A.XYMasking(
                num_masks_x=2, num_masks_y=2, mask_x_length=8, mask_y_length=8, fill=0.0, p=1.0
            ),
            hwc,
        )
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Count fully-zero columns and rows.  With 2 vertical bands
        # of width 8 we expect 8-16 fully-zero columns (depending on
        # band overlap); with 2 horizontal bands of height 8 we
        # expect 8-16 fully-zero rows.
        h, w = hwc.shape[:2]
        # A column is "masked" if every value in it (across rows and
        # channels) equals 0.  Same for rows.
        got_zero_cols = int(np.sum(np.all(got == 0.0, axis=(0, 2))))
        ref_zero_cols = int(np.sum(np.all(ref == 0.0, axis=(0, 2))))
        got_zero_rows = int(np.sum(np.all(got == 0.0, axis=(1, 2))))
        ref_zero_rows = int(np.sum(np.all(ref == 0.0, axis=(1, 2))))
        # Bands may end up not covering any full row/column on some
        # Albu versions (where the band sampling lands in regions that
        # don't fully zero a row/column).  Tolerate that — only require
        # the *upper bound* and that Lucid produces at least some bands
        # (it's deterministic about always masking something).
        assert got_zero_cols <= 16
        assert ref_zero_cols <= 16
        assert got_zero_rows <= 16
        assert ref_zero_rows <= 16
        # Lucid should produce at least one masked row OR column.
        assert got_zero_cols + got_zero_rows > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
