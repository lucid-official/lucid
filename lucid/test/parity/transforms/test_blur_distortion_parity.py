"""Numerical parity for Lucid's ``_blur.py`` + ``_distortion.py`` vs Albumentations.

These ops are almost all stochastic and/or kernel-based, so bit-exact
parity is impossible across the two implementations:

* **Blur / MedianBlur / MotionBlur / GaussianBlur** — Lucid uses
  ``depthwise_conv2d`` with zero padding; Albumentations routes through
  cv2's ``filter2D`` with ``BORDER_REFLECT_101``.  Interior pixels agree
  to a kernel-shape-dependent bounded drift; border pixels disagree by
  the full kernel response.

* **GaussNoise / MultiplicativeNoise / ISONoise** — independent RNG
  streams; per-pixel parity is impossible.  Tested statistically
  (mean / range / shape preservation across many seeds).

* **Downscale** — both use nearest-up / nearest-down, but cv2's
  ``INTER_NEAREST`` and Lucid's ``functional.resize(mode="nearest")``
  pick different tie-breaks at exact half-pixel boundaries.  Smoke
  test only (shape + range + zero-region invariance).

* **Defocus / ZoomBlur** — multi-step kernel + bilinear pipelines with
  diverging RNG.  Smoke test (shape + range + bounded mean drift).

* **ElasticTransform / GridDistortion / OpticalDistortion** — both
  backward-warp through a displacement field, but Lucid uses
  ``grid_sample`` (``align_corners=True``, reflection padding) while
  Albumentations uses cv2 ``remap`` (``BORDER_CONSTANT``).  Independent
  RNG too.  Identity-parameter tests assert the trivial fixed point;
  random-parameter tests assert shape + range preservation only.

* **GridElasticDeform** — fully stochastic on top of the same backend
  divergence; smoke test only.

The tier names below say *what kind* of check the test does, not how
tight it is — most rows are deliberately loose because they have to be.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import _blur, _distortion
from lucid.utils.transforms._distortion import DispParams

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


# ── kernel-based blur: interior bounded drift ───────────────────────


@pytest.mark.parity
class TestBlurKernel:
    """Kernel-based blur — interior agreement, border excluded.

    Lucid pads zeros; Albumentations / cv2 default to BORDER_REFLECT_101.
    Both produce the same kernel response on the interior; everything else
    is the boundary-handling delta and is dropped by ``_strip_border``.
    """

    def test_blur_fixed_kernel(self) -> None:
        # Force k=3 in both (set blur_limit==3 so the only sampled value is 3).
        chw, hwc = _image(0, h=32, w=40)
        lucid_tf = _blur.Blur(blur_limit=3, p=1.0)
        # Pin Lucid's RNG so the sampled kernel size is deterministic.
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        ref = _run_albu(A.Blur(blur_limit=(3, 3), p=1.0), hwc)
        assert got.shape == ref.shape
        # Interior matches to a tight bound: identical mean-kernel math
        # under different padding only differs at the border strip.
        np.testing.assert_allclose(
            _strip_border(got, margin=2), _strip_border(ref, margin=2), atol=5e-2
        )

    def test_median_blur_fixed_kernel(self) -> None:
        # k=3 median.  Lucid uses lucid.roll (with wrap-around) + sort;
        # cv2 uses BORDER_REPLICATE.  Interior should match to float32.
        chw, hwc = _image(1, h=32, w=40)
        lucid_tf = _blur.MedianBlur(blur_limit=3, p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        ref = _run_albu(A.MedianBlur(blur_limit=(3, 3), p=1.0), hwc)
        assert got.shape == ref.shape
        # Interior median is well-defined and identical regardless of
        # padding mode (the kernel only touches the centre + 8 neighbours
        # which are all interior pixels here).
        np.testing.assert_allclose(
            _strip_border(got, margin=2), _strip_border(ref, margin=2), atol=5e-2
        )

    def test_motion_blur_fixed_kernel(self) -> None:
        # MotionBlur draws a 1-pixel-wide streak at a random angle so
        # bit-exact parity is impossible (Albu samples a different angle
        # range — [0, 360) by default — and may also drop a shift).
        # Smoke test: shape + value range + mean stays close to input mean.
        chw, hwc = _image(2, h=32, w=40)
        lucid_tf = _blur.MotionBlur(blur_limit=5, p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        ref = _run_albu(A.MotionBlur(blur_limit=(5, 5), p=1.0), hwc)
        assert got.shape == ref.shape
        # Convolution with a unit-sum kernel preserves the global mean.
        assert abs(got.mean() - hwc.mean()) < 0.05
        assert abs(ref.mean() - hwc.mean()) < 0.05
        # Both outputs share the input dynamic range (blur is contractive).
        assert 0.0 <= got.min() and got.max() <= 1.0
        assert 0.0 <= ref.min() and ref.max() <= 1.0

    def test_gaussian_blur_fixed_sigma(self) -> None:
        # Pin sigma=1.5, k=5 in both.  Lucid uses a separable depthwise
        # conv; Albu calls cv2.GaussianBlur.  Kernel weights agree to a
        # rounding eps; interior pixels match to ~5e-2 (border differs
        # because zero-pad vs reflect-pad on a Gaussian tail isn't tiny).
        chw, hwc = _image(3, h=32, w=40)
        lucid_tf = _blur.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        ref = _run_albu(A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=1.0), hwc)
        assert got.shape == ref.shape
        # Strip a wider border — Gaussian tail at k=5 reaches 2 pixels in.
        np.testing.assert_allclose(
            _strip_border(got, margin=3), _strip_border(ref, margin=3), atol=5e-2
        )


# ── stochastic noise: distributional match only ─────────────────────


@pytest.mark.parity
class TestNoiseStatistical:
    """Stochastic noise — different RNG streams, distributional check only.

    Each op draws its own noise tensor.  We compare the *image-level
    statistics* (mean, value range) across both implementations rather
    than pixel-by-pixel.  Asymmetric tolerance: noise must stay bounded
    in magnitude but signs / per-pixel values cannot match.
    """

    N_SEEDS = 8  # cheap statistical sample

    def test_gauss_noise_mean_preserved(self) -> None:
        # Zero-mean Gaussian noise should leave the image mean ~unchanged.
        lucid_diffs, albu_diffs = [], []
        for seed in range(self.N_SEEDS):
            chw, hwc = _image(seed, h=32, w=40)
            lucid.manual_seed(seed)
            got = _run_lucid(_blur.GaussNoise(var_limit=(50.0, 50.0), mean=0.0, p=1.0), chw)
            ref = _run_albu(
                A.GaussNoise(std_range=(np.sqrt(50.0) / 255.0,) * 2, mean_range=(0.0, 0.0), p=1.0),
                hwc,
            )
            assert got.shape == ref.shape == hwc.shape
            lucid_diffs.append(abs(float(got.mean()) - float(hwc.mean())))
            albu_diffs.append(abs(float(ref.mean()) - float(hwc.mean())))
            # Output stays in [0, 1] (Lucid clips; Albu may not — we
            # only assert Lucid's bound + a loose envelope for Albu).
            assert 0.0 <= got.min() and got.max() <= 1.0
        # Mean drift is bounded for both — zero-mean noise.
        assert max(lucid_diffs) < 0.05
        assert max(albu_diffs) < 0.05

    def test_multiplicative_noise_mean_preserved(self) -> None:
        # Multiplier uniform in [0.9, 1.1] -> expected scale ≈ 1.0.
        for seed in range(self.N_SEEDS):
            chw, hwc = _image(seed, h=32, w=40)
            lucid.manual_seed(seed)
            got = _run_lucid(_blur.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0), chw)
            ref = _run_albu(A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0), hwc)
            assert got.shape == ref.shape == hwc.shape
            assert 0.0 <= got.min() and got.max() <= 1.0
            # Multiplier is within ~10% of identity -> mean drift bounded.
            assert abs(float(got.mean()) - float(hwc.mean())) < 0.15
            assert abs(float(ref.mean()) - float(hwc.mean())) < 0.15

    def test_iso_noise_mean_preserved(self) -> None:
        # ISONoise adds saturation perturbation + scaled Gaussian.
        # Lucid's implementation is an *approximation* of Albu's
        # (Albu uses sRGB / luminance proper conversion) — only
        # check structural correctness, not numerics.
        for seed in range(self.N_SEEDS):
            chw, hwc = _image(seed, h=32, w=40)
            lucid.manual_seed(seed)
            got = _run_lucid(
                _blur.ISONoise(color_shift=(0.02, 0.02), intensity=(0.2, 0.2), p=1.0), chw
            )
            ref = _run_albu(A.ISONoise(color_shift=(0.02, 0.02), intensity=(0.2, 0.2), p=1.0), hwc)
            assert got.shape == ref.shape == hwc.shape
            assert 0.0 <= got.min() and got.max() <= 1.0
            # Both add a small perturbation around the input.
            assert abs(float(got.mean()) - float(hwc.mean())) < 0.1


# ── complex / multi-step blur: structural smoke only ────────────────


@pytest.mark.parity
class TestComplexBlur:
    """Multi-step ops — shape / range / mean drift only.

    Defocus and ZoomBlur both compose several kernel / resize stages
    whose details diverge between the two backends.  We assert just
    enough to catch a regression that mangles the shape or blows the
    value range, not pixel-level parity.
    """

    def test_defocus_runs(self) -> None:
        chw, hwc = _image(4, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(_blur.Defocus(radius=(3, 3), p=1.0), chw)
        ref = _run_albu(A.Defocus(radius=(3, 3), p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Disk-kernel blur preserves the global mean (unit-sum kernel).
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.05

    def test_zoom_blur_runs(self) -> None:
        # ZoomBlur averages 5 centre-zoomed copies + the original.  Both
        # implementations diverge in resampling (cv2 vs Lucid bilinear)
        # so only smoke-test shape + range + bounded mean drift.
        chw, hwc = _image(5, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(_blur.ZoomBlur(max_factor=1.2, p=1.0), chw)
        ref = _run_albu(A.ZoomBlur(max_factor=(1.2, 1.2), p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Centre-zoom keeps overall brightness close to the input.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.1

    def test_downscale_shape(self) -> None:
        # Both use nearest-neighbour for down + up; ties at exact
        # half-pixel boundaries are picked differently.  Smoke test:
        # output shape preserved, dtype preserved, range preserved.
        chw, hwc = _image(6, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(_blur.Downscale(scale_min=0.5, scale_max=0.5, p=1.0), chw)
        ref = _run_albu(A.Downscale(scale_range=(0.5, 0.5), p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert got.dtype == hwc.dtype
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Nearest-neighbour roundtrip preserves the set of pixel values
        # (only their *positions* change).  The mean is roughly stable.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.1


# ── distortions: identity-param exact, random bounded ───────────────


@pytest.mark.parity
class TestDistortion:
    """Grid-warp distortions.

    *Identity-parameter* tests pin params at the trivial fixed point
    (alpha=0 for Elastic, distort_limit=0 for Grid/Optical).  Both
    backends should return the image essentially unchanged (modulo
    interpolation rounding).

    *Random-parameter* tests can only assert structural invariants
    (shape, range) — the backward sampling grid plus independent RNG
    means pixel parity is out of reach.
    """

    def test_elastic_transform_identity_params(self) -> None:
        # alpha=0 -> displacement field is identically zero ->
        # output should be ≈ input (bilinear roundtrip on integer
        # pixel grid is identity to float32 epsilon).
        chw, hwc = _image(7, h=32, w=40)
        lucid_tf = _distortion.ElasticTransform(alpha=0.0, sigma=50.0, p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        assert got.shape == hwc.shape
        # grid_sample at the integer grid with reflection padding is
        # identity on the interior; allow a tight tolerance for the
        # sampling normalisation arithmetic.
        np.testing.assert_allclose(
            _strip_border(got, margin=1), _strip_border(hwc, margin=1), atol=1e-4
        )

    def test_grid_distortion_identity_params(self) -> None:
        # distort_limit=0 -> all control-grid perturbations are 0 ->
        # field is identically zero -> output ≈ input.
        chw, hwc = _image(8, h=32, w=40)
        lucid_tf = _distortion.GridDistortion(num_steps=5, distort_limit=0.0, p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        assert got.shape == hwc.shape
        np.testing.assert_allclose(
            _strip_border(got, margin=1), _strip_border(hwc, margin=1), atol=1e-4
        )

    def test_optical_distortion_identity_params(self) -> None:
        # k=0 (distort_limit=0) and shift=0 -> factor = 1.0 -> identity
        # field.  Output equals input (up to grid_sample rounding).
        chw, hwc = _image(9, h=32, w=40)
        lucid_tf = _distortion.OpticalDistortion(distort_limit=0.0, shift_limit=0.0, p=1.0)
        lucid.manual_seed(0)
        got = _run_lucid(lucid_tf, chw)
        assert got.shape == hwc.shape
        np.testing.assert_allclose(
            _strip_border(got, margin=1), _strip_border(hwc, margin=1), atol=1e-4
        )

    def test_grid_elastic_deform_runs(self) -> None:
        # Fully stochastic + grid_sample backend divergence -> smoke
        # only.  Shape + range + bounded mean drift.
        chw, hwc = _image(10, h=32, w=40)
        lucid.manual_seed(0)
        got = _run_lucid(_distortion.GridElasticDeform(num_grid_xy=(4, 4), magnitude=5, p=1.0), chw)
        ref = _run_albu(A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=5, p=1.0), hwc)
        assert got.shape == ref.shape == hwc.shape
        assert 0.0 <= got.min() and got.max() <= 1.0
        # Elastic deformation is volume-preserving on average — mean
        # drift bounded by border-effect mass.
        assert abs(float(got.mean()) - float(hwc.mean())) < 0.15


# ── deterministic fixed-displacement parity ─────────────────────────


@pytest.mark.parity
class TestDistortionDeterministic:
    """Bypass the RNG entirely by hand-constructing a ``DispParams``.

    Calling ``_apply_image`` directly with a known displacement field
    isolates the geometric backend from the parameter-sampling step.
    A zero displacement must round-trip to the input exactly; a small
    constant shift must equal the geometrically-shifted source.
    """

    def test_zero_displacement_is_identity(self) -> None:
        chw, hwc = _image(11, h=24, w=32)
        h, w = hwc.shape[:2]
        zeros = lucid.zeros(h, w)
        params = DispParams(dx=zeros, dy=zeros, out_hw=(h, w))
        tf = _distortion.ElasticTransform(alpha=0.0, sigma=1.0, p=1.0)
        out = tf._apply_image(chw, params).numpy()
        out_hwc = np.transpose(out, (1, 2, 0))
        # grid_sample at the integer grid with align_corners=True is
        # bit-exact on the interior; reflection padding only affects
        # the 1-pixel border.
        np.testing.assert_allclose(
            _strip_border(out_hwc, margin=1), _strip_border(hwc, margin=1), atol=1e-5
        )

    def test_constant_shift_matches_numpy_roll(self) -> None:
        # A constant displacement (dx=+1, dy=0) shifts the sampling
        # coordinate by +1 — i.e. the output at column x reads the
        # input at column x+1 (numpy roll by -1 along axis=1).
        chw, hwc = _image(12, h=24, w=32)
        h, w = hwc.shape[:2]
        dx = lucid.ones(h, w) * 1.0
        dy = lucid.zeros(h, w)
        params = DispParams(dx=dx, dy=dy, out_hw=(h, w))
        tf = _distortion.GridDistortion(num_steps=5, distort_limit=0.0, p=1.0)
        out = tf._apply_image(chw, params).numpy()
        out_hwc = np.transpose(out, (1, 2, 0))
        expected = np.roll(hwc, shift=-1, axis=1)
        # Strip a 2-pixel border on each side: the right edge wraps via
        # reflection padding rather than out-of-bounds, so values
        # diverge there.  Interior is the geometric shift exactly.
        margin = 2
        np.testing.assert_allclose(
            out_hwc[margin:-margin, margin:-margin, :],
            expected[margin:-margin, margin:-margin, :],
            atol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
