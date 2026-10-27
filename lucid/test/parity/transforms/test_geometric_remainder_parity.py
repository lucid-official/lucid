"""Parity for _geometric.py stochastic crop + Resize bilinear modes
vs Albumentations.

Covers the geometric transforms that weren't already validated in
``test_albumentations_parity.py``:

* :class:`Resize` — both ``nearest`` and ``bilinear`` modes;
* :class:`SmallestMaxSize` / :class:`LongestMaxSize` — ``bilinear``;
* :class:`RandomCrop` — pinned to deterministic position by sizing
  the crop to the full input;
* :class:`RandomResizedCrop` — pinned to ``scale=(1, 1) / ratio=(1, 1)``
  so the sampled window covers the whole image.

Random crops use parameter pinning rather than seed-pinning so the
comparison is deterministic without needing matching RNG state across
the two frameworks.  Resize bilinear lands on cv2 in both stacks, so
agreement is to float32 epsilon.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


def _image(seed: int = 0, h: int = 24, w: int = 32) -> tuple[lucid.Tensor, np.ndarray]:
    """A matched (Lucid CHW tensor, Albumentations HWC array) image pair."""
    hwc = np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf: T.Transform, chw: lucid.Tensor) -> np.ndarray:
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug: object, hwc: np.ndarray) -> np.ndarray:
    return aug(image=hwc)["image"]  # type: ignore[operator]


# ── Resize ──────────────────────────────────────────────────────────


@pytest.mark.parity
class TestResize:
    def test_resize_nearest(self) -> None:
        chw, hwc = _image(0, h=24, w=32)
        got = _run_lucid(T.Resize(12, 16, interpolation="nearest", p=1.0), chw)
        ref = _run_albu(A.Resize(12, 16, interpolation=0, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=0.0)

    def test_resize_bilinear(self) -> None:
        chw, hwc = _image(0)
        for hh, ww in [(12, 16), (40, 50)]:
            got = _run_lucid(T.Resize(hh, ww, interpolation="bilinear", p=1.0), chw)
            ref = _run_albu(A.Resize(hh, ww, interpolation=1, p=1.0), hwc)
            np.testing.assert_allclose(got, ref, atol=1e-5)


# ── SmallestMaxSize ─────────────────────────────────────────────────


@pytest.mark.parity
class TestSmallestMaxSize:
    def test_bilinear(self) -> None:
        chw, hwc = _image(0)
        got = _run_lucid(T.SmallestMaxSize(16, interpolation="bilinear", p=1.0), chw)
        ref = _run_albu(A.SmallestMaxSize(16, interpolation=1, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-5)


# ── LongestMaxSize ──────────────────────────────────────────────────


@pytest.mark.parity
class TestLongestMaxSize:
    def test_bilinear(self) -> None:
        chw, hwc = _image(0)
        got = _run_lucid(T.LongestMaxSize(20, interpolation="bilinear", p=1.0), chw)
        ref = _run_albu(A.LongestMaxSize(20, interpolation=1, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-5)


# ── RandomCrop (pinned to full-image position) ──────────────────────


@pytest.mark.parity
class TestRandomCrop:
    def test_full_image_crop_is_identity(self) -> None:
        # crop size == image size → sampled (top, left) is the only legal
        # value (0, 0); deterministic across RNG state.
        chw, hwc = _image(0, h=24, w=24)
        got = _run_lucid(T.RandomCrop(24, 24, p=1.0), chw)
        ref = _run_albu(A.RandomCrop(24, 24, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=0.0)

    def test_smaller_crop_shape(self) -> None:
        # We can't pin the offset without coupling RNGs, but we can
        # verify the output shape and value range stay correct.
        chw, _ = _image(0, h=24, w=24)
        got = _run_lucid(T.RandomCrop(16, 16, p=1.0), chw)
        assert got.shape == (16, 16, 3)
        assert got.min() >= 0.0 and got.max() <= 1.0


# ── RandomResizedCrop (pinned to identity scale/ratio) ──────────────


@pytest.mark.parity
class TestRandomResizedCrop:
    def test_identity_scale_ratio(self) -> None:
        # scale=(1.0, 1.0) + ratio=(1.0, 1.0) on a square input → the
        # sampled window is the whole image, and the resize target
        # equals the input size, so the output is identity.
        chw, hwc = _image(0, h=24, w=24)
        got = _run_lucid(
            T.RandomResizedCrop(
                24, 24, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation="bilinear", p=1.0
            ),
            chw,
        )
        np.testing.assert_allclose(got, hwc, atol=1e-5)

    def test_identity_scale_ratio_with_downscale(self) -> None:
        # Same crop pin but resize to a smaller target — output should
        # match cv2 bilinear of the full image to float32 epsilon.
        chw, hwc = _image(1, h=24, w=24)
        got = _run_lucid(
            T.RandomResizedCrop(
                12, 12, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation="bilinear", p=1.0
            ),
            chw,
        )
        ref = _run_albu(A.Resize(12, 12, interpolation=1, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
