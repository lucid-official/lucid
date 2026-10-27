"""G4b/c — segmentation-mask label preservation + multi-channel coverage.

Two robustness invariants the existing suite was thin on:

* every geometric transform must keep an integer-label mask *integer-
  valued* (nearest interpolation, no halo of synthetic in-between
  labels);
* photometric transforms that don't gate on channel count must accept
  arbitrary ``C`` (1, 4, 8 …) without crashing — those that do gate
  (RGB-only) must raise a clear :class:`ValueError`.
"""

from typing import Callable

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._base import Transform


def _label_mask(h: int, w: int, n_labels: int = 5) -> T.Mask:
    """Random integer-label mask in ``{0, 1, …, n_labels - 1}``."""
    lucid.manual_seed(0)
    data = lucid.rand(1, h, w)
    # Quantise floats into discrete bins → integer labels stored as float.
    bins = lucid.floor(data * float(n_labels))
    return T.Mask(lucid.clip(bins, 0.0, float(n_labels - 1)))


def _label_set(mask: T.Mask) -> set[int]:
    return {int(round(v)) for v in mask.data.numpy().reshape(-1).tolist()}


# ── G4b — mask label preservation through every geometric transform ──


_GEOMETRIC_FIXTURES: list[tuple[str, Callable[[], Transform]]] = [
    ("HorizontalFlip", lambda: T.HorizontalFlip(p=1.0)),
    ("VerticalFlip", lambda: T.VerticalFlip(p=1.0)),
    ("Transpose", lambda: T.Transpose(p=1.0)),
    ("Flip", lambda: T.Flip(p=1.0)),
    ("CenterCrop", lambda: T.CenterCrop(20, 24, p=1.0)),
    ("Crop", lambda: T.Crop(2, 3, 22, 27, p=1.0)),
    ("Resize", lambda: T.Resize(20, 24, p=1.0)),
    ("SmallestMaxSize", lambda: T.SmallestMaxSize(16, p=1.0)),
    ("LongestMaxSize", lambda: T.LongestMaxSize(20, p=1.0)),
    ("RandomCrop", lambda: T.RandomCrop(20, 24, p=1.0)),
    ("RandomResizedCrop", lambda: T.RandomResizedCrop(20, 24, p=1.0)),
    ("PadIfNeeded", lambda: T.PadIfNeeded(64, 64, p=1.0)),
    ("Rotate", lambda: T.Rotate(limit=45, p=1.0)),
    ("Affine", lambda: T.Affine(translate_percent=0.1, scale=0.9, p=1.0)),
    ("RandomRotate90", lambda: T.RandomRotate90(p=1.0)),
    ("D4", lambda: T.D4(p=1.0)),
    ("RandomScale", lambda: T.RandomScale(scale_limit=0.2, p=1.0)),
    ("ShiftScaleRotate", lambda: T.ShiftScaleRotate(p=1.0)),
    ("ElasticTransform", lambda: T.ElasticTransform(alpha=10.0, sigma=3.0, p=1.0)),
    ("GridDistortion", lambda: T.GridDistortion(p=1.0)),
    ("OpticalDistortion", lambda: T.OpticalDistortion(p=1.0)),
    ("Perspective", lambda: T.Perspective(p=1.0)),
]


class TestMaskLabelPreservation:
    @pytest.mark.parametrize(
        "name, make_tf", _GEOMETRIC_FIXTURES, ids=[n for n, _ in _GEOMETRIC_FIXTURES]
    )
    def test_integer_labels_preserved(self, name: str, make_tf: Callable[[], Transform]) -> None:
        # Build a 5-label mask + a paired image (the transform's sampler
        # needs an image-shaped reference for shape-derived params).
        lucid.manual_seed(0)
        h, w = 32, 40
        sample = {
            "image": T.Image(lucid.rand(3, h, w)),
            "mask": _label_mask(h, w, n_labels=5),
        }
        original_labels = _label_set(sample["mask"])
        out = make_tf()(sample)
        out_labels = _label_set(out["mask"])
        new = out_labels - original_labels
        assert not new, (
            f"{name}: mask gained synthetic labels {sorted(new)} "
            f"(input labels were {sorted(original_labels)}, "
            f"output labels {sorted(out_labels)}) — interpolation leaked."
        )


# ── G4c — multi-channel coverage for photometric transforms ─────────


# Transforms that internally enforce 3-channel RGB and should raise on
# 1- or 4-channel input.
_RGB_ONLY: list[Callable[[], Transform]] = [
    lambda: T.HueSaturationValue(p=1.0),
    lambda: T.RGBShift(p=1.0),
    lambda: T.ChannelShuffle(p=1.0),
    lambda: T.ToGray(p=1.0),
    lambda: T.ToSepia(p=1.0),
    lambda: T.FancyPCA(p=1.0),
]

# Transforms that should accept arbitrary channel count.
_CHANNEL_AGNOSTIC: list[Callable[[], Transform]] = [
    lambda: T.Solarize(p=1.0),
    lambda: T.Posterize(num_bits=4, p=1.0),
    lambda: T.InvertImg(p=1.0),
    lambda: T.RandomBrightness(p=1.0),
    lambda: T.RandomContrast(p=1.0),
    lambda: T.RandomBrightnessContrast(p=1.0),
    lambda: T.RandomGamma(p=1.0),
    lambda: T.GaussNoise(p=1.0),
    lambda: T.MultiplicativeNoise(p=1.0),
    lambda: T.Blur(blur_limit=3, p=1.0),
    lambda: T.GaussianBlur(blur_limit=(3, 7), p=1.0),
    lambda: T.MedianBlur(blur_limit=3, p=1.0),
    lambda: T.PixelDropout(p=1.0),
]


class TestMultiChannelCoverage:
    @pytest.mark.parametrize("make_tf", _RGB_ONLY, ids=[f.__qualname__ for f in _RGB_ONLY])
    @pytest.mark.parametrize("channels", [1, 4])
    def test_rgb_only_rejects_non_three(
        self, make_tf: Callable[[], Transform], channels: int
    ) -> None:
        lucid.manual_seed(0)
        img = lucid.rand(channels, 16, 20)
        with pytest.raises(ValueError):
            make_tf()(T.Image(img))

    @pytest.mark.parametrize(
        "make_tf",
        _CHANNEL_AGNOSTIC,
        ids=[f.__qualname__ for f in _CHANNEL_AGNOSTIC],
    )
    @pytest.mark.parametrize("channels", [1, 3, 4, 7])
    def test_channel_agnostic_accepts_any(
        self, make_tf: Callable[[], Transform], channels: int
    ) -> None:
        lucid.manual_seed(channels)
        img = lucid.rand(channels, 16, 20)
        out = make_tf()(T.Image(img)).data
        assert tuple(out.shape) == (channels, 16, 20)
        assert 0.0 <= float(out.min().item()) and float(out.max().item()) <= 1.0 + 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
