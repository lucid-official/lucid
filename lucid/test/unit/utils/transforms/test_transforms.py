"""lucid.utils.transforms — inference substrate (Albumentations API).

Covers Compose + deterministic transforms (Resize / SmallestMaxSize /
LongestMaxSize / CenterCrop / Normalize), the functional API, and the
ImageClassification preset.  Tensor-native (no numpy / PIL).
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import functional as F


class TestFunctional:
    def test_resize_exact_hw(self) -> None:
        assert tuple(F.resize(lucid.rand(3, 200, 400), (64, 48)).shape) == (3, 64, 48)

    def test_resize_int_shorter_side(self) -> None:
        assert tuple(F.resize(lucid.rand(3, 200, 400), 100).shape) == (3, 100, 200)

    def test_center_crop(self) -> None:
        assert tuple(F.center_crop(lucid.rand(3, 256, 256), 224).shape) == (3, 224, 224)

    def test_hflip_involution(self) -> None:
        x = lucid.rand(1, 4, 4)
        assert float((F.hflip(F.hflip(x)) - x).abs().max().item()) < 1e-6

    def test_pad(self) -> None:
        assert tuple(F.pad(lucid.rand(3, 8, 8), 2).shape) == (3, 12, 12)

    def test_normalize(self) -> None:
        out = F.normalize(lucid.ones(3, 4, 4) * 0.5, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(float(out.mean().item())) < 1e-6


class TestResizeFamily:
    def test_resize_exact(self) -> None:
        assert tuple(T.Resize(50, 60)(lucid.rand(3, 100, 120)).shape) == (3, 50, 60)

    def test_smallest_max_size(self) -> None:
        # shorter side (100) -> 64; 120 -> 77
        assert tuple(T.SmallestMaxSize(64)(lucid.rand(3, 100, 120)).shape) == (3, 64, 77)

    def test_longest_max_size(self) -> None:
        # longer side (120) -> 60; 100 -> 50
        assert tuple(T.LongestMaxSize(60)(lucid.rand(3, 100, 120)).shape) == (3, 50, 60)

    def test_interp_int_code(self) -> None:
        assert T.Resize(4, 4, interpolation=0).interpolation == "nearest"
        assert T.Resize(4, 4, interpolation=1).interpolation == "bilinear"


class TestCenterCropNormalize:
    def test_center_crop(self) -> None:
        assert tuple(T.CenterCrop(80, 80)(lucid.rand(3, 100, 120)).shape) == (3, 80, 80)

    def test_normalize_max_pixel_value(self) -> None:
        # mean=0.5, std=0.5, mpv=1.0, const 0.5 -> 0
        tf = T.Normalize((0.5,), (0.5,), max_pixel_value=1.0)
        assert abs(float(tf(lucid.ones(1, 4, 4) * 0.5).mean().item())) < 1e-6

    def test_normalize_mpv_255(self) -> None:
        # const 127.5 with mpv 255 -> (0.5-0.5)/0.5 = 0
        tf = T.Normalize((0.5,), (0.5,), max_pixel_value=255.0)
        assert abs(float(tf(lucid.ones(1, 4, 4) * 127.5).mean().item())) < 1e-4


class TestCompose:
    def test_chain(self) -> None:
        tf = T.Compose(
            [
                T.SmallestMaxSize(256),
                T.CenterCrop(224, 224),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        assert tuple(tf(lucid.rand(3, 300, 400)).shape) == (3, 224, 224)

    def test_repr(self) -> None:
        r = repr(T.Compose([T.Resize(8, 8)]))
        assert "Compose" in r and "Resize" in r


class TestImageClassificationPreset:
    def test_unbatched(self) -> None:
        out = T.ImageClassification(crop_size=224, resize_size=256)(lucid.rand(3, 300, 400))
        assert tuple(out.shape) == (3, 224, 224)

    def test_default_imagenet_stats(self) -> None:
        tf = T.ImageClassification(crop_size=224)
        assert tf.mean == (0.485, 0.456, 0.406)
        assert tf.std == (0.229, 0.224, 0.225)


class TestProbabilityGate:
    def test_p0_identity(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.HorizontalFlip(p=0.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            T.HorizontalFlip(p=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
