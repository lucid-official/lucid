"""Colour / pixel-level transforms (B5)."""

import pytest

import lucid
import lucid.utils.transforms as T

_COLOR = [
    T.RandomBrightnessContrast(p=1.0),
    T.RandomGamma(p=1.0),
    T.HueSaturationValue(p=1.0),
    T.RGBShift(p=1.0),
    T.ChannelShuffle(p=1.0),
    T.ChannelDropout(p=1.0),
    T.Equalize(p=1.0),
    T.CLAHE(p=1.0),
    T.Solarize(p=1.0),
    T.Posterize(p=1.0),
    T.InvertImg(p=1.0),
    T.ToGray(p=1.0),
    T.ToSepia(p=1.0),
    T.Sharpen(p=1.0),
    T.Emboss(p=1.0),
    T.RandomToneCurve(p=1.0),
]


class TestColorShapesAndRange:
    @pytest.mark.parametrize("tf", _COLOR, ids=lambda t: type(t).__name__)
    def test_shape_and_range(self, tf: T.Transform) -> None:
        lucid.manual_seed(0)
        out = tf(lucid.rand(3, 16, 16))
        assert tuple(out.shape) == (3, 16, 16)
        assert float(out.min().item()) >= -1e-4
        assert float(out.max().item()) <= 1.0 + 1e-4


class TestColorSemantics:
    def test_invert(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((T.InvertImg(p=1.0)(x) - (1.0 - x)).abs().max().item()) < 1e-6

    def test_togray_constant_channels(self) -> None:
        g = T.ToGray(p=1.0)(lucid.rand(3, 8, 8))
        assert float((g - g[0:1]).abs().max().item()) < 1e-6

    def test_brightness_contrast_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0), contrast_limit=(0.0, 0.0), p=1.0
        )(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_channel_shuffle_preserves_pixels(self) -> None:
        # shuffling channels keeps the multiset of channel values
        x = lucid.rand(3, 8, 8)
        out = T.ChannelShuffle(p=1.0)(x)
        assert abs(float(out.sum().item()) - float(x.sum().item())) < 1e-3

    def test_dropout_leaves_boxes(self) -> None:
        s = {
            "image": T.Image(lucid.rand(3, 16, 16)),
            "boxes": T.BoundingBoxes(lucid.tensor([[1.0, 2.0, 8.0, 9.0]]), "xyxy", (16, 16)),
        }
        out = T.Solarize(p=1.0)(s)
        from lucid.utils.transforms._datatypes import to_xyxy

        assert to_xyxy(out["boxes"]).numpy().reshape(-1).tolist() == [1.0, 2.0, 8.0, 9.0]


class TestReproducibility:
    def test_rbc_seeded(self) -> None:
        lucid.manual_seed(3)
        a = T.RandomBrightnessContrast(p=1.0)(lucid.rand(3, 16, 16))
        lucid.manual_seed(3)
        b = T.RandomBrightnessContrast(p=1.0)(lucid.rand(3, 16, 16))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
