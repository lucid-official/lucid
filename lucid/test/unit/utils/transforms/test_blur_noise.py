"""Blur + noise transforms (B6)."""

import pytest

import lucid
import lucid.utils.transforms as T

_BN = [
    T.Blur(p=1.0),
    T.MedianBlur(p=1.0),
    T.MotionBlur(p=1.0),
    T.GaussianBlur(p=1.0),
    T.GaussNoise(p=1.0),
    T.MultiplicativeNoise(p=1.0),
    T.ISONoise(p=1.0),
    T.Downscale(p=1.0),
]


class TestBlurNoise:
    @pytest.mark.parametrize("tf", _BN, ids=lambda t: type(t).__name__)
    def test_shape_and_range(self, tf: T.Transform) -> None:
        lucid.manual_seed(0)
        out = tf(lucid.rand(3, 24, 24))
        assert tuple(out.shape) == (3, 24, 24)
        assert float(out.min().item()) >= -1e-4
        assert float(out.max().item()) <= 1.0 + 1e-4

    def test_blur_preserves_constant_interior(self) -> None:
        # box blur preserves a constant in the interior (borders zero-pad)
        x = lucid.ones(3, 16, 16) * 0.5
        out = T.Blur(blur_limit=3, p=1.0)(x)
        interior = out[:, 2:14, 2:14]
        assert float((interior - 0.5).abs().max().item()) < 1e-4

    def test_median_constant(self) -> None:
        x = lucid.ones(3, 16, 16) * 0.3
        out = T.MedianBlur(blur_limit=3, p=1.0)(x)
        assert float((out - 0.3).abs().max().item()) < 1e-5

    def test_gaussnoise_changes(self) -> None:
        lucid.manual_seed(0)
        x = lucid.ones(3, 16, 16) * 0.5
        out = T.GaussNoise(p=1.0)(x)
        assert float((out - x).abs().max().item()) > 0.0

    def test_downscale_shape(self) -> None:
        out = T.Downscale(scale_min=0.5, scale_max=0.5, p=1.0)(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_noise_leaves_boxes(self) -> None:
        from lucid.utils.transforms._datatypes import to_xyxy

        s = {
            "image": T.Image(lucid.rand(3, 16, 16)),
            "boxes": T.BoundingBoxes(lucid.tensor([[1.0, 2.0, 8.0, 9.0]]), "xyxy", (16, 16)),
        }
        out = T.GaussNoise(p=1.0)(s)
        assert to_xyxy(out["boxes"]).numpy().reshape(-1).tolist() == [1.0, 2.0, 8.0, 9.0]


class TestReproducibility:
    def test_gaussnoise_seeded(self) -> None:
        lucid.manual_seed(4)
        a = T.GaussNoise(p=1.0)(lucid.rand(3, 16, 16))
        lucid.manual_seed(4)
        b = T.GaussNoise(p=1.0)(lucid.rand(3, 16, 16))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
