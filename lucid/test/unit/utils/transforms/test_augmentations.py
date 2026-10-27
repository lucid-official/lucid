"""Randomized augmentations (Albumentations API) — geometric + color.

Shape correctness, p-extremes, manual_seed reproducibility, and the
photometric functional helpers.
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import functional as F


class TestGeometricAug:
    def test_random_resized_crop(self) -> None:
        out = T.RandomResizedCrop(64, 64)(lucid.rand(3, 80, 100))
        assert tuple(out.shape) == (3, 64, 64)

    def test_random_crop(self) -> None:
        out = T.RandomCrop(20, 24)(lucid.rand(3, 32, 40))
        assert tuple(out.shape) == (3, 20, 24)

    def test_hflip_p1(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((T.HorizontalFlip(p=1.0)(x) - F.hflip(x)).abs().max().item()) < 1e-6

    def test_vflip_p1(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((T.VerticalFlip(p=1.0)(x) - F.vflip(x)).abs().max().item()) < 1e-6


class TestColorJitter:
    def test_shape(self) -> None:
        out = T.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0)(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_zero_is_noop(self) -> None:
        x = lucid.rand(3, 16, 16)
        out = T.ColorJitter(0, 0, 0, 0, p=1.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_in_range(self) -> None:
        out = T.ColorJitter(0.5, 0.5, 0.5, 0.2, p=1.0)(lucid.rand(3, 16, 16))
        assert float(out.min().item()) >= -1e-5
        assert float(out.max().item()) <= 1.0 + 1e-5


class TestPhotometricFunctional:
    def test_brightness_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((F.adjust_brightness(x, 1.0) - x).abs().max().item()) < 1e-6

    def test_grayscale_constant_channels(self) -> None:
        g = F.rgb_to_grayscale(lucid.rand(3, 8, 8), keep_channels=True)
        assert float((g - g[0:1]).abs().max().item()) < 1e-6

    def test_hue_roundtrip(self) -> None:
        x = lucid.rand(3, 8, 8)
        back = F.adjust_hue(F.adjust_hue(x, 0.3), -0.3)
        assert float((back - x).abs().max().item()) < 1e-4


class TestReproducibility:
    def test_rrc_seeded(self) -> None:
        lucid.manual_seed(123)
        a = T.RandomResizedCrop(64, 64)(lucid.rand(3, 128, 128))
        lucid.manual_seed(123)
        b = T.RandomResizedCrop(64, 64)(lucid.rand(3, 128, 128))
        assert float((a - b).abs().max().item()) == 0.0

    def test_colorjitter_seeded(self) -> None:
        lucid.manual_seed(5)
        a = T.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0)(lucid.rand(3, 16, 16))
        lucid.manual_seed(5)
        b = T.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0)(lucid.rand(3, 16, 16))
        assert float((a - b).abs().max().item()) == 0.0

    def test_p_gate_seeded(self) -> None:
        lucid.manual_seed(2)
        a = T.HorizontalFlip(p=0.5)(lucid.rand(3, 8, 8))
        lucid.manual_seed(2)
        b = T.HorizontalFlip(p=0.5)(lucid.rand(3, 8, 8))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
