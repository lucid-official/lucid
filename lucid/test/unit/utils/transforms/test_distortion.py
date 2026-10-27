"""Displacement-field distortions — Elastic / Grid / Optical."""

import pytest

import lucid
import lucid.utils.transforms as T


def _sample(h: int = 40, w: int = 40) -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, h, w)),
        "mask": T.Mask(lucid.rand(1, h, w)),
        "boxes": T.BoundingBoxes(lucid.tensor([[5.0, 5.0, 35.0, 35.0]]), "xyxy", (h, w)),
        "kp": T.Keypoints(lucid.tensor([[20.0, 20.0]]), (h, w)),
    }


class TestDistortionShapes:
    @pytest.mark.parametrize(
        "tf",
        [
            T.ElasticTransform(alpha=30, sigma=4, p=1.0),
            T.GridDistortion(p=1.0),
            T.OpticalDistortion(distort_limit=0.2, p=1.0),
        ],
    )
    def test_image_shape(self, tf: T.Transform) -> None:
        lucid.manual_seed(0)
        out = tf(lucid.rand(3, 32, 40))
        assert tuple(out.shape) == (3, 32, 40)

    def test_multitarget(self) -> None:
        lucid.manual_seed(1)
        out = T.ElasticTransform(alpha=20, sigma=4, p=1.0)(_sample())
        assert tuple(out["image"].data.shape) == (3, 40, 40)
        assert tuple(out["mask"].data.shape) == (1, 40, 40)
        assert out["boxes"].canvas_size == (40, 40)
        assert out["kp"].canvas_size == (40, 40)

    def test_optical_zero_is_near_identity(self) -> None:
        x = lucid.rand(3, 24, 24)
        out = T.OpticalDistortion(distort_limit=(0.0, 0.0), shift_limit=(0.0, 0.0), p=1.0)(x)
        assert float((out - x).abs().max().item()) < 1e-4

    def test_grid_canvas_preserved(self) -> None:
        lucid.manual_seed(2)
        out = T.GridDistortion(p=1.0)(_sample(48, 32))
        assert out["boxes"].canvas_size == (48, 32)


class TestReproducibility:
    def test_elastic_seeded(self) -> None:
        lucid.manual_seed(7)
        a = T.ElasticTransform(alpha=30, sigma=4, p=1.0)(lucid.rand(3, 24, 24))
        lucid.manual_seed(7)
        b = T.ElasticTransform(alpha=30, sigma=4, p=1.0)(lucid.rand(3, 24, 24))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
