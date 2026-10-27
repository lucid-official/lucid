"""Multi-target transforms — Image / Mask / BoundingBoxes / Keypoints.

A single transform moves an image, its mask, its boxes, and its
keypoints consistently; photometric transforms leave non-image targets
untouched; plain tensors pass through as images.
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import to_xyxy


def _sample(h: int = 100, w: int = 120) -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, h, w)),
        "mask": T.Mask(lucid.rand(1, h, w)),
        "boxes": T.BoundingBoxes(lucid.tensor([[10.0, 20.0, 60.0, 80.0]]), "xyxy", (h, w)),
        "kp": T.Keypoints(lucid.tensor([[10.0, 20.0], [60.0, 80.0]]), (h, w)),
    }


def _box(out: dict[str, object]) -> list[float]:
    bb = out["boxes"]
    assert isinstance(bb, T.BoundingBoxes)
    return to_xyxy(bb).numpy().reshape(-1).tolist()


def _kp(out: dict[str, object]) -> list[list[float]]:
    kp = out["kp"]
    assert isinstance(kp, T.Keypoints)
    return kp.data.numpy().tolist()


class TestDataTypes:
    def test_bbox_format_validation(self) -> None:
        with pytest.raises(ValueError, match="unknown format"):
            T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "nope", (10, 10))

    def test_xywh_roundtrip(self) -> None:
        b = T.BoundingBoxes(lucid.tensor([[10.0, 20.0, 30.0, 40.0]]), "xywh", (100, 100))
        assert to_xyxy(b).numpy().reshape(-1).tolist() == [10.0, 20.0, 40.0, 60.0]


class TestResize:
    def test_all_targets(self) -> None:
        out = T.Resize(50, 60)(_sample())
        assert tuple(out["image"].data.shape) == (3, 50, 60)
        assert tuple(out["mask"].data.shape) == (1, 50, 60)
        assert _box(out) == [5.0, 10.0, 30.0, 40.0]
        assert _kp(out) == [[5.0, 10.0], [30.0, 40.0]]
        assert out["boxes"].canvas_size == (50, 60)
        assert out["kp"].canvas_size == (50, 60)


class TestFlip:
    def test_hflip(self) -> None:
        out = T.HorizontalFlip(p=1.0)(_sample())
        assert _box(out) == [60.0, 20.0, 110.0, 80.0]
        assert _kp(out) == [[110.0, 20.0], [60.0, 80.0]]

    def test_vflip(self) -> None:
        out = T.VerticalFlip(p=1.0)(_sample())
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]
        assert _kp(out) == [[10.0, 80.0], [60.0, 20.0]]


class TestCrop:
    def test_center_crop(self) -> None:
        out = T.CenterCrop(80, 80)(_sample())
        assert tuple(out["image"].data.shape) == (3, 80, 80)
        assert _box(out) == [0.0, 10.0, 40.0, 70.0]
        assert _kp(out) == [[-10.0, 10.0], [40.0, 70.0]]  # keypoints not clipped
        assert out["boxes"].canvas_size == (80, 80)


class TestPhotometricUntouched:
    def test_normalize(self) -> None:
        s = _sample()
        out = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), max_pixel_value=1.0)(s)
        assert float((out["mask"].data - s["mask"].data).abs().max().item()) == 0.0
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]
        assert _kp(out) == [[10.0, 20.0], [60.0, 80.0]]

    def test_colorjitter(self) -> None:
        out = T.ColorJitter(0.4, 0.4, p=1.0)(_sample())
        assert _box(out) == [10.0, 20.0, 60.0, 80.0]


class TestPlainAndCompose:
    def test_plain_tensor(self) -> None:
        assert tuple(T.Resize(50, 60)(lucid.rand(3, 100, 120)).shape) == (3, 50, 60)

    def test_compose_multitarget(self) -> None:
        lucid.manual_seed(1)
        tf = T.Compose([T.Resize(80, 80), T.HorizontalFlip(p=1.0)])
        out = tf(_sample())
        assert tuple(out["image"].data.shape) == (3, 80, 80)
        assert out["boxes"].canvas_size == (80, 80)

    def test_no_image_raises(self) -> None:
        with pytest.raises(ValueError, match="no image"):
            T.Resize(5, 5)(
                {"boxes": T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "xyxy", (10, 10))}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
