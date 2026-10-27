"""Crop / pad / dropout family (B4)."""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import to_xyxy


def _sample(h: int = 40, w: int = 40) -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, h, w)),
        "mask": T.Mask(lucid.rand(1, h, w)),
        "boxes": T.BoundingBoxes(lucid.tensor([[5.0, 5.0, 35.0, 35.0]]), "xyxy", (h, w)),
        "kp": T.Keypoints(lucid.tensor([[10.0, 10.0]]), (h, w)),
    }


class TestCropPad:
    def test_crop_shape(self) -> None:
        assert tuple(T.Crop(2, 2, 30, 34)(lucid.rand(3, 40, 40)).shape) == (3, 32, 28)

    def test_pad_if_needed(self) -> None:
        assert tuple(T.PadIfNeeded(50, 50)(lucid.rand(3, 40, 30)).shape) == (3, 50, 50)

    def test_pad_if_needed_no_op_when_large(self) -> None:
        out = T.PadIfNeeded(10, 10)(lucid.rand(3, 40, 40))
        assert tuple(out.shape) == (3, 40, 40)

    def test_random_sized_crop(self) -> None:
        lucid.manual_seed(0)
        out = T.RandomSizedCrop((20, 30), 64, 64)(lucid.rand(3, 40, 40))
        assert tuple(out.shape) == (3, 64, 64)

    def test_crop_and_pad(self) -> None:
        assert tuple(T.CropAndPad(4)(lucid.rand(3, 32, 32)).shape) == (3, 40, 40)
        assert tuple(T.CropAndPad(-4)(lucid.rand(3, 32, 32)).shape) == (3, 24, 24)

    def test_pad_multitarget_canvas(self) -> None:
        out = T.PadIfNeeded(50, 50)(_sample())
        assert out["boxes"].canvas_size == (50, 50)
        assert out["kp"].canvas_size == (50, 50)
        # box shifted by the centering pad (5 each side).
        assert to_xyxy(out["boxes"]).numpy().reshape(-1).tolist() == [
            10.0,
            10.0,
            40.0,
            40.0,
        ]

    @pytest.mark.parametrize(
        "position, dx, dy",
        [
            ("center", 5, 5),
            ("top_left", 0, 0),
            ("top_right", 10, 0),
            ("bottom_left", 0, 10),
            ("bottom_right", 10, 10),
        ],
    )
    def test_pad_position_moves_image_and_targets_together(
        self, position: str, dx: int, dy: int
    ) -> None:
        img = lucid.rand(3, 40, 40)
        sample = _sample()
        sample["image"] = T.Image(img)
        out = T.PadIfNeeded(50, 50, position=position)(sample)
        assert to_xyxy(out["boxes"]).numpy().reshape(-1).tolist() == [
            5.0 + dx,
            5.0 + dy,
            35.0 + dx,
            35.0 + dy,
        ]
        placed = out["image"].data[:, dy : dy + 40, dx : dx + 40]
        assert bool((placed == img).all())

    def test_pad_position_is_validated(self) -> None:
        with pytest.raises(ValueError, match="position"):
            T.PadIfNeeded(50, 50, position="middle")


class TestDropout:
    def test_coarse_dropout_blanks(self) -> None:
        out = T.CoarseDropout(max_holes=5, max_height=8, max_width=8, p=1.0)(lucid.ones(3, 32, 32))
        assert float(out.min().item()) == 0.0

    def test_coarse_dropout_p0(self) -> None:
        x = lucid.rand(3, 32, 32)
        out = T.CoarseDropout(p=0.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_grid_dropout(self) -> None:
        out = T.GridDropout(ratio=0.5, unit_size=8, p=1.0)(lucid.ones(3, 32, 32))
        assert float(out.min().item()) == 0.0

    def test_dropout_leaves_boxes(self) -> None:
        out = T.CoarseDropout(p=1.0)(_sample())
        assert to_xyxy(out["boxes"]).numpy().reshape(-1).tolist() == [
            5.0,
            5.0,
            35.0,
            35.0,
        ]


class TestValueScaling:
    def test_to_float(self) -> None:
        assert abs(float(T.ToFloat()(lucid.ones(3, 4, 4) * 255).max().item()) - 1.0) < 1e-5

    def test_from_float(self) -> None:
        assert abs(float(T.FromFloat()(lucid.ones(3, 4, 4)).max().item()) - 255.0) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
