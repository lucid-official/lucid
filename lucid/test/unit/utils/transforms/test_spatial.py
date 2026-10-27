"""Spatial augmentations — flips/transpose/rot90 + affine warps.

Shape correctness, canvas tracking, multi-target consistency, and
manual_seed reproducibility for the warp family.
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import to_xyxy


def _box_sample(h: int = 40, w: int = 40) -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, h, w)),
        "mask": T.Mask(lucid.rand(1, h, w)),
        "boxes": T.BoundingBoxes(lucid.tensor([[5.0, 5.0, 35.0, 35.0]]), "xyxy", (h, w)),
        "kp": T.Keypoints(lucid.tensor([[20.0, 5.0]]), (h, w)),
    }


def _box(out: dict[str, object]) -> list[float]:
    return to_xyxy(out["boxes"]).numpy().reshape(-1).tolist()


class TestExactSpatial:
    def test_transpose_shape_and_canvas(self) -> None:
        out = T.Transpose(p=1.0)(_box_sample(32, 40))
        assert tuple(out["image"].data.shape) == (3, 40, 32)
        assert out["boxes"].canvas_size == (40, 32)

    def test_transpose_box_swaps_xy(self) -> None:
        out = T.Transpose(p=1.0)(_box_sample(40, 40))
        assert _box(out) == [5.0, 5.0, 35.0, 35.0]  # symmetric box, swapped

    def test_flip_runs(self) -> None:
        lucid.manual_seed(0)
        out = T.Flip(p=1.0)(lucid.rand(3, 16, 20))
        assert tuple(out.shape) == (3, 16, 20)

    def test_rotate90_canvas(self) -> None:
        lucid.manual_seed(1)
        out = T.RandomRotate90(p=1.0)(_box_sample(32, 48))
        # k is random; canvas is either (32,48) or (48,32)
        assert out["boxes"].canvas_size in {(32, 48), (48, 32)}


class TestWarpFamily:
    @pytest.mark.parametrize(
        "tf",
        [
            T.Rotate(45, p=1.0),
            T.ShiftScaleRotate(p=1.0),
            T.Affine(scale=(0.8, 1.2), rotate=30, shear=10, p=1.0),
            T.Perspective(p=1.0),
        ],
    )
    def test_shape_preserved(self, tf: T.Transform) -> None:
        lucid.manual_seed(0)
        out = tf(lucid.rand(3, 32, 40))
        assert tuple(out.shape) == (3, 32, 40)

    def test_rotate_multitarget(self) -> None:
        lucid.manual_seed(0)
        out = T.Rotate((90, 90), p=1.0)(_box_sample(40, 40))
        assert tuple(out["image"].data.shape) == (3, 40, 40)
        assert tuple(out["mask"].data.shape) == (1, 40, 40)
        # 90° rotation of a centered box stays a valid box within canvas.
        x1, y1, x2, y2 = _box(out)
        assert 0 <= x1 <= x2 <= 40 and 0 <= y1 <= y2 <= 40

    def test_rotate_zero_is_identity_ish(self) -> None:
        x = lucid.rand(3, 16, 16)
        out = T.Rotate((0, 0), p=1.0)(x)
        assert float((out - x).abs().max().item()) < 1e-4


class TestReproducibility:
    def test_affine_seeded(self) -> None:
        lucid.manual_seed(7)
        a = T.Affine(scale=(0.8, 1.2), rotate=30, p=1.0)(lucid.rand(3, 24, 24))
        lucid.manual_seed(7)
        b = T.Affine(scale=(0.8, 1.2), rotate=30, p=1.0)(lucid.rand(3, 24, 24))
        assert float((a - b).abs().max().item()) == 0.0


@pytest.mark.parametrize(("src_points", "dst_points"), [(4, 3), (4, 5), (5, 5)])
def test_perspective_matrix_wants_four_corners_a_side(src_points: int, dst_points: int) -> None:
    """Anything but four corners a side says so, naming both counts.

    A fifth destination used to be dropped without a word, and five on
    each side failed inside the solver as a non-square matrix.
    """
    import lucid.utils.transforms.functional as F

    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 2.0]]
    with pytest.raises(ValueError, match=f"got {src_points} and {dst_points}$"):
        F.perspective_matrix(square[:src_points], square[:dst_points])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
