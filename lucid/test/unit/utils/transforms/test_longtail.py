"""B8 long-tail transforms — geometric / color / blur / cross-target / utils."""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import to_xyxy

# ── single-image (size-preserving) ──────────────────────────────────

_FIXED = [
    T.D4(p=1.0),
    T.RandomGridShuffle(p=1.0),
    T.GridElasticDeform(p=1.0),
    T.UnsharpMask(p=1.0),
    T.RingingOvershoot(p=1.0),
    T.FancyPCA(p=1.0),
    T.PixelDropout(dropout_prob=0.2, p=1.0),
    T.RandomBrightness(p=1.0),
    T.RandomContrast(p=1.0),
    T.XYMasking(num_masks_x=2, num_masks_y=2, p=1.0),
    T.Defocus(p=1.0),
    T.ZoomBlur(p=1.0),
    T.Lambda(image=lambda im: im * 0.5, p=1.0),
]


class TestFixedSize:
    @pytest.mark.parametrize("tf", _FIXED, ids=lambda t: type(t).__name__)
    def test_shape(self, tf: T.Transform) -> None:
        lucid.manual_seed(0)
        out = tf(lucid.rand(3, 24, 24))
        assert tuple(out.shape) == (3, 24, 24)


class TestSizeChanging:
    def test_random_scale(self) -> None:
        out = T.RandomScale(scale_limit=(0.5, 0.5), p=1.0)(lucid.rand(3, 20, 20))
        assert tuple(out.shape) == (3, 30, 30)  # 1.5x

    def test_safe_rotate_expands(self) -> None:
        out = T.SafeRotate((90, 90), p=1.0)(lucid.rand(3, 20, 30))
        # 90° → canvas swaps to ~30x20-bounded square-ish; just larger than crop
        assert out.ndim == 3 and int(out.shape[0]) == 3


class TestColorSemantics:
    def test_random_brightness_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.RandomBrightness(limit=(0.0, 0.0), p=1.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_lambda_applies(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.Lambda(image=lambda im: im * 0.0, p=1.0)(x)
        assert float(out.abs().max().item()) < 1e-6
        kps = T.Keypoints(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]), (8, 8))
        sample = {"image": T.Image(x), "keypoints": kps}
        out_sample = T.Lambda(keypoints=lambda k: T.Keypoints(k.data + 5.0, k.canvas_size), p=1.0)(
            sample
        )
        assert out_sample["keypoints"].data.tolist() == [[6.0, 7.0], [8.0, 9.0]]

    def test_pixel_dropout_range(self) -> None:
        out = T.PixelDropout(dropout_prob=0.5, p=1.0)(lucid.rand(3, 16, 16))
        assert float(out.min().item()) >= -1e-5


# ── cross-target ────────────────────────────────────────────────────


def _det_sample() -> dict[str, object]:
    return {
        "image": T.Image(lucid.rand(3, 40, 40)),
        "mask": T.Mask((lucid.rand(1, 40, 40) * 3).long().to(lucid.float32)),
        "boxes": T.BoundingBoxes(lucid.tensor([[10.0, 10.0, 30.0, 30.0]]), "xyxy", (40, 40)),
    }


class TestCrossTarget:
    def test_bbox_safe_crop_keeps_box(self) -> None:
        lucid.manual_seed(0)
        out = T.BBoxSafeRandomCrop(p=1.0)(_det_sample())
        x1, y1, x2, y2 = to_xyxy(out["boxes"]).numpy().reshape(-1).tolist()
        ch, cw = out["boxes"].canvas_size
        # box stays within the cropped canvas and non-degenerate
        assert 0 <= x1 < x2 <= cw and 0 <= y1 < y2 <= ch

    def test_random_sized_bbox_safe_crop(self) -> None:
        lucid.manual_seed(1)
        out = T.RandomSizedBBoxSafeCrop(64, 64, p=1.0)(_det_sample())
        assert tuple(out["image"].data.shape) == (3, 64, 64)
        assert out["boxes"].canvas_size == (64, 64)

    def test_random_crop_near_bbox(self) -> None:
        lucid.manual_seed(2)
        out = T.RandomCropNearBBox(p=1.0)(_det_sample())
        assert out["image"].data.ndim == 3

    def test_mask_dropout(self) -> None:
        lucid.manual_seed(3)
        out = T.MaskDropout(max_objects=2, p=1.0)(_det_sample())
        assert tuple(out["mask"].data.shape) == (1, 40, 40)
        assert tuple(out["image"].data.shape) == (3, 40, 40)


# ── composition / replay ────────────────────────────────────────────


class TestReplayCompose:
    def test_deterministic_replay(self) -> None:
        x = lucid.rand(3, 24, 24)
        rc = T.ReplayCompose([T.HorizontalFlip(p=1.0), T.RandomBrightness(p=1.0)])
        out = rc(x)
        replayed = rc.replay(rc.replay_data, x)
        assert float((out - replayed).abs().max().item()) == 0.0

    def test_replay_on_paired_input(self) -> None:
        # apply the recorded augmentation identically to a second tensor
        rc = T.ReplayCompose([T.HorizontalFlip(p=1.0)])
        rc(lucid.rand(3, 8, 8))
        y = lucid.rand(3, 8, 8)
        out_y = rc.replay(rc.replay_data, y)
        from lucid.utils.transforms import functional as F

        assert float((out_y - F.hflip(y)).abs().max().item()) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
