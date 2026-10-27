"""G1 — detection-grade bbox: normalized formats + filtering + labels."""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms._datatypes import box_areas, filter_boxes, to_xyxy

# ── normalized formats ──────────────────────────────────────────────


class TestFormats:
    def test_yolo_to_xyxy(self) -> None:
        b = T.BoundingBoxes(lucid.tensor([[0.5, 0.5, 0.5, 0.5]]), "yolo", (100, 200))
        assert to_xyxy(b).numpy().reshape(-1).tolist() == [50.0, 25.0, 150.0, 75.0]

    def test_albumentations_to_xyxy(self) -> None:
        b = T.BoundingBoxes(lucid.tensor([[0.1, 0.2, 0.6, 0.8]]), "albumentations", (100, 200))
        got = to_xyxy(b).numpy().reshape(-1).tolist()
        assert got == pytest.approx([20.0, 20.0, 120.0, 80.0], abs=1e-3)

    def test_albu_aliases(self) -> None:
        assert (
            T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "pascal_voc", (8, 8)).format
            == "xyxy"
        )
        assert (
            T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 1.0, 1.0]]), "coco", (8, 8)).format == "xywh"
        )

    def test_yolo_roundtrip_through_hflip(self) -> None:
        b = {
            "image": T.Image(lucid.rand(3, 100, 200)),
            "boxes": T.BoundingBoxes(lucid.tensor([[0.25, 0.5, 0.2, 0.2]]), "yolo", (100, 200)),
        }
        out = T.HorizontalFlip(p=1.0)(b)
        cx = out["boxes"].data.numpy().reshape(-1).tolist()[0]
        assert abs(cx - 0.75) < 1e-5  # cx mirrored, still normalized


# ── filtering helpers ───────────────────────────────────────────────


class TestFilter:
    def test_drop_degenerate_and_small(self) -> None:
        b = T.BoundingBoxes(
            lucid.tensor(
                [
                    [10.0, 10.0, 40.0, 40.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 10.0, 12.0, 12.0],
                ]
            ),
            "xyxy",
            (100, 100),
            labels=lucid.tensor([1.0, 2.0, 3.0]),
        )
        out = filter_boxes(b, min_area=20.0)
        assert int(out.data.shape[0]) == 1
        assert out.labels.numpy().tolist() == [1.0]

    def test_min_visibility(self) -> None:
        b = T.BoundingBoxes(lucid.tensor([[40.0, 40.0, 90.0, 90.0]]), "xyxy", (100, 100))
        orig = box_areas(b)  # 2500
        # simulate crop to [0,50): visible area 10x10=100 → 100/2500 = 0.04
        cropped_box = T.BoundingBoxes(lucid.tensor([[40.0, 40.0, 50.0, 50.0]]), "xyxy", (50, 50))
        out = filter_boxes(cropped_box, orig_areas=orig, min_visibility=0.5)
        assert int(out.data.shape[0]) == 0


# ── Compose(bbox_params=...) end-to-end ─────────────────────────────


class TestComposeBboxParams:
    def _sample(self) -> dict[str, object]:
        return {
            "image": T.Image(lucid.rand(3, 100, 100)),
            "boxes": T.BoundingBoxes(
                lucid.tensor([[10.0, 10.0, 40.0, 40.0], [80.0, 80.0, 95.0, 95.0]]),
                "xyxy",
                (100, 100),
                labels=lucid.tensor([1.0, 2.0]),
            ),
        }

    def test_crop_drops_out_of_frame_box_and_label(self) -> None:
        tf = T.Compose([T.Crop(0, 0, 50, 50)], bbox_params=T.BboxParams(min_area=1.0))
        out = tf(self._sample())
        assert to_xyxy(out["boxes"]).numpy().tolist() == [[10.0, 10.0, 40.0, 40.0]]
        assert out["boxes"].labels.numpy().tolist() == [1.0]

    def test_min_visibility_drops_partial(self) -> None:
        s = {
            "image": T.Image(lucid.rand(3, 100, 100)),
            "boxes": T.BoundingBoxes(lucid.tensor([[40.0, 40.0, 90.0, 90.0]]), "xyxy", (100, 100)),
        }
        tf = T.Compose([T.Crop(0, 0, 50, 50)], bbox_params=T.BboxParams(min_visibility=0.5))
        out = tf(s)
        assert int(out["boxes"].data.shape[0]) == 0

    def test_no_bbox_params_keeps_all(self) -> None:
        tf = T.Compose([T.Crop(0, 0, 50, 50)])  # no filtering
        out = tf(self._sample())
        # both boxes present (2nd is degenerate-clipped but not dropped)
        assert int(out["boxes"].data.shape[0]) == 2

    def test_labels_preserved_through_geometric(self) -> None:
        s = self._sample()
        out = T.HorizontalFlip(p=1.0)(s)
        assert out["boxes"].labels.numpy().tolist() == [1.0, 2.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
