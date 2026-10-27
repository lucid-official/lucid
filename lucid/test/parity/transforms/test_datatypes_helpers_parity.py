"""Parity for ``_datatypes.py`` bounding-box + keypoint geometric
helpers vs hand-computed expected coordinates (cross-checked against
Albumentations / OpenCV semantics where applicable).

All 14 helpers are pure coordinate transformations on axis-aligned
geometry — bit-exact parity expected (``atol = 1e-5``).

Covered (BBox):
    flip_boxes, resize_boxes, crop_boxes, pad_boxes,
    transpose_boxes, rot90_boxes, affine_boxes

Covered (Keypoints):
    flip_keypoints, resize_keypoints, crop_keypoints, pad_keypoints,
    transpose_keypoints, rot90_keypoints, affine_keypoints
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import functional as TF
from lucid.utils.transforms._datatypes import (
    affine_boxes,
    affine_keypoints,
    crop_boxes,
    crop_keypoints,
    flip_boxes,
    flip_keypoints,
    pad_boxes,
    pad_keypoints,
    resize_boxes,
    resize_keypoints,
    rot90_boxes,
    rot90_keypoints,
    transpose_boxes,
    transpose_keypoints,
)

pytest.importorskip("albumentations")
pytest.importorskip("cv2")


# ── helpers ─────────────────────────────────────────────────────────


def _make_boxes(coords_xyxy: list[list[float]], h: int = 100, w: int = 100) -> T.BoundingBoxes:
    """Build a ``BoundingBoxes`` (xyxy) on an ``(h, w)`` canvas."""
    return T.BoundingBoxes(lucid.tensor(coords_xyxy), "xyxy", (h, w))


def _make_kps(coords: list[list[float]], h: int = 100, w: int = 100) -> T.Keypoints:
    """Build a ``Keypoints`` instance; columns 0/1 are x, y; extras pass through."""
    return T.Keypoints(lucid.tensor(coords), (h, w))


# ── BBox: flip ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestFlipBoxes:
    def test_horizontal(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = flip_boxes(boxes, horizontal=True)
        # x → W - x → (70, 20, 90, 40)
        np.testing.assert_allclose(out.data.numpy(), [[70.0, 20.0, 90.0, 40.0]], atol=1e-5)
        assert out.canvas_size == (100, 100)

    def test_vertical(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = flip_boxes(boxes, horizontal=False)
        # y → H - y → (10, 60, 30, 80)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 60.0, 30.0, 80.0]], atol=1e-5)

    def test_horizontal_multi(self) -> None:
        boxes = _make_boxes([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 80.0, 90.0]], h=100, w=100)
        out = flip_boxes(boxes, horizontal=True)
        expected = [[90.0, 0.0, 100.0, 10.0], [20.0, 50.0, 50.0, 90.0]]
        np.testing.assert_allclose(out.data.numpy(), expected, atol=1e-5)


# ── BBox: resize ────────────────────────────────────────────────────


@pytest.mark.parity
class TestResizeBoxes:
    def test_double(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = resize_boxes(boxes, new_h=200, new_w=200)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 40.0, 60.0, 80.0]], atol=1e-5)
        assert out.canvas_size == (200, 200)

    def test_non_uniform(self) -> None:
        # 100x100 → 50x200: sx = 2.0, sy = 0.5
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = resize_boxes(boxes, new_h=50, new_w=200)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 10.0, 60.0, 20.0]], atol=1e-5)
        assert out.canvas_size == (50, 200)


# ── BBox: crop ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestCropBoxes:
    def test_inside(self) -> None:
        boxes = _make_boxes([[20.0, 30.0, 50.0, 60.0]], h=100, w=100)
        out = crop_boxes(boxes, top=10, left=15, height=70, width=70)
        np.testing.assert_allclose(out.data.numpy(), [[5.0, 20.0, 35.0, 50.0]], atol=1e-5)
        assert out.canvas_size == (70, 70)

    def test_clipped(self) -> None:
        # box spans the crop edge → clipped at 0 / width / height
        boxes = _make_boxes([[5.0, 5.0, 60.0, 60.0]], h=100, w=100)
        out = crop_boxes(boxes, top=10, left=10, height=40, width=40)
        # raw shifted: (-5, -5, 50, 50) → clip to (0, 0, 40, 40)
        np.testing.assert_allclose(out.data.numpy(), [[0.0, 0.0, 40.0, 40.0]], atol=1e-5)


# ── BBox: pad ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestPadBoxes:
    def test_translate(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = pad_boxes(boxes, left=5, top=8, new_h=108, new_w=110)
        np.testing.assert_allclose(out.data.numpy(), [[15.0, 28.0, 35.0, 48.0]], atol=1e-5)
        assert out.canvas_size == (108, 110)


# ── BBox: transpose ─────────────────────────────────────────────────


@pytest.mark.parity
class TestTransposeBoxes:
    def test_swap_axes(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=80)
        out = transpose_boxes(boxes)
        # (x1, y1, x2, y2) → (y1, x1, y2, x2); canvas (H, W) → (W, H)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 10.0, 40.0, 30.0]], atol=1e-5)
        assert out.canvas_size == (80, 100)


# ── BBox: rot90 ─────────────────────────────────────────────────────


@pytest.mark.parity
class TestRot90Boxes:
    def test_k0_noop(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = rot90_boxes(boxes, k=0)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0, 30.0, 40.0]], atol=1e-5)

    def test_k1_ccw(self) -> None:
        # H=100, W=100. CCW one quarter:
        # (x1, y1, x2, y2) → (y1, (W-1)-x2, y2, (W-1)-x1)
        # = (20, 69, 40, 89)
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = rot90_boxes(boxes, k=1)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 69.0, 40.0, 89.0]], atol=1e-5)
        assert out.canvas_size == (100, 100)

    def test_k2_full_round_trip(self) -> None:
        # k=2: (H, W) stays the same; box maps to (W-1-x2, H-1-y2, W-1-x1, H-1-y1)
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        out = rot90_boxes(boxes, k=2)
        expected = [[69.0, 59.0, 89.0, 79.0]]
        np.testing.assert_allclose(out.data.numpy(), expected, atol=1e-5)

    def test_k4_identity(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=80)
        out = rot90_boxes(boxes, k=4)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0, 30.0, 40.0]], atol=1e-5)
        assert out.canvas_size == (100, 80)


# ── BBox: affine ────────────────────────────────────────────────────


@pytest.mark.parity
class TestAffineBoxes:
    def test_identity(self) -> None:
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        eye = lucid.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        out = affine_boxes(boxes, eye, out_hw=(100, 100))
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0, 30.0, 40.0]], atol=1e-5)

    def test_translation(self) -> None:
        # Pure translation by (+5, +8); enlarge output canvas so no clip.
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        mat = lucid.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]])
        out = affine_boxes(boxes, mat, out_hw=(200, 200))
        np.testing.assert_allclose(out.data.numpy(), [[15.0, 28.0, 35.0, 48.0]], atol=1e-5)
        assert out.canvas_size == (200, 200)

    def test_90deg_rotation(self) -> None:
        # CCW 90° about origin: (x, y) → (-y, x).
        # Box (10, 20, 30, 40) corners → after rotation enclosing aabb is
        # x ∈ [-40, -20], y ∈ [10, 30]. Output canvas (100, 100) clips
        # the negative-x corners to 0.
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        mat = lucid.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = affine_boxes(boxes, mat, out_hw=(100, 100))
        # x clipped to 0; y unchanged
        np.testing.assert_allclose(out.data.numpy(), [[0.0, 10.0, 0.0, 30.0]], atol=1e-5)

    def test_rotation_via_helper(self) -> None:
        # ``rotation_matrix(angle_deg=90)`` follows the image-space CW
        # convention (matches cv2 ``getRotationMatrix2D``).  For (x, y)
        # rotated about (49.5, 49.5) by CW 90°: (x, y) → (y, 99 - x).
        # Four corners of (10, 20, 30, 40):
        #   (10, 20) → (20, 89)
        #   (30, 20) → (20, 69)
        #   (30, 40) → (40, 69)
        #   (10, 40) → (40, 89)
        # AABB: x ∈ [20, 40], y ∈ [69, 89]
        boxes = _make_boxes([[10.0, 20.0, 30.0, 40.0]], h=100, w=100)
        mat = TF.rotation_matrix(angle_deg=90.0, cx=49.5, cy=49.5)
        out = affine_boxes(boxes, mat, out_hw=(100, 100))
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 69.0, 40.0, 89.0]], atol=1e-4)


# ── Keypoints: flip ─────────────────────────────────────────────────


@pytest.mark.parity
class TestFlipKeypoints:
    def test_horizontal(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = flip_keypoints(kps, horizontal=True)
        np.testing.assert_allclose(out.data.numpy(), [[90.0, 20.0]], atol=1e-5)

    def test_vertical(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = flip_keypoints(kps, horizontal=False)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 80.0]], atol=1e-5)

    def test_extra_columns_preserved(self) -> None:
        # (x, y, visibility) — visibility column should pass through untouched
        kps = _make_kps([[10.0, 20.0, 1.0]], h=100, w=100)
        out = flip_keypoints(kps, horizontal=True)
        np.testing.assert_allclose(out.data.numpy(), [[90.0, 20.0, 1.0]], atol=1e-5)


# ── Keypoints: resize ───────────────────────────────────────────────


@pytest.mark.parity
class TestResizeKeypoints:
    def test_double(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = resize_keypoints(kps, new_h=200, new_w=200)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 40.0]], atol=1e-5)
        assert out.canvas_size == (200, 200)

    def test_non_uniform_with_extras(self) -> None:
        # 100x100 → 50x200: sx=2.0, sy=0.5; extra col is preserved
        kps = _make_kps([[10.0, 20.0, 0.7]], h=100, w=100)
        out = resize_keypoints(kps, new_h=50, new_w=200)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 10.0, 0.7]], atol=1e-5)


# ── Keypoints: crop ─────────────────────────────────────────────────


@pytest.mark.parity
class TestCropKeypoints:
    def test_translate_no_clip(self) -> None:
        # crop_keypoints does NOT clip — points outside the window keep
        # their (possibly negative) coords.
        kps = _make_kps([[20.0, 30.0], [5.0, 5.0]], h=100, w=100)
        out = crop_keypoints(kps, top=10, left=15, height=70, width=70)
        np.testing.assert_allclose(out.data.numpy(), [[5.0, 20.0], [-10.0, -5.0]], atol=1e-5)
        assert out.canvas_size == (70, 70)


# ── Keypoints: pad ──────────────────────────────────────────────────


@pytest.mark.parity
class TestPadKeypoints:
    def test_translate(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = pad_keypoints(kps, left=5, top=8, new_h=108, new_w=110)
        np.testing.assert_allclose(out.data.numpy(), [[15.0, 28.0]], atol=1e-5)
        assert out.canvas_size == (108, 110)


# ── Keypoints: transpose ────────────────────────────────────────────


@pytest.mark.parity
class TestTransposeKeypoints:
    def test_swap_axes(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=80)
        out = transpose_keypoints(kps)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 10.0]], atol=1e-5)
        assert out.canvas_size == (80, 100)

    def test_extras_preserved(self) -> None:
        kps = _make_kps([[10.0, 20.0, 0.5, 3.0]], h=100, w=80)
        out = transpose_keypoints(kps)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 10.0, 0.5, 3.0]], atol=1e-5)


# ── Keypoints: rot90 ────────────────────────────────────────────────


@pytest.mark.parity
class TestRot90Keypoints:
    def test_k0_noop(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = rot90_keypoints(kps, k=0)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0]], atol=1e-5)

    def test_k1_ccw(self) -> None:
        # (x, y) → (y, (W-1) - x); canvas (H, W) → (W, H)
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        out = rot90_keypoints(kps, k=1)
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 89.0]], atol=1e-5)
        assert out.canvas_size == (100, 100)

    def test_k4_identity(self) -> None:
        kps = _make_kps([[10.0, 20.0, 0.9]], h=100, w=80)
        out = rot90_keypoints(kps, k=4)
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0, 0.9]], atol=1e-5)
        assert out.canvas_size == (100, 80)


# ── Keypoints: affine ───────────────────────────────────────────────


@pytest.mark.parity
class TestAffineKeypoints:
    def test_identity(self) -> None:
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        eye = lucid.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        out = affine_keypoints(kps, eye, out_hw=(100, 100))
        np.testing.assert_allclose(out.data.numpy(), [[10.0, 20.0]], atol=1e-5)

    def test_translation_preserves_extras(self) -> None:
        kps = _make_kps([[10.0, 20.0, 1.0]], h=100, w=100)
        mat = lucid.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]])
        out = affine_keypoints(kps, mat, out_hw=(100, 100))
        np.testing.assert_allclose(out.data.numpy(), [[15.0, 28.0, 1.0]], atol=1e-5)
        assert out.canvas_size == (100, 100)

    def test_rotation_about_center(self) -> None:
        # ``rotation_matrix(angle_deg=90)`` is image-space CW (matches cv2).
        # (x, y) → (y, 99 - x) about (49.5, 49.5).  (10, 20) → (20, 89).
        kps = _make_kps([[10.0, 20.0]], h=100, w=100)
        mat = TF.rotation_matrix(angle_deg=90.0, cx=49.5, cy=49.5)
        out = affine_keypoints(kps, mat, out_hw=(100, 100))
        np.testing.assert_allclose(out.data.numpy(), [[20.0, 89.0]], atol=1e-4)
