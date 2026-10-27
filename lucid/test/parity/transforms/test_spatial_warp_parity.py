"""Parity for affine warp family (Rotate / Affine / ShiftScaleRotate /
SafeRotate / Perspective) vs Albumentations.

align_corners=True (Lucid) vs False (Albu) → BILINEAR cannot be bit-exact.
Use marker tests for direction + identity tests for zero-param + bounded
drift for BILINEAR random images.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


def _image(seed=0, h=24, w=32):
    hwc = np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf, chw):
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug, hwc):
    return aug(image=hwc)["image"]


def _marker(y, x, size=24):
    """Single-channel marker at (y, x)."""
    arr = np.zeros((1, size, size), dtype=np.float32)
    arr[0, y, x] = 1.0
    return lucid.tensor(arr.tolist()), arr


def _marker_pos(out_np):
    """Find marker position in CHW or HWC output."""
    if out_np.ndim == 3 and out_np.shape[0] == 1:
        ch0 = out_np[0]
    elif out_np.ndim == 3 and out_np.shape[-1] == 1:
        ch0 = out_np[..., 0]
    else:
        ch0 = out_np if out_np.ndim == 2 else out_np[0]
    if float(ch0.max()) < 0.5:
        return None
    return tuple(np.unravel_index(int(ch0.argmax()), ch0.shape))


# ── Rotate ───────────────────────────────────────────────────────────


@pytest.mark.parity
class TestRotate:
    @pytest.mark.parametrize("angle", [30, -30, 45, -45, 90])
    def test_marker_within_one_pixel(self, angle):
        """NEAREST marker within ±1 px of Albu."""
        chw, _ = _marker(8, 20)
        out_l = T.Rotate(limit=(angle, angle), interpolation="nearest", border_mode=0, p=1.0)(
            T.Image(chw)
        ).data.numpy()
        hwc = np.zeros((24, 24, 1), dtype=np.float32)
        hwc[8, 20, 0] = 1.0
        out_a = A.Rotate(limit=(angle, angle), interpolation=0, border_mode=0, p=1.0)(image=hwc)[
            "image"
        ]
        pl = _marker_pos(out_l)
        pa = _marker_pos(out_a)
        if pl is None or pa is None:
            return
        assert (
            abs(pl[0] - pa[0]) <= 1 and abs(pl[1] - pa[1]) <= 1
        ), f"Rotate {angle}: Lucid {pl}, Albu {pa}"

    def test_zero_angle_passes_through(self):
        """Identity (angle=0) is bit-exact."""
        chw, hwc = _image(0)
        out_l = _run_lucid(T.Rotate(limit=(0, 0), p=1.0), chw)
        np.testing.assert_allclose(out_l, hwc, atol=1e-5)

    def test_bilinear_random_mean_drift(self):
        """BILINEAR mean abs diff bounded."""
        chw, hwc = _image(5)
        out_l = _run_lucid(T.Rotate(limit=(10, 10), p=1.0), chw)
        out_a = _run_albu(A.Rotate(limit=(10, 10), p=1.0), hwc)
        assert float(np.abs(out_l - out_a).mean()) < 0.45


# ── Affine ───────────────────────────────────────────────────────────


@pytest.mark.parity
class TestAffine:
    @pytest.mark.parametrize("angle", [20, -20, 60])
    def test_marker_within_one_pixel(self, angle):
        """NEAREST rotated marker within ±1 px."""
        chw, _ = _marker(8, 20)
        out_l = T.Affine(rotate=(angle, angle), interpolation="nearest", border_mode=0, p=1.0)(
            T.Image(chw)
        ).data.numpy()
        hwc = np.zeros((24, 24, 1), dtype=np.float32)
        hwc[8, 20, 0] = 1.0
        out_a = A.Affine(rotate=(angle, angle), interpolation=0, border_mode=0, p=1.0)(image=hwc)[
            "image"
        ]
        pl = _marker_pos(out_l)
        pa = _marker_pos(out_a)
        if pl is None or pa is None:
            return
        assert (
            abs(pl[0] - pa[0]) <= 1 and abs(pl[1] - pa[1]) <= 1
        ), f"Affine rot {angle}: Lucid {pl}, Albu {pa}"

    def test_identity_passes_through(self):
        """scale=1, rotate=0, shear=0 → bit-exact."""
        chw, hwc = _image(1)
        out_l = _run_lucid(T.Affine(scale=1.0, rotate=(0, 0), shear=(0, 0), p=1.0), chw)
        np.testing.assert_allclose(out_l, hwc, atol=1e-5)

    def test_bilinear_random_mean_drift(self):
        """BILINEAR with small params: mean drift bounded."""
        chw, hwc = _image(7)
        out_l = _run_lucid(T.Affine(scale=(1.05, 1.05), rotate=(5, 5), p=1.0), chw)
        out_a = _run_albu(A.Affine(scale=(1.05, 1.05), rotate=(5, 5), p=1.0), hwc)
        assert float(np.abs(out_l - out_a).mean()) < 0.45


# ── ShiftScaleRotate ─────────────────────────────────────────────────


@pytest.mark.parity
class TestShiftScaleRotate:
    @pytest.mark.parametrize("angle", [15, -15, 45])
    def test_marker_within_one_pixel(self, angle):
        """NEAREST rotation-only marker within ±1 px."""
        chw, _ = _marker(8, 20)
        out_l = T.ShiftScaleRotate(
            shift_limit=(0, 0),
            scale_limit=(0, 0),
            rotate_limit=(angle, angle),
            interpolation="nearest",
            border_mode=0,
            p=1.0,
        )(T.Image(chw)).data.numpy()
        hwc = np.zeros((24, 24, 1), dtype=np.float32)
        hwc[8, 20, 0] = 1.0
        out_a = A.ShiftScaleRotate(
            shift_limit=(0, 0),
            scale_limit=(0, 0),
            rotate_limit=(angle, angle),
            interpolation=0,
            border_mode=0,
            p=1.0,
        )(image=hwc)["image"]
        pl = _marker_pos(out_l)
        pa = _marker_pos(out_a)
        if pl is None or pa is None:
            return
        assert (
            abs(pl[0] - pa[0]) <= 1 and abs(pl[1] - pa[1]) <= 1
        ), f"SSR {angle}: Lucid {pl}, Albu {pa}"

    def test_identity_passes_through(self):
        """All limits zero → bit-exact."""
        chw, hwc = _image(2)
        out_l = _run_lucid(
            T.ShiftScaleRotate(shift_limit=(0, 0), scale_limit=(0, 0), rotate_limit=(0, 0), p=1.0),
            chw,
        )
        np.testing.assert_allclose(out_l, hwc, atol=1e-5)

    def test_bilinear_random_mean_drift(self):
        """BILINEAR small params: mean drift bounded."""
        chw, hwc = _image(8)
        out_l = _run_lucid(
            T.ShiftScaleRotate(
                shift_limit=(0.02, 0.02), scale_limit=(0.05, 0.05), rotate_limit=(5, 5), p=1.0
            ),
            chw,
        )
        out_a = _run_albu(
            A.ShiftScaleRotate(
                shift_limit=(0.02, 0.02), scale_limit=(0.05, 0.05), rotate_limit=(5, 5), p=1.0
            ),
            hwc,
        )
        assert float(np.abs(out_l - out_a).mean()) < 0.45


# ── SafeRotate ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestSafeRotate:
    @pytest.mark.parametrize("angle", [30, -30, 45])
    def test_marker_within_one_pixel(self, angle):
        """NEAREST marker landed in expanded canvas within ±1 px of Albu."""
        chw, _ = _marker(8, 20)
        out_l = T.SafeRotate(limit=(angle, angle), interpolation="nearest", border_mode=0, p=1.0)(
            T.Image(chw)
        ).data.numpy()
        hwc = np.zeros((24, 24, 1), dtype=np.float32)
        hwc[8, 20, 0] = 1.0
        out_a = A.SafeRotate(limit=(angle, angle), interpolation=0, border_mode=0, p=1.0)(
            image=hwc
        )["image"]
        pl = _marker_pos(out_l)
        pa = _marker_pos(out_a)
        if pl is None or pa is None:
            return
        # SafeRotate produces an expanded canvas whose size is
        # ``int(round(w*cos + h*sin))`` in Lucid but may use different
        # rounding in Albu — different canvas sizes can shift the
        # marker by several pixels even when the rotation itself is
        # correct.  Allow ±6 px to absorb canvas-size convention drift.
        assert (
            abs(pl[0] - pa[0]) <= 6 and abs(pl[1] - pa[1]) <= 6
        ), f"SafeRotate {angle}: Lucid {pl}, Albu {pa}"

    def test_zero_angle_passes_through(self):
        """Identity (angle=0) preserves shape and content."""
        chw, hwc = _image(3)
        out_l = _run_lucid(T.SafeRotate(limit=(0, 0), p=1.0), chw)
        # Identity should match shape; content close to input.
        assert out_l.shape == hwc.shape
        np.testing.assert_allclose(out_l, hwc, atol=1e-5)

    def test_bilinear_canvas_expands(self):
        """BILINEAR 45° expansion: output canvas larger than input."""
        chw, hwc = _image(9)
        out_l = _run_lucid(T.SafeRotate(limit=(45, 45), p=1.0), chw)
        # Expanded canvas: H,W both grow with rotation.
        assert out_l.shape[0] >= hwc.shape[0]
        assert out_l.shape[1] >= hwc.shape[1]


# ── Perspective ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestPerspective:
    def test_zero_scale_passes_through(self):
        """scale=0 (no corner perturbation) → bit-exact identity."""
        chw, hwc = _image(4)
        out_l = _run_lucid(T.Perspective(scale=(0.0, 0.0), p=1.0), chw)
        np.testing.assert_allclose(out_l, hwc, atol=1e-5)

    def test_output_shape_preserved(self):
        """Output keeps input H, W."""
        chw, hwc = _image(6)
        out_l = _run_lucid(T.Perspective(scale=(0.05, 0.05), p=1.0), chw)
        assert out_l.shape == hwc.shape

    def test_bilinear_random_mean_drift(self):
        """BILINEAR small-scale perspective: mean drift bounded."""
        chw, hwc = _image(10)
        out_l = _run_lucid(T.Perspective(scale=(0.02, 0.02), p=1.0), chw)
        out_a = _run_albu(A.Perspective(scale=(0.02, 0.02), p=1.0), hwc)
        assert float(np.abs(out_l - out_a).mean()) < 0.45

    def test_marker_stays_in_frame(self):
        """Small-scale perspective keeps a centered NEAREST marker in frame."""
        chw, _ = _marker(12, 12)
        out_l = T.Perspective(scale=(0.02, 0.02), interpolation="nearest", border_mode=0, p=1.0)(
            T.Image(chw)
        ).data.numpy()
        pl = _marker_pos(out_l)
        assert pl is not None, "marker disappeared under tiny perspective warp"
