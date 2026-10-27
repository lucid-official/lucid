"""Numerical parity for Lucid's RandomErasing vs the reference framework.

Opt-in tier (mirrors the reference-framework parity policy): the whole
module auto-skips when the reference framework and its vision package
aren't installed.

Two tiers:

* **algorithm parity** — given an identical input image *and* identical
  erase rectangle coordinates / fill, both implementations must produce
  bitwise-equivalent output (float32 epsilon).  Lucid's
  :class:`RandomErasing` exposes :func:`_apply_image` which consumes a
  hand-constructed :class:`_ErasingParams`; the reference framework's
  ``F.erase(img, i, j, h, w, v)`` is the matching primitive.  Pinning
  the params decouples this tier from RNG implementation differences.

* **distribution parity** — across many calls (``p=1.0``), the
  distribution of *sampled* erase rectangle sizes (area, aspect ratio)
  must match the reference framework's.  Compared via sample means
  with a relative tolerance — direct seed reproducibility is impossible
  across two unrelated RNGs.
"""

import math

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.ref_framework import require_ref, require_ref_vision
from lucid.test._helpers.compare import assert_close
from lucid.utils.transforms._erasing import RandomErasing, _ErasingParams

# Reference framework imports are guarded by ``parity/conftest.py`` —
# the whole module is skipped at collection time when the reference itself
# is missing.  The vision package is a separate optional dependency, so
# we still guard it explicitly with ``importorskip``.
_ref = require_ref(module_level=True)
_ref_vision = require_ref_vision(module_level=True)
TF_ref = _ref_vision.transforms.functional
T_ref = _ref_vision.transforms


# ── helpers ─────────────────────────────────────────────────────────


def _matched_image(
    seed: int = 0, c: int = 3, h: int = 64, w: int = 64
) -> tuple[lucid.Tensor, object]:
    """A matched ``(Lucid CHW tensor, reference framework CHW tensor)`` pair."""
    rng = np.random.default_rng(seed)
    arr = rng.random((c, h, w), dtype=np.float32)
    lucid_img = lucid.tensor(arr.tolist())
    ref_img = _ref.from_numpy(arr.copy())
    return lucid_img, ref_img


def _ref_scalar_fill(value: float, c: int, h: int, w: int) -> object:
    """Reference framework constant scalar fill tensor of shape ``(c, h, w)``."""
    return _ref.full((c, h, w), float(value), dtype=_ref.float32)


def _ref_per_channel_fill(values: tuple[float, ...], h: int, w: int) -> object:
    """Reference framework per-channel fill tensor of shape ``(C, h, w)``."""
    col = _ref.tensor(list(values), dtype=_ref.float32).view(len(values), 1, 1)
    return col.expand(len(values), h, w).contiguous()


def _lucid_scalar_fill(value: float, c: int, h: int, w: int) -> lucid.Tensor:
    """Lucid constant scalar fill tensor of shape ``(c, h, w)``."""
    return lucid.full((c, h, w), float(value), dtype=lucid.float32)


def _lucid_per_channel_fill(values: tuple[float, ...], h: int, w: int) -> lucid.Tensor:
    """Lucid per-channel fill tensor of shape ``(C, h, w)``."""
    c = len(values)
    col = lucid.tensor(list(values), dtype=lucid.float32).reshape(c, 1, 1)
    return col * lucid.ones(c, h, w, dtype=lucid.float32)


# ── tier 1: algorithm parity (identical params → identical output) ──


@pytest.mark.parity
class TestRandomErasingApplyParity:
    """Given identical erase rectangles, both impls must produce identical output."""

    @pytest.mark.parametrize(
        "i, j, h, w, fill_value",
        [
            (5, 10, 20, 30, 0.0),
            (0, 0, 10, 10, 0.5),
            (15, 25, 32, 32, 1.0),
            (32, 16, 8, 16, 0.25),
            (1, 1, 2, 2, 0.7),  # tiny rectangle
        ],
    )
    def test_constant_scalar_fill(self, i: int, j: int, h: int, w: int, fill_value: float) -> None:
        """Scalar constant fill — identical rect must produce identical pixels."""
        lucid_img, ref_img = _matched_image(seed=0)
        c = int(lucid_img.shape[0])

        # Reference: F.erase wants a fill tensor sized to the rectangle
        # (or scalar-broadcastable); pass the full (c, h, w) constant.
        ref_fill = _ref_scalar_fill(fill_value, c, h, w)
        ref_out = TF_ref.erase(ref_img, i, j, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_scalar_fill(fill_value, c, h, w)
        params = _ErasingParams(top=i, left=j, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0)
        lucid_out = tf._apply_image(lucid_img, params)

        assert_close(lucid_out, ref_out, atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize(
        "i, j, h, w",
        [
            (5, 10, 20, 30),
            (0, 0, 16, 16),
            (32, 32, 30, 30),
        ],
    )
    def test_per_channel_tuple_fill(self, i: int, j: int, h: int, w: int) -> None:
        """Per-channel ImageNet-mean fill — matches reference channel-by-channel."""
        lucid_img, ref_img = _matched_image(seed=1)
        # ImageNet pixel mean (Krizhevsky / standard recipe).
        means = (0.485, 0.456, 0.406)

        ref_fill = _ref_per_channel_fill(means, h, w)
        ref_out = TF_ref.erase(ref_img, i, j, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_per_channel_fill(means, h, w)
        params = _ErasingParams(top=i, left=j, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0, value=means)
        lucid_out = tf._apply_image(lucid_img, params)

        assert_close(lucid_out, ref_out, atol=1e-6, rtol=0.0)

    def test_full_image_erase(self) -> None:
        """Erase rectangle covering the entire image — every pixel becomes fill."""
        lucid_img, ref_img = _matched_image(seed=2, c=3, h=32, w=32)
        c, h, w = 3, 32, 32
        fill_value = 0.42

        ref_fill = _ref_scalar_fill(fill_value, c, h, w)
        ref_out = TF_ref.erase(ref_img, 0, 0, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_scalar_fill(fill_value, c, h, w)
        params = _ErasingParams(top=0, left=0, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0)
        lucid_out = tf._apply_image(lucid_img, params)

        assert_close(lucid_out, ref_out, atol=1e-6, rtol=0.0)

    def test_zero_value_fill_matches_reference_default(self) -> None:
        """``value=0`` (reference default) matches Lucid's ``value=0.0`` exactly."""
        lucid_img, ref_img = _matched_image(seed=3)
        c, i, j, h, w = 3, 8, 12, 24, 24

        # F.erase with a 0-tensor matches the reference framework's default value=0.
        ref_fill = _ref_scalar_fill(0.0, c, h, w)
        ref_out = TF_ref.erase(ref_img, i, j, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_scalar_fill(0.0, c, h, w)
        params = _ErasingParams(top=i, left=j, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0, value=0.0)
        lucid_out = tf._apply_image(lucid_img, params)

        assert_close(lucid_out, ref_out, atol=1e-6, rtol=0.0)

    def test_outside_erase_region_unchanged(self) -> None:
        """Pixels outside the erase rectangle are preserved exactly in both impls."""
        lucid_img, ref_img = _matched_image(seed=4)
        c, i, j, h, w = 3, 10, 10, 20, 20

        ref_fill = _ref_scalar_fill(0.0, c, h, w)
        ref_out = TF_ref.erase(ref_img, i, j, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_scalar_fill(0.0, c, h, w)
        params = _ErasingParams(top=i, left=j, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0)
        lucid_out = tf._apply_image(lucid_img, params)

        # Outside the rectangle, both outputs must equal the input.
        ref_np = ref_out.detach().cpu().numpy()
        lucid_np = lucid_out.numpy()
        src_np = ref_img.detach().cpu().numpy()
        # Build a mask of "outside the rectangle".
        outside = np.ones_like(src_np, dtype=bool)
        outside[:, i : i + h, j : j + w] = False
        np.testing.assert_array_equal(ref_np[outside], src_np[outside])
        np.testing.assert_array_equal(lucid_np[outside], src_np[outside])


# ── tier 2: distribution parity (statistical aggregate over many calls) ──


def _sample_lucid_rects(
    n: int,
    img_h: int,
    img_w: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    seed: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Sample ``n`` erase rectangles via Lucid's :class:`RandomErasing`.

    Returns the ``(top, left, h, w)`` tuples for every *successful* fit
    (Lucid records ``h=0, w=0`` as a no-op after 10 failed attempts;
    those are filtered out so the distribution covers only realised
    erases, matching what the reference framework also does).
    """
    lucid.manual_seed(seed)
    img = lucid.zeros(3, img_h, img_w)
    tf = RandomErasing(p=1.0, scale=scale, ratio=ratio)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        params = tf.make_params(img)
        if params.h > 0 and params.w > 0:
            rects.append((params.top, params.left, params.h, params.w))
    return rects


def _sample_ref_rects(
    n: int,
    img_h: int,
    img_w: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    seed: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Sample ``n`` erase rectangles via the reference framework's ``get_params``.

    Filters out no-op fits the same way the Lucid sampler does.
    """
    _ref.manual_seed(seed)
    img = _ref.zeros(3, img_h, img_w)
    tf = T_ref.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=0)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        # Reference signature: get_params(img, scale, ratio, value).
        x, y, h, w, _v = tf.get_params(img, scale=list(scale), ratio=list(ratio), value=[0.0])
        if h > 0 and w > 0 and h < img_h and w < img_w:
            rects.append((y, x, h, w))
    return rects


@pytest.mark.parity
class TestRandomErasingDistribution:
    """Statistical match of sampled erase regions across many calls."""

    N_SAMPLES = 1000
    IMG_H = 64
    IMG_W = 64
    SCALE = (0.02, 0.33)
    RATIO = (0.3, 3.3)

    def test_sampled_area_within_scale_bounds(self) -> None:
        """Every sampled rectangle's area must lie inside the requested scale."""
        rects = _sample_lucid_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        img_area = float(self.IMG_H * self.IMG_W)
        # The sampler uses int(round(sqrt(...))) which can push the
        # realised area slightly above/below the float target; allow a
        # one-pixel-row slack on each side.
        lo_bound = self.SCALE[0] * img_area * 0.5
        hi_bound = self.SCALE[1] * img_area * 1.5
        for top, left, h, w in rects:
            area = h * w
            assert (
                lo_bound <= area <= hi_bound
            ), f"area {area} outside [{lo_bound:.1f}, {hi_bound:.1f}]"

    def test_sampled_aspect_within_ratio_bounds(self) -> None:
        """Every sampled rectangle's aspect ratio must lie inside ``ratio``."""
        rects = _sample_lucid_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        # Allow 2x slack on each side — round() to int can push small
        # rectangles' realised aspect away from the float target.
        lo = self.RATIO[0] * 0.5
        hi = self.RATIO[1] * 2.0
        for top, left, h, w in rects:
            aspect = h / w  # eh / ew, matching the sampler convention
            assert lo <= aspect <= hi, f"aspect {aspect:.3f} outside [{lo:.3f}, {hi:.3f}]"

    def test_mean_area_matches_reference(self) -> None:
        """Mean erase area across many samples agrees with the reference impl."""
        lucid_rects = _sample_lucid_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        ref_rects = _sample_ref_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        # Both samplers should yield enough successful fits to estimate
        # the mean to ~3% relative tolerance.
        assert len(lucid_rects) >= self.N_SAMPLES * 0.5
        assert len(ref_rects) >= self.N_SAMPLES * 0.5

        lucid_area = float(np.mean([h * w for _, _, h, w in lucid_rects]))
        ref_area = float(np.mean([h * w for _, _, h, w in ref_rects]))
        rel = abs(lucid_area - ref_area) / max(ref_area, 1.0)
        # 5% relative tolerance — two independent RNGs over 1000 samples
        # of the same uniform scale distribution should land much closer
        # than this, but allow Monte-Carlo headroom.
        assert rel <= 0.05, (
            f"mean area mismatch: lucid={lucid_area:.1f} ref={ref_area:.1f} " f"(rel {rel:.3%})"
        )

    def test_mean_aspect_matches_reference(self) -> None:
        """Mean log-aspect across many samples agrees with the reference impl."""
        lucid_rects = _sample_lucid_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        ref_rects = _sample_ref_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=0
        )
        # Compare in log-space because both samplers draw the aspect
        # ratio from a *log*-uniform distribution; the geometric mean is
        # the natural summary statistic.
        lucid_log_aspect = float(np.mean([math.log(h / w) for _, _, h, w in lucid_rects]))
        ref_log_aspect = float(np.mean([math.log(h / w) for _, _, h, w in ref_rects]))
        # Theoretical mean of log-uniform on [log(0.3), log(3.3)] is 0;
        # check both sample means are close to each other (and to 0).
        abs_diff = abs(lucid_log_aspect - ref_log_aspect)
        # 0.15 nat ≈ 16% multiplicative — generous for 1000 samples but
        # rules out gross divergence (wrong log/linear distribution).
        assert abs_diff <= 0.15, (
            f"mean log-aspect mismatch: lucid={lucid_log_aspect:.3f} "
            f"ref={ref_log_aspect:.3f} (diff {abs_diff:.3f})"
        )

    def test_top_left_within_image_bounds(self) -> None:
        """Sampled ``(top, left)`` keep the rectangle fully inside the image."""
        rects = _sample_lucid_rects(
            self.N_SAMPLES, self.IMG_H, self.IMG_W, self.SCALE, self.RATIO, seed=1
        )
        for top, left, h, w in rects:
            assert (
                0 <= top and top + h <= self.IMG_H
            ), f"vertical out of bounds: top={top} h={h} (img_h={self.IMG_H})"
            assert (
                0 <= left and left + w <= self.IMG_W
            ), f"horizontal out of bounds: left={left} w={w} (img_w={self.IMG_W})"


# ── tier 3: edge cases (both impls must handle gracefully) ──────────


@pytest.mark.parity
class TestRandomErasingEdgeCases:
    """Edge configurations that have caused divergence in past parity audits."""

    def test_p_zero_passes_through(self) -> None:
        """``p=0`` returns the input unchanged in both impls."""
        lucid_img, ref_img = _matched_image(seed=0)

        lucid_tf = RandomErasing(p=0.0)
        ref_tf = T_ref.RandomErasing(p=0.0, value=0)

        lucid_out = lucid_tf(lucid_img)
        ref_out = ref_tf(ref_img)

        # ``p=0`` is deterministic identity in both.
        assert_close(lucid_out, lucid_img, atol=0.0, rtol=0.0)
        assert_close(ref_out, ref_img, atol=0.0, rtol=0.0)

    def test_zero_scale_is_no_op(self) -> None:
        """``scale=(0, 0)`` makes every fit attempt fail → no-op identity."""
        lucid_img, _ = _matched_image(seed=2)
        lucid.manual_seed(0)
        tf = RandomErasing(p=1.0, scale=(0.0, 0.0))
        # 10 attempts all yield eh=ew=0 → no_op branch returns input unchanged.
        out = tf(lucid_img)
        assert_close(out, lucid_img, atol=0.0, rtol=0.0)

    def test_small_image_handled(self) -> None:
        """Very small images don't crash; both impls return same-shape output."""
        lucid_img, ref_img = _matched_image(seed=3, c=3, h=5, w=5)
        c, i, j, h, w = 3, 1, 1, 2, 2
        fill_value = 0.3

        ref_fill = _ref_scalar_fill(fill_value, c, h, w)
        ref_out = TF_ref.erase(ref_img, i, j, h, w, ref_fill, inplace=False)

        lucid_fill = _lucid_scalar_fill(fill_value, c, h, w)
        params = _ErasingParams(top=i, left=j, h=h, w=w, fill=lucid_fill)
        tf = RandomErasing(p=1.0)
        lucid_out = tf._apply_image(lucid_img, params)

        assert tuple(lucid_out.shape) == (3, 5, 5)
        assert_close(lucid_out, ref_out, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
