"""Numerical parity for Lucid's dropout / crop / composition vs Albumentations.

Opt-in tier — auto-skips when ``albumentations`` / ``cv2`` aren't installed.

Three families with very different parity expectations:

* **Crop / pad (deterministic)** — ``Crop``, ``PadIfNeeded``,
  ``CropAndPad`` are pure indexing / constant-pad slicing.  Lucid and
  Albumentations land on the same pixels modulo float-cast: bit-exact
  agreement (``atol=0``).

* **Dropout (mask paste)** — ``CoarseDropout`` / ``GridDropout`` blank
  rectangular regions with a constant fill.  The two RNGs disagree on
  hole placement, so we *cannot* compare pixels.  Instead we pin
  the contract that both impls promise:

      - the dropped pixels are exactly ``fill_value`` (no blending),
      - the count of holes / size of holes lies in the requested range,
      - GridDropout's hole pattern is periodic at ``unit_size`` and
        every hole spans ``round(unit_size * ratio)`` pixels.

* **Crop (stochastic)** — ``RandomSizedCrop``, ``BBoxSafeRandomCrop``,
  ``RandomSizedBBoxSafeCrop``, ``RandomCropNearBBox`` cannot reach
  pixel parity (different RNGs).  We assert shape / bounds / box
  containment invariants instead.

* **Composition (meta)** — ``Compose`` / ``OneOf`` / ``SomeOf`` /
  ``Sequential`` / ``OneOrOther`` / ``ReplayCompose`` orchestrate
  children.  These are not pixel-comparable to Albu directly; we test
  *structural correctness* (correct number of stages, probability gates
  respected, replay reproducibility) using Lucid-only fixtures, and
  spot-check ``Compose`` against Albu where both reduce to a
  deterministic pipeline.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


# ── helpers ─────────────────────────────────────────────────────────


def _image(seed: int = 0, h: int = 40, w: int = 40, c: int = 3) -> tuple[lucid.Tensor, np.ndarray]:
    """A matched ``(Lucid CHW tensor, Albu HWC array)`` random image pair."""
    hwc = np.random.default_rng(seed).random((h, w, c), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _const_image(
    value: float = 0.5, h: int = 40, w: int = 40, c: int = 3
) -> tuple[lucid.Tensor, np.ndarray]:
    """A matched constant-fill image pair — used for dropout mask tests."""
    hwc = np.full((h, w, c), value, dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid_img(tf: T.Transform, chw: lucid.Tensor) -> np.ndarray:
    """Apply a Lucid transform to a CHW tensor; return HWC numpy."""
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug: object, hwc: np.ndarray) -> np.ndarray:
    """Apply an Albumentations transform; return HWC numpy."""
    return aug(image=hwc)["image"]  # type: ignore[operator]


# ── tier 1: deterministic crop / pad — bit-exact ────────────────────


@pytest.mark.parity
class TestCropDeterministic:
    """Pure indexing / constant-pad ops — both impls land on the same pixels."""

    @pytest.mark.parametrize(
        "x_min, y_min, x_max, y_max",
        [
            (0, 0, 20, 20),
            (2, 3, 18, 22),
            (5, 5, 35, 30),
            (0, 0, 40, 40),  # full-image (identity)
        ],
    )
    def test_crop_exact(self, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        """``Crop`` returns exactly ``image[y_min:y_max, x_min:x_max]``."""
        chw, hwc = _image(seed=0, h=40, w=40)
        got = _run_lucid_img(T.Crop(x_min, y_min, x_max, y_max, p=1.0), chw)
        ref = _run_albu(A.Crop(x_min, y_min, x_max, y_max, p=1.0), hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_crop_output_dims_match(self) -> None:
        """Output dimensions match ``(y_max - y_min, x_max - x_min)``."""
        chw, _ = _image(seed=1, h=50, w=60)
        x_min, y_min, x_max, y_max = 5, 7, 45, 35
        out = T.Crop(x_min, y_min, x_max, y_max, p=1.0)(T.Image(chw)).data
        # Lucid stores (C, H, W); the Crop's _hw returns (y_max-y_min, x_max-x_min).
        assert tuple(out.shape) == (3, y_max - y_min, x_max - x_min)

    @pytest.mark.parametrize(
        "min_h, min_w, src_h, src_w",
        [
            (20, 16, 10, 8),  # needs padding both axes
            (30, 30, 30, 30),  # no padding needed (identity)
            (24, 18, 12, 18),  # only height needs padding
        ],
    )
    def test_pad_if_needed_centered(self, min_h: int, min_w: int, src_h: int, src_w: int) -> None:
        """``PadIfNeeded`` centers + constant-pads to ``(min_h, min_w)``."""
        chw, hwc = _image(seed=2, h=src_h, w=src_w)
        got = _run_lucid_img(
            T.PadIfNeeded(min_height=min_h, min_width=min_w, value=0.0, p=1.0), chw
        )
        ref = _run_albu(
            A.PadIfNeeded(
                min_height=min_h,
                min_width=min_w,
                border_mode=0,
                fill=0.0,
                position="center",
                p=1.0,
            ),
            hwc,
        )
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_pad_if_needed_no_pad_is_identity(self) -> None:
        """When already large enough, ``PadIfNeeded`` is the identity."""
        chw, _ = _image(seed=3, h=30, w=30)
        out = T.PadIfNeeded(min_height=20, min_width=20, p=1.0)(T.Image(chw)).data
        np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("px", [1, 2, 4, 8])
    def test_crop_and_pad_positive(self, px: int) -> None:
        """``CropAndPad(px=+k)`` pads ``k`` pixels on every side with constant fill."""
        chw, hwc = _image(seed=4, h=16, w=20)
        got = _run_lucid_img(T.CropAndPad(px=px, value=0.0, p=1.0), chw)
        ref = _run_albu(
            A.CropAndPad(
                px=px,
                fill=0.0,
                sample_independently=False,
                keep_size=False,
                p=1.0,
            ),
            hwc,
        )
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("px", [-1, -2, -3])
    def test_crop_and_pad_negative(self, px: int) -> None:
        """``CropAndPad(px=-k)`` crops ``k`` pixels off every side."""
        chw, hwc = _image(seed=5, h=20, w=24)
        got = _run_lucid_img(T.CropAndPad(px=px, p=1.0), chw)
        ref = _run_albu(
            A.CropAndPad(
                px=px,
                sample_independently=False,
                keep_size=False,
                p=1.0,
            ),
            hwc,
        )
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_crop_and_pad_zero_is_identity(self) -> None:
        """``CropAndPad(px=0)`` is the identity (no crop, no pad)."""
        chw, _ = _image(seed=6, h=12, w=14)
        out = T.CropAndPad(px=0, p=1.0)(T.Image(chw)).data
        np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0, rtol=0.0)


# ── tier 2: stochastic crops — shape / bounds invariants only ───────


@pytest.mark.parity
class TestCropStochasticShape:
    """Random crops can't reach pixel parity — check shape / bound contracts."""

    def test_random_sized_crop_output_size(self) -> None:
        """``RandomSizedCrop`` always returns ``(C, height, width)``."""
        chw, _ = _image(seed=10, h=40, w=40)
        tf = T.RandomSizedCrop(min_max_height=(10, 30), height=12, width=16, p=1.0)
        for seed in range(20):
            lucid.manual_seed(seed)
            out = tf(T.Image(chw)).data
            assert tuple(out.shape) == (
                3,
                12,
                16,
            ), f"seed {seed}: shape {tuple(out.shape)} != (3, 12, 16)"

    def test_random_sized_crop_pixels_in_source_range(self) -> None:
        """Resampled pixels stay within the source's value range."""
        # Input is uniform in [0, 1); resized-crop with bilinear must
        # produce values in the same range (no extrapolation).
        chw, _ = _image(seed=11, h=40, w=40)
        tf = T.RandomSizedCrop(min_max_height=(10, 30), height=12, width=16, p=1.0)
        lucid.manual_seed(0)
        out = tf(T.Image(chw)).data.numpy()
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_bbox_safe_crop_preserves_boxes(self) -> None:
        """``BBoxSafeRandomCrop`` returns a crop fully containing every box."""
        img = lucid.full((3, 100, 100), 0.5, dtype=lucid.float32)
        boxes = T.BoundingBoxes(
            lucid.tensor([[20.0, 30.0, 60.0, 80.0]]),
            "xyxy",
            (100, 100),
        )
        sample = {"image": T.Image(img), "boxes": boxes}
        tf = T.BBoxSafeRandomCrop(erosion_rate=0.0, p=1.0)
        for seed in range(20):
            lucid.manual_seed(seed)
            out = tf(sample)
            new_boxes = out["boxes"]
            # After a "safe" crop, the original box must lie fully
            # inside the new canvas — i.e. the (cropped, clipped) box's
            # area must equal the original area (40 * 50 = 2000).
            xy = new_boxes.data.numpy()
            box_w = float(xy[0, 2] - xy[0, 0])
            box_h = float(xy[0, 3] - xy[0, 1])
            assert box_w * box_h == pytest.approx(
                40.0 * 50.0
            ), f"seed {seed}: box clipped (area {box_w * box_h} != 2000)"

    def test_random_sized_bbox_safe_crop_output_size(self) -> None:
        """``RandomSizedBBoxSafeCrop`` always returns ``(C, height, width)``."""
        img = lucid.full((3, 100, 100), 0.5, dtype=lucid.float32)
        boxes = T.BoundingBoxes(
            lucid.tensor([[20.0, 30.0, 60.0, 80.0]]),
            "xyxy",
            (100, 100),
        )
        sample = {"image": T.Image(img), "boxes": boxes}
        tf = T.RandomSizedBBoxSafeCrop(height=24, width=32, p=1.0)
        for seed in range(10):
            lucid.manual_seed(seed)
            out = tf(sample)
            assert tuple(out["image"].data.shape) == (3, 24, 32)
            # Box canvas matches output image canvas.
            assert out["boxes"].canvas_size == (24, 32)

    def test_random_crop_near_bbox_within_image(self) -> None:
        """``RandomCropNearBBox`` returns a crop staying inside the source image."""
        h, w = 100, 100
        img = lucid.full((3, h, w), 0.5, dtype=lucid.float32)
        boxes = T.BoundingBoxes(
            lucid.tensor([[30.0, 30.0, 70.0, 70.0]]),
            "xyxy",
            (h, w),
        )
        sample = {"image": T.Image(img), "boxes": boxes}
        tf = T.RandomCropNearBBox(max_part_shift=0.3, p=1.0)
        for seed in range(20):
            lucid.manual_seed(seed)
            out = tf(sample)
            oh, ow = out["image"].data.shape[-2:]
            # Crop window cannot exceed the source image.
            assert 0 < int(oh) <= h, f"seed {seed}: height {int(oh)} out of [1, {h}]"
            assert 0 < int(ow) <= w, f"seed {seed}: width {int(ow)} out of [1, {w}]"


# ── tier 3: dropout — mask paste / fill / pattern contracts ─────────


@pytest.mark.parity
class TestDropoutPatch:
    """Mask paste contracts: dropped pixels = fill, count + size in range.

    We do *not* compare pixel-for-pixel against Albumentations because
    the two RNGs disagree on hole placement; instead we pin the
    promised contract that both implementations expose.
    """

    def test_coarse_dropout_count_respected(self) -> None:
        """With ``min_holes == max_holes``, exactly that many holes are blanked."""
        # Pin every hole shape so the area can be computed exactly.
        chw, _ = _const_image(value=1.0, h=40, w=40)
        n, hh, ww = 3, 4, 5
        lucid.manual_seed(0)
        tf = T.CoarseDropout(
            max_holes=n,
            max_height=hh,
            max_width=ww,
            min_holes=n,
            min_height=hh,
            min_width=ww,
            fill_value=0.0,
            p=1.0,
        )
        params = tf.make_params(chw)
        assert len(params.holes) == n
        for top, left, h, w in params.holes:
            assert h == hh and w == ww, "hole size must match min==max range"
            assert 0 <= top and top + h <= 40
            assert 0 <= left and left + w <= 40

    def test_coarse_dropout_max_holes_respected(self) -> None:
        """Number of holes is always inside ``[min_holes, max_holes]``."""
        chw, _ = _const_image(value=1.0, h=40, w=40)
        tf = T.CoarseDropout(
            max_holes=5,
            max_height=4,
            max_width=4,
            min_holes=1,
            min_height=4,
            min_width=4,
            fill_value=0.0,
            p=1.0,
        )
        for seed in range(20):
            lucid.manual_seed(seed)
            params = tf.make_params(chw)
            # ``_random.randint`` with span 0 collapses to low — the
            # sampler draws from ``[min_holes, max_holes + 1)``.
            assert (
                1 <= len(params.holes) <= 5
            ), f"seed {seed}: {len(params.holes)} holes outside [1, 5]"

    def test_coarse_dropout_fill_value(self) -> None:
        """Pixels inside every hole are exactly the requested fill value."""
        chw, _ = _const_image(value=1.0, h=24, w=24)
        fill = 0.25
        n, hh, ww = 2, 3, 4
        lucid.manual_seed(0)
        tf = T.CoarseDropout(
            max_holes=n,
            max_height=hh,
            max_width=ww,
            min_holes=n,
            min_height=hh,
            min_width=ww,
            fill_value=fill,
            p=1.0,
        )
        params = tf.make_params(chw)
        out = tf._apply_image(chw, params).numpy()
        # Each hole pixel must equal ``fill`` exactly; outside pixels
        # must equal the original 1.0.
        for top, left, h, w in params.holes:
            patch = out[:, top : top + h, left : left + w]
            np.testing.assert_array_equal(
                patch,
                np.full_like(patch, fill),
                err_msg=f"hole at ({top}, {left}, {h}, {w}) is not fill",
            )

    def test_coarse_dropout_total_dropped_area(self) -> None:
        """When holes don't overlap, total dropped count = sum of areas."""
        # Choose hole sizes that very rarely overlap on a 40x40 canvas
        # (small holes, small count) and assert the dropped count is
        # bounded by ``n * hh * ww * c`` (equality when no overlap).
        chw, _ = _const_image(value=1.0, h=40, w=40, c=3)
        n, hh, ww = 2, 3, 3
        lucid.manual_seed(123)
        tf = T.CoarseDropout(
            max_holes=n,
            max_height=hh,
            max_width=ww,
            min_holes=n,
            min_height=hh,
            min_width=ww,
            fill_value=0.0,
            p=1.0,
        )
        params = tf.make_params(chw)
        out = tf._apply_image(chw, params).numpy()
        # Dropped pixels are exactly the zeros (source was all-1).
        n_zero = int(np.sum(out == 0.0))
        upper = n * hh * ww * 3  # per-channel × 3
        # With possible overlap, count <= upper.
        assert (
            n_zero <= upper
        ), f"zero-count {n_zero} > upper bound {upper} (impossible overlap math)"
        # And at minimum, one hole worth of pixels was dropped.
        assert n_zero >= hh * ww * 3

    def test_grid_dropout_periodic_pattern(self) -> None:
        """``GridDropout`` produces a periodic hole pattern of period ``unit_size``."""
        # Constant input + zero fill: every dropped pixel becomes 0,
        # every kept pixel stays at 1.  Verify that the keep-mask is
        # periodic and that each hole is exactly hole=round(unit*ratio).
        chw, _ = _const_image(value=1.0, h=32, w=32, c=1)
        unit, ratio = 8, 0.5
        hole = int(round(unit * ratio))  # 4
        lucid.manual_seed(0)
        tf = T.GridDropout(ratio=ratio, unit_size=unit, fill_value=0.0, p=1.0)
        params = tf.make_params(chw)
        out = tf._apply_image(chw, params).numpy()[0]
        # First ``hole`` rows of each unit are zeros, next (unit-hole) rows are ones.
        for row_start in range(0, 32, unit):
            # zero rows
            zero_band = out[row_start : row_start + hole, :]
            np.testing.assert_array_equal(
                zero_band[:, :hole],
                np.zeros((hole, hole), dtype=np.float32),
                err_msg=f"zero band missing at row {row_start}",
            )
            # kept rows below the zero band
            keep_band = out[row_start + hole : row_start + unit, hole:unit]
            np.testing.assert_array_equal(
                keep_band,
                np.ones_like(keep_band),
                err_msg=f"keep band corrupted at row {row_start}",
            )

    def test_grid_dropout_fill_value(self) -> None:
        """Pixels inside every grid cell are exactly the requested fill value."""
        chw, _ = _const_image(value=1.0, h=16, w=16, c=1)
        fill = 0.7
        lucid.manual_seed(0)
        tf = T.GridDropout(ratio=0.5, unit_size=8, fill_value=fill, p=1.0)
        params = tf.make_params(chw)
        out = tf._apply_image(chw, params).numpy()[0]
        # All "hole" pixels (top-left 4x4 of each 8x8 unit) must equal fill.
        for r0 in (0, 8):
            for c0 in (0, 8):
                patch = out[r0 : r0 + 4, c0 : c0 + 4]
                np.testing.assert_array_equal(
                    patch,
                    np.full_like(patch, fill),
                    err_msg=f"hole at ({r0}, {c0}) not fill {fill}",
                )

    def test_dropout_p_zero_passes_through(self) -> None:
        """``p=0`` makes both dropout ops the identity."""
        chw, _ = _image(seed=20, h=16, w=16)
        # Same orig must come back exactly.
        for tf in (
            T.CoarseDropout(max_holes=5, fill_value=0.5, p=0.0),
            T.GridDropout(ratio=0.5, unit_size=8, fill_value=0.5, p=0.0),
        ):
            out = tf(T.Image(chw)).data
            np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0)


# ── tier 4: composition — structural correctness ────────────────────


@pytest.mark.parity
class TestComposition:
    """Meta-transforms: probability gates, child selection, replay reproducibility.

    These containers are not pixel-comparable to Albumentations directly
    (the two RNGs disagree on every coin flip).  Instead we test the
    *contract*: ``Compose`` applies in order, ``OneOf`` picks exactly
    one, ``SomeOf`` picks the requested count, and ``ReplayCompose``
    can reproduce its own decisions on a different input.
    """

    def test_compose_runs_in_order(self) -> None:
        """``Compose`` applies children left-to-right."""
        chw, _ = _image(seed=30, h=16, w=16)
        # H(V(x)) != V(H(x)) for crops with offset; but for flips the
        # composition is commutative, so build a deterministic
        # non-commutative chain instead.
        crop = T.Crop(0, 0, 8, 8, p=1.0)
        pad = T.CropAndPad(px=2, p=1.0)
        composed = T.Compose([crop, pad])
        got = composed(T.Image(chw)).data.numpy()
        # Manually apply in order.
        expected = pad(T.Image(crop(T.Image(chw)).data)).data.numpy()
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)
        # Shape: (3, 8, 8) cropped then +2 pad on each side -> (3, 12, 12).
        assert got.shape == (3, 12, 12)

    def test_compose_vs_albu_deterministic(self) -> None:
        """When every child is deterministic, Lucid ``Compose`` matches Albu."""
        chw, hwc = _image(seed=31, h=32, w=32)
        lucid_pipe = T.Compose([T.Crop(2, 3, 26, 22, p=1.0), T.HorizontalFlip(p=1.0)])
        albu_pipe = A.Compose(
            [A.Crop(2, 3, 26, 22, p=1.0), A.HorizontalFlip(p=1.0)],
            p=1.0,
        )
        got = lucid_pipe(T.Image(chw)).data.numpy()
        ref = albu_pipe(image=hwc)["image"]
        # Lucid CHW vs Albu HWC.
        got_hwc = np.transpose(got, (1, 2, 0))
        np.testing.assert_allclose(got_hwc, ref, atol=0.0, rtol=0.0)

    def test_one_of_picks_exactly_one(self) -> None:
        """``OneOf`` applies exactly one child per call."""
        chw, _ = _image(seed=32, h=16, w=16)
        # Build two children whose outputs are easily distinguishable
        # from the input and from each other.
        hflip = T.HorizontalFlip(p=1.0)
        vflip = T.VerticalFlip(p=1.0)
        one_of = T.OneOf([hflip, vflip], p=1.0)

        # Pre-compute the two possible outcomes.
        h_out = hflip(T.Image(chw)).data.numpy()
        v_out = vflip(T.Image(chw)).data.numpy()

        for seed in range(20):
            lucid.manual_seed(seed)
            got = one_of(T.Image(chw)).data.numpy()
            # Output must equal one of the two candidates exactly.
            is_h = np.allclose(got, h_out, atol=0.0)
            is_v = np.allclose(got, v_out, atol=0.0)
            assert is_h or is_v, f"seed {seed}: OneOf output matches neither child"
            # And not both (they differ for a generic image).
            assert not (is_h and is_v), f"seed {seed}: both branches indistinguishable (test bug?)"

    def test_one_of_gate_pass_through(self) -> None:
        """``OneOf(p=0)`` is the identity (gate fails)."""
        chw, _ = _image(seed=33, h=16, w=16)
        # p=0 means the container's gate always fails -> input returns unchanged.
        one_of = T.OneOf([T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0)], p=0.0)
        for seed in range(5):
            lucid.manual_seed(seed)
            out = one_of(T.Image(chw)).data.numpy()
            np.testing.assert_allclose(out, chw.numpy(), atol=0.0)

    def test_some_of_count_matches(self) -> None:
        """``SomeOf(n=k)`` applies exactly ``k`` of its children."""
        chw, _ = _image(seed=34, h=16, w=16)
        # Three children, each idempotent on the others (a flip + a
        # rot90 + another flip can be replayed by counting orientation).
        # Use crops: each crop adds 2 px of crop, so two applications
        # of Crop(0, 0, h-2, w-2) shrinks by 2 px each time.
        # Simpler: build children that each multiply output by a known
        # amount, then check the cascade.
        # We use Lambda transforms for unambiguous multiplicative effect.
        scale_a = T.Lambda(image=lambda x: x * 2.0, p=1.0)
        scale_b = T.Lambda(image=lambda x: x * 3.0, p=1.0)
        scale_c = T.Lambda(image=lambda x: x * 5.0, p=1.0)

        # When SomeOf picks ``n=2`` of 3 children, the output is the
        # input times exactly two of (2, 3, 5).  Possible products:
        # {6, 10, 15}.  Check the realised product lies in that set.
        some_of = T.SomeOf([scale_a, scale_b, scale_c], n=2, p=1.0)
        valid_products = {6.0, 10.0, 15.0}
        for seed in range(10):
            lucid.manual_seed(seed)
            got = some_of(T.Image(chw)).data.numpy()
            ratio = float(got.flat[0] / chw.numpy().flat[0])
            assert ratio in valid_products, (
                f"seed {seed}: product {ratio} not in {valid_products} "
                "(SomeOf applied wrong number of children)"
            )

    def test_some_of_n_zero_is_identity(self) -> None:
        """``SomeOf(n=0)`` applies no children."""
        chw, _ = _image(seed=35, h=12, w=12)
        some_of = T.SomeOf([T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0)], n=0, p=1.0)
        for seed in range(5):
            lucid.manual_seed(seed)
            out = some_of(T.Image(chw)).data
            np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0)

    def test_sequential_always_applies(self) -> None:
        """``Sequential(p=1.0)`` applies every child in order."""
        chw, _ = _image(seed=36, h=16, w=16)
        # Two unambiguous multiplications: deterministic, easy to verify.
        seq = T.Sequential(
            [
                T.Lambda(image=lambda x: x * 2.0, p=1.0),
                T.Lambda(image=lambda x: x + 1.0, p=1.0),
            ],
            p=1.0,
        )
        got = seq(T.Image(chw)).data.numpy()
        expected = chw.numpy() * 2.0 + 1.0
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_sequential_gate_blocks_all(self) -> None:
        """``Sequential(p=0.0)`` skips every child (identity)."""
        chw, _ = _image(seed=37, h=12, w=12)
        seq = T.Sequential(
            [
                T.Lambda(image=lambda x: x * 0.0, p=1.0),  # would zero out
            ],
            p=0.0,
        )
        for seed in range(5):
            lucid.manual_seed(seed)
            out = seq(T.Image(chw)).data
            np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0)

    def test_one_or_other_branches(self) -> None:
        """``OneOrOther`` runs ``first`` with prob ``p``, else ``second``."""
        chw, _ = _image(seed=38, h=16, w=16)
        # Make first and second produce distinguishable outputs.
        first = T.Lambda(image=lambda x: x * 2.0, p=1.0)
        second = T.Lambda(image=lambda x: x * 5.0, p=1.0)
        oro = T.OneOrOther(first, second, p=1.0)  # always pick first
        for seed in range(5):
            lucid.manual_seed(seed)
            got = oro(T.Image(chw)).data.numpy()
            np.testing.assert_allclose(got, chw.numpy() * 2.0, atol=1e-6)

        oro_never = T.OneOrOther(first, second, p=0.0)  # always pick second
        for seed in range(5):
            lucid.manual_seed(seed)
            got = oro_never(T.Image(chw)).data.numpy()
            np.testing.assert_allclose(got, chw.numpy() * 5.0, atol=1e-6)

    def test_one_or_other_distribution(self) -> None:
        """At ``p=0.5`` the two branches are picked with roughly equal frequency."""
        chw, _ = _image(seed=39, h=8, w=8)
        first = T.Lambda(image=lambda x: x * 2.0, p=1.0)
        second = T.Lambda(image=lambda x: x * 5.0, p=1.0)
        oro = T.OneOrOther(first, second, p=0.5)
        n_first = 0
        n_trials = 200
        lucid.manual_seed(42)
        ref = chw.numpy()
        for _ in range(n_trials):
            got = oro(T.Image(chw)).data.numpy()
            ratio = float(got.flat[0] / ref.flat[0])
            if abs(ratio - 2.0) < 1e-3:
                n_first += 1
        # Loose Monte-Carlo: 1/4 ± 1/4 of n_trials should be safely
        # within the first-branch count (binomial std ~7 at n=200, p=0.5).
        assert (
            n_trials // 4 <= n_first <= 3 * n_trials // 4
        ), f"first-branch count {n_first}/{n_trials} far from 50/50"

    def test_replay_compose_reproducibility(self) -> None:
        """``ReplayCompose.replay`` reproduces the exact same param sequence."""
        chw1, _ = _image(seed=40, h=24, w=24)
        chw2, _ = _image(seed=41, h=24, w=24)
        # Use stochastic transforms so saving / replaying matters.
        rc = T.ReplayCompose(
            [
                T.Crop(2, 3, 18, 22, p=1.0),
                T.HorizontalFlip(p=1.0),
                T.CoarseDropout(
                    max_holes=2,
                    max_height=4,
                    max_width=4,
                    min_holes=2,
                    min_height=4,
                    min_width=4,
                    fill_value=0.0,
                    p=1.0,
                ),
            ]
        )
        lucid.manual_seed(7)
        rc(T.Image(chw1))  # populates rc.replay_data (output itself unused)
        saved = list(rc.replay_data)
        # Replay on a *different* input — the same indices / holes /
        # crop window must be applied (i.e. the output should be
        # produced by re-running with identical saved params).
        replayed_on_2 = rc.replay(saved, T.Image(chw2)).data
        # Re-build the same pipeline manually from saved params.
        # Use the saved tuples directly: each Transform.dispatch with
        # the saved params is what ``replay`` invokes.
        from lucid.utils.transforms import Image as _Image

        manual = _Image(chw2)
        for tf, params, applied in saved:
            if not applied or params is None:
                continue
            manual = tf._dispatch(manual, params)  # type: ignore[attr-defined]
        manual_out = manual.data if hasattr(manual, "data") else manual
        np.testing.assert_allclose(replayed_on_2.numpy(), manual_out.numpy(), atol=0.0)

    def test_replay_compose_saves_per_child(self) -> None:
        """``replay_data`` records one entry per child after a call."""
        chw, _ = _image(seed=42, h=16, w=16)
        rc = T.ReplayCompose(
            [
                T.HorizontalFlip(p=1.0),
                T.VerticalFlip(p=1.0),
                T.Crop(0, 0, 10, 10, p=1.0),
            ]
        )
        lucid.manual_seed(0)
        rc(T.Image(chw))
        assert (
            len(rc.replay_data) == 3
        ), f"replay_data has {len(rc.replay_data)} entries, expected 3"
        for tf, params, applied in rc.replay_data:
            assert applied is True, "p=1.0 children should always apply"

    def test_replay_compose_skipped_child_recorded(self) -> None:
        """Children skipped by their own ``p`` gate get ``applied=False``."""
        chw, _ = _image(seed=43, h=12, w=12)
        # Force at least one child to fail its gate.
        rc = T.ReplayCompose([T.HorizontalFlip(p=0.0), T.VerticalFlip(p=1.0)])
        lucid.manual_seed(0)
        rc(T.Image(chw))
        # First child (p=0) must be marked not applied; second must be applied.
        assert rc.replay_data[0][2] is False
        assert rc.replay_data[1][2] is True


# ── tier 5: edge cases / no-op contracts ────────────────────────────


@pytest.mark.parity
class TestEdgeCases:
    """No-op contracts (empty containers, fully-covering crops, etc.)."""

    def test_compose_empty_is_identity(self) -> None:
        """An empty ``Compose`` passes the input through unchanged."""
        chw, _ = _image(seed=50, h=12, w=12)
        out = T.Compose([])(T.Image(chw)).data
        np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0)

    def test_one_of_empty_is_identity(self) -> None:
        """An empty ``OneOf`` passes the input through unchanged."""
        chw, _ = _image(seed=51, h=12, w=12)
        out = T.OneOf([], p=1.0)(T.Image(chw))
        # OneOf with no children returns the input unchanged (Image wrap).
        np.testing.assert_allclose(out.data.numpy(), chw.numpy(), atol=0.0)

    def test_some_of_empty_is_identity(self) -> None:
        """An empty ``SomeOf`` passes the input through unchanged."""
        chw, _ = _image(seed=52, h=12, w=12)
        out = T.SomeOf([], n=2, p=1.0)(T.Image(chw))
        np.testing.assert_allclose(out.data.numpy(), chw.numpy(), atol=0.0)

    def test_coarse_dropout_zero_holes_is_identity(self) -> None:
        """``max_holes=min_holes=0`` produces no holes (identity)."""
        chw, _ = _image(seed=53, h=20, w=20)
        lucid.manual_seed(0)
        tf = T.CoarseDropout(
            max_holes=0,
            max_height=4,
            max_width=4,
            min_holes=0,
            min_height=4,
            min_width=4,
            fill_value=0.5,
            p=1.0,
        )
        out = tf(T.Image(chw)).data
        np.testing.assert_allclose(out.numpy(), chw.numpy(), atol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
