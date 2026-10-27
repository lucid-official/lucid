"""``interpolate`` modes, refusals and coordinate mapping, without the reference.

The values are matched against the reference in
``lucid/test/parity/nn/test_interpolate_parity.py``; this tier pins what
can be stated by hand: where a fractional scale factor puts the samples,
that ``nearest-exact`` is not ``nearest``, which arguments are refused and
with what, and when the size-only engine kernels may stand in for the
gathers.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.nn.functional import _interpolate
from lucid.test._helpers.resample_paths import PATHS, forced_path


def _ramp(length: int) -> lucid.Tensor:
    return lucid.arange(length, dtype=lucid.float32).reshape(1, 1, length)


class TestCoordinateMapping:
    def test_a_fractional_scale_steps_by_its_reciprocal(self) -> None:
        # 5 * 1.7 -> 8 samples.  Stepping by 1/1.7 (not 5/8) puts sample 1
        # at (1.5 / 1.7) - 0.5 = 0.38235...
        out = F.interpolate(_ramp(5), scale_factor=1.7, mode="linear")
        assert out.shape == (1, 1, 8)
        assert float(out[0, 0, 1].item()) == pytest.approx(1.5 / 1.7 - 0.5, abs=1e-6)

    def test_recompute_steps_by_the_size_ratio(self) -> None:
        out = F.interpolate(_ramp(5), scale_factor=1.7, mode="linear", recompute_scale_factor=True)
        assert float(out[0, 0, 1].item()) == pytest.approx(1.5 * 5 / 8 - 0.5, abs=1e-6)

    def test_nearest_exact_is_not_nearest(self) -> None:
        near = F.interpolate(_ramp(5), size=9, mode="nearest").reshape(-1).tolist()
        exact = F.interpolate(_ramp(5), size=9, mode="nearest-exact").reshape(-1).tolist()
        assert near == [0, 0, 1, 1, 2, 2, 3, 3, 4]
        assert exact == [0, 0, 1, 1, 2, 3, 3, 4, 4]

    def test_area_is_adaptive_average_pooling(self) -> None:
        out = F.interpolate(_ramp(5), scale_factor=0.6, mode="area")
        assert out.reshape(-1).tolist() == pytest.approx([0.5, 2.0, 3.5])

    def test_nearest_keeps_an_integer_dtype(self) -> None:
        x = lucid.arange(6, dtype=lucid.int64).reshape(1, 1, 2, 3)
        out = F.interpolate(x, scale_factor=1.5, mode="nearest")
        assert out.dtype == lucid.int64
        assert out.shape == (1, 1, 3, 4)

    @pytest.mark.parametrize(
        ("shape", "scale", "expected"),
        [
            # The same-size and double-size shortcuts are the reference's 2-D
            # kernel's alone; 1-D and 3-D step by float32 1/scale.
            ((1, 1, 4), 1.1, [0, 0, 1, 2]),
            ((1, 1, 5), 2.1, [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]),
            ((1, 1, 1, 4), 1.1, [0, 1, 2, 3]),
            ((1, 1, 1, 5), 2.1, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        ],
    )
    def test_nearest_pins_same_and_double_size_only_in_two_d(
        self, shape: tuple[int, ...], scale: float, expected: list[int]
    ) -> None:
        n = shape[-1]
        x = lucid.arange(n, dtype=lucid.float32).reshape(*shape)
        scales = scale if len(shape) == 3 else (1.0, scale)
        out = F.interpolate(x, scale_factor=scales, mode="nearest")
        assert out.reshape(-1).tolist() == [float(i) for i in expected]

    def test_antialias_leaves_a_same_size_axis_alone(self) -> None:
        x = lucid.randn(1, 1, 3, 8)
        out = F.interpolate(x, scale_factor=(1.1, 1.0), mode="bilinear", antialias=True)
        assert out.reshape(-1).tolist() == x.reshape(-1).tolist()

    def test_same_size_returns_a_new_tensor(self) -> None:
        x = _ramp(5)
        out = F.interpolate(x, scale_factor=1.1, mode="linear")
        assert out is not x
        assert out.reshape(-1).tolist() == x.reshape(-1).tolist()


class TestEngineStandIn:
    """The size-only kernels step by ``in / out``, by the reference's rule
    for a size on both devices (LCD-403): allowed only when a scale factor
    puts every sample where that does, in the tensor's accumulation type
    (``wide`` for float64)."""

    @pytest.mark.parametrize(
        ("mode", "spatial", "sizes", "scales", "corners", "wide", "expected"),
        [
            ("bilinear", [5, 7], [8, 11], [1.7, 1.7], False, False, False),
            ("bilinear", [5, 7], [8, 11], [1.7, 1.7], True, False, True),
            ("bilinear", [5, 7], [8, 11], [None, None], False, False, True),
            ("bilinear", [4, 6], [8, 12], [2.0, 2.0], False, False, True),
            ("nearest", [5, 7], [8, 11], [1.7, 1.7], False, False, False),
            ("nearest", [4, 6], [8, 12], [2.0, 2.0], False, False, True),
            ("nearest", [5], [8], [None], False, False, True),
            # float32 in/out lands sample 5382 on 2153; the reference's
            # float32 1/2.5 lands it on 2152.  In double both land on 2152.
            ("nearest", [1, 2155], [1, 5387], [1.0, 2.5], False, False, False),
            ("nearest", [1, 2155], [1, 5387], [1.0, 2.5], False, True, True),
            # With a size the reference also steps by in/out, which the
            # kernels now do on both devices.
            ("nearest", [1, 2111], [1, 5277], [None, None], False, False, True),
            # x1.1 on 4521 samples: in float32 the two steps agree, in
            # double sample 4962 reads 4511 by in/out and 4510 by 1/1.1.
            ("nearest", [4521], [4973], [1.1], False, False, True),
            ("nearest", [4521], [4973], [1.1], False, True, False),
            # 1-D: no same-size shortcut in the reference, so x1.1 resamples.
            ("nearest", [4], [4], [1.1], False, False, False),
        ],
    )
    def test_when_the_kernel_may_run(
        self,
        mode: str,
        spatial: list[int],
        sizes: list[int],
        scales: list[float | None],
        corners: bool,
        wide: bool,
        expected: bool,
    ) -> None:
        got = _interpolate._engine_maps_like(mode, spatial, sizes, scales, corners, wide)
        assert got is expected

    @pytest.mark.parametrize(
        ("shape", "kw"),
        [
            ((1, 1, 1, 2155), {"scale_factor": (1.0, 2.5)}),
            ((1, 1, 1, 2111), {"size": (1, 5277)}),
        ],
        ids=["scale-step", "size-step"],
    )
    def test_each_device_reads_what_the_mapping_says(
        self, device: str, shape: tuple[int, ...], kw: dict[str, Any]
    ) -> None:
        n = shape[-1]
        x = lucid.arange(n, dtype=lucid.float32, device=device).reshape(*shape)
        out = F.interpolate(x, mode="nearest", **kw).reshape(-1).to("cpu").tolist()
        scale = kw["scale_factor"][1] if "scale_factor" in kw else None
        expected = _interpolate._nearest_indices(n, len(out), scale, False, 2, False)
        assert out == [float(i) for i in expected]


class TestRefusals:
    @pytest.mark.parametrize(
        ("shape", "kw", "error", "match"),
        [
            ((1, 1, 4, 4), {"size": 2, "mode": "lanczos3"}, ValueError, "unknown mode"),
            ((1, 1, 4, 4), {"size": 2, "mode": "linear"}, ValueError, "spatial axes"),
            ((1, 1, 4), {"size": 2, "mode": "bilinear"}, ValueError, "spatial axes"),
            ((1, 1, 4, 4), {"size": 2, "scale_factor": 2.0}, ValueError, "not both"),
            ((1, 1, 4, 4), {}, ValueError, "either size"),
            ((1, 1, 4, 4), {"size": (2, 3, 4)}, ValueError, "entries"),
            ((1, 1, 4, 4), {"scale_factor": (2.0,)}, ValueError, "entries"),
            ((1, 1, 4, 4), {"scale_factor": 0.1}, ValueError, "positive"),
            (
                (1, 1, 4, 4),
                {"size": 2, "recompute_scale_factor": True},
                ValueError,
                "recompute_scale_factor",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "nearest", "align_corners": False},
                ValueError,
                "align_corners",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "area", "align_corners": True},
                ValueError,
                "align_corners",
            ),
            (
                (1, 1, 4, 4, 4),
                {"size": 2, "mode": "trilinear", "antialias": True},
                ValueError,
                "antialias",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "nearest", "antialias": True},
                ValueError,
                "antialias",
            ),
            (
                (1, 1, 4, 4),
                {"size": 2, "mode": "lanczos"},
                ValueError,
                "Lanczos mode requires antialias=True",
            ),
            (
                (1, 1, 4, 4),
                {
                    "size": 2,
                    "mode": "lanczos",
                    "antialias": True,
                    "align_corners": True,
                },
                ValueError,
                "Lanczos mode does not support align_corners=True",
            ),
            (
                (1, 1, 4),
                {"size": 2, "mode": "lanczos", "antialias": True},
                ValueError,
                "spatial axes",
            ),
        ],
    )
    def test_refused_with(
        self,
        shape: tuple[int, ...],
        kw: dict[str, Any],
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            F.interpolate(lucid.zeros(*shape), **kw)

    def test_interpolating_an_integer_input_is_a_type_error(self) -> None:
        x = lucid.arange(16, dtype=lucid.int64).reshape(1, 1, 4, 4)
        with pytest.raises(TypeError, match="floating-point"):
            F.interpolate(x, size=2, mode="bilinear")


@pytest.mark.parametrize(
    ("mode", "scale", "antialias", "path"),
    [
        *[
            (mode, scale, antialias, path)
            for mode, scale, antialias in (
                ("bilinear", 1.7, False),
                ("bicubic", 1.7, False),
                ("bicubic", 0.6, True),
                ("lanczos", 0.6, True),
            )
            for path in PATHS
        ],
        ("nearest-exact", 1.7, False, None),
    ],
)
def test_the_composite_differentiates_twice(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    scale: float,
    antialias: bool,
    path: str | None,
) -> None:
    # A fractional scale runs the composite rather than a kernel — as a
    # matrix product or as gathers, depending on the axis length, so each
    # weighted mode runs on both.  A gradient penalty through it must still
    # see a second derivative.
    if path is None:
        _second_derivative_matches(mode, scale, antialias)
        return
    with forced_path(monkeypatch, path):
        _second_derivative_matches(mode, scale, antialias)


def _second_derivative_matches(mode: str, scale: float, antialias: bool) -> None:
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, 2, 5, 7))
    out = F.interpolate(lucid.tensor(x0), scale_factor=scale, mode=mode, antialias=antialias)
    r0 = lucid.tensor(rng.standard_normal(tuple(out.shape)))
    u = rng.standard_normal(x0.shape)
    e = rng.standard_normal(x0.shape)

    def loss(x: lucid.Tensor) -> lucid.Tensor:
        y = F.interpolate(x, scale_factor=scale, mode=mode, antialias=antialias)
        return (r0 * y * y).sum()

    def projected_gradient(values: np.ndarray) -> float:
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(x), [x])
        return float((np.asarray(g.numpy()) * u).sum())

    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(loss(x), [x], create_graph=True)
    (hvp,) = grad((g * lucid.tensor(u)).sum(), [x])
    h = 1e-6
    numeric = (projected_gradient(x0 + h * e) - projected_gradient(x0 - h * e)) / (2 * h)
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6)


class TestUpsampleForwardsEveryOption:
    def test_recompute_scale_factor_reaches_interpolate(self) -> None:
        x = lucid.randn(1, 2, 5, 7)
        up = nn.Upsample(scale_factor=1.7, mode="bilinear", recompute_scale_factor=True)
        expected = F.interpolate(x, scale_factor=1.7, mode="bilinear", recompute_scale_factor=True)
        assert float((up(x) - expected).abs().max().item()) == 0.0
        assert (
            float((up(x) - F.interpolate(x, scale_factor=1.7, mode="bilinear")).abs().max().item())
            > 1e-3
        )
