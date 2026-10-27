"""Every ``interpolate`` setting against the reference, values and gradients.

A scale factor that does not divide the input evenly used to be rounded
to an output size, and the sample positions then derived from the size,
so every mode read the wrong input samples — ``1.7`` on a 5-pixel axis
steps by ``1/1.7``, not ``5/8``.  ``nearest-exact`` computed ``nearest``,
and ``linear``, 1-D nearest and ``area`` were refused.

The matrix is the product the reference accepts: every mode at every
spatial rank it takes, sized by a whole, a fractional up and a
fractional down scale factor, a per-axis tuple and an explicit size,
with ``recompute_scale_factor`` unset and ``True``, both corner
conventions where they apply, and ``antialias`` at several downsampling
ratios — on every device, with the input gradient.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close
from lucid.test._helpers.resample_paths import PATHS, forced_path

_SHAPES = {1: (2, 3, 7), 2: (2, 3, 5, 7), 3: (1, 2, 4, 5, 6)}
_RANKS = {
    "nearest": (1, 2, 3),
    "nearest-exact": (1, 2, 3),
    "area": (1, 2, 3),
    "linear": (1,),
    "bilinear": (2,),
    "bicubic": (2,),
    "trilinear": (3,),
}
_INTERPOLATING = {"linear", "bilinear", "bicubic", "trilinear"}


def _sizings(rank: int) -> list[tuple[str, dict[str, Any]]]:
    per_axis = (1.7, 0.6, 2.5)[:rank]
    size = (9, 4, 3)[:rank]
    return [
        ("x2", {"scale_factor": 2.0}),
        ("x1.7", {"scale_factor": 1.7}),
        ("x0.6", {"scale_factor": 0.6}),
        ("tuple", {"scale_factor": per_axis}),
        ("x1.7-recompute", {"scale_factor": 1.7, "recompute_scale_factor": True}),
        ("x0.6-recompute", {"scale_factor": 0.6, "recompute_scale_factor": True}),
        ("size", {"size": size}),
        # floor(n * s) lands on n and 2n with s neither 1 nor 2: the
        # reference's 2-D nearest pins these, its 1-D and 3-D do not.
        ("x1.1-same-size", {"scale_factor": 1.1}),
        ("x2.1-double-size", {"scale_factor": 2.1}),
    ]


def _cases() -> list[Any]:
    cases = []
    for mode, ranks in _RANKS.items():
        corner_options = (True, False) if mode in _INTERPOLATING else (None,)
        for rank in ranks:
            for tag, kw in _sizings(rank):
                for corners in corner_options:
                    ctag = {True: "corners", False: "edges", None: "-"}[corners]
                    cases.append(
                        pytest.param(
                            mode,
                            rank,
                            {**kw, "align_corners": corners},
                            id=f"{mode}-{rank}d-{tag}-{ctag}",
                        )
                    )
    return cases


_AA_SHAPE = (2, 3, 11, 14)
_ANTIALIAS = [
    pytest.param(mode, shape, kw, corners, id=f"{mode}-{tag}-{'corners' if corners else 'edges'}")
    for mode in ("bilinear", "bicubic", "lanczos")
    for tag, shape, kw in (
        ("x0.5", _AA_SHAPE, {"scale_factor": 0.5}),
        ("x0.3", _AA_SHAPE, {"scale_factor": 0.3}),
        ("x0.37-recompute", _AA_SHAPE, {"scale_factor": 0.37, "recompute_scale_factor": True}),
        ("tuple", _AA_SHAPE, {"scale_factor": (0.75, 0.21)}),
        ("size", _AA_SHAPE, {"size": (4, 3)}),
        # H = 4 at x1.1 keeps its size; the reference leaves such an axis
        # untouched whatever the scale factor said.
        ("x1.1-same-height", (2, 3, 4, 14), {"scale_factor": (1.1, 0.5)}),
        ("x1.1-same-size", (1, 2, 3, 8), {"scale_factor": (1.1, 1.0)}),
        ("x1.7-up", _AA_SHAPE, {"scale_factor": 1.7}),
    )
    for corners in ((False,) if mode == "lanczos" else (True, False))
]


def _compare(ref: Any, device: str, shape: tuple[int, ...], kw: dict[str, Any], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    lx = lucid.tensor(x, device=device, requires_grad=True)
    rx = ref.tensor(x, requires_grad=True)
    l_out = F.interpolate(lx, **kw)
    r_out = ref.nn.functional.interpolate(rx, **kw)
    assert tuple(l_out.shape) == tuple(r_out.shape)
    assert_close(l_out, r_out.detach(), atol=1e-5, msg="value")

    # A random cotangent, so no gradient term cancels against another.
    weight = rng.standard_normal(tuple(r_out.shape)).astype(np.float32)
    (l_out * lucid.tensor(weight, device=device)).sum().backward()
    assert lx.grad is not None
    probe = rng.standard_normal(shape)
    if _reference_backward_is_adjoint(ref, probe, weight, kw):
        (r_out * ref.tensor(weight)).sum().backward()
        assert_close(lx.grad, rx.grad, atol=1e-5, msg="input grad")
        return
    # The reference's backward is not the adjoint of its own forward here
    # (an axis that keeps its size under a scale factor other than one:
    # its backward copies while its forward resamples).  The forwards
    # agree, so the gradient is held to the adjoint of that forward.
    probed = F.interpolate(lucid.tensor(probe.astype(np.float32), device=device), **kw)
    lhs = float((probed * lucid.tensor(weight, device=device)).sum().item())
    rhs = float((np.asarray(lx.grad.to("cpu").numpy()) * probe).sum())
    assert abs(lhs - rhs) <= 1e-4 * (1.0 + abs(lhs)), "input grad is not the adjoint"


def _reference_backward_is_adjoint(
    ref: Any, probe: np.ndarray, weight: np.ndarray, kw: dict[str, Any]
) -> bool:
    """``<J v, w> == <v, J^T w>`` for the reference, in double precision."""
    x = ref.tensor(probe, requires_grad=True)
    cotangent = ref.tensor(weight.astype(np.float64))
    out = ref.nn.functional.interpolate(x, **kw)
    (out * cotangent).sum().backward()
    lhs = float((out.detach() * cotangent).sum())
    rhs = float((x.grad * x.detach()).sum())
    return abs(lhs - rhs) <= 1e-9 * (1.0 + abs(lhs))


@pytest.mark.parity
class TestInterpolateMatrix:
    @pytest.mark.parametrize(("mode", "rank", "kw"), _cases())
    def test_values_and_gradient(
        self, ref: Any, device: str, mode: str, rank: int, kw: dict[str, Any]
    ) -> None:
        _compare(ref, device, _SHAPES[rank], {**kw, "mode": mode}, seed=rank)


@pytest.mark.parity
class TestInterpolateAntialias:
    @pytest.mark.parametrize(("mode", "shape", "kw", "align_corners"), _ANTIALIAS)
    def test_values_and_gradient(
        self,
        ref: Any,
        device: str,
        mode: str,
        shape: tuple[int, ...],
        kw: dict[str, Any],
        align_corners: bool,
    ) -> None:
        kw = {**kw, "mode": mode, "align_corners": align_corners, "antialias": True}
        _compare(ref, device, shape, kw, seed=7)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic", "lanczos"])
    def test_an_output_width_of_one_is_not_a_constant_column(
        self, ref: Any, device: str, mode: str
    ) -> None:
        # Left out of the matrix on purpose: with antialias and an output
        # width of one, the reference repeats one value down the column.
        # That is not mirrored.  Resizing the transposed image (height one,
        # which it gets right) gives the answer to hold Lucid to instead.
        x = np.random.default_rng(3).standard_normal((1, 2, 6, 2)).astype(np.float32)
        got = F.interpolate(lucid.tensor(x, device=device), size=(3, 1), mode=mode, antialias=True)
        want = ref.nn.functional.interpolate(
            ref.tensor(x).transpose(2, 3), size=(1, 3), mode=mode, antialias=True
        ).transpose(2, 3)
        assert_close(got, want, atol=1e-5)


@pytest.mark.parity
class TestUpsampleModules:
    @pytest.mark.parametrize(
        ("module", "kw"),
        [
            ("Upsample", {"scale_factor": 0.6, "mode": "bilinear"}),
            ("Upsample", {"scale_factor": 1.7, "mode": "nearest"}),
            ("Upsample", {"scale_factor": 1.7, "mode": "bicubic", "recompute_scale_factor": True}),
            ("UpsamplingNearest2d", {"scale_factor": 1.7}),
            ("UpsamplingBilinear2d", {"scale_factor": 1.7}),
        ],
        ids=[
            "upsample-bilinear",
            "upsample-nearest",
            "upsample-recompute",
            "nearest2d",
            "bilinear2d",
        ],
    )
    def test_modules_delegate(self, ref: Any, device: str, module: str, kw: dict[str, Any]) -> None:
        x = np.random.default_rng(0).standard_normal((2, 3, 5, 7)).astype(np.float32)
        l_out = getattr(lucid.nn, module)(**kw)(lucid.tensor(x, device=device))
        r_out = getattr(ref.nn, module)(**kw)(ref.tensor(x))
        assert_close(l_out, r_out, atol=1e-5)


# The matrix's inputs are short enough that every weighted axis takes the
# dense product; long axes in real use take the gathers.  This subset runs
# on each path with the other one forbidden.
# Only rows that reach the composite: a fractional scale with half-pixel
# centres (with corners pinned, or a recomputed scale, the linear modes
# run the engine kernel).
_PATH_TAGS = ("-x1.7-edges", "-x0.6-edges", "-tuple-edges")
_PATH_CASES = [
    case
    for case in _cases()
    if case.values[0] in _INTERPOLATING and str(case.id).endswith(_PATH_TAGS)
]
# Every antialias row but the one whose axes all keep their size.
_PATH_ANTIALIAS = [case for case in _ANTIALIAS if "-same-size-" not in str(case.id)]


@pytest.mark.parity
@pytest.mark.parametrize("path", PATHS)
class TestBothCompositePaths:
    @pytest.mark.parametrize(("mode", "rank", "kw"), _PATH_CASES)
    def test_matrix_subset(
        self,
        ref: Any,
        device: str,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        mode: str,
        rank: int,
        kw: dict[str, Any],
    ) -> None:
        with forced_path(monkeypatch, path):
            _compare(ref, device, _SHAPES[rank], {**kw, "mode": mode}, seed=rank)

    @pytest.mark.parametrize(("mode", "shape", "kw", "align_corners"), _PATH_ANTIALIAS)
    def test_antialias(
        self,
        ref: Any,
        device: str,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        mode: str,
        shape: tuple[int, ...],
        kw: dict[str, Any],
        align_corners: bool,
    ) -> None:
        kw = {**kw, "mode": mode, "align_corners": align_corners, "antialias": True}
        with forced_path(monkeypatch, path):
            _compare(ref, device, shape, kw, seed=7)
