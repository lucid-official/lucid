"""Where ``interpolate``'s engine kernels read, on both devices, in every dtype.

The CPU nearest kernel divided exactly and the Metal one multiplied by
``in / out`` in float32, so on a long axis the two copied different
samples (1137 axis pairs between 1000 and 9000 long).  The Metal linear
kernels built their coordinates in the input's dtype: a bfloat16 bilinear
resize was off by 1.13.  Both now ask one rule, the reference's
(``lucid/_C/backend/SamplePositions.h``, LCD-403).

Nearest is held to the reference's indices exactly — each output sample
reports the index it copied.  The linear modes are held to the
reference's values in the same dtype, and their gradients to the
reference's, within that dtype's rounding.  Sizes go through the engine
kernels (a size is the mapping they compute).
"""

import math
import random
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_DTYPES = ["float32", "float16", "bfloat16", "float64"]
# Every value a resize can produce from standard-normal input, rounded the
# way the dtype rounds: blends of a few samples, so a few ulp of ~4.
_ATOL = {"float32": 2e-6, "float16": 4e-3, "bfloat16": 3.2e-2, "float64": 1e-12}


def _skip_unsupported(device: str, dtype: str) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal has no float64")


def _long_axes() -> list[tuple[int, int]]:
    rng = random.Random(403)
    pairs = [(2155, 5387), (2111, 5277), (9000, 1001), (1000, 8999)]
    pairs += [(rng.randint(1000, 9000), rng.randint(1000, 12000)) for _ in range(8)]
    return pairs


def _host(t: Any) -> np.ndarray:
    """``t`` (either framework) as float64 numpy."""
    if isinstance(t, lucid.Tensor):
        return np.asarray(t.to("cpu").to(lucid.float64).numpy())
    return t.detach().double().numpy()


def _copied_indices(
    ref: Any, device: str, dtype: str, n: int, m: int, rank: int
) -> tuple[list[int], list[int]]:
    """The input index each output sample copies, Lucid's and the reference's.

    The input holds its own index split in two (``i % 64`` and ``i // 64``),
    each exact in every dtype for an axis below 16384.
    """
    shape = (1, 1) + (1,) * (rank - 1) + (n,)
    size = (1,) * (rank - 1) + (m,)
    i = np.arange(n, dtype=np.float64)
    out: dict[str, list[int]] = {}
    for name in ("lucid", "ref"):
        parts = []
        for part in (i % 64, i // 64):
            data = part.reshape(shape)
            if name == "lucid":
                x = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device)
                parts.append(_host(F.interpolate(x, size=size, mode="nearest")))
            else:
                x = ref.tensor(data, dtype=getattr(ref, dtype))
                parts.append(_host(ref.nn.functional.interpolate(x, size=size, mode="nearest")))
        out[name] = (parts[1] * 64 + parts[0]).astype(np.int64).ravel().tolist()
    return out["lucid"], out["ref"]


@pytest.mark.parity
class TestNearestIndices:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_long_axes_copy_the_reference_sample(
        self, ref: Any, device: str, dtype: str, rank: int
    ) -> None:
        _skip_unsupported(device, dtype)
        for n, m in _long_axes():
            got, want = _copied_indices(ref, device, dtype, n, m, rank)
            assert got == want, f"n={n} m={m}"

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_gradient_lands_on_the_copied_sample(self, device: str, dtype: str) -> None:
        # The adjoint of the forward: each input sample gathers the
        # gradient of every output that copied it.  (The reference's own
        # float64 backward steps by a float32 scale its forward does not
        # use, so it is no oracle there.)
        _skip_unsupported(device, dtype)
        n, m = 2155, 5387
        x = lucid.zeros(1, 1, 1, n, dtype=getattr(lucid, dtype), device=device)
        x.requires_grad_(True)
        F.interpolate(x, size=(1, m), mode="nearest").sum().backward()
        # Positions are computed in float32 for every dtype but float64,
        # whose index probe must then be float64 too.
        wide = lucid.float64 if dtype == "float64" else lucid.float32
        index = lucid.arange(n, dtype=wide).reshape(1, 1, 1, n)
        copied = _host(F.interpolate(index.to(device), size=(1, m), mode="nearest"))
        counts = np.bincount(copied.astype(np.int64).ravel(), minlength=n)
        assert x.grad is not None
        assert _host(x.grad).ravel().tolist() == counts.astype(np.float64).tolist()


def _linear_cases() -> list[Any]:
    cases = []
    for mode, shapes in (
        ("linear", [((2, 3, 9), (31,)), ((1, 1, 2155), (5387,))]),
        (
            "bilinear",
            [
                ((2, 3, 40, 50), (68, 85)),
                ((1, 2, 7, 9), (3, 31)),
                ((1, 1, 1, 2155), (1, 5387)),
            ],
        ),
        ("trilinear", [((1, 2, 3, 4, 5), (5, 7, 3)), ((2, 1, 4, 3, 6), (2, 6, 13))]),
    ):
        for shape, size in shapes:
            for corners in (False, True):
                cases.append(
                    pytest.param(mode, shape, size, corners, id=f"{mode}-{size}-{corners}")
                )
    return cases


@pytest.mark.parity
class TestLinearValues:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize(("mode", "shape", "size", "corners"), _linear_cases())
    def test_values_and_gradient_in_the_input_dtype(
        self,
        ref: Any,
        device: str,
        dtype: str,
        mode: str,
        shape: tuple[int, ...],
        size: tuple[int, ...],
        corners: bool,
    ) -> None:
        _skip_unsupported(device, dtype)
        rng = np.random.default_rng(len(shape) * 100 + size[-1])
        data = rng.standard_normal(shape)
        cotangent = rng.standard_normal(shape[:2] + size)
        lx = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
        rx = ref.tensor(data, dtype=getattr(ref, dtype), requires_grad=True)
        kw = {"size": size, "mode": mode, "align_corners": corners}
        l_out = F.interpolate(lx, **kw)
        r_out = ref.nn.functional.interpolate(rx, **kw)
        atol = _ATOL[dtype]
        np.testing.assert_allclose(_host(l_out), _host(r_out), atol=atol, rtol=0)
        l_out.backward(lucid.tensor(cotangent, dtype=getattr(lucid, dtype), device=device))
        r_out.backward(ref.tensor(cotangent, dtype=getattr(ref, dtype)))
        assert lx.grad is not None
        # A gradient sample sums every output that read it: a few ulp each.
        np.testing.assert_allclose(_host(lx.grad), _host(rx.grad), atol=8 * atol, rtol=0)


# A scale factor that does not divide the axis maps by ``1 / scale``, which
# the Python composite computes; it has to use the same accumulation type
# as the kernels (double for float64), or a float64 resize reads float32
# positions — and ``x1.1`` on 4521 samples read sample 4511 for 4510.  No
# 2-D axis here keeps its size or doubles it: the reference pins those
# below an output-size line, which ``TestNearestCompatibilityShortcuts``
# sweeps.
_SCALED = [
    ("nearest", (1, 1, 4521), {"scale_factor": 1.1}),
    ("nearest", (1, 1, 2155), {"scale_factor": 2.5}),
    ("nearest", (1, 1, 3, 997, 1501), {"scale_factor": (1.5, 0.37, 1.7)}),
    ("nearest-exact", (1, 1, 56, 100), {"size": (56, 115)}),
    ("nearest-exact", (1, 1, 4521), {"scale_factor": 1.1}),
    ("linear", (1, 1, 2835), {"scale_factor": 0.379}),
    ("linear", (1, 1, 4521), {"scale_factor": 1.1}),
    ("bilinear", (1, 2, 37, 2155), {"scale_factor": (1.7, 2.3)}),
    ("bicubic", (1, 2, 37, 1999), {"scale_factor": (1.7, 0.43)}),
    ("trilinear", (1, 1, 3, 41, 1503), {"scale_factor": (1.5, 0.61, 1.3)}),
]


def _scaled_cases() -> list[Any]:
    cases = []
    for mode, shape, kw in _SCALED:
        interpolating = "linear" in mode or mode == "bicubic"
        for corners in (False, True) if interpolating else (None,):
            tag = {None: "-", False: "edges", True: "corners"}[corners]
            cases.append(
                pytest.param(
                    mode,
                    shape,
                    {**kw, "align_corners": corners},
                    id=f"{mode}-{shape[-1]}-{tag}",
                )
            )
    return cases


@pytest.mark.parity
class TestScaleFactorPositions:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize(("mode", "shape", "kw"), _scaled_cases())
    def test_values_match_the_reference(
        self,
        ref: Any,
        device: str,
        dtype: str,
        mode: str,
        shape: tuple[int, ...],
        kw: dict[str, Any],
    ) -> None:
        _skip_unsupported(device, dtype)
        data = np.random.default_rng(sum(shape)).standard_normal(shape)
        x = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device)
        got = F.interpolate(x, mode=mode, **kw)
        want = ref.nn.functional.interpolate(
            ref.tensor(data, dtype=getattr(ref, dtype)), mode=mode, **kw
        )
        np.testing.assert_allclose(_host(got), _host(want), atol=_ATOL[dtype], rtol=0)


# Plain nearest on a 2-D input copies an axis that keeps its length and
# halves the index of one that doubles it, whatever the scale factor said —
# but the reference does so only on the kernel it takes for a small output
# (height plus width at most 128); a larger output steps by 1 / scale_factor
# (LCD-429).  Each row sweeps the output height across that line: below it
# the kept or doubled axis is pinned, above it resampled.
_SHORTCUT_ROWS = [
    # (width, width scale, height scale): a kept or doubled width.
    pytest.param(32, 1.031, 1.694, id="keep-32"),
    pytest.param(8, 1.031, 1.694, id="keep-8"),
    pytest.param(10, 2.01, 1.694, id="double-10"),
    pytest.param(32, 1.031, 2.01, id="keep-32-tall"),
]
_SHORTCUT_LIMIT = 128


def _heights_across(width_out: int, height_scale: float) -> list[int]:
    """Input heights whose output height lands on either side of the line."""
    line = _SHORTCUT_LIMIT - width_out
    heights = [h for h in range(2, 200) if abs(math.floor(h * height_scale) - line) <= 6]
    assert any(math.floor(h * height_scale) <= line for h in heights)
    assert any(math.floor(h * height_scale) > line for h in heights)
    return heights


@pytest.mark.parity
class TestNearestCompatibilityShortcuts:
    @pytest.mark.parametrize(("width", "width_scale", "height_scale"), _SHORTCUT_ROWS)
    def test_values_follow_the_reference_across_the_line(
        self,
        ref: Any,
        device: str,
        width: int,
        width_scale: float,
        height_scale: float,
    ) -> None:
        width_out = math.floor(width * width_scale)
        kw = {"scale_factor": (height_scale, width_scale), "mode": "nearest"}
        for height in _heights_across(width_out, height_scale):
            data = np.arange(height * width, dtype=np.float32).reshape(1, 1, height, width)
            got = F.interpolate(lucid.tensor(data, device=device), **kw)
            want = ref.nn.functional.interpolate(ref.tensor(data), **kw)
            np.testing.assert_array_equal(_host(got), _host(want), err_msg=f"{height}")

    @pytest.mark.parametrize(("width", "width_scale", "height_scale"), _SHORTCUT_ROWS)
    def test_gradient_is_the_adjoint_of_the_forward(
        self, device: str, width: int, width_scale: float, height_scale: float
    ) -> None:
        # The reference's backward pins at every size, so above the line it
        # is not the adjoint of its own forward and is no oracle: each input
        # sample gathers the gradient of every output that copied it.
        width_out = math.floor(width * width_scale)
        kw = {"scale_factor": (height_scale, width_scale), "mode": "nearest"}
        for height in _heights_across(width_out, height_scale):
            x = lucid.zeros(1, 1, height, width, device=device, requires_grad=True)
            F.interpolate(x, **kw).sum().backward()
            index = lucid.arange(float(height * width)).reshape(1, 1, height, width)
            copied = _host(F.interpolate(index.to(device), **kw))
            counts = np.bincount(copied.astype(np.int64).ravel(), minlength=height * width)
            assert x.grad is not None
            assert _host(x.grad).ravel().tolist() == counts.tolist(), f"{height}"
