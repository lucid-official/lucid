"""``x[key] = v`` and ``lucid.scatter`` overwrite (CHA-158), in every dtype (CHA-27).

The engine's ``scatter`` — which every int, slice, mask and advanced-index
assignment reaches — was a scatter-add of the delta ``src - base[idx]``.
That is not an overwrite:

* NaN or inf in ``base`` survived: ``x[isnan(x)] = 0`` did nothing, and
  ``x[isinf(x)] = 0`` turned inf into NaN;
* a value much smaller than the one it replaced rounded away:
  ``x[0] = 1e-30`` wrote 0;
* bool could not become False, and a half overflowed in the subtraction;
* a repeated index summed its deltas: ``x[[1, 1]] = 3`` wrote -4 into 10.

It most likely existed because the CPU overwrite kernel took f32 and f64
only, so ``index_copy`` / ``index_put`` / ``put`` / ``masked_scatter``
refused bool, the integers and the halves on the CPU — and int64 and
complex everywhere, Metal included (MLX scatters no 8-byte element there,
and its CPU fallback hit the same refusal).  Now the kernel moves bytes by
element width, every dtype, and ``scatter`` is a true overwrite.

A position named twice keeps the value written last on the CPU (each line
along ``dim`` is written in index order, as the reference's CPU kernel
does); on Metal the writes race and one of the values survives, which one
unspecified — the reference's GPU behaviour too.
"""

import json
import math
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_ALL_DTYPES = (
    lucid.bool,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
)

_NP = {
    lucid.bool: np.bool_,
    lucid.int8: np.int8,
    lucid.int16: np.int16,
    lucid.int32: np.int32,
    lucid.int64: np.int64,
    lucid.float16: np.float16,
    lucid.bfloat16: np.float32,  # small integers are exact in bfloat16
    lucid.float32: np.float32,
    lucid.float64: np.float64,
    lucid.complex64: np.complex64,
    lucid.complex128: np.complex128,
}

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


def _t(values: object, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    t = lucid.tensor(np.asarray(values, dtype=_NP[dtype]), device=device)
    return t.to(dtype) if t.dtype != dtype else t


def _i(values: object, device: str) -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.int64), device=device)


def _same(got: lucid.Tensor, want: np.ndarray) -> None:
    assert got.tolist() == want.tolist()


# ── the reported cases, written out ─────────────────────────────────────────


class TestTheReportedCases:
    def test_a_mask_write_clears_nan_and_inf(self, device: str) -> None:
        x = lucid.tensor([1.0, math.nan, math.inf, -math.inf], device=device)
        x[lucid.isnan(x)] = 0.0
        x[lucid.isinf(x)] = 0.0
        assert x.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_an_int_write_over_nan_is_the_value(self, device: str) -> None:
        x = lucid.tensor([math.nan, math.inf, 2.0], device=device)
        x[0] = 5.0
        x[1:2] = lucid.tensor([-1.0], device=device)
        assert x.tolist() == [5.0, -1.0, 2.0]

    def test_a_tiny_value_is_not_rounded_away(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0], device=device)
        x[0] = 1e-30
        assert x.tolist() == [float(np.float32(1e-30)), 2.0]
        big = lucid.tensor([1e30, 3.0], device=device)
        big[0] = 1.0
        assert big.tolist() == [1.0, 3.0]

    def test_bool_can_become_false(self, device: str) -> None:
        b = lucid.tensor([True, True, True, True], device=device)
        b[0] = False
        b[lucid.tensor([False, True, False, False], device=device)] = False
        b[2:3] = lucid.tensor([False], device=device)
        assert b.tolist() == [False, False, False, True]

    def test_a_half_write_does_not_overflow(self, device: str) -> None:
        x = lucid.tensor([60000.0, 1.0], dtype=lucid.float16, device=device)
        x[0] = -60000.0
        assert x.tolist() == [-60000.0, 1.0]
        y = lucid.tensor([3e38, 1.0], dtype=lucid.bfloat16, device=device)
        y[0] = -3e38
        assert y[0].item() < -1e38 and math.isfinite(y[0].item())

    def test_a_repeated_index_writing_one_value_writes_it_once(self, device: str) -> None:
        x = lucid.full((3,), 10.0, device=device)
        x[_i([1, 1], device)] = 3.0
        assert x.tolist() == [10.0, 3.0, 10.0]
        y = lucid.zeros(3, device=device)
        i = _i([1, 1, 1], device)
        y[i] += 1
        assert y.tolist() == [0.0, 1.0, 0.0]


class TestARepeatedIndex:
    def test_the_cpu_keeps_the_last_write(self) -> None:
        x = lucid.full((3,), 10.0)
        x[_i([1, 1], "cpu")] = lucid.tensor([3.0, 5.0])
        assert x.tolist() == [10.0, 5.0, 10.0]
        out = lucid.scatter(lucid.full((3,), 10.0), 0, _i([1, 1], "cpu"), lucid.tensor([5.0, 7.0]))
        assert out.tolist() == [10.0, 7.0, 10.0]
        # Along an inner axis, each line in index order.
        base = lucid.zeros(2, 3)
        out = lucid.scatter(
            base, 1, _i([[2, 2, 0], [1, 1, 1]], "cpu"), lucid.arange(6.0).reshape(2, 3)
        )
        assert out.tolist() == [[2.0, 0.0, 1.0], [0.0, 5.0, 0.0]]

    @_needs_metal
    def test_metal_writes_one_of_the_values(self) -> None:
        out = lucid.scatter(
            lucid.full((4,), 10.0, device="metal"),
            0,
            _i([1, 1, 3], "metal"),
            lucid.tensor([5.0, 7.0, 1.0], device="metal"),
        ).tolist()
        assert out[0] == 10.0 and out[2] == 10.0 and out[3] == 1.0
        assert out[1] in (5.0, 7.0)


# ── every dtype, every way in ───────────────────────────────────────────────

# ``(name, write)``: ``write(x, idx, val)`` assigns into ``x`` — a NumPy array
# or a Lucid tensor — with ``idx`` making an index array and ``val`` a value
# array of ``x``'s dtype in the same library.
_Writer = Callable[[object, Callable[[object], object], Callable[[object], object]], None]

_SETITEM_1D: list[tuple[str, _Writer]] = [
    ("int", lambda x, idx, val: x.__setitem__(2, 0)),
    ("int-tensor", lambda x, idx, val: x.__setitem__(4, val([0])[0])),
    ("slice", lambda x, idx, val: x.__setitem__(slice(1, 3), val([5, 0]))),
    ("step", lambda x, idx, val: x.__setitem__(slice(None, None, 2), val([0, 5, 0]))),
    (
        "mask",
        lambda x, idx, val: x.__setitem__(
            idx([True, False, False, True, True, False]), val([0, 5, 0])
        ),
    ),
    (
        "mask-scalar",
        lambda x, idx, val: x.__setitem__(idx([False, True, False, True, False, True]), 0),
    ),
    ("advanced", lambda x, idx, val: x.__setitem__(idx([4, 0, 2]), val([1, 0, 3]))),
    ("negative", lambda x, idx, val: x.__setitem__(idx([-1, 1]), val([0, 2]))),
]

_SETITEM_2D: list[tuple[str, _Writer]] = [
    ("column", lambda x, idx, val: x.__setitem__((slice(None), 1), val([5, 0, 5]))),
    ("row-scalar", lambda x, idx, val: x.__setitem__(1, 0)),
    (
        "rows-block",
        lambda x, idx, val: x.__setitem__((idx([2, 0]), slice(1, 3)), val([[1, 0], [0, 4]])),
    ),
    (
        "pairs",
        lambda x, idx, val: x.__setitem__((idx([0, 2]), idx([3, 0])), val([0, 7])),
    ),
]


def _run_setitem(
    write: _Writer, base: list, dtype: lucid.dtype, device: str
) -> tuple[lucid.Tensor, np.ndarray]:
    npdt = _NP[dtype]
    want = np.asarray(base, dtype=npdt)

    def np_idx(v: object) -> np.ndarray:
        a = np.asarray(v)
        return a if a.dtype == np.bool_ else a.astype(np.int64)

    write(want, np_idx, lambda v: np.asarray(v, dtype=npdt))
    got = _t(base, dtype, device)

    def lu_idx(v: object) -> lucid.Tensor:
        a = np.asarray(v)
        return lucid.tensor(a if a.dtype == np.bool_ else a.astype(np.int64), device=device)

    write(got, lu_idx, lambda v: _t(v, dtype, device))
    return got, want


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_ALL_DTYPES))
@pytest.mark.parametrize(("kind", "write"), _SETITEM_1D, ids=[k for k, _ in _SETITEM_1D])
def test_setitem_writes_every_dtype(
    device: str, dtype: lucid.dtype, kind: str, write: _Writer
) -> None:
    got, want = _run_setitem(write, [1, 2, 3, 4, 5, 6], dtype, device)
    assert got.dtype == dtype
    _same(got, want)


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_ALL_DTYPES))
@pytest.mark.parametrize(("kind", "write"), _SETITEM_2D, ids=[k for k, _ in _SETITEM_2D])
def test_setitem_writes_every_dtype_in_two_dims(
    device: str, dtype: lucid.dtype, kind: str, write: _Writer
) -> None:
    base = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]]
    got, want = _run_setitem(write, base, dtype, device)
    assert got.dtype == dtype
    _same(got, want)


_BASE = [1, 2, 3, 4, 5, 6]
_POS = [4, 0, 2]
_SRC = [0, 7, 1]


def _expected(dtype: lucid.dtype, pos: list[int] = _POS) -> np.ndarray:
    out = np.asarray(_BASE, dtype=_NP[dtype])
    out[pos] = np.asarray(_SRC, dtype=_NP[dtype])
    return out


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_ALL_DTYPES))
class TestTheOverwritesTakeEveryDtype:
    """``index_copy`` / ``index_put`` / ``put`` refused int, half and bool on
    the CPU (CHA-27), and int64 and complex64 on Metal too."""

    def test_scatter(self, device: str, dtype: lucid.dtype) -> None:
        out = lucid.scatter(_t(_BASE, dtype, device), 0, _i(_POS, device), _t(_SRC, dtype, device))
        assert out.dtype == dtype
        _same(out, _expected(dtype))

    def test_index_copy(self, device: str, dtype: lucid.dtype) -> None:
        out = _t(_BASE, dtype, device).index_copy(0, _i(_POS, device), _t(_SRC, dtype, device))
        _same(out, _expected(dtype))

    def test_index_put(self, device: str, dtype: lucid.dtype) -> None:
        out = lucid.index_put(
            _t(_BASE, dtype, device), (_i(_POS, device),), _t(_SRC, dtype, device)
        )
        _same(out, _expected(dtype))

    def test_put(self, device: str, dtype: lucid.dtype) -> None:
        out = lucid.put(_t(_BASE, dtype, device), _i(_POS, device), _t(_SRC, dtype, device))
        _same(out, _expected(dtype))

    def test_masked_scatter(self, device: str, dtype: lucid.dtype) -> None:
        mask = lucid.tensor([True, False, True, False, True, False], device=device)
        out = lucid.masked_scatter(_t(_BASE, dtype, device), mask, _t(_SRC, dtype, device))
        _same(out, _expected(dtype, [0, 2, 4]))


# ── scatter's shapes ────────────────────────────────────────────────────────


class TestScatterShapes:
    def test_an_index_shorter_than_base_writes_its_corner(self, device: str) -> None:
        base = np.arange(24, dtype=np.float32).reshape(4, 6)
        src = np.arange(100, 115, dtype=np.float32).reshape(3, 5)
        idx = np.array([[3, 0, 1], [2, 2, 0]])
        out = lucid.scatter(
            lucid.tensor(base, device=device),
            0,
            _i(idx, device),
            lucid.tensor(src, device=device),
        )
        want = base.copy()
        for r in range(2):
            for c in range(3):
                want[idx[r, c], c] = src[r, c]
        _same(out, want)

    def test_a_view_base_is_read_as_its_elements(self, device: str) -> None:
        base = lucid.arange(12.0, device=device).reshape(3, 4).mT  # (4, 3), non-contiguous
        out = lucid.scatter(
            base,
            0,
            _i([[3, 0, 1]], device),
            lucid.tensor([[-1.0, -2.0, -3.0]], device=device),
        )
        want = np.arange(12.0, dtype=np.float32).reshape(3, 4).T.copy()
        want[3, 0], want[0, 1], want[1, 2] = -1.0, -2.0, -3.0
        _same(out, want)

    def test_a_0d_tensor_scatters_its_one_element(self, device: str) -> None:
        out = lucid.scatter(
            lucid.zeros((), device=device),
            0,
            _i(0, device),
            lucid.tensor(5.0, device=device),
        )
        assert out.shape == () and out.item() == 5.0
        out = lucid.scatter(
            lucid.zeros(3, device=device),
            0,
            _i(2, device),
            lucid.tensor(5.0, device=device),
        )
        assert out.tolist() == [0.0, 0.0, 5.0]

    def test_an_empty_index_writes_nothing(self, device: str) -> None:
        base = lucid.arange(6.0, device=device).reshape(2, 3)
        out = lucid.scatter(
            base,
            0,
            lucid.zeros(0, 3, dtype=lucid.int64, device=device),
            lucid.zeros(0, 3, device=device),
        )
        assert out.tolist() == base.tolist()

    def test_a_source_of_another_dtype_is_refused(self, device: str) -> None:
        # As the reference refuses it, and as every reducing scatter does:
        # the cast is the caller's to make (LCD-336).  DtypeMismatch is a
        # TypeError.
        with pytest.raises(TypeError, match="src must have self's dtype"):
            lucid.scatter(lucid.zeros(3, device=device), 0, _i([1], device), _i([7], device))

    def test_the_cpu_refuses_an_index_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            lucid.scatter(lucid.zeros(3), 0, _i([3], "cpu"), lucid.ones(1))
        with pytest.raises(IndexError):
            lucid.scatter(lucid.zeros(3), 0, _i([-4], "cpu"), lucid.ones(1))

    def test_an_index_into_an_empty_axis_is_refused(self, device: str) -> None:
        with pytest.raises(IndexError):
            lucid.scatter(
                lucid.zeros(0, device=device),
                0,
                _i([0], device),
                lucid.ones(1, device=device),
            )


# ``index_copy`` with an index outside the axis ended the process on the CPU:
# into an empty axis a SIGSEGV, at index 1000000 a SIGBUS — the reduce loop
# it shared wrote wherever the index pointed.  Each case runs in a child
# interpreter, so a regression fails its test instead of the whole run.
_OUT_OF_RANGE = {
    "empty-axis": "lucid.zeros(0, 5, device=D).index_copy("
    "0, lucid.tensor([0, 1], device=D), lucid.ones(2, 5, device=D))",
    "far": "lucid.zeros(3, 4, device=D).index_copy("
    "0, lucid.tensor([1000000], device=D), lucid.ones(1, 4, device=D))",
    "far-negative": "lucid.zeros(3, 4, device=D).index_copy("
    "0, lucid.tensor([-4], device=D), lucid.ones(1, 4, device=D))",
}

_CHILD = """
import json
import lucid
D = {device!r}
try:
    out = {{"value": ({expr}).tolist()}}
except IndexError:
    out = {{"raised": "IndexError"}}
except Exception as e:
    out = {{"raised": type(e).__name__}}
print(json.dumps(out))
"""


def _run_child(expr: str, device: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(device=device, expr=expr)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the child died ({proc.returncode}): {proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


class TestIndexCopyOutOfRange:
    @pytest.mark.parametrize("case", sorted(_OUT_OF_RANGE))
    def test_the_cpu_raises(self, case: str) -> None:
        assert _run_child(_OUT_OF_RANGE[case], "cpu") == {"raised": "IndexError"}

    @_needs_metal
    def test_metal_drops_the_write(self) -> None:
        # An empty axis is refused from the shapes alone.  An index value out
        # of range is dropped on Metal (policy B, LCD-228): the base keeps its
        # value.  -4 on an axis of 3 used to wrap twice, once by the backend
        # and once more by MLX, and wrote the last row.
        got = {case: _run_child(e, "metal") for case, e in _OUT_OF_RANGE.items()}
        zero_row = [0.0] * 4
        assert got == {
            "empty-axis": {"raised": "IndexError"},
            "far": {"value": [zero_row] * 3},
            "far-negative": {"value": [zero_row] * 3},
        }


# ── gradients ───────────────────────────────────────────────────────────────


class TestGradients:
    def test_setitem_routes_the_gradient(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True, device=device)
        v = lucid.tensor([10.0, 20.0, 30.0], requires_grad=True, device=device)
        y = x * 1.0
        y[_i([1, 1, 3], device)] = v
        (y * lucid.arange(5.0, device=device)).sum().backward()
        assert x.grad is not None and v.grad is not None
        # The written positions take no gradient back to ``x``; every element
        # of ``v`` aimed at a position gathers that position's gradient.
        assert x.grad.tolist() == [0.0, 0.0, 2.0, 0.0, 4.0]
        assert v.grad.tolist() == [1.0, 1.0, 3.0]

    def test_scatter_routes_the_gradient(self, device: str) -> None:
        b = lucid.ones(2, 4, requires_grad=True, device=device)
        s = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)
        w = lucid.arange(8.0, device=device).reshape(2, 4)
        out = lucid.scatter(b * 1.0, 1, _i([[3, 3], [0, 2]], device), s)
        (out * w).sum().backward()
        assert b.grad is not None and s.grad is not None
        assert b.grad.tolist() == [[0.0, 1.0, 2.0, 0.0], [0.0, 5.0, 0.0, 7.0]]
        assert s.grad.tolist() == [[3.0, 3.0], [4.0, 6.0]]

    def test_a_larger_source_sends_its_gradient_to_the_corner_it_gave(self, device: str) -> None:
        s = lucid.ones(2, 3, requires_grad=True, device=device)
        out = lucid.scatter(lucid.zeros(3, 3, device=device), 0, _i([[2, 0]], device), s)
        (out * 2.0).sum().backward()
        assert s.grad is not None
        assert s.grad.tolist() == [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]

    def test_the_gradient_is_differentiable_again(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        y = x * x
        y2 = y + 0.0
        y2[1] = 0.0
        (g,) = lucid.autograd.grad(y2.sum(), x, create_graph=True)
        (gg,) = lucid.autograd.grad(g.sum(), x)
        assert g.tolist() == [2.0, 0.0, 6.0]
        assert gg.tolist() == [2.0, 0.0, 2.0]

    def test_index_copy_is_differentiable_again(self, device: str) -> None:
        s = lucid.tensor([2.0, 3.0], requires_grad=True, device=device)
        out = lucid.zeros(4, device=device).index_copy(0, _i([3, 0], device), s * s)
        (g,) = lucid.autograd.grad((out * out).sum(), s, create_graph=True)
        (gg,) = lucid.autograd.grad(g.sum(), s)
        assert g.tolist() == [32.0, 108.0]  # d(s^4)/ds = 4 s^3
        assert gg.tolist() == [48.0, 108.0]  # 12 s^2


# ── against the reference ───────────────────────────────────────────────────


@pytest.mark.parity
class TestAgainstTheReference:
    def test_the_reported_cases(self, ref: object, device: str) -> None:
        r = ref  # type: ignore[assignment]
        x = lucid.tensor([1.0, math.nan, math.inf], device=device)
        x[lucid.isnan(x)] = 0.0
        x[lucid.isinf(x)] = 0.0
        rx = r.tensor([1.0, math.nan, math.inf])  # type: ignore[attr-defined]
        rx[r.isnan(rx)] = 0.0  # type: ignore[attr-defined]
        rx[r.isinf(rx)] = 0.0  # type: ignore[attr-defined]
        assert x.tolist() == rx.tolist()

        h = lucid.tensor([60000.0], dtype=lucid.float16, device=device)
        h[0] = -60000.0
        rh = r.tensor([60000.0], dtype=r.float16)  # type: ignore[attr-defined]
        rh[0] = -60000.0
        assert h.tolist() == rh.tolist()

        y = lucid.zeros(3, device=device)
        y[_i([1, 1, 1], device)] += 1
        ry = r.zeros(3)  # type: ignore[attr-defined]
        ry[r.tensor([1, 1, 1])] += 1  # type: ignore[attr-defined]
        assert y.tolist() == ry.tolist()

    def test_scatter_with_a_repeated_index_on_the_cpu(self, ref: object) -> None:
        r = ref  # type: ignore[assignment]
        base = np.zeros((3, 4), np.float32)
        idx = np.array([[2, 2, 0, 1], [0, 0, 0, 0]])
        src = np.arange(8, dtype=np.float32).reshape(2, 4)
        got = lucid.scatter(lucid.tensor(base), 0, lucid.tensor(idx), lucid.tensor(src))
        want = r.tensor(base).scatter(0, r.tensor(idx), r.tensor(src))  # type: ignore[attr-defined]
        assert got.tolist() == want.tolist()

    def test_gradients(self, ref: object, device: str) -> None:
        r = ref  # type: ignore[assignment]
        base = np.random.default_rng(3).standard_normal((3, 5)).astype(np.float32)
        src = np.random.default_rng(4).standard_normal((2, 5)).astype(np.float32)
        w = np.random.default_rng(5).standard_normal((3, 5)).astype(np.float32)
        idx = np.array([[2, 0, 2, 1, 0], [1, 0, 2, 1, 1]])

        b = lucid.tensor(base, requires_grad=True, device=device)
        s = lucid.tensor(src, requires_grad=True, device=device)
        out = lucid.scatter(b * 1.0, 0, _i(idx, device), s)
        (out * lucid.tensor(w, device=device)).sum().backward()

        rb = r.tensor(base, requires_grad=True)  # type: ignore[attr-defined]
        rs = r.tensor(src, requires_grad=True)  # type: ignore[attr-defined]
        rout = (rb * 1.0).scatter(0, r.tensor(idx), rs)  # type: ignore[attr-defined]
        (rout * r.tensor(w)).sum().backward()  # type: ignore[attr-defined]

        assert b.grad is not None and s.grad is not None
        np.testing.assert_allclose(b.grad.numpy(), rb.grad.numpy(), rtol=1e-6)
        np.testing.assert_allclose(s.grad.numpy(), rs.grad.numpy(), rtol=1e-6)

        # And through ``__setitem__`` with a repeated index.
        v = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        y = lucid.zeros(5, device=device) + 0.0
        y[_i([1, 1, 3], device)] = v
        (y * lucid.arange(5.0, device=device)).sum().backward()
        rv = r.tensor([1.0, 2.0, 3.0], requires_grad=True)  # type: ignore[attr-defined]
        ry = r.zeros(5) + 0.0  # type: ignore[attr-defined]
        ry[r.tensor([1, 1, 3])] = rv  # type: ignore[attr-defined]
        (ry * r.arange(5.0)).sum().backward()  # type: ignore[attr-defined]
        assert v.grad is not None
        assert v.grad.tolist() == rv.grad.tolist()
