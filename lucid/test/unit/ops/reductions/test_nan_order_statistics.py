"""NaN-skipping order statistics, one slice at a time (LCD-376).

``nanmedian`` along a dim used to pick one sorted position for every slice
from the *average* non-NaN count, so a slice with a different count got
another slice's median — silently.  It built that position on the CPU,
which a Metal input refused; it filled NaNs with ``inf``, which an integer
input refused; and it read the count back to the host.  Every slice now
reads its own count, a tensor on the input's device, and the median is
gathered from the input so the gradient lands on the entry ``indices``
names.

``nanquantile`` shared the inf fill and, over the whole tensor, the host
read; ``quantile`` filled an integer ``interpolation="lower"`` result with
NaN.  The cases below hold all three to the reference: values, indices
and gradients, on every dim with and without ``keepdim``, for data with no
NaN, some, an all-NaN slice and ties, in float32, float16 and int64, on
CPU and Metal.
"""

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid

pytestmark = pytest.mark.parity

_NAN = math.nan

# Distinct numbers, so the reference's indices are not a tie-break choice.
_DATA: dict[str, list[list[float]]] = {
    "no_nan": [[3, 7, 1, 9], [4, 0, 6, 2], [11, 5, 8, 10]],
    "some_nan": [[3, _NAN, 1, 9], [4, 0, _NAN, _NAN], [_NAN, 5, 8, 10]],
    # Row 0 and column 2 are all NaN; the other slices hold 1 to 3 numbers.
    "all_nan_slice": [
        [_NAN, _NAN, _NAN, _NAN],
        [4, _NAN, _NAN, 2],
        [_NAN, 5, _NAN, 10],
    ],
    # Both even and odd counts per slice, so the lower median is exercised.
    "uneven_counts": [[1, _NAN, _NAN, _NAN], [1, 2, 3, _NAN], [6, 4, 2, 0]],
}
_FLOAT_ONLY = {"some_nan", "all_nan_slice", "uneven_counts"}
_TIES = [[2, 1, 2, 2], [5, 5, 5, 5], [0, 3, 3, 1]]

_DTYPES = ("float32", "float16", "int64")
_DIMS = (None, 0, 1, -1)


def _cases() -> list[object]:
    return [
        pytest.param(name, dtype, id=f"{name}-{dtype}")
        for name in _DATA
        for dtype in _DTYPES
        if not (dtype == "int64" and name in _FLOAT_ONLY)
    ]


def _pair(
    ref: ModuleType, data: object, dtype: str, device: str, grad: bool
) -> tuple[lucid.Tensor, object]:
    x = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device)
    want = ref.tensor(data, dtype=getattr(ref, dtype))
    if grad:
        x.requires_grad_(True)
        want.requires_grad_(True)
    return x, want


def _np(t: object) -> np.ndarray:
    return np.asarray(t.detach().cpu().float().numpy())


def _same(got: object, want: object) -> None:
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(_np(got), _np(want))


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("name,dtype", _cases())
def test_nanmedian_matches_reference(
    ref: ModuleType, device: str, name: str, dtype: str, dim: int | None, keepdim: bool
) -> None:
    grad = dtype != "int64"
    x, want_x = _pair(ref, _DATA[name], dtype, device, grad)
    if dim is None:
        got, want = lucid.nanmedian(x), ref.nanmedian(want_x)
        _same(got, want)
        outs = (got, want)
    else:
        result = lucid.nanmedian(x, dim, keepdim)
        assert isinstance(result, tuple)
        values, indices = result
        assert values is result.values and indices is result.indices
        m_val, m_idx = x.nanmedian(dim, keepdim)
        _same(m_val, values)
        _same(m_idx, indices)
        want_values, want_indices = ref.nanmedian(want_x, dim, keepdim)
        _same(values, want_values)
        _same(indices, want_indices)
        assert indices.dtype == lucid.int64
        assert values.device == x.device and indices.device == x.device
        outs = (values, want_values)
    if grad:
        outs[0].sum().backward()
        outs[1].sum().backward()
        assert isinstance(x.grad, lucid.Tensor)
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("name,dtype", _cases())
def test_median_matches_reference(
    ref: ModuleType, device: str, name: str, dtype: str, dim: int | None, keepdim: bool
) -> None:
    grad = dtype != "int64"
    x, want_x = _pair(ref, _DATA[name], dtype, device, grad)
    if dim is None:
        got, want = lucid.median(x), ref.median(want_x)
        _same(got, want)
        _same(x.median(), want)
        outs = (got, want)
    else:
        result = lucid.median(x, dim, keepdim)
        assert isinstance(result, tuple)
        values, indices = result
        assert values is result.values and indices is result.indices
        m_val, m_idx = x.median(dim, keepdim)
        _same(m_val, values)
        _same(m_idx, indices)
        want_values, want_indices = ref.median(want_x, dim, keepdim)
        _same(values, want_values)
        _same(indices, want_indices)
        assert indices.dtype == lucid.int64
        assert values.device == x.device and indices.device == x.device
        outs = (values, want_values)
    if grad:
        outs[0].sum().backward()
        outs[1].sum().backward()
        assert isinstance(x.grad, lucid.Tensor)
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("dtype", ("float32", "float16"))
def test_whole_tensor_median_shares_its_gradient_between_ties(
    ref: ModuleType, device: str, dtype: str
) -> None:
    for data in (_TIES, [[math.inf, math.inf, _NAN, 1.0]]):
        x, want_x = _pair(ref, data, dtype, device, True)
        got, want = lucid.nanmedian(x), ref.nanmedian(want_x)
        _same(got, want)
        got.backward()
        want.backward()
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)
    for data in (_TIES, [[3.0, _NAN, 1.0, _NAN]]):
        x, want_x = _pair(ref, data, dtype, device, True)
        got, want = lucid.median(x), ref.median(want_x)
        _same(got, want)
        got.backward()
        want.backward()
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("dim", (0, 1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_tied_median_names_an_entry_holding_it(
    ref: ModuleType, device: str, dtype: str, dim: int
) -> None:
    # Which of several equal entries the reference names is its own choice;
    # what holds is the value, and that ``indices`` points at it.
    x, want_x = _pair(ref, _TIES, dtype, device, False)
    for fn_l, fn_r in ((lucid.nanmedian, ref.nanmedian), (lucid.median, ref.median)):
        values, indices = fn_l(x, dim, True)
        _same(values, fn_r(want_x, dim, True).values)
        _same(lucid.take_along_dim(x, indices, dim), values)


@pytest.mark.parametrize("dtype", ("float32", "int64"))
def test_an_empty_input_matches_reference(ref: ModuleType, device: str, dtype: str) -> None:
    x = lucid.zeros(0, 4, dtype=getattr(lucid, dtype), device=device)
    want_x = ref.zeros(0, 4, dtype=getattr(ref, dtype))
    for fn_l, fn_r in (
        (lucid.nanmedian, ref.nanmedian),
        (lucid.median, ref.median),
        (lucid.mode, ref.mode),
    ):
        for dim in (1, -1):
            values, indices = fn_l(x, dim)
            want_values, want_indices = fn_r(want_x, dim)
            _same(values, want_values)
            _same(indices, want_indices)
        with pytest.raises(IndexError):
            fn_r(want_x, 0)
        with pytest.raises(IndexError):
            fn_l(x, 0)
    if dtype == "float32":
        assert math.isnan(lucid.nanmedian(x).item())
        assert math.isnan(lucid.median(x).item())


def test_a_zero_dim_input_is_its_own_median(ref: ModuleType, device: str) -> None:
    x = lucid.tensor(5.0, device=device)
    for fn_l, fn_r in (
        (lucid.nanmedian, ref.nanmedian),
        (lucid.median, ref.median),
        (lucid.mode, ref.mode),
    ):
        for dim in (0, -1):
            for keepdim in (False, True):
                values, indices = fn_l(x, dim, keepdim)
                want_values, want_indices = fn_r(ref.tensor(5.0), dim, keepdim)
                _same(values, want_values)
                _same(indices, want_indices)
        with pytest.raises(IndexError):
            fn_l(x, 1)


def test_bool_and_out_of_range_dims_are_refused(device: str) -> None:
    for fn in (lucid.nanmedian, lucid.median):
        with pytest.raises(NotImplementedError):
            fn(lucid.tensor([True, False], device=device))
        with pytest.raises(IndexError):
            fn(lucid.zeros(2, 3, device=device), 2)


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", (0, 1, -1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", (*_DTYPES, "bool_"))
def test_mode_matches_reference(
    ref: ModuleType, device: str, dtype: str, dim: int, keepdim: bool
) -> None:
    raw = [[True, False, True, False], [False, False, True, False]] if dtype == "bool_" else _TIES
    ref_dt = "bool" if dtype == "bool_" else dtype
    grad = dtype in ("float32", "float16")
    x = lucid.tensor(raw, dtype=getattr(lucid, dtype), device=device)
    want_x = ref.tensor(raw, dtype=getattr(ref, ref_dt))
    if grad:
        x.requires_grad_(True)
        want_x.requires_grad_(True)
    result = lucid.mode(x, dim, keepdim)
    values, indices = result
    assert values is result.values and indices is result.indices
    m_val, m_idx = x.mode(dim, keepdim)
    _same(m_val, values)
    _same(m_idx, indices)
    want_res = ref.mode(want_x, dim, keepdim)
    _same(values, want_res.values)
    assert indices.dtype == lucid.int64
    idx_along = indices if keepdim else indices.unsqueeze(dim)
    val_along = values if keepdim else values.unsqueeze(dim)
    _same(lucid.take_along_dim(x, idx_along, dim), val_along)
    if grad:
        values.sum().backward()
        assert isinstance(x.grad, lucid.Tensor)
        assert float(x.grad.sum().item()) == float(values.numel())


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", (0, 1, -1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_kthvalue_returns_values_and_indices_namedtuple(
    ref: ModuleType, device: str, dtype: str, dim: int, keepdim: bool
) -> None:
    grad = dtype != "int64"
    x, want_x = _pair(ref, _DATA["no_nan"], dtype, device, grad)
    result = lucid.kthvalue(x, 2, dim, keepdim)
    values, indices = result
    assert values is result.values and indices is result.indices
    m_val, m_idx = x.kthvalue(2, dim, keepdim)
    _same(m_val, values)
    _same(m_idx, indices)
    want_res = ref.kthvalue(want_x, 2, dim, keepdim)
    _same(values, want_res.values)
    _same(indices, want_res.indices)
    if grad:
        values.sum().backward()
        want_res.values.sum().backward()
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
@pytest.mark.parametrize("dim", (0, 1, -1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sort_returns_values_and_indices_namedtuple(
    ref: ModuleType, device: str, dtype: str, dim: int, descending: bool
) -> None:
    grad = dtype != "int64"
    x, want_x = _pair(ref, _TIES, dtype, device, grad)
    result = lucid.sort(x, dim=dim, descending=descending)
    values, indices = result
    assert values is result.values and indices is result.indices
    m_res = x.sort(dim=dim, descending=descending)
    _same(m_res.values, values)
    _same(m_res.indices, indices)
    want_res = ref.sort(want_x, dim=dim, descending=descending, stable=True)
    _same(values, want_res.values)
    _same(indices, want_res.indices)
    if grad:
        weights = lucid.arange(1, x.numel() + 1, dtype=x.dtype, device=device).reshape(x.shape)
        r_weights = ref.arange(1, want_x.numel() + 1, dtype=want_x.dtype).reshape(want_x.shape)
        (values * weights).sum().backward()
        (want_res.values * r_weights).sum().backward()
        np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


@pytest.mark.parametrize("op", ("max", "min"))
@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", (0, 1, -1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_max_and_min_dim_return_values_and_indices_namedtuple(
    ref: ModuleType, device: str, dtype: str, dim: int, keepdim: bool, op: str
) -> None:
    x, want_x = _pair(ref, _DATA["no_nan"], dtype, device, False)
    fn_l = getattr(lucid, op)
    fn_r = getattr(ref, op)
    result = fn_l(x, dim=dim, keepdim=keepdim)
    values, indices = result
    assert values is result.values and indices is result.indices
    m_res = getattr(x, op)(dim=dim, keepdim=keepdim)
    _same(m_res.values, values)
    _same(m_res.indices, indices)
    want_res = fn_r(want_x, dim=dim, keepdim=keepdim)
    _same(values, want_res.values)
    _same(indices, want_res.indices)
    assert isinstance(fn_l(x), lucid.Tensor)
    assert isinstance(fn_l(x, dim=(0, 1), keepdim=keepdim), lucid.Tensor)


@pytest.mark.parametrize("op", ("cummax", "cummin"))
@pytest.mark.parametrize("dim", (0, 1, -1), ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cummax_and_cummin_return_values_and_indices_namedtuple(
    ref: ModuleType, device: str, dtype: str, dim: int, op: str
) -> None:
    for data in (_TIES, _DATA["no_nan"]):
        x, want_x = _pair(ref, data, dtype, device, False)
        fn_l = getattr(lucid, op)
        fn_r = getattr(ref, op)
        result = fn_l(x, dim=dim)
        values, indices = result
        assert values is result.values and indices is result.indices
        m_res = getattr(x, op)(dim=dim)
        _same(m_res.values, values)
        _same(m_res.indices, indices)
        want_res = fn_r(want_x, dim=dim)
        _same(values, want_res.values)
        _same(indices, want_res.indices)
    if dtype != "int64":
        x, want_x = _pair(ref, _DATA["some_nan"], dtype, device, False)
        result = getattr(lucid, op)(x, dim=dim)
        want_res = getattr(ref, op)(want_x, dim=dim)
        _same(result.values, want_res.values)
        _same(result.indices, want_res.indices)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_msort_matches_reference(ref: ModuleType, device: str, dtype: str) -> None:
    grad = dtype != "int64"
    for data in (5, [3, 1, 2], _DATA["no_nan"]):
        x, want_x = _pair(ref, data, dtype, device, grad)
        got = lucid.msort(x)
        want = ref.msort(want_x)
        _same(got, want)
        _same(x.msort(), want)
        if grad:
            got.sum().backward()
            want.sum().backward()
            np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-3)


# ── nanquantile / quantile ────────────────────────────────────────────────────


@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
@pytest.mark.parametrize("name", list(_DATA))
def test_nanquantile_matches_reference(
    ref: ModuleType, device: str, name: str, dim: int | None, keepdim: bool
) -> None:
    x, want_x = _pair(ref, _DATA[name], "float32", device, True)
    got = lucid.nanquantile(x, 0.4, dim, keepdim=keepdim)
    want = ref.nanquantile(want_x, 0.4, dim, keepdim=keepdim)
    assert tuple(got.shape) == tuple(want.shape)
    assert tuple(x.nanquantile(0.4, dim, keepdim=keepdim).shape) == tuple(want.shape)
    np.testing.assert_allclose(_np(got), _np(want), rtol=1e-6, equal_nan=True)
    got.sum().backward()
    want.sum().backward()
    np.testing.assert_allclose(_np(x.grad), _np(want_x.grad), rtol=1e-6)


@pytest.mark.parametrize("q", [0.4, [0.25, 0.75]], ids=["scalar_q", "list_q"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["drop", "keep"])
@pytest.mark.parametrize("data", [5.0, [3.0, 1.0, 2.0], _DATA["no_nan"]])
def test_quantile_keepdim_and_method_match_reference(
    ref: ModuleType, device: str, data: object, keepdim: bool, q: object
) -> None:
    x, want_x = _pair(ref, data, "float32", device, False)
    q_ref = ref.tensor(q) if isinstance(q, list) else q
    for fn_l, m_name, fn_r in (
        (lucid.quantile, "quantile", ref.quantile),
        (lucid.nanquantile, "nanquantile", ref.nanquantile),
    ):
        got = fn_l(x, q, None, keepdim=keepdim)
        got_m = getattr(x, m_name)(q, None, keepdim=keepdim)
        want = fn_r(want_x, q_ref, None, keepdim=keepdim)
        assert tuple(got.shape) == tuple(want.shape)
        assert tuple(got_m.shape) == tuple(want.shape)
        np.testing.assert_allclose(_np(got), _np(want), rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("dim", _DIMS, ids=lambda d: f"dim{d}")
def test_integer_quantiles_need_no_nan_handling(
    ref: ModuleType, device: str, dim: int | None
) -> None:
    data = _DATA["no_nan"]
    x = lucid.tensor(data, dtype=lucid.int64, device=device)
    as_float = ref.tensor(data, dtype=ref.float32)
    # The reference refuses an integer quantile; Lucid answers what the same
    # numbers give in floating point, with or without NaN skipping.
    want = ref.quantile(as_float, 0.4, dim)
    for fn in (lucid.quantile, lucid.nanquantile):
        np.testing.assert_allclose(_np(fn(x, 0.4, dim)), _np(want), rtol=1e-6)
    lower = lucid.quantile(x, 0.4, dim, interpolation="lower")
    want_lower = ref.quantile(as_float, 0.4, dim, interpolation="lower")
    assert lower.dtype == lucid.int64
    np.testing.assert_array_equal(_np(lower), _np(want_lower))


# ── no host reads ─────────────────────────────────────────────────────────────

_CALLS: dict[str, Callable[[lucid.Tensor], object]] = {
    "nanmedian_dim": lambda x: lucid.nanmedian(x, 1),
    "nanmedian_all": lambda x: lucid.nanmedian(x),
    "median_dim": lambda x: lucid.median(x, 1),
    "median_all": lambda x: lucid.median(x),
    "mode_dim": lambda x: lucid.mode(x, 1),
    "nanquantile_dim": lambda x: lucid.nanquantile(x, 0.5, 1),
    "nanquantile_all": lambda x: lucid.nanquantile(x, 0.5),
    "quantile_all": lambda x: lucid.quantile(x, 0.5),
}


@pytest.mark.parametrize("call", list(_CALLS))
def test_no_value_is_read_back_to_the_host(
    monkeypatch: pytest.MonkeyPatch, device: str, call: str
) -> None:
    x = lucid.tensor(_DATA["all_nan_slice"], device=device).requires_grad_(True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{call} read a value back to the host")

    with monkeypatch.context() as guard:
        for method in ("numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
            guard.setattr(lucid.Tensor, method, forbidden)
        _CALLS[call](x)
