"""Integer and bool results keep the reference's dtype (LCD-668).

``sum`` has always widened ``bool`` / ``int8`` / ``int16`` / ``int32`` to ``int64``
before adding, but ``trace`` kept the narrow input dtype (so ``int8`` overflowed
where the reference does not), and ``square`` of a ``bool`` answered ``bool`` where
the reference answers the ``int64`` of ``x ** 2``.  The matrix below holds every
op that adds up or multiplies integer elements to the reference's result dtype
and values, on both streams.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_DTYPES = [lucid.bool, lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _matrix(dtype: lucid.dtype) -> list[list[int]]:
    return [[1, 0], [1, 1]] if dtype == lucid.bool else [[1, 2], [3, 4]]


_OPS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], Callable[[object], object]]] = {
    "sum": (lambda x: lucid.sum(x), lambda x: x.sum()),
    "sum_dim": (lambda x: lucid.sum(x, 0, keepdim=True), lambda x: x.sum(0, keepdim=True)),
    "prod": (lambda x: lucid.prod(x), lambda x: x.prod()),
    "cumsum": (lambda x: lucid.cumsum(x, 0), lambda x: x.cumsum(0)),
    "cumprod": (lambda x: lucid.cumprod(x, 1), lambda x: x.cumprod(1)),
    "nansum": (lambda x: lucid.nansum(x), lambda x: x.nansum()),
    "trace": (lambda x: lucid.trace(x), lambda x: x.trace()),
    "square": (lambda x: lucid.square(x), lambda x: x.square()),
}


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("name", list(_OPS))
def test_integer_result_dtype_and_values_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, name: str
) -> None:
    """One result-dtype rule for every op that accumulates integer elements."""
    lucid_fn, ref_fn = _OPS[name]
    rx = ref.tensor(_matrix(dtype), dtype=_ref_dtype(ref, dtype))
    x = lucid.tensor(_matrix(dtype), dtype=dtype, device=device)
    try:
        want = ref_fn(rx)
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            lucid_fn(x)
        return
    got = lucid_fn(x)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES[:3]))
def test_trace_adds_up_in_int64(device: str, dtype: lucid.dtype) -> None:
    """The sum of the diagonal has int64 headroom, as ``sum`` does."""
    limit = {lucid.int8: 100, lucid.int16: 30000, lucid.int32: 2_000_000_000}[dtype]
    x = lucid.tensor([[limit, 0], [0, limit]], dtype=dtype, device=device)
    out = lucid.trace(x)
    assert out.dtype == lucid.int64
    assert out.item() == 2 * limit
    assert out.item() == lucid.sum(lucid.diagonal(x)).item()


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_batched_trace_follows_the_two_dimensional_rule(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    """A trace over leading matrix dimensions widens like the 2-D one."""
    vals = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    got = lucid.trace(lucid.tensor(vals, dtype=dtype, device=device))
    want = ref.tensor(vals, dtype=_ref_dtype(ref, dtype)).diagonal(0, 0, 1).sum(-1)
    assert got.dtype == lucid.int64
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_trace_of_bool_is_refused_as_in_the_reference(ref: ModuleType, device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    with pytest.raises(NotImplementedError):
        ref.tensor([[True, False], [True, True]]).trace()
    with pytest.raises(NotImplementedError):
        lucid.trace(lucid.tensor([[True, False], [True, True]], device=device))


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_square_of_bool_is_an_int64_power(ref: ModuleType, device: str) -> None:
    """``square`` is ``x ** 2`` (int64 for a bool), not the logical and of ``x * x``."""
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    vals = [True, False, True, True]
    b = lucid.tensor(vals, device=device)
    rb = ref.tensor(vals)
    for got, want in ((lucid.square(b), rb.square()), (b.square(), rb.square())):
        assert got.dtype == lucid.int64
        assert got.tolist() == want.tolist()
    assert (b * b).dtype == lucid.bool
    assert (rb * rb).dtype == ref.bool

    # In place the int64 answer cannot be written back into a bool, for ``cube_``
    # (``x ** 3``) as for ``square_``.
    for method in ("square_", "cube_"):
        with pytest.raises(RuntimeError, match="can't be cast"):
            getattr(b.clone(), method)()
    with pytest.raises(RuntimeError, match="can't be cast"):
        rb.clone().square_()
    with pytest.raises(RuntimeError, match="can't be cast"):
        rb.clone().pow_(3)
