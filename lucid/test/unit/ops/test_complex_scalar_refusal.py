"""A complex ``alpha`` / ``beta`` / ``value`` / ``weight`` on a tensor that cannot hold it (LCD-1076).

``lucid.add(x, y, alpha=1j)`` on real tensors answered ``x + y``: the scale cast the ``1j`` to the
tensor's dtype and the imaginary part was gone, with nothing to say so.  ``addmm``, ``addr``,
``addcmul`` and the rest scaled a tensor the same way (or promoted a real result to complex
through the operator), where the reference refuses the number.  The reference reads a Python
scalar two ways, and ``lucid/_ops/_scalar_cast.py`` holds both:

* ``add``, ``sub``, ``rsub`` (and ``addr`` on an integral tensor) judge a complex ``alpha`` by
  its type: refused on every tensor that is not complex, a zero imaginary part included;
* every other scaled op converts the scalar to the tensor's element type: a complex number with
  no imaginary part is its real part, one with an imaginary part is refused.

The sweep is op x tensor dtype x scalar x device and is checked against the reference, not a
table: a cell either is refused by both, with one exception class and one message, or is
answered by both, with one dtype and one value.  What the sweep leaves out, it leaves out because
the reference answers it differently for a reason that is not this one (a bool tensor, a bool
scalar, a complex matmul, a complex ``lerp``); the table below says which.
"""

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, NamedTuple

import pytest

import lucid
from lucid.test._fixtures.devices import devices_supporting

pytestmark = pytest.mark.parity

_SCALARS: dict[str, complex] = {
    "1j": 1j,
    "0j": 0j,
    "1+2j": 1 + 2j,
    "1+0j": 1 + 0j,
    "2.5": 2.5,
    "2": 2,
}
_ALL = tuple(_SCALARS)
_HALF = ("float16", "bfloat16")
_FLOATS = ("float16", "float32", "float64")
_COMPLEX = ("complex64", "complex128")
# Metal holds neither float64 nor complex128 (``devices_supporting`` leaves those cells out).
_ADD_DTYPES = ("int8", "int32", *_FLOATS, *_COMPLEX)
# The engine has no complex ``matmul`` / ``outer`` kernel, so a scaled product of complex tensors
# is refused whatever its scalar is; the complex scalar on a complex *accumulator* is pinned in
# ``test_zero_dim_operand_ops.py``.
_PRODUCT_DTYPES = ("int32", *_FLOATS)


class _Side:
    """One library, one dtype and one device: builds the operands of a cell."""

    def __init__(self, lib: Any, dtype: str, device: str) -> None:
        self.dtype = getattr(lib, dtype)
        self.lib = lib
        self.kwargs: dict[str, Any] = {"device": device} if lib is lucid else {}

    def ones(self, *shape: int) -> Any:
        return self.lib.ones(shape, dtype=self.dtype, **self.kwargs)

    def positions(self, *values: int) -> Any:
        return self.lib.tensor(list(values), **self.kwargs)


class _Op(NamedTuple):
    call: Callable[[_Side, Any], Any]
    dtypes: tuple[str, ...]


Shape = tuple[int, ...]


def _spelling(
    form: str, fn: str, shapes: Sequence[Shape], keyword: str | None = "alpha"
) -> Callable[[_Side, Any], Any]:
    """One spelling of ``fn`` over ones of ``shapes``, the scalar passed as ``keyword``.

    ``form`` is the free function (``"free"``), the ``Tensor`` method (``"method"``) or its
    in-place twin (``"inplace"``, which answers the tensor it wrote).  ``keyword=None`` passes
    the scalar positionally, as ``lerp`` takes its ``weight``.
    """

    def call(side: _Side, scalar: Any) -> Any:
        operands = [side.ones(*shape) for shape in shapes]
        named = {} if keyword is None else {keyword: scalar}
        positional = [scalar] if keyword is None else []
        if form == "free":
            return getattr(side.lib, fn)(*operands, *positional, **named)
        head, *rest = operands
        out = getattr(head, fn)(*rest, *positional, **named)
        return head if form == "inplace" else out

    return call


def _beside_a_number(fn: str) -> Callable[[_Side, Any], Any]:
    """``fn(ones(2), 2, alpha=scalar)``: the other operand is a Python number too."""

    def call(side: _Side, scalar: Any) -> Any:
        return getattr(side.lib, fn)(side.ones(2), 2, alpha=scalar)

    return call


def _index_add(side: Any, scalar: Any) -> Any:
    return side.lib.index_add(side.ones(3), 0, side.positions(0, 1), side.ones(2), alpha=scalar)


_PRODUCTS: dict[str, tuple[Shape, ...]] = {
    "addmm": ((2, 2), (2, 2), (2, 2)),
    "addmv": ((2,), (2, 2), (2,)),
    "addbmm": ((2, 2), (3, 2, 2), (3, 2, 2)),
    "baddbmm": ((3, 2, 2), (3, 2, 2), (3, 2, 2)),
    "addr": ((2, 2), (2,), (2,)),
}
_VALUED = ("int32", *_FLOATS, *_COMPLEX)


def _ops() -> dict[str, _Op]:
    ops: dict[str, _Op] = {}
    for fn in ("add", "sub", "rsub", "subtract"):
        ops[fn] = _Op(_spelling("free", fn, [(2,), (2,)]), _ADD_DTYPES)
        ops[f"{fn}-number"] = _Op(_beside_a_number(fn), _ADD_DTYPES)
    for fn in ("add", "sub"):
        ops[f"{fn}-method"] = _Op(_spelling("method", fn, [(2,), (2,)]), _ADD_DTYPES)
        ops[f"{fn}_"] = _Op(_spelling("inplace", f"{fn}_", [(2,), (2,)]), _ADD_DTYPES)
    for fn, shapes in _PRODUCTS.items():
        for keyword in ("alpha", "beta"):
            ops[f"{fn}-{keyword}"] = _Op(_spelling("free", fn, shapes, keyword), _PRODUCT_DTYPES)
    ops["addmm-method"] = _Op(
        _spelling("method", "addmm", _PRODUCTS["addmm"], "beta"), _PRODUCT_DTYPES
    )
    ops["addmm_"] = _Op(
        _spelling("inplace", "addmm_", _PRODUCTS["addmm"], "alpha"), _PRODUCT_DTYPES
    )
    ops["addr_"] = _Op(_spelling("inplace", "addr_", _PRODUCTS["addr"], "beta"), _PRODUCT_DTYPES)
    for fn in ("addcmul", "addcdiv"):
        dtypes = _VALUED if fn == "addcmul" else _VALUED[1:]
        ops[fn] = _Op(_spelling("free", fn, [(2,)] * 3, "value"), dtypes)
        ops[f"{fn}_"] = _Op(_spelling("inplace", f"{fn}_", [(2,)] * 3, "value"), dtypes)
    ops["lerp"] = _Op(_spelling("free", "lerp", [(2,), (2,)], None), _FLOATS)
    ops["lerp_"] = _Op(_spelling("inplace", "lerp_", [(2,), (2,)], None), _FLOATS)
    ops["index_add"] = _Op(_index_add, ("int32", *_FLOATS))
    return ops


_OPS = _ops()


def _cells() -> list[Any]:
    return [
        pytest.param(name, dtype, device, scalar, id=f"{name}-{dtype}-{device}-{scalar}")
        for name, op in _OPS.items()
        for dtype in op.dtypes
        for device in devices_supporting(getattr(lucid, dtype))
        for scalar in _ALL
    ]


def _outcome(call: Callable[[], Any]) -> tuple[str, Any, Any]:
    """``("raised", class, message)`` or ``("answered", dtype name, values)``."""
    try:
        out = call()
    except (RuntimeError, TypeError, ValueError) as err:
        return "raised", type(err), str(err)
    return "answered", str(out.dtype).rsplit(".", 1)[-1], out.tolist()


def _same_message(dtype: str, got: str, want: str) -> bool:
    """Exact, but for a half tensor, where the reference names the type its kernel reads.

    ``addmm`` reads a half scalar as ``float`` and ``addr`` / ``addmv`` as ``c10::Half``, a
    storage type no Lucid message should name; Lucid says ``float`` for every half tensor, so
    only the stem is compared there.
    """
    stem = "value cannot be converted to type"
    if dtype in _HALF and got.startswith(stem) and want.startswith(stem):
        return True
    return got == want


@pytest.mark.parametrize(("name", "dtype", "device", "scalar"), _cells())
def test_scalar_is_read_as_the_reference_reads_it(
    name: str, dtype: str, device: str, scalar: str, ref: ModuleType
) -> None:
    value = _SCALARS[scalar]
    op = _OPS[name]
    got = _outcome(lambda: op.call(_Side(lucid, dtype, device), value))
    want = _outcome(lambda: op.call(_Side(ref, dtype, device), value))
    assert got[0] == want[0], f"{got} vs {want}"
    if got[0] == "raised":
        assert got[1] is want[1]
        assert _same_message(dtype, got[2], want[2]), f"{got[2]!r} vs {want[2]!r}"
    else:
        assert got[1] == want[1]
        assert got[2] == want[2]


# ── what the reference raises, pinned without the reference ──────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_add_with_a_complex_alpha_on_a_float_tensor_is_refused(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(
        RuntimeError,
        match="For non-complex input tensors, argument alpha must not be a complex number.",
    ):
        lucid.add(x, x, alpha=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.int32))
def test_add_with_a_complex_alpha_on_an_integer_tensor_is_refused(device: str) -> None:
    x = lucid.tensor([1, 2], device=device)
    with pytest.raises(
        RuntimeError,
        match="For integral input tensors, argument alpha must not be a floating point number.",
    ):
        lucid.add(x, x, alpha=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_addmm_with_a_complex_beta_on_a_float_tensor_is_refused(device: str) -> None:
    m = lucid.ones((2, 2), device=device)
    with pytest.raises(
        RuntimeError, match="value cannot be converted to type float without overflow"
    ):
        lucid.addmm(m, m, m, beta=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_refused_in_place_write_leaves_the_tensor_alone(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(RuntimeError):
        x.add_(lucid.ones(2, device=device), alpha=1j)
    with pytest.raises(RuntimeError):
        x.addcmul_(lucid.ones(2, device=device), lucid.ones(2, device=device), value=1j)
    assert x.tolist() == [1.0, 2.0]


def test_a_complex_scalar_with_no_imaginary_part_keeps_a_real_result_real() -> None:
    """``addcmul``'s ``value`` is converted, not promoted: a float tensor stays float."""
    out = lucid.addcmul(lucid.ones(2), lucid.ones(2), lucid.ones(2), value=2 + 0j)
    assert out.dtype is lucid.float32
    assert out.tolist() == [3.0, 3.0]


# ── what still works: the scalar a tensor can hold ──────────────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.complex64))
def test_a_complex_alpha_on_a_complex_tensor_scales_it(device: str) -> None:
    c = lucid.tensor([1 + 1j, 2 - 1j], dtype=lucid.complex64, device=device)
    d = lucid.tensor([0.5, 3j], dtype=lucid.complex64, device=device)
    assert lucid.add(c, d, alpha=1j).tolist() == [1 + 1.5j, -1 - 1j]
    assert lucid.sub(c, d, alpha=1j).tolist() == [1 + 0.5j, 5 - 1j]
    assert lucid.rsub(c, d, alpha=1j).tolist() == [1.5 - 1j, -1 + 1j]
    out = lucid.addcmul(c, d, d, value=1j)
    assert out.tolist() == [1 + 1.25j, 2 - 10j]


@pytest.mark.parametrize("device", devices_supporting(lucid.complex128))
def test_complex_alpha_on_complex128_keeps_double_precision(device: str, ref: ModuleType) -> None:
    values = [1 + 1j, 2 - 1j]
    other = [0.5, 3j]
    got = lucid.add(
        lucid.tensor(values, dtype=lucid.complex128, device=device),
        lucid.tensor(other, dtype=lucid.complex128, device=device),
        alpha=0.1 + 0.2j,
    )
    want = ref.add(
        ref.tensor(values, dtype=ref.complex128),
        ref.tensor(other, dtype=ref.complex128),
        alpha=0.1 + 0.2j,
    )
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_real_alpha_on_a_real_tensor_is_unchanged(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    assert lucid.add(x, x, alpha=2).tolist() == [3.0, 6.0]
    assert lucid.sub(x, x, alpha=0.5).tolist() == [0.5, 1.0]
    assert lucid.rsub(x, 10.0, alpha=2).tolist() == [8.0, 6.0]
    assert lucid.rsub(x, 10).tolist() == [9.0, 8.0]
