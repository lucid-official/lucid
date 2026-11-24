"""The op table behind the scale-factor sweeps (LCD-1076, LCD-1078, LCD-1088).

Every op that takes a scale factor (``alpha``, ``beta``, ``value``, ``weight``), in every spelling
the reference has for it -- the free function, the ``Tensor`` method, the in-place twin -- as a
``call(side, scalar)`` that builds its operands on one side (Lucid or the reference, one dtype, one
device) and passes the scalar in.  ``test_scale_factor_reading.py`` passes Python numbers,
``test_scale_factor_tensor.py`` 0-d tensors, and both hold the answer to the reference's.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import lucid
from lucid.test._fixtures.devices import devices_supporting

INTS = ("int8", "int16", "int32", "int64")
HALF = ("float16", "bfloat16")
#: One unit in the last place of each half type, relative: what a double rounding can cost.
HALF_RTOL = {"float16": 2.0**-10, "bfloat16": 2.0**-7}
FLOATS = (*HALF, "float32", "float64")
COMPLEX = ("complex64", "complex128")
# Metal holds neither float64 nor complex128 (``devices_supporting`` leaves those cells out).
ADD_DTYPES = (*INTS, *FLOATS, *COMPLEX)
PRODUCT_DTYPES = (*INTS, *FLOATS, *COMPLEX)


class Side:
    """One library, one dtype and one device: builds the operands of a cell."""

    def __init__(self, lib: Any, dtype: str, device: str) -> None:
        self.dtype = getattr(lib, dtype)
        self.lib = lib
        self.kwargs: dict[str, Any] = {"device": device} if lib is lucid else {}

    def ones(self, *shape: int) -> Any:
        return self.lib.ones(shape, dtype=self.dtype, **self.kwargs)

    def zeros(self, *shape: int) -> Any:
        return self.lib.zeros(shape, dtype=self.dtype, **self.kwargs)

    def positions(self, *values: int) -> Any:
        return self.lib.tensor(list(values), **self.kwargs)

    def scalar(self, dtype: str, value: complex, *, requires_grad: bool = False) -> Any:
        """A 0-d tensor of ``dtype`` holding ``value``: on this side's device when it can hold the
        dtype there (Metal has no float64 or complex128), else on the CPU."""
        kwargs = dict(self.kwargs)
        if self.lib is lucid and kwargs["device"] not in devices_supporting(getattr(lucid, dtype)):
            kwargs["device"] = "cpu"
        return self.lib.tensor(
            value, dtype=getattr(self.lib, dtype), requires_grad=requires_grad, **kwargs
        )

    def tensor(self, values: Any, dtype: str) -> Any:
        """A tensor of ``dtype`` with ``values`` (a nested list) on this side's device."""
        return self.lib.tensor(values, dtype=getattr(self.lib, dtype), **self.kwargs)


class Op(NamedTuple):
    call: Callable[[Side, Any], Any]
    dtypes: tuple[str, ...]
    #: Whether the reference accumulates a half tensor's scalar in float32 (see the docstring).
    op_math: bool = False


Shape = tuple[int, ...]


def _spelling(
    form: str, fn: str, shapes: Sequence[Shape], keyword: str | None = "alpha"
) -> Callable[[Side, Any], Any]:
    """One spelling of ``fn`` over ones of ``shapes``, the scalar passed as ``keyword``.

    ``form`` is the free function (``"free"``), the ``Tensor`` method (``"method"``) or its
    in-place twin (``"inplace"``, which answers the tensor it wrote).  ``keyword=None`` passes
    the scalar positionally, as ``lerp`` takes its ``weight``.
    """

    def call(side: Side, scalar: Any) -> Any:
        operands = [side.ones(*shape) for shape in shapes]
        named = {} if keyword is None else {keyword: scalar}
        positional = [scalar] if keyword is None else []
        if form == "free":
            return getattr(side.lib, fn)(*operands, *positional, **named)
        head, *rest = operands
        out = getattr(head, fn)(*rest, *positional, **named)
        return head if form == "inplace" else out

    return call


def _beside_a_number(fn: str) -> Callable[[Side, Any], Any]:
    """``fn(ones(2), 2, alpha=scalar)``: the other operand is a Python number too."""

    def call(side: Side, scalar: Any) -> Any:
        return getattr(side.lib, fn)(side.ones(2), 2, alpha=scalar)

    return call


def _index_add(side: Side, scalar: Any) -> Any:
    """Zeros take the source, so a bool tensor shows the scalar's truth, not a saturated ``True``."""
    return side.lib.index_add(side.zeros(3), 0, side.positions(0, 1), side.ones(2), alpha=scalar)


_PRODUCTS: dict[str, tuple[Shape, ...]] = {
    "addmm": ((2, 2), (2, 2), (2, 2)),
    "addmv": ((2,), (2, 2), (2,)),
    "addbmm": ((2, 2), (3, 2, 2), (3, 2, 2)),
    "baddbmm": ((3, 2, 2), (3, 2, 2), (3, 2, 2)),
    "addr": ((2, 2), (2,), (2,)),
}
# ``addr`` takes a bool tensor, as ``add`` does; the other products have no bool kernel.
_ACCUMULATING = ("addmm", "addbmm", "baddbmm")
_VALUED = (*INTS, *FLOATS, *COMPLEX)


def _ops() -> dict[str, Op]:
    ops: dict[str, Op] = {}
    for fn in ("add", "sub", "rsub", "subtract"):
        # a bool ``sub`` is refused by name, which the two spell differently
        dtypes = ("bool", *ADD_DTYPES) if fn == "add" else ADD_DTYPES
        ops[fn] = Op(_spelling("free", fn, [(2,), (2,)]), dtypes)
        ops[f"{fn}-number"] = Op(_beside_a_number(fn), ADD_DTYPES)
    for fn in ("add", "sub"):
        dtypes = ("bool", *ADD_DTYPES) if fn == "add" else ADD_DTYPES
        ops[f"{fn}-method"] = Op(_spelling("method", fn, [(2,), (2,)]), dtypes)
        ops[f"{fn}_"] = Op(_spelling("inplace", f"{fn}_", [(2,), (2,)]), dtypes)
    for fn, shapes in _PRODUCTS.items():
        for keyword in ("alpha", "beta"):
            ops[f"{fn}-{keyword}"] = Op(
                _spelling("free", fn, shapes, keyword),
                ("bool", *PRODUCT_DTYPES) if fn == "addr" else PRODUCT_DTYPES,
                op_math=fn in _ACCUMULATING,
            )
    ops["addmm-method"] = Op(
        _spelling("method", "addmm", _PRODUCTS["addmm"], "beta"), PRODUCT_DTYPES, op_math=True
    )
    ops["addmm_"] = Op(
        _spelling("inplace", "addmm_", _PRODUCTS["addmm"], "alpha"), PRODUCT_DTYPES, op_math=True
    )
    ops["addr_"] = Op(_spelling("inplace", "addr_", _PRODUCTS["addr"], "beta"), PRODUCT_DTYPES)
    for fn in ("addcmul", "addcdiv"):
        dtypes = _VALUED if fn == "addcmul" else (*FLOATS, *COMPLEX)
        ops[fn] = Op(_spelling("free", fn, [(2,)] * 3, "value"), dtypes, op_math=True)
        ops[f"{fn}_"] = Op(
            _spelling("inplace", f"{fn}_", [(2,)] * 3, "value"), dtypes, op_math=True
        )
    ops["lerp"] = Op(_spelling("free", "lerp", [(2,), (2,)], None), FLOATS, op_math=True)
    ops["lerp_"] = Op(_spelling("inplace", "lerp_", [(2,), (2,)], None), FLOATS, op_math=True)
    ops["index_add"] = Op(_index_add, ("bool", *INTS, *FLOATS, *COMPLEX))
    return ops


OPS = _ops()


def outcome(call: Callable[[], Any]) -> tuple[str, Any, Any]:
    """``("raised", class, message)`` or ``("answered", dtype name, values)``."""
    try:
        out = call()
    except (RuntimeError, TypeError, ValueError, OverflowError) as err:
        return "raised", type(err), str(err)
    return "answered", str(out.dtype).rsplit(".", 1)[-1], out.tolist()


def same_values(got: Any, want: Any, rtol: float = 0.0) -> bool:
    """Equal nested lists of numbers to within ``rtol``, a NaN equal to a NaN (in a complex too)."""
    if isinstance(got, list):
        return (
            isinstance(want, list)
            and len(got) == len(want)
            and all(same_values(g, w, rtol) for g, w in zip(got, want, strict=True))
        )
    if isinstance(got, complex):
        return same_values(got.real, want.real, rtol) and same_values(got.imag, want.imag, rtol)
    if isinstance(got, float) and math.isnan(got):
        return isinstance(want, float) and math.isnan(want)
    return bool(got == want or abs(got - want) <= rtol * abs(want))
