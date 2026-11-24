"""``max(a, b)`` / ``min(a, b)``: the elementwise form of the extrema (LCD-1073).

The reference overloads ``max`` and ``min`` three ways -- ``f(input)``,
``f(input, dim, keepdim=False)`` and ``f(input, other)`` -- and tells the last two
apart by what the second argument *is*: a tensor, in its place or as ``other=``, is the
other operand; anything else is ``dim``.  Lucid read every second argument as ``dim``,
so ``max(a, b)`` of two tensors failed with "item() can only be called on a tensor with
one element" and the keyword ``other=`` was refused.

The defect class: an op the reference overloads by the *type* of its second argument
read that argument by position only.  The reference's schemas show ``max`` / ``min`` as
the one pair of public ops with a tensor second argument next to a ``dim`` one, so the
table below is the whole class: both ops, the free function and the method, every
spelling, every dtype pair, the broadcast, NaN and gradient rules of ``maximum`` /
``minimum`` (the elementwise form *is* that function), the reductions it must leave
alone, and the invalid mixes -- held to the reference's error class.
"""

import ast
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import lucid
from lucid import Tensor
from lucid._ops._reduce import _ValuesIndices
from lucid.test._fixtures.devices import device_dtype_params

_NAN = math.nan
_OPS = ("max", "min")
_ELEMENTWISE = {"max": "maximum", "min": "minimum"}

# Every spelling of the elementwise call: free function and method, the operand
# positional or named.
_SPELLINGS: dict[str, Callable[[ModuleType, str, Tensor, Tensor], Tensor]] = {
    "fn-positional": lambda fw, op, a, b: getattr(fw, op)(a, b),
    "fn-other": lambda fw, op, a, b: getattr(fw, op)(a, other=b),
    "fn-input-other": lambda fw, op, a, b: getattr(fw, op)(input=a, other=b),
    "method-positional": lambda fw, op, a, b: getattr(a, op)(b),
    "method-other": lambda fw, op, a, b: getattr(a, op)(other=b),
}

_A = [1.0, 5.0, -2.0, 0.0]
_B = [3.0, 2.0, -2.0, -1.0]
_WANT = {"max": [3.0, 5.0, -2.0, 0.0], "min": [1.0, 2.0, -2.0, -1.0]}


def _pair(device: str) -> tuple[Tensor, Tensor]:
    return lucid.tensor(_A, device=device), lucid.tensor(_B, device=device)


def _name(dtype: object) -> str:
    """``lucid.float32`` -> ``"float32"``, the spelling both frameworks share."""
    return str(dtype).rsplit(".", 1)[-1]


def _typed(fw: ModuleType, dtype: object, values: list[float], device: str | None) -> Tensor:
    """``values`` as a tensor of ``fw`` — read through the calls both frameworks share."""
    kwargs = {} if device is None else {"device": device}
    made: Tensor = fw.tensor(values, dtype=getattr(fw, _name(dtype)), **kwargs)
    return made


# ── the elementwise form ────────────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", _SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_two_tensors_are_the_elementwise_op(device: str, op: str, spelling: str) -> None:
    a, b = _pair(device)
    out = _SPELLINGS[spelling](lucid, op, a, b)
    assert out.tolist() == _WANT[op]
    assert out.device == a.device
    assert out.tolist() == getattr(lucid, _ELEMENTWISE[op])(a, b).tolist()


def test_the_repro_of_the_defect_computes(device: str) -> None:
    a, b = lucid.tensor([1.0, 5.0], device=device), lucid.tensor([3.0, 2.0], device=device)
    assert lucid.max(a, b).tolist() == [3.0, 5.0]
    assert lucid.min(a, b).tolist() == [1.0, 2.0]


_DTYPES = (
    lucid.bool,
    lucid.int8,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
)
_DTYPE_CELLS = device_dtype_params(_DTYPES, _DTYPES)


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("device", "left", "right"), _DTYPE_CELLS)
def test_a_dtype_pair_promotes_as_the_elementwise_op_does(
    op: str, device: str, left: lucid.dtype, right: lucid.dtype
) -> None:
    a = _typed(lucid, left, [1, 0, 1, 1], device)
    b = _typed(lucid, right, [0, 1, 1, 0], device)
    got = getattr(lucid, op)(a, b)
    want = getattr(lucid, _ELEMENTWISE[op])(a, b)
    assert got.dtype == want.dtype
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("device", "left", "right"), _DTYPE_CELLS)
def test_a_dtype_pair_matches_the_reference(
    ref: ModuleType, op: str, device: str, left: lucid.dtype, right: lucid.dtype
) -> None:
    got = getattr(lucid, op)(
        _typed(lucid, left, [1, 0, 1, 1], device), _typed(lucid, right, [0, 1, 1, 0], device)
    )
    want = getattr(ref, op)(
        _typed(ref, left, [1, 0, 1, 1], None), _typed(ref, right, [0, 1, 1, 0], None)
    )
    assert _name(got.dtype) == _name(want.dtype)
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(
    ("left", "right", "left_shape", "right_shape"),
    [
        (lucid.float16, lucid.float32, (3,), ()),
        (lucid.int64, lucid.float32, (), (3,)),
        (lucid.int32, lucid.float16, (2, 1), (3,)),
        (lucid.float32, lucid.float32, (), ()),
    ],
    ids=str,
)
def test_a_zero_dim_operand_ranks_below_a_dimensioned_one(
    ref: ModuleType,
    device: str,
    op: str,
    left: lucid.dtype,
    right: lucid.dtype,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    n_left, n_right = math.prod(left_shape), math.prod(right_shape)
    left_values, right_values = [1.0, 2.0, 3.0][:n_left], [3.0, 1.0, 2.0][:n_right]
    got = getattr(lucid, op)(
        _typed(lucid, left, left_values, device).reshape(left_shape),
        _typed(lucid, right, right_values, device).reshape(right_shape),
    )
    want = getattr(ref, op)(
        _typed(ref, left, left_values, None).reshape(left_shape),
        _typed(ref, right, right_values, None).reshape(right_shape),
    )
    assert _name(got.dtype) == _name(want.dtype)
    assert tuple(got.shape) == tuple(want.shape)
    assert got.tolist() == want.tolist()


# ── broadcasting ────────────────────────────────────────────────────────────────

_SHAPES = [((3, 1), (1, 4)), ((), (2, 3)), ((2, 3), (3,)), ((0, 3), (3,)), ((1,), (5,))]


def _values(shape: tuple[int, ...], offset: float) -> list[float]:
    return [((i * 7 + offset) % 5.0) - 2.0 for i in range(math.prod(shape))]


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("left", "right"), _SHAPES, ids=str)
def test_the_operands_broadcast(
    device: str, op: str, left: tuple[int, ...], right: tuple[int, ...]
) -> None:
    a = lucid.tensor(_values(left, 0.0), device=device).reshape(left)
    b = lucid.tensor(_values(right, 3.0), device=device).reshape(right)
    got = getattr(lucid, op)(a, b)
    want = getattr(lucid, _ELEMENTWISE[op])(a, b)
    assert tuple(got.shape) == tuple(want.shape)
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("left", "right"), _SHAPES, ids=str)
def test_the_broadcast_matches_the_reference(
    ref: ModuleType, device: str, op: str, left: tuple[int, ...], right: tuple[int, ...]
) -> None:
    a_values, b_values = _values(left, 0.0), _values(right, 3.0)
    got = getattr(lucid, op)(
        lucid.tensor(a_values, device=device).reshape(left),
        lucid.tensor(b_values, device=device).reshape(right),
    )
    want = getattr(ref, op)(ref.tensor(a_values).reshape(left), ref.tensor(b_values).reshape(right))
    assert tuple(got.shape) == tuple(want.shape)
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("op", _OPS)
def test_an_unbroadcastable_pair_is_refused_as_the_reference_refuses_it(
    ref: ModuleType, device: str, op: str
) -> None:
    with pytest.raises(Exception) as want:
        getattr(ref, op)(ref.ones(2, 3), ref.ones(4))
    with pytest.raises(Exception) as got:
        getattr(lucid, op)(lucid.ones(2, 3, device=device), lucid.ones(4, device=device))
    assert issubclass(got.type, want.type)


# ── NaN ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", _SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_a_nan_in_either_operand_propagates(device: str, op: str, spelling: str) -> None:
    a = lucid.tensor([_NAN, 1.0, _NAN, 2.0], device=device)
    b = lucid.tensor([1.0, _NAN, _NAN, 3.0], device=device)
    got = _SPELLINGS[spelling](lucid, op, a, b).tolist()
    assert [math.isnan(v) for v in got] == [True, True, True, False]
    assert got[3] == {"max": 3.0, "min": 2.0}[op]


# ── gradients ───────────────────────────────────────────────────────────────────

_X, _Y = [1.0, 5.0, 2.0, 0.0], [3.0, 2.0, 2.0, -1.0]


def _grads(
    fw: ModuleType, op: str, spelling: str, device: str | None
) -> tuple[list[float], list[float]]:
    kwargs = {} if device is None else {"device": device}
    a = fw.tensor(_X, requires_grad=True, **kwargs)
    b = fw.tensor(_Y, requires_grad=True, **kwargs)
    weights = fw.tensor([1.0 + k for k in range(len(_X))], **kwargs)
    (_SPELLINGS[spelling](fw, op, a, b) * weights).sum().backward()
    assert a.grad is not None and b.grad is not None
    return a.grad.tolist(), b.grad.tolist()


@pytest.mark.parametrize("spelling", _SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_the_gradient_goes_to_the_larger_or_smaller_and_a_tie_splits(
    device: str, op: str, spelling: str
) -> None:
    got = _grads(lucid, op, spelling, device)
    if op == "max":
        assert got == ([0.0, 2.0, 1.5, 4.0], [1.0, 0.0, 1.5, 0.0])
    else:
        assert got == ([1.0, 0.0, 1.5, 0.0], [0.0, 2.0, 1.5, 4.0])


@pytest.mark.parity
@pytest.mark.parametrize("spelling", _SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_the_gradient_matches_the_reference(
    ref: ModuleType, device: str, op: str, spelling: str
) -> None:
    assert _grads(lucid, op, spelling, device) == _grads(ref, op, spelling, None)


@pytest.mark.parametrize("op", _OPS)
def test_a_broadcast_operand_gets_a_gradient_of_its_own_shape(device: str, op: str) -> None:
    a = lucid.tensor([[1.0], [3.0], [2.0]], device=device, requires_grad=True)
    b = lucid.tensor([[2.0, 0.0, 4.0, 3.0]], device=device, requires_grad=True)
    getattr(lucid, op)(a, b).sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.shape == a.shape and b.grad.shape == b.shape
    # Each of the twelve outputs sends one unit to the operands (half each on a tie).
    assert math.isclose(sum(a.grad.reshape(-1).tolist()) + sum(b.grad.reshape(-1).tolist()), 12.0)


# ── the reductions are left alone ───────────────────────────────────────────────

_GRID = [[1.0, 4.0], [3.0, 2.0]]


def _reduced(result: object) -> _ValuesIndices[Tensor]:
    assert isinstance(result, _ValuesIndices)
    return result


def _tensor(result: object) -> Tensor:
    assert isinstance(result, Tensor)
    return result


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("via", ["fn", "method"])
def test_the_reduction_overloads_are_unchanged(device: str, op: str, via: str) -> None:
    x = lucid.tensor(_GRID, device=device)

    def call(*args: object, **kwargs: object) -> object:
        if via == "fn":
            return getattr(lucid, op)(x, *args, **kwargs)
        return getattr(x, op)(*args, **kwargs)

    top = 4.0 if op == "max" else 1.0
    rows = ([4.0, 3.0], [1, 0]) if op == "max" else ([1.0, 2.0], [0, 1])
    cols = ([3.0, 4.0], [1, 0]) if op == "max" else ([1.0, 2.0], [0, 1])

    assert _tensor(call()).item() == top
    assert _tensor(call(None)).item() == top
    assert _tensor(call((0, 1))).item() == top
    for result, (values, indices) in (
        (call(1), rows),
        (call(dim=1), rows),
        (call(-1), rows),
        (call(0, False), cols),
    ):
        pair = _reduced(result)
        assert pair.values.tolist() == values
        assert pair.indices.tolist() == indices
        assert pair[0] is pair.values and pair[1] is pair.indices
    kept = _reduced(call(0, keepdim=True))
    assert kept.values.shape == (1, 2) and kept.indices.shape == (1, 2)
    assert _tensor(call(dim=(0, 1), keepdim=True)).shape == (1, 1)
    assert _reduced(call(1, True)).values.shape == (2, 1)


@pytest.mark.parametrize("op", _OPS)
def test_a_zero_dim_tensor_is_an_operand_not_an_axis(device: str, op: str) -> None:
    x = lucid.tensor([1.0, 5.0], device=device)
    axis = lucid.tensor(1, device=device)
    # The reference reads it as ``other``: a 0-d integer tensor broadcast against ``x``.
    want = getattr(lucid, _ELEMENTWISE[op])(x, axis)
    assert getattr(lucid, op)(x, axis).tolist() == want.tolist()


# A 0-d integer tensor is a ``dim`` as well, as ``__index__`` reads it -- by keyword, or in the
# second place when ``keepdim`` follows (a lone tensor there is ``other``, above).  The call
# taking ``d`` is spelled once and run with a tensor and with the int it holds.
_AXIS_SPELLINGS: dict[str, Callable[[ModuleType, str, Tensor, object], _ValuesIndices[Tensor]]] = {
    "fn-dim": lambda fw, op, x, d: getattr(fw, op)(x, dim=d),
    "fn-dim-keepdim": lambda fw, op, x, d: getattr(fw, op)(x, dim=d, keepdim=True),
    "fn-positional-keepdim": lambda fw, op, x, d: getattr(fw, op)(x, d, True),
    "fn-positional-keepdim-keyword": lambda fw, op, x, d: getattr(fw, op)(x, d, keepdim=True),
    "method-dim": lambda fw, op, x, d: getattr(x, op)(dim=d),
    "method-dim-keepdim": lambda fw, op, x, d: getattr(x, op)(dim=d, keepdim=True),
    "method-positional-keepdim": lambda fw, op, x, d: getattr(x, op)(d, True),
    "method-positional-keepdim-keyword": lambda fw, op, x, d: getattr(x, op)(d, keepdim=True),
}
_AXIS_DTYPES = (lucid.int8, lucid.int32, lucid.int64)


def _grid(fw: ModuleType, device: str | None) -> Tensor:
    kwargs = {} if device is None else {"device": device}
    made: Tensor = fw.tensor(_GRID, **kwargs)
    return made


def _axis(fw: ModuleType, dtype: lucid.dtype, value: int, device: str | None) -> Tensor:
    kwargs = {} if device is None else {"device": device}
    made: Tensor = fw.tensor(value, dtype=getattr(fw, _name(dtype)), **kwargs)
    return made


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("dtype", _AXIS_DTYPES, ids=str)
@pytest.mark.parametrize("spelling", _AXIS_SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_a_zero_dim_integer_tensor_is_an_axis(
    device: str, op: str, spelling: str, dtype: lucid.dtype, axis: int
) -> None:
    call = _AXIS_SPELLINGS[spelling]
    x = _grid(lucid, device)
    got = call(lucid, op, x, _axis(lucid, dtype, axis, device))
    want = call(lucid, op, x, axis)
    assert got.values.shape == want.values.shape
    assert got.values.tolist() == want.values.tolist()
    assert got.indices.tolist() == want.indices.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("dtype", _AXIS_DTYPES, ids=str)
@pytest.mark.parametrize("spelling", _AXIS_SPELLINGS)
@pytest.mark.parametrize("op", _OPS)
def test_a_zero_dim_integer_axis_matches_the_reference(
    ref: ModuleType, device: str, op: str, spelling: str, dtype: lucid.dtype, axis: int
) -> None:
    call = _AXIS_SPELLINGS[spelling]
    got = call(lucid, op, _grid(lucid, device), _axis(lucid, dtype, axis, device))
    want = call(ref, op, _grid(ref, None), _axis(ref, dtype, axis, None))
    assert tuple(got.values.shape) == tuple(want.values.shape)
    assert got.values.tolist() == want.values.tolist()
    assert got.indices.tolist() == want.indices.tolist()


@pytest.mark.parametrize("op", _OPS)
def test_a_zero_dim_integer_axis_out_of_range_is_an_index_error(device: str, op: str) -> None:
    x = _grid(lucid, device)
    with pytest.raises(IndexError):
        getattr(lucid, op)(x, dim=_axis(lucid, lucid.int64, 5, device))


# ── invalid mixes ───────────────────────────────────────────────────────────────

_Call = Callable[[ModuleType, str, Tensor, Tensor], object]

#: Calls both frameworks refuse: not one of the three overloads.
_INVALID: dict[str, _Call] = {
    "other-with-keepdim": lambda fw, op, a, b: getattr(fw, op)(a, b, keepdim=True),
    "other-with-dim": lambda fw, op, a, b: getattr(fw, op)(a, b, dim=0),
    "other-then-dim": lambda fw, op, a, b: getattr(fw, op)(a, b, 0),
    "dim-then-other": lambda fw, op, a, b: getattr(fw, op)(a, 0, b),
    "two-tensors-after-input": lambda fw, op, a, b: getattr(fw, op)(a, b, b),
    "tensor-named-dim": lambda fw, op, a, b: getattr(fw, op)(a, dim=b),
    "keyword-other-with-dim": lambda fw, op, a, b: getattr(fw, op)(a, other=b, dim=0),
    "dim-and-keyword-other": lambda fw, op, a, b: getattr(fw, op)(a, 0, other=b),
    "number-as-other": lambda fw, op, a, b: getattr(fw, op)(a, other=1.0),
    "none-as-other": lambda fw, op, a, b: getattr(fw, op)(a, other=None),
    "float-as-dim": lambda fw, op, a, b: getattr(fw, op)(a, 1.0),
    "too-many-positionals": lambda fw, op, a, b: getattr(fw, op)(a, 0, True, 1),
    "unknown-keyword": lambda fw, op, a, b: getattr(fw, op)(a, nonsense=1),
    "number-as-input": lambda fw, op, a, b: getattr(fw, op)(5, b),
    "method-other-with-keepdim": lambda fw, op, a, b: getattr(a, op)(b, keepdim=True),
    "method-other-with-dim": lambda fw, op, a, b: getattr(a, op)(b, 0),
    "method-tensor-named-dim": lambda fw, op, a, b: getattr(a, op)(dim=b),
    "method-keyword-other-with-dim": lambda fw, op, a, b: getattr(a, op)(other=b, dim=0),
    # Only a 0-d integer tensor is an axis: not a float, a bool or a vector -- by keyword or
    # in the second place with ``keepdim`` after it -- and it is no axis beside ``dim=`` /
    # ``other=``, nor a ``keepdim``.
    "float-zero-dim-as-dim": lambda fw, op, a, b: getattr(fw, op)(a, dim=fw.tensor(0.0)),
    "bool-zero-dim-as-dim": lambda fw, op, a, b: getattr(fw, op)(a, dim=fw.tensor(False)),
    "integer-vector-as-dim": lambda fw, op, a, b: getattr(fw, op)(a, dim=fw.tensor([0])),
    "float-zero-dim-with-keepdim": lambda fw, op, a, b: getattr(fw, op)(a, fw.tensor(0.0), True),
    "bool-zero-dim-with-keepdim": lambda fw, op, a, b: getattr(fw, op)(a, fw.tensor(False), True),
    "integer-vector-with-keepdim": lambda fw, op, a, b: getattr(fw, op)(
        a, fw.tensor([0]), keepdim=True
    ),
    "integer-zero-dim-with-dim": lambda fw, op, a, b: getattr(fw, op)(a, fw.tensor(0), dim=0),
    "integer-zero-dim-with-other": lambda fw, op, a, b: getattr(fw, op)(a, fw.tensor(0), other=b),
    "integer-zero-dim-as-dim-and-other": lambda fw, op, a, b: getattr(fw, op)(
        a, dim=fw.tensor(0), other=b
    ),
    "integer-zero-dim-as-keepdim": lambda fw, op, a, b: getattr(fw, op)(a, 0, fw.tensor(0)),
    "integer-zero-dim-twice": lambda fw, op, a, b: getattr(fw, op)(a, fw.tensor(0), fw.tensor(0)),
    "integer-zero-dim-keepdim-tensor": lambda fw, op, a, b: getattr(fw, op)(
        a, dim=fw.tensor(0), keepdim=fw.tensor(True)
    ),
    "method-float-zero-dim-as-dim": lambda fw, op, a, b: getattr(a, op)(dim=fw.tensor(0.0)),
    "method-bool-zero-dim-as-dim": lambda fw, op, a, b: getattr(a, op)(dim=fw.tensor(False)),
    "method-float-zero-dim-with-keepdim": lambda fw, op, a, b: getattr(a, op)(fw.tensor(0.0), True),
    "method-integer-zero-dim-with-dim": lambda fw, op, a, b: getattr(a, op)(fw.tensor(0), dim=0),
}


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("case", _INVALID)
def test_an_invalid_mix_is_a_type_error(device: str, op: str, case: str) -> None:
    a, b = _pair(device)
    with pytest.raises(TypeError):
        _INVALID[case](lucid, op, a, b)


@pytest.mark.parity
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("case", _INVALID)
def test_an_invalid_mix_is_refused_as_the_reference_refuses_it(
    ref: ModuleType, device: str, op: str, case: str
) -> None:
    with pytest.raises(Exception) as want:
        _INVALID[case](ref, op, ref.tensor(_A), ref.tensor(_B))
    a, b = _pair(device)
    with pytest.raises(Exception) as got:
        _INVALID[case](lucid, op, a, b)
    assert issubclass(got.type, want.type)


@pytest.mark.parametrize("op", _OPS)
def test_a_refused_call_leaves_the_operands_alone(device: str, op: str) -> None:
    a, b = _pair(device)
    with pytest.raises(TypeError):
        getattr(lucid, op)(a, b, keepdim=True)
    assert a.tolist() == _A and b.tolist() == _B


@pytest.mark.parametrize("op", _OPS)
def test_an_out_of_range_dim_is_still_an_index_error(device: str, op: str) -> None:
    a, _ = _pair(device)
    with pytest.raises(IndexError):
        getattr(lucid, op)(a, 3)


# ── the stubs advertise the form ────────────────────────────────────────────────


def _second_parameters(path: Path, op: str, cls: str | None) -> set[str]:
    """The name of the parameter after the operand (``input`` or ``self``) in each overload of
    ``op`` in a stub."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return {
        fn.args.args[1].arg
        for fn in body
        if isinstance(fn, ast.FunctionDef) and fn.name == op and len(fn.args.args) > 1
    }


@pytest.mark.parametrize("op", _OPS)
def test_the_stubs_declare_the_other_overload(op: str) -> None:
    root = Path(lucid.__file__).parent
    assert _second_parameters(root / "__init__.pyi", op, None) == {"dim", "other"}
    assert _second_parameters(root / "_tensor" / "tensor.pyi", op, "Tensor") == {"dim", "other"}
