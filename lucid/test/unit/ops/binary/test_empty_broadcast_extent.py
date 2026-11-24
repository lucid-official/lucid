"""A broadcast that comes out empty leaves a result whose storage matches its shape (LCD-1062).

On Metal the fused binary ops (``maximum``/``minimum`` of a float pair, integer ``pow``, the
shifts) are compiled with ``shapeless=True``.  MLX's shapeless trace broadcasts a ``0`` extent
against a ``1`` as ``1``, so ``maximum`` of a ``(0,)`` and a ``(1,)`` tensor came back labelled
``(0,)`` over a one-element buffer: ``storage_nbytes()`` was 4 where ``nbytes()`` was 0, a
shift's in-place form failed ``copy_from`` with ``nbytes mismatch (GPU)``, and the other ops
carried the wrong buffer without a word (``maximum_`` left its receiver on it).  A call with an
empty operand now runs the steps eagerly (``backend/gpu/ShapelessFused.h``), and ``helpers::fresh``
— the function every forward builds its result through, kernel families and hand-built ops alike
(LCD-1072) — refuses a GPU result whose byte count disagrees with its shape
(``core/ResultExtent.h``).  ``test_forward_result_extent.py`` holds the same count over the ops that
build by hand.

What is held here is the class, not the ops it was found in: every binary op the engine
dispatches, out of place and in place, over the broadcasts that come out empty, on whichever
streams are available.  The CPU stream runs the same cases as the control.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

import lucid
import lucid.nn.functional as F

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

#: (receiver shape, other shape, broadcast result shape).  A 0 against a 1 on either side, in one,
#: two and three dimensions, a scalar partner, and two equal empty shapes.
EMPTY_BROADCASTS = [
    ((0,), (1,), (0,)),
    ((1,), (0,), (0,)),
    ((0, 3), (1, 3), (0, 3)),
    ((2, 0), (2, 1), (2, 0)),
    ((3, 1, 0), (1, 4, 1), (3, 4, 0)),
    ((3, 4, 0), (3, 1, 1), (3, 4, 0)),
    ((0,), (), (0,)),
    ((0,), (0,), (0,)),
]

#: The cases an in-place op can take: it writes the result into a receiver of its own shape, so
#: the receiver has to be the side that broadcasts to the result.  Generated, not skipped.
IN_PLACE_BROADCASTS = [case for case in EMPTY_BROADCASTS if case[0] == case[2]]

_FLOATS = (lucid.float32, lucid.float16, lucid.bfloat16)
_INTS = (lucid.int32, lucid.int64, lucid.int16, lucid.int8)

_FLOAT_OPS = (
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "atan2",
    "remainder",
    "fmod",
    "floor_divide",
    "hypot",
    "copysign",
    "logaddexp",
    "xlogy",
    "heaviside",
)
_INT_OPS = (
    "add",
    "sub",
    "mul",
    "pow",
    "maximum",
    "minimum",
    "remainder",
    "fmod",
    "floor_divide",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "gcd",
    "lcm",
)


def _op_dtype_params() -> list[ParameterSet]:
    return [
        pytest.param(op, dt, id=f"{op}-{dt}")
        for ops, dtypes in ((_FLOAT_OPS, _FLOATS), (_INT_OPS, _INTS))
        for dt in dtypes
        for op in ops
    ]


def _in_place_params() -> list[ParameterSet]:
    return [p for p in _op_dtype_params() if hasattr(lucid.Tensor, f"{p.values[0]}_")]


def _consistent(t: lucid.Tensor) -> None:
    """The storage holds exactly the elements the shape names."""
    assert t._impl.storage_nbytes() == t._impl.nbytes(), (
        f"shape {tuple(t.shape)} {t.dtype}: storage {t._impl.storage_nbytes()} B, "
        f"shape needs {t._impl.nbytes()} B"
    )


def _ones(shape: tuple[int, ...], dtype: lucid.dtype, device: str) -> lucid.Tensor:
    return lucid.ones(shape, dtype=dtype, device=device)


@pytest.mark.parametrize(("a_shape", "b_shape", "want"), EMPTY_BROADCASTS, ids=str)
@pytest.mark.parametrize(("op", "dtype"), _op_dtype_params())
def test_an_empty_broadcast_result_has_matching_storage(
    device: str,
    op: str,
    dtype: lucid.dtype,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    want: tuple[int, ...],
) -> None:
    out = getattr(lucid, op)(_ones(a_shape, dtype, device), _ones(b_shape, dtype, device))
    assert tuple(out.shape) == want
    _consistent(out)


@pytest.mark.parametrize(("a_shape", "b_shape", "want"), IN_PLACE_BROADCASTS, ids=str)
@pytest.mark.parametrize(("op", "dtype"), _in_place_params())
def test_an_in_place_op_over_an_empty_broadcast_completes(
    device: str,
    op: str,
    dtype: lucid.dtype,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    want: tuple[int, ...],
) -> None:
    recv = _ones(a_shape, dtype, device)
    getattr(recv, f"{op}_")(_ones(b_shape, dtype, device))
    assert tuple(recv.shape) == want
    _consistent(recv)


# ── the paths the empty case now bypasses still compute ───────────────────────────────────────

#: (op, dtypes, a, b, expected): small enough to be exact in every dtype listed.
_CONTROL_TABLE = (
    (
        "maximum",
        _FLOATS,
        [[1.0], [-2.0], [3.0]],
        [[0.0, 2.0, 5.0, -1.0]],
        [[1.0, 2.0, 5.0, 1.0], [0.0, 2.0, 5.0, -1.0], [3.0, 3.0, 5.0, 3.0]],
    ),
    (
        "minimum",
        _FLOATS,
        [[1.0], [-2.0], [3.0]],
        [[0.0, 2.0, 5.0, -1.0]],
        [[0.0, 1.0, 1.0, -1.0], [-2.0, -2.0, -2.0, -2.0], [0.0, 2.0, 3.0, -1.0]],
    ),
    (
        "pow",
        _INTS,
        [[2], [3], [-1]],
        [[0, 1, 2, 3]],
        [[1, 2, 4, 8], [1, 3, 9, 27], [1, -1, 1, -1]],
    ),
    (
        "bitwise_left_shift",
        _INTS,
        [[1], [2]],
        [[0, 1, 2, 3]],
        [[1, 2, 4, 8], [2, 4, 8, 16]],
    ),
    (
        "bitwise_right_shift",
        _INTS,
        [[8], [16]],
        [[0, 1, 2, 3]],
        [[8, 4, 2, 1], [16, 8, 4, 2]],
    ),
)
_CONTROL = [
    (op, dt, a, b, expected) for op, dtypes, a, b, expected in _CONTROL_TABLE for dt in dtypes
]


@pytest.mark.parametrize(
    ("op", "dtype", "a", "b", "expected"),
    _CONTROL,
    ids=[f"{c[0]}-{c[1]}" for c in _CONTROL],
)
def test_a_non_empty_broadcast_still_computes(
    device: str,
    op: str,
    dtype: lucid.dtype,
    a: list[list[float]],
    b: list[list[float]],
    expected: list[list[float]],
) -> None:
    # An empty call first: the empty case must not leave the fused path unable to trace.
    getattr(lucid, op)(_ones((0,), dtype, device), _ones((1,), dtype, device))
    out = getattr(lucid, op)(
        lucid.tensor(a, dtype=dtype, device=device), lucid.tensor(b, dtype=dtype, device=device)
    )
    _consistent(out)
    assert out.tolist() == expected


# ── the other fused and clamped paths on empty operands ───────────────────────────────────────

#: (x shape, bound shape, result shape): the bound broadcasts against an empty x.
_CLAMP_BROADCASTS = [
    ((0, 3), (1, 3), (0, 3)),
    ((2, 0), (2, 1), (2, 0)),
    ((0,), (1,), (0,)),
    ((1,), (0,), (0,)),
]


@pytest.mark.parametrize(("x_shape", "bound_shape", "want"), _CLAMP_BROADCASTS, ids=str)
@pytest.mark.parametrize("dtype", _FLOATS, ids=str)
def test_clamp_with_tensor_bounds_over_an_empty_broadcast(
    device: str,
    dtype: lucid.dtype,
    x_shape: tuple[int, ...],
    bound_shape: tuple[int, ...],
    want: tuple[int, ...],
) -> None:
    x = _ones(x_shape, dtype, device)
    lo = lucid.zeros(bound_shape, dtype=dtype, device=device)
    hi = _ones(bound_shape, dtype=dtype, device=device)
    out = lucid.clamp(x, min=lo, max=hi)
    assert tuple(out.shape) == want
    _consistent(out)
    if x_shape == want:
        x.clamp_(min=lo, max=hi)
        assert tuple(x.shape) == want
        _consistent(x)


@pytest.mark.parametrize("dtype", _FLOATS, ids=str)
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4)], ids=str)
@pytest.mark.parametrize("name", ["silu", "gelu", "softplus", "selu", "mish"])
def test_a_fused_activation_and_its_gradient_on_an_empty_input(
    device: str, name: str, shape: tuple[int, ...], dtype: lucid.dtype
) -> None:
    act: Callable[[lucid.Tensor], lucid.Tensor] = getattr(F, name)
    x = _ones(shape, dtype, device).requires_grad_(True)
    y = act(x)
    assert tuple(y.shape) == shape
    _consistent(y)
    y.sum().backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == shape
    _consistent(x.grad)


@pytest.mark.parametrize("dtype", (lucid.float16, lucid.bfloat16), ids=str)
def test_half_norms_over_an_empty_batch(device: str, dtype: lucid.dtype) -> None:
    """The half-accumulation fusions (``HalfAccumulation.h``) take an empty operand eagerly."""
    x = _ones((0, 3, 4, 4), dtype, device).requires_grad_(True)
    weight = _ones((3,), dtype, device)
    bias = lucid.zeros((3,), dtype=dtype, device=device)
    y = F.batch_norm(x, None, None, weight, bias, training=True)
    assert tuple(y.shape) == (0, 3, 4, 4)
    _consistent(y)
    y.sum().backward()
    assert x.grad is not None
    _consistent(x.grad)
    _consistent(F.layer_norm(_ones((0, 8), dtype, device), (8,)))


@pytest.mark.parametrize("dtype", (lucid.float32, lucid.int32), ids=str)
def test_an_index_op_with_an_empty_index(device: str, dtype: lucid.dtype) -> None:
    """The index checks (``AxisIndex.h``) take an empty index eagerly."""
    src = lucid.arange(12, dtype=dtype, device=device).reshape(3, 4)
    none = lucid.zeros((0,), dtype=lucid.int64, device=device)
    for out in (
        lucid.index_select(src, 0, none),
        lucid.gather(src, 0, lucid.zeros((0, 4), dtype=lucid.int64, device=device)),
        lucid.take(src, none),
    ):
        _consistent(out)
