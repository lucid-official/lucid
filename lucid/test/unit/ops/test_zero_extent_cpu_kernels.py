"""CPU kernels on a zero-length axis with other axes around it (LCD-1047).

A CPU tensor with no elements has a null buffer.  A kernel that walks ``outer`` x ``len`` x
``inner`` and offsets its input by the *inner* index before looking at ``len`` forms
``nullptr + index`` when ``len`` is 0 -- undefined behaviour, whatever the loop then does with
the pointer.  The answer is right every time (``p_norm`` over ``(2, 0, 3)`` was 0 in every
nightly that halted on it), so a value assertion alone cannot catch it: the nightly sanitizer job
runs this directory under UBSan, and what is asserted here is what each op must give.

``test_an_empty_reduction_is_zero`` in ``unit/linalg/test_p_norm_cpu_kernel.py`` is the engine-level
case.  This is the class: every reduction, normalisation, scan, sort, shuffle and contraction the
CPU stream has, on shapes that put the empty axis in the middle (outer and inner both non-trivial,
the inner one wide enough to take a kernel's tiled path), first, and last, forward and backward.

Reductions that refuse an empty axis (``max``, ``argmax``, ``median``) must keep refusing: a
kernel that never ran cannot read past nothing.
"""

import math
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.linalg as LA
import lucid.nn.functional as F

Op = Callable[[lucid.Tensor, int], lucid.Tensor]

#: ``(shape, the empty axis)``: the empty axis in the middle with a narrow and a wide inner extent,
#: leading, trailing, and a rank-4 pair.  The wide inner (16) is where a kernel tiles its columns.
EMPTY_AXIS = [
    ((2, 0, 3), 1),
    ((3, 0, 16), 1),
    ((1, 0, 9), 1),
    ((0, 3), 0),
    ((0, 16), 0),
    ((3, 4, 0), 2),
    ((2, 0, 3, 4), 1),
    ((2, 3, 0, 4), 2),
]


def _poison() -> None:
    """Leave 7.0 in every small block the allocator will hand out next.

    A kernel that skips writing a result it owes then returns 7.0, not the 0.0 a fresh page
    would happen to give.
    """
    for numel in range(1, 65):
        lucid.full((numel,), 7.0)


def _same_values(got: lucid.Tensor, want: float | bool) -> bool:
    values = got.reshape(-1).tolist()
    if isinstance(want, float) and math.isnan(want):
        return all(math.isnan(v) for v in values)
    return all(v == want for v in values)


def _reduced_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


def _vector_norm(p: float) -> Op:
    return lambda x, d: LA.vector_norm(x, p, dim=d)


#: Reductions over the empty axis and what the empty sum of each is.
REDUCTIONS: dict[str, tuple[Op, float | bool]] = {
    "sum": (lambda x, d: lucid.sum(x, dim=d), 0.0),
    "prod": (lambda x, d: lucid.prod(x, dim=d), 1.0),
    "logsumexp": (lambda x, d: lucid.logsumexp(x, dim=d), -math.inf),
    "any": (lambda x, d: lucid.any(x, dim=d), False),
    "all": (lambda x, d: lucid.all(x, dim=d), True),
    "count_nonzero": (lambda x, d: lucid.count_nonzero(x, dim=d), 0),
    **{
        f"vector_norm[{p}]": (_vector_norm(p), 0.0)
        for p in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.7, 3.0, 4.0, 6.0, 7.5)
    },
}

#: Means and variances of nothing are NaN.
NAN_REDUCTIONS: dict[str, Op] = {
    "mean": lambda x, d: lucid.mean(x, dim=d),
    "var": lambda x, d: lucid.var(x, dim=d),
    "std": lambda x, d: lucid.std(x, dim=d),
}

#: Ops that keep the shape (an empty one) or build one from the empty axis.
SHAPE_KEEPING: dict[str, Op] = {
    "normalize[1]": lambda x, d: F.normalize(x, p=1.0, dim=d),
    "normalize[2]": lambda x, d: F.normalize(x, p=2.0, dim=d),
    "normalize[3]": lambda x, d: F.normalize(x, p=3.0, dim=d),
    "normalize[0.5]": lambda x, d: F.normalize(x, p=0.5, dim=d),
    "softmax": lambda x, d: F.softmax(x, dim=d),
    "log_softmax": lambda x, d: F.log_softmax(x, dim=d),
    "cumsum": lambda x, d: lucid.cumsum(x, dim=d),
    "cumprod": lambda x, d: lucid.cumprod(x, dim=d),
    "cummax": lambda x, d: lucid.cummax(x, dim=d)[0],
    "cummin": lambda x, d: lucid.cummin(x, dim=d)[0],
    "sort": lambda x, d: lucid.sort(x, dim=d)[0],
    "argsort": lambda x, d: lucid.argsort(x, dim=d),
    "topk[0]": lambda x, d: lucid.topk(x, 0, dim=d)[0],
    "flip": lambda x, d: lucid.flip(x, [d]),
    "roll": lambda x, d: lucid.roll(x, 1, d),
    "diff": lambda x, d: lucid.diff(x, dim=d),
    "tril": lambda x, d: lucid.tril(x),
    "clone": lambda x, d: x.clone(),
    "cat": lambda x, d: lucid.cat([x, x], dim=d),
    "narrow": lambda x, d: x.narrow(d, 0, 0),
}

#: Ops that refuse to reduce nothing.
REFUSING: dict[str, Op] = {
    "amax": lambda x, d: lucid.amax(x, dim=d),
    "argmax": lambda x, d: lucid.argmax(x, dim=d),
    "median": lambda x, d: lucid.median(x, dim=d),
    "vector_norm[inf]": _vector_norm(math.inf),
}


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize("name", list(REDUCTIONS))
def test_a_reduction_over_nothing_is_its_identity(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    op, identity = REDUCTIONS[name]
    x = lucid.zeros(*shape)
    _poison()
    got = op(x, axis)
    assert tuple(got.shape) == _reduced_shape(shape, axis)
    assert _same_values(got, identity), got.reshape(-1).tolist()


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize("name", list(NAN_REDUCTIONS))
def test_a_mean_or_variance_of_nothing_is_nan(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    x = lucid.zeros(*shape)
    _poison()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # var and std warn that no degree of freedom is left
        got = NAN_REDUCTIONS[name](x, axis)
    assert tuple(got.shape) == _reduced_shape(shape, axis)
    assert _same_values(got, math.nan)


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize("name", list(SHAPE_KEEPING))
def test_an_op_over_a_zero_length_axis_answers_empty(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    x = lucid.zeros(*shape)
    _poison()
    got = SHAPE_KEEPING[name](x, axis)
    assert tuple(got.shape) == shape
    assert got.numel() == 0


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize("name", list(REFUSING))
def test_a_reduction_that_has_no_identity_refuses_nothing_to_reduce(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    with pytest.raises(IndexError):
        REFUSING[name](lucid.zeros(*shape), axis)


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize(
    "name", ["sum", "prod", "logsumexp", "vector_norm[1.5]", "vector_norm[3.0]", "vector_norm[0.5]"]
)
def test_a_reduction_over_nothing_has_an_empty_gradient(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    op, _ = REDUCTIONS[name]
    x = lucid.zeros(*shape, requires_grad=True)
    _poison()
    op(x, axis).sum().backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == shape


@pytest.mark.parametrize(("shape", "axis"), EMPTY_AXIS)
@pytest.mark.parametrize(
    "name",
    [
        "normalize[1]",
        "normalize[2]",
        "normalize[3]",
        "normalize[0.5]",
        "softmax",
        "log_softmax",
        "cumsum",
        "cumprod",
        "flip",
        "roll",
    ],
)
def test_an_op_over_a_zero_length_axis_has_an_empty_gradient(
    device_cpu_only: str, name: str, shape: tuple[int, ...], axis: int
) -> None:
    x = lucid.zeros(*shape, requires_grad=True)
    _poison()
    SHAPE_KEEPING[name](x, axis).sum().backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == shape


@pytest.mark.parametrize(
    ("shape", "axes", "kept"),
    [
        # reduced axes that are one contiguous run, and ones that are not (gathered first)
        ((2, 0, 3, 4), [1, 3], (2, 3)),
        ((2, 0, 3, 4), [0, 1, 3], (3,)),
        ((2, 3, 0, 4), [0, 2], (3, 4)),
        ((2, 3, 0, 4), [0, 3], (3, 0)),
        ((2, 0, 3, 4), [0, 2], (0, 4)),
        ((0, 3, 2, 4), [1, 3], (0, 2)),
    ],
)
@pytest.mark.parametrize("p", [0.5, 1.5, 3.0, 7.5])
def test_a_norm_over_axes_that_are_not_a_run_and_include_an_empty_one(
    device_cpu_only: str, shape: tuple[int, ...], axes: list[int], kept: tuple[int, ...], p: float
) -> None:
    """The kept axes' extent is the number of slices, not ``numel / reduced length``: the quotient
    is 0 once a reduced axis is empty, and the slices were then never written."""
    x = lucid.zeros(*shape)
    _poison()
    got = LA.vector_norm(x, p, dim=axes)
    assert tuple(got.shape) == kept
    assert _same_values(got, 0.0)


@pytest.mark.parametrize(("batch", "rows", "cols"), [(2, 3, 3), (3, 16, 16), (1, 1, 9), (2, 5, 2)])
def test_a_contraction_over_a_zero_length_axis_is_zero(
    device_cpu_only: str, batch: int, rows: int, cols: int
) -> None:
    left = lucid.zeros(batch, rows, 0)
    right = lucid.zeros(batch, 0, cols)
    _poison()
    got = lucid.matmul(left, right)
    assert tuple(got.shape) == (batch, rows, cols)
    assert _same_values(got, 0.0)
    flat = lucid.matmul(lucid.zeros(rows, 0), lucid.zeros(0, cols))
    assert tuple(flat.shape) == (rows, cols)
    assert _same_values(flat, 0.0)


@pytest.mark.parametrize("batch_first", [False, True])
def test_an_lstm_over_an_empty_batch_answers_empty(device_cpu_only: str, batch_first: bool) -> None:
    # The CPU kernel's gate buffer is null for a zero batch; the output-gate slice of every
    # step (forward and backward) once offset it, which UBSan reports in the nightly job.
    _poison()
    lstm = lucid.nn.LSTM(3, 4, num_layers=2, batch_first=batch_first)
    x = lucid.zeros(0, 5, 3) if batch_first else lucid.zeros(5, 0, 3)
    x.requires_grad_(True)
    out, (h, c) = lstm(x)
    assert tuple(out.shape) == ((0, 5, 4) if batch_first else (5, 0, 4))
    assert tuple(h.shape) == (2, 0, 4) and tuple(c.shape) == (2, 0, 4)
    out.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == tuple(x.shape)
    weight = lstm.weight_hh_l0.grad
    assert weight is not None and all(v == 0.0 for v in weight.reshape(-1).tolist())


@pytest.mark.parametrize("dtype", [lucid.complex64])
def test_a_conjugate_of_nothing_is_empty(device_cpu_only: str, dtype: lucid.dtype) -> None:
    # ``vzconj_c64`` negated the imag halves through ``out + 1`` on a null buffer.
    got = lucid.conj_physical(lucid.zeros(0, 3, dtype=dtype))
    assert tuple(got.shape) == (0, 3) and got.dtype == dtype
