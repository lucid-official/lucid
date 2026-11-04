"""Parity guard for :func:`lucid.nn._shadow.shadow_alloc` against eager execution.

Verifies that under ``shadow_alloc()``, phantom tensors report the exact same
``.dtype`` and ``.shape`` as eager tensors across:
- binary operators (``+``, ``-``, ``*``, ``/``, ``//``, ``**``, ``@``, reflected
  scalar forms ``2 - a``, ``0.5 * a``, comparisons, and in-place forms)
- free functions (``lucid.add``, ``lucid.sub``, ``lucid.mul``, ``lucid.div``,
  ``lucid.floor_divide``, ``lucid.pow``, ``lucid.matmul``, unary math, and
  reductions)
- ``Tensor`` methods (``.add``, ``.sub``, ``.mul``, ``.div``, ``.floor_divide``,
  ``.pow``, ``.matmul``, layout/reduction methods)
across same-dtype pairs, cross-dtype tensor pairs, ``int64`` + float scalar
promotion, broadcast shapes, and reversed scalar operands (LCD-346 / #419).

Every comparison runs on the CPU and on Metal, and adds ``.device``: the Metal
path used to read its own result's shape off an MLX array the phantom does not
have (LCD-850).  ``test_shadow_alloc_surface.py`` sweeps the rest of the op
surface the same way.  What Metal refuses for real it refuses as a phantom too:
a phantom float64 Metal tensor would blind every shadow-built check to it.
"""

from collections.abc import Callable
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._dtype import dtype as DType
from lucid._tensor.tensor import Tensor
from lucid.nn._shadow import is_shadow, shadow_alloc
from lucid.test._fixtures.devices import (
    METAL_UNSUPPORTED_DTYPES,
    device_supports,
    devices_supporting,
    metal_available,
)


def _assert_same_answer(phantom: Tensor, eager: Tensor) -> None:
    assert is_shadow(phantom)
    assert phantom.shape == eager.shape
    assert phantom.dtype == eager.dtype
    assert phantom.device == eager.device


# ── Binary operators, free functions, and Tensor methods (tensor × tensor) ────

_BINARY_SPECS: list[tuple[str, Callable[[Tensor, Tensor], Tensor]]] = [
    ("op_add", lambda a, b: a + b),
    ("op_sub", lambda a, b: a - b),
    ("op_mul", lambda a, b: a * b),
    ("op_truediv", lambda a, b: a / b),
    ("op_floordiv", lambda a, b: a // b),
    ("op_pow", lambda a, b: a**b),
    ("fn_add", lucid.add),
    ("fn_sub", lucid.sub),
    ("fn_mul", lucid.mul),
    ("fn_div", lucid.div),
    ("fn_floor_divide", lucid.floor_divide),
    ("fn_pow", lucid.pow),
    ("fn_maximum", lucid.maximum),
    ("fn_minimum", lucid.minimum),
    ("fn_atan2", lucid.atan2),
    ("fn_hypot", lucid.hypot),
    ("fn_eq", lucid.eq),
    ("fn_less", lucid.less),
    ("method_add", lambda a, b: a.add(b)),
    ("method_sub", lambda a, b: a.sub(b)),
    ("method_mul", lambda a, b: a.mul(b)),
    ("method_div", lambda a, b: a.div(b)),
    ("method_floor_divide", lambda a, b: a.floor_divide(b)),
    ("method_pow", lambda a, b: a.pow(b)),
    ("method_maximum", lambda a, b: a.maximum(b)),
    ("method_minimum", lambda a, b: a.minimum(b)),
    ("method_eq", lambda a, b: a.eq(b)),
]

_TENSOR_DTYPE_PAIRS: list[tuple[DType, DType]] = [
    (lucid.float32, lucid.float32),
    (lucid.int64, lucid.int64),
    (lucid.int64, lucid.float32),
    (lucid.float32, lucid.int64),
    (lucid.int32, lucid.float16),
    (lucid.float16, lucid.bfloat16),
    (lucid.int8, lucid.int32),
    (lucid.float32, lucid.float64),
]

# Metal holds no float64, and eagerly refuses it, so the pair runs on the CPU.
_DEVICE_DTYPE_PAIRS = [
    pytest.param(device, a, b, id=f"{device}-{a}-{b}")
    for a, b in _TENSOR_DTYPE_PAIRS
    for device in devices_supporting(a)
    if device_supports(device, b)
]

_BROADCAST_SHAPE_PAIRS: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3, 4), (3, 4)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (4,)),
    ((), (3, 4)),
    ((3, 4), ()),
]


_FLOAT_DTYPES = (lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64)


@pytest.mark.parametrize("name,op", _BINARY_SPECS, ids=[s[0] for s in _BINARY_SPECS])
@pytest.mark.parametrize("on,dt_a,dt_b", _DEVICE_DTYPE_PAIRS)
@pytest.mark.parametrize(
    "shape_a,shape_b",
    _BROADCAST_SHAPE_PAIRS,
    ids=[f"{sa}x{sb}" for sa, sb in _BROADCAST_SHAPE_PAIRS],
)
def test_binary_tensor_pair_shape_and_dtype_parity(
    name: str,
    op: Callable[[Tensor, Tensor], Tensor],
    on: str,
    dt_a: DType,
    dt_b: DType,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> None:
    if name == "fn_hypot" and dt_a not in _FLOAT_DTYPES and dt_b not in _FLOAT_DTYPES:
        pytest.skip("hypot requires at least one floating-point operand in eager mode")
    eager_a = lucid.ones(shape_a, dtype=dt_a, device=on)
    eager_b = lucid.ones(shape_b, dtype=dt_b, device=on)
    eager_out = op(eager_a, eager_b)

    with shadow_alloc():
        ph_a = lucid.ones(shape_a, dtype=dt_a, device=on)
        ph_b = lucid.ones(shape_b, dtype=dt_b, device=on)
        ph_out = op(ph_a, ph_b)

    _assert_same_answer(ph_out, eager_out)


# ── Binary ops with Python scalars (forward and reflected) ───────────────────

_SCALAR_BINARY_SPECS: list[tuple[str, Callable[[Tensor, int | float], Tensor]]] = [
    ("op_add_r", lambda a, s: a + s),
    ("op_add_l", lambda a, s: s + a),
    ("op_sub_r", lambda a, s: a - s),
    ("op_sub_l", lambda a, s: s - a),
    ("op_mul_r", lambda a, s: a * s),
    ("op_mul_l", lambda a, s: s * a),
    ("op_div_r", lambda a, s: a / s),
    ("op_div_l", lambda a, s: s / a),
    ("op_floordiv_r", lambda a, s: a // s),
    ("op_floordiv_l", lambda a, s: s // a),
    ("op_pow_r", lambda a, s: a**s),
    ("op_pow_l", lambda a, s: s**a),
    ("fn_add_r", lambda a, s: lucid.add(a, s)),
    ("fn_add_l", lambda a, s: lucid.add(s, a)),
    ("fn_sub_r", lambda a, s: lucid.sub(a, s)),
    ("fn_sub_l", lambda a, s: lucid.sub(s, a)),
    ("fn_mul_r", lambda a, s: lucid.mul(a, s)),
    ("fn_mul_l", lambda a, s: lucid.mul(s, a)),
    ("fn_div_r", lambda a, s: lucid.div(a, s)),
    ("fn_div_l", lambda a, s: lucid.div(s, a)),
    ("fn_floor_divide_r", lambda a, s: lucid.floor_divide(a, s)),
    ("fn_floor_divide_l", lambda a, s: lucid.floor_divide(s, a)),
    ("fn_pow_r", lambda a, s: lucid.pow(a, s)),
    ("fn_pow_l", lambda a, s: lucid.pow(s, a)),
    ("method_add", lambda a, s: a.add(s)),
    ("method_sub", lambda a, s: a.sub(s)),
    ("method_mul", lambda a, s: a.mul(s)),
    ("method_div", lambda a, s: a.div(s)),
    ("method_floor_divide", lambda a, s: a.floor_divide(s)),
    ("method_pow", lambda a, s: a.pow(s)),
]


@pytest.mark.parametrize("name,op", _SCALAR_BINARY_SPECS, ids=[s[0] for s in _SCALAR_BINARY_SPECS])
@pytest.mark.parametrize(
    "dt", [lucid.int8, lucid.int32, lucid.int64, lucid.float16, lucid.bfloat16, lucid.float32]
)
@pytest.mark.parametrize("scalar", [2, 0.5], ids=["int_scalar", "float_scalar"])
@pytest.mark.parametrize("shape", [(3, 4), (4,), ()], ids=["2d", "1d", "0d"])
def test_binary_scalar_shape_and_dtype_parity(
    name: str,
    op: Callable[[Tensor, int | float], Tensor],
    dt: DType,
    scalar: int | float,
    shape: tuple[int, ...],
    device: str,
) -> None:
    del name
    eager_a = lucid.ones(shape, dtype=dt, device=device)
    eager_out = op(eager_a, scalar)

    with shadow_alloc():
        ph_a = lucid.ones(shape, dtype=dt, device=device)
        ph_out = op(ph_a, scalar)

    _assert_same_answer(ph_out, eager_out)


# ── Matrix multiplication (@, lucid.matmul, Tensor.matmul, mm, bmm) ──────────

_MATMUL_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((3,), (3, 4)),
    ((2, 3), (3, 4)),
    ((5, 1, 2, 3), (1, 4, 3, 6)),
    ((5, 2, 3), (3,)),
    ((3,), (5, 3, 4)),
]

_MATMUL_CALLERS: list[tuple[str, Callable[[Tensor, Tensor], Tensor]]] = [
    ("op_matmul", lambda a, b: a @ b),
    ("op_rmatmul", lambda a, b: b.__rmatmul__(a)),
    ("fn_matmul", lucid.matmul),
    ("method_matmul", lambda a, b: a.matmul(b)),
]


@pytest.mark.parametrize("caller_name,caller", _MATMUL_CALLERS, ids=[c[0] for c in _MATMUL_CALLERS])
@pytest.mark.parametrize(
    "shape_a,shape_b", _MATMUL_SHAPES, ids=[f"{sa}@{sb}" for sa, sb in _MATMUL_SHAPES]
)
@pytest.mark.parametrize(
    "dt_a,dt_b",
    [
        (lucid.float32, lucid.float32),
        (lucid.float16, lucid.float16),
        (lucid.bfloat16, lucid.bfloat16),
        (lucid.int64, lucid.int64),
    ],
)
def test_matmul_shape_and_dtype_parity(
    caller_name: str,
    caller: Callable[[Tensor, Tensor], Tensor],
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    dt_a: DType,
    dt_b: DType,
    device: str,
) -> None:
    del caller_name
    eager_a = lucid.zeros(shape_a, dtype=dt_a, device=device)
    eager_b = lucid.zeros(shape_b, dtype=dt_b, device=device)
    eager_out = caller(eager_a, eager_b)

    with shadow_alloc():
        ph_a = lucid.zeros(shape_a, dtype=dt_a, device=device)
        ph_b = lucid.zeros(shape_b, dtype=dt_b, device=device)
        ph_out = caller(ph_a, ph_b)

    _assert_same_answer(ph_out, eager_out)


# ── Unary, reduction, and layout ops parity ──────────────────────────────────


@pytest.mark.parametrize(
    "fn",
    [
        lucid.neg,
        lucid.abs,
        lucid.exp,
        lucid.log,
        lucid.sqrt,
        lucid.rsqrt,
        lucid.sin,
        lucid.cos,
        lucid.tanh,
        lucid.sigmoid,
        lucid.relu,
        lucid.isnan,
        lucid.isfinite,
        lucid.logical_not,
    ],
)
@pytest.mark.parametrize("dt", [lucid.bool_, lucid.int64, lucid.float16, lucid.float32])
def test_unary_free_fn_and_method_parity(
    fn: Callable[[Tensor], Tensor], dt: DType, device: str
) -> None:
    if fn is lucid.neg and dt == lucid.bool_:
        pytest.skip("neg is not supported on bool tensors in eager mode")
    eager_x = lucid.ones((2, 3), dtype=dt, device=device)
    eager_out = fn(eager_x)

    with shadow_alloc():
        ph_x = lucid.ones((2, 3), dtype=dt, device=device)
        ph_out = fn(ph_x)
        method_out = getattr(ph_x, fn.__name__)() if hasattr(ph_x, fn.__name__) else ph_out

    _assert_same_answer(ph_out, eager_out)
    _assert_same_answer(method_out, eager_out)


@pytest.mark.parametrize(
    "op_name",
    [
        "sum",
        "mean",
        "prod",
        "max",
        "min",
        "var",
        "std",
        "argmax",
        "argmin",
        "any",
        "all",
        "logsumexp",
    ],
)
@pytest.mark.parametrize("dt", [lucid.int32, lucid.int64, lucid.float16, lucid.float32])
@pytest.mark.parametrize("dim", [None, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduction_free_fn_and_method_parity(
    op_name: str, dt: DType, dim: int | None, keepdim: bool, device: str
) -> None:
    fn = getattr(lucid, op_name)
    eager_x = lucid.ones((2, 3, 4), dtype=dt, device=device)
    eager_out = fn(eager_x, dim=dim, keepdim=keepdim)

    with shadow_alloc():
        ph_x = lucid.ones((2, 3, 4), dtype=dt, device=device)
        ph_out = fn(ph_x, dim=dim, keepdim=keepdim)
        method_out = getattr(ph_x, op_name)(dim=dim, keepdim=keepdim)

    if isinstance(eager_out, tuple):
        for got in (ph_out, method_out):
            assert isinstance(got, tuple) and len(got) == len(eager_out)
            for g, e in zip(got, eager_out, strict=True):
                _assert_same_answer(g, e)
    else:
        _assert_same_answer(ph_out, eager_out)
        _assert_same_answer(method_out, eager_out)


def test_direct_reproducers_from_issue_419(device: str) -> None:
    """Explicit guard for the three reproducers reported in LCD-346 / #419."""
    with shadow_alloc():
        a = lucid.arange(4, device=device)  # int64, shape (4,)
        b = lucid.zeros((3, 4), dtype=lucid.float32, device=device)

        # 1. lucid.add(phantom, phantom) works and broadcasts + promotes
        added = lucid.add(a, b)
        assert added.shape == (3, 4)
        assert added.dtype == lucid.float32

        # 2. a * 0.5 promotes int64 -> float32 instead of keeping first operand's int64
        scaled = a * 0.5
        assert scaled.shape == (4,)
        assert scaled.dtype == lucid.float32

        # 3. 2 - b keeps b's shape (3, 4) instead of 0-d scalar's ()
        rsub = 2 - b
        assert rsub.shape == (3, 4)
        assert rsub.dtype == lucid.float32
        rsub_int = 2 - a
        assert rsub_int.shape == (4,)
        assert rsub_int.dtype == lucid.int64


# ── What Metal refuses for real, it refuses as a phantom ─────────────────────

# Every way onto Metal: the factories, a cast of a Metal tensor, a move of a CPU
# one, both at once, and a layer's own parameters.
_WAYS_ONTO_METAL: dict[str, Callable[[DType], object]] = {
    "zeros": lambda dt: lucid.zeros(2, dtype=dt, device="metal"),
    "ones": lambda dt: lucid.ones(2, dtype=dt, device="metal"),
    "empty": lambda dt: lucid.empty(2, dtype=dt, device="metal"),
    "full": lambda dt: lucid.full((2,), 1.0, dtype=dt, device="metal"),
    "eye": lambda dt: lucid.eye(2, dtype=dt, device="metal"),
    "tensor": lambda dt: lucid.tensor([1.0, 2.0], dtype=dt, device="metal"),
    "zeros_like": lambda dt: lucid.zeros_like(lucid.ones(2, device="metal"), dtype=dt),
    "new_zeros": lambda dt: lucid.ones(2, device="metal").new_zeros(2, dtype=dt),
    "cast": lambda dt: lucid.zeros(2, device="metal").to(dt),
    "move": lambda dt: lucid.zeros(2, dtype=dt).to("metal"),
    "move_and_cast": lambda dt: lucid.zeros(2).to("metal", dt),
}

_REAL_NUMBER_WAYS: dict[str, Callable[[DType], object]] = {
    "arange": lambda dt: lucid.arange(0.0, 3.0, 1.0, dtype=dt, device="metal"),
    "linspace": lambda dt: lucid.linspace(0.0, 1.0, 3, dtype=dt, device="metal"),
    "logspace": lambda dt: lucid.logspace(0.0, 1.0, 3, dtype=dt, device="metal"),
    "rand": lambda dt: lucid.rand(2, dtype=dt, device="metal"),
    "randn": lambda dt: lucid.randn(2, dtype=dt, device="metal"),
    "linear": lambda dt: nn.Linear(2, 2, dtype=dt, device="metal"),
    "embedding": lambda dt: nn.Embedding(4, 2, padding_idx=0, dtype=dt, device="metal"),
}

_REFUSED_ON_METAL = [
    *(
        pytest.param(way, dt, id=f"{name}-{dt}")
        for name, way in _WAYS_ONTO_METAL.items()
        for dt in sorted(METAL_UNSUPPORTED_DTYPES, key=str)
    ),
    *(pytest.param(way, lucid.float64, id=name) for name, way in _REAL_NUMBER_WAYS.items()),
]


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
@pytest.mark.parametrize("way, dt", _REFUSED_ON_METAL)
def test_a_dtype_metal_refuses_is_refused_as_a_phantom(
    way: Callable[[DType], object], dt: DType
) -> None:
    with pytest.raises(_C_engine.NotImplementedError, match="is not supported on GPU") as eager:
        way(dt)
    with shadow_alloc(), pytest.raises(type(eager.value)) as ghost:
        way(dt)
    assert str(ghost.value) == str(eager.value)


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_moving_a_real_float64_tensor_to_metal_is_refused_under_the_mode() -> None:
    real = lucid.zeros(2, dtype=lucid.float64)
    with shadow_alloc(), pytest.raises(_C_engine.NotImplementedError, match="float64"):
        real.to("metal")


@pytest.mark.parametrize("dt", sorted(METAL_UNSUPPORTED_DTYPES, key=str))
def test_the_cpu_still_holds_what_metal_refuses(dt: DType) -> None:
    with shadow_alloc():
        ghost = lucid.zeros(2, dtype=dt, device="cpu")
    _assert_same_answer(ghost, lucid.zeros(2, dtype=dt, device="cpu"))
