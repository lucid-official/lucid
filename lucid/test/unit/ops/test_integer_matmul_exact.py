"""An integer matrix product wraps at the dtype's width and is exact (LCD-1067).

Neither device has an integer GEMM -- Accelerate and MLX both refuse one -- so the
engine used to take the product through a floating one and cast it back:

* Metal widened to float32, which holds 24 bits, and the cast back *saturated*: an
  int8 ``[[100, 100], [100, 100]]`` squared came out 127 where the CPU and the
  reference wrap to 32, and ``mm([[16777217]], [[1]])`` came out 16777216.
* The CPU widened to float64, which holds 53 bits: ``mm([[2**53 + 1]], [[1]])`` came
  out ``2**53`` on both devices, the low bit gone before the wrap could run.

Each device now owns one integer product (``backend/cpu/IntGemm.h``,
``backend/gpu/IntGemm.h``) that wraps two's-complement at the operand width, and every
member of the matmul family reaches it: the product of two integer matrices is an
integer matrix, and its low ``width`` bits depend only on the low ``width`` bits of
its terms, so the expected value of every cell below is the exact Python-integer
result reduced modulo ``2**width``.

The cells cross every op that reaches the engine's ``matmul`` (``matmul``, ``mm``,
``bmm``, ``mv``, the ``add*`` family, ``outer``, ``inner``, ``linalg.dot``,
``linalg.multi_dot``, the CPU and GPU ``tensordot``) with the four integer dtypes and
both devices.  The operands are drawn over the whole range of the dtype, so every
dot product overflows its own width and passes 2**24 and, for int64, 2**53.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params, metal_available

type Nested = int | list[Nested]

_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_BITS = {lucid.int8: 8, lucid.int16: 16, lucid.int32: 32, lucid.int64: 64}


def _wrap(value: int, bits: int) -> int:
    """``value`` as a two's-complement integer of ``bits`` bits."""
    half = 1 << (bits - 1)
    return ((value + half) % (1 << bits)) - half


def _wrap_all(x: Nested, bits: int) -> Nested:
    if isinstance(x, int):
        return _wrap(x, bits)
    return [_wrap_all(e, bits) for e in x]


def _random(rng: random.Random, shape: tuple[int, ...], bits: int) -> Nested:
    """Nested lists of ``shape``, uniform over the whole range of a ``bits``-bit integer."""
    if not shape:
        return rng.randint(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return [_random(rng, shape[1:], bits) for _ in range(shape[0])]


def _t(a: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*a, strict=True)]


def _mm(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """The exact (unwrapped) product of two matrices."""
    cols = _t(b)
    return [[sum(x * y for x, y in zip(row, col, strict=True)) for col in cols] for row in a]


def _bmm(a: list[list[list[int]]], b: list[list[list[int]]]) -> list[list[list[int]]]:
    return [_mm(x, y) for x, y in zip(a, b, strict=True)]


def _zip_with(f: Callable[[int, int], int], x: Nested, y: Nested) -> Nested:
    if isinstance(x, int) and isinstance(y, int):
        return f(x, y)
    assert isinstance(x, list) and isinstance(y, list)
    return [_zip_with(f, p, q) for p, q in zip(x, y, strict=True)]


def _add(x: Nested, y: Nested) -> Nested:
    return _zip_with(lambda p, q: p + q, x, y)


def _scaled(x: Nested, factor: int) -> Nested:
    if isinstance(x, int):
        return factor * x
    return [_scaled(e, factor) for e in x]


def _col(v: list[int]) -> list[list[int]]:
    return [[x] for x in v]


def _tensordot2(a: list[list[list[int]]], b: list[list[list[int]]]) -> list[list[int]]:
    """``tensordot(a, b, dims=2)`` of ``[i, j, k]`` and ``[j, k, q]``: sum over ``j, k``."""
    flat_a = [[x for row in plane for x in row] for plane in a]
    flat_b = [
        [b[j][k][q] for j in range(len(b)) for k in range(len(b[0]))] for q in range(len(b[0][0]))
    ]
    return [[sum(x * y for x, y in zip(ra, rb, strict=True)) for rb in flat_b] for ra in flat_a]


@dataclass(frozen=True)
class _Case:
    """One op: operand shapes, the call, and its exact (unwrapped) answer in Python integers."""

    name: str
    shapes: tuple[tuple[int, ...], ...]
    call: Callable[..., lucid.Tensor]
    expect: Callable[..., Nested]


_CASES = [
    _Case("matmul", ((3, 5), (5, 4)), lambda a, b: lucid.matmul(a, b), _mm),
    _Case("matmul_operator", ((3, 5), (5, 4)), lambda a, b: a @ b, _mm),
    _Case("matmul_batched", ((2, 3, 5), (2, 5, 4)), lambda a, b: lucid.matmul(a, b), _bmm),
    _Case(
        "matmul_broadcast_batch",
        ((2, 3, 5), (5, 4)),
        lambda a, b: lucid.matmul(a, b),
        lambda a, b: [_mm(x, b) for x in a],
    ),
    _Case(
        "matmul_vector_rhs",
        ((3, 5), (5,)),
        lambda a, v: lucid.matmul(a, v),
        lambda a, v: [row[0] for row in _mm(a, _col(v))],
    ),
    _Case(
        "matmul_vector_lhs",
        ((5,), (5, 4)),
        lambda v, b: lucid.matmul(v, b),
        lambda v, b: _mm([v], b)[0],
    ),
    _Case(
        "matmul_transposed_view",
        ((5, 3), (5, 4)),
        lambda a, b: lucid.matmul(a.mT, b),
        lambda a, b: _mm(_t(a), b),
    ),
    _Case(
        "matmul_strided_slice",
        ((3, 10), (5, 4)),
        lambda a, b: lucid.matmul(a[:, ::2], b),
        lambda a, b: _mm([row[::2] for row in a], b),
    ),
    _Case("mm", ((3, 5), (5, 4)), lambda a, b: lucid.mm(a, b), _mm),
    _Case("bmm", ((2, 3, 5), (2, 5, 4)), lambda a, b: lucid.bmm(a, b), _bmm),
    _Case(
        "mv",
        ((3, 5), (5,)),
        lambda a, v: lucid.mv(a, v),
        lambda a, v: [row[0] for row in _mm(a, _col(v))],
    ),
    _Case(
        "addmm",
        ((3, 4), (3, 5), (5, 4)),
        lambda c, a, b: lucid.addmm(c, a, b),
        lambda c, a, b: _add(c, _mm(a, b)),
    ),
    _Case(
        "addmm_scaled",
        ((3, 4), (3, 5), (5, 4)),
        lambda c, a, b: lucid.addmm(c, a, b, beta=2, alpha=3),
        lambda c, a, b: _add(_scaled(c, 2), _scaled(_mm(a, b), 3)),
    ),
    _Case(
        "addmv",
        ((3,), (3, 5), (5,)),
        lambda c, a, v: lucid.addmv(c, a, v),
        lambda c, a, v: _add(c, [row[0] for row in _mm(a, _col(v))]),
    ),
    _Case(
        "baddbmm",
        ((2, 3, 4), (2, 3, 5), (2, 5, 4)),
        lambda c, a, b: lucid.baddbmm(c, a, b),
        lambda c, a, b: _add(c, _bmm(a, b)),
    ),
    _Case(
        "addbmm",
        ((3, 4), (2, 3, 5), (2, 5, 4)),
        lambda c, a, b: lucid.addbmm(c, a, b),
        lambda c, a, b: _add(c, reduce(_add, _bmm(a, b))),
    ),
    _Case("linalg_dot", ((3, 5), (5, 4)), lambda a, b: lucid.linalg.dot(a, b), _mm),
    _Case(
        "linalg_multi_dot",
        ((3, 5), (5, 4), (4, 2)),
        lambda a, b, c: lucid.linalg.multi_dot([a, b, c]),
        lambda a, b, c: _mm(_mm(a, b), c),
    ),
    _Case(
        "outer", ((3,), (4,)), lambda a, b: a.outer(b), lambda a, b: [[x * y for y in b] for x in a]
    ),
    _Case(
        "addr",
        ((3, 4), (3,), (4,)),
        lambda c, a, b: lucid.addr(c, a, b),
        lambda c, a, b: _add(c, [[x * y for y in b] for x in a]),
    ),
    _Case("inner", ((3, 5), (4, 5)), lambda a, b: a.inner(b), lambda a, b: _mm(a, _t(b))),
    _Case("tensordot_dims1", ((3, 5), (5, 4)), lambda a, b: lucid.tensordot(a, b, dims=1), _mm),
    _Case(
        "tensordot_dims2",
        ((2, 3, 4), (3, 4, 5)),
        lambda a, b: lucid.tensordot(a, b, dims=2),
        _tensordot2,
    ),
    _Case(
        "einsum_contraction",
        ((3, 5), (5, 4)),
        lambda a, b: lucid.einops.einsum("ij,jk->ik", a, b),
        _mm,
    ),
]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_every_integer_product_wraps_at_the_dtype_width_on_both_devices(
    case: _Case, device: str, dtype: lucid.dtype
) -> None:
    bits = _BITS[dtype]
    rng = random.Random(f"{case.name}-{dtype}")
    operands = [_random(rng, shape, bits) for shape in case.shapes]
    tensors = [lucid.tensor(x, dtype=dtype, device=device) for x in operands]
    out = case.call(*tensors)
    assert out.dtype == dtype
    assert out.tolist() == _wrap_all(case.expect(*operands), bits)


# The products the issue reported, with the answers the reference gives.
_PINNED = [
    pytest.param(
        lucid.int8,
        [[100, 100], [100, 100]],
        [[100, 100], [100, 100]],
        [[32, 32], [32, 32]],
        id="int8-wraps-where-metal-saturated-127",
    ),
    pytest.param(
        lucid.int16,
        [[30000, 30000], [30000, 30000]],
        [[30000, 30000], [30000, 30000]],
        [[-11776, -11776], [-11776, -11776]],
        id="int16-wraps-where-metal-saturated-32767",
    ),
    pytest.param(
        lucid.int32,
        [[2**30, 2**30], [2**30, 2**30]],
        [[2**30, 2**30], [2**30, 2**30]],
        [[0, 0], [0, 0]],
        id="int32-wraps-where-metal-saturated-2147483647",
    ),
    pytest.param(lucid.int32, [[2**24 + 1]], [[1]], [[2**24 + 1]], id="int32-past-2**24"),
    pytest.param(lucid.int64, [[2**53 + 1]], [[1]], [[2**53 + 1]], id="int64-past-2**53"),
    pytest.param(
        lucid.int64,
        [[2**62 + 1, 7]],
        [[3], [5]],
        [[-4611686018427387866]],
        id="int64-low-bits-a-double-drops",
    ),
    pytest.param(
        lucid.int64, [[-(2**63), 2**63 - 1]], [[-(2**63)], [2**63 - 1]], [[1]], id="int64-extremes"
    ),
]

_PINNED_OPS: dict[str, Callable[..., lucid.Tensor]] = {
    "mm": lambda a, b: lucid.mm(a, b),
    "matmul": lambda a, b: lucid.matmul(a, b),
    "bmm": lambda a, b: lucid.bmm(a[None], b[None])[0],
    "tensordot": lambda a, b: lucid.tensordot(a, b, dims=1),
    "linalg_dot": lambda a, b: lucid.linalg.dot(a, b),
}


@pytest.mark.parametrize("op", sorted(_PINNED_OPS))
@pytest.mark.parametrize(("dtype", "a", "b", "want"), _PINNED)
def test_the_reported_products_have_the_reference_answer(
    op: str, dtype: lucid.dtype, a: Nested, b: Nested, want: Nested, device: str
) -> None:
    out = _PINNED_OPS[op](
        lucid.tensor(a, dtype=dtype, device=device), lucid.tensor(b, dtype=dtype, device=device)
    )
    assert out.dtype == dtype
    assert out.tolist() == want


# A bool has no matrix product (``matmul`` refuses it, and so does each backend's integer GEMM),
# but its outer product is ``a[i] & b[j]``, which the engine takes as the elementwise product of
# the two views.
_BOOL_OUTERS: dict[str, Callable[..., lucid.Tensor]] = {
    "outer": lambda a, b: a.outer(b),
    "linalg_outer": lambda a, b: lucid.linalg.outer(a, b),
    "ger": lambda a, b: lucid.ger(a, b),
}

_BOOL_A = [True, False, True]
_BOOL_B = [True, True, False, False, True]


@pytest.mark.parametrize("name", sorted(_BOOL_OUTERS))
def test_a_bool_outer_product_is_the_elementwise_and(name: str, device: str) -> None:
    a = lucid.tensor(_BOOL_A, device=device)
    b = lucid.tensor(_BOOL_B, device=device)
    out = _BOOL_OUTERS[name](a, b)
    assert out.dtype == lucid.bool_
    assert out.tolist() == [[x and y for y in _BOOL_B] for x in _BOOL_A]


@pytest.mark.parametrize("name", sorted(_BOOL_OUTERS))
def test_an_empty_bool_outer_product_keeps_its_shape(name: str, device: str) -> None:
    empty = lucid.zeros((0,), dtype=lucid.bool_, device=device)
    full = lucid.tensor(_BOOL_B, device=device)
    assert _BOOL_OUTERS[name](empty, full).shape == (0, len(_BOOL_B))
    assert _BOOL_OUTERS[name](full, empty).shape == (len(_BOOL_B), 0)


def test_addr_of_a_bool_matrix_adds_the_and_with_or(device: str) -> None:
    base = [[bool((i + j) % 2) for j in range(len(_BOOL_B))] for i in range(len(_BOOL_A))]
    out = lucid.addr(
        lucid.tensor(base, device=device),
        lucid.tensor(_BOOL_A, device=device),
        lucid.tensor(_BOOL_B, device=device),
    )
    assert out.dtype == lucid.bool_
    assert out.tolist() == [
        [base[i][j] or (x and y) for j, y in enumerate(_BOOL_B)] for i, x in enumerate(_BOOL_A)
    ]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_an_empty_integer_product_keeps_its_shape_and_an_empty_sum_is_zero(
    device: str, dtype: lucid.dtype
) -> None:
    def zeros(*shape: int) -> lucid.Tensor:
        return lucid.zeros(shape, dtype=dtype, device=device)

    empty_sum = lucid.matmul(zeros(3, 0), zeros(0, 4))
    assert empty_sum.dtype == dtype
    assert empty_sum.tolist() == [[0] * 4] * 3
    assert lucid.matmul(zeros(0, 3), zeros(3, 4)).shape == (0, 4)
    assert lucid.matmul(zeros(3, 5), zeros(5, 0)).shape == (3, 0)
    assert lucid.bmm(zeros(0, 2, 3), zeros(0, 3, 2)).shape == (0, 2, 2)
    assert lucid.tensordot(zeros(3, 0), zeros(0, 4), dims=1).tolist() == [[0] * 4] * 3


@pytest.mark.skipif(not metal_available(), reason="no Metal device")
@pytest.mark.parametrize(
    ("m", "k", "n"), [(140, 2, 16000), (3, 2100, 2100)], ids=["row-blocks", "k-blocks"]
)
def test_a_product_too_large_for_one_block_matches_the_cpu(m: int, k: int, n: int) -> None:
    """The GPU forms ``batch * M * K * N`` products in blocks of at most 2**22 elements.

    Both shapes exceed one block: the first splits the rows, the second one row's
    ``K * N`` and so ``K`` too.  The CPU kernel is a different implementation, and it is
    pinned against Python integers above.
    """
    lucid.manual_seed(1067)
    a = lucid.randint(-(2**30), 2**30, (m, k), dtype=lucid.int32, device="cpu")
    b = lucid.randint(-(2**30), 2**30, (k, n), dtype=lucid.int32, device="cpu")
    want = lucid.matmul(a, b)
    got = lucid.matmul(a.to("metal"), b.to("metal")).to("cpu")
    assert got.dtype == lucid.int32
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=[str(d) for d in _INT_DTYPES])
@pytest.mark.parametrize("op", ["mm", "matmul", "bmm", "mv"])
def test_integer_products_match_the_reference_on_the_cpu(
    ref: ModuleType, op: str, dtype: lucid.dtype
) -> None:
    bits = _BITS[dtype]
    rng = random.Random(f"parity-{op}-{dtype}")
    shapes = {
        "mm": ((6, 9), (9, 7)),
        "matmul": ((6, 9), (9, 7)),
        "bmm": ((3, 6, 9), (3, 9, 7)),
        "mv": ((6, 9), (9,)),
    }[op]
    a, b = (_random(rng, shape, bits) for shape in shapes)
    ref_dtype = getattr(ref, str(dtype).split(".")[-1])
    got = getattr(lucid, op)(lucid.tensor(a, dtype=dtype), lucid.tensor(b, dtype=dtype))
    want = getattr(ref, op)(ref.tensor(a, dtype=ref_dtype), ref.tensor(b, dtype=ref_dtype))
    assert got.tolist() == want.tolist()
