"""``//``, ``%``, ``fmod`` and ``remainder`` are exact, on both devices (LCD-159).

They used to divide in float.  Integer ``//`` divided in double on the CPU
and float32 on Metal, so ``(2**53 + 1) // 1`` was 2**53 and Metal's
``100000001 // 3`` was 33333334.  ``fmod`` and ``remainder`` were
``a - trunc(a / b) * b`` and ``a - floor(a / b) * b``, which round the
quotient: ``(2**62 + 3) % 7`` was 259, and Metal's ``arange(2e7) % 3``
left half a million values outside ``[0, 2]``.  An integer divided by zero
answered -1 or INT64_MAX instead of raising, and the float forms got the
infinite divisors wrong (``-5 % inf`` was NaN, not inf).

An integer zero divisor now raises on the CPU, as the reference's CPU does.
Metal does not read the divisor back (no device sync, as on the
reference's accelerator) and answers a defined value instead.

Integers are held to Python's own integer arithmetic, bit for bit,
wrapped to the dtype where the quotient overflows (``min // -1``).
Floats are held to the reference framework, bit for bit, signed zeros
and NaN positions included.  Every integer dtype, the four float dtypes,
both devices (Metal has no float64).

Metal flushes float32 subnormals to zero, so no value here is subnormal.
"""

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import device_dtype_params, metal_available

Op = Callable[[lucid.Tensor, lucid.Tensor | float], lucid.Tensor]

#: Every spelling of the family — ``div``'s rounding modes (LCD-485) too:
#: ``"floor"`` is ``floor_divide``'s path, ``"trunc"`` rounds toward zero.
OPS: dict[str, Op] = {
    "floordiv": lambda a, b: a // b,
    "floor_divide": lucid.floor_divide,
    "div_floor": lambda a, b: lucid.div(a, b, rounding_mode="floor"),
    "Tensor.div_floor": lambda a, b: a.div(b, rounding_mode="floor"),
    "div_trunc": lambda a, b: lucid.div(a, b, rounding_mode="trunc"),
    "Tensor.div_trunc": lambda a, b: a.div(b, rounding_mode="trunc"),
    "divide_trunc": lambda a, b: lucid.divide(a, b, rounding_mode="trunc"),
    "mod": lambda a, b: a % b,
    "remainder": lucid.remainder,
    "fmod": lucid.fmod,
}
_KIND = {
    "floordiv": "floor",
    "floor_divide": "floor",
    "div_floor": "floor",
    "Tensor.div_floor": "floor",
    "div_trunc": "trunc",
    "Tensor.div_trunc": "trunc",
    "divide_trunc": "trunc",
    "mod": "remainder",
    "remainder": "remainder",
    "fmod": "fmod",
}

_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_FLOAT_DTYPES = [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64]
_BITS = {lucid.int8: 8, lucid.int16: 16, lucid.int32: 32, lucid.int64: 64}


# ── integers: Python's arithmetic is the oracle ──────────────────────────────


def _wrap(v: int, bits: int) -> int:
    half = 1 << (bits - 1)
    return ((v + half) % (1 << bits)) - half


def _int_oracle(kind: str, a: int, b: int, bits: int) -> int:
    if kind == "floor":
        return _wrap(a // b, bits)
    if kind == "trunc":
        q = abs(a) // abs(b)
        return _wrap(-q if (a < 0) != (b < 0) else q, bits)
    if kind == "remainder":
        return a % b
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _int_operands(bits: int) -> tuple[list[int], list[int]]:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    near = [0, 1, -1, 2, -2, 3, -3, 7, -7, 100, -100, 100_000_001]
    powers = [2**24 - 1, 2**24 + 1, 2**31 - 1, 2**53 - 1, 2**53 + 1, 2**62 + 3]
    edges = [lo, lo + 1, hi, hi - 1]
    nums = [v for v in near + powers + [-p for p in powers] + edges if lo <= v <= hi]
    dens = [
        v
        for v in [1, -1, 2, -2, 3, -3, 7, -7, 2**24 + 1, -(2**31 - 1), lo, hi]
        if lo <= v <= hi and v != 0
    ]
    pairs = [(a, b) for a in nums for b in dens]
    return [a for a, _ in pairs], [b for _, b in pairs]


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_integer_results_are_exact(device: str, dtype: lucid.dtype, op: str) -> None:
    bits = _BITS[dtype]
    nums, dens = _int_operands(bits)
    a = lucid.tensor(nums, dtype=dtype, device=device)
    b = lucid.tensor(dens, dtype=dtype, device=device)
    out = OPS[op](a, b)
    assert out.dtype == dtype
    want = [_int_oracle(_KIND[op], x, y, bits) for x, y in zip(nums, dens, strict=True)]
    assert out.tolist() == want


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_integer_zero_divisor_raises_on_the_cpu(dtype: lucid.dtype, op: str) -> None:
    a = lucid.tensor([[5, -5, 6]], dtype=dtype)
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        OPS[op](a, lucid.tensor([[3], [0]], dtype=dtype))
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        OPS[op](a, lucid.tensor(0, dtype=dtype))


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_integer_zero_divisor_on_metal_is_defined_and_unsynced(dtype: lucid.dtype, op: str) -> None:
    """Metal reads nothing back.

    ``a // 0`` and its truncated form are -1; ``a % 0`` and ``fmod(a, 0)``
    are ``a``.
    """
    if not metal_available():
        pytest.skip("Metal unavailable")
    nums = [5, -5, 6, 0, -(2**7)]
    a = lucid.tensor([nums], dtype=dtype, device="metal")
    b = lucid.tensor([[3], [0]], dtype=dtype, device="metal")
    lucid.eval(a, b)
    before = _C_engine.host_sync_count()
    out = OPS[op](a, b)
    lucid.eval(out)
    assert _C_engine.host_sync_count() == before
    at_zero = [-1] * len(nums) if _KIND[op] in ("floor", "trunc") else nums
    at_three = [_int_oracle(_KIND[op], x, 3, _BITS[dtype]) for x in nums]
    assert out.tolist() == [at_three, at_zero]


def test_the_reported_cases(device: str) -> None:
    def t(*v: int) -> lucid.Tensor:
        return lucid.tensor(list(v), dtype=lucid.int64, device=device)

    assert (t(100000001) // 3).tolist() == [33333333]
    assert (t(100000001) % 3).tolist() == [2]
    assert (t(2**53 + 1) // 1).tolist() == [2**53 + 1]
    assert (t(2**62 + 3) % 7).tolist() == [(2**62 + 3) % 7]
    assert (t(-(2**63)) // -1).tolist() == [-(2**63)]


def test_large_metal_remainder_is_exact() -> None:
    if not metal_available():
        pytest.skip("Metal unavailable")
    n = 20_000_000
    x = lucid.arange(n, dtype=lucid.int64, device="metal")
    want = np.arange(n, dtype=np.int64)
    np.testing.assert_array_equal((x % 3).numpy(), want % 3)
    np.testing.assert_array_equal((x // 3).numpy(), want // 3)


def test_metal_broadcast_and_wide_int64_divmod_are_exact(ref: ModuleType) -> None:
    """Contiguous, scalar-broadcast, and 2-D strided-broadcast paths (LCD-448)."""
    if not metal_available():
        pytest.skip("Metal unavailable")
    nums = [-(2**63), -(2**62) - 3, -(2**40) - 7, 2**40 + 7, 2**62 + 3, 2**63 - 1]
    dens = [-7, 3, -(2**33) - 5, 2**33 + 5]
    a = lucid.tensor([nums], dtype=lucid.int64, device="metal")
    b = lucid.tensor([[d] for d in dens], dtype=lucid.int64, device="metal")
    for op in ("floordiv", "mod", "fmod", "div_trunc"):
        got = OPS[op](a, b).tolist()
        want = [[_int_oracle(_KIND[op], x, y, 64) for x in nums] for y in dens]
        assert got == want

    fa = lucid.tensor([[0.0, -0.0, 5.0, -5.0, _INF, -_INF]], dtype=lucid.float32, device="metal")
    fb = lucid.tensor([[1.0], [-1.0], [_INF], [-_INF]], dtype=lucid.float32, device="metal")
    rfa = ref.tensor([[0.0, -0.0, 5.0, -5.0, _INF, -_INF]], dtype=ref.float32)
    rfb = ref.tensor([[1.0], [-1.0], [_INF], [-_INF]], dtype=ref.float32)
    for op in ("floordiv", "mod", "fmod"):
        _assert_same_floats(OPS[op](fa, fb), _ref_op(ref, op)(rfa, rfb))


# ── floats: the reference is the oracle ──────────────────────────────────────

_INF, _NAN = math.inf, math.nan
_FLOAT_NUMS = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 0.5, 2.5, -2.5, 7.5, -7.5, 3.0]
_FLOAT_NUMS += [2.0**24 - 1, 2.0**24 + 1, 2.0**53 + 1, -(2.0**53) - 1, 1e30, -1e30]
_FLOAT_NUMS += [_INF, -_INF, _NAN]
_FLOAT_DENS = [1.0, -1.0, 0.1, -0.1, 3.0, -3.0, 0.3, 7.0, 2.0**24 + 1, 1e-3, 1e30]
_FLOAT_DENS += [_INF, -_INF, 0.0, -0.0, _NAN]


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _ref_op(ref: ModuleType, op: str) -> Callable[..., object]:
    ops: dict[str, Callable[..., object]] = {
        "floordiv": lambda a, b: a // b,
        "floor_divide": ref.floor_divide,
        "div_floor": lambda a, b: ref.div(a, b, rounding_mode="floor"),
        "Tensor.div_floor": lambda a, b: ref.div(a, b, rounding_mode="floor"),
        "div_trunc": lambda a, b: ref.div(a, b, rounding_mode="trunc"),
        "Tensor.div_trunc": lambda a, b: ref.div(a, b, rounding_mode="trunc"),
        "divide_trunc": lambda a, b: ref.div(a, b, rounding_mode="trunc"),
        "mod": lambda a, b: a % b,
        "remainder": ref.remainder,
        "fmod": ref.fmod,
    }
    return ops[op]


_HALF = (lucid.float16, lucid.bfloat16)


def _half_oracle(ref: ModuleType, op: str, a: object, b: object, rdt: object) -> object:
    """The exact answer for 16-bit operands, rounded once into their dtype.

    The reference's own half kernels disagree with each other: for bfloat16
    ``7.5 // 1e-3`` its scalar loop answers 7488 and its vector loop 7520
    (the exact quotient is 7504, a tie that rounds to even, 7488).  Lucid
    computes in float32 and rounds once, so it is held to the exact value:
    the reference's float64 answer on the same 16-bit values, rounded.
    """
    wide = _ref_op(ref, op)(getattr(a, "double")(), getattr(b, "double")())
    return getattr(wide, "to")(rdt)


def _bits(values: np.ndarray) -> np.ndarray:
    """Float values as comparable bit patterns; every NaN as one pattern."""
    wide = values.astype(np.float64)
    wide[np.isnan(wide)] = np.nan
    return wide.view(np.int64)


def _assert_same_floats(got: lucid.Tensor, want: object) -> None:
    g = got.to(lucid.float64 if got.device == "cpu" else lucid.float32).numpy()
    w = np.asarray(getattr(want, "double")().numpy())
    np.testing.assert_array_equal(_bits(g), _bits(w))


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_FLOAT_DTYPES))
def test_float_results_match_the_reference(
    ref: ModuleType,
    device: str,
    dtype: lucid.dtype,
    op: str,
) -> None:
    pairs = [(a, b) for a in _FLOAT_NUMS for b in _FLOAT_DENS]
    nums, dens = [a for a, _ in pairs], [b for _, b in pairs]
    rdt = _ref_dtype(ref, dtype)
    ra, rb = ref.tensor(nums, dtype=rdt), ref.tensor(dens, dtype=rdt)
    # A truncated quotient is the rounded 16-bit quotient's ``trunc``, not
    # the exact one's, so the reference's own 16-bit kernel is the oracle.
    if dtype in _HALF and _KIND[op] != "trunc":
        want = _half_oracle(ref, op, ra, rb, rdt)
    else:
        want = _ref_op(ref, op)(ra, rb)
    a = lucid.tensor(nums, dtype=dtype, device=device)
    b = lucid.tensor(dens, dtype=dtype, device=device)
    out = OPS[op](a, b)
    assert out.dtype == dtype
    _assert_same_floats(out, want)


@pytest.mark.parametrize("divisor", [2.0, -0.3, 1e-3, _INF, -_INF, 0.0])
@pytest.mark.parametrize("op", ["floordiv", "div_floor", "div_trunc", "mod"])
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float16, lucid.bfloat16]))
def test_half_tensor_over_a_number(
    ref: ModuleType, device: str, dtype: lucid.dtype, op: str, divisor: float
) -> None:
    """A half tensor and a Python number.

    ``//`` and both rounding divisions read the number as float32
    (op-math); ``%`` rounds it into the tensor's dtype first, as the
    reference does for every op but mul, div and floor division.  A
    truncated quotient is truncated in float32 and rounded once, which is
    the reference's own kernel on the number.
    """
    nums = [0.0, -0.0, 1.0, -1.0, 7.5, -7.5, 1000.0, -3.0, 0.1]
    rdt = _ref_dtype(ref, dtype)
    if _KIND[op] == "trunc":
        want = _ref_op(ref, op)(ref.tensor(nums, dtype=rdt), divisor)
    else:
        as_dtype = ref.float32 if _KIND[op] == "floor" else rdt
        den = ref.full((len(nums),), divisor, dtype=as_dtype)
        want = _half_oracle(ref, op, ref.tensor(nums, dtype=rdt), den, rdt)
    out = OPS[op](lucid.tensor(nums, dtype=dtype, device=device), divisor)
    assert out.dtype == dtype
    _assert_same_floats(out, want)


# ── gradients ────────────────────────────────────────────────────────────────

_GRAD_A = [7.5, -7.5, 5.25, -5.25, 0.75, 100.5]
_GRAD_B = [2.0, 2.0, -1.5, -1.5, 0.5, 3.0]


@pytest.mark.parametrize("op", ["fmod", "remainder"])
def test_gradients_match_the_reference(ref: ModuleType, device: str, op: str) -> None:
    a = lucid.tensor(_GRAD_A, device=device, requires_grad=True)
    b = lucid.tensor(_GRAD_B, device=device, requires_grad=True)
    ra = ref.tensor(_GRAD_A, requires_grad=True)
    rb = ref.tensor(_GRAD_B, requires_grad=True)
    (OPS[op](a, b) * lucid.tensor(_GRAD_B, device=device)).sum().backward()
    (_ref_op(ref, op)(ra, rb) * ref.tensor(_GRAD_B)).sum().backward()
    assert a.grad is not None and b.grad is not None
    np.testing.assert_array_equal(a.grad.numpy(), ra.grad.numpy())
    np.testing.assert_array_equal(b.grad.numpy(), rb.grad.numpy())


@pytest.mark.parametrize("op", ["fmod", "remainder"])
def test_gradients_build_a_graph(ref: ModuleType, device: str, op: str) -> None:
    """``create_graph=True`` differentiates the gradient again."""
    a = lucid.tensor(_GRAD_A, device=device, requires_grad=True)
    b = lucid.tensor(_GRAD_B, device=device, requires_grad=True)
    ra = ref.tensor(_GRAD_A, requires_grad=True)
    rb = ref.tensor(_GRAD_B, requires_grad=True)
    ga, gb = lucid.autograd.grad(OPS[op](a, b).sum(), [a, b], create_graph=True)
    rout = _ref_op(ref, op)(ra, rb)
    rga, rgb = ref.autograd.grad(getattr(rout, "sum")(), [ra, rb], create_graph=True)
    np.testing.assert_array_equal(gb.numpy(), rgb.detach().numpy())
    (second,) = lucid.autograd.grad((ga * gb).sum(), [b])
    (rsecond,) = ref.autograd.grad((rga * rgb).sum(), [rb], allow_unused=True)
    want = np.zeros(len(_GRAD_B), np.float32) if rsecond is None else rsecond.numpy()
    np.testing.assert_array_equal(second.numpy(), want)


# ── compiled graphs fall back without losing exactness ───────────────────────


class _FmodPlusOne(nn.Module):
    def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return lucid.fmod(a, b) + 1


@pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning")
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.int64])
def test_compiled_fmod_matches_eager_bit_for_bit(dtype: lucid.dtype) -> None:
    """Compiled or not, ``fmod`` is the exact one.

    No compile emitter for ``fmod`` yet (LCD-444), so the graph runs eager;
    an emitter must keep these values when it lands.
    """
    if not metal_available():
        pytest.skip("Metal unavailable")
    if dtype == lucid.int64:
        nums: list[float] = [2**62 + 3, -(2**62) - 3, 7, -7]
        dens: list[float] = [7, 7, -3, 3]
    else:
        nums, dens = [3.0e9 + 128, -7.5, 1e30, 0.75], [0.3, 2.0, 7.0, -0.5]
    a = lucid.tensor(nums, dtype=dtype, device="metal")
    b = lucid.tensor(dens, dtype=dtype, device="metal")
    model = _FmodPlusOne()
    eager = model(a, b).numpy()
    compiled = lucid.compile(model)(a, b).numpy()
    np.testing.assert_array_equal(compiled.view(np.uint8), eager.view(np.uint8))
    if dtype == lucid.int64:
        want = [
            _int_oracle("fmod", int(x), int(y), 64) + 1 for x, y in zip(nums, dens, strict=True)
        ]
        assert compiled.tolist() == want
    else:
        a32 = np.array(nums, np.float32)
        want32 = np.fmod(a32, np.array(dens, np.float32)) + np.float32(1)
        np.testing.assert_array_equal(compiled, want32)
