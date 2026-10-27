"""Parity guard for dtype edge cases across CPU and Metal (LCD-343).

Covers:
1. ``lucid.can_cast(from_, to)`` matches the reference across all 11x11 pairs.
2. ``t == None`` and ``t != None`` (plus arbitrary non-Tensor, non-Scalar objects)
   return ``False`` / ``True`` as plain Python ``bool``.
3. Half and narrow-integer scalar exponent overflow in ``pow`` raises ``RuntimeError``
   across every spelling (``**``, ``lucid.pow``, ``Tensor.pow``, ``**=``, ``pow_``).
4. Complex comparisons on CPU and Metal: ``==`` and ``!=`` work on ``complex64`` and
   ``complex128`` (including NaN semantics), while ``<, <=, >, >=`` raise
   ``NotImplementedError`` on both backends.
5. ``lucid.tensor(scalar, dtype=dt)`` and ``lucid.full((), scalar, dtype=dt)`` return
   a 0-D tensor ``()`` for every dtype including ``bfloat16`` and ``complex128``.
"""

from types import ModuleType
from typing import Any

import pytest

import lucid

pytestmark = pytest.mark.parity

_DTYPES = [
    lucid.bool_,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
]
_CPU_ONLY = {lucid.float64, lucid.complex128}


def _ref_dt(ref: ModuleType, d: lucid.dtype) -> Any:
    name = str(d).removeprefix("lucid.").rstrip("_")
    return getattr(ref, name)


def test_can_cast_matches_reference(ref: ModuleType) -> None:
    for da in _DTYPES:
        for db in _DTYPES:
            got = lucid.can_cast(da, db)
            want = ref.can_cast(_ref_dt(ref, da), _ref_dt(ref, db))
            assert got is want, f"can_cast({da}, {db}): got {got}, want {want}"


@pytest.mark.parametrize("other", [None, "abc", object(), [1, 2]])
def test_equality_against_non_tensor_non_scalar(other: Any, device: str, ref: ModuleType) -> None:
    t = lucid.tensor([1.0, 2.0], device=device)
    rt = ref.tensor([1.0, 2.0])
    eq_got = t == other
    ne_got = t != other
    assert isinstance(eq_got, bool) and eq_got is (rt == other)
    assert isinstance(ne_got, bool) and ne_got is (rt != other)
    req_got = other == t
    rne_got = other != t
    assert isinstance(req_got, bool) and req_got is (other == rt)
    assert isinstance(rne_got, bool) and rne_got is (other != rt)


@pytest.mark.parametrize(
    ("dt", "scalar"),
    [
        (lucid.float16, 100000),
        (lucid.float16, -100000.0),
        (lucid.bfloat16, 1e40),
        (lucid.int8, 200),
        (lucid.int8, -200),
        (lucid.int16, 100000),
        (lucid.int32, 1 << 40),
        (lucid.int64, 1 << 70),
    ],
)
def test_pow_scalar_overflow_raises_runtime_error(
    dt: lucid.dtype, scalar: int | float, device: str, ref: ModuleType
) -> None:
    rt = ref.tensor([1, 2], dtype=_ref_dt(ref, dt))
    with pytest.raises((RuntimeError, OverflowError)):
        _ = rt**scalar

    t = lucid.tensor([1, 2], dtype=lucid.float32, device=device).to(dt)
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = t**scalar
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = lucid.pow(t, scalar)
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = t.pow(scalar)
    with pytest.raises(RuntimeError, match="without overflow"):
        t_ip = t.clone()
        t_ip **= scalar


@pytest.mark.parametrize("dt", [lucid.complex64, lucid.complex128])
def test_complex_comparisons_match_reference(dt: lucid.dtype, device: str, ref: ModuleType) -> None:
    if device != "cpu" and dt in _CPU_ONLY:
        pytest.skip(f"{dt} is CPU-only")
    vals_a = [1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 3.0j, complex(float("nan"), 1.0)]
    vals_b = [1.0 + 2.0j, 1.0 + 4.0j, 2.0 + 3.0j, complex(float("nan"), 1.0)]
    a = lucid.tensor(vals_a, dtype=dt, device=device)
    b = lucid.tensor(vals_b, dtype=dt, device=device)
    ra = ref.tensor(vals_a, dtype=_ref_dt(ref, dt))
    rb = ref.tensor(vals_b, dtype=_ref_dt(ref, dt))

    for fn_l, fn_r in [
        (lambda x, y: x == y, lambda x, y: x == y),
        (lucid.eq, ref.eq),
        (lambda x, y: x != y, lambda x, y: x != y),
        (lucid.ne, ref.ne),
    ]:
        out = fn_l(a, b)
        want = fn_r(ra, rb)
        assert out.dtype == lucid.bool_
        assert out.tolist() == want.tolist()

    for op in [
        lambda x, y: x < y,
        lambda x, y: x <= y,
        lambda x, y: x > y,
        lambda x, y: x >= y,
        lucid.lt,
        lucid.le,
        lucid.gt,
        lucid.ge,
    ]:
        with pytest.raises(NotImplementedError):
            op(a, b)


@pytest.mark.parametrize("dt", _DTYPES)
def test_zero_dim_scalar_tensor_and_full_preserve_shape(
    dt: lucid.dtype, device: str, ref: ModuleType
) -> None:
    if device != "cpu" and dt in _CPU_ONLY:
        pytest.skip(f"{dt} is CPU-only")
    val: bool | int | float | complex = True if dt is lucid.bool_ else 2
    t = lucid.tensor(val, dtype=dt, device=device)
    rt = ref.tensor(val, dtype=_ref_dt(ref, dt))
    assert t.shape == rt.shape == ()
    assert t.ndim == 0
    assert t.dtype == dt

    f = lucid.full((), val, dtype=dt, device=device)
    assert f.shape == ()
    assert f.ndim == 0
    assert f.dtype == dt


@pytest.mark.parametrize(
    ("da", "db"),
    [
        (lucid.float16, lucid.float32),
        (lucid.float32, lucid.float16),
        (lucid.bfloat16, lucid.float32),
        (lucid.float16, lucid.bfloat16),
        (lucid.int32, lucid.float16),
        (lucid.int64, lucid.float32),
    ],
)
def test_composite_mixed_dtype_promotion_matches_reference(
    da: lucid.dtype, db: lucid.dtype, device: str, ref: ModuleType
) -> None:
    a = lucid.tensor([1, 2], dtype=da, device=device)
    b = lucid.tensor([-3, 4], dtype=db, device=device)
    ra = ref.tensor([1, 2], dtype=_ref_dt(ref, da))
    rb = ref.tensor([-3, 4], dtype=_ref_dt(ref, db))

    for fn_l, fn_r in [
        (lucid.kron, ref.kron),
        (lambda x, y: x.kron(y), lambda x, y: x.kron(y)),
        (lucid.copysign, ref.copysign),
        (lucid.fmax, ref.fmax),
        (lucid.fmin, ref.fmin),
    ]:
        got = fn_l(a, b)
        want = fn_r(ra, rb)
        assert _ref_dt(ref, got.dtype) == want.dtype
        assert got.tolist() == pytest.approx(want.tolist(), rel=1e-3, abs=1e-3)


def test_copysign_scalar_and_integral_promotion_matches_reference(
    device: str, ref: ModuleType
) -> None:
    f16 = lucid.tensor([1.0, -2.0], dtype=lucid.float16, device=device)
    rf16 = ref.tensor([1.0, -2.0], dtype=ref.float16)
    for scalar in (-1.0, -1, 2.0):
        got = lucid.copysign(f16, scalar)
        want = ref.copysign(rf16, scalar)
        assert _ref_dt(ref, got.dtype) == want.dtype
        assert got.tolist() == want.tolist()

    i32 = lucid.tensor([1, -2], dtype=lucid.int32, device=device)
    i64 = lucid.tensor([-3, 4], dtype=lucid.int64, device=device)
    ri32 = ref.tensor([1, -2], dtype=ref.int32)
    ri64 = ref.tensor([-3, 4], dtype=ref.int64)
    got_ii = lucid.copysign(i32, i64)
    want_ii = ref.copysign(ri32, ri64)
    assert _ref_dt(ref, got_ii.dtype) == want_ii.dtype
    assert got_ii.tolist() == want_ii.tolist()


def test_addcmul_preserves_integer_dtype_and_addcdiv_refuses_integer_division(
    device: str, ref: ModuleType
) -> None:
    a = lucid.tensor([1, 2], dtype=lucid.int64, device=device)
    b = lucid.tensor([3, 4], dtype=lucid.int64, device=device)
    c = lucid.tensor([5, 6], dtype=lucid.int64, device=device)
    ra = ref.tensor([1, 2], dtype=ref.int64)
    rb = ref.tensor([3, 4], dtype=ref.int64)
    rc = ref.tensor([5, 6], dtype=ref.int64)

    got = lucid.addcmul(a, b, c)
    want = ref.addcmul(ra, rb, rc)
    assert got.dtype == lucid.int64
    assert _ref_dt(ref, got.dtype) == want.dtype
    assert got.tolist() == want.tolist()

    f = lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=device)
    rf = ref.tensor([1.0, 2.0], dtype=ref.float32)
    for x_l, x_r in [(f, rf), (a, ra)]:
        with pytest.raises(RuntimeError, match="Integer division with addcdiv"):
            ref.addcdiv(x_r, rb, rc)
        with pytest.raises(RuntimeError, match="Integer division with addcdiv"):
            lucid.addcdiv(x_l, b, c)
    f_ip = f.clone()
    with pytest.raises(RuntimeError, match="Integer division with addcdiv"):
        f_ip.addcdiv_(b, c)


def test_clamp_preserves_int64_precision_above_2_pow_53(device: str, ref: ModuleType) -> None:
    big = (1 << 53) + 1
    x = lucid.tensor([big - 5, big, big + 5], dtype=lucid.int64, device=device)
    rx = ref.tensor([big - 5, big, big + 5], dtype=ref.int64)

    got = x.clamp(big, big)
    want = rx.clamp(big, big)
    assert got.dtype == lucid.int64
    assert got.tolist() == want.tolist() == [big, big, big]

    x_ip = x.clone()
    x_ip.clamp_(big, big)
    assert x_ip.tolist() == [big, big, big]


def test_bool_subtraction_refused_for_any_bool_operand(device: str, ref: ModuleType) -> None:
    f = lucid.tensor([2.0], device=device)
    i = lucid.tensor([2], device=device)
    b = lucid.tensor([True], device=device)
    rf = ref.tensor([2.0])
    rb = ref.tensor([True])

    for call_r in [lambda: rf - True, lambda: True - rf, lambda: rf - rb, lambda: rb - rf]:
        with pytest.raises(RuntimeError, match="bool"):
            call_r()

    for call_l in [
        lambda: f - True,
        lambda: True - f,
        lambda: f - b,
        lambda: b - f,
        lambda: i - True,
        lambda: b - 1,
        lambda: lucid.sub(f, True),
        lambda: lucid.sub(f, b),
        lambda: f.sub(True),
    ]:
        with pytest.raises(RuntimeError, match="bool"):
            call_l()

    f_ip = f.clone()
    with pytest.raises(RuntimeError, match="bool"):
        f_ip -= True
    assert f_ip.tolist() == [2.0]


def test_lerp_on_integer_tensors_raises_not_implemented_error(device: str, ref: ModuleType) -> None:
    a = lucid.tensor([0, 10], dtype=lucid.int64, device=device)
    b = lucid.tensor([10, 20], dtype=lucid.int64, device=device)
    ra = ref.tensor([0, 10], dtype=ref.int64)
    rb = ref.tensor([10, 20], dtype=ref.int64)

    with pytest.raises(NotImplementedError):
        ref.lerp(ra, rb, 0.5)
    with pytest.raises(NotImplementedError):
        lucid.lerp(a, b, 0.5)
    with pytest.raises(NotImplementedError):
        a.lerp(b, 0.5)
    with pytest.raises(NotImplementedError):
        a.lerp_(b, 0.5)


def test_tensor_clamp_and_clip_stubs_declare_tensor_bounds() -> None:
    from pathlib import Path

    stub = Path(lucid.__file__).resolve().parent / "_tensor" / "tensor.pyi"
    text = stub.read_text(encoding="utf-8")
    assert (
        "def clip(self, min: Tensor | Scalar | None = None, max: Tensor | Scalar | None = None) -> Tensor:"
        in text
    )
    assert (
        "def clamp(self, min: Tensor | Scalar | None = None, max: Tensor | Scalar | None = None) -> Tensor:"
        in text
    )
