"""Integer ``pow`` stays in its integer dtype and computes exact modular powers (LCD-43).

``PowBackward::schema_v1`` used to carry ``AmpPolicy::ForceFP32``, which caused
``SchemaGuard`` to promote every integer input to ``float32`` before dispatch
(so ``lucid.tensor([9], dtype=lucid.int64) ** 2`` returned ``float32 [81.0000076]``),
and ``CpuBackend::pow`` used ``static_cast<int64_t>(std::pow(a, b))``, losing
precision above $2^{53}$.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_BITS = {lucid.int8: 8, lucid.int16: 16, lucid.int32: 32, lucid.int64: 64}


def _wrap_signed(val: int, bits: int) -> int:
    half = 1 << (bits - 1)
    return ((val + half) % (1 << bits)) - half


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("MPS not available in reference framework")
        return "mps"
    return "cpu"


def test_reported_int64_pow_is_exact(device: str) -> None:
    x = lucid.tensor([9], dtype=lucid.int64, device=device)
    out = x**2
    assert out.dtype == lucid.int64
    assert out.tolist() == [81]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_integer_pow_tensor_and_scalar_spellings_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    bits = _BITS[dtype]
    rdev = _ref_device(ref, device)
    rdt = _ref_dtype(ref, dtype)

    bases = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 9]
    exps = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 2, 3, 2, 3, 2]
    want = [_wrap_signed(b**e, bits) for b, e in zip(bases, exps, strict=True)]

    t = lucid.tensor(bases, dtype=dtype, device=device)
    e = lucid.tensor(exps, dtype=dtype, device=device)
    rt = ref.tensor(bases, dtype=rdt, device=rdev)
    re = ref.tensor(exps, dtype=rdt, device=rdev)

    for fn_l, fn_r in [
        (lambda a, b: a**b, lambda a, b: a**b),
        (lambda a, b: a.pow(b), lambda a, b: a.pow(b)),
        (lucid.pow, ref.pow),
    ]:
        got_t = fn_l(t, e)
        ref_t = fn_r(rt, re)
        assert got_t.dtype == dtype
        assert got_t.tolist() == ref_t.cpu().tolist() == want

    for scalar_exp in (0, 1, 2, 3):
        want_s = [_wrap_signed(b**scalar_exp, bits) for b in bases]
        for spell in (
            lambda a, s=scalar_exp: a**s,
            lambda a, s=scalar_exp: a.pow(s),
            lambda a, s=scalar_exp: lucid.pow(a, s),
        ):
            got_s = spell(t)
            ref_s = rt**scalar_exp
            assert got_s.dtype == dtype
            assert got_s.tolist() == ref_s.cpu().tolist() == want_s

        ip1 = t.clone()
        ip1 **= scalar_exp
        ip2 = t.clone()
        ip2.pow_(scalar_exp)
        assert ip1.dtype == dtype and ip1.tolist() == want_s
        assert ip2.dtype == dtype and ip2.tolist() == want_s

    rpow_got = 2**e
    rpow_ref = 2**re
    assert rpow_got.dtype == dtype
    assert rpow_got.tolist() == rpow_ref.cpu().tolist()


def test_int64_pow_above_float64_mantissa_is_exact(ref: ModuleType, device: str) -> None:
    """Powers above $2^{53}$ lose low bits if routed through ``double``."""
    rdev = _ref_device(ref, device)
    bases = [
        3,
        5,
        7,
        9,
        11,
        13,
        (1 << 26) + 1,
        (1 << 27) + 3,
        (1 << 30) + 5,
        -3,
        -5,
        -7,
        2,
        -2,
        15,
        17,
    ]
    exps = [35, 25, 20, 19, 17, 16, 2, 2, 2, 35, 25, 21, 62, 62, 15, 14]
    want = [_wrap_signed(b**e, 64) for b, e in zip(bases, exps, strict=True)]

    t = lucid.tensor(bases, dtype=lucid.int64, device=device)
    e = lucid.tensor(exps, dtype=lucid.int64, device=device)
    rt = ref.tensor(bases, dtype=ref.int64, device=rdev)
    re = ref.tensor(exps, dtype=ref.int64, device=rdev)

    got = t**e
    assert got.dtype == lucid.int64
    assert got.tolist() == (rt**re).cpu().tolist() == want

    squares = t**2
    want_sq = [_wrap_signed(b**2, 64) for b in bases]
    assert squares.dtype == lucid.int64
    assert squares.tolist() == (rt**2).cpu().tolist() == want_sq


def test_bool_pow_promotion_and_values(ref: ModuleType, device: str) -> None:
    rdev = _ref_device(ref, device)
    vals = [True, False, True, True, False, False, True, False] * 2
    b = lucid.tensor(vals, dtype=lucid.bool, device=device)
    rb = ref.tensor(vals, dtype=ref.bool, device=rdev)

    for exp in (0, 1, 2, 3):
        got = b**exp
        want = rb**exp
        assert got.dtype == lucid.int64
        assert got.tolist() == want.cpu().tolist()

    for flag in (True, False):
        got_b = b**flag
        want_b = rb**flag
        assert got_b.dtype == lucid.bool
        assert got_b.tolist() == want_b.cpu().tolist()

    b2_vals = [True, True, False, False, True, False, True, False] * 2
    b2 = lucid.tensor(b2_vals, dtype=lucid.bool, device=device)
    got_bb = b**b2
    assert got_bb.dtype == lucid.bool
    assert got_bb.tolist() == [bool(x**y) for x, y in zip(vals, b2_vals, strict=True)]


@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_negative_integer_exponent_raises_on_cpu(ref: ModuleType, dtype: lucid.dtype) -> None:
    rdt = _ref_dtype(ref, dtype)
    rt = ref.arange(1, 17, dtype=rdt)
    with pytest.raises(RuntimeError, match="Integers to negative integer powers"):
        _ = rt**-1

    t = lucid.arange(1, 17, dtype=dtype, device="cpu")
    neg_t = lucid.full((16,), -1, dtype=dtype, device="cpu")
    calls: list[Callable[[], object]] = [
        lambda: t**-1,
        lambda: t.pow(-1),
        lambda: lucid.pow(t, -1),
        lambda: t.clone().__ipow__(-1),
        lambda: t.clone().pow_(-1),
        lambda: t**neg_t,
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="Integers to negative integer powers"):
            call()


@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_negative_integer_exponent_on_metal_is_unsynced(dtype: lucid.dtype) -> None:
    if not metal_available():
        pytest.skip("Metal unavailable")
    bases = [1, -1, 2, -2, 3, -3, 5, 9] * 2
    exps = [-1, -2, -1, -3, -2, -1, 2, 2] * 2
    a = lucid.tensor(bases, dtype=dtype, device="metal")
    b = lucid.tensor(exps, dtype=dtype, device="metal")
    lucid.eval(a, b)
    before = _C_engine.host_sync_count()
    out = a**b
    lucid.eval(out)
    assert _C_engine.host_sync_count() == before
    assert out.dtype == dtype
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 25, 81] * 2


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_integer_pow_inside_autocast_keeps_dtype(device: str, dtype: lucid.dtype) -> None:
    amp_dt = lucid.float16 if device == "metal" else lucid.bfloat16
    x = lucid.arange(1, 17, dtype=dtype, device=device)
    e = lucid.full((16,), 2, dtype=dtype, device=device)
    with autocast(device_type=device, dtype=amp_dt):
        y1 = x**2
        y2 = x**e
    want = [_wrap_signed(v * v, _BITS[dtype]) for v in range(1, 17)]
    assert y1.dtype == dtype and y1.tolist() == want
    assert y2.dtype == dtype and y2.tolist() == want
