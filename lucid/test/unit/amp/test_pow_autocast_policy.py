"""``pow`` under autocast answers as the reference does, and integer ``pow`` stays exact (LCD-462).

ce8fc7f19 moved ``PowBackward`` from ``ForceFP32`` to ``Promote`` so that an
integer power would keep its integer dtype (LCD-43).  ``Promote`` took the
floating powers with it: under ``autocast(metal, float16)`` ``x ** y`` of two
float32 tensors came back float16 while the reference runs ``pow`` in float32,
and ``x ** 2.0`` (``pow_scalar``) still answered float32, so two spellings of
one power disagreed.  ``pow`` is ``ForceFP32`` again; the schema's
``integral_closed`` flag is what keeps an integer or bool input out of the
policy's int-to-float32 promotion.

Only float32 inputs are held to the reference under CPU autocast: the
reference's CPU autocast list leaves ``pow`` (like ``exp`` and ``softmax``)
alone, so a half input keeps its dtype there, while Lucid's policy table is the
same on both devices.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_BASES = [0.5, 1.25, 2.0, 3.0, 0.75, 1.5, 4.0, 2.5]
_EXPS = [2.0, 0.5, 1.5, -1.0, 3.0, 0.25, -0.5, 1.0]

_metal_only = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")

Spelling = Callable[[object, object, object], object]

# ``(lib, base, exponent) -> result`` for each way to write a tensor power.
_TENSOR_SPELLINGS: dict[str, Spelling] = {
    "dunder": lambda lib, a, b: a**b,
    "method": lambda lib, a, b: a.pow(b),
    "function": lambda lib, a, b: lib.pow(a, b),
    "rpow": lambda lib, a, b: 2.0**a,
}


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("MPS not available in reference framework")
        return "mps"
    return "cpu"


def _amp_dtype(device: str) -> lucid.dtype:
    return lucid.float16 if device == "metal" else lucid.bfloat16


def _assert_matches(got: lucid.Tensor, want: object) -> None:
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    np.testing.assert_allclose(
        got.float().numpy(),
        want.float().cpu().numpy(),
        rtol=2e-6,
        atol=0.0,
    )


def _run_both(
    ref: ModuleType,
    device: str,
    dtype: lucid.dtype,
    spell: Spelling,
) -> tuple[lucid.Tensor, object]:
    rdev = _ref_device(ref, device)
    a = lucid.tensor(_BASES, dtype=dtype, device=device)
    b = lucid.tensor(_EXPS, dtype=dtype, device=device)
    ra = ref.tensor(_BASES, dtype=_ref_dtype(ref, dtype), device=rdev)
    rb = ref.tensor(_EXPS, dtype=_ref_dtype(ref, dtype), device=rdev)
    amp = _amp_dtype(device)
    with autocast(device_type=device, dtype=amp):
        got = spell(lucid, a, b)
    with ref.autocast(device_type=rdev, dtype=_ref_dtype(ref, amp)):
        want = spell(ref, ra, rb)
    return got, want


def test_pow_schema_is_force_fp32() -> None:
    schema = _C_engine.op_lookup("pow")
    assert schema is not None
    assert schema.amp_policy == _C_engine.AmpPolicy.ForceFP32


@_metal_only
@pytest.mark.parametrize("spelling", sorted(_TENSOR_SPELLINGS))
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16], ids=str)
def test_tensor_pow_under_metal_autocast_runs_in_float32(
    ref: ModuleType, dtype: lucid.dtype, spelling: str
) -> None:
    got, want = _run_both(ref, "metal", dtype, _TENSOR_SPELLINGS[spelling])
    assert got.dtype == lucid.float32
    _assert_matches(got, want)


@pytest.mark.parametrize("spelling", sorted(_TENSOR_SPELLINGS))
def test_float32_tensor_pow_under_cpu_autocast_stays_float32(
    ref: ModuleType, spelling: str
) -> None:
    got, want = _run_both(ref, "cpu", lucid.float32, _TENSOR_SPELLINGS[spelling])
    assert got.dtype == lucid.float32
    _assert_matches(got, want)


@pytest.mark.parametrize("exponent", [2, 2.0, 0.5, 3, -1.0, 1.5])
def test_float32_scalar_pow_agrees_with_tensor_pow_under_autocast(
    ref: ModuleType, device: str, exponent: float
) -> None:
    def scalar(lib: object, a: object, b: object) -> object:
        del b  # the exponent is the Python number, not the tensor
        return a**exponent

    got, want = _run_both(ref, device, lucid.float32, scalar)
    assert got.dtype == lucid.float32
    _assert_matches(got, want)

    a = lucid.tensor(_BASES, device=device)
    e = lucid.full((len(_BASES),), float(exponent), device=device)
    with autocast(device_type=device, dtype=_amp_dtype(device)):
        tensor_form = a**e
    assert tensor_form.dtype == got.dtype
    np.testing.assert_allclose(tensor_form.numpy(), got.numpy(), rtol=2e-6)


@_metal_only
@pytest.mark.parametrize("exponent", [2.0, 1.1])
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=str)
def test_half_scalar_pow_under_metal_autocast_runs_in_float32(
    ref: ModuleType, dtype: lucid.dtype, exponent: float
) -> None:
    got, want = _run_both(ref, "metal", dtype, lambda lib, a, b: a**exponent)
    assert got.dtype == lucid.float32
    _assert_matches(got, want)

    x = lucid.tensor([0.5, 1.25, 2.0, 3.0], dtype=dtype, device="metal", requires_grad=True)
    with autocast(device_type="metal", dtype=lucid.float16):
        y_pow = _C_engine.pow_scalar(x._impl, exponent)
        y_rpow = _C_engine.rpow_scalar(2.0, x._impl)
    assert y_pow.dtype == _C_engine.Dtype.F32
    assert y_rpow.dtype == _C_engine.Dtype.F32


@_metal_only
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16], ids=str)
def test_local_response_norm_under_metal_autocast_runs_in_float32(
    ref: ModuleType, dtype: lucid.dtype
) -> None:
    rdev = _ref_device(ref, "metal")
    rdt = _ref_dtype(ref, dtype)
    vals = [[[0.5, 1.25], [2.0, 3.0], [0.75, 1.5], [4.0, 2.5]]]
    x = lucid.tensor(vals, dtype=dtype, device="metal", requires_grad=True)
    rx = ref.tensor(vals, dtype=rdt, device=rdev, requires_grad=True)
    with autocast(device_type="metal", dtype=lucid.float16):
        got = lucid.nn.functional.local_response_norm(x, size=2)
    with ref.autocast(device_type=rdev, dtype=ref.float16):
        want_amp = ref.nn.functional.local_response_norm(rx.detach(), size=2)
    want = ref.nn.functional.local_response_norm(rx.float(), size=2)
    assert got.dtype == lucid.float32
    assert want_amp.dtype == ref.float32
    _assert_matches(got.detach(), want.detach())
    got.sum().backward()
    want.sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype
    np.testing.assert_allclose(
        x.grad.float().numpy(), rx.grad.float().cpu().numpy(), rtol=1e-2, atol=1e-3
    )


@_metal_only
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=str)
def test_half_pow_gradient_under_metal_autocast(ref: ModuleType, dtype: lucid.dtype) -> None:
    rdev = _ref_device(ref, "metal")
    rdt = _ref_dtype(ref, dtype)
    a = lucid.tensor(_BASES, dtype=dtype, device="metal", requires_grad=True)
    b = lucid.tensor(_EXPS, dtype=dtype, device="metal", requires_grad=True)
    ra = ref.tensor(_BASES, dtype=rdt, device=rdev, requires_grad=True)
    rb = ref.tensor(_EXPS, dtype=rdt, device=rdev, requires_grad=True)
    with autocast(device_type="metal", dtype=lucid.float16):
        (a**b).sum().backward()
    with ref.autocast(device_type=rdev, dtype=ref.float16):
        (ra**rb).sum().backward()
    for got, want in ((a.grad, ra.grad), (b.grad, rb.grad)):
        assert got is not None and got.dtype == dtype
        np.testing.assert_allclose(
            got.float().numpy(), want.float().cpu().numpy(), rtol=1e-2, atol=1e-3
        )


_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
@pytest.mark.parametrize("amp", [False, True], ids=["eager", "autocast"])
def test_integer_pow_keeps_its_dtype_and_is_exact(
    device: str, dtype: lucid.dtype, amp: bool
) -> None:
    bits = int(str(dtype).split("int")[-1])
    half = 1 << (bits - 1)
    bases = [0, 1, -1, 2, -2, 3, 5, 7]
    exps = [3, 0, 5, bits - 2, 3, 4, 3, 2]
    want = [((x**e + half) % (1 << bits)) - half for x, e in zip(bases, exps, strict=True)]
    a = lucid.tensor(bases, dtype=dtype, device=device)
    b = lucid.tensor(exps, dtype=dtype, device=device)
    with autocast(device_type=device, dtype=_amp_dtype(device), enabled=amp):
        got = a**b
        got_scalar = a**2
    assert got.dtype == dtype and got.tolist() == want
    assert got_scalar.dtype == dtype
    assert got_scalar.tolist() == [((x * x + half) % (1 << bits)) - half for x in bases]


@pytest.mark.parametrize("amp", [False, True], ids=["eager", "autocast"])
def test_int64_pow_past_float64_mantissa_is_exact(device: str, amp: bool) -> None:
    bases = [3, -3, 7, (1 << 27) + 3]
    exps = [39, 39, 22, 2]
    a = lucid.tensor(bases, dtype=lucid.int64, device=device)
    b = lucid.tensor(exps, dtype=lucid.int64, device=device)
    with autocast(device_type=device, dtype=_amp_dtype(device), enabled=amp):
        got = a**b
    want = [((x**e + (1 << 63)) % (1 << 64)) - (1 << 63) for x, e in zip(bases, exps, strict=True)]
    assert got.dtype == lucid.int64 and got.tolist() == want


@pytest.mark.parametrize("amp", [False, True], ids=["eager", "autocast"])
def test_bool_pow_stays_bool(device: str, amp: bool) -> None:
    x = lucid.tensor([True, True, False, False], device=device)
    y = lucid.tensor([True, False, True, False], device=device)
    with autocast(device_type=device, dtype=_amp_dtype(device), enabled=amp):
        got = x**y
    assert got.dtype == lucid.bool
    assert got.tolist() == [True, True, False, True]
