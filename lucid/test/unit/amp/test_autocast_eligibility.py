"""Autocast casts only eligible floating dtypes (F16, BF16, F32) (LCD-78).

``SchemaGuard`` previously applied the active ``AmpPolicy`` to every input
dtype indiscriminately, casting ``float64`` down to ``float32`` / ``bfloat16``
and reinterpreting or truncating ``complex64`` / ``complex128`` and integer
inputs into real floats.  Only ``float16``, ``bfloat16``, and ``float32`` are
autocast-eligible; ``float64``, ``complex64``, ``complex128``, integers, and
``bool`` stay in their own dtype (except that integer and boolean inputs to
``real_valued`` / ``ForceFP32`` ops still promote to ``float32``, as outside
autocast).
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.amp import autocast
from lucid.test._fixtures.devices import device_dtype_params, metal_available


def _amp_dtype(device: str) -> lucid.dtype:
    return lucid.float16 if device == "metal" else lucid.bfloat16


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("MPS not available in reference framework")
        return "mps"
    return "cpu"


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def test_reported_float64_and_complex64_reproducers() -> None:
    a = lucid.linspace(1.0, 16.0, 16, dtype=lucid.float64)
    with autocast(device_type="cpu", dtype=lucid.float16):
        prod_a = a * a
    assert prod_a.dtype == lucid.float64
    assert prod_a.tolist() == [float(i * i) for i in range(1, 17)]

    c_vals = [complex(float(i), float(-i)) for i in range(1, 17)]
    c = lucid.tensor(c_vals, dtype=lucid.complex64)
    with autocast(device_type="cpu", dtype=lucid.float16):
        prod_c = c * c
    assert prod_c.dtype == lucid.complex64
    np.testing.assert_allclose(
        prod_c.numpy(), np.asarray([z * z for z in c_vals], dtype=np.complex64)
    )


@pytest.mark.parametrize("amp_dt", [lucid.float16, lucid.bfloat16])
def test_float64_stays_float64_across_policies_and_backward(
    ref: ModuleType, amp_dt: lucid.dtype
) -> None:
    r_amp_dt = _ref_dtype(ref, amp_dt)
    grid = [[float(4 * r + c) * 0.25 + 0.5 for c in range(4)] for r in range(4)]

    x = lucid.tensor(grid, dtype=lucid.float64, requires_grad=True)
    w = lucid.tensor(grid, dtype=lucid.float64, requires_grad=True)
    b = lucid.tensor([0.1, 0.2, 0.3, 0.4], dtype=lucid.float64, requires_grad=True)

    rx = ref.tensor(grid, dtype=ref.float64, requires_grad=True)
    rw = ref.tensor(grid, dtype=ref.float64, requires_grad=True)
    rb = ref.tensor([0.1, 0.2, 0.3, 0.4], dtype=ref.float64, requires_grad=True)

    with autocast(device_type="cpu", dtype=amp_dt):
        # Rows LOWER on the CPU
        y_mm = x @ w
        y_lin = F.linear(x, w, b)
        # No row
        y_mul = x * w
        y_pow = x**2
        # Rows that are FP32 on Metal and KEEP on the CPU
        y_exp = lucid.exp(x)
        y_ln = F.layer_norm(y_lin, (4,), b, b)
        y_sm = F.softmax(y_ln, dim=-1)
        y_sum = y_sm.sum()

    with ref.autocast(device_type="cpu", dtype=r_amp_dt):
        ry_mul = rx * rw
        ry_mm = rx @ rw
        ry_lin = ref.nn.functional.linear(rx, rw, rb)
        ry_pow = rx**2
        ry_exp = ref.exp(rx)
        ry_ln = ref.nn.functional.layer_norm(ry_lin, (4,), rb, rb)
        ry_sm = ref.nn.functional.softmax(ry_ln, dim=-1)
        ry_sum = ry_sm.sum()

    for got, want in [
        (y_mul, ry_mul),
        (y_mm, ry_mm),
        (y_lin, ry_lin),
        (y_pow, ry_pow),
        (y_exp, ry_exp),
        (y_ln, ry_ln),
        (y_sm, ry_sm),
        (y_sum, ry_sum),
    ]:
        assert got.dtype == lucid.float64
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-12, atol=1e-12)

    loss = (y_mm + y_exp + y_ln).sum()
    rloss = (ry_mm + ry_exp + ry_ln).sum()
    loss.backward()
    rloss.backward()
    assert x.grad is not None and x.grad.dtype == lucid.float64
    assert w.grad is not None and w.grad.dtype == lucid.float64
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", lucid.complex64),
        ("cpu", lucid.complex128),
        pytest.param(
            "metal",
            lucid.complex64,
            marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
        ),
    ],
)
def test_complex_stays_complex_under_autocast(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    rdt = _ref_dtype(ref, dtype)
    amp_dt = _amp_dtype(device)
    r_amp_dt = _ref_dtype(ref, amp_dt)

    vals_a = [complex(0.1 * i + 0.5, -0.05 * i + 0.2) for i in range(16)]
    vals_b = [complex(0.2 * i - 0.3, 0.1 * i + 0.1) for i in range(16)]

    a = lucid.tensor(vals_a, dtype=dtype, device=device, requires_grad=True)
    b = lucid.tensor(vals_b, dtype=dtype, device=device, requires_grad=True)
    ra = ref.tensor(vals_a, dtype=rdt, requires_grad=True)
    rb = ref.tensor(vals_b, dtype=rdt, requires_grad=True)

    with autocast(device_type=device, dtype=amp_dt):
        y_add = a + b
        y_sub = a - b
        y_mul = a * b
        y_div = a / b
        y_conj = lucid.conj(y_mul)

    with ref.autocast(device_type="cpu", dtype=r_amp_dt):
        ry_add = ra + rb
        ry_sub = ra - rb
        ry_mul = ra * rb
        ry_div = ra / rb
        ry_conj = ref.conj(ry_mul).resolve_conj()

    for got, want in [
        (y_add, ry_add),
        (y_sub, ry_sub),
        (y_mul, ry_mul),
        (y_div, ry_div),
        (y_conj, ry_conj),
    ]:
        assert got.dtype == dtype
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-5, atol=1e-5)

    lucid.real(y_mul).sum().backward()
    ref.real(ry_mul).sum().backward()
    assert a.grad is not None and a.grad.dtype == dtype
    np.testing.assert_allclose(a.grad.numpy(), ra.grad.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("device", "dtype"),
    device_dtype_params([lucid.int8, lucid.int16, lucid.int32, lucid.int64, lucid.bool]),
)
def test_integer_and_bool_dtypes_under_autocast(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    rdev = _ref_device(ref, device)
    rdt = _ref_dtype(ref, dtype)
    amp_dt = _amp_dtype(device)
    r_amp_dt = _ref_dtype(ref, amp_dt)

    raw = [True, False] * 8 if dtype == lucid.bool else list(range(1, 17))
    x = lucid.tensor(raw, dtype=dtype, device=device).reshape(4, 4)
    rx = ref.tensor(raw, dtype=rdt, device=rdev).reshape(4, 4)

    with autocast(device_type=device, dtype=amp_dt):
        y_exp = lucid.exp(x)
        y_mean = x.mean()
        if dtype != lucid.bool:
            y_add = x + x
            y_mul = x * x
            y_pow = x**2

    with ref.autocast(device_type=rdev, dtype=r_amp_dt):
        ry_exp = ref.exp(rx)
        ry_mean = rx.float().mean()
        if dtype != lucid.bool:
            ry_add = rx + rx
            ry_mul = rx * rx
            ry_pow = rx**2

    assert y_exp.dtype == lucid.float32
    assert y_mean.dtype == lucid.float32
    np.testing.assert_allclose(y_exp.numpy(), ry_exp.cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_mean.numpy(), ry_mean.cpu().numpy(), rtol=1e-5, atol=1e-5)

    if dtype != lucid.bool:
        assert y_add.dtype == dtype and y_add.tolist() == ry_add.cpu().tolist()
        assert y_mul.dtype == dtype and y_mul.tolist() == ry_mul.cpu().tolist()
        assert y_pow.dtype == dtype and y_pow.tolist() == ry_pow.cpu().tolist()
