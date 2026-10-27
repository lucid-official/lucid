"""CPU bfloat16 ops that compute in float32 come back as bfloat16.

The CPU has no 16-bit float kernels, so those ops widen to float32, compute
and narrow back.  Six of them narrowed with a helper whose target dtype
defaulted to float16, and wrote float16 bits under a bfloat16 label:
``linear`` was off by 1.8e5, ``pow_scalar`` / ``rpow_scalar`` returned
nonsense, and through them the gradient of ``x ** 3`` was 128 times too
small.  float16 was right, which is why nothing noticed.  The target is now
a required argument.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap

_HALVES = [lucid.float16, lucid.bfloat16]


def _close(got: lucid.Tensor, ref: lucid.Tensor, dtype: lucid.dtype) -> None:
    assert got.dtype == dtype
    tol = 2e-2 if dtype == lucid.bfloat16 else 2e-3
    np.testing.assert_allclose(got.float().numpy(), ref.float().numpy(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _HALVES)
def test_linear(dtype: lucid.dtype) -> None:
    lucid.manual_seed(0)
    x, w, b = lucid.randn(4, 8), lucid.randn(3, 8), lucid.randn(3)
    ref = F.linear(x.to(dtype).float(), w.to(dtype).float(), b.to(dtype).float())
    _close(F.linear(x.to(dtype), w.to(dtype), b.to(dtype)), ref, dtype)


@pytest.mark.parametrize("dtype", _HALVES)
def test_scalar_kernels(dtype: lucid.dtype) -> None:
    v = lucid.arange(1.0, 5.0).to(dtype)
    f = v.float()
    _close(_wrap(_C_engine.pow_scalar(v._impl, 3.0)), f**3, dtype)
    _close(_wrap(_C_engine.rpow_scalar(2.0, v._impl)), 2.0**f, dtype)
    _close(v**2, f**2, dtype)


@pytest.mark.parametrize("dtype", _HALVES)
def test_gradients_through_scalar_kernels(dtype: lucid.dtype) -> None:
    q = lucid.arange(1.0, 5.0).to(dtype).requires_grad_()
    (q**3).sum().backward()
    assert q.grad.dtype == dtype
    assert q.grad.float().tolist() == [3.0, 12.0, 27.0, 48.0]
    p = lucid.arange(1.0, 7.0).reshape(2, 3).to(dtype).requires_grad_()
    (p.sum(dim=1) * lucid.tensor([2.0, 3.0]).to(dtype)).sum().backward()
    assert p.grad.float().tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]


@pytest.mark.parametrize("dtype", _HALVES)
def test_linear_layer_trains(dtype: lucid.dtype) -> None:
    lucid.manual_seed(0)
    layer = lucid.nn.Linear(8, 1).to(dtype)
    x, y = lucid.randn(32, 8).to(dtype), lucid.randn(32, 1).to(dtype)
    first = None
    for _ in range(10):
        layer.zero_grad()
        loss = ((layer(x) - y) ** 2).mean()
        loss.backward()
        first = first if first is not None else loss.item()
        with lucid.no_grad():
            for p in layer.parameters():
                p -= 0.05 * p.grad
    assert loss.item() < first


_DEVICES = [
    "cpu",
    pytest.param(
        "metal", marks=pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
def test_floor_division_keeps_bfloat16_and_its_nan(device: str) -> None:
    # Left off the floating list, bfloat16 took the integer path: int64 out,
    # and the NaN read back as 0.
    x = lucid.tensor([7.0, 2.5, float("nan"), -3.0]).to(lucid.bfloat16).to(device)
    out = x // 2.0
    assert out.dtype == lucid.bfloat16
    got = out.float().tolist()
    assert got[0] == 3.0 and got[1] == 1.0 and got[3] == -2.0
    assert got[2] != got[2]


@pytest.mark.parametrize("device", _DEVICES)
def test_bfloat16_argmax_points_at_the_first_nan(device: str) -> None:
    # Metal's NaN rule listed float16 but not bfloat16.
    y = lucid.tensor([1.0, float("nan"), 5.0, float("nan")]).to(lucid.bfloat16)
    y = y.to(device)
    assert y.argmax().item() == 1 and y.argmin().item() == 1
