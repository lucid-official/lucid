"""A dtype-cast module must still receive gradients.

Found 2026-07-26 while trying to build a float64 reference for a gradient
audit — the reference itself produced no gradients.

``Module._apply`` stored ``fn(param)._impl`` directly.  A real dtype cast is a
**differentiable op**, so its output is a NON-LEAF; the parameter ended up
hanging off a cast node.  Backward then ran without error, flowed *through* the
parameter to the discarded pre-cast tensor, and accumulated nothing: ``.grad``
stayed ``None`` and training a ``.double()`` model was a silent no-op.

``.to(float32)`` on an already-float32 module hid it — that path is a no-op and
returns the original leaf, so only an actual dtype *change* was affected.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._dispatch import _unwrap

DTYPES = [("float32", lucid.float32, np.float32), ("float64", lucid.float64, np.float64)]


def test_complex128_module_cast_preserves_integer_buffers_and_parameter_identity():
    module = nn.Module()
    module.register_parameter(
        "weight", nn.Parameter(lucid.tensor([1 + 2j], dtype=lucid.complex128))
    )
    module.register_buffer("counter", lucid.tensor([3], dtype=lucid.int64))
    module.register_buffer("phase", lucid.tensor([2 + 3j], dtype=lucid.complex128))
    weight = module.weight
    module.to(lucid.complex64)
    assert module.weight is weight
    assert module.weight.dtype == lucid.complex64
    assert module.phase.dtype == lucid.complex64
    assert module.counter.dtype == lucid.int64
    module.to(lucid.complex128)
    assert module.weight is weight
    assert module.counter.dtype == lucid.int64
    assert module.phase.dtype == lucid.complex128
    assert module.weight.is_leaf
    np.testing.assert_array_equal(module.weight.numpy(), [1 + 2j])


def _train_step(module, dt, npdt, device="cpu"):
    x = lucid.tensor(np.ones((2, 4), dtype=npdt), device=device, dtype=dt)
    out = module(x)
    (out * lucid.ones_like(out)).sum().backward()
    return out


@pytest.mark.parametrize("name,dt,npdt", DTYPES)
def test_cast_parameter_stays_a_leaf(name, dt, npdt):
    module = nn.Linear(4, 3).to(dt)
    weight = module.weight
    assert _unwrap(weight).is_leaf, f"{name}: parameter is no longer a leaf"
    assert weight.requires_grad
    assert weight.dtype == dt


@pytest.mark.parametrize("name,dt,npdt", DTYPES)
def test_cast_module_receives_gradients(name, dt, npdt):
    module = nn.Linear(4, 3).to(dt)
    _train_step(module, dt, npdt)
    assert module.weight.grad is not None, f"{name}: weight got no gradient"
    assert module.bias.grad is not None, f"{name}: bias got no gradient"
    assert module.weight.grad.dtype == dt


def test_float64_gradient_matches_float32():
    """The f64 path must agree with f32, not merely be non-None."""
    lucid.manual_seed(0)
    m32 = nn.Linear(4, 3)
    w = m32.weight.numpy().copy()
    b = m32.bias.numpy().copy()
    _train_step(m32, lucid.float32, np.float32)

    lucid.manual_seed(0)
    m64 = nn.Linear(4, 3).to(lucid.float64)
    assert np.allclose(m64.weight.numpy(), w)
    assert np.allclose(m64.bias.numpy(), b)
    _train_step(m64, lucid.float64, np.float64)

    assert np.abs(m32.weight.grad.numpy() - m64.weight.grad.numpy()).max() < 1e-5
    assert np.abs(m32.bias.grad.numpy() - m64.bias.grad.numpy()).max() < 1e-5


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_device_move_keeps_gradients(device):
    """``.to(device)`` must not regress the same way."""
    module = nn.Linear(4, 3).to(device)
    assert _unwrap(module.weight).is_leaf
    _train_step(module, lucid.float32, np.float32, device=device)
    assert module.weight.grad is not None
    assert str(module.weight.device) == f"device('{device}')"


def test_chained_device_then_dtype_cast():
    module = nn.Linear(4, 3).to("metal").to(lucid.float32)
    assert _unwrap(module.weight).is_leaf
    _train_step(module, lucid.float32, np.float32, device="metal")
    assert module.weight.grad is not None


def test_frozen_parameter_stays_frozen_through_a_cast():
    """requires_grad=False must survive; the fix restores the original flag."""
    module = nn.Linear(4, 3)
    module.weight.requires_grad = False
    module = module.to(lucid.float64)
    assert module.weight.requires_grad is False
    assert module.bias.requires_grad is True
    _train_step(module, lucid.float64, np.float64)
    assert module.weight.grad is None
    assert module.bias.grad is not None


def test_optimizer_actually_updates_a_float64_model():
    """End of the chain: a .double() model must train, not silently no-op."""
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    model = model.to(lucid.float64)
    before = model[0].weight.numpy().copy()
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.1)
    x = lucid.tensor(np.random.default_rng(0).standard_normal((16, 4)), dtype=lucid.float64)
    y = lucid.tensor(np.random.default_rng(1).standard_normal((16, 1)), dtype=lucid.float64)
    for _ in range(5):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = model[0].weight.numpy()
    assert np.abs(after - before).max() > 1e-6, "float64 training did not move the weights"


@pytest.mark.parametrize(
    "cast_fn,expected_dtype",
    [
        (lambda m, _lib: m.double(), lucid.float64),
        (lambda m, _lib: m.half(), lucid.float16),
        (lambda m, _lib: m.bfloat16(), lucid.bfloat16),
        (lambda m, lib: m.to(dtype=lib.float64), lucid.float64),
    ],
)
def test_dtype_cast_preserves_and_converts_existing_grad(ref, cast_fn, expected_dtype) -> None:
    """LCD-74 (#146): Module.to() / .double() / .half() / .bfloat16() keep param.grad."""
    m = nn.Linear(2, 1)
    rm = ref.nn.Linear(2, 1)
    with ref.no_grad():
        rm.weight.copy_(ref.tensor(m.weight.numpy()))
        rm.bias.copy_(ref.tensor(m.bias.numpy()))

    m(lucid.ones(1, 2)).sum().backward()
    rm(ref.ones(1, 2)).sum().backward()

    cast_fn(m, lucid)
    cast_fn(rm, ref)

    for p, rp in zip(m.parameters(), rm.parameters(), strict=True):
        assert p.grad is not None
        assert p.grad.dtype == expected_dtype
        assert p.grad.is_leaf
        np.testing.assert_allclose(
            p.grad.to(lucid.float32).numpy(), rp.grad.to(ref.float32).numpy(), atol=1e-3
        )


def test_device_move_preserves_existing_grad() -> None:
    """LCD-74 (#146): Moving a module across CPU and Metal keeps param.grad."""
    m = nn.Linear(3, 2)
    m(lucid.ones(2, 3)).sum().backward()
    w_grad_cpu = m.weight.grad.numpy().copy()
    b_grad_cpu = m.bias.grad.numpy().copy()

    m.to("metal")
    assert m.weight.grad is not None and m.bias.grad is not None
    assert str(m.weight.grad.device) == "device('metal')"
    assert str(m.bias.grad.device) == "device('metal')"
    np.testing.assert_allclose(m.weight.grad.cpu().numpy(), w_grad_cpu, atol=1e-6)
    np.testing.assert_allclose(m.bias.grad.cpu().numpy(), b_grad_cpu, atol=1e-6)

    m.to("cpu")
    assert m.weight.grad is not None and m.bias.grad is not None
    assert str(m.weight.grad.device) == "device('cpu')"
    np.testing.assert_allclose(m.weight.grad.numpy(), w_grad_cpu, atol=1e-6)
