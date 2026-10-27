"""Activation gradients at NaN, signed zero, infinity and the bounds (LCD-428).

A piecewise backward picks its branch by a comparison on the input, and a
NaN fails every comparison, so which side a NaN lands on is decided by the
comparison the backward makes.  The reference answers with its vectorised
kernels on any real input and with a scalar loop on a tensor too short to
fill a vector, and the two disagree at NaN (relu6: 0 against 1, elu: NaN
against 1).  The oracle is the vectorised path — what training hits — so
the reference's gradient is read off a tensor of ``_WIDTH`` elements.
Lucid's relu dropped a NaN's gradient where both paths pass it, leaky_relu
gave 1 at 0 where the reference gives the slope, and selu and elu with
alpha != 1 took the positive side at 0.  Each activation is held to the
reference through ``backward()`` and ``grad(create_graph=True)`` on both
devices.
"""

import math

import pytest

import lucid
import lucid.nn.functional as F

_VALUES = [
    float("nan"),
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    1.0,
    -1.0,
    3.0,
    -3.0,
    6.0,
    0.5,
    -0.5,
]

# name -> call(F-like module, x); the reference's functional module and
# Lucid's take the same arguments for every entry.
_ACTIVATIONS = {
    "relu": lambda f, x: f.relu(x),
    "relu6": lambda f, x: f.relu6(x),
    "leaky_relu": lambda f, x: f.leaky_relu(x, 0.1),
    "elu": lambda f, x: f.elu(x),
    "elu_alpha": lambda f, x: f.elu(x, 0.5),
    "selu": lambda f, x: f.selu(x),
    "hardswish": lambda f, x: f.hardswish(x),
    "hardsigmoid": lambda f, x: f.hardsigmoid(x),
    "threshold": lambda f, x: f.threshold(x, 0.5, -2.0),
    "softplus": lambda f, x: f.softplus(x),
    "silu": lambda f, x: f.silu(x),
    "gelu": lambda f, x: f.gelu(x),
    "mish": lambda f, x: f.mish(x),
    "hardtanh": lambda f, x: f.hardtanh(x, -1.0, 1.0),
    "celu": lambda f, x: f.celu(x),
    "hardshrink": lambda f, x: f.hardshrink(x, 0.5),
    "softshrink": lambda f, x: f.softshrink(x, 0.5),
}

# name -> call(library, x) for the tensor-level ops.
_TENSOR_OPS = {
    "clamp": lambda lib, x: lib.clamp(x, -1.0, 1.0),
    "clamp_min": lambda lib, x: lib.clamp(x, min=0.0),
    "clamp_max": lambda lib, x: lib.clamp(x, max=0.0),
    "maximum": lambda lib, x: lib.maximum(x, lib.zeros_like(x)),
    "minimum": lambda lib, x: lib.minimum(x, lib.zeros_like(x)),
}


def _ids(v):
    return str(v)


def _same(got, want):
    if math.isnan(want):
        return math.isnan(got)
    return got == pytest.approx(want, rel=1e-5, abs=1e-6)


# Wide enough that the reference runs its vectorised kernel, not the
# scalar loop it keeps for the tail of a short tensor.
_WIDTH = 32


def _reference_fwd_and_grad(ref, call, value, dtype):
    x = ref.tensor([value] * _WIDTH, dtype=dtype, requires_grad=True)
    y = call(x)
    y.sum().backward()
    return float(y[0].item()), float(x.grad[0].item())


def _lucid_fwd_and_grads(call, value, device, dtype):
    x = lucid.tensor([value] * _WIDTH, dtype=dtype, device=device, requires_grad=True)
    y = call(x)
    y.sum().backward()
    eager = float(x.grad[0].item())
    x2 = lucid.tensor([value] * _WIDTH, dtype=dtype, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(call(x2).sum(), x2, create_graph=True)
    return float(y[0].item()), eager, float(g[0].item())


def _check(ref, device, dtype_name, value, lucid_call, ref_call):
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal has no float64")
    want_fwd, want = _reference_fwd_and_grad(ref, ref_call, value, getattr(ref, dtype_name))
    got_fwd, eager, graph = _lucid_fwd_and_grads(
        lucid_call, value, device, getattr(lucid, dtype_name)
    )
    assert _same(got_fwd, want_fwd), f"forward: {got_fwd} != reference {want_fwd}"
    assert _same(eager, want), f"backward(): {eager} != reference {want}"
    assert _same(graph, want), f"grad(create_graph=True): {graph} != reference {want}"


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("value", _VALUES, ids=_ids)
@pytest.mark.parametrize("name", list(_ACTIVATIONS))
def test_activation_gradient_matches_reference(ref, device, name, value, dtype_name):
    call = _ACTIVATIONS[name]
    _check(
        ref,
        device,
        dtype_name,
        value,
        lambda x: call(F, x),
        lambda x: call(ref.nn.functional, x),
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("value", _VALUES, ids=_ids)
@pytest.mark.parametrize("name", list(_TENSOR_OPS))
def test_tensor_op_gradient_matches_reference(ref, device, name, value, dtype_name):
    call = _TENSOR_OPS[name]
    _check(
        ref,
        device,
        dtype_name,
        value,
        lambda x: call(lucid, x),
        lambda x: call(ref, x),
    )
