"""Backward hooks, and the gradients they are allowed to replace.

``nn/module.py`` sat at 63.7%, and the largest dark region was the
backward-hook machinery: ``register_full_backward_hook``, its legacy and
pre- variants, the global registries, and the two paths that apply them
(the Python ``autograd.Function`` fallback and the C++ ``ModuleHookNode``
barrier).

Firing is the easy half.  The half that matters is that a hook's
*return value* replaces the gradient — that is what a gradient-reversal
layer, a clipping hook or a masking hook is, and a framework that ran
the hook and discarded what it returned would look identical from the
outside while doing nothing.  So every replacement here is checked
against an identical module without the hook, not against a constant.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _v(x):
    return np.asarray(x.numpy())


def _x(shape=(2, 4)):
    """A fresh input each time.

    Not a module-level constant: ``.grad`` accumulates, and a shared
    input makes a hook that correctly zeroes its contribution look like a
    hook that was ignored.  (It did, once, while this file was written.)
    """
    return lucid.tensor(np.ones(shape, dtype=np.float32), requires_grad=True)


def _twins():
    """Two identical Linears — one to hook, one to compare against."""
    hooked, plain = nn.Linear(4, 3), nn.Linear(4, 3)
    plain.load_state_dict(hooked.state_dict())
    return hooked, plain


def _grads(hooked, plain):
    a, b = _x(), _x()
    hooked(a).sum().backward()
    plain(b).sum().backward()
    return _v(a.grad), _v(b.grad)


# ── firing ────────────────────────────────────────────────────────────────────


def test_a_full_backward_hook_fires():
    seen = []
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: seen.append(mod))
    layer(_x()).sum().backward()
    assert seen == [layer]


def test_the_handle_removes_the_hook():
    seen = []
    layer = nn.Linear(4, 3)
    handle = layer.register_full_backward_hook(lambda mod, gi, go: seen.append(1))
    handle.remove()
    layer(_x()).sum().backward()
    assert not seen


def test_hooks_fire_in_registration_order():
    order = []
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: order.append("first"))
    layer.register_full_backward_hook(lambda mod, gi, go: order.append("second"))
    layer(_x()).sum().backward()
    assert order == ["first", "second"]


def test_the_legacy_backward_hook_fires():
    seen = []
    layer = nn.Linear(4, 3)
    layer.register_backward_hook(lambda mod, gi, go: seen.append(1))
    layer(_x()).sum().backward()
    assert seen


def test_a_backward_pre_hook_fires():
    seen = []
    layer = nn.Linear(4, 3)
    layer.register_full_backward_pre_hook(lambda mod, go: seen.append(1))
    layer(_x()).sum().backward()
    assert seen


def test_a_hook_on_a_nested_child_fires():
    seen = []
    inner = nn.Linear(4, 3)
    inner.register_full_backward_hook(lambda mod, gi, go: seen.append(1))
    nn.Sequential(inner, nn.ReLU())(_x()).sum().backward()
    assert seen


def test_a_hook_fires_on_a_convolution():
    seen = []
    layer = nn.Conv2d(3, 4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: seen.append(1))
    layer(_x((1, 3, 8, 8))).sum().backward()
    assert seen


def test_a_global_backward_hook_fires_for_every_module():
    seen = []
    handle = nn.register_module_full_backward_hook(
        lambda mod, gi, go: seen.append(type(mod).__name__)
    )
    try:
        nn.Sequential(nn.Linear(4, 3), nn.ReLU())(_x()).sum().backward()
    finally:
        handle.remove()
    assert "Linear" in seen


# ── what the hook is handed ───────────────────────────────────────────────────


def test_grad_output_is_the_derivative_of_the_loss_by_the_output():
    seen = {}
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: seen.update(go=go))
    layer(_x()).sum().backward()
    assert np.allclose(_v(seen["go"][0]), 1.0)  # d(sum)/d(out)


def test_grad_input_is_the_derivative_of_the_loss_by_the_input():
    seen = {}
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: seen.update(gi=gi))
    layer(_x()).sum().backward()
    assert np.allclose(_v(seen["gi"][0]), _v(layer.weight).sum(axis=0), atol=1e-5)


# ── replacing the gradient ────────────────────────────────────────────────────


def test_a_returned_gradient_replaces_the_real_one():
    """Everything below rests on this.  A framework that fired the hook
    and dropped its return value would pass every test above."""
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(lucid.zeros_like(g) for g in gi))
    got, _ = _grads(hooked, plain)
    assert np.allclose(got, 0.0)


def test_gradient_reversal_negates_exactly():
    """The domain-adaptation layer, and the reason the return value
    exists at all."""
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(-g for g in gi))
    got, want = _grads(hooked, plain)
    assert np.allclose(got, -want, atol=1e-6)
    assert not np.allclose(got, want)  # the two really do differ


@pytest.mark.parametrize("factor", [0.5, 3.0, -2.0])
def test_a_scaling_hook_scales_exactly(factor):
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(g * factor for g in gi))
    got, want = _grads(hooked, plain)
    assert np.allclose(got, factor * want, atol=1e-6)


def test_returning_none_leaves_the_gradient_alone():
    """The common case — a hook that only observes."""
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: None)
    got, want = _grads(hooked, plain)
    assert np.allclose(got, want, atol=1e-6)


def test_two_hooks_compose():
    """The second has to see the first's output, not the original — the
    difference between 10x and 5x, and invisible without a comparison."""
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(g * 2.0 for g in gi))
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(g * 5.0 for g in gi))
    got, want = _grads(hooked, plain)
    assert np.allclose(got, 10.0 * want, atol=1e-6)


def test_a_pre_hooks_returned_grad_output_propagates():
    hooked, plain = _twins()
    hooked.register_full_backward_pre_hook(lambda mod, go: tuple(g * 4.0 for g in go))
    got, want = _grads(hooked, plain)
    assert np.allclose(got, 4.0 * want, atol=1e-6)


def test_replacing_grad_input_leaves_the_parameter_gradients_alone():
    """``grad_input`` is what flows *further back*.  The module's own
    weight gradient was already computed from ``grad_output`` and must
    not move — otherwise a masking hook silently stops the layer it is
    attached to from learning."""
    hooked, plain = _twins()
    hooked.register_full_backward_hook(lambda mod, gi, go: tuple(g * 0 for g in gi))
    hooked(_x()).sum().backward()
    plain(_x()).sum().backward()
    assert np.allclose(_v(hooked.weight.grad), _v(plain.weight.grad), atol=1e-6)


def test_a_replacement_reaches_layers_further_back():
    """The point of replacing ``grad_input``: it is what the layer before
    receives."""
    first = nn.Linear(4, 4)
    second = nn.Linear(4, 3)
    second.register_full_backward_hook(lambda mod, gi, go: tuple(lucid.zeros_like(g) for g in gi))
    nn.Sequential(first, second)(_x()).sum().backward()
    assert np.allclose(_v(first.weight.grad), 0.0)


# ── interaction with the rest ─────────────────────────────────────────────────


def test_a_module_without_hooks_is_unaffected():
    """The hook machinery wraps the inputs when hooks exist; when they do
    not it must be exactly the plain path."""
    hooked, plain = _twins()
    handle = hooked.register_full_backward_hook(lambda mod, gi, go: None)
    handle.remove()
    got, want = _grads(hooked, plain)
    assert np.allclose(got, want, atol=1e-7)


def test_hooks_survive_a_second_backward():
    counted = []
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: counted.append(1))
    for _ in range(3):
        layer(_x()).sum().backward()
    assert len(counted) == 3


def test_an_input_that_does_not_require_grad_still_runs_the_hook():
    """The no-input-gradient path — a different branch, and one where an
    early return would skip the hook entirely."""
    seen = []
    layer = nn.Linear(4, 3)
    layer.register_full_backward_hook(lambda mod, gi, go: seen.append(1))
    frozen = lucid.tensor(np.ones((2, 4), dtype=np.float32))
    layer(frozen).sum().backward()
    assert seen
