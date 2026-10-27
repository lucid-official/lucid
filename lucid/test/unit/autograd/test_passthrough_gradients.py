"""Ops that pass a value through unchanged must pass its gradient through too.

``nan_to_num`` is the identity on finite values and ``nextafter(a, b)`` moves
``a`` by one unit in the last place.  Both used to return a detached tensor,
so the gradient simply went missing — no error, and ``nan_to_num(x) + x``
differentiated to 1 instead of 2.  The reference framework's rules, asserted
here by value on both devices: ``nan_to_num`` passes the gradient where the
input is finite and gives zero where a value was replaced; ``nextafter``
passes it to ``a`` and gives ``b`` zero.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]


@pytest.mark.parametrize("device", DEVICES)
def test_nan_to_num_passes_the_gradient_where_the_input_is_finite(device):
    x = lucid.tensor(
        [1.0, float("nan"), float("inf"), -float("inf"), -2.0],
        requires_grad=True,
        device=device,
    )
    weights = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    (lucid.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0) * weights).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1.0, 0.0, 0.0, 0.0, 5.0])


@pytest.mark.parametrize("device", DEVICES)
def test_nan_to_num_adds_to_a_second_path(device):
    x = lucid.tensor([0.5, 3.0], requires_grad=True, device=device)
    (lucid.nan_to_num(x) + x).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2.0, 2.0])


def test_nan_to_num_is_differentiable_twice():
    x = lucid.tensor([1.0, 2.0], requires_grad=True)
    (g,) = lucid.autograd.grad((lucid.nan_to_num(x) ** 2).sum(), [x], create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), [x])
    np.testing.assert_array_equal(g.detach().numpy(), [2.0, 4.0])
    np.testing.assert_array_equal(h.numpy(), [2.0, 2.0])


@pytest.mark.parametrize("device", DEVICES)
def test_nextafter_follows_its_first_argument(device):
    a = lucid.tensor([1.0, 2.0, -3.0], requires_grad=True, device=device)
    b = lucid.tensor([2.0, 0.0, 5.0], requires_grad=True, device=device)
    (lucid.nextafter(a, b) * lucid.tensor([1.0, 2.0, 3.0], device=device)).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(b.grad.numpy(), [0.0, 0.0, 0.0])


def test_nextafter_broadcasts_its_gradient_back():
    a = lucid.tensor([[1.0], [2.0]], requires_grad=True)
    b = lucid.tensor([0.0, 5.0, 9.0], requires_grad=True)
    lucid.nextafter(a, b).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [[3.0], [3.0]])
    np.testing.assert_array_equal(b.grad.numpy(), [0.0, 0.0, 0.0])
