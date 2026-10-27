"""``trunc`` is a step function of its own, and ``fmod`` / ``frac`` differentiate twice.

``trunc`` was ``where(x >= 0, floor(x), ceil(x))`` over tensors.  Once
``floor`` and ``ceil`` kept the graph with their zero gradient, that put a
``where`` node on the path, and ``where`` refuses a second derivative — so
``fmod`` and ``frac``, which are built on ``trunc``, lost theirs.  It picks
between the two on storages now, with a zero-gradient node of its own.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]
_VALUES = [2.7, -2.7, -0.5, 0.5, 0.0, -0.0, float("inf"), -float("inf"), float("nan")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16], ids=str)
def test_trunc_matches_numpy_down_to_the_sign_of_zero(device: str, dtype: object) -> None:
    got = lucid.trunc(lucid.tensor(_VALUES, device=device).to(dtype)).float().numpy()
    want = np.trunc(np.array(_VALUES, dtype=np.float32))
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(want))


@pytest.mark.parametrize("device", DEVICES)
def test_trunc_leaves_integers_as_they_are(device: str) -> None:
    t = lucid.tensor([5, -5, 0], dtype=lucid.int64, device=device)
    out = lucid.trunc(t)
    assert out.dtype == lucid.int64
    np.testing.assert_array_equal(out.numpy(), [5, -5, 0])


@pytest.mark.parametrize("device", DEVICES)
def test_trunc_stays_in_the_graph_with_a_zero_gradient(device: str) -> None:
    x = lucid.tensor([2.5, -1.7, 3.2], requires_grad=True, device=device)
    y = lucid.trunc(x)
    assert y.requires_grad
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "fn",
    [lambda x: lucid.frac(x), lambda x: lucid.fmod(x, lucid.tensor([1.5, 0.7, 1.1]))],
    ids=["frac", "fmod"],
)
def test_second_derivative_through_trunc(fn: object) -> None:
    x = lucid.tensor([2.5, -1.7, 3.2], requires_grad=True)
    (g,) = lucid.autograd.grad((fn(x) ** 2).sum(), [x], create_graph=True)  # type: ignore[operator]
    (h,) = lucid.autograd.grad(g.sum(), [x])
    # (y**2)'' = 2 y'**2 + 2 y y'' with y' = 1 and y'' = 0.
    np.testing.assert_allclose(h.numpy(), [2.0, 2.0, 2.0])
