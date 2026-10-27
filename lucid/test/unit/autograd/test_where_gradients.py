"""``where`` routes its gradient right, to a broadcast operand and a second time.

Two defects, both silent.

On Metal MLX broadcasts ``where``'s operands natively, and the gradient of a
smaller operand was handed over at the output's shape: a ``(1, 3)`` operand
of a ``(2, 3)`` result read the first row of it, not the column sums.

And ``where`` refused ``create_graph``, because its graph-mode backward had
once made ``cdist``'s second derivative wrong.  The fault was upstream:
``where`` keeps no reference to its operands, so ``sqrt``'s output was
dropped, and ``sqrt``'s graph-mode backward rebuilt it as a leaf — its
formula ``g / 2y`` then treated ``y`` as a constant.  The rebuild keeps the
output's history now, and ``where`` differentiates twice.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]


def _mask(device: str) -> lucid.Tensor:
    return (lucid.arange(6).reshape(2, 3) % 2 == 0).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("x_shape", "want"),
    [((1, 3), [[1.0, 1.0, 1.0]]), ((3,), [1.0, 1.0, 1.0]), ((), 3.0)],
    ids=["row", "vector", "scalar"],
)
def test_a_broadcast_operand_receives_the_summed_gradient(
    device: str, x_shape: tuple, want: object
) -> None:
    x = (lucid.ones(*x_shape) if x_shape else lucid.tensor(1.0)).to(device)
    x.requires_grad_()
    y = lucid.zeros(2, 3, device=device, requires_grad=True)
    lucid.where(_mask(device), x, y).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), want)
    np.testing.assert_array_equal(y.grad.numpy(), [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
def test_the_other_operand_broadcast_too(device: str) -> None:
    x = lucid.zeros(2, 3, device=device, requires_grad=True)
    y = lucid.ones(1, 3, device=device, requires_grad=True)
    lucid.where(_mask(device), x, y).sum().backward()
    np.testing.assert_array_equal(y.grad.numpy(), [[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("op", "second"),
    [
        (lucid.sqrt, lambda v: -0.25 * v**-1.5),
        (lucid.exp, np.exp),
        (lucid.tanh, lambda v: -2.0 * np.tanh(v) * (1.0 - np.tanh(v) ** 2)),
    ],
    ids=["sqrt", "exp", "tanh"],
)
def test_an_op_read_back_from_its_output_differentiates_twice_under_where(
    device: str, op: object, second: object
) -> None:
    """Nothing holds ``op(x)`` but ``where``, which keeps no reference to it."""
    values = np.array([0.5, 1.5, 3.0], dtype=np.float32)
    x = lucid.tensor(values, requires_grad=True, device=device)
    y = lucid.where(x > 100.0, lucid.zeros_like(x), op(x)).sum()  # type: ignore[operator]
    (g,) = lucid.autograd.grad(y, [x], create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), [x])
    # One ulp of Metal's tanh near 1 is 1e-5 relative in 1 - tanh², so the
    # tolerance is float32's, not the formula's.
    np.testing.assert_allclose(h.numpy(), second(values), rtol=1e-4, atol=1e-6)  # type: ignore[operator]


def _explicit_cdist(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    return (((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(dim=-1)).sqrt()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wrt", [0, 1], ids=["x1", "x2"])
def test_cdist_second_derivative_matches_the_explicit_formula(device: str, wrt: int) -> None:
    rng = np.random.default_rng(0)
    arrays = [
        rng.standard_normal((3, 2)).astype(np.float32),
        rng.standard_normal((4, 2)).astype(np.float32),
    ]
    v = rng.standard_normal((3, 4)).astype(np.float32)
    w = rng.standard_normal(arrays[wrt].shape).astype(np.float32)

    def hessian_vector(fn: object) -> np.ndarray:
        leaves = [lucid.tensor(a, requires_grad=True, device=device) for a in arrays]
        out = (fn(*leaves) * lucid.tensor(v, device=device)).sum()  # type: ignore[operator]
        (g,) = lucid.autograd.grad(out, [leaves[wrt]], create_graph=True)
        (h,) = lucid.autograd.grad((g * lucid.tensor(w, device=device)).sum(), [leaves[wrt]])
        return h.numpy()

    np.testing.assert_allclose(
        hessian_vector(lucid.cdist), hessian_vector(_explicit_cdist), atol=1e-5
    )
