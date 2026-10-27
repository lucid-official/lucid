"""Bilinear and trilinear resizing differentiate twice.

A gradient penalty through a decoder that upsamples with
``interpolate(mode="bilinear")`` — a U-Net, an FPN, most GAN generators —
stopped at the resize: its backward had no graph-mode form.  Linear
resampling is linear in its input and separable, so its adjoint is one
matmul per axis against the 1-D resampling matrices, built from the same
source coordinates the forward kernel computes.

Checked against central differences of the first gradient in float64, on a
loss quadratic in the resize so the second derivative is not trivially
zero.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

CASES = [
    ("bilinear up", (2, 3, 4, 5), (7, 9), "bilinear", False),
    ("bilinear down", (1, 2, 9, 8), (4, 3), "bilinear", False),
    ("bilinear corners", (1, 2, 4, 6), (7, 5), "bilinear", True),
    ("trilinear", (1, 2, 3, 4, 3), (5, 3, 6), "trilinear", False),
    ("trilinear corners", (1, 1, 4, 3, 2), (2, 5, 4), "trilinear", True),
]


def _resize(x: lucid.Tensor, size: tuple[int, ...], mode: str, align: bool) -> lucid.Tensor:
    return F.interpolate(x, size=size, mode=mode, align_corners=align)


@pytest.mark.parametrize("name,shape,size,mode,align", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    name: str, shape: tuple[int, ...], size: tuple[int, ...], mode: str, align: bool
) -> None:
    rng = np.random.default_rng(len(name))
    x0 = rng.standard_normal(shape)
    r0 = rng.standard_normal(shape[:2] + size)
    u = rng.standard_normal(shape)

    def loss(x: lucid.Tensor) -> lucid.Tensor:
        y = _resize(x, size, mode, align)
        return (lucid.tensor(r0) * y * y).sum()

    def projected_gradient(values: np.ndarray) -> float:
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(x), [x])
        return float((np.asarray(g.numpy()) * u).sum())

    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(loss(x), [x], create_graph=True)
    eager = projected_gradient(x0)
    assert float((np.asarray(g.numpy()) * u).sum()) == pytest.approx(eager, rel=1e-12)

    (hvp,) = grad((g * lucid.tensor(u)).sum(), [x])
    e = rng.standard_normal(shape)
    h = 1e-6
    numeric = (projected_gradient(x0 + h * e) - projected_gradient(x0 - h * e)) / (2 * h)
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6, abs=1e-6)
    assert np.abs(np.asarray(hvp.numpy())).max() > 1e-3  # not a zero that passes


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "name,shape,size,mode,align", [CASES[0], CASES[3]], ids=[CASES[0][0], CASES[3][0]]
)
def test_metal_differentiates_twice_as_the_cpu_does(
    name: str, shape: tuple[int, ...], size: tuple[int, ...], mode: str, align: bool
) -> None:
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal(shape).astype(np.float32)
    answers = {}
    for device in ("cpu", "metal"):
        x = lucid.tensor(x0, device=device, requires_grad=True)
        y = _resize(x, size, mode, align)
        (g,) = grad((y * y).sum(), [x], create_graph=True)
        (hvp,) = grad((g * g).sum(), [x])
        answers[device] = hvp.numpy()
    np.testing.assert_allclose(answers["metal"], answers["cpu"], rtol=1e-4, atol=1e-4)
