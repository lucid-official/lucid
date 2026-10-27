"""Convolutions differentiate twice in every configuration they run in.

Graph-mode backward through a convolution — what a gradient penalty or a
meta-learning step needs — covered only the ungrouped, undilated 2-D case
and refused grouped, dilated, 1-D and 3-D convolutions and every
transposed one.  Most real networks have at least one of those (a depthwise
block, a dilated context layer, a decoder's upsampling), so the penalty
failed on the first real model it met.

A convolution's adjoint is its transposed convolution and the other way
round, and each weight gradient is a correlation of the input with the
output gradient — a convolution again, with batch and channel swapped and
stride and dilation trading places.  Where the forward's last window
stopped short of the padded input that correlation runs on past the
kernel and is cut back to it; a transposed convolution whose output
padding reaches its stride leaves the adjoint one window long, cut the
same way.  The shapes below reach both cuts.

Checked against central differences of the first gradient in float64, so
no oracle is involved: for a projection ``<grad_first L, u>`` its gradient
with respect to both the input and the weight — the cross term included —
is compared along a random direction.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

Conv = Callable[..., lucid.Tensor]

# (name, op, input shape, weight shape, bias length, keyword arguments)
CASES = [
    (
        "conv1d strided dilated grouped",
        F.conv1d,
        (2, 4, 11),
        (6, 2, 3),
        6,
        {"stride": 2, "padding": 1, "dilation": 2, "groups": 2},
    ),
    ("conv2d stride past the last window", F.conv2d, (2, 2, 8, 7), (3, 2, 3, 3), 3, {"stride": 2}),
    (
        "conv2d mixed settings grouped",
        F.conv2d,
        (1, 4, 7, 8),
        (4, 2, 3, 2),
        4,
        {"stride": (2, 1), "padding": (1, 0), "dilation": (1, 2), "groups": 2},
    ),
    ("conv2d depthwise", F.conv2d, (2, 3, 6, 6), (3, 1, 3, 3), 3, {"padding": 1, "groups": 3}),
    (
        "conv3d strided dilated",
        F.conv3d,
        (1, 2, 6, 5, 6),
        (2, 2, 2, 2, 2),
        2,
        {"stride": 2, "padding": 1, "dilation": (1, 2, 1)},
    ),
    (
        "conv_transpose1d grouped",
        F.conv_transpose1d,
        (2, 4, 5),
        (4, 3, 3),
        6,
        {"stride": 2, "padding": 1, "output_padding": 1, "groups": 2},
    ),
    (
        "conv_transpose2d output padding past the stride",
        F.conv_transpose2d,
        (1, 2, 4, 5),
        (2, 3, 3, 3),
        3,
        {"stride": 1, "dilation": 2, "output_padding": 1},
    ),
    (
        "conv_transpose2d strided dilated",
        F.conv_transpose2d,
        (2, 2, 4, 3),
        (2, 2, 3, 2),
        2,
        {"stride": 2, "padding": 1, "dilation": (2, 1)},
    ),
    (
        "conv_transpose3d",
        F.conv_transpose3d,
        (1, 2, 3, 3, 2),
        (2, 2, 2, 3, 2),
        2,
        {"stride": 2, "padding": (0, 1, 0), "output_padding": (1, 0, 1)},
    ),
]


def _loss(
    op: Conv,
    x: lucid.Tensor,
    w: lucid.Tensor,
    b: lucid.Tensor,
    r: lucid.Tensor,
    kw: dict[str, object],
) -> lucid.Tensor:
    y = op(x, w, b, **kw)
    return (r * y * y).sum()


def _t(values: np.ndarray, device: str = "cpu", grad_: bool = False) -> lucid.Tensor:
    dtype = lucid.float64 if device == "cpu" else lucid.float32
    return lucid.tensor(values, dtype=dtype, device=device, requires_grad=grad_)


@pytest.mark.parametrize("first", ["input", "weight"])
@pytest.mark.parametrize("name,op,xs,ws,nb,kw", CASES, ids=[case[0] for case in CASES])
def test_the_second_derivative_matches_central_differences(
    first: str,
    name: str,
    op: Conv,
    xs: tuple[int, ...],
    ws: tuple[int, ...],
    nb: int,
    kw: dict[str, object],
) -> None:
    rng = np.random.default_rng(len(name))
    x0, w0, b0 = (rng.standard_normal(xs), rng.standard_normal(ws), rng.standard_normal(nb))
    r0 = rng.standard_normal(op(_t(x0), _t(w0), _t(b0), **kw).shape)
    u = rng.standard_normal(xs if first == "input" else ws)

    def projected_gradient(xv: np.ndarray, wv: np.ndarray) -> float:
        # The first derivative from the eager path, which is not under test.
        x, w = _t(xv, grad_=True), _t(wv, grad_=True)
        (g,) = grad(_loss(op, x, w, _t(b0), _t(r0), kw), [x if first == "input" else w])
        return float((np.asarray(g.numpy()) * u).sum())

    x, w = _t(x0, grad_=True), _t(w0, grad_=True)
    (g,) = grad(
        _loss(op, x, w, _t(b0), _t(r0), kw), [x if first == "input" else w], create_graph=True
    )
    by_x, by_w = grad((g * _t(u)).sum(), [x, w], allow_unused=True)

    h = 1e-6
    for label, analytic, (xv, wv), base in (
        ("input", by_x, (x0, w0), x0),
        ("weight", by_w, (x0, w0), w0),
    ):
        e = rng.standard_normal(base.shape)
        step = (e * h, 0) if label == "input" else (0, e * h)
        plus = projected_gradient(xv + step[0], wv + step[1])
        minus = projected_gradient(xv - step[0], wv - step[1])
        numeric = (plus - minus) / (2 * h)
        exact = 0.0 if analytic is None else float((np.asarray(analytic.numpy()) * e).sum())
        assert exact == pytest.approx(numeric, rel=1e-6, abs=1e-6), f"d/d{label} <d/d{first} L, u>"


def test_the_checks_above_see_a_curved_loss() -> None:
    # Guard the guard: for a loss bilinear in (x, W) every second
    # derivative through one argument alone would be zero, and a formula
    # returning zeros would pass.
    name, op, xs, ws, nb, kw = CASES[2]
    rng = np.random.default_rng(0)
    x, w = _t(rng.standard_normal(xs), grad_=True), _t(rng.standard_normal(ws), grad_=True)
    b = _t(rng.standard_normal(nb))
    r = _t(rng.standard_normal(op(x, w, b, **kw).shape))
    (g,) = grad(_loss(op, x, w, b, r, kw), [x], create_graph=True)
    (by_x,) = grad(g.sum(), [x])
    assert np.abs(np.asarray(by_x.numpy())).max() > 1e-3


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "name,op,xs,ws,nb,kw", [CASES[0], CASES[2], CASES[6]], ids=[CASES[i][0] for i in (0, 2, 6)]
)
def test_metal_differentiates_twice_as_the_cpu_does(
    name: str, op: Conv, xs: tuple[int, ...], ws: tuple[int, ...], nb: int, kw: dict[str, object]
) -> None:
    rng = np.random.default_rng(1)
    x0, w0, b0 = (rng.standard_normal(xs), rng.standard_normal(ws), rng.standard_normal(nb))
    x0, w0, b0 = (a.astype(np.float32) for a in (x0, w0, b0))
    answers = {}
    for device in ("cpu", "metal"):
        x = lucid.tensor(x0, device=device, requires_grad=True)
        w = lucid.tensor(w0, device=device, requires_grad=True)
        y = op(x, w, lucid.tensor(b0, device=device), **kw)
        (g,) = grad((y * y).sum(), [x], create_graph=True)
        by_x, by_w = grad(g.sum(), [x, w])
        answers[device] = (by_x.numpy(), by_w.numpy())
    for got, want in zip(answers["metal"], answers["cpu"]):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
