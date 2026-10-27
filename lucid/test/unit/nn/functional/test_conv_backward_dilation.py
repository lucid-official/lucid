"""Convolution backward regression — dilated conv weight gradient on Metal.

The GPU ``conv_nd_backward`` computes ``dW`` with the channel-permute trick: a
convolution of ``x`` by ``grad``.  The mapping is

    dW[k] = sum_o x[k*dilation + o*stride] * grad[o]

so that convolution must stride by the ORIGINAL DILATION (output index ``k``
steps by ``dilation``) while the ``grad`` kernel is dilated by the ORIGINAL
STRIDE.  It instead used ``stride=1`` with ``input_dilation=dilation``, which
coincides with the correct mapping **only when dilation == 1**.  Every dilated
convolution therefore trained on silently wrong weight gradients (measured
rel maxdiff ~1.0 vs CPU; ~0.76 vs finite differences).  Forward and ``dx`` were
correct, so nothing surfaced — dilated/atrous backbones just learned wrong.
Fixed 2026-07-13.

Oracles: the CPU (Accelerate) backward, plus finite differences for the 2-D
dilated case (conv is smooth, so FD is reliable here).
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _grads(fn, xa, wa, device, seed):
    x = lucid.tensor(xa, dtype=lucid.float32, device=device)
    w = lucid.tensor(wa, dtype=lucid.float32, device=device)
    x.requires_grad = True
    w.requires_grad = True
    out = fn(x, w)
    g = np.random.default_rng(seed).standard_normal(out.shape).astype(np.float32)
    (out * lucid.tensor(g, dtype=lucid.float32, device=device)).sum().backward()
    return x.grad.numpy(), w.grad.numpy()


@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (1, 1), (2, 1)])
def test_conv2d_dilated_backward_metal_matches_cpu(
    dilation: int, stride: int, padding: int
) -> None:
    r = np.random.default_rng(dilation * 31 + stride * 7 + padding)
    xa = r.standard_normal((2, 3, 11, 11)).astype(np.float32)
    wa = r.standard_normal((4, 3, 3, 3)).astype(np.float32)
    fn = lambda x, w: F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)

    dxc, dwc = _grads(fn, xa, wa, "cpu", 7)
    dxm, dwm = _grads(fn, xa, wa, "metal", 7)
    np.testing.assert_allclose(dxm, dxc, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(dwm, dwc, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_conv2d_dilated_grouped_backward_metal_matches_cpu(groups: int) -> None:
    """Dilation composes with the per-group backward loop."""
    ci = co = 6
    r = np.random.default_rng(groups)
    xa = r.standard_normal((2, ci, 9, 9)).astype(np.float32)
    wa = r.standard_normal((co, ci // groups, 3, 3)).astype(np.float32)
    fn = lambda x, w: F.conv2d(x, w, padding=1, dilation=2, groups=groups)

    dxc, dwc = _grads(fn, xa, wa, "cpu", 3)
    dxm, dwm = _grads(fn, xa, wa, "metal", 3)
    np.testing.assert_allclose(dxm, dxc, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(dwm, dwc, atol=1e-4, rtol=1e-3)


def test_conv1d_and_conv3d_dilated_backward_metal_matches_cpu() -> None:
    """The dW mapping is N-D generic — guard the 1-D and 3-D paths too."""
    r = np.random.default_rng(11)
    x1 = r.standard_normal((2, 3, 13)).astype(np.float32)
    w1 = r.standard_normal((4, 3, 3)).astype(np.float32)
    f1 = lambda x, w: F.conv1d(x, w, dilation=2)
    np.testing.assert_allclose(
        _grads(f1, x1, w1, "metal", 5)[1], _grads(f1, x1, w1, "cpu", 5)[1], atol=1e-4, rtol=1e-3
    )

    x3 = r.standard_normal((1, 2, 7, 7, 7)).astype(np.float32)
    w3 = r.standard_normal((3, 2, 3, 3, 3)).astype(np.float32)
    f3 = lambda x, w: F.conv3d(x, w, dilation=2)
    np.testing.assert_allclose(
        _grads(f3, x3, w3, "metal", 5)[1], _grads(f3, x3, w3, "cpu", 5)[1], atol=1e-4, rtol=1e-3
    )


def test_conv2d_dilated_dweight_matches_finite_differences() -> None:
    """Absolute check: the Metal dW must equal the numerical gradient, not just
    agree with CPU (guards against both backends drifting together)."""
    r = np.random.default_rng(0)
    xa = r.standard_normal((1, 2, 7, 7)).astype(np.float32)
    wa = r.standard_normal((2, 2, 3, 3)).astype(np.float32)
    ga = np.random.default_rng(1).standard_normal((1, 2, 3, 3)).astype(np.float32)

    def loss(w_np: np.ndarray) -> float:
        x = lucid.tensor(xa, dtype=lucid.float32, device="cpu")
        w = lucid.tensor(w_np, dtype=lucid.float32, device="cpu")
        out = F.conv2d(x, w, dilation=2)
        return float((out * lucid.tensor(ga, dtype=lucid.float32)).sum().item())

    eps = 1e-3
    fd = np.zeros_like(wa)
    it = np.nditer(wa, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        wp, wm = wa.copy(), wa.copy()
        wp[i] += eps
        wm[i] -= eps
        fd[i] = (loss(wp) - loss(wm)) / (2 * eps)
        it.iternext()

    x = lucid.tensor(xa, dtype=lucid.float32, device="metal")
    w = lucid.tensor(wa, dtype=lucid.float32, device="metal")
    w.requires_grad = True
    out = F.conv2d(x, w, dilation=2)
    (out * lucid.tensor(ga, dtype=lucid.float32, device="metal")).sum().backward()

    np.testing.assert_allclose(w.grad.numpy(), fd, atol=2e-2, rtol=2e-2)
