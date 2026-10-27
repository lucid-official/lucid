"""Pooling backward regression — the family had no numerical backward test, which
let a silent metal bug hide: the non-overlapping max_pool2d fast path
reconstructed dx at (Oh*Kh, Ow*Kw) and mis-placed gradients whenever the input
was not an exact multiple of the kernel (odd feature maps).  Fixed 2026-07-10.

Oracle = the CPU (Accelerate) backend, which uses a straightforward scatter and
is verified correct.  (Finite-difference gradcheck is unreliable for max: the
gradient is piecewise-constant, so an FD step that crosses an argmax boundary
disagrees with the true subgradient — it "fails" on the correct CPU path too.)
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


def _grad(fn, shape, device, seed):
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    x = lucid.tensor(a, dtype=lucid.float32, device=device)
    x.requires_grad = True
    fn(x).sum().backward()
    return x.grad.numpy()


# Non-divisible sizes are the regression: (5,5)/(7,7)/(15,15) with kernel 2 were
# silently wrong on metal; even sizes happened to be fine.
@pytest.mark.parametrize("hw", [(5, 5), (7, 7), (8, 8), (15, 15), (16, 16), (6, 9), (9, 6)])
def test_max_pool2d_backward_metal_matches_cpu(hw: tuple) -> None:
    if not metal_available():
        pytest.skip("metal backend unavailable")
    fn = lambda t: F.max_pool2d(t, kernel_size=2, stride=2)
    seed = hw[0] * 31 + hw[1]
    np.testing.assert_allclose(
        _grad(fn, (2, 3, *hw), "metal", seed),
        _grad(fn, (2, 3, *hw), "cpu", seed),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "k,s,hw", [(3, 2, (9, 9)), (3, 3, (7, 7)), (2, 2, (10, 7)), (3, 1, (6, 6))]
)
def test_max_pool2d_variants_metal_matches_cpu(k: int, s: int, hw: tuple) -> None:
    """Overlapping (stride != kernel, scatter path) + more non-divisible cases."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    fn = lambda t: F.max_pool2d(t, kernel_size=k, stride=s)
    np.testing.assert_allclose(
        _grad(fn, (2, 3, *hw), "metal", 7),
        _grad(fn, (2, 3, *hw), "cpu", 7),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# ceil_mode / count_include_pad regression
#
# Both parameters were accepted by the Python layer and stored on the module,
# but the engine binding never carried them, so every ``ceil_mode=True`` in the
# codebase was a silent no-op: the caller asked for ceiling output sizing and
# got floor geometry.  GoogLeNet had to hand-roll a padded work-alike and
# ResNeSt's downsample was quietly pooling one row and column short.  These
# tests pin the plumbing so it cannot go dead again.
# ---------------------------------------------------------------------------


def _expected_out(size: int, k: int, stride: int, pad: int, ceil: bool) -> int:
    num = size + 2 * pad - k + (stride - 1 if ceil else 0)
    out = num // stride + 1
    if ceil and (out - 1) * stride >= size + pad:
        out -= 1
    return out


@pytest.mark.parametrize(
    "size,k,stride,pad",
    [
        (5, 2, 2, 0),
        (7, 2, 2, 0),
        (110, 3, 2, 0),
        (26, 3, 2, 0),
        (9, 3, 2, 1),
        (8, 2, 2, 0),
    ],
)
@pytest.mark.parametrize("ceil", [False, True])
def test_max_pool2d_honours_ceil_mode(size, k, stride, pad, ceil):
    x = lucid.zeros(1, 1, size, size)
    y = F.max_pool2d(x, k, stride=stride, padding=pad, ceil_mode=ceil)
    want = _expected_out(size, k, stride, pad, ceil)
    assert tuple(y.shape[2:]) == (want, want)


@pytest.mark.parametrize("ceil", [False, True])
def test_avg_pool2d_honours_ceil_mode(ceil):
    x = lucid.zeros(1, 1, 5, 5)
    y = F.avg_pool2d(x, 2, stride=2, ceil_mode=ceil)
    want = _expected_out(5, 2, 2, 0, ceil)
    assert tuple(y.shape[2:]) == (want, want)


def test_avg_pool2d_ceil_divisor_excludes_the_overhang():
    # Every window of an all-ones input must average to exactly 1.0, including
    # the clipped edge windows.  Dividing by the full kernel area instead would
    # give 1/2 along the edges and 1/4 in the corner.
    x = lucid.ones(1, 1, 3, 3)
    y = F.avg_pool2d(x, 2, stride=2, ceil_mode=True)
    assert tuple(y.shape[2:]) == (2, 2)
    for i in range(2):
        for j in range(2):
            assert abs(float(y[0, 0, i, j].item()) - 1.0) < 1e-6


def test_avg_pool2d_count_include_pad_changes_the_divisor():
    x = lucid.ones(1, 1, 4, 4)
    inc = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=True)
    exc = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
    # Excluding padding, an all-ones input averages to 1.0 everywhere; counting
    # it, the corner window sees 4 real elements out of 9.
    assert abs(float(exc[0, 0, 0, 0].item()) - 1.0) < 1e-6
    assert abs(float(inc[0, 0, 0, 0].item()) - 4.0 / 9.0) < 1e-6


def test_max_pool2d_ceil_routes_gradient_to_the_trailing_row():
    # With floor sizing the last row and column are never pooled, so they get
    # no gradient at all; ceil sizing must reach them.
    x = lucid.arange(0, 25, dtype=lucid.float32).reshape(1, 1, 5, 5)
    x.requires_grad = True
    F.max_pool2d(x, 2, stride=2, ceil_mode=True).sum().backward()
    assert float(x.grad[0, 0, 4, 4].item()) > 0.0
