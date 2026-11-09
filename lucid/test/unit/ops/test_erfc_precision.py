"""erfc keeps its relative precision in the tail.

Written as ``1 - erf(x)`` it lost everything past x ≈ 4: ``erfc(4)`` came back
0, where the answer is 1.54e-8.
"""

import math

import pytest

import lucid

_XS = [-6.0, -2.0, -0.5, 0.0, 0.3, 1.0, 2.5, 4.0, 6.0, 9.0]


def test_erfc_matches_double_precision_relatively() -> None:
    got = lucid.erfc(lucid.tensor(_XS)).tolist()
    for x, value in zip(_XS, got):
        assert value == pytest.approx(math.erfc(x), rel=3e-6, abs=0), x


def test_erfc_gradient_is_the_gaussian_slope() -> None:
    xs = [-1.0, 0.0, 0.5, 3.0]
    x = lucid.tensor(xs, requires_grad=True)
    lucid.erfc(x).sum().backward()
    assert x.grad is not None
    for xi, gi in zip(xs, x.grad.tolist()):
        want = -2.0 / math.sqrt(math.pi) * math.exp(-xi * xi)
        assert gi == pytest.approx(want, rel=1e-4, abs=1e-9), xi


def test_erfc_in_float64_is_double_precision() -> None:
    # The Chebyshev fit is float32 grade; float64 takes 1 - erf below 3 and
    # a continued fraction above it.
    got = lucid.erfc(lucid.tensor(_XS, dtype=lucid.float64)).tolist()
    for x, value in zip(_XS, got):
        assert value == pytest.approx(math.erfc(x), rel=1e-10, abs=0), x
