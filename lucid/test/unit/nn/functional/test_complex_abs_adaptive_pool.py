"""Regression tests for complex ``abs`` and non-divisible adaptive max pooling.

Both were logged by the 2026-07-13 backward sweep as "loud, low-severity"
gaps.  One of them was neither loud nor low-severity:

1. ``abs()`` on a complex tensor.  ``|z| = sqrt(Re² + Im²)`` is a complex→REAL
   map, but it went through ``UnaryKernel``, which tags the output with the
   *input* dtype.  On Metal the MLX kernel computed the correct real magnitudes
   and they were then labelled C64 — so the reader paired consecutive
   magnitudes into ``(re, im)`` and ran off the end of the buffer.  Silently
   wrong, not a crash.  On the CPU the same path raised ``NotImplementedError``.
   Now composed from the ``real``/``imag`` primitives (like the sibling
   ``angle``) with overflow-safe scaling, so the full F32 range is exact.
   Like ``real``/``imag``/``angle``, this path is forward-only — Lucid has no
   complex autograd.  ``abs`` on a *real* tensor keeps its gradient.
2. ``adaptive_max_pool{1,2,3}d`` rejected non-divisible output sizes while
   ``adaptive_avg_pool*`` accepted them — the engine op lowers adaptive pooling
   to a fixed-kernel pool, and only avg had the Python fallback wired up.  The
   ``adaptive_max_pool2d`` docstring even advertised ``(1, 64, 11, 13) → (3, 3)``,
   which raised.  Both families now run through ``_adaptive_pool`` in
   ``lucid.nn.functional._pool_core``, so they cannot drift apart again.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

DEVICES = ["cpu", "metal"]


# ── 1. complex abs ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(3, 8), (2, 4, 16), (5,)])
def test_fft_abs_matches_reference(device, shape):
    """``fft(x).abs()`` raised on the CPU and was silently wrong on Metal."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape).astype(np.float32)
    ref = np.abs(np.fft.fft(base, axis=-1))
    got = lucid.fft.fft(lucid.tensor(base, device=device)).abs()
    assert got.dtype == lucid.float32, "complex abs must return a REAL dtype"
    assert np.abs(got.numpy() - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_complex_abs_entry_points_agree(device):
    """Method, free function and ``__abs__`` must all take the same path."""
    z = lucid.fft.fft(lucid.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), device=device))
    a, b, c = z.abs(), lucid.abs(z), abs(z)
    assert a.dtype == b.dtype == c.dtype == lucid.float32
    assert np.array_equal(a.numpy(), b.numpy())
    assert np.array_equal(a.numpy(), c.numpy())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "values",
    [
        [0 + 0j, 0 + 0j],  # exact zero — must not divide 0/0
        [3 + 0j, -3 + 0j],  # pure real
        [0 + 3j, 0 - 3j],  # pure imaginary
        [3 + 4j, -5 + 12j],  # exact Pythagorean triples
        [1e-8 + 1e-8j, 1e-20 + 0j],  # small
    ],
)
def test_complex_abs_edge_values(device, values):
    z = np.array(values, dtype=np.complex64)
    got = lucid.tensor(z, device=device).abs().numpy()
    assert np.abs(got - np.abs(z)).max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_complex_abs_is_overflow_safe(device):
    """Naive sqrt(re²+im²) overflowed to inf / underflowed to 0 in F32."""
    z = np.array([1e20 + 1e20j, 1e-25 + 1e-25j, 1e30 + 0j], dtype=np.complex64)
    got = lucid.tensor(z, device=device).abs().numpy()
    ref = np.abs(z)
    assert np.isfinite(got).all(), "must not overflow for representable results"
    assert (got > 0).all(), "must not underflow to zero"
    assert np.abs(got / ref - 1.0).max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_complex_abs_propagates_infinity(device):
    """hypot's contract: |z| is inf when either lane is — not NaN.

    Scaling by ``m = max(|Re|, |Im|)`` would give ``inf / inf = NaN`` if the
    divisor were unclamped, which is why the upper clamp is FLT_MAX.

    Not asserted: ``|NaN + inf·i|``.  IEEE-754 hypot calls that ``inf``; Lucid
    propagates NaN because ``maximum`` does.  A NaN lane already means the
    producing computation broke, so the distinction is not worth a special case.
    """
    z = np.array(
        [complex(np.inf, 1), complex(1, np.inf), complex(np.inf, np.inf)],
        dtype=np.complex64,
    )
    got = lucid.tensor(z, device=device).abs().numpy()
    assert not np.isnan(got).any()
    assert np.isinf(got).all()


@pytest.mark.parametrize("device", DEVICES)
def test_real_abs_still_differentiable(device):
    """The complex branch must not disturb abs on real tensors."""
    base = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
    x = lucid.tensor(base, device=device)
    x.requires_grad = True
    out = x.abs()
    (out * lucid.ones_like(out)).sum().backward()
    assert out.dtype == lucid.float32
    assert np.array_equal(out.numpy(), np.abs(base))
    assert x.grad is not None
    # sign(x), with the standard 0 subgradient at the kink.
    assert np.array_equal(x.grad.numpy(), np.array([-1, -1, 0, 1, 1], np.float32))


# ── 2. adaptive pooling with non-divisible sizes ─────────────────────────────


def _reference_adaptive(x, out, mode):
    """Reference contract: start = floor(i·In/Out), end = ceil((i+1)·In/Out)."""
    n = len(out)
    ins = x.shape[-n:]

    def spans(ax):
        return [
            ((i * ins[ax]) // out[ax], -(-(i + 1) * ins[ax] // out[ax])) for i in range(out[ax])
        ]

    red = (lambda w, ax: w.max(axis=ax)) if mode == "max" else (lambda w, ax: w.mean(axis=ax))
    if n == 1:
        return np.stack([red(x[..., a:b], -1) for a, b in spans(0)], axis=-1)
    if n == 2:
        return np.stack(
            [
                np.stack([red(x[..., a:b, c:d], (-2, -1)) for c, d in spans(1)], axis=-1)
                for a, b in spans(0)
            ],
            axis=-2,
        )
    return np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [red(x[..., a:b, c:d, e:f], (-3, -2, -1)) for e, f in spans(2)],
                        axis=-1,
                    )
                    for c, d in spans(1)
                ],
                axis=-2,
            )
            for a, b in spans(0)
        ],
        axis=-3,
    )


ADAPTIVE_2D = [
    ((1, 2, 7, 7), (3, 3)),  # non-divisible both axes
    ((1, 2, 7, 7), (2, 2)),
    ((2, 3, 11, 13), (3, 3)),  # the docstring's own example
    ((1, 1, 9, 5), (4, 2)),
    ((1, 2, 8, 8), (4, 4)),  # divisible — engine fast path, must stay identical
    ((1, 2, 7, 7), (1, 1)),  # global pool
]


@pytest.mark.parametrize("shape,out", ADAPTIVE_2D)
@pytest.mark.parametrize("mode", ["max", "avg"])
@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_pool2d_matches_reference(shape, out, mode, device):
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape).astype(np.float32)
    ref = _reference_adaptive(base.astype(np.float64), out, mode)
    fn = F.adaptive_max_pool2d if mode == "max" else F.adaptive_avg_pool2d
    got = fn(lucid.tensor(base, device=device), out).numpy()
    assert got.shape == tuple(shape[:2]) + tuple(out)
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["max", "avg"])
def test_adaptive_pool1d_non_divisible(device, mode):
    rng = np.random.default_rng(1)
    base = rng.standard_normal((1, 2, 7)).astype(np.float32)
    ref = _reference_adaptive(base.astype(np.float64), (3,), mode)
    fn = F.adaptive_max_pool1d if mode == "max" else F.adaptive_avg_pool1d
    got = fn(lucid.tensor(base, device=device), 3).numpy()
    assert got.shape == (1, 2, 3)
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["max", "avg"])
def test_adaptive_pool3d_non_divisible(device, mode):
    rng = np.random.default_rng(2)
    base = rng.standard_normal((1, 2, 5, 7, 3)).astype(np.float32)
    ref = _reference_adaptive(base.astype(np.float64), (2, 3, 2), mode)
    fn = F.adaptive_max_pool3d if mode == "max" else F.adaptive_avg_pool3d
    got = fn(lucid.tensor(base, device=device), (2, 3, 2)).numpy()
    assert got.shape == (1, 2, 2, 3, 2)
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,out", [((1, 1, 7, 7), (3, 3)), ((1, 1, 8, 8), (4, 4))])
def test_adaptive_max_pool2d_backward(device, shape, out):
    """Finite-difference check on both the fallback and the fast path."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal(shape).astype(np.float64)
    probe = F.adaptive_max_pool2d(lucid.tensor(base, dtype=lucid.float64), out)
    w = rng.standard_normal(probe.shape).astype(np.float64)

    dt = lucid.float64 if device == "cpu" else lucid.float32
    data = base if device == "cpu" else base.astype(np.float32)
    weight = w if device == "cpu" else w.astype(np.float32)
    x = lucid.tensor(data, device=device, dtype=dt)
    x.requires_grad = True
    (F.adaptive_max_pool2d(x, out) * lucid.tensor(weight, device=device, dtype=dt))
    (F.adaptive_max_pool2d(x, out) * lucid.tensor(weight, device=device, dtype=dt)).sum().backward()
    assert x.grad is not None
    got = np.asarray(x.grad.numpy(), dtype=np.float64)

    def value(b):
        t = lucid.tensor(b, dtype=lucid.float64)
        prod = F.adaptive_max_pool2d(t, out) * lucid.tensor(w, dtype=lucid.float64)
        return float(prod.sum().item())

    ref = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        plus = base.copy()
        plus[idx] += 1e-4
        minus = base.copy()
        minus[idx] -= 1e-4
        ref[idx] = (value(plus) - value(minus)) / 2e-4
        it.iternext()

    scale = max(float(np.abs(ref).max()), 1e-12)
    tol = 1e-7 if device == "cpu" else 1e-4
    assert np.abs(got - ref).max() / scale < tol


@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_max_pool_modules_accept_non_divisible(device):
    rng = np.random.default_rng(4)
    x2 = lucid.tensor(rng.standard_normal((1, 2, 7, 7)).astype(np.float32), device=device)
    assert nn.AdaptiveMaxPool2d((3, 3))(x2).shape == (1, 2, 3, 3)
    x1 = lucid.tensor(rng.standard_normal((1, 2, 7)).astype(np.float32), device=device)
    assert nn.AdaptiveMaxPool1d(3)(x1).shape == (1, 2, 3)
    x3 = lucid.tensor(rng.standard_normal((1, 2, 5, 7, 3)).astype(np.float32), device=device)
    assert nn.AdaptiveMaxPool3d((2, 3, 2))(x3).shape == (1, 2, 2, 3, 2)


def test_adaptive_max_pool2d_docstring_example():
    """The documented example used to raise (11 and 13 are not divisible by 3)."""
    x = lucid.randn(1, 64, 11, 13)
    assert F.adaptive_max_pool2d(x, output_size=(3, 3)).shape == (1, 64, 3, 3)
