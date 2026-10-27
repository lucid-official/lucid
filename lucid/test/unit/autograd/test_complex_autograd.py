"""A complex tensor used to be where every gradient stopped.

``real``, ``imag`` and ``conj`` built their output and returned it, with
no backward of any kind, so:

    real(fft(x)).sum().backward()   ->   x.grad is None

with nothing raised.  ``fft`` itself is wired correctly; the chain broke
one step later, which meant any loss written through a frequency-domain
transform trained on nothing at all.  ``abs`` of a complex input is a
composite of ``real`` and ``imag``, so it was unreachable for the same
reason.

The header on ``Real.h`` said this belonged to "the Python autograd
layer".  There was no such layer.

Conventions are the reference's, measured rather than derived:

    d/dz real(z) = g + 0i        d/dz imag(z) = 0 + g i
    d/dz conj(z) = conj(g)       d/dz |z|     = g z / |z|,  0 at z = 0
"""

import numpy as np
import pytest

import lucid

Z = np.array([1 + 2j, 3 - 1j], dtype=np.complex64)
SIGNAL = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


def _grad(build, arr=Z):
    z = lucid.tensor(arr.copy(), requires_grad=True)
    build(z).sum().backward()
    assert z.grad is not None, "no gradient reached the input"
    return np.asarray(z.grad.numpy())


# ── the projections ───────────────────────────────────────────────────────────


def test_real_sends_the_gradient_to_the_real_lane() -> None:
    assert np.allclose(_grad(lucid.real), [1 + 0j, 1 + 0j])


def test_imag_sends_it_to_the_imaginary_lane() -> None:
    assert np.allclose(_grad(lucid.imag), [0 + 1j, 0 + 1j])


def test_conj_conjugates_the_gradient() -> None:
    assert np.allclose(_grad(lambda z: lucid.real(lucid.conj(z))), [1 + 0j, 1 + 0j])


def test_abs_is_the_unit_vector_in_z() -> None:
    got = _grad(lucid.abs)
    expected = Z / np.abs(Z)
    assert np.allclose(got, expected)


def test_conj_of_a_real_tensor_still_has_no_node_to_report() -> None:
    """Conjugation is the identity on a real input; the backend returns
    the argument itself, so putting a node on it would double-count."""
    x = lucid.tensor(np.array([1.0, -2.0]), requires_grad=True)
    (lucid.conj(x) * 2.0).sum().backward()
    assert np.allclose(np.asarray(x.grad.numpy()), [2.0, 2.0])


# ── abs at the one point its composite could not survive ──────────────────────


def test_abs_gradient_at_exactly_zero_is_zero_not_nan() -> None:
    """The overflow-safe composite divides by a clamped magnitude, and the
    divide's own backward forms ``-re / m**2`` — ``FLT_MIN`` squared
    underflows to zero in float32, so ``0 / 0`` gave NaN.  It spread:
    ``abs(fft2(x))`` was all-NaN whenever one coefficient was zero, which
    for most real inputs one is.
    """
    got = _grad(lucid.abs, np.array([0 + 0j, 3 + 4j], dtype=np.complex64))
    assert not np.isnan(got).any(), got
    assert got[0] == 0
    assert np.allclose(got[1], 0.6 + 0.8j)


@pytest.mark.parametrize(
    "value,expected", [(1e20 + 1e20j, np.sqrt(2.0) * 1e20), (1e-30 + 0j, 1e-30), (3 + 4j, 5.0)]
)
def test_the_magnitude_itself_is_unchanged(value, expected) -> None:
    """Guard the guard: the dedicated backward must not have cost the
    forward its overflow safety."""
    got = float(lucid.abs(lucid.tensor(np.array([value], dtype=np.complex64))).item())
    assert np.isfinite(got)
    assert abs(got - expected) / expected < 1e-6


# ── accumulation, which nothing had exercised ─────────────────────────────────


def test_two_complex_contributions_accumulate() -> None:
    """``abs`` reads ``z`` through both projections, so its two gradients
    must sum — and ``accumulate_into`` had no complex case at all."""
    z = lucid.tensor(Z.copy(), requires_grad=True)
    (lucid.real(z) + lucid.imag(z)).sum().backward()
    assert np.allclose(np.asarray(z.grad.numpy()), [1 + 1j, 1 + 1j])


def test_complex_builds_from_two_real_operands() -> None:
    re = lucid.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    im = lucid.tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
    lucid.abs(lucid.complex(re, im)).sum().backward()
    magnitude = np.hypot([1.0, 2.0], [3.0, 4.0])
    assert np.allclose(np.asarray(re.grad.numpy()), np.array([1.0, 2.0]) / magnitude)
    assert np.allclose(np.asarray(im.grad.numpy()), np.array([3.0, 4.0]) / magnitude)


# ── the point of all of it ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda x: lucid.real(lucid.fft.fft(x)), [4.0, 0.0, 0.0, 0.0]),
        (lambda x: lucid.imag(lucid.fft.fft(x)), [0.0, 0.0, 0.0, 0.0]),
        (lambda x: lucid.abs(lucid.fft.fft(x)), [-1.41421, 0.58579, 1.41421, 3.41421]),
    ],
)
def test_a_gradient_survives_a_round_trip_through_the_frequency_domain(build, expected) -> None:
    x = lucid.tensor(SIGNAL.copy(), requires_grad=True)
    build(x).sum().backward()
    assert x.grad is not None
    assert np.allclose(np.asarray(x.grad.numpy()), expected, atol=1e-4)


def test_a_real_leaf_receives_a_real_gradient() -> None:
    """``fft`` of a real input is complex, so the gradient coming back is
    too, while ``d/dx`` of a real parameter has to be real.  Taking the
    real part is a projection, not a cast — it arrived at the leaf as
    ``astype: complex64 -> float32``, which is not implemented and never
    should have been the operation."""
    x = lucid.tensor(SIGNAL.copy(), requires_grad=True)
    lucid.real(lucid.fft.fft(x)).sum().backward()
    assert "float32" in str(x.grad.dtype)
    assert not x.grad.is_complex()


def test_a_spectral_loss_actually_trains() -> None:
    """The shape the defect took: a loss that could never move."""
    x = lucid.tensor(SIGNAL.copy(), requires_grad=True)
    target = lucid.tensor(np.full(4, 2.0, dtype=np.float32))
    first = None
    for _ in range(40):
        loss = ((lucid.abs(lucid.fft.fft(x)) - target) ** 2).sum()
        if first is None:
            first = float(loss.item())
        x.grad = None
        loss.backward()
        with lucid.no_grad():
            x -= 0.02 * x.grad
    assert float(loss.item()) < first * 0.1
