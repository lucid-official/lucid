"""The adjoint of a half-spectrum transform is not its inverse.

``rfft`` maps a real signal of length ``n`` to ``n // 2 + 1`` complex
bins, relying on Hermitian symmetry to drop the rest.  ``irfft`` inverts
that by *reconstructing* the dropped half, so it reads every interior bin
twice — once as itself, once as the conjugate mirror.

Backward passes need adjoints, and both were written as inverses:

* ``rfft``'s backward called ``irfft``, which doubles.  The adjoint must
  not: those conjugate entries are not inputs a gradient can flow to.
* ``irfft``'s backward called ``rfft``, which does not double.  The
  adjoint must, for the same reason read the other way.

Every bin strictly between DC and Nyquist was therefore off by exactly a
factor of two, in opposite directions.  DC and Nyquist were right, being
their own mirrors — the worst way to be wrong, since those are the
entries a spot check reads first.

Unreachable until complex tensors carried a gradient at all, so these
were newly visible rather than newly broken.
"""

import numpy as np
import pytest

import lucid


def _grad(fn, arr, **kwargs) -> np.ndarray:
    x = lucid.tensor(arr.copy(), requires_grad=True)
    lucid.abs(fn(x, **kwargs)).sum().backward()
    assert x.grad is not None, "no gradient reached the input"
    return np.asarray(x.grad.numpy())


def _numpy_adjoint_of_rfft(arr: np.ndarray) -> np.ndarray:
    """``Re(A^H G)`` written out, with ``A[k, j] = exp(-2i pi j k / n)``."""
    n = arr.size
    spectrum = np.fft.rfft(np.asarray(arr, dtype=np.float64))
    cotangent = spectrum / np.abs(spectrum)  # d|X| / dX for a sum
    k = np.arange(spectrum.size)[:, None]
    j = np.arange(n)[None, :]
    return np.real((cotangent[:, None] * np.exp(2j * np.pi * j * k / n)).sum(axis=0))


# ── rfft: each bin counted once ───────────────────────────────────────────────


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 12])
def test_rfft_adjoint_matches_the_written_out_matrix(n) -> None:
    arr = np.arange(1.0, n + 1, dtype=np.float32)
    assert np.allclose(_grad(lucid.fft.rfft, arr), _numpy_adjoint_of_rfft(arr), atol=1e-4)


def test_rfft_interior_bins_are_not_doubled() -> None:
    """The concrete number the old code produced, so a regression is
    recognisable rather than merely unequal."""
    arr = np.arange(1.0, 7.0, dtype=np.float32)
    got = _grad(lucid.fft.rfft, arr)
    assert np.allclose(got, [-1.36603, 1.0, 0.36603, 1.63397, 1.0, 3.36603], atol=1e-4)
    doubled_interior = [-2.73205, 0.0, 0.73205, 1.26795, 2.0, 4.73205]
    assert not np.allclose(got, doubled_interior, atol=1e-3)


# ── irfft: each interior bin counted twice ────────────────────────────────────


@pytest.mark.parametrize("n", [4, 5, 6, 8])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_irfft_adjoint_doubles_the_interior(n, norm) -> None:
    """A finite difference is the arbiter: it knows nothing about which
    transform is the inverse of which.

    The loss is a sum of squares rather than of absolute values.  The
    output of ``irfft`` is real, so ``abs`` puts a kink at every zero
    crossing and a central difference straddling one reports the average
    of two different slopes — a fact about the probe, not the adjoint.
    """
    arr = np.arange(1.0, n + 1, dtype=np.float32)
    x = lucid.tensor(arr.copy(), requires_grad=True)
    (lucid.fft.irfft(x, norm=norm) ** 2).sum().backward()
    analytic = np.asarray(x.grad.numpy())

    def loss(v: np.ndarray) -> float:
        probe = lucid.tensor(v.astype(np.float32))
        return float((lucid.fft.irfft(probe, norm=norm) ** 2).sum().item())

    step = 1e-3
    wide = arr.astype(np.float64)
    numeric = np.empty_like(wide)
    for i in range(wide.size):
        up, down = wide.copy(), wide.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (loss(up) - loss(down)) / (2 * step)
    # A float32 transform differenced at 1e-3 loses about three digits to
    # cancellation, so the tolerance is the probe's, not the adjoint's.
    assert np.allclose(analytic, numeric, rtol=3e-2, atol=1e-2), (analytic, numeric)


# ── a real input is a spectrum with no imaginary part ─────────────────────────


@pytest.mark.parametrize("name", ["irfft", "irfft2", "irfftn", "hfft"])
def test_the_inverse_real_transforms_accept_a_real_input(name) -> None:
    """They raised ``irfftn requires C64 input, got float32``.  The
    reference accepts one, and reading a real array as a spectrum with no
    imaginary part is the only sensible interpretation."""
    fn = getattr(lucid.fft, name)
    arr = np.arange(1.0, 13.0, dtype=np.float32)
    arr = arr.reshape(3, 4) if name.endswith(("2", "n")) else arr
    out = fn(lucid.tensor(arr))
    assert out is not None and np.isfinite(np.asarray(out.numpy())).all()


def test_ihfft_carries_a_gradient() -> None:
    """It routed through a bare engine ``_conj_complex`` with no autograd
    node, so the chain ended there and ``x.grad`` stayed ``None``."""
    arr = np.arange(1.0, 7.0, dtype=np.float32)
    assert np.abs(_grad(lucid.fft.ihfft, arr)).sum() > 0.0


# ── the round trips still hold ────────────────────────────────────────────────


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("n", [6, 7])
def test_irfft_inverts_rfft(norm, n) -> None:
    """Guard the guard: fixing the adjoints must not disturb the values."""
    arr = np.arange(1.0, n + 1, dtype=np.float32)
    x = lucid.tensor(arr)
    back = lucid.fft.irfft(lucid.fft.rfft(x, norm=norm), n=n, norm=norm)
    assert np.allclose(np.asarray(back.numpy()), arr, atol=1e-4)


@pytest.mark.parametrize("name", ["fft", "ifft", "rfft"])
def test_forward_values_are_unchanged(name) -> None:
    arr = np.arange(1.0, 7.0, dtype=np.float32)
    got = np.asarray(getattr(lucid.fft, name)(lucid.tensor(arr)).numpy())
    expected = getattr(np.fft, name)(arr.astype(np.float64))
    assert np.allclose(got, expected, atol=1e-4)
