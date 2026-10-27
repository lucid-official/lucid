"""The window functions, checked against their closed forms.

``signal/windows.py`` sat at 48.5%.  What was dark was not exotic: the
``sym=False`` periodic form of every window, the degenerate lengths, and
``exponential`` entirely.

Windows are multiplied into a frame before it reaches an FFT, so a wrong
one does not fail — it attenuates.  A length-1 Bartlett that returns
``0.0`` instead of ``1.0`` deletes its frame, and the spectrum that comes
back is a perfectly well-formed spectrum of nothing.

Each window is checked against the cosine sum or exponential that
defines it, written out here rather than taken from the implementation.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.signal.windows as W

LENGTHS = [2, 3, 8, 33, 64]


def _v(x):
    return np.asarray(x.numpy())


def _n(M, sym):
    """The sample positions the definitions are stated over.

    A periodic window of length ``M`` is the first ``M`` samples of the
    symmetric window of length ``M + 1`` — that is what makes it tile.
    """
    N = M if sym else M + 1
    return np.arange(N, dtype=np.float64), N


def _cosine_sum(M, sym, coeffs):
    n, N = _n(M, sym)
    w = np.zeros_like(n)
    for k, a in enumerate(coeffs):
        w += a * np.cos(2.0 * np.pi * k * n / (N - 1)) * (-1) ** k
    return w[:M]


# ── the cosine-sum family ─────────────────────────────────────────────────────


COSINE_WINDOWS = {
    "hann": [0.5, 0.5],
    "hamming": [0.54, 0.46],
    "blackman": [0.42, 0.5, 0.08],
    "nuttall": [0.3635819, 0.4891775, 0.1365995, 0.0106411],
}


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("name", sorted(COSINE_WINDOWS))
def test_the_cosine_sum_windows_match_their_definition(name, M, sym):
    got = _v(getattr(W, name)(M, sym=sym))
    assert got.shape == (M,)
    assert np.allclose(got, _cosine_sum(M, sym, COSINE_WINDOWS[name]), atol=1e-6)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
def test_bartlett_is_the_triangle(M, sym):
    n, N = _n(M, sym)
    want = (1.0 - np.abs(2.0 * n / (N - 1) - 1.0))[:M]
    assert np.allclose(_v(W.bartlett(M, sym=sym)), want, atol=1e-6)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
def test_cosine_is_a_half_period_sine(M, sym):
    n, N = _n(M, sym)
    want = np.sin(np.pi * (n + 0.5) / N)[:M]
    assert np.allclose(_v(W.cosine(M, sym=sym)), want, atol=1e-6)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("std", [1.0, 3.0, 7.0])
def test_gaussian_is_the_bell(M, sym, std):
    n, N = _n(M, sym)
    want = np.exp(-0.5 * ((n - (N - 1) / 2.0) / std) ** 2)[:M]
    assert np.allclose(_v(W.gaussian(M, std=std, sym=sym)), want, atol=1e-6)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("tau", [1.0, 3.0, 10.0])
def test_exponential_decays_from_the_centre(M, sym, tau):
    """The periodic form used to be refused outright — ``exponential``
    was the one window in the module with no ``sym=False`` at all, on
    the grounds that it had no canonical centre.  It has the same one
    the symmetric form uses, on the working length."""
    n, N = _n(M, sym)
    want = np.exp(-np.abs(n - (N - 1) / 2.0) / tau)[:M]
    assert np.allclose(_v(W.exponential(M, tau=tau, sym=sym)), want, atol=1e-6)


def test_gaussian_default_std_is_one():
    assert np.allclose(_v(W.gaussian(8)), _v(W.gaussian(8, std=1.0)), atol=1e-6)


def test_exponential_accepts_an_explicit_centre():
    got = _v(W.exponential(9, center=2.0, tau=2.0, sym=False))
    want = np.exp(-np.abs(np.arange(9.0) - 2.0) / 2.0)
    assert np.allclose(got, want, atol=1e-6)
    assert np.argmax(got) == 2


@pytest.mark.parametrize("M", [0, 1, 9])
def test_exponential_refuses_explicit_centre_when_symmetric(M):
    with pytest.raises(ValueError, match="center must be None for symmetric windows"):
        W.exponential(M, center=2.0, sym=True)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("beta", [0.0, 8.6, 14.0])
def test_kaiser_is_the_bessel_ratio(M, sym, beta):
    n, N = _n(M, sym)
    alpha = (N - 1) / 2.0
    ratio = (n - alpha) / alpha
    want = (np.i0(beta * np.sqrt(1.0 - ratio**2)) / np.i0(beta))[:M]
    assert np.allclose(_v(W.kaiser(M, beta=beta, sym=sym)), want, atol=1e-5)


def test_a_zero_beta_kaiser_is_rectangular():
    assert np.allclose(_v(W.kaiser(16, beta=0.0)), 1.0, atol=1e-6)


# ── the parameterised generalisations ─────────────────────────────────────────


def test_general_hamming_reproduces_hamming_and_hann():
    """The two named windows are the same family at two alphas; if the
    generalisation disagrees with them, one of the three is wrong."""
    assert np.allclose(_v(W.general_hamming(32)), _v(W.hamming(32)), atol=1e-6)
    assert np.allclose(_v(W.general_hamming(32, alpha=0.54)), _v(W.hamming(32)), atol=1e-6)
    assert np.allclose(_v(W.general_hamming(32, alpha=0.5)), _v(W.hann(32)), atol=1e-6)


def test_general_cosine_reproduces_blackman():
    assert np.allclose(_v(W.general_cosine(32, [0.42, 0.5, 0.08])), _v(W.blackman(32)), atol=1e-6)


def test_general_cosine_reproduces_nuttall():
    assert np.allclose(
        _v(W.general_cosine(32, COSINE_WINDOWS["nuttall"])), _v(W.nuttall(32)), atol=1e-6
    )


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_general_gaussian_at_p_one_is_gaussian(p):
    got = _v(W.general_gaussian(32, p=p, sig=5.0))
    n = np.arange(32.0) - 31 / 2.0
    want = np.exp(-0.5 * np.abs(n / 5.0) ** (2 * p))
    assert np.allclose(got, want, atol=1e-6)
    if p == 1.0:
        assert np.allclose(got, _v(W.gaussian(32, std=5.0)), atol=1e-6)


# ── shape properties every window shares ──────────────────────────────────────


ALL_WINDOWS = [
    ("bartlett", {}),
    ("blackman", {}),
    ("cosine", {}),
    ("exponential", {"tau": 3.0}),
    ("gaussian", {"std": 3.0}),
    ("general_gaussian", {"p": 1.5, "sig": 5.0}),
    ("general_hamming", {"alpha": 0.6}),
    ("hamming", {}),
    ("hann", {}),
    ("kaiser", {"beta": 8.6}),
    ("nuttall", {}),
]


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
@pytest.mark.parametrize("sym", [True, False])
def test_every_window_returns_the_requested_length(name, kw, sym):
    for M in LENGTHS:
        assert _v(getattr(W, name)(M, sym=sym, **kw)).shape == (M,)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_a_symmetric_window_is_symmetric(name, kw):
    got = _v(getattr(W, name)(33, sym=True, **kw))
    assert np.allclose(got, got[::-1], atol=1e-6)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_a_periodic_window_is_the_head_of_the_longer_symmetric_one(name, kw):
    """That is the definition of the periodic form, and the property
    that makes overlap-add reconstruct."""
    periodic = _v(getattr(W, name)(32, sym=False, **kw))
    symmetric = _v(getattr(W, name)(33, sym=True, **kw))
    assert np.allclose(periodic, symmetric[:32], atol=1e-6)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
@pytest.mark.parametrize("sym", [True, False])
def test_a_single_sample_window_carries_no_taper(name, kw, sym):
    """``M <= 1`` has no shape to taper, so it is ``ones(M)``.

    The guard used to read the *working* length rather than the
    requested one.  ``sym=False`` computes ``M + 1`` samples and keeps
    the first, so ``M = 1`` slipped past and came back as whatever the
    leading sample of a two-sample window happened to be: ``0.0`` for a
    Bartlett or a Hann — which deletes the frame it was meant to taper —
    and ``0.08`` for a Hamming, worse for being plausible.
    """
    assert np.allclose(_v(getattr(W, name)(1, sym=sym, **kw)), 1.0)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
@pytest.mark.parametrize("sym", [True, False])
def test_a_zero_length_window_is_empty(name, kw, sym):
    assert _v(getattr(W, name)(0, sym=sym, **kw)).shape == (0,)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_a_negative_length_is_refused(name, kw):
    with pytest.raises(ValueError, match=">= 0"):
        getattr(W, name)(-1, **kw)


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_the_taper_never_exceeds_one_or_goes_negative(name, kw):
    got = _v(getattr(W, name)(64, **kw))
    assert got.max() <= 1.0 + 1e-6
    assert got.min() >= -1e-6


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_an_odd_symmetric_window_reaches_one_at_its_centre(name, kw):
    """Only at odd lengths — an even-length window straddles the peak
    and its largest sample sits just below it."""
    got = _v(getattr(W, name)(65, **kw))
    assert np.isclose(got[32], 1.0, atol=1e-5)
    assert np.argmax(got) == 32
    assert _v(getattr(W, name)(64, **kw)).max() < 1.0


@pytest.mark.parametrize("name,kw", ALL_WINDOWS, ids=[c[0] for c in ALL_WINDOWS])
def test_the_dtype_and_device_are_honoured(name, kw):
    got = getattr(W, name)(16, dtype=lucid.float64, **kw)
    assert got.dtype is lucid.float64
    assert got.device.type == "cpu"


# ── the property that makes a window worth applying ───────────────────────────


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "nuttall"])
def test_tapering_suppresses_spectral_leakage(name):
    """A sinusoid at a non-integer bin smears across the spectrum of a
    rectangular frame.  Every one of these windows exists to reduce
    that, so each has to beat the untapered frame on far-field energy.
    """
    n = np.arange(256)
    signal = np.sin(2 * np.pi * (10.5 / 256) * n)
    tapered = signal * _v(getattr(W, name)(256, sym=False))

    def far_field(x):
        mag = np.abs(np.fft.rfft(x))
        return mag[40:].sum() / mag.sum()

    assert far_field(tapered) < far_field(signal)


def test_the_windows_are_ordered_by_how_hard_they_taper():
    """Nuttall tapers hardest, then Blackman, then Hann — the ordering
    the coefficient tables exist to produce."""
    energy = {
        name: _v(getattr(W, name)(64)).sum() for name in ("nuttall", "blackman", "hann", "hamming")
    }
    assert energy["nuttall"] < energy["blackman"] < energy["hann"] < energy["hamming"]


def test_the_module_exports_what_it_documents():
    assert math.isclose(float(_v(W.hann(8)).sum()), 3.5, abs_tol=1e-6)
