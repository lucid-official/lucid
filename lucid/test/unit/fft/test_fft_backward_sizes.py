"""An FFT with ``n`` / ``s`` differentiates back to its input's size.

``s`` resizes each transformed axis before the transform: an axis longer
than its entry is cropped, a shorter one zero-padded.  The gradient has to
undo that resize — the cropped tail of the input received nothing, and the
padding was never an input — but every backward here returned the gradient
at the *transform* size.  The engine then read that buffer as if it had
the input's shape: short (wrong values) or past its end (values that
changed from run to run).  ``x = ones(2, 5)`` through ``real(fft(x, n=4))``
gave ``[[4, 0, 0, 0, 4], [0, 0, 0, 4, 0]]`` where the answer is
``[[4, 0, 0, 0, 0], [4, 0, 0, 0, 0]]``.

The forward values were right throughout, and every case without ``n`` /
``s`` was right, so the defect showed only once a size was asked for.
Custom ``Function`` gradients are now checked against their input's shape,
so the same mistake would be refused rather than read.

The sweep below holds every transform to the reference: the six 1-D
transforms and their 2-D / N-D forms, ``n`` / ``s`` smaller, equal, larger
and odd, each norm, more than one ``dim``, real and complex input, on each
device.

Two inputs the reference accepts were refused outright and are checked
at the end: an integer or boolean signal (promoted to the default float
dtype), and an ``s`` shorter than the rank with ``dim`` omitted (the last
``len(s)`` axes).
"""

import numpy as np
import pytest

import lucid

ONE_D = ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]
MULTI_D = [
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "hfft2",
    "ihfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "hfftn",
    "ihfftn",
]
#: Transforms that take a real signal; the reference refuses complex input.
REAL_ONLY = {"rfft", "ihfft", "rfft2", "ihfft2", "rfftn", "ihfftn"}
NORMS = [None, "forward", "ortho"]


def _data(seed: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    )


def _lucid_grad(name, re, im, kwargs, device, complex_in, weights):
    if complex_in:
        x = lucid.complex(lucid.tensor(re), lucid.tensor(im)).to(device)
        x.requires_grad_(True)
    else:
        x = lucid.tensor(re.copy(), device=device, requires_grad=True)
    out = getattr(lucid.fft, name)(x, **kwargs)
    w_re, w_im = (lucid.tensor(w, device=device) for w in weights(tuple(out.shape)))
    if out.is_complex():
        loss = (lucid.real(out) * w_re + lucid.imag(out) * w_im).sum()
    else:
        loss = (out * w_re).sum()
    loss.backward()
    assert x.grad is not None, "no gradient reached the input"
    g = x.grad.to("cpu")
    if g.is_complex():
        return lucid.real(g).numpy() + 1j * lucid.imag(g).numpy()
    return g.numpy()


def _ref_grad(ref, name, re, im, kwargs, complex_in, weights):
    if complex_in:
        x = ref.complex(ref.tensor(re), ref.tensor(im)).requires_grad_(True)
    else:
        x = ref.tensor(re.copy(), requires_grad=True)
    out = getattr(ref.fft, name)(x, **kwargs)
    w_re, w_im = (ref.tensor(w) for w in weights(tuple(out.shape)))
    if out.is_complex():
        loss = (out.real * w_re + out.imag * w_im).sum()
    else:
        loss = (out * w_re).sum()
    loss.backward()
    return x.grad.detach().resolve_conj().numpy()


def _check(ref, name, shape, kwargs, device, complex_in, seed) -> None:
    re, im = _data(seed, shape)

    def weights(out_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        # A loss that weights every output lane differently, so a gradient
        # in the wrong place cannot cancel out.
        return _data(seed + 1, out_shape)

    got = _lucid_grad(name, re, im, kwargs, device, complex_in, weights)
    want = _ref_grad(ref, name, re, im, kwargs, complex_in, weights)
    assert got.shape == want.shape, (name, kwargs, got.shape, want.shape)
    np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3, err_msg=f"{name} {kwargs}")


# ── The reported case, by its numbers ────────────────────────────────────────


def test_fft_with_a_shorter_n_crops_the_gradient(device: str) -> None:
    x = lucid.ones(2, 5, device=device, requires_grad=True)
    lucid.real(lucid.fft.fft(x, n=4)).sum().backward()
    assert x.grad.tolist() == [[4.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0]]


def test_fft_with_a_longer_n_drops_the_padding(device: str) -> None:
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    lucid.real(lucid.fft.fft(x, n=6)).sum().backward()
    assert x.grad.tolist() == [[6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]


# ── Every transform comes back to the input's shape ──────────────────────────


@pytest.mark.parametrize("name", ONE_D)
@pytest.mark.parametrize("n", [3, 5, 8, 11])
def test_one_d_gradient_has_the_input_shape(name: str, n: int, device: str) -> None:
    x = lucid.randn(3, 5, device=device, requires_grad=True)
    out = getattr(lucid.fft, name)(x, n=n)
    (lucid.abs(out) if out.is_complex() else out).sum().backward()
    assert tuple(x.grad.shape) == (3, 5)


@pytest.mark.parametrize("name", MULTI_D)
@pytest.mark.parametrize("s", [(2, 3), (4, 9), (3, 6)])
def test_multi_d_gradient_has_the_input_shape(name: str, s: tuple[int, int], device: str) -> None:
    x = lucid.randn(2, 3, 6, device=device, requires_grad=True)
    out = getattr(lucid.fft, name)(x, s=s)
    (lucid.abs(out) if out.is_complex() else out).sum().backward()
    assert tuple(x.grad.shape) == (2, 3, 6)


# ── Parity sweep ─────────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("name", ONE_D)
@pytest.mark.parametrize("n", [None, 3, 6, 9, 11], ids=lambda n: f"n{n}")
def test_one_d_gradients_match_the_reference(name: str, n: int | None, device: str, ref) -> None:
    """Input length 6 along the transformed axis: ``n`` shorter, equal,
    longer, odd — and for ``irfft`` / ``hfft`` 6 is a bin count, so the
    same values cover cropping and padding the half-spectrum."""
    complex_cases = [False] if name in REAL_ONLY else [False, True]
    seed = 0
    for norm in NORMS:
        for dim, shape in ((-1, (3, 6)), (0, (6, 4))):
            for complex_in in complex_cases:
                seed += 2
                kwargs = {"n": n, "dim": dim, "norm": norm}
                _check(ref, name, shape, kwargs, device, complex_in, seed)


@pytest.mark.parity
@pytest.mark.parametrize("name", MULTI_D)
@pytest.mark.parametrize("s", [None, (3, 4), (4, 6), (5, 9), (3, 9)], ids=lambda s: f"s{s}")
def test_multi_d_gradients_match_the_reference(name: str, s, device: str, ref) -> None:
    """Input ``(4, 3, 6)``.  The 2-D forms transform the last two axes;
    the N-D forms transform axes ``(0, 2)`` explicitly and, when ``s`` is
    given, the last ``len(s)`` axes when ``dim`` is omitted."""
    complex_cases = [False] if name in REAL_ONLY else [False, True]
    dims: list[object] = [(-2, -1)] if name.endswith("2") else [(0, 2), None]
    seed = 100
    for norm in NORMS:
        for dim in dims:
            for complex_in in complex_cases:
                seed += 2
                kwargs = {"s": s, "dim": dim, "norm": norm}
                _check(ref, name, (4, 3, 6), kwargs, device, complex_in, seed)


# ── Integer input, and an ``s`` shorter than the rank ────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("name", ONE_D)
@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_integer_and_bool_input_is_promoted(name: str, dtype: str, device: str, ref) -> None:
    values = [3, 0, 1, 4, 1, 5]
    if dtype == "bool":
        values = [v % 2 == 1 for v in values]
    got = getattr(lucid.fft, name)(lucid.tensor(values, device=device))
    want = getattr(ref.fft, name)(ref.tensor(values))
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    got_np = got.to("cpu")
    a = (
        lucid.real(got_np).numpy() + 1j * lucid.imag(got_np).numpy()
        if got_np.is_complex()
        else got_np.numpy()
    )
    np.testing.assert_allclose(a, want.resolve_conj().numpy(), atol=1e-4)


@pytest.mark.parity
@pytest.mark.parametrize("name", ["fftn", "ifftn", "rfftn", "irfftn", "hfftn", "ihfftn"])
def test_a_short_s_without_dim_names_the_last_axes(name: str, device: str, ref) -> None:
    re, _ = _data(7, (2, 3, 4))
    got = getattr(lucid.fft, name)(lucid.tensor(re, device=device), s=(3, 5))
    want = getattr(ref.fft, name)(ref.tensor(re), s=(3, 5))
    assert tuple(got.shape) == tuple(want.shape)
    got_np = got.to("cpu")
    a = (
        lucid.real(got_np).numpy() + 1j * lucid.imag(got_np).numpy()
        if got_np.is_complex()
        else got_np.numpy()
    )
    np.testing.assert_allclose(a, want.resolve_conj().numpy(), atol=1e-4)


def test_an_s_longer_than_the_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds the input's 2 dimensions"):
        lucid.fft.fftn(lucid.ones(2, 3), s=(1, 2, 3))
