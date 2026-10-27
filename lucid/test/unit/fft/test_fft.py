"""``lucid.fft`` — DFT / Hermitian / shift / freq."""

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine


class TestFFT:
    def test_constant_input(self) -> None:
        # FFT of a real constant signal — only DC bin is non-zero.
        t = lucid.tensor([1.0, 1.0, 1.0, 1.0])
        out = lucid.fft.fft(t).numpy()
        assert abs(out[0]) > 0
        for i in range(1, 4):
            assert abs(out[i]) < 1e-5

    def test_inverse_round_trip(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = lucid.fft.ifft(lucid.fft.fft(t)).numpy()
        # Imaginary part should vanish; real part must match.
        np.testing.assert_allclose(out.real, t.numpy(), atol=1e-4)


class TestRFFT:
    def test_round_trip(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = lucid.fft.irfft(lucid.fft.rfft(t), n=8).numpy()
        np.testing.assert_allclose(out, t.numpy(), atol=1e-4)


class TestFFT2:
    def test_basic_shape(self) -> None:
        t = lucid.zeros(4, 8)
        out = lucid.fft.fft2(t)
        assert out.shape == t.shape


class TestFFTShift:
    def test_basic(self) -> None:
        t = lucid.tensor([0.0, 1.0, 2.0, 3.0])
        out = lucid.fft.fftshift(t).numpy()
        np.testing.assert_array_equal(out, [2.0, 3.0, 0.0, 1.0])


class TestFreqs:
    def test_fftfreq(self) -> None:
        out = lucid.fft.fftfreq(4).numpy()
        # Standard fftfreq for n=4 is [0, 0.25, -0.5, -0.25].
        np.testing.assert_allclose(out, [0.0, 0.25, -0.5, -0.25], atol=1e-6)

    def test_rfftfreq(self) -> None:
        out = lucid.fft.rfftfreq(4).numpy()
        # For n=4 rfftfreq → [0, 0.25, 0.5].
        np.testing.assert_allclose(out, [0.0, 0.25, 0.5], atol=1e-6)


class TestIHFFT:
    """``ihfft`` once raised ``broadcast: dtype not supported`` for every
    real input — the whole family had never produced a value.  These pin
    the values, not just the fact that a call returns."""

    def test_matches_the_reference_1d(self) -> None:
        a = np.arange(8, dtype=np.float32) - 3.5
        out = lucid.fft.ihfft(lucid.tensor(a)).numpy()
        np.testing.assert_allclose(out, np.fft.ihfft(a.astype(np.float64)), atol=1e-6)

    def test_every_norm(self) -> None:
        a = np.arange(8, dtype=np.float32)
        for norm in ("backward", "ortho", "forward"):
            out = lucid.fft.ihfft(lucid.tensor(a), norm=norm).numpy()
            np.testing.assert_allclose(
                out, np.fft.ihfft(a.astype(np.float64), norm=norm), atol=1e-6
            )

    def test_ihfft2_and_ihfftn(self) -> None:
        # The reference has no ihfft2/ihfftn, so use the identity they are
        # defined by: ihfft(x) = conj(rfft(x)) / N.
        a = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
        expected = np.conj(np.fft.rfft2(a.astype(np.float64))) / a.size
        for fn in (lucid.fft.ihfft2, lucid.fft.ihfftn):
            np.testing.assert_allclose(fn(lucid.tensor(a)).numpy(), expected, atol=1e-6)

    def test_hermitian_round_trip(self) -> None:
        a = np.arange(8, dtype=np.float32) - 3.5
        spectrum = lucid.fft.ihfft(lucid.tensor(a))
        np.testing.assert_allclose(lucid.fft.hfft(spectrum, n=8).numpy(), a, atol=1e-4)


# ── Double-precision CPU FFT / RFFT / HFFT parity (LCD-458) ──────────────────

_F64_LENGTHS = [1, 2, 3, 5, 8, 12, 97, 1024]
_F64_NORMS = [None, "backward", "ortho", "forward"]


@pytest.mark.parity
@pytest.mark.parametrize("n", _F64_LENGTHS)
@pytest.mark.parametrize("norm", _F64_NORMS)
def test_f64_and_c128_1d_fft_family_matches_reference(n: int, norm: str | None, ref) -> None:
    rng = np.random.default_rng(n + 17)
    re = rng.standard_normal(n).astype(np.float64)
    im = rng.standard_normal(n).astype(np.float64)
    xr = lucid.tensor(re)
    xc = lucid.complex(xr, lucid.tensor(im))
    txr = ref.tensor(re)
    txc = ref.complex(txr, ref.tensor(im))

    for fn_name in ("fft", "ifft"):
        for l_in, t_in in ((xr, txr), (xc, txc)):
            got = getattr(lucid.fft, fn_name)(l_in, norm=norm)
            want = getattr(ref.fft, fn_name)(t_in, norm=norm)
            assert got.dtype == lucid.complex128
            np.testing.assert_allclose(
                got.numpy(), want.resolve_conj().numpy(), rtol=1e-11, atol=1e-11
            )

    for fn_name in ("rfft", "ihfft"):
        got = getattr(lucid.fft, fn_name)(xr, norm=norm)
        want = getattr(ref.fft, fn_name)(txr, norm=norm)
        assert got.dtype == lucid.complex128
        np.testing.assert_allclose(got.numpy(), want.resolve_conj().numpy(), rtol=1e-11, atol=1e-11)

    half_len = n // 2 + 1
    h_re = rng.standard_normal(half_len).astype(np.float64)
    h_im = rng.standard_normal(half_len).astype(np.float64)
    lh = lucid.complex(lucid.tensor(h_re), lucid.tensor(h_im))
    th = ref.complex(ref.tensor(h_re), ref.tensor(h_im))
    for fn_name in ("irfft", "hfft"):
        got = getattr(lucid.fft, fn_name)(lh, n=n, norm=norm)
        want = getattr(ref.fft, fn_name)(th, n=n, norm=norm)
        assert got.dtype == lucid.float64
        np.testing.assert_allclose(got.numpy(), want.resolve_conj().numpy(), rtol=1e-11, atol=1e-11)


@pytest.mark.parity
@pytest.mark.parametrize("s", [(3, 5), (8, 12), (5, 7)])
@pytest.mark.parametrize("dim", [(0, 1), (-1, 0), (1, 2)])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_f64_and_c128_nd_fft_and_gradients_match_reference(
    s: tuple[int, int], dim: tuple[int, int], norm: str, ref
) -> None:
    rng = np.random.default_rng(42)
    re = rng.standard_normal((4, 6, 8)).astype(np.float64)
    im = rng.standard_normal((4, 6, 8)).astype(np.float64)

    for fn_name in ("fftn", "ifftn", "rfftn", "irfftn", "hfftn", "ihfftn"):
        complex_in = fn_name in ("ifftn", "irfftn", "hfftn")
        if complex_in:
            lx = lucid.complex(lucid.tensor(re), lucid.tensor(im)).requires_grad_(True)
            tx = ref.complex(ref.tensor(re), ref.tensor(im)).requires_grad_(True)
        else:
            lx = lucid.tensor(re.copy(), requires_grad=True)
            tx = ref.tensor(re.copy(), requires_grad=True)

        ly = getattr(lucid.fft, fn_name)(lx, s=s, dim=dim, norm=norm)
        ty = getattr(ref.fft, fn_name)(tx, s=s, dim=dim, norm=norm)
        expected_dt = lucid.float64 if fn_name in ("irfftn", "hfftn") else lucid.complex128
        assert ly.dtype == expected_dt
        np.testing.assert_allclose(
            ly.numpy(), ty.detach().resolve_conj().numpy(), rtol=1e-11, atol=1e-11
        )

        w_re = rng.standard_normal(tuple(ly.shape)).astype(np.float64)
        w_im = rng.standard_normal(tuple(ly.shape)).astype(np.float64)
        if ly.is_complex():
            l_loss = (
                lucid.real(ly) * lucid.tensor(w_re) + lucid.imag(ly) * lucid.tensor(w_im)
            ).sum()
            t_loss = (ty.real * ref.tensor(w_re) + ty.imag * ref.tensor(w_im)).sum()
        else:
            l_loss = (ly * lucid.tensor(w_re)).sum()
            t_loss = (ty * ref.tensor(w_re)).sum()
        l_loss.backward()
        t_loss.backward()
        assert lx.grad is not None
        np.testing.assert_allclose(
            lx.grad.numpy(), tx.grad.resolve_conj().numpy(), rtol=1e-11, atol=1e-11
        )


def test_non_positive_fft_length_raises_invalid_argument() -> None:
    for dtype in (lucid.float32, lucid.float64):
        x = lucid.ones(4, dtype=dtype)
        for bad_n in (0, -3):
            with pytest.raises(_C_engine.InvalidArgument):
                lucid.fft.fft(x, n=bad_n)
            with pytest.raises(_C_engine.InvalidArgument):
                lucid.fft.rfft(x, n=bad_n)
            with pytest.raises(_C_engine.InvalidArgument):
                lucid.fft.irfft(lucid.fft.rfft(x), n=bad_n)
