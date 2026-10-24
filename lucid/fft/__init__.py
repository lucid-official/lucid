"""lucid.fft — discrete Fourier transform ops.

Mirrors the standard FFT surface: 18 transform variants
(``fft`` / ``ifft`` / ``fft2`` / ``ifft2`` / ``fftn`` / ``ifftn`` and the
``r``- and ``h``-prefixed counterparts) plus the four utility helpers
``fftfreq`` / ``rfftfreq`` / ``fftshift`` / ``ifftshift``.

Backend: every transform delegates to ``mlx::core::fft`` through the four
engine primitives ``_C_engine.fft.{fftn,ifftn,rfftn,irfftn}``.  MLX
itself wraps Apple Accelerate vDSP under the hood, so both ``Device::CPU``
and ``Device::GPU`` tensors land on the same fast CPU FFT kernel — see
``lucid/_C/ops/fft/_Detail.h`` for the carve-out rationale.

Normalisation: matches the standard convention.  ``norm='backward'`` (the
default) leaves ``fft`` unscaled and divides ``ifft`` by ``N``;
``norm='forward'`` swaps those; ``norm='ortho'`` divides both by
``sqrt(N)``.  MLX's transforms always produce the ``'backward'`` form,
so non-default norms are reached by post-multiplying the result.

Autograd: the base ops are ``lucid.autograd.Function`` subclasses, in
``_transforms``.  The backward of every transform is the *dual*
transform with the *dual* normalisation (``backward ↔ forward``,
``ortho ↔ ortho``), taken at the transform size and then cropped or
zero-padded back to the input's size along every transformed axis — the
adjoint of what ``n`` / ``s`` does to the input.  The dual is itself
taken through its ``Function``, so a gradient differentiates again under
``create_graph``.

Input: an integer or boolean tensor is promoted to the default floating
dtype, and with ``dim`` omitted an ``s`` shorter than the rank names the
last ``len(s)`` axes.
"""

from typing import Sequence, TYPE_CHECKING

import lucid
from lucid._types import DeviceLike, DTypeLike
from lucid.fft._transforms import (
    _FftnAutograd,
    _IrfftnAutograd,
    _RfftnAutograd,
    _as_axis_list,
    _check_norm,
    _conj,
    _dual_norm,
    _input_sizes_along_axes,
    _normalise_axes,
    _prepare,
    _promote,
    _transform,
    _transform_size,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Public API: complex FFT (fft / fft2 / fftn) ──────────────────────────────


def fftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional discrete Fourier transform.

    Computes the N-dimensional discrete Fourier transform (DFT) by applying
    1-D FFTs over each of the specified axes in succession.  For a single
    axis of length :math:`N`, the DFT is defined as:

    .. math::

        X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i 2\pi k n / N},
        \quad k = 0, 1, \ldots, N-1.

    For multiple axes the transforms are applied sequentially, one axis at
    a time (the final result is equivalent to a single multi-dimensional
    transform).

    The output is always a complex tensor (``complex64``).  Its shape is
    identical to ``input`` except along the transformed axes, where axis
    ``i`` has length ``s[i]`` when ``s`` is provided (the input is
    zero-padded or truncated to that size first).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.  May be real or complex; an integer or
        boolean input is promoted to the default floating dtype first.
    s : int or sequence of int, optional
        Signal length(s) along each transformed axis.  When given, each
        axis is zero-padded (if ``s[i] > input.shape[dim[i]]``) or
        truncated (if smaller) before the transform.  ``len(s)`` must
        equal ``len(dim)`` when both are sequences.  If ``None``, each
        axis is transformed at its current size.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the transform.  Negative
        indices are supported.  ``None`` (the default) means every axis,
        or the last ``len(s)`` axes when ``s`` is given.
    norm : str or None, optional
        Normalisation mode.  One of:

        - ``"backward"`` (default, same as ``None``) — forward transform
          is unscaled; inverse divides by :math:`N`.
        - ``"forward"`` — forward transform divides by :math:`N`; inverse
          is unscaled.
        - ``"ortho"`` — both forward and inverse divide by
          :math:`\sqrt{N}`, making the transforms mutually unitary.

        Here :math:`N` is the product of the lengths of all transformed
        axes (after padding/truncation).

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that each transformed axis ``i`` has length ``s[i]`` (or
        ``input.shape[dim[i]]`` when ``s`` is ``None``).

    Notes
    -----
    **Frequency ordering** — the output follows the standard FFT ordering.
    For an axis of length :math:`N`, the output bins represent frequencies:

    .. math::

        f_k = \frac{k}{N}, \quad k = 0, 1, \ldots, N-1.

    Bins :math:`k = 0, \ldots, \lfloor N/2 \rfloor` are the non-negative
    frequencies; bins :math:`k = \lfloor N/2 \rfloor + 1, \ldots, N-1`
    wrap around and represent negative frequencies
    :math:`(k-N)/N`.  Use :func:`fftshift` to reorder so that the
    zero-frequency bin sits at the centre of each axis.

    **Autograd** — the backward pass computes
    :math:`\text{grad}_x = \text{ifft}(\text{grad}_{\text{out}},
    \text{dual}(\text{norm}))` where ``dual`` swaps ``"backward"`` and
    ``"forward"`` and leaves ``"ortho"`` unchanged.

    Examples
    --------
    1-D DFT of a length-8 signal:

    >>> x = lucid.fft.fft(lucid.ones(8))
    >>> x.shape
    (8,)

    2-D DFT over the last two axes with explicit output sizes:

    >>> x = lucid.ones(4, 4)
    >>> X = lucid.fft.fftn(x, s=[8, 8], dim=[-2, -1])
    >>> X.shape
    (8, 8)

    Orthonormal convention (``"ortho"``) makes the transform unitary:

    >>> x = lucid.randn(16)
    >>> X = lucid.fft.fftn(x, norm="ortho")
    >>> # lucid.fft.ifftn(X, norm="ortho") recovers x exactly
    """
    norm_v = _check_norm(norm)
    input, axes, s_list = _prepare(input, s, dim, "fftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    N = _transform_size(s_list, in_sizes)
    return _transform(_FftnAutograd, input, s_list, axes, norm_v, N, False)


def ifftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional inverse discrete Fourier transform.

    Recovers the original signal from its frequency-domain representation
    produced by :func:`fftn`.  For a single axis of length :math:`N`, the
    inverse DFT is:

    .. math::

        x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{+i 2\pi k n / N},
        \quad n = 0, 1, \ldots, N-1.

    The :math:`1/N` factor shown above corresponds to ``norm='backward'``
    (the default).  With ``norm='forward'`` the factor becomes 1 (unscaled),
    and with ``norm='ortho'`` it becomes :math:`1/\sqrt{N}`.

    Parameters
    ----------
    input : Tensor
        Complex input tensor (frequency domain), of any shape.
    s : int or sequence of int, optional
        Output signal length(s) along each transformed axis.  The
        frequency-domain tensor is zero-padded or truncated to match
        before inverting.  ``len(s)`` must equal ``len(dim)`` when both
        are sequences.  If ``None``, each axis keeps its current size.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the inverse transform.
        Negative indices are supported.  ``None`` (the default) means
        every axis, or the last ``len(s)`` axes when ``s`` is given.
    norm : str or None, optional
        Normalisation mode.  One of:

        - ``"backward"`` (default, same as ``None``) — divides by
          :math:`N` (standard inverse-DFT convention).
        - ``"forward"`` — no division; assumes the forward transform
          divided by :math:`N`.
        - ``"ortho"`` — divides by :math:`\sqrt{N}`, matching
          :func:`fftn` with ``norm='ortho'``.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that each transformed axis ``i`` has length ``s[i]`` (or
        the input size when ``s`` is ``None``).

    Notes
    -----
    **Round-trip property** — for any complex tensor ``x``:

    .. math::

        \text{ifftn}(\text{fftn}(x, \text{norm}=m), \text{norm}=m) \approx x

    up to floating-point rounding, for any valid ``norm`` mode ``m``.

    **Conjugate symmetry** — if the input to :func:`fftn` was a real
    tensor, its DFT satisfies :math:`X[N-k] = X^*[k]`.  In that
    case :func:`irfftn` is preferred because it exploits this symmetry
    to produce a real output more efficiently.

    Examples
    --------
    Round-trip through the frequency domain:

    >>> x = lucid.randn(32)
    >>> X = lucid.fft.fftn(x)
    >>> x_rec = lucid.fft.ifftn(X)
    >>> # x_rec.real ≈ x  (imaginary part is ≈ 0 for a real input)

    Inverse of a 2-D transform:

    >>> X = lucid.fft.fftn(lucid.ones(8, 8))
    >>> x = lucid.fft.ifftn(X)
    >>> x.shape
    (8, 8)
    """
    norm_v = _check_norm(norm)
    input, axes, s_list = _prepare(input, s, dim, "ifftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    N = _transform_size(s_list, in_sizes)
    return _transform(_FftnAutograd, input, s_list, axes, norm_v, N, True)


def fft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D discrete Fourier transform along a single axis.

    Computes the one-dimensional DFT of the input tensor along the axis
    ``dim``.  For an input sequence :math:`x[0], x[1], \ldots, x[N-1]`,
    the output is:

    .. math::

        X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i 2\pi k n / N},
        \quad k = 0, 1, \ldots, N-1.

    The output bin :math:`X[0]` is the DC component (sum of all samples);
    :math:`X[1]` through :math:`X[\lfloor N/2 \rfloor]` are the positive
    frequencies; :math:`X[\lfloor N/2 \rfloor + 1]` through
    :math:`X[N-1]` are the negative frequencies.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.  May be real or complex.
    n : int, optional
        Length of the transformed axis.  If larger than the input axis,
        the input is zero-padded; if smaller, it is truncated.  If
        ``None``, the axis length is used as-is.
    dim : int, optional
        Axis over which to compute the transform.  Default is ``-1``
        (last axis).  Negative indices are supported.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that axis ``dim`` has length ``n`` (or the original length
        when ``n`` is ``None``).

    Notes
    -----
    **Computational efficiency** — under the hood this calls :func:`fftn`
    restricted to one axis, which in turn delegates to the MLX/Accelerate
    FFT kernel.  For a length-:math:`N` axis where :math:`N` is a power
    of two, the Cooley–Tukey radix-2 algorithm is used, reducing the
    :math:`O(N^2)` DFT to :math:`O(N \log N)`.

    **DC and Nyquist** — for an even-length transform of a real signal:

    - ``X[0]`` is real and equals the mean of ``x`` times :math:`N`.
    - ``X[N//2]`` is real and is the Nyquist component.
    - All other bins come in conjugate pairs:
      ``X[N-k] == conj(X[k])``.

    Use :func:`rfft` to exploit this conjugate symmetry and halve the
    output size.

    Examples
    --------
    DFT of a pure cosine at the third harmonic:

    >>> N = 16
    >>> k0 = 3
    >>> n = lucid.arange(N)
    >>> # x[n] = cos(2π k0 n / N) — should spike at bins k0 and N-k0
    >>> x = lucid.cos(n * (2 * 3.14159 * k0 / N))
    >>> X = lucid.fft.fft(x)
    >>> X.shape
    (16,)

    Zero-padding to the next power of two for spectral interpolation:

    >>> x = lucid.randn(100)
    >>> X = lucid.fft.fft(x, n=128)   # pads from 100 → 128
    >>> X.shape
    (128,)
    """
    return fftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def ifft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D inverse discrete Fourier transform along a single axis.

    Recovers the time-domain signal from its frequency-domain
    representation computed by :func:`fft`.  For a complex frequency
    sequence :math:`X[0], \ldots, X[N-1]`, the inverse DFT is:

    .. math::

        x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{+i 2\pi k n / N},
        \quad n = 0, 1, \ldots, N-1.

    The :math:`1/N` factor above corresponds to ``norm='backward'``
    (default).  With ``norm='ortho'`` both :func:`fft` and :func:`ifft`
    divide by :math:`\sqrt{N}`, making the pair a unitary transform.

    Parameters
    ----------
    input : Tensor
        Complex input tensor (frequency domain) of any shape.
    n : int, optional
        Length of the output (time-domain) axis.  The frequency-domain
        axis is zero-padded or truncated to match before inverting.
        If ``None``, the axis keeps its current length.
    dim : int, optional
        Axis over which to compute the inverse transform.  Default is
        ``-1`` (last axis).
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that axis ``dim`` has length ``n`` (or the input length
        when ``n`` is ``None``).

    Notes
    -----
    **Round-trip** — for any complex tensor ``x``:

    .. math::

        \text{ifft}(\text{fft}(x)) \approx x

    up to floating-point rounding.

    **Real inputs** — if ``input`` represents the spectrum of a real
    signal (conjugate-symmetric), the imaginary part of the output will
    be negligibly small.  Use :func:`irfft` to obtain a strictly real
    output more efficiently when you know the input is conjugate-symmetric.

    Examples
    --------
    Round-trip reconstruction:

    >>> x = lucid.randn(64)
    >>> X = lucid.fft.fft(x)
    >>> x_rec = lucid.fft.ifft(X)
    >>> x_rec.shape
    (64,)
    >>> # x_rec.real ≈ x

    Deconvolution in the frequency domain:

    >>> H = lucid.fft.fft(lucid.randn(32))   # frequency response
    >>> Y = lucid.fft.fft(lucid.randn(32))   # observed output spectrum
    >>> x_est = lucid.fft.ifft(Y / H)        # estimated input
    """
    return ifftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def fft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    r"""2-D discrete Fourier transform over a pair of axes.

    Computes the two-dimensional DFT of the input tensor by applying
    :func:`fft` along the two axes specified by ``dim`` (default: the
    last two axes).  For an :math:`M \times N` input array, the 2-D DFT
    is:

    .. math::

        X[p, q] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1}
                  x[m, n]\, e^{-i 2\pi (pm/M + qn/N)},

    where :math:`p = 0, \ldots, M-1` and :math:`q = 0, \ldots, N-1`.

    This is equivalent to calling ``fftn(input, s=s, dim=dim, norm=norm)``
    with a two-element ``dim`` — it is provided as a convenience for the
    common case of 2-D signals such as images or spatial fields.

    Parameters
    ----------
    input : Tensor
        Input tensor with at least 2 dimensions.  May be real or complex.
    s : int or sequence of int, optional
        Output size ``(M_out, N_out)`` along each transformed axis.
        The input is zero-padded or truncated to this size before the
        transform.  ``len(s)`` must match ``dim``.  If ``None``, the
        axes are transformed at their current sizes.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` (the last two) by default; a
        single int transforms that one axis.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  :math:`N` in the scaling formula is the product
        :math:`M_{\text{out}} \times N_{\text{out}}`.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that the two transformed axes have lengths ``s[0]`` and
        ``s[1]`` (or the original axis sizes when ``s`` is ``None``).

    Notes
    -----
    **Spatial frequency interpretation** — in image processing, the
    2-D DFT maps a spatial image :math:`x[m, n]` to its spatial-frequency
    content :math:`X[p, q]`.  The bin :math:`(p, q) = (0, 0)` is the
    DC component (average pixel value times :math:`MN`).  Radial distance
    from the origin in the frequency plane corresponds to spatial frequency
    magnitude; :func:`fftshift` is typically applied before displaying the
    spectrum so that low frequencies sit at the centre.

    **Separability** — the 2-D DFT is separable: the row-wise 1-D DFTs
    followed by the column-wise 1-D DFTs yield the same result as a
    single 2-D DFT.  The engine exploits this to run two passes of the
    1-D algorithm.

    Examples
    --------
    2-D DFT of a real image-like tensor:

    >>> x = lucid.randn(64, 64)
    >>> X = lucid.fft.fft2(x)
    >>> X.shape
    (64, 64)

    Zero-pad to a larger grid for spectral interpolation:

    >>> X_padded = lucid.fft.fft2(x, s=[128, 128])
    >>> X_padded.shape
    (128, 128)

    Centre the spectrum for visualisation:

    >>> X_shifted = lucid.fft.fftshift(lucid.fft.fft2(x))
    """
    return fftn(input, s=s, dim=dim, norm=norm)


def ifft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    r"""2-D inverse discrete Fourier transform over a pair of axes.

    Recovers the 2-D spatial-domain signal from its frequency-domain
    representation produced by :func:`fft2`.  For an :math:`M \times N`
    frequency array, the 2-D inverse DFT is:

    .. math::

        x[m, n] = \frac{1}{MN} \sum_{p=0}^{M-1} \sum_{q=0}^{N-1}
                  X[p, q]\, e^{+i 2\pi (pm/M + qn/N)}.

    The :math:`1/(MN)` factor corresponds to ``norm='backward'`` (default).

    Parameters
    ----------
    input : Tensor
        Complex input tensor (2-D frequency domain) with at least 2
        dimensions.
    s : int or sequence of int, optional
        Output size ``(M_out, N_out)`` along each transformed axis.
        The frequency tensor is zero-padded or truncated before inverting.
        ``len(s)`` must match ``dim``.  If ``None``, the axes keep
        their current sizes.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` by default; a single int
        transforms that one axis.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that the two transformed axes have lengths ``s[0]`` and
        ``s[1]`` (or the input sizes when ``s`` is ``None``).

    Notes
    -----
    **Round-trip** — for any 2-D complex tensor ``x``:

    .. math::

        \text{ifft2}(\text{fft2}(x)) \approx x

    **Real outputs** — if ``input`` is the 2-D DFT of a real image
    (conjugate-symmetric in 2-D), the imaginary part of the output will
    be negligibly small.  Use :func:`irfft2` for strictly real output.

    Examples
    --------
    Round-trip through the frequency domain:

    >>> x = lucid.randn(32, 32)
    >>> X = lucid.fft.fft2(x)
    >>> x_rec = lucid.fft.ifft2(X)
    >>> x_rec.shape
    (32, 32)

    Low-pass filtering in the frequency domain:

    >>> X = lucid.fft.fft2(lucid.randn(64, 64))
    >>> # Zero out high frequencies, then invert
    >>> x_smooth = lucid.fft.ifft2(X)
    """
    return ifftn(input, s=s, dim=dim, norm=norm)


# ── Public API: real-input FFT (rfft / rfft2 / rfftn) ────────────────────────


def rfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional FFT of a real-valued input, exploiting conjugate symmetry.

    When the input is real, its N-dimensional DFT satisfies the
    Hermitian (conjugate-symmetric) property:

    .. math::

        X[k_1, \ldots, k_d] = X^*[N_1 - k_1, \ldots, N_d - k_d].

    Only the non-redundant half of the spectrum along the *last*
    transformed axis needs to be stored.  For a last-axis length
    :math:`N`, the output contains only :math:`\lfloor N/2 \rfloor + 1`
    unique complex values along that axis.  All other axes are full-length
    (:math:`N_i` bins each).

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor of any shape.  Passing a complex tensor
        is accepted but the imaginary part is silently discarded by the
        engine.
    s : int or sequence of int, optional
        Signal length(s) along each transformed axis.  When given, the
        last element ``s[-1]`` determines the last-axis size before the
        real FFT; the output last axis will be ``s[-1] // 2 + 1``.
        ``len(s)`` must equal ``len(dim)`` when both are sequences.
        If ``None``, each axis is transformed at its current size.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the transform.  The *last*
        element of ``dim`` is the axis that gets the conjugate-symmetry
        compression.  ``None`` (the default) means every axis, or the
        last ``len(s)`` axes when ``s`` is given.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  :math:`N` is the product of the *full* (uncompressed)
        lengths of all transformed axes.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except:

        - All transformed axes except the last have their original lengths
          (or ``s[i]`` when specified).
        - The *last* transformed axis has length
          :math:`\lfloor N_{\text{last}} / 2 \rfloor + 1`.

    Notes
    -----
    **Why n//2 + 1?** — A real-to-complex 1-D DFT of length :math:`N`
    has the symmetry :math:`X[N-k] = X^*[k]`.  The unique bins are
    therefore :math:`k = 0, 1, \ldots, \lfloor N/2 \rfloor`, giving
    :math:`\lfloor N/2 \rfloor + 1` complex values.  For even :math:`N`
    this is :math:`N/2 + 1`; for odd :math:`N` it is :math:`(N+1)/2`.

    **Nyquist theorem** — for a real discrete signal sampled at rate
    :math:`f_s`, the highest representable frequency is the Nyquist
    frequency :math:`f_s / 2`.  The bin at index
    :math:`\lfloor N/2 \rfloor` corresponds exactly to this frequency.
    Any signal energy above :math:`f_s / 2` aliases back into the
    :math:`[0, f_s/2]` range — this is the aliasing artefact that the
    sampling theorem warns against.

    **Memory saving** — compared to :func:`fftn`, ``rfftn`` stores
    roughly half the complex coefficients along the last axis, reducing
    peak memory and compute by ~2× for large real tensors.

    Examples
    --------
    1-D real FFT:

    >>> x = lucid.randn(128)
    >>> X = lucid.fft.rfftn(x)
    >>> X.shape   # 128 // 2 + 1 = 65
    (65,)

    3-D real FFT over all axes:

    >>> x = lucid.randn(16, 32, 64)
    >>> X = lucid.fft.rfftn(x)
    >>> X.shape   # last axis: 64 // 2 + 1 = 33
    (16, 32, 33)

    Explicit output size for zero-padding:

    >>> X = lucid.fft.rfftn(lucid.randn(60), s=[64])
    >>> X.shape   # 64 // 2 + 1 = 33
    (33,)
    """
    norm_v = _check_norm(norm)
    input, axes, s_list = _prepare(input, s, dim, "rfftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    full_sizes: list[int] = list(s_list) if s_list else list(in_sizes)
    N = _transform_size(s_list, in_sizes)
    return _transform(_RfftnAutograd, input, s_list, axes, norm_v, N, full_sizes)


def irfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional inverse FFT for a Hermitian-symmetric spectrum.

    Computes the inverse N-dimensional FFT assuming the input is
    conjugate-symmetric (Hermitian), as produced by :func:`rfftn`.
    The output is guaranteed to be real-valued.

    For the last transformed axis, the full-length signal size must be
    specified or inferred: given an input last-axis of length
    :math:`m = \lfloor N/2 \rfloor + 1`, the default assumed full length
    is :math:`N = 2(m - 1)` (even).  Supply ``s[-1]`` explicitly when
    the original signal length was odd.

    Parameters
    ----------
    input : Tensor
        Complex Hermitian tensor (frequency domain) of any shape,
        typically the output of :func:`rfftn`.
    s : int or sequence of int, optional
        Full output signal length(s) along each transformed axis.  The
        last element is the real (time-domain) length of the last
        transformed axis.  For all other axes ``s[i]`` may equal the
        corresponding input axis length.  If ``None``, the last axis
        output length defaults to :math:`2 (m - 1)` where :math:`m` is
        the input last-axis length; all other axes keep their input sizes.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the inverse transform.
        ``None`` (the default) means every axis, or the last ``len(s)``
        axes when ``s`` is given.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Real tensor (``float32``) whose shape matches ``input`` except:

        - The last transformed axis has length ``s[-1]`` (or
          :math:`2(m-1)` by default, where :math:`m` is the input
          last-axis length).
        - All other transformed axes have lengths ``s[i]`` (or input
          sizes when ``s`` is ``None``).

    Notes
    -----
    **Specifying odd-length signals** — if the original real signal had
    an odd length :math:`N` (so the rfft output had :math:`(N+1)/2`
    bins), you must pass ``s=[N]`` explicitly.  Without it the default
    :math:`2(m-1)` rule would give :math:`N-1` (even), producing the
    wrong output length.

    **Round-trip** — for a real tensor ``x``:

    .. math::

        \text{irfftn}(\text{rfftn}(x, \text{norm}=m), s, \text{norm}=m)
        \approx x

    where ``s`` should contain the original axis sizes to avoid the
    even/odd ambiguity.

    Examples
    --------
    Round-trip through the real FFT:

    >>> x = lucid.randn(128)
    >>> X = lucid.fft.rfftn(x)
    >>> X.shape
    (65,)
    >>> x_rec = lucid.fft.irfftn(X, s=[128])
    >>> x_rec.shape
    (128,)

    3-D round-trip:

    >>> x = lucid.randn(16, 32, 64)
    >>> X = lucid.fft.rfftn(x)
    >>> x_rec = lucid.fft.irfftn(X, s=[16, 32, 64])
    >>> x_rec.shape
    (16, 32, 64)
    """
    norm_v = _check_norm(norm)
    input, axes, s_list = _prepare(input, s, dim, "irfftn")
    out_sizes: list[int]
    if s_list:
        out_sizes = list(s_list)
    else:
        # Default: last axis becomes 2*(in_last - 1); others keep size.
        out_sizes = _input_sizes_along_axes(input.shape, axes)
        out_sizes[-1] = (out_sizes[-1] - 1) * 2
    N = 1
    for v in out_sizes:
        N *= int(v)
    return _transform(_IrfftnAutograd, input, s_list, axes, norm_v, N, out_sizes)


def rfft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D FFT of a real-valued input along a single axis.

    Computes the one-dimensional DFT of a real signal and returns only
    the non-redundant complex output bins.  For a real input of length
    :math:`N`, the DFT satisfies :math:`X[N-k] = X^*[k]`, so only
    :math:`\lfloor N/2 \rfloor + 1` unique complex values need to be
    stored.

    The mathematical definition is identical to :func:`fft`:

    .. math::

        X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i 2\pi k n / N},
        \quad k = 0, 1, \ldots, \lfloor N/2 \rfloor.

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor of any shape.
    n : int, optional
        Number of input samples used (along ``dim``).  The axis is
        zero-padded or truncated to this length before the transform.
        If ``None``, the full axis length is used.
    dim : int, optional
        Axis over which to compute the transform.  Default is ``-1``
        (last axis).
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that axis ``dim`` has length :math:`\lfloor n/2 \rfloor + 1`
        (where :math:`n` is the effective input length along that axis).

    Notes
    -----
    **Output size** — for an input axis of length :math:`N` the output
    has :math:`N//2 + 1` complex bins:

    - Bin 0 is the DC component (purely real for a real input).
    - Bins 1 through :math:`N//2 - 1` are complex (positive frequencies).
    - Bin :math:`N//2` is the Nyquist component (purely real for even
      :math:`N`).

    **Inverse** — use :func:`irfft` to reconstruct the real signal.
    You must pass the original length ``n`` to resolve the even/odd
    ambiguity when the output of ``rfft`` has an odd number of bins.

    Examples
    --------
    Real FFT of a length-128 signal:

    >>> x = lucid.randn(128)
    >>> X = lucid.fft.rfft(x)
    >>> X.shape   # 128 // 2 + 1 = 65
    (65,)

    Round-trip:

    >>> x_rec = lucid.fft.irfft(X, n=128)
    >>> x_rec.shape
    (128,)
    """
    return rfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def irfft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D inverse FFT for a Hermitian-symmetric spectrum.

    Computes the inverse one-dimensional FFT of a conjugate-symmetric
    (Hermitian) spectrum, as produced by :func:`rfft`, and returns a
    real-valued output.

    Given an input of length :math:`m = \lfloor N/2 \rfloor + 1`, the
    full-length real output has length :math:`N`.  The default assumption
    is that :math:`N = 2(m-1)` (even); supply ``n`` explicitly when the
    original signal length was odd.

    Parameters
    ----------
    input : Tensor
        Complex Hermitian input tensor (the one-sided spectrum produced
        by :func:`rfft`), of any shape.
    n : int, optional
        Length of the output (real) axis.  If ``None``, the output
        length defaults to :math:`2(m - 1)` where :math:`m` is the
        input axis length.  Pass the original signal length to avoid
        the even/odd ambiguity.
    dim : int, optional
        Axis over which to compute the inverse transform.  Default is
        ``-1`` (last axis).
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Real tensor (``float32``) with the same shape as ``input``
        except that axis ``dim`` has length ``n`` (or :math:`2(m-1)`
        when ``n`` is ``None``).

    Notes
    -----
    **Odd-length signals** — if the original real signal had odd length
    :math:`N`, the ``rfft`` output has :math:`(N+1)/2` bins.  The
    default :math:`2(m-1)` rule gives :math:`N-1` instead of :math:`N`.
    Always pass ``n=N`` explicitly when the original length was odd.

    **Normalisation parity** — to faithfully invert :func:`rfft`, use
    the same ``norm`` value for both the forward and inverse calls.

    Examples
    --------
    Round-trip with explicit length:

    >>> x = lucid.randn(100)              # odd length
    >>> X = lucid.fft.rfft(x)
    >>> X.shape                           # 51
    (51,)
    >>> x_rec = lucid.fft.irfft(X, n=100)
    >>> x_rec.shape
    (100,)

    Default (even-length assumption):

    >>> X = lucid.fft.rfft(lucid.randn(128))
    >>> lucid.fft.irfft(X).shape          # 2*(65-1) = 128
    (128,)
    """
    return irfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def rfft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    r"""2-D FFT of a real-valued input over a pair of axes.

    Computes the two-dimensional DFT of a real tensor and returns only
    the non-redundant complex coefficients.  For a real input of shape
    :math:`(M, N)` (along the two transformed axes), the output shape
    is :math:`(M, N//2 + 1)` — full along the first transformed axis
    and compressed along the last.

    This is a convenience wrapper around :func:`rfftn` restricted to
    two axes (default: the last two).

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor with at least 2 dimensions.
    s : int or sequence of int, optional
        Output sizes ``(M_out, N_out)`` along the two transformed axes.
        The last value ``s[-1]`` is the full signal length along the last
        axis before the real FFT; the output last axis will be
        ``s[-1] // 2 + 1``.  If ``None``, current axis sizes are used.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` by default; a single int
        transforms that one axis.  The *last* one receives the
        conjugate-symmetry compression.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except:

        - The first transformed axis has length ``s[0]`` (or input size).
        - The *last* transformed axis has length
          :math:`\lfloor s[-1]/2 \rfloor + 1`.

    Notes
    -----
    **Typical use** — ``rfft2`` is the standard choice for 2-D image
    convolution via the convolution theorem:

    .. math::

        (f * g)[m, n] = \mathcal{F}^{-1}\{\mathcal{F}\{f\} \cdot
                        \mathcal{F}\{g\}\}[m, n],

    where :math:`\mathcal{F}` denotes the 2-D DFT.  Performing the
    multiplication in the frequency domain reduces the cost of a full
    convolution from :math:`O(M^2 N^2)` to :math:`O(MN \log(MN))`.

    Examples
    --------
    2-D real FFT of an image-like tensor:

    >>> x = lucid.randn(64, 64)
    >>> X = lucid.fft.rfft2(x)
    >>> X.shape   # last axis: 64 // 2 + 1 = 33
    (64, 33)

    Round-trip:

    >>> x_rec = lucid.fft.irfft2(X, s=[64, 64])
    >>> x_rec.shape
    (64, 64)
    """
    return rfftn(input, s=s, dim=dim, norm=norm)


def irfft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    r"""2-D inverse FFT for a Hermitian-symmetric spectrum.

    Recovers a real 2-D signal from its one-sided 2-D spectrum as
    produced by :func:`rfft2`.  The output is guaranteed to be
    real-valued.

    For an input of shape :math:`(M, m)` where :math:`m = \lfloor N/2
    \rfloor + 1`, the default full output size is :math:`(M, 2(m-1))`
    (assuming an even last-axis length :math:`N`).

    Parameters
    ----------
    input : Tensor
        Complex Hermitian tensor (one-sided 2-D spectrum) with at least
        2 dimensions, typically the output of :func:`rfft2`.
    s : int or sequence of int, optional
        Full output sizes ``(M_out, N_out)`` along the two transformed
        axes.  Pass the original signal shape to avoid the even/odd
        ambiguity in the last axis.  If ``None``, the last axis
        defaults to :math:`2(m-1)`.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` by default; a single int
        transforms that one axis.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Real tensor (``float32``) with the same shape as ``input``
        except the two transformed axes have lengths ``s[0]`` and
        ``s[1]`` (or inferred sizes when ``s`` is ``None``).

    Notes
    -----
    **Odd-length last axis** — if the original image had an odd number
    of columns :math:`N`, always pass ``s=[M, N]`` explicitly.

    Examples
    --------
    Round-trip for a 128×256 image:

    >>> x = lucid.randn(128, 256)
    >>> X = lucid.fft.rfft2(x)
    >>> X.shape   # (128, 129)
    (128, 129)
    >>> x_rec = lucid.fft.irfft2(X, s=[128, 256])
    >>> x_rec.shape
    (128, 256)
    """
    return irfftn(input, s=s, dim=dim, norm=norm)


# ── Public API: Hermitian FFT (hfft / hfft2 / hfftn) ─────────────────────────
#
# Identity used (matches numpy & the standard reference framework):
#     hfft(x, n, norm)  =  irfft(conj(x), n, norm=dual(norm))
#     ihfft(x, n, norm) =  conj(rfft(x, n, norm=dual(norm)))
# where ``dual`` swaps 'backward' ↔ 'forward' and leaves 'ortho' fixed.
# This works because hfft uses the FFT sign convention on a Hermitian
# (conjugate-symmetric) signal — flipping conj on the input flips the
# sign in the exponent, turning the inverse-real transform back into a
# forward-real one.


def hfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional FFT of a Hermitian-symmetric complex signal.

    Computes the N-dimensional DFT of a signal that is known to be
    Hermitian (conjugate-symmetric), producing a real-valued output.
    The input is the *half-spectrum* as conventionally stored by
    :func:`rfftn` or constructed manually: the first :math:`n//2 + 1`
    bins along the last transformed axis, with no redundancy stored.

    The implementation uses the identity:

    .. math::

        \text{hfftn}(x, n, \text{norm}) =
        \text{irfftn}(x^*, n, \text{norm}=\text{dual}(\text{norm})),

    where :math:`x^*` is the element-wise complex conjugate and
    ``dual`` swaps ``"backward"`` ↔ ``"forward"`` while leaving
    ``"ortho"`` unchanged.  This is mathematically equivalent to a
    forward DFT on a Hermitian signal: conjugating the input flips the
    sign in the exponent, converting the inverse-real transform into a
    forward-real one.

    Parameters
    ----------
    input : Tensor
        Complex Hermitian half-spectrum tensor of any shape.  The last
        transformed axis is the compressed one-sided axis (length
        :math:`\lfloor N/2 \rfloor + 1`).
    s : int or sequence of int, optional
        Full output signal length(s) along each transformed axis.  The
        last value gives the full real length of the last axis.  If
        ``None``, the last axis defaults to :math:`2(m-1)` where
        :math:`m` is the input last-axis length.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the transform.  ``None`` (the
        default) means every axis, or the last ``len(s)`` axes when ``s``
        is given.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  Note that the dual normalisation is applied
        internally; the semantics mirror those of :func:`fftn`.

    Returns
    -------
    Tensor
        Real tensor (``float32``) whose shape matches ``input`` except
        that the last transformed axis has length ``s[-1]`` (or
        :math:`2(m-1)` when ``s`` is ``None``), and all other
        transformed axes have lengths ``s[i]``.

    Notes
    -----
    **Relationship to rfft / irfft** — the Hermitian FFT pair
    (``hfftn`` / ``ihfftn``) is the *Fourier-domain dual* of the real
    FFT pair (``rfftn`` / ``irfftn``):

    - ``rfftn`` takes a **real** time-domain signal → **half complex**
      spectrum.
    - ``hfftn`` takes a **half complex** Hermitian signal (frequency
      domain) → **real** output in the transform domain.

    **Sign convention** — :func:`hfftn` uses the *forward* FFT sign
    convention (:math:`e^{-i2\pi kn/N}`) applied to the Hermitian input.
    This means applying ``hfftn`` followed by ``ihfftn`` recovers the
    original half-spectrum (up to floating-point rounding).

    Examples
    --------
    Recover a real spectrum from a Hermitian half-spectrum:

    >>> x_complex = lucid.randn(33)   # half-spectrum for N=64
    >>> y = lucid.fft.hfftn(x_complex, s=[64])
    >>> y.shape
    (64,)

    N-dimensional example:

    >>> x = lucid.randn(16, 33)           # last axis: 33 = 64//2+1
    >>> y = lucid.fft.hfftn(x, s=[16, 64])
    >>> y.shape
    (16, 64)
    """
    norm_v = _check_norm(norm)
    return irfftn(_conj(_promote(input)), s=s, dim=dim, norm=_dual_norm(norm_v))


def ihfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    r"""N-dimensional inverse FFT of a real-valued signal, giving a Hermitian output.

    Computes the inverse of :func:`hfftn`: given a real-valued input,
    produces the conjugate-symmetric half-spectrum.  The implementation
    uses the identity:

    .. math::

        \text{ihfftn}(x, s, \text{norm}) =
        \overline{\text{rfftn}(x, s, \text{norm}=\text{dual}(\text{norm}))},

    where :math:`\overline{(\cdot)}` denotes complex conjugation.

    Equivalently, ``ihfftn`` computes the *backward*-sign DFT
    (:math:`e^{+i2\pi kn/N}`) of the real input and returns only the
    first :math:`\lfloor N/2 \rfloor + 1` bins along the last
    transformed axis, exactly mirroring the output format of :func:`rfft`.

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor of any shape.
    s : int or sequence of int, optional
        Signal length(s) along each transformed axis.  The last value
        determines the full real length, and the output last axis will
        be ``s[-1] // 2 + 1``.  If ``None``, current axis sizes are used.
    dim : int or sequence of int, optional
        Axis or axes over which to compute the transform.  ``None`` (the
        default) means every axis, or the last ``len(s)`` axes when ``s``
        is given.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that the last transformed axis has length
        :math:`\lfloor N/2 \rfloor + 1`.

    Notes
    -----
    **Round-trip with hfftn** — for a real tensor ``x``:

    .. math::

        \text{hfftn}(\text{ihfftn}(x, s, \text{norm}=m),
        s, \text{norm}=m) \approx x.

    **Relationship to rfft** — the output of ``ihfftn`` with
    ``norm='backward'`` has the same element magnitudes as ``rfft`` with
    ``norm='backward'`` but the signs of the imaginary parts are
    negated (due to the conjugation in the identity above).

    Examples
    --------
    Compute the Hermitian half-spectrum of a real signal:

    >>> x = lucid.randn(128)
    >>> H = lucid.fft.ihfftn(x)
    >>> H.shape   # 128 // 2 + 1 = 65
    (65,)

    Round-trip:

    >>> x_rec = lucid.fft.hfftn(H, s=[128])
    >>> x_rec.shape
    (128,)
    """
    norm_v = _check_norm(norm)
    return _conj(rfftn(input, s=s, dim=dim, norm=_dual_norm(norm_v)))


def hfft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D FFT of a Hermitian-symmetric complex signal.

    Computes the one-dimensional DFT of a Hermitian (conjugate-symmetric)
    signal stored as a one-sided half-spectrum, returning a real-valued
    output.  This is the 1-D specialisation of :func:`hfftn`.

    A Hermitian signal satisfies :math:`x[N-k] = x^*[k]`, so only
    :math:`\lfloor N/2 \rfloor + 1` complex values need to be given as
    input.  The transform is mathematically:

    .. math::

        y[n] = \sum_{k=0}^{N-1} x[k]\, e^{-i 2\pi k n / N},
        \quad n = 0, \ldots, N-1,

    where the full spectrum :math:`x[k]` is reconstructed from the
    half-spectrum by Hermitian extension before the sum is evaluated.

    Parameters
    ----------
    input : Tensor
        Complex half-spectrum tensor of any shape.  Axis ``dim`` is
        expected to have length :math:`\lfloor N/2 \rfloor + 1`.
    n : int, optional
        Full length of the output (real) axis.  If ``None``, the output
        defaults to :math:`2(m-1)` where :math:`m` is the input axis
        length.  Specify ``n`` explicitly when the original signal
        length was odd.
    dim : int, optional
        Axis over which to compute the transform.  Default is ``-1``.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Real tensor (``float32``) with the same shape as ``input``
        except that axis ``dim`` has length ``n`` (or :math:`2(m-1)`
        when ``n`` is ``None``).

    Notes
    -----
    **Usage pattern** — ``hfft`` is rarely needed in signal processing
    pipelines that use the standard ``rfft`` / ``irfft`` pair.  It
    becomes useful when working in the frequency domain: if you have
    synthesised or modified a Hermitian spectrum directly and want to
    recover the implied real time-domain signal, ``hfft`` does it in
    one call without needing to materialise the full (redundant) spectrum.

    Examples
    --------
    Construct a Hermitian half-spectrum and recover the real signal:

    >>> half_spec = lucid.randn(33)  # half-spectrum for N=64
    >>> y = lucid.fft.hfft(half_spec, n=64)
    >>> y.shape
    (64,)

    Round-trip with ``ihfft``:

    >>> x = lucid.randn(64)
    >>> H = lucid.fft.ihfft(x)
    >>> x_rec = lucid.fft.hfft(H, n=64)
    >>> x_rec.shape
    (64,)
    """
    return hfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def ihfft(
    input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
) -> Tensor:
    r"""1-D inverse FFT of a real signal, giving a Hermitian half-spectrum.

    Computes the inverse of :func:`hfft`: given a real-valued input,
    returns the conjugate of its one-sided real FFT.  The output is
    complex and has length :math:`\lfloor N/2 \rfloor + 1` along
    axis ``dim``.

    Mathematically:

    .. math::

        Y[k] = \overline{\text{rfft}(x)[k]},
        \quad k = 0, 1, \ldots, \lfloor N/2 \rfloor,

    where :math:`\overline{(\cdot)}` denotes complex conjugation and
    the dual normalisation is applied to match ``hfft``.

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor of any shape.
    n : int, optional
        Number of input samples to use.  If ``None``, the full axis
        length is used.
    dim : int, optional
        Axis over which to compute the transform.  Default is ``-1``.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except that axis ``dim`` has length :math:`\lfloor n/2 \rfloor + 1`.

    Notes
    -----
    **Round-trip with hfft** — for a real tensor ``x`` of length
    :math:`N`:

    .. math::

        \text{hfft}(\text{ihfft}(x, N), N) \approx x.

    **Negative imaginary parts** — because ``ihfft`` conjugates the
    ``rfft`` output, the imaginary parts of the result are the negatives
    of those from ``rfft``.  For a real input, ``rfft(x)[0]`` and
    ``rfft(x)[N//2]`` (for even :math:`N`) are both real, so
    ``ihfft(x)[0]`` and ``ihfft(x)[N//2]`` are also real.

    Examples
    --------
    Compute the Hermitian half-spectrum:

    >>> x = lucid.randn(64)
    >>> H = lucid.fft.ihfft(x)
    >>> H.shape   # 64 // 2 + 1 = 33
    (33,)

    Round-trip check:

    >>> x_rec = lucid.fft.hfft(H, n=64)
    >>> x_rec.shape
    (64,)
    """
    return ihfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def hfft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """2-D FFT of a Hermitian-symmetric complex signal over a pair of axes.

    Computes the two-dimensional DFT of a Hermitian signal stored as a
    one-sided half-spectrum (as produced by :func:`ihfft2`), returning a
    real-valued output.  This is the 2-D specialisation of :func:`hfftn`.

    For an input of shape :math:`(M, m)` where :math:`m = N//2 + 1`,
    the output shape is :math:`(M, N)`.

    Parameters
    ----------
    input : Tensor
        Complex Hermitian half-spectrum tensor with at least 2 dimensions.
    s : int or sequence of int, optional
        Full output sizes ``(M_out, N_out)`` along the two transformed
        axes.  If ``None``, the last axis defaults to :math:`2(m-1)`.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` by default; a single int
        transforms that one axis.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Real tensor (``float32``) with the same shape as ``input``
        except the last transformed axis has length ``s[-1]`` (or
        :math:`2(m-1)` by default).

    Notes
    -----
    **Odd-length last axis** — always pass ``s`` explicitly when the
    original last axis had an odd length to avoid the default
    :math:`2(m-1)` even-length assumption.

    Examples
    --------
    Recover a real 2-D signal from its Hermitian half-spectrum:

    >>> H = lucid.randn(64, 33)   # half-spectrum for 64×64 real image
    >>> y = lucid.fft.hfft2(H, s=[64, 64])
    >>> y.shape
    (64, 64)
    """
    return hfftn(input, s=s, dim=dim, norm=norm)


def ihfft2(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    r"""2-D inverse FFT of a real signal, giving a Hermitian half-spectrum.

    Computes the inverse of :func:`hfft2`: given a real-valued 2-D input,
    returns the conjugated one-sided 2-D spectrum (the Hermitian
    half-spectrum).  This is the 2-D specialisation of :func:`ihfftn`.

    For an input of shape :math:`(M, N)`, the output shape is
    :math:`(M, N//2 + 1)` — full along the first axis and compressed
    along the last.

    Parameters
    ----------
    input : Tensor
        Real-valued input tensor with at least 2 dimensions.
    s : int or sequence of int, optional
        Signal sizes ``(M_out, N_out)`` along the two transformed axes.
        If ``None``, current axis sizes are used.
    dim : int or sequence of int, optional
        The axes to transform, ``(-2, -1)`` by default; a single int
        transforms that one axis.
    norm : str or None, optional
        Normalisation mode — ``"backward"`` (default), ``"forward"``, or
        ``"ortho"``.  See :func:`fftn` for the full description.

    Returns
    -------
    Tensor
        Complex tensor (``complex64``) with the same shape as ``input``
        except the last transformed axis has length
        :math:`\lfloor N/2 \rfloor + 1`.

    Notes
    -----
    **Round-trip with hfft2** — for a real tensor ``x`` of shape
    :math:`(M, N)`:

    .. math::

        \text{hfft2}(\text{ihfft2}(x), s=[M, N]) \approx x.

    Examples
    --------
    Compute the Hermitian half-spectrum of a 2-D real image:

    >>> x = lucid.randn(64, 64)
    >>> H = lucid.fft.ihfft2(x)
    >>> H.shape   # last axis: 64 // 2 + 1 = 33
    (64, 33)

    Round-trip:

    >>> x_rec = lucid.fft.hfft2(H, s=[64, 64])
    >>> x_rec.shape
    (64, 64)
    """
    return ihfftn(input, s=s, dim=dim, norm=norm)


# ── Public API: utility functions ────────────────────────────────────────────


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Frequency bin centres for the output of :func:`fft` / :func:`fftn`.

    Returns a 1-D tensor of length ``n`` containing the normalised
    discrete frequencies corresponding to the output bins of a length-``n``
    DFT computed with sample spacing ``d``.  The values follow the
    standard FFT frequency ordering:

    .. math::

        f_k = \frac{k}{n \cdot d}, \quad
        k \in \{0, 1, \ldots, \lfloor n/2 \rfloor - 1,
                  -\lfloor n/2 \rfloor, \ldots, -1\}.

    Equivalently the output tensor is:

    .. code-block:: text

        [0, 1, 2, ..., n/2-1, -n/2, ..., -2, -1] / (d * n)   # n even
        [0, 1, 2, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d * n) # n odd

    Parameters
    ----------
    n : int
        Window length (number of DFT points).  Must be positive.
    d : float, optional
        Sample spacing (inverse of the sample rate).  Default is ``1.0``,
        which gives frequencies in units of *cycles per sample*.
        Pass ``d = 1/fs`` where ``fs`` is the sample rate in Hz to get
        frequencies in Hz.
    dtype : DTypeLike, optional
        Data type of the output tensor.  Default uses the framework's
        default floating-point type.
    device : DeviceLike, optional
        Device on which to allocate the output tensor.
    requires_grad : bool, optional
        Whether autograd records operations on the result.

    Returns
    -------
    Tensor
        1-D float tensor of shape ``(n,)`` containing the frequency bin
        centres in the same order as the DFT output.

    Notes
    -----
    **Physical frequency interpretation** — for a signal sampled at
    rate :math:`f_s = 1/d`:

    - The first bin (:math:`k=0`) is DC (zero frequency).
    - Bins :math:`k = 1, \ldots, \lfloor n/2 \rfloor - 1` are positive
      frequencies :math:`k f_s / n`.
    - Bin :math:`k = \lfloor n/2 \rfloor` is the *Nyquist frequency*
      :math:`f_s / 2` (for even :math:`n`).  This is the highest
      representable frequency without aliasing.
    - Bins :math:`k = \lfloor n/2 \rfloor + 1, \ldots, n-1` are
      aliased negative frequencies.

    **Centred display** — to get frequencies in monotonically increasing
    order (centred at zero), apply :func:`fftshift` to both the frequency
    vector and the DFT output.

    Examples
    --------
    Frequencies for a length-8 DFT at sample spacing 1:

    >>> lucid.fft.fftfreq(8)
    tensor([0., 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125])

    Frequencies in Hz for a 1 kHz sample rate:

    >>> fs = 1000.0
    >>> freqs = lucid.fft.fftfreq(256, d=1.0/fs)
    >>> freqs.shape
    (256,)
    >>> # freqs[0] == 0.0,  freqs[128] == -500.0 (Nyquist)

    Centred frequency axis for plotting:

    >>> freqs_centred = lucid.fft.fftshift(lucid.fft.fftfreq(256, d=1.0/1000.0))
    >>> # Now ranges from -500 Hz to ~496 Hz, centred at 0
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"fftfreq requires n > 0, got {n}")
    half = (n - 1) // 2 + 1  # number of non-negative frequency bins
    pos = lucid.arange(0.0, float(half), 1.0, dtype=dtype, device=device)
    neg = lucid.arange(float(-(n // 2)), 0.0, 1.0, dtype=dtype, device=device)
    full = lucid.cat([pos, neg], 0)
    return (full * (1.0 / (float(d) * n))).requires_grad_(requires_grad)


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Frequency bin centres for the output of :func:`rfft` / :func:`rfftn`.

    Returns a 1-D tensor of length :math:`\lfloor n/2 \rfloor + 1`
    containing the non-negative discrete frequencies corresponding to
    the output bins of a length-``n`` real DFT.  Because :func:`rfft`
    exploits conjugate symmetry and returns only the unique half of the
    spectrum, all returned frequencies are non-negative:

    .. math::

        f_k = \frac{k}{n \cdot d},
        \quad k = 0, 1, \ldots, \lfloor n/2 \rfloor.

    Parameters
    ----------
    n : int
        Length of the original (real) signal, not the length of the
        ``rfft`` output.  Must be positive.
    d : float, optional
        Sample spacing (inverse of the sample rate).  Default is ``1.0``
        (cycles per sample).  Pass ``d = 1/fs`` to get frequencies in Hz.
    dtype : DTypeLike, optional
        Data type of the output tensor.
    device : DeviceLike, optional
        Device on which to allocate the output tensor.
    requires_grad : bool, optional
        Whether autograd records operations on the result.

    Returns
    -------
    Tensor
        1-D float tensor of shape :math:`(\lfloor n/2 \rfloor + 1,)`
        containing the non-negative frequency bin centres, in ascending
        order from 0 to the Nyquist frequency :math:`1/(2d)`.

    Notes
    -----
    **Non-negative only** — unlike :func:`fftfreq`, ``rfftfreq`` returns
    only non-negative frequencies.  The last bin is the Nyquist frequency
    :math:`f_{n//2} = 1/(2d) = f_s/2`.  This is because for a real input
    the negative-frequency bins are redundant (they are the complex
    conjugates of the positive-frequency bins).

    **Bin count** — for even :math:`n`, there are :math:`n/2 + 1` bins
    (including DC and Nyquist); for odd :math:`n`, there are
    :math:`(n+1)/2` bins (including DC but no exact Nyquist bin).

    **Monotone ordering** — the output is always in ascending order,
    making it directly usable as an :math:`x`-axis for power spectra.
    No :func:`fftshift` is needed.

    Examples
    --------
    Frequencies for a length-8 real DFT:

    >>> lucid.fft.rfftfreq(8)
    tensor([0., 0.125, 0.25, 0.375, 0.5])

    Frequencies in Hz at 44100 Hz sample rate:

    >>> fs = 44100.0
    >>> freqs = lucid.fft.rfftfreq(2048, d=1.0/fs)
    >>> freqs.shape
    (1025,)
    >>> # freqs[-1] ≈ 22050.0 Hz  (Nyquist)

    Pair with rfft output:

    >>> x = lucid.randn(1024)
    >>> X = lucid.fft.rfft(x)
    >>> freqs = lucid.fft.rfftfreq(1024)
    >>> # len(freqs) == X.shape[-1] == 513
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"rfftfreq requires n > 0, got {n}")
    bins = lucid.arange(0.0, float(n // 2 + 1), 1.0, dtype=dtype, device=device)
    return (bins * (1.0 / (float(d) * n))).requires_grad_(requires_grad)


def fftshift(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    r"""Shift the zero-frequency component to the centre of the spectrum.

    Rearranges the output of :func:`fft`, :func:`fft2`, or :func:`fftn`
    so that the zero-frequency bin (DC component) moves to the centre
    of each specified axis.  For an axis of length :math:`N`, the
    operation performs a circular shift of :math:`\lfloor N/2 \rfloor`
    positions:

    - Bins :math:`\lfloor N/2 \rfloor, \ldots, N-1` (the negative
      frequencies) wrap around to the *left* half of the axis.
    - Bins :math:`0, \ldots, \lfloor N/2 \rfloor - 1` (DC and positive
      frequencies) move to the *right* half.

    The result is a spectrum ordered from the most-negative frequency
    to the most-positive, with DC at index :math:`\lfloor N/2 \rfloor`.

    Parameters
    ----------
    input : Tensor
        Tensor containing DFT output (or any data in FFT frequency
        ordering) of any shape.
    dim : int or sequence of int, optional
        Axis or axes along which to shift.  Negative indices are
        supported.  Defaults to all axes when ``None``.

    Returns
    -------
    Tensor
        Tensor with the same shape and dtype as ``input``, rearranged
        so that zero frequency is centred along each specified axis.

    Notes
    -----
    **Visual motivation** — the standard FFT ordering places the DC
    component at index 0 and the Nyquist frequency at index
    :math:`N//2`.  When displaying spectra (e.g. 2-D power spectra of
    images), it is conventional to show the DC component at the centre
    of the image.  ``fftshift`` performs exactly this reordering so
    that the spatial-frequency origin is visually centred.

    **Frequency correspondence** — after shifting, use
    :func:`fftfreq` with the same ``n`` and apply the same shift to
    obtain a matching, monotonically increasing frequency axis:

    .. math::

        f_{\text{centred}} =
        \text{fftshift}(\text{fftfreq}(N, d)).

    **Inverse** — :func:`ifftshift` undoes ``fftshift`` exactly.
    For even :math:`N`, ``fftshift`` and ``ifftshift`` are
    self-inverse (applying either twice returns the original).  For
    odd :math:`N` the two differ by one position.

    Examples
    --------
    Centre the spectrum of a 1-D signal:

    >>> x = lucid.randn(8)
    >>> X = lucid.fft.fft(x)
    >>> X_centred = lucid.fft.fftshift(X)
    >>> # X_centred[4] == X[0]  (DC at centre for N=8)

    Centre both the 2-D spectrum and its frequency axes:

    >>> x = lucid.randn(64, 64)
    >>> X = lucid.fft.fft2(x)
    >>> X_c = lucid.fft.fftshift(X)           # centred spectrum
    >>> freqs = lucid.fft.fftshift(lucid.fft.fftfreq(64))  # centred frequency axis
    """
    rank = input.ndim
    dims = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    shifts = [int(input.shape[a]) // 2 for a in dims]
    return lucid.roll(input, shifts=shifts, dims=dims)


def ifftshift(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    r"""Undo the zero-frequency centring performed by :func:`fftshift`.

    Performs the inverse of :func:`fftshift`: given a centred spectrum
    (with DC at the middle of each axis), restores the standard FFT
    frequency ordering (DC at index 0).

    For an axis of length :math:`N`, the operation is a circular shift
    of :math:`-(N//2)` positions, which is exactly :math:`\lceil N/2 \rceil`
    positions in the forward direction (i.e., the *ceiling* variant
    rather than the *floor* used by :func:`fftshift`).  For even
    :math:`N` the two shifts have the same magnitude; for odd :math:`N`
    they differ by one.

    Parameters
    ----------
    input : Tensor
        Tensor in centred-spectrum ordering (typically the output of
        :func:`fftshift`), of any shape.
    dim : int or sequence of int, optional
        Axis or axes along which to shift.  Defaults to all axes.

    Returns
    -------
    Tensor
        Tensor with the same shape and dtype as ``input``, rearranged
        back to the standard FFT ordering (DC at index 0 of each
        specified axis).

    Notes
    -----
    **Exact inverse** — for any tensor ``x`` and any axis selection
    ``dim``:

    .. math::

        \text{ifftshift}(\text{fftshift}(x, \text{dim}), \text{dim}) = x,

    .. math::

        \text{fftshift}(\text{ifftshift}(x, \text{dim}), \text{dim}) = x.

    **Even vs. odd** — for even :math:`N`, ``fftshift`` and
    ``ifftshift`` are identical (both shift by :math:`N/2`).  For
    odd :math:`N`, ``fftshift`` shifts by :math:`\lfloor N/2 \rfloor`
    and ``ifftshift`` shifts by :math:`-\lfloor N/2 \rfloor` (= shift
    by :math:`\lceil N/2 \rceil` forward), so they are distinct
    operations.  Always use the matching call to guarantee an exact
    round-trip.

    **Typical use** — when a signal has been constructed or modified in
    centred-frequency space (e.g. a 2-D optical transfer function defined
    as centred), ``ifftshift`` must be applied before feeding it into
    :func:`ifftn` to recover the correct spatial-domain signal.

    Examples
    --------
    Round-trip through fftshift / ifftshift:

    >>> X = lucid.fft.fft(lucid.randn(16))
    >>> X_c = lucid.fft.fftshift(X)
    >>> X_back = lucid.fft.ifftshift(X_c)
    >>> # X_back ≈ X  (exact, not approximate)

    Typical pipeline for constructing a centred filter:

    >>> N = 64
    >>> freqs = lucid.fft.fftshift(lucid.fft.fftfreq(N))
    >>> # Build H_centred (e.g. Gaussian low-pass) in centred coords
    >>> # H = ...
    >>> # Then convert back to FFT ordering before inverting:
    >>> # x_filtered = lucid.fft.ifft(lucid.fft.ifftshift(H))
    """
    rank = input.ndim
    dims = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    shifts = [-(int(input.shape[a]) // 2) for a in dims]
    return lucid.roll(input, shifts=shifts, dims=dims)


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
