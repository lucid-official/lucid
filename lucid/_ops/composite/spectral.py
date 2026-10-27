"""Short-time Fourier transforms and the reference's window factories.

``stft`` frames a signal (``unfold``), tapers each frame with a window and
transforms it (``lucid.fft``); ``istft`` inverts each frame and overlap-adds
them back, dividing out the summed squared window.  Both are composites of
differentiable ops, so autograd reaches the signal and the window alike.

The five ``*_window`` factories are the reference's top-level spelling
(``window_length``, ``periodic``), and only that: the window math lives in
:mod:`lucid.signal.windows`, which these forward to.  ``periodic=True`` is
``sym=False`` there.
"""

import warnings

import lucid
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike

# ``lucid.fft``, ``lucid.signal`` and ``lucid.nn`` are imported where they
# are used: this module loads with ``import lucid``, which must not load them.


# ── window factories ─────────────────────────────────────────────────────────


def _as_factory(window: Tensor, requires_grad: bool) -> Tensor:
    return window.requires_grad_(True) if requires_grad else window


def hann_window(
    window_length: int,
    periodic: bool = True,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Hann window, in the periodic form ``stft`` expects by default.

    Parameters
    ----------
    window_length : int
        Number of samples; ``>= 0``.
    periodic : bool, optional
        If ``True`` (default) return the first ``window_length`` samples of
        the symmetric window of length ``window_length + 1``, which tiles
        under a hop that divides the length.  If ``False`` return the
        symmetric window.
    dtype : DTypeLike, optional
        A floating-point dtype; the default dtype when omitted.
    device : DeviceLike, optional
        Target device.
    requires_grad : bool, optional
        Whether the returned window records gradients.

    Returns
    -------
    Tensor
        1-D tensor of length ``window_length``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        ``window_length`` is negative.
    lucid._C.engine.DtypeMismatch
        ``dtype`` is not a floating-point dtype.

    Notes
    -----
    .. math::

        w[n] = \frac{1}{2}\left(1 - \cos\frac{2\pi n}{N - 1}\right),

    with :math:`N` the window length, plus one when ``periodic``.  The
    same window as :func:`lucid.signal.windows.hann` with ``sym`` set to
    ``not periodic``.

    Examples
    --------
    >>> import lucid
    >>> w = lucid.hann_window(4)  # zero at n = 0 up to float32 round-off
    >>> lucid.allclose(w, lucid.tensor([0.0, 0.5, 1.0, 0.5]), atol=1e-6)
    True
    """
    from lucid.signal.windows import hann

    w = hann(window_length, sym=not periodic, dtype=dtype, device=device)
    return _as_factory(w, requires_grad)


def hamming_window(
    window_length: int,
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Hamming window, or the two-term cosine window ``alpha``/``beta`` give.

    Parameters
    ----------
    window_length, periodic : int, bool
        The length, and the periodic or symmetric form, as for
        :func:`hann_window`.
    alpha : float, optional
        Constant term; ``0.54`` (default) for Hamming.
    beta : float, optional
        Cosine coefficient; ``0.46`` (default) for Hamming.
    dtype, device, requires_grad : DTypeLike, DeviceLike, bool
        As for :func:`hann_window`.

    Returns
    -------
    Tensor
        1-D tensor of length ``window_length``; refusals as for
        :func:`hann_window`.

    Notes
    -----
    .. math::

        w[n] = \alpha - \beta \cos\frac{2\pi n}{N - 1},

    computed by :func:`lucid.signal.windows.general_cosine` with
    coefficients ``[alpha, beta]``.  ``alpha = beta = 0.5`` is the Hann
    window.

    Examples
    --------
    >>> import lucid
    >>> lucid.hamming_window(4)
    tensor([0.08, 0.54, 1., 0.54])
    """
    from lucid.signal.windows import general_cosine

    w = general_cosine(window_length, [alpha, beta], sym=not periodic, dtype=dtype, device=device)
    return _as_factory(w, requires_grad)


def blackman_window(
    window_length: int,
    periodic: bool = True,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Blackman window, in the periodic form ``stft`` expects by default.

    Parameters
    ----------
    window_length : int
        Number of samples; ``>= 0``.
    periodic : bool, optional
        ``True`` (default) for the form that tiles under ``stft``, ``False``
        for the symmetric one.
    dtype, device, requires_grad : DTypeLike, DeviceLike, bool
        The factory options :func:`hann_window` documents.

    Returns
    -------
    Tensor
        The window, shape ``(window_length,)``.  What it refuses is what
        :func:`hann_window` refuses.

    Notes
    -----
    .. math::

        w[n] = 0.42 - 0.5 \cos\frac{2\pi n}{N - 1}
               + 0.08 \cos\frac{4\pi n}{N - 1},

    as :func:`lucid.signal.windows.blackman` with ``sym=not periodic``.

    Examples
    --------
    >>> import lucid
    >>> w = lucid.blackman_window(5, periodic=False)
    >>> lucid.allclose(w, lucid.tensor([0.0, 0.34, 1.0, 0.34, 0.0]), atol=1e-6)
    True
    """
    from lucid.signal.windows import blackman

    w = blackman(window_length, sym=not periodic, dtype=dtype, device=device)
    return _as_factory(w, requires_grad)


def bartlett_window(
    window_length: int,
    periodic: bool = True,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Bartlett (triangular) window, periodic by default.

    Parameters
    ----------
    window_length, periodic : int, bool
        The length, and which of the two forms, as for :func:`hann_window`.
    dtype, device, requires_grad : DTypeLike, DeviceLike, bool
        Output dtype (floating point), device and gradient flag, as for
        :func:`hann_window`.

    Returns
    -------
    Tensor
        A 1-D tensor of ``window_length`` samples.  It refuses what
        :func:`hann_window` refuses.

    Notes
    -----
    .. math::

        w[n] = 1 - \left|\frac{2n}{N - 1} - 1\right|,

    as :func:`lucid.signal.windows.bartlett` with ``sym=not periodic``.

    Examples
    --------
    >>> import lucid
    >>> lucid.bartlett_window(4)
    tensor([0., 0.5, 1., 0.5])
    """
    from lucid.signal.windows import bartlett

    w = bartlett(window_length, sym=not periodic, dtype=dtype, device=device)
    return _as_factory(w, requires_grad)


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Kaiser window with shape parameter ``beta``, periodic by default.

    Parameters
    ----------
    window_length, periodic : int, bool
        Number of samples and the form (periodic by default); see
        :func:`hann_window`.
    beta : float, optional
        Shape parameter; ``12.0`` by default.  Larger values widen the main
        lobe and lower the side lobes.
    dtype, device, requires_grad : DTypeLike, DeviceLike, bool
        See :func:`hann_window`.

    Returns
    -------
    Tensor
        The ``window_length`` samples, 1-D; the refusals are
        :func:`hann_window`'s.

    Notes
    -----
    .. math::

        w[n] = \frac{I_0\left(\beta\sqrt{1 - (2n/(N - 1) - 1)^2}\right)}
                    {I_0(\beta)},

    as :func:`lucid.signal.windows.kaiser` with ``sym=not periodic``.

    Examples
    --------
    >>> import lucid
    >>> lucid.kaiser_window(3, periodic=False, beta=0.0)
    tensor([1., 1., 1.])
    """
    from lucid.signal.windows import kaiser

    w = kaiser(window_length, beta, sym=not periodic, dtype=dtype, device=device)
    return _as_factory(w, requires_grad)


# ── shared argument checks ───────────────────────────────────────────────────


def _invalid(op: str, message: str) -> _C_engine.InvalidArgument:
    return _C_engine.InvalidArgument(f"{op}: {message}")


def _frame_lengths(
    op: str, n_fft: int, hop_length: int | None, win_length: int | None
) -> tuple[int, int]:
    """``hop_length`` and ``win_length`` with their defaults, checked."""
    if n_fft <= 0:
        raise _invalid(op, f"expected n_fft > 0, got n_fft={n_fft}")
    hop = n_fft // 4 if hop_length is None else hop_length
    win = n_fft if win_length is None else win_length
    if hop <= 0:
        raise _invalid(op, f"expected hop_length > 0, got hop_length={hop}")
    if not 0 < win <= n_fft:
        raise _invalid(op, f"expected 0 < win_length <= n_fft={n_fft}, got win_length={win}")
    return hop, win


def _padded_window(
    op: str, window: Tensor | None, n_fft: int, win_length: int, like: Tensor
) -> Tensor | None:
    """The window zero-padded to ``n_fft``, centred; ``None`` means all ones."""
    if window is not None and (window.ndim != 1 or window.shape[0] != win_length):
        raise _C_engine.ShapeMismatch(
            f"{op}: expected a 1-D window of length win_length={win_length}, "
            f"got shape {tuple(window.shape)}"
        )
    if win_length == n_fft:
        return window
    if window is None:
        window = _ones(win_length, like)
    from lucid.nn.functional import pad

    left = (n_fft - win_length) // 2
    return pad(window, (left, n_fft - win_length - left))


def _ones(n: int, like: Tensor) -> Tensor:
    """A rectangular window in ``like``'s real dtype, on its device."""
    dtype: DTypeLike = like.dtype
    if like.dtype == lucid.complex64:
        dtype = lucid.float32
    elif like.dtype == lucid.complex128:
        dtype = lucid.float64
    return lucid.ones(n, dtype=dtype, device=like.device)


def _norm(normalized: bool) -> str:
    return "ortho" if normalized else "backward"


# ── stft ─────────────────────────────────────────────────────────────────────


def _check_signal(input: Tensor, center: bool, align_to_window: bool | None) -> None:
    if center and align_to_window is not None:
        raise _invalid("stft", "align_to_window may be set only when center=False")
    if not (input.is_floating_point() or input.is_complex()):
        raise _C_engine.DtypeMismatch(
            f"stft: expected a floating-point or complex signal, got {input.dtype}"
        )
    if input.ndim not in (1, 2):
        raise _C_engine.ShapeMismatch(
            f"stft: expected a 1-D or 2-D (batch, time) signal, got " f"shape {tuple(input.shape)}"
        )


def _frames(
    signal: Tensor,
    n_fft: int,
    hop: int,
    win: int,
    center: bool,
    pad_mode: str,
    align_to_window: bool,
) -> Tensor:
    """``signal`` (batch, time) cut into ``(batch, n_frames, n_fft)`` frames."""
    from lucid.nn.functional import pad

    if center:
        signal = pad(signal, (n_fft // 2, n_fft // 2), mode=pad_mode)
    length = int(signal.shape[-1])
    if n_fft > length:
        raise _invalid(
            "stft",
            f"expected n_fft <= the signal length {length}"
            f"{' after centre padding' if center else ''}, got n_fft={n_fft}",
        )
    if align_to_window:
        # The window starts at t = 0: the zero margins of the centred window
        # hang off the signal's ends instead of eating into it.  Padding by
        # n_fft - win in all gives unfold the reference's frame count,
        # 1 + (length - win) // hop, odd margins included.
        left = (n_fft - win) // 2
        signal = pad(signal, (left, n_fft - win - left))
    return signal.unfold(-1, n_fft, hop)


def stft(
    input: Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool | None = None,
    return_complex: bool | None = None,
    align_to_window: bool | None = None,
) -> Tensor:
    r"""Short-time Fourier transform.

    Cuts ``input`` into frames of ``n_fft`` samples, ``hop_length`` apart,
    multiplies each by ``window`` and returns the DFT of every frame.

    Parameters
    ----------
    input : Tensor
        Real or complex signal, ``(time,)`` or ``(batch, time)``.
    n_fft : int
        Frame and transform length.
    hop_length : int, optional
        Distance between frame starts; ``n_fft // 4`` by default.
    win_length : int, optional
        Window length, ``<= n_fft`` (default ``n_fft``); a shorter window
        is zero-padded on both sides to ``n_fft``.
    window : Tensor, optional
        1-D, of length ``win_length``; rectangular when omitted.
    center : bool, optional
        Pad ``input`` by ``n_fft // 2`` at both ends first (default), so
        frame ``t`` is centred on sample ``t * hop_length``.
    pad_mode : str, optional
        That padding: ``"reflect"`` (default), ``"constant"``,
        ``"replicate"`` or ``"circular"``.
    normalized : bool, optional
        Scale by :math:`1/\sqrt{n_{fft}}`, the unitary transform.
    onesided : bool, optional
        Keep the ``n_fft // 2 + 1`` non-negative frequencies: the default
        for a real signal and window, refused for a complex one.
    return_complex : bool, optional
        Required for a real signal and window.  ``False`` returns the real
        and imaginary parts in a trailing axis, as :func:`lucid.view_as_real`.
    align_to_window : bool, optional
        With ``center=False``, start the first frame where the window
        starts rather than where its zero-padded ``n_fft`` frame does.

    Returns
    -------
    Tensor
        ``([batch,] n_freq, n_frames)``: ``n_freq`` is ``n_fft // 2 + 1``
        one-sided and ``n_fft`` otherwise, ``n_frames`` is
        ``1 + (L - n_fft) // hop_length`` for the padded length ``L``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        A length out of range, or flags that contradict each other.
    lucid._C.engine.ShapeMismatch, lucid._C.engine.DtypeMismatch
        ``input`` is not a 1-D or 2-D floating or complex signal, or
        ``window`` is not 1-D of length ``win_length``.

    Notes
    -----
    .. math::

        X[\omega, m] = \sum_{k=0}^{n_{fft}-1}
            w[k]\, x[m \cdot \text{hop} + k]\, e^{-2\pi i\, \omega k / n_{fft}}.

    Differentiable in ``input`` and ``window``.  A ``float64`` signal on CPU
    produces a ``complex128`` spectrum via the Accelerate double-precision
    FFT path.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 1000)
    >>> spec = lucid.stft(x, 256, window=lucid.hann_window(256), return_complex=True)
    >>> spec.shape
    (2, 129, 16)
    """
    from lucid import fft as _fft

    _check_signal(input, center, align_to_window)
    hop, win = _frame_lengths("stft", n_fft, hop_length, win_length)
    padded_window = _padded_window("stft", window, n_fft, win, input)
    as_complex, one_sided = _stft_output(input, window, return_complex, onesided)

    signal = input.unsqueeze(0) if input.ndim == 1 else input
    frames = _frames(signal, n_fft, hop, win, center, pad_mode, bool(align_to_window))
    if padded_window is not None:
        frames = frames * padded_window
    transform = _fft.rfft if one_sided else _fft.fft
    out = transform(frames, dim=-1, norm=_norm(normalized)).transpose(-1, -2)
    if input.ndim == 1:
        out = out.squeeze(0)
    return out if as_complex else lucid.view_as_real(out)


def _stft_output(
    input: Tensor, window: Tensor | None, return_complex: bool | None, onesided: bool | None
) -> tuple[bool, bool]:
    """``return_complex`` and ``onesided`` resolved against the frames' dtype."""
    complex_frames = input.is_complex() or (window is not None and window.is_complex())
    if return_complex is None and not complex_frames:
        raise _invalid(
            "stft",
            "return_complex must be given for a real signal and window; "
            "pass return_complex=True",
        )
    one_sided = not complex_frames if onesided is None else onesided
    if one_sided and complex_frames:
        raise _invalid("stft", "a complex signal or window has no one-sided spectrum")
    return (True if return_complex is None else return_complex), one_sided


# ── istft ────────────────────────────────────────────────────────────────────


def _overlap_add(frames: Tensor, hop: int) -> Tensor:
    """``(batch, n_frames, n_fft)`` frames summed at ``hop`` spacing.

    The adjoint of ``unfold`` — a 1-D ``fold`` — so the gradient of every
    output sample flows back to each frame that covered it.  The fold
    kernel is real, so a complex signal is folded lane by lane.
    """
    from lucid.nn.functional import fold

    if frames.is_complex():
        return lucid.complex(
            _overlap_add(lucid.real(frames), hop), _overlap_add(lucid.imag(frames), hop)
        )
    batch, n_frames, n_fft = (int(s) for s in frames.shape)
    length = n_fft + hop * (n_frames - 1)
    folded = fold(
        frames.transpose(1, 2), output_size=(1, length), kernel_size=(1, n_fft), stride=(1, hop)
    )
    return folded.reshape(batch, length)


def _check_spectrum(input: Tensor, n_fft: int, onesided: bool | None) -> bool:
    """Validate a spectrum for ``istft``; return whether it is one-sided."""
    if not input.is_complex():
        raise _C_engine.DtypeMismatch(
            "istft: expected a complex spectrum, as stft returns with "
            f"return_complex=True, got {input.dtype}"
        )
    if input.ndim not in (2, 3):
        raise _C_engine.ShapeMismatch(
            "istft: expected a (n_freq, n_frames) or (batch, n_freq, n_frames) "
            f"spectrum, got shape {tuple(input.shape)}"
        )
    if input.numel() == 0:
        raise _invalid("istft", "the spectrum is empty")
    n_freq = int(input.shape[-2])
    one_sided = n_freq != n_fft if onesided is None else onesided
    expected = n_fft // 2 + 1 if one_sided else n_fft
    if n_freq != expected:
        raise _C_engine.ShapeMismatch(
            f"istft: a {'one' if one_sided else 'two'}-sided spectrum of n_fft={n_fft} "
            f"has {expected} frequency bins, got {n_freq}"
        )
    return one_sided


def _frames_of(
    spectrum: Tensor,
    n_fft: int,
    one_sided: bool,
    normalized: bool,
    return_complex: bool,
    window: Tensor | None,
) -> Tensor:
    """The inverse transform of every frame: ``(batch, n_frames, n_fft)``."""
    from lucid import fft as _fft

    frames = spectrum.transpose(-1, -2)
    norm = _norm(normalized)
    if return_complex:
        if one_sided:
            raise _invalid("istft", "a complex output needs a two-sided spectrum")
        return _fft.ifft(frames, dim=-1, norm=norm)
    if window is not None and window.is_complex():
        raise _invalid("istft", "a complex window needs return_complex=True")
    if not one_sided:
        frames = frames.narrow(-1, 0, n_fft // 2 + 1)
    return _fft.irfft(frames, n=n_fft, dim=-1, norm=norm)


def _envelope(window: Tensor | None, frames: Tensor, hop: int) -> Tensor:
    """The overlap-added squared window that ``istft`` divides out."""
    _, n_frames, n_fft = (int(s) for s in frames.shape)
    if window is None:
        window = _ones(n_fft, frames)
    squared = (window * window).reshape(1, 1, n_fft).expand(1, n_frames, n_fft)
    return _overlap_add(squared, hop)


def _output_span(
    n_fft: int, hop: int, n_frames: int, center: bool, length: int | None
) -> tuple[int, int, int]:
    """``(start, end, span)``: the output is ``[start, end)`` of the frames' span."""
    span = n_fft + hop * (n_frames - 1)
    start = n_fft // 2 if center else 0
    # Centring padded both ends by ``n_fft // 2``; without a length both go.
    end = span - start if length is None else start + length
    if end <= start:
        raise _invalid("istft", f"the output would be empty (length={length})")
    return start, end, span


def _overlap_add_normalised(
    frames: Tensor, window: Tensor | None, hop: int, start: int, kept: int
) -> Tensor:
    """Samples ``[start, start + kept)`` of the frames' overlap-add, divided by
    the window's — refused where that is zero (the NOLA condition)."""
    # Imported here: ``lucid.compile`` imports the tensor this module builds on.
    from lucid.compile import _host_check

    signal = _overlap_add(frames, hop).narrow(-1, start, kept)
    envelope = _envelope(window, frames, hop).narrow(-1, start, kept)
    # The read only decides whether to raise; the division below uses the
    # envelope itself, so a trace does not count the read as a dependency.
    if not envelope._impl.is_phantom:
        with _host_check():
            lowest = float(envelope.abs().min().item())
        if lowest < 1e-11:
            raise _invalid(
                "istft",
                "the window's overlap-add is zero somewhere in the output (it fails "
                "the NOLA condition); use a window and hop whose overlap covers "
                "every sample",
            )
    return signal / envelope


def _pad_tail(signal: Tensor, missing: int) -> Tensor:
    """``signal`` with ``missing`` zeros after it, and a warning that says so."""
    warnings.warn(
        f"istft: length runs {missing} samples past what the frames cover; the "
        "tail is padded with zeros",
        stacklevel=3,
    )
    from lucid.nn.functional import pad

    return pad(signal, (0, missing))


def istft(
    input: Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: Tensor | None = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool | None = None,
    length: int | None = None,
    return_complex: bool = False,
) -> Tensor:
    r"""Inverse short-time Fourier transform.

    Inverts every frame of a spectrum ``stft`` produced, multiplies it by
    ``window`` again, overlap-adds the frames and divides by the
    overlap-added squared window, so ``istft(stft(x))`` returns ``x`` for
    any window that satisfies the non-zero overlap-add (NOLA) condition.

    Parameters
    ----------
    input : Tensor
        Complex spectrum, ``(n_freq, n_frames)`` or
        ``(batch, n_freq, n_frames)``.
    n_fft : int
        Frame length the spectrum was computed with.
    hop_length : int, optional
        Frame distance, ``<= win_length``; ``n_fft // 4`` by default.
    win_length : int, optional
        Window length, ``<= n_fft``; ``n_fft`` by default.
    window : Tensor, optional
        The forward transform's window; rectangular when omitted.
    center, normalized : bool, optional
        Whether ``input`` came from a centred (default) or normalised
        ``stft``; centring's padding is trimmed again.
    onesided : bool, optional
        Whether ``input`` holds only the non-negative frequencies; when
        omitted, unless it has ``n_fft`` bins.
    length : int, optional
        Output length: trims the signal, or pads it with zeros (with a
        warning) past what the frames cover.
    return_complex : bool, optional
        Return a complex signal, from a two-sided spectrum.  ``False``
        (default) returns a real one, reading the spectrum as Hermitian.

    Returns
    -------
    Tensor
        ``([batch,] length)``; without ``length``, the frames' span
        ``n_fft + hop_length * (n_frames - 1)``, less ``2 * (n_fft // 2)``
        when ``center``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        A length out of range, an empty spectrum or output, a window that
        fails NOLA, or ``return_complex`` against ``onesided`` or the window.
    lucid._C.engine.ShapeMismatch, lucid._C.engine.DtypeMismatch
        ``input`` is not a complex spectrum of the right rank and bin count,
        or ``window`` has the wrong shape.

    Notes
    -----
    With :math:`y_m` the inverse transform of frame :math:`m`, hop :math:`H`:

    .. math::

        x[t] = \frac{\sum_m w[t - mH]\, y_m[t - mH]}{\sum_m w[t - mH]^2}.

    Differentiable in ``input`` and ``window``.  The NOLA check reads one
    value back to the host, as the reference does.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(1000)
    >>> w = lucid.hann_window(256)
    >>> spec = lucid.stft(x, 256, window=w, return_complex=True)
    >>> y = lucid.istft(spec, 256, window=w, length=1000)
    >>> bool(lucid.allclose(x, y, atol=1e-5))
    True
    """
    one_sided = _check_spectrum(input, n_fft, onesided)
    hop, win = _frame_lengths("istft", n_fft, hop_length, win_length)
    if hop > win:
        raise _invalid("istft", f"expected hop_length <= win_length={win}, got hop_length={hop}")
    spectrum = input.unsqueeze(0) if input.ndim == 2 else input
    frames = _frames_of(spectrum, n_fft, one_sided, normalized, return_complex, window)
    padded_window = _padded_window("istft", window, n_fft, win, frames)
    if padded_window is not None:
        frames = frames * padded_window
    start, end, span = _output_span(n_fft, hop, int(frames.shape[-2]), center, length)
    signal = _overlap_add_normalised(frames, padded_window, hop, start, min(end, span) - start)
    if input.ndim == 2:
        signal = signal.squeeze(0)
    return _pad_tail(signal, end - span) if end > span else signal


__all__ = [
    "bartlett_window",
    "blackman_window",
    "hamming_window",
    "hann_window",
    "istft",
    "kaiser_window",
    "stft",
]
