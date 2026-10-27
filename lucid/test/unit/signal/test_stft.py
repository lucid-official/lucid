"""``lucid.stft`` / ``lucid.istft`` and the ``*_window`` factories (LCD-109).

Lucid had none of these.  ``stft`` frames, windows and transforms a
signal; ``istft`` inverts the frames, overlap-adds them and divides out the
summed squared window; the five factories forward to
``lucid.signal.windows``, the one owner of the window math.

What holds on every run: ``stft`` equals the DFT of each windowed frame
written out with numpy, ``istft(stft(x))`` returns ``x``, the factories
equal their closed forms, and every argument the reference refuses is
refused here with the class it shares with the reference.  Under
``-m parity`` the reference is the oracle: a matrix over ``n_fft``, hop,
window length, centring and its pad mode, normalisation, one-sidedness,
window and batch, both transforms, both devices, plus gradients through
the signal and the window.
"""

import itertools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._helpers.compare import assert_close

_WINDOWS = ["hann", "hamming", "blackman", "bartlett", "kaiser"]


def _signal(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _factory(name: str) -> Callable[..., lucid.Tensor]:
    fn: Callable[..., lucid.Tensor] = getattr(lucid, f"{name}_window")
    return fn


# ── window factories ─────────────────────────────────────────────────────────


def _closed_form(name: str, n: int, periodic: bool) -> np.ndarray:
    """The symmetric window of length ``N`` (``n + 1`` when periodic), cut to ``n``."""
    if n <= 1:
        return np.ones(n)
    N = n + 1 if periodic else n
    k = np.arange(N, dtype=np.float64)
    x = 2 * np.pi * k / (N - 1)
    if name == "hann":
        w = 0.5 - 0.5 * np.cos(x)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(x)
    elif name == "blackman":
        w = 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)
    elif name == "bartlett":
        w = 1 - np.abs(2 * k / (N - 1) - 1)
    else:
        w = np.i0(12.0 * np.sqrt(1 - (2 * k / (N - 1) - 1) ** 2)) / np.i0(12.0)
    return w[:n]


@pytest.mark.parametrize("name", _WINDOWS)
@pytest.mark.parametrize("n", [0, 1, 2, 5, 16])
@pytest.mark.parametrize("periodic", [True, False])
def test_a_window_is_its_closed_form(name: str, n: int, periodic: bool, device: str) -> None:
    w = _factory(name)(n, periodic, device=device)
    assert w.shape == (n,) and w.dtype == lucid.float32 and w.device.type == device
    np.testing.assert_allclose(w.numpy(), _closed_form(name, n, periodic), atol=2e-6)


def test_the_factories_forward_to_signal_windows() -> None:
    windows = lucid.signal.windows
    for periodic in (True, False):
        pairs = [
            (lucid.hann_window(9, periodic), windows.hann(9, sym=not periodic)),
            (
                lucid.hamming_window(9, periodic, 0.6, 0.3),
                windows.general_cosine(9, [0.6, 0.3], sym=not periodic),
            ),
            (lucid.blackman_window(9, periodic), windows.blackman(9, sym=not periodic)),
            (lucid.bartlett_window(9, periodic), windows.bartlett(9, sym=not periodic)),
            (lucid.kaiser_window(9, periodic, 5.0), windows.kaiser(9, 5.0, sym=not periodic)),
        ]
        for top, owner in pairs:
            np.testing.assert_array_equal(top.numpy(), owner.numpy())


@pytest.mark.parametrize("name", _WINDOWS)
def test_a_window_takes_dtype_and_requires_grad(name: str) -> None:
    w = _factory(name)(6, dtype=lucid.float64, requires_grad=True)
    assert w.dtype == lucid.float64 and w.requires_grad and w.is_leaf
    assert not _factory(name)(6).requires_grad


@pytest.mark.parametrize("name", _WINDOWS)
@pytest.mark.parametrize("dtype", ["int32", "bool_", "complex64"])
def test_a_window_refuses_a_dtype_it_cannot_be_computed_in(name: str, dtype: str) -> None:
    with pytest.raises(_C_engine.DtypeMismatch):
        _factory(name)(4, dtype=getattr(lucid, dtype))
    with pytest.raises(TypeError):
        getattr(lucid.signal.windows, name)(4, dtype=getattr(lucid, dtype))


@pytest.mark.parametrize("name", _WINDOWS)
def test_a_negative_window_length_is_refused(name: str) -> None:
    with pytest.raises(_C_engine.InvalidArgument, match=">= 0"):
        _factory(name)(-1)


# ── stft against its definition ──────────────────────────────────────────────


def _numpy_stft(
    x: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    center: bool,
    onesided: bool,
    normalized: bool,
) -> np.ndarray:
    """``stft`` of a 1-D signal, frame by frame, with numpy's DFT."""
    if center:
        x = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    win = np.zeros(n_fft)
    left = (n_fft - len(window)) // 2
    win[left : left + len(window)] = window
    frames = np.stack([x[m * hop : m * hop + n_fft] * win for m in range(n_frames)])
    spec = np.fft.rfft(frames, axis=-1) if onesided else np.fft.fft(frames, axis=-1)
    if normalized:
        spec = spec / np.sqrt(n_fft)
    return spec.T


_DEFINITION_CASES = [
    # n_fft, hop, win_length, center, onesided, normalized
    (16, 4, 16, True, True, False),
    (16, 4, 16, False, True, False),
    (16, 5, 11, True, True, True),
    (15, 3, 15, True, False, False),
    (8, 8, 8, False, False, True),
]


@pytest.mark.parametrize(
    "n_fft, hop, win, center, onesided, normalized",
    _DEFINITION_CASES,
    ids=[f"n{c[0]}-h{c[1]}-w{c[2]}-c{c[3]:d}-o{c[4]:d}-z{c[5]:d}" for c in _DEFINITION_CASES],
)
def test_stft_is_the_dft_of_each_windowed_frame(
    n_fft: int, hop: int, win: int, center: bool, onesided: bool, normalized: bool, device: str
) -> None:
    x = _signal((61,))
    window = lucid.hann_window(win, device=device)
    got = lucid.stft(
        lucid.tensor(x, device=device),
        n_fft,
        hop,
        win,
        window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    expected = _numpy_stft(x, n_fft, hop, window.numpy(), center, onesided, normalized)
    assert got.dtype == lucid.complex64 and got.device.type == device
    assert_close(got, expected, atol=1e-4)


def test_stft_batches_the_leading_axis(device: str) -> None:
    x = _signal((3, 50))
    w = lucid.hann_window(16, device=device)
    batched = lucid.stft(lucid.tensor(x, device=device), 16, window=w, return_complex=True)
    assert batched.shape == (3, 9, 13)
    for b in range(3):
        alone = lucid.stft(lucid.tensor(x[b], device=device), 16, window=w, return_complex=True)
        assert_close(batched[b], alone, atol=1e-5)


def test_stft_without_return_complex_is_real_pairs() -> None:
    x = lucid.tensor(_signal((40,)))
    pairs = lucid.stft(x, 8, return_complex=False)
    spec = lucid.stft(x, 8, return_complex=True)
    assert pairs.shape == (*spec.shape, 2) and pairs.dtype == lucid.float32
    assert_close(pairs, lucid.view_as_real(spec), atol=0.0)


# ── istft inverts stft ───────────────────────────────────────────────────────

_ROUND_TRIPS = [
    # n_fft, hop, win_length, center, normalized, batch
    (16, 4, None, True, False, False),
    (16, 4, None, True, True, True),
    (16, 8, 12, True, False, True),
    (15, 5, None, True, False, False),
    (32, 8, None, True, False, True),
]


@pytest.mark.parametrize(
    "n_fft, hop, win, center, normalized, batch",
    _ROUND_TRIPS,
    ids=[f"n{c[0]}-h{c[1]}-w{c[2]}-c{c[3]:d}-z{c[4]:d}-b{c[5]:d}" for c in _ROUND_TRIPS],
)
def test_istft_of_stft_is_the_signal(
    n_fft: int, hop: int, win: int | None, center: bool, normalized: bool, batch: bool, device: str
) -> None:
    x = lucid.tensor(_signal((2, 100) if batch else (100,)), device=device)
    w = lucid.hann_window(win or n_fft, device=device)
    spec = lucid.stft(
        x, n_fft, hop, win, w, center=center, normalized=normalized, return_complex=True
    )
    y = lucid.istft(spec, n_fft, hop, win, w, center=center, normalized=normalized, length=100)
    assert y.shape == x.shape and y.device.type == device
    assert_close(y, x, atol=2e-5)


def test_istft_inverts_an_uncentred_two_sided_complex_transform(device: str) -> None:
    re, im = _signal((64,), 1), _signal((64,), 2)
    x = lucid.complex(lucid.tensor(re, device=device), lucid.tensor(im, device=device))
    w = lucid.hann_window(16, device=device) + 0.1  # non-zero at the edges
    spec = lucid.stft(x, 16, 4, window=w, center=False)
    assert spec.shape == (16, 13)
    y = lucid.istft(spec, 16, 4, window=w, center=False, return_complex=True)
    assert_close(y, x.narrow(-1, 0, y.shape[-1]), atol=2e-5)


def test_istft_pads_a_long_length_with_zeros_and_says_so() -> None:
    x = lucid.tensor(_signal((40,)))
    w = lucid.hann_window(8)
    spec = lucid.stft(x, 8, 2, window=w, return_complex=True)
    # 21 frames span 8 + 2 * 20 = 48 samples; centring trims the first 4, so
    # the frames reach sample 44 and the rest of a length-50 output is padding.
    with pytest.warns(UserWarning, match="padded with zeros"):
        y = lucid.istft(spec, 8, 2, window=w, length=50)
    assert y.shape == (50,)
    assert_close(y.narrow(0, 0, 40), x, atol=2e-5)
    assert float(y.narrow(0, 44, 6).abs().max().item()) == 0.0


def test_the_nola_check_leaves_a_trace_supported(device: str) -> None:
    # The NOLA check reads the envelope's minimum on the host only to decide
    # whether to raise; a read of a traced value would otherwise mark the
    # recording unsupported and send every compiled istft back to eager.
    w = lucid.hann_window(16, device=device)
    spec = lucid.stft(
        lucid.tensor(_signal((64,)), device=device), 16, 4, window=w, return_complex=True
    )
    with lucid.no_grad(), lucid.compile._tracing(watch=[spec]) as tracer:
        lucid.istft(spec, 16, 4, window=w, length=64)
    assert len(tracer.graph.ops) > 0
    assert tracer.graph.unsupported == ""


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradients_reach_the_signal_and_the_window(device: str) -> None:
    x = lucid.tensor(_signal((2, 40)), device=device, requires_grad=True)
    w = lucid.tensor(np.hanning(8).astype(np.float32) + 0.1, device=device)
    w.requires_grad_(True)
    spec = lucid.stft(x, 8, 2, window=w, return_complex=True)
    y = lucid.istft(spec, 8, 2, window=w, length=40)
    (y * y).sum().backward()
    assert x.grad is not None and w.grad is not None
    # istft(stft(x)) is x whatever the window, so the loss is sum(x^2): its
    # gradient is 2x in the signal and zero in the window.
    assert_close(x.grad, 2 * x.detach(), atol=1e-4)
    assert_close(w.grad, lucid.zeros_like(w), atol=1e-4)


# ── refusals ─────────────────────────────────────────────────────────────────


def _x() -> lucid.Tensor:
    return lucid.zeros(40)


def _spec() -> lucid.Tensor:
    return lucid.stft(_x(), 8, return_complex=True)


_Invalid, _Shape, _Dtype = (
    _C_engine.InvalidArgument,
    _C_engine.ShapeMismatch,
    _C_engine.DtypeMismatch,
)

# (what, call, class).  The engine classes are also the ``RuntimeError`` the
# reference raises; the two padding refusals come from ``nn.functional.pad``.
_REFUSALS: list[tuple[str, Callable[[], object], type[Exception]]] = [
    ("no return_complex", lambda: lucid.stft(_x(), 8), _Invalid),
    (
        "align_to_window with center",
        lambda: lucid.stft(_x(), 8, return_complex=True, align_to_window=True),
        _Invalid,
    ),
    (
        "n_fft longer than the signal",
        lambda: lucid.stft(_x(), 64, center=False, return_complex=True),
        _Invalid,
    ),
    ("n_fft zero", lambda: lucid.stft(_x(), 0, return_complex=True), _Invalid),
    ("hop zero", lambda: lucid.stft(_x(), 8, 0, return_complex=True), _Invalid),
    ("win over n_fft", lambda: lucid.stft(_x(), 8, 2, 9, return_complex=True), _Invalid),
    (
        "window length",
        lambda: lucid.stft(_x(), 8, window=lucid.ones(7), return_complex=True),
        _Shape,
    ),
    ("3-D signal", lambda: lucid.stft(lucid.zeros(2, 2, 40), 8, return_complex=True), _Shape),
    (
        "integer signal",
        lambda: lucid.stft(lucid.zeros(40, dtype=lucid.int32), 8, return_complex=True),
        _Dtype,
    ),
    (
        "one-sided complex",
        lambda: lucid.stft(lucid.complex(_x(), _x()), 8, onesided=True),
        _Invalid,
    ),
    (
        "reflect past the signal",
        lambda: lucid.stft(lucid.zeros(3), 8, return_complex=True),
        ValueError,
    ),
    (
        "unknown pad mode",
        lambda: lucid.stft(_x(), 8, pad_mode="mirror", return_complex=True),
        NotImplementedError,
    ),
    ("istft real input", lambda: lucid.istft(lucid.zeros(5, 21), 8), _Dtype),
    ("istft 1-D", lambda: lucid.istft(lucid.complex(_x(), _x()), 8), _Shape),
    ("istft bin count", lambda: lucid.istft(_spec(), 16), _Shape),
    ("istft hop over win", lambda: lucid.istft(_spec(), 8, 9), _Invalid),
    ("istft one-sided complex", lambda: lucid.istft(_spec(), 8, return_complex=True), _Invalid),
    ("istft NOLA", lambda: lucid.istft(_spec(), 8, window=lucid.zeros(8)), _Invalid),
    ("istft empty", lambda: lucid.istft(_spec(), 8, length=0), _Invalid),
]


@pytest.mark.parametrize("call, kind", [r[1:] for r in _REFUSALS], ids=[r[0] for r in _REFUSALS])
def test_a_bad_argument_is_refused(call: Callable[[], object], kind: type[Exception]) -> None:
    with pytest.raises(kind) as info:
        call()
    if issubclass(kind, _C_engine.LucidError):
        assert isinstance(info.value, RuntimeError)


def test_a_float64_signal_gives_a_complex128_spectrum() -> None:
    x = lucid.tensor(_signal((40,)).astype(np.float64))
    spec = lucid.stft(x, 8, window=lucid.hann_window(8, dtype=lucid.float64), return_complex=True)
    assert spec.dtype == lucid.complex128


def test_a_complex128_spectrum_inverts_to_float64() -> None:
    spec = lucid.complex(
        lucid.zeros(5, 11, dtype=lucid.float64), lucid.zeros(5, 11, dtype=lucid.float64)
    )
    assert (
        lucid.istft(spec, 8, window=lucid.hann_window(8, dtype=lucid.float64)).dtype
        == lucid.float64
    )


# ── parity ───────────────────────────────────────────────────────────────────

_STFT_GRID = [
    c
    for c in itertools.product(
        [8, 15, 16],  # n_fft
        [None, 3],  # hop_length
        [None, 7],  # win_length
        [
            (True, "reflect"),
            (True, "constant"),
            (True, "replicate"),
            (True, "circular"),
            (False, None),
            (False, "align"),
        ],
        [False, True],  # normalized
        [None, False],  # onesided
        [True, False],  # a window, or the rectangular default
        [False, True],  # batched
    )
    # Thin the product: every centring mode against every other axis would be
    # 1,152 cases; one normalised / batched combination per mode is enough
    # to cover the mode, and the full product runs on the reflect default.
    if c[3] == (True, "reflect") or (c[4] == c[7] and c[5] is None)
]


def _stft_id(c: tuple[Any, ...]) -> str:
    n_fft, hop, win, (center, mode), norm, onesided, windowed, batched = c
    return (
        f"n{n_fft}-h{hop}-w{win}-{'c-' + mode if center else 'nc-' + str(mode)}"
        f"-z{norm:d}-o{onesided}-{'win' if windowed else 'rect'}-{'b' if batched else '1d'}"
    )


@pytest.mark.parity
@pytest.mark.filterwarnings("ignore:A window was not provided")
@pytest.mark.parametrize("case", _STFT_GRID, ids=[_stft_id(c) for c in _STFT_GRID])
def test_stft_matches_the_reference(case: tuple[Any, ...], device: str, ref: Any) -> None:
    n_fft, hop, win, (center, mode), normalized, onesided, windowed, batched = case
    x = _signal((3, 50) if batched else (50,))
    kwargs: dict[str, Any] = dict(
        hop_length=hop,
        win_length=win,
        center=center,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    if center:
        kwargs["pad_mode"] = mode
    elif mode == "align":
        kwargs["align_to_window"] = True
    length = win or n_fft
    theirs = ref.hann_window(length) if windowed else None
    ours = lucid.hann_window(length, device=device) if windowed else None
    expected = ref.stft(ref.tensor(x), n_fft, window=theirs, **kwargs)
    got = lucid.stft(lucid.tensor(x, device=device), n_fft, window=ours, **kwargs)
    assert got.dtype == lucid.complex64
    assert_close(got, expected.numpy(), atol=1e-4)


_ISTFT_GRID = list(
    itertools.product(
        [8, 16],  # n_fft
        [2, 4],  # hop_length
        [None, 6],  # win_length
        [True, False],  # center
        [False, True],  # normalized
        [None, 64, 30, 80],  # length
        [False, True],  # batched
    )
)


@pytest.mark.parity
@pytest.mark.parametrize(
    "case",
    _ISTFT_GRID,
    ids=[f"n{c[0]}-h{c[1]}-w{c[2]}-c{c[3]:d}-z{c[4]:d}-l{c[5]}-b{c[6]:d}" for c in _ISTFT_GRID],
)
def test_istft_matches_the_reference(case: tuple[Any, ...], device: str, ref: Any) -> None:
    n_fft, hop, win, center, normalized, length, batched = case
    x = _signal((2, 64) if batched else (64,))
    # An uncentred Hann window is zero at t = 0, which fails NOLA.
    w = ref.hann_window(win or n_fft).numpy() + (0.0 if center else 0.1)
    kwargs: dict[str, Any] = dict(
        hop_length=hop, win_length=win, center=center, normalized=normalized
    )
    spec = ref.stft(ref.tensor(x), n_fft, window=ref.tensor(w), return_complex=True, **kwargs)

    def ours_() -> lucid.Tensor:
        return lucid.istft(
            lucid.tensor(spec.numpy(), device=device),
            n_fft,
            window=lucid.tensor(w, device=device),
            length=length,
            **kwargs,
        )

    # The reference warns about a zero-padded tail once per process, so the
    # warning is checked against the frames' reach rather than against it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            expected = ref.istft(spec, n_fft, window=ref.tensor(w), length=length, **kwargs)
        except RuntimeError:
            # A window shorter than n_fft leaves the uncentred ends with no
            # overlap at all (NOLA), which both refuse.
            expected = None
    if expected is None:
        with pytest.raises(_C_engine.InvalidArgument, match="NOLA"):
            ours_()
        return
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = ours_()
    assert_close(got, expected.numpy(), atol=1e-4)
    start = n_fft // 2 if center else 0
    reach = n_fft + hop * (int(spec.shape[-1]) - 1) - start
    padded = any("padded with zeros" in str(m.message) for m in seen)
    assert padded == (length is not None and length > reach)


@pytest.mark.parity
@pytest.mark.parametrize("name", _WINDOWS)
@pytest.mark.parametrize("n", [0, 1, 2, 7, 32])
@pytest.mark.parametrize("periodic", [True, False])
def test_a_window_matches_the_reference(
    name: str, n: int, periodic: bool, device: str, ref: Any
) -> None:
    expected = getattr(ref, f"{name}_window")(n, periodic)
    assert_close(_factory(name)(n, periodic, device=device), expected, atol=2e-6)


@pytest.mark.parity
def test_the_window_parameters_match_the_reference(ref: Any) -> None:
    assert_close(
        lucid.hamming_window(9, False, 0.6, 0.3), ref.hamming_window(9, False, 0.6, 0.3), atol=1e-6
    )
    assert_close(lucid.kaiser_window(9, True, 5.0), ref.kaiser_window(9, True, 5.0), atol=1e-6)
    expected64 = ref.hann_window(9, dtype=ref.float64)
    assert_close(lucid.hann_window(9, dtype=lucid.float64), expected64, atol=1e-12)


@pytest.mark.parity
def test_a_complex_signal_or_window_matches_the_reference(device: str, ref: Any) -> None:
    re, im = _signal((40,), 1), _signal((40,), 2)
    xc = re + 1j * im
    w = np.hanning(8).astype(np.float32)
    expected = ref.stft(ref.tensor(xc.astype(np.complex64)), 8, window=ref.tensor(w))
    x = lucid.complex(lucid.tensor(re, device=device), lucid.tensor(im, device=device))
    got = lucid.stft(x, 8, window=lucid.tensor(w, device=device))
    assert_close(got, expected.numpy(), atol=1e-4)

    cw = ref.tensor(w) * (1 + 1j)
    expected = ref.stft(ref.tensor(re), 8, window=cw)
    ours_w = lucid.complex(lucid.tensor(w, device=device), lucid.tensor(w, device=device))
    got = lucid.stft(lucid.tensor(re, device=device), 8, window=ours_w)
    assert_close(got, expected.numpy(), atol=1e-4)

    spec = ref.stft(ref.tensor(xc.astype(np.complex64)), 8, window=ref.tensor(w))
    ours = lucid.tensor(spec.numpy(), device=device)
    for return_complex in (True, False):
        expected = ref.istft(spec, 8, window=ref.tensor(w), return_complex=return_complex)
        got = lucid.istft(
            ours, 8, window=lucid.tensor(w, device=device), return_complex=return_complex
        )
        assert_close(got, expected.numpy(), atol=1e-4)


@pytest.mark.parity
@pytest.mark.parametrize("inverse", [False, True], ids=["stft", "istft"])
def test_gradients_match_the_reference(inverse: bool, device: str, ref: Any) -> None:
    x = _signal((2, 40))
    w = np.hanning(8).astype(np.float32) + 0.05
    tx, tw = ref.tensor(x, requires_grad=True), ref.tensor(w, requires_grad=True)
    lx = lucid.tensor(x, device=device, requires_grad=True)
    lw = lucid.tensor(w, device=device, requires_grad=True)
    ts = ref.stft(tx, 8, 2, window=tw, return_complex=True)
    ls = lucid.stft(lx, 8, 2, window=lw, return_complex=True)
    if inverse:
        # A scaled spectrum, so the round trip is not the identity and the
        # window's gradient is not zero.
        tout = ref.istft(ts * 1.5 + 0.25, 8, 2, window=tw, length=40)
        lout = lucid.istft(ls * 1.5 + 0.25, 8, 2, window=lw, length=40)
    else:
        tout, lout = ts.abs(), lucid.abs(ls)
    g = _signal(tuple(tout.shape), 3)
    (tout * ref.tensor(g)).sum().backward()
    (lout * lucid.tensor(g, device=device)).sum().backward()
    assert_close(lx.grad, tx.grad, atol=1e-4, msg="signal gradient")
    assert_close(lw.grad, tw.grad, atol=1e-4, msg="window gradient")
