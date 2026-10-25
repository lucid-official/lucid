"""``lucid.fft`` takes the argument forms the reference's parser takes (LCD-431).

The 2-D transforms coerced ``dim`` with ``list(dim)``, so the single int
the reference's ``int[1]`` reads as a one-axis list raised ``TypeError``;
``fftfreq`` and ``rfftfreq`` refused the ``requires_grad`` the reference's
factories take.
"""

from types import ModuleType

import pytest

import lucid

_DATA = [[1.0, 2.0, 0.5, -1.0], [0.0, 3.0, -2.0, 1.5], [4.0, -1.0, 2.0, 0.25]]

#: Each 2-D transform and the 1-D one it is along a single axis.
_PAIRS = [
    ("fft2", "fft", False),
    ("ifft2", "ifft", False),
    ("rfft2", "rfft", False),
    ("ihfft2", "ihfft", False),
    ("irfft2", "irfft", True),
    ("hfft2", "hfft", True),
]


def _input(device: str, spectrum: bool) -> lucid.Tensor:
    x = lucid.tensor(_DATA, device=device)
    return lucid.fft.fft(x) if spectrum else x


def _close(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return a.shape == b.shape and bool((a - b).abs().max().item() < 1e-4)


@pytest.mark.parametrize("two_d, one_d, spectrum", _PAIRS, ids=[p[0] for p in _PAIRS])
def test_a_two_dim_transform_takes_a_single_int_dim(
    device: str, two_d: str, one_d: str, spectrum: bool
) -> None:
    x = _input(device, spectrum)
    for dim in (0, -1):
        got = getattr(lucid.fft, two_d)(x, dim=dim)
        assert _close(got, getattr(lucid.fft, one_d)(x, dim=dim))


@pytest.mark.parametrize("two_d, one_d, spectrum", _PAIRS, ids=[p[0] for p in _PAIRS])
def test_a_single_int_s_goes_with_a_single_int_dim(
    device: str, two_d: str, one_d: str, spectrum: bool
) -> None:
    x = _input(device, spectrum)
    got = getattr(lucid.fft, two_d)(x, s=6, dim=1)
    assert _close(got, getattr(lucid.fft, one_d)(x, n=6, dim=1))


@pytest.mark.parametrize("fn", ["fft2", "rfft2"])
def test_one_size_for_two_axes_is_refused(fn: str) -> None:
    with pytest.raises(ValueError, match="len"):
        getattr(lucid.fft, fn)(lucid.tensor(_DATA), s=5)


@pytest.mark.parametrize("fn, n", [("fftfreq", 4), ("rfftfreq", 4)])
def test_the_frequency_factories_take_requires_grad(fn: str, n: int) -> None:
    out = getattr(lucid.fft, fn)(n, 0.5, requires_grad=True)
    assert out.requires_grad and out.is_leaf
    assert not getattr(lucid.fft, fn)(n).requires_grad


@pytest.mark.parity
def test_fftfreq_and_rfftfreq_zero_match_reference(
    ref: ModuleType, device: str
) -> None:
    f0 = lucid.fft.fftfreq(0, device=device)
    rf0 = ref.fft.fftfreq(0)
    assert f0.shape == tuple(rf0.shape) == (0,)
    r0 = lucid.fft.rfftfreq(0, device=device)
    rr0 = ref.fft.rfftfreq(0)
    assert r0.shape == tuple(rr0.shape) == (1,)
    assert bool(lucid.isnan(r0).all().item()) and bool(ref.isnan(rr0).all().item())
    for fn in ("fftfreq", "rfftfreq"):
        with pytest.raises(ValueError, match=">= 0"):
            getattr(lucid.fft, fn)(-1, device=device)


@pytest.mark.parity
@pytest.mark.parametrize(
    "fn, args, kwargs",
    [
        ("fft2", (_DATA,), {"dim": 0}),
        ("ifft2", (_DATA,), {"dim": -1}),
        ("rfft2", (_DATA,), {"dim": 0}),
        ("ihfft2", (_DATA,), {"dim": 0}),
        ("fft2", (_DATA,), {"s": 6, "dim": 1}),
        ("fftfreq", (5,), {"d": 0.5, "requires_grad": True}),
        ("rfftfreq", (6,), {"requires_grad": True}),
    ],
    ids=["fft2", "ifft2", "rfft2", "ihfft2", "fft2-s", "fftfreq", "rfftfreq"],
)
def test_the_reference_gives_the_same_answer(
    ref: ModuleType, fn: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    def operands(m: ModuleType) -> list[object]:
        return [m.tensor(a) if isinstance(a, list) else a for a in args]

    got = getattr(lucid.fft, fn)(*operands(lucid), **kwargs)
    want = getattr(ref.fft, fn)(*operands(ref), **kwargs)
    assert got.shape == tuple(want.shape)
    assert got.requires_grad == want.requires_grad
    flat = want.detach().flatten().tolist()
    for g, w in zip(got.detach().flatten().tolist(), flat, strict=True):
        assert abs(complex(g) - complex(w)) < 1e-4
