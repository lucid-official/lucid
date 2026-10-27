"""Complex arithmetic works on the CPU as it does on Metal.

The CPU stream refused ``+``, ``-`` and ``/`` for complex input ("dtype not
supported"), could not broadcast a complex scalar, and could not cast a real
tensor to complex — so ``a / 2`` and ``a * real`` failed too, while Metal ran
all of them.  Only ``*`` worked.
"""

from collections.abc import Callable

import pytest

import lucid

_DEVICES = [
    "cpu",
    pytest.param(
        "metal", marks=pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
    ),
]
_A = [1 + 2j, 3 - 1j]
_B = [2 - 1j, 0.5 + 0.5j]
_R = [2.0, 4.0]

_CASES: dict[str, Callable[[object, object, object], object]] = {
    "a+b": lambda a, b, r: a + b,
    "a-b": lambda a, b, r: a - b,
    "a*b": lambda a, b, r: a * b,
    "a/b": lambda a, b, r: a / b,
    "a/2": lambda a, b, r: a / 2,
    "2/a": lambda a, b, r: 2 / a,
    "a+r": lambda a, b, r: a + r,
    "a*r": lambda a, b, r: a * r,
    "a/r": lambda a, b, r: a / r,
    "r/a": lambda a, b, r: r / a,
}


def _expected(name: str) -> list[complex]:
    return [complex(_CASES[name](a, b, r)) for a, b, r in zip(_A, _B, _R)]  # type: ignore[arg-type]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", list(_CASES))
def test_complex_arithmetic_matches_python(device: str, name: str) -> None:
    a = lucid.tensor(_A, device=device)
    b = lucid.tensor(_B, device=device)
    r = lucid.tensor(_R, device=device)
    got = _CASES[name](a, b, r).tolist()  # type: ignore[attr-defined]
    for g, w in zip(got, _expected(name)):
        assert complex(g) == pytest.approx(w, rel=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_real_and_complex_casts(device: str) -> None:
    r = lucid.tensor(_R, device=device)
    assert r.to(lucid.complex64).tolist() == [2 + 0j, 4 + 0j]
    a = lucid.tensor(_A, device=device)
    assert a.to(lucid.float32).tolist() == [1.0, 3.0]


def test_bfloat16_broadcasts_on_the_cpu() -> None:
    x = lucid.ones(2, 3).to(lucid.bfloat16)
    y = lucid.tensor([1.0, 2.0, 3.0]).to(lucid.bfloat16)
    assert (x + y).to(lucid.float32).tolist() == [[2.0, 3.0, 4.0]] * 2
