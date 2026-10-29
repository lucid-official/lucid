"""``ldexp`` is ``input * 2**other``: the same dtype, and the imaginary part kept.

It answered a complex input's real part alone, so ``ldexp([1+2j, 3-1j], [1, 2])``
was ``[2+0j, 12+0j]`` where the reference gives ``[2+4j, 12-4j]``, and it took
its result dtype from the mantissa alone: an integral mantissa over a half
exponent came back float32, a float16 one over a float32 exponent stayed
float16, a float32 one over a complex exponent stayed real (LCD-662).

The matrix below runs every pair of operand dtypes the reference accepts
against the reference itself, on every device: the answer's dtype by name and
its values to the dtype's own rounding.  The exactness of the real float
paths is ``test_ldexp_frexp_exact.py``'s.
"""

import cmath
import math
import warnings
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import device_supports

_DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
#: The reference refuses a bool exponent over a float mantissa, and any
#: complex mantissa over a float16 exponent (it has no half-precision complex pow).
_EXPONENTS = [name for name in _DTYPES if name != "bool"]

_KIND = {"bool": "b", "complex64": "c", "complex128": "c"}
_X = {
    "b": [True, False, True, True],
    "i": [1, 3, -2, 7],
    "f": [1.0, 3.0, -2.0, 7.0],
    "c": [1 + 2j, 3 - 1j, -0.5 + 0j, 7 + 0.25j],
}
_E = {"i": [1, 2, -1, 3], "f": [1.0, 2.5, -1.5, 0.0], "c": [1 + 0.5j, 2 - 1j, -1.5 + 0.25j, 0j]}
#: Relative tolerance by dtype: each is the dtype's own rounding.
_RTOL = {
    "float16": 5e-3,
    "bfloat16": 4e-2,
    "float32": 2e-5,
    "float64": 1e-12,
    "complex64": 2e-5,
    "complex128": 1e-12,
}


def _kind(name: str) -> str:
    return _KIND.get(name, "i" if name.startswith("int") else "f")


def _name(dtype: object) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _values(table: dict[str, list[object]], name: str) -> list[object]:
    return table[_kind(name)]


def _case_ids() -> list[tuple[str, str]]:
    return [(x, e) for x in _DTYPES for e in _EXPONENTS if not (_kind(x) == "c" and e == "float16")]


@pytest.mark.parametrize(("x_name", "e_name"), _case_ids())
def test_ldexp_matches_the_reference_over_every_dtype_pair(
    ref: ModuleType, device: str, x_name: str, e_name: str
) -> None:
    if not (
        device_supports(device, getattr(lucid, x_name))
        and device_supports(device, getattr(lucid, e_name))
    ):
        pytest.skip(f"{device} cannot hold {x_name} / {e_name}")
    xs, es = _values(_X, x_name), _values(_E, e_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the reference warns about half-precision complex
        want = ref.ldexp(
            ref.tensor(xs, dtype=getattr(ref, x_name)), ref.tensor(es, dtype=getattr(ref, e_name))
        )
    got = lucid.ldexp(
        lucid.tensor(xs, dtype=getattr(lucid, x_name), device=device),
        lucid.tensor(es, dtype=getattr(lucid, e_name), device=device),
    )
    assert _name(got.dtype) == _name(want.dtype)
    assert got.shape == tuple(want.shape)
    # The reference forms 2**other in the exponent's own dtype, whatever the answer's.
    rtol = max(_RTOL[_name(want.dtype)], _RTOL.get(e_name, 0.0))
    for g, w in zip(got.to("cpu").tolist(), want.tolist(), strict=True):
        assert cmath.isclose(g, w, rel_tol=rtol, abs_tol=1e-30), f"got {g!r}, want {w!r}"


def test_ldexp_keeps_the_imaginary_part(device: str) -> None:
    z = lucid.tensor([1 + 2j, 3 - 1j], dtype=lucid.complex64, device=device)
    e = lucid.tensor([1, 2], dtype=lucid.int32, device=device)
    assert lucid.ldexp(z, e).to("cpu").tolist() == [2 + 4j, 12 - 4j]
    assert z.ldexp(e).to("cpu").tolist() == [2 + 4j, 12 - 4j]
    # A Python number is as an integral exponent: the mantissa keeps its dtype.
    out = lucid.ldexp(z, 3)
    assert out.dtype == lucid.complex64
    assert out.to("cpu").tolist() == [8 + 16j, 24 - 8j]


def test_ldexp_with_a_complex_exponent_turns_the_product(device: str) -> None:
    # 2**(i * pi / ln 2) = e**(i pi) = -1, so the phase alone negates.
    turn = math.pi / math.log(2.0)
    z = lucid.tensor([1 + 2j, 3 - 1j], dtype=lucid.complex64, device=device)
    e = lucid.tensor([1j * turn, 1 + 1j * turn], dtype=lucid.complex64, device=device)
    got = lucid.ldexp(z, e).to("cpu").tolist()
    for g, w in zip(got, [-1 - 2j, -6 + 2j], strict=True):
        assert cmath.isclose(g, w, rel_tol=1e-5, abs_tol=1e-5)


def test_ldexp_complex_input_gradient_reaches_both_parts(device: str) -> None:
    re = lucid.tensor([1.0, 3.0], device=device, requires_grad=True)
    im = lucid.tensor([2.0, -1.0], device=device, requires_grad=True)
    e = lucid.tensor([1, 2], dtype=lucid.int32, device=device)
    out = lucid.ldexp(lucid.complex(re, im), e)
    (lucid.real(out) + 3.0 * lucid.imag(out)).sum().backward()
    assert re.grad is not None and im.grad is not None
    assert re.grad.to("cpu").tolist() == [2.0, 4.0]
    assert im.grad.to("cpu").tolist() == [6.0, 12.0]
