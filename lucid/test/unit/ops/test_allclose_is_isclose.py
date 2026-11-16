"""``allclose`` is ``isclose`` reduced with ``all`` (LCD-1024).

``allclose`` was its own predicate, written from arithmetic -- ``|a - b| <= atol + rtol * |b|`` --
and the arithmetic has corners the element-wise ``isclose`` does not.  A bool operand has no
subtraction, so ``allclose(mask, mask)`` raised ``LucidError`` where the reference answers.  An
infinity does not subtract from itself either: ``inf - inf`` is NaN, so equal infinities were
"not close", and ``inf`` against ``-inf`` came out close.  There is one predicate now, and the
answer is its ``all``.
"""

from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid

_INF = float("inf")
_NAN = float("nan")


@dataclass(frozen=True)
class _Case:
    """Two vectors, the options, and the reference's answer."""

    a: list[float]
    b: list[float]
    want: bool
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False


_CASES: dict[str, _Case] = {
    "equal": _Case([1.0, 2.0], [1.0, 2.0], True),
    "within-rtol": _Case([1.0, 2.0], [1.0, 2.0000001], True),
    "apart": _Case([1.0, 2.0], [1.0, 2.1], False),
    "loose-rtol": _Case([1.0, 2.0], [1.0, 2.1], True, rtol=0.1),
    "loose-atol": _Case([1.0, 2.0], [1.0, 2.1], True, rtol=0.0, atol=0.5),
    "equal-infinities": _Case([_INF, 1.0], [_INF, 1.0], True),
    "equal-negative-infinities": _Case([-_INF], [-_INF], True),
    "opposite-infinities": _Case([_INF, 1.0], [-_INF, 1.0], False),
    "nan": _Case([_NAN, 1.0], [_NAN, 1.0], False),
    "equal-nan": _Case([_NAN, 1.0], [_NAN, 1.0], True, equal_nan=True),
    "empty": _Case([], [], True),
}


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_allclose_answers_what_isclose_does(device: str, name: str, dtype: str) -> None:
    if device != "cpu" and dtype == "float64":
        pytest.skip("Metal has no float64")
    case = _CASES[name]
    x = lucid.tensor(case.a, dtype=getattr(lucid, dtype), device=device)
    y = lucid.tensor(case.b, dtype=getattr(lucid, dtype), device=device)
    options = (case.rtol, case.atol, case.equal_nan)
    assert lucid.allclose(x, y, *options) is case.want
    assert x.allclose(y, *options) is case.want
    assert bool(lucid.isclose(x, y, *options).all().item()) is case.want


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_CASES))
def test_the_reference_answers_the_same(ref: ModuleType, name: str) -> None:
    case = _CASES[name]
    got = ref.allclose(ref.tensor(case.a), ref.tensor(case.b), case.rtol, case.atol, case.equal_nan)
    assert got is case.want


def test_complex_operands_compare_by_modulus(device: str) -> None:
    z = lucid.tensor([1 + 2j, 3j], device=device)
    assert lucid.allclose(z, z)
    assert not lucid.allclose(z, lucid.tensor([1 + 2j, 3j + 1], device=device))


def test_a_pair_of_different_widths_meets_at_the_wider(device: str) -> None:
    # ``isclose`` compares one dtype, so ``allclose`` promotes a mixed pair first, as the
    # subtraction it replaces did.  (The reference refuses the pair; callers have relied on this.)
    narrow = lucid.tensor([1.0, 2.0], dtype=lucid.float16, device=device)
    wide = lucid.tensor([1.0, 2.0], device=device)
    assert lucid.allclose(narrow, wide)
    assert lucid.allclose(wide, lucid.tensor([1, 2], device=device))


@pytest.mark.parametrize("other", ["float32", "int64"])
def test_a_bool_beside_another_dtype_is_refused(device: str, other: str) -> None:
    # A dtype mismatch, as the reference's: a ``RuntimeError`` that is not a ``NotImplementedError``.
    mask = lucid.tensor([True, False], device=device)
    number = lucid.tensor([1, 0], dtype=getattr(lucid, other), device=device)
    for first, second in ((mask, number), (number, mask)):
        with pytest.raises(RuntimeError) as caught:
            lucid.allclose(first, second)
        assert not isinstance(caught.value, NotImplementedError)


@pytest.mark.parity
@pytest.mark.parametrize("other", ["float32", "int64"])
def test_the_reference_refuses_a_bool_beside_another_dtype(ref: ModuleType, other: str) -> None:
    mask = ref.tensor([True, False])
    number = ref.tensor([1, 0], dtype=getattr(ref, other))
    with pytest.raises(RuntimeError):
        ref.allclose(mask, number)
