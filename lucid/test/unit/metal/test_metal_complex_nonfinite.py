"""isinf / isfinite / nan_to_num of complex64 on Metal look at each part.

MLX's ``isinf`` and ``isfinite`` judged a complex number as a whole:
``isinf`` was true only for (inf + 0j), so (1 + inf j) and (-inf + 1j)
read as finite, and ``nan_to_num`` — built on the same test and on a
comparison of the complex value with zero — left every one of them in
place.  The reference takes a complex number as NaN or infinite when either
part is, and replaces each part on its own.  (``isnan`` was already right;
it is pinned here alongside.)  The CPU side is covered by
``unit/ops/test_nonfinite_probes_dtype.py``.
"""

import math

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_NAN, _INF = float("nan"), math.inf
_TOP = 3.4028234663852886e38  # float32's largest finite value

_VALUES = [
    complex(_NAN, 1.0),
    complex(1.0, _INF),
    complex(-_INF, _NAN),
    complex(1.0, -2.0),
    complex(0.0, -_INF),
    complex(_NAN, _NAN),
    complex(_INF, 0.0),
    complex(-_INF, 1.0),
    complex(-0.0, 5e-40),
]

_DEFINITION = {
    "isnan": lambda z: math.isnan(z.real) or math.isnan(z.imag),
    "isinf": lambda z: math.isinf(z.real) or math.isinf(z.imag),
    "isfinite": lambda z: math.isfinite(z.real) and math.isfinite(z.imag),
}


@pytest.fixture(autouse=True)
def _require_metal() -> None:
    if not metal_available():
        pytest.skip("Metal not available on this host")


def _metal(values: list[complex]) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.complex64).to("metal")


def _same(got: list[complex], want: list[complex]) -> bool:
    """Equal part by part, NaN where NaN."""

    def eq(a: float, b: float) -> bool:
        return (math.isnan(a) and math.isnan(b)) or a == b

    return len(got) == len(want) and all(
        eq(g.real, w.real) and eq(g.imag, w.imag) for g, w in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("op", sorted(_DEFINITION))
def test_probe_takes_either_part(op: str) -> None:
    got = getattr(lucid, op)(_metal(_VALUES))
    assert got.dtype == lucid.bool
    assert got.to("cpu").tolist() == [_DEFINITION[op](z) for z in _VALUES]


@pytest.mark.parametrize("op", sorted(_DEFINITION))
def test_probe_agrees_with_cpu(op: str) -> None:
    cpu = getattr(lucid, op)(lucid.tensor(_VALUES, dtype=lucid.complex64))
    metal = getattr(lucid, op)(_metal(_VALUES))
    assert metal.to("cpu").tolist() == cpu.tolist()


def test_probe_keeps_shape() -> None:
    x = _metal(_VALUES[:8]).reshape(2, 4)
    assert lucid.isinf(x).shape == (2, 4)
    assert lucid.isfinite(x[0, 1]).shape == ()


def test_nan_to_num_replaces_each_part() -> None:
    got = lucid.nan_to_num(_metal(_VALUES))
    assert got.dtype == lucid.complex64

    def part(v: float) -> float:
        return 0.0 if math.isnan(v) else _TOP if v == _INF else -_TOP if v == -_INF else v

    # Each value as complex64 holds it (5e-40 is a float32 subnormal).
    held = lucid.tensor(_VALUES, dtype=lucid.complex64).tolist()
    want = [complex(part(z.real), part(z.imag)) for z in held]
    assert _same(got.to("cpu").tolist(), want)


def test_nan_to_num_explicit_values() -> None:
    got = lucid.nan_to_num(_metal(_VALUES[:3]), nan=0.5, posinf=7.0, neginf=-7.0)
    assert _same(got.to("cpu").tolist(), [complex(0.5, 1.0), complex(1.0, 7.0), complex(-7.0, 0.5)])


def test_nan_to_num_keeps_shape_and_finite_values() -> None:
    x = _metal([complex(1.5, -2.5), complex(0.0, 3.0), complex(_INF, 1.0), complex(2.0, 2.0)])
    got = lucid.nan_to_num(x.reshape(2, 2))
    assert got.shape == (2, 2)
    assert _same(
        got.to("cpu").flatten().tolist(),
        [complex(1.5, -2.5), complex(0.0, 3.0), complex(_TOP, 1.0), complex(2.0, 2.0)],
    )


@pytest.mark.parametrize("op", [*sorted(_DEFINITION), "nan_to_num"])
def test_matches_reference(ref: object, op: str) -> None:
    got = getattr(lucid, op)(_metal(_VALUES)).to("cpu").tolist()
    want = getattr(ref, op)(ref.tensor(_VALUES, dtype=ref.complex64)).tolist()  # type: ignore[attr-defined]
    if op == "nan_to_num":
        assert _same(got, want)
    else:
        assert got == want
