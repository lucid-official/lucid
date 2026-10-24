"""Binary composites broadcast, and the integer-only ones take integers (LCD-634).

``lucid.gcd`` / ``lucid.lcm`` walked the two operands element by element
at ``input``'s size: ``gcd([12, 18], [8])`` read past ``other`` and raised
``IndexError``, and ``gcd([8], [12, 18])`` answered one element.  They
also truncated a float operand without a word, and failed with an
``AttributeError`` from deep inside the loop on a Python number.  Now
both operands are refused unless they are integer tensors, as the
reference refuses them, then brought to their common dtype and broadcast
shape before the loop.  Held for every binary composite: the broadcast
table below — a scalar, a size-1 axis, trailing and leading dims, an
empty axis — answers the reference's shape and values, on either device.
"""

from types import ModuleType

import pytest

import lucid

#: Shape pairs every binary op broadcasts.
_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((2, 3), ()),
    ((2, 3), (1,)),
    ((2, 3), (3,)),
    ((2, 3), (1, 3)),
    ((2, 1), (1, 3)),
    ((3,), (2, 3)),
    ((), (2, 3)),
    ((0, 3), (3,)),
]

_INTEGER_OPS = (
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "gcd",
    "lcm",
)
_FLOAT_OPS = (
    "arctan2",
    "copysign",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "heaviside",
    "hypot",
    "ldexp",
    "logaddexp",
    "nextafter",
    "remainder",
    "xlogy",
)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _data(shape: tuple[int, ...], integer: bool, seed: int) -> list[object]:
    """Flat operand values: small positive integers, or spread floats."""
    n = _numel(shape)
    if integer:
        return [(i * 5 + seed) % 7 + 1 for i in range(n)]
    return [0.3 + 0.7 * i + seed * 0.5 for i in range(n)]


def _operands(
    mod: ModuleType, name: str, sa: tuple[int, ...], sb: tuple[int, ...], device: str
) -> tuple[object, object]:
    integer = name in _INTEGER_OPS
    a_dtype = mod.int64 if integer else mod.float32
    b_dtype = mod.int64 if integer or name == "ldexp" else mod.float32
    a = mod.tensor(_data(sa, integer, 1), dtype=a_dtype).reshape(sa)
    b = mod.tensor(_data(sb, integer or name == "ldexp", 2), dtype=b_dtype)
    b = b.reshape(sb)
    if mod is lucid:
        return a.to(device), b.to(device)
    return a, b


@pytest.mark.parity
@pytest.mark.parametrize("shapes", _SHAPES, ids=lambda s: f"{s[0]}-{s[1]}")
@pytest.mark.parametrize("name", _INTEGER_OPS + _FLOAT_OPS)
def test_every_binary_composite_broadcasts_as_the_reference(
    ref: ModuleType, name: str, shapes: tuple[tuple[int, ...], ...], device: str
) -> None:
    sa, sb = shapes
    theirs = getattr(ref, name)(*_operands(ref, name, sa, sb, "cpu"))
    a, b = _operands(lucid, name, sa, sb, device)
    if name == "float_power" and device == "metal":
        # float_power answers float64, which Metal does not hold.
        with pytest.raises(NotImplementedError, match="float64"):
            getattr(lucid, name)(a, b)
        return
    ours = getattr(lucid, name)(a, b)
    assert tuple(ours.shape) == tuple(theirs.shape)
    expected = ref.tensor(ours.tolist(), dtype=ref.float64).reshape(theirs.shape)
    assert ref.allclose(expected, theirs.double(), atol=1e-5, equal_nan=True)


_GCD_CASES: list[tuple[str, list[int], list[int] | int, list[int]]] = [
    ("gcd", [12, 18], [8], [4, 2]),
    ("gcd", [8], [12, 18], [4, 2]),
    ("gcd", [0, -4, 9], [0, 6, -6], [0, 2, 3]),
    ("lcm", [4, 6], [6], [12, 6]),
    ("lcm", [0, -4, 3], [0, 6, -5], [0, 12, 15]),
]


@pytest.mark.parametrize("name,a,b,want", _GCD_CASES)
def test_gcd_and_lcm_broadcast(
    name: str, a: list[int], b: list[int], want: list[int], device: str
) -> None:
    x = lucid.tensor(a, device=device)
    y = lucid.tensor(b, device=device)
    out = getattr(lucid, name)(x, y)
    assert out.tolist() == want
    assert out.device == x.device
    assert getattr(x, name)(y).tolist() == want


@pytest.mark.parametrize("name", ["gcd", "lcm"])
def test_gcd_and_lcm_answer_the_promoted_dtype(name: str, device: str) -> None:
    x = lucid.tensor([12, 18], dtype=lucid.int32, device=device)
    y = lucid.tensor([8], dtype=lucid.int64, device=device)
    assert getattr(lucid, name)(x, y).dtype == lucid.int64
    narrow = lucid.tensor([8], dtype=lucid.int16, device=device)
    assert getattr(lucid, name)(x, narrow).dtype == lucid.int32


@pytest.mark.parametrize("name", ["gcd", "lcm"])
@pytest.mark.parametrize(
    "a,b,what",
    [
        (([4.0, 6.0], "float32"), ([6.0, 9.0], "float32"), "integer dtype"),
        (([4, 6], "int64"), ([6.0, 9.0], "float32"), "integer dtype"),
        (([True], "bool_"), ([True], "bool_"), "integer dtype"),
        (([3], "int64"), 2, "must be a Tensor"),
        (3, ([2], "int64"), "must be a Tensor"),
    ],
    ids=["float", "int-float", "bool", "scalar-other", "scalar-input"],
)
def test_gcd_and_lcm_refuse_what_is_not_an_integer_tensor(
    name: str, a: object, b: object, what: str
) -> None:
    def build(spec: object) -> object:
        if isinstance(spec, tuple):
            data, dtype = spec
            return lucid.tensor(data, dtype=getattr(lucid, dtype))
        return spec

    with pytest.raises(TypeError, match=f"lucid.{name}.*{what}"):
        getattr(lucid, name)(build(a), build(b))


@pytest.mark.parity
@pytest.mark.parametrize("name", ["gcd", "lcm"])
def test_the_reference_refuses_the_same_operands(ref: ModuleType, name: str) -> None:
    fn = getattr(ref, name)
    with pytest.raises((TypeError, NotImplementedError, RuntimeError)):
        fn(ref.tensor([4.0, 6.0]), ref.tensor([6.0, 9.0]))
    with pytest.raises((TypeError, NotImplementedError, RuntimeError)):
        fn(ref.tensor([True]), ref.tensor([True]))
    with pytest.raises(TypeError):
        fn(ref.tensor([3]), 2)
