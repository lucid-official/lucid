"""Principal complex inverse hyperbolic values and conjugate gradients on Metal."""

import cmath
import math
import struct
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="requires Metal")
_FUNCTIONS = [(lucid.asinh, cmath.asinh), (lucid.acosh, cmath.acosh), (lucid.atanh, cmath.atanh)]


def _float(x: float) -> float:
    return struct.unpack(">f", struct.pack(">f", x))[0]


def _component(got: float, expected: float) -> None:
    want = _float(expected)
    if math.isnan(want):
        assert math.isnan(got)
        return
    if want == 0.0:
        assert got == 0.0 and math.copysign(1.0, got) == math.copysign(1.0, want), (got, want)
        return
    assert math.copysign(1.0, got) == math.copysign(1.0, want)
    a, b = (int.from_bytes(struct.pack(">f", v), "big") for v in (got, want))
    assert abs(a - b) <= 5, (got, want, abs(a - b))


@pytest.mark.parametrize(("op", "reference"), _FUNCTIONS)
def test_principal_values(
    op: Callable[[lucid.Tensor], lucid.Tensor], reference: Callable[[complex], complex]
) -> None:
    # Include the adjacent representable float on both sides of either branch point.
    lattice = [0.0, 1e-20, 1e-8, 0.3, 0.5, 1.0 - 2**-24, 1.0, 1.0 + 2**-23, 2.0, 3.0, 1e5, 1e20]
    points = [
        complex(_float(sx * x), _float(sy * y))
        for x in lattice
        for y in lattice
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        if not (reference is cmath.atanh and x == 1.0 and y == 0.0)
    ]
    special = [0.0, -0.0, 2.0, -2.0, math.inf, -math.inf, math.nan]
    points.extend(
        complex(x, y)
        for x in special
        for y in special
        if not (math.isfinite(x) and math.isfinite(y))
    )
    out = op(lucid.tensor(points, dtype=lucid.complex64, device="metal")).tolist()
    for z, value in zip(points, out, strict=True):
        expected = reference(z)
        try:
            _component(value.real, expected.real)
            _component(value.imag, expected.imag)
        except AssertionError as exc:
            raise AssertionError((reference.__name__, z, value, expected)) from exc


@pytest.mark.parametrize(("op", "reference"), _FUNCTIONS)
def test_conjugate_derivative(
    op: Callable[[lucid.Tensor], lucid.Tensor], reference: Callable[[complex], complex]
) -> None:
    points = [complex(x, y) for x in (-1e5, -2.0, -0.5, 0.5, 2.0, 1e5) for y in (-0.25, 0.25)]
    points.extend(complex(x, y) for x in (-2.0, -0.5, 0.5, 2.0) for y in (-0.0, 0.0))
    points.extend(complex(x, y) for x in (-0.0, 0.0) for y in (-2.0, -0.5, 0.5, 2.0))
    points.extend([complex(1e-20, -1e-20), complex(-1e-8, 1e-8)])
    z = lucid.tensor(points, dtype=lucid.complex64, device="metal", requires_grad=True)
    seed = complex(0.625, -0.875)
    cotangent = lucid.full_like(z, seed)
    (gradient,) = lucid.autograd.grad(op(z), [z], grad_outputs=cotangent)
    assert gradient is not None
    for item, got in zip(points, gradient.tolist(), strict=True):
        if reference is cmath.asinh:
            # Construct the rotated components directly to retain the side of a cut at -0.
            p = cmath.sqrt(complex(1.0 - item.imag, item.real))
            m = cmath.sqrt(complex(1.0 + item.imag, -item.real))
            slope = 1.0 / (p * m)
        elif reference is cmath.acosh:
            slope = 1.0 / (cmath.sqrt(item - 1.0) * cmath.sqrt(item + 1.0))
        else:
            slope = 1.0 / ((1.0 - item) * (1.0 + item))
        expected = seed * slope.conjugate()
        assert abs(got - expected) <= 8e-7 * abs(expected), (item, got, expected)


@pytest.mark.parametrize(("op", "reference"), _FUNCTIONS)
def test_compiled_principal_values(
    op: Callable[[lucid.Tensor], lucid.Tensor], reference: Callable[[complex], complex]
) -> None:
    points = [complex(-0.0, -0.0), -0.5 + 0.3j, -1e5 + 0j, 1e-8 + 1e-8j, 1e20 + 1e-20j]
    x = lucid.tensor(points, dtype=lucid.complex64, device="metal")
    values = lucid.compile(lambda z: op(z))(x).tolist()
    for z, value in zip(x.tolist(), values, strict=True):
        expect = reference(z)
        _component(value.real, expect.real)
        _component(value.imag, expect.imag)


def test_atanh_poles_keep_the_reference_zero_sign(ref: ModuleType) -> None:
    points = [complex(x, y) for x in (-1.0, 1.0) for y in (-0.0, 0.0)]
    x = lucid.tensor(points, dtype=lucid.complex64, device="metal")
    expected = ref.atanh(ref.tensor(points, dtype=ref.complex64)).tolist()
    for got, want in zip(lucid.atanh(x).tolist(), expected, strict=True):
        _component(got.real, want.real)
        _component(got.imag, want.imag)


@pytest.mark.parametrize(("op", "reference"), _FUNCTIONS)
def test_second_derivative_where_a_square_root_operand_is_imaginary(
    op: Callable[[lucid.Tensor], lucid.Tensor], reference: Callable[[complex], complex]
) -> None:
    # At Re z=1, z-1 is imaginary. At Im z=1, so is 1+i*z.
    points = [1 + 2j, 2 + 1j, -1 + 2j, 2 - 1j]
    z = lucid.tensor(points, device="metal", dtype=lucid.complex64, requires_grad=True)
    (first,) = lucid.autograd.grad(op(z).real.sum(), z, create_graph=True)
    assert first is not None
    (second,) = lucid.autograd.grad(first.real.sum(), z)
    assert second is not None
    for x, value in zip(points, second.tolist(), strict=True):
        if reference is cmath.asinh:
            slope = 1 / (cmath.sqrt(1 + 1j * x) * cmath.sqrt(1 - 1j * x))
            expected = -x * slope**3
        elif reference is cmath.acosh:
            slope = 1 / (cmath.sqrt(x + 1) * cmath.sqrt(x - 1))
            expected = -x * slope**3
        else:
            expected = 2 * x / (1 - x * x) ** 2
        assert abs(value - expected.conjugate()) <= 1e-6 * abs(expected)
