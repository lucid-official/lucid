"""CPU double-complex arithmetic cannot silently narrow either component."""

import operator

import numpy as np
import pytest

import lucid


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("broadcast", [False, True])
def test_complex128_arithmetic(operation, broadcast: bool) -> None:
    a = np.array([[1.0000000001 + 2.0000000001j, -3 + 4j]], dtype=np.complex128)
    b = np.array([[2 - 3j], [4 + 2j]] if broadcast else [[2 - 3j, 4 + 2j]])
    result = operation(lucid.tensor(a), lucid.tensor(b))
    assert result.dtype == lucid.complex128
    np.testing.assert_allclose(result.numpy(), operation(a, b), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128])
def test_float64_complex_promotion_matches_public_contract(dtype: lucid.dtype) -> None:
    a = lucid.tensor([1 + 2j], dtype=dtype)
    b = lucid.tensor([1.0000000001], dtype=lucid.float64)
    assert lucid.promote_types(dtype, b.dtype) == lucid.complex128
    assert lucid.promote_types(b.dtype, dtype) == lucid.complex128
    for output in (a + b, b + a):
        assert output.dtype == lucid.complex128
        np.testing.assert_allclose(output.numpy(), [2.0000000001 + 2j], rtol=1e-14)


def test_complex_cast_preserves_imaginary_lanes() -> None:
    values = np.array([1.25 + 2.5j, -3.5 + 4.75j], dtype=np.complex128)
    x = lucid.tensor(values)
    np.testing.assert_array_equal(x.to(lucid.complex64).to(lucid.complex128).numpy(), values)
    real = lucid.tensor([1.0000000001], dtype=lucid.float64)
    np.testing.assert_array_equal(real.to(lucid.complex128).to(lucid.float64).numpy(), real.numpy())


def test_complex128_factories_and_scalar_broadcast() -> None:
    np.testing.assert_array_equal(lucid.ones(2, dtype=lucid.complex128).numpy(), [1 + 0j] * 2)
    values = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    np.testing.assert_array_equal((lucid.tensor(values) + 2).numpy(), values + 2)


@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128])
def test_complex_truth_cast_observes_the_imaginary_lane(dtype: lucid.dtype) -> None:
    x = lucid.tensor([0j, 2j, 3 + 0j, complex(0, float("nan"))], dtype=dtype)
    np.testing.assert_array_equal(x.to(lucid.bool).numpy(), [False, True, True, True])
