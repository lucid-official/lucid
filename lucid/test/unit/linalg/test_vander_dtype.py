"""``linalg.vander`` orders its powers as the reference does (LCD-431).

The default was decreasing powers — NumPy's ``vander`` — and the docstring
called that a convention; the reference's ``linalg.vander`` has no
``increasing`` argument and always puts ``x^0`` first, so code ported from
it read every column backwards.  Increasing is the default now, and
``increasing=False`` reverses the columns.

The powers are a running product of ``x``, as the reference builds them.
``pow`` against an exponent row had raised for an f64 input and then for a
Metal one, rounded ``3**4`` to 81.000008 in float32, widened an integer
input to float32, and cast a complex input to float32 without a word.
"""

from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_SQUARES = [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0]]


def test_the_default_is_increasing_powers(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    assert lucid.linalg.vander(x).tolist() == _SQUARES
    assert lucid.linalg.vander(x, increasing=False).tolist() == [row[::-1] for row in _SQUARES]


def test_n_sets_the_number_of_columns(device: str) -> None:
    x = lucid.tensor([-2.0, 0.0, 0.5], device=device)
    assert lucid.linalg.vander(x, N=4).tolist() == [
        [1.0, -2.0, 4.0, -8.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.25, 0.125],
    ]
    assert lucid.linalg.vander(x, N=1).tolist() == [[1.0], [1.0], [1.0]]
    assert lucid.linalg.vander(x, N=0).shape == (3, 0)


def test_a_batch_of_vectors_gives_a_batch_of_matrices(device: str) -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 0.0]], device=device)
    assert lucid.linalg.vander(x, N=3).tolist() == [
        [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]],
        [[1.0, 3.0, 9.0], [1.0, 0.0, 0.0]],
    ]


def test_a_0d_input_is_a_vector_of_one(device: str) -> None:
    out = lucid.linalg.vander(lucid.tensor(2.0, device=device), N=3)
    assert out.tolist() == [[1.0, 2.0, 4.0]]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_a_floating_input_keeps_its_dtype(device: str, dtype: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], dtype=getattr(lucid, dtype), device=device)
    out = lucid.linalg.vander(x)
    assert out.dtype == x.dtype
    assert out.to(lucid.float32).tolist() == _SQUARES


def test_a_float64_input_keeps_its_dtype() -> None:
    out = lucid.linalg.vander(lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.float64))
    assert out.dtype == lucid.float64
    assert out.tolist() == _SQUARES


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_an_integer_input_has_exact_int64_powers(device: str, dtype: str) -> None:
    x = lucid.tensor([3, -7], dtype=getattr(lucid, dtype), device=device)
    out = lucid.linalg.vander(x, N=12)
    assert out.dtype == lucid.int64
    assert out.tolist() == [[3**k for k in range(12)], [(-7) ** k for k in range(12)]]


def test_a_bool_input_is_refused() -> None:
    with pytest.raises(TypeError, match="vander"):
        lucid.linalg.vander(lucid.tensor([True, False]))


def test_a_negative_n_is_refused() -> None:
    with pytest.raises(ValueError, match="N must be non-negative"):
        lucid.linalg.vander(lucid.tensor([1.0, 2.0]), N=-1)


def test_the_derivative_at_zero_is_exact(device: str) -> None:
    # d/dx (1 + x + x^2 + x^3) = 1 + 2x + 3x^2: 1 at 0, 17 at 2.  ``pow``'s
    # derivative of x^0 at 0 is 0 * 0^-1.
    x = lucid.tensor([0.0, 2.0], device=device, requires_grad=True)
    lucid.linalg.vander(x, N=4).sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [1.0, 17.0]


def test_the_second_derivative_at_zero_is_exact(device: str) -> None:
    # d^2/dx^2 (1 + x + x^2 + x^3) = 2 + 6x: 2 at 0, 14 at 2 (LCD-637).
    x = lucid.tensor([0.0, 2.0], device=device, requires_grad=True)
    (g1,) = lucid.autograd.grad(lucid.linalg.vander(x, N=4).sum(), [x], create_graph=True)
    (g2,) = lucid.autograd.grad(g1.sum(), [x])
    assert g1.tolist() == [1.0, 17.0]
    assert g2.tolist() == [2.0, 14.0]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "metal",
            marks=pytest.mark.skipif(not metal_available(), reason="needs Metal"),
        ),
    ],
)
def test_a_complex_input_keeps_its_imaginary_part(device: str) -> None:
    x = lucid.complex(
        lucid.tensor([1.0, 2.0], device=device), lucid.tensor([1.0, 0.0], device=device)
    )
    out = lucid.linalg.vander(x)
    assert out.dtype == lucid.complex64
    assert out.tolist() == [[1 + 0j, 1 + 1j], [1 + 0j, 2 + 0j]]


@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128])
def test_cpu_complex_cumsum_and_cumprod_preserve_dtype(dtype: lucid.dtype) -> None:
    x = lucid.tensor([1 + 2j, 3 - 1j, 0 + 1j], dtype=dtype)
    assert lucid.cumsum(x, 0).dtype == dtype
    assert lucid.cumsum(x, 0).tolist() == [1 + 2j, 4 + 1j, 4 + 2j]
    assert lucid.cumprod(x, 0).dtype == dtype
    assert lucid.cumprod(x, 0).tolist() == [1 + 2j, 5 + 5j, -5 + 5j]
    assert lucid.linalg.vander(x, N=3).dtype == dtype


@pytest.mark.parity
@pytest.mark.parametrize(
    "data, n_cols",
    [
        ([1.0, 2.0, 3.0], None),
        ([0.5, -1.5, 2.0, 0.0], 6),
        ([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], 4),
        ([float("nan"), float("inf")], 3),
    ],
)
def test_the_reference_gives_the_same_matrix(
    ref: ModuleType, data: list[float], n_cols: int | None
) -> None:
    got = lucid.linalg.vander(lucid.tensor(data), N=n_cols)
    want = ref.linalg.vander(ref.tensor(data), N=n_cols)
    assert got.shape == tuple(want.shape)
    assert str(got.tolist()) == str(want.tolist())


@pytest.mark.parity
def test_the_reference_gives_the_same_integer_matrix(ref: ModuleType) -> None:
    got = lucid.linalg.vander(lucid.tensor([1, 2, 3]))
    want = ref.linalg.vander(ref.tensor([1, 2, 3]))
    assert got.tolist() == want.tolist()
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
