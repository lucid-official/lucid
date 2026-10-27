"""Two-argument special functions take what the reference's overloads take (LCD-431).

``special.zeta(x, 2.0)`` raised ``TypeError``: the reference has a
``Scalar`` overload for either operand.  The orthogonal polynomials took
their degree ``n`` as a Python int alone, where the reference takes a
tensor of degrees — one per element, broadcast against ``x`` — a Python
number for ``x``, and a float degree, truncated toward zero; a negative
degree raised ``ValueError`` where the reference answers the zero
polynomial.  Two Python numbers are refused, as the reference refuses them.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

_POLYNOMIALS = sorted(n for n in lucid.special.__all__ if "polynomial" in n)

_ZETA_2 = math.pi**2 / 6
_APERY = 1.2020569031595942


def _close(got: lucid.Tensor, want: list[float], tol: float = 1e-5) -> bool:
    values = got.flatten().tolist()
    return len(values) == len(want) and all(
        abs(g - w) <= tol * max(1.0, abs(w)) for g, w in zip(values, want, strict=True)
    )


# ── zeta ─────────────────────────────────────────────────────────────────────


def test_zeta_takes_a_python_number_for_other(device: str) -> None:
    x = lucid.tensor([2.0, 3.0], device=device)
    for q in (1.0, 1):
        got = lucid.special.zeta(x, q)
        assert got.device == x.device and got.dtype == lucid.float32
        assert _close(got, [_ZETA_2, _APERY])
    assert _close(lucid.special.zeta(input=x, other=1.0), [_ZETA_2, _APERY])


def test_zeta_takes_a_python_number_for_input(device: str) -> None:
    q = lucid.tensor([1.0, 2.0], device=device)
    got = lucid.special.zeta(2.0, q)
    assert got.device == q.device
    assert _close(got, [_ZETA_2, _ZETA_2 - 1.0])


def test_zeta_answers_in_the_tensor_operands_dtype() -> None:
    x = lucid.tensor([2.0, 3.0], dtype=lucid.float64)
    assert lucid.special.zeta(x, 1.0).dtype == lucid.float64
    # An integer tensor is evaluated at the default float.
    assert lucid.special.zeta(lucid.tensor([2, 3]), 1.0).dtype == lucid.float32


def test_zeta_is_differentiable_through_the_tensor_operand() -> None:
    q = lucid.tensor([1.0, 2.0], requires_grad=True)
    lucid.special.zeta(2.0, q).sum().backward()
    # d/dq zeta(x, q) = -x zeta(x + 1, q).
    assert q.grad is not None
    assert _close(q.grad, [-2.0 * _APERY, -2.0 * (_APERY - 1.0)], tol=1e-4)


@pytest.mark.parametrize(
    "call",
    [lambda: lucid.special.zeta(2.0, 1.0), lambda: lucid.special.zeta(lucid.tensor([2.0]), 1j)],
    ids=["two-numbers", "complex-number"],
)
def test_zeta_refuses_what_the_reference_refuses(call: Callable[[], object]) -> None:
    with pytest.raises(TypeError):
        call()


@pytest.mark.parity
@pytest.mark.parametrize(
    "make",
    [
        lambda m: (m.tensor([2.0, 3.0, 4.5]), 2.0),
        lambda m: (2.0, m.tensor([1.0, 2.0, 0.5])),
        lambda m: (m.tensor([2, 3]), 2),
        lambda m: (m.tensor([2.0, 3.0], dtype=m.float64), 0.5),
    ],
    ids=["tensor-number", "number-tensor", "int-tensor", "float64"],
)
def test_zeta_matches_the_reference(
    ref: ModuleType, make: Callable[[ModuleType], tuple[object, object]]
) -> None:
    got = lucid.special.zeta(*make(lucid))
    want = ref.special.zeta(*make(ref))
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert _close(got, want.tolist())


# ── orthogonal polynomials ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", _POLYNOMIALS)
def test_a_tensor_of_degrees_gives_each_element_its_own(name: str, device: str) -> None:
    fn = getattr(lucid.special, name)
    x = lucid.tensor([-0.5, 0.25, 0.8], device=device)
    degrees = [0, 3, 6]
    got = fn(x, lucid.tensor(degrees, device=device))
    assert got.device == x.device and got.shape == (3,)
    want = [fn(x, k).tolist()[i] for i, k in enumerate(degrees)]
    assert _close(got, want)


@pytest.mark.parametrize("name", _POLYNOMIALS)
def test_the_degrees_broadcast_against_x(name: str) -> None:
    fn = getattr(lucid.special, name)
    x = lucid.tensor([0.1, 0.9])
    got = fn(x, lucid.tensor([[1], [4]]))
    assert got.shape == (2, 2)
    assert _close(got, fn(x, 1).tolist() + fn(x, 4).tolist())


@pytest.mark.parametrize("name", _POLYNOMIALS)
def test_x_may_be_a_python_number(name: str) -> None:
    fn = getattr(lucid.special, name)
    got = fn(0.3, lucid.tensor([0, 2, 5]))
    want = [fn(lucid.tensor([0.3]), k).item() for k in (0, 2, 5)]
    assert got.dtype == lucid.float32
    assert _close(got, want)


@pytest.mark.parametrize("name", _POLYNOMIALS)
def test_a_degree_truncates_toward_zero_and_a_negative_one_is_zero(name: str) -> None:
    fn = getattr(lucid.special, name)
    x = lucid.tensor([0.5, 0.2])
    assert fn(x, 2.9).tolist() == fn(x, 2).tolist()
    assert fn(x, -0.5).tolist() == [1.0, 1.0]
    assert fn(x, -1).tolist() == [0.0, 0.0]
    assert fn(x, lucid.tensor([-3, 0])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("name", _POLYNOMIALS)
def test_two_python_numbers_are_refused(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        getattr(lucid.special, name)(0.5, 2)


@pytest.mark.parametrize("name", _POLYNOMIALS)
@pytest.mark.parametrize("n", ["2", None, 1j], ids=["str", "none", "complex"])
def test_a_degree_that_is_not_a_real_number_is_refused(name: str, n: object) -> None:
    # ``int(n)`` read ``"2"`` as degree 2.
    with pytest.raises(TypeError, match=name):
        getattr(lucid.special, name)(lucid.tensor([0.5]), n)


@pytest.mark.parametrize("name", _POLYNOMIALS)
@pytest.mark.parametrize(
    "n",
    [float("nan"), float("inf"), lucid.tensor([1.0, float("nan")])],
    ids=["nan", "inf", "nan-in-tensor"],
)
def test_a_degree_that_is_not_finite_is_refused(name: str, n: object) -> None:
    with pytest.raises(ValueError, match=f"{name}: expected a finite degree"):
        getattr(lucid.special, name)(lucid.tensor([0.5, 0.2]), n)


def test_the_per_element_path_is_differentiable_in_x() -> None:
    x = lucid.tensor([0.3, -0.4, 0.7], requires_grad=True)
    lucid.special.legendre_polynomial_p(x, lucid.tensor([0, 2, 5])).sum().backward()
    # P_0' = 0, P_2' = 3x, P_5' = (315 x^4 - 210 x^2 + 15) / 8.
    assert x.grad is not None
    assert _close(x.grad, [0.0, 3 * -0.4, (315 * 0.7**4 - 210 * 0.7**2 + 15) / 8])


@pytest.mark.parity
@pytest.mark.parametrize("name", _POLYNOMIALS)
@pytest.mark.parametrize(
    "make",
    [
        lambda m: (m.tensor([-0.5, 0.25, 0.8]), m.tensor([0, 2, 5])),
        lambda m: (m.tensor([-0.5, 0.25, 0.8]), m.tensor([0.0, 2.9, 6.0])),
        lambda m: (0.5, m.tensor([0, 2, 5])),
        lambda m: (m.tensor([-0.5, 0.25, 0.8]), 3.0),
        lambda m: (m.tensor([-0.5, 0.25, 0.8]), -1),
        lambda m: (m.tensor([1, 2]), m.tensor([2, 3])),
        lambda m: (m.tensor([0.5]), m.tensor([[1], [2], [3]])),
    ],
    ids=["tensors", "float-degrees", "number-x", "float-n", "negative-n", "ints", "bcast"],
)
def test_the_polynomials_match_the_reference(
    ref: ModuleType, name: str, make: Callable[[ModuleType], tuple[object, object]]
) -> None:
    got = getattr(lucid.special, name)(*make(lucid))
    want = getattr(ref.special, name)(*make(ref))
    assert got.shape == tuple(want.shape)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert _close(got, want.flatten().tolist(), tol=1e-4)
