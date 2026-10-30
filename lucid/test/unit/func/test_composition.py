"""``lucid.func`` transforms nest: any two compose, to any depth.

The derivative taken inside another transform's function is an
intermediate of the outer derivative, so it has to keep its autograd graph.
It used to be dropped (``create_graph=False``), which turned the inner
result into ``None`` or zeros — ``grad(grad(f))`` returned ``None``.

Every case differentiates ``s(a) = a ** 3`` twice, so the answer is
``6 a`` whatever the two transforms are: it is checked against that closed
form on every device, and — under ``-m parity`` — against the reference
framework's ``func`` running the same composition.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.func as func
from lucid import Tensor
from lucid._ctx import transform_depth

Fn = Callable[[Tensor], Tensor]
Lift = Callable[[ModuleType, Fn], Fn]

_XS = [1.0, -2.0, 0.5]


def _cube(a: Tensor) -> Tensor:
    return a**3


def _vector(m: ModuleType, values: list[float], device: str) -> Tensor:
    """``values`` as a tensor of module ``m`` — the reference lives on the CPU."""
    return lucid.tensor(values, device=device) if m is lucid else m.tensor(values)


# ── the derivative operators: a scalar function in, its derivative out ───────


def _grad(m: ModuleType, f: Fn) -> Fn:
    return m.func.grad(f)


def _jacrev(m: ModuleType, f: Fn) -> Fn:
    return m.func.jacrev(f)


def _jacfwd(m: ModuleType, f: Fn) -> Fn:
    return m.func.jacfwd(f)


def _vjp(m: ModuleType, f: Fn) -> Fn:
    def derivative(a: Tensor) -> Tensor:
        y, pullback = m.func.vjp(f, a)
        (g,) = pullback(m.ones_like(y))
        return g

    return derivative


def _jvp(m: ModuleType, f: Fn) -> Fn:
    def derivative(a: Tensor) -> Tensor:
        _, t = m.func.jvp(f, (a,), (m.ones_like(a),))
        return t

    return derivative


_DERIVATIVES: dict[str, Lift] = {
    "grad": _grad,
    "jacrev": _jacrev,
    "jacfwd": _jacfwd,
    "vjp": _vjp,
    "jvp": _jvp,
}


def _vmap(m: ModuleType, f: Fn) -> Fn:
    return m.func.vmap(f)


def _close(got: Tensor, want: Tensor) -> None:
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    assert float((got - want).abs().max().item()) < 1e-4, (got, want)


# ── derivative ∘ derivative, on one scalar ───────────────────────────────────


@pytest.mark.parametrize("inner", _DERIVATIVES)
@pytest.mark.parametrize("outer", _DERIVATIVES)
def test_second_derivative_of_a_scalar(device: str, outer: str, inner: str) -> None:
    f = _DERIVATIVES[outer](lucid, _DERIVATIVES[inner](lucid, _cube))
    x = _vector(lucid, [2.0], device)[0]
    _close(f(x), 6.0 * x)


@pytest.mark.parametrize("inner", _DERIVATIVES)
@pytest.mark.parametrize("outer", _DERIVATIVES)
def test_vmap_of_a_second_derivative(device: str, outer: str, inner: str) -> None:
    """``vmap(D(D'(f)))`` — per-example second derivatives, the case that
    used to fail with ``stack: empty input``."""
    f = _vmap(lucid, _DERIVATIVES[outer](lucid, _DERIVATIVES[inner](lucid, _cube)))
    xs = _vector(lucid, _XS, device)
    _close(f(xs), 6.0 * xs)


# ── derivative ∘ vmap ────────────────────────────────────────────────────────


def _total(m: ModuleType, f: Fn) -> Fn:
    return lambda xs: f(xs).sum()


#: A derivative applied to a vector function, with what it must give for the
#: elementwise ``6 x`` it is applied to: ``(build, expected)``.
_ON_VECTORS: dict[str, tuple[Callable[[ModuleType, Fn], Fn], Callable[[Tensor], Tensor]]] = {
    "grad": (lambda m, v: m.func.grad(_total(m, v)), lambda xs: 6.0 * xs),
    "jacrev": (
        lambda m, v: m.func.jacrev(v),
        lambda xs: lucid.eye(xs.shape[0], device=xs.device) * (6.0 * xs),
    ),
    "jacfwd": (
        lambda m, v: m.func.jacfwd(v),
        lambda xs: lucid.eye(xs.shape[0], device=xs.device) * (6.0 * xs),
    ),
    "vjp": (
        lambda m, v: lambda xs: m.func.vjp(v, xs)[1](m.ones_like(v(xs)))[0],
        lambda xs: 6.0 * xs,
    ),
    "jvp": (
        lambda m, v: lambda xs: m.func.jvp(v, (xs,), (m.ones_like(xs),))[1],
        lambda xs: 6.0 * xs,
    ),
}


@pytest.mark.parametrize("inner", _DERIVATIVES)
@pytest.mark.parametrize("outer", _ON_VECTORS)
def test_derivative_of_a_vmapped_derivative(device: str, outer: str, inner: str) -> None:
    build, expected = _ON_VECTORS[outer]
    f = build(lucid, _vmap(lucid, _DERIVATIVES[inner](lucid, _cube)))
    xs = _vector(lucid, _XS, device)
    _close(f(xs), expected(xs))


@pytest.mark.parametrize("inner", _DERIVATIVES)
def test_vmap_of_vmap_of_a_derivative(device: str, inner: str) -> None:
    f = _vmap(lucid, _vmap(lucid, _DERIVATIVES[inner](lucid, _cube)))
    xs = lucid.tensor([_XS, [2.0, 3.0, -1.0]], device=device)
    _close(f(xs), 3.0 * xs * xs)


# ── parity with the reference framework ──────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("inner", _DERIVATIVES)
@pytest.mark.parametrize("outer", _DERIVATIVES)
def test_second_derivative_matches_reference(ref: ModuleType, outer: str, inner: str) -> None:
    got = _vmap(lucid, _DERIVATIVES[outer](lucid, _DERIVATIVES[inner](lucid, _cube)))
    want = _vmap(ref, _DERIVATIVES[outer](ref, _DERIVATIVES[inner](ref, _cube)))
    _close(got(_vector(lucid, _XS, "cpu")), lucid.tensor(want(_vector(ref, _XS, "cpu")).tolist()))


@pytest.mark.parity
@pytest.mark.parametrize("inner", _DERIVATIVES)
@pytest.mark.parametrize("outer", _ON_VECTORS)
def test_derivative_of_vmap_matches_reference(ref: ModuleType, outer: str, inner: str) -> None:
    build, _ = _ON_VECTORS[outer]
    got = build(lucid, _vmap(lucid, _DERIVATIVES[inner](lucid, _cube)))
    want = build(ref, _vmap(ref, _DERIVATIVES[inner](ref, _cube)))
    _close(got(_vector(lucid, _XS, "cpu")), lucid.tensor(want(_vector(ref, _XS, "cpu")).tolist()))


# ── depth three, and the compositions the issue named ────────────────────────


def test_third_derivative(device: str) -> None:
    f = func.grad(func.grad(func.grad(_cube)))
    x = lucid.tensor(1.5, device=device)
    _close(f(x), lucid.tensor(6.0, device=device))


def test_jacrev_of_jacrev_is_the_hessian(device: str) -> None:
    f = lambda a: (a**3).sum()  # noqa: E731
    x = lucid.tensor(_XS, device=device)
    want = lucid.eye(3, device=device) * (6.0 * x)
    _close(func.jacrev(func.jacrev(f))(x), want)
    _close(func.jacfwd(func.jacrev(f))(x), want)
    _close(func.jacrev(func.grad(f))(x), want)
    _close(func.hessian(f)(x), want)


def test_grad_of_a_function_that_calls_grad(device: str) -> None:
    """The user-written composition: the inner ``grad`` is called inside the
    outer function, not wrapped around it."""
    x = lucid.tensor(_XS, device=device)
    inner = func.grad(lambda b: (b**3).sum())
    _close(func.grad(lambda a: inner(a).sum())(x), 6.0 * x)


def test_inner_derivative_through_a_captured_outer_tensor(device: str) -> None:
    """The inner transform's own argument does not require grad; what makes
    its result differentiable is the outer variable it captures."""
    c = lucid.tensor([1.0, 2.0], device=device)

    def f(a: Tensor) -> Tensor:
        # d/dc sum(a * c ** 2) = 2 a c, so the outer function is sum(2 a c).
        return func.grad(lambda b: (a * b**2).sum())(c).sum()

    a = lucid.tensor([3.0, 4.0], device=device)
    _close(func.grad(f)(a), 2.0 * c)


def test_jvp_of_grad_is_a_hessian_vector_product(device: str) -> None:
    x = lucid.tensor(_XS, device=device)
    g = func.grad(lambda a: (a**3).sum())
    y, hv = func.jvp(g, (x,), (lucid.ones_like(x),))
    _close(y, 3.0 * x * x)
    _close(hv, 6.0 * x)
    assert not hv.requires_grad and not y.requires_grad


# ── what carries a graph, and what does not ──────────────────────────────────


def test_gradient_at_an_input_that_requires_grad_is_differentiable(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    g = func.grad(lambda x: (x**3).sum())(a)
    assert g.requires_grad
    g.sum().backward()
    assert a.grad is not None
    _close(a.grad, 6.0 * a.detach())


def test_gradient_at_a_plain_input_is_a_plain_tensor(device: str) -> None:
    """Nothing asks for a second derivative, so no graph is kept."""
    x = lucid.tensor(_XS, device=device)
    f = lambda a: (a**3).sum()  # noqa: E731
    assert not func.grad(f)(x).requires_grad
    assert not func.grad(func.grad(lambda a: a**3))(x[0]).requires_grad
    assert not func.jacrev(func.jacrev(f))(x).requires_grad
    assert not func.jacfwd(func.jacrev(f))(x).requires_grad
    assert not func.vmap(func.grad(func.grad(_cube)))(x).requires_grad


def test_no_grad_keeps_no_graph(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    with lucid.no_grad():
        g = func.grad(lambda x: (x**3).sum())(a)
    assert not g.requires_grad


def test_grad_and_value_keeps_the_value_attached_inside_another_transform(device: str) -> None:
    f = lambda a: (a**3).sum()  # noqa: E731
    x = lucid.tensor(_XS, device=device)
    # d/da value(a) = f'(a): the value must still be on the graph.
    _close(func.grad(lambda a: func.grad_and_value(f)(a)[1])(x), 3.0 * x * x)


def test_the_transform_depth_is_back_at_zero_after_a_failure() -> None:
    def boom(a: Tensor) -> Tensor:
        raise ZeroDivisionError

    assert transform_depth() == 0
    with pytest.raises(ZeroDivisionError):
        func.grad(boom)(lucid.tensor(1.0))
    with pytest.raises(ZeroDivisionError):
        func.jacfwd(boom)(lucid.tensor(1.0))
    assert transform_depth() == 0


def test_a_user_function_sees_its_own_depth() -> None:
    seen: list[int] = []

    def probe(a: Tensor) -> Tensor:
        seen.append(transform_depth())
        return (a**2).sum()

    x = lucid.tensor([1.0, 2.0])
    func.grad(probe)(x)
    func.grad(lambda a: func.grad(probe)(a).sum())(x)
    func.jvp(probe, (x,), (x,))
    assert seen[0] == 1 and seen[-1] >= 1
    assert 2 in seen
    assert transform_depth() == 0
