"""``autograd.jacobian / hessian / vjp / jvp`` are exact and honour ``create_graph`` (LCD-303).

``jvp`` was a central finite difference with ``eps=1e-4`` — off in the
eighth digit on a smooth function, and not differentiable at all.
``hessian`` read its rows back from ``.grad`` as plain values, so
``create_graph=True`` returned a constant.  All four now go through the
graph with ``autograd.grad``: ``jvp`` by differentiating a vector-Jacobian
product with respect to its cotangent (the double-vjp trick).

Values are checked against the reference in float64 at ``1e-10`` — tighter
than any finite difference can reach — and on Metal in float32.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.autograd import hessian, jacobian, jvp, vjp

_X = [0.3, -0.7, 1.1]
_V = [0.5, 2.0, -1.5]
_TIGHT = 1e-10


def _flat(t: object) -> list[float]:
    if isinstance(t, (list, tuple)):
        return [v for item in t for v in _flat(item)]
    values = getattr(t, "detach")().cpu().reshape(-1).tolist()
    return [float(v) for v in values]


def _close(got: list[float], want: list[float], tol: float) -> bool:
    return len(got) == len(want) and all(
        math.isclose(g, w, rel_tol=tol, abs_tol=tol) for g, w in zip(got, want, strict=True)
    )


def _vector_f(m: ModuleType) -> Callable[[object], object]:
    # Its third derivative is large, so a finite difference misses by ~1e-7.
    return lambda x: m.exp(3.0 * x) * m.sin(x) + x * x.sum()


def _scalar_f(m: ModuleType) -> Callable[[object], object]:
    return lambda x: (m.exp(2.0 * x) * m.cos(x)).sum() + (x * x).sum() * x[0]


def _two_inputs_f(m: ModuleType) -> Callable[[object, object], object]:
    return lambda x, y: (m.sin(x) * y.exp()).sum() + (x * x * y).sum()


def _as(m: ModuleType, values: list[float], device: str, dtype_name: str) -> object:
    return m.tensor(values, dtype=getattr(m, dtype_name), device=device)


_CASES = {
    "jvp": lambda m, x, v: m_jvp(m)(_vector_f(m), x, v)[1],
    "vjp": lambda m, x, v: m_vjp(m)(_vector_f(m), x, v)[1],
    "jacobian": lambda m, x, v: m_jac(m)(_vector_f(m), x),
    "hessian": lambda m, x, v: m_hess(m)(_scalar_f(m), x),
}


def m_jvp(m: ModuleType) -> Callable[..., tuple[object, object]]:
    return jvp if m is lucid else m.autograd.functional.jvp


def m_vjp(m: ModuleType) -> Callable[..., tuple[object, object]]:
    return vjp if m is lucid else m.autograd.functional.vjp


def m_jac(m: ModuleType) -> Callable[..., object]:
    return jacobian if m is lucid else m.autograd.functional.jacobian


def m_hess(m: ModuleType) -> Callable[..., object]:
    return hessian if m is lucid else m.autograd.functional.hessian


@pytest.mark.parametrize("name", list(_CASES))
def test_float64_matches_the_reference_exactly(name: str, ref: ModuleType) -> None:
    got = _CASES[name](lucid, _as(lucid, _X, "cpu", "float64"), _as(lucid, _V, "cpu", "float64"))
    want = _CASES[name](ref, _as(ref, _X, "cpu", "float64"), _as(ref, _V, "cpu", "float64"))
    assert _close(_flat(got), _flat(want), _TIGHT), (_flat(got), _flat(want))


@pytest.mark.parametrize("name", list(_CASES))
def test_float32_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    got = _CASES[name](lucid, _as(lucid, _X, device, "float32"), _as(lucid, _V, device, "float32"))
    want = _CASES[name](ref, _as(ref, _X, "cpu", "float32"), _as(ref, _V, "cpu", "float32"))
    assert _close(_flat(got), _flat(want), 1e-5), (_flat(got), _flat(want))


def test_jvp_is_exact_without_the_reference() -> None:
    # d/dt f(x + t v) for f = exp(3x) sin(x) + x * sum(x), worked by hand.
    x = _as(lucid, _X, "cpu", "float64")
    v = _as(lucid, _V, "cpu", "float64")
    _, tangent = jvp(_vector_f(lucid), x, v)
    sv = sum(_V)
    sx = sum(_X)
    want = [
        (3 * math.exp(3 * a) * math.sin(a) + math.exp(3 * a) * math.cos(a)) * b + b * sx + a * sv
        for a, b in zip(_X, _V, strict=True)
    ]
    assert _close(_flat(tangent), want, _TIGHT)


def test_two_inputs_and_cross_hessian_blocks(ref: ModuleType) -> None:
    def run(m: ModuleType) -> list[float]:
        x = _as(m, [0.2, 0.4], "cpu", "float64")
        y = _as(m, [1.5, -0.5], "cpu", "float64")
        f = _two_inputs_f(m)
        _, t = m_jvp(m)(f, (x, y), (x * 0 + 1, y * 0 + 2))
        _, g = m_vjp(m)(f, (x, y), m.tensor(1.0, dtype=m.float64))
        return _flat([t, list(g), list(m_jac(m)(f, (x, y))), m_hess(m)(f, (x, y))])

    assert _close(run(lucid), run(ref), _TIGHT)


def test_tuple_outputs(ref: ModuleType) -> None:
    def run(m: ModuleType) -> list[float]:
        x = _as(m, _X, "cpu", "float64")
        v = _as(m, _V, "cpu", "float64")
        out, t = m_jvp(m)(lambda a: (a.sin(), (a * a).sum()), x, v)
        assert isinstance(out, tuple) and isinstance(t, tuple)
        return _flat([out, t])

    assert _close(run(lucid), run(ref), _TIGHT)


@pytest.mark.parametrize("name", ["jvp", "vjp", "jacobian", "hessian"])
def test_create_graph_results_differentiate_back_to_the_input(name: str, ref: ModuleType) -> None:
    def run(m: ModuleType) -> list[float]:
        x = _as(m, _X, "cpu", "float64").requires_grad_(True)
        v = _as(m, _V, "cpu", "float64")
        if name == "jvp":
            out = m_jvp(m)(_vector_f(m), x, v, create_graph=True)[1]
        elif name == "vjp":
            out = m_vjp(m)(_vector_f(m), x, v, create_graph=True)[1]
        elif name == "jacobian":
            out = m_jac(m)(_vector_f(m), x, create_graph=True)
        else:
            out = m_hess(m)(_scalar_f(m), x, create_graph=True)
        out = out[0] if isinstance(out, tuple) else out
        assert out.requires_grad
        (g,) = m.autograd.grad((out * out).sum(), x)
        return _flat(g)

    assert _close(run(lucid), run(ref), 1e-9)


def test_third_order_through_hessian() -> None:
    # f = sum(x^4): H = diag(12 x^2), d/dx sum(H) = 24 x.
    x = lucid.tensor([0.5, -1.0, 2.0], dtype=lucid.float64, requires_grad=True)
    H = hessian(lambda a: (a**4).sum(), x, create_graph=True)
    (g,) = lucid.autograd.grad(H.sum(), x)
    assert _close(_flat(g), [12.0, -24.0, 48.0], _TIGHT)


def test_gradcheck_through_a_create_graph_hessian() -> None:
    x = lucid.tensor([0.3, -0.7], dtype=lucid.float64, requires_grad=True)
    w = lucid.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=lucid.float64)
    assert lucid.autograd.gradcheck(
        lambda a: (hessian(_scalar_f(lucid), a, create_graph=True) * w).sum(), [x]
    )


def test_create_graph_off_gives_constants(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    out, t = jvp(lambda a: a * a, x, lucid.ones_like(x))
    assert not out.requires_grad and not t.requires_grad
    assert not hessian(lambda a: (a**3).sum(), x).requires_grad


def test_an_output_independent_of_the_input_has_zero_derivatives(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    c = lucid.tensor([5.0, 6.0], device=device)
    _, t = jvp(lambda a: c * 2, x, lucid.ones_like(x))
    assert t.tolist() == [0.0, 0.0]
    assert jacobian(lambda a: c * 2, x).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert hessian(lambda a: a.sum(), x).tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_runs_under_no_grad(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with lucid.no_grad():
        assert jacobian(lambda a: a * a, x).tolist() == [[2.0, 0.0], [0.0, 4.0]]
        assert jvp(lambda a: a * a, x, lucid.ones_like(x))[1].tolist() == [2.0, 4.0]


@pytest.mark.parametrize("module", ["lucid", "ref"])
def test_a_cotangent_of_another_shape_is_refused(
    module: str, request: pytest.FixtureRequest
) -> None:
    # vjp used to reshape a one-element v to the output's shape on its own,
    # ahead of the seed rules every other entry point follows.
    m: ModuleType = lucid if module == "lucid" else request.getfixturevalue("ref")
    vjp_of = m_vjp(m)
    x = m.tensor([2.0])
    with pytest.raises(RuntimeError):
        vjp_of(lambda a: a * a, x, m.tensor(1.0))


def test_a_tangent_of_another_shape_is_refused() -> None:
    x = lucid.tensor([1.0, 2.0])
    with pytest.raises(lucid._C.engine.ShapeMismatch):
        jvp(lambda a: a * a, x, lucid.ones(3))


def test_hessian_refuses_a_non_scalar_function() -> None:
    with pytest.raises(RuntimeError, match="one element"):
        hessian(lambda a: a * a, lucid.tensor([1.0, 2.0]))
