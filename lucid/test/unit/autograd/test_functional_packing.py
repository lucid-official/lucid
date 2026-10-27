"""The functional derivatives pack their results as the reference does (LCD-409).

``vjp(f, x, v)`` with one bare tensor ``x`` returned its gradient as a
1-tuple where the reference returns the tensor, gave ``None`` for an input
the outputs do not reach where the reference gives zeros, and ignored
``strict``; ``hvp`` and ``vhp`` did not exist.  All six now follow one
rule — a result per input is packed like ``inputs``, a result per output
like what ``func`` returned — and ``strict`` refuses what would otherwise
be filled with zeros.  Every combination of function, input packing,
``create_graph`` and ``strict`` is held to the reference, structure and
values in float64.  ``jacobian`` and ``hessian`` blocks take the
reference's shapes too — ``out.shape + x.shape``, one block per output —
where they were flattened to ``(numel(out), numel(x))`` and several
outputs' rows were concatenated.
"""

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.autograd as AG

_X = [[0.3, -0.7, 1.1], [0.5, 0.2, -0.4]]
_Y = [1.5, -0.5, 0.25]
_TIGHT = 1e-10

_FUNCTIONS = ["vjp", "jvp", "jacobian", "hessian", "hvp", "vhp"]
_SECOND_ORDER = {"hessian", "hvp", "vhp"}
#: Taken with one tensor output only: the second-order functions require it.
_ONE_OUTPUT = _SECOND_ORDER
#: How ``inputs`` is passed: one bare tensor, a 1-tuple, a pair.
_PACKINGS = ["tensor", "one_tuple", "pair"]


def _namespace(m: ModuleType) -> Any:
    return AG if m is lucid else m.autograd.functional


def _vector_f(m: ModuleType, outputs: str) -> Callable[..., Any]:
    def f(a: Any, b: Any = None) -> Any:
        y = m.sin(a) * a if b is None else m.sin(a) * b
        return (y, (a * a).sum()) if outputs == "tuple" else y

    return f


def _scalar_f(m: ModuleType) -> Callable[..., Any]:
    def f(a: Any, b: Any = None) -> Any:
        s = (a**3).sum() + (m.exp(a) * a).sum()
        return s if b is None else s + (a.sum() * b * b).sum()

    return f


def _inputs(m: ModuleType, packing: str, requires_grad: bool) -> Any:
    x = m.tensor(_X, dtype=m.float64, requires_grad=requires_grad)
    if packing == "tensor":
        return x
    if packing == "one_tuple":
        return (x,)
    return (x, m.tensor(_Y, dtype=m.float64, requires_grad=requires_grad))


def _like(t: Any, scale: float) -> Any:
    return (t.detach() * 0 + scale).detach()


def _seed(values: Any, scale: float) -> Any:
    """A seed packed exactly as ``values`` is."""
    if isinstance(values, tuple):
        return tuple(_like(t, scale + 0.25 * i) for i, t in enumerate(values))
    return _like(values, scale)


def _call(
    m: ModuleType,
    name: str,
    packing: str,
    outputs: str,
    create_graph: bool,
    strict: bool,
) -> Any:
    ns = _namespace(m)
    inputs = _inputs(m, packing, requires_grad=create_graph)
    flags = {"create_graph": create_graph, "strict": strict}
    if name in _SECOND_ORDER:
        f = _scalar_f(m)
        if name == "hessian":
            return ns.hessian(f, inputs, **flags)
        return getattr(ns, name)(f, inputs, _seed(inputs, 0.3), **flags)
    f = _vector_f(m, outputs)
    if name == "jacobian":
        return ns.jacobian(f, inputs, **flags)
    if name == "jvp":
        return ns.jvp(f, inputs, _seed(inputs, 0.3), **flags)
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    return ns.vjp(f, inputs, _seed(f(*args), 0.7), **flags)


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """``("value", result)``, or ``("raised", None)`` for a refusal."""
    try:
        return "value", call()
    except RuntimeError:
        return "raised", None


def _structure(r: Any) -> Any:
    """The nesting of ``r`` and each tensor's shape."""
    if isinstance(r, tuple):
        return tuple(_structure(x) for x in r)
    return tuple(r.shape)


def _flat(r: Any) -> list[float]:
    if isinstance(r, tuple):
        return [v for x in r for v in _flat(x)]
    return [float(v) for v in r.detach().reshape(-1).tolist()]


def _grad_flags(r: Any) -> list[bool]:
    if isinstance(r, tuple):
        return [f for x in r for f in _grad_flags(x)]
    return [bool(r.requires_grad)]


def _cases() -> list[Any]:
    cases = []
    for name in _FUNCTIONS:
        for packing in _PACKINGS:
            outs = ["tensor"] if name in _ONE_OUTPUT else ["tensor", "tuple"]
            for outputs in outs:
                cases.append(
                    pytest.param(name, packing, outputs, id=f"{name}-{packing}-{outputs}-out")
                )
    return cases


@pytest.mark.parametrize("strict", [False, True], ids=["lax", "strict"])
@pytest.mark.parametrize("create_graph", [False, True], ids=["plain", "create_graph"])
@pytest.mark.parametrize(("name", "packing", "outputs"), _cases())
def test_matches_the_reference(
    ref: ModuleType,
    name: str,
    packing: str,
    outputs: str,
    create_graph: bool,
    strict: bool,
) -> None:
    kind, got = _outcome(lambda: _call(lucid, name, packing, outputs, create_graph, strict))
    want_kind, want = _outcome(lambda: _call(ref, name, packing, outputs, create_graph, strict))
    assert kind == want_kind
    if kind == "raised":
        return
    assert _structure(got) == _structure(want)
    g, w = _flat(got), _flat(want)
    assert len(g) == len(w)
    assert all(
        math.isclose(a, b, rel_tol=_TIGHT, abs_tol=_TIGHT) for a, b in zip(g, w, strict=True)
    ), (g, w)
    assert _grad_flags(got) == _grad_flags(want)


@pytest.mark.parametrize("name", _FUNCTIONS)
def test_strict_refuses_an_input_the_function_ignores(ref: ModuleType, name: str) -> None:
    for m in (lucid, ref):
        ns = _namespace(m)
        x, y = _inputs(m, "pair", requires_grad=False)
        if name in _SECOND_ORDER:

            def f(a: Any, b: Any) -> Any:
                return (a**3).sum()

            call = (
                (lambda: ns.hessian(f, (x, y), strict=True))
                if name == "hessian"
                else (lambda: getattr(ns, name)(f, (x, y), (x, y), strict=True))
            )
        else:

            def g(a: Any, b: Any) -> Any:
                return a * a

            call = {
                "vjp": lambda: ns.vjp(g, (x, y), x, strict=True),
                "jvp": lambda: ns.jvp(g, (x, y), (x, y), strict=True),
                "jacobian": lambda: ns.jacobian(g, (x, y), strict=True),
            }[name]
        with pytest.raises(RuntimeError, match="strict mode"):
            call()


@pytest.mark.parametrize("name", ["vjp", "hvp", "vhp"])
def test_without_strict_an_ignored_input_gets_zeros(name: str, device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    y = lucid.tensor([3.0], device=device)

    def f(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return (a**3).sum()

    v = (lucid.ones_like(x), lucid.ones_like(y))
    _, result = getattr(AG, name)(
        f, (x, y), lucid.tensor(1.0, device=device) if name == "vjp" else v
    )
    assert isinstance(result, tuple)
    assert result[0].tolist() == ([3.0, 12.0] if name == "vjp" else [6.0, 12.0])
    assert result[1].tolist() == [0.0]


@pytest.mark.parametrize("name", ["hvp", "vhp"])
def test_hessian_products_run_on_every_device(name: str, device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    out, product = getattr(AG, name)(
        lambda a: (a**3).sum(),
        x,
        lucid.tensor([1.0, -1.0], device=device),
        create_graph=True,
    )
    assert product.device == x.device
    assert product.tolist() == [6.0, -12.0]
    assert float(out.item()) == 9.0
    (second,) = lucid.autograd.grad(product.sum(), x)
    assert second.tolist() == [6.0, -6.0]


@pytest.mark.parametrize("name", ["hvp", "vhp"])
def test_hessian_products_refuse_a_function_of_several_elements(name: str) -> None:
    with pytest.raises(RuntimeError, match="one element"):
        getattr(AG, name)(lambda a: a * a, lucid.tensor([1.0, 2.0]), lucid.ones(2))


@pytest.mark.parametrize("name", ["hvp", "vhp"])
def test_hessian_products_seed_a_single_element_input(ref: ModuleType, name: str) -> None:
    got = getattr(AG, name)(lambda a: a**3, lucid.tensor(2.0, dtype=lucid.float64))
    want = getattr(ref.autograd.functional, name)(
        lambda a: a**3, ref.tensor(2.0, dtype=ref.float64)
    )
    assert _flat(got) == _flat(want)
    with pytest.raises(RuntimeError, match="v can only be None"):
        getattr(AG, name)(lambda a: (a**3).sum(), lucid.tensor([1.0, 2.0]))


@pytest.mark.parametrize("name", ["hessian", "jacobian"])
def test_strict_refuses_a_linear_function(ref: ModuleType, name: str) -> None:
    for m in (ref, lucid):
        ns = _namespace(m)
        x = m.tensor([1.0, 2.0], dtype=m.float64)
        calls = {
            "jacobian": lambda: ns.jacobian(lambda a: a * 2, x, create_graph=True, strict=True),
            "hessian": lambda: ns.hessian(lambda a: (a * 2).sum(), x, strict=True),
        }
        with pytest.raises(RuntimeError, match="strict mode"):
            calls[name]()
