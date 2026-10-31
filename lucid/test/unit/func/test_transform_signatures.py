"""The signatures of ``lucid.func``'s transforms, as the reference framework has them.

``jvp`` takes ``has_aux``; a tensor the output does not depend on has a
gradient of zeros, not ``None``; ``grad`` differentiates a 0-d output and
says so when it is not; the side output of a ``has_aux`` function and the
outputs of ``vjp`` come back plain unless the caller's graph runs through
them.  Each is a table of calls against the closed form and, under
``-m parity``, the reference framework.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.func as func
from lucid import Tensor
from lucid.func._function import ScalarOutputError
from lucid.func._pytree import tree_map

Fn = Callable[..., object]


def _t(m: ModuleType, values: object, device: str = "cpu", *, grad: bool = False) -> Tensor:
    t = lucid.tensor(values, device=device) if m is lucid else m.tensor(values)
    return t.requires_grad_(True) if grad else t


def _close(got: object, want: object) -> None:
    assert isinstance(got, Tensor) and isinstance(want, Tensor)
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    assert float((got - want).abs().max().item()) < 1e-5, (got, want)


# ── jvp(has_aux=) ────────────────────────────────────────────────────────────


def _jvp_aux(m: ModuleType, device: str, *, grad: bool) -> tuple[object, ...]:
    x = _t(m, [1.0, 2.0, 3.0], device, grad=grad)
    t = _t(m, [1.0, 0.5, 2.0], device)
    return m.func.jvp(lambda a: (a**2, {"half": a * 0.5}), (x,), (t,), has_aux=True)


def test_jvp_with_aux_returns_the_primal_the_tangent_and_the_aux(device: str) -> None:
    out, tangent, aux = _jvp_aux(lucid, device, grad=False)
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    _close(out, x**2)
    _close(tangent, 2.0 * x * lucid.tensor([1.0, 0.5, 2.0], device=device))
    assert isinstance(aux, dict)
    _close(aux["half"], x * 0.5)


def test_jvp_without_aux_still_returns_a_pair(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    got = func.jvp(lambda a: a**2, (x,), (x,))
    assert len(got) == 2
    got = func.jvp(lambda a: a**2, (x,), (x,), has_aux=False)
    assert len(got) == 2


def test_jvp_with_aux_needs_a_pair_from_func(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(ValueError, match="has_aux=True requires"):
        func.jvp(lambda a: a**2, (x,), (x,), has_aux=True)


def test_jvp_aux_is_any_structure(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    _, _, aux = func.jvp(lambda a: (a, 7), (x,), (x,), has_aux=True)
    assert aux == 7


@pytest.mark.parity
@pytest.mark.parametrize("grad", [False, True], ids=["plain", "requires_grad"])
def test_jvp_with_aux_matches_reference(ref: ModuleType, grad: bool) -> None:
    got, want = _jvp_aux(lucid, "cpu", grad=grad), _jvp_aux(ref, "cpu", grad=grad)
    assert len(got) == len(want) == 3
    for g, w in zip(got[:2], want[:2], strict=True):
        assert isinstance(g, Tensor)
        assert g.requires_grad == w.requires_grad  # type: ignore[attr-defined]
        assert g.tolist() == w.tolist()  # type: ignore[attr-defined]
    assert isinstance(got[2], dict) and isinstance(want[2], dict)
    assert got[2]["half"].tolist() == want[2]["half"].tolist()


# ── a tensor the output does not depend on has gradient zeros ───────────────


def _unused_cases(m: ModuleType, device: str) -> dict[str, Callable[[], object]]:
    a = _t(m, [1.0, 2.0], device)
    b = _t(m, [1.0, 2.0, 3.0], device)
    return {
        "grad-tuple": lambda: m.func.grad(lambda x, y: (x**2).sum(), argnums=(0, 1))(a, b),
        "grad-dict": lambda: m.func.grad(lambda d: (d["a"] ** 2).sum())({"a": a, "b": b}),
        "grad-list": lambda: m.func.grad(lambda d: (d[0] * 3.0).sum())([a, b]),
        "grad_and_value": lambda: m.func.grad_and_value(lambda x, y: x.sum(), argnums=(0, 1))(a, b)[
            0
        ],
        "grad-aux": lambda: m.func.grad(lambda x, y: (x.sum(), x), argnums=(0, 1), has_aux=True)(
            a, b
        )[0],
        "vjp": lambda: m.func.vjp(lambda x, y: x * 2.0, a, b)[1](_t(m, [1.0, 1.0], device)),
        "vjp-dict": lambda: m.func.vjp(lambda d: d["a"] * 2.0, {"a": a, "b": b})[1](
            _t(m, [1.0, 1.0], device)
        ),
        "disconnected-by-detach": lambda: m.func.grad(
            lambda x, y: (x.detach() * y).sum(), argnums=0
        )(a, a),
    }


def _leaves(tree: object) -> list[Tensor]:
    """The tensors of a result of either framework, in order."""
    if isinstance(tree, (tuple, list)):
        return [leaf for part in tree for leaf in _leaves(part)]
    if isinstance(tree, dict):
        return [leaf for part in tree.values() for leaf in _leaves(part)]
    return [tree]  # type: ignore[list-item]


#: What each case above gives, leaf by leaf: ``a = [1, 2]``, ``b = [1, 2, 3]``.
_UNUSED_WANT: dict[str, list[list[float]]] = {
    "grad-tuple": [[2.0, 4.0], [0.0, 0.0, 0.0]],
    "grad-dict": [[2.0, 4.0], [0.0, 0.0, 0.0]],
    "grad-list": [[3.0, 3.0], [0.0, 0.0, 0.0]],
    "grad_and_value": [[1.0, 1.0], [0.0, 0.0, 0.0]],
    "grad-aux": [[1.0, 1.0], [0.0, 0.0, 0.0]],
    "vjp": [[2.0, 2.0], [0.0, 0.0, 0.0]],
    "vjp-dict": [[2.0, 2.0], [0.0, 0.0, 0.0]],
    "disconnected-by-detach": [[0.0, 0.0]],
}


@pytest.mark.parametrize("name", list(_UNUSED_WANT))
def test_a_gradient_the_output_does_not_depend_on_is_zeros(device: str, name: str) -> None:
    result = _unused_cases(lucid, device)[name]()
    assert [leaf.tolist() for leaf in _leaves(result)] == _UNUSED_WANT[name]


def test_the_unused_gradient_has_the_shape_dtype_and_device_of_the_argument(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device)
    b = lucid.ones(2, 3, device=device, dtype=lucid.float16)
    _, gb = func.grad(lambda x, y: x.sum(), argnums=(0, 1))(a, b)
    assert isinstance(gb, Tensor)
    assert tuple(gb.shape) == (2, 3) and gb.dtype == b.dtype and gb.device == b.device
    assert float(gb.abs().sum().item()) == 0.0


def test_the_unused_gradient_is_zeros_inside_another_transform(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device)
    b = lucid.tensor([1.0, 2.0, 3.0], device=device)

    def inner(x: Tensor, y: Tensor) -> Tensor:
        gx, gy = func.grad(lambda p, q: (p**2).sum(), argnums=(0, 1))(x, y)
        return gx.sum() + gy.sum()

    _close(func.grad(inner)(a, b), lucid.full((2,), 2.0, device=device))


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_UNUSED_WANT))
def test_the_gradient_of_what_the_output_ignores_matches_reference(
    ref: ModuleType, name: str
) -> None:
    got = [leaf.tolist() for leaf in _leaves(_unused_cases(lucid, "cpu")[name]())]
    want = [leaf.tolist() for leaf in _leaves(_unused_cases(ref, "cpu")[name]())]
    assert got == want == _UNUSED_WANT[name]


# ── grad needs a 0-d output ──────────────────────────────────────────────────

_NON_SCALAR = {"(1,)": (1,), "(3,)": (3,), "(2, 2)": (2, 2), "(1, 1)": (1, 1), "(0,)": (0,)}


@pytest.mark.parametrize("transform", ["grad", "grad_and_value"])
@pytest.mark.parametrize("shape", list(_NON_SCALAR))
def test_grad_refuses_an_output_that_is_not_0d(device: str, transform: str, shape: str) -> None:
    dims = _NON_SCALAR[shape]
    x = lucid.ones(*dims, device=device)
    f = getattr(func, transform)(lambda a: a * 2.0)
    with pytest.raises(RuntimeError, match=rf"scalar Tensor, got tensor with {len(dims)} dims"):
        f(x)


def test_the_refusal_names_the_alternatives_and_is_also_a_value_error(device: str) -> None:
    with pytest.raises(ScalarOutputError, match="vjp or jacrev"):
        func.grad(lambda a: a)(lucid.ones(2, device=device))
    assert issubclass(ScalarOutputError, (RuntimeError, ValueError))


def test_grad_accepts_a_0d_output_however_it_is_made(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    for f in (lambda a: a.sum(), lambda a: (a**2).mean(), lambda a: a[0] * 2.0, lambda a: a.max()):
        func.grad(f)(x)


def test_vmap_of_grad_gives_each_example_a_scalar(device: str) -> None:
    x = lucid.arange(6.0, device=device).reshape(3, 2)
    _close(func.vmap(func.grad(lambda r: (r**2).sum()))(x), 2.0 * x)


def test_hessian_and_jacrev_still_take_a_non_scalar_output(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    _close(func.jacrev(lambda a: a * 3.0)(x), 3.0 * lucid.eye(2, device=device))


@pytest.mark.parity
@pytest.mark.parametrize("shape", list(_NON_SCALAR))
def test_grad_of_a_non_scalar_fails_as_the_reference_does(ref: ModuleType, shape: str) -> None:
    dims = _NON_SCALAR[shape]
    for m in (lucid, ref):
        with pytest.raises(RuntimeError, match="scalar Tensor, got tensor with"):
            m.func.grad(lambda a: a * 2.0)(m.ones(*dims))


# ── has_aux: the aux is plain unless the graph is kept ──────────────────────


def _aux_cases(m: ModuleType, device: str, *, grad: bool) -> dict[str, Callable[[], object]]:
    x = _t(m, [1.0, 2.0, 3.0], device, grad=grad)
    t = _t(m, [1.0, 1.0, 1.0], device)

    def loss(a: Tensor) -> tuple[Tensor, Tensor]:
        return (a**2).sum(), a * 3.0

    def vec(a: Tensor) -> tuple[Tensor, Tensor]:
        return a**2, a * 3.0

    return {
        "grad": lambda: m.func.grad(loss, has_aux=True)(x)[1],
        "grad_and_value": lambda: m.func.grad_and_value(loss, has_aux=True)(x)[1][1],
        "vjp": lambda: m.func.vjp(vec, x, has_aux=True)[0][1],
        "jacrev": lambda: m.func.jacrev(vec, has_aux=True)(x)[1],
        "jacfwd": lambda: m.func.jacfwd(vec, has_aux=True)(x)[1],
        "jvp": lambda: m.func.jvp(vec, (x,), (t,), has_aux=True)[2],
    }


@pytest.mark.parametrize("name", list(_aux_cases(lucid, "cpu", grad=False)))
def test_the_aux_of_a_plain_call_is_detached(device: str, name: str) -> None:
    aux = _aux_cases(lucid, device, grad=False)[name]()
    assert isinstance(aux, Tensor)
    assert not aux.requires_grad
    _close(aux, lucid.tensor([3.0, 6.0, 9.0], device=device))


@pytest.mark.parametrize("name", list(_aux_cases(lucid, "cpu", grad=True)))
def test_the_aux_of_a_call_at_an_argument_that_requires_grad_keeps_its_graph(
    device: str, name: str
) -> None:
    aux = _aux_cases(lucid, device, grad=True)[name]()
    assert isinstance(aux, Tensor)
    assert aux.requires_grad


def test_the_aux_inside_another_transform_stays_differentiable(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)

    def outer(a: Tensor) -> Tensor:
        _, aux = func.grad(lambda p: ((p**2).sum(), p * 3.0), has_aux=True)(a)
        assert isinstance(aux, Tensor)
        return aux.sum()

    _close(func.grad(outer)(x), lucid.full((3,), 3.0, device=device))


@pytest.mark.parity
@pytest.mark.parametrize("grad", [False, True], ids=["plain", "requires_grad"])
@pytest.mark.parametrize("name", list(_aux_cases(lucid, "cpu", grad=False)))
def test_the_aux_is_attached_where_the_reference_attaches_it(
    ref: ModuleType, name: str, grad: bool
) -> None:
    got = _aux_cases(lucid, "cpu", grad=grad)[name]()
    want = _aux_cases(ref, "cpu", grad=grad)[name]()
    assert isinstance(got, Tensor)
    assert got.requires_grad == want.requires_grad  # type: ignore[attr-defined]


# ── vjp's outputs ────────────────────────────────────────────────────────────


_VJP_STRUCTURES: dict[str, tuple[Fn, float]] = {
    "tensor": (lambda a: a * 2.0, 2.0),
    "tuple": (lambda a: (a * 2.0, a + 1.0), 3.0),
    "dict": (lambda a: {"u": a * 2.0, "v": a + 1.0}, 3.0),
}


@pytest.mark.parametrize("structure", list(_VJP_STRUCTURES))
def test_the_outputs_of_vjp_have_no_graph_when_the_caller_has_none(
    device: str, structure: str
) -> None:
    f, slope = _VJP_STRUCTURES[structure]
    x = lucid.tensor([1.0, 2.0], device=device)
    outputs, vjp_fn = func.vjp(f, x)
    assert not any(leaf.requires_grad for leaf in _leaves(outputs))
    # the pullback keeps the graph it needs
    (g,) = vjp_fn(tree_map(lambda t: lucid.ones_like(t), outputs))  # type: ignore[arg-type]
    _close(g, lucid.full((2,), slope, device=device))


def test_the_outputs_of_vjp_keep_the_graph_at_an_argument_that_requires_grad(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device).requires_grad_(True)
    out, _ = func.vjp(lambda a: a * 2.0, x)
    assert isinstance(out, Tensor)
    assert out.requires_grad
    out.sum().backward()
    _close(x.grad, lucid.full((2,), 2.0, device=device))


def test_the_outputs_of_vjp_keep_the_graph_inside_another_transform(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)

    def outer(a: Tensor) -> Tensor:
        out, _ = func.vjp(lambda p: p**2, a)
        assert isinstance(out, Tensor)
        return out.sum()

    _close(func.grad(outer)(x), 2.0 * x)


@pytest.mark.parity
@pytest.mark.parametrize("grad", [False, True], ids=["plain", "requires_grad"])
def test_the_outputs_of_vjp_are_attached_where_the_reference_attaches_them(
    ref: ModuleType, grad: bool
) -> None:
    flags = []
    for m in (lucid, ref):
        out, _ = m.func.vjp(lambda a: a * 2.0, _t(m, [1.0, 2.0], grad=grad))
        flags.append(out.requires_grad)
    assert flags[0] == flags[1]
