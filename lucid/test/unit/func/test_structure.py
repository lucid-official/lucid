"""``lucid.func`` takes pytrees in and gives pytrees back.

Tuples, lists, dicts, namedtuples and nests of them are walked leaf by
leaf by every transform — ``vmap`` used to pass a dict through unbatched
(a silently wrong scalar), ``jacfwd`` kept only the first of a tuple of
outputs, and ``grad`` / ``vjp`` / ``jvp`` / ``jacrev`` refused anything but
a tensor.  The matrices below cross {tensor, tuple, list, dict, namedtuple,
nested} with each transform, once for the input and once for the output,
against the closed form and (under ``-m parity``) the reference framework.
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from types import ModuleType

import pytest

import lucid
import lucid.func as func
from lucid import Tensor
from lucid.func._pytree import tree_flatten, tree_map

Pair = namedtuple("Pair", ["u", "v"])  # noqa: PYI024

Builder = Callable[[Sequence[object]], object]

#: Structures to put tensors in: a builder from a list of leaves, and how
#: many leaves it takes.
_SHAPES: dict[str, tuple[Builder, int]] = {
    "tensor": (lambda leaf: leaf[0], 1),
    "tuple": (lambda leaf: (leaf[0], leaf[1]), 2),
    "list": (lambda leaf: [leaf[0], leaf[1]], 2),
    "dict": (lambda leaf: {"a": leaf[0], "b": leaf[1]}, 2),
    "namedtuple": (lambda leaf: Pair(leaf[0], leaf[1]), 2),
    "nested": (lambda leaf: {"x": (leaf[0], [leaf[1]]), "y": leaf[2]}, 3),
}

_LEAF_VALUES = ([1.0, 2.0, 3.0], [4.0, 5.0], 2.0)


def _tensor(m: ModuleType, values: object, device: str) -> Tensor:
    return lucid.tensor(values, device=device) if m is lucid else m.tensor(values)


def _leaves(tree: object) -> list[object]:
    return tree_flatten(tree)[0]


def _close(got: Tensor, want: Tensor) -> None:
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    assert float((got - want).abs().max().item()) < 1e-4, (got, want)


def _same(got: object, want: object) -> None:
    """Same structure, same shapes, same values."""
    got_leaves, got_spec = tree_flatten(got)
    want_leaves, want_spec = tree_flatten(want)
    assert got_spec == want_spec, f"{got_spec} != {want_spec}"
    for g, w in zip(got_leaves, want_leaves, strict=True):
        assert isinstance(g, Tensor) and isinstance(w, Tensor)
        _close(g, w)


def _python(x: object) -> object:
    """A result of either framework as builtins, container types kept."""
    if isinstance(x, tuple):
        return (type(x).__name__, [_python(i) for i in x])
    if isinstance(x, list):
        return ("list", [_python(i) for i in x])
    if isinstance(x, dict):
        return ("dict", {k: _python(v) for k, v in x.items()})
    return x.tolist() if hasattr(x, "tolist") else x


def _agree(got: object, want: object) -> None:
    if isinstance(got, float) or isinstance(want, float):
        assert isinstance(got, float) and isinstance(want, float)
        assert abs(got - want) < 1e-4
    elif isinstance(got, (tuple, list)):
        assert isinstance(want, (tuple, list)) and len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _agree(g, w)
    elif isinstance(got, dict):
        assert isinstance(want, dict) and got.keys() == want.keys()
        for k in got:
            _agree(got[k], want[k])
    else:
        assert got == want


def _inputs(m: ModuleType, shape: str, device: str) -> object:
    build, n = _SHAPES[shape]
    return build([_tensor(m, v, device) for v in _LEAF_VALUES[:n]])


# ── pytree in: a scalar function of a tree ───────────────────────────────────


def _total(tree: object) -> Tensor:
    """The sum of the cubes of every leaf — the same function for any tree."""
    leaves = _leaves(tree)
    total = (leaves[0] ** 3).sum()
    for leaf in leaves[1:]:
        total = total + (leaf**3).sum()
    return total


def _batched(m: ModuleType, shape: str, device: str) -> object:
    """The tree with a batch of two along the first axis of every leaf."""
    tree = _inputs(m, shape, device)
    return tree_map(lambda t: m.stack([t, t * 2.0]), tree)


def _hessian_want(tree: object) -> object:
    """``tree`` of ``tree`` of blocks: ``diag(6 x)`` on the diagonal, zeros off it."""
    leaves, spec = tree_flatten(tree)
    rows = []
    for k, a in enumerate(leaves):
        assert isinstance(a, Tensor)
        row = []
        for j, b in enumerate(leaves):
            assert isinstance(b, Tensor)
            if j == k:
                eye = lucid.eye(a.numel(), device=a.device)
                row.append((eye * (6.0 * a.reshape(-1))).reshape(tuple(a.shape) + tuple(a.shape)))
            else:
                row.append(lucid.zeros(tuple(a.shape) + tuple(b.shape), device=a.device))
        rows.append(spec.unflatten(row))
    return spec.unflatten(rows)


def _vjp_of(m: ModuleType, tree: object) -> object:
    out, pullback = m.func.vjp(_total, tree)
    return pullback(m.ones_like(out))


def _jvp_of(m: ModuleType, tree: object) -> object:
    return m.func.jvp(_total, (tree,), (tree_map(m.ones_like, tree),))


_ON_TREES: dict[str, Callable[[ModuleType, object], object]] = {
    "grad": lambda m, tree: m.func.grad(_total)(tree),
    "vjp": _vjp_of,
    "jvp": _jvp_of,
    "jacrev": lambda m, tree: m.func.jacrev(_total)(tree),
    "jacfwd": lambda m, tree: m.func.jacfwd(_total)(tree),
    "hessian": lambda m, tree: m.func.hessian(_total)(tree),
}


def _want_on_trees(transform: str, tree: object) -> object:
    cubes = tree_map(lambda t: 3.0 * t * t, tree)
    if transform in ("grad", "jacrev", "jacfwd"):
        return cubes
    if transform == "vjp":
        return (cubes,)
    if transform == "jvp":
        total = _total(tree)
        slope = sum(float(t.sum().item()) for t in _leaves(cubes))
        return (total, lucid.tensor(slope, device=total.device))
    return _hessian_want(tree)


@pytest.mark.parametrize("transform", _ON_TREES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_transform_of_a_tree_argument(device: str, shape: str, transform: str) -> None:
    tree = _inputs(lucid, shape, device)
    _same(_ON_TREES[transform](lucid, tree), _want_on_trees(transform, tree))


@pytest.mark.parity
@pytest.mark.parametrize("transform", _ON_TREES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_transform_of_a_tree_argument_matches_reference(
    ref: ModuleType, shape: str, transform: str
) -> None:
    got = _ON_TREES[transform](lucid, _inputs(lucid, shape, "cpu"))
    want = _ON_TREES[transform](ref, _inputs(ref, shape, "cpu"))
    _agree(_python(got), _python(want))


@pytest.mark.parametrize("shape", _SHAPES)
def test_vmap_of_a_tree_argument(device: str, shape: str) -> None:
    """Every leaf is batched; a container is never passed through whole."""
    batch = _batched(lucid, shape, device)
    got = func.vmap(_total)(batch)
    examples = [tree_map(lambda t, i=i: t[i], batch) for i in range(2)]
    _close(got, lucid.stack([_total(e) for e in examples]))


@pytest.mark.parity
@pytest.mark.parametrize("shape", _SHAPES)
def test_vmap_of_a_tree_argument_matches_reference(ref: ModuleType, shape: str) -> None:
    got = func.vmap(_total)(_batched(lucid, shape, "cpu"))
    want = ref.func.vmap(_total)(_batched(ref, shape, "cpu"))
    _agree(_python(got), _python(want))


# ── pytree out: a function of a tensor returning a tree ──────────────────────

_LEAF_FUNCTIONS: tuple[Callable[[Tensor], Tensor], ...] = (
    lambda x: x * 2.0,
    lambda x: x**2,
    lambda x: x.sum(),
)

_X = [1.0, 2.0, 3.0]


def _g(shape: str) -> Callable[[Tensor], object]:
    build, n = _SHAPES[shape]
    return lambda x: build([f(x) for f in _LEAF_FUNCTIONS[:n]])


def _jacobian_blocks(x: Tensor) -> list[Tensor]:
    """``d f_i / d x`` for the three leaf functions, by hand."""
    eye = lucid.eye(3, device=x.device)
    return [2.0 * eye, eye * (2.0 * x), lucid.ones(3, device=x.device)]


def _want_of_outputs(transform: str, shape: str, x: Tensor) -> object:
    build, n = _SHAPES[shape]
    jac = _jacobian_blocks(x)[:n]
    if transform in ("jacrev", "jacfwd"):
        return build(jac)
    ones = lucid.ones(3, device=x.device)
    if transform == "jvp":
        tangent = build([j @ ones if j.ndim == 2 else (j * ones).sum() for j in jac])
        return (_g(shape)(x), tangent)
    # vjp with a cotangent of ones: sum_i J_i^T 1
    pulled = sum((j.sum(dim=0) if j.ndim == 2 else j for j in jac), lucid.zeros(3, device=x.device))
    return (pulled,)


def _vjp_out(m: ModuleType, shape: str, x: Tensor) -> object:
    out, pullback = m.func.vjp(_g(shape), x)
    return pullback(tree_map(m.ones_like, out))


_OF_TREES: dict[str, Callable[[ModuleType, str, Tensor], object]] = {
    "jacrev": lambda m, shape, x: m.func.jacrev(_g(shape))(x),
    "jacfwd": lambda m, shape, x: m.func.jacfwd(_g(shape))(x),
    "jvp": lambda m, shape, x: m.func.jvp(_g(shape), (x,), (m.ones_like(x),)),
    "vjp": _vjp_out,
}


@pytest.mark.parametrize("transform", _OF_TREES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_transform_of_a_tree_output(device: str, shape: str, transform: str) -> None:
    x = _tensor(lucid, _X, device)
    _same(_OF_TREES[transform](lucid, shape, x), _want_of_outputs(transform, shape, x))


@pytest.mark.parity
@pytest.mark.parametrize("transform", _OF_TREES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_transform_of_a_tree_output_matches_reference(
    ref: ModuleType, shape: str, transform: str
) -> None:
    got = _OF_TREES[transform](lucid, shape, _tensor(lucid, _X, "cpu"))
    want = _OF_TREES[transform](ref, shape, _tensor(ref, _X, "cpu"))
    _agree(_python(got), _python(want))


@pytest.mark.parametrize("shape", _SHAPES)
def test_vmap_of_a_tree_output(device: str, shape: str) -> None:
    build, n = _SHAPES[shape]
    xs = lucid.tensor([_X, [4.0, 5.0, 6.0]], device=device)
    got = func.vmap(_g(shape))(xs)
    per_example = [f(xs[0]) for f in _LEAF_FUNCTIONS[:n]], [f(xs[1]) for f in _LEAF_FUNCTIONS[:n]]
    want = build([lucid.stack([a, b]) for a, b in zip(*per_example, strict=True)])
    _same(got, want)


@pytest.mark.parity
@pytest.mark.parametrize("shape", _SHAPES)
def test_vmap_of_a_tree_output_matches_reference(ref: ModuleType, shape: str) -> None:
    rows = [_X, [4.0, 5.0, 6.0]]
    got = func.vmap(_g(shape))(_tensor(lucid, rows, "cpu"))
    want = ref.func.vmap(_g(shape))(_tensor(ref, rows, "cpu"))
    _agree(_python(got), _python(want))


# ── tree in and tree out together, and several arguments ─────────────────────


def _mixed(d: dict[str, Tensor], e: Tensor) -> dict[str, object]:
    return {"y": d["a"] * e, "s": (d["a"].sum(), [e * 2.0])}


def _mixed_jacobian(m: ModuleType, transform: str) -> object:
    a = _tensor(m, _X, "cpu")
    return getattr(m.func, transform)(_mixed, argnums=(0, 1))({"a": a}, a)


@pytest.mark.parametrize("transform", ["jacrev", "jacfwd"])
def test_jacobian_of_a_tree_with_respect_to_several_arguments(device: str, transform: str) -> None:
    """The result has the output's structure; each leaf holds, per argument,
    the Jacobian block in that argument's structure."""
    a = lucid.tensor(_X, device=device)
    eye = lucid.eye(3, device=device)
    ones = lucid.ones(3, device=device)
    zeros = lucid.zeros(3, 3, device=device)
    want = {
        "y": ({"a": eye * a}, eye * a),
        "s": (({"a": ones}, lucid.zeros(3, device=device)), [({"a": zeros}, 2.0 * eye)]),
    }
    got = getattr(func, transform)(_mixed, argnums=(0, 1))({"a": a}, a)
    _same(got, want)


@pytest.mark.parity
@pytest.mark.parametrize("transform", ["jacrev", "jacfwd"])
def test_jacobian_of_a_tree_matches_reference(ref: ModuleType, transform: str) -> None:
    _agree(_python(_mixed_jacobian(lucid, transform)), _python(_mixed_jacobian(ref, transform)))


def _hessian_of_two_trees(m: ModuleType, device: str) -> object:
    a = _tensor(m, [1.0, 2.0], device)
    b = _tensor(m, [3.0, 4.0], device)

    def f(p: dict[str, Tensor], q: list[Tensor]) -> Tensor:
        return (p["u"] * q[0] * q[0]).sum()

    return m.func.hessian(f, argnums=(0, 1))({"u": a}, [b])


def test_hessian_with_respect_to_two_tree_arguments(device: str) -> None:
    """The structure of the gradient, with the structure of the arguments
    at each of its leaves: ``H[leaf of grad][argument][leaf of argument]``."""
    a = lucid.tensor([1.0, 2.0], device=device)
    b = lucid.tensor([3.0, 4.0], device=device)
    eye = lucid.eye(2, device=device)
    h_ab = eye * (2.0 * b)  # d2f / da db
    want = (
        {"u": ({"u": lucid.zeros(2, 2, device=device)}, [h_ab])},
        [({"u": h_ab}, [eye * (2.0 * a)])],
    )
    _same(_hessian_of_two_trees(lucid, device), want)


@pytest.mark.parity
def test_hessian_with_respect_to_two_tree_arguments_matches_reference(ref: ModuleType) -> None:
    _agree(_python(_hessian_of_two_trees(lucid, "cpu")), _python(_hessian_of_two_trees(ref, "cpu")))


# ── has_aux carries a tree through untouched ─────────────────────────────────


def _aux_fn(d: dict[str, Tensor]) -> tuple[Tensor, dict[str, object]]:
    return (d["a"] ** 2).sum(), {"k": d["a"] * 10.0, "n": [1, "tag"]}


@pytest.mark.parametrize("transform", ["grad", "jacrev", "jacfwd"])
def test_has_aux_with_a_tree_aux(device: str, transform: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device)
    out, aux = getattr(func, transform)(_aux_fn, has_aux=True)({"a": a})
    _same(out, {"a": 2.0 * a})
    assert list(aux) == ["k", "n"] and aux["n"] == [1, "tag"]
    _close(aux["k"], 10.0 * a)


def test_grad_and_value_with_a_tree_aux(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device)
    grads, (value, aux) = func.grad_and_value(_aux_fn, has_aux=True)({"a": a})
    _same(grads, {"a": 2.0 * a})
    _close(value, lucid.tensor(5.0, device=device))
    assert aux["n"] == [1, "tag"]


def test_vjp_with_a_tree_aux(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device)
    (out, aux), pullback = func.vjp(
        lambda d: ({"y": d["a"] * 3.0}, {"z": d["a"]}), {"a": a}, has_aux=True
    )
    _same(out, {"y": 3.0 * a})
    _same(aux, {"z": a})
    _same(pullback({"y": lucid.ones_like(a)}), ({"a": lucid.full((2,), 3.0, device=device)},))


# ── vjp cotangents, jvp tangents, linearize ──────────────────────────────────


def _two_outputs(x: Tensor) -> dict[str, Tensor]:
    return {"p": x * 2.0, "q": x**2}


def test_vjp_cotangent_has_the_structure_of_the_output(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    _, pullback = func.vjp(_two_outputs, x)
    (g,) = pullback({"p": lucid.ones(3, device=device), "q": 2.0 * lucid.ones(3, device=device)})
    _close(g, 2.0 + 4.0 * x)


def test_vjp_cotangents_may_be_given_leaf_by_leaf(device: str) -> None:
    """The form every earlier release took, and a single tensor for a
    single-leaf output."""
    x = lucid.tensor(_X, device=device)
    ones = lucid.ones(3, device=device)
    _, pullback = func.vjp(lambda a: (a * 2.0, a**2), x)
    _close(pullback(ones, ones)[0], 2.0 + 2.0 * x)
    _close(pullback((ones, ones))[0], 2.0 + 2.0 * x)
    _, single = func.vjp(lambda a: (a * 2.0,), x)
    _close(single(ones)[0], 2.0 * ones)
    _, scalar = func.vjp(lambda a: (a**2).sum(), x)
    _close(scalar()[0], 2.0 * x)


def test_vjp_refuses_a_cotangent_of_the_wrong_structure(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    _, pullback = func.vjp(_two_outputs, x)
    with pytest.raises(ValueError, match="structure"):
        pullback({"p": lucid.ones(3, device=device)})
    with pytest.raises(TypeError, match="cotangent"):
        pullback({"p": 1.0, "q": 2.0})


def test_jvp_refuses_tangents_of_the_wrong_structure(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    with pytest.raises(ValueError, match="structure"):
        func.jvp(lambda d: d["a"], ({"a": x},), ({"b": x},))
    with pytest.raises(ValueError, match="structure"):
        func.jvp(lambda d: d["a"], ({"a": x},), (x,))


def test_jvp_holds_a_non_tensor_leaf_fixed(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    y, dy = func.jvp(
        lambda d: d["a"] * d["scale"], ({"a": x, "scale": 3.0},), ({"a": x, "scale": 0.0},)
    )
    _close(y, 3.0 * x)
    _close(dy, 3.0 * x)


def test_linearize_with_trees(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    out, linear = func.linearize(lambda d: {"y": d["a"] ** 2, "w": (d["a"].sum(),)}, {"a": x})
    _same(out, {"y": x**2, "w": (x.sum(),)})
    _same(
        linear({"a": lucid.ones(3, device=device)}),
        {"y": 2.0 * x, "w": (lucid.tensor(3.0, device=device),)},
    )
    with pytest.raises(RuntimeError, match="one tangent per primal"):
        linear({"a": x}, {"a": x})


# ── vmap: in_dims / out_dims as pytrees ──────────────────────────────────────


def _vmap_cases(m: ModuleType) -> dict[str, Callable[[], object]]:
    """Calls to ``vmap`` that both frameworks accept, by what they exercise."""
    x = _tensor(m, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], "cpu")
    w = _tensor(m, [1.0, 2.0], "cpu")
    v = m.func.vmap
    return {
        "dict-arg": lambda: v(lambda d: d["x"].sum())({"x": x}),
        "list-arg": lambda: v(lambda l: l[0].sum() + l[1].sum())([x, x * 2.0]),
        "in_dims-prefix": lambda: v(
            lambda d, y: d["x"] * d["w"] + y.sum(), in_dims=({"x": 0, "w": None}, 1)
        )({"x": x, "w": w}, x.T),
        "in_dims-int-for-all": lambda: v(lambda d: d["x"] + d["y"], in_dims=1)(
            {"x": x.T, "y": x.T}
        ),
        "in_dims-list": lambda: v(lambda a, b: a + b[0] + b[1], in_dims=(0, [0, None]))(x, [x, w]),
        "out-dict": lambda: v(lambda a: {"o": a * 2.0, "t": (a.sum(), a)})(x),
        "out_dims-tuple": lambda: v(lambda a: (a, a * 2.0), out_dims=(1, 0))(x),
        "out_dims-dict": lambda: v(lambda a: {"p": a, "q": a * 2.0}, out_dims={"p": 1, "q": 0})(x),
        "out_dims-nested": lambda: v(lambda a: (a, {"k": a * 2.0}), out_dims=(1, {"k": 0}))(x),
        "kwarg-unbatched": lambda: v(lambda a, k: a + k)(x, k=w),
        "chunk-dict": lambda: v(lambda d: {"s": d["x"] * 2.0}, chunk_size=2)({"x": x}),
        "chunk-tuple-out": lambda: v(lambda a: (a, a.sum()), chunk_size=2)(x),
        "unbatched-arg": lambda: v(lambda a, b: a * b, in_dims=(0, None))(x, w),
    }


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_vmap_cases(lucid)))
def test_vmap_pytree_dims_match_reference(ref: ModuleType, case: str) -> None:
    got = _vmap_cases(lucid)[case]()
    want = _vmap_cases(ref)[case]()
    _agree(_python(got), _python(want))


@pytest.mark.parametrize("chunk", [None, 2])
@pytest.mark.parametrize("strategy", ["auto", "isolated", "vectorized"])
def test_vmap_strategies_agree_on_trees(device: str, strategy: str, chunk: int | None) -> None:
    x = lucid.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], device=device)
    f = lambda d: {"o": d["a"] * 2.0, "t": (d["b"][0] + 1.0,)}  # noqa: E731
    got = func.vmap(f, strategy=strategy, chunk_size=chunk)({"a": x, "b": [x]})
    _same(got, {"o": 2.0 * x, "t": (x + 1.0,)})


def test_vmap_over_a_tree_of_one_batch_size_checks_all_leaves(device: str) -> None:
    a = lucid.zeros(3, 2, device=device)
    b = lucid.zeros(4, 2, device=device)
    with pytest.raises(ValueError, match="inconsistent batch sizes"):
        func.vmap(lambda d: d["a"] + d["b"])({"a": a, "b": b})


def test_vmap_refuses_a_non_tensor_it_was_asked_to_batch(device: str) -> None:
    """Passing it through unbatched is the silent wrong answer."""
    x = lucid.zeros(3, 2, device=device)
    with pytest.raises(TypeError, match="only Tensors can be batched"):
        func.vmap(lambda d: d["x"].sum())({"x": x, "tag": "t"})
    assert (
        float(
            func.vmap(lambda d: d["x"].sum(), in_dims=({"x": 0, "tag": None},))(
                {"x": x, "tag": "t"}
            )
            .sum()
            .item()
        )
        == 0.0
    )


def test_vmap_refuses_in_dims_that_do_not_fit_the_arguments(device: str) -> None:
    x = lucid.zeros(3, 2, device=device)
    with pytest.raises(ValueError, match="cannot be given a prefix"):
        func.vmap(lambda d: d["x"].sum(), in_dims=({"y": 0},))({"x": x})
    with pytest.raises(TypeError, match="ints or None"):
        func.vmap(lambda d: d["x"].sum(), in_dims=({"x": "0"},))({"x": x})


def test_vmap_refuses_results_whose_structure_differs_between_examples(device: str) -> None:
    x = lucid.arange(4.0, device=device).reshape(2, 2)

    def f(a: Tensor) -> object:
        return (a, a) if float(a[0].item()) == 0.0 else a

    with pytest.raises(ValueError, match="for one batch element and"):
        func.vmap(f)(x)


def test_vmap_refuses_a_result_that_is_not_made_of_tensors(device: str) -> None:
    x = lucid.arange(4.0, device=device).reshape(2, 2)
    with pytest.raises(TypeError, match="func must return Tensors"):
        func.vmap(lambda a: {"k": 3.0})(x)


def test_vmap_out_dims_without_a_batched_output_axis_is_refused(device: str) -> None:
    x = lucid.arange(4.0, device=device).reshape(2, 2)
    with pytest.raises(TypeError, match="out_dims must hold ints"):
        func.vmap(lambda a: a, out_dims=None)(x)


def test_vmap_a_one_tuple_out_dims_for_a_single_tensor(device: str) -> None:
    x = lucid.arange(6.0, device=device).reshape(3, 2)
    _close(func.vmap(lambda a: a, out_dims=(1,))(x), x.T)


# ── what a transform refuses ─────────────────────────────────────────────────


def test_grad_refuses_a_leaf_that_is_not_a_tensor(device: str) -> None:
    a = lucid.tensor(_X, device=device)
    with pytest.raises(TypeError, match="pytree of Tensors"):
        func.grad(lambda d: (d["a"] ** 2).sum())({"a": a, "n": 2})
    with pytest.raises(TypeError, match="pytree of Tensors"):
        func.jacfwd(lambda d: d["a"])({"a": a, "n": 2})


def test_grad_leaves_the_arguments_it_does_not_differentiate_alone(device: str) -> None:
    a = lucid.tensor(_X, device=device)
    got = func.grad(lambda x, cfg: (x ** cfg["p"]).sum())(a, {"p": 2})
    _close(got, 2.0 * a)


def test_argnums_out_of_range_is_an_index_error(device: str) -> None:
    a = lucid.tensor(_X, device=device)
    with pytest.raises(IndexError, match="argnums 2 is out of range"):
        func.grad(lambda x: x.sum(), argnums=2)(a)
    _close(func.grad(lambda x, y: (x * y).sum(), argnums=-1)(a, a), a)


def test_a_transform_that_needs_one_tensor_refuses_anything_else(device: str) -> None:
    a = lucid.tensor(_X, device=device)
    with pytest.raises(TypeError, match="single Tensor, got a tuple of 2"):
        func.grad(lambda x: (x.sum(), x.sum()))(a)
    with pytest.raises(TypeError, match="single Tensor, got float"):
        func.grad(lambda x: 1.0)(a)
    with pytest.raises(TypeError, match="Tensor or a pytree of Tensors, found a float"):
        func.jacrev(lambda x: {"k": 1.0})(a)


def test_has_aux_needs_a_pair(device: str) -> None:
    a = lucid.tensor(_X, device=device)
    with pytest.raises(ValueError, match="has_aux=True requires"):
        func.grad(lambda x: (x**2).sum(), has_aux=True)(a)


def test_an_empty_tree_has_nothing_to_differentiate(device: str) -> None:
    got = func.grad(lambda d, x: (x**2).sum(), argnums=0)({}, lucid.tensor(_X, device=device))
    assert got == {}


def test_transforms_are_isolated_for_vmap() -> None:
    from lucid.func._transform import Isolated

    f = lambda x: (x**2).sum()  # noqa: E731
    for t in (func.grad(f), func.jacrev(f), func.jacfwd(f), func.hessian(f), func.vmap(f)):
        assert isinstance(t, Isolated)
    _, pullback = func.vjp(f, lucid.tensor(_X))
    assert isinstance(pullback, Isolated)


def test_vmap_with_nothing_to_batch_just_calls_func(device: str) -> None:
    w = lucid.tensor(_X, device=device)
    got = func.vmap(lambda a, b: a * b, in_dims=(None, None))(w, w)
    _close(got, w * w)


def test_jvp_needs_a_tangent_per_primal(device: str) -> None:
    x = lucid.tensor(_X, device=device)
    with pytest.raises(RuntimeError, match="one tangent per primal"):
        func.jvp(lambda a, b: a * b, (x, x), (x,))


def test_jacrev_refuses_a_chunk_size_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        func.jacrev(lambda a: a, chunk_size=0)
