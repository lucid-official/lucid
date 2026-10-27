"""A second pass over a graph is refused only when the first freed what it reads (LCD-350).

A pass without ``retain_graph`` frees what every node it ran saved for
backward, and a later pass must not read that memory.  Whether a node's
release freed anything was read off the capacity of its version list, which
every kernel filled — including ``add``, ``sum`` and the views, which save
nothing — so ``(a + 1).sum()`` refused a second ``backward()`` the reference
runs.  Each node now records what its release freed: values its eager
backward reads (refused by every pass), or only the tensors graph-mode
backward reads (refused by a ``create_graph`` pass).  Values count as saved
whatever their size, and whether they sit in the node's slots or in members
of its own (a mask, an index), which the node declares.  The root is cut from
its graph only when something an eager pass reads was freed in it.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Build = Callable[[ModuleType, object], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 5) for v in t.detach().cpu().reshape(-1).tolist()]


def _outcome(m: ModuleType, device: str, scenario: Callable[[ModuleType, str], object]) -> object:
    try:
        return {"value": scenario(m, device)}
    except RuntimeError as e:
        return {"second_pass": "second time" in str(e).lower()}


# Graphs that save no value their backward reads.
_SAVE_NOTHING: dict[str, Build] = {
    "add-scalar": lambda m, x: (x + 1.0).sum(),
    "sub-scalar": lambda m, x: (x - 2.0).mean(),
    "neg": lambda m, x: (-x).sum(),
    "add-tensors": lambda m, x: (x + x).sum(),
    "reshape-transpose": lambda m, x: x.reshape(3, 2).transpose(0, 1).sum(),
    "permute": lambda m, x: x.reshape(3, 2).permute(1, 0).sum(),
    "expand": lambda m, x: x.reshape(1, 6).expand(2, 6).sum(),
    "slice": lambda m, x: x[1:4].sum(),
    "as-strided": lambda m, x: x.as_strided((2,), (2,)).sum(),
    "cat": lambda m, x: m.cat([x, x - 1.0], 0).mean(),
    "stack": lambda m, x: m.stack([x, x + 1.0], 0).sum(),
    "flip": lambda m, x: m.flip(x, [0]).sum(),
    "pad": lambda m, x: m.nn.functional.pad(x, (1, 2)).sum(),
}


def _index(m: ModuleType, device: object) -> object:
    return m.tensor([0, 2, 2], device=device)


# Graphs with a node whose backward reads a value it saved — in its slots, or
# in a member of its own (a mask, an index), which only that node can say.
_SAVE_VALUES: dict[str, Build] = {
    "mul": lambda m, x: (x * x).sum(),
    "exp": lambda m, x: x.exp().sum(),
    "sin-after-add": lambda m, x: (x + 1.0).sin().sum(),
    "matmul": lambda m, x: (x.reshape(2, 3) @ x.reshape(3, 2)).sum(),
    "where": lambda m, x: m.where(x > 2, x, -x).sum(),
    "masked-fill": lambda m, x: x.masked_fill(x > 2, 0.0).sum(),
    "gather": lambda m, x: m.gather(x, 0, _index(m, x.device)).sum(),
    "index-select": lambda m, x: m.index_select(x, 0, _index(m, x.device)).sum(),
    "advanced-indexing": lambda m, x: x[_index(m, x.device)].sum(),
    "sort": lambda m, x: m.sort(x)[0].sum(),
    "topk": lambda m, x: m.topk(x, 2)[0].sum(),
    "max-pool1d": lambda m, x: m.nn.functional.max_pool1d(x.reshape(1, 1, 6), 2).sum(),
    "embedding": lambda m, x: m.nn.functional.embedding(
        m.tensor([0, 2], device=x.device), x.reshape(3, 2)
    ).sum(),
}

# Graphs whose saving node saved a tensor of no elements: still saved, and
# freed by the first pass.
_SAVE_EMPTY: dict[str, Build] = {
    "mul": lambda m, x: (x * x).sum(),
    "exp": lambda m, x: x.exp().sum(),
    "div": lambda m, x: (x / (x + 1.0)).sum(),
    "matmul": lambda m, x: (x.reshape(0, 2) @ m.ones(2, 3, device=x.device)).sum(),
    "softmax": lambda m, x: m.nn.functional.softmax(x.reshape(0, 3), dim=1).sum(),
    "log-softmax": lambda m, x: m.nn.functional.log_softmax(x.reshape(0, 3), dim=1).sum(),
}


def _leaf(m: ModuleType, device: str) -> object:
    return m.arange(6, dtype=m.float32, device=device).requires_grad_(True)


def _backward_twice(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        y.backward()
        y.backward()
        return _values(x.grad)

    return scenario


def _grad_twice(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        m.autograd.grad(y, x)
        return [_values(g) for g in m.autograd.grad(y, x)]

    return scenario


def _grad_then_backward(build: Build) -> Callable[[ModuleType, str], object]:
    def scenario(m: ModuleType, device: str) -> object:
        x = _leaf(m, device)
        y = build(m, x)
        m.autograd.grad(y, x)
        y.backward()
        return _values(x.grad)

    return scenario


_PASSES = {
    "backward-twice": _backward_twice,
    "grad-twice": _grad_twice,
    "grad-then-backward": _grad_then_backward,
}


@pytest.mark.parametrize("graph", list(_SAVE_NOTHING))
@pytest.mark.parametrize("passes", list(_PASSES))
def test_a_graph_that_saves_no_values_runs_again(
    graph: str, passes: str, ref: ModuleType, device: str
) -> None:
    scenario = _PASSES[passes](_SAVE_NOTHING[graph])
    want = _outcome(ref, "cpu", scenario)
    assert "value" in want
    assert _outcome(lucid, device, scenario) == want


@pytest.mark.parametrize("graph", list(_SAVE_VALUES))
@pytest.mark.parametrize("passes", list(_PASSES))
def test_a_graph_that_saves_values_is_still_refused(
    graph: str, passes: str, ref: ModuleType, device: str
) -> None:
    scenario = _PASSES[passes](_SAVE_VALUES[graph])
    assert _outcome(ref, "cpu", scenario) == {"second_pass": True}
    assert _outcome(lucid, device, scenario) == {"second_pass": True}


@pytest.mark.parametrize("graph", list(_SAVE_EMPTY))
def test_a_graph_that_saved_a_tensor_of_no_elements_is_still_refused(
    graph: str, ref: ModuleType, device: str
) -> None:
    # A slot holding a tensor of no elements holds no bytes, but it was
    # filled: on Metal the second pass read it as the GPU buffer the release
    # had replaced (bad_variant_access), and on the CPU it ran where the
    # reference refuses.
    def scenario(m: ModuleType, dev: str) -> object:
        x = m.zeros(0, device=dev).requires_grad_(True)
        y = _SAVE_EMPTY[graph](m, x)
        y.backward()
        y.backward()
        return _values(x.grad)

    assert _outcome(ref, "cpu", scenario) == {"second_pass": True}
    assert _outcome(lucid, device, scenario) == {"second_pass": True}


@pytest.mark.parametrize("graph", list(_SAVE_NOTHING))
def test_the_same_graph_without_a_second_pass_is_unchanged(graph: str, device: str) -> None:
    x = _leaf(lucid, device)
    _SAVE_NOTHING[graph](lucid, x).backward()
    once = _values(x.grad)
    x.grad = None
    y = _SAVE_NOTHING[graph](lucid, x)
    y.backward(retain_graph=True)
    assert _values(x.grad) == once


@pytest.mark.parametrize(
    "build",
    [
        _SAVE_NOTHING["add-scalar"],
        _SAVE_NOTHING["cat"],
        lambda m, x: m.nn.functional.linear(x.reshape(2, 3), m.ones(4, 3, device=x.device)).sum(),
        lambda m, x: m.nn.functional.mse_loss(x, m.zeros(6, device=x.device)),
    ],
    ids=["add-scalar", "cat-of-add", "linear", "mse-loss"],
)
def test_create_graph_after_a_release_refuses_rather_than_reads_freed_inputs(
    build: Build, device: str
) -> None:
    # Graph-mode backward reads the tensors each node was handed, and a
    # release frees them all, so a create_graph pass refuses any released
    # node — where it read null, or the reference answers from the values
    # it never freed (add) — rather than crash.
    x = _leaf(lucid, device)
    y = build(lucid, x)
    y.backward()
    with pytest.raises(RuntimeError, match="second time"):
        lucid.autograd.grad(y, x, create_graph=True)


def test_a_save_free_pass_keeps_the_root_attached(device: str) -> None:
    x = _leaf(lucid, device)
    y = (x + 1.0).sum()
    fn = y.grad_fn
    y.backward()
    assert y.grad_fn is not None and y.grad_fn.name() == fn.name()


def test_a_pass_that_freed_values_cuts_at_the_freed_frontier(device: str) -> None:
    x = _leaf(lucid, device)
    h = x * x
    y = h.sum()
    y.backward()
    assert y.grad_fn is not None and y.grad_fn.name() == "sum"
    assert h.grad_fn is not None and h.grad_fn.name() == "mul"
    with pytest.raises(RuntimeError, match="second time"):
        y.backward()
    with pytest.raises(RuntimeError, match="second time"):
        lucid.autograd.grad(y, x)


def test_grad_to_an_interior_tensor_behind_save_free_nodes(ref: ModuleType, device: str) -> None:
    def scenario(m: ModuleType, dev: str) -> object:
        a = m.ones(2, requires_grad=True, device=dev)
        h = a * a
        y = h.sum()
        y.backward()
        return [_values(g) for g in m.autograd.grad(y, h)]

    want = _outcome(ref, "cpu", scenario)
    assert want == {"value": [[1.0, 1.0]]}
    assert _outcome(lucid, device, scenario) == want


def test_grad_to_an_unvisited_branch_after_a_partial_grad_pass(
    ref: ModuleType, device: str
) -> None:
    def scenario(m: ModuleType, dev: str) -> object:
        x = m.tensor([2.0, 3.0], requires_grad=True, device=dev)
        w = m.tensor([4.0, 5.0], requires_grad=True, device=dev)
        y = (x * x).sum() + (w * w).sum()
        gx = [_values(g) for g in m.autograd.grad(y, [x])]
        gw = [_values(g) for g in m.autograd.grad(y, [w])]
        return gx, gw

    want = _outcome(ref, "cpu", scenario)
    assert want == {"value": ([[4.0, 6.0]], [[8.0, 10.0]])}
    assert _outcome(lucid, device, scenario) == want

    x = lucid.tensor([2.0, 3.0], requires_grad=True, device=device)
    w = lucid.tensor([4.0, 5.0], requires_grad=True, device=device)
    y = (x * x).sum() + (w * w).sum()
    lucid.autograd.grad(y, [x])
    with pytest.raises(RuntimeError, match="second time"):
        lucid.autograd.grad(y, [x])
