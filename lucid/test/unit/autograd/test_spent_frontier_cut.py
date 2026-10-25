"""The cut after a pass that frees its graph is exact and linear (LCD-542).

A pass without ``retain_graph`` cuts each node whose eager-read values it
freed from the subgraph below it, and leaves a marker there that says what
that subgraph held, so a later pass past the node refuses where the
reference does (LCD-493).  Each marker kept its own copy of everything below
it, made by walking that subgraph again: a chain of N saving nodes held
N²/2 numbers, and ``y.sum().backward()`` took 1 s on a 16,000-node chain
that ``retain_graph=True`` ran in 15 ms.  The markers now share the parts
below where their subgraphs overlap, and one walk builds them.

The scenarios over overlapping frontiers are written once over a module
``m`` and run on Lucid and the reference: each marker must answer for its own
subgraph only, neither refusing a pass the reference answers nor answering
one it refuses.
"""

import gc
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.memory import live_bytes

Scenario = Callable[[ModuleType, str], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 3) for v in t.detach().cpu().reshape(-1).tolist()]


def _outcome(m: ModuleType, device: str, scenario: Scenario) -> dict[str, object]:
    try:
        return {"value": scenario(m, device)}
    except RuntimeError as e:
        return {"second_pass": "second time" in str(e).lower()}


def _grads(m: ModuleType, outputs: object, inputs: object, **kw: object) -> object:
    return [_values(g) for g in m.autograd.grad(outputs, inputs, **kw)]


# ── two saving nodes over one save-free subgraph ─────────────────────────────
# p = (x + 1) * w and q = exp(x + 1): both reach x + 1 and x, only p reaches w.


def _shared(m: ModuleType, dev: str) -> tuple[object, ...]:
    x = m.tensor([0.5, 1.0], requires_grad=True, device=dev)
    w = m.tensor([2.0, 3.0], requires_grad=True, device=dev)
    g = x + 1.0
    p = g * w
    q = g.exp()
    y = p.sum() + q.sum()
    y.backward()
    return x, w, g, p, q, y


def _shared_unreached_input(m: ModuleType, dev: str) -> object:
    _, w, _, _, q, _ = _shared(m, dev)
    return _grads(m, q.sum(), w, allow_unused=True)


def _shared_beside_a_fresh_branch(m: ModuleType, dev: str) -> object:
    _, w, _, _, q, _ = _shared(m, dev)
    return _grads(m, q.sum() + (w * 2.0).sum(), w)


def _shared_input(m: ModuleType, dev: str) -> object:
    x, _, _, _, q, _ = _shared(m, dev)
    return _grads(m, q.sum(), x)


def _shared_own_input(m: ModuleType, dev: str) -> object:
    _, w, _, p, _, _ = _shared(m, dev)
    return _grads(m, p.sum(), w)


def _shared_interior_below(m: ModuleType, dev: str) -> object:
    _, _, g, _, _, y = _shared(m, dev)
    return _grads(m, y, g)


def _shared_interior_frontier(m: ModuleType, dev: str) -> object:
    _, _, _, p, q, y = _shared(m, dev)
    return _grads(m, y, [p, q])


def _shared_frontier_and_below(m: ModuleType, dev: str) -> object:
    x, _, _, _, q, y = _shared(m, dev)
    return _grads(m, y, [q, x])


# ── a chain of saving nodes ──────────────────────────────────────────────────


def _chain(m: ModuleType, dev: str) -> tuple[list[object], object]:
    hs = [m.tensor([1.0, 2.0], requires_grad=True, device=dev)]
    for _ in range(64):
        hs.append(hs[-1] * 1.01)
    y = hs[-1].sum()
    y.backward()
    return hs, y


def _chain_last(m: ModuleType, dev: str) -> object:
    hs, y = _chain(m, dev)
    return _grads(m, y, hs[-1])


def _chain_middle(m: ModuleType, dev: str) -> object:
    hs, y = _chain(m, dev)
    return _grads(m, y, hs[32], allow_unused=True)


def _chain_leaf(m: ModuleType, dev: str) -> object:
    hs, y = _chain(m, dev)
    return _grads(m, y, hs[0], allow_unused=True)


def _chain_new_loss_at_a_spent_tensor(m: ModuleType, dev: str) -> object:
    hs, _ = _chain(m, dev)
    return _grads(m, (hs[32] * 2.0).sum(), hs[32])


def _chain_new_loss_past_a_spent_tensor(m: ModuleType, dev: str) -> object:
    hs, _ = _chain(m, dev)
    return _grads(m, (hs[32] * 2.0).sum(), hs[16], allow_unused=True)


# ── a later pass that frees a node over an earlier cut ───────────────────────


def _recut(m: ModuleType, dev: str) -> tuple[object, ...]:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=dev)
    h = a * a
    h.sum().backward()
    z = h.exp().sum()
    first = _grads(m, z, h)
    return a, h, z, first


def _recut_first(m: ModuleType, dev: str) -> object:
    return _recut(m, dev)[3]


def _recut_again(m: ModuleType, dev: str) -> object:
    _, h, z, _ = _recut(m, dev)
    return _grads(m, z, h)


def _recut_leaf(m: ModuleType, dev: str) -> object:
    a, _, z, _ = _recut(m, dev)
    return _grads(m, z, a, allow_unused=True)


def _recut_unrelated(m: ModuleType, dev: str) -> object:
    _, _, z, _ = _recut(m, dev)
    w = m.ones(2, requires_grad=True, device=dev)
    return _grads(m, (w * z).sum(), w)


# ── a diamond of saving nodes ────────────────────────────────────────────────


def _diamond(m: ModuleType, dev: str) -> tuple[object, ...]:
    x = m.tensor([0.1, 0.2], requires_grad=True, device=dev)
    a = x.exp()
    b = a * a
    c = a.sin()
    y = b.sum() + c.sum()
    y.backward()
    return x, a, b, c, y


def _diamond_top(m: ModuleType, dev: str) -> object:
    _, _, b, c, y = _diamond(m, dev)
    return _grads(m, y, [b, c])


def _diamond_shared(m: ModuleType, dev: str) -> object:
    _, a, _, _, y = _diamond(m, dev)
    return _grads(m, y, a)


def _diamond_one_side(m: ModuleType, dev: str) -> object:
    x, _, _, c, _ = _diamond(m, dev)
    return _grads(m, c.sum(), x)


_REFUSED = {"second_pass": True}

# Without the reference installed, what the reference answers.
_EXPECTED: dict[str, tuple[Scenario, dict[str, object]]] = {
    "shared-unreached-input": (_shared_unreached_input, {"value": [None]}),
    "shared-beside-a-fresh-branch": (
        _shared_beside_a_fresh_branch,
        {"value": [[2.0, 2.0]]},
    ),
    "shared-input": (_shared_input, _REFUSED),
    "shared-own-input": (_shared_own_input, _REFUSED),
    "shared-interior-below": (_shared_interior_below, _REFUSED),
    "shared-interior-frontier": (
        _shared_interior_frontier,
        {"value": [[1.0, 1.0], [1.0, 1.0]]},
    ),
    "shared-frontier-and-below": (_shared_frontier_and_below, _REFUSED),
    "chain-last": (_chain_last, {"value": [[1.0, 1.0]]}),
    "chain-middle": (_chain_middle, _REFUSED),
    "chain-leaf": (_chain_leaf, _REFUSED),
    "chain-new-loss-at-a-spent-tensor": (
        _chain_new_loss_at_a_spent_tensor,
        {"value": [[2.0, 2.0]]},
    ),
    "chain-new-loss-past-a-spent-tensor": (
        _chain_new_loss_past_a_spent_tensor,
        _REFUSED,
    ),
    "recut-first": (_recut_first, {"value": [[2.718, 54.598]]}),
    "recut-again": (_recut_again, _REFUSED),
    "recut-leaf": (_recut_leaf, _REFUSED),
    "recut-unrelated": (_recut_unrelated, {"value": [[57.316, 57.316]]}),
    "diamond-top": (_diamond_top, {"value": [[1.0, 1.0], [1.0, 1.0]]}),
    "diamond-shared": (_diamond_shared, _REFUSED),
    "diamond-one-side": (_diamond_one_side, _REFUSED),
}


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    scenario = _EXPECTED[name][0]
    assert _outcome(lucid, device, scenario) == _outcome(ref, "cpu", scenario)


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_the_outcome(name: str, device: str) -> None:
    scenario, expected = _EXPECTED[name]
    assert _outcome(lucid, device, scenario) == expected


# ── what the pass frees ──────────────────────────────────────────────────────

_ELEMS = 1 << 15  # 128 KiB of float32 per saved tensor: far above the noise
_STEPS = 8


@pytest.mark.parametrize("retain", [False, True], ids=["default", "retain-graph"])
def test_a_pass_that_does_not_retain_its_graph_frees_what_it_saved(
    retain: bool, device: str
) -> None:
    x = lucid.randn(_ELEMS, requires_grad=True, device=device)
    y = x
    for _ in range(_STEPS):
        y = y * 1.0001
    loss = y.sum()
    lucid.eval(loss)
    gc.collect()
    held = live_bytes(device)
    loss.backward(retain_graph=retain)
    lucid.eval(x.grad)
    gc.collect()
    freed = held - live_bytes(device)
    # Seven intermediates are held only by the nodes that saved them; the
    # pass adds x.grad.
    buffer = _ELEMS * 4
    if retain:
        assert freed <= 0, f"retain_graph=True freed {freed} bytes"
    else:
        assert freed >= 5 * buffer, f"freed {freed} bytes, saved {7 * buffer}"


# ── how the cut scales ───────────────────────────────────────────────────────


def _chain_loss(n: int) -> lucid.Tensor:
    y = lucid.randn(64, requires_grad=True)
    for _ in range(n):
        y = y * 1.0001
    return y.sum()


# A saving node between every two save-free ones: the cut walks through
# nodes it keeps wired as well as the ones it cuts.
def _affine_chain_loss(n: int) -> lucid.Tensor:
    y = lucid.randn(64, dtype=lucid.float64, requires_grad=True)
    for _ in range(n):
        y = y * 0.999 + 0.001
    return y.sum()


def _gru_loss(steps: int) -> lucid.Tensor:
    lucid.manual_seed(0)
    cell = nn.GRUCell(8, 16)
    xs = lucid.randn(steps, 4, 8)
    h = lucid.zeros(4, 16)
    for t in range(steps):
        h = cell(xs[t], h)
    return h.sum()


def _backward_seconds(
    build: Callable[[int], lucid.Tensor], size: int, retain: bool
) -> float:
    best = float("inf")
    for _ in range(5):
        loss = build(size)
        gc.collect()
        start = time.perf_counter()
        loss.backward(retain_graph=retain)
        best = min(best, time.perf_counter() - start)
        del loss
    return best


@pytest.mark.parametrize("retain", [False, True], ids=["default", "retain-graph"])
@pytest.mark.parametrize(
    ("build", "size"),
    [(_chain_loss, 20000), (_affine_chain_loss, 15000), (_gru_loss, 1200)],
    ids=["mul-chain", "affine-f64-chain", "gru-cell-bptt"],
)
def test_a_backward_is_linear_in_the_graph(
    build: Callable[[int], lucid.Tensor], size: int, retain: bool
) -> None:
    # The cut was quadratic: 16 ms at a 2,000-node chain and 62 ms at
    # 4,000, 0.33 s and 1.35 s at 4,000 and 8,000 steps of y * 0.999 +
    # 0.001, and a GRUCell unrolled 500 steps took 0.6 s where
    # retain_graph=True took 8 ms.  Every pass also ran a fusion pass that
    # fused nothing and erased from the middle of a vector the size of the
    # graph.  Doubling a linear pass doubles its time; the slack absorbs a
    # loaded machine.  Both are host-side work, the same on every device.
    #
    # LCD-640: at the old sizes ``small`` sat at 1-5 ms, where fixed
    # scheduler/timer noise (worth a millisecond or so, independent of the
    # graph) is a sizeable fraction of what is measured — a shared CI
    # runner pushed the ratio over the bound by about a millisecond.  Sizes
    # are picked so ``small`` is tens of milliseconds even for the cheap
    # retain_graph=True path, making that fixed noise a small fraction of
    # either measurement, and ``best`` is now of 5 repeats, not 3, so a
    # single stalled rep cannot stand in for the fastest one.
    small = _backward_seconds(build, size, retain)
    large = _backward_seconds(build, 2 * size, retain)
    assert (
        large < 3 * small + 0.01
    ), f"{size}: {small:.4f} s, {2 * size}: {large:.4f} s"


_DEEP = """
import threading

import lucid


def run():
    y = lucid.randn(4, requires_grad=True)
    for _ in range(20000):
        y = y * 1.0001
    loss = y.sum()
    del y
    loss.backward()
    del loss
    print("survived", flush=True)


# A small stack: what the cut leaves must be released by a loop, as the
# cut itself is, not by a recursion as deep as the graph.
threading.stack_size(256 * 1024)
worker = threading.Thread(target=run)
worker.start()
worker.join()
"""


def test_what_a_long_chain_leaves_is_released_without_recursion() -> None:
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(_DEEP)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0 and "survived" in proc.stdout, (
        f"exit {proc.returncode} (negative: killed by that signal)\n"
        f"{proc.stderr[-3000:]}"
    )
