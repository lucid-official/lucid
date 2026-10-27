"""Collecting a wrapper never touches the graph it shares, and costs linear time.

The engine objects behind Python wrappers have C++ owners the garbage
collector cannot see: a tensor saved by a later op, a node on another
tensor's edges.  LCD-392's first hooks reported every Python reference
reachable from a wrapper as that wrapper's own, and cleared the C++ object
when the wrapper was found garbage (LCD-459):

* a tensor whose wrapper sat in a garbage cycle lost its ``grad_fn`` while
  a live graph had saved it, so a second derivative read ``[2, 2, 2]`` where
  the reference gives ``[10, 10, 10]``, and without the second term the
  graph was "not reachable" at all;
* every wrapper re-walked the graph behind it: 2,000 kept chained tensors
  made one collection take 2 s and 4,000 took 8 s.

A wrapper now reports and clears only what it owns alone.
"""

import gc
import time
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid


def _cycle(*objs: object) -> list[object]:
    """A list holding ``objs`` and itself: garbage once the caller drops it."""
    cycle: list[object] = [*objs]
    cycle.append(cycle)
    return cycle


def _second_derivative(z: lucid.Tensor, x: lucid.Tensor) -> list[float]:
    (g,) = lucid.autograd.grad(z, x, create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), x)
    return h.numpy().tolist()


@pytest.mark.parametrize(
    ("with_leaf_term", "expected"),
    [(True, [10.0, 10.0, 10.0]), (False, [8.0, 8.0, 8.0])],
    ids=["y-and-x-terms", "y-term-only"],
)
def test_a_stranded_tensor_wrapper_leaves_its_saved_graph_intact(
    with_leaf_term: bool, expected: list[float], device: str
) -> None:
    # z = sum((2x)^2) [+ sum(x^2)], so d/dx sum(dz/dx) = 8 [+ 2] per element.
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    y = x * 2
    z = (y * y).sum()
    if with_leaf_term:
        z = z + (x * x).sum()
    cycle = _cycle(y)
    del y, cycle
    gc.collect()
    assert _second_derivative(z, x) == expected


def test_the_second_derivative_after_a_collection_matches_the_reference(ref: ModuleType) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = (y * y).sum() + (x * x).sum()
    cycle = _cycle(y)
    del y, cycle
    gc.collect()
    x_ref = ref.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_ref = x_ref * 2
    z_ref = (y_ref * y_ref).sum() + (x_ref * x_ref).sum()
    (g_ref,) = ref.autograd.grad(z_ref, x_ref, create_graph=True)
    (h_ref,) = ref.autograd.grad(g_ref.sum(), x_ref)
    assert _second_derivative(z, x) == h_ref.tolist()


def test_a_stranded_node_wrapper_leaves_its_graph_intact(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    y = x * 2
    z = (y * y).sum() + (x * x).sum()
    cycle = _cycle(y.grad_fn, z.grad_fn)
    del cycle
    gc.collect()
    assert _second_derivative(z, x) == [10.0, 10.0, 10.0]


def test_a_stranded_function_ctx_leaves_its_graph_intact(device: str) -> None:
    class Square(lucid.autograd.Function):
        @staticmethod
        def forward(ctx: object, x: lucid.Tensor) -> lucid.Tensor:
            ctx.save_for_backward(x)  # type: ignore[attr-defined]
            ctx.tag = [1, 2, 3]  # type: ignore[attr-defined]
            return x * x

        @staticmethod
        def backward(ctx: object, grad: lucid.Tensor) -> lucid.Tensor:
            (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
            assert ctx.tag == [1, 2, 3]  # type: ignore[attr-defined]
            return grad * 2 * x

    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    y = Square.apply(x)
    assert isinstance(y, lucid.Tensor)
    gc.collect()
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.numpy().tolist() == [2.0, 4.0, 6.0]


def _collect_seconds(build: Callable[[int], list[lucid.Tensor]], n: int) -> float:
    kept = build(n)
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        gc.collect()
        best = min(best, time.perf_counter() - start)
    del kept
    return best


def _chain(n: int) -> list[lucid.Tensor]:
    kept = [lucid.randn(8, requires_grad=True)]
    for _ in range(n):
        kept.append(kept[-1] * 1.0001)
    return kept


def test_collection_time_grows_linearly_with_kept_graph_tensors() -> None:
    # Quadratic was 0.26 s at 1,000 and 3.9 s at 4,000 (15x); linear is a
    # few milliseconds either way.  The slack absorbs a loaded machine.
    #
    # LCD-640: unlike the backward-cost sibling in this directory, this one
    # is not pushed to a larger chain to shrink timer noise — measured at
    # 60,000 vs 240,000 nodes, cPython's own cyclic collector got
    # noticeably super-linear (observed ratios 3-5x over several trials,
    # against the already-generous 6x + 50 ms bound here), so a bigger
    # graph trades one flake risk for another instead of removing it.  The
    # existing sizes keep both measurements in the low milliseconds, where
    # the fixed +50 ms slack — not the 6x multiplier — is what actually
    # carries a loaded runner; best-of-5 (was 3) only adds cheap margin.
    small = _collect_seconds(_chain, 1000)
    large = _collect_seconds(_chain, 4000)
    assert large < 6 * small + 0.05, f"1000: {small:.3f} s, 4000: {large:.3f} s"
