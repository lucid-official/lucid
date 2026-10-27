"""A backward node says which tensors it holds (``Node::for_each_owned_tensor``).

The cyclic garbage collector reaches the autograd graph only through the
engine, and a tensor a node keeps for backward — the inputs graph-mode
backward reads, an index, a pivot vector — was invisible to it: a cycle
closed through one could not be collected.  Each node class reports exactly
the tensors it holds a strong reference to, and none once a pass has
released them.
"""

import lucid


def _owned(t: lucid.Tensor) -> list[object]:
    assert t.grad_fn is not None
    return list(t.grad_fn._owned_tensors())


def _same(got: list[object], want: list[lucid.Tensor]) -> bool:
    return len(got) == len(want) and all(g is w.impl for g, w in zip(got, want, strict=True))


def test_a_kernel_node_holds_its_inputs(device: str) -> None:
    a = lucid.ones(3, requires_grad=True, device=device)
    b = lucid.ones(3, requires_grad=True, device=device)
    assert _same(_owned(a * b), [a, b])
    assert _same(_owned(a + b), [a, b])
    assert _same(_owned(a.exp()), [a])


def test_a_released_node_holds_nothing(device: str) -> None:
    a = lucid.ones(3, requires_grad=True, device=device)
    y = a * a
    z = y.sum()
    z.backward()
    assert _owned(y) == []


def test_a_retained_node_still_holds_its_inputs(device: str) -> None:
    a = lucid.ones(3, requires_grad=True, device=device)
    y = a * a
    y.sum().backward(retain_graph=True)
    assert _same(_owned(y), [a, a])


def test_scatter_nodes_hold_their_index(device: str) -> None:
    base = lucid.zeros(5, requires_grad=True, device=device)
    src = lucid.ones(3, requires_grad=True, device=device)
    index = lucid.tensor([0, 2, 2], device=device)
    assert len(_owned(base.scatter_add(0, index, src))) == 1
    assert len(_owned(base.scatter(0, index, src))) == 1


def test_scatter_reduce_holds_operands_and_index_but_not_its_result(
    device: str,
) -> None:
    # The result owns the node through grad_fn; a strong hold back is a cycle
    # only a backward breaks (LCD-578).
    base = lucid.ones(5, requires_grad=True, device=device)
    src = lucid.full((3,), 2.0, requires_grad=True, device=device)
    index = lucid.tensor([0, 2, 2], device=device)
    for mode in ("prod", "amax", "amin"):
        out = base.scatter_reduce(0, index, src, reduce=mode)
        owned = _owned(out)
        assert len(owned) == 3, mode
        assert all(o is not out.impl for o in owned), mode


def test_lu_solve_holds_its_inputs_and_the_pivots() -> None:
    a = lucid.randn(3, 3, requires_grad=True)
    b = lucid.randn(3, 2, requires_grad=True)
    lu, pivots = lucid.linalg.lu_factor(a)
    owned = _owned(lucid.linalg.lu_solve(lu, pivots, b))
    assert len(owned) == 3
    assert owned[0] is lu.impl and owned[1] is b.impl


def test_a_custom_function_holds_nothing_of_its_own(device: str) -> None:
    class Twice(lucid.autograd.Function):
        @staticmethod
        def forward(ctx: object, x: lucid.Tensor) -> lucid.Tensor:
            return x * 2

        @staticmethod
        def backward(ctx: object, g: lucid.Tensor) -> lucid.Tensor:
            return g * 2

    a = lucid.ones(2, requires_grad=True, device=device)
    # What backward saved belongs to ctx, which the collector reaches itself.
    assert _owned(Twice.apply(a)) == []
