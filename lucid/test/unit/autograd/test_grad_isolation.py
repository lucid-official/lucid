"""``lucid.autograd.grad`` must not touch ``.grad`` on anything.

The documented contract is that ``grad`` returns cotangents and leaves every
leaf's gradient state alone.  It used to hold only for the tensors named in
``inputs``: the implementation ran a full backward and restored ``.grad``
afterwards, so every *other* leaf kept whatever the traversal deposited.  That
is silent corruption in exactly the workflows ``grad`` exists for — a
meta-learning inner loop asks for a subset of parameters and the rest are
quietly poisoned before the next optimizer step.
"""

import math

import pytest

import lucid


class TestGradLeavesGradAlone:
    def test_requested_input_grad_stays_none(self) -> None:
        x = lucid.tensor([3.0], requires_grad=True)
        lucid.autograd.grad((x * x).sum(), [x])
        assert x.grad is None

    def test_unrequested_leaf_grad_stays_none(self) -> None:
        # The regression: k is in the graph but was not asked for.
        k = lucid.tensor([2.0], requires_grad=True)
        x = lucid.tensor([3.0], requires_grad=True)
        lucid.autograd.grad(k * x, [x], grad_outputs=[lucid.tensor([1.0])])
        assert k.grad is None
        assert x.grad is None

    def test_preexisting_grads_survive_unchanged(self) -> None:
        k = lucid.tensor([2.0], requires_grad=True)
        x = lucid.tensor([3.0], requires_grad=True)
        (k * x).sum().backward()
        before_k, before_x = k.grad.tolist(), x.grad.tolist()  # type: ignore[union-attr]

        lucid.autograd.grad((k * x).sum(), [x])

        assert k.grad.tolist() == before_k  # type: ignore[union-attr]
        assert x.grad.tolist() == before_x  # type: ignore[union-attr]

    def test_repeated_calls_do_not_accumulate(self) -> None:
        # Accumulation would show up as a growing .grad on the untouched leaf.
        k = lucid.tensor([2.0], requires_grad=True)
        x = lucid.tensor([3.0], requires_grad=True)
        for _ in range(5):
            lucid.autograd.grad((k * x).sum(), [x])
        assert k.grad is None

    def test_a_subset_of_parameters_leaves_the_rest_clean(self) -> None:
        # The meta-learning shape: differentiate w.r.t. some parameters only.
        params = [lucid.tensor([float(i + 1)], requires_grad=True) for i in range(4)]
        loss = sum((p * p for p in params), lucid.tensor([0.0])).sum()
        got = lucid.autograd.grad(loss, [params[0], params[2]])

        assert got[0].tolist() == pytest.approx([2.0])  # type: ignore[union-attr]
        assert got[1].tolist() == pytest.approx([6.0])  # type: ignore[union-attr]
        assert all(p.grad is None for p in params)


class TestGradStillCorrect:
    def test_matches_backward(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad((x * x).sum(), [x])

        y = lucid.tensor([1.0, 2.0, 3.0], dtype=lucid.float64, requires_grad=True)
        (y * y).sum().backward()
        assert g.tolist() == pytest.approx(y.grad.tolist())  # type: ignore[union-attr]

    def test_higher_order(self) -> None:
        x = lucid.tensor([3.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad((x**3).sum(), [x], create_graph=True)
        assert g.tolist() == pytest.approx([27.0])
        (gg,) = lucid.autograd.grad(g.sum(), [x])
        assert gg.tolist() == pytest.approx([18.0])
        assert x.grad is None

    def test_accumulates_across_branches(self) -> None:
        x = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad((x * x).sum(), [x])
        assert g.tolist() == pytest.approx([4.0])

    def test_multiple_outputs_sum(self) -> None:
        a = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad(
            [a * a, a * 3], [a], grad_outputs=[lucid.tensor([1.0]), lucid.tensor([1.0])]
        )
        assert g.tolist() == pytest.approx([7.0])

    def test_interior_tensor(self) -> None:
        x = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        mid = x * 3
        g_mid, g_x = lucid.autograd.grad((mid * mid).sum(), [mid, x])
        assert g_mid.tolist() == pytest.approx([12.0])  # type: ignore[union-attr]
        assert g_x.tolist() == pytest.approx([36.0])  # type: ignore[union-attr]

    def test_explicit_seed(self) -> None:
        x = lucid.tensor([1.0, 2.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad(
            x * x, [x], grad_outputs=[lucid.tensor([1.0, 10.0], dtype=lucid.float64)]
        )
        assert g.tolist() == pytest.approx([2.0, 40.0])

    def test_allow_unused(self) -> None:
        u = lucid.tensor([1.0], requires_grad=True)
        v = lucid.tensor([1.0], requires_grad=True)
        got = lucid.autograd.grad((u * u).sum(), [u, v], allow_unused=True)
        assert got[0] is not None
        assert got[1] is None

    def test_unreachable_input_raises_without_allow_unused(self) -> None:
        u = lucid.tensor([1.0], requires_grad=True)
        v = lucid.tensor([1.0], requires_grad=True)
        with pytest.raises(RuntimeError, match="not reachable"):
            lucid.autograd.grad((u * u).sum(), [u, v])

    def test_scalar_output_needs_no_seed(self) -> None:
        x = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        (g,) = lucid.autograd.grad(lucid.exp(x).sum(), [x])
        assert g.tolist() == pytest.approx([math.exp(2.0)])


class TestSeedUnderCreateGraph:
    """``grad_outputs`` is part of the graph ``create_graph`` builds.

    The seed used to be detached unconditionally, so ``J^T u`` came back with
    no path to ``u``: the double-backward form of a JVP, or any seed computed
    from a parameter, differentiated to ``None`` — and with
    ``allow_unused=True`` that read as a silent zero.
    """

    def test_seed_is_differentiable(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        w = lucid.tensor([0.5, -1.0, 2.0], device=device, requires_grad=True)
        (g,) = lucid.autograd.grad(x**3, [x], grad_outputs=[w * 2.0], create_graph=True)
        assert g.tolist() == pytest.approx([3.0, -24.0, 108.0])
        # d/dw sum(3x² · 2w) = 6x²
        (gw,) = lucid.autograd.grad(g.sum(), [w])
        assert gw.tolist() == pytest.approx([6.0, 24.0, 54.0])
        assert x.grad is None and w.grad is None

    def test_seed_stays_detached_without_create_graph(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        w = lucid.tensor([3.0, 4.0], device=device, requires_grad=True)
        (g,) = lucid.autograd.grad(x * x, [x], grad_outputs=[w])
        assert g.tolist() == pytest.approx([6.0, 16.0])
        assert not g.requires_grad
