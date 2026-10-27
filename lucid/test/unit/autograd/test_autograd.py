"""Autograd: backward / leaf semantics / retain_graph / functional API."""

import numpy as np
import pytest

import lucid


class TestBackwardScalar:
    def test_simple(self) -> None:
        x = lucid.tensor([3.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [6.0], atol=1e-6)

    def test_chain(self) -> None:
        x = lucid.tensor([2.0], requires_grad=True)
        y = (x.exp().log()).sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], atol=1e-5)


class TestBackwardVector:
    def test_with_explicit_seed(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        seed = lucid.tensor([1.0, 1.0, 1.0])
        y.backward(seed)
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0], atol=1e-6)


class TestLeafSemantics:
    def test_leaf_with_requires_grad(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        assert x.is_leaf

    def test_non_leaf_intermediate(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        y = x * 2.0
        # ``y`` is a non-leaf result of an op.
        assert not y.is_leaf

    def test_grad_of_non_leaf_default_none(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        y = x * 2.0
        z = y.sum()
        z.backward()
        # Without retain_grad, intermediate ``y`` should not hold a grad.
        assert y.grad is None


class TestRequireGradTransitions:
    def test_requires_grad_setter(self) -> None:
        x = lucid.tensor([1.0])
        assert not x.requires_grad
        x.requires_grad = True
        assert x.requires_grad

    def test_requires_grad_method(self) -> None:
        x = lucid.tensor([1.0])
        x.requires_grad_(True)
        assert x.requires_grad


class TestDetach:
    def test_detach_breaks_graph(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        y = (x * 2.0).detach()
        assert not y.requires_grad


class TestZeroGrad:
    def test_set_grad_none(self) -> None:
        x = lucid.tensor([1.0], requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None
        x.grad = None
        assert x.grad is None


class TestMultiInput:
    def test_two_var_sum(self) -> None:
        a = lucid.tensor([1.0, 2.0], requires_grad=True)
        b = lucid.tensor([3.0, 4.0], requires_grad=True)
        ((a + b) ** 2).sum().backward()
        # d/da [(a+b)²] = 2(a+b).
        np.testing.assert_allclose(a.grad.numpy(), [8.0, 12.0], atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), [8.0, 12.0], atol=1e-5)


class TestFunctionalAutograd:
    def test_jacobian_present(self) -> None:
        # Just verify the lucid.autograd surface includes the
        # higher-order helpers — values tested under functional/.
        for name in ("backward", "grad"):
            assert hasattr(lucid.autograd, name) or hasattr(lucid, name)


class TestSecondOrder:
    def test_grad_of_grad_x_cubed(self) -> None:
        # d²/dx² [x³] = 6x; at x=2 → 12.
        x = lucid.tensor([2.0], requires_grad=True)
        y = x * x * x
        (g1,) = lucid.autograd.grad(y, x, create_graph=True)
        (g2,) = lucid.autograd.grad(g1, x)
        assert abs(g2.item() - 12.0) < 1e-4

    def test_hessian_quartic(self) -> None:
        # f(x) = sum(x^4); H_ii = 12 x_i², off-diag 0.  At [1, 1] → diag(12, 12).
        H = lucid.autograd.hessian(lambda x: (x**4).sum(), lucid.tensor([1.0, 1.0]))
        np.testing.assert_allclose(H.numpy(), [[12.0, 0.0], [0.0, 12.0]], atol=1e-3)


class TestAnomalyToggle:
    def test_set_and_query(self) -> None:
        try:
            lucid.autograd.set_detect_anomaly(True)
            assert lucid.autograd.is_anomaly_enabled()
            lucid.autograd.set_detect_anomaly(False)
            assert not lucid.autograd.is_anomaly_enabled()
        finally:
            lucid.autograd.set_detect_anomaly(False)


class TestAutogradGraph:
    def test_allow_mutation_on_saved_flag_toggles(self) -> None:
        from lucid._C import engine as _C_engine

        assert not _C_engine.is_mutation_on_saved_allowed()
        with lucid.autograd.graph.allow_mutation_on_saved_tensors():
            assert _C_engine.is_mutation_on_saved_allowed()
        assert not _C_engine.is_mutation_on_saved_allowed()

    def test_save_on_cpu_callable_stub(self) -> None:
        # Stub: must enter/exit cleanly and not affect backward correctness.
        with lucid.autograd.graph.save_on_cpu():
            x = lucid.tensor([1.0, 2.0], requires_grad=True)
            (x * x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0], atol=1e-5)


class TestCustomFunction:
    """Tests for lucid.autograd.Function (PythonBackwardNode bridge)."""

    def test_relu_forward_backward(self) -> None:
        """Custom ReLU Function: forward mask + bool-gradient promotion."""
        from lucid.autograd import Function, FunctionCtx

        class MyReLU(Function):
            @staticmethod
            def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
                ctx.save_for_backward(x)
                return lucid.relu(x)

            @staticmethod
            def backward(ctx: FunctionCtx, grad_output: lucid.Tensor) -> lucid.Tensor:
                (x,) = ctx.saved_tensors
                # (x > 0) is a bool tensor — promotion to float32 must happen
                # transparently inside the * operator.
                return grad_output * (x > 0)

        x = lucid.tensor([-1.0, 2.0, -0.5, 3.0], requires_grad=True)
        y = MyReLU.apply(x)
        np.testing.assert_allclose(y.numpy(), [0.0, 2.0, 0.0, 3.0], atol=1e-6)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0, 1.0], atol=1e-6)

    def test_scale_forward_backward(self) -> None:
        """Custom Scale Function: single learnable scalar."""
        from lucid.autograd import Function, FunctionCtx

        class Scale(Function):
            @staticmethod
            def forward(ctx: FunctionCtx, x: lucid.Tensor, s: lucid.Tensor) -> lucid.Tensor:
                ctx.save_for_backward(x, s)
                return x * s

            @staticmethod
            def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
                x, s = ctx.saved_tensors
                return grad * s, (grad * x).sum()

        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = lucid.tensor(2.0, requires_grad=True)
        y = Scale.apply(x, s)
        np.testing.assert_allclose(y.numpy(), [2.0, 4.0, 6.0], atol=1e-6)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(float(s.grad.item()), 6.0, atol=1e-6)

    def test_none_grad_passthrough(self) -> None:
        """backward returning None for a non-differentiable input is valid."""
        from lucid.autograd import Function, FunctionCtx

        class AddWithMask(Function):
            @staticmethod
            def forward(ctx: FunctionCtx, x: lucid.Tensor, mask: lucid.Tensor) -> lucid.Tensor:
                ctx.save_for_backward(mask)
                return x * mask.to(dtype=x.dtype)

            @staticmethod
            def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> tuple[lucid.Tensor, None]:
                (mask,) = ctx.saved_tensors
                return grad * mask.to(dtype=grad.dtype), None

        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask = lucid.tensor([1, 0, 1], dtype=lucid.int32)
        y = AddWithMask.apply(x, mask)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 0.0, 1.0], atol=1e-6)


class TestAutogradProfilerAlias:
    def test_alias_matches_lucid_profiler(self) -> None:
        import lucid.profiler as P

        assert lucid.autograd.profiler.profile is P.profile
        assert lucid.autograd.profiler.OpEvent is P.OpEvent

    def test_profile_context_manager_runs(self) -> None:
        with lucid.autograd.profiler.profile():
            t = lucid.tensor([1.0, 2.0])
            _ = t * t


class TestRegisterHook:
    """Tests for Tensor.register_hook."""

    def test_hook_fires_with_grad(self) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        captured: list[lucid.Tensor] = []
        x.register_hook(lambda g: captured.append(g.clone()) or None)
        (x * 2).sum().backward()
        assert len(captured) == 1
        np.testing.assert_allclose(captured[0].numpy(), [2.0, 2.0, 2.0])

    def test_hook_modifies_grad(self) -> None:
        """Hook returning a tensor replaces the gradient."""
        x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.register_hook(lambda g: g * 0.5)
        (x * 4).sum().backward()
        # raw grad = 4; hook scales by 0.5 → effective grad = 2
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])

    def test_multiple_hooks_chain(self) -> None:
        """Multiple hooks on the same tensor fire in registration order."""
        x = lucid.tensor([1.0], requires_grad=True)
        log: list[int] = []
        x.register_hook(lambda g: log.append(1) or None)
        x.register_hook(lambda g: log.append(2) or None)
        x.sum().backward()
        assert log == [1, 2]

    def test_handle_remove(self) -> None:
        """Removed hook does not fire."""
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        calls: list[int] = []
        h = x.register_hook(lambda g: calls.append(1) or None)
        h.remove()
        (x * 3).sum().backward()
        assert calls == []

    def test_handle_context_manager(self) -> None:
        """RemovableHandle works as a context manager."""
        x = lucid.tensor([1.0], requires_grad=True)
        calls: list[int] = []
        with x.register_hook(lambda g: calls.append(1) or None):
            x.sum().backward()
        # hook fires once while inside the context
        assert calls == [1]

    def test_not_requires_grad_raises(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError):
            x.register_hook(lambda g: None)


class TestCheckpoint:
    """Tests for lucid.autograd.checkpoint."""

    def test_output_matches_direct(self) -> None:
        """Checkpointed output equals non-checkpointed output."""
        import lucid.nn.functional as F
        from lucid.autograd import checkpoint

        W = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])

        def block(x: lucid.Tensor) -> lucid.Tensor:
            return F.relu(x @ W)

        lucid.manual_seed(0)
        x = lucid.randn(2, 2)
        x.requires_grad_(True)

        y_direct = block(x.detach().requires_grad_(True))
        y_ckpt = checkpoint(block, x)
        np.testing.assert_allclose(y_ckpt.numpy(), y_direct.detach().numpy(), atol=1e-5)

    def test_gradients_correct(self) -> None:
        """Gradients through checkpoint match direct backward."""
        from lucid.autograd import checkpoint

        W = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        x = lucid.tensor([[1.0, 2.0]], requires_grad=True)

        def seg(inp: lucid.Tensor) -> lucid.Tensor:
            return inp @ W

        y = checkpoint(seg, x)
        y.sum().backward()

        W_ref = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        x_ref = lucid.tensor([[1.0, 2.0]], requires_grad=True)
        (x_ref @ W_ref).sum().backward()

        np.testing.assert_allclose(W.grad.numpy(), W_ref.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(x.grad.numpy(), x_ref.grad.numpy(), atol=1e-5)

    def test_deep_model_checkpoint(self) -> None:
        """Checkpoint a 3-layer chain; gradients must be non-zero."""
        import lucid.nn as nn
        from lucid.autograd import checkpoint

        lucid.manual_seed(7)
        l1 = nn.Linear(4, 4)
        l2 = nn.Linear(4, 4)
        l3 = nn.Linear(4, 2)

        def segment(x: lucid.Tensor) -> lucid.Tensor:
            return l2(lucid.nn.functional.relu(l1(x)))

        x = lucid.randn(3, 4)
        x.requires_grad_(True)
        h = checkpoint(segment, x)
        out = l3(h).sum()
        out.backward()

        for name, p in list(l1.named_parameters()) + list(l2.named_parameters()):
            assert p.grad is not None, f"{name}.grad is None"
            assert float((p.grad * p.grad).sum().item()) > 0.0, f"{name}.grad is zero"


class TestMultiOutputFunction:
    """A ``Function`` whose ``forward`` returns several tensors.

    Each output is a separate entry point into the same node, so their
    gradients arrive at different points in the traversal.  The node has to
    wait for all of them and call ``backward`` once with the full set --
    which is why these tests care about *which* outputs were used, not only
    about the values.
    """

    @staticmethod
    def _split_class() -> type:
        from lucid.autograd import Function, FunctionCtx

        class Split(Function):
            """``x -> (2x, 3x)``, so the combined gradient is ``2ga + 3gb``."""

            @staticmethod
            def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
                ctx.save_for_backward(x)
                # Detached, as a hand-written backward normally would be:
                # if the node were skipped, these would carry no gradient at
                # all and the failure would be silent.
                return (x * 2.0).detach(), (x * 3.0).detach()

            @staticmethod
            def backward(ctx: FunctionCtx, ga: lucid.Tensor, gb: lucid.Tensor) -> lucid.Tensor:
                return ga * 2.0 + gb * 3.0

        return Split

    def test_custom_backward_actually_runs(self) -> None:
        split = self._split_class()
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        a, b = split.apply(x)

        assert a.requires_grad and b.requires_grad
        (a.sum() + b.sum()).backward()

        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0], atol=1e-6)

    def test_unused_output_contributes_zero(self) -> None:
        # Only the first output is differentiated.  The second never receives
        # a gradient, and ``backward`` must still be handed a real tensor for
        # it rather than being skipped or given ``None``.
        split = self._split_class()
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        a, _ = split.apply(x)
        a.sum().backward()

        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0], atol=1e-6)

    def test_outputs_weighted_separately(self) -> None:
        # Different coefficients on each output prove the gradients are kept
        # in their own slots rather than summed on arrival.
        split = self._split_class()
        x = lucid.tensor([1.0], requires_grad=True)
        a, b = split.apply(x)
        (a.sum() * 10.0 + b.sum()).backward()

        assert x.grad is not None
        # 2 * 10 + 3 * 1
        np.testing.assert_allclose(x.grad.numpy(), [23.0], atol=1e-6)

    def test_one_output_reaching_the_loss_by_two_paths(self) -> None:
        # A slot can be written more than once; the second arrival has to
        # accumulate onto the first, not replace it.
        split = self._split_class()
        x = lucid.tensor([1.0], requires_grad=True)
        a, _ = split.apply(x)
        (a.sum() + a.sum() * 4.0).backward()

        assert x.grad is not None
        # 2 * (1 + 4)
        np.testing.assert_allclose(x.grad.numpy(), [10.0], atol=1e-6)

    def test_single_output_is_unaffected(self) -> None:
        # The single-output path must stay on the engine's direct route.
        from lucid.autograd import Function, FunctionCtx

        class Double(Function):
            @staticmethod
            def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
                return (x * 2.0).detach()

            @staticmethod
            def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
                return g * 2.0

        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        Double.apply(x).sum().backward()

        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0], atol=1e-6)


class TestCheckpointMultiOutput:
    """``lucid.utils.checkpoint`` on a block that returns several tensors.

    This is the shape attention and recurrent layers return, and it used to
    lose gradients entirely: the outputs came back detached and ``backward``
    reported ``None`` without raising.
    """

    def test_gradients_survive(self) -> None:
        from lucid.utils.checkpoint import checkpoint

        def block(x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
            return x * 2.0, x * 3.0

        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        a, b = checkpoint(block, x)

        assert a.requires_grad and b.requires_grad
        (a.sum() + b.sum()).backward()

        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0], atol=1e-6)

    def test_matches_the_uncheckpointed_gradient(self) -> None:
        from lucid.utils.checkpoint import checkpoint

        def block(x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
            return lucid.sin(x), x * x

        raw = [0.3, -1.2, 2.0]

        direct = lucid.tensor(raw, requires_grad=True)
        a, b = block(direct)
        (a.sum() * 2.0 + b.sum()).backward()

        saved = lucid.tensor(raw, requires_grad=True)
        c, d = checkpoint(block, saved)
        (c.sum() * 2.0 + d.sum()).backward()

        assert direct.grad is not None and saved.grad is not None
        np.testing.assert_allclose(saved.grad.numpy(), direct.grad.numpy(), atol=1e-6)


class TestCheckpointNonReentrant:
    """``use_reentrant=False`` — the mode that keeps closed-over parameters
    in the graph.

    The reentrant form takes the output's ``requires_grad`` from its
    positional inputs.  Feed it constants — raw data into a first layer, a
    block behind a frozen stem — and the output carries no graph, so the
    parameters it closed over can get no gradient.  ``backward()`` used to
    return quietly on such an output and the layer simply stopped
    training; it now refuses, as the reference framework does.
    """

    @staticmethod
    def _run(use_reentrant: bool | None):
        import lucid.nn as nn
        from lucid.autograd import checkpoint

        lucid.manual_seed(0)
        layer = nn.Linear(3, 3)
        x = lucid.randn(2, 3)  # deliberately not requiring grad
        if use_reentrant is None:
            out = layer(x)
        else:
            out = checkpoint(lambda t: layer(t), x, use_reentrant=use_reentrant)
        out.sum().backward()
        grad = layer.weight.grad
        return None if grad is None else np.asarray(grad.numpy())

    def test_closed_over_parameters_still_get_gradients(self) -> None:
        direct = self._run(None)
        non_reentrant = self._run(False)
        assert non_reentrant is not None
        assert np.allclose(non_reentrant, direct, atol=1e-6)

    def test_the_reentrant_form_refuses_rather_than_training_nothing(self) -> None:
        """Its semantics are kept — the output takes ``requires_grad`` from
        the positional inputs — but the silent part is gone: backward on
        that output raises instead of leaving the parameters untouched."""
        with pytest.raises(RuntimeError, match="does not require grad"):
            self._run(True)

    def test_it_matches_the_reentrant_result_when_inputs_do_require_grad(
        self,
    ) -> None:
        from lucid.autograd import checkpoint

        def block(t):
            return (t * t).sum()

        grads = []
        for reentrant in (True, False):
            x = lucid.tensor([[1.0, -2.0]], requires_grad=True)
            checkpoint(block, x, use_reentrant=reentrant).backward()
            grads.append(np.asarray(x.grad.numpy()))
        assert np.allclose(grads[0], grads[1], atol=1e-6)
        assert np.allclose(grads[0], [[2.0, -4.0]], atol=1e-6)

    def test_the_anchor_does_not_change_the_value(self) -> None:
        """Non-reentrant mode passes an extra scalar the segment never
        reads; the output must be identical either way."""
        from lucid.autograd import checkpoint

        def block(t):
            return t * 3.0 + 1.0

        x = lucid.tensor([[1.0, -2.0]], requires_grad=True)
        a = checkpoint(block, x, use_reentrant=True)
        b = checkpoint(block, x, use_reentrant=False)
        assert np.allclose(np.asarray(a.numpy()), np.asarray(b.numpy()))

    def test_a_segment_returning_a_scalar_can_be_checkpointed(self) -> None:
        """Pre-existing, and not specific to either mode.

        ``Function`` seeds a 0-d output's gradient as shape ``(1,)`` — fine
        for an ordinary backward, which broadcasts — but ``checkpoint``
        handed that seed to ``Tensor.backward``, which checks the shape
        exactly.  Any segment ending in a reduction (a loss block, a pooled
        embedding) raised instead of running.
        """
        from lucid.autograd import checkpoint

        for reentrant in (True, False):
            x = lucid.tensor([[1.0, -2.0]], requires_grad=True)
            out = checkpoint(lambda t: (t * t).sum(), x, use_reentrant=reentrant)
            assert tuple(out.shape) == ()
            out.backward()
            assert np.allclose(np.asarray(x.grad.numpy()), [[2.0, -4.0]], atol=1e-6)
