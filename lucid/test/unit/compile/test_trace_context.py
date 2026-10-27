"""Tests for ``lucid.compile._core.trace_context`` and Card E1 contracts.

Covers:
1. ``compile_attempt`` snapshots and restores the default RNG stream and live
   :class:`~lucid.Generator` instances in LIFO order across nested attempts.
2. ``ensure_attention_workaround_probed`` uses an isolated ``Generator(0)`` and
   leaves the default RNG stream and live ``Generator`` states unchanged.
3. ``recording`` keeps the caller's grad mode by default (or honours an explicit
   ``grad_mode``), exposes ``recording_active() == True`` inside the block, and
   scans ``retained_values`` on exit for tensor gradient hooks
   (``register_hook``) and intermediate ``retain_grad()`` calls, marking the
   tracer unsupported.
4. ``is_tensor_hook_error`` / ``notice_if_tensor_hook_error`` classify the
   ``register_hook`` error raised on an ungrad tensor.
5. ``compiled_step`` records in the caller's grad mode, no longer accepts
   ``use_grad_mode``, advances RNG streams once on a successful step (and zero
   times when compilation raises), refuses an ``is_grad_enabled()``-gated STE
   with :class:`CustomBackwardInTrace`, compiles a ``Function``-less
   ``is_grad_enabled()`` / ``requires_grad`` branch with gradients matching
   eager, and refuses a ``requires_grad``-guarded ``register_hook`` or
   ``retain_grad()``.
6. ``diagnose`` records in the caller's grad mode without perturbing RNG streams.
"""

import copy
import inspect
from typing import override

import numpy as np
import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.compile._core import attention_probe
from lucid.compile._core.custom_backward import CustomBackwardInTrace
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._core.trace_context import (
    compile_attempt,
    is_tensor_hook_error,
    notice_if_tensor_hook_error,
    recording,
    recording_active,
)
from lucid.compile._debug.diagnose import diagnose
from lucid.compile._entry.function import compiled_step
import lucid.nn as nn


@pytest.fixture(autouse=True)
def _clear_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LUCID_COMPILE_VERBOSE`` left by earlier matrix runs does not suppress warnings."""
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)


class _RoundSTE(Function):
    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        return x.round()

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor]:
        return (grad_out,)


def test_compile_attempt_restores_default_and_generator_streams_lifo() -> None:
    lucid.manual_seed(1234)
    gen = lucid.Generator(5678)
    s0 = lucid.get_rng_state().numpy().copy()
    g0 = gen.get_state().numpy().copy()

    with compile_attempt():
        _ = lucid.randn(4)
        _ = lucid.randn(4, generator=gen)
        s1 = lucid.get_rng_state().numpy().copy()
        g1 = gen.get_state().numpy().copy()
        assert not np.array_equal(s0, s1)
        assert not np.array_equal(g0, g1)

        with compile_attempt():
            _ = lucid.randn(4)
            _ = lucid.randn(4, generator=gen)
            assert not np.array_equal(s1, lucid.get_rng_state().numpy())
            assert not np.array_equal(g1, gen.get_state().numpy())

        np.testing.assert_array_equal(lucid.get_rng_state().numpy(), s1)
        np.testing.assert_array_equal(gen.get_state().numpy(), g1)

    np.testing.assert_array_equal(lucid.get_rng_state().numpy(), s0)
    np.testing.assert_array_equal(gen.get_state().numpy(), g0)


def test_attention_probe_leaves_rng_streams_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_probe, "_probed", False)
    lucid.manual_seed(2026)
    gen = lucid.Generator(99)
    _ = lucid.randn(3)
    _ = lucid.randn(3, generator=gen)
    before_default = lucid.get_rng_state().numpy().copy()
    before_gen = gen.get_state().numpy().copy()

    attention_probe.ensure_attention_workaround_probed()

    np.testing.assert_array_equal(lucid.get_rng_state().numpy(), before_default)
    np.testing.assert_array_equal(gen.get_state().numpy(), before_gen)


def test_recording_preserves_caller_grad_mode_and_respects_override() -> None:
    assert not recording_active()
    with lucid.enable_grad():
        with recording() as tracer:
            assert recording_active()
            assert lucid.is_grad_enabled()
            assert not tracer.graph.unsupported
        with recording(grad_mode=False):
            assert not lucid.is_grad_enabled()
    with lucid.no_grad():
        with recording():
            assert not lucid.is_grad_enabled()
        with recording(grad_mode=True):
            assert lucid.is_grad_enabled()
    assert not recording_active()


def test_recording_scans_retained_values_for_hooks_and_retain_grad() -> None:
    x = lucid.randn(2, 3, requires_grad=True)
    with recording(watch=(x,)) as tracer:
        h = x * 2.0
        if h.requires_grad:
            h.register_hook(lambda g: g * 2.0)
        _ = h + 1.0
    assert tracer.graph.unsupported
    assert "register_hook" in tracer.graph.unsupported

    with recording(watch=(x,)) as tracer2:
        h2 = x * 3.0
        h2.retain_grad()
        _ = h2 + 1.0
    assert tracer2.graph.unsupported
    assert "retain_grad" in tracer2.graph.unsupported


def test_tensor_hook_error_helpers() -> None:
    x = lucid.randn(2, 3)
    with pytest.raises(RuntimeError) as exc_info:
        x.register_hook(lambda g: g)
    err = exc_info.value
    assert is_tensor_hook_error(err)
    assert not is_tensor_hook_error(RuntimeError("unrelated failure"))
    _NOTICED.discard("tensor-hook")
    with pytest.warns(CompileFallbackWarning, match="register_hook"):
        assert notice_if_tensor_hook_error(err)
    assert not notice_if_tensor_hook_error(RuntimeError("unrelated failure"))


def test_compiled_step_signature_drops_use_grad_mode() -> None:
    sig = inspect.signature(compiled_step)
    assert list(sig.parameters) == ["model", "x", "loss_fn"]


def test_compiled_step_refuses_grad_gated_ste_and_restores_rng() -> None:
    class _GatedSTEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            noise = lucid.randn(x.shape, device=x.device)
            h = self.l1(x + noise)
            if lucid.is_grad_enabled():
                h = _RoundSTE.apply(h)
            return self.l2(h)

    model = _GatedSTEModel().to("metal")
    x = lucid.randn(3, 4, device="metal")
    lucid.manual_seed(77)
    gen = lucid.Generator(88)
    s0 = lucid.get_rng_state().numpy().copy()
    g0 = gen.get_state().numpy().copy()

    with pytest.raises(CustomBackwardInTrace):
        compiled_step(model, x, lambda y: (y * y).sum())

    np.testing.assert_array_equal(lucid.get_rng_state().numpy(), s0)
    np.testing.assert_array_equal(gen.get_state().numpy(), g0)


def test_compiled_step_compiles_function_less_grad_branch_and_advances_rng_once() -> None:
    class _GradBranchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            noise = lucid.randn(x.shape, device=x.device)
            h = self.l1(x) + 0.1 * noise
            if lucid.is_grad_enabled() and h.requires_grad:
                h = h * 2.5
            else:
                h = h * 0.0
            return self.l2(h)

    lucid.manual_seed(10)
    eager_model = _GradBranchModel().to("metal")
    comp_model = copy.deepcopy(eager_model)
    x = lucid.randn(3, 4, device="metal")

    lucid.manual_seed(321)
    out_e = eager_model(x)
    loss_e = (out_e * out_e).sum()
    loss_e.backward()
    eager_rng = lucid.get_rng_state().numpy().copy()

    lucid.manual_seed(321)
    loss_c = compiled_step(comp_model, x, lambda y: (y * y).sum())
    comp_rng = lucid.get_rng_state().numpy().copy()

    np.testing.assert_array_equal(comp_rng, eager_rng)
    np.testing.assert_allclose(loss_c.cpu().numpy(), loss_e.cpu().numpy(), rtol=1e-5)
    for p_e, p_c in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
        assert p_e.grad is not None and p_c.grad is not None
        np.testing.assert_allclose(
            p_c.grad.cpu().numpy(), p_e.grad.cpu().numpy(), rtol=1e-5, atol=1e-5
        )


def test_compiled_step_refuses_requires_grad_hook_and_retain_grad() -> None:
    class _HookedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            h = self.lin(x)
            if h.requires_grad:
                h.register_hook(lambda g: g * 2.0)
            return h

    class _RetainGradModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: Tensor) -> Tensor:
            h = self.lin(x)
            h.retain_grad()
            return h * 2.0

    x = lucid.randn(2, 4, device="metal")
    with pytest.raises(RuntimeError, match="register_hook"):
        compiled_step(_HookedModel().to("metal"), x, lambda y: y.sum())
    with pytest.raises(RuntimeError, match="retain_grad"):
        compiled_step(_RetainGradModel().to("metal"), x, lambda y: y.sum())


def test_diagnose_records_in_caller_grad_mode_and_preserves_rng() -> None:
    seen_grad: list[bool] = []

    def fn(x: Tensor) -> Tensor:
        seen_grad.append(lucid.is_grad_enabled())
        noise = lucid.randn(x.shape)
        if lucid.is_grad_enabled():
            return (x + noise).sin()
        return x + noise

    x = lucid.randn(2, 3, requires_grad=True)
    lucid.manual_seed(42)
    s0 = lucid.get_rng_state().numpy().copy()

    with lucid.enable_grad():
        report_grad = diagnose(fn, x)
    with lucid.no_grad():
        report_nograd = diagnose(fn, x)

    assert seen_grad == [True, False]
    assert any(op.name == "sin" for op in report_grad.registered)
    assert not any(op.name == "sin" for op in report_nograd.registered)
    np.testing.assert_array_equal(lucid.get_rng_state().numpy(), s0)
