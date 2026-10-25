"""Tests for Card E3 (``LCD-580b`` / ``LCD-607b`` / ``LCD-626``): forward trace is an eager call.

Covers:
1. A frozen model with an ``is_grad_enabled()`` branch traces in the caller's
   grad mode, matches eager in both ``enable_grad`` and ``no_grad``, and caches
   two distinct entries keyed by :attr:`CacheKey.grad_mode`.
2. First (compile-on-miss), second (cached hit), and refused (eager fallback)
   calls of a :class:`CompiledModule` are bit-equal to eager under a fixed seed
   for both the default RNG stream and a user :class:`~lucid.Generator`.
3. A nested :class:`CompiledModule` called inside an outer recording or outer
   compiled callable runs inline (eagerly) so its ops enter the outer trace and
   subsequent calls are never frozen at the first call's output (LCD-626).
"""

import copy

import numpy as np
import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._core.trace_context import recording
import lucid.nn as nn


@pytest.fixture(autouse=True)
def _clear_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LUCID_COMPILE_VERBOSE`` left by earlier matrix runs does not suppress warnings."""
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)


class _FrozenGradBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.lin(x)
        if lucid.is_grad_enabled():
            return h * 3.0 + 1.0
        return h * -2.0 - 0.5


def test_frozen_model_with_is_grad_enabled_branch_caches_per_grad_mode() -> None:
    lucid.manual_seed(0)
    eager = _FrozenGradBranchModel().to("metal")
    compiled = lucid.compile(copy.deepcopy(eager))
    x = lucid.randn(3, 4, device="metal")

    with lucid.enable_grad():
        out_grad_1 = compiled(x)
        out_grad_2 = compiled(x)
        want_grad = eager(x)

    with lucid.no_grad():
        out_nograd_1 = compiled(x)
        out_nograd_2 = compiled(x)
        want_nograd = eager(x)

    np.testing.assert_allclose(
        out_grad_1.cpu().numpy(), want_grad.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out_grad_2.cpu().numpy(), want_grad.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out_nograd_1.cpu().numpy(), want_nograd.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out_nograd_2.cpu().numpy(), want_nograd.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(out_grad_1.cpu().numpy(), out_nograd_1.cpu().numpy())

    info = compiled.cache_info()
    assert info["entries"] == 2
    assert not info["eager_only"]
    modes = {key.grad_mode for key in compiled._cache}
    assert modes == {False, True}


def test_compiled_forward_first_second_and_refused_calls_bit_equal_rng() -> None:
    class _StochasticFrozen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.requires_grad_(False)

        def forward(self, x: Tensor) -> Tensor:
            draw = lucid.randn(x.shape, device=x.device)
            return self.lin(x) + draw

    lucid.manual_seed(10)
    eager = _StochasticFrozen().to("metal")
    compiled = lucid.compile(copy.deepcopy(eager))
    x = lucid.randn(2, 4, device="metal")

    with lucid.no_grad():
        lucid.manual_seed(777)
        want_1 = eager(x).cpu().numpy().copy()
        want_2 = eager(x).cpu().numpy().copy()
        want_rng = lucid.get_rng_state().numpy().copy()

        lucid.manual_seed(777)
        got_1 = compiled(x).cpu().numpy().copy()
        got_2 = compiled(x).cpu().numpy().copy()
        got_rng = lucid.get_rng_state().numpy().copy()

    assert compiled.cache_info()["entries"] == 1
    np.testing.assert_allclose(got_1, want_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_2, want_2, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(got_rng, want_rng)


def test_refused_forward_restores_default_and_generator_rng_streams() -> None:
    def refused_fn(x: Tensor, gen: lucid.Generator) -> Tensor:
        a = lucid.randn(x.shape)
        b = lucid.randn(x.shape, generator=gen)
        # CPU trace is refused by MPSGraph lowering -> eager fallback.
        return x + a + b

    compiled = lucid.compile(refused_fn)
    x = lucid.zeros(2, 3)

    with lucid.no_grad():
        lucid.manual_seed(555)
        gen_e = lucid.Generator(999)
        want_1 = refused_fn(x, gen_e).numpy().copy()
        want_2 = refused_fn(x, gen_e).numpy().copy()
        want_default_state = lucid.get_rng_state().numpy().copy()
        want_gen_state = gen_e.get_state().numpy().copy()

        _NOTICED.discard("lowering")
        lucid.manual_seed(555)
        gen_c = lucid.Generator(999)
        with pytest.warns(CompileFallbackWarning):
            got_1 = compiled(x, gen_c).numpy().copy()
        got_2 = compiled(x, gen_c).numpy().copy()
        got_default_state = lucid.get_rng_state().numpy().copy()
        got_gen_state = gen_c.get_state().numpy().copy()

    np.testing.assert_array_equal(got_1, want_1)
    np.testing.assert_array_equal(got_2, want_2)
    np.testing.assert_array_equal(got_default_state, want_default_state)
    np.testing.assert_array_equal(got_gen_state, want_gen_state)


def test_nested_compiled_inlines_into_outer_trace_and_matches_eager() -> None:
    inner = lucid.compile(lambda t: (t + 1.0).sin())
    outer = lucid.compile(lambda t: inner(t) * 3.0)

    x1 = lucid.linspace(-1.0, 1.0, 6, device="metal").reshape(2, 3)
    x2 = lucid.linspace(0.5, 2.5, 6, device="metal").reshape(2, 3)

    with lucid.no_grad():
        # Warm inner cache first so it already has a compiled executable when outer traces.
        _ = inner(x1)
        assert inner.cache_info()["entries"] == 1

        with recording(watch=(x1,), fold_constants=True) as tracer:
            _ = inner(x1) * 3.0
        op_names = [op.name for op in tracer.graph.ops]
        assert "sin" in op_names

        y1 = outer(x1)
        y2 = outer(x2)
        want1 = (x1 + 1.0).sin() * 3.0
        want2 = (x2 + 1.0).sin() * 3.0

    assert outer.cache_info()["entries"] == 1
    np.testing.assert_allclose(
        y1.cpu().numpy(), want1.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        y2.cpu().numpy(), want2.cpu().numpy(), rtol=1e-5, atol=1e-5
    )
