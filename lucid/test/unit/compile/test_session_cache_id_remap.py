"""Session-cache canonical TensorId remapping and non-silent eager fallback tests (LCD-454).

Covers two invariants:

1. ``ExecutableCache::session()`` keys on canonical first-appearance tensor IDs,
   while each trace draws raw ``TensorId`` values that depend on how many ops or
   RNG draws ran before the surviving graph nodes (e.g. ``lucid.randn`` pops its
   op node and mints an RNG feed ID before the consuming op enters).
   ``compile_or_cached`` must canonicalize ``input_ids`` / ``output_ids`` when
   storing into the session cache and localize them back into each caller's raw
   ``TensorId`` space on a cache hit — preserving slot order even for
   non-commutative ops whose raw feed IDs sort in opposite order.
2. Every path in ``CompiledModule._compile_for`` that returns ``None`` and adds
   a signature to ``_eager_only`` must emit a ``CompileFallbackWarning`` via
   ``_fallback_notice`` so no compiled callable silently runs eager.
"""

from collections.abc import Iterator
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._core.fallback import CompileFallbackWarning
import lucid.nn as nn
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001
        return False
    return True


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


@pytest.fixture(autouse=True)
def _clean_session_cache_and_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    yield
    _C_engine.compile.session_cache_clear()


class TestSessionCacheIdRemap:
    def test_constant_feed_then_rng_feed_both_compile(self) -> None:
        """``x + c`` followed by ``x + randn(...)`` hits the session cache and compiles."""
        c = lucid.ones(4, device=COMPILE_DEVICE)
        x = lucid.ones(4, device=COMPILE_DEVICE)

        f1 = lucid.compile.compile(lambda t: t + c)
        f2 = lucid.compile.compile(lambda t: t + lucid.randn(4, device=t.device))

        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            out1 = f1(x)
            _ = f2(x)  # first call traces (1 draw) + runs (1 draw)
            lucid.manual_seed(123)
            out2_a = f2(x)
            out2_b = f2(x)

        assert f1.cache_info()["entries"] == 1
        assert not f1.cache_info()["eager_only"]
        assert f2.cache_info()["entries"] == 1
        assert not f2.cache_info()["eager_only"]
        assert _C_engine.compile.session_cache_size() == 1

        assert np.allclose(out1.to("cpu").numpy(), (x + c).to("cpu").numpy())
        lucid.manual_seed(123)
        want2_a = x + lucid.randn(4, device=COMPILE_DEVICE)
        want2_b = x + lucid.randn(4, device=COMPILE_DEVICE)
        assert np.allclose(out2_a.to("cpu").numpy(), want2_a.to("cpu").numpy(), atol=1e-5)
        assert np.allclose(out2_b.to("cpu").numpy(), want2_b.to("cpu").numpy(), atol=1e-5)
        assert not np.allclose(out2_a.to("cpu").numpy(), out2_b.to("cpu").numpy())

    def test_rng_feed_then_constant_feed_both_compile(self) -> None:
        """Reverse order: ``x + randn(...)`` first, then ``x + c``."""
        c = lucid.ones(4, device=COMPILE_DEVICE) * 3.0
        x = lucid.ones(4, device=COMPILE_DEVICE) * 2.0

        f_rng = lucid.compile.compile(lambda t: t + lucid.randn(4, device=t.device))
        f_const = lucid.compile.compile(lambda t: t + c)

        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            _ = f_rng(x)
            lucid.manual_seed(77)
            out_rng = f_rng(x)
            out_const = f_const(x)

        assert f_rng.cache_info()["entries"] == 1
        assert not f_rng.cache_info()["eager_only"]
        assert f_const.cache_info()["entries"] == 1
        assert not f_const.cache_info()["eager_only"]
        assert _C_engine.compile.session_cache_size() == 1

        lucid.manual_seed(77)
        want_rng = x + lucid.randn(4, device=COMPILE_DEVICE)
        assert np.allclose(out_rng.to("cpu").numpy(), want_rng.to("cpu").numpy(), atol=1e-5)
        assert np.allclose(out_const.to("cpu").numpy(), (x + c).to("cpu").numpy(), atol=1e-5)

    @pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
    def test_three_same_graph_functions_with_distinct_raw_id_offsets(
        self, order: tuple[int, int, int]
    ) -> None:
        """Three callables with identical canonical graphs but different raw ID gaps all compile."""
        c1 = metal_tensor(2, 4)
        c2 = metal_tensor(2, 4)
        x = metal_tensor(2, 4)

        # All three trace to two chained ``add`` ops with 3 feeds, but each draws
        # different raw input_ids and output_ids depending on where RNG draws happen:
        #   f0: no RNG draw -> raw feeds {1, 2, 4}, output 3
        #   f1: first operand is RNG draw -> raw feeds {1, 3, 5}, output 4
        #   f2: both constants are RNG draws -> raw feeds {1, 3, 5}, output 6
        f0 = lucid.compile.compile(lambda t: (t + c1) + c2)
        f1 = lucid.compile.compile(lambda t: (t + lucid.randn(2, 4, device=t.device)) + c2)
        f2 = lucid.compile.compile(
            lambda t: (t + lucid.randn(2, 4, device=t.device)) + lucid.rand(2, 4, device=t.device)
        )
        fns = [f0, f1, f2]

        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            for idx in order:
                _ = fns[idx](x)

        assert _C_engine.compile.session_cache_size() == 1
        for fn in fns:
            info = fn.cache_info()
            assert info["entries"] == 1, info
            assert not info["eager_only"], info

        assert np.allclose(f0(x).to("cpu").numpy(), ((x + c1) + c2).to("cpu").numpy(), atol=1e-5)
        lucid.manual_seed(314)
        got_f2 = f2(x)
        lucid.manual_seed(314)
        want_f2 = (x + lucid.randn(2, 4, device=COMPILE_DEVICE)) + lucid.rand(
            2, 4, device=COMPILE_DEVICE
        )
        assert np.allclose(got_f2.to("cpu").numpy(), want_f2.to("cpu").numpy(), atol=1e-5)

    def test_non_commutative_op_preserves_operand_order_across_id_inversion(
        self,
    ) -> None:
        """``t - randn`` vs ``t - c`` vs ``randn - t`` must bind LHS and RHS to the right slots.

        In ``t - randn(...)``, ``randn`` draws raw ID 1 (RHS, canonical 1) while
        ``t`` gets raw ID 3 (LHS, canonical 0), so ``MpsBuilder`` sorts the
        placeholders as ``[raw 1 (RHS), raw 3 (LHS)]``.  In ``t - c``, ``t`` has
        raw ID 1 (LHS, canonical 0) and ``c`` has raw ID 2 (RHS, canonical 1).
        In ``randn(...) - t``, ``randn`` has raw ID 1 (LHS, canonical 0) and
        ``t`` has raw ID 3 (RHS, canonical 1).  All three share one session
        cache entry and must bind each caller's LHS and RHS to the right
        executable slots regardless of compilation order.
        """
        x = lucid.full((4,), 10.0, device=COMPILE_DEVICE)
        c = lucid.full((4,), 3.0, device=COMPILE_DEVICE)

        f_sub_rng = lucid.compile.compile(lambda t: t - lucid.randn(4, device=t.device))
        f_sub_const = lucid.compile.compile(lambda t: t - c)
        f_rng_sub = lucid.compile.compile(lambda t: lucid.randn(4, device=t.device) - t)

        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            # First call compiles/binds each callable.
            _ = f_sub_rng(x)
            _ = f_sub_const(x)
            _ = f_rng_sub(x)

            # Subsequent calls hit the compiled executables directly (1 RNG draw per call).
            lucid.manual_seed(99)
            got_sub_rng = f_sub_rng(x)
            got_sub_const = f_sub_const(x)
            lucid.manual_seed(99)
            got_rng_sub = f_rng_sub(x)

        assert _C_engine.compile.session_cache_size() == 1
        for fn in (f_sub_rng, f_sub_const, f_rng_sub):
            assert fn.cache_info()["entries"] == 1
            assert not fn.cache_info()["eager_only"]

        lucid.manual_seed(99)
        r = lucid.randn(4, device=COMPILE_DEVICE)
        assert np.allclose(got_sub_rng.to("cpu").numpy(), (x - r).to("cpu").numpy(), atol=1e-5)
        assert np.allclose(got_sub_const.to("cpu").numpy(), (x - c).to("cpu").numpy(), atol=1e-5)
        assert np.allclose(got_rng_sub.to("cpu").numpy(), (r - x).to("cpu").numpy(), atol=1e-5)

    def test_unused_rng_feed_does_not_collide_with_used_only_trace(self) -> None:
        """A trace with an unread RNG feed has an extra placeholder and must not poison a 1-feed trace."""
        x = lucid.ones(4, device=COMPILE_DEVICE) * 4.0

        def with_unused_rng(t: lucid.Tensor) -> lucid.Tensor:
            _ = lucid.randn(4, device=t.device)
            return t * 2.0

        def plain(t: lucid.Tensor) -> lucid.Tensor:
            return t * 2.0

        f_unused = lucid.compile.compile(with_unused_rng)
        f_plain = lucid.compile.compile(plain)

        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            out1 = f_unused(x)
            out2 = f_plain(x)

        assert f_unused.cache_info()["entries"] == 1
        assert not f_unused.cache_info()["eager_only"]
        assert f_plain.cache_info()["entries"] == 1
        assert not f_plain.cache_info()["eager_only"]
        assert np.allclose(out1.to("cpu").numpy(), (x * 2.0).to("cpu").numpy())
        assert np.allclose(out2.to("cpu").numpy(), (x * 2.0).to("cpu").numpy())


class TestNoSilentEagerOnlyTransitions:
    def test_empty_trace_emits_compile_fallback_warning(self) -> None:
        """An identity callable with no traced ops warns when transitioning to ``_eager_only``."""
        x = metal_tensor(4)
        fn = lucid.compile.compile(lambda t: t)

        with pytest.warns(CompileFallbackWarning, match="no operations"):
            out = fn(x)

        assert np.allclose(out.to("cpu").numpy(), x.to("cpu").numpy())
        assert len(fn.cache_info()["eager_only"]) == 1

    def test_training_batchnorm_forward_only_emits_compile_fallback_warning(
        self,
    ) -> None:
        """Forward-only ``lucid.compile`` on a training-mode BatchNorm warns on fallback."""
        bn = nn.BatchNorm1d(4).to(COMPILE_DEVICE).train()
        compiled_bn = lucid.compile.compile(bn)
        x = metal_tensor(8, 4)

        with pytest.warns(CompileFallbackWarning, match="BatchNorm"):
            _ = compiled_bn(x)

        assert len(compiled_bn.cache_info()["eager_only"]) == 1

    def test_unresolvable_feed_id_emits_compile_fallback_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If an executable ever reports an input_id absent from external_feeds, it warns."""
        monkeypatch.setattr(
            _C_engine.compile,
            "compile_or_cached",
            lambda *_a, **_kw: SimpleNamespace(input_ids=[999999], output_ids=[0]),
        )
        x = metal_tensor(4)
        fn = lucid.compile.compile(lambda t: t + 1.0)

        with pytest.warns(CompileFallbackWarning, match="external_feeds"):
            out = fn(x)

        assert np.allclose(out.to("cpu").numpy(), (x + 1.0).to("cpu").numpy())
        assert len(fn.cache_info()["eager_only"]) == 1
