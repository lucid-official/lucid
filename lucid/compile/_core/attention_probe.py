"""
lucid.compile._core.attention_probe — hardware capability probe for the
MPSGraph fused-attention miscompile.

Some GPU/OS combinations (observed: M1 Pro / macOS 26; NOT M4 Max / macOS 27)
pattern-match ``softmax(QKᵀ) @ V`` onto an internal fused-attention kernel that
silently produces WRONG results for a window of shapes (batch >= 3, sequence
length in [17, 24], head dim 64).  The compile emitters dodge it by emitting the
value-matmul transposed — ``a @ b == (bᵀ @ aᵀ)ᵀ`` — which breaks the pattern
match, but costs ~+70% on the attention path and is pure waste where the bug is
absent.

This probe runs once per process, compiles a known-bad envelope with the
workaround forced OFF, compares each result against the eager (always-correct)
reference, and pins the engine flag so the transpose only fires on affected
hardware.  It is **conservative by construction**: the flag defaults to -1
(apply the transpose), and this probe clears it to 0 (skip) only when EVERY
envelope shape matches eager.  Any divergence, compile fallthrough, or
unexpected error leaves the workaround ON.
"""

import threading
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._C.engine.compile import TraceGraph

_lock = threading.Lock()
_probed = False

# Envelope spanning the known-bad region (batch >= 3, seq in [17, 24], head dim
# 64, assorted head counts).  When a GPU miscompiles attention at all the defect
# is systematic across this region (as M1 Pro demonstrates), so an all-clean
# sweep is strong evidence the GPU is unaffected.
_ENVELOPE: tuple[tuple[int, int, int, int], ...] = (
    (4, 12, 17, 64),
    (3, 12, 20, 64),
    (8, 12, 24, 64),
    (4, 8, 18, 64),
    (8, 4, 22, 64),
    (3, 16, 23, 64),
)


# Ops whose emitters consult the workaround flag.  ``softmax`` covers manual
# attention (softmax feeding a matmul); ``scaled_dot_product_attention`` covers
# the fused op, whose graph contains no softmax node at all — without it a
# fused-SDPA graph never probes, so the flag stays at its conservative -1 and
# unaffected hardware pays the transpose tax (and the native fused emitter,
# which is gated on the same flag, never engages).
_PROBE_TRIGGER_OPS: frozenset[str] = frozenset({"softmax", "scaled_dot_product_attention"})


def maybe_probe_for_graph(graph: TraceGraph) -> None:
    """Run the capability probe iff ``graph`` contains an attention op.

    The probe only matters for attention, so skip it entirely for
    attention-free graphs (pure CNN / MLP) — they never hit the workaround and
    should not pay the one-time probe cost.
    """
    if _probed:
        return
    for node in graph.ops:
        if node.name in _PROBE_TRIGGER_OPS:
            ensure_attention_workaround_probed()
            return


def ensure_attention_workaround_probed() -> None:
    """Run the capability probe once; pin the engine workaround flag.

    Idempotent and thread-safe.  Compiles a handful of tiny attention graphs;
    runs at most once per process.
    """
    global _probed
    if _probed:
        return
    with _lock:
        if _probed:
            return
        _probed = True
        _run_probe()


def _run_probe() -> None:
    """Detect whether this GPU miscompiles fused attention; set the flag."""
    # Respect a state already pinned by a test / env override (anything other
    # than the -1 "unprobed" sentinel).
    if _C_engine.compile.attention_workaround_state() != -1:
        return

    import lucid
    import lucid.nn.functional as F

    def _attn(q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(q @ k.permute(0, 1, 3, 2), dim=-1) @ v

    affected = False
    # Force the workaround OFF so the probe observes the raw hardware result.
    _C_engine.compile.set_attention_workaround_state(0)
    try:
        gen = lucid.Generator(0)
        for b, h, n, d in _ENVELOPE:
            q = lucid.randn(b, h, n, d, generator=gen).to("metal")
            k = lucid.randn(b, h, n, d, generator=gen).to("metal")
            v = lucid.randn(b, h, n, d, generator=gen).to("metal")
            ref = _attn(q, k, v)  # eager (MLX, correct)
            cm = lucid.compile(_attn)
            out = cm(q, k, v)
            # If the attention fell back to eager (no compiled entry), the
            # comparison is vacuous — cannot certify the GPU, assume affected.
            if not getattr(cm, "_cache", None):
                affected = True
                break
            if float((out - ref).abs().max().item()) > 1e-2:
                affected = True
                break
    except RuntimeError:
        # An engine, Metal or compile failure (all RuntimeError subclasses,
        # LucidError included) → cannot certify the GPU clean → keep the
        # workaround on.
        affected = True
    finally:
        _C_engine.compile.set_attention_workaround_state(1 if affected else 0)
        # The probe compiled attention with the workaround forced off; on an
        # affected GPU those executables are WRONG, so purge the session cache
        # to ensure no probe artifact serves a later real attention compile.
        if affected:
            _C_engine.compile.session_cache_clear()
