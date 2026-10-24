"""Regression test for the compile-path output-buffer lifecycle bug.

Discovered 2026-05-26 while investigating the
``perf-compile-vs-eager-2026-05-26`` parity divergence at gpt2_block
@ BS=128.  The async ``encodeToCommandBuffer:`` + ``commit`` path in
``CompiledExecutable.mm::run_executable`` was silently overwriting
output MTLBuffers to zeros after roughly N consecutive ``cm(x)``
calls without intermediate materialisation (N ≈ 14 for transformer-
block-sized outputs, ~50 for CIFAR-block-sized outputs).  The
computation completed correctly but the output buffer lifecycle got
tangled in MLX's async tracker.

Fix (2026-05-26): default to the synchronous
``runWithMTLCommandQueue:waitUntilCompleted:YES`` path.  ``LUCID_COMPILE_ASYNC=1``
opts back into the async path for users who know their workload doesn't
hit the threshold.

This test exercises the GPT-2-block reproducer at N=20 (well above
the failure threshold of 14) and asserts the last call's output
still has the correct max value.  Under the buggy async-default code,
this test failed deterministically.  Under the sync default, it
passes.

If you tighten this test (lower N, smaller workload) be aware: the
bug threshold depends on output footprint, so a much smaller model
may not hit it.  GPT-2-block @ BS=32 is the sweet spot — small
enough to run fast, large enough that the buffer pool exhausts
quickly under async dispatch.

See also
--------
- ``obsidian/engine/engine-compile-output-buffer-pool-exhaustion-2026-05-26.md``
- ``obsidian/perf/perf-compile-vs-eager-2026-05-26.md``
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.metal as metal
from lucid._C import engine as _C_engine

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _MiniGPT2Block(nn.Module):
    """Single transformer block — d=768, 12 heads, seq=128.

    Same shape as the bench harness's ``_GPT2Block`` workload.  Used
    here because the buffer-pool bug threshold scales inversely with
    per-call output footprint; this block is large enough to expose
    the bug at N=14 under the prior async-default code.
    """

    def __init__(self) -> None:
        super().__init__()
        d_model = 768
        n_heads = 12
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x))


def _run_n_calls_and_return_last(n_calls: int) -> float:
    """Build a fresh compiled GPT-2 block, dispatch N times, return last max."""
    _C_engine.compile.session_cache_clear()
    lucid.manual_seed(0)
    model = _MiniGPT2Block().to(COMPILE_DEVICE)
    model.eval()
    x = lucid.randn(32, 128, 768).to(COMPILE_DEVICE)

    cm = lucid.compile(model)
    last_out: lucid.Tensor | None = None
    for _ in range(n_calls):
        last_out = cm(x)
    assert last_out is not None
    # Force materialisation + sync before reading the value.
    _ = float(last_out.sum().item())
    metal.synchronize()
    return float(last_out.abs().max().item())


def test_compile_output_survives_many_consecutive_calls() -> None:
    """20 consecutive ``cm(x)`` calls — last output must NOT be all zeros.

    Under the prior async-default code, this asserted-against value
    was ``0.0`` (silently corrupted buffer).  Under the sync default,
    it's the correct ~5.27.

    A 20-iter loop sits comfortably above the GPT-2-block failure
    threshold (N=14) so the test catches any regression of the fix
    even with some variance.
    """
    c_max = _run_n_calls_and_return_last(20)
    # Reference value from eager: ~5.27 for this exact seed + shape.
    # We accept anything > 1.0 (any plausibly-correct non-zero output)
    # to be robust to upstream init changes.
    assert c_max > 1.0, (
        f"compile output is all zeros (or near-zero: {c_max}) after 20 "
        "consecutive calls — the async-dispatch buffer-pool bug regressed. "
        "Check CompiledExecutable.mm::run_executable's sync/async default. "
        "See engine-compile-output-buffer-pool-exhaustion-2026-05-26.md."
    )


def test_compile_output_survives_threshold_boundary() -> None:
    """N=14 — the exact failure threshold from the original investigation."""
    c_max = _run_n_calls_and_return_last(14)
    assert c_max > 1.0, (
        f"compile output is all zeros at N=14 (got {c_max}) — "
        "the historic threshold of the async-dispatch buffer bug. "
        "See engine-compile-output-buffer-pool-exhaustion-2026-05-26.md."
    )
