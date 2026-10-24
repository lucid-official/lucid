"""Compiled single-token decode over a StaticCache.

``compiled_decode_step`` compiles the decode forward ONCE into a reused
executable and rolls the StaticCache buffers forward in place.  These tests use a
minimal masked self-attention decoder (the same full-buffer + cache_position-mask
shape a real GPT-2 decode compiles into) and assert:

* the compiled decode is token-identical to the eager decode, and
* a single executable serves every decode step (stable signature — the
  ``cache_position`` is a fixed-shape ``(1,)`` int64 runtime feed).

They decode under ``no_grad``, as generation does: under grad mode a step
whose model has trainable parameters runs eagerly, so that its result
requires grad as eager's does (LCD-618).
"""

import math

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.utils.cache import DynamicCache, StaticCache

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

pytestmark = pytest.mark.usefixtures("inference")

_B, _H, _DH, _L = 1, 2, 4, 8
_D = _H * _DH


class _MaskedDecoder(nn.Module):
    """One masked self-attention layer that reads/writes a StaticCache.

    Attends a single query over the FULL fixed-shape buffer with a
    ``cache_position``-derived mask (future slots set to -inf) — the
    compile-friendly shape (constant every step), not a growing narrowed view.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(_D, _D)
        self.k = nn.Linear(_D, _D)
        self.v = nn.Linear(_D, _D)
        self.o = nn.Linear(_D, _D)

    def forward(
        self, hidden: lucid.Tensor, cache: StaticCache, cache_position: lucid.Tensor
    ) -> lucid.Tensor:
        bsz = hidden.shape[0]

        def split(x: lucid.Tensor) -> lucid.Tensor:
            return x.reshape(bsz, 1, _H, _DH).permute(0, 2, 1, 3)

        q, k, v = split(self.q(hidden)), split(self.k(hidden)), split(self.v(hidden))
        kk, vv = cache.update(k, v, 0, {"cache_position": cache_position})
        scores = (q @ kk.permute(0, 1, 3, 2)) / math.sqrt(_DH)
        idx = lucid.arange(_L, device=hidden.device.type).long().reshape(1, 1, 1, _L)
        valid = idx <= cache_position.reshape(1, 1, 1, 1)
        neg = lucid.tensor(-1e30, device=hidden.device.type)
        scores = lucid.where(valid, scores, neg)
        ctx = F.softmax(scores, dim=-1) @ vv
        return self.o(ctx.permute(0, 2, 1, 3).reshape(bsz, 1, _D))


def _decode(
    model: _MaskedDecoder, compiled: bool, n_steps: int
) -> tuple[list[lucid.Tensor], int]:
    cache = StaticCache(_L)

    def fwd(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
        return model(h, cache, cp)

    step = compiled_decode_step(fwd, cache) if compiled else fwd
    outs: list[lucid.Tensor] = []
    for pos in range(n_steps):
        h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE) * ((pos + 1) * 0.1)
        cp = lucid.tensor([pos], device=COMPILE_DEVICE).long()
        out = step(h, cp)
        out.eval()
        outs.append(out)
    n_exe = len(getattr(step, "cache", {})) if compiled else 0
    return outs, n_exe


def test_compiled_decode_token_identical() -> None:
    lucid.manual_seed(0)
    model = _MaskedDecoder().to(COMPILE_DEVICE).eval()
    eager, _ = _decode(model, compiled=False, n_steps=8)
    comp, n_exe = _decode(model, compiled=True, n_steps=8)
    worst = max(float((e - c).abs().max().item()) for e, c in zip(eager, comp))
    assert worst < 1e-4, f"compiled decode diverged from eager: {worst:.3e}"


def test_compiled_decode_single_executable() -> None:
    # The prefill step (empty cache) runs eager to allocate the buffers; every
    # subsequent decode step shares ONE compiled executable (fixed signature).
    lucid.manual_seed(0)
    model = _MaskedDecoder().to(COMPILE_DEVICE).eval()
    _, n_exe = _decode(model, compiled=True, n_steps=8)
    assert n_exe == 1, f"expected 1 cached decode executable, got {n_exe}"


def test_gpt2_compiled_decode_matches_eager() -> None:
    # Real-model end-to-end: a GPT-2 LM decoded one token at a time through the
    # compiled StaticCache path must produce logits token-identical to the eager
    # DynamicCache path, with a single reused executable across all steps.
    lucid.manual_seed(3)
    cfg = M.GPT2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    gpt = M.GPT2LMHeadModel(cfg).to(COMPILE_DEVICE).eval()
    prompt = lucid.tensor([[5, 9, 2, 7]], device=COMPILE_DEVICE).long()
    t_prompt, max_len = 4, 32
    feed = [11, 3, 40, 22, 8, 1, 55, 17]

    # Eager reference (DynamicCache).
    ce = DynamicCache()
    gpt(prompt, past_key_values=ce, use_cache=True)
    ref = []
    for t in feed:
        tok = lucid.tensor([[t]], device=COMPILE_DEVICE).long()
        ref.append(gpt(tok, past_key_values=ce, use_cache=True).logits[:, -1])

    # Compiled (StaticCache prefill + compiled decode), same feed.
    cs = StaticCache(max_len)
    cp0 = lucid.arange(0, t_prompt, device=COMPILE_DEVICE).long()
    gpt(prompt, past_key_values=cs, use_cache=True, cache_position=cp0)

    def fwd(ids: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
        return gpt(ids, past_key_values=cs, use_cache=True, cache_position=cp).logits

    step = compiled_decode_step(fwd, cs)
    comp = []
    pos = t_prompt
    for t in feed:
        tok = lucid.tensor([[t]], device=COMPILE_DEVICE).long()
        cp = lucid.tensor([pos], device=COMPILE_DEVICE).long()
        comp.append(step(tok, cp)[:, -1])
        pos += 1

    worst = max(float((r - c).abs().max().item()) for r, c in zip(ref, comp))
    assert worst < 1e-4, f"GPT-2 compiled decode diverged from eager: {worst:.3e}"
    assert len(getattr(step, "cache", {})) == 1


@pytest.mark.parametrize("frozen", [False, True])
def test_a_step_under_grad_mode_answers_as_eager(frozen: bool) -> None:
    """A step that needs gradients runs eagerly, and only that one (LCD-618).

    The executable records no autograd graph: from the first compiled step on
    the decode returned results that did not require grad.  A frozen model
    needs none and stays compiled; after grad mode, a step under ``no_grad``
    still compiles.
    """
    runs = []
    for compiled in (True, False):
        lucid.manual_seed(0)
        model = _MaskedDecoder().to(COMPILE_DEVICE).eval().requires_grad_(not frozen)
        cache = StaticCache(_L)

        def fwd(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
            return model(h, cache, cp)

        step = compiled_decode_step(fwd, cache) if compiled else fwd
        outs = []
        with lucid.enable_grad():
            for pos in range(4):
                h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE) * ((pos + 1) * 0.1)
                outs.append(step(h, lucid.tensor([pos], device=COMPILE_DEVICE).long()))
        runs.append(outs)
        if compiled:
            assert len(step.cache) == (1 if frozen else 0)  # type: ignore[attr-defined]
            h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE)
            step(h, lucid.tensor([4], device=COMPILE_DEVICE).long())
            assert len(step.cache) == 1  # type: ignore[attr-defined]
    got, want = runs
    assert [t.requires_grad for t in got] == [t.requires_grad for t in want]
    assert [t.requires_grad for t in want] == [not frozen] * 4
    worst = max(float((g - w).abs().max().item()) for g, w in zip(got, want))
    assert worst < 1e-4


class _GainedDecoder(_MaskedDecoder):
    """Reads a buffer of its own on the host — a value the trace would bake."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gain", lucid.tensor(2.0))

    def forward(
        self, hidden: lucid.Tensor, cache: StaticCache, cache_position: lucid.Tensor
    ) -> lucid.Tensor:
        return super().forward(hidden, cache, cache_position) * self.gain.item()


def test_a_host_read_of_the_models_state_runs_the_step_eagerly() -> None:
    """The step's trace watches the model ``forward_fn`` reaches, as a
    compiled module's does: the gain read on the first step stayed baked in."""
    runs = []
    for compiled in (True, False):
        lucid.manual_seed(0)
        model = _GainedDecoder().to(COMPILE_DEVICE).eval()
        cache = StaticCache(_L)

        def fwd(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
            return model(h, cache, cp)

        step = compiled_decode_step(fwd, cache) if compiled else fwd
        outs = []
        for pos in range(4):
            if pos == 2:
                model.get_buffer("gain").fill_(5.0)
            h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE) * ((pos + 1) * 0.1)
            outs.append(step(h, lucid.tensor([pos], device=COMPILE_DEVICE).long()))
        runs.append(outs)
    got, want = runs
    worst = max(float((g - w).abs().max().item()) for g, w in zip(got, want))
    assert worst < 1e-4
