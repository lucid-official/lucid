"""Compiled scaled-dot-product-attention — causal path + mask safety.

``is_causal=True`` used to force eager fallback in the MPSGraph emitter.
It now emits a lower-triangular −∞ additive mask before the softmax, so
autoregressive decoders built on the fused SDPA op (``nn.MultiheadAttention``
/ ``nn.TransformerDecoderLayer`` with ``is_causal=True`` and no Python-built
mask) compile into a single executable.

Two correctness properties are pinned here:

  1. **Causal compiles, parity-exact vs eager** — square self-attention and
     the non-square (Lq < Lk) case.  Non-square causality is aligned
     top-left (query ``i`` sees keys ``0..i``); ``F.scaled_dot_product_attention``
     folds it into an additive mask before the engine, which only takes
     ``is_causal`` for a square score matrix.
  2. **Additive float masks compile** — the tracer wires the auxiliary mask
     tensor into the graph (a non-differentiable SDPA input), so masked
     attention (BERT padding masks, cross-attention) compiles and adds the
     mask before the softmax.  A bool mask compiles too: it is normalised to
     the equivalent additive mask before the tracer sees it, so there is only
     one mask form for the emitter to handle.
"""

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


def _qkv(B: int, H: int, Lq: int, Lk: int, D: int) -> tuple[lucid.Tensor, ...]:
    return (
        metal_tensor(B, H, Lq, D),
        metal_tensor(B, H, Lk, D),
        metal_tensor(B, H, Lk, D),
    )


class _Causal(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _Plain(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class _MaskOnly(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)


class _CausalPlusMask(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=True)


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_causal_square_compiles() -> None:
    m = _Causal().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_causal_non_square_decode_shape_compiles() -> None:
    # Lq < Lk: the top-left triangle arrives as an additive mask and compiles.
    m = _Causal().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(1, 2, 4, 64, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_plain_compiles() -> None:
    m = _Plain().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_additive_float_mask_compiles() -> None:
    # An additive float mask is wired into the trace (a non-differentiable
    # auxiliary SDPA input) and added before the softmax, so masked attention
    # — e.g. BERT padding masks — compiles instead of falling back to eager.
    m = _MaskOnly().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_causal_plus_mask_applies_both() -> None:
    # Both apply: the causal triangle is folded into the additive mask before
    # the engine sees it, so the emitter adds one mask and matches eager.
    m = _CausalPlusMask().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_bool_mask_compiles_like_the_additive_one() -> None:
    # A bool mask used to reach the emitter as a *set*-mask, which it
    # could not reproduce with a plain add, so it bailed to eager rather
    # than emit a wrong executable.  ``scaled_dot_product_attention`` now
    # converts bool to the equivalent additive mask up front — it had to,
    # because the fused and dropout paths were reading ``True`` in
    # opposite directions — so what the emitter sees is the additive form
    # it already handles, and the fallback is no longer needed.
    class _BoolMask(nn.Module):
        def forward(
            self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
        ) -> lucid.Tensor:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    m = _BoolMask().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 8, 8, 16)
    bool_mask = lucid.tril(lucid.ones(8, 8, device=COMPILE_DEVICE)).to(lucid.bool_)
    eager = m(q, k, v, bool_mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, bool_mask)

    assert not cm.cache_info()["eager_only"], "bool-mask SDPA no longer needs eager"
    assert _maxdiff(eager, out) < 1e-5

    # And it is the *right* answer, not merely a self-consistent one: a
    # lower-triangular keep-mask is causal attention.
    causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _maxdiff(causal, out) < 1e-5


def test_multihead_attention_causal_compiles() -> None:
    class _MHA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(
                embed_dim=32, num_heads=4, batch_first=True
            )

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            out, _ = self.mha(x, x, x, is_causal=True, need_weights=False)
            return out

    m = _MHA().to(COMPILE_DEVICE).eval()
    x = metal_tensor(2, 16, 32)
    eager = m(x)
    cm = lucid.compile(m)
    out = cm(x)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_bert_with_padding_mask_compiles() -> None:
    # The flagship payoff: BERT calls F.scaled_dot_product_attention with an
    # additive padding mask, so wiring the mask into the trace lets a real
    # masked transformer compile into one executable (was eager-fallback).
    cfg = M.BERTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    model = M.BERTModel(cfg).to(COMPILE_DEVICE).eval()

    b, t = 2, 8
    # token ids in [0, b*t) < vocab_size (64); arange is float, embedding
    # needs integer indices.
    ids = lucid.arange(0, b * t, device=COMPILE_DEVICE).reshape(b, t).to(lucid.int64)
    mask = lucid.ones(b, t, device=COMPILE_DEVICE)
    mask[0, 6:] = 0.0  # pad the tail of the first sequence

    eager = model(ids, attention_mask=mask).last_hidden_state
    cm = lucid.compile(model)
    out = cm(ids, attention_mask=mask).last_hidden_state
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5
