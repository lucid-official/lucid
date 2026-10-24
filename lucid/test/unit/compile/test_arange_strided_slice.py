"""Compiled ``arange`` + strided slicing (and the RoPE / RoFormer payoff).

``arange`` was a compile stub (eager fallback).  Because a strided slice
``x[..., ::step]`` lowers to a ``gather`` over an ``arange`` index, *any*
strided indexing forced the whole graph to eager — and with it RoFormer's
interleaved RoPE (``_rotate_interleaved`` uses ``x[..., 0::2]``).

``arange`` now bakes its sequence (``start + i*step``, all static) as a
``constantWithData`` tensor, so it — and everything built on it — compiles
bit-exactly.
"""

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


class _Arange(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        rng = lucid.arange(0, x.shape[-1], device=COMPILE_DEVICE).reshape(1, -1)
        return x + rng


class _StridedSlice(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x[..., 0::2] - x[..., 1::2]


class _RopeInterleaved(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, cos: lucid.Tensor, sin: lucid.Tensor
    ) -> lucid.Tensor:
        qr, kr = F.apply_rotary_emb(q, k, cos, sin, interleaved=True)
        return qr + kr


def test_arange_compiles_bit_exact() -> None:
    m = _Arange().to(COMPILE_DEVICE).eval()
    x = metal_tensor(4, 8)
    eager = m(x)
    cm = lucid.compile(m)
    out = cm(x)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    # The arange is a baked constant → identical to the eager fill.
    assert _maxdiff(eager, out) == 0.0


def test_strided_slice_compiles() -> None:
    m = _StridedSlice().to(COMPILE_DEVICE).eval()
    x = metal_tensor(2, 4, 8, 16)
    eager = m(x)
    cm = lucid.compile(m)
    out = cm(x)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) == 0.0


class _ReturnedSlice(nn.Module):
    """``return x[a:b]`` — a non-zero split_at piece as the graph output."""

    def __init__(self, lo: int, hi: int) -> None:
        super().__init__()
        self._lo, self._hi = lo, hi

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x[self._lo : self._hi]


def test_returned_middle_slice_compiles() -> None:
    # A returned split_at piece that isn't piece-0 used to go unbound (it was
    # never marked "consumed"), forcing eager fallback.  Now it compiles.
    for lo, hi in ((1, 2), (2, 4), (1, 3)):
        m = _ReturnedSlice(lo, hi).to(COMPILE_DEVICE).eval()
        x = metal_tensor(4, 4, 8)
        eager = m(x)
        cm = lucid.compile(m)
        out = cm(x)
        info = cm.cache_info()
        assert info["entries"] == 1 and not info["eager_only"], (lo, hi, info)
        assert _maxdiff(eager, out) == 0.0


def test_interleaved_rope_compiles() -> None:
    # RoFormer's RoPE layout: _rotate_interleaved slices x[..., 0::2] / [1::2].
    m = _RopeInterleaved().to(COMPILE_DEVICE).eval()
    b, h, s, d = 2, 4, 8, 16
    q, k = metal_tensor(b, h, s, d), metal_tensor(b, h, s, d)
    cos, sin = metal_tensor(s, d), metal_tensor(s, d)
    eager = m(q, k, cos, sin)
    cm = lucid.compile(m)
    out = cm(q, k, cos, sin)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) == 0.0


def test_roformer_compiles() -> None:
    # The model-level payoff: RoFormer (interleaved RoPE) compiles into one
    # executable instead of eager-falling-back at every rotary slice.
    cfg = M.RoFormerConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = M.RoFormerModel(cfg).to(COMPILE_DEVICE).eval()
    b, t = 2, 8
    ids = lucid.arange(0, b * t, device=COMPILE_DEVICE).reshape(b, t).to(lucid.int64)
    eager = model(ids).last_hidden_state
    cm = lucid.compile(model)
    # Under grad mode a trainable model's call runs eagerly (LCD-582).
    with lucid.no_grad():
        out = cm(ids).last_hidden_state
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5
