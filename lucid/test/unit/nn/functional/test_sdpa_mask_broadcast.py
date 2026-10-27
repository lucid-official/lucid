"""SDPA additive-mask broadcasting — CPU-stream correctness regression.

The CPU (Accelerate) SDPA kernel flattens the leading (batch, head) dims into a
single batch and only broadcasts a mask that is EITHER fully shared over that
batch (numel == Lq·Lk) OR fully materialized (numel == B·H·Lq·Lk).  A *partial*
broadcast — exactly one of batch/head equal to 1, e.g. a ``(1, H, Lq, Lk)``
relative-position bias or a ``(B, 1, Lq, Lk)`` padding mask — has neither numel
and was mis-indexed past the mask buffer, producing silently-wrong scores
(measured maxdiff ~0.5-2.5 vs the reference softmax).  This hit every masked
attention on CPU: BERT's padding mask, and the CoAtNet / MaxViT / EfficientFormer
relative-position biases once they moved onto fused SDPA.

Fixed 2026-07-13 in the functional wrapper: on the CPU stream a partially
broadcast mask is materialized to the full score shape (free there — the scores
are already dense).  Metal/MLX broadcasts any-rank masks correctly and is left
untouched so the fused kernel keeps its O(N) memory win.

Oracle = the explicit ``softmax((q·kᵀ)·scale + mask)·v`` recomputation.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_B, _H, _T, _E = 2, 4, 16, 8


def _manual(q: object, k: object, v: object, m: object, scale: float) -> object:
    a = (q @ k.permute(0, 1, 3, 2)) * scale
    if m is not None:
        a = a + m
    return F.softmax(a, dim=-1) @ v


# Every leading-dim broadcast pattern of a (·, ·, T, S) additive mask.
_MASK_SHAPES = [
    (_B, _H, _T, _T),  # full
    (1, _H, _T, _T),  # batch-broadcast  (was broken on CPU)
    (_B, 1, _T, _T),  # head-broadcast   (was broken on CPU)
    (1, 1, _T, _T),  # fully broadcast
    (_B, 1, 1, _T),  # key-padding style (query-broadcast + head-broadcast)
]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("mshape", _MASK_SHAPES)
def test_sdpa_additive_mask_broadcast_matches_manual(device: str, mshape: tuple) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("metal backend unavailable")
    lucid.manual_seed(0)
    scale = _E**-0.5
    q = lucid.randn(_B, _H, _T, _E, device=device)
    k = lucid.randn(_B, _H, _T, _E, device=device)
    v = lucid.randn(_B, _H, _T, _E, device=device)
    m = lucid.randn(*mshape, device=device) * 0.3

    got = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale)
    ref = _manual(q, k, v, m, scale)
    assert float((got - ref).abs().max().item()) < 1e-4


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_sdpa_relpos_bias_shape_matches_manual(device: str) -> None:
    """The exact ``(1, H, N, N)`` relative-position-bias pattern the migrated
    CoAtNet / MaxViT / EfficientFormer attentions feed through SDPA."""
    if device == "metal" and not metal_available():
        pytest.skip("metal backend unavailable")
    lucid.manual_seed(1)
    B, H, N, E = 3, 4, 16, 8
    scale = E**-0.5
    q = lucid.randn(B, H, N, E, device=device)
    k = lucid.randn(B, H, N, E, device=device)
    v = lucid.randn(B, H, N, E, device=device)
    bias = lucid.randn(1, H, N, N, device=device)

    got = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    a = (q @ k.permute(0, 1, 3, 2)) * scale + bias
    ref = F.softmax(a, dim=-1) @ v
    assert float((got - ref).abs().max().item()) < 1e-4
