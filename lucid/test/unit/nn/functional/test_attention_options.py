"""Multi-head attention options that used to crash or raise.

``add_bias_kv`` and ``add_zero_attn`` append key rows but left the caller's
masks as they were, so any mask made the module fail on a reshape.  The
functional form rejected ``add_zero_attn`` and separate projection weights
outright.  The parity tier compares the numbers against the reference.
"""

from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

E, H, L, S, N = 8, 2, 3, 5, 2


def _masks() -> tuple[lucid.Tensor, lucid.Tensor]:
    key_padding = lucid.tensor([[False] * (S - 1) + [True], [False] * S])
    attn = lucid.tensor([[False, True] + [False] * (S - 2)] + [[False] * S] * (L - 1))
    return key_padding, attn


@pytest.mark.parametrize(
    "cfg",
    [
        {"add_bias_kv": True},
        {"add_zero_attn": True},
        {"add_bias_kv": True, "add_zero_attn": True},
    ],
    ids=["bias_kv", "zero_attn", "both"],
)
def test_appended_keys_widen_the_masks(cfg: dict[str, Any]) -> None:
    mha = nn.MultiheadAttention(E, H, **cfg).eval()
    key_padding, attn = _masks()
    q, kv = lucid.randn(L, N, E), lucid.randn(S, N, E)
    out, weights = mha(q, kv, kv, key_padding_mask=key_padding, attn_mask=attn)
    appended = sum(bool(v) for v in cfg.values())
    assert tuple(out.shape) == (L, N, E)
    assert weights is not None
    assert tuple(weights.shape) == (N, L, S + appended)


def test_separate_projections_need_all_three_weights() -> None:
    x = lucid.randn(L, N, E)
    with pytest.raises(ValueError, match="q_proj_weight"):
        F.multi_head_attention_forward(
            x,
            x,
            x,
            E,
            H,
            use_separate_proj_weight=True,
            q_proj_weight=lucid.randn(E, E),
        )


def test_keys_already_split_per_head_are_still_refused() -> None:
    x = lucid.randn(L, N, E)
    with pytest.raises(NotImplementedError, match="static_k"):
        F.multi_head_attention_forward(
            x, x, x, E, H, static_k=lucid.randn(N * H, S, E // H)
        )


@pytest.mark.parametrize("dtype", [lucid.float64, lucid.float16, lucid.bfloat16])
def test_partial_weights_match_query_dtype(dtype: lucid.DType) -> None:
    q = lucid.randn(L, N, E, dtype=dtype)
    wi = lucid.randn(3 * E, E, dtype=dtype)
    bi = lucid.randn(3 * E, dtype=dtype)
    out, weights = F.multi_head_attention_forward(q, q, q, E, H, wi, bi)
    assert out.dtype == dtype
    assert weights is not None and weights.dtype == dtype
    assert tuple(out.shape) == (L, N, E)

