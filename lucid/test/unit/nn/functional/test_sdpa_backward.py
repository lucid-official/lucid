"""SDPA backward gradcheck — the fused (MLX / metal) engine path.

Regression guard for a latent bug in the GPU backward: the hand-rolled
``softmax(QKᵀ·s)`` recompute silently dropped the causal / additive mask, so
``is_causal=True`` training on metal produced gradients as if attention were
bidirectional (measured dV error ~0.37 before the fix).  The backward now
differentiates the fused kernel via ``mx::vjp`` (fixed 2026-07-07), which
captures the mask exactly.

Ground truth is finite differences (``gradcheck``); a secondary check confirms
the metal grads agree with the CPU reference implementation.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available
from lucid.test._helpers.grad_check import gradcheck

B, H, T, D = 1, 2, 4, 3
_N = B * H * T * D

# Deterministic, device-independent inputs (no RNG so metal == cpu exactly).
_QV = [math.sin(i * 0.7) * 0.5 for i in range(_N)]
_KV = [math.cos(i * 0.5) * 0.5 for i in range(_N)]
_VV = [math.sin(i * 0.3 + 1.0) * 0.5 for i in range(_N)]


def _mk(vals, device):
    return lucid.tensor(vals, dtype=lucid.float32, device=device).reshape(B, H, T, D)


def _additive_mask(device):
    # Finite negatives (not -inf) so finite differences stay well-conditioned;
    # upper triangle disallowed, diagonal + below allowed.
    rows = [[0.0 if j <= i else -20.0 for j in range(T)] for i in range(T)]
    return lucid.tensor(rows, dtype=lucid.float32, device=device).reshape(1, 1, T, T)


_MASK_CONFIGS = ["none", "causal", "additive"]


def _call(q, k, v, config, device):
    if config == "none":
        return F.scaled_dot_product_attention(q, k, v)
    if config == "causal":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if config == "additive":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=_additive_mask(device))
    raise ValueError(config)


@pytest.mark.parametrize("config", _MASK_CONFIGS)
def test_sdpa_backward_gradcheck_metal(config: str) -> None:
    """dQ/dK/dV from autograd must match finite differences on the fused path."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    dev = "metal"

    # Full-Jacobian gradcheck per input (helper differentiates inputs[0] only).
    # dQ
    k = _mk(_KV, dev)
    v = _mk(_VV, dev)
    gradcheck(
        lambda qq: _call(qq, k, v, config, dev), [_mk(_QV, dev)], eps=1e-3, atol=2e-2, rtol=2e-2
    )
    # dK
    q = _mk(_QV, dev)
    v = _mk(_VV, dev)
    gradcheck(
        lambda kk: _call(q, kk, v, config, dev), [_mk(_KV, dev)], eps=1e-3, atol=2e-2, rtol=2e-2
    )
    # dV
    q = _mk(_QV, dev)
    k = _mk(_KV, dev)
    gradcheck(
        lambda vv: _call(q, k, vv, config, dev), [_mk(_VV, dev)], eps=1e-3, atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("config", _MASK_CONFIGS)
def test_sdpa_backward_metal_matches_cpu(config: str) -> None:
    """Metal grads must match the CPU reference backward (secondary oracle)."""
    if not metal_available():
        pytest.skip("metal backend unavailable")

    def grads(device: str) -> tuple:
        q = _mk(_QV, device)
        q.requires_grad = True
        k = _mk(_KV, device)
        k.requires_grad = True
        v = _mk(_VV, device)
        v.requires_grad = True
        _call(q, k, v, config, device).sum().backward()
        return (q.grad.numpy(), k.grad.numpy(), v.grad.numpy())

    gq_c, gk_c, gv_c = grads("cpu")
    gq_m, gk_m, gv_m = grads("metal")
    np.testing.assert_allclose(gq_m, gq_c, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(gk_m, gk_c, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(gv_m, gv_c, atol=1e-4, rtol=1e-3)


def test_sdpa_causal_query0_grad_is_zero_metal() -> None:
    """Smoking-gun assertion: under causal masking query 0 attends only to key 0
    (softmax over one element), so dL/dq[...,0,:] must be exactly zero — the
    signature that exposed the dropped-mask bug (non-zero before the fix)."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    q = _mk(_QV, "metal")
    q.requires_grad = True
    k = _mk(_KV, "metal")
    v = _mk(_VV, "metal")
    F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
    dq0 = q.grad.numpy()[:, :, 0, :]
    np.testing.assert_allclose(dq0, np.zeros_like(dq0), atol=1e-6)
