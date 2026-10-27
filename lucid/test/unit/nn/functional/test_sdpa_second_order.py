"""Scaled dot-product attention: second derivatives, and one meaning per call.

Two families of silent disagreement are pinned here.

* **Second order.** The fused backward returned raw storage, so a derivative
  of a gradient through attention — a gradient penalty, the double-backward
  form of a JVP — had nothing to differentiate.  The engine now carries a
  graph-mode backward; every case compares it with the same attention written
  out in elementary ops.
* **Masks.** The same call answered differently per device: with an
  ``attn_mask`` the Metal kernel ignored ``is_causal`` (attending to the
  future), a non-square causal triangle was aligned bottom-right on Metal and
  top-left on the CPU, and the engine read a boolean mask as "keep" on Metal
  and "mask out" on the CPU.  ``F.scaled_dot_product_attention`` now folds
  those causal cases into the additive mask (top-left, the reference
  framework's alignment), the engine refuses them, and a boolean mask keeps
  on every device.
"""

import math

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

_TOL = 2e-4


def _qkv(device: str, lq: int = 4, lk: int = 4, d: int = 8) -> list[lucid.Tensor]:
    lucid.manual_seed(0)
    shapes = ((2, 2, lq, d), (2, 2, lk, d), (2, 2, lk, d))
    return [lucid.randn(*s).to(device).detach().requires_grad_(True) for s in shapes]


def _explicit(
    q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, mask: lucid.Tensor | None, causal: bool
) -> lucid.Tensor:
    """Attention in elementary ops, causal triangle aligned top-left."""
    scores = lucid.matmul(q, k.mT) * (1.0 / math.sqrt(q.shape[-1]))
    if mask is not None:
        scores = scores + mask
    if causal:
        lq, lk = int(q.shape[-2]), int(k.shape[-2])
        keep = lucid.tril(lucid.ones((lq, lk), device=q.device.type))
        scores = scores + (keep - 1.0) * 1e9
    return lucid.matmul(F.softmax(scores, dim=-1), v)


def _close(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return float((a - b).abs().max().item()) <= _TOL * (1.0 + float(b.abs().max().item()))


def _second_order(out: lucid.Tensor, wrt: list[lucid.Tensor]) -> list[lucid.Tensor]:
    """∂/∂wrt of Σ‖∂(Σ out²)/∂wrt‖² — a gradient penalty through attention."""
    firsts = lucid.autograd.grad((out * out).sum(), wrt, create_graph=True)
    penalty = sum(((g * g).sum() for g in firsts), lucid.zeros(()).to(out.device))
    return list(lucid.autograd.grad(penalty, wrt))


@pytest.mark.parametrize("case", ["plain", "causal", "additive", "keep"])
def test_second_derivative_matches_the_explicit_form(device: str, case: str) -> None:
    q, k, v = _qkv(device)
    wrt = [q, k, v]
    mask: lucid.Tensor | None = None
    if case == "additive":
        lucid.manual_seed(1)
        mask = (lucid.randn(4, 4) * 0.5).to(device).detach().requires_grad_(True)
        wrt.append(mask)
    fused_mask = mask
    explicit_mask = mask
    if case == "keep":
        keep = lucid.tensor([[j <= i or j == 3 for j in range(4)] for i in range(4)])
        fused_mask = keep.to(device)
        explicit_mask = ((keep.to(lucid.float32) - 1.0) * 1e9).to(device)
    causal = case == "causal"

    got = _second_order(
        F.scaled_dot_product_attention(q, k, v, attn_mask=fused_mask, is_causal=causal), wrt
    )
    want = _second_order(_explicit(q, k, v, explicit_mask, causal), wrt)
    for name, g, w in zip("qkvm", got, want):
        assert _close(g, w), f"{case}: d²/d{name} differs"


def test_create_graph_first_order_equals_the_eager_backward(device: str) -> None:
    q, k, v = _qkv(device)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    graph = lucid.autograd.grad((out * out).sum(), [q, k, v], create_graph=True)
    eager = lucid.autograd.grad((out * out).sum(), [q, k, v])
    for g, e in zip(graph, eager):
        assert _close(g, e)


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("lq,lk", [(4, 4), (2, 5), (5, 2)])
def test_causal_means_one_thing_on_every_device(
    device: str, lq: int, lk: int, with_mask: bool
) -> None:
    q, k, v = _qkv(device, lq, lk)
    mask = None
    if with_mask:
        lucid.manual_seed(2)
        mask = lucid.randn(lq, lk).to(device)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
    assert _close(got, _explicit(q, k, v, mask, True))


@pytest.mark.parametrize("lq,lk,with_mask", [(2, 5, False), (4, 4, True)])
def test_engine_refuses_an_ambiguous_causal_call(
    device: str, lq: int, lk: int, with_mask: bool
) -> None:
    q, k, v = _qkv(device, lq, lk)
    mask = _unwrap(lucid.zeros(lq, lk).to(device)) if with_mask else None
    with pytest.raises(Exception, match="is_causal needs a square"):
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(q), _unwrap(k), _unwrap(v), mask, 0.5, True
        )


def test_engine_bool_mask_keeps_on_every_device(device: str) -> None:
    q, k, v = (t.detach() for t in _qkv(device))
    keep = lucid.tensor([[j <= i for j in range(4)] for i in range(4)]).to(device)
    by_mask = _wrap(
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(q), _unwrap(k), _unwrap(v), _unwrap(keep), 0.5, False
        )
    )
    by_flag = _wrap(
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(q), _unwrap(k), _unwrap(v), None, 0.5, True
        )
    )
    assert _close(by_mask, by_flag)
