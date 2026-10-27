"""SDPA forward — memory-efficient fused (MLX / metal) path (Phase 1).

Additive-mask attention now routes through the fused kernel instead of
materializing the ``(B, H, Lq, Lk)`` score matrix.  Checks:

* **output parity** — fused metal additive/bool/causal output matches the CPU
  reference (which rolls the full softmax by hand);
* **memory** — the fused forward allocates its output and no ``(Lq, Lk)``
  buffer, for every head dim up to 128: one MLX does not fuse is padded to
  one it does, and the padded call matches the CPU reference;
* **weights variant** — ``scaled_dot_product_attention_with_weights`` still
  returns correct dense weights on metal for none/causal/additive/bool.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

B, H, T, D = 2, 3, 16, 8
_N = B * H * T * D


def _mk(freq: float, device: str, n: int = _N, shape: tuple = (B, H, T, D)):
    vals = [math.sin(i * freq + 0.3) * 0.5 for i in range(n)]
    return lucid.tensor(vals, dtype=lucid.float32, device=device).reshape(*shape)


def _additive_mask(device: str, tq: int = T, tk: int = T):
    rows = [[0.0 if j <= i else -1.0e4 for j in range(tk)] for i in range(tq)]
    return lucid.tensor(rows, dtype=lucid.float32, device=device).reshape(1, 1, tq, tk)


def _bool_mask(device: str, tq: int = T, tk: int = T):
    rows = [[1.0 if j <= i else 0.0 for j in range(tk)] for i in range(tq)]
    t = lucid.tensor(rows, dtype=lucid.float32, device=device).reshape(1, 1, tq, tk)
    return t > 0.5


@pytest.mark.parametrize("config", ["none", "causal", "additive"])
def test_fused_forward_matches_cpu(config: str) -> None:
    """Fused metal SDPA output must match the CPU reference (manual softmax)."""
    if not metal_available():
        pytest.skip("metal backend unavailable")

    def out(device: str):
        q, k, v = _mk(0.7, device), _mk(0.5, device), _mk(0.3, device)
        if config == "none":
            r = F.scaled_dot_product_attention(q, k, v)
        elif config == "causal":
            r = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            r = F.scaled_dot_product_attention(q, k, v, attn_mask=_additive_mask(device))
        return r.numpy()

    np.testing.assert_allclose(out("metal"), out("cpu"), atol=1e-5, rtol=1e-4)


def test_query_broadcast_padding_mask_applied() -> None:
    """Regression: a (B, 1, 1, S) key-padding mask (query dim broadcast) must be
    applied correctly on the fused metal path.  The engine mis-applied such masks
    (masked keys kept weight) until the functional expands the query dim."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    Bt, Ht, Tq, Sk, Dh = 1, 4, 7, 7, 8
    n = Bt * Ht * Tq * Dh
    q = _mk(0.31, "metal", n=n, shape=(Bt, Ht, Tq, Dh))
    k = _mk(0.41, "metal", n=n, shape=(Bt, Ht, Sk, Dh))
    v = _mk(0.51, "metal", n=n, shape=(Bt, Ht, Sk, Dh))
    # mask out the last two key positions for every query (broadcast over query).
    rows = [0.0] * (Sk - 2) + [float("-inf")] * 2
    mask = lucid.tensor(rows, dtype=lucid.float32, device="metal").reshape(1, 1, 1, Sk)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).numpy()
    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    s = np.einsum("bhqd,bhkd->bhqk", qn, kn) * (1.0 / math.sqrt(Dh)) + mask.numpy()
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    w = e / e.sum(-1, keepdims=True)
    ref = np.einsum("bhqk,bhkd->bhqd", w, vn)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)


# The head dims MLX fuses for more than 8 queries.  The engine zero-pads a
# narrower q, k and v to the next one (LCD-498).
_FUSED_WIDTHS = (64, 72, 80, 96, 128)


def _fused_width(*dims: int) -> int:
    return next(w for w in _FUSED_WIDTHS if w >= max(dims))


def _forward_peak_above_inputs(q, k, v, is_causal: bool) -> tuple[int, int]:
    """Peak bytes the forward allocates above its live inputs, and its output's size.

    The GPU is synchronised before each reading.  MLX hands a command
    buffer's temporaries back from its completion handler, after ``.item()``
    has already returned, so a peak reset right after one is charged with
    the previous call's scratch.
    """
    _C_engine.synchronize_gpu()
    before = _C_engine.metal_active_memory()
    _C_engine.metal_reset_peak_memory()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    _C_engine.synchronize_gpu()
    return _C_engine.metal_peak_memory() - before, out.numel() * 4


def _assert_no_score_matrix(dims: tuple[int, int], is_causal: bool) -> None:
    head_dim, value_dim = dims
    b, h, lq = 1, 4, 1024
    q, k = (lucid.randn(b, h, lq, head_dim, device="metal") for _ in range(2))
    v = lucid.randn(b, h, lq, value_dim, device="metal")
    extra, out_bytes = _forward_peak_above_inputs(q, k, v, is_causal)

    width = _fused_width(head_dim, value_dim)
    # A padded call copies q, k, v and the output at the fused width.
    padded = 0 if width == head_dim == value_dim else 4 * b * h * lq * width * 4
    # Any (Lq, Lk) buffer, even a one-byte mask, is at least lq * lq bytes.
    assert extra < out_bytes + padded + lq * lq // 2, (
        f"dims={dims} is_causal={is_causal}: the fused forward peaked {extra} "
        f"bytes above its inputs for a {out_bytes}-byte output and {padded} "
        f"bytes of padded copies; one (B, H, Lq, Lk) float32 score matrix is "
        f"{b * h * lq * lq * 4} bytes"
    )


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("head_dim", [16, 32, 48, 64, 72, 80, 88, 96, 128])
def test_fused_forward_allocates_no_score_matrix(head_dim: int, is_causal: bool) -> None:
    """The fused forward allocates its output and no ``(Lq, Lk)`` buffer.

    The bound is absolute.  It used to be "below the hand-rolled softmax
    path minus one score matrix", which measured that path as much as the
    kernel: MLX has no fused kernel for head dim 32, and the test passed only
    while ``scores * scale`` also materialised the broadcast scale.  When
    that copy went (LCD-299), the same 18.9 MB failed.

    At a head dim MLX does not fuse it ran unfused and held the scores
    (LCD-496); the engine now pads it to the next fused width, which costs
    copies linear in the sequence length.
    """
    if not metal_available():
        pytest.skip("metal backend unavailable")
    _assert_no_score_matrix((head_dim, head_dim), is_causal)


@pytest.mark.parametrize("dims", [(64, 32), (32, 64)])
def test_unequal_value_width_allocates_no_score_matrix(dims: tuple[int, int]) -> None:
    """MLX fuses only when q and v share a width, so both are padded to one."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    _assert_no_score_matrix(dims, is_causal=False)


@pytest.mark.parametrize("mode", ["none", "causal", "additive", "bool"])
@pytest.mark.parametrize("lk", [1024, 2048])
@pytest.mark.parametrize("dims", [(16, 16), (48, 48), (100, 100), (64, 32), (32, 64)])
def test_padded_head_dim_matches_cpu(dims: tuple[int, int], lk: int, mode: str) -> None:
    """A head dim padded to a fused width gives the CPU kernel's answer.

    The lengths are long enough for the engine to pad.  With ``lk != lq``
    the functional folds a causal triangle into an additive mask.
    """
    if not metal_available():
        pytest.skip("metal backend unavailable")
    head_dim, value_dim = dims
    lucid.manual_seed(0)
    b, h, lq = 1, 2, 1024
    q = lucid.randn(b, h, lq, head_dim)
    k = lucid.randn(b, h, lk, head_dim)
    v = lucid.randn(b, h, lk, value_dim)
    mask = None
    if mode == "additive":
        mask = lucid.randn(1, 1, lq, lk)
    elif mode == "bool":
        mask = lucid.randn(1, 1, lq, lk) > -1.0

    def out(device: str) -> np.ndarray:
        m = mask.to(device) if mask is not None else None
        return F.scaled_dot_product_attention(
            q.to(device),
            k.to(device),
            v.to(device),
            attn_mask=m,
            is_causal=mode == "causal",
        ).numpy()

    np.testing.assert_allclose(out("metal"), out("cpu"), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize(("dtype", "tol"), [(lucid.float16, 1e-3), (lucid.bfloat16, 8e-3)])
def test_padded_head_dim_half_matches_cpu(dtype, tol: float, is_causal: bool) -> None:
    """A padded half-precision call agrees with the CPU kernel, which widens.

    ``tol`` is about one unit in the last place of the dtype at 1.0.
    """
    if not metal_available():
        pytest.skip("metal backend unavailable")
    lucid.manual_seed(0)
    q, k, v = (lucid.randn(1, 2, 1024, 32).to(dtype) for _ in range(3))

    def out(device: str) -> np.ndarray:
        r = F.scaled_dot_product_attention(
            q.to(device), k.to(device), v.to(device), is_causal=is_causal
        )
        return r.to(lucid.float32).numpy()

    np.testing.assert_allclose(out("metal"), out("cpu"), atol=tol, rtol=tol)


@pytest.mark.parametrize("is_causal", [True, False])
def test_padded_head_dim_takes_head_transposed_operands(is_causal: bool) -> None:
    """Operands built ``(B, L, H, D)`` and transposed, as multi-head attention does."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    lucid.manual_seed(0)
    q, k = (lucid.randn(2, 1024, 4, 32, device="metal").transpose(1, 2) for _ in range(2))
    v = lucid.randn(2, 1024, 4, 48, device="metal").transpose(1, 2)

    def out(q, k, v) -> np.ndarray:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal).numpy()

    cpu = out(q.to("cpu"), k.to("cpu"), v.to("cpu"))
    np.testing.assert_allclose(out(q, k, v), cpu, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("config", ["none", "causal", "additive", "bool"])
def test_with_weights_op_correct_on_metal(config: str) -> None:
    """The weights variant materializes correct dense W + output on metal."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    dev = "metal"
    q, k, v = _mk(0.7, dev), _mk(0.5, dev), _mk(0.3, dev)
    scale = 1.0 / math.sqrt(D)

    mask = None
    is_causal = False
    if config == "causal":
        is_causal = True
    elif config == "additive":
        mask = _additive_mask(dev)
    elif config == "bool":
        mask = _bool_mask(dev)

    out_i, w_i = _C_engine.nn.scaled_dot_product_attention_with_weights(
        _unwrap(q),
        _unwrap(k),
        _unwrap(v),
        attn_mask=(_unwrap(mask) if mask is not None else None),
        scale=scale,
        is_causal=is_causal,
    )
    w = _wrap(w_i).numpy()
    out = _wrap(out_i).numpy()

    # numpy reference.
    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    scores = np.einsum("bhqd,bhkd->bhqk", qn, kn) * scale
    if config == "causal":
        tri = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(tri[None, None], -np.inf, scores)
    elif config == "additive":
        scores = scores + _additive_mask(dev).numpy()
    elif config == "bool":
        keep = _bool_mask(dev).numpy()
        scores = np.where(keep, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    w_ref = e / e.sum(axis=-1, keepdims=True)
    out_ref = np.einsum("bhqk,bhkd->bhqd", w_ref, vn)

    np.testing.assert_allclose(w, w_ref, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(out, out_ref, atol=1e-5, rtol=1e-4)
