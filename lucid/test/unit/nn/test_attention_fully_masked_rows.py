"""A query row with every key masked out attends to nothing (LCD-129).

Attention's masked softmax had no rule for a row whose every score is
``-inf``, so the answer depended on the device, the dtype and the spelling
of the mask: a boolean mask became a finite ``-1e9`` and the row averaged
V; a ``-inf`` float mask gave ``0`` on the CPU in float32, the mean of V
in float16, and NaN on Metal; a fully padded batch in
``MultiheadAttention`` came back NaN on Metal.  The reference returns a
zero row with zero gradients for every mask kind.

One core (``nn.functional.attention._masked_attention``) now owns the
rule for the fused kernel, the dropout path and the weights path alike.
The invariant tests below run everywhere; the ``parity`` class holds the
whole matrix — {SDPA, MHA} x {bool, float, key padding, causal + mask} x
{no row, one row, every row masked} x {CPU, Metal} x {float32, float16},
values and query / key / value gradients — to the reference.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

N, H, L, S, E = 2, 2, 3, 5, 8
D = E // H

_DTYPES = {"float32": lucid.float32, "float16": lucid.float16}
_TOL = {"float32": 2e-5, "float16": 2e-2}
_KINDS = ["bool", "float", "key_padding", "causal_mask"]
_ROWS = ["none", "one", "all"]


def _drop(lq: int, lk: int, kind: str, rows: str) -> np.ndarray:
    """``(N, lq, lk)`` bool, ``True`` where a key is masked out.

    ``none`` masks some keys but leaves every row a key; ``one`` empties
    one row (one whole batch for key padding, which drops keys per batch);
    ``all`` empties every row.  ``causal_mask`` lists only the extra keys:
    the causal triangle is added by the caller.
    """
    drop = np.zeros((N, lq, lk), dtype=bool)
    if kind == "key_padding":
        drop[0, :, -1] = True
        if rows in ("one", "all"):
            drop[1] = True
        if rows == "all":
            drop[0] = True
        return drop
    if kind == "causal_mask":
        # Top-left causality leaves row 0 only key 0; banning key 0 empties it.
        drop[:, 1, 0] = True
        if rows in ("one", "all"):
            drop[1, 0, 0] = True
        if rows == "all":
            drop[:] = True
        return drop
    drop[:, 0, 1] = True
    if rows in ("one", "all"):
        drop[1, 1, :] = True
    if rows == "all":
        drop[:] = True
    return drop


def _draw(rng: np.random.Generator, dtype: str, *shape: int) -> np.ndarray:
    # Rounded through the tested dtype, so the reference sees the same inputs.
    return rng.standard_normal(shape).astype(dtype).astype(np.float32)


def _assert_close(got: lucid.Tensor | None, want: Any, dtype: str, what: str) -> None:
    assert got is not None, f"{what}: no gradient"
    g = got.detach().to(lucid.float32).cpu().numpy()
    w = want.detach().numpy()
    assert np.isfinite(g).all(), f"{what}: non-finite {g}"
    np.testing.assert_allclose(
        g, w, rtol=0, atol=_TOL[dtype] * max(1.0, np.abs(w).max()), err_msg=what
    )


# ── the rule, without the reference ────────────────────────────────────────


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
@pytest.mark.parametrize("spelling", ["bool", "float"])
def test_sdpa_zeroes_a_row_with_no_key(
    device: str, spelling: str, dropout_p: float, dtype: str
) -> None:
    rng = np.random.default_rng(0)
    q, k, v = (
        lucid.tensor(
            _draw(rng, dtype, N, H, n, D),
            dtype=_DTYPES[dtype],
            device=device,
            requires_grad=True,
        )
        for n in (L, S, S)
    )
    drop = _drop(L, S, "bool", "one")[:, None]
    if spelling == "bool":
        mask = lucid.tensor(~drop, device=device)
    else:
        mask = lucid.tensor(np.where(drop, -np.inf, 0.0), dtype=_DTYPES[dtype], device=device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
    out.sum().backward()
    o = out.to(lucid.float32).numpy()
    assert np.isfinite(o).all()
    assert (o[1, :, 1] == 0).all(), "the empty row did not come back zero"
    assert np.abs(o[1, :, 0]).max() > 0, "the probe is vacuous: a live row is zero"
    for name, t in (("query", q), ("key", k), ("value", v)):
        assert t.grad is not None
        g = t.grad.to(lucid.float32).numpy()
        assert np.isfinite(g).all(), f"{name} gradient is not finite"
    assert q.grad is not None
    assert (q.grad.to(lucid.float32).numpy()[1, :, 1] == 0).all()


def test_a_learnable_bias_gets_no_gradient_from_an_empty_row(device: str) -> None:
    rng = np.random.default_rng(1)
    q, k, v = (lucid.tensor(_draw(rng, "float32", 1, 1, n, D), device=device) for n in (L, S, S))
    bias_np = rng.standard_normal((L, S)).astype(np.float32)
    bias_np[1] = -np.inf
    bias = lucid.tensor(bias_np, device=device, requires_grad=True)
    F.scaled_dot_product_attention(q, k, v, attn_mask=bias).sum().backward()
    assert bias.grad is not None
    g = bias.grad.numpy()
    assert np.isfinite(g).all()
    assert (g[1] == 0).all()
    assert np.abs(g[0]).max() > 0


def test_a_finite_large_negative_is_not_a_mask_out(device: str) -> None:
    """``-1e9`` everywhere still normalises — to the mean of V, as in the reference."""
    rng = np.random.default_rng(2)
    q, k, v = (lucid.tensor(_draw(rng, "float32", 1, 1, n, D), device=device) for n in (L, S, S))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.full((L, S), -1e9, device=device))
    np.testing.assert_allclose(
        out.numpy()[0, 0], np.broadcast_to(v.numpy()[0, 0].mean(0), (L, D)), atol=1e-5
    )


@pytest.mark.parametrize("training", [False, True])
def test_mha_weights_of_a_padded_batch_are_zero(device: str, training: bool) -> None:
    """The weights path shares the rule: zero weights, not NaN.

    The reference's ``need_weights=True`` path runs a plain softmax and
    returns NaN here; its fused path returns zero.  Lucid answers zero on
    both, so asking for the weights cannot poison a padded batch.
    """
    mha = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.5).to(device)
    mha.train(training)
    x = lucid.randn(N, S, E, device=device, requires_grad=True)
    pad = lucid.tensor(_drop(S, S, "key_padding", "one")[:, 0], device=device)
    out, w = mha(x, x, x, key_padding_mask=pad, need_weights=True)
    out.sum().backward()
    assert w is not None
    assert np.isfinite(out.numpy()).all()
    assert (w.numpy()[1] == 0).all()
    assert x.grad is not None and np.isfinite(x.grad.numpy()).all()


# ── the matrix, against the reference ──────────────────────────────────────


def _sdpa_case(ref: Any, device: str, dtype: str, kind: str, rows: str) -> None:
    lq = S if kind == "causal_mask" else L
    rng = np.random.default_rng(3)
    arrays = [_draw(rng, dtype, N, H, n, D) for n in (lq, S, S)]
    upstream = rng.standard_normal((N, H, lq, D)).astype(np.float32)
    drop = _drop(lq, S, kind, rows)[:, None]
    if kind == "key_padding":
        drop = drop[:, :, :1]  # (N, 1, 1, S): one row per batch, broadcast
    causal = kind == "causal_mask"
    if kind == "float":
        mask_np: np.ndarray = np.where(drop, -np.inf, 0.0).astype(np.float32)
        ours_mask = lucid.tensor(mask_np, dtype=_DTYPES[dtype], device=device)
        ref_mask = ref.from_numpy(mask_np)
    else:
        ours_mask = lucid.tensor(~drop, device=device)
        ref_mask = ref.from_numpy(~drop)

    theirs = [ref.from_numpy(a).requires_grad_() for a in arrays]
    want = ref.nn.functional.scaled_dot_product_attention(
        *theirs, attn_mask=ref_mask, is_causal=causal
    )
    (want * ref.from_numpy(upstream)).sum().backward()

    ours = [
        lucid.tensor(a, dtype=_DTYPES[dtype], device=device, requires_grad=True) for a in arrays
    ]
    got = F.scaled_dot_product_attention(*ours, attn_mask=ours_mask, is_causal=causal)
    (got * lucid.tensor(upstream, dtype=_DTYPES[dtype], device=device)).sum().backward()

    _assert_close(got, want, dtype, "output")
    for name, o, t in zip(("query", "key", "value"), ours, theirs, strict=True):
        _assert_close(o.grad, t.grad, dtype, f"{name} gradient")


def _mha_case(ref: Any, device: str, dtype: str, kind: str, rows: str) -> None:
    rng = np.random.default_rng(4)
    theirs = ref.nn.MultiheadAttention(E, H, batch_first=True)
    with ref.no_grad():
        for p in theirs.parameters():
            p.copy_(ref.from_numpy(_draw(rng, dtype, *p.shape)))
    ours = nn.MultiheadAttention(E, H, batch_first=True).to(device, dtype=_DTYPES[dtype])
    state = theirs.state_dict()
    for mine, name in (
        ("in_proj_weight", "in_proj_weight"),
        ("in_proj_bias", "in_proj_bias"),
        ("out_proj_weight", "out_proj.weight"),
        ("out_proj_bias", "out_proj.bias"),
    ):
        getattr(ours, mine)._impl = lucid.tensor(
            state[name].numpy(), dtype=_DTYPES[dtype], device=device
        )._impl

    lq = S if kind == "causal_mask" else L
    arrays = [_draw(rng, dtype, N, n, E) for n in (lq, S, S)]
    upstream = rng.standard_normal((N, lq, E)).astype(np.float32)
    drop = _drop(lq, S, kind, rows)
    ours_kw: dict[str, Any] = {}
    ref_kw: dict[str, Any] = {}
    if kind == "key_padding":
        pad = drop[:, 0]
        ours_kw["key_padding_mask"] = lucid.tensor(pad, device=device)
        ref_kw["key_padding_mask"] = ref.from_numpy(pad)
    else:
        # MHA takes one (L, S) or per-head (N * H, L, S) mask, True = drop.
        per_head = np.repeat(drop, H, axis=0)
        if kind == "causal_mask":
            # The reference reads ``is_causal`` as a promise that the mask
            # already holds the triangle, so the triangle goes in the mask;
            # an all-False padding mask keeps it from dropping the mask.
            per_head = per_head | np.triu(np.ones((lq, S), dtype=bool), 1)
            ours_kw["is_causal"] = ref_kw["is_causal"] = True
            no_pad = np.zeros((N, S), dtype=bool)
            ours_kw["key_padding_mask"] = lucid.tensor(no_pad, device=device)
            ref_kw["key_padding_mask"] = ref.from_numpy(no_pad)
        if kind == "float":
            add = np.where(per_head, -np.inf, 0.0).astype(np.float32)
            ours_kw["attn_mask"] = lucid.tensor(add, dtype=_DTYPES[dtype], device=device)
            ref_kw["attn_mask"] = ref.from_numpy(add)
        else:
            ours_kw["attn_mask"] = lucid.tensor(per_head, device=device)
            ref_kw["attn_mask"] = ref.from_numpy(per_head)

    # ``need_weights=False`` is the reference's path that defines the rule;
    # its weights path returns NaN for an empty row (see the test above).
    ref_in = [ref.from_numpy(a).requires_grad_() for a in arrays]
    want, _ = theirs(*ref_in, need_weights=False, **ref_kw)
    (want * ref.from_numpy(upstream)).sum().backward()

    our_in = [
        lucid.tensor(a, dtype=_DTYPES[dtype], device=device, requires_grad=True) for a in arrays
    ]
    got, _ = ours(*our_in, need_weights=False, **ours_kw)
    (got * lucid.tensor(upstream, dtype=_DTYPES[dtype], device=device)).sum().backward()

    _assert_close(got, want, dtype, "output")
    for name, o, t in zip(("query", "key", "value"), our_in, ref_in, strict=True):
        _assert_close(o.grad, t.grad, dtype, f"{name} gradient")


@pytest.mark.parity
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("device", ["cpu", "metal"] if metal_available() else ["cpu"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("entry", ["sdpa", "mha"])
def test_matches_the_reference(
    ref: Any, entry: str, kind: str, device: str, dtype: str, rows: str
) -> None:
    case = _sdpa_case if entry == "sdpa" else _mha_case
    case(ref, device, dtype, kind, rows)
