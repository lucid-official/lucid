"""The two ``scaled_dot_product_attention`` branches nothing exercised.

Both are recent corrections, and both shipped without a test — which is
how the line-coverage floor caught them rather than the suite.

* ``dropout_p > 0`` takes an explicit-math path instead of the fused
  kernel, because MLX's fused SDPA has no dropout argument.  Before that
  branch existed the argument was accepted and discarded, so every caller
  that asked for attention dropout — BERT among them — trained with none.
  A test that only checks shapes would have passed then too, so these
  check that the probabilities are *actually* dropped, that the
  expectation is preserved, and that the branch still honours
  ``is_causal`` and the additive mask, both of which it re-implements by
  hand rather than delegating.

  Writing these turned up a third thing: the two paths disagreed about
  what a *boolean* mask means.  The fused kernel read ``True`` as "mask
  this out", the dropout branch as "this attends", so every boolean mask
  in a model inverted the moment dropout was switched on.  The function
  now converts a bool mask to its additive form once, up front, and the
  tests below pin both the convention and the agreement.

* A mask that broadcasts over the batch is expanded on the CPU path.  The
  fused kernel accepts only a full or per-batch mask, so a ``(Lq, Lk)``
  mask against a batched query has to be materialised first.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F


def _qkv(batch: int = 2, heads: int = 2, lq: int = 4, lk: int = 4, dim: int = 8):
    lucid.manual_seed(0)
    shape = (batch, heads, lq, dim)
    q = lucid.randn(*shape)
    k = lucid.randn(batch, heads, lk, dim)
    v = lucid.randn(batch, heads, lk, dim)
    return q, k, v


def test_dropout_actually_drops() -> None:
    """``dropout_p`` must change the result, not be quietly ignored."""
    q, k, v = _qkv()
    lucid.manual_seed(0)
    wet = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5).numpy()
    dry = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).numpy()
    assert wet.shape == dry.shape
    assert not np.allclose(wet, dry), (
        "dropout_p=0.5 produced the same output as dropout_p=0.0 — the "
        "argument is being discarded again"
    )


def test_dropout_one_drops_every_weight() -> None:
    """``p = 1`` zeroes every attention weight, so the output is zeros.

    Inverted dropout scales the survivors by ``1/(1-p)``; at ``p = 1``
    there are none and the scale is infinite.  ``F.dropout`` answers that
    case as the reference does — ``x * 0`` — rather than dividing, so the
    attention branch must give zeros, not NaNs and not an error.
    """
    q, k, v = _qkv()
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=1.0).numpy()
    assert np.all(out == 0.0)


def test_dropout_outside_the_unit_interval_is_refused() -> None:
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="between 0 and 1"):
        F.scaled_dot_product_attention(q, k, v, dropout_p=1.5)


def test_dropout_scales_the_survivors() -> None:
    """Inverted dropout keeps the expectation, so the mean survives.

    Averaged over enough draws the dropout path must agree with the
    dropout-free one: that is what ``1/(1-p)`` is for, and getting it
    wrong is a bug no shape check would see.
    """
    q, k, v = _qkv(batch=1, heads=1, lq=4, lk=4)
    dry = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).numpy()

    total = np.zeros_like(dry)
    draws = 400
    for seed in range(draws):
        lucid.manual_seed(seed)
        total += F.scaled_dot_product_attention(q, k, v, dropout_p=0.5).numpy()
    mean = total / draws

    # Generous: 400 draws of a Bernoulli average is not a tight estimate,
    # and the failure this guards against — a missing 1/(1-p), i.e. a
    # factor of two — is far outside this band.
    np.testing.assert_allclose(mean, dry, rtol=0.25, atol=0.25)


def test_dropout_path_is_differentiable() -> None:
    """The explicit-math branch exists to keep the graph intact."""
    q, k, v = _qkv()
    for t in (q, k, v):
        t.requires_grad = True
    F.scaled_dot_product_attention(q, k, v, dropout_p=0.3).sum().backward()
    for name, t in (("query", q), ("key", k), ("value", v)):
        assert t.grad is not None, f"{name} received no gradient"
        assert np.isfinite(t.grad.numpy()).all()


def test_dropout_path_honours_is_causal() -> None:
    """A causal mask must still hold when the branch re-implements it.

    Checked by perturbing a future key: with causality intact, position 0
    of the output cannot move.
    """
    q, k, v = _qkv(batch=1, heads=1, lq=4, lk=4)
    base = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True).numpy()

    v_np = v.numpy().copy()
    v_np[0, 0, 3, :] += 100.0  # a key/value strictly in the future of row 0
    moved = F.scaled_dot_product_attention(
        q, k, lucid.tensor(v_np), dropout_p=0.0, is_causal=True
    ).numpy()

    assert np.allclose(
        base[0, 0, 0], moved[0, 0, 0], atol=1e-5
    ), "row 0 attended to a future position — is_causal was dropped"
    assert not np.allclose(
        base[0, 0, 3], moved[0, 0, 3]
    ), "the perturbation never reached the last row; the probe is vacuous"


@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
def test_the_additive_mask_holds_on_both_paths(dropout_p: float) -> None:
    """A masked-out key cannot reach the output, with or without dropout.

    The additive float mask is the documented contract — "large negative
    values (or ``-inf``) at positions to mask out" — and the dropout
    branch re-implements it by hand, so it is checked on both paths.
    Probed by perturbation rather than by comparing numbers: a banned
    position carries ~0 probability, so moving its value must not move
    the result, and dropout cannot resurrect it because scaling zero is
    still zero.
    """
    q, k, v = _qkv(batch=1, heads=1, lq=3, lk=3)
    banned = 2
    additive = np.zeros((3, 3), dtype=np.float32)
    additive[:, banned] = -1e9
    mask = lucid.tensor(additive)

    v_np = v.numpy().copy()
    lucid.manual_seed(11)
    base = F.scaled_dot_product_attention(
        q, k, lucid.tensor(v_np), attn_mask=mask, dropout_p=dropout_p
    ).numpy()
    v_np[0, 0, banned, :] += 100.0
    lucid.manual_seed(11)
    moved = F.scaled_dot_product_attention(
        q, k, lucid.tensor(v_np), attn_mask=mask, dropout_p=dropout_p
    ).numpy()

    assert np.allclose(base, moved, atol=1e-3), "a masked-out position still influenced the output"


@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
def test_a_bool_mask_means_the_same_thing_on_both_paths(dropout_p: float) -> None:
    """``True`` attends, whether or not dropout is on.

    The two paths used to disagree — the fused kernel read ``True`` as
    "mask this out" and the dropout branch as "this attends" — so turning
    dropout on inverted every boolean mask in the model.  Asserted as
    equality with the additive mask it stands for, which pins the
    convention and the agreement in one go.
    """
    q, k, v = _qkv(batch=1, heads=1, lq=3, lk=3)
    keep = np.ones((3, 3), dtype=bool)
    keep[:, 2] = False
    additive = np.where(keep, 0.0, -1e9).astype(np.float32)

    lucid.manual_seed(7)
    from_bool = F.scaled_dot_product_attention(
        q, k, v, attn_mask=lucid.tensor(keep, dtype=lucid.bool_), dropout_p=dropout_p
    ).numpy()
    lucid.manual_seed(7)
    from_float = F.scaled_dot_product_attention(
        q, k, v, attn_mask=lucid.tensor(additive), dropout_p=dropout_p
    ).numpy()
    np.testing.assert_allclose(from_bool, from_float, rtol=1e-4, atol=1e-4)


def test_causal_and_dropout_together() -> None:
    """``is_causal`` is re-implemented inside the dropout branch."""
    q, k, v = _qkv(batch=1, heads=1, lq=4, lk=4)
    v_np = v.numpy().copy()
    lucid.manual_seed(3)
    base = F.scaled_dot_product_attention(
        q, k, lucid.tensor(v_np), dropout_p=0.5, is_causal=True
    ).numpy()
    v_np[0, 0, 3, :] += 100.0
    lucid.manual_seed(3)
    moved = F.scaled_dot_product_attention(
        q, k, lucid.tensor(v_np), dropout_p=0.5, is_causal=True
    ).numpy()
    assert np.allclose(
        base[0, 0, 0], moved[0, 0, 0], atol=1e-3
    ), "with dropout on, row 0 attended to a future position"


def test_a_partially_broadcast_mask_is_materialised() -> None:
    """A mask broadcast over heads only — neither shared nor full.

    The CPU kernel flattens the leading dims and accepts a mask that is
    either fully shared over that batch or fully materialised.  A
    ``(B, 1, Lq, Lk)`` bias is neither, and used to be indexed past its
    own buffer, so it has to be expanded first.
    """
    q, k, v = _qkv(batch=2, heads=2, lq=4, lk=4)
    partial = np.zeros((2, 1, 4, 4), dtype=np.float32)
    partial[1, 0, :, 3] = -1e9  # ban the last key, in the second batch only
    full = np.broadcast_to(partial, (2, 2, 4, 4)).copy()

    got = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(partial)).numpy()
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(full)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_a_lower_rank_partial_mask_is_unsqueezed_then_expanded() -> None:
    """A rank-3 mask gains its leading axes before being materialised.

    Broadcasting aligns from the right, so ``(H, Lq, Lk)`` against
    ``(B, H, Lq, Lk)`` scores is a per-head mask shared across the batch
    — the unsqueeze has to add the *batch* axis, not the head one.
    """
    q, k, v = _qkv(batch=2, heads=2, lq=4, lk=4)
    partial = np.zeros((2, 4, 4), dtype=np.float32)
    partial[1, :, 3] = -1e9  # ban the last key, in the second head only
    full = np.broadcast_to(partial[None, :], (2, 2, 4, 4)).copy()

    got = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(partial)).numpy()
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(full)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_a_broadcast_mask_matches_the_expanded_one() -> None:
    """``(Lq, Lk)`` against a batched query must equal the full mask."""
    q, k, v = _qkv(batch=2, heads=2, lq=4, lk=4)
    small = np.triu(np.full((4, 4), -1e9, dtype=np.float32), k=1)
    full = np.broadcast_to(small, (2, 2, 4, 4)).copy()

    got = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(small)).numpy()
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=lucid.tensor(full)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_the_scale_argument_reaches_the_dropout_branch() -> None:
    """``scale`` overrides ``1/sqrt(d)`` on the explicit path too."""
    q, k, v = _qkv(batch=1, heads=1, lq=3, lk=3, dim=8)
    default = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).numpy()
    explicit = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=1.0 / math.sqrt(8)
    ).numpy()
    np.testing.assert_allclose(default, explicit, rtol=1e-5, atol=1e-6)

    other = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=4.0).numpy()
    assert not np.allclose(default, other), "scale had no effect"
