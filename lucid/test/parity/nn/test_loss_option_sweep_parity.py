"""Every loss, across the options that select a code path — the reference half.

``nn/functional/loss.py`` sat at 35.4%, and the missing lines were the
options rather than the formulas: the three reductions, per-class
``weight``, ``label_smoothing``, ``ignore_index``, and the logit-space
variants.  The default call ran; nothing else did.

This file holds the checks against the reference, for every option it
shares; the checks against the definition written out, which need no
oracle, stay in ``lucid/test/unit/nn/test_loss_option_sweep.py`` so the
fast tier runs them.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

pytestmark = pytest.mark.parity

RNG = np.random.default_rng(0)
LOGITS = RNG.standard_normal((6, 4))
TARGET = np.array([0, 3, 1, 2, 3, 0])
A = RNG.standard_normal((6, 4))
B = RNG.standard_normal((6, 4))
PROB = RNG.uniform(0.1, 0.9, (6, 4))
REDUCTIONS = ["mean", "sum", "none"]


def _t(arr, **kw):
    return lucid.tensor(arr.copy(), **kw)


def _v(x):
    return np.asarray(x.numpy())


# ── the regression losses, every reduction ────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize(
    "name,fn",
    [
        ("mse_loss", F.mse_loss),
        ("l1_loss", F.l1_loss),
        ("smooth_l1_loss", F.smooth_l1_loss),
        ("huber_loss", F.huber_loss),
    ],
)
def test_regression_loss_reductions(name, fn, reduction, ref):
    got = _v(fn(_t(A), _t(B), reduction=reduction))
    want = np.asarray(
        getattr(ref.nn.functional, name)(
            ref.from_numpy(A.copy()), ref.from_numpy(B.copy()), reduction=reduction
        ).tolist()
    )
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-6)


# ── classification ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_cross_entropy_reductions(reduction, ref):
    got = _v(F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), reduction=reduction))
    want = np.asarray(
        ref.nn.functional.cross_entropy(
            ref.from_numpy(LOGITS.copy()), ref.from_numpy(TARGET).long(), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


def test_cross_entropy_with_class_weights(ref):
    weight = np.array([0.5, 2.0, 1.0, 0.25])
    got = float(
        F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), weight=_t(weight)).item()
    )
    want = float(
        ref.nn.functional.cross_entropy(
            ref.from_numpy(LOGITS.copy()),
            ref.from_numpy(TARGET).long(),
            weight=ref.from_numpy(weight),
        )
    )
    assert np.isclose(got, want, atol=1e-6)


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_cross_entropy_label_smoothing(smoothing, ref):
    got = float(
        F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), label_smoothing=smoothing).item()
    )
    want = float(
        ref.nn.functional.cross_entropy(
            ref.from_numpy(LOGITS.copy()), ref.from_numpy(TARGET).long(), label_smoothing=smoothing
        )
    )
    assert np.isclose(got, want, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_nll_loss_reductions(reduction, ref):
    log_probs = np.log(np.exp(LOGITS) / np.exp(LOGITS).sum(axis=1, keepdims=True))
    got = _v(F.nll_loss(_t(log_probs), _t(TARGET, dtype=lucid.int32), reduction=reduction))
    want = np.asarray(
        ref.nn.functional.nll_loss(
            ref.from_numpy(log_probs), ref.from_numpy(TARGET).long(), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


# ── binary ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_binary_cross_entropy_reductions(reduction, ref):
    target = (RNG.uniform(size=(6, 4)) > 0.5).astype(np.float64)
    got = _v(F.binary_cross_entropy(_t(PROB), _t(target), reduction=reduction))
    want = np.asarray(
        ref.nn.functional.binary_cross_entropy(
            ref.from_numpy(PROB.copy()), ref.from_numpy(target), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


# ── pair and margin losses ────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize(
    "name,fn",
    [("hinge_embedding_loss", F.hinge_embedding_loss), ("soft_margin_loss", F.soft_margin_loss)],
)
def test_sign_target_losses(name, fn, reduction, ref):
    sign = np.where(RNG.uniform(size=(6, 4)) > 0.5, 1.0, -1.0)
    got = _v(fn(_t(A), _t(sign), reduction=reduction))
    want = np.asarray(
        getattr(ref.nn.functional, name)(
            ref.from_numpy(A.copy()), ref.from_numpy(sign), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_margin_ranking_loss(reduction, ref):
    sign = np.where(RNG.uniform(size=(6,)) > 0.5, 1.0, -1.0)
    x1, x2 = A[:, 0], B[:, 0]
    got = _v(F.margin_ranking_loss(_t(x1), _t(x2), _t(sign), reduction=reduction))
    want = np.asarray(
        ref.nn.functional.margin_ranking_loss(
            ref.from_numpy(x1.copy()),
            ref.from_numpy(x2.copy()),
            ref.from_numpy(sign),
            reduction=reduction,
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("margin", [0.5, 1.0, 2.5])
def test_triplet_margin_loss_margin(margin, ref):
    anchor, positive, negative = A, B, RNG.standard_normal((6, 4))
    got = float(F.triplet_margin_loss(_t(anchor), _t(positive), _t(negative), margin=margin).item())
    want = float(
        ref.nn.functional.triplet_margin_loss(
            ref.from_numpy(anchor.copy()),
            ref.from_numpy(positive.copy()),
            ref.from_numpy(negative),
            margin=margin,
        )
    )
    assert np.isclose(got, want, atol=1e-6)


# ── distribution losses ───────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_kl_div_reductions(reduction, ref):
    log_input = np.log(PROB / PROB.sum(axis=1, keepdims=True))
    target = PROB / PROB.sum(axis=1, keepdims=True)
    got = _v(F.kl_div(_t(log_input), _t(target), reduction=reduction))
    want = np.asarray(
        ref.nn.functional.kl_div(
            ref.from_numpy(log_input), ref.from_numpy(target), reduction=reduction
        ).tolist()
    )
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("log_input", [True, False])
def test_poisson_nll_loss(log_input, ref):
    rate = PROB if not log_input else np.log(PROB)
    target = np.abs(RNG.standard_normal((6, 4)))
    got = float(F.poisson_nll_loss(_t(rate), _t(target), log_input=log_input).item())
    want = float(
        ref.nn.functional.poisson_nll_loss(
            ref.from_numpy(rate), ref.from_numpy(target), log_input=log_input
        )
    )
    assert np.isclose(got, want, atol=1e-5)


def test_gaussian_nll_loss(ref):
    var = np.abs(RNG.standard_normal((6, 4))) + 0.5
    got = float(F.gaussian_nll_loss(_t(A), _t(B), _t(var)).item())
    want = float(
        ref.nn.functional.gaussian_nll_loss(
            ref.from_numpy(A.copy()), ref.from_numpy(B.copy()), ref.from_numpy(var)
        )
    )
    assert np.isclose(got, want, atol=1e-6)
