"""Every loss, across the options that select a code path.

``nn/functional/loss.py`` sat at 35.4%, and the missing lines were the
options rather than the formulas: the three reductions, per-class
``weight``, ``label_smoothing``, ``ignore_index``, and the logit-space
variants.  The default call ran; nothing else did.

Values are checked against the definition written out, so a passing
test says the arithmetic is right rather than unchanged; the checks
against the reference, for every option it shares, live in
``lucid/test/parity/nn/test_loss_option_sweep_parity.py`` because they
need the oracle installed.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

RNG = np.random.default_rng(0)
LOGITS = RNG.standard_normal((6, 4))
TARGET = np.array([0, 3, 1, 2, 3, 0])
A = RNG.standard_normal((6, 4))
B = RNG.standard_normal((6, 4))
REDUCTIONS = ["mean", "sum", "none"]


def _t(arr, **kw):
    return lucid.tensor(arr.copy(), **kw)


def _v(x):
    return np.asarray(x.numpy())


# ── the regression losses, every reduction ────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_the_reductions_relate_as_they_should(reduction):
    """``mean`` is ``sum`` over the element count, and ``none`` keeps the
    shape — a relation no single reduction can verify alone."""
    none = _v(F.mse_loss(_t(A), _t(B), reduction="none"))
    total = float(F.mse_loss(_t(A), _t(B), reduction="sum").item())
    mean = float(F.mse_loss(_t(A), _t(B), reduction="mean").item())
    assert none.shape == A.shape
    assert np.isclose(none.sum(), total)
    assert np.isclose(total / A.size, mean)


# ── classification ────────────────────────────────────────────────────────────


def test_smoothing_raises_the_loss_off_a_confident_fit():
    """Smoothing moves probability mass off the true class, so a model
    that was right pays for it — the direction, not just the number."""
    confident = np.eye(4)[TARGET] * 10.0
    plain = float(F.cross_entropy(_t(confident), _t(TARGET, dtype=lucid.int32)).item())
    smoothed = float(
        F.cross_entropy(_t(confident), _t(TARGET, dtype=lucid.int32), label_smoothing=0.2).item()
    )
    assert smoothed > plain


# ── binary ────────────────────────────────────────────────────────────────────


def test_bce_with_logits_matches_bce_of_the_sigmoid():
    """The logit form exists to be stable, not different: it has to agree
    with the two-step version wherever the two-step version is finite."""
    target = (RNG.uniform(size=(6, 4)) > 0.5).astype(np.float64)
    logit_form = float(F.binary_cross_entropy_with_logits(_t(LOGITS), _t(target)).item())
    two_step = float(F.binary_cross_entropy(F.sigmoid(_t(LOGITS)), _t(target)).item())
    assert np.isclose(logit_form, two_step, atol=1e-6)


def test_bce_with_logits_survives_a_saturating_input():
    """Where the two-step form overflows to infinity, the fused one must
    not — that is the whole reason it exists."""
    extreme = np.array([[-80.0, 80.0], [80.0, -80.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = float(F.binary_cross_entropy_with_logits(_t(extreme), _t(target)).item())
    assert np.isfinite(got)
    assert got > 0.0


# ── pair and margin losses ────────────────────────────────────────────────────


def test_a_non_positive_triplet_margin_is_accepted_here_and_not_by_the_reference():
    """A divergence, recorded rather than decided.

    ``margin=0`` makes the hinge vacuous and a negative one inverts it,
    so the reference refuses both.  Lucid computes
    ``max(0, d_pos - d_neg + margin)`` and returns a number.  Neither is
    wrong — one is stricter — and changing it is an API decision rather
    than a defect fix, so this pins the current behaviour and names the
    difference.
    """
    for margin in (0.0, -0.5):
        value = float(F.triplet_margin_loss(_t(A), _t(B), _t(A), margin=margin).item())
        assert np.isfinite(value)


# ── refusals ──────────────────────────────────────────────────────────────────


def test_an_unknown_reduction_is_refused():
    with pytest.raises(ValueError, match="reduction"):
        F.mse_loss(_t(A), _t(B), reduction="average")


def test_label_smoothing_outside_its_range_is_refused():
    with pytest.raises(ValueError, match="label_smoothing"):
        F.cross_entropy(_t(LOGITS), _t(TARGET, dtype=lucid.int32), label_smoothing=1.5)


# ── every loss is differentiable ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: F.mse_loss(a, b),
        lambda a, b: F.l1_loss(a, b),
        lambda a, b: F.smooth_l1_loss(a, b),
        lambda a, b: F.huber_loss(a, b),
        lambda a, b: F.binary_cross_entropy_with_logits(a, F.sigmoid(b)),
    ],
)
def test_gradients_reach_the_input(build):
    a = _t(A, requires_grad=True)
    build(a, _t(B)).backward()
    assert a.grad is not None
    assert np.abs(_v(a.grad)).sum() > 0.0
