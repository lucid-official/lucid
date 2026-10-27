"""The losses, against the formulas that define them.

``nn/functional/loss.py`` sat at 62.8% and ``nn/modules/loss.py`` at
63.2%.  A loss is the objective: if it is off by a factor, training still
runs, still converges, and converges to the wrong place.  Nothing
downstream can tell, because the loss *is* the thing everything else is
measured against.

Two defects were here, both invisible at the default arguments — which
is why they lasted.  So the parameters are swept rather than left at
their defaults, and each expectation is written out from the definition
rather than taken from the implementation.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

RNG = np.random.default_rng(0)
X = RNG.standard_normal((6, 4)).astype(np.float32)
Y = RNG.standard_normal((6, 4)).astype(np.float32)
LOGITS = RNG.standard_normal((6, 5)).astype(np.float32)
CLASSES = RNG.integers(0, 5, 6)
PROBS = (RNG.random((6, 4)) * 0.8 + 0.1).astype(np.float32)
BINARY = (RNG.random((6, 4)) > 0.5).astype(np.float32)


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float32))


def _i(a):
    return lucid.tensor(np.asarray(a, dtype=np.int32), dtype=lucid.int32)


def _v(x):
    return np.asarray(x.numpy())


def _f(x):
    return float(_v(x))


# ── the elementwise regressions ───────────────────────────────────────────────


def test_mse_is_the_mean_square():
    assert np.isclose(_f(F.mse_loss(_t(X), _t(Y))), ((X - Y) ** 2).mean(), atol=1e-6)


def test_l1_is_the_mean_absolute():
    assert np.isclose(_f(F.l1_loss(_t(X), _t(Y))), np.abs(X - Y).mean(), atol=1e-6)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_the_reductions_are_what_they_say(reduction):
    got = _v(F.mse_loss(_t(X), _t(Y), reduction=reduction))
    squared = (X - Y) ** 2
    if reduction == "mean":
        assert np.isclose(got, squared.mean(), atol=1e-6)
    elif reduction == "sum":
        assert np.isclose(got, squared.sum(), atol=1e-4)
    else:
        assert got.shape == squared.shape
        assert np.allclose(got, squared, atol=1e-6)


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_is_quadratic_inside_delta_and_linear_outside(delta):
    diff = X - Y
    want = np.where(np.abs(diff) < delta, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    assert np.allclose(
        _v(F.huber_loss(_t(X), _t(Y), delta=delta, reduction="none")), want, atol=1e-6
    )


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_smooth_l1_divides_the_quadratic_region_by_beta(beta):
    """The defect, and why it survived.

    Smooth L1 is ``0.5 x²/beta`` inside the transition and
    ``|x| - 0.5·beta`` outside; Huber is ``0.5 x²`` and
    ``beta(|x| - 0.5 beta)``.  The two differ by a factor of ``beta``
    *everywhere*, and this was implemented as a bare call to Huber.

    At ``beta = 1`` — the default — the factor is 1 and the answer is
    right, so every test that used the default passed.  Every other
    ``beta`` was scaled by it, silently: a detection head with
    ``beta = 1/9`` was optimising an objective nine times smaller than
    the one it asked for.
    """
    diff = X - Y
    want = np.where(np.abs(diff) < beta, 0.5 * diff**2 / beta, np.abs(diff) - 0.5 * beta)
    assert np.allclose(
        _v(F.smooth_l1_loss(_t(X), _t(Y), beta=beta, reduction="none")),
        want,
        atol=1e-6,
    )


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_meets_itself_at_the_transition(beta):
    """What the ``1/beta`` is for: the quadratic and the linear branch have
    to agree in value *and* slope where they meet, or the loss has a kink
    the optimiser feels.

    Asserted by halving the step and watching the gap halve with it — a
    fixed tolerance would either pass a discontinuity or fail on the
    step itself.
    """

    def at(value):
        return _f(F.smooth_l1_loss(_t([value]), _t([0.0]), beta=beta))

    gaps = []
    for step in (1e-3, 5e-4, 2.5e-4):
        gaps.append(abs(at(beta + step) - at(beta - step)))
    # Same slope on both sides, so the gap is ``2·step·slope`` and halves
    # with the step.  A value discontinuity would leave a constant floor.
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.6 * gaps[0]
    # The slope it meets at is 1 — that is what makes ``beta`` a
    # transition point rather than an overall scale, and it is exactly
    # what the missing ``1/beta`` used to destroy.
    slope = (at(beta + 1e-2) - at(beta)) / 1e-2
    assert np.isclose(slope, 1.0, atol=1e-3)
    # Value at the transition: both branches give ``0.5·beta`` there.
    assert np.isclose(at(beta), 0.5 * beta, atol=1e-5)


def test_smooth_l1_at_beta_zero_is_l1():
    """The degenerate limit, and the one where dividing by ``beta`` would
    otherwise be a division by zero."""
    assert np.allclose(
        _v(F.smooth_l1_loss(_t(X), _t(Y), beta=0.0, reduction="none")),
        np.abs(X - Y),
        atol=1e-6,
    )


def test_smooth_l1_at_beta_one_is_huber_at_delta_one():
    assert np.allclose(
        _v(F.smooth_l1_loss(_t(X), _t(Y), beta=1.0, reduction="none")),
        _v(F.huber_loss(_t(X), _t(Y), delta=1.0, reduction="none")),
        atol=1e-7,
    )


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_the_smooth_l1_gradient_follows_the_value(beta):
    """A scale error in the value is a scale error in the gradient, which
    is what actually reaches the weights."""
    x = lucid.tensor(X, requires_grad=True)
    F.smooth_l1_loss(x, _t(Y), beta=beta, reduction="sum").backward()
    diff = X - Y
    want = np.where(np.abs(diff) < beta, diff / beta, np.sign(diff))
    assert np.allclose(_v(x.grad), want, atol=1e-5)


# ── the classification losses ─────────────────────────────────────────────────


def _log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def test_cross_entropy_is_the_negative_log_softmax_of_the_true_class():
    logp = _log_softmax(LOGITS)
    want = -logp[np.arange(len(CLASSES)), CLASSES].mean()
    assert np.isclose(_f(F.cross_entropy(_t(LOGITS), _i(CLASSES))), want, atol=1e-5)


def test_cross_entropy_weights_each_class_and_normalises_by_the_weights():
    """The subtle half: the divisor is the sum of the *used* weights, not
    the batch size.  Dividing by N instead scales the loss by the mean
    weight and is invisible whenever the weights average to one."""
    weights = (RNG.random(5) + 0.5).astype(np.float32)
    logp = _log_softmax(LOGITS)
    picked = -logp[np.arange(len(CLASSES)), CLASSES]
    w = weights[CLASSES]
    want = (picked * w).sum() / w.sum()
    assert np.isclose(
        _f(F.cross_entropy(_t(LOGITS), _i(CLASSES), weight=_t(weights))),
        want,
        atol=1e-5,
    )


def test_cross_entropy_skips_the_ignored_index_entirely():
    targets = CLASSES.copy()
    targets[2] = -100
    kept = [i for i in range(len(targets)) if targets[i] != -100]
    logp = _log_softmax(LOGITS)
    want = -logp[kept, targets[kept]].mean()
    assert np.isclose(
        _f(F.cross_entropy(_t(LOGITS), _i(targets), ignore_index=-100)),
        want,
        atol=1e-5,
    )


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5])
def test_label_smoothing_mixes_in_the_uniform_target(smoothing):
    logp = _log_softmax(LOGITS)
    picked = -logp[np.arange(len(CLASSES)), CLASSES]
    uniform = -logp.mean(axis=1)
    want = ((1 - smoothing) * picked + smoothing * uniform).mean()
    assert np.isclose(
        _f(F.cross_entropy(_t(LOGITS), _i(CLASSES), label_smoothing=smoothing)),
        want,
        atol=1e-5,
    )


def test_nll_loss_expects_log_probabilities_already():
    logp = _log_softmax(LOGITS)
    want = -logp[np.arange(len(CLASSES)), CLASSES].mean()
    assert np.isclose(_f(F.nll_loss(_t(logp), _i(CLASSES))), want, atol=1e-5)


def test_cross_entropy_is_log_softmax_then_nll():
    """Two entry points for one objective; if they disagree, one is
    wrong and no single test of either would say so."""
    assert np.isclose(
        _f(F.cross_entropy(_t(LOGITS), _i(CLASSES))),
        _f(F.nll_loss(_t(_log_softmax(LOGITS)), _i(CLASSES))),
        atol=1e-6,
    )


def test_binary_cross_entropy_is_the_bernoulli_log_likelihood():
    want = -(BINARY * np.log(PROBS) + (1 - BINARY) * np.log(1 - PROBS)).mean()
    assert np.isclose(_f(F.binary_cross_entropy(_t(PROBS), _t(BINARY))), want, atol=1e-5)


def test_bce_with_logits_matches_sigmoid_then_bce():
    logits = X
    probs = 1.0 / (1.0 + np.exp(-logits))
    assert np.isclose(
        _f(nn.BCEWithLogitsLoss()(_t(logits), _t(BINARY))),
        _f(F.binary_cross_entropy(_t(probs), _t(BINARY))),
        atol=1e-5,
    )


def test_bce_with_logits_survives_a_saturated_logit():
    """The reason it exists as its own function: ``sigmoid`` then ``log``
    loses the answer at ±40, the fused form does not."""
    got = _v(nn.BCEWithLogitsLoss(reduction="none")(_t([[-60.0, 60.0]]), _t([[0.0, 1.0]])))
    assert np.isfinite(got).all()
    assert np.allclose(got, 0.0, atol=1e-6)


# ── Poisson ───────────────────────────────────────────────────────────────────


def test_poisson_nll_is_exp_minus_target_times_input():
    target = np.abs(Y) + 0.1
    want = (np.exp(X) - target * X).mean()
    assert np.isclose(_f(F.poisson_nll_loss(_t(X), _t(target))), want, atol=1e-5)


def test_poisson_nll_takes_the_rate_directly_when_told_to():
    rate = np.abs(X) + 0.1
    target = np.abs(Y) + 0.1
    want = (rate - target * np.log(rate + 1e-8)).mean()
    assert np.isclose(
        _f(F.poisson_nll_loss(_t(rate), _t(target), log_input=False)), want, atol=1e-5
    )


def test_full_adds_the_stirling_term_rather_than_nothing():
    """``full`` was accepted and changed the answer by exactly zero.

    It names the ``log(target!)`` the short form drops as constant in the
    parameters, approximated by Stirling as
    ``target·log(target) - target + ½log(2π·target)`` — and only where
    ``target > 1``, since ``log(0!) = log(1!) = 0`` and the
    approximation is not.
    """
    target = np.array([[0.0, 1.0, 2.0, 5.0]], dtype=np.float32)
    inputs = np.array([[0.5, 0.5, 1.0, 1.5]], dtype=np.float32)

    short = np.exp(inputs) - target * inputs
    stirling = np.where(
        target > 1,
        target * np.log(np.maximum(target, 1.0))
        - np.maximum(target, 1.0)
        + 0.5 * np.log(2 * math.pi * np.maximum(target, 1.0)),
        0.0,
    )

    without = _v(F.poisson_nll_loss(_t(inputs), _t(target), reduction="none"))
    with_full = _v(F.poisson_nll_loss(_t(inputs), _t(target), full=True, reduction="none"))
    assert np.allclose(without, short, atol=1e-5)
    assert np.allclose(with_full, short + stirling, atol=1e-5)
    assert not np.allclose(without, with_full)


def test_the_stirling_term_is_zero_at_target_zero_and_one():
    target = np.array([[0.0, 1.0]], dtype=np.float32)
    inputs = np.array([[0.3, 0.7]], dtype=np.float32)
    assert np.allclose(
        _v(F.poisson_nll_loss(_t(inputs), _t(target), reduction="none")),
        _v(F.poisson_nll_loss(_t(inputs), _t(target), full=True, reduction="none")),
        atol=1e-7,
    )


def test_full_stays_finite_at_target_zero():
    """``target·log(target)`` is where a naive Stirling term produces a
    NaN and poisons the whole batch."""
    got = _v(F.poisson_nll_loss(_t([[0.5, 0.5]]), _t([[0.0, 0.0]]), full=True, reduction="none"))
    assert np.isfinite(got).all()


def test_the_full_gradient_is_unchanged_by_a_term_that_does_not_involve_the_input():
    """Stirling depends only on the target, so it shifts the value and
    must leave the gradient alone."""
    target = np.abs(Y) + 1.5
    grads = []
    for full in (False, True):
        x = lucid.tensor(X, requires_grad=True)
        F.poisson_nll_loss(x, _t(target), full=full, reduction="sum").backward()
        grads.append(_v(x.grad))
    assert np.allclose(grads[0], grads[1], atol=1e-6)


# ── KL divergence ─────────────────────────────────────────────────────────────


def test_kl_div_batchmean_divides_by_the_batch_and_not_the_support():
    logp = np.log(PROBS)
    want = (PROBS * (np.log(PROBS) - logp)).sum() / PROBS.shape[0]
    assert np.isclose(_f(F.kl_div(_t(logp), _t(PROBS), reduction="batchmean")), want, atol=1e-5)


def test_kl_div_of_a_distribution_with_itself_is_zero():
    logp = np.log(PROBS / PROBS.sum(axis=1, keepdims=True))
    probs = np.exp(logp)
    assert abs(_f(F.kl_div(_t(logp), _t(probs), reduction="batchmean"))) < 1e-5


def test_kl_div_accepts_a_log_target():
    logp = np.log(PROBS)
    assert abs(_f(F.kl_div(_t(logp), _t(logp), log_target=True))) < 1e-6


# ── the module wrappers agree with the functions ───────────────────────────────


MODULE_PAIRS = [
    ("MSELoss", lambda: nn.MSELoss(), lambda a, b: F.mse_loss(a, b)),
    ("L1Loss", lambda: nn.L1Loss(), lambda a, b: F.l1_loss(a, b)),
    (
        "HuberLoss",
        lambda: nn.HuberLoss(delta=2.0),
        lambda a, b: F.huber_loss(a, b, delta=2.0),
    ),
    (
        "SmoothL1Loss",
        lambda: nn.SmoothL1Loss(beta=0.5),
        lambda a, b: F.smooth_l1_loss(a, b, beta=0.5),
    ),
    (
        "SoftMarginLoss",
        lambda: nn.SoftMarginLoss(),
        lambda a, b: F.soft_margin_loss(a, b),
    ),
]


@pytest.mark.parametrize(
    "build,call", [(p[1], p[2]) for p in MODULE_PAIRS], ids=[p[0] for p in MODULE_PAIRS]
)
def test_a_loss_module_is_its_function(build, call):
    """The wrapper must not quietly drop the arguments it was given —
    ``SmoothL1Loss(beta=0.5)`` forwarding a default ``beta`` would be
    exactly the defect this file opened with, one level up."""
    signs = np.where(BINARY > 0, 1.0, -1.0).astype(np.float32)
    left, right = _t(X), _t(signs if "Margin" in str(build) else Y)
    assert np.isclose(_f(build()(left, right)), _f(call(left, right)), atol=1e-6)


def test_the_smooth_l1_module_forwards_its_beta():
    """Directly, because the parametrised check above would pass on a
    module that ignored ``beta`` if the function did too."""
    assert not np.isclose(
        _f(nn.SmoothL1Loss(beta=0.5)(_t(X), _t(Y))),
        _f(nn.SmoothL1Loss(beta=2.0)(_t(X), _t(Y))),
    )


def test_the_poisson_module_forwards_full():
    target = np.abs(Y) + 1.5
    assert not np.isclose(
        _f(nn.PoissonNLLLoss(full=False)(_t(X), _t(target))),
        _f(nn.PoissonNLLLoss(full=True)(_t(X), _t(target))),
    )


# ── every loss trains ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "loss",
    [
        lambda p, t: F.mse_loss(p, t),
        lambda p, t: F.l1_loss(p, t),
        lambda p, t: F.huber_loss(p, t, delta=0.5),
        lambda p, t: F.smooth_l1_loss(p, t, beta=0.5),
        lambda p, t: F.smooth_l1_loss(p, t, beta=4.0),
    ],
    ids=["mse", "l1", "huber", "smooth-l1-small-beta", "smooth-l1-large-beta"],
)
def test_a_regression_loss_descends(loss):
    lucid.manual_seed(0)
    model = nn.Linear(4, 4)
    optimiser = lucid.optim.SGD(model.parameters(), lr=0.05)
    inputs, targets = _t(X), _t(Y)
    first = _f(loss(model(inputs), targets))
    for _ in range(30):
        optimiser.zero_grad()
        loss(model(inputs), targets).backward()
        optimiser.step()
    assert _f(loss(model(inputs), targets)) < first


# ── the target dtype a caller actually has ────────────────────────────────────


@pytest.mark.parametrize(
    "dtype",
    [lucid.int64, lucid.int32, lucid.int16],
    ids=["int64-the-default", "int32", "int16"],
)
def test_multilabel_margin_accepts_any_integer_target(dtype):
    """``lucid.tensor`` of ints gives int64, and int64 used to raise.

    The index columns are compared against int32 constants inside the
    loss, so a target that was not already int32 died on a dtype
    mismatch — meaning the loss rejected the dtype its own callers
    naturally produce, and only the docstring's explicit
    ``dtype=lucid.int32`` ever worked.
    """
    scores = lucid.tensor(np.array([[1.0, 0.5, -0.3, 0.2]], dtype=np.float32))
    target = lucid.tensor(np.array([[0, 1, -1, -1]], dtype=np.int32), dtype=dtype)
    # Positives {0,1}, negatives {2,3}: hinges 0, 0.2, 0.2, 0.7 over C=4.
    assert _f(F.multilabel_margin_loss(scores, target)) == pytest.approx(1.1 / 4, rel=1e-6)


def test_multilabel_margin_1d_accepts_a_default_int_target():
    scores = lucid.tensor(np.array([1.0, 0.5, -0.3, 0.2], dtype=np.float32))
    target = lucid.tensor(np.array([0, 1, -1, -1], dtype=np.int64), dtype=lucid.int64)
    assert _f(F.multilabel_margin_loss(scores, target)) == pytest.approx(1.1 / 4, rel=1e-6)
