"""Regression tests: moments outside the range their formula is valid on.

Found 2026-08-04 by the audit's ``distribution`` axis, after it stopped
treating a non-finite moment as a defect.  That check was answering the
wrong question — whether a moment is finite is a property of the
*parameters*, and the axis did not choose them: Cauchy has no mean at any
parameter, Pareto none for α ≤ 1, StudentT no variance for ν ≤ 2.  The
reference returns NaN and infinity for exactly those, because a divergent
integral is an answer.

What it checks instead holds of every distribution at every parameter: a
variance is not negative, and a standard deviation is its square root.
Both survive infinity — and both caught real defects immediately, because
three distributions evaluated their closed form outside the range it is
valid on and returned a *negative* variance rather than the divergence it
stands for.
"""

import numpy as np
import pytest

import lucid
import lucid.distributions as D


def _T(*values: float) -> lucid.Tensor:
    return lucid.tensor(np.array(values, dtype=np.float64))


def _scalar(value: object) -> float:
    return float(np.asarray(value.numpy(), dtype=np.float64).ravel()[0])


# ── the closed form evaluated where it does not hold ─────────────────────────


@pytest.mark.parametrize(
    "df,expected",
    [
        (0.5, "nan"),  # no second moment, and no sign to diverge in
        (1.0, "nan"),
        (1.5, "inf"),  # diverges, but upward
        (2.0, "inf"),
        (3.0, 3.0),  # ν / (ν - 2)
        (10.0, 1.25),
    ],
)
def test_student_t_variance_across_its_thresholds(df: float, expected) -> None:
    """``ν/(ν−2)`` is only the answer above ν = 2.

    Evaluated anyway it changes sign rather than diverging: -0.333 at
    ν = 0.5, which is not a number any distribution has.
    """
    got = _scalar(D.StudentT(df=_T(df)).variance)
    if expected == "nan":
        assert np.isnan(got), got
    elif expected == "inf":
        assert np.isposinf(got), got
    else:
        assert np.isclose(got, expected), got


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
def test_pareto_moments_never_go_negative(alpha: float) -> None:
    """Below α = 1 the mean diverges; below α = 2 the variance does."""
    dist = D.Pareto(scale=_T(1.0), alpha=_T(alpha))
    mean, variance = _scalar(dist.mean), _scalar(dist.variance)
    assert mean > 0.0, mean
    assert variance > 0.0, variance
    if alpha <= 1.0:
        assert np.isposinf(mean), mean
    if alpha <= 2.0:
        assert np.isposinf(variance), variance


@pytest.mark.parametrize("concentration", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
def test_inverse_gamma_moments_never_go_negative(concentration: float) -> None:
    """Same shape as Pareto, and it had the same defect."""
    dist = D.InverseGamma(concentration=_T(concentration), rate=_T(1.0))
    mean, variance = _scalar(dist.mean), _scalar(dist.variance)
    assert mean > 0.0, mean
    assert variance > 0.0, variance
    if concentration <= 1.0:
        assert np.isposinf(mean), mean
    if concentration <= 2.0:
        assert np.isposinf(variance), variance


# ── moments that do not exist still have to answer ───────────────────────────


@pytest.mark.parametrize(
    "name,build,mean,variance",
    [
        ("Cauchy", lambda: D.Cauchy(loc=_T(0.0), scale=_T(1.0)), "nan", "inf"),
        ("HalfCauchy", lambda: D.HalfCauchy(scale=_T(1.0)), "inf", "inf"),
        ("Categorical", lambda: D.Categorical(probs=_T(0.5, 0.5)), "nan", "nan"),
    ],
)
def test_undefined_moments_answer_rather_than_raise(
    name: str, build, mean: str, variance: str
) -> None:
    """``NotImplementedError`` reads as "not got round to it".

    The truth is that the integral diverges, and NaN or infinity says so
    in a form that carries through an expression — which is what the
    reference returns for every one of these.
    """
    dist = build()
    check = {"nan": np.isnan, "inf": np.isposinf}
    assert check[mean](_scalar(dist.mean)), f"{name}.mean"
    assert check[variance](_scalar(dist.variance)), f"{name}.variance"


def test_one_hot_categorical_has_the_moments_categorical_cannot() -> None:
    """The one-hot encoding supplies the vector space the expectation needs.

    ``Categorical``'s labels carry no metric, so its mean is NaN — but a
    one-hot sample is a vector of indicators, and the expectation of an
    indicator is the probability of what it indicates.
    """
    dist = D.OneHotCategorical(probs=_T(0.25, 0.75))
    assert np.allclose(np.asarray(dist.mean.numpy()).ravel(), [0.25, 0.75])
    assert np.allclose(np.asarray(dist.variance.numpy()).ravel(), [0.1875, 0.1875])
    assert np.allclose(np.asarray(dist.stddev.numpy()).ravel(), np.sqrt([0.1875, 0.1875]))


# ── the round trip a distribution exists for ─────────────────────────────────


@pytest.mark.parametrize(
    "probs,sample_shape",
    [([0.25, 0.75], (64,)), ([[0.25, 0.75], [0.6, 0.4]], (5,)), ([[0.25, 0.75], [0.6, 0.4]], ())],
)
def test_categorical_can_score_its_own_samples(probs, sample_shape) -> None:
    """It could not, when the batch shape was empty.

    ``log_prob`` forced ``value`` into ``batch_shape``, which cannot
    express a sample_shape in front of it: ``probs`` of shape (2,) has no
    batch shape, so ``sample((64,))`` gave a (64,) tensor that had nowhere
    to be reshaped to, and scoring it raised a rank mismatch out of
    ``gather``.  The probabilities broadcast onto the value, not the
    other way round.
    """
    dist = D.Categorical(probs=lucid.tensor(np.array(probs, dtype=np.float64)))
    drawn = dist.sample(sample_shape)
    scored = dist.log_prob(drawn)
    assert tuple(scored.shape) == tuple(drawn.shape)
    assert np.isfinite(np.asarray(scored.numpy())).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_relaxed_families_refuse_a_non_positive_temperature(temperature: float) -> None:
    """The temperature divides the logits, so zero is a division by zero.

    Declared ``real``, it was never checked, and the draw came back
    non-finite with nothing raised — as it does in the reference, which
    does not check it either.
    """
    with pytest.raises(ValueError):
        D.RelaxedOneHotCategorical(temperature=_T(temperature), probs=_T(0.5, 0.5)).rsample((4,))


# ── scoring at a degenerate probability ──────────────────────────────────────


@pytest.mark.parametrize(
    "total,prob,value,expected",
    [
        (0.0, 1.0, 0.0, 0.0),  # the only outcome there is
        (0.0, 0.0, 0.0, 0.0),
        (5.0, 1.0, 5.0, 0.0),  # certain
        (5.0, 1.0, 0.0, "-inf"),  # impossible
        (5.0, 0.0, 0.0, 0.0),
        (5.0, 0.0, 5.0, "-inf"),
        (5.0, 0.5, 2.0, -1.1631508098056809),
    ],
)
def test_binomial_scores_a_degenerate_probability(
    total: float, prob: float, value: float, expected
) -> None:
    """A degenerate p makes the logit infinite, and both terms use it.

    ``k·l − n·softplus(l)`` is ``5·inf − 5·inf`` for ``Binomial(5, p=1)``
    at k=5 — NaN, for an outcome that is certain.  The equivalent
    ``−k·softplus(−l) − (n−k)·softplus(l)`` puts each infinity in the term
    whose count is zero, where the guard reaches it before the multiply.
    """
    dist = D.Binomial(total_count=_T(total), probs=_T(prob))
    got = _scalar(dist.log_prob(_T(value)))
    if expected == "-inf":
        assert np.isneginf(got), got
    else:
        assert np.isclose(got, expected, atol=1e-9), got


def test_binomial_scores_its_own_samples() -> None:
    """Including at p = 1, where the only draw is the certain one."""
    for total, prob in ((0.0, 1.0), (5.0, 0.5), (10.0, 0.3)):
        dist = D.Binomial(total_count=_T(total), probs=_T(prob))
        drawn = dist.sample((32,))
        scored = np.asarray(dist.log_prob(drawn).numpy(), dtype=np.float64)
        assert np.isfinite(scored).all(), (total, prob, scored)


def test_binomial_keeps_its_precision_at_a_tiny_probability() -> None:
    """``log C(20,10) + 10·log(1e-8)`` is -172.08, and that is the answer.

    The reference reports -147.3 here: it carries the probability through
    float32 and 1e-8 clamps to the epsilon, 1.19e-7.  The check is against
    the arithmetic rather than against the other framework.
    """
    dist = D.Binomial(total_count=_T(20.0), probs=_T(1e-8))
    got = _scalar(dist.log_prob(_T(10.0)))
    expected = float(np.log(184756.0) + 10.0 * np.log(1e-8) + 10.0 * np.log1p(-1e-8))
    assert np.isclose(got, expected, rtol=1e-9), (got, expected)
