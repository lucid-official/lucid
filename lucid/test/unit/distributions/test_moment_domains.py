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

import itertools

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.test._fixtures.tolerance_guard import noise_when_below


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


# ── LCD-680: a moment is nan or +inf outside its domain, never a number ──────
#
# ``FisherSnedecor.mean`` and ``.variance`` and ``StudentT.mean`` evaluated
# their closed form with no domain mask: F(5, 1.5) had mean -3.0 and F(5, 3)
# variance -21.6, F(5, 2) mean +inf where the reference answers nan, and
# StudentT(df=1) the location where the integral does not exist.  The table
# below states each family's thresholds once; the parity sweep underneath it
# holds Lucid to the reference over a wider grid than the table.

_NAN = "nan"
_INF = "inf"

DEVICES = ["cpu", "metal"]


def _tt(values: object, device: str) -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.float32), device=device)


def _check(got: float, expected: object, what: str) -> None:
    if expected == _NAN:
        assert np.isnan(got), (what, got)
    elif expected == _INF:
        assert np.isposinf(got), (what, got)
    else:
        assert np.isclose(got, expected, rtol=1e-4, atol=1e-6), (what, got, expected)


# (family, constructor kwargs, moment, expected).  Thresholds of F and
# StudentT first, then the other families whose moment diverges or does not
# exist at a valid parameter.
_MOMENT_TABLE = [
    # F: mean needs d2 > 2, variance needs d2 > 4; nan on and below the edge.
    ("FisherSnedecor", {"df1": 5.0, "df2": 0.25}, "mean", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 1.0}, "mean", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 1.5}, "mean", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 2.0}, "mean", _NAN),
    ("FisherSnedecor", {"df1": 3.0, "df2": 3.0}, "mean", 3.0),
    ("FisherSnedecor", {"df1": 5.0, "df2": 4.0}, "mean", 2.0),
    ("FisherSnedecor", {"df1": 3.0, "df2": 1.0}, "variance", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 0.5}, "variance", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 2.0}, "variance", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 3.0}, "variance", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 4.0}, "variance", _NAN),
    ("FisherSnedecor", {"df1": 5.0, "df2": 6.0}, "variance", 2 * 36 * 9 / (5 * 16 * 2)),
    ("FisherSnedecor", {"df1": 5.0, "df2": 6.0}, "stddev", np.sqrt(2 * 36 * 9 / (5 * 16 * 2))),
    ("FisherSnedecor", {"df1": 5.0, "df2": 3.0}, "stddev", _NAN),
    # StudentT: mean needs df > 1; variance needs df > 2 (inf on (1, 2]).
    ("StudentT", {"df": 0.5, "loc": 2.0}, "mean", _NAN),
    ("StudentT", {"df": 1.0, "loc": 2.0}, "mean", _NAN),
    ("StudentT", {"df": 1.5, "loc": 2.0}, "mean", 2.0),
    ("StudentT", {"df": 5.0, "loc": 2.0}, "mean", 2.0),
    ("StudentT", {"df": 1.0}, "variance", _NAN),
    ("StudentT", {"df": 1.5}, "variance", _INF),
    ("StudentT", {"df": 2.0}, "variance", _INF),
    ("StudentT", {"df": 1.0}, "stddev", _NAN),
    ("StudentT", {"df": 2.0}, "stddev", _INF),
    # Pareto: mean needs alpha > 1, variance needs alpha > 2.
    ("Pareto", {"scale": 1.0, "alpha": 0.5}, "mean", _INF),
    ("Pareto", {"scale": 1.0, "alpha": 1.0}, "mean", _INF),
    ("Pareto", {"scale": 1.0, "alpha": 2.0}, "mean", 2.0),
    ("Pareto", {"scale": 1.0, "alpha": 2.0}, "variance", _INF),
    ("Pareto", {"scale": 1.0, "alpha": 3.0}, "variance", 0.75),
    # InverseGamma: mean needs concentration > 1, variance > 2.
    ("InverseGamma", {"concentration": 1.0, "rate": 1.0}, "mean", _INF),
    ("InverseGamma", {"concentration": 2.0, "rate": 1.0}, "variance", _INF),
    ("InverseGamma", {"concentration": 3.0, "rate": 1.0}, "variance", 0.25),
    # GeneralizedPareto: mean needs concentration < 1, variance < 1/2; nan beyond.
    ("GeneralizedPareto", {"loc": 0.0, "scale": 1.0, "concentration": 1.0}, "mean", _NAN),
    ("GeneralizedPareto", {"loc": 0.0, "scale": 1.0, "concentration": 0.5}, "variance", _NAN),
    ("GeneralizedPareto", {"loc": 0.0, "scale": 1.0, "concentration": 0.25}, "mean", 4.0 / 3.0),
    # No first moment at any parameter.
    ("HalfCauchy", {"scale": 1.0}, "mean", _INF),
    ("Cauchy", {"loc": 0.0, "scale": 1.0}, "mean", _NAN),
    # Weibull's mode is 0 up to k = 1 (the density is not increasing there).
    ("Weibull", {"scale": 1.0, "concentration": 0.5}, "mode", 0.0),
    ("Weibull", {"scale": 2.0, "concentration": 1.0}, "mode", 0.0),
]


def _table_id(row: tuple) -> str:
    family, params, moment, _ = row
    return f"{family}-{'-'.join(f'{k}{v}' for k, v in params.items())}-{moment}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row", _MOMENT_TABLE, ids=[_table_id(r) for r in _MOMENT_TABLE])
def test_moment_is_nan_or_inf_outside_its_domain(device: str, row: tuple) -> None:
    """Each moment is its closed form inside the domain, nan/+inf outside."""
    family, params, moment, expected = row
    dist = getattr(D, family)(**{k: _tt(v, device) for k, v in params.items()})
    _check(_scalar(getattr(dist, moment)), expected, f"{family}{params}.{moment}")


@pytest.mark.parametrize("device", DEVICES)
def test_f_and_student_t_masks_are_per_element(device: str) -> None:
    """A batch mixes valid and invalid parameters; each element is judged alone."""
    f = D.FisherSnedecor(_tt([5.0] * 6, device), _tt([1.0, 2.0, 3.0, 4.0, 5.0, 8.0], device))
    mean = np.asarray(f.mean.numpy(), dtype=np.float64)
    var = np.asarray(f.variance.numpy(), dtype=np.float64)
    assert np.isnan(mean[:2]).all() and np.allclose(mean[2:], [3.0, 2.0, 5.0 / 3.0, 4.0 / 3.0])
    assert np.isnan(var[:4]).all() and np.isfinite(var[4:]).all() and (var[4:] > 0).all()

    t = D.StudentT(df=_tt([0.5, 1.0, 1.5, 3.0], device), loc=_tt(7.0, device))
    got = np.asarray(t.mean.numpy(), dtype=np.float64)
    assert np.isnan(got[:2]).all() and np.allclose(got[2:], 7.0)


def test_moment_masks_keep_the_parameter_dtype() -> None:
    """The nan branch must not promote a float64 parameter to float32."""
    f = D.FisherSnedecor(_T(5.0, 5.0), _T(1.0, 8.0))
    t = D.StudentT(df=_T(1.0, 3.0))
    for tensor in (f.mean, f.variance, t.mean):
        assert tensor.dtype == lucid.float64, tensor.dtype


# ── the reference answers the same over a wider grid ─────────────────────────

_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
_UNIT = [0.1, 0.5, 1.0, 2.0, 5.0]
_MOMENTS = ["mean", "variance", "mode", "stddev", "entropy"]

# family -> parameter name -> values; the sweep is their product.
_PARITY_GRIDS: dict[str, dict[str, list[float]]] = {
    "FisherSnedecor": {"df1": [1.0, 5.0], "df2": _GRID},
    "StudentT": {"df": _GRID},
    "Pareto": {"scale": [1.0], "alpha": _GRID},
    "InverseGamma": {"concentration": _GRID, "rate": [1.0]},
    "GeneralizedPareto": {
        "loc": [0.0],
        "scale": [1.0],
        "concentration": [-1.0, -0.1, 0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 1.0, 2.0],
    },
    "Weibull": {"scale": [1.0], "concentration": _GRID},
    "Gamma": {"concentration": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0], "rate": [1.0]},
    "Chi2": {"df": _GRID},
    "Beta": {"concentration1": _UNIT, "concentration0": _UNIT},
    "Kumaraswamy": {"concentration1": _UNIT, "concentration0": _UNIT},
    "HalfCauchy": {"scale": [0.5, 2.0]},
    "LogNormal": {"loc": [-1.0, 1.0], "scale": [0.1, 3.0]},
    "VonMises": {"loc": [0.0], "concentration": [0.01, 1.0, 50.0]},
    "NegativeBinomial": {"total_count": [1.0, 5.0], "probs": [0.3, 0.7]},
    "Binomial": {"total_count": [0.0, 5.0], "probs": [0.0, 0.3, 1.0]},
}

# Differences kept on purpose, with the reason.  Both are the reference
# evaluating a different expression from the one the mode is.
_MODE_DIFFERS = {
    "Weibull": (
        lambda p: p["concentration"] < 1.0,
        "below k = 1 the mode is 0; the reference raises a negative base to a "
        "fractional power and returns 81 at k = 0.25",
    ),
    "Kumaraswamy": (
        lambda p: True,
        "Lucid answers ((a-1)/(ab-1))^(1/a); the reference's expression is nan "
        "for every concentration0 > 1",
    ),
}


def _parity_cases(family: str) -> list[dict[str, float]]:
    grid = _PARITY_GRIDS[family]
    names = list(grid)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*(grid[n] for n in names))
    ]


def _moment(dist, name: str):
    """The five moments are properties except ``entropy``, a method."""
    return dist.entropy() if name == "entropy" else getattr(dist, name)


def _outcome(compute) -> np.ndarray | type[Exception]:
    """The moment as float64, or the exception class that stood in for it."""
    try:
        return np.asarray(compute(), dtype=np.float64)
    except NotImplementedError as exc:
        return type(exc)


def _agree(ours: np.ndarray | type, theirs: np.ndarray | type) -> bool:
    """Two answers agree when both are the same refusal or the same array."""
    if isinstance(ours, type) or isinstance(theirs, type):
        return ours is theirs
    if ours.shape != theirs.shape:
        return False
    # An entropy that is 0 in closed form comes back as 1e-5 from the reference's clamping
    # and 5e-7 from float32 roundoff: that is the comparison, not a small value being missed.
    with noise_when_below(theirs, 1e-4, "moments that are 0 in closed form"):
        return bool(np.allclose(ours, theirs, rtol=1e-3, atol=1e-4, equal_nan=True))


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", list(_PARITY_GRIDS))
def test_moments_match_the_reference_over_a_grid(family: str, device: str, ref) -> None:
    """mean, variance, mode, stddev and entropy agree element for element.

    ``nan`` must meet ``nan`` and ``inf`` meet ``inf``: a number where the
    reference has a divergence is the defect this file exists for.
    """
    mismatches: list[str] = []
    for params in _parity_cases(family):
        ours_dist = getattr(D, family)(**{k: _tt(v, device) for k, v in params.items()})
        theirs_dist = getattr(ref.distributions, family)(
            **{k: ref.tensor(np.float32(v)) for k, v in params.items()}
        )
        for moment in _MOMENTS:
            if moment == "mode" and family in _MODE_DIFFERS and _MODE_DIFFERS[family][0](params):
                continue
            ours = _outcome(lambda: _moment(ours_dist, moment).numpy())
            theirs = _outcome(lambda: _moment(theirs_dist, moment).detach().cpu().numpy())
            if not _agree(ours, theirs):
                mismatches.append(f"{params} {moment}: {ours} vs {theirs}")
    assert not mismatches, "\n".join(mismatches)
