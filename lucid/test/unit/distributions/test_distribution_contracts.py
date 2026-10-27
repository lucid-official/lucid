"""Contracts every distribution owes, found by the audit's distribution axis.

Three defects, and the third is the one that hid the other two.

**Exponential's CDF left [0, 1].**  ``1 - exp(-λx)`` is the formula *on*
the support; evaluated below it the exponential grows without bound and
``cdf(-1)`` answered -1.718, while the docstring one line above promised
a value in [0, 1].  A CDF is defined on the whole line and is zero to the
left of its support.

**Laplace's CDF ran backwards.**  The sign in front of the second term
was ``-`` rather than ``+``, in the implementation and in the formula
above it, which makes the function decreasing: ``cdf(-3)`` gave 0.975 and
``cdf(3)`` gave 0.025.  It stayed inside [0, 1] and was symmetric about
0.5 at the median, so nothing looked wrong until something checked that a
CDF increases.

**Parameter validation was written and never armed.**  ``arg_constraints``
and ``validate_args`` were both present; ``_validate_args`` defaulted to
``False``.  So ``Beta(0.0, 1.0)`` — a concentration outside the
distribution's own declared constraint — constructed happily and every
``log_prob`` against it answered nan.  Eleven distributions reported
"log_prob is not finite on the distribution's own samples", which was
true, and was the parameters rather than the density.
"""

import numpy as np
import pytest

import lucid
import lucid.distributions as D


def test_exponential_cdf_is_zero_below_the_support() -> None:
    # Validation refuses a value below the support outright; the formula is
    # what this pins, so it is read with validation off.
    dist = D.Exponential(lucid.tensor(np.array([1.0])), validate_args=False)
    values = lucid.tensor(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]))
    got = dist.cdf(values).numpy()
    assert ((got >= 0.0) & (got <= 1.0)).all(), "cdf left [0, 1]"
    assert np.allclose(got[:3], 0.0)
    assert np.allclose(got[3:], 1.0 - np.exp(-np.array([0.5, 1.0, 2.0])))


def test_laplace_cdf_increases() -> None:
    dist = D.Laplace(lucid.tensor(np.array([0.0])), lucid.tensor(np.array([1.0])))
    grid = np.linspace(-3.0, 3.0, 13)
    got = dist.cdf(lucid.tensor(grid)).numpy()
    assert np.all(np.diff(got) >= -1e-12), "cdf is not monotone"
    expected = np.where(grid < 0, 0.5 * np.exp(grid), 1.0 - 0.5 * np.exp(-grid))
    assert np.allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "build",
    [
        lambda: D.Beta(lucid.tensor(np.array(0.0)), lucid.tensor(np.array(1.0))),
        lambda: D.Gamma(lucid.tensor(np.array(0.0)), lucid.tensor(np.array(1.0))),
        lambda: D.Normal(lucid.tensor(np.array(0.0)), lucid.tensor(np.array(-1.0))),
    ],
)
def test_parameters_outside_their_constraint_are_refused(build) -> None:
    """Silent nan is worse than a loud refusal."""
    with pytest.raises(ValueError, match="constraint"):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: D.Beta(lucid.tensor(np.array(2.0)), lucid.tensor(np.array(3.0))),
        lambda: D.Gamma(lucid.tensor(np.array(2.0)), lucid.tensor(np.array(1.0))),
        lambda: D.Normal(lucid.tensor(np.array(0.0)), lucid.tensor(np.array(1.0))),
    ],
)
def test_valid_parameters_still_build_and_score(build) -> None:
    """Guard the instrument: validation must not reject the legitimate case."""
    lucid.manual_seed(0)
    dist = build()
    assert np.isfinite(np.asarray(dist.log_prob(dist.sample()).numpy())).all()


def test_validation_can_be_turned_off() -> None:
    """The cost is opt-out, for a caller who has already checked."""
    dist = D.Beta(lucid.tensor(np.array(0.0)), lucid.tensor(np.array(1.0)), validate_args=False)
    assert dist is not None


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_enumerate_support_contract(device: str) -> None:
    """Every distribution declaring ``has_enumerate_support`` enumerates a
    support whose probabilities sum to 1, and every other raises
    ``NotImplementedError`` (LCD-453)."""
    with pytest.raises(NotImplementedError):
        D.Distribution().enumerate_support()

    t = lambda v: lucid.tensor(np.asarray(v, dtype=np.float32)).to(device)
    enumerable = [
        (D.Bernoulli(probs=t([[0.2, 0.7], [0.4, 0.9]])), (2, 2, 2), (2, 1, 1)),
        (D.Binomial(5, probs=t([0.3, 0.6])), (6, 2), (6, 1)),
        (D.Categorical(probs=t([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])), (3, 2), (3, 1)),
        (D.OneHotCategorical(probs=t([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])), (3, 2, 3), (3, 1, 3)),
        (D.Independent(D.Bernoulli(probs=t([0.3, 0.7])), 0), (2, 2), (2, 1)),
    ]
    for dist, expanded_shape, unexpanded_shape in enumerable:
        assert dist.has_enumerate_support
        full = dist.enumerate_support(expand=True)
        slim = dist.enumerate_support(expand=False)
        assert full.device.type == device
        assert slim.device.type == device
        assert tuple(full.shape) == expanded_shape
        assert tuple(slim.shape) == unexpanded_shape
        assert np.allclose(dist.log_prob(full).exp().sum(dim=0).numpy(), 1.0, atol=1e-5)
        assert np.allclose(dist.log_prob(slim).exp().sum(dim=0).numpy(), 1.0, atol=1e-5)

    ind_1 = D.Independent(D.Bernoulli(probs=t([0.3, 0.7])), 1)
    assert not ind_1.has_enumerate_support
    with pytest.raises(NotImplementedError):
        ind_1.enumerate_support()

    inhom = D.Binomial(t([2.0, 5.0]), probs=t([0.3, 0.7]))
    with pytest.raises(NotImplementedError):
        inhom.enumerate_support()
