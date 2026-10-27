"""Distributions must sample and score on the device their parameters live on.

Found 2026-07-26 by coverage-directed probing (``lucid/distributions`` was at
67%).  Three families raised ``DeviceMismatch`` for Metal parameters:

* ``Chi2`` and ``StudentT`` build a host-derived scalar constant
  (``rate=_as_tensor(0.5)``, the StudentT defaults) and then combine it with the
  user's parameter.  ``_broadcast_pair`` is the single point where the two meet,
  so it now reconciles devices there rather than threading a device through all
  ~67 ``_as_tensor`` call sites.
* ``Poisson.sample`` passed the right device down, but ``lucid.poisson`` itself
  returned a CPU tensor regardless of its input — Knuth sampling runs on the
  host and the result was never brought back.  ``log_prob`` then combined a CPU
  sample with a Metal ``rate``.  Same family as the ``nonzero``/``unique``/
  ``bincount`` device leaks.
"""

import numpy as np
import pytest

import lucid
import lucid.distributions as D

DEVICES = ["cpu", "metal"]

# (name, constructor kwargs) — one per shipped continuous/discrete family.
SPECS = [
    ("Normal", dict(loc=0.0, scale=1.0)),
    ("Uniform", dict(low=0.0, high=1.0)),
    ("Bernoulli", dict(probs=0.5)),
    ("Exponential", dict(rate=1.0)),
    ("Laplace", dict(loc=0.0, scale=1.0)),
    ("Cauchy", dict(loc=0.0, scale=1.0)),
    ("Gamma", dict(concentration=2.0, rate=1.0)),
    ("Beta", dict(concentration1=2.0, concentration0=3.0)),
    ("Poisson", dict(rate=2.0)),
    ("StudentT", dict(df=5.0)),
    ("Chi2", dict(df=3.0)),
    ("Geometric", dict(probs=0.3)),
    ("HalfNormal", dict(scale=1.0)),
    ("LogNormal", dict(loc=0.0, scale=1.0)),
    ("Pareto", dict(scale=1.0, alpha=2.0)),
    ("Weibull", dict(scale=1.0, concentration=2.0)),
    ("Gumbel", dict(loc=0.0, scale=1.0)),
]


def _build(name, kwargs, device):
    cls = getattr(D, name)
    return cls(
        **{k: lucid.tensor(np.array(v, dtype=np.float32), device=device) for k, v in kwargs.items()}
    )


@pytest.mark.parametrize("name,kwargs", SPECS, ids=[s[0] for s in SPECS])
@pytest.mark.parametrize("device", DEVICES)
def test_sample_stays_on_the_parameter_device(name, kwargs, device):
    dist = _build(name, kwargs, device)
    lucid.manual_seed(0)
    sample = dist.sample((32,))
    assert str(sample.device) == f"device('{device}')", f"{name} left {device}"


@pytest.mark.parametrize("name,kwargs", SPECS, ids=[s[0] for s in SPECS])
@pytest.mark.parametrize("device", DEVICES)
def test_log_prob_of_own_sample_is_finite(name, kwargs, device):
    """The device leak surfaced here: a CPU sample against a Metal parameter."""
    dist = _build(name, kwargs, device)
    lucid.manual_seed(0)
    sample = dist.sample((32,))
    log_prob = dist.log_prob(sample)
    assert str(log_prob.device) == f"device('{device}')"
    assert np.isfinite(log_prob.numpy()).all(), f"{name}: non-finite log_prob"


@pytest.mark.parametrize("device", DEVICES)
def test_poisson_factory_rides_the_input_device(device):
    """``lucid.poisson`` samples on the host but must return on the input's device."""
    rate = lucid.tensor(np.full(16, 3.0, dtype=np.float32), device=device)
    out = lucid.poisson(rate)
    assert str(out.device) == f"device('{device}')"
    assert out.shape == rate.shape
    assert (out.numpy() >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_scalar_constant_broadcast_reconciles_devices(device):
    """Chi2/StudentT mix a host scalar with the user's parameter."""
    for name, kwargs in (("Chi2", dict(df=3.0)), ("StudentT", dict(df=5.0))):
        dist = _build(name, kwargs, device)
        lucid.manual_seed(0)
        sample = dist.sample((8,))
        assert str(sample.device) == f"device('{device}')", name


def test_cauchy_has_no_mean():
    """Not a bug — the Cauchy mean is undefined, and NaN is how it says so.

    This used to assert ``NotImplementedError``, which reads as "Lucid has
    not got round to this" when the truth is that the defining integral
    diverges.  The reference returns NaN for the mean and +inf for the
    variance, and that is the more useful shape: the value carries through
    an expression and can be tested at the end, where an exception has to
    be caught at the call.
    """
    dist = _build("Cauchy", dict(loc=0.0, scale=1.0), "cpu")
    mean = np.asarray(dist.mean.numpy(), dtype=np.float64)
    assert np.isnan(mean).all(), mean
    variance = np.asarray(dist.variance.numpy(), dtype=np.float64)
    assert np.isposinf(variance).all(), variance
