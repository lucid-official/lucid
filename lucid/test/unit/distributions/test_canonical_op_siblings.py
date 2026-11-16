"""LCD-1035 sibling sweep: distribution formulas that spelled out what ``expm1`` / ``log1p`` are for.

Each is ``exp(t) - 1`` or ``log(1 - p)`` of a small number, where the sum rounds to 1 and the answer
is the number itself:

* ``Exponential.cdf`` was ``1 - exp(-rate x)``: 0 at ``x = 1e-9`` (float32), 19% off at ``1e-7``;
* ``LogNormal.variance`` was ``(exp(s^2) - 1) exp(2 mu + s^2)``: 0 at ``sigma = 1e-4``, 5% off at ``1e-3``;
* ``Geometric.sample`` divided by ``log(1 - p)``: 0 for ``p`` below 6e-8, so every sample was ``-inf``.

The oracles are plain float64 closed forms (``math``).
"""

import math

import pytest

import lucid
import lucid.distributions as D
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_REL = {lucid.float32: 2e-6, lucid.float64: 1e-12}


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_exponential_cdf_keeps_small_arguments(device: str, dtype: lucid.dtype) -> None:
    rate = 2.0
    xs = [0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 0.5, 30.0, 1e3, math.inf]
    d = D.Exponential(rate=lucid.tensor(rate, dtype=dtype, device=device))
    x = lucid.tensor(xs, dtype=dtype, device=device)
    for xv, got in zip(x.tolist(), d.cdf(x).tolist(), strict=True):
        want = -math.expm1(-rate * xv)
        assert math.isclose(got, want, rel_tol=_REL[dtype], abs_tol=1e-300), (xv, got, want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_lognormal_variance_keeps_a_small_sigma(device: str, dtype: lucid.dtype) -> None:
    mu = 0.3
    for sigma in (1e-4, 1e-3, 1e-2, 0.5, 1.0):
        d = D.LogNormal(
            loc=lucid.tensor(mu, dtype=dtype, device=device),
            scale=lucid.tensor(sigma, dtype=dtype, device=device),
        )
        s, m = d.scale.item(), d.loc.item()
        want = math.expm1(s * s) * math.exp(2.0 * m + s * s)
        assert math.isclose(d.variance.item(), want, rel_tol=4 * _REL[dtype]), (sigma, want)


@pytest.mark.parametrize(
    ("device", "dtype", "prob"),
    [
        (dev, dt, 1e-9 if dt == lucid.float32 else 1e-18)
        for dev, dt in [p.values[:2] for p in _FLOATS]
    ],
)
def test_geometric_samples_for_a_tiny_probability(
    device: str, dtype: lucid.dtype, prob: float
) -> None:
    """``log(1 - p)`` is 0 below the dtype's epsilon: ``-inf`` samples.  The median is ``ln 2 / p``."""
    lucid.manual_seed(0)
    d = D.Geometric(probs=lucid.tensor(prob, dtype=dtype, device=device))
    samples = sorted(d.sample((4001,)).tolist())
    assert all(math.isfinite(v) and v >= 0.0 for v in samples)
    assert math.isclose(samples[2000], math.log(2.0) / prob, rel_tol=0.15)
