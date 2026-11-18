"""Samplers bring their parameters to the sample shape as views (LCD-1044).

``sample`` and ``rsample`` of ``Bernoulli``, ``Poisson``, ``Binomial``,
``NegativeBinomial``, ``ContinuousBernoulli`` and the relaxed laws widened a
parameter to ``(*sample_shape, *batch_shape)`` by adding a tensor of zeros to
it.  They now take a ``broadcast_to`` view (``_util._broadcast_to``), which is
arithmetic-free.  Each law is drawn here from parameters that have to be
broadcast, in both forms the constructors accept: a number beside a vector, and
a vector beside a matrix.  The draw has the shape and the dtype of the
parameters and stays in its support; where the law lets a lane hold a NaN, that
lane does not reach the others.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

_SAMPLES = 4
_VECTOR = [0.2, 0.5, 0.8]
_MATRIX = [[0.2, 0.5, 0.8], [0.3, 0.4, 0.6]]
_TOTALS = [3.0, 5.0, 7.0]

_Maker = Callable[[Any], lucid.Tensor]
_Draw = np.ndarray[Any, Any]
_Check = Callable[[_Draw, Any], None]


def _binary(draw: _Draw, dist: Any) -> None:
    assert set(np.unique(draw).tolist()) <= {0.0, 1.0}


def _counts(draw: _Draw, dist: Any) -> None:
    assert (draw >= 0).all()
    np.testing.assert_array_equal(draw, np.floor(draw))


def _binomial_counts(draw: _Draw, dist: Any) -> None:
    _counts(draw, dist)
    assert (draw <= np.asarray(dist.total_count.numpy(), dtype=np.float64)).all()


def _unit_interval(draw: _Draw, dist: Any) -> None:
    assert ((draw >= 0.0) & (draw <= 1.0)).all()


def _simplex(draw: _Draw, dist: Any) -> None:
    _unit_interval(draw, dist)
    np.testing.assert_allclose(draw.sum(axis=-1), 1.0, rtol=1e-4)


@dataclass(frozen=True)
class _Sampler:
    """One law drawn from parameters that need broadcasting."""

    build: Callable[[_Maker], Any]
    method: str
    check: _Check


_SAMPLERS: dict[str, _Sampler] = {
    "Bernoulli-vector": _Sampler(lambda m: D.Bernoulli(probs=m(_VECTOR)), "sample", _binary),
    "Bernoulli-matrix": _Sampler(lambda m: D.Bernoulli(probs=m(_MATRIX)), "sample", _binary),
    "Poisson-vector": _Sampler(lambda m: D.Poisson(m([1.5, 4.0, 9.0])), "sample", _counts),
    "Poisson-matrix": _Sampler(lambda m: D.Poisson(m([[1.5, 4.0], [9.0, 0.5]])), "sample", _counts),
    "Binomial-number-beside-vector": _Sampler(
        lambda m: D.Binomial(5, probs=m(_VECTOR)), "sample", _binomial_counts
    ),
    "Binomial-vector-beside-matrix": _Sampler(
        lambda m: D.Binomial(m(_TOTALS), probs=m(_MATRIX)), "sample", _binomial_counts
    ),
    "Binomial-integer-count-beside-vector": _Sampler(
        lambda m: D.Binomial(lucid.tensor(5, dtype=lucid.int64), probs=m(_VECTOR)),
        "sample",
        _binomial_counts,
    ),
    "NegativeBinomial-number-beside-vector": _Sampler(
        lambda m: D.NegativeBinomial(3.5, probs=m(_VECTOR)), "sample", _counts
    ),
    "NegativeBinomial-vector-beside-matrix": _Sampler(
        lambda m: D.NegativeBinomial(m(_TOTALS), probs=m(_MATRIX)), "sample", _counts
    ),
    "NegativeBinomial-integer-count-beside-vector": _Sampler(
        lambda m: D.NegativeBinomial(lucid.tensor(5, dtype=lucid.int64), probs=m(_VECTOR)),
        "sample",
        _counts,
    ),
    "ContinuousBernoulli-vector": _Sampler(
        lambda m: D.ContinuousBernoulli(probs=m(_VECTOR)), "rsample", _unit_interval
    ),
    "ContinuousBernoulli-matrix": _Sampler(
        lambda m: D.ContinuousBernoulli(probs=m(_MATRIX)), "rsample", _unit_interval
    ),
    "RelaxedBernoulli-number-beside-vector": _Sampler(
        lambda m: D.RelaxedBernoulli(0.5, probs=m(_VECTOR)), "rsample", _unit_interval
    ),
    "RelaxedBernoulli-vector-beside-matrix": _Sampler(
        lambda m: D.RelaxedBernoulli(m(_TOTALS), probs=m(_MATRIX)), "rsample", _unit_interval
    ),
    "RelaxedOneHotCategorical-number-beside-matrix": _Sampler(
        lambda m: D.RelaxedOneHotCategorical(0.5, probs=m([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])),
        "rsample",
        _simplex,
    ),
    "RelaxedOneHotCategorical-vector-beside-matrix": _Sampler(
        lambda m: D.RelaxedOneHotCategorical(
            m([0.5, 0.6]), probs=m([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
        ),
        "rsample",
        _simplex,
    ),
}


def _maker(dtype: lucid.dtype, device: str) -> _Maker:
    return lambda value: lucid.tensor(value, dtype=dtype, device=device)


@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_a_draw_from_broadcast_parameters_has_the_batch_shape_and_the_parameter_dtype(
    name: str, device: str, float_dtype: lucid.dtype
) -> None:
    sampler = _SAMPLERS[name]
    dist = sampler.build(_maker(float_dtype, device))
    lucid.manual_seed(0)
    draw = getattr(dist, sampler.method)((_SAMPLES,))
    assert tuple(draw.shape) == (_SAMPLES, *dist.batch_shape, *dist.event_shape)
    assert draw.dtype == float_dtype
    assert draw.device.type == device
    values = np.asarray(draw.numpy(), dtype=np.float64)
    assert np.isfinite(values).all()
    sampler.check(values, dist)


@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_a_draw_is_repeatable_under_a_seed(name: str) -> None:
    sampler = _SAMPLERS[name]
    dist = sampler.build(_maker(lucid.float32, "cpu"))
    lucid.manual_seed(7)
    first = getattr(dist, sampler.method)((_SAMPLES,)).numpy()
    lucid.manual_seed(7)
    second = getattr(dist, sampler.method)((_SAMPLES,)).numpy()
    np.testing.assert_array_equal(first, second)


def test_an_empty_sample_shape_draws_one_value_per_batch_element() -> None:
    lucid.manual_seed(0)
    probs = lucid.tensor(_MATRIX)
    assert tuple(D.Bernoulli(probs=probs).sample().shape) == (2, 3)
    assert tuple(D.Binomial(5, probs=probs).sample().shape) == (2, 3)
    assert tuple(D.RelaxedBernoulli(0.5, probs=probs).rsample().shape) == (2, 3)


_NAN = float("nan")


def test_a_nan_probability_does_not_reach_the_other_bernoulli_lanes() -> None:
    lucid.manual_seed(0)
    dist = D.Bernoulli(probs=lucid.tensor([1.0, _NAN, 0.0]), validate_args=False)
    draw = dist.sample((64,)).numpy()
    np.testing.assert_array_equal(draw[:, 0], np.ones(64, dtype=np.float32))
    np.testing.assert_array_equal(draw[:, 2], np.zeros(64, dtype=np.float32))


def test_a_nan_probability_does_not_reach_the_other_binomial_lanes() -> None:
    lucid.manual_seed(0)
    dist = D.Binomial(5, probs=lucid.tensor([0.2, _NAN, 0.8]), validate_args=False)
    draw = dist.sample((64,)).numpy()
    for lane in (0, 2):
        assert ((draw[:, lane] >= 0) & (draw[:, lane] <= 5)).all()
        np.testing.assert_array_equal(draw[:, lane], np.floor(draw[:, lane]))


def test_a_nan_probability_does_not_reach_the_other_continuous_bernoulli_lanes() -> None:
    lucid.manual_seed(0)
    dist = D.ContinuousBernoulli(probs=lucid.tensor([0.2, _NAN, 0.8]), validate_args=False)
    draw = dist.rsample((64,)).numpy()
    for lane in (0, 2):
        assert ((draw[:, lane] >= 0.0) & (draw[:, lane] <= 1.0)).all()


def test_a_nan_probability_does_not_reach_the_other_relaxed_bernoulli_lanes() -> None:
    lucid.manual_seed(0)
    dist = D.RelaxedBernoulli(0.5, probs=lucid.tensor([0.2, _NAN, 0.8]), validate_args=False)
    draw = dist.rsample((64,)).numpy()
    for lane in (0, 2):
        assert ((draw[:, lane] >= 0.0) & (draw[:, lane] <= 1.0)).all()


def test_a_nan_row_does_not_reach_the_other_relaxed_one_hot_rows() -> None:
    lucid.manual_seed(0)
    probs = lucid.tensor([[0.2, 0.3, 0.5], [_NAN, _NAN, _NAN], [0.1, 0.2, 0.7]])
    dist = D.RelaxedOneHotCategorical(0.5, probs=probs, validate_args=False)
    draw = dist.rsample((64,)).numpy()
    for row in (0, 2):
        np.testing.assert_allclose(draw[:, row].sum(axis=-1), 1.0, rtol=1e-4)
        assert ((draw[:, row] >= 0.0) & (draw[:, row] <= 1.0)).all()


@pytest.mark.parametrize(
    "law", [D.Poisson, D.NegativeBinomial], ids=["Poisson", "NegativeBinomial"]
)
def test_a_non_finite_rate_is_refused_not_spread_over_the_batch(law: Any) -> None:
    """The count laws hand a rate to ``lucid.poisson``, which refuses a non-finite one whole."""
    if law is D.Poisson:
        dist = law(lucid.tensor([1.0, float("inf"), 3.0]), validate_args=False)
    else:
        dist = law(3.5, probs=lucid.tensor([0.2, _NAN, 0.8]), validate_args=False)
    with pytest.raises(ValueError, match="finite"):
        dist.sample((_SAMPLES,))
