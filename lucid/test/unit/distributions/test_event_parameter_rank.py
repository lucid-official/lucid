"""A parameter with fewer axes than its event is refused at construction (LCD-414).

``Dirichlet(lucid.tensor(0.5))`` constructed, then raised ``IndexError`` from
``entropy``; a 0-d ``Categorical`` raised it from its constructor.  The
``Distribution`` base now refuses, with ``ValueError``, any tensor stored
under a parameter whose constraint reads more trailing axes than it has —
with validation on or off, as it is a shape requirement.  The reference
comparison is in ``lucid/test/parity/test_distribution_constraints_parity.py``.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.distributions as D


def _t(v: object) -> lucid.Tensor:
    return lucid.tensor(v, dtype=lucid.float64)


_VEC = [0.2, 0.3, 0.5]
_MAT = [[2.0, 0.5], [0.5, 1.0]]

# name -> builder taking ``validate_args``; each passes one parameter short.
_SHORT: dict[str, Callable[[bool], object]] = {
    "Dirichlet": lambda va: D.Dirichlet(_t(0.5), validate_args=va),
    "Categorical.probs": lambda va: D.Categorical(probs=_t(0.5), validate_args=va),
    "Categorical.logits": lambda va: D.Categorical(logits=_t(0.5), validate_args=va),
    "OneHotCategorical": lambda va: D.OneHotCategorical(_t(0.5), validate_args=va),
    "RelaxedOneHotCategorical": lambda va: D.RelaxedOneHotCategorical(
        _t(0.5), logits=_t(0.5), validate_args=va
    ),
    "Multinomial": lambda va: D.Multinomial(3, probs=_t(0.5), validate_args=va),
    "MultivariateNormal.loc": lambda va: D.MultivariateNormal(
        _t(0.0), covariance_matrix=_t([[1.0]]), validate_args=va
    ),
    "MultivariateNormal.scale_tril": lambda va: D.MultivariateNormal(
        _t([0.0]), scale_tril=_t([1.0]), validate_args=va
    ),
    "Wishart": lambda va: D.Wishart(_t(3.0), covariance_matrix=_t(1.0), validate_args=va),
}


@pytest.mark.parametrize("validate", [True, False], ids=["validate", "no-validate"])
@pytest.mark.parametrize("name", list(_SHORT))
def test_short_parameter_is_refused(name: str, validate: bool) -> None:
    with pytest.raises(ValueError, match="must be at least"):
        _SHORT[name](validate)


def test_full_rank_and_batch_scalars_still_construct() -> None:
    # The happy paths, and a per-batch scalar whose constraint has no event.
    D.Dirichlet(_t(_VEC)).entropy()
    D.Categorical(logits=_t(_VEC)).entropy()
    D.MultivariateNormal(_t([0.0, 0.0]), covariance_matrix=_t(_MAT)).entropy()
    D.Wishart(_t(3.0), covariance_matrix=_t(_MAT)).sample()
    D.LKJCholesky(3, _t(0.5)).sample()
