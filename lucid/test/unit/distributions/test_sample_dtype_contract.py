"""A discrete draw answers in the dtype the reference gives it (LCD-815).

``Poisson``, ``Binomial`` and ``Multinomial`` answered ``int64`` and
``lucid.poisson`` did too, where the reference returns the parameter's floating
dtype.  The rule is the same for every discrete law: a *count* keeps the
parameter's float dtype, and only an *index* -- ``Categorical`` -- is ``int64``
(``OneHotCategorical`` is a one-hot float vector).  The wrappers inherit the
dtype of what they wrap, so ``Independent`` and ``MixtureSameFamily`` over a
count law answer in the float dtype too.

The guard walks every discrete law and wrapper rather than the three that were
found, on both devices and both float widths, and checks the rule as well as
the reference (the rule alone survives a run without the reference installed).
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.distributions as D

# Builds a distribution from a namespace of its framework's tensor constructor
# and distributions module, so one table drives both frameworks.
_Make = Callable[[Any, Callable[..., Any]], Any]

_CASES: dict[str, _Make] = {
    "Bernoulli": lambda m, t: m.Bernoulli(probs=t([0.3, 0.6])),
    "Geometric": lambda m, t: m.Geometric(probs=t([0.3, 0.6])),
    "Categorical": lambda m, t: m.Categorical(probs=t([0.2, 0.3, 0.5])),
    "OneHotCategorical": lambda m, t: m.OneHotCategorical(probs=t([0.2, 0.3, 0.5])),
    "OneHotCategoricalStraightThrough": lambda m, t: m.OneHotCategoricalStraightThrough(
        probs=t([0.2, 0.3, 0.5])
    ),
    "Poisson": lambda m, t: m.Poisson(t([2.0, 30.0])),
    "Binomial": lambda m, t: m.Binomial(10, probs=t([0.3, 0.6])),
    "NegativeBinomial": lambda m, t: m.NegativeBinomial(5, probs=t([0.3, 0.6])),
    "Multinomial": lambda m, t: m.Multinomial(4, probs=t([0.2, 0.3, 0.5])),
    "Independent(Poisson)": lambda m, t: m.Independent(m.Poisson(t([[2.0, 3.0], [4.0, 5.0]])), 1),
    "Independent(Binomial)": lambda m, t: m.Independent(
        m.Binomial(10, probs=t([[0.3, 0.4], [0.5, 0.6]])), 1
    ),
    "MixtureSameFamily(Poisson)": lambda m, t: m.MixtureSameFamily(
        m.Categorical(probs=t([0.5, 0.5])), m.Poisson(t([2.0, 20.0]))
    ),
    "MixtureSameFamily(Binomial)": lambda m, t: m.MixtureSameFamily(
        m.Categorical(probs=t([0.5, 0.5])), m.Binomial(10, probs=t([0.3, 0.6]))
    ),
}

# The draws that are indices; every other discrete draw is a float.
_INDEX_VALUED = frozenset({"Categorical"})


def _name(dtype: object) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _float_dtypes(device: str) -> list[str]:
    return ["float32"] if device == "metal" else ["float32", "float64"]


def _lucid_dist(case: str, dtype: str, device: str) -> D.Distribution:
    def t(v: Any) -> lucid.Tensor:
        return lucid.tensor(v, dtype=getattr(lucid, dtype), device=device)

    return _CASES[case](D, t)


def _expected(case: str, dtype: str) -> str:
    return "int64" if case in _INDEX_VALUED else dtype


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_discrete_sample_dtype_follows_the_rule(case: str, dtype: str, device: str) -> None:
    if dtype not in _float_dtypes(device):
        pytest.skip(f"{dtype} is not available on {device}")
    lucid.manual_seed(0)
    draw = _lucid_dist(case, dtype, device).sample((3,))
    assert _name(draw.dtype) == _expected(case, dtype), f"{case} sampled {draw.dtype}"
    assert draw.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_discrete_sample_dtype_matches_the_reference(
    case: str, dtype: str, device: str, ref: ModuleType
) -> None:
    if dtype not in _float_dtypes(device):
        pytest.skip(f"{dtype} is not available on {device}")

    def t(v: Any) -> Any:
        return ref.tensor(v, dtype=getattr(ref, dtype))

    theirs = _CASES[case](ref.distributions, t).sample((3,))
    lucid.manual_seed(0)
    mine = _lucid_dist(case, dtype, device).sample((3,))
    assert _name(mine.dtype) == _name(theirs.dtype), f"{case}: {mine.dtype} vs {theirs.dtype}"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_poisson_factory_answers_in_the_input_dtype(dtype: str, device: str) -> None:
    if dtype not in _float_dtypes(device):
        pytest.skip(f"{dtype} is not available on {device}")
    rate = lucid.full((8,), 4.0, dtype=getattr(lucid, dtype), device=device)
    assert _name(lucid.poisson(rate).dtype) == dtype


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_poisson_factory_matches_the_reference_dtype(
    dtype: str, device: str, ref: ModuleType
) -> None:
    if dtype not in _float_dtypes(device):
        pytest.skip(f"{dtype} is not available on {device}")
    theirs = ref.poisson(ref.full((8,), 4.0, dtype=getattr(ref, dtype)))
    mine = lucid.poisson(lucid.full((8,), 4.0, dtype=getattr(lucid, dtype), device=device))
    assert _name(mine.dtype) == _name(theirs.dtype)


def test_poisson_factory_of_an_integer_rate_uses_the_default_float(device: str) -> None:
    out = lucid.poisson(lucid.tensor([2, 3], dtype=lucid.int64, device=device))
    assert out.dtype == lucid.get_default_dtype()


def test_count_samples_are_integral_and_in_range(device: str) -> None:
    lucid.manual_seed(1)
    binom = D.Binomial(10, probs=lucid.tensor([0.3, 0.9], device=device)).sample((500,))
    pois = D.Poisson(lucid.tensor([0.5, 40.0], device=device)).sample((500,))
    multi = D.Multinomial(4, probs=lucid.tensor([0.2, 0.3, 0.5], device=device)).sample((500,))
    for draw in (binom, pois, multi):
        assert bool((draw == draw.floor()).all().item()) and bool((draw >= 0).all().item())
    assert bool((binom <= 10).all().item())
    assert bool((multi.sum(dim=-1) == 4).all().item())
