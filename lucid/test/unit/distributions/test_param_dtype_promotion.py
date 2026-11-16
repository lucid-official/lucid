"""A parameter given as a Python number takes the dtype of the tensors beside it (LCD-1040).

A distribution parameter handed over as a number was turned into a tensor on
its own, before anything said what it would be combined with, so it came out
in the default dtype.  Against a float64 tensor parameter that has two costs: a
number that is not exact in float32 (``0.1``, ``3.3``) is rounded before the
float64 arithmetic ever sees it, and where the two parameters share a shape
nothing promotes the float32 one at all, so the whole score runs in float32.
``Binomial(10_000_000, probs=float64)`` scored ``log P(X = 5e6)`` as 2.23 (true
value -8.28) because ``lgamma(n + 1)`` ran in float32.

The rule is the reference framework's, applied where parameters are turned into
tensors: a number takes the floating dtype and the device of the first tensor
parameter beside it, and the default dtype when there is none.  The guard walks
every distribution rather than the one that was found: each scalar parameter in
turn is the tensor and the others are numbers, so no constructor can go back to
converting its parameters one at a time unseen.
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions._util import _promote_params

_Build = Callable[[Any, dict[str, Any]], Any]


@dataclass(frozen=True)
class _Case:
    """How to build one distribution for the sweep.

    ``numbers`` are the scalar parameters, each a Python number unless it is
    the anchor; ``tensors`` can only be tensors.  The numbers are inexact in
    float32 so that a rounded parameter shows.
    """

    numbers: dict[str, float]
    value: Any
    tensors: dict[str, Any] = field(default_factory=dict)
    build: _Build | None = None
    xfail: str | None = None
    # The reference cannot be built, or cannot score, in this form.
    no_reference: str | None = None


_PROBS = [0.2, 0.3, 0.5]
_COV = [[2.0, 0.5], [0.5, 1.0]]
# StudentT converts its parameters in a file another change holds.  ``strict``
# turns this into a failure the day it is fixed, so the marker cannot linger.
_STUDENT = "LCD-1040: StudentT converts its parameters in student.py (held by LCD-1037)"
_TEMPERATURE = "the reference takes a tensor temperature only"
_GENERALIZED_PARETO = "the reference mixes float32 and float64 in GeneralizedPareto.log_prob"


def _independent(M: Any, kw: dict[str, Any]) -> Any:
    return M.Independent(M.Normal(kw["loc"], kw["scale"]), 1)


def _mixture(M: Any, kw: dict[str, Any]) -> Any:
    return M.MixtureSameFamily(M.Categorical(probs=kw["probs"]), M.Normal(kw["loc"], kw["scale"]))


def _lkj(M: Any, kw: dict[str, Any]) -> Any:
    return M.LKJCholesky(2, kw["concentration"])


def _logistic(M: Any, kw: dict[str, Any]) -> Any:
    return M.LogisticNormal(kw["loc"], kw["scale"])


def _multinomial(M: Any, kw: dict[str, Any]) -> Any:
    # The reference takes an ``int`` count; this framework takes a tensor too.
    total = kw["total_count"]
    return M.Multinomial(total if M is D else int(total), probs=kw["probs"])


_CASES: dict[str, _Case] = {
    "Normal": _Case({"loc": 0.1, "scale": 1.3}, 0.7),
    "LogNormal": _Case({"loc": 0.1, "scale": 1.3}, 1.3),
    "Uniform": _Case({"low": 0.1, "high": 2.3}, 0.7),
    "Laplace": _Case({"loc": 0.1, "scale": 1.3}, 0.7),
    "Cauchy": _Case({"loc": 0.1, "scale": 1.3}, 0.7),
    "Gumbel": _Case({"loc": 0.1, "scale": 1.3}, 0.7),
    "StudentT": _Case({"df": 3.1, "loc": 0.1, "scale": 1.3}, 0.7, xfail=_STUDENT),
    "Pareto": _Case({"scale": 1.1, "alpha": 2.3}, 1.7),
    "GeneralizedPareto": _Case(
        {"loc": 0.1, "scale": 1.3, "concentration": 0.3}, 1.7, no_reference=_GENERALIZED_PARETO
    ),
    "Weibull": _Case({"scale": 1.1, "concentration": 1.7}, 0.7),
    "FisherSnedecor": _Case({"df1": 3.1, "df2": 5.3}, 0.7),
    "VonMises": _Case({"loc": 0.1, "concentration": 1.3}, 0.7),
    "Gamma": _Case({"concentration": 2.1, "rate": 1.3}, 0.7),
    "Beta": _Case({"concentration1": 2.1, "concentration0": 3.3}, 0.3),
    "InverseGamma": _Case({"concentration": 2.1, "rate": 1.3}, 0.7),
    "Kumaraswamy": _Case({"concentration1": 2.1, "concentration0": 3.3}, 0.3),
    "Binomial": _Case({"total_count": 5, "probs": 0.3}, 2.0),
    "NegativeBinomial": _Case({"total_count": 3.1, "probs": 0.3}, 2.0),
    "RelaxedBernoulli": _Case({"temperature": 0.7, "probs": 0.3}, 0.4, no_reference=_TEMPERATURE),
    "Multinomial": _Case(
        {"total_count": 5}, [1.0, 2.0, 2.0], {"probs": _PROBS}, build=_multinomial
    ),
    "RelaxedOneHotCategorical": _Case(
        {"temperature": 0.7}, _PROBS, {"probs": _PROBS}, no_reference=_TEMPERATURE
    ),
    "Wishart": _Case({"df": 3.1}, [[2.0, 0.3], [0.3, 1.0]], {"covariance_matrix": _COV}),
    "LogisticNormal": _Case({"scale": 1.3}, _PROBS, {"loc": [0.0, 0.5]}, build=_logistic),
    "Independent": _Case(
        {"scale": 1.3}, [0.1, 0.9, 2.2], {"loc": [0.0, 1.0, 2.0]}, build=_independent
    ),
    "MixtureSameFamily": _Case(
        {"scale": 1.3}, 0.7, {"probs": [0.3, 0.7], "loc": [0.0, 1.0]}, build=_mixture
    ),
    # One scalar parameter and no tensor beside it: nothing to follow, so these
    # only say what a number alone becomes.
    "Chi2": _Case({"df": 3.1}, 0.7),
    "Exponential": _Case({"rate": 0.7}, 0.7),
    "HalfNormal": _Case({"scale": 1.3}, 0.7),
    "HalfCauchy": _Case({"scale": 1.3}, 0.7),
    "Poisson": _Case({"rate": 1.7}, 2.0),
    "Bernoulli": _Case({"probs": 0.3}, 1.0),
    "Geometric": _Case({"probs": 0.3}, 2.0),
    "ContinuousBernoulli": _Case({"probs": 0.3}, 0.4),
    "LKJCholesky": _Case({"concentration": 1.3}, [[1.0, 0.0], [0.6, 0.8]], build=_lkj),
}

# Count and relaxed laws take their success probability as ``logits`` as well.
_LOGIT_CASES: dict[str, _Case] = {
    "Binomial(logits)": _Case({"total_count": 5, "logits": 0.3}, 2.0),
    "NegativeBinomial(logits)": _Case({"total_count": 3.1, "logits": 0.3}, 2.0),
    "RelaxedBernoulli(logits)": _Case(
        {"temperature": 0.7, "logits": 0.3}, 0.4, no_reference=_TEMPERATURE
    ),
}

# Laws whose parameters are all tensors — a probability vector, a matrix — or
# that hold none of their own.  Listing them is what makes a new law declare
# its scalar parameters here.
_NO_SCALARS = frozenset(
    {
        "Categorical",
        "OneHotCategorical",
        "OneHotCategoricalStraightThrough",
        "Dirichlet",
        "MultivariateNormal",
        "LowRankMultivariateNormal",
        "TransformedDistribution",
    }
)

_ALL: dict[str, _Case] = {**_CASES, **_LOGIT_CASES}


def _cls(name: str) -> str:
    return name.split("(")[0]


def _anchors(case: _Case) -> list[str | None]:
    """The scalar that is a tensor in each build; ``None`` when a tensor-only
    parameter is the one beside the numbers, and nothing when no tensor can be."""
    if case.tensors:
        return [None]
    if len(case.numbers) > 1:
        return list(case.numbers)
    return []


# Every build that has at least one number beside a tensor.
_MIXED = [(n, a) for n, c in _ALL.items() for a in _anchors(c)]
_ALONE = [n for n, c in _ALL.items() if not c.tensors]


def _mixed(rows: list[tuple[str, str | None]] = _MIXED, *, float64: bool = False) -> list[Any]:
    """Parameters for a sweep; ``float64`` carries the xfail of a law that still
    rounds its numbers — the float32 and structure rows of it are fine."""

    def marks(name: str) -> list[pytest.MarkDecorator]:
        reason = _ALL[name].xfail
        return [pytest.mark.xfail(strict=True, reason=reason)] if reason and float64 else []

    return [pytest.param(n, a, marks=marks(n), id=f"{n}-{a or 'tensors'}") for n, a in rows]


def _tensor(v: Any, dtype: Any, device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor(v, dtype=dtype, device=device)


def _make(
    M: Any,
    name: str,
    anchor: str | None,
    make_tensor: Callable[[Any], Any],
    *,
    as_tensors: bool = False,
) -> Any:
    """Build ``name`` with ``anchor`` a tensor and its other scalars numbers.

    ``as_tensors`` makes every scalar a tensor instead — the answer the
    numbers must reproduce.
    """
    case = _ALL[name]
    kw: dict[str, Any] = dict(case.numbers)
    if as_tensors:
        kw = {k: make_tensor(float(v)) for k, v in kw.items()}
    elif anchor is not None:
        kw[anchor] = make_tensor(float(case.numbers[anchor]))
    kw |= {k: make_tensor(v) for k, v in case.tensors.items()}
    if case.build is not None:
        return case.build(M, kw)
    return getattr(M, _cls(name))(**kw)


def _tensors_of(dist: Any) -> Iterator[tuple[str, Any]]:
    """Every tensor a distribution holds, and those of the ones it wraps."""
    for key, val in vars(dist).items():
        if isinstance(val, lucid.Tensor):
            yield key, val
        elif isinstance(val, D.Distribution):
            yield from _tensors_of(val)


def _float_tensors(dist: Any) -> dict[str, Any]:
    return {k: t for k, t in _tensors_of(dist) if t.is_floating_point()}


_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "log_prob": lambda d, v: d.log_prob(v),
    "mean": lambda d, v: d.mean,
    "variance": lambda d, v: d.variance,
    "entropy": lambda d, v: d.entropy(),
}


def _outcome(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except NotImplementedError:
        return NotImplementedError


def _same(got: Any, want: Any, *, rtol: float, atol: float, what: str) -> None:
    """Equal to a tolerance, an infinity or a NaN matching its twin."""
    np.testing.assert_allclose(
        np.asarray(got.numpy(), dtype=np.float64),
        np.asarray(want.numpy(), dtype=np.float64),
        rtol=rtol,
        atol=atol,
        err_msg=what,
    )


def _f64(v: Any) -> lucid.Tensor:
    return _tensor(v, lucid.float64)


def _f32(v: Any) -> lucid.Tensor:
    return _tensor(v, lucid.float32)


def test_every_distribution_is_swept() -> None:
    swept = {_cls(n) for n in _ALL} | _NO_SCALARS
    every = {
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    }
    assert swept == every


@pytest.mark.parametrize("name,anchor", _mixed(float64=True))
def test_numbers_take_the_float64_of_the_tensors_beside_them(name: str, anchor: str | None) -> None:
    dist = _make(D, name, anchor, _f64)
    for key, t in _float_tensors(dist).items():
        assert t.dtype == lucid.float64, f"{name}.{key} is {t.dtype}"


@pytest.mark.parametrize("name,anchor", _mixed(float64=True))
def test_a_number_keeps_its_float64_value(name: str, anchor: str | None) -> None:
    """``0.1`` is held as ``0.1``, not as the float32 nearest to it."""
    dist = _make(D, name, anchor, _f64)
    held = {k.lstrip("_"): t for k, t in _float_tensors(dist).items()}
    for param, number in _ALL[name].numbers.items():
        if param == anchor or param not in held or held[param].ndim != 0:
            continue
        assert float(held[param].item()) == float(number), f"{name}.{param}"


@pytest.mark.parametrize("name,anchor", _mixed(float64=True))
def test_float64_results_match_the_all_tensor_build(name: str, anchor: str | None) -> None:
    """Numbers and float64 tensors in one signature score as all float64 tensors."""
    mixed = _make(D, name, anchor, _f64)
    full = _make(D, name, anchor, _f64, as_tensors=True)
    value = _f64(_ALL[name].value)
    for op, fn in _OPERATIONS.items():
        want = _outcome(lambda: fn(full, value))
        got = _outcome(lambda: fn(mixed, value))
        if want is NotImplementedError:
            assert got is NotImplementedError, f"{name}.{op} exists only for numbers"
            continue
        assert got is not NotImplementedError, f"{name}.{op} is missing for numbers"
        assert got.dtype == lucid.float64, f"{name}.{op} answered {got.dtype}"
        assert got.shape == want.shape, f"{name}.{op} shape"
        _same(got, want, rtol=1e-12, atol=0.0, what=f"{name}.{op}")


@pytest.mark.parametrize("name,anchor", _mixed())
def test_float32_is_unchanged(name: str, anchor: str | None) -> None:
    """The float32 build reproduces the all-tensor float32 build."""
    mixed = _make(D, name, anchor, _f32)
    full = _make(D, name, anchor, _f32, as_tensors=True)
    value = _f32(_ALL[name].value)
    for key, t in _float_tensors(mixed).items():
        assert t.dtype == lucid.float32, f"{name}.{key} is {t.dtype}"
    for op, fn in _OPERATIONS.items():
        want = _outcome(lambda: fn(full, value))
        if want is NotImplementedError:
            continue
        got = fn(mixed, value)
        assert got.dtype == lucid.float32, f"{name}.{op} answered {got.dtype}"
        _same(got, want, rtol=1e-5, atol=1e-6, what=f"{name}.{op}")


_SCALAR_ANCHORS = [(n, a) for n, a in _MIXED if a is not None and not _ALL[n].build]


@pytest.mark.parametrize("name,anchor", _mixed(_SCALAR_ANCHORS))
def test_a_batched_tensor_sets_the_batch_and_the_numbers_follow_its_dtype(
    name: str, anchor: str
) -> None:
    """Broadcasting is unchanged: the batch is the tensor's, the dtype float64."""
    case = _ALL[name]
    kw: dict[str, Any] = dict(case.numbers)
    kw[anchor] = _f64([float(case.numbers[anchor]) * f for f in (1.0, 2.0, 3.0)])
    dist = getattr(D, _cls(name))(**kw)
    assert tuple(dist.batch_shape) == (3,)
    for key, t in _float_tensors(dist).items():
        assert t.dtype == lucid.float64, f"{name}.{key} is {t.dtype}"


@pytest.mark.parametrize("name", _ALONE)
def test_numbers_alone_take_the_default_dtype(name: str) -> None:
    """With no tensor to follow, a number is a tensor of the default dtype."""
    previous = lucid.get_default_dtype()
    try:
        for dtype in (lucid.float64, lucid.float32):
            lucid.set_default_dtype(dtype)
            floats = _float_tensors(_make(D, name, None, _f32))
            assert floats, f"{name} holds no floating tensor"
            for key, t in floats.items():
                assert t.dtype == dtype, f"{name}.{key} is {t.dtype}, default {dtype}"
    finally:
        lucid.set_default_dtype(previous)


def test_the_reported_binomial_scores_in_float64() -> None:
    """``Binomial(10_000_000, probs=float64)`` — the case that scored 2.23."""
    probs = _f64(0.5)
    n = 10_000_000
    k = _f64(n / 2)
    got = D.Binomial(n, probs=probs).log_prob(k)
    as_tensor = D.Binomial(_f64(float(n)), probs=probs).log_prob(k)
    exact = math.lgamma(n + 1) - 2 * math.lgamma(n // 2 + 1) + n * math.log(0.5)
    assert got.dtype == lucid.float64
    assert got.item() == as_tensor.item()
    assert abs(got.item() - exact) < 1e-6


@pytest.mark.parametrize("n", [10_000_000, 16_777_217, 123_456_789])
def test_a_large_count_is_not_rounded_to_float32(n: int) -> None:
    """``n`` above ``2**24`` is not representable in float32; float64 holds it."""
    dist = D.Binomial(n, probs=_f64(0.25))
    assert dist.total_count.dtype == lucid.float64
    assert int(dist.total_count.item()) == n


def test_multinomial_total_count_follows_the_probabilities() -> None:
    dist = D.Multinomial(7, probs=_f64([0.2, 0.3, 0.5]))
    assert dist._total_count.dtype == lucid.float64
    assert dist.mean.dtype == lucid.float64


def test_chi2_rate_follows_the_degrees_of_freedom() -> None:
    dist = D.Chi2(_f64(3.1))
    assert dist.rate.dtype == lucid.float64
    assert dist.concentration.dtype == lucid.float64


def test_an_integer_count_tensor_keeps_a_numeric_probability() -> None:
    """The reference truncates ``0.3`` to ``0`` beside an integer tensor; Lucid must not."""
    dist = D.Binomial(lucid.tensor(5, dtype=lucid.int64), probs=0.3)
    assert abs(float(dist.probs.item()) - 0.3) < 1e-6


class TestPromoteParams:
    """The helper every distribution constructor goes through."""

    def test_a_number_takes_the_dtype_of_the_first_floating_tensor(self) -> None:
        a, b, c = _promote_params(0.1, _f64(1.0), _f32(2.0))
        assert (a.dtype, b.dtype, c.dtype) == (lucid.float64, lucid.float64, lucid.float32)
        assert a.item() == 0.1

    def test_the_first_tensor_wins_when_two_tensors_differ(self) -> None:
        a, _, _ = _promote_params(0.1, _f32(1.0), _f64(2.0))
        assert a.dtype == lucid.float32

    def test_an_integer_tensor_does_not_make_a_number_an_integer(self) -> None:
        count, p = _promote_params(_tensor(5, lucid.int64), 0.3)
        assert count.dtype == lucid.int64
        assert p.dtype == lucid.get_default_dtype()
        assert abs(p.item() - 0.3) < 1e-6

    def test_a_floating_tensor_after_an_integer_one_is_the_one_followed(self) -> None:
        _, p, _ = _promote_params(_tensor(5, lucid.int64), 0.3, _f64(1.0))
        assert p.dtype == lucid.float64

    def test_without_a_tensor_numbers_take_the_default_dtype(self) -> None:
        a, b = _promote_params(0.1, 3)
        assert a.dtype == b.dtype == lucid.get_default_dtype()
        assert a.ndim == b.ndim == 0

    def test_a_tensor_passes_through_untouched(self) -> None:
        t = _f64([1.0, 2.0])
        (out,) = _promote_params(t)
        assert out is t

    def test_numbers_are_zero_dimensional(self) -> None:
        a, _ = _promote_params(0.1, _f64([1.0, 2.0]))
        assert a.shape == ()

    def test_a_number_follows_the_tensor_to_its_device(self, device: str) -> None:
        a, _ = _promote_params(0.1, _tensor([1.0, 2.0], lucid.float32, device))
        assert a.device.type == device

    @pytest.mark.parametrize("bad", ["0.1", None, [0.1]])
    def test_a_non_number_is_refused(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="Tensor or a number"):
            _promote_params(bad, _f64(1.0))


def test_numbers_follow_the_tensor_to_its_device(device: str) -> None:
    """A number beside a Metal tensor is created there, in the tensor's dtype."""
    dist = D.Normal(0.1, _tensor(1.3, lucid.float32, device))
    assert dist.loc.device.type == device
    assert dist.loc.dtype == lucid.float32
    binom = D.Binomial(5, probs=_tensor(0.3, lucid.float32, device))
    assert binom.total_count.device.type == device
    assert binom.log_prob(_tensor(2.0, lucid.float32, device)).device.type == device


_REFERENCE_ROWS = [
    (n, a) for n, a in _MIXED if _ALL[n].no_reference is None and _ALL[n].build is not _lkj
]


def _dtype_name(t: Any) -> str:
    return str(t.dtype).rsplit(".", 1)[-1]


def _held_dtypes(dist: Any) -> dict[str, str]:
    """The dtype of each declared parameter that is held as a tensor."""
    held = {k: getattr(dist, k, None) for k in dist.arg_constraints}
    return {k: _dtype_name(v) for k, v in held.items() if hasattr(v, "dtype")}


@pytest.mark.parity
@pytest.mark.parametrize("name,anchor", _mixed(_REFERENCE_ROWS, float64=True))
def test_held_dtypes_match_the_reference(name: str, anchor: str | None, ref: ModuleType) -> None:
    """The reference holds the same parameters in float64 for the same build."""

    def ref_tensor(v: Any) -> Any:
        return ref.tensor(v, dtype=ref.float64)

    theirs = _make(ref.distributions, name, anchor, ref_tensor)
    ours = _make(D, name, anchor, _f64)
    value = _ALL[name].value
    mine = ours.log_prob(_f64(value))
    want = theirs.log_prob(ref_tensor(value))
    assert _dtype_name(mine) == _dtype_name(want), name
    if not hasattr(ours, "arg_constraints") or not ours.arg_constraints:
        return
    ours_held, theirs_held = _held_dtypes(ours), _held_dtypes(theirs)
    shared = ours_held.keys() & theirs_held.keys()
    assert {k: ours_held[k] for k in shared} == {k: theirs_held[k] for k in shared}, name
