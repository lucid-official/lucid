"""Broadcasting a distribution's parameters must not touch their values (LCD-1044).

``_broadcast_pair`` brought two parameters to one shape by adding ``a * 0 + b * 0``
to each.  Beside an infinity that is ``inf * 0``, a NaN, so ``Uniform(0.0,
[1.0, inf])`` held ``low = [0, nan]`` and ``Normal(0.1, [1.0, inf])`` held
``loc = [0.1, nan]``: a finite parameter spoiled by its neighbour's lane.  The
same arithmetic tied every broadcast parameter to the autograd graph
(``Normal(0.1, leaf).mean.requires_grad`` was true, and the distribution could
not be deep-copied).

A parameter is broadcast as a view now.  The guard walks every law that takes
two or more broadcast parameters: each parameter in turn is a vector holding an
infinity, a negative infinity or a NaN in one lane and finite values in the
others, and the rest are numbers or 0-dim tensors.  Every held parameter must
come out bit for bit what went in.
"""

import copy
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid._C import engine as _C_engine

# Each law's broadcast parameters in constructor order, with a finite value each.
_LAWS: dict[str, dict[str, float]] = {
    "Normal": {"loc": 0.5, "scale": 1.5},
    "LogNormal": {"loc": 0.5, "scale": 1.5},
    "Uniform": {"low": 0.5, "high": 1.5},
    "Laplace": {"loc": 0.5, "scale": 1.5},
    "Cauchy": {"loc": 0.5, "scale": 1.5},
    "Gumbel": {"loc": 0.5, "scale": 1.5},
    "StudentT": {"df": 3.5, "loc": 0.5, "scale": 1.5},
    "Pareto": {"scale": 1.5, "alpha": 2.5},
    "GeneralizedPareto": {"loc": 0.5, "scale": 1.5, "concentration": 0.25},
    "Weibull": {"scale": 1.5, "concentration": 2.5},
    "FisherSnedecor": {"df1": 3.5, "df2": 5.5},
    "VonMises": {"loc": 0.5, "concentration": 1.5},
    "Gamma": {"concentration": 2.5, "rate": 1.5},
    "Beta": {"concentration1": 2.5, "concentration0": 3.5},
    "InverseGamma": {"concentration": 2.5, "rate": 1.5},
    "Kumaraswamy": {"concentration1": 2.5, "concentration0": 3.5},
    "Binomial": {"total_count": 5.0, "probs": 0.3},
    "NegativeBinomial": {"total_count": 3.5, "probs": 0.3},
    "RelaxedBernoulli": {"temperature": 0.5, "probs": 0.3},
}

# The other laws, each with a reason it has no pair of scalar parameters to broadcast.
_ELSEWHERE = frozenset(
    {
        # One scalar parameter: nothing to broadcast it against.
        "Bernoulli",
        "Geometric",
        "ContinuousBernoulli",
        "Poisson",
        "Exponential",
        "HalfNormal",
        "HalfCauchy",
        "Chi2",
        "LKJCholesky",
        # Parameters are vectors or matrices, joined by their event shape.
        "Categorical",
        "OneHotCategorical",
        "OneHotCategoricalStraightThrough",
        "Dirichlet",
        "Multinomial",
        "RelaxedOneHotCategorical",
        "MultivariateNormal",
        "LowRankMultivariateNormal",
        "Wishart",
        # Wrappers: the parameters are the wrapped law's, swept above.
        "LogisticNormal",
        "Independent",
        "MixtureSameFamily",
        "TransformedDistribution",
    }
)

_LANES = 3
_BAD_LANE = 1
_NON_FINITE = [float("inf"), float("-inf"), float("nan")]
_NON_FINITE_IDS = ["inf", "-inf", "nan"]
_ROWS = [(name, anchor) for name, params in _LAWS.items() for anchor in params]
_ROW_IDS = [f"{name}-{anchor}" for name, anchor in _ROWS]


def _bits(t: lucid.Tensor) -> np.ndarray[Any, Any]:
    """The bit pattern of a floating tensor: equal only if every lane is, NaN included."""
    arr = np.ascontiguousarray(t.detach().numpy())
    return arr.view(np.uint64 if arr.dtype.itemsize == 8 else np.uint32)


def _lanes(value: float, bad: float | None = None) -> list[float]:
    lanes = [value] * _LANES
    if bad is not None:
        lanes[_BAD_LANE] = bad
    return lanes


def _build(
    name: str, anchor: str, vector: lucid.Tensor, *, form: str, dtype: lucid.dtype, device: str
) -> Any:
    """Law ``name`` with ``anchor`` the vector and the rest numbers or 0-dim tensors."""
    kw: dict[str, Any] = {}
    for param, value in _LAWS[name].items():
        if param == anchor:
            kw[param] = vector
        elif form == "number":
            kw[param] = value
        else:
            kw[param] = lucid.tensor(value, dtype=dtype, device=device)
    return getattr(D, name)(**kw, validate_args=False)


def test_every_distribution_is_swept_or_has_no_pair_to_broadcast() -> None:
    every = {
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    }
    assert not set(_LAWS) & _ELSEWHERE
    assert set(_LAWS) | _ELSEWHERE == every


@pytest.mark.parametrize("form", ["number", "tensor"])
@pytest.mark.parametrize("bad", _NON_FINITE, ids=_NON_FINITE_IDS)
@pytest.mark.parametrize("name,anchor", _ROWS, ids=_ROW_IDS)
def test_broadcasting_keeps_every_parameter_bit_for_bit(
    name: str, anchor: str, bad: float, form: str, device: str, float_dtype: lucid.dtype
) -> None:
    """An infinity or a NaN in one lane is that lane's alone; the rest stay finite."""
    finite = _LAWS[name][anchor]
    vector = lucid.tensor(_lanes(finite, bad), dtype=float_dtype, device=device)
    dist = _build(name, anchor, vector, form=form, dtype=float_dtype, device=device)
    assert tuple(dist.batch_shape) == (_LANES,)
    for param, value in _LAWS[name].items():
        held = getattr(dist, param)
        want = vector if param == anchor else lucid.tensor(_lanes(value), dtype=float_dtype)
        assert tuple(held.shape) == (_LANES,), f"{name}.{param}"
        assert held.dtype == float_dtype, f"{name}.{param} is {held.dtype}"
        assert held.device.type == device, f"{name}.{param} is on {held.device}"
        np.testing.assert_array_equal(_bits(held), _bits(want), err_msg=f"{name}.{param}")


# The reference builds an inner law of these parameters and validates it whatever
# ``validate_args`` says, so only the non-finite values that inner law accepts can be
# built there: +inf for a positive parameter, none for a probability.
_REFERENCE_VALIDATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("StudentT", "df"): ("inf",),
    ("FisherSnedecor", "df1"): ("inf",),
    ("FisherSnedecor", "df2"): ("inf",),
    ("RelaxedBernoulli", "probs"): (),
}


def _reference_rows() -> list[Any]:
    """``(law, parameter, bad value)`` rows the reference can be built for."""
    rows = []
    for name, anchor in _ROWS:
        if name == "GeneralizedPareto":
            continue  # the reference has no such law
        buildable = _REFERENCE_VALIDATES.get((name, anchor), _NON_FINITE_IDS)
        for bad, label in zip(_NON_FINITE, _NON_FINITE_IDS, strict=True):
            if label in buildable:
                rows.append(pytest.param(name, anchor, bad, id=f"{name}-{anchor}-{label}"))
    return rows


@pytest.mark.parity
@pytest.mark.parametrize("name,anchor,bad", _reference_rows())
def test_held_parameters_are_the_references(
    name: str, anchor: str, bad: float, ref: ModuleType
) -> None:
    """The reference holds the same parameters, lane for lane, for the same inputs."""

    def theirs_of(values: Any) -> Any:
        return ref.tensor(values, dtype=ref.float32)

    kw = {
        param: theirs_of(_lanes(value, bad) if param == anchor else value)
        for param, value in _LAWS[name].items()
    }
    theirs = getattr(ref.distributions, name)(**kw, validate_args=False)
    vector = lucid.tensor(_lanes(_LAWS[name][anchor], bad), dtype=lucid.float32)
    ours = _build(name, anchor, vector, form="tensor", dtype=lucid.float32, device="cpu")
    for param in _LAWS[name]:
        np.testing.assert_array_equal(
            getattr(ours, param).numpy(),
            getattr(theirs, param).detach().numpy(),
            err_msg=f"{name}.{param}",
        )


@pytest.mark.parametrize("name,anchor", _ROWS, ids=_ROW_IDS)
def test_a_number_beside_a_leaf_stays_detached(name: str, anchor: str, device: str) -> None:
    """A parameter nobody asked to differentiate must not join the graph.

    The leaf needs no broadcast, so it stays the very tensor it was; the
    numbers beside it are views of constants and carry no gradient.
    """
    leaf = lucid.tensor(_lanes(_LAWS[name][anchor]), device=device, requires_grad=True)
    dist = _build(name, anchor, leaf, form="number", dtype=lucid.float32, device=device)
    for param in _LAWS[name]:
        held = getattr(dist, param)
        if param == anchor:
            assert held is leaf, f"{name}.{param} was copied"
        else:
            assert not held.requires_grad, f"{name}.{param} joined the graph"


def test_a_mean_that_is_a_number_stays_detached_beside_a_leaf() -> None:
    """``Normal(0.1, leaf).mean.requires_grad`` was true: ``loc`` went through ``leaf * 0``."""
    leaf = lucid.tensor([1.0, 2.0], requires_grad=True)
    assert not D.Normal(0.1, leaf).mean.requires_grad


@pytest.mark.parametrize("name,anchor", _ROWS, ids=_ROW_IDS)
def test_a_broadcast_leaf_keeps_its_gradient(name: str, anchor: str, device: str) -> None:
    """Expanding a leaf keeps it in the graph, as the reference does."""
    leaf = lucid.tensor(_LAWS[name][anchor], device=device, requires_grad=True)
    other = next(p for p in _LAWS[name] if p != anchor)
    vector = lucid.tensor(_lanes(_LAWS[name][other]), device=device)
    kw: dict[str, Any] = {p: v for p, v in _LAWS[name].items() if p not in (anchor, other)}
    kw[anchor], kw[other] = leaf, vector
    dist = getattr(D, name)(**kw, validate_args=False)
    held = getattr(dist, anchor)
    assert tuple(held.shape) == (_LANES,)
    assert held.requires_grad, f"{name}.{anchor} left the graph"
    held.sum().backward()
    assert leaf.grad is not None
    assert float(leaf.grad.item()) == _LANES
    assert getattr(dist, other) is vector
    assert not vector.requires_grad


@pytest.mark.parametrize("requires_grad", [False, True], ids=["detached", "leaf"])
@pytest.mark.parametrize("name,anchor", _ROWS, ids=_ROW_IDS)
def test_a_distribution_can_be_deep_copied(
    name: str, anchor: str, requires_grad: bool, device: str
) -> None:
    """Deep-copying raised for any tensor in the graph that is not a user leaf."""
    vector = lucid.tensor(
        _lanes(_LAWS[name][anchor], float("inf")), device=device, requires_grad=requires_grad
    )
    dist = _build(name, anchor, vector, form="number", dtype=lucid.float32, device=device)
    twin = copy.deepcopy(dist)
    assert type(twin) is type(dist)
    for param in _LAWS[name]:
        got, want = getattr(twin, param), getattr(dist, param)
        assert got is not want, f"{name}.{param} was shared, not copied"
        np.testing.assert_array_equal(_bits(got), _bits(want), err_msg=f"{name}.{param}")


def test_a_host_scalar_tensor_follows_a_metal_parameter(device_gpu_only: str) -> None:
    """A 0-dim host *tensor* beside a Metal vector is a scalar; it moves, and keeps its value."""
    vector = lucid.tensor([1.0, float("inf"), 3.0], device="metal")
    dist = D.Normal(lucid.tensor(0.25), vector, validate_args=False)
    assert dist.loc.device.type == "metal"
    np.testing.assert_array_equal(dist.loc.numpy(), np.full(3, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(dist.scale.numpy(), vector.numpy())


def test_parameters_on_different_devices_are_refused_at_construction(
    cross_device_pair: tuple[str, str],
) -> None:
    """A view would hide the mismatch until the first op, far from the call that caused it."""
    cpu, metal = cross_device_pair
    with pytest.raises(_C_engine.DeviceMismatch, match="share a device"):
        D.Normal(lucid.tensor([0.0, 1.0], device=cpu), lucid.tensor([1.0, 2.0], device=metal))


def test_student_t_mean_with_an_infinite_df_is_the_location() -> None:
    """``loc + 0 * df`` was NaN at ``df = inf``; the mean there is the Normal's."""
    dist = D.StudentT(lucid.tensor([3.0, float("inf"), 0.5]), 0.25, validate_args=False)
    got = dist.mean.numpy()
    np.testing.assert_array_equal(got, np.array([0.25, 0.25, np.nan], dtype=np.float32))


@pytest.mark.parametrize("value", _NON_FINITE, ids=_NON_FINITE_IDS)
def test_uniform_scores_a_non_finite_value_as_outside_its_support(value: float) -> None:
    """``low + value * 0`` was NaN for such a value; the density is ``-inf`` outside."""
    dist = D.Uniform(0.0, lucid.tensor([2.0, 4.0]), validate_args=False)
    got = dist.log_prob(lucid.tensor([value, value]))
    np.testing.assert_array_equal(got.numpy(), np.full(2, -np.inf, dtype=np.float32))


def test_uniform_scores_an_integer_value() -> None:
    """The ``-inf`` fill took the value's dtype, so an integer value could not be scored."""
    dist = D.Uniform(0.0, lucid.tensor([2.0, 4.0]), validate_args=False)
    got = dist.log_prob(lucid.tensor([1, 5]))
    np.testing.assert_allclose(got.numpy(), [-np.log(2.0), -np.inf], rtol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("value", [*_NON_FINITE, 0.0, 0.5, 2.0, 7.0])
def test_uniform_log_prob_is_the_references(value: float, ref: ModuleType) -> None:
    ours = D.Uniform(0.0, lucid.tensor([2.0, 4.0]), validate_args=False)
    theirs = ref.distributions.Uniform(0.0, ref.tensor([2.0, 4.0]), validate_args=False)
    np.testing.assert_allclose(
        ours.log_prob(lucid.tensor([value, value])).numpy(),
        theirs.log_prob(ref.tensor([value, value])).detach().numpy(),
        rtol=1e-6,
    )


def test_uniform_scores_in_the_dtype_the_parameters_and_value_promote_to() -> None:
    """A float64 value beside float32 parameters is scored in float64, as ``Normal`` does."""
    dist = D.Uniform(0.0, lucid.tensor([2.0, 4.0]), validate_args=False)
    assert dist.log_prob(lucid.tensor([1.0, 1.0], dtype=lucid.float64)).dtype == lucid.float64
    assert dist.log_prob(lucid.tensor([1.0, 1.0])).dtype == lucid.float32


def test_the_unit_interval_density_is_positive_zero() -> None:
    """``-log(1)`` is ``-0``; the density of Uniform(0, 1) prints and compares as ``0``."""
    got = D.Uniform(0.0, 1.0).log_prob(lucid.tensor(0.5)).numpy()
    assert got == 0.0
    assert not np.signbit(got)
