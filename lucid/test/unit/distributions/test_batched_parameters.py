"""A distribution's batch shape is the broadcast of *all* its parameters (LCD-678).

``MultivariateNormal(zeros(2), covariance_matrix=stack([I, 2 * I]))`` reported
``batch_shape == ()`` — it read the batch off ``loc`` alone — so ``sample((4,))``
raised, ``entropy`` came out the wrong shape, and ``Independent(mvn, 1).entropy``
was halved.  ``Wishart(df=tensor([5., 6.]), covariance_matrix=2 * I)`` made the same
mistake from ``covariance_matrix``.

The invariant is a property of every distribution, so it is swept over all of
them: each parameter batched alone, then all together, and the batch shape and
event shape, the moments, the entropy, the density of a point, and the shape of
a draw must match the reference.
"""

import math
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.distributions as D

_S2 = [[1.0, 0.2], [0.2, 2.0]]
_S2_ALT = [[2.0, 0.3], [0.3, 1.0]]
_L2 = [[1.0, 0.0], [0.2, 1.4]]
_L2_ALT = [[1.5, 0.0], [0.3, 0.9]]
_P2 = [[1.0, -0.1], [-0.1, 0.6]]
_P2_ALT = [[0.8, 0.2], [0.2, 1.1]]

# name -> (parameters {name: (base, alt)}, fixed kwargs, a point for log_prob and its dtype)
_Spec = tuple[dict[str, tuple[Any, Any]], dict[str, Any], Any]

_SPECS: dict[str, _Spec] = {
    "Normal": ({"loc": (0.3, -0.7), "scale": (1.2, 2.0)}, {}, 0.5),
    "LogNormal": ({"loc": (0.3, -0.7), "scale": (1.2, 2.0)}, {}, 1.5),
    "Uniform": ({"low": (0.0, -1.0), "high": (2.0, 3.0)}, {}, 0.5),
    "Exponential": ({"rate": (1.5, 0.7)}, {}, 0.5),
    "Laplace": ({"loc": (0.3, -0.7), "scale": (1.2, 2.0)}, {}, 0.4),
    "Cauchy": ({"loc": (0.3, -0.7), "scale": (1.2, 2.0)}, {}, 0.4),
    "StudentT": ({"df": (3.0, 5.5), "loc": (0.1, -0.2), "scale": (1.1, 0.8)}, {}, 0.3),
    "Pareto": ({"scale": (1.0, 2.0), "alpha": (3.0, 4.5)}, {}, 2.5),
    "GeneralizedPareto": (
        {"loc": (0.0, 0.5), "scale": (1.0, 1.5), "concentration": (0.2, 0.4)},
        {},
        1.2,
    ),
    "Weibull": ({"scale": (1.0, 2.0), "concentration": (1.5, 2.5)}, {}, 0.8),
    "HalfNormal": ({"scale": (1.0, 2.0)}, {}, 0.7),
    "HalfCauchy": ({"scale": (1.0, 2.0)}, {}, 0.7),
    "FisherSnedecor": ({"df1": (5.0, 7.0), "df2": (9.0, 11.0)}, {}, 1.2),
    "VonMises": ({"loc": (0.3, -0.7), "concentration": (1.0, 2.0)}, {}, 0.3),
    "Gamma": ({"concentration": (2.0, 3.0), "rate": (1.5, 0.8)}, {}, 1.0),
    "Chi2": ({"df": (3.0, 4.0)}, {}, 1.5),
    "Beta": ({"concentration1": (2.0, 3.0), "concentration0": (3.0, 1.5)}, {}, 0.4),
    "Dirichlet": ({"concentration": ([1.5, 2.5, 3.0], [2.0, 1.0, 4.0])}, {}, [0.2, 0.3, 0.5]),
    "Bernoulli": ({"probs": (0.3, 0.6)}, {}, 1.0),
    "Geometric": ({"probs": (0.3, 0.6)}, {}, 2.0),
    "Poisson": ({"rate": (2.0, 3.5)}, {}, 2.0),
    "Binomial": ({"total_count": (5.0, 7.0), "probs": (0.4, 0.6)}, {}, 2.0),
    "NegativeBinomial": ({"total_count": (5.0, 7.0), "probs": (0.4, 0.6)}, {}, 2.0),
    "Binomial-logits": ({"total_count": (5.0, 7.0), "logits": (0.3, -0.5)}, {}, 2.0),
    "NegativeBinomial-logits": ({"total_count": (5.0, 7.0), "logits": (0.3, -0.5)}, {}, 2.0),
    "Bernoulli-logits": ({"logits": (0.3, -0.5)}, {}, 1.0),
    "Categorical-logits": ({"logits": ([0.1, 0.3, -0.2], [0.4, -0.3, 0.2])}, {}, 1.0),
    "RelaxedBernoulli-logits": ({"temperature": (0.5, 0.8), "logits": (0.3, -0.5)}, {}, 0.4),
    "Gumbel": ({"loc": (0.3, -0.7), "scale": (1.2, 2.0)}, {}, 0.5),
    "InverseGamma": ({"concentration": (3.0, 4.0), "rate": (2.0, 1.5)}, {}, 0.7),
    "Kumaraswamy": ({"concentration1": (2.0, 3.0), "concentration0": (3.0, 2.0)}, {}, 0.4),
    "LogisticNormal": (
        {"loc": ([0.1, 0.2], [0.3, -0.1]), "scale": ([1.0, 1.2], [0.8, 1.1])},
        {},
        [0.2, 0.3, 0.5],
    ),
    "ContinuousBernoulli": ({"probs": (0.3, 0.6)}, {}, 0.4),
    "RelaxedBernoulli": ({"temperature": (0.5, 0.8), "probs": (0.3, 0.6)}, {}, 0.4),
    "RelaxedOneHotCategorical": (
        {"temperature": (0.5, 0.8), "probs": ([0.2, 0.3, 0.5], [0.5, 0.3, 0.2])},
        {},
        [0.2, 0.3, 0.5],
    ),
    "Categorical": ({"probs": ([0.2, 0.3, 0.5], [0.5, 0.3, 0.2])}, {}, 1.0),
    "OneHotCategorical": ({"probs": ([0.2, 0.3, 0.5], [0.5, 0.3, 0.2])}, {}, [0.0, 1.0, 0.0]),
    "Multinomial": (
        {"probs": ([0.2, 0.3, 0.5], [0.5, 0.3, 0.2])},
        {"total_count": 3},
        [1.0, 1.0, 1.0],
    ),
    "MVN-covariance": (
        {"loc": ([0.0, 1.0], [0.5, -0.5]), "covariance_matrix": (_S2, _S2_ALT)},
        {},
        [0.3, 0.4],
    ),
    "MVN-scale_tril": (
        {"loc": ([0.0, 1.0], [0.5, -0.5]), "scale_tril": (_L2, _L2_ALT)},
        {},
        [0.3, 0.4],
    ),
    "MVN-precision": (
        {"loc": ([0.0, 1.0], [0.5, -0.5]), "precision_matrix": (_P2, _P2_ALT)},
        {},
        [0.3, 0.4],
    ),
    "LowRankMultivariateNormal": (
        {
            "loc": ([0.0, 1.0], [0.5, -0.5]),
            "cov_factor": ([[1.0], [0.5]], [[0.7], [0.3]]),
            "cov_diag": ([1.0, 2.0], [1.5, 0.5]),
        },
        {},
        [0.1, 0.2],
    ),
    "Wishart-covariance": (
        {"df": (4.0, 5.0), "covariance_matrix": (_S2, _S2_ALT)},
        {},
        [[2.0, 0.3], [0.3, 1.5]],
    ),
    "Wishart-scale_tril": (
        {"df": (4.0, 5.0), "scale_tril": (_L2, _L2_ALT)},
        {},
        [[2.0, 0.3], [0.3, 1.5]],
    ),
    "Wishart-precision": (
        {"df": (4.0, 5.0), "precision_matrix": (_P2, _P2_ALT)},
        {},
        [[2.0, 0.3], [0.3, 1.5]],
    ),
    "LKJCholesky": (
        {"concentration": (1.0, 2.0)},
        {"dim": 3},
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]],
    ),
}

_INTEGER_POINT = {"Categorical"}
# ``mode`` is left out: where it is undefined the two frameworks differ on
# whether it is ``nan`` or a value, which is not the batch shape's business.
_ATTRIBUTES = ("mean", "variance")

# (class, parameter) pairs whose batching the reference itself cannot do, so
# there is nothing to compare against: only Lucid's own invariant is checked.
#  - RelaxedBernoulli.temperature: the reference reads the batch from ``probs``
#    alone, so a batched temperature alone is a batch of ``()`` that then
#    fails to broadcast when sampled.
_REFERENCE_CANNOT = {("RelaxedBernoulli", "temperature")}
# Not batched at all: the temperature divides the *event* axis, so a batched
# one has no defined meaning in either framework.
_NOT_BATCHABLE = {("RelaxedOneHotCategorical", "temperature")}
# Known parity gap, pinned in its own xfail below.
_NO_ENTROPY = {"Wishart"}


def _class_name(spec_name: str) -> str:
    stem = spec_name.split("-")[0]
    return "MultivariateNormal" if stem == "MVN" else stem


def _build(lib: ModuleType, device: str, name: str, batched: set[str]) -> Any:
    params, fixed, _ = _SPECS[name]
    kwargs: dict[str, Any] = dict(fixed)
    for key, (base, alt) in params.items():
        value = [base, alt] if key in batched else base
        kwargs[key] = lib.tensor(value, dtype=lib.float32, device=device)
    return getattr(lib.distributions, _class_name(name))(**kwargs)


def _point(lib: ModuleType, device: str, name: str) -> Any:
    point = _SPECS[name][2]
    dtype = lib.int64 if _class_name(name) in _INTEGER_POINT else lib.float32
    return lib.tensor(point, dtype=dtype, device=device)


def _values(x: Any) -> list[float]:
    return x.detach().cpu().reshape(-1).tolist()


def _read(call: Any) -> Any:
    try:
        return call()
    except NotImplementedError:
        return None


def _agree(got: Any, want: Any, what: str) -> None:
    assert tuple(got.shape) == tuple(want.shape), (what, tuple(got.shape), tuple(want.shape))
    for g, w in zip(_values(got), _values(want), strict=True):
        if math.isnan(w):
            assert math.isnan(g), what
        else:
            assert g == pytest.approx(w, rel=2e-3, abs=2e-4), what


def _cases() -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, (params, _, _) in _SPECS.items():
        batchable = [k for k in params if (_class_name(name), k) not in _NOT_BATCHABLE]
        for key in batchable:
            out.append((name, (key,)))
        if len(batchable) > 1:
            out.append((name, tuple(batchable)))
    return out


_CASES = _cases()
_IDS = [f"{name}[{'+'.join(keys)}]" for name, keys in _CASES]


def _own_invariants(mine: Any, device: str, name: str) -> None:
    """What holds of the batch without any reference: shapes follow the broadcast."""
    batch = tuple(mine.batch_shape)
    assert batch == (2,), batch
    event = tuple(mine.event_shape)
    assert tuple(mine.sample((4,)).shape) == (4, *batch, *event)
    if mine.has_rsample:
        assert tuple(mine.rsample((4,)).shape) == (4, *batch, *event)
    assert tuple(mine.log_prob(_point(lucid, device, name)).shape) == batch
    assert tuple(mine.expand((3, *batch)).batch_shape) == (3, *batch)


@pytest.mark.parametrize(("name", "keys"), _CASES, ids=_IDS)
def test_the_batch_is_the_broadcast_of_every_parameter(
    name: str, keys: tuple[str, ...], device: str
) -> None:
    _own_invariants(_build(lucid, device, name, set(keys)), device, name)


@pytest.mark.parity
@pytest.mark.parametrize(("name", "keys"), _CASES, ids=_IDS)
def test_batch_shape_and_values_match_the_reference(
    name: str, keys: tuple[str, ...], ref: ModuleType, device: str
) -> None:
    cls = _class_name(name)
    if any((cls, k) in _REFERENCE_CANNOT for k in keys):
        pytest.skip("the reference cannot batch this parameter")
    mine = _build(lucid, device, name, set(keys))
    theirs = _build(ref, "cpu", name, set(keys))
    assert tuple(mine.batch_shape) == tuple(theirs.batch_shape)
    assert tuple(mine.event_shape) == tuple(theirs.event_shape)

    for attribute in _ATTRIBUTES:
        want = _read(lambda a=attribute: getattr(theirs, a))
        if want is not None:
            _agree(getattr(mine, attribute), want, attribute)

    if cls not in _NO_ENTROPY:
        want_entropy = _read(theirs.entropy)
        if want_entropy is not None:
            _agree(mine.entropy(), want_entropy, "entropy")

    _agree(
        mine.log_prob(_point(lucid, device, name)),
        theirs.log_prob(_point(ref, "cpu", name)),
        "log_prob",
    )

    lucid.manual_seed(0)
    ref.manual_seed(0)
    assert tuple(mine.sample((4,)).shape) == tuple(theirs.sample((4,)).shape)
    if mine.has_rsample:
        assert tuple(mine.rsample((4,)).shape) == tuple(theirs.rsample((4,)).shape)


@pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="Wishart.entropy is not implemented"
)
def test_wishart_entropy_is_implemented() -> None:
    d = D.Wishart(lucid.tensor(4.0), covariance_matrix=lucid.tensor(_S2))
    d.entropy()


# ── the reported cases, verbatim ────────────────────────────────────────────


def _eye_stack(device: str):
    eye = lucid.eye(2, device=device)
    return lucid.stack([eye, 2.0 * eye])


@pytest.mark.parametrize("parametrisation", ["covariance_matrix", "scale_tril", "precision_matrix"])
def test_a_stack_of_matrices_over_one_mean_is_a_batch(parametrisation: str, device: str) -> None:
    d = D.MultivariateNormal(lucid.zeros(2, device=device), **{parametrisation: _eye_stack(device)})
    assert d.batch_shape == (2,) and d.event_shape == (2,)
    assert d.sample((4,)).shape == (4, 2, 2)
    assert d.entropy().shape == (2,)
    assert d.mean.shape == d.variance.shape == (2, 2)
    assert d.log_prob(lucid.zeros(2, device=device)).shape == (2,)


def test_independent_over_a_batched_mvn_sums_both_entropies(device: str) -> None:
    mvn = D.MultivariateNormal(lucid.zeros(2, device=device), covariance_matrix=_eye_stack(device))
    wrapped = D.Independent(mvn, 1)
    assert wrapped.batch_shape == () and wrapped.event_shape == (2, 2)
    assert float(wrapped.entropy().item()) == pytest.approx(float(mvn.entropy().sum().item()))


def test_wishart_df_over_one_scale_is_a_batch(device: str) -> None:
    eye3 = 2.0 * lucid.eye(3, device=device)
    d = D.Wishart(lucid.tensor([5.0, 6.0], device=device), covariance_matrix=eye3)
    assert d.batch_shape == (2,) and d.event_shape == (3, 3)
    assert d.sample((4,)).shape == (4, 2, 3, 3)
    assert d.mean.shape == d.variance.shape == (2, 3, 3)
    # Each df scales its own row of the mean.
    assert float(d.mean[1, 0, 0].item()) == pytest.approx(6.0 * 2.0)
    assert float(d.mean[0, 0, 0].item()) == pytest.approx(5.0 * 2.0)


def test_the_batched_covariance_is_trained_through(device: str) -> None:
    cov = _eye_stack(device).requires_grad_(True)
    d = D.MultivariateNormal(lucid.zeros(2, device=device), covariance_matrix=cov)
    d.log_prob(lucid.ones(2, device=device)).sum().backward()
    assert cov.grad is not None and cov.grad.shape == cov.shape


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([1.0, 1.0], "at least two-dimensional"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square matrix"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "3 x 3"),
    ],
    ids=["one-dimensional", "not square", "wrong size for loc"],
)
def test_a_matrix_that_does_not_fit_is_refused_at_construction(matrix: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        D.MultivariateNormal(lucid.zeros(2), covariance_matrix=lucid.tensor(matrix))


def test_a_scalar_loc_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one-dimensional"):
        D.MultivariateNormal(lucid.tensor(0.0), covariance_matrix=lucid.eye(2))


def test_wishart_refuses_a_matrix_that_is_not_square() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        D.Wishart(lucid.tensor(5.0), covariance_matrix=lucid.ones(2, 3))
