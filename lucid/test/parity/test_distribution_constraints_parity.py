"""Every distribution accepts exactly the reference's parameters and support.

A distribution whose constraint is stricter than the reference's refuses a
parameter the reference accepts — ``Poisson(rate=0)`` was a ``ValueError``
under the default ``validate_args`` — and one whose constraint is looser
constructs happily and then answers NaN.  Either way the closed forms at the
edge of the domain become unreachable or wrong without anything raising.

The comparison is by behaviour, not by constraint class: each candidate is
offered to both libraries and the two verdicts must agree.

* ``test_parameter_domain`` builds the distribution with the candidate —
  Lucid under its default validation, the reference with
  ``validate_args=True`` — so the domain compared includes checks that live
  outside a constraint table (the reference's ``Geometric`` refuses
  ``probs == 0`` in its constructor, ``Uniform`` refuses ``low >= high``).
* ``test_support`` offers candidate values to ``support.check``.
* ``test_constraint_table`` compares the names in ``arg_constraints`` and the
  ``is_discrete`` / ``event_dim`` of the support.
* ``test_scored_value_validation`` offers the same values to ``log_prob`` and
  ``cdf`` with validation on: Lucid must refuse — with ``ValueError`` — exactly
  the values the reference refuses.  The check lives in the ``Distribution``
  base, so a family that slipped past it fails here.
* ``test_scored_values_without_validation`` scores, with validation off, every
  value the reference's support admits, and the answers must agree.
* ``test_mode`` asks both for ``mode``: present in one exactly when present in
  the other, and equal.
* ``test_own_samples_pass_validation`` scores a distribution's own draws with
  validation on, which must not refuse them.
* ``test_method_presence`` holds the family × {cdf, icdf, entropy} matrix:
  each is implemented exactly where the reference implements it — neither
  missing (``NotImplementedError`` where the reference answers) nor invented.
* ``test_method_values`` compares those three on a grid — every support value
  the reference admits for ``cdf``, probabilities from 0 to 1 for ``icdf`` —
  in float32 and float64 on CPU and float32 on Metal.
* ``test_method_gradients`` takes the gradient of each with respect to each
  parameter in turn: equal to the reference's, or refused by both (the
  reference's incomplete gamma is not differentiable in its shape).

The value and mode tests run on CPU and on Metal.

``_EXCEPTIONS`` lists every difference kept on purpose, with the reason.  A
listed difference must still be there — fixing one fails until its entry is
removed — and a difference that is not listed fails outright.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

pytestmark = [
    pytest.mark.parity,
    # The reference's Wishart warns about a small df on the way to refusing it.
    pytest.mark.filterwarnings("ignore:Low df values detected:UserWarning"),
]

_INF = math.inf
_NAN = math.nan

# Scalar candidates: both sides of 0 and 1, a non-integer above 1, an integer
# above 1, the infinities and NaN.
_SCALARS: list[float] = [-_INF, -1.0, -0.5, 0.0, 0.3, 1.0, 1.5, 3.0, _INF, _NAN]

_PROB_VECTORS: list[list[float]] = [
    [0.2, 0.3, 0.5],
    [0.0, 0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.5, 0.6, 0.1],
    [2.0, 3.0, 5.0],
    [-0.1, 0.6, 0.5],
    [0.0, 0.0, 0.0],
    [_INF, 1.0, 1.0],
    [_NAN, 0.5, 0.5],
]

_LOGIT_VECTORS: list[list[float]] = [
    [0.0, 1.0, 2.0],
    [-_INF, 0.0, 1.0],
    [-_INF, -_INF, -_INF],
    [_INF, 0.0, 0.0],
    [_NAN, 0.0, 0.0],
]

_CONCENTRATION_VECTORS: list[list[float]] = [
    [1.0, 2.0, 3.0],
    [0.5, 0.5, 0.5],
    [0.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [_INF, 1.0, 1.0],
    [_NAN, 1.0, 1.0],
]

_COVARIANCES: list[list[list[float]]] = [
    [[2.0, 0.5], [0.5, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]],
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[_NAN, 0.0], [0.0, 1.0]],
]

_SCALE_TRILS: list[list[list[float]]] = [
    [[1.0, 0.0], [0.5, 2.0]],
    [[1.0, 0.3], [0.5, 2.0]],
    [[-1.0, 0.0], [0.5, 2.0]],
    [[0.0, 0.0], [0.5, 1.0]],
]

_SCALAR_VALUES: list[float] = [
    -_INF,
    -1.0,
    -0.5,
    0.0,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.0,
    _INF,
    _NAN,
]

_VECTOR_VALUES: list[list[float]] = _PROB_VECTORS + [[0.0, 1.0, 0.0]]

_COUNT_VALUES: list[list[float]] = [
    [1.0, 2.0, 2.0],
    [0.0, 0.0, 5.0],
    [1.0, 1.0, 1.0],
    [3.0, 3.0, 0.0],
    [-1.0, 3.0, 3.0],
    [1.5, 1.5, 2.0],
]

_MATRIX_VALUES: list[list[list[float]]] = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[2.0, 0.5], [0.5, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]],
    [[1.0, 0.0], [0.6, 0.8]],
    [[1.0, 0.0], [0.5, 0.5]],
    [[1.0, 0.6], [0.0, 0.8]],
    [[1.0, 0.0], [-0.6, -0.8]],
    [[0.0, 0.0], [0.0, 0.0]],
]

_EVENT_VALUES: list[list[float]] = [[0.0, 1.0, 2.0], [-1.0, 0.0, 0.0], [_NAN, 0.0, 0.0]]

# ``build(M, t, validate_args, **override)`` constructs a distribution in the
# library ``M`` (``lucid.distributions`` or the reference's), with ``t``
# turning a nested list into that library's tensor and ``override``
# replacing one nominal parameter.
_Builder = Callable[..., Any]


class _Case:
    """How to build one distribution, what to offer each parameter, and
    which values to offer its support.  ``nominal`` holds the parameters a
    plain build uses, where they are scalars or vectors a gradient can be
    taken with respect to."""

    def __init__(
        self,
        build: _Builder,
        params: dict[str, list[Any]],
        values: list[Any],
        nominal: dict[str, Any] | None = None,
    ) -> None:
        self.build = build
        self.params = params
        self.values = values
        self.nominal = nominal or {}


def _plain(name: str, nominal: dict[str, Any], values: list[Any] | None = None) -> _Case:
    """A distribution whose parameters are all scalars passed by keyword."""

    def build(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
        kw = {k: t(o.get(k, v)) for k, v in nominal.items()}
        return getattr(M, name)(**kw, validate_args=va)

    return _Case(build, {k: _SCALARS for k in nominal}, values or _SCALAR_VALUES, nominal)


def _dual(
    name: str, nominal: Any, extra: dict[str, Any], params: dict[str, list[Any]], values: list[Any]
) -> _Case:
    """A distribution taking ``probs`` or ``logits``, plus other parameters."""

    def build(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
        kw = {
            k: (v if k == "total_count" and isinstance(v, int) else t(v))
            for k, v in (extra | {k: v for k, v in o.items() if k in extra}).items()
        }
        if "logits" in o:
            kw["logits"] = t(o["logits"])
        else:
            kw["probs"] = t(o.get("probs", nominal))
        return getattr(M, name)(**kw, validate_args=va)

    plain = {k: v for k, v in extra.items() if not isinstance(v, int)}
    return _Case(build, params, values, {"probs": nominal} | plain)


def _binary(name: str, extra: dict[str, Any] | None = None) -> _Case:
    """``probs`` / ``logits`` scalars, as ``Bernoulli``."""
    extra = extra or {}
    params = {"probs": _SCALARS, "logits": _SCALARS} | {k: _SCALARS for k in extra}
    return _dual(name, 0.3, extra, params, _SCALAR_VALUES)


def _categorical(name: str, extra: dict[str, Any] | None = None) -> _Case:
    """``probs`` / ``logits`` vectors, as ``Categorical``."""
    extra = extra or {}
    params: dict[str, list[Any]] = {"probs": _PROB_VECTORS, "logits": _LOGIT_VECTORS}
    params |= {k: _SCALARS for k in extra if k != "total_count"}
    return _dual(name, [0.2, 0.3, 0.5], extra, params, _VECTOR_VALUES)


def _dirichlet(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.Dirichlet(t(o.get("concentration", [1.0, 2.0, 3.0])), validate_args=va)


def _mvn(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    loc = t(o.get("loc", [0.0, 0.0]))
    for key in ("precision_matrix", "scale_tril"):
        if key in o:
            return M.MultivariateNormal(loc, **{key: t(o[key])}, validate_args=va)
    cov = t(o.get("covariance_matrix", [[2.0, 0.5], [0.5, 1.0]]))
    return M.MultivariateNormal(loc, covariance_matrix=cov, validate_args=va)


def _wishart(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    df = t(o.get("df", 3.0))
    for key in ("precision_matrix", "scale_tril"):
        if key in o:
            return M.Wishart(df, **{key: t(o[key])}, validate_args=va)
    cov = t(o.get("covariance_matrix", [[2.0, 0.5], [0.5, 1.0]]))
    return M.Wishart(df, covariance_matrix=cov, validate_args=va)


def _lkj(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.LKJCholesky(2, t(o.get("concentration", 1.5)), validate_args=va)


def _independent(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    base = M.Normal(t([0.0, 1.0, 2.0]), t([1.0, 1.0, 2.0]))
    return M.Independent(base, 1, validate_args=va)


def _transformed(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    base = M.Normal(t(0.0), t(1.0))
    return M.TransformedDistribution(base, [M.transforms.ExpTransform()], validate_args=va)


def _mixture(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.MixtureSameFamily(
        M.Categorical(probs=t([0.3, 0.7])), M.Normal(t([0.0, 1.0]), t([1.0, 2.0])), validate_args=va
    )


def _low_rank_mvn(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    loc = t(o.get("loc", [0.0, 1.0]))
    cov_factor = t(o.get("cov_factor", [[1.0], [0.5]]))
    cov_diag = t(o.get("cov_diag", [1.0, 2.0]))
    return M.LowRankMultivariateNormal(loc, cov_factor, cov_diag, validate_args=va)


def _logistic_normal(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    loc = t(o.get("loc", [0.0, 0.5]))
    scale = t(o.get("scale", [1.0, 1.5]))
    return M.LogisticNormal(loc, scale, validate_args=va)


_MATRIX_PARAMS: dict[str, list[Any]] = {
    "covariance_matrix": _COVARIANCES,
    "precision_matrix": _COVARIANCES,
    "scale_tril": _SCALE_TRILS,
}

CASES: dict[str, _Case] = {
    "Normal": _plain("Normal", {"loc": 0.5, "scale": 1.5}),
    "LogNormal": _plain("LogNormal", {"loc": 0.5, "scale": 1.5}),
    "Uniform": _plain("Uniform", {"low": 0.0, "high": 2.0}),
    "Exponential": _plain("Exponential", {"rate": 1.5}),
    "Laplace": _plain("Laplace", {"loc": 0.5, "scale": 1.5}),
    "Cauchy": _plain("Cauchy", {"loc": 0.5, "scale": 1.5}),
    "StudentT": _plain("StudentT", {"df": 3.0, "loc": 0.5, "scale": 1.5}),
    "Pareto": _plain("Pareto", {"scale": 1.0, "alpha": 2.0}),
    "GeneralizedPareto": _plain(
        "GeneralizedPareto", {"loc": 0.5, "scale": 1.5, "concentration": 0.25}
    ),
    "Weibull": _plain("Weibull", {"scale": 1.0, "concentration": 1.5}),
    "HalfNormal": _plain("HalfNormal", {"scale": 1.5}),
    "HalfCauchy": _plain("HalfCauchy", {"scale": 1.5}),
    "FisherSnedecor": _plain("FisherSnedecor", {"df1": 3.0, "df2": 5.0}),
    "VonMises": _plain("VonMises", {"loc": 0.5, "concentration": 1.5}),
    "Gamma": _plain("Gamma", {"concentration": 2.0, "rate": 1.5}),
    "Chi2": _plain("Chi2", {"df": 3.0}),
    "Beta": _plain("Beta", {"concentration1": 2.0, "concentration0": 3.0}),
    "Gumbel": _plain("Gumbel", {"loc": 0.5, "scale": 1.5}),
    "InverseGamma": _plain("InverseGamma", {"concentration": 2.0, "rate": 1.5}),
    "Kumaraswamy": _plain("Kumaraswamy", {"concentration1": 2.0, "concentration0": 3.0}),
    "LogisticNormal": _Case(
        _logistic_normal,
        {
            "loc": [[0.0, 0.5], [_INF, 0.0], [_NAN, 0.0]],
            "scale": [[1.0, 1.5], [0.0, 1.0], [-1.0, 1.0], [_NAN, 1.0]],
        },
        _VECTOR_VALUES,
        {"loc": [0.0, 0.5], "scale": [1.0, 1.5]},
    ),
    "Poisson": _plain("Poisson", {"rate": 1.5}),
    "Dirichlet": _Case(_dirichlet, {"concentration": _CONCENTRATION_VECTORS}, _VECTOR_VALUES),
    "Bernoulli": _binary("Bernoulli"),
    "Geometric": _binary("Geometric"),
    "ContinuousBernoulli": _binary("ContinuousBernoulli"),
    "Binomial": _binary("Binomial", {"total_count": 5.0}),
    "NegativeBinomial": _binary("NegativeBinomial", {"total_count": 3.0}),
    "RelaxedBernoulli": _binary("RelaxedBernoulli", {"temperature": 0.5}),
    "Categorical": _categorical("Categorical"),
    "OneHotCategorical": _categorical("OneHotCategorical"),
    "OneHotCategoricalStraightThrough": _categorical("OneHotCategoricalStraightThrough"),
    "RelaxedOneHotCategorical": _categorical("RelaxedOneHotCategorical", {"temperature": 0.5}),
    "Multinomial": _categorical("Multinomial", {"total_count": 5}),
    "MultivariateNormal": _Case(
        _mvn, {"loc": [[0.0, 0.0], [_INF, 0.0], [_NAN, 0.0]]} | _MATRIX_PARAMS, [[0.0, 1.0]]
    ),
    "LowRankMultivariateNormal": _Case(
        _low_rank_mvn,
        {
            "loc": [[0.0, 1.0], [_INF, 0.0], [_NAN, 0.0]],
            "cov_factor": [[[1.0], [0.5]], [[_INF], [0.5]], [[_NAN], [0.5]]],
            "cov_diag": [[1.0, 2.0], [0.0, 1.0], [-1.0, 1.0], [_NAN, 1.0]],
        },
        [[0.0, 1.0], [-1.0, 0.5], [_NAN, 0.0]],
        {"loc": [0.0, 1.0], "cov_factor": [[1.0], [0.5]], "cov_diag": [1.0, 2.0]},
    ),
    "Wishart": _Case(_wishart, {"df": _SCALARS} | _MATRIX_PARAMS, _MATRIX_VALUES),
    "LKJCholesky": _Case(_lkj, {"concentration": _SCALARS}, _MATRIX_VALUES),
    "Independent": _Case(_independent, {}, _EVENT_VALUES),
    "TransformedDistribution": _Case(_transformed, {}, _SCALAR_VALUES),
    "MixtureSameFamily": _Case(_mixture, {}, _SCALAR_VALUES),
}
CASES["Multinomial"].values = _COUNT_VALUES

# Every difference from the reference that is kept on purpose, keyed by
# (distribution, aspect) — the aspect is ``param:<name>``, ``support``,
# ``table``, ``validates:<method>``, ``values:<method>`` or ``mode`` — with the
# reason.
_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("Kumaraswamy", "values:log_prob"): "at 0 and 1 the density is 0 and Lucid "
    "answers -inf; the reference answers nan",
    ("Kumaraswamy", "mode"): "Lucid answers the closed form "
    "((a-1)/(ab-1))^(1/a); the reference evaluates another expression, nan "
    "for every concentration0 > 1",
    ("Multinomial", "table"): "total_count is in the table: it may be a tensor "
    "here, where the reference takes only an int and refuses a negative one "
    "through the Binomial it builds",
    ("RelaxedBernoulli", "param:temperature"): "the reference does not validate "
    "it; a non-positive temperature divides the logits by zero or flips them",
    ("RelaxedBernoulli", "table"): "temperature is in the table",
    ("RelaxedOneHotCategorical", "param:temperature"): "as RelaxedBernoulli",
    ("RelaxedOneHotCategorical", "table"): "temperature is in the table",
    # The reference builds these families as transforms of a base and, in
    # ``cdf``, validates the value carried back to the base against the
    # *base's* support rather than its own.  Lucid checks a cdf's value
    # against the family's support, as both libraries check log_prob's.
    ("Gumbel", "validates:cdf"): "the reference's base is a uniform on "
    "[tiny, 1 - eps], which ±inf map outside; ±inf are in Gumbel's real "
    "support and Lucid scores them 0 and 1",
    ("InverseGamma", "validates:cdf"): "the reference admits -inf and 0, whose "
    "reciprocals -0 and inf are in its base Gamma's support; neither is in "
    "the positive support",
    ("Kumaraswamy", "validates:cdf"): "the reference admits -1 and -0.5: with "
    "concentration1 = 2 their power x^a lands in the base's support",
    ("LogNormal", "validates:cdf"): "the reference admits 0, whose log -inf "
    "is in its base Normal's support; 0 is not in the positive support",
    ("Weibull", "validates:cdf"): "the reference admits -inf and 0 through its "
    "base Exponential; neither is in the positive support",
}


def _distribution_names() -> list[str]:
    return sorted(
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    )


def _lucid_tensor(v: Any) -> lucid.Tensor:
    return lucid.tensor(np.asarray(v, dtype=np.float32))


def _ref_tensor(ref: Any) -> Callable[[Any], Any]:
    return lambda v: ref.tensor(np.asarray(v, dtype=np.float32))


def _accepts(build: Callable[[], Any]) -> bool:
    try:
        build()
    except Exception:  # noqa: BLE001 — any refusal counts
        return False
    return True


def _admits(constraint: Any, value: Any) -> bool:
    try:
        return bool(constraint.check(value).all().item())
    except Exception:  # noqa: BLE001 — a check that cannot run admits nothing
        return False


def _judge(name: str, aspect: str, differences: list[Any]) -> None:
    reason = _EXCEPTIONS.get((name, aspect))
    if reason is None:
        assert not differences, f"{name} {aspect} differs from the reference: {differences}"
    else:
        assert differences, (
            f"{name} {aspect} now matches the reference; remove its entry from "
            f"_EXCEPTIONS ({reason})"
        )


def test_every_distribution_has_a_case() -> None:
    assert sorted(CASES) == _distribution_names()


def test_every_exception_names_a_case() -> None:
    for name, aspect in _EXCEPTIONS:
        assert name in CASES, name
        kind, _, param = aspect.partition(":")
        assert kind in ("param", "support", "table", "validates", "values", "mode", "grad"), aspect
        if kind == "param":
            assert param in CASES[name].params, aspect
        if kind == "validates":
            assert param in _SCORERS, aspect
        if kind in ("values", "grad"):
            assert param in _SCORERS + _METHODS, aspect


_PARAMS = [(n, p) for n in sorted(CASES) for p in CASES[n].params]


@pytest.mark.parametrize("name,param", _PARAMS, ids=[f"{n}.{p}" for n, p in _PARAMS])
def test_parameter_domain(name: str, param: str, ref: Any) -> None:
    case = CASES[name]
    rt = _ref_tensor(ref)
    differences = []
    for v in case.params[param]:
        ours = _accepts(lambda: case.build(D, _lucid_tensor, None, **{param: v}))
        theirs = _accepts(lambda: case.build(ref.distributions, rt, True, **{param: v}))
        if ours != theirs:
            differences.append((v, "Lucid accepts" if ours else "Lucid refuses"))
    _judge(name, f"param:{param}", differences)


@pytest.mark.parametrize("name", sorted(CASES))
def test_support(name: str, ref: Any) -> None:
    case = CASES[name]
    rt = _ref_tensor(ref)
    ours = case.build(D, _lucid_tensor, None).support
    theirs = case.build(ref.distributions, rt, True).support
    differences = []
    for v in case.values:
        a = ours is not None and _admits(ours, _lucid_tensor(v))
        b = _admits(theirs, rt(v))
        if a != b:
            differences.append((v, "Lucid admits" if a else "Lucid refuses"))
    _judge(name, "support", differences)


@pytest.mark.parametrize("name", sorted(CASES))
def test_constraint_table(name: str, ref: Any) -> None:
    case = CASES[name]
    ours = case.build(D, _lucid_tensor, None)
    theirs = case.build(ref.distributions, _ref_tensor(ref), True)
    differences: list[Any] = []
    if set(ours.arg_constraints) != set(theirs.arg_constraints):
        differences.append(("arg_constraints", sorted(ours.arg_constraints)))
    if ours.has_enumerate_support != theirs.has_enumerate_support:
        differences.append(("has_enumerate_support", ours.has_enumerate_support))
    if ours.support is None:
        differences.append(("support", None))
    else:
        for attr in ("is_discrete", "event_dim"):
            if getattr(ours.support, attr) != getattr(theirs.support, attr):
                differences.append((attr, getattr(ours.support, attr)))
    _judge(name, "table", differences)


# ── values given to the scorers, and the mode ─────────────────────────────────

_DEVICES = ["cpu", "metal"]
_SCORERS = ("log_prob", "cdf")


def _lucid_on(device: str) -> Callable[[Any], lucid.Tensor]:
    # Built on the host and moved: ``lucid.tensor(<0-d array>, device=...)``
    # comes out with shape (1,) rather than ().
    return lambda v: _lucid_tensor(v).to(device)


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", values)``, ``("refused", None)`` for a ``ValueError``,
    ``("absent", None)`` where the method is not implemented, or
    ``("error", name)`` for any other exception."""
    try:
        out = call()
    except ValueError:
        return "refused", None
    except NotImplementedError:
        return "absent", None
    except Exception as e:  # noqa: BLE001 — any other failure is its own verdict
        return "error", type(e).__name__
    return "ok", np.asarray(out.numpy(), dtype=np.float64)


def _agree(ours: tuple[str, Any], theirs: tuple[str, Any]) -> bool:
    if ours[0] != theirs[0]:
        return False
    if ours[0] != "ok":
        return True
    a, b = ours[1], theirs[1]
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True))


_SCORER_PARAMS = [
    pytest.param(name, method, device, id=f"{name}.{method}-{device}")
    for name in sorted(CASES)
    for method in _SCORERS
    for device in _DEVICES
]


@pytest.mark.parametrize("name,method,device", _SCORER_PARAMS)
def test_scored_value_validation(name: str, method: str, device: str, ref: Any) -> None:
    case = CASES[name]
    lt, rt = _lucid_on(device), _ref_tensor(ref)
    ours_dist = case.build(D, lt, True)
    theirs_dist = case.build(ref.distributions, rt, True)
    differences = []
    for v in case.values:
        ours = _outcome(lambda: getattr(ours_dist, method)(lt(v)))
        theirs = _outcome(lambda: getattr(theirs_dist, method)(rt(v)))
        if "absent" in (ours[0], theirs[0]):
            continue  # a scorer one side lacks is a separate gap
        if (ours[0] == "refused") != (theirs[0] == "refused"):
            differences.append((v, "Lucid refuses" if ours[0] == "refused" else ours))
    _judge(name, f"validates:{method}", differences)


@pytest.mark.parametrize("name,method,device", _SCORER_PARAMS)
def test_scored_values_without_validation(name: str, method: str, device: str, ref: Any) -> None:
    case = CASES[name]
    lt, rt = _lucid_on(device), _ref_tensor(ref)
    ours_dist = case.build(D, lt, False)
    theirs_dist = case.build(ref.distributions, rt, False)
    theirs_support = case.build(ref.distributions, rt, True).support
    differences = []
    for v in case.values:
        if not _admits(theirs_support, rt(v)):
            continue  # off the support each library answers in its own way
        ours = _outcome(lambda: getattr(ours_dist, method)(lt(v)))
        theirs = _outcome(lambda: getattr(theirs_dist, method)(rt(v)))
        if "absent" in (ours[0], theirs[0]):
            continue
        if not _agree(ours, theirs):
            differences.append((v, ours, theirs))
    _judge(name, f"values:{method}", differences)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", sorted(CASES))
def test_mode(name: str, device: str, ref: Any) -> None:
    case = CASES[name]
    ours = _outcome(lambda: case.build(D, _lucid_on(device), True).mode)
    theirs = _outcome(lambda: case.build(ref.distributions, _ref_tensor(ref), True).mode)
    _judge(name, "mode", [] if _agree(ours, theirs) else [(ours, theirs)])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", sorted(CASES))
def test_own_samples_pass_validation(name: str, device: str) -> None:
    """The guard's other side: validation that refused a distribution's own
    draws would break every likelihood computed from them."""
    lucid.manual_seed(0)
    dist = CASES[name].build(D, _lucid_on(device), True)
    try:
        draws = dist.sample((256,))
    except NotImplementedError:
        pytest.skip(f"{name} has no sampler")
    dist.log_prob(draws)


# ── parameters one axis short of their event (LCD-414) ───────────────────────

# Each event-shaped family's parameters, given with fewer axes than the
# event they describe: a scalar where a probability vector belongs, a vector
# where a matrix does.  The reference refuses every one with ``ValueError``
# in its constructor, validation or not; ``LKJCholesky``'s concentration is
# per batch element, so a scalar is a batch of one and both accept it.
_SUB_RANK: list[tuple[str, str, Any]] = [
    ("Dirichlet", "concentration", 0.5),
    *(
        (family, param, 0.5)
        for family in (
            "Categorical",
            "OneHotCategorical",
            "RelaxedOneHotCategorical",
            "Multinomial",
        )
        for param in ("probs", "logits")
    ),
    ("MultivariateNormal", "loc", 0.0),
    *(
        (family, param, [2.0, 1.0])
        for family in ("MultivariateNormal", "Wishart")
        for param in ("covariance_matrix", "precision_matrix", "scale_tril")
    ),
    ("LKJCholesky", "concentration", 0.5),
]


def _construction(build: Callable[[], Any]) -> str:
    """``"ok"``, or the name of the exception the constructor raised."""
    try:
        build()
    except Exception as e:  # noqa: BLE001 — the exception class is the verdict
        return type(e).__name__
    return "ok"


@pytest.mark.parametrize("validate", [True, False], ids=["validate", "no-validate"])
@pytest.mark.parametrize("name,param,value", _SUB_RANK, ids=[f"{n}.{p}" for n, p, _ in _SUB_RANK])
def test_sub_rank_parameter(name: str, param: str, value: Any, validate: bool, ref: Any) -> None:
    case = CASES[name]
    ours = _construction(lambda: case.build(D, _lucid_tensor, validate, **{param: value}))
    theirs = _construction(
        lambda: case.build(ref.distributions, _ref_tensor(ref), validate, **{param: value})
    )
    assert ours == theirs


# ── cdf / icdf / entropy: where each exists, its values, its gradients ────────

_METHODS = ("cdf", "icdf", "entropy")

# Where the reference answers and Lucid does not yet: each is pinned so that
# implementing it shows up as an XPASS and the entry has to go.
_PENDING: dict[tuple[str, str], str] = {
    ("TransformedDistribution", "cdf"): "LCD-397: transforms declare no "
    "sign/codomain to monotonize the base cdf with",
    ("TransformedDistribution", "icdf"): "LCD-397: as cdf",
    ("Wishart", "entropy"): "Wishart entropy (matrix.py, held by LCD-410)",
}

_PROBABILITIES: list[float] = [0.0, 1e-6, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1.0]

# (device, dtype) pairs a value is compared on; Metal has no float64.
_PLACES = [("cpu", np.float32), ("cpu", np.float64), ("metal", np.float32)]
_PLACE_IDS = ["cpu-f32", "cpu-f64", "metal-f32"]


def _lucid_at(device: str, dtype: Any) -> Callable[[Any], lucid.Tensor]:
    return lambda v: lucid.tensor(np.asarray(v, dtype=dtype)).to(device)


def _ref_at(ref: Any, dtype: Any) -> Callable[[Any], Any]:
    return lambda v: ref.tensor(np.asarray(v, dtype=dtype))


def _close(ours: tuple[str, Any], theirs: tuple[str, Any], dtype: Any) -> bool:
    if ours[0] != theirs[0]:
        return False
    if ours[0] != "ok":
        return True
    a, b = ours[1], theirs[1]
    # float64 is held to 1e-7, not to its own rounding: the reference's
    # Multinomial entropy is 2e-8 off the exact value, its Temme expansion
    # of the incomplete gamma takes pi in single precision (1e-9), and
    # Lucid's agree with the exact values in both.
    rtol, atol = (1e-7, 1e-10) if dtype == np.float64 else (1e-4, 1e-5)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _admitted(case: _Case, ref: Any) -> list[Any]:
    """The values offered to ``case``'s support that the reference admits."""
    rt = _ref_tensor(ref)
    support = case.build(ref.distributions, rt, True).support
    return [v for v in case.values if _admits(support, rt(v))]


def _call(dist: Any, method: str, t: Callable[[Any], Any], v: Any) -> Any:
    return dist.entropy() if method == "entropy" else getattr(dist, method)(t(v))


def _probe_point(case: _Case, method: str, ref: Any) -> Any:
    """One argument each method can be asked about: a support point for
    ``cdf``, the median for ``icdf``."""
    if method == "icdf":
        return 0.5
    admitted = _admitted(case, ref)
    return admitted[0] if admitted else case.values[0]


def _matrix_params() -> list[Any]:
    params = []
    for name in sorted(CASES):
        for method in _METHODS:
            reason = _PENDING.get((name, method))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            params.append(pytest.param(name, method, marks=marks, id=f"{name}.{method}"))
    return params


@pytest.mark.parametrize("name,method", _matrix_params())
def test_method_presence(name: str, method: str, ref: Any) -> None:
    case = CASES[name]
    v = _probe_point(case, method, ref)
    rt = _ref_tensor(ref)
    ours_dist = case.build(D, _lucid_tensor, False)
    theirs_dist = case.build(ref.distributions, rt, False)
    ours = _outcome(lambda: _call(ours_dist, method, _lucid_tensor, v))
    theirs = _outcome(lambda: _call(theirs_dist, method, rt, v))
    assert (ours[0] == "absent") == (
        theirs[0] == "absent"
    ), f"{name}.{method}: Lucid {ours[0]}, the reference {theirs[0]}"


def _value_grid(case: _Case, method: str, ref: Any) -> list[Any]:
    if method == "entropy":
        return [None]
    if method == "icdf":
        return _PROBABILITIES
    return _admitted(case, ref)


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_method_values(name: str, method: str, place: Any, ref: Any) -> None:
    device, dtype = place
    case = CASES[name]
    # In float64 the reference's GeneralizedPareto.log_survival_function raises
    # RuntimeError ('Double did not match Float') because it passes an untyped
    # scalar 0.0 tensor to isclose; compare against its float32 answer.
    ref_dtype = np.float32 if (name == "GeneralizedPareto" and method == "cdf") else dtype
    lt, rt = _lucid_at(device, dtype), _ref_at(ref, ref_dtype)
    ours_dist = case.build(D, lt, False)
    theirs_dist = case.build(ref.distributions, rt, False)
    differences = []
    # In float32 the reference's own answer in float64 is accepted too: its
    # float32 tails lose digits a careful evaluation keeps (the Gumbel and
    # Kumaraswamy quantiles at 1e-6 are 4e-4 and 4e-2 off), while its
    # float32 edges are float32's (the Gumbel quantile at 0 is set by
    # float32's smallest normal).
    rt64 = _ref_at(ref, np.float64)
    wide_dist = case.build(ref.distributions, rt64, False)
    for v in _value_grid(case, method, ref):
        ours = _outcome(lambda: _call(ours_dist, method, lt, v))
        theirs = _outcome(lambda: _call(theirs_dist, method, rt, v))
        if "absent" in (ours[0], theirs[0]):
            continue  # presence is test_method_presence's question
        if _close(ours, theirs, ref_dtype):
            continue
        if dtype == np.float32:
            wide = _outcome(lambda: _call(wide_dist, method, rt64, v))
            if _close(ours, wide, dtype):
                continue
        differences.append((v, ours, theirs))
    _judge(name, f"values:{method}", differences)


class _Leaf:
    """A nominal parameter to be built as a leaf that requires a gradient."""

    def __init__(self, value: Any) -> None:
        self.value = value


def _grad_outcome(
    build: Callable[[Callable[[Any], Any]], Any],
    make: Callable[[Any], Any],
    method: str,
    points: list[Any],
) -> tuple[str, Any]:
    """``("ok", gradient)`` of the summed method over ``points`` with respect
    to the one ``_Leaf`` parameter, ``("absent", None)`` if the method is not
    implemented, ``("refused", None)`` if its backward raises
    ``NotImplementedError``."""
    leaves: list[Any] = []

    def t(v: Any) -> Any:
        if isinstance(v, _Leaf):
            leaf = make(v.value)
            leaf.requires_grad_(True)
            leaves.append(leaf)
            return leaf
        return make(v)

    dist = build(t)
    try:
        outs = [_call(dist, method, make, p) for p in points]
    except NotImplementedError:
        return "absent", None
    total = outs[0].sum()
    for o in outs[1:]:
        total = total + o.sum()
    shape = np.shape(np.asarray(leaves[0].detach().numpy()))
    if not total.requires_grad:
        return "ok", np.zeros(shape)  # the parameter does not enter it
    try:
        total.backward()
    except NotImplementedError:
        return "refused", None
    grad = leaves[0].grad
    if grad is None:
        return "ok", np.zeros(shape)
    return "ok", np.asarray(grad.detach().numpy(), dtype=np.float64)


def _grad_points(case: _Case, method: str, ref: Any) -> list[Any]:
    if method == "entropy":
        return [None]
    if method == "icdf":
        return [0.1, 0.5, 0.9]
    # Inside the support, away from its edges, where a derivative exists.
    return [v for v in _admitted(case, ref) if np.all(np.isfinite(v)) and v not in (0.0, 1.0)]


_GRAD_PARAMS = [(n, m, p) for n in sorted(CASES) for m in _METHODS for p in CASES[n].nominal]
_GRAD_IDS = [f"{n}.{m}.{p}" for n, m, p in _GRAD_PARAMS]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name,method,param", _GRAD_PARAMS, ids=_GRAD_IDS)
def test_method_gradients(name: str, method: str, param: str, device: str, ref: Any) -> None:
    case = CASES[name]
    dtype = np.float64 if device == "cpu" else np.float32
    ref_dtype = np.float32 if (name == "GeneralizedPareto" and method == "cdf") else dtype
    points = _grad_points(case, method, ref)
    if not points:
        pytest.skip(f"{name} offers no interior support point")
    leaf = {param: _Leaf(case.nominal[param])}
    ours = _grad_outcome(
        lambda t: case.build(D, t, False, **leaf), _lucid_at(device, dtype), method, points
    )
    theirs = _grad_outcome(
        lambda t: case.build(ref.distributions, t, False, **leaf),
        _ref_at(ref, ref_dtype),
        method,
        points,
    )
    if "absent" in (ours[0], theirs[0]):
        return  # presence is test_method_presence's question
    differences = [] if _close(ours, theirs, ref_dtype) else [(param, ours, theirs)]
    _judge(name, f"grad:{method}", differences)


# ── enumerate_support (LCD-453) ───────────────────────────────────────────────


def _check_support_enumeration(ours_dist: Any, theirs_dist: Any, device: str, expand: bool) -> None:
    assert ours_dist.has_enumerate_support == theirs_dist.has_enumerate_support
    if not theirs_dist.has_enumerate_support:
        with pytest.raises(NotImplementedError):
            ours_dist.enumerate_support(expand=expand)
        with pytest.raises(NotImplementedError):
            theirs_dist.enumerate_support(expand=expand)
        return
    ours_out = ours_dist.enumerate_support(expand=expand)
    theirs_out = theirs_dist.enumerate_support(expand=expand)
    assert ours_out.device.type == device
    assert tuple(ours_out.shape) == tuple(theirs_out.shape)
    assert str(ours_out.dtype).split(".")[-1] == str(theirs_out.dtype).split(".")[-1]
    assert np.array_equal(ours_out.numpy(), theirs_out.numpy())
    # Scoring the enumerated support under validation must succeed and sum to 1.
    mass = ours_dist.log_prob(ours_out).exp().sum(dim=0).numpy()
    assert np.allclose(mass, 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("expand", [True, False], ids=["expand", "no-expand"])
@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_enumerate_support(name: str, place: Any, expand: bool, ref: Any) -> None:
    device, dtype = place
    case = CASES[name]
    ours_dist = case.build(D, _lucid_at(device, dtype), True)
    theirs_dist = case.build(ref.distributions, _ref_at(ref, dtype), True)
    _check_support_enumeration(ours_dist, theirs_dist, device, expand)


_ENUM_BATCH_BUILDERS: list[tuple[str, Callable[[Any, Callable[[Any], Any]], Any]]] = [
    ("Bernoulli.probs.scalar", lambda M, t: M.Bernoulli(probs=t(0.3))),
    ("Bernoulli.logits.1d", lambda M, t: M.Bernoulli(logits=t([-1.0, 0.0, 2.0]))),
    ("Bernoulli.probs.2d", lambda M, t: M.Bernoulli(probs=t([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]]))),
    ("Binomial.0.scalar", lambda M, t: M.Binomial(0, probs=t(0.3))),
    ("Binomial.probs.1d", lambda M, t: M.Binomial(4, probs=t([0.1, 0.5, 0.9]))),
    ("Binomial.logits.2d", lambda M, t: M.Binomial(3, logits=t([[-1.0, 0.0], [1.0, 2.0]]))),
    ("Binomial.inhomogeneous", lambda M, t: M.Binomial(t([2.0, 5.0]), probs=t([0.3, 0.7]))),
    ("Categorical.probs.1d", lambda M, t: M.Categorical(probs=t([0.2, 0.3, 0.5]))),
    (
        "Categorical.logits.2d",
        lambda M, t: M.Categorical(logits=t([[[0.0, 1.0, 2.0, -1.0], [1.0, 0.0, -1.0, 2.0]]])),
    ),
    ("OneHotCategorical.probs.1d", lambda M, t: M.OneHotCategorical(probs=t([0.2, 0.3, 0.5]))),
    (
        "OneHotCategorical.logits.2d",
        lambda M, t: M.OneHotCategorical(logits=t([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])),
    ),
    ("Independent.Bernoulli.0", lambda M, t: M.Independent(M.Bernoulli(probs=t([0.2, 0.8])), 0)),
    ("Independent.Bernoulli.1", lambda M, t: M.Independent(M.Bernoulli(probs=t([0.2, 0.8])), 1)),
]


@pytest.mark.parametrize("expand", [True, False], ids=["expand", "no-expand"])
@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize(
    "label,build", _ENUM_BATCH_BUILDERS, ids=[b[0] for b in _ENUM_BATCH_BUILDERS]
)
def test_enumerate_support_batches(
    label: str,
    build: Callable[[Any, Callable[[Any], Any]], Any],
    place: Any,
    expand: bool,
    ref: Any,
) -> None:
    del label
    device, dtype = place
    ours_dist = build(D, _lucid_at(device, dtype))
    theirs_dist = build(ref.distributions, _ref_at(ref, dtype))
    if not theirs_dist.has_enumerate_support:
        assert not ours_dist.has_enumerate_support
        with pytest.raises(NotImplementedError):
            ours_dist.enumerate_support(expand=expand)
        with pytest.raises(NotImplementedError):
            theirs_dist.enumerate_support(expand=expand)
        return
    try:
        theirs_out = theirs_dist.enumerate_support(expand=expand)
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            ours_dist.enumerate_support(expand=expand)
        return
    _check_support_enumeration(ours_dist, theirs_dist, device, expand)
    del theirs_out
