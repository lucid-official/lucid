"""Every distribution works on Metal when its other parameters are numbers.

A parameter given as a Python number is held as a 0-dim host tensor, since
nothing at construction says which device the computation will run on.
Where such a constant met a Metal tensor, the result was ``DeviceMismatch``
(CHA-148).  That issue was guarded case by case, five symptoms in all; this
sweep covers the defect class instead.

For every distribution ``lucid.distributions`` exports, each scalar
parameter in turn is a Metal tensor and every other scalar parameter is a
Python number.  Parameters that can only be tensors (a probability vector, a
covariance) sit on Metal with it.  Each build then runs ``log_prob`` on a
Metal value, ``sample``, ``entropy``, ``mean`` and ``variance``.  Every
result must be on Metal and equal to what the same build gives on CPU — and,
in the parity tier, to what the reference framework gives — and an operation
the family does not implement must raise ``NotImplementedError`` on both
devices.

A distribution added without an entry in ``SPECS`` fails
``test_every_distribution_has_a_spec``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

# ``build(M, kw)`` constructs the distribution from the module ``M`` —
# ``lucid.distributions`` or the reference's — and the resolved arguments.
_Build = Callable[[Any, dict[str, Any]], Any]


class _Spec:
    """How to build one distribution for the sweep.

    ``numbers`` are the scalar parameters, each passed as a Python number
    unless it is the one on the device under test.  ``tensors`` can only be
    tensors and are always on that device.  ``value`` is a point of the
    support to score, and ``build`` turns the resolved keyword arguments
    into the distribution (the class of the same name unless it wraps
    another).
    """

    def __init__(
        self,
        numbers: dict[str, Any],
        value: Any,
        tensors: dict[str, Any] | None = None,
        build: _Build | None = None,
    ) -> None:
        self.numbers = numbers
        self.tensors = tensors or {}
        self.value = value
        self.build = build


_PROBS = [0.2, 0.3, 0.5]
_COV = [[2.0, 0.5], [0.5, 1.0]]


def _transformed(M: Any, kw: dict[str, Any]) -> Any:
    """A Normal through transforms that hold numbers: ``exp(2 + 3x) ** 2.5``."""
    return M.TransformedDistribution(
        M.Normal(kw["loc"], kw["scale"]),
        [
            M.transforms.AffineTransform(2.0, 3.0),
            M.transforms.ExpTransform(),
            M.transforms.PowerTransform(2.5),
        ],
    )


SPECS: dict[str, _Spec] = {
    "Normal": _Spec({"loc": 0.5, "scale": 1.5}, 0.7),
    "LogNormal": _Spec({"loc": 0.5, "scale": 1.5}, 1.3),
    "Uniform": _Spec({"low": 0.0, "high": 2.0}, 0.7),
    "Exponential": _Spec({"rate": 1.5}, 0.7),
    "Laplace": _Spec({"loc": 0.5, "scale": 1.5}, 0.7),
    "Cauchy": _Spec({"loc": 0.5, "scale": 1.5}, 0.7),
    "Gumbel": _Spec({"loc": 0.5, "scale": 1.5}, 0.7),
    "StudentT": _Spec({"df": 3.0, "loc": 0.5, "scale": 1.5}, 0.7),
    "Pareto": _Spec({"scale": 1.0, "alpha": 2.0}, 1.7),
    "GeneralizedPareto": _Spec({"loc": 0.5, "scale": 1.5, "concentration": 0.25}, 1.7),
    "Weibull": _Spec({"scale": 1.0, "concentration": 1.5}, 0.7),
    "HalfNormal": _Spec({"scale": 1.5}, 0.7),
    "HalfCauchy": _Spec({"scale": 1.5}, 0.7),
    "FisherSnedecor": _Spec({"df1": 3.0, "df2": 5.0}, 0.7),
    "VonMises": _Spec({"loc": 0.5, "concentration": 1.5}, 0.7),
    "Gamma": _Spec({"concentration": 2.0, "rate": 1.5}, 0.7),
    "Chi2": _Spec({"df": 3.0}, 0.7),
    "Beta": _Spec({"concentration1": 2.0, "concentration0": 3.0}, 0.3),
    "InverseGamma": _Spec({"concentration": 2.0, "rate": 1.5}, 0.7),
    "Kumaraswamy": _Spec({"concentration1": 2.0, "concentration0": 3.0}, 0.3),
    "LogisticNormal": _Spec({}, _PROBS, {"loc": [0.0, 0.5], "scale": [1.0, 1.5]}),
    "Poisson": _Spec({"rate": 1.5}, 2.0),
    "Bernoulli": _Spec({"probs": 0.3}, 1.0),
    "Geometric": _Spec({"probs": 0.3}, 2.0),
    "ContinuousBernoulli": _Spec({"probs": 0.3}, 0.4),
    "Binomial": _Spec({"total_count": 5, "probs": 0.3}, 2.0),
    "NegativeBinomial": _Spec({"total_count": 3.0, "probs": 0.3}, 2.0),
    "RelaxedBernoulli": _Spec({"temperature": 0.5, "probs": 0.3}, 0.4),
    "Dirichlet": _Spec({}, _PROBS, {"concentration": [1.0, 2.0, 3.0]}),
    "Categorical": _Spec({}, 1, {"probs": _PROBS}),
    "OneHotCategorical": _Spec({}, [0.0, 1.0, 0.0], {"probs": _PROBS}),
    "OneHotCategoricalStraightThrough": _Spec({}, [0.0, 1.0, 0.0], {"probs": _PROBS}),
    "Multinomial": _Spec({"total_count": 5}, [1.0, 2.0, 2.0], {"probs": _PROBS}),
    "RelaxedOneHotCategorical": _Spec({"temperature": 0.5}, _PROBS, {"probs": _PROBS}),
    "MultivariateNormal": _Spec({}, [0.3, -0.2], {"loc": [0.0, 0.0], "covariance_matrix": _COV}),
    "LowRankMultivariateNormal": _Spec(
        {}, [0.3, -0.2], {"loc": [0.0, 1.0], "cov_factor": [[1.0], [0.5]], "cov_diag": [1.0, 2.0]}
    ),
    "Wishart": _Spec({"df": 3.0}, [[2.0, 0.3], [0.3, 1.0]], {"covariance_matrix": _COV}),
    "LKJCholesky": _Spec(
        {"concentration": 1.5},
        [[1.0, 0.0], [0.6, 0.8]],
        build=lambda M, kw: M.LKJCholesky(2, kw["concentration"]),
    ),
    "Independent": _Spec(
        {"scale": 1.5},
        [0.1, 0.9, 2.2],
        {"loc": [0.0, 1.0, 2.0]},
        build=lambda M, kw: M.Independent(M.Normal(kw["loc"], kw["scale"]), 1),
    ),
    "TransformedDistribution": _Spec({"loc": 0.5, "scale": 1.5}, 2.5, build=_transformed),
    "MixtureSameFamily": _Spec(
        {"scale": 1.5},
        0.7,
        {"probs": [0.3, 0.7], "loc": [0.0, 1.0]},
        build=lambda M, kw: M.MixtureSameFamily(
            M.Categorical(probs=kw["probs"]), M.Normal(kw["loc"], kw["scale"])
        ),
    ),
}

_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "log_prob": lambda d, v: d.log_prob(v),
    "entropy": lambda d, v: d.entropy(),
    "mean": lambda d, v: d.mean,
    "variance": lambda d, v: d.variance,
}


def _distribution_names() -> list[str]:
    return sorted(
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    )


def _anchors(name: str) -> list[str | None]:
    """Each scalar parameter in turn; ``None`` when there are none."""
    return list(SPECS[name].numbers) or [None]


_CASES = [(n, a) for n in sorted(SPECS) for a in _anchors(n)]
_IDS = [f"{n}-{a or 'tensors'}" for n, a in _CASES]


def _on(device: str) -> Callable[[Any], lucid.Tensor]:
    """A converter to a Lucid tensor on ``device``; whole numbers stay integral."""

    def tensor(v: Any) -> lucid.Tensor:
        if isinstance(v, int):
            return lucid.tensor(v, dtype=lucid.int64, device=device)
        return lucid.tensor(np.asarray(v, dtype=np.float32), device=device)

    return tensor


def _build(M: Any, name: str, anchor: str | None, tensor: Callable[[Any], Any]) -> Any:
    """Build ``name`` with ``anchor`` as a tensor and the other scalars as numbers.

    With ``anchor`` ``None`` every floating scalar is a tensor — the form the
    reference framework takes for all of them (its ``RelaxedBernoulli``
    needs a tensor temperature, its ``Multinomial`` an ``int`` count).
    """
    spec = SPECS[name]
    kw: dict[str, Any] = dict(spec.numbers)
    if anchor is not None:
        kw[anchor] = tensor(float(spec.numbers[anchor]))
    else:
        kw = {k: v if isinstance(v, int) else tensor(v) for k, v in kw.items()}
    kw |= {k: tensor(v) for k, v in spec.tensors.items()}
    if spec.build is not None:
        return spec.build(M, kw)
    return getattr(M, name)(**kw)


def _outcome(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except NotImplementedError:
        return NotImplementedError


def _close(got: Any, want: Any, what: str) -> None:
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=1e-4,
        atol=1e-5,
        err_msg=what,
    )


def test_every_distribution_has_a_spec() -> None:
    assert sorted(SPECS) == _distribution_names()


@pytest.mark.parametrize("name,anchor", _CASES, ids=_IDS)
def test_scores_on_metal_like_on_cpu(name: str, anchor: str | None) -> None:
    value = SPECS[name].value
    on_metal = _build(D, name, anchor, _on("metal"))
    on_cpu = _build(D, name, anchor, _on("cpu"))
    for op, fn in _OPERATIONS.items():
        want = _outcome(lambda: fn(on_cpu, _on("cpu")(value)))
        got = _outcome(lambda: fn(on_metal, _on("metal")(value)))
        if want is NotImplementedError:
            assert got is NotImplementedError, f"{op} exists on Metal only"
            continue
        assert got is not NotImplementedError, f"{op} exists on CPU only"
        assert got.device.type == "metal", f"{op} came back on {got.device}"
        _close(got.numpy(), want.numpy(), f"{name}.{op}")


@pytest.mark.parametrize("name,anchor", _CASES, ids=_IDS)
def test_samples_on_metal(name: str, anchor: str | None) -> None:
    d = _build(D, name, anchor, _on("metal"))
    lucid.manual_seed(0)
    sample = d.sample((3,))
    assert sample.device.type == "metal"
    assert tuple(sample.shape) == (3, *d.batch_shape, *d.event_shape)
    assert np.isfinite(np.asarray(sample.numpy(), dtype=np.float64)).all()


@pytest.mark.parity
@pytest.mark.parametrize("name,anchor", _CASES, ids=_IDS)
def test_scores_on_metal_like_the_reference(name: str, anchor: str | None, ref: Any) -> None:
    """The values the Metal build reports are the reference's.

    An operation only one side implements is not compared here.
    """
    value = SPECS[name].value

    def ref_tensor(v: Any) -> Any:
        if isinstance(v, int):
            return ref.tensor(v)
        return ref.tensor(np.asarray(v, dtype=np.float32))

    ours = _build(D, name, anchor, _on("metal"))
    theirs = _build(ref.distributions, name, None, ref_tensor)
    for op, fn in _OPERATIONS.items():
        want = _outcome(lambda: fn(theirs, ref_tensor(value)))
        got = _outcome(lambda: fn(ours, _on("metal")(value)))
        if want is NotImplementedError or got is NotImplementedError:
            continue
        _close(got.numpy(), want.detach().cpu().numpy(), f"{name}.{op}")


@pytest.mark.parametrize(
    "transform",
    [D.transforms.AffineTransform(2.0, -3.0), D.transforms.PowerTransform(2.5)],
    ids=["affine", "power"],
)
def test_number_parameter_transforms_on_metal(transform: Any) -> None:
    """A transform's number parameters meet the Metal tensor it maps."""
    x_metal = _on("metal")([1.0, 2.0])
    x_cpu = _on("cpu")([1.0, 2.0])
    for fn in (
        lambda t: transform(t),
        lambda t: transform.inv(t),
        lambda t: transform.log_abs_det_jacobian(t, transform(t)),
    ):
        got = fn(x_metal)
        assert got.device.type == "metal"
        _close(got.numpy(), fn(x_cpu).numpy(), repr(transform))
