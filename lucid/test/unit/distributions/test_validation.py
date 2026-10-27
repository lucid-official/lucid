"""Argument validation is the ``Distribution`` base's, not each family's.

``Poisson(1.5).log_prob(-1)`` scored a count of minus one as a number, and
only four families checked the value they were given at all: the check was
opted into per subclass, so most never did.  The base now wraps every
``log_prob`` / ``cdf`` / ``icdf`` a subclass defines, and each family only
declares ``arg_constraints`` and ``support``.

What the reference framework refuses is compared family by family in
``lucid/test/parity/test_distribution_constraints_parity.py``; this file pins
the mechanism and the reported cases without it.
"""

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions.distribution import _is_validated

DEVICES = ["cpu", "metal"]
_SCORERS = ("log_prob", "cdf", "icdf")


def _t(values: object, device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.float32)).to(device)


def _families() -> list[type]:
    found: list[type] = []
    stack = [D.Distribution]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("lucid.distributions") and sub not in found:
                found.append(sub)
                stack.append(sub)
    return sorted(found, key=lambda c: c.__qualname__)


# ── the mechanism ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", _families(), ids=lambda c: c.__qualname__)
def test_every_scorer_a_family_defines_is_validated(cls: type) -> None:
    for name in _SCORERS:
        method = vars(cls).get(name)
        if method is not None:
            assert _is_validated(method), f"{cls.__qualname__}.{name}"


def test_no_family_calls_the_checks_itself() -> None:
    """A per-family call is the opt-in the base replaced; it would also
    check twice."""
    root = Path(D.__file__).parent
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if path.name != "distribution.py"
        and any(
            f"._validate_{kind}(" in path.read_text()
            for kind in ("sample", "probability", "value_shape")
        )
    ]
    assert offenders == []


def test_the_wrapper_keeps_the_method_it_wraps() -> None:
    method = D.Normal.log_prob
    inner = method.__wrapped__
    assert inspect.signature(method) == inspect.signature(inner)
    assert method.__doc__ == inner.__doc__
    assert method.__qualname__ == "Normal.log_prob"
    assert method.__annotations__ == inner.__annotations__


class _Counting(D.Normal):
    """A Normal that counts its sample checks and scores through ``super``."""

    checks = 0

    def _validate_sample(self, value: lucid.Tensor) -> None:
        type(self).checks += 1
        super()._validate_sample(value)

    def log_prob(self, value: lucid.Tensor) -> lucid.Tensor:
        return super().log_prob(value) + 0.0


class _Weighted(D.Normal):
    """A user family whose scorer takes more than the value."""

    def log_prob(self, value: lucid.Tensor, weight: float = 1.0) -> lucid.Tensor:
        return super().log_prob(value) * weight


@pytest.mark.parametrize("validate", [True, False])
def test_a_scorer_with_extra_arguments_still_takes_them(validate: bool) -> None:
    dist = _Weighted(0.0, 1.0, validate_args=validate)
    base = dist.log_prob(_t(0.5)).item()
    assert dist.log_prob(_t(0.5), weight=2.0).item() == pytest.approx(2 * base)
    assert dist.log_prob(_t(0.5), 3.0).item() == pytest.approx(3 * base)
    assert list(inspect.signature(_Weighted.log_prob).parameters) == [
        "self",
        "value",
        "weight",
    ]


def test_init_subclass_forwards_class_keywords() -> None:
    seen: dict[str, object] = {}

    class _Mixin:
        def __init_subclass__(cls, **kwargs: object) -> None:
            seen.update(kwargs)
            super().__init_subclass__()

    class _Tagged(D.Normal, _Mixin, tag="x"):
        pass

    assert issubclass(_Tagged, D.Normal)
    assert seen == {"tag": "x"}


def test_a_scorer_that_calls_super_is_checked_once() -> None:
    _Counting.checks = 0
    _Counting(0.0, 1.0).log_prob(_t(0.5))
    assert _Counting.checks == 1


def test_validation_off_runs_no_check() -> None:
    _Counting.checks = 0
    out = _Counting(0.0, 1.0, validate_args=False).log_prob(_t(float("nan")))
    assert _Counting.checks == 0
    assert np.isnan(out.item())


def test_a_value_that_is_not_a_tensor_is_refused() -> None:
    not_a_tensor: Any = 0.5
    with pytest.raises(TypeError, match="Tensor"):
        D.Normal(0.0, 1.0).log_prob(not_a_tensor)


@pytest.mark.parametrize(
    "build,value",
    [
        (lambda: D.Normal(_t([0.0, 1.0, 2.0]), _t([1.0, 1.0, 1.0])), [0.0, 1.0]),
        (lambda: D.MultivariateNormal(_t([0.0, 0.0]), _t(np.eye(2))), [0.0, 0.0, 0.0]),
    ],
    ids=["batch", "event"],
)
def test_a_value_of_the_wrong_shape_is_refused(
    build: Callable[[], Any], value: list[float]
) -> None:
    with pytest.raises(ValueError, match="shape"):
        build().log_prob(_t(value))


def test_the_default_can_be_turned_off_and_on() -> None:
    built_before = D.Exponential(1.0)
    pinned_on = D.Exponential(1.0, validate_args=True)
    try:
        D.Distribution.set_default_validate_args(False)
        assert D.Normal(0.0, -1.0).scale.item() == -1.0
        # An instance that took the default follows it at once.
        assert built_before.log_prob(_t(-1.0)).item() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="support"):
            pinned_on.log_prob(_t(-1.0))
    finally:
        D.Distribution.set_default_validate_args(True)
    with pytest.raises(ValueError, match="constraint"):
        D.Normal(0.0, -1.0)
    not_a_bool: Any = 1
    with pytest.raises(TypeError):
        D.Distribution.set_default_validate_args(not_a_bool)


# ── icdf takes a probability ──────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_icdf_refuses_what_is_not_a_probability(device: str) -> None:
    """Stricter than the reference, which answers nan; on purpose."""
    normal = D.Normal(_t(0.0, device), _t(1.0, device))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normal.icdf(_t(1.5, device))
    off = D.Normal(_t(0.0, device), _t(1.0, device), validate_args=False)
    assert np.isnan(off.icdf(_t(1.5, device)).item())


@pytest.mark.parametrize("device", DEVICES)
def test_icdf_is_checked_against_the_unit_interval_not_the_support(device: str) -> None:
    uniform = D.Uniform(_t(2.0, device), _t(3.0, device))
    assert uniform.icdf(_t(0.5, device)).item() == pytest.approx(2.5)


# ── the reported cases ────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "build,value",
    [
        (lambda d: D.Poisson(_t(1.5, d)), -1.0),
        (lambda d: D.Weibull(_t(1.0, d), _t(1.0, d)), 0.0),
        (lambda d: D.Bernoulli(_t(0.3, d)), 0.5),
        (lambda d: D.Categorical(_t([0.2, 0.3, 0.5], d)), -1.0),
        (lambda d: D.HalfNormal(_t(1.0, d)), -1.0),
        (lambda d: D.Exponential(_t(1.0, d)), -1.0),
    ],
    ids=["poisson", "weibull", "bernoulli", "categorical", "half_normal", "exp"],
)
def test_a_value_outside_the_support_is_refused(
    build: Callable[[str], Any], value: float, device: str
) -> None:
    with pytest.raises(ValueError, match="support"):
        build(device).log_prob(_t(value, device))


@pytest.mark.parametrize("device", DEVICES)
def test_half_normal_scores_below_zero_as_impossible(device: str) -> None:
    for family in (D.HalfNormal, D.HalfCauchy):
        dist = family(_t(1.5, device), validate_args=False)
        assert dist.log_prob(_t(-1.0, device)).item() == float("-inf")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("probs", [[-0.5, 1.5], [float("nan"), 0.5, 0.5]], ids=["negative", "nan"])
def test_multinomial_validates_its_probabilities(probs: list[float], device: str) -> None:
    with pytest.raises(ValueError, match="probs"):
        D.Multinomial(5, probs=_t(probs, device))


@pytest.mark.parametrize("device", DEVICES)
def test_multinomial_scores_like_the_reference(device: str) -> None:
    """The coefficient counts the arrangements of the counts given."""
    dist = D.Multinomial(5, probs=_t([0.2, 0.3, 0.5], device))
    # 3! / (1! 1! 1!) · 0.2 · 0.3 · 0.5
    want = np.log(6 * 0.2 * 0.3 * 0.5)
    assert dist.log_prob(_t([1.0, 1.0, 1.0], device)).item() == pytest.approx(want, rel=1e-5)
    with pytest.raises(ValueError, match="support"):
        dist.log_prob(_t([3.0, 3.0, 0.0], device))


@pytest.mark.parametrize("device", DEVICES)
def test_a_covariance_that_is_not_positive_definite(device: str) -> None:
    loc, cov = _t([0.0, 0.0], device), _t([[1.0, 2.0], [2.0, 1.0]], device)
    with pytest.raises(ValueError, match="covariance_matrix"):
        D.MultivariateNormal(loc, covariance_matrix=cov)
    with pytest.raises(lucid.linalg.LinAlgError):
        D.MultivariateNormal(loc, covariance_matrix=cov, validate_args=False)
    with pytest.raises(ValueError, match="precision_matrix"):
        D.Wishart(_t(3.0, device), precision_matrix=cov)


@pytest.mark.parametrize("device", DEVICES)
def test_a_scale_tril_that_is_not_a_cholesky_factor(device: str) -> None:
    tril = _t([[-1.0, 0.0], [0.5, 2.0]], device)
    with pytest.raises(ValueError, match="scale_tril"):
        D.MultivariateNormal(_t([0.0, 0.0], device), scale_tril=tril)
    with pytest.raises(ValueError, match="scale_tril"):
        D.Wishart(_t(3.0, device), scale_tril=tril)


@pytest.mark.parametrize("device", DEVICES)
def test_the_given_matrix_and_its_duals_agree(device: str) -> None:
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    mvn = D.MultivariateNormal(_t([0.0, 0.0], device), covariance_matrix=_t(cov, device))
    np.testing.assert_allclose(mvn.precision_matrix.numpy(), np.linalg.inv(cov), rtol=1e-5)
    np.testing.assert_allclose(mvn.scale_tril.numpy(), np.linalg.cholesky(cov), rtol=1e-5)
    wishart = D.Wishart(_t(3.0, device), precision_matrix=_t(np.linalg.inv(cov), device))
    np.testing.assert_allclose(wishart.covariance_matrix.numpy(), cov, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_geometric_and_negative_binomial_have_a_mode(device: str) -> None:
    assert D.Geometric(_t(0.3, device)).mode.item() == 0.0
    # floor((r - 1) · p / (1 - p)) = floor(2 · 0.7 / 0.3) = 4
    assert D.NegativeBinomial(_t(3.0, device), _t(0.7, device)).mode.item() == 4.0
    assert D.NegativeBinomial(_t(3.0, device), _t(0.3, device)).mode.item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_positive_definite_checks_every_matrix_in_mixed_batch(device: str) -> None:
    batch = _t(
        [
            [[1.0, 0.0], [0.0, -1.0]],  # symmetric, indefinite -> False
            [[1.0, 2.0], [3.0, 4.0]],  # non-symmetric -> False
            [[1.0, 2.0], [0.0, 1.0]],  # non-symmetric with SPD lower triangle -> False
            [[2.0, 0.5], [0.5, 1.0]],  # symmetric positive-definite -> True
        ],
        device,
    )
    got = D.constraints.positive_definite.check(batch)
    per_matrix = [
        bool(D.constraints.positive_definite.check(batch[i]).item()) for i in range(batch.shape[0])
    ]
    assert got.device == device
    assert got.dtype == lucid.bool
    assert got.tolist() == per_matrix == [False, False, False, True]
