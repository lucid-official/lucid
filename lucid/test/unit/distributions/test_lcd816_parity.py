"""Constraint-registry transforms and the Relaxed families against the reference (LCD-816).

Two gaps LCD-684 left:

* ``LowerCholeskyTransform`` put ``softplus`` on the diagonal where the
  reference puts ``exp``, so ``transform_to(lower_cholesky)``,
  ``transform_to(positive_definite)`` and ``PositiveDefiniteTransform`` mapped
  one unconstrained input to a different matrix.  Every transform the
  ``biject_to`` / ``transform_to`` registries hand out is held here to the
  reference's: which constraints each answers for, and its forward value,
  inverse and log-abs-det.
* ``RelaxedBernoulli`` and ``RelaxedOneHotCategorical`` are
  ``TransformedDistribution`` subclasses in the reference — a logit-space (or
  log-space) base distribution pushed through ``SigmoidTransform`` (or
  ``ExpTransform``) — and were plain ``Distribution`` subclasses here.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.distributions.transforms as T
from lucid.distributions import constraints as C

pytestmark = pytest.mark.parity

_PLACES = [("cpu", np.float32), ("cpu", np.float64), ("metal", np.float32)]
_PLACE_IDS = ["cpu-f32", "cpu-f64", "metal-f32"]


def _lt(value: Any, device: str = "cpu", dtype: Any = np.float32) -> lucid.Tensor:
    return lucid.tensor(np.asarray(value, dtype=dtype)).to(device)


def _rt(ref: Any, value: Any, dtype: Any = np.float32) -> Any:
    return ref.tensor(np.asarray(value, dtype=dtype))


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)


def _tol(dtype: Any) -> dict[str, float]:
    return {"rtol": 1e-12, "atol": 1e-12} if dtype == np.float64 else {"rtol": 3e-4, "atol": 1e-5}


def _outcome(fn: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", value)``, or the name of the exception class the call raised."""
    try:
        out = fn()
    except (NotImplementedError, ValueError, TypeError, AssertionError) as exc:
        return type(exc).__name__, None
    return "ok", _arr(out)


# ── The registry's transforms ────────────────────────────────────────────────

_ELEMENTWISE = [-1.5, -0.3, 0.4, 2.0]
_VEC = [[0.3, -0.8, 1.1, 0.2], [-1.2, 0.5, 0.0, 0.9]]
_MAT = [
    [[0.5, 0.0, 0.0], [0.3, -1.0, 0.0], [0.2, 0.7, 0.4]],
    [[-0.6, 1.0, 2.0], [0.1, 0.9, -0.5], [-0.3, 0.2, -1.1]],
]
_CORR = [[0.2, -0.4, 0.6], [-0.7, 0.1, 0.3]]

# Name -> (constraint builder taking a constraints module and a tensor
# function, unconstrained input).  One builder serves both frameworks.
_REGISTRY_CASES: dict[str, tuple[Callable[[Any, Callable[[Any], Any]], Any], Any]] = {
    "real": (lambda C_, t: C_.real, _ELEMENTWISE),
    "positive": (lambda C_, t: C_.positive, _ELEMENTWISE),
    "nonnegative": (lambda C_, t: C_.nonnegative, _ELEMENTWISE),
    "greater_than": (lambda C_, t: C_.greater_than(2.0), _ELEMENTWISE),
    "greater_than(tensor)": (lambda C_, t: C_.greater_than(t([1.0, 2.0, -1.0, 0.5])), _ELEMENTWISE),
    "greater_than_eq": (lambda C_, t: C_.greater_than_eq(-1.0), _ELEMENTWISE),
    "less_than": (lambda C_, t: C_.less_than(5.0), _ELEMENTWISE),
    "unit_interval": (lambda C_, t: C_.unit_interval, _ELEMENTWISE),
    "interval": (lambda C_, t: C_.interval(-1.0, 3.0), _ELEMENTWISE),
    "half_open_interval": (lambda C_, t: C_.half_open_interval(-1.0, 3.0), _ELEMENTWISE),
    "simplex": (lambda C_, t: C_.simplex, _VEC),
    "lower_cholesky": (lambda C_, t: C_.lower_cholesky, _MAT),
    "positive_definite": (lambda C_, t: C_.positive_definite, _MAT),
    "positive_semidefinite": (lambda C_, t: C_.positive_semidefinite, _MAT),
    "corr_cholesky": (lambda C_, t: C_.corr_cholesky, _CORR),
    "independent(positive,1)": (lambda C_, t: C_.independent(C_.positive, 1), _VEC),
    "independent(interval,1)": (lambda C_, t: C_.independent(C_.interval(0.0, 2.0), 1), _VEC),
    "independent(real,2)": (lambda C_, t: C_.independent(C_.real, 2), _MAT),
    "stack": (lambda C_, t: C_.stack([C_.positive, C_.unit_interval], dim=0), _VEC),
    "cat": (
        lambda C_, t: C_.cat([C_.positive, C_.unit_interval], dim=-1),
        [[0.2, -0.3], [0.5, 0.1], [-1.0, 1.0]],
    ),
}
_REGISTRIES = ["biject_to", "transform_to"]

# Where the reference defines no ``log_abs_det_jacobian`` (it does not call
# these maps bijections) Lucid keeps one; it is held to the numerical
# Jacobian instead, below.  The simplex is not among them: ``SoftmaxTransform``
# has no determinant and refuses in both (LCD-818).
_LUCID_ONLY_LADJ = {"lower_cholesky", "positive_definite", "positive_semidefinite"}


def _pair(name: str, registry: str, ref: Any, device: str, dtype: Any) -> tuple[Any, Any]:
    """``(lucid transform, reference transform)`` for ``name`` from ``registry``."""
    build, _ = _REGISTRY_CASES[name]
    ours = getattr(D, registry)(build(C, lambda v: _lt(v, device, dtype)))
    theirs = getattr(ref.distributions, registry)(
        build(ref.distributions.constraints, lambda v: _rt(ref, v, dtype))
    )
    return ours, theirs


def _refusals(name: str, registry: str, ref: Any) -> tuple[str, str]:
    build, _ = _REGISTRY_CASES[name]
    ours = _outcome(
        lambda: getattr(D, registry)(build(C, lambda v: _lt(v)))(_lt(_REGISTRY_CASES[name][1]))
    )
    theirs = _outcome(
        lambda: getattr(ref.distributions, registry)(
            build(ref.distributions.constraints, lambda v: _rt(ref, v))
        )(_rt(ref, _REGISTRY_CASES[name][1]))
    )
    return ours[0], theirs[0]


@pytest.mark.parametrize("registry", _REGISTRIES)
@pytest.mark.parametrize("name", list(_REGISTRY_CASES))
def test_registry_answers_for_the_same_constraints_as_the_reference(
    name: str, registry: str, ref: Any
) -> None:
    ours, theirs = _refusals(name, registry, ref)
    assert ours == theirs, (name, registry, ours, theirs)


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("registry", _REGISTRIES)
@pytest.mark.parametrize("name", list(_REGISTRY_CASES))
def test_registry_forward_matches_reference(name: str, registry: str, place: Any, ref: Any) -> None:
    device, dtype = place
    if _refusals(name, registry, ref)[1] != "ok":
        pytest.skip("the reference has no such transform")
    ours, theirs = _pair(name, registry, ref, device, dtype)
    x = _REGISTRY_CASES[name][1]
    got = _arr(ours(_lt(x, device, dtype)))
    want = _arr(theirs(_rt(ref, x, dtype)))
    np.testing.assert_allclose(got, want, err_msg=f"{name}/{registry}", **_tol(dtype))


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("registry", _REGISTRIES)
@pytest.mark.parametrize("name", list(_REGISTRY_CASES))
def test_registry_inverse_matches_reference(name: str, registry: str, place: Any, ref: Any) -> None:
    device, dtype = place
    if _refusals(name, registry, ref)[1] != "ok":
        pytest.skip("the reference has no such transform")
    ours, theirs = _pair(name, registry, ref, device, dtype)
    x = _REGISTRY_CASES[name][1]
    y = _arr(theirs(_rt(ref, x, dtype)))
    # The point is a value of the constraint, taken from the reference.
    got = _outcome(lambda: ours.inv(_lt(y, device, dtype)))
    want = _outcome(lambda: theirs.inv(_rt(ref, y, dtype)))
    assert got[0] == want[0], (name, registry, got[0], want[0])
    if want[0] == "ok":
        np.testing.assert_allclose(got[1], want[1], err_msg=f"{name}/{registry}", **_tol(dtype))


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("registry", _REGISTRIES)
@pytest.mark.parametrize("name", list(_REGISTRY_CASES))
def test_registry_log_abs_det_matches_reference(
    name: str, registry: str, place: Any, ref: Any
) -> None:
    device, dtype = place
    if _refusals(name, registry, ref)[1] != "ok" or name in _LUCID_ONLY_LADJ:
        pytest.skip("the reference has no such transform, or no log-det for it")
    ours, theirs = _pair(name, registry, ref, device, dtype)
    x = _REGISTRY_CASES[name][1]
    xl, xr = _lt(x, device, dtype), _rt(ref, x, dtype)
    got = _outcome(lambda: ours.log_abs_det_jacobian(xl, ours(xl)))
    want = _outcome(lambda: theirs.log_abs_det_jacobian(xr, theirs(xr)))
    assert got[0] == want[0], (name, registry, got[0], want[0])
    if want[0] == "ok":
        assert got[1].shape == want[1].shape
        np.testing.assert_allclose(got[1], want[1], err_msg=f"{name}/{registry}", **_tol(dtype))


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("registry", _REGISTRIES)
@pytest.mark.parametrize("name", list(_REGISTRY_CASES))
def test_registry_transform_lands_in_its_constraint_and_round_trips(
    name: str, registry: str, place: Any, ref: Any
) -> None:
    device, dtype = place
    build, x = _REGISTRY_CASES[name]
    if _refusals(name, registry, ref)[1] != "ok":
        pytest.skip("the reference has no such transform")
    transform = getattr(D, registry)(build(C, lambda v: _lt(v, device, dtype)))
    constraint = build(C, lambda v: _lt(v, device, dtype))
    y = transform(_lt(x, device, dtype))
    assert bool(constraint.check(y).all().item()), name
    # ``transform_to`` is onto the constraint, not one-to-one, so the round
    # trip that must hold is the one through the constraint's value.
    again = transform(transform.inv(y))
    np.testing.assert_allclose(_arr(again), _arr(y), err_msg=f"{name}/{registry}", **_tol(dtype))
    if registry == "biject_to":
        np.testing.assert_allclose(
            _arr(transform.inv(y)), np.asarray(x, dtype=np.float64), **_tol(dtype)
        )


def test_lower_cholesky_is_the_references_not_a_softplus_one(ref: Any) -> None:
    """The reproduction from the issue: ``[[0.5, 0], [0.3, -1.0]]``."""
    x = [[0.5, 0.0], [0.3, -1.0]]
    got = _arr(T.LowerCholeskyTransform()(_lt(x)))
    want = _arr(ref.distributions.transforms.LowerCholeskyTransform()(_rt(ref, x)))
    np.testing.assert_allclose(got, want, rtol=1e-5)
    np.testing.assert_allclose(got, [[np.exp(0.5), 0.0], [0.3, np.exp(-1.0)]], rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_one_input_maps_to_one_matrix_through_every_matrix_route(device: str, ref: Any) -> None:
    from lucid.distributions.constraint_registry import PositiveDefiniteTransform

    x = _lt(_MAT[0], device)
    tril = D.transform_to(C.lower_cholesky)(x)
    np.testing.assert_allclose(_arr(tril), _arr(T.LowerCholeskyTransform()(x)), rtol=1e-6)
    spd = D.transform_to(C.positive_definite)(x)
    np.testing.assert_allclose(_arr(spd), _arr(PositiveDefiniteTransform()(x)), rtol=1e-6)
    np.testing.assert_allclose(_arr(spd), _arr(tril @ tril.mT), rtol=1e-5)
    want = ref.distributions.transform_to(ref.distributions.constraints.positive_definite)(
        _rt(ref, _MAT[0])
    )
    np.testing.assert_allclose(_arr(spd), _arr(want), rtol=1e-5)


def _numeric_log_det(fn: Callable[[np.ndarray], np.ndarray], free: np.ndarray, dim: int) -> float:
    """``log|det|`` of ``fn`` on the lower-triangle coordinates, by central differences."""
    rows, cols = np.tril_indices(dim)

    def flat(v: np.ndarray) -> np.ndarray:
        m = np.zeros((dim, dim))
        m[rows, cols] = v
        return fn(m)[rows, cols]

    h = 1e-6
    jac = np.stack(
        [
            (flat(free + h * np.eye(len(free))[k]) - flat(free - h * np.eye(len(free))[k]))
            / (2 * h)
            for k in range(len(free))
        ],
        axis=1,
    )
    return float(np.linalg.slogdet(jac)[1])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("name", ["lower_cholesky", "positive_definite"])
def test_matrix_transform_log_det_is_the_jacobian(name: str, dim: int) -> None:
    """Lucid keeps a log-det the reference lacks; it is the true Jacobian of the lower triangle."""
    transform = D.transform_to(getattr(C, name))
    rows, cols = np.tril_indices(dim)
    rng = np.random.RandomState(7)
    free = rng.randn(len(rows)) * 0.6

    def forward(m: np.ndarray) -> np.ndarray:
        return _arr(transform(_lt(m, dtype=np.float64)))

    m = np.zeros((dim, dim))
    m[rows, cols] = free
    y = transform(_lt(m, dtype=np.float64))
    got = float(_arr(transform.log_abs_det_jacobian(_lt(m, dtype=np.float64), y)))
    np.testing.assert_allclose(got, _numeric_log_det(forward, free, dim), rtol=1e-5)


# ── Relaxed families as TransformedDistribution ──────────────────────────────


def _bernoulli_args(ref: Any, device: str, dtype: Any, kw: str, batched: bool) -> tuple[Any, Any]:
    p = [0.3, 0.8] if batched else 0.35
    if kw == "logits":
        p = np.log(np.asarray(p) / (1.0 - np.asarray(p)))
    return (
        D.RelaxedBernoulli(_lt(0.6, device, dtype), **{kw: _lt(p, device, dtype)}),
        ref.distributions.RelaxedBernoulli(_rt(ref, 0.6, dtype), **{kw: _rt(ref, p, dtype)}),
    )


def _categorical_args(ref: Any, device: str, dtype: Any, kw: str, batched: bool) -> tuple[Any, Any]:
    p = np.asarray([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]] if batched else [0.2, 0.3, 0.5])
    if kw == "logits":
        p = np.log(p)
    return (
        D.RelaxedOneHotCategorical(_lt(0.6, device, dtype), **{kw: _lt(p, device, dtype)}),
        ref.distributions.RelaxedOneHotCategorical(
            _rt(ref, 0.6, dtype), **{kw: _rt(ref, p, dtype)}
        ),
    )


_FAMILIES = {"RelaxedBernoulli": _bernoulli_args, "RelaxedOneHotCategorical": _categorical_args}
_FAMILY_BASES = {
    "RelaxedBernoulli": (D.relaxed.LogitRelaxedBernoulli, "SigmoidTransform"),
    "RelaxedOneHotCategorical": (D.relaxed.ExpRelaxedCategorical, "ExpTransform"),
}


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_family_is_a_transformed_distribution_like_the_reference(
    family: str, ref: Any
) -> None:
    ours, theirs = _FAMILIES[family](ref, "cpu", np.float32, "probs", False)
    assert isinstance(ours, D.TransformedDistribution)
    assert isinstance(theirs, ref.distributions.TransformedDistribution)
    base_cls, transform_name = _FAMILY_BASES[family]
    assert isinstance(ours.base_dist, base_cls)
    assert type(ours.base_dist).__name__ == type(theirs.base_dist).__name__
    assert [type(t).__name__ for t in ours.transforms] == [
        type(t).__name__ for t in theirs.transforms
    ]
    assert [type(t).__name__ for t in ours.transforms] == [transform_name]


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_family_surface_matches_reference(family: str, ref: Any) -> None:
    ours_cls = getattr(D, family)
    theirs_cls = getattr(ref.distributions, family)
    assert [(n, p.default) for n, p in inspect.signature(ours_cls.__init__).parameters.items()] == [
        (n, p.default) for n, p in inspect.signature(theirs_cls.__init__).parameters.items()
    ]
    ours, theirs = _FAMILIES[family](ref, "cpu", np.float32, "probs", True)
    assert ours.has_rsample is theirs.has_rsample is True
    assert type(ours.support).__name__ == type(theirs.support).__name__
    assert ours.batch_shape == tuple(theirs.batch_shape)
    assert ours.event_shape == tuple(theirs.event_shape)
    assert set(ours.arg_constraints) >= set(theirs.arg_constraints)
    for attr in ("temperature", "probs", "logits"):
        np.testing.assert_allclose(
            _arr(getattr(ours, attr)), _arr(getattr(theirs, attr)), rtol=1e-5, err_msg=attr
        )


@pytest.mark.parametrize("batched", [False, True], ids=["scalar", "batched"])
@pytest.mark.parametrize("kw", ["probs", "logits"])
@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_log_prob_matches_reference(
    family: str, place: Any, kw: str, batched: bool, ref: Any
) -> None:
    device, dtype = place
    ours, theirs = _FAMILIES[family](ref, device, dtype, kw, batched)
    # The reference draws the points, so each is a value of the support.
    points = theirs.sample((4,)).detach().numpy().astype(dtype)
    if family == "RelaxedBernoulli":
        points = np.concatenate([points, np.zeros_like(points[:1]), np.ones_like(points[:1])])
    got = _arr(ours.log_prob(_lt(points, device, dtype)))
    want = _arr(theirs.log_prob(_rt(ref, points, dtype)))
    np.testing.assert_allclose(got, want, rtol=2e-3 if dtype == np.float32 else 1e-8, atol=2e-3)


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_base_dist_log_prob_matches_reference(family: str, place: Any, ref: Any) -> None:
    device, dtype = place
    ours, theirs = _FAMILIES[family](ref, device, dtype, "probs", True)
    points = theirs.base_dist.sample((4,)).detach().numpy().astype(dtype)
    got = _arr(ours.base_dist.log_prob(_lt(points, device, dtype)))
    want = _arr(theirs.base_dist.log_prob(_rt(ref, points, dtype)))
    np.testing.assert_allclose(got, want, rtol=2e-3 if dtype == np.float32 else 1e-8, atol=2e-3)


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("batched", [False, True], ids=["scalar", "batched"])
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_rsample_matches_reference_in_shape_range_and_moments(
    family: str, batched: bool, device: str, ref: Any
) -> None:
    ours, theirs = _FAMILIES[family](ref, device, np.float32, "probs", batched)
    n = 20000
    mine = _arr(ours.rsample((n,)))
    other = theirs.rsample((n,)).detach().numpy().astype(np.float64)
    assert mine.shape == other.shape == (n, *ours.batch_shape, *ours.event_shape)
    # Closed: a float32 draw rounds to exactly 0 or 1 at the tails.
    assert np.all(mine >= 0.0) and np.all(mine <= 1.0)
    np.testing.assert_allclose(mine.mean(axis=0), other.mean(axis=0), atol=0.02)
    np.testing.assert_allclose(mine.std(axis=0), other.std(axis=0), atol=0.02)
    if family == "RelaxedOneHotCategorical":
        np.testing.assert_allclose(mine.sum(axis=-1), 1.0, atol=1e-5)
    assert tuple(ours.sample((3,)).shape) == (3, *ours.batch_shape, *ours.event_shape)


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_rsample_carries_gradients_to_the_parameters(family: str) -> None:
    logits = lucid.tensor([0.3, -0.2, 0.5]) if family != "RelaxedBernoulli" else lucid.tensor(0.4)
    temperature = lucid.tensor(0.7)
    logits.requires_grad_(True)
    temperature.requires_grad_(True)
    dist = getattr(D, family)(temperature, logits=logits)
    dist.rsample((8,))[..., 0].sum().backward()
    assert logits.grad is not None and temperature.grad is not None
    assert bool(lucid.isfinite(logits.grad).all().item())
    assert float(temperature.grad.abs().sum().item()) > 0.0


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_expand_keeps_the_structure_and_matches_reference(family: str, ref: Any) -> None:
    ours, theirs = _FAMILIES[family](ref, "cpu", np.float32, "probs", False)
    mine, other = ours.expand((4, 2)), theirs.expand((4, 2))
    assert type(mine) is type(ours)
    assert isinstance(mine, D.TransformedDistribution)
    assert type(mine.base_dist) is type(ours.base_dist)
    assert mine.batch_shape == tuple(other.batch_shape) == (4, 2)
    assert mine.event_shape == tuple(other.event_shape)
    assert tuple(mine.rsample().shape) == tuple(other.rsample().shape)
    point = other.sample().detach().numpy()
    np.testing.assert_allclose(
        _arr(mine.log_prob(_lt(point))), _arr(other.log_prob(_rt(ref, point))), rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_log_prob_refuses_a_point_off_its_support_like_the_reference(
    family: str, ref: Any
) -> None:
    ours, theirs = _FAMILIES[family](ref, "cpu", np.float32, "probs", False)
    bad = [1.5] if family == "RelaxedBernoulli" else [0.5, 0.5, 0.5]
    bad = bad[0] if family == "RelaxedBernoulli" else bad
    got = _outcome(lambda: ours.log_prob(_lt(bad)))
    want = _outcome(lambda: theirs.log_prob(_rt(ref, bad)))
    assert got[0] == want[0] == "ValueError"


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_relaxed_repr_names_the_family(family: str, ref: Any) -> None:
    ours, _ = _FAMILIES[family](ref, "cpu", np.float32, "logits", False)
    text = repr(ours)
    assert text.startswith(family)
    assert "logits" in text and "probs" not in text
