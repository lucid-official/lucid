"""Distribution surface gaps against the reference (LCD-684).

Five things the reference answers and Lucid did not:

* ``Transform.sign`` — ``AffineTransform(0., -1.).sign`` was ``1`` and
  ``PowerTransform`` ignored its exponent; a transform with no meaningful
  sign (``AbsTransform``, the vector-valued ones) claimed ``+1`` instead of
  refusing.  The sign is the contract ``TransformedDistribution`` orients its
  CDF with, so it belongs to ``Transform`` and every transform reports its own.
* ``TransformedDistribution.cdf`` / ``icdf``, which flip by that sign.
* ``Wishart.entropy`` (and so ``perplexity``).
* ``ContinuousBernoulli(lims=...)``.
* The constraints missing from ``constraints.__all__``.

Each is held to the reference by behaviour: the same inputs to both, and the
same answer — or the same refusal — from each.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.distributions.transforms as T
from lucid.distributions import constraints

pytestmark = pytest.mark.parity


def _lt(value: Any, device: str = "cpu", dtype: Any = np.float32) -> lucid.Tensor:
    return lucid.tensor(np.asarray(value, dtype=dtype)).to(device)


def _rt(ref: Any, value: Any, dtype: Any = np.float32) -> Any:
    return ref.tensor(np.asarray(value, dtype=dtype))


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)


def _outcome(fn: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", value)``, or the name of the exception class the call raised."""
    try:
        out = fn()
    except (NotImplementedError, ValueError, TypeError, AssertionError) as exc:
        return type(exc).__name__, None
    return "ok", np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out)


# ── Transform.sign ───────────────────────────────────────────────────────────


def _transform_cases(ref: Any) -> dict[str, tuple[Callable[[], Any], Callable[[], Any]]]:
    """Name -> (Lucid transform factory, reference transform factory)."""
    RT = ref.distributions.transforms
    exps = [-2.0, 3.0]
    return {
        "Exp": (lambda: T.ExpTransform(), lambda: RT.ExpTransform()),
        "Sigmoid": (lambda: T.SigmoidTransform(), lambda: RT.SigmoidTransform()),
        "Tanh": (lambda: T.TanhTransform(), lambda: RT.TanhTransform()),
        "Softplus": (
            lambda: D.constraint_registry.SoftplusTransform(),
            lambda: RT.SoftplusTransform(),
        ),
        "Affine(+)": (lambda: T.AffineTransform(0.0, 2.0), lambda: RT.AffineTransform(0.0, 2.0)),
        "Affine(-)": (lambda: T.AffineTransform(0.0, -1.0), lambda: RT.AffineTransform(0.0, -1.0)),
        "Affine(0)": (lambda: T.AffineTransform(0.0, 0.0), lambda: RT.AffineTransform(0.0, 0.0)),
        "Affine(int)": (lambda: T.AffineTransform(0, -3), lambda: RT.AffineTransform(0, -3)),
        "Affine(tensor)": (
            lambda: T.AffineTransform(lucid.tensor([0.0, 0.0]), lucid.tensor([-1.0, 2.0])),
            lambda: RT.AffineTransform(ref.tensor([0.0, 0.0]), ref.tensor([-1.0, 2.0])),
        ),
        "Power(-)": (
            lambda: T.PowerTransform(lucid.tensor(exps[0])),
            lambda: RT.PowerTransform(ref.tensor(exps[0])),
        ),
        "Power(+)": (
            lambda: T.PowerTransform(lucid.tensor(exps[1])),
            lambda: RT.PowerTransform(ref.tensor(exps[1])),
        ),
        "Abs": (lambda: T.AbsTransform(), lambda: RT.AbsTransform()),
        "Softmax": (lambda: T.SoftmaxTransform(), lambda: RT.SoftmaxTransform()),
        "StickBreaking": (lambda: T.StickBreakingTransform(), lambda: RT.StickBreakingTransform()),
        "LowerCholesky": (lambda: T.LowerCholeskyTransform(), lambda: RT.LowerCholeskyTransform()),
        "PositiveDefinite": (
            lambda: D.constraint_registry.PositiveDefiniteTransform(),
            lambda: RT.PositiveDefiniteTransform(),
        ),
        "CorrCholesky": (lambda: T.CorrCholeskyTransform(), lambda: RT.CorrCholeskyTransform()),
        "Reshape": (
            lambda: T.ReshapeTransform((2, 3), (6,)),
            lambda: RT.ReshapeTransform((2, 3), (6,)),
        ),
        "CDF": (
            lambda: T.CumulativeDistributionTransform(D.Normal(0.0, 1.0)),
            lambda: RT.CumulativeDistributionTransform(ref.distributions.Normal(0.0, 1.0)),
        ),
        "Independent(Exp)": (
            lambda: T.IndependentTransform(T.ExpTransform(), 1),
            lambda: RT.IndependentTransform(RT.ExpTransform(), 1),
        ),
        "Independent(Affine-)": (
            lambda: T.IndependentTransform(T.AffineTransform(0.0, -1.0), 1),
            lambda: RT.IndependentTransform(RT.AffineTransform(0.0, -1.0), 1),
        ),
        "Independent(Softmax)": (
            lambda: T.IndependentTransform(T.SoftmaxTransform(), 1),
            lambda: RT.IndependentTransform(RT.SoftmaxTransform(), 1),
        ),
        "Compose(Exp,Affine-)": (
            lambda: T.ComposeTransform([T.ExpTransform(), T.AffineTransform(0.0, -1.0)]),
            lambda: RT.ComposeTransform([RT.ExpTransform(), RT.AffineTransform(0.0, -1.0)]),
        ),
        "Compose(Affine-,Affine-)": (
            lambda: T.ComposeTransform(
                [T.AffineTransform(0.0, -1.0), T.AffineTransform(1.0, -2.0)]
            ),
            lambda: RT.ComposeTransform(
                [RT.AffineTransform(0.0, -1.0), RT.AffineTransform(1.0, -2.0)]
            ),
        ),
        "Compose(Exp,Softmax)": (
            lambda: T.ComposeTransform([T.ExpTransform(), T.SoftmaxTransform()]),
            lambda: RT.ComposeTransform([RT.ExpTransform(), RT.SoftmaxTransform()]),
        ),
        "Inverse(Affine-)": (
            lambda: T.AffineTransform(0.0, -1.0).inv,
            lambda: RT.AffineTransform(0.0, -1.0).inv,
        ),
        "Inverse(Abs)": (lambda: T.AbsTransform().inv, lambda: RT.AbsTransform().inv),
        "Cat": (
            lambda: D.constraint_registry.CatTransform([T.ExpTransform(), T.ExpTransform()], 0),
            lambda: RT.CatTransform([RT.ExpTransform(), RT.ExpTransform()], 0),
        ),
        "Stack": (
            lambda: D.constraint_registry.StackTransform([T.ExpTransform(), T.ExpTransform()], 0),
            lambda: RT.StackTransform([RT.ExpTransform(), RT.ExpTransform()], 0),
        ),
    }


_TRANSFORM_NAMES = [
    "Exp",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Affine(+)",
    "Affine(-)",
    "Affine(0)",
    "Affine(int)",
    "Affine(tensor)",
    "Power(-)",
    "Power(+)",
    "Abs",
    "Softmax",
    "StickBreaking",
    "LowerCholesky",
    "PositiveDefinite",
    "CorrCholesky",
    "Reshape",
    "CDF",
    "Independent(Exp)",
    "Independent(Affine-)",
    "Independent(Softmax)",
    "Compose(Exp,Affine-)",
    "Compose(Affine-,Affine-)",
    "Compose(Exp,Softmax)",
    "Inverse(Affine-)",
    "Inverse(Abs)",
    "Cat",
    "Stack",
]


@pytest.mark.parametrize("name", _TRANSFORM_NAMES)
def test_transform_sign_matches_reference(name: str, ref: Any) -> None:
    ours_factory, theirs_factory = _transform_cases(ref)[name]
    ours = _outcome(lambda: ours_factory().sign)
    theirs = _outcome(lambda: theirs_factory().sign)
    assert ours[0] == theirs[0], f"{name}: Lucid {ours[0]}, the reference {theirs[0]}"
    if ours[0] == "ok":
        np.testing.assert_array_equal(ours[1], theirs[1])


def test_the_base_transform_has_no_sign() -> None:
    with pytest.raises(NotImplementedError):
        D.Transform().sign


# ── TransformedDistribution.cdf / icdf ───────────────────────────────────────


def _td_cases(ref: Any) -> dict[str, Callable[[Any, Any, Any], Any]]:
    """Name -> builder(distributions module, transforms module, tensor fn)."""

    def normal(M: Any, t: Any) -> Any:
        return M.Normal(t(0.3), t(1.2))

    def exponential(M: Any, t: Any) -> Any:
        return M.Exponential(t(1.5))

    return {
        "Normal|Exp": lambda M, TR, t: M.TransformedDistribution(normal(M, t), [TR.ExpTransform()]),
        "Normal|Affine+": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.AffineTransform(t(1.0), t(2.0))]
        ),
        "Normal|Affine-": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.AffineTransform(t(1.0), t(-2.0))]
        ),
        "Normal|Affine-float": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.AffineTransform(0.0, -1.0)]
        ),
        "Normal|Sigmoid": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.SigmoidTransform()]
        ),
        "Normal|Tanh": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.TanhTransform()]
        ),
        "Normal|Exp,Affine-": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.ExpTransform(), TR.AffineTransform(t(0.0), t(-1.0))]
        ),
        "Normal|Affine-,Exp": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.AffineTransform(t(0.0), t(-1.0)), TR.ExpTransform()]
        ),
        "Normal|Affine-,Affine-": lambda M, TR, t: M.TransformedDistribution(
            normal(M, t), [TR.AffineTransform(t(0.0), t(-1.0)), TR.AffineTransform(t(1.0), t(-3.0))]
        ),
        "Exponential|Power-": lambda M, TR, t: M.TransformedDistribution(
            exponential(M, t), [TR.PowerTransform(t(-2.0))]
        ),
        "Exponential|Power+": lambda M, TR, t: M.TransformedDistribution(
            exponential(M, t), [TR.PowerTransform(t(2.0))]
        ),
    }


_TD_NAMES = list(_td_cases(None))
_PLACES = [("cpu", np.float32), ("cpu", np.float64), ("metal", np.float32)]
_PLACE_IDS = ["cpu-f32", "cpu-f64", "metal-f32"]


def _lucid_td(name: str, device: str, dtype: Any, ref: Any) -> Any:
    return _td_cases(ref)[name](D, T, lambda v: _lt(v, device, dtype))


def _ref_td(name: str, dtype: Any, ref: Any) -> Any:
    return _td_cases(ref)[name](
        ref.distributions, ref.distributions.transforms, lambda v: _rt(ref, v, dtype)
    )


# Points inside every case's codomain: the tanh and sigmoid cases need (-1, 1)
# and (0, 1), the power and exp cases a positive point, and the negated ones
# take anything the scaled Normal reaches, so one grid per support.
_VALUES = {
    "Normal|Exp": [0.2, 0.7, 1.0, 2.5],
    "Normal|Affine+": [-2.0, 0.0, 1.5, 4.0],
    "Normal|Affine-": [-3.0, -0.5, 1.0, 3.0],
    "Normal|Affine-float": [-1.5, -0.3, 0.4, 2.0],
    "Normal|Sigmoid": [0.1, 0.4, 0.5, 0.9],
    "Normal|Tanh": [-0.8, -0.2, 0.3, 0.9],
    "Normal|Exp,Affine-": [-3.0, -1.0, -0.4, -0.1],
    "Normal|Affine-,Exp": [0.3, 0.9, 1.4, 3.0],
    "Normal|Affine-,Affine-": [-4.0, 0.0, 1.0, 5.0],
    "Exponential|Power-": [0.3, 0.7, 1.4, 3.0],
    "Exponential|Power+": [0.5, 2.0, 5.0, 12.0],
}
_PROBS = [0.05, 0.3, 0.5, 0.8, 0.95]


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("name", _TD_NAMES)
def test_transformed_cdf_matches_reference(name: str, place: Any, ref: Any) -> None:
    device, dtype = place
    ours = _lucid_td(name, device, dtype, ref)
    theirs = _ref_td(name, dtype, ref)
    for v in _VALUES[name]:
        got = _arr(ours.cdf(_lt(v, device, dtype)))
        want = _arr(theirs.cdf(_rt(ref, v, dtype)))
        rtol = 1e-7 if dtype == np.float64 else 2e-4
        np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-6, err_msg=f"{name}.cdf({v})")


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("name", _TD_NAMES)
def test_transformed_icdf_matches_reference(name: str, place: Any, ref: Any) -> None:
    device, dtype = place
    ours = _lucid_td(name, device, dtype, ref)
    theirs = _ref_td(name, dtype, ref)
    for p in _PROBS:
        got = _arr(ours.icdf(_lt(p, device, dtype)))
        want = _arr(theirs.icdf(_rt(ref, p, dtype)))
        rtol = 1e-6 if dtype == np.float64 else 5e-4
        np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-5, err_msg=f"{name}.icdf({p})")


@pytest.mark.parametrize("name", _TD_NAMES)
def test_transformed_cdf_is_monotone_increasing(name: str) -> None:
    """A decreasing transform must not leave the CDF decreasing."""
    td = _lucid_td(name, "cpu", np.float64, None)
    grid = np.sort(np.asarray(_VALUES[name], dtype=np.float64))
    cdf = _arr(td.cdf(_lt(grid, "cpu", np.float64)))
    assert np.all(np.diff(cdf) >= 0.0), (name, cdf)


@pytest.mark.parametrize("name", _TD_NAMES)
def test_transformed_icdf_inverts_cdf(name: str) -> None:
    td = _lucid_td(name, "cpu", np.float64, None)
    for v in _VALUES[name]:
        x = _lt(v, "cpu", np.float64)
        np.testing.assert_allclose(_arr(td.icdf(td.cdf(x))), v, rtol=1e-6, atol=1e-8)


def test_transformed_cdf_refuses_a_chain_without_a_sign(ref: Any) -> None:
    ours = D.TransformedDistribution(D.Normal(0.0, 1.0), [T.AbsTransform()])
    theirs = ref.distributions.TransformedDistribution(
        ref.distributions.Normal(0.0, 1.0), [ref.distributions.transforms.AbsTransform()]
    )
    for method, arg in (("cdf", 0.5), ("icdf", 0.5)):
        got = _outcome(lambda: getattr(ours, method)(lucid.tensor(arg)))
        want = _outcome(lambda: getattr(theirs, method)(ref.tensor(arg)))
        assert got[0] == want[0] == "NotImplementedError", (method, got, want)


def test_transformed_cdf_of_a_base_without_a_cdf_refuses(ref: Any) -> None:
    ours = D.TransformedDistribution(
        D.Categorical(probs=lucid.tensor([0.5, 0.5])), [T.ExpTransform()]
    )
    theirs = ref.distributions.TransformedDistribution(
        ref.distributions.Categorical(probs=ref.tensor([0.5, 0.5])),
        [ref.distributions.transforms.ExpTransform()],
    )
    got = _outcome(lambda: ours.cdf(lucid.tensor(1.0)))
    want = _outcome(lambda: theirs.cdf(ref.tensor(1.0)))
    assert got[0] == want[0] == "NotImplementedError"


def test_relaxed_families_have_no_cdf_in_either(ref: Any) -> None:
    """The reference builds ``RelaxedBernoulli`` / ``RelaxedOneHotCategorical``
    on a logit-relaxed base that has no CDF, so neither answers ``cdf`` or
    ``icdf``; Lucid must refuse in the same way rather than invent one."""
    RD = ref.distributions
    pairs = {
        "RelaxedBernoulli": (
            D.RelaxedBernoulli(lucid.tensor(0.5), probs=lucid.tensor(0.3)),
            RD.RelaxedBernoulli(ref.tensor(0.5), probs=ref.tensor(0.3)),
            0.4,
        ),
        "RelaxedOneHotCategorical": (
            D.RelaxedOneHotCategorical(lucid.tensor(0.5), probs=lucid.tensor([0.3, 0.7])),
            RD.RelaxedOneHotCategorical(ref.tensor(0.5), probs=ref.tensor([0.3, 0.7])),
            [0.4, 0.6],
        ),
    }
    for name, (ours, theirs, point) in pairs.items():
        for method in ("cdf", "icdf"):
            got = _outcome(lambda: getattr(ours, method)(lucid.tensor(point)))
            want = _outcome(lambda: getattr(theirs, method)(ref.tensor(point)))
            assert got[0] == want[0], f"{name}.{method}: Lucid {got[0]}, the reference {want[0]}"


# ── Wishart.entropy ──────────────────────────────────────────────────────────


def _spd(p: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((p, p))
    return a @ a.T + p * np.eye(p)


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("excess", [0.5, 2.3, 11.0])
def test_wishart_entropy_matches_reference(p: int, excess: float, place: Any, ref: Any) -> None:
    device, dtype = place
    df = p - 1 + excess
    cov = _spd(p, seed=10 * p)
    ours = D.Wishart(_lt(df, device, dtype), covariance_matrix=_lt(cov, device, dtype))
    theirs = ref.distributions.Wishart(_rt(ref, df, dtype), covariance_matrix=_rt(ref, cov, dtype))
    rtol = 1e-7 if dtype == np.float64 else 2e-4
    np.testing.assert_allclose(_arr(ours.entropy()), _arr(theirs.entropy()), rtol=rtol, atol=1e-6)
    # perplexity is exp(entropy) in both; compared as the entropy it carries so
    # the float32 exponential's range does not decide the verdict.
    if dtype == np.float64:
        np.testing.assert_allclose(
            np.log(_arr(ours.perplexity())), np.log(_arr(theirs.perplexity())), rtol=rtol, atol=1e-5
        )


def test_wishart_entropy_is_batched_and_parameterisation_free(ref: Any) -> None:
    """A vector of degrees of freedom over one scale matrix is a batch, and the
    three ways to give the scale are one distribution."""
    cov = _spd(3, seed=5)
    chol = np.linalg.cholesky(cov)
    prec = np.linalg.inv(cov)
    df = np.array([2.5, 3.5, 7.0])
    theirs = _arr(
        ref.distributions.Wishart(
            _rt(ref, df, np.float64), covariance_matrix=_rt(ref, cov, np.float64)
        ).entropy()
    )
    for key, mat in (("covariance_matrix", cov), ("scale_tril", chol), ("precision_matrix", prec)):
        ours = D.Wishart(_lt(df, dtype=np.float64), **{key: _lt(mat, dtype=np.float64)})
        assert ours.entropy().shape == (3,)
        np.testing.assert_allclose(_arr(ours.entropy()), theirs, rtol=1e-6)


def test_wishart_entropy_is_differentiable_in_df(ref: Any) -> None:
    cov = _spd(2, seed=1)
    ours_df = _lt(4.5, dtype=np.float64)
    ours_df.requires_grad_(True)
    D.Wishart(ours_df, covariance_matrix=_lt(cov, dtype=np.float64)).entropy().backward()
    theirs_df = _rt(ref, 4.5, np.float64)
    theirs_df.requires_grad_(True)
    ref.distributions.Wishart(
        theirs_df, covariance_matrix=_rt(ref, cov, np.float64)
    ).entropy().backward()
    np.testing.assert_allclose(_arr(ours_df.grad), _arr(theirs_df.grad), rtol=1e-6)


# ── ContinuousBernoulli(lims=...) ────────────────────────────────────────────


def test_continuous_bernoulli_signature_matches_reference(ref: Any) -> None:
    ours = inspect.signature(D.ContinuousBernoulli.__init__)
    theirs = inspect.signature(ref.distributions.ContinuousBernoulli.__init__)
    assert [(n, p.default) for n, p in ours.parameters.items()] == [
        (n, p.default) for n, p in theirs.parameters.items()
    ]


def test_continuous_bernoulli_lims_defaults_to_the_references(ref: Any) -> None:
    assert D.ContinuousBernoulli(probs=0.3)._lims == (0.499, 0.501)
    assert D.ContinuousBernoulli(probs=0.3).expand((2,))._lims == (0.499, 0.501)
    assert D.ContinuousBernoulli(probs=0.3, lims=(0.4, 0.6)).expand((2,))._lims == (0.4, 0.6)


_CB_PROBS = [0.05, 0.2, 0.39, 0.45, 0.4999, 0.5, 0.5003, 0.55, 0.61, 0.8, 0.97]
_CB_LIMS = [(0.499, 0.501), (0.48, 0.52), (0.45, 0.55), (0.4, 0.6)]


@pytest.mark.parametrize("lims", _CB_LIMS, ids=[str(w) for w in _CB_LIMS])
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_continuous_bernoulli_lims_agree_with_reference(
    lims: tuple[float, float], device: str, ref: Any
) -> None:
    """Inside the window the reference takes its low-order Taylor series, so a
    wide window is compared where that series is still the value to a tolerance
    its own truncation sets; the moments, normaliser and entropy all go through
    ``lims``.  ``cdf`` / ``icdf`` do not (Lucid's are exact there)."""
    dtype = np.float64 if device == "cpu" else np.float32
    probs = np.asarray(_CB_PROBS)
    ours = D.ContinuousBernoulli(probs=_lt(probs, device, dtype), lims=lims)
    theirs = ref.distributions.ContinuousBernoulli(probs=_rt(ref, probs, dtype), lims=lims)
    atol = 5e-5 if device == "cpu" else 2e-4
    for attr in ("mean", "variance"):
        np.testing.assert_allclose(
            _arr(getattr(ours, attr)),
            _arr(getattr(theirs, attr)),
            rtol=1e-3,
            atol=atol,
            err_msg=attr,
        )
    np.testing.assert_allclose(_arr(ours.entropy()), _arr(theirs.entropy()), rtol=1e-3, atol=atol)
    x = np.asarray([0.1, 0.5, 0.9])[:, None]
    np.testing.assert_allclose(
        _arr(ours.log_prob(_lt(x, device, dtype))),
        _arr(theirs.log_prob(_rt(ref, x, dtype))),
        rtol=1e-3,
        atol=atol,
    )


def test_continuous_bernoulli_wider_lims_only_widens_the_series_region() -> None:
    """``lims`` changes which formula is used, not what the distribution is:
    narrow, default and wide windows agree on every moment to the series'
    accuracy."""
    probs = _lt(_CB_PROBS, dtype=np.float64)
    base = D.ContinuousBernoulli(probs=probs)
    wide = D.ContinuousBernoulli(probs=probs, lims=(0.45, 0.55))
    for attr in ("mean", "variance"):
        np.testing.assert_allclose(_arr(getattr(base, attr)), _arr(getattr(wide, attr)), atol=1e-5)
    np.testing.assert_allclose(_arr(base.entropy()), _arr(wide.entropy()), atol=1e-5)


@pytest.mark.parametrize("lims", [(0.6, 0.4), (-0.1, 0.5), (0.5, 1.2)])
def test_continuous_bernoulli_refuses_an_unordered_or_out_of_range_lims(
    lims: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="lims"):
        D.ContinuousBernoulli(probs=0.3, lims=lims)


# ── the constraints missing from __all__ ─────────────────────────────────────


def test_constraints_all_covers_the_references(ref: Any) -> None:
    reference_all = set(ref.distributions.constraints.__all__)
    ours_all = set(constraints.__all__)
    assert reference_all <= ours_all, sorted(reference_all - ours_all)
    for name in constraints.__all__:
        assert hasattr(constraints, name), name


_MATRICES: dict[str, np.ndarray] = {
    "identity": np.eye(3),
    "spd": _spd(3, seed=3),
    "singular_psd": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "indefinite": np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "lower": np.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.1, 0.2, 3.0]]),
    "lower_neg_diag": np.array([[-1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.1, 0.2, 3.0]]),
    "upper": np.array([[1.0, 0.5, 0.1], [0.0, 2.0, 0.2], [0.0, 0.0, 3.0]]),
    "asymmetric": np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "batch": np.stack([np.eye(3), np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]),
    "batch_mixed": np.stack([np.eye(3), np.arange(9.0).reshape(3, 3)]),
    "rectangular": np.ones((2, 3)),
    "rectangular_batch": np.ones((4, 3, 2)),
    "zero": np.zeros((3, 3)),
}

_MATRIX_CONSTRAINTS = [
    "lower_triangular",
    "square",
    "symmetric",
    "positive_semidefinite",
    "positive_definite",
    "lower_cholesky",
    "corr_cholesky",
]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("matrix", sorted(_MATRICES))
@pytest.mark.parametrize("name", _MATRIX_CONSTRAINTS)
def test_matrix_constraint_check_matches_reference(
    name: str, matrix: str, device: str, ref: Any
) -> None:
    value = _MATRICES[matrix]
    ours = _outcome(lambda: getattr(constraints, name).check(_lt(value, device)))
    theirs = _outcome(lambda: getattr(ref.distributions.constraints, name).check(_rt(ref, value)))
    assert (
        ours[0] == theirs[0]
    ), f"{name}.check({matrix}): Lucid {ours[0]}, the reference {theirs[0]}"
    if ours[0] != "ok":
        return
    assert ours[1].shape == theirs[1].shape
    if matrix == "batch_mixed" and name in {"positive_semidefinite", "positive_definite"}:
        # The reference returns the symmetry verdict alone as soon as one
        # matrix in the batch is not symmetric, so a symmetric but indefinite
        # matrix beside it reads as admitted; Lucid decides each matrix on its
        # own, which is the answer for every matrix the reference decides alone.
        assert ours[1].tolist() == [True, False]
        return
    np.testing.assert_array_equal(ours[1], theirs[1], err_msg=f"{name}.check({matrix})")


@pytest.mark.parametrize("matrix", sorted(_MATRICES))
@pytest.mark.parametrize(
    "name", ["lower_triangular", "square", "symmetric", "positive_semidefinite"]
)
def test_matrix_constraint_table_attributes(name: str, matrix: str, ref: Any) -> None:
    ours = getattr(constraints, name)
    theirs = getattr(ref.distributions.constraints, name)
    assert ours.event_dim == theirs.event_dim == 2
    assert ours.is_discrete == theirs.is_discrete is False


def test_positive_integer_matches_reference(ref: Any) -> None:
    values = [-np.inf, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 7.0, np.inf, np.nan]
    ours = constraints.positive_integer.check(_lt(values))
    theirs = ref.distributions.constraints.positive_integer.check(_rt(ref, values))
    np.testing.assert_array_equal(_arr(ours), _arr(theirs))
    assert (
        constraints.positive_integer.is_discrete
        is ref.distributions.constraints.positive_integer.is_discrete
    )
    assert constraints.positive_integer.event_dim == 0


def test_dependent_matches_reference(ref: Any) -> None:
    RC = ref.distributions.constraints
    for module in (constraints, RC):
        assert module.is_dependent(module.dependent)
        assert not module.is_dependent(module.positive)
        assert not module.is_dependent(module.real_vector)
        for attr in ("is_discrete", "event_dim"):
            with pytest.raises(NotImplementedError):
                getattr(module.dependent, attr)
        with pytest.raises(ValueError):
            module.dependent.check(None)
        declared = module.dependent(is_discrete=True, event_dim=1)
        assert (declared.is_discrete, declared.event_dim) == (True, 1)
        assert module.is_dependent(declared)
        # A later call keeps what was declared and overrides what it names.
        again = declared(event_dim=2)
        assert (again.is_discrete, again.event_dim) == (True, 2)


def test_dependent_property_matches_reference(ref: Any) -> None:
    RC = ref.distributions.constraints

    def build(module: Any) -> Any:
        class Box:
            def __init__(self, high: float) -> None:
                self.high = high

            @module.dependent_property(is_discrete=False, event_dim=0)
            def support(self) -> Any:
                return module.interval(0.0, self.high)

            @module.dependent_property
            def bare(self) -> Any:
                return module.real

        return Box

    for module in (constraints, RC):
        Box = build(module)
        assert module.is_dependent(Box.support)
        assert module.is_dependent(Box.bare)
        assert (Box.support.is_discrete, Box.support.event_dim) == (False, 0)
        with pytest.raises(NotImplementedError):
            Box.bare.event_dim
        assert Box(3.0).support.upper_bound == 3.0
        assert repr(Box(3.0).bare) == repr(module.real)


def test_mixture_same_family_constraint_matches_reference(ref: Any) -> None:
    RC = ref.distributions.constraints
    low = np.array([[0.0, 1.0], [0.0, 2.0]])
    high = np.array([[2.0, 3.0], [1.5, 4.0]])
    ours = constraints.MixtureSameFamilyConstraint(constraints.interval(_lt(low), _lt(high)))
    theirs = RC.MixtureSameFamilyConstraint(RC.interval(_rt(ref, low), _rt(ref, high)))
    assert (ours.event_dim, ours.is_discrete) == (theirs.event_dim, theirs.is_discrete)
    for value in ([1.2, 1.4], [1.2, 0.7], [0.5, 1.7], [-1.0, 2.0]):
        np.testing.assert_array_equal(
            _arr(ours.check(_lt(value))), _arr(theirs.check(_rt(ref, value))), err_msg=str(value)
        )
    # a vector event
    simplex_ours = constraints.MixtureSameFamilyConstraint(constraints.simplex)
    simplex_theirs = RC.MixtureSameFamilyConstraint(RC.simplex)
    assert simplex_ours.event_dim == simplex_theirs.event_dim == 1
    point = [[0.2, 0.8], [0.5, 0.6]]
    np.testing.assert_array_equal(
        _arr(simplex_ours.check(_lt(point))), _arr(simplex_theirs.check(_rt(ref, point)))
    )
    with pytest.raises(TypeError):
        constraints.MixtureSameFamilyConstraint("positive")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simplex_ours.check(_lt(0.5))


def test_mixture_support_is_the_mixture_constraint_and_validates_like_the_reference(
    ref: Any,
) -> None:
    low, high = [0.0, 1.0], [2.0, 3.0]
    ours = D.MixtureSameFamily(
        D.Categorical(probs=lucid.tensor([0.4, 0.6])),
        D.Uniform(lucid.tensor(low), lucid.tensor(high)),
    )
    theirs = ref.distributions.MixtureSameFamily(
        ref.distributions.Categorical(probs=ref.tensor([0.4, 0.6])),
        ref.distributions.Uniform(ref.tensor(low), ref.tensor(high)),
    )
    assert isinstance(ours.support, constraints.MixtureSameFamilyConstraint)
    for value in (1.5, 0.5, 2.5, -1.0):
        got = _outcome(lambda: ours.log_prob(lucid.tensor(value)))
        want = _outcome(lambda: theirs.log_prob(ref.tensor(value)))
        assert got[0] == want[0], f"log_prob({value}): Lucid {got[0]}, the reference {want[0]}"


# ── the registry: positive_semidefinite maps to a transform ──────────────────


def test_transform_to_positive_semidefinite_lands_in_the_constraint(ref: Any) -> None:
    ours = D.transform_to(constraints.positive_semidefinite)
    theirs = ref.distributions.transform_to(ref.distributions.constraints.positive_semidefinite)
    assert type(ours).__name__ == type(theirs).__name__
    x = np.random.default_rng(0).standard_normal((3, 3))
    y = ours(_lt(x, dtype=np.float64))
    assert bool(constraints.positive_semidefinite.check(y).item())
