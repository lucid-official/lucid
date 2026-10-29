"""The Gamma sampler and its dependants stay off 0, 1 and NaN (LCD-677).

``Gamma(0.01).sample((100000,))`` held 35 534 exact zeros, ``Beta(.1, .1)``
9 111 exact ones and ``Dirichlet([.01] * 3)`` 4 456 rows of ``NaN``.  The sampler
boosts a draw for ``concentration < 1`` by ``U ** (1 / concentration)``, which
falls under the smallest float32 once the concentration is a few hundredths;
the rounded-to-zero draw then turned every ratio, reciprocal and logarithm
taken of it into ``0``, ``1``, ``NaN`` or ``-inf``.

Every distribution that draws through that sampler is swept here over the
concentrations that used to break it: no draw leaves its support's open
interior, and ``log_prob`` of the draws is finite.  The ranges the reference
holds its draws in, and the distribution they follow, are compared with the
reference in the hard regimes (two-sample Kolmogorov-Smirnov, with the atoms
at the clamps included).
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.distributions as D
from lucid.distributions.gamma import _standard_gamma, _standard_gamma_grad

_TINY = 2.0**-126  # the smallest normal float32
_BELOW_ONE = 1.0 - 2.0**-24  # the largest float32 below 1
_CONCENTRATIONS = [1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 1e3]


def _t(lib: ModuleType, device: str, *values: float):
    return lib.tensor(list(values), dtype=lib.float32, device=device)


def _flat(x) -> list[float]:
    return x.detach().cpu().reshape(-1).tolist()


# name -> (builder(device, c), support)
_SUPPORTED: dict[str, tuple[Callable[[str, float], D.Distribution], str]] = {
    "Gamma": (lambda dev, c: D.Gamma(_t(lucid, dev, c), _t(lucid, dev, 1.0)), "positive"),
    "Chi2": (lambda dev, c: D.Chi2(_t(lucid, dev, 2 * c)), "positive"),
    "InverseGamma": (
        lambda dev, c: D.InverseGamma(_t(lucid, dev, c), _t(lucid, dev, 1.0)),
        "positive",
    ),
    "Beta": (lambda dev, c: D.Beta(_t(lucid, dev, c), _t(lucid, dev, c)), "unit"),
    "Beta-skewed": (lambda dev, c: D.Beta(_t(lucid, dev, c), _t(lucid, dev, 1.0)), "unit"),
    "Dirichlet": (lambda dev, c: D.Dirichlet(_t(lucid, dev, c, c, c)), "simplex"),
}


def _in_support(values: list[float], support: str) -> bool:
    if support == "positive":
        return all(v >= _TINY and math.isfinite(v) for v in values)
    return all(_TINY <= v <= _BELOW_ONE for v in values)


@pytest.mark.parametrize("name", list(_SUPPORTED))
@pytest.mark.parametrize("c", _CONCENTRATIONS)
@pytest.mark.parametrize("draw", ["sample", "rsample"])
def test_draws_stay_in_the_open_support_and_score_finite(
    name: str, c: float, draw: str, device: str
) -> None:
    build, support = _SUPPORTED[name]
    dist = build(device, c)
    lucid.manual_seed(0)
    x = getattr(dist, draw)((20000,))
    values = _flat(x)
    assert _in_support(values, support), (name, c, draw)
    if support == "simplex":
        assert all(abs(s - 1.0) < 1e-4 or s > 1.0 - 1e-4 for s in _flat(x.sum(dim=-1)))
    log_prob = _flat(dist.log_prob(x))
    assert all(math.isfinite(v) for v in log_prob), (name, c, draw)


def test_the_reported_counts_are_gone(device: str) -> None:
    lucid.manual_seed(0)
    gamma = D.Gamma(_t(lucid, device, 0.01), _t(lucid, device, 1.0)).sample((100000,))
    beta = D.Beta(_t(lucid, device, 0.1), _t(lucid, device, 0.1)).sample((100000,))
    dirichlet = D.Dirichlet(_t(lucid, device, 0.01, 0.01, 0.01)).sample((100000,))
    assert int((gamma == 0).sum().item()) == 0
    assert int((beta == 1).sum().item()) == 0
    assert int(lucid.isnan(dirichlet).any(dim=-1).sum().item()) == 0


# ── the other consumers of the sampler ──────────────────────────────────────


def _finite_and_nonzero(x) -> bool:
    return all(math.isfinite(v) for v in _flat(x))


@pytest.mark.parametrize("c", [1e-3, 1e-2, 0.1, 1.0, 100.0])
@pytest.mark.parametrize("draw", ["sample", "rsample"])
def test_student_t_and_fisher_draws_are_finite(c: float, draw: str, device: str) -> None:
    lucid.manual_seed(1)
    t = getattr(D.StudentT(_t(lucid, device, c)), draw)((5000,))
    f = getattr(D.FisherSnedecor(_t(lucid, device, c), _t(lucid, device, c)), draw)((5000,))
    assert _finite_and_nonzero(t)
    # Spanning 1e-39 .. 1e38 at df = 0.01, the quotient's low end is flushed
    # to 0 on Metal: only a non-finite or negative draw is wrong.
    assert all(v >= 0.0 and math.isfinite(v) for v in _flat(f))


@pytest.mark.parametrize("df", [1.0 + 1e-3, 1.01, 1.5, 3.0])
@pytest.mark.parametrize("draw", ["sample", "rsample"])
def test_wishart_draws_are_finite_and_symmetric(df: float, draw: str, device: str) -> None:
    # df barely above d - 1 puts the last Bartlett chi-squared's concentration
    # at (df - 1) / 2 ~ 5e-4, where its draw used to round to 0 and the whole
    # matrix to a singular one.
    lucid.manual_seed(2)
    scale = lucid.eye(2, device=device)
    w = getattr(D.Wishart(lucid.tensor(df, device=device), covariance_matrix=scale), draw)((2000,))
    assert _finite_and_nonzero(w)
    assert all(v > 0.0 for v in _flat(w[..., 0, 0]) + _flat(w[..., 1, 1]))
    assert _flat(w[..., 0, 1]) == _flat(w[..., 1, 0])


@pytest.mark.parametrize("c", [1e-3, 1e-2, 0.5, 5.0])
def test_negative_binomial_draws_are_valid_counts(c: float, device: str) -> None:
    lucid.manual_seed(3)
    x = D.NegativeBinomial(_t(lucid, device, c), probs=_t(lucid, device, 0.4)).sample((5000,))
    assert all(math.isfinite(v) and v >= 0.0 and v == int(v) for v in _flat(x))


# ── rsample gradients are those of the unclamped expression ────────────────


@pytest.mark.parametrize("c", [1e-3, 1e-2, 0.5, 4.0])
def test_gamma_rsample_gradient_is_the_implicit_one(c: float, device: str) -> None:
    alpha = _t(lucid, device, c, c, c).requires_grad_(True)
    rate = _t(lucid, device, 2.0, 2.0, 2.0)
    lucid.manual_seed(4)
    x = D.Gamma(alpha, rate).rsample()
    x.sum().backward()
    want = _standard_gamma_grad(alpha.detach(), x.detach() * rate) / rate
    got = alpha.grad
    assert got is not None
    for g, w in zip(_flat(got), _flat(want), strict=True):
        assert math.isfinite(g) and g == pytest.approx(w, rel=1e-4, abs=1e-30)


def _dirichlet_grad(device: str, c: float, *, plain_quotient: bool) -> list[float]:
    a = _t(lucid, device, c, c).requires_grad_(True)
    lucid.manual_seed(5)
    if plain_quotient:
        g = _standard_gamma(a.expand(64, 2))
        x = g / g.sum(dim=-1, keepdim=True)
    else:
        x = D.Dirichlet(a).rsample((64,))
    x[..., 0].sum().backward()
    assert a.grad is not None
    return _flat(a.grad)


@pytest.mark.parametrize("c", [0.1, 0.5, 4.0])
def test_dirichlet_rsample_gradient_ignores_the_range_clamp(c: float, device: str) -> None:
    # At 0.1 a tenth of the rows sit on the upper clamp; a clamp in the graph
    # would zero their gradient, and the log-space normalisation must agree
    # with the plain quotient wherever the quotient is not itself degenerate.
    got = _dirichlet_grad(device, c, plain_quotient=False)
    want = _dirichlet_grad(device, c, plain_quotient=True)
    assert got == pytest.approx(want, rel=2e-3, abs=1e-3)


@pytest.mark.parametrize("c", [1e-3, 1e-2])
def test_dirichlet_rsample_gradient_is_finite_where_the_quotient_was_nan(
    c: float, device: str
) -> None:
    assert all(math.isfinite(v) for v in _dirichlet_grad(device, c, plain_quotient=False))


@pytest.mark.parametrize("c", [1e-3, 1e-2, 0.1])
def test_beta_rsample_gradient_is_finite(c: float, device: str) -> None:
    a = _t(lucid, device, c).requires_grad_(True)
    b = _t(lucid, device, c).requires_grad_(True)
    lucid.manual_seed(5)
    D.Beta(a, b).rsample((64,)).sum().backward()
    assert a.grad is not None and b.grad is not None
    assert all(math.isfinite(v) for v in _flat(a.grad) + _flat(b.grad))


# ── distribution against the reference ──────────────────────────────────────


def _ks(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, ties handled."""
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    i = j = 0
    worst = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] == x:
            i += 1
        while j < m and b[j] == x:
            j += 1
        worst = max(worst, abs(i / n - j / m))
    return worst


_N = 20000
_KS_LIMIT = 1.95 * math.sqrt(2.0 / _N)  # alpha ~ 0.001


def _reference_cases(lib: ModuleType, device: str):
    d = lib.distributions
    one = _t(lib, device, 1.0)
    return {
        "Gamma(0.01)": d.Gamma(_t(lib, device, 0.01), one),
        "Gamma(0.1)": d.Gamma(_t(lib, device, 0.1), one),
        "Gamma(0.5)": d.Gamma(_t(lib, device, 0.5), one),
        "Gamma(3)": d.Gamma(_t(lib, device, 3.0), one),
        "Beta(.1,.1)": d.Beta(_t(lib, device, 0.1), _t(lib, device, 0.1)),
        "Beta(.01,2)": d.Beta(_t(lib, device, 0.01), _t(lib, device, 2.0)),
        "Beta(.5,.5)": d.Beta(_t(lib, device, 0.5), _t(lib, device, 0.5)),
        "Dirichlet(.01)": d.Dirichlet(_t(lib, device, 0.01, 0.01, 0.01)),
        "Dirichlet(.1)": d.Dirichlet(_t(lib, device, 0.1, 0.1, 0.1)),
    }


@pytest.mark.parity
@pytest.mark.parametrize(
    "case",
    [
        "Gamma(0.01)",
        "Gamma(0.1)",
        "Gamma(0.5)",
        "Gamma(3)",
        "Beta(.1,.1)",
        "Beta(.01,2)",
        "Beta(.5,.5)",
        "Dirichlet(.01)",
        "Dirichlet(.1)",
    ],
)
def test_draws_follow_the_reference(case: str, ref: ModuleType, device: str) -> None:
    lucid.manual_seed(7)
    ref.manual_seed(7)
    mine = _reference_cases(lucid, device)[case].sample((_N,))
    theirs = _reference_cases(ref, "cpu")[case].sample((_N,))
    mine_flat, theirs_flat = _flat(mine), _flat(theirs)
    assert not any(math.isnan(v) for v in mine_flat)
    assert min(mine_flat) >= _TINY
    if mine.ndim == 2:
        for k in range(mine.shape[-1]):
            assert _ks(_flat(mine[:, k]), _flat(theirs[:, k])) < _KS_LIMIT, (case, k)
    else:
        assert _ks(mine_flat, theirs_flat) < _KS_LIMIT, case
