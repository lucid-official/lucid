"""Distribution formulas checked against oracles that do not share their code.

Each class is one issue.  A test either compares with a value from an
independent oracle (a closed form evaluated in NumPy, SciPy constants, a
numerical integral of the density) and runs everywhere, or compares with
the reference framework through the ``ref`` fixture and is marked
``parity``.  Every check runs on CPU and on Metal.

* ``TestFormulaErrors`` (CHA-143) — four closed forms that were silently
  wrong on both devices.
* ``TestSupportBoundaries`` (CHA-144) — ``-inf`` from overflowing
  ``log(1 + exp(l))``, NaN from ``0 · log 0`` at the edge of a support, and
  ``-inf`` logits (masked categories) refused by ``constraints.real``.
* ``TestReferenceDomains`` (CHA-236) — parameter domains as wide as the
  reference's under default validation, and the closed forms at their
  edges.

CHA-148's cases — Python numbers held as 0-dim host tensors that raised
``DeviceMismatch`` against Metal tensors — are now one sweep over every
distribution, ``test_distributions_metal_scalar_params.py``.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

DEVICES = ["cpu", "metal"]


def _t(values: Any, device: str) -> lucid.Tensor:
    return lucid.tensor(np.asarray(values, dtype=np.float32), device=device)


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.numpy(), dtype=np.float64)


def _ref_np(t: Any) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


# ── CHA-143: formula errors ──────────────────────────────────────────────────

_PRECISION = np.array([[2.0, 0.6], [0.6, 1.0]])
_PRECISION_3 = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])


def _mvn_log_prob_from_precision(x: np.ndarray, loc: np.ndarray, precision: np.ndarray) -> float:
    """``log N(x; loc, P⁻¹)`` written with ``P`` directly — no factorisation."""
    diff = x - loc
    _, logdet = np.linalg.slogdet(precision)
    d = len(loc)
    return float(-0.5 * diff @ precision @ diff + 0.5 * logdet - 0.5 * d * np.log(2 * np.pi))


def _continuous_bernoulli_moments(lam: float) -> tuple[float, float]:
    """Mean and variance of ``ContinuousBernoulli(λ)`` by integrating its density."""
    x = np.linspace(0.0, 1.0, 200_001)
    density = lam**x * (1.0 - lam) ** (1.0 - x)
    z = _trapezoid(density, x)
    m1 = _trapezoid(x * density, x) / z
    m2 = _trapezoid(x * x * density, x) / z
    return m1, m2 - m1 * m1


@pytest.mark.parametrize("device", DEVICES)
class TestFormulaErrors:
    """CHA-143 — MVN(precision), Dirichlet entropy, RelaxedBernoulli density,
    ContinuousBernoulli variance."""

    @pytest.mark.parametrize(
        "precision,loc,x",
        [
            (_PRECISION, [0.5, -1.0], [1.0, 0.3]),
            (_PRECISION_3, [0.0, 0.0, 0.0], [0.2, -0.4, 1.1]),
        ],
        ids=["2d", "3d"],
    )
    def test_mvn_precision_log_prob(
        self, device: str, precision: np.ndarray, loc: list[float], x: list[float]
    ) -> None:
        """``inv(chol(P)).mT`` is upper-triangular; ``log_prob`` read it as lower.

        Before: -2.53343 for the 2-d case, against -3.07553.
        """
        d = D.MultivariateNormal(_t(loc, device), precision_matrix=_t(precision, device))
        got = float(_np(d.log_prob(_t(x, device))))
        want = _mvn_log_prob_from_precision(np.array(x), np.array(loc), precision)
        assert got == pytest.approx(want, abs=1e-5)

    def test_mvn_precision_agrees_with_covariance(self, device: str) -> None:
        cov = np.linalg.inv(_PRECISION_3)
        x = _t([[0.2, -0.4, 1.1], [1.0, 0.0, -2.0]], device)
        by_precision = D.MultivariateNormal(
            _t([0.1, 0.2, 0.3], device), precision_matrix=_t(_PRECISION_3, device)
        )
        by_covariance = D.MultivariateNormal(
            _t([0.1, 0.2, 0.3], device), covariance_matrix=_t(cov, device)
        )
        np.testing.assert_allclose(
            _np(by_precision.log_prob(x)), _np(by_covariance.log_prob(x)), atol=1e-5
        )
        np.testing.assert_allclose(
            _np(by_precision.entropy()), _np(by_covariance.entropy()), atol=1e-5
        )
        tril = _np(by_precision.scale_tril)
        np.testing.assert_allclose(np.triu(tril, 1), 0.0, atol=0.0)
        np.testing.assert_allclose(tril, np.linalg.cholesky(cov), atol=1e-5)

    def test_mvn_precision_batched(self, device: str) -> None:
        precisions = np.stack([_PRECISION, np.array([[1.5, -0.3], [-0.3, 0.8]])])
        d = D.MultivariateNormal(_t([0.0, 0.0], device), precision_matrix=_t(precisions, device))
        got = _np(d.log_prob(_t([1.0, 0.3], device)))
        want = [
            _mvn_log_prob_from_precision(np.array([1.0, 0.3]), np.zeros(2), p) for p in precisions
        ]
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_dirichlet_entropy(self, device: str) -> None:
        """ψ(α₀) was counted twice.  Before: ``[3.874, -1.217]``.

        Expected values: ``scipy.stats.dirichlet(α).entropy()``.
        """
        d = D.Dirichlet(_t([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], device))
        np.testing.assert_allclose(
            _np(d.entropy()), [-1.2443445622221003, -1.1621229335906549], atol=1e-5
        )

    def test_dirichlet_entropy_of_the_flat_simplex(self, device: str) -> None:
        """``Dirichlet(1, …, 1)`` is uniform on the simplex: ``H = −log((K−1)!)``."""
        for k in (2, 3, 5):
            got = float(_np(D.Dirichlet(_t(np.ones(k), device)).entropy()))
            assert got == pytest.approx(-np.log(float(np.prod(np.arange(1, k)))), abs=1e-5)

    def test_relaxed_bernoulli_log_prob(self, device: str) -> None:
        """Before: ``[-0.357, -3.251, -6.947]`` — an exponent of ``τ+1`` and no
        sigmoid Jacobian."""
        d = D.RelaxedBernoulli(_t(0.7, device), probs=_t(0.3, device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.1, 0.5, 0.9], device))),
            [0.54798782, -0.53102827, -0.51020932],
            atol=1e-5,
        )

    @pytest.mark.parametrize("temperature,logit", [(2.0, -1.5), (1.0, 0.7), (3.5, 2.0)])
    def test_relaxed_bernoulli_density_integrates_to_one(
        self, device: str, temperature: float, logit: float
    ) -> None:
        """Bounded at both ends for ``τ ≥ 1``, so integrated in ``y`` directly."""
        y = np.linspace(1e-6, 1.0 - 1e-6, 400_001)
        d = D.RelaxedBernoulli(_t(temperature, device), logits=_t(logit, device))
        density = np.exp(_np(d.log_prob(_t(y, device))))
        assert _trapezoid(density, y) == pytest.approx(1.0, abs=2e-4)

    @pytest.mark.parametrize("temperature,prob", [(0.6, 0.4), (0.7, 0.5), (0.9, 0.8)])
    def test_relaxed_bernoulli_density_integrates_to_one_below_unit_temperature(
        self, device: str, temperature: float, prob: float
    ) -> None:
        """For ``τ < 1`` the density blows up at 0 and 1, so integrate over
        ``z = logit(y)``: ``∫ p(y) dy = ∫ p(σ(z)) σ(z)(1 − σ(z)) dz``.

        Float32 cannot hold a ``y`` closer to 1 than ``σ(16.6)``; the
        parameters keep the mass beyond ``|z| = 16`` under 2e-4.
        """
        z = np.linspace(-16.0, 16.0, 400_001)
        y = (1.0 / (1.0 + np.exp(-z))).astype(np.float32).astype(np.float64)
        y = y[(y > 0.0) & (y < 1.0)]
        z = np.log(y) - np.log1p(-y)  # the z each representable y stands for
        d = D.RelaxedBernoulli(_t(temperature, device), probs=_t(prob, device))
        log_p = _np(d.log_prob(_t(y, device)))
        integrand = np.exp(log_p + np.log(y) + np.log1p(-y))
        assert _trapezoid(integrand, z) == pytest.approx(1.0, abs=1e-3)

    def test_continuous_bernoulli_variance(self, device: str) -> None:
        """Before: ``[0.0695, 0.0833, 0.0]`` — a wrong ``E[X²]`` with the
        negative results clamped to 0."""
        probs = [0.01, 0.2, 0.3, 0.45, 0.4999, 0.5, 0.5003, 0.55, 0.65, 0.7, 0.99]
        d = D.ContinuousBernoulli(probs=_t(probs, device))
        want = [_continuous_bernoulli_moments(p)[1] for p in probs]
        np.testing.assert_allclose(_np(d.variance), want, atol=2e-6)

    def test_continuous_bernoulli_mean_near_one_half(self, device: str) -> None:
        """The closed-form mean cancels catastrophically near ½ — in float32 it
        was off by 0.5 at λ = 0.4999 and by 0.25 at λ = 0.5001."""
        probs = [0.38, 0.45, 0.4999, 0.5001, 0.5002, 0.501, 0.505, 0.62]
        d = D.ContinuousBernoulli(probs=_t(probs, device))
        want = [_continuous_bernoulli_moments(p)[0] for p in probs]
        np.testing.assert_allclose(_np(d.mean), want, atol=2e-6)

    def test_continuous_bernoulli_moments_from_large_logits(self, device: str) -> None:
        """``sigmoid(±100)`` rounds to 0 or 1; the moments come from the logit."""
        d = D.ContinuousBernoulli(logits=_t([100.0, -100.0], device))
        np.testing.assert_allclose(_np(d.mean), [0.99, 0.01], atol=1e-6)
        np.testing.assert_allclose(_np(d.variance), [1e-4, 1e-4], atol=1e-7)

    def test_continuous_bernoulli_moment_gradients_are_finite_at_one_half(
        self, device: str
    ) -> None:
        """The closed form is evaluated everywhere and masked; its pole at ½
        must not reach the gradient."""
        p = _t([0.2, 0.5, 0.45], device)
        p.requires_grad_(True)
        d = D.ContinuousBernoulli(probs=p)
        (d.mean.sum() + d.variance.sum()).backward()
        grad = _np(p.grad)
        assert np.isfinite(grad).all(), grad
        assert grad[1] == pytest.approx(1.0 / 3.0, abs=1e-5)

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731

        lucid_mvn = D.MultivariateNormal(
            _t([0.5, -1.0], device), precision_matrix=_t(_PRECISION, device)
        )
        ref_mvn = RD.MultivariateNormal(rt([0.5, -1.0]), precision_matrix=rt(_PRECISION))
        np.testing.assert_allclose(
            _np(lucid_mvn.log_prob(_t([1.0, 0.3], device))),
            _ref_np(ref_mvn.log_prob(rt([1.0, 0.3]))),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            _np(lucid_mvn.scale_tril), _ref_np(ref_mvn.scale_tril), atol=1e-5
        )

        conc = [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]
        np.testing.assert_allclose(
            _np(D.Dirichlet(_t(conc, device)).entropy()),
            _ref_np(RD.Dirichlet(rt(conc)).entropy()),
            atol=1e-5,
        )

        y = [0.05, 0.3, 0.5, 0.9]
        np.testing.assert_allclose(
            _np(D.RelaxedBernoulli(_t(0.7, device), probs=_t(0.3, device)).log_prob(_t(y, device))),
            _ref_np(RD.RelaxedBernoulli(rt(0.7), probs=rt(0.3)).log_prob(rt(y))),
            atol=1e-5,
        )

        # Away from the reference's own float32 cancellation band (0.4–0.499).
        probs = [0.05, 0.2, 0.3, 0.5, 0.65, 0.7, 0.95]
        lucid_cb = D.ContinuousBernoulli(probs=_t(probs, device))
        ref_cb = RD.ContinuousBernoulli(probs=rt(probs))
        np.testing.assert_allclose(_np(lucid_cb.mean), _ref_np(ref_cb.mean), atol=1e-5)
        np.testing.assert_allclose(_np(lucid_cb.variance), _ref_np(ref_cb.variance), atol=1e-5)


# ── CHA-144: support boundaries ──────────────────────────────────────────────

_INF = float("inf")


@pytest.mark.parametrize("device", DEVICES)
class TestSupportBoundaries:
    """CHA-144 — overflow at large logits, ``0 · log 0`` NaN, and masked logits.

    Expected values are the reference framework's (it holds an exact 0 or 1
    probability one epsilon inside ``[0, 1]``, hence the ``1e-7``-sized
    residues), or SciPy's where noted.
    """

    def test_bernoulli_large_logits(self, device: str) -> None:
        """``log(1 + exp(l))`` overflowed at ``l ≈ 89``: before, ``[-inf]``
        and an entropy of ``inf``."""
        d = D.Bernoulli(logits=_t([100.0, -100.0], device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([1.0, 1.0], device))), [0.0, -100.0], atol=1e-5
        )
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 0.0], device))), [-100.0, 0.0], atol=1e-5
        )
        np.testing.assert_allclose(_np(d.entropy()), [0.0, 0.0], atol=1e-5)

    def test_bernoulli_degenerate_probs(self, device: str) -> None:
        """``probs=0`` gave an infinite logit and ``0 · (-inf) = NaN``."""
        zero = D.Bernoulli(probs=_t([0.0], device))
        np.testing.assert_allclose(
            _np(zero.log_prob(_t([0.0, 1.0], device))), [0.0, -15.94238472], atol=1e-5
        )
        np.testing.assert_allclose(_np(zero.entropy()), [0.0], atol=1e-6)
        one = D.Bernoulli(probs=_t([1.0], device))
        np.testing.assert_allclose(
            _np(one.log_prob(_t([1.0, 0.0], device))), [0.0, -15.94238472], atol=1e-5
        )

    def test_bernoulli_gradient_at_large_logits(self, device: str) -> None:
        logits = _t([100.0, -100.0, 0.0], device)
        logits.requires_grad_(True)
        d = D.Bernoulli(logits=logits)
        d.log_prob(_t([1.0, 1.0, 1.0], device)).sum().backward()
        grad = _np(logits.grad)
        # d/dl [l − softplus(l)] = 1 − σ(l).
        np.testing.assert_allclose(grad, [0.0, 1.0, 0.5], atol=1e-5)

    def test_gamma_family_at_zero(self, device: str) -> None:
        """``(α − 1) log x`` at ``α = 1, x = 0`` was NaN."""
        gamma = D.Gamma(_t(1.0, device), _t(1.0, device))
        np.testing.assert_allclose(_np(gamma.log_prob(_t(0.0, device))), 0.0, atol=1e-6)
        gamma = D.Gamma(_t(1.0, device), _t(2.5, device))
        np.testing.assert_allclose(
            _np(gamma.log_prob(_t([0.0, 1.0], device))),
            [0.91629076, -1.58370924],
            atol=1e-5,
        )
        chi2 = D.Chi2(_t(2.0, device))
        np.testing.assert_allclose(_np(chi2.log_prob(_t(0.0, device))), -0.69314718, atol=1e-5)

    def test_beta_and_dirichlet_at_the_boundary(self, device: str) -> None:
        flat = D.Beta(_t(1.0, device), _t(1.0, device))
        np.testing.assert_allclose(
            _np(flat.log_prob(_t([0.0, 1.0], device))), [0.0, 0.0], atol=1e-6
        )
        beta = D.Beta(_t(1.0, device), _t(3.0, device))
        np.testing.assert_allclose(
            _np(beta.log_prob(_t([0.0, 0.5], device))),
            [1.09861231, -0.28768206],
            atol=1e-5,
        )
        # Dirichlet(1, 1, 2) on a face of the simplex:
        # log Γ(4) − log Γ(2) + (2 − 1) log 0.5 = log 6 − log 2.
        d = D.Dirichlet(_t([1.0, 1.0, 2.0], device))
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 0.5, 0.5], device))), np.log(3.0), atol=1e-5
        )

    def test_gradients_stay_finite_at_the_boundary(self, device: str) -> None:
        """The guard swaps the operand, so no ``0 · inf`` reaches backward."""
        a = _t(1.0, device)
        b = _t(1.0, device)
        a.requires_grad_(True)
        b.requires_grad_(True)
        D.Beta(a, b).log_prob(_t([0.0, 1.0], device)).sum().backward()
        assert np.isfinite([_np(a.grad), _np(b.grad)]).all()

        alpha = _t(1.0, device)
        alpha.requires_grad_(True)
        D.Gamma(alpha, _t(1.0, device)).log_prob(_t([0.0, 2.0], device)).sum().backward()
        # Σ over both values of d/dα = log β + log x − ψ(α), at α = β = 1:
        # x = 0 contributes only −ψ(1) (its log x term is held at 0), x = 2
        # contributes log 2 − ψ(1).
        euler = 0.5772156649015329
        assert float(_np(alpha.grad).sum()) == pytest.approx(np.log(2.0) + 2 * euler, abs=1e-4)

    def test_poisson_geometric_negative_binomial_at_zero(self, device: str) -> None:
        """Built with default validation: these domains are the reference's
        (CHA-236), so the boundary is reachable without opting out."""
        poisson = D.Poisson(_t(0.0, device))
        np.testing.assert_allclose(
            _np(poisson.log_prob(_t([0.0, 1.0], device))), [0.0, -_INF], atol=2e-6
        )
        geom = D.Geometric(_t(1.0, device))
        np.testing.assert_allclose(_np(geom.log_prob(_t(0.0, device))), 0.0, atol=1e-6)
        np.testing.assert_allclose(_np(geom.entropy()), 0.0, atol=1e-6)
        nb = D.NegativeBinomial(_t(3.0, device), probs=_t(0.0, device))
        np.testing.assert_allclose(_np(nb.log_prob(_t(0.0, device))), 0.0, atol=2e-6)

    def test_negative_binomial_large_logits(self, device: str) -> None:
        """Through ``probs`` the answer was ``-inf`` for both."""
        nb = D.NegativeBinomial(_t(3.0, device), logits=_t([-100.0, 100.0], device))
        np.testing.assert_allclose(
            _np(nb.log_prob(_t(1.0, device))), [-98.90139008, -298.90139771], atol=1e-4
        )
        nb = D.NegativeBinomial(_t(3.0, device), probs=_t(0.4, device))
        np.testing.assert_allclose(
            _np(nb.log_prob(_t([0.0, 2.0, 7.0], device))),
            [-1.5324769, -1.57329893, -4.36299324],
            atol=1e-5,
        )

    def test_fisher_snedecor_at_zero(self, device: str) -> None:
        """``scipy.stats.f.logpdf([0, 1], 2, 5)``.  The reference itself gives
        NaN at 0; the density there is exactly 1.  The support is ``x > 0``,
        so validation refuses 0; the limit is read with it off."""
        d = D.FisherSnedecor(_t(2.0, device), _t(5.0, device), validate_args=False)
        np.testing.assert_allclose(
            _np(d.log_prob(_t([0.0, 1.0], device))), [0.0, -1.17765283], atol=1e-5
        )

    def test_categorical_entropy_with_zero_probability(self, device: str) -> None:
        log2 = np.log(2.0)
        cat = D.Categorical(probs=_t([0.0, 0.5, 0.5], device))
        np.testing.assert_allclose(_np(cat.entropy()), log2, atol=1e-6)
        ohc = D.OneHotCategorical(probs=_t([0.0, 0.5, 0.5], device))
        np.testing.assert_allclose(_np(ohc.entropy()), log2, atol=1e-6)
        np.testing.assert_allclose(
            _np(ohc.log_prob(_t([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device))),
            [-log2, -_INF],
            atol=1e-6,
        )

    def test_masked_logits(self, device: str) -> None:
        """The action-mask idiom: ``real`` refused ``-inf``, so this raised."""
        logits = _t([-_INF, 0.0, 1.0], device)
        logits.requires_grad_(True)
        d = D.Categorical(logits=logits)
        index = lucid.tensor([1, 0], dtype=lucid.int64, device=device)
        np.testing.assert_allclose(_np(d.log_prob(index)), [-np.log1p(np.e), -_INF], atol=1e-5)
        entropy = d.entropy()
        p = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
        np.testing.assert_allclose(_np(entropy), -(p * np.log(p)).sum(), atol=1e-6)
        entropy.backward()
        grad = _np(logits.grad)
        assert np.isfinite(grad).all(), grad
        assert grad[0] == 0.0

    def test_real_constraint_rejects_only_nan(self, device: str) -> None:
        check = D.constraints.real.check(_t([-_INF, _INF, 0.0, np.nan], device))
        assert [bool(v) for v in check.numpy()] == [True, True, True, False]
        with pytest.raises(ValueError):
            D.Categorical(logits=_t([np.nan, 0.0], device))

    def test_categorical_kl_with_zero_probability(self, device: str) -> None:
        p = D.Categorical(probs=_t([0.0, 0.5, 0.5], device))
        q = D.Categorical(probs=_t([0.2, 0.3, 0.5], device))
        np.testing.assert_allclose(_np(D.kl_divergence(p, q)), 0.25541282, atol=1e-6)
        assert np.isposinf(_np(D.kl_divergence(q, p)))
        masked = _t([-_INF, 0.0, 1.0], device)
        masked.requires_grad_(True)
        D.kl_divergence(D.Categorical(logits=masked), q).backward()
        assert np.isfinite(_np(masked.grad)).all(), _np(masked.grad)

    def test_multinomial_with_zero_probability(self, device: str) -> None:
        by_probs = D.Multinomial(5, probs=_t([0.0, 0.5, 0.5], device))
        by_logits = D.Multinomial(5, logits=_t([-_INF, 0.0, 0.0], device))
        counts = _t([0.0, 2.0, 3.0], device)
        np.testing.assert_allclose(_np(by_probs.log_prob(counts)), -1.16315079, atol=1e-5)
        np.testing.assert_allclose(_np(by_logits.log_prob(counts)), -1.16315079, atol=1e-5)

    def test_continuous_bernoulli_large_logits(self, device: str) -> None:
        """``log(1 + exp(l))`` and a normaliser computed through ``λ`` gave
        ``inf``.  The exact value is ``0.3 l − softplus(l) + log|l|``; the
        reference holds ``λ`` one epsilon inside ``[0, 1]`` and answers
        ``[-67.23, -27.23]`` instead."""
        d = D.ContinuousBernoulli(logits=_t([100.0, -100.0], device))
        want = [0.3 * 100 - 100 + np.log(100.0), -0.3 * 100 + np.log(100.0)]
        np.testing.assert_allclose(_np(d.log_prob(_t(0.3, device))), want, atol=1e-4)

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731

        def same(lucid_value: lucid.Tensor, ref_value: Any) -> None:
            np.testing.assert_allclose(_np(lucid_value), _ref_np(ref_value), atol=1e-5, rtol=1e-5)

        for logits in ([100.0, -100.0, 3.0], [0.0, 20.0, -40.0]):
            for v in ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
                same(
                    D.Bernoulli(logits=_t(logits, device)).log_prob(_t(v, device)),
                    RD.Bernoulli(logits=rt(logits)).log_prob(rt(v)),
                )
            same(
                D.Bernoulli(logits=_t(logits, device)).entropy(),
                RD.Bernoulli(logits=rt(logits)).entropy(),
            )
        probs, v = [0.0, 1.0, 0.3], [0.0, 1.0, 1.0]
        same(
            D.Bernoulli(probs=_t(probs, device)).log_prob(_t(v, device)),
            RD.Bernoulli(probs=rt(probs)).log_prob(rt(v)),
        )
        same(
            D.Bernoulli(probs=_t(probs, device)).entropy(),
            RD.Bernoulli(probs=rt(probs)).entropy(),
        )
        same(
            D.Gamma(_t([1.0, 2.0], device), _t([2.5, 1.0], device)).log_prob(
                _t([0.0, 0.0], device)
            ),
            RD.Gamma(rt([1.0, 2.0]), rt([2.5, 1.0])).log_prob(rt([0.0, 0.0])),
        )
        same(
            D.Beta(_t([1.0, 1.0], device), _t([1.0, 3.0], device)).log_prob(_t([0.0, 1.0], device)),
            RD.Beta(rt([1.0, 1.0]), rt([1.0, 3.0])).log_prob(rt([0.0, 1.0])),
        )
        same(
            D.Poisson(_t([0.0, 2.0], device)).log_prob(_t([0.0, 3.0], device)),
            RD.Poisson(rt([0.0, 2.0])).log_prob(rt([0.0, 3.0])),
        )
        same(
            D.NegativeBinomial(_t(3.0, device), logits=_t([-100.0, 100.0, 0.5], device)).log_prob(
                _t(2.0, device)
            ),
            RD.NegativeBinomial(rt(3.0), logits=rt([-100.0, 100.0, 0.5])).log_prob(rt(2.0)),
        )
        same(
            D.FisherSnedecor(_t(7.0, device), _t(3.0, device)).log_prob(
                _t([0.2, 1.0, 3.5], device)
            ),
            RD.FisherSnedecor(rt(7.0), rt(3.0)).log_prob(rt([0.2, 1.0, 3.5])),
        )
        q_probs = [0.2, 0.3, 0.5]
        for kwargs in ({"probs": [0.0, 0.5, 0.5]}, {"logits": [-_INF, 0.0, 1.0]}):
            lucid_kw = {k: _t(v, device) for k, v in kwargs.items()}
            ref_kw = {k: rt(v) for k, v in kwargs.items()}
            same(
                D.Categorical(**lucid_kw).entropy(),
                RD.Categorical(**ref_kw).entropy(),
            )
            same(
                D.OneHotCategorical(**lucid_kw).entropy(),
                RD.OneHotCategorical(**ref_kw).entropy(),
            )
            same(
                D.kl_divergence(
                    D.Categorical(**lucid_kw),
                    D.Categorical(probs=_t(q_probs, device)),
                ),
                RD.kl_divergence(RD.Categorical(**ref_kw), RD.Categorical(probs=rt(q_probs))),
            )
        same(
            D.ContinuousBernoulli(logits=_t([0.5, -3.0, 8.0], device)).log_prob(_t(0.3, device)),
            RD.ContinuousBernoulli(logits=rt([0.5, -3.0, 8.0])).log_prob(rt(0.3)),
        )


# ── CHA-236: the reference's domains, and the closed forms at their edges ────


def _softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


def _flags(t: lucid.Tensor) -> list[bool]:
    return [bool(b) for b in np.atleast_1d(t.numpy())]


@pytest.mark.parametrize("device", DEVICES)
class TestReferenceDomains:
    """CHA-236 — every parameter domain is the reference's under default
    validation, and the closed forms hold at its edges.

    The domain comparison itself, class by class, is
    ``lucid/test/parity/test_distribution_constraints_parity.py``; the
    Python-number device sweep that replaced CHA-148's five cases is
    ``test_distributions_metal_scalar_params.py``.
    """

    def test_widened_domains_construct(self, device: str) -> None:
        D.Poisson(_t([0.0, 2.0], device))
        D.Geometric(_t([1.0, 0.5], device))
        D.NegativeBinomial(_t([0.0, 3.0], device), probs=_t([0.5, 0.0], device))
        D.RelaxedBernoulli(_t(0.5, device), probs=_t([0.0, 1.0], device))
        for refused in (
            lambda: D.Poisson(_t(-1.0, device)),
            lambda: D.Geometric(_t(0.0, device)),
            lambda: D.NegativeBinomial(_t(3.0, device), probs=_t(1.0, device)),
            lambda: D.NegativeBinomial(_t(-1.0, device), probs=_t(0.5, device)),
            lambda: D.RelaxedBernoulli(_t(0.5, device), probs=_t(1.5, device)),
        ):
            with pytest.raises(ValueError):
                refused()

    def test_parameters_the_reference_refuses(self, device: str) -> None:
        for refused in (
            lambda: D.Uniform(_t(2.0, device), _t(1.0, device)),
            lambda: D.ContinuousBernoulli(probs=_t(1.5, device)),
            lambda: D.ContinuousBernoulli(logits=_t(np.nan, device)),
            lambda: D.Categorical(logits=_t([-_INF, -_INF, -_INF], device)),
            lambda: D.Categorical(logits=_t([_INF, 0.0, 0.0], device)),
            lambda: D.Binomial(_t(_INF, device), probs=_t(0.5, device)),
        ):
            with pytest.raises(ValueError):
                refused()
        # The stored logits are log-probabilities, as the reference stores them.
        logits = D.Categorical(logits=_t([-_INF, 0.0, 1.0], device)).logits
        np.testing.assert_allclose(_np(logits.exp()).sum(), 1.0, atol=1e-6)

    def test_bernoulli_kl_at_the_boundary(self, device: str) -> None:
        """A 1e-7 clip made ``KL(Bernoulli(0.3) || Bernoulli(0))`` 4.22."""
        p = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0]
        q = [0.0, 0.3, 1.0, 0.0, 0.3, 1.0, 0.0, 0.3, 1.0]
        kl = D.kl_divergence(D.Bernoulli(probs=_t(p, device)), D.Bernoulli(probs=_t(q, device)))
        want = [0.0, -np.log(0.7), _INF, _INF, 0.0, _INF, _INF, -np.log(0.3), 0.0]
        np.testing.assert_allclose(_np(kl), want, atol=1e-6)

    def test_bernoulli_kl_from_large_logits(self, device: str) -> None:
        """``1 - sigmoid(15)`` keeps one digit in float32; ``sigmoid(-15)``
        keeps all of them."""
        kl = D.kl_divergence(
            D.Bernoulli(logits=_t(0.0, device)), D.Bernoulli(logits=_t(15.0, device))
        )
        want = np.log(0.5) + 0.5 * (_softplus(-15.0) + _softplus(15.0))
        np.testing.assert_allclose(_np(kl), want, atol=1e-5)

    def test_bernoulli_kl_gradient_is_finite_at_zero(self, device: str) -> None:
        probs = _t([0.0, 0.3], device)
        probs.requires_grad_(True)
        D.kl_divergence(
            D.Bernoulli(probs=probs), D.Bernoulli(probs=_t([0.4, 0.4], device))
        ).sum().backward()
        assert np.isfinite(_np(probs.grad)).all()

    def test_poisson_and_geometric_kl_at_the_boundary(self, device: str) -> None:
        """At rate 0 the reference answers NaN, and ``inf`` for
        ``Geometric(1)`` against itself.  These are the exact divergences."""
        poisson = D.kl_divergence(
            D.Poisson(_t([0.0, 0.0, 1.5], device)),
            D.Poisson(_t([1.5, 0.0, 0.0], device)),
        )
        np.testing.assert_allclose(_np(poisson), [1.5, 0.0, _INF], atol=1e-6)
        geometric = D.kl_divergence(
            D.Geometric(_t([1.0, 1.0, 0.3], device)),
            D.Geometric(_t([0.3, 1.0, 1.0], device)),
        )
        np.testing.assert_allclose(_np(geometric), [-np.log(0.3), 0.0, _INF], atol=1e-6)

    def test_negative_binomial_with_no_successes_to_wait_for(self, device: str) -> None:
        """``r = 0`` is the point mass at 0: ``lgamma(0) - lgamma(0)`` was NaN."""
        r = _t(0.0, device)
        r.requires_grad_(True)
        nb = D.NegativeBinomial(r, probs=_t(0.5, device))
        np.testing.assert_allclose(
            _np(nb.log_prob(_t([0.0, 1.0, 2.0], device))),
            [0.0, -_INF, -_INF],
            atol=1e-6,
        )
        np.testing.assert_allclose(_np(nb.mean), 0.0)
        np.testing.assert_allclose(_np(nb.variance), 0.0)
        nb.log_prob(_t(0.0, device)).backward()
        assert np.isfinite(_np(r.grad)).all()

    def test_relaxed_bernoulli_at_degenerate_probs_and_the_ends(self, device: str) -> None:
        """An unclamped probability of 0 or 1 gave an infinite logit, and the
        ends of the support ``log 0``: NaN either way."""
        d = D.RelaxedBernoulli(_t(0.5, device), probs=_t([0.0, 1.0, 0.3], device))
        for v in (0.0, 0.3, 1.0):
            assert np.isfinite(_np(d.log_prob(_t(v, device)))).all(), v

    def test_kumaraswamy_at_the_ends(self, device: str) -> None:
        """The density's limits, by shape; ``(a, b) = (1, 1)`` is uniform."""
        d = D.Kumaraswamy(_t([1.0, 0.5, 2.0, 2.0], device), _t([1.0, 2.0, 0.5, 3.0], device))
        np.testing.assert_allclose(_np(d.log_prob(_t(0.0, device))), [0.0, _INF, -_INF, -_INF])
        np.testing.assert_allclose(_np(d.log_prob(_t(1.0, device))), [0.0, -_INF, _INF, -_INF])

    def test_supports_have_the_reference_edges(self, device: str) -> None:
        # ``Exponential.log_prob`` validates its value; ``positive`` refused 0.
        np.testing.assert_allclose(
            _np(D.Exponential(_t(2.0, device)).log_prob(_t(0.0, device))), np.log(2.0)
        )

        def admits(dist: Any, v: Any) -> list[bool]:
            return _flags(dist.support.check(_t(v, device)))

        gamma = D.Gamma(_t(2.0, device), _t(1.0, device))
        assert admits(gamma, [0.0, -1.0]) == [True, False]
        kumaraswamy = D.Kumaraswamy(_t(2.0, device), _t(3.0, device))
        assert admits(kumaraswamy, [0.0, 1.0]) == [True, True]
        weibull = D.Weibull(_t(1.0, device), _t(2.0, device))
        assert admits(weibull, [0.0, 1.0]) == [False, True]
        binomial = D.Binomial(_t([2.0, 5.0], device), probs=_t(0.5, device))
        assert admits(binomial, [3.0, 3.0]) == [False, True]
        pareto = D.Pareto(_t([1.0, 2.0], device), _t(3.0, device))
        assert admits(pareto, [1.5, 1.5]) == [True, False]
        uniform = D.Uniform(_t(0.0, device), _t(2.0, device))
        assert admits(uniform, [-0.1, 2.0]) == [False, True]
        one_hot = D.OneHotCategorical(probs=_t([0.2, 0.3, 0.5], device))
        assert admits(one_hot, [[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]]) == [True, False]
        lkj = D.LKJCholesky(3, _t(1.5, device))
        lucid.manual_seed(0)
        assert all(_flags(lkj.support.check(lkj.sample((4,)))))

    def test_integer_constraints_refuse_infinity(self, device: str) -> None:
        c = D.constraints
        values = _t([0.0, 2.0, _INF, 1.5, -1.0], device)
        assert _flags(c.nonnegative_integer.check(values)) == [
            True,
            True,
            False,
            False,
            False,
        ]
        assert _flags(c.integer_interval(0, 5).check(_t(_INF, device))) == [False]

    def test_positive_definite_requires_symmetry(self, device: str) -> None:
        """Cholesky reads the lower triangle only, so a Cholesky factor passed."""
        pd = D.constraints.positive_definite
        assert _flags(pd.check(_t([[1.0, 0.0], [0.6, 0.8]], device))) == [False]
        batch = _t([[[2.0, 0.5], [0.5, 1.0]], [[1.0, 2.0], [2.0, 1.0]]], device)
        assert _flags(pd.check(batch)) == [True, False]

    def test_reference_constraints_added(self, device: str) -> None:
        c = D.constraints
        half_open = c.half_open_interval(0.0, 1.0)
        assert _flags(half_open.check(_t([0.0, 0.5, 1.0], device))) == [
            True,
            True,
            False,
        ]
        assert c.one_hot.is_discrete and c.one_hot.event_dim == 1
        vectors = _t([[0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], device)
        assert _flags(c.one_hot.check(vectors)) == [True, False, False]
        assert c.corr_cholesky.event_dim == 2
        factors = _t([[[1.0, 0.0], [0.6, 0.8]], [[1.0, 0.0], [0.5, 0.5]]], device)
        assert _flags(c.corr_cholesky.check(factors)) == [True, False]
        counts = c.independent(c.nonnegative_integer, 1)
        assert counts.is_discrete and counts.event_dim == 1
        assert _flags(counts.check(_t([[0.0, 2.0], [1.0, 0.5]], device))) == [
            True,
            False,
        ]
        with pytest.raises(ValueError):
            c.independent(c.real, -1)

    @pytest.mark.parity
    def test_parity(self, device: str, ref: Any) -> None:
        RD = ref.distributions
        rt = lambda v: ref.tensor(np.asarray(v, dtype=np.float32))  # noqa: E731

        def same(lucid_value: lucid.Tensor, ref_value: Any) -> None:
            np.testing.assert_allclose(_np(lucid_value), _ref_np(ref_value), atol=1e-5, rtol=1e-5)

        p = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0]
        q = [0.0, 0.3, 1.0, 0.0, 0.3, 1.0, 0.0, 0.3, 1.0]
        same(
            D.kl_divergence(D.Bernoulli(probs=_t(p, device)), D.Bernoulli(probs=_t(q, device))),
            RD.kl_divergence(RD.Bernoulli(probs=rt(p)), RD.Bernoulli(probs=rt(q))),
        )
        # Not a ``q`` whose probability rounds to 1: the reference answers
        # ``inf`` there, from ``q.probs == 1``, where the logits say ~20.
        lp, lq = [0.0, 3.0, -20.0, 15.0], [15.0, -4.0, 10.0, 0.0]
        same(
            D.kl_divergence(D.Bernoulli(logits=_t(lp, device)), D.Bernoulli(logits=_t(lq, device))),
            RD.kl_divergence(RD.Bernoulli(logits=rt(lp)), RD.Bernoulli(logits=rt(lq))),
        )
        # Only where the reference is right: it answers NaN at rate 0 and
        # ``inf`` for KL(Geometric(1) || Geometric(1)).
        same(
            D.kl_divergence(D.Poisson(_t([1.5, 2.0], device)), D.Poisson(_t([0.0, 0.5], device))),
            RD.kl_divergence(RD.Poisson(rt([1.5, 2.0])), RD.Poisson(rt([0.0, 0.5]))),
        )
        same(
            D.kl_divergence(
                D.Geometric(_t([1.0, 0.3], device)), D.Geometric(_t([0.3, 1.0], device))
            ),
            RD.kl_divergence(RD.Geometric(rt([1.0, 0.3])), RD.Geometric(rt([0.3, 1.0]))),
        )
        counts = [0.0, 1.0, 2.0]
        for r, probs in ((0.0, 0.5), (3.0, 0.0)):
            same(
                D.NegativeBinomial(_t(r, device), probs=_t(probs, device)).log_prob(
                    _t(counts, device)
                ),
                RD.NegativeBinomial(rt(r), probs=rt(probs)).log_prob(rt(counts)),
            )
        same(
            D.Poisson(_t(0.0, device)).log_prob(_t(counts, device)),
            RD.Poisson(rt(0.0)).log_prob(rt(counts)),
        )
        same(
            D.Geometric(_t(1.0, device)).log_prob(_t(counts, device)),
            RD.Geometric(rt(1.0)).log_prob(rt(counts)),
        )
        values = [0.0, 0.3, 1.0]
        for probs in (0.0, 1.0, 0.3):
            same(
                D.RelaxedBernoulli(_t(0.5, device), probs=_t(probs, device)).log_prob(
                    _t(values, device)
                ),
                RD.RelaxedBernoulli(rt(0.5), probs=rt(probs)).log_prob(rt(values)),
            )
        same(
            D.Exponential(_t(2.0, device)).log_prob(_t(0.0, device)),
            RD.Exponential(rt(2.0)).log_prob(rt(0.0)),
        )
        logits = [-_INF, 0.0, 1.0]
        same(
            D.Categorical(logits=_t(logits, device)).logits,
            RD.Categorical(logits=rt(logits)).logits,
        )
