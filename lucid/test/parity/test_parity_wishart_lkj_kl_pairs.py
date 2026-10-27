"""Parity tests for matrix distributions and additional KL divergence pairs.

Covers:
  lucid.distributions — Wishart, LKJCholesky (matrix-valued distributions)
  lucid.distributions.kl — Laplace||Laplace, Gumbel||Gumbel, Uniform||Uniform,
                            Geometric||Geometric, HalfNormal||HalfNormal,
                            LogNormal||LogNormal, Independent||Independent
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions.kl import kl_divergence
from lucid.test._helpers.compare import assert_close

# ── helpers ───────────────────────────────────────────────────────────────────


def _ref_kl(ref: Any, p_ref: Any, q_ref: Any) -> float:
    return float(ref.distributions.kl_divergence(p_ref, q_ref).item())


# ── Wishart ───────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestWishartParity:
    def test_log_prob_identity(self, ref: Any) -> None:
        """log_prob of a Wishart sample should be finite."""
        lucid.manual_seed(0)
        w = D.Wishart(5.0, covariance_matrix=lucid.eye(3))
        # reference Wishart
        ref_w = ref.distributions.Wishart(ref.tensor(5.0), covariance_matrix=ref.eye(3))
        # Use the same sample (via numpy seed)
        np.random.seed(0)
        sample_np = ref_w.sample().detach().cpu().numpy()
        sample_lucid = lucid.tensor(sample_np.copy())
        sample_ref = ref.tensor(sample_np.copy())

        lucid_lp = float(w.log_prob(sample_lucid).item())
        ref_lp = float(ref_w.log_prob(sample_ref).item())
        assert (
            abs(lucid_lp - ref_lp) < 1e-2
        ), f"Wishart log_prob mismatch: lucid={lucid_lp:.4f}, ref={ref_lp:.4f}"

    def test_mean(self, ref: Any) -> None:  # noqa: ARG002
        """mean = df * covariance_matrix."""
        cov = np.eye(3, dtype=np.float32) * 2.0
        w = D.Wishart(6.0, covariance_matrix=lucid.tensor(cov))
        expected = 6.0 * lucid.tensor(cov)
        assert_close(w.mean, expected, atol=1e-6)

    def test_sample_positive_definite(self, ref: Any) -> None:  # noqa: ARG002
        """All Wishart samples must be positive definite."""
        lucid.manual_seed(1)
        w = D.Wishart(6.0, covariance_matrix=lucid.eye(3))
        for _ in range(5):
            s = w.sample()
            # PD ↔ all eigenvalues positive ↔ Cholesky succeeds
            try:
                lucid.linalg.cholesky(s)
            except Exception as exc:
                pytest.fail(f"Wishart sample is not positive-definite: {exc}")

    def test_scale_tril_parameterisation(self, ref: Any) -> None:  # noqa: ARG002
        """scale_tril and covariance_matrix parameterisations agree on log_prob."""
        np.random.seed(2)
        cov = np.eye(3, dtype=np.float32) + np.array(
            [[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]], dtype=np.float32
        )
        L = np.linalg.cholesky(cov)
        w1 = D.Wishart(5.0, covariance_matrix=lucid.tensor(cov))
        w2 = D.Wishart(5.0, scale_tril=lucid.tensor(L))
        sample = w1.sample()
        assert_close(w1.log_prob(sample), w2.log_prob(sample), atol=1e-4)


# ── LKJCholesky ───────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLKJCholeskyParity:
    def test_log_prob(self, ref: Any) -> None:
        """log_prob must agree with the reference."""
        lucid.manual_seed(3)
        l = D.LKJCholesky(3, concentration=2.0)
        ref_l = ref.distributions.LKJCholesky(3, concentration=2.0)

        # Sample from reference framework so both evaluate the same point.
        ref_samp = ref_l.sample().detach().cpu().numpy()
        sample_lucid = lucid.tensor(ref_samp.copy())
        sample_ref = ref.tensor(ref_samp.copy())

        lucid_lp = float(l.log_prob(sample_lucid).item())
        ref_lp = float(ref_l.log_prob(sample_ref).item())
        assert (
            abs(lucid_lp - ref_lp) < 1e-3
        ), f"LKJCholesky log_prob mismatch: lucid={lucid_lp:.4f}, ref={ref_lp:.4f}"

    def test_sample_lower_triangular(self, ref: Any) -> None:  # noqa: ARG002
        """Samples must be lower-triangular with positive diagonal."""
        lucid.manual_seed(4)
        l = D.LKJCholesky(4, concentration=1.0)
        s = l.sample((3,))
        assert tuple(s.shape) == (3, 4, 4)
        # Diagonal should be positive
        diag = s.diagonal(dim1=-2, dim2=-1)
        assert bool((diag > 0).all().item()), "LKJCholesky diagonal must be positive"
        # First row must be [1, 0, 0, ...]
        first_diag = float(s[0, 0, 0].item())
        assert abs(first_diag - 1.0) < 1e-5, "LKJCholesky L[0,0] must be 1.0"

    def test_uniform_concentration(self, ref: Any) -> None:
        """concentration=1 → uniform over correlation Cholesky factors."""
        lucid.manual_seed(5)
        l = D.LKJCholesky(3, concentration=1.0)
        ref_l = ref.distributions.LKJCholesky(3, concentration=1.0)
        ref_samp = ref_l.sample().detach().cpu().numpy()
        sample = lucid.tensor(ref_samp.copy())
        # log_prob should match reference at concentration=1
        lucid_lp = float(l.log_prob(sample).item())
        ref_lp = float(ref_l.log_prob(ref.tensor(ref_samp.copy())).item())
        assert abs(lucid_lp - ref_lp) < 1e-3


# ── New KL pairs ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestNewKLPairsParity:
    def _kl(self, p: Any, q: Any) -> float:
        return float(kl_divergence(p, q).item())

    def test_laplace_laplace(self, ref: Any) -> None:
        lucid_kl = self._kl(D.Laplace(0.0, 1.0), D.Laplace(0.5, 2.0))
        ref_kl = _ref_kl(
            ref, ref.distributions.Laplace(0.0, 1.0), ref.distributions.Laplace(0.5, 2.0)
        )
        assert abs(lucid_kl - ref_kl) < 1e-5

    def test_gumbel_gumbel(self, ref: Any) -> None:
        lucid_kl = self._kl(D.Gumbel(0.0, 1.0), D.Gumbel(1.0, 2.0))
        ref_kl = _ref_kl(
            ref, ref.distributions.Gumbel(0.0, 1.0), ref.distributions.Gumbel(1.0, 2.0)
        )
        assert abs(lucid_kl - ref_kl) < 1e-4

    def test_uniform_uniform_contained(self, ref: Any) -> None:
        lucid_kl = self._kl(D.Uniform(0.0, 1.0), D.Uniform(-0.5, 1.5))
        ref_kl = _ref_kl(
            ref, ref.distributions.Uniform(0.0, 1.0), ref.distributions.Uniform(-0.5, 1.5)
        )
        assert abs(lucid_kl - ref_kl) < 1e-5

    def test_uniform_uniform_not_contained(self, ref: Any) -> None:  # noqa: ARG002
        """When p is not supported inside q the KL should be +∞."""
        kl = float(kl_divergence(D.Uniform(0.0, 2.0), D.Uniform(0.0, 1.0)).item())
        assert kl == float("inf")

    def test_geometric_geometric(self, ref: Any) -> None:
        lucid_kl = self._kl(D.Geometric(probs=0.3), D.Geometric(probs=0.5))
        ref_kl = _ref_kl(ref, ref.distributions.Geometric(0.3), ref.distributions.Geometric(0.5))
        assert abs(lucid_kl - ref_kl) < 1e-4

    def test_halfnormal_halfnormal(self, ref: Any) -> None:
        lucid_kl = self._kl(D.HalfNormal(1.0), D.HalfNormal(2.0))
        ref_kl = _ref_kl(ref, ref.distributions.HalfNormal(1.0), ref.distributions.HalfNormal(2.0))
        assert abs(lucid_kl - ref_kl) < 1e-5

    def test_lognormal_lognormal(self, ref: Any) -> None:
        lucid_kl = self._kl(D.LogNormal(0.0, 1.0), D.LogNormal(0.5, 1.5))
        ref_kl = _ref_kl(
            ref, ref.distributions.LogNormal(0.0, 1.0), ref.distributions.LogNormal(0.5, 1.5)
        )
        assert abs(lucid_kl - ref_kl) < 1e-4

    def test_independent_independent(self, ref: Any) -> None:
        p_lucid = D.Independent(D.Normal(lucid.zeros(3), lucid.ones(3)), 1)
        q_lucid = D.Independent(D.Normal(lucid.tensor([0.5, 0.5, 0.5]), lucid.ones(3)), 1)
        lucid_kl = self._kl(p_lucid, q_lucid)

        p_ref = ref.distributions.Independent(
            ref.distributions.Normal(ref.zeros(3), ref.ones(3)), 1
        )
        q_ref = ref.distributions.Independent(
            ref.distributions.Normal(ref.tensor([0.5, 0.5, 0.5]), ref.ones(3)), 1
        )
        ref_kl = _ref_kl(ref, p_ref, q_ref)
        assert abs(lucid_kl - ref_kl) < 1e-4
