"""Parity tests for additional distributions, linalg.matrix_exp, and nn.utils.skip_init.

Covers:
  lucid.distributions — Gumbel, InverseGamma, Kumaraswamy, Multinomial,
                        ContinuousBernoulli
  lucid.linalg        — matrix_exp  (Padé [6/6] + scaling-and-squaring)
  lucid.nn.utils      — skip_init
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.linalg as LA
import lucid.nn as nn
import lucid.nn.utils as nnu

# ── helpers ───────────────────────────────────────────────────────────────────


def _lp(dist: Any, x: Any) -> float:
    """Scalar log_prob as Python float."""
    return float(dist.log_prob(x).item())


def _ref_lp(ref_dist: Any, x: Any) -> float:
    return float(ref_dist.log_prob(x).detach().cpu().item())


# ── Gumbel ────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestGumbelParity:
    def test_log_prob(self, ref: Any) -> None:
        loc, scale = 1.0, 2.0
        x = np.array([-2.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32)
        lucid_lp = D.Gumbel(loc, scale).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.Gumbel(loc, scale)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-5)

    def test_mean(self, ref: Any) -> None:
        loc, scale = 0.5, 1.5
        lm = float(D.Gumbel(loc, scale).mean.item())
        rm = float(ref.distributions.Gumbel(loc, scale).mean.item())
        assert abs(lm - rm) < 1e-5

    def test_variance(self, ref: Any) -> None:
        loc, scale = 0.0, 2.0
        lv = float(D.Gumbel(loc, scale).variance.item())
        rv = float(ref.distributions.Gumbel(loc, scale).variance.item())
        assert abs(lv - rv) < 1e-4

    def test_entropy(self, ref: Any) -> None:
        loc, scale = 0.0, 1.0
        le = float(D.Gumbel(loc, scale).entropy().item())
        re = float(ref.distributions.Gumbel(loc, scale).entropy().item())
        assert abs(le - re) < 1e-5

    def test_sample_shape(self, ref: Any) -> None:  # noqa: ARG002
        lucid.manual_seed(0)
        samp = D.Gumbel(0.0, 1.0).rsample((50,))
        assert tuple(samp.shape) == (50,)


# ── InverseGamma ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestInverseGammaParity:
    def test_log_prob(self, ref: Any) -> None:
        conc, rate = 3.0, 2.0
        x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
        lucid_lp = D.InverseGamma(conc, rate).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.InverseGamma(conc, rate)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-5)

    def test_mean(self, ref: Any) -> None:
        conc, rate = 4.0, 3.0
        lm = float(D.InverseGamma(conc, rate).mean.item())
        rm = float(ref.distributions.InverseGamma(conc, rate).mean.item())
        assert abs(lm - rm) < 1e-5

    def test_sample_positive(self, ref: Any) -> None:  # noqa: ARG002
        lucid.manual_seed(1)
        samp = D.InverseGamma(2.0, 1.0).sample((40,))
        assert bool((samp > 0).all().item()), "All InverseGamma samples must be positive"


# ── Kumaraswamy ───────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestKumaraswamyParity:
    def test_log_prob(self, ref: Any) -> None:
        a, b = 2.0, 3.0
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        lucid_lp = D.Kumaraswamy(a, b).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.Kumaraswamy(a, b)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-5)

    def test_mean(self, ref: Any) -> None:
        a, b = 2.0, 5.0
        lm = float(D.Kumaraswamy(a, b).mean.item())
        rm = float(ref.distributions.Kumaraswamy(a, b).mean.item())
        assert abs(lm - rm) < 1e-4

    def test_entropy(self, ref: Any) -> None:
        a, b = 3.0, 2.0
        le = float(D.Kumaraswamy(a, b).entropy().item())
        re = float(ref.distributions.Kumaraswamy(a, b).entropy().item())
        assert abs(le - re) < 1e-4

    def test_sample_in_unit_interval(self, ref: Any) -> None:  # noqa: ARG002
        lucid.manual_seed(2)
        samp = D.Kumaraswamy(2.0, 3.0).rsample((60,))
        assert bool((samp > 0).all().item()) and bool(
            (samp < 1).all().item()
        ), "All Kumaraswamy samples must be in (0, 1)"


# ── Multinomial ───────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestMultinomialParity:
    def test_log_prob_uniform(self, ref: Any) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        counts = np.array([3.0, 2.0, 1.0, 4.0], dtype=np.float32)
        lucid_lp = float(
            D.Multinomial(10, probs=lucid.tensor(probs.copy()))
            .log_prob(lucid.tensor(counts.copy()))
            .item()
        )
        ref_lp = float(
            ref.distributions.Multinomial(10, probs=ref.tensor(probs.copy()))
            .log_prob(ref.tensor(counts.copy()))
            .detach()
            .item()
        )
        assert abs(lucid_lp - ref_lp) < 1e-4

    def test_log_prob_nonuniform(self, ref: Any) -> None:
        probs = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
        counts = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
        lucid_lp = float(
            D.Multinomial(10, probs=lucid.tensor(probs.copy()))
            .log_prob(lucid.tensor(counts.copy()))
            .item()
        )
        ref_lp = float(
            ref.distributions.Multinomial(10, probs=ref.tensor(probs.copy()))
            .log_prob(ref.tensor(counts.copy()))
            .detach()
            .item()
        )
        assert abs(lucid_lp - ref_lp) < 1e-4

    def test_mean(self, ref: Any) -> None:
        probs = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        lm = D.Multinomial(5, probs=lucid.tensor(probs.copy())).mean.numpy()
        rm = (
            ref.distributions.Multinomial(5, probs=ref.tensor(probs.copy()))
            .mean.detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lm, rm, atol=1e-5)

    def test_sample_counts_sum(self, ref: Any) -> None:  # noqa: ARG002
        lucid.manual_seed(3)
        probs = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        n = 10
        samp = D.Multinomial(n, probs=lucid.tensor(probs)).sample()
        total = int(samp.to(lucid.float32).sum().item())
        assert total == n, "Multinomial sample counts must sum to total_count"


# ── ContinuousBernoulli ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestContinuousBernoulliParity:
    def test_log_prob_probs(self, ref: Any) -> None:
        p = 0.3
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        lucid_lp = D.ContinuousBernoulli(probs=p).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.ContinuousBernoulli(probs=p)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-4)

    def test_log_prob_far_from_half(self, ref: Any) -> None:
        p = 0.8
        x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        lucid_lp = D.ContinuousBernoulli(probs=p).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.ContinuousBernoulli(probs=p)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-4)

    def test_log_prob_near_half(self, ref: Any) -> None:
        p = 0.5
        x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        lucid_lp = D.ContinuousBernoulli(probs=p).log_prob(lucid.tensor(x.copy())).numpy()
        ref_lp = (
            ref.distributions.ContinuousBernoulli(probs=p)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(lucid_lp, ref_lp, atol=1e-4)

    def test_mean(self, ref: Any) -> None:
        p = 0.7
        lm = float(D.ContinuousBernoulli(probs=p).mean.item())
        rm = float(ref.distributions.ContinuousBernoulli(probs=p).mean.item())
        assert abs(lm - rm) < 1e-4

    def test_sample_in_unit_interval(self, ref: Any) -> None:  # noqa: ARG002
        lucid.manual_seed(4)
        samp = D.ContinuousBernoulli(probs=0.6).rsample((50,))
        assert bool((samp >= 0).all().item()) and bool(
            (samp <= 1).all().item()
        ), "All ContinuousBernoulli samples must be in [0, 1]"


# ── linalg.matrix_exp ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestMatrixExpParity:
    def test_diagonal_matrix(self, ref: Any) -> None:
        """exp of a diagonal matrix D = diag(d) should be diag(exp(d))."""
        d = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        A = np.diag(d)
        lucid_out = LA.matrix_exp(lucid.tensor(A.copy())).numpy()
        ref_out = ref.linalg.matrix_exp(ref.tensor(A.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(lucid_out, ref_out, atol=1e-4)

    def test_skew_symmetric(self, ref: Any) -> None:
        """exp of a skew-symmetric matrix should be orthogonal."""
        A = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -3.0], [-2.0, 3.0, 0.0]], dtype=np.float32)
        lucid_out = LA.matrix_exp(lucid.tensor(A.copy())).numpy()
        ref_out = ref.linalg.matrix_exp(ref.tensor(A.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(lucid_out, ref_out, atol=1e-3)

    def test_random_matrix(self, ref: Any) -> None:
        np.random.seed(10)
        # Small entries to stay within the Padé approximant's accuracy region.
        A = (np.random.standard_normal((4, 4)) * 0.5).astype(np.float32)
        lucid_out = LA.matrix_exp(lucid.tensor(A.copy())).numpy()
        ref_out = ref.linalg.matrix_exp(ref.tensor(A.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(lucid_out, ref_out, atol=1e-3)

    def test_identity_exp_is_e_times_I(self, ref: Any) -> None:  # noqa: ARG002
        """exp(I) = e · I."""
        import math

        A = np.eye(3, dtype=np.float32)
        out = LA.matrix_exp(lucid.tensor(A)).numpy()
        expected = np.eye(3, dtype=np.float32) * math.e
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_zero_matrix_is_identity(self, ref: Any) -> None:  # noqa: ARG002
        """exp(0) = I."""
        A = np.zeros((3, 3), dtype=np.float32)
        out = LA.matrix_exp(lucid.tensor(A)).numpy()
        np.testing.assert_allclose(out, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_large_norm_matrix(self, ref: Any) -> None:
        """Scaling-and-squaring path: matrix with norm > theta_6."""
        np.random.seed(11)
        A = (np.random.standard_normal((3, 3)) * 3.0).astype(np.float32)
        lucid_out = LA.matrix_exp(lucid.tensor(A.copy())).numpy()
        ref_out = ref.linalg.matrix_exp(ref.tensor(A.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(lucid_out, ref_out, atol=1e-2)


# ── nn.utils.skip_init ────────────────────────────────────────────────────────


@pytest.mark.parity
class TestSkipInitParity:
    def test_returns_module_instance(self, ref: Any) -> None:  # noqa: ARG002
        module = nnu.skip_init(nn.Linear, 8, 4)
        assert isinstance(module, nn.Linear)

    def test_correct_shape(self, ref: Any) -> None:  # noqa: ARG002
        module = nnu.skip_init(nn.Linear, 16, 8)
        assert tuple(module.weight.shape) == (8, 16)
        assert tuple(module.bias.shape) == (8,)

    def test_parameters_are_uninitialized(self, ref: Any) -> None:  # noqa: ARG002
        """skip_init must return a module where parameters have the right
        shape and dtype but their values may be arbitrary (empty memory)."""
        module = nnu.skip_init(nn.Linear, 4, 2)
        # Only verify that the parameter tensor exists and has finite dtype.
        assert module.weight.dtype == lucid.float32
        assert tuple(module.weight.shape) == (2, 4)

    def test_conv2d_skip_init(self, ref: Any) -> None:  # noqa: ARG002
        module = nnu.skip_init(nn.Conv2d, 3, 8, 3)
        assert tuple(module.weight.shape) == (8, 3, 3, 3)

    def test_parity_structure_matches_ref(self, ref: Any) -> None:
        """skip_init result must have the same parameter names as the
        reference framework's equivalent."""
        lucid_module = nnu.skip_init(nn.Linear, 6, 3)
        ref_module = ref.nn.utils.skip_init(ref.nn.Linear, 6, 3)
        lucid_names = sorted(n for n, _ in lucid_module.named_parameters())
        ref_names = sorted(ref_module.state_dict().keys())
        assert lucid_names == ref_names
