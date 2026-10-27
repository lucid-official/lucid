"""Unit tests for deferred distribution additions:
• New transforms: AbsTransform, IndependentTransform, ReshapeTransform,
  CorrCholeskyTransform, CumulativeDistributionTransform, StackTransform,
  CatTransform
• New KL pairs: StudentT-StudentT, Cauchy-Cauchy, Normal-Laplace
• StudentT.rsample
"""

import math

import pytest

import lucid
import lucid.distributions as D
from lucid._C import engine as _C_engine

# ── AbsTransform ──────────────────────────────────────────────────────────────


class TestAbsTransform:
    def test_forward(self) -> None:
        t = D.AbsTransform()
        x = lucid.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
        y = t(x)
        expected = lucid.tensor([3.0, 1.0, 0.0, 2.0, 5.0])
        assert lucid.allclose(y, expected)

    def test_inverse_identity(self) -> None:
        t = D.AbsTransform()
        y = lucid.tensor([1.0, 2.0, 3.0])
        assert lucid.allclose(t._inverse(y), y)

    def test_ladj_zero(self) -> None:
        t = D.AbsTransform()
        x = lucid.randn(5)
        y = t(x)
        ladj = t.log_abs_det_jacobian(x, y)
        assert lucid.allclose(ladj, lucid.zeros_like(ladj))

    def test_not_bijective(self) -> None:
        assert not D.AbsTransform.bijective


# ── ReshapeTransform ──────────────────────────────────────────────────────────


class TestReshapeTransform:
    def test_forward_inverse(self) -> None:
        t = D.ReshapeTransform((2, 3), (6,))
        x = lucid.randn(2, 3)
        y = t(x)
        assert y.shape == (6,)
        x_back = t._inverse(y)
        assert x_back.shape == (2, 3)
        assert lucid.allclose(x, x_back)

    def test_batched(self) -> None:
        t = D.ReshapeTransform((3,), (1, 3))
        x = lucid.randn(4, 3)
        y = t(x)
        assert y.shape == (4, 1, 3)

    def test_ladj_zero(self) -> None:
        t = D.ReshapeTransform((2, 3), (6,))
        x = lucid.randn(2, 3)
        y = t(x)
        ladj = t.log_abs_det_jacobian(x, y)
        assert ladj.shape == ()  # scalar batch

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="elements"):
            D.ReshapeTransform((2, 3), (5,))


# ── IndependentTransform ──────────────────────────────────────────────────────


class TestIndependentTransform:
    def test_forward_inverse(self) -> None:
        t = D.IndependentTransform(D.ExpTransform(), reinterpreted_batch_ndims=1)
        x = lucid.randn(4)
        y = t(x)
        x_back = t._inverse(y)
        assert lucid.allclose(x, x_back, atol=1e-5)

    def test_ladj_sums_batch(self) -> None:
        inner = D.ExpTransform()
        t = D.IndependentTransform(inner, reinterpreted_batch_ndims=1)
        x = lucid.randn(3)
        y = t(x)
        ladj = t.log_abs_det_jacobian(x, y)
        # IndependentTransform should sum the inner ladj over the reinterpreted dim.
        inner_ladj = inner.log_abs_det_jacobian(x, y)
        assert lucid.allclose(ladj, inner_ladj.sum(dim=-1))

    def test_event_dim(self) -> None:
        t = D.IndependentTransform(D.ExpTransform(), reinterpreted_batch_ndims=2)
        assert t.event_dim == 2  # ExpTransform.event_dim=0 + 2


# ── CorrCholeskyTransform ─────────────────────────────────────────────────────


class TestCorrCholeskyTransform:
    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_produces_corr_cholesky(self, d: int) -> None:
        n_free = d * (d - 1) // 2
        t = D.CorrCholeskyTransform(d)
        x = lucid.randn(n_free)
        L = t(x)
        assert L.shape == (d, d)
        # First diagonal must be 1
        diag = L.diagonal(dim1=-2, dim2=-1)
        assert abs(float(diag[0].item()) - 1.0) < 1e-5

    def test_invalid_dim(self) -> None:
        with pytest.raises(ValueError, match="dim must be ≥ 2"):
            D.CorrCholeskyTransform(1)

    def test_inverse_roundtrip(self) -> None:
        t = D.CorrCholeskyTransform(3)
        x = lucid.randn(3) * 0.5  # small values for stable tanh inversion
        L = t(x)
        x_back = t._inverse(L)
        assert lucid.allclose(x, x_back, atol=1e-4)

    def test_upper_triangle_zero(self) -> None:
        t = D.CorrCholeskyTransform(3)
        L = t(lucid.randn(3))
        upper = L.triu(1)
        assert lucid.allclose(upper, lucid.zeros_like(upper), atol=1e-6)


# ── CumulativeDistributionTransform ──────────────────────────────────────────


class TestCumulativeDistributionTransform:
    def test_standard_normal_cdf(self) -> None:
        t = D.CumulativeDistributionTransform(D.Normal(0.0, 1.0))
        x = lucid.tensor([0.0])
        y = t(x)
        # CDF of N(0,1) at 0 = 0.5
        assert abs(float(y.item()) - 0.5) < 1e-5

    def test_inverse_icdf(self) -> None:
        t = D.CumulativeDistributionTransform(D.Normal(0.0, 1.0))
        x = lucid.tensor([-1.0, 0.0, 1.0])
        y = t(x)
        x_back = t._inverse(y)
        assert lucid.allclose(x, x_back, atol=1e-5)

    def test_ladj_equals_log_prob(self) -> None:
        dist = D.Normal(0.0, 1.0)
        t = D.CumulativeDistributionTransform(dist)
        x = lucid.tensor([0.5])
        y = t(x)
        assert lucid.allclose(t.log_abs_det_jacobian(x, y), dist.log_prob(x))


# ── StackTransform ────────────────────────────────────────────────────────────


class TestStackTransform:
    def test_forward_inverse(self) -> None:
        t = D.StackTransform([D.ExpTransform(), D.AffineTransform(0.0, 2.0)], dim=0)
        x = lucid.randn(2, 3)
        y = t(x)
        x_back = t._inverse(y)
        assert lucid.allclose(x, x_back, atol=1e-5)

    def test_wrong_slice_count(self) -> None:
        t = D.StackTransform([D.ExpTransform()], dim=0)
        x = lucid.randn(2, 3)
        with pytest.raises(ValueError, match="slices"):
            t(x)


# ── CatTransform ──────────────────────────────────────────────────────────────


class TestCatTransform:
    def test_forward_inverse(self) -> None:
        t = D.CatTransform([D.ExpTransform(), D.AffineTransform(1.0, 2.0)], dim=-1, lengths=[3, 4])
        x = lucid.randn(7)
        y = t(x)
        assert y.shape == (7,)
        x_back = t._inverse(y)
        assert lucid.allclose(x, x_back, atol=1e-5)

    def test_default_lengths_are_one_per_transform(self) -> None:
        """The reference's default (LCD-495), not equal parts of the axis."""
        t = D.CatTransform([D.ExpTransform(), D.ExpTransform()], dim=-1)
        assert t.lengths == [1, 1]
        assert t(lucid.randn(2)).shape == (2,)
        with pytest.raises(_C_engine.InvalidArgument, match="sum exactly to 6"):
            t(lucid.randn(6))


# ── StudentT rsample ──────────────────────────────────────────────────────────


class TestStudentTRsample:
    def test_rsample_shape(self) -> None:
        dist = D.StudentT(df=3.0)
        s = dist.rsample((5,))
        assert s.shape == (5,)

    def test_rsample_mean_converges(self) -> None:
        # For df=10, loc=2, the mean should be ~2.
        dist = D.StudentT(df=10.0, loc=2.0, scale=1.0)
        s = dist.rsample((2000,))
        mean_val = float(s.mean().item())
        assert abs(mean_val - 2.0) < 0.2


# ── New KL pairs ──────────────────────────────────────────────────────────────


class TestNewKLPairs:
    def test_kl_studentt_studentt(self) -> None:
        p = D.StudentT(df=3.0, loc=0.0, scale=1.0)
        q = D.StudentT(df=3.0, loc=0.0, scale=1.0)
        with pytest.raises(NotImplementedError):
            D.kl_divergence(p, q)

    def test_kl_cauchy_cauchy(self) -> None:
        p = D.Cauchy(loc=0.0, scale=1.0)
        q = D.Cauchy(loc=0.0, scale=1.0)
        kl = D.kl_divergence(p, q)
        assert abs(float(kl.item())) < 1e-6

    def test_kl_normal_laplace_finite(self) -> None:
        p = D.Normal(0.0, 1.0)
        q = D.Laplace(0.0, 1.0)
        kl = D.kl_divergence(p, q)
        val = float(kl.item())
        assert math.isfinite(val) and val >= 0.0

    def test_unregistered_raises_not_implemented(self) -> None:
        pw = D.Weibull(scale=1.0, concentration=2.0)
        qn = D.Normal(1.0, 1.0)
        with pytest.raises(NotImplementedError):
            D.kl_divergence(pw, qn)
