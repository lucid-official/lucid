"""Reference parity for deferred distribution additions:
AbsTransform, IndependentTransform, ReshapeTransform,
CorrCholeskyTransform, CumulativeDistributionTransform,
StackTransform, CatTransform, StudentT.rsample, new KL pairs.
"""

from typing import Any

import pytest

import lucid
import lucid.distributions as D
from lucid.test._helpers.compare import assert_close

# ── helpers ───────────────────────────────────────────────────────────────────


def _np(x: Any) -> Any:  # lucid.Tensor → numpy scalar for comparison
    try:
        return x.detach().numpy()
    except Exception:
        return x


# ── AbsTransform ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestAbsTransformParity:
    def test_forward(self, ref: Any) -> None:
        lt = D.AbsTransform()
        rt = ref.distributions.transforms.AbsTransform()
        x_l = lucid.tensor([-3.0, -1.0, 0.0, 2.0])
        x_r = ref.tensor([-3.0, -1.0, 0.0, 2.0])
        assert_close(lt(x_l), rt(x_r))

    def test_ladj_zero(self, ref: Any) -> None:
        # The reference framework raises NotImplementedError for AbsTransform.ladj.
        # Lucid's convention: return 0. (same convention for non-bijective transforms.)
        lt = D.AbsTransform()
        x_l = lucid.randn(5)
        y_l = lt(x_l)
        ladj = lt.log_abs_det_jacobian(x_l, y_l)
        assert lucid.allclose(ladj, lucid.zeros_like(ladj))


# ── ReshapeTransform ──────────────────────────────────────────────────────────


@pytest.mark.parity
class TestReshapeTransformParity:
    def test_forward(self, ref: Any) -> None:
        lt = D.ReshapeTransform((2, 3), (6,))
        rt = ref.distributions.transforms.ReshapeTransform((2, 3), (6,))
        import numpy as np

        data = np.random.randn(2, 3).astype("float32")
        x_l = lucid.tensor(data.tolist())
        x_r = ref.tensor(data)
        assert_close(lt(x_l), rt(x_r))

    def test_inverse(self, ref: Any) -> None:
        lt = D.ReshapeTransform((2, 3), (6,))
        rt = ref.distributions.transforms.ReshapeTransform((2, 3), (6,))
        import numpy as np

        data = np.random.randn(6).astype("float32")
        y_l = lucid.tensor(data.tolist())
        y_r = ref.tensor(data)
        assert_close(lt._inverse(y_l), rt._inverse(y_r))


# ── CumulativeDistributionTransform ──────────────────────────────────────────


@pytest.mark.parity
class TestCumulativeDistributionTransformParity:
    def test_forward_normal(self, ref: Any) -> None:
        lt = D.CumulativeDistributionTransform(D.Normal(0.0, 1.0))
        rt = ref.distributions.transforms.CumulativeDistributionTransform(
            ref.distributions.Normal(0.0, 1.0)
        )
        import numpy as np

        pts = np.array([-1.0, 0.0, 0.5, 1.0], dtype="float32")
        x_l = lucid.tensor(pts.tolist())
        x_r = ref.tensor(pts)
        assert_close(lt(x_l), rt(x_r), atol=1e-5)


# ── KL(Normal || Laplace) analytical ─────────────────────────────────────────


@pytest.mark.parity
class TestKLNormalLaplaceParity:
    def test_analytical_vs_ref(self, ref: Any) -> None:
        for mu, sigma, m, b in [(0.0, 1.0, 0.0, 1.0), (1.0, 0.5, 0.5, 2.0), (-1.0, 2.0, 0.0, 0.5)]:
            kl_l = D.kl_divergence(D.Normal(mu, sigma), D.Laplace(m, b))
            kl_r = ref.distributions.kl_divergence(
                ref.distributions.Normal(mu, sigma), ref.distributions.Laplace(m, b)
            )
            assert_close(kl_l, kl_r, atol=1e-5, rtol=1e-4)


# ── StudentT.rsample shape ────────────────────────────────────────────────────


@pytest.mark.parity
class TestStudentTRsampleParity:
    def test_rsample_shape(self, ref: Any) -> None:
        dist_l = D.StudentT(df=3.0, loc=1.0, scale=2.0)
        dist_r = ref.distributions.StudentT(df=3.0, loc=1.0, scale=2.0)
        s_l = dist_l.rsample((10,))
        s_r = dist_r.rsample((10,))
        assert tuple(s_l.shape) == tuple(s_r.shape)

    def test_has_rsample_flag(self, ref: Any) -> None:
        assert D.StudentT.has_rsample == ref.distributions.StudentT.has_rsample


# ── StackTransform ────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestStackTransformParity:
    def test_forward(self, ref: Any) -> None:
        import numpy as np

        data = np.random.randn(2, 3).astype("float32")
        x_l = lucid.tensor(data.tolist())
        x_r = ref.tensor(data)
        lt = D.StackTransform([D.ExpTransform(), D.AffineTransform(0.0, 2.0)], dim=0)
        rt = ref.distributions.transforms.StackTransform(
            [
                ref.distributions.transforms.ExpTransform(),
                ref.distributions.transforms.AffineTransform(0.0, 2.0),
            ],
            dim=0,
        )
        assert_close(lt(x_l), rt(x_r), atol=1e-5)


# ── CatTransform ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestCatTransformParity:
    def test_forward(self, ref: Any) -> None:
        import numpy as np

        data = np.random.randn(6).astype("float32")
        x_l = lucid.tensor(data.tolist())
        x_r = ref.tensor(data)
        lt = D.CatTransform([D.ExpTransform(), D.AffineTransform(1.0, 2.0)], dim=-1, lengths=[2, 4])
        rt = ref.distributions.transforms.CatTransform(
            [
                ref.distributions.transforms.ExpTransform(),
                ref.distributions.transforms.AffineTransform(1.0, 2.0),
            ],
            dim=-1,
            lengths=[2, 4],
        )
        assert_close(lt(x_l), rt(x_r), atol=1e-5)
