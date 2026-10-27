"""Parity tests for composite ops and nn extension functions.

Covers:
  lucid.special      — gammaln / psi / expit / modified_bessel_i0 / i1 aliases
  lucid top-level    — softmax, log_softmax
  lucid.nn.utils     — fuse_conv_bn_eval, get_total_norm
  lucid.nn.init      — non-inplace aliases (xavier_uniform, constant, normal, …)
  lucid composites   — allclose, amax, amin, argwhere, floor_divide,
                       std_mean, var_mean, logdet, diag_embed
  lucid.nn.functional — inplace activations: relu_, elu_, selu_,
                        leaky_relu_, hardtanh_, threshold_
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
import lucid.nn.init as init
import lucid.nn.utils as nnu
import lucid.special as S
from lucid.test._helpers.compare import assert_close

# ── Tier 1: lucid.special aliases ────────────────────────────────────────────


@pytest.mark.parity
class TestSpecialAliasesParity:
    def test_gammaln(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
        assert_close(
            S.gammaln(lucid.tensor(x.copy())), ref.special.gammaln(ref.tensor(x.copy())), atol=1e-5
        )

    def test_psi(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        assert_close(
            S.psi(lucid.tensor(x.copy())), ref.special.psi(ref.tensor(x.copy())), atol=1e-4
        )

    def test_expit(self, ref: Any) -> None:
        x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
        assert_close(
            S.expit(lucid.tensor(x.copy())), ref.special.expit(ref.tensor(x.copy())), atol=1e-5
        )

    def test_modified_bessel_i0(self, ref: Any) -> None:
        x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        assert_close(
            S.modified_bessel_i0(lucid.tensor(x.copy())),
            ref.special.modified_bessel_i0(ref.tensor(x.copy())),
            atol=1e-4,
        )

    def test_modified_bessel_i1(self, ref: Any) -> None:
        x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        assert_close(
            S.modified_bessel_i1(lucid.tensor(x.copy())),
            ref.special.modified_bessel_i1(ref.tensor(x.copy())),
            atol=1e-4,
        )


# ── Tier 1: top-level softmax / log_softmax ───────────────────────────────────


@pytest.mark.parity
class TestTopLevelSoftmaxParity:
    def test_softmax_dim1(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        assert_close(
            lucid.softmax(lucid.tensor(x.copy()), dim=1),
            ref.softmax(ref.tensor(x.copy()), dim=1),
            atol=1e-5,
        )

    def test_log_softmax_dim1(self, ref: Any) -> None:
        np.random.seed(1)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        assert_close(
            lucid.log_softmax(lucid.tensor(x.copy()), dim=1),
            ref.log_softmax(ref.tensor(x.copy()), dim=1),
            atol=1e-5,
        )


# ── Tier 1: nn.utils fusion top-level ────────────────────────────────────────


@pytest.mark.parity
class TestNNUtilsFusionTopLevelParity:
    def test_fuse_conv_bn_eval_accessible(self, ref: Any) -> None:
        import lucid.nn as nn

        np.random.seed(0)
        w = np.random.standard_normal((4, 2, 3, 3)).astype(np.float32)
        conv = nn.Conv2d(2, 4, 3, bias=True)
        conv.weight = nn.Parameter(lucid.tensor(w.copy()))
        bn = nn.BatchNorm2d(4)
        bn.eval()
        conv.eval()
        fused_via_utils = nnu.fuse_conv_bn_eval(conv, bn)
        fused_via_module = nnu.fusion.fuse_conv_bn_eval(conv, bn)
        x = lucid.tensor(np.random.standard_normal((1, 2, 8, 8)).astype(np.float32))
        assert_close(fused_via_utils(x), fused_via_module(x), atol=0.0)

    def test_get_total_norm(self, ref: Any) -> None:  # noqa: ARG002
        import lucid.nn as nn

        np.random.seed(7)
        linear = nn.Linear(4, 2)
        x = lucid.tensor(np.random.standard_normal((3, 4)).astype(np.float32))
        linear(x).sum().backward()
        grads = [p.grad for p in linear.parameters() if p.grad is not None]
        lucid_norm = nnu.get_total_norm(grads).item()
        # Verify manually: sqrt(sum(grad**2 for each param))
        expected = (
            float(
                sum(
                    p.grad.numpy().flatten() ** 2 for p in linear.parameters() if p.grad is not None
                ).__class__.__mro__  # trick to get the sum
            )
            if False
            else float(
                np.sqrt(
                    sum(
                        float((p.grad * p.grad).sum().item())
                        for p in linear.parameters()
                        if p.grad is not None
                    )
                )
            )
        )
        assert abs(lucid_norm - expected) < 1e-4


# ── Tier 1: nn.init aliases ───────────────────────────────────────────────────


@pytest.mark.parity
class TestInitAliasesParity:
    def test_xavier_uniform_alias(self, ref: Any) -> None:  # noqa: ARG002
        t = lucid.zeros([4, 8])
        init.xavier_uniform(t)
        assert not (t == lucid.zeros([4, 8])).all().item()

    def test_constant_alias(self, ref: Any) -> None:  # noqa: ARG002
        t = lucid.zeros([3, 3])
        init.constant(t, 7.0)
        assert_close(t, lucid.full([3, 3], 7.0), atol=0.0)

    def test_normal_alias(self, ref: Any) -> None:  # noqa: ARG002
        t = lucid.zeros([100])
        init.normal(t, mean=0.0, std=1.0)
        assert not (t == lucid.zeros([100])).all().item()


# ── Tier 2: top-level composites ─────────────────────────────────────────────


@pytest.mark.parity
class TestTier2CompositesParity:
    def test_allclose_true(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert lucid.allclose(lucid.tensor(x), lucid.tensor(x)) is True

    def test_allclose_false(self, ref: Any) -> None:
        a = np.array([1.0], dtype=np.float32)
        b = np.array([2.0], dtype=np.float32)
        assert lucid.allclose(lucid.tensor(a), lucid.tensor(b)) is False

    def test_amax(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        assert_close(
            lucid.amax(lucid.tensor(x.copy()), dim=1),
            ref.amax(ref.tensor(x.copy()), dim=1),
            atol=1e-6,
        )

    def test_amin(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        assert_close(
            lucid.amin(lucid.tensor(x.copy()), dim=1),
            ref.amin(ref.tensor(x.copy()), dim=1),
            atol=1e-6,
        )

    def test_argwhere(self, ref: Any) -> None:
        x = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)
        l = lucid.argwhere(lucid.tensor(x.copy())).numpy()
        r = ref.argwhere(ref.tensor(x.copy())).numpy()
        np.testing.assert_array_equal(l, r)

    def test_floor_divide(self, ref: Any) -> None:
        a = np.array([7.0, -7.0, 10.0], dtype=np.float32)
        b = np.array([2.0, 2.0, 3.0], dtype=np.float32)
        assert_close(
            lucid.floor_divide(lucid.tensor(a.copy()), lucid.tensor(b.copy())),
            ref.floor_divide(ref.tensor(a.copy()), ref.tensor(b.copy())),
            atol=0.0,
        )

    def test_std_mean(self, ref: Any) -> None:
        np.random.seed(2)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        l_s, l_m = lucid.std_mean(lucid.tensor(x.copy()), dim=1)
        r_s, r_m = ref.std_mean(ref.tensor(x.copy()), dim=1)
        assert_close(l_s, r_s, atol=1e-4)
        assert_close(l_m, r_m, atol=1e-5)

    def test_var_mean(self, ref: Any) -> None:
        np.random.seed(3)
        x = np.random.standard_normal((4, 5)).astype(np.float32)
        l_v, l_m = lucid.var_mean(lucid.tensor(x.copy()), dim=1)
        r_v, r_m = ref.var_mean(ref.tensor(x.copy()), dim=1)
        assert_close(l_v, r_v, atol=1e-4)
        assert_close(l_m, r_m, atol=1e-5)

    def test_logdet(self, ref: Any) -> None:
        np.random.seed(4)
        M = np.random.standard_normal((4, 4)).astype(np.float32)
        A = (M @ M.T + 4 * np.eye(4)).astype(np.float32)
        assert_close(
            lucid.logdet(lucid.tensor(A.copy())), ref.logdet(ref.tensor(A.copy())), atol=1e-3
        )

    def test_diag_embed_1d(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_close(
            lucid.diag_embed(lucid.tensor(x.copy())), ref.diag_embed(ref.tensor(x.copy())), atol=0.0
        )

    def test_diag_embed_batch(self, ref: Any) -> None:
        np.random.seed(5)
        x = np.random.standard_normal((3, 4)).astype(np.float32)
        assert_close(
            lucid.diag_embed(lucid.tensor(x.copy())), ref.diag_embed(ref.tensor(x.copy())), atol=0.0
        )

    def test_diag_embed_offset(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_close(
            lucid.diag_embed(lucid.tensor(x.copy()), offset=1),
            ref.diag_embed(ref.tensor(x.copy()), offset=1),
            atol=0.0,
        )


# ── Tier 2: F.inplace activations ────────────────────────────────────────────


@pytest.mark.parity
class TestInplaceActivationsParity:
    def _x(self) -> np.ndarray:
        return np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    def test_relu_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.relu_(t)
        ref.nn.functional.relu_(r)
        assert_close(t, r, atol=0.0)

    def test_elu_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.elu_(t)
        ref.nn.functional.elu_(r)
        assert_close(t, r, atol=1e-5)

    def test_selu_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.selu_(t)
        ref.nn.functional.selu_(r)
        assert_close(t, r, atol=1e-5)

    def test_leaky_relu_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.leaky_relu_(t, negative_slope=0.1)
        ref.nn.functional.leaky_relu_(r, negative_slope=0.1)
        assert_close(t, r, atol=1e-5)

    def test_hardtanh_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.hardtanh_(t, min_val=-1.0, max_val=1.0)
        ref.nn.functional.hardtanh_(r, min_val=-1.0, max_val=1.0)
        assert_close(t, r, atol=0.0)

    def test_threshold_(self, ref: Any) -> None:
        x = self._x()
        t = lucid.tensor(x.copy())
        r = ref.tensor(x.copy())
        F.threshold_(t, threshold=0.0, value=-1.0)
        ref.nn.functional.threshold_(r, threshold=0.0, value=-1.0)
        assert_close(t, r, atol=0.0)
