"""``lucid.special`` — special-math sub-package."""

import math

import numpy as np
import pytest

import lucid

# Top-level erf/erfc tested in unit/ops/unary; this file covers the
# special-only entries that don't have a top-level alias.


class TestErfcx:
    def test_at_zero(self) -> None:
        # erfcx(0) = exp(0)·erfc(0) = 1.0.
        assert abs(lucid.special.erfcx(lucid.tensor([0.0])).item() - 1.0) < 1e-5

    def test_between_two_and_four(self) -> None:
        # The continued fraction used to take over only at x = 4, leaving
        # [2, 4) to exp(x²)·erfc(x) — where float32 erfc has lost its
        # relative precision, so erfcx(3.9) came back as 0.240, not 0.140.
        xs = [2.0, 2.5, 3.0, 3.5, 3.9]
        got = lucid.special.erfcx(lucid.tensor(xs)).tolist()
        for x, g in zip(xs, got):
            ref = math.exp(x * x) * math.erfc(x)
            assert abs(g - ref) / ref < 1e-5, (x, g, ref)


class TestI0e:
    def test_at_zero(self) -> None:
        # i0e(0) = exp(0)·I_0(0) = 1.
        assert abs(lucid.special.i0e(lucid.tensor([0.0])).item() - 1.0) < 1e-5


class TestI1:
    def test_at_zero(self) -> None:
        # I_1(0) = 0.
        assert abs(lucid.special.i1(lucid.tensor([0.0])).item()) < 1e-6


class TestI1e:
    def test_at_zero(self) -> None:
        assert abs(lucid.special.i1e(lucid.tensor([0.0])).item()) < 1e-6


class TestNdtr:
    def test_at_zero(self) -> None:
        # Φ(0) = 0.5.
        assert abs(lucid.special.ndtr(lucid.tensor([0.0])).item() - 0.5) < 1e-5

    def test_extreme(self) -> None:
        assert lucid.special.ndtr(lucid.tensor([5.0])).item() > 0.99
        assert lucid.special.ndtr(lucid.tensor([-5.0])).item() < 0.01


class TestNdtri:
    def test_inverse_of_half(self) -> None:
        # ndtri(0.5) = 0.
        assert abs(lucid.special.ndtri(lucid.tensor([0.5])).item()) < 1e-5

    def test_round_trip(self) -> None:
        x = lucid.tensor([-1.5, -0.5, 0.5, 1.5])
        round_trip = lucid.special.ndtri(lucid.special.ndtr(x)).numpy()
        np.testing.assert_allclose(round_trip, x.numpy(), atol=1e-3)


def _log_ndtr_double(x: float) -> float:
    """``log Φ(x)`` in double precision, from ``math.erfc``.

    ``erfc`` itself underflows in double below x ≈ -37.5, so the far left
    tail uses the asymptotic Mills-ratio series, which is converged to
    double precision within a few terms at |x| > 37.
    """
    if x < -37.0:
        s, term = 1.0, 1.0
        for k in range(1, 8):
            term *= -(2 * k - 1) / (x * x)
            s += term
        return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + math.log(s)
    if x < 0.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    return math.log1p(-0.5 * math.erfc(x / math.sqrt(2.0)))


class TestLogNdtr:
    def test_at_zero(self) -> None:
        # log Φ(0) = log 0.5 = -log 2.
        v = lucid.special.log_ndtr(lucid.tensor([0.0])).item()
        assert abs(v - (-math.log(2.0))) < 1e-5

    def test_deep_left_tail(self) -> None:
        # The tail branch was log(½·erfc(-x/√2)), and float32 erfc is 0
        # well before x = -10: log_ndtr(-10) was -inf, not -53.23.
        xs = [-40.0, -10.0, -5.5]
        refs = [-804.60844, -53.231285, -17.779379]
        got = lucid.special.log_ndtr(lucid.tensor(xs)).tolist()
        for x, g, r in zip(xs, got, refs):
            assert abs(g - r) / abs(r) < 1e-5, (x, g, r)

    def test_gradient_finite_on_both_sides_of_the_cut(self) -> None:
        # ``where`` evaluates both branches; a -inf in the losing one
        # would surface as a NaN gradient.
        x = lucid.tensor([-40.0, -10.0, -1.0, 0.0, 10.0], requires_grad=True)
        lucid.special.log_ndtr(x).sum().backward()
        assert x.grad is not None
        g = x.grad.tolist()
        assert all(math.isfinite(v) for v in g)
        # d/dx log Φ(x) = φ(x)/Φ(x), which tends to -x as x → -∞.
        assert abs(g[0] - 40.02498) < 1e-3

    def test_matches_double_precision_from_minus_40_to_40(self, device: str) -> None:
        # The upper tail was the gap: there log Φ(x) ≈ -Q with Q = 1 - Φ(x)
        # small, and forming Φ first left Q to rounding — float32 answered
        # -2.98e-7 at x = 5 for -2.867e-7, and 0 from x ≈ 5.4 on.
        xs = [i / 8.0 for i in range(-320, 321)]
        got = lucid.special.log_ndtr(lucid.tensor(xs, device=device)).tolist()
        # Relative to the answer, down to float32's smallest normal: beyond
        # x ≈ 13 the true value is smaller than that, and whether it comes
        # back subnormal or flushed to zero is the device's business.
        tiny = float(np.finfo(np.float32).tiny)
        for x, g in zip(xs, got):
            r = _log_ndtr_double(x)
            assert abs(g - r) <= max(3e-5 * abs(r), tiny), (x, g, r)

    def test_float64_matches_double_precision(self) -> None:
        # Held to 1e-7 rather than float64's own precision by the lower
        # tail, whose erfcx goes through the engine's float64 erfc.
        xs = [i / 8.0 for i in range(-320, 321)]
        got = lucid.special.log_ndtr(lucid.tensor(xs, dtype=lucid.float64)).tolist()
        tiny = float(np.finfo(np.float64).tiny)
        for x, g in zip(xs, got):
            r = _log_ndtr_double(x)
            assert abs(g - r) <= max(1e-7 * abs(r), tiny), (x, g, r)

    def test_gradient_continuous_across_the_cuts(self) -> None:
        # d/dx log Φ(x) = φ(x)/Φ(x) on either side of each branch point; a
        # mismatch between two branches would show here as a jump.
        upper = 2.0 * math.sqrt(2.0)
        xs = [-1.0 - 1e-4, -1.0, -1.0 + 1e-4, upper - 1e-4, upper, upper + 1e-4]
        xs += [6.0, 20.0]
        x = lucid.tensor(xs, dtype=lucid.float64, requires_grad=True)
        lucid.special.log_ndtr(x).sum().backward()
        assert x.grad is not None
        for xv, g in zip(xs, x.grad.tolist()):
            log_pdf = -0.5 * xv * xv - 0.5 * math.log(2.0 * math.pi)
            ref = math.exp(log_pdf - _log_ndtr_double(xv))
            assert abs(g - ref) <= 1e-5 * ref, (xv, g, ref)


class TestXlog1py:
    def test_zero_zero(self) -> None:
        # 0·log(1+0) = 0.
        out = lucid.special.xlog1py(lucid.tensor([0.0]), lucid.tensor([0.0])).item()
        assert out == 0.0

    def test_known(self) -> None:
        # 2·log(1+1) = 2·log 2.
        out = lucid.special.xlog1py(lucid.tensor([2.0]), lucid.tensor([1.0])).item()
        assert abs(out - 2.0 * math.log(2.0)) < 1e-5


class TestEntr:
    def test_at_one(self) -> None:
        # entr(1) = -1·log(1) = 0.
        assert abs(lucid.special.entr(lucid.tensor([1.0])).item()) < 1e-6

    def test_at_half(self) -> None:
        # entr(0.5) = -0.5·log(0.5) = 0.5·log 2.
        v = lucid.special.entr(lucid.tensor([0.5])).item()
        assert abs(v - 0.5 * math.log(2.0)) < 1e-5


class TestPolygamma:
    def test_n0_eq_digamma(self) -> None:
        x = lucid.tensor([1.0, 2.0, 5.0])
        np.testing.assert_allclose(
            lucid.special.polygamma(0, x).numpy(), lucid.digamma(x).numpy(), atol=1e-5
        )

    def test_n1_at_one_pi2_over_6(self) -> None:
        # ψ¹(1) = π²/6.
        v = lucid.special.polygamma(1, lucid.tensor([1.0])).item()
        assert abs(v - math.pi**2 / 6.0) < 1e-3

    @staticmethod
    def _direct(n: int, x: float, terms: int = 200_000) -> float:
        """psi^(n)(x) = (-1)^(n+1) n! sum_k (x+k)^-(n+1), summed head-on.

        Independent of the asymptotic machinery under test — no Bernoulli
        numbers, no recurrence shift — so agreement is evidence rather than
        a restatement.  Euler-Maclaurin tail keeps truncation off the
        result.
        """
        total = math.fsum(1.0 / (x + k) ** (n + 1) for k in range(terms))
        a = x + terms
        total += 1.0 / (n * a**n) - 0.5 / a ** (n + 1)
        return (1.0 if n % 2 else -1.0) * math.factorial(n) * total

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 12])
    def test_matches_a_direct_summation(self, n: int) -> None:
        """n >= 4 used to raise; the coefficients were tabulated by hand
        for n = 1, 2, 3 even though they are closed-form in n."""
        xs = [0.3, 1.0, 2.5, 7.0]
        got = lucid.special.polygamma(n, lucid.tensor(xs)).numpy()
        for i, x in enumerate(xs):
            ref = self._direct(n, x)
            assert abs(float(got[i]) - ref) <= 4e-6 * abs(ref), (n, x)

    @pytest.mark.parametrize("n", [1, 4, 7, 10])
    def test_satisfies_the_recurrence(self, n: int) -> None:
        """psi^(n)(x+1) - psi^(n)(x) = (-1)^n n! / x^(n+1), exactly.

        Holds whatever the evaluation method, so it catches an error that
        a reference sharing the same derivation would not.
        """
        x = lucid.tensor([0.6, 1.3, 4.0])
        lhs = (lucid.special.polygamma(n, x + 1.0) - lucid.special.polygamma(n, x)).numpy()
        rhs = ((-1.0) ** n * math.factorial(n) / (x ** (n + 1))).numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=4e-6)

    def test_large_orders_overflow_rather_than_going_nan(self) -> None:
        """psi^(35)(1) = 35! zeta(36) ~ 1e40 genuinely exceeds float32.

        Reaching that as ``inf`` is the dtype's limit honestly reported;
        it used to arrive as ``nan`` from n = 33, because an intermediate
        coefficient overflowed while the answer was still representable.
        """
        assert np.isfinite(lucid.special.polygamma(34, lucid.tensor([1.0])).item())
        assert np.isinf(lucid.special.polygamma(35, lucid.tensor([1.0])).item())

    def test_negative_order_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            lucid.special.polygamma(-1, lucid.tensor([1.0]))


class TestSphericalBesselJ0:
    def test_at_zero(self) -> None:
        # j₀(0) = 1 by continuous extension.
        assert abs(lucid.special.spherical_bessel_j0(lucid.tensor([0.0])).item() - 1.0) < 1e-5

    def test_at_pi(self) -> None:
        # sin(π)/π = 0.
        v = lucid.special.spherical_bessel_j0(lucid.tensor([math.pi])).item()
        assert abs(v) < 1e-5


class TestMultigammaln:
    def test_p1_eq_lgamma(self) -> None:
        # multigammaln(a, p=1) = lgamma(a).
        x = lucid.tensor([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            lucid.special.multigammaln(x, 1).numpy(), lucid.lgamma(x).numpy(), atol=1e-5
        )


# ── Orthogonal polynomials ──────────────────────────────────────────────


class TestOrthogonalPolynomials:
    def test_chebyshev_t_known(self) -> None:
        x = lucid.tensor([0.5, 0.0, 1.0])
        # T_0=1, T_1=x, T_2=2x²-1, T_3=4x³-3x.
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 0).numpy(), [1.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 2).numpy(), [-0.5, -1.0, 1.0]
        )
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_t(x, 3).numpy(), [-1.0, 0.0, 1.0]
        )

    def test_chebyshev_u_known(self) -> None:
        # U_2(x) = 4x²-1; at 0.5 → 0.
        np.testing.assert_allclose(
            lucid.special.chebyshev_polynomial_u(lucid.tensor([0.5]), 2).numpy(), [0.0], atol=1e-6
        )

    def test_shifted_chebyshev_at_one(self) -> None:
        # T*_n(1) = T_n(1) = 1 for all n.
        x = lucid.tensor([1.0])
        for n in range(4):
            np.testing.assert_allclose(
                lucid.special.shifted_chebyshev_polynomial_t(x, n).numpy(), [1.0], atol=1e-6
            )

    def test_hermite_h_known(self) -> None:
        # H_2(x)=4x²-2; H_3(x)=8x³-12x.  At x=1 → 2 and -4.
        x = lucid.tensor([1.0])
        np.testing.assert_allclose(lucid.special.hermite_polynomial_h(x, 2).numpy(), [2.0])
        np.testing.assert_allclose(lucid.special.hermite_polynomial_h(x, 3).numpy(), [-4.0])

    def test_hermite_he_known(self) -> None:
        # He_2=x²-1, He_3=x³-3x at x=2.
        x = lucid.tensor([2.0])
        np.testing.assert_allclose(lucid.special.hermite_polynomial_he(x, 2).numpy(), [3.0])
        np.testing.assert_allclose(lucid.special.hermite_polynomial_he(x, 3).numpy(), [2.0])

    def test_legendre_p_known(self) -> None:
        # P_2(x) = (3x²-1)/2; at 0.5 → -0.125.
        np.testing.assert_allclose(
            lucid.special.legendre_polynomial_p(lucid.tensor([0.5]), 2).numpy(), [-0.125], atol=1e-6
        )

    def test_laguerre_l_known(self) -> None:
        # L_0=1, L_1=1-x, L_2=(x²-4x+2)/2.  At x=1 → 1, 0, -0.5.
        x = lucid.tensor([1.0])
        for n, expected in [(0, 1.0), (1, 0.0), (2, -0.5)]:
            np.testing.assert_allclose(
                lucid.special.laguerre_polynomial_l(x, n).numpy(), [expected], atol=1e-6
            )


# ── Bessel ──────────────────────────────────────────────────────────────


class TestBessel:
    def test_j0_known(self) -> None:
        # scipy.special.j0 reference values.
        x = lucid.tensor([0.0, 1.0, 5.0, 2.4048])
        out = lucid.special.bessel_j0(x).numpy()
        np.testing.assert_allclose(out, [1.0, 0.7651977, -0.1775968, 0.0], atol=1e-4)

    def test_j1_known(self) -> None:
        x = lucid.tensor([0.0, 1.0, -1.0, 5.0])
        out = lucid.special.bessel_j1(x).numpy()
        np.testing.assert_allclose(out, [0.0, 0.4400506, -0.4400506, -0.3275791], atol=1e-4)

    def test_y0_y1_known(self) -> None:
        x = lucid.tensor([1.0, 5.0, 10.0])
        np.testing.assert_allclose(
            lucid.special.bessel_y0(x).numpy(), [0.08825696, -0.30851763, 0.05567117], atol=1e-4
        )
        np.testing.assert_allclose(
            lucid.special.bessel_y1(x).numpy(), [-0.78121282, 0.14786314, 0.24901543], atol=1e-4
        )

    def test_modified_k0_k1_known(self) -> None:
        x = lucid.tensor([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            lucid.special.modified_bessel_k0(x).numpy(),
            [0.92441907, 0.42102443, 0.11389387],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            lucid.special.modified_bessel_k1(x).numpy(),
            [1.65644112, 0.60190724, 0.13986588],
            atol=1e-4,
        )

    def test_scaled_modified_k_consistent(self) -> None:
        x_np = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        x = lucid.tensor(x_np)
        unscaled = lucid.special.modified_bessel_k0(x).numpy()
        scaled = lucid.special.scaled_modified_bessel_k0(x).numpy()
        np.testing.assert_allclose(scaled, unscaled * np.exp(x_np), atol=1e-3)


# ── Hurwitz zeta ────────────────────────────────────────────────────────


class TestZeta:
    def test_riemann_special_values(self) -> None:
        import math

        z2 = lucid.special.zeta(lucid.tensor([2.0]), lucid.tensor([1.0])).item()
        z4 = lucid.special.zeta(lucid.tensor([4.0]), lucid.tensor([1.0])).item()
        assert abs(z2 - math.pi**2 / 6.0) < 1e-3
        assert abs(z4 - math.pi**4 / 90.0) < 1e-3

    def test_hurwitz_shift_identity(self) -> None:
        # ζ(s, q+1) = ζ(s, q) - q^{-s}.
        s = lucid.tensor([2.0])
        z_at_1 = lucid.special.zeta(s, lucid.tensor([1.0])).item()
        z_at_2 = lucid.special.zeta(s, lucid.tensor([2.0])).item()
        assert abs((z_at_1 - 1.0) - z_at_2) < 1e-3

    @pytest.mark.parametrize("x", [2.0, 3.0, 5.0, 8.0])
    def test_negative_q_equals_the_three_positive_evaluations(self, x: float) -> None:
        # zeta(x, q) = zeta(x, f) + (-1)^x [zeta(x, 1 - f) - zeta(x, 1 - q)] for a negative
        # non-integer q = f - m.  The three terms share one stacked series evaluation (one
        # graph to compile on Metal, not three); each must still equal its own call.
        q_np = np.array([-0.3, -1.7, -2.5, -6.25, -9.9], dtype=np.float32)
        f_np = q_np - np.floor(q_np)
        z = lucid.special.zeta
        xs = lucid.full((5,), x)
        got = z(xs, lucid.tensor(q_np)).numpy()
        direct = (
            z(xs, lucid.tensor(f_np))
            + (-1.0) ** x * (z(xs, lucid.tensor(1.0 - f_np)) - z(xs, lucid.tensor(1.0 - q_np)))
        ).numpy()
        np.testing.assert_array_equal(got, direct)

    def test_values_are_pinned(self) -> None:
        # The Euler-Maclaurin evaluation order is fixed: restructuring it for compile time
        # (LCD-794) left every bit unchanged, and these are the values from before that change.
        x = lucid.tensor([1.5, 2.0, 3.0, 10.0, 6.0])
        q = lucid.tensor([0.25, 1.0, 7.5, 0.5, 20.0])
        expected = np.array(
            [10.213055610656738, 1.644934058189392, 0.010152626782655716, 1024.0174560546875]
            + [7.070222096672296e-08],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(lucid.special.zeta(x, q).numpy(), expected)
