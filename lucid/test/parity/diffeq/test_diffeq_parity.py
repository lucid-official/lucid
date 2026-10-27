"""Reference parity for ``lucid.diffeq`` fixed-step Runge-Kutta integration.

Two independent references are used.  The first is a hand-written RK loop
built from reference-framework tensor ops — it pins Lucid's fused
``rk_combine`` against the same arithmetic done the unfused way, which is
exactly what the fusion is supposed to be equivalent to.  The second is
the reference framework's ODE package when it happens to be installed,
which pins the tableau coefficients themselves against the established
implementation.
"""

from typing import Any, Callable, Sequence

import numpy as np
import pytest

import lucid
import lucid.diffeq as diffeq
from lucid.diffeq import _adaptive, _fused
from lucid.diffeq._tableau import _METHODS
from lucid.test._fixtures.ref_framework import require_ref_ode
from lucid.test._helpers.compare import assert_close

METHODS = ["euler", "midpoint", "heun2", "heun3", "rk4"]


def _grid(n: int, t1: float = 1.0) -> list[float]:
    return [i * t1 / n for i in range(n + 1)]


def _ref_odeint(
    ref: Any, func: Callable[[Any, Any], Any], y0: Any, t: Sequence[float], method: str
) -> Any:
    """Unfused reference implementation of the same fixed-step RK loop."""
    tableau = _METHODS[method]
    y = y0
    trajectory = [y0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        ks: list[Any] = []
        for stage in range(tableau.stages):
            stage_y = y
            for j, coeff in enumerate(tableau.a[stage]):
                if coeff:
                    stage_y = stage_y + dt * coeff * ks[j]
            stage_t = ref.tensor(t[i] + tableau.c[stage] * dt, dtype=y0.dtype)
            ks.append(func(stage_t, stage_y))
        for j, coeff in enumerate(tableau.b):
            if coeff:
                y = y + dt * coeff * ks[j]
        trajectory.append(y)
    return ref.stack(trajectory, dim=0)


@pytest.mark.parity
class TestOdeintParity:
    @pytest.fixture
    def y0_pair(self, ref: Any) -> tuple[lucid.Tensor, Any]:
        rng = np.random.default_rng(0)
        y0 = rng.standard_normal(size=(3, 2)).astype(np.float64)
        return (
            lucid.tensor(y0.copy(), dtype=lucid.float64),
            ref.tensor(y0.copy(), dtype=ref.float64),
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_trajectory_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        # A time-dependent, non-linear RHS so every tableau entry matters.
        grid = _grid(16)
        lucid_traj = diffeq.odeint(lambda t, y: -y + lucid.sin(t), y0_pair[0], grid, method=method)
        ref_traj = _ref_odeint(ref, lambda t, y: -y + ref.sin(t), y0_pair[1], grid, method)
        assert_close(lucid_traj, ref_traj, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_final_state_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        grid = _grid(16)
        lucid_final = diffeq.odeint(
            lambda t, y: y * y - t, y0_pair[0], grid, method=method, return_trajectory=False
        )
        ref_final = _ref_odeint(ref, lambda t, y: y * y - t, y0_pair[1], grid, method)[-1]
        assert_close(lucid_final, ref_final, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_descending_grid_matches_unfused_loop(
        self, method: str, y0_pair: tuple[lucid.Tensor, Any], ref: Any
    ) -> None:
        grid = [1.0 - i / 16 for i in range(17)]
        lucid_traj = diffeq.odeint(lambda t, y: -y, y0_pair[0], grid, method=method)
        ref_traj = _ref_odeint(ref, lambda t, y: -y, y0_pair[1], grid, method)
        assert_close(lucid_traj, ref_traj, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_gradient_matches_unfused_loop(self, method: str, ref: Any) -> None:
        rng = np.random.default_rng(1)
        raw = rng.standard_normal(size=(4,)).astype(np.float64)

        lucid_y0 = lucid.tensor(raw.copy(), dtype=lucid.float64, requires_grad=True)
        ref_y0 = ref.tensor(raw.copy(), dtype=ref.float64, requires_grad=True)
        grid = _grid(12)

        lucid_out = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t), lucid_y0, grid, method=method, return_trajectory=False
        )
        (lucid_out * lucid_out).sum().backward()

        ref_out = _ref_odeint(ref, lambda t, y: -y + ref.sin(t), ref_y0, grid, method)[-1]
        (ref_out * ref_out).sum().backward()

        assert lucid_y0.grad is not None
        assert_close(lucid_y0.grad, ref_y0.grad, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_second_order_matches_unfused_loop(self, method: str, ref: Any) -> None:
        # Double backward through the solver.  The reference loop is built
        # from plain arithmetic, so this pins the fused op's graph-recording
        # backward against the unfused chain it stands in for.
        grid = _grid(8)

        lucid_y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        lucid_k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        lucid_out = diffeq.odeint(
            lambda t, y: -lucid_k * y, lucid_y0, grid, method=method, return_trajectory=False
        )
        (lucid_g,) = lucid.autograd.grad(lucid_out.sum(), lucid_y0, create_graph=True)
        (lucid_mixed,) = lucid.autograd.grad(lucid_g.sum(), lucid_k)

        ref_y0 = ref.tensor([1.0], dtype=ref.float64, requires_grad=True)
        ref_k = ref.tensor([0.5], dtype=ref.float64, requires_grad=True)
        ref_out = _ref_odeint(ref, lambda t, y: -ref_k * y, ref_y0, grid, method)[-1]
        (ref_g,) = ref.autograd.grad(ref_out.sum(), ref_y0, create_graph=True)
        (ref_mixed,) = ref.autograd.grad(ref_g.sum(), ref_k)

        assert lucid_mixed is not None
        assert_close(lucid_mixed, ref_mixed, atol=1e-10, rtol=1e-9)


@pytest.mark.parity
class TestReferenceOdeParity:
    """Cross-check the tableau coefficients against the established library.

    Skipped unless the reference ODE package is installed — it is not a
    Lucid test dependency, only an opportunistic second opinion.
    """

    @pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
    def test_matches_the_reference_solver(self, method: str, ref: Any) -> None:
        ref_ode = require_ref_ode()

        rng = np.random.default_rng(2)
        raw = rng.standard_normal(size=(5,)).astype(np.float64)
        n = 16
        grid = _grid(n)

        lucid_out = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t),
            lucid.tensor(raw.copy(), dtype=lucid.float64),
            grid,
            method=method,
        )
        ref_out = ref_ode.odeint(
            lambda t, y: -y + ref.sin(t),
            ref.tensor(raw.copy(), dtype=ref.float64),
            ref.tensor(grid, dtype=ref.float64),
            method=method,
            options={"step_size": 1.0 / n},
        )
        assert_close(lucid_out, ref_out, atol=1e-10, rtol=1e-9)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_event_time_gradient_matches_the_reference_solver(
        self, reverse: bool, ref: Any
    ) -> None:
        """The implicit-function rerouting has to agree, not merely exist.

        A gradient obtained this way is easy to get plausibly wrong -- a
        dropped ``dg/dt`` or a sign on the reverse-time branch still produces
        a finite, believable number.  The event function used here depends on
        both arguments so neither term can be dropped unnoticed.
        """
        ref_ode = require_ref_ode()
        target = 2.0 if reverse else 0.5

        k = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        event_t, _ = diffeq.odeint_event(
            lambda t, y: -k * y,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            event_fn=lambda t, y: y[0] - target,
            reverse_time=reverse,
            rtol=1e-12,
            atol=1e-14,
        )
        event_t.backward()

        ref_k = ref.tensor([2.0], dtype=ref.float64, requires_grad=True)
        ref_event_t, _ = ref_ode.odeint_event(
            lambda t, y: -ref_k * y,
            ref.tensor([1.0], dtype=ref.float64),
            ref.tensor(0.0, dtype=ref.float64),
            event_fn=lambda t, y: y[0] - target,
            reverse_time=reverse,
            rtol=1e-12,
            atol=1e-14,
        )
        ref_event_t.backward()

        assert float(event_t.item()) == pytest.approx(float(ref_event_t.item()), rel=1e-9)
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(float(ref_k.grad.item()), rel=1e-9)

    def test_dopri8_tableau_is_transcribed_exactly(self) -> None:
        """Pin all ~130 dopri8 coefficients against the reference's own.

        Dopri8 is the one tableau here that is neither derived nor short
        enough to check by eye: thirteen stages of published ratios.  A single
        wrong digit does not raise and does not cost the method its order in
        any way a convergence test would notice -- it shifts the answer at a
        place no other check looks.  Since the reference stores the same
        ratios, every coefficient can be held to bitwise equality, which is
        the strongest statement available and the reason transcribing it was
        safe to do at all.
        """
        dopri8 = require_ref_ode("_impl.dopri8")
        tab = diffeq.DOPRI8
        ref_tab, ref_mid = dopri8._DOPRI8_TABLEAU, dopri8._C_mid

        # The reference drops the leading zero from ``c`` and the empty first
        # row from ``a``; both are implied by stage 0 being ``f(t0, y0)``.
        assert tab.c == (0.0,) + tuple(ref_tab.alpha.tolist())
        assert tab.a == ((),) + tuple(tuple(r.tolist()) for r in ref_tab.beta)
        assert tab.b == tuple(ref_tab.c_sol.tolist())
        assert tab.b_error == tuple(ref_tab.c_error.tolist())
        assert tab.mid == tuple(ref_mid.tolist())


ADAPTIVE_METHODS = ["dopri5", "dopri8", "tsit5", "bosh3", "fehlberg2", "adaptive_heun"]

# Step count for an adaptive method scales like ``tol ** (-1 / order)``, so a
# single tight tolerance would make the second-order methods take minutes
# while dopri5 finishes instantly.  Ask each for the accuracy its order can
# reach at comparable cost, and assert against that.
ADAPTIVE_TOL = {
    "dopri5": (1e-12, 1e-14, 1e-9),
    # dopri8 solves to eighth order but interpolates to output times with the
    # same quartic as the rest, so what a caller reads off a coarse grid is
    # limited by the interpolant, not by the steps.  Asking it for 1e-9 at an
    # interior point would fail on that alone.
    "dopri8": (1e-12, 1e-14, 1e-6),
    "tsit5": (1e-12, 1e-14, 1e-9),
    "bosh3": (1e-11, 1e-13, 1e-7),
    "fehlberg2": (1e-9, 1e-11, 1e-7),
    "adaptive_heun": (1e-9, 1e-11, 1e-7),
}


@pytest.mark.parity
class TestFusedPrimitiveParity:
    """Pin the fused engine ops against the same arithmetic done unfused.

    The adaptive loop's algorithm is covered by behavioural unit tests; what
    parity has to establish is that fusing the per-step arithmetic into single
    kernels did not change the numbers.
    """

    @pytest.fixture
    def state(self, ref: Any) -> tuple[list[lucid.Tensor], list[Any]]:
        rng = np.random.default_rng(7)
        # Two states plus one stage derivative per stage of the widest tableau
        # -- dopri8's fourteen.  Sized from the registry so adding a method
        # cannot silently leave the fixture short.
        n_slots = 2 + max(t.stages for t in _METHODS.values())
        raw = [rng.standard_normal(size=(3, 4)).astype(np.float64) for _ in range(n_slots)]
        return (
            [lucid.tensor(a.copy(), dtype=lucid.float64) for a in raw],
            [ref.tensor(a.copy(), dtype=ref.float64) for a in raw],
        )

    @pytest.mark.parametrize("method", ADAPTIVE_METHODS)
    def test_error_norm_matches_unfused(
        self, method: str, state: tuple[list[lucid.Tensor], list[Any]], ref: Any
    ) -> None:
        tableau = _METHODS[method]
        lucid_ts, ref_ts = state
        n = tableau.stages
        dt, rtol, atol = 0.037, 1e-5, 1e-8

        got = _fused.error_ratio(
            lucid_ts[0], lucid_ts[1], lucid_ts[2 : 2 + n][:n], tableau.b_error[:n], dt, rtol, atol
        )

        y0, y1 = ref_ts[0], ref_ts[1]
        ks = ref_ts[2 : 2 + n]
        err = sum(c * k for c, k in zip(tableau.b_error, ks)) * dt
        tol = atol + rtol * ref.max(y0.abs(), y1.abs())
        want = float((err / tol).abs().pow(2).mean().sqrt())

        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

    def test_interpolation_matches_unfused_polynomial(
        self, state: tuple[list[lucid.Tensor], list[Any]], ref: Any
    ) -> None:
        lucid_ts, ref_ts = state
        y0, y1, y_mid, f0, f1 = lucid_ts[:5]
        dt, t0, t1 = 0.25, 1.5, 1.75

        coeffs = _adaptive.interp_fit(y0, y1, y_mid, f0, f1, dt)
        for frac in (0.0, 0.1, 0.5, 0.9, 1.0):
            got = _adaptive.interp_evaluate(coeffs, t0, t1, t0 + frac * (t1 - t0))

            ry0, ry1, rmid, rf0, rf1 = ref_ts[:5]
            ra = 2 * dt * (rf1 - rf0) - 8 * (ry0 + ry1) + 16 * rmid
            rb = dt * (5 * rf0 - 3 * rf1) + 18 * ry0 + 14 * ry1 - 32 * rmid
            rc = dt * (rf1 - 4 * rf0) - 11 * ry0 - 5 * ry1 + 16 * rmid
            rd = dt * rf0
            x = frac
            want = ((((ra * x + rb) * x + rc) * x + rd) * x) + ry0

            assert_close(got, want, atol=1e-12, rtol=1e-11)

    def test_interpolant_reproduces_its_anchors(
        self, state: tuple[list[lucid.Tensor], list[Any]], ref: Any
    ) -> None:
        # The quartic is pinned by five conditions; check the three value ones
        # directly so a mistyped coefficient cannot hide behind a small error.
        lucid_ts, _ = state
        y0, y1, y_mid, f0, f1 = lucid_ts[:5]
        coeffs = _adaptive.interp_fit(y0, y1, y_mid, f0, f1, 0.25)
        assert_close(_adaptive.interp_evaluate(coeffs, 0.0, 1.0, 0.0), y0, atol=1e-12)
        assert_close(_adaptive.interp_evaluate(coeffs, 0.0, 1.0, 1.0), y1, atol=1e-12)
        assert_close(_adaptive.interp_evaluate(coeffs, 0.0, 1.0, 0.5), y_mid, atol=1e-12)


@pytest.mark.parity
class TestAdaptiveAgreement:
    """Every method must land on the same solution when told to be accurate."""

    @pytest.mark.parametrize("method", ADAPTIVE_METHODS)
    def test_agrees_with_the_reference_framework_closed_form(self, method: str, ref: Any) -> None:
        # y' = -y + sin(t) has the closed form below; solving it with a tight
        # tolerance must reproduce that, computed in the reference framework.
        rtol, atol, want_tol = ADAPTIVE_TOL[method]
        y0_val = 0.75
        t1 = 1.3
        y0 = lucid.tensor([y0_val], dtype=lucid.float64)
        got = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t),
            y0,
            [0.0, t1],
            method=method,
            rtol=rtol,
            atol=atol,
            return_trajectory=False,
        )
        rt = ref.tensor(t1, dtype=ref.float64)
        want = (ref.tensor(y0_val, dtype=ref.float64) + 0.5) * ref.exp(-rt) + 0.5 * (
            ref.sin(rt) - ref.cos(rt)
        )
        assert float(got.item()) == pytest.approx(float(want), abs=want_tol)

    @pytest.mark.parametrize("method", ADAPTIVE_METHODS)
    def test_agrees_with_a_fine_fixed_step_solve(self, method: str, ref: Any) -> None:
        rng = np.random.default_rng(11)
        raw = rng.standard_normal(size=(4,)).astype(np.float64)
        y0 = lucid.tensor(raw.copy(), dtype=lucid.float64)

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return -y + lucid.sin(t)

        rtol, atol, want_tol = ADAPTIVE_TOL[method]
        adaptive = diffeq.odeint(
            rhs, y0, [0.0, 1.0], method=method, rtol=rtol, atol=atol, return_trajectory=False
        )
        fixed = diffeq.odeint(
            rhs, y0, [i / 512 for i in range(513)], method="rk4", return_trajectory=False
        )
        assert_close(adaptive, fixed, atol=want_tol, rtol=want_tol * 10)


@pytest.mark.parity
class TestOdeintDenseParity:
    """The continuous solution must agree with the grid solve and the truth."""

    @pytest.mark.parametrize("method", ADAPTIVE_METHODS)
    @pytest.mark.filterwarnings("ignore:method .*interpolates:RuntimeWarning")
    def test_matches_the_reference_framework_closed_form(self, method: str, ref: Any) -> None:
        rtol, atol, want_tol = ADAPTIVE_TOL[method]
        y0_val = 0.75
        y0 = lucid.tensor([y0_val], dtype=lucid.float64)
        dense = diffeq.odeint_dense(
            lambda t, y: -y + lucid.sin(t), y0, 0.0, 1.3, method=method, rtol=rtol, atol=atol
        )
        # A time inside a step carries the interpolant's error on top of the
        # solver's, and the two are not always comparable: measured here,
        # bosh3's interior error is ~90x its endpoint error because its
        # midpoint weights are coarse, while dopri5's is 1x.  The endpoint is
        # asserted tightly by test_agrees_with_the_reference_framework_closed_form.
        interior_tol = want_tol * 100
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            t = 1.3 * frac
            rt = ref.tensor(t, dtype=ref.float64)
            want = (ref.tensor(y0_val, dtype=ref.float64) + 0.5) * ref.exp(-rt) + 0.5 * (
                ref.sin(rt) - ref.cos(rt)
            )
            assert float(dense(t).item()) == pytest.approx(float(want), abs=interior_tol)

    def test_matches_odeint_on_the_same_solve(self, ref: Any) -> None:
        # A bounded right-hand side: y * y - t blows up wherever |y| > 1, and
        # the resulting step collapse costs 20s for no extra coverage.
        rng = np.random.default_rng(21)
        raw = rng.standard_normal(size=(3,)).astype(np.float64)
        y0 = lucid.tensor(raw.copy(), dtype=lucid.float64)
        rhs = lambda t, y: -y + lucid.sin(t)  # noqa: E731
        times = [0.0, 0.11, 0.37, 0.6, 0.9]

        traj = diffeq.odeint(rhs, y0, times, rtol=1e-12, atol=1e-14)
        dense = diffeq.odeint_dense(rhs, y0, 0.0, 0.9, rtol=1e-12, atol=1e-14)
        for t, row in zip(times, traj.tolist()):
            assert_close(dense(t), lucid.tensor(row, dtype=lucid.float64), atol=1e-9)

    def test_fixed_cubic_matches_a_fine_fixed_solve(self, ref: Any) -> None:
        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64)
        rhs = lambda t, y: -y + lucid.sin(t)  # noqa: E731
        dense = diffeq.odeint_dense(
            rhs, y0, 0.0, 1.0, method="rk4", options={"step_size": 1 / 64, "interp": "cubic"}
        )
        fine = diffeq.odeint(rhs, y0, [i / 1024 for i in range(1025)], method="rk4")
        for i in (0, 256, 512, 1024):
            assert_close(dense(i / 1024), fine[i], atol=1e-9, rtol=1e-8)


@pytest.mark.parity
class TestAdjointParity:
    """The adjoint gradient must land where direct differentiation does."""

    @pytest.mark.parametrize("method", ["dopri5", "tsit5", "bosh3"])
    def test_matches_direct_differentiation(self, method: str, ref: Any) -> None:
        rng = np.random.default_rng(31)
        raw = rng.standard_normal(size=(4,)).astype(np.float64)
        grid = [0.0, 0.4, 1.0]

        def run(adjoint: bool) -> tuple[list[float], list[float]]:
            k = lucid.tensor([0.6], dtype=lucid.float64, requires_grad=True)
            y0 = lucid.tensor(raw.copy(), dtype=lucid.float64, requires_grad=True)
            solve = diffeq.odeint_adjoint if adjoint else diffeq.odeint
            extra = {"adjoint_params": [k]} if adjoint else {}
            ys = solve(
                lambda t, y: -k * y + lucid.sin(t),
                y0,
                grid,
                method=method,
                rtol=1e-12,
                atol=1e-14,
                **extra,
            )
            (ys * ys).sum().backward()
            assert y0.grad is not None and k.grad is not None
            return y0.grad.tolist(), k.grad.tolist()

        adj_y0, adj_k = run(True)
        dir_y0, dir_k = run(False)
        assert adj_y0 == pytest.approx(dir_y0, abs=1e-6)
        assert adj_k == pytest.approx(dir_k, abs=1e-6)

    def test_matches_the_reference_framework_closed_form(self, ref: Any) -> None:
        # y' = -k y has y(T) = y0 exp(-k T); differentiate that in the
        # reference framework and require the adjoint to reproduce it.
        y0_val, k_val, horizon = 1.25, 0.6, 1.4
        rk = ref.tensor(k_val, dtype=ref.float64, requires_grad=True)
        ry0 = ref.tensor(y0_val, dtype=ref.float64, requires_grad=True)
        (ry0 * ref.exp(-rk * horizon)).backward()

        k = lucid.tensor([k_val], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([y0_val], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint_adjoint(
            lambda t, y: -k * y, y0, [0.0, horizon], rtol=1e-12, atol=1e-14, adjoint_params=[k]
        )[-1].sum().backward()

        assert float(y0.grad.item()) == pytest.approx(float(ry0.grad), abs=1e-8)
        assert float(k.grad.item()) == pytest.approx(float(rk.grad), abs=1e-8)


@pytest.mark.parity
class TestEventParity:
    """Event times must match the closed forms, computed independently."""

    # Only the methods whose midpoint weights are accurate.  bosh3,
    # fehlberg2 and adaptive_heun carry first-order ``mid`` coefficients, so
    # their interpolant — and therefore any event time read off it — is far
    # coarser than their steps.  Pinned by
    # TestInterpolantQuality::test_midpoint_accuracy_varies_by_tableau.
    @pytest.mark.parametrize("method", ["dopri5", "tsit5"])
    def test_free_fall_impact_time(self, method: str, ref: Any) -> None:
        # Height h(t) = h0 - g t^2 / 2 reaches zero at sqrt(2 h0 / g).
        h0, g = 12.5, 9.8
        y0 = lucid.tensor([h0, 0.0], dtype=lucid.float64)

        def fall(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], lucid.tensor(-g, dtype=y.dtype)], dim=0)

        event_t, sol = diffeq.odeint_event(
            fall, y0, 0.0, event_fn=lambda t, y: y[0], method=method, rtol=1e-12, atol=1e-14
        )
        want = ref.sqrt(ref.tensor(2 * h0 / g, dtype=ref.float64))
        assert float(event_t.item()) == pytest.approx(float(want), abs=1e-7)
        # Impact speed is g * t_event.
        assert sol[-1].tolist()[1] == pytest.approx(-g * float(want), abs=1e-6)

    def test_threshold_crossing_on_exponential_decay(self, ref: Any) -> None:
        # y = y0 exp(-t) crosses a threshold c at t = ln(y0 / c).
        y0_val, threshold = 3.0, 0.4
        y0 = lucid.tensor([y0_val], dtype=lucid.float64)
        event_t, sol = diffeq.odeint_event(
            lambda t, y: -y, y0, 0.0, event_fn=lambda t, y: y[0] - threshold, rtol=1e-12, atol=1e-14
        )
        want = ref.log(ref.tensor(y0_val / threshold, dtype=ref.float64))
        assert float(event_t.item()) == pytest.approx(float(want), abs=1e-9)
        assert sol[-1].tolist()[0] == pytest.approx(threshold, abs=1e-9)

    def test_state_at_the_event_matches_odeint_at_that_time(self, ref: Any) -> None:
        # Solving to the discovered event time by the ordinary path must give
        # the same state — the interpolant is not allowed to drift from it.
        y0 = lucid.tensor([2.0, -1.0], dtype=lucid.float64)
        rhs = lambda t, y: -y + lucid.sin(t)  # noqa: E731
        event_t, sol = diffeq.odeint_event(
            rhs, y0, 0.0, event_fn=lambda t, y: y[0] - 0.9, rtol=1e-12, atol=1e-14
        )
        direct = diffeq.odeint(
            rhs, y0, [0.0, float(event_t.item())], rtol=1e-12, atol=1e-14, return_trajectory=False
        )
        assert_close(sol[-1], direct, atol=1e-9)


# Published Adams coefficients, written out as the exact fractions they are.
# Lucid derives its own rather than tabulating them, so these are genuinely
# independent: a slip in the derivation shows up here as a mismatch.
_BASHFORTH: dict[int, list[float]] = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
}
_MOULTON: dict[int, list[float]] = {
    1: [1.0],
    2: [1 / 2, 1 / 2],
    3: [5 / 12, 8 / 12, -1 / 12],
    4: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
    5: [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
}


def _ref_adams(
    ref: Any,
    func: Callable[[Any, Any], Any],
    y0: Any,
    t: Sequence[float],
    max_order: int,
    implicit: bool,
    max_iters: int = 4,
) -> Any:
    """Unfused reference Adams loop over the published coefficient tables."""
    tableau = _METHODS["rk4"]
    y = y0
    trajectory = [y0]
    history: list[Any] = []

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        f0 = func(t[i], y)
        history.insert(0, f0)
        del history[max_order:]
        order = min(len(history), max_order)

        if order < 4:
            ks = [f0]
            for stage in range(1, tableau.stages):
                stage_y = y
                for j, coeff in enumerate(tableau.a[stage]):
                    if coeff != 0.0:
                        stage_y = stage_y + dt * coeff * ks[j]
                ks.append(func(t[i] + tableau.c[stage] * dt, stage_y))
            y_next = y
            for j, coeff in enumerate(tableau.b):
                if coeff != 0.0:
                    y_next = y_next + dt * coeff * ks[j]
        else:
            y_next = y
            for j, coeff in enumerate(_BASHFORTH[order]):
                y_next = y_next + dt * coeff * history[j]
            if implicit:
                for _ in range(max_iters):
                    f_next = func(t[i + 1], y_next)
                    updated = y
                    stencil = [f_next, *history[: order - 1]]
                    for j, coeff in enumerate(_MOULTON[order]):
                        updated = updated + dt * coeff * stencil[j]
                    y_next = updated

        trajectory.append(y_next)
        y = y_next
    return ref.stack(trajectory, dim=0)


@pytest.mark.parity
class TestMultistepParity:
    """Adams stepping against an unfused loop over the published tables."""

    @pytest.fixture
    def problem(self, ref: Any) -> tuple[Any, Any]:
        return ref.tensor([1.0, -2.0], dtype=ref.float64), ref.float64

    @pytest.mark.parametrize("max_order", [4, 5])
    @pytest.mark.parametrize(
        ("method", "implicit"),
        [("explicit_adams", False), ("implicit_adams", True), ("fixed_adams", True)],
    )
    def test_matches_the_unfused_loop(
        self, ref: Any, method: str, implicit: bool, max_order: int
    ) -> None:
        grid = _grid(60)
        y0_val = [1.0, -2.0]

        got = diffeq.odeint(
            lambda t, y: -y + lucid.sin(t),
            lucid.tensor(y0_val, dtype=lucid.float64),
            grid,
            method=method,
            options={"max_order": max_order},
        )
        want = _ref_adams(
            ref,
            lambda t, y: -y + ref.sin(ref.tensor(t, dtype=ref.float64)),
            ref.tensor(y0_val, dtype=ref.float64),
            grid,
            max_order,
            implicit,
        )
        assert_close(got, want, atol=1e-11)

    @pytest.mark.parametrize("max_order", [1, 2, 3])
    def test_low_max_order_runs_the_runge_kutta_fallback(self, ref: Any, max_order: int) -> None:
        # Below order 4 there is nothing to gain over RK4, so the solver never
        # applies Adams weights at all and must reproduce rk4 exactly.
        grid = _grid(40)
        y0_val = [0.5]
        got = diffeq.odeint(
            lambda t, y: -y,
            lucid.tensor(y0_val, dtype=lucid.float64),
            grid,
            method="explicit_adams",
            options={"max_order": max_order},
        )
        want = diffeq.odeint(
            lambda t, y: -y, lucid.tensor(y0_val, dtype=lucid.float64), grid, method="rk4"
        )
        assert float((got - want).abs().max().item()) == 0.0

    def test_matches_the_reference_solver(self, ref: Any) -> None:
        ref_ode = require_ref_ode()
        grid = _grid(60)
        y0_val = [1.0, -2.0]
        got = diffeq.odeint(
            lambda t, y: -y,
            lucid.tensor(y0_val, dtype=lucid.float64),
            grid,
            method="implicit_adams",
            options={"max_order": 5},
        )
        ref_out = ref_ode.odeint(
            lambda t, y: -y,
            ref.tensor(y0_val, dtype=ref.float64),
            ref.tensor(grid, dtype=ref.float64),
            method="implicit_adams",
            options={"max_order": 5},
        )
        assert_close(got, ref_out, atol=1e-8)


IMPLICIT_METHODS = [
    "implicit_euler",
    "implicit_midpoint",
    "trapezoid",
    "radauIIA3",
    "radauIIA5",
    "gl4",
    "gl6",
    "sdirk2",
    "trbdf2",
]


def _stability_step(ref: Any, method: str, z: float) -> float:
    r"""One step of ``y' = lambda*y`` from the tableau, in closed form.

    For a linear problem the stage equations stop being nonlinear: with
    ``z = lambda*dt`` they reduce to ``(I - z*A) K = lambda*y*1``, so the step
    multiplies the state by the method's stability function

    .. math::
        R(z) = 1 + z\,b^{T}(I - zA)^{-1}\mathbf{1}

    computed here with the reference framework's linear algebra.  Nothing
    about Lucid's nonlinear solver enters, which is what makes this an
    independent check on both the derived tableau and the solve that is
    supposed to land on it.
    """
    tableau = _METHODS[method]
    stages = tableau.stages
    a = ref.tensor([list(row) for row in tableau.a], dtype=ref.float64)
    b = ref.tensor(list(tableau.b), dtype=ref.float64)
    ones = ref.ones(stages, dtype=ref.float64)
    lhs = ref.eye(stages, dtype=ref.float64) - z * a
    solved = ref.linalg.solve(lhs, ones)
    return float(1.0 + z * ref.dot(b, solved))


@pytest.mark.parity
class TestImplicitParity:
    """Implicit stepping against the closed-form linear solution."""

    @pytest.mark.parametrize("method", IMPLICIT_METHODS)
    @pytest.mark.parametrize("rate", [-1.0, -50.0, 2.0])
    def test_matches_the_stability_function(self, ref: Any, method: str, rate: float) -> None:
        steps = 20
        dt = 1.0 / steps
        got = diffeq.odeint(
            lambda t, y: rate * y,
            lucid.tensor([1.0], dtype=lucid.float64),
            _grid(steps),
            method=method,
            return_trajectory=False,
        )
        want = _stability_step(ref, method, rate * dt) ** steps
        assert float(got.item()) == pytest.approx(want, rel=1e-9)

    @pytest.mark.parametrize("method", IMPLICIT_METHODS)
    def test_matches_the_closed_form_on_a_coupled_system(self, ref: Any, method: str) -> None:
        # A 2x2 system, so the stage solve is genuinely multidimensional
        # rather than a scalar equation dressed up as one.
        matrix = [[-3.0, 1.0], [1.0, -3.0]]
        y0_val = [1.0, 0.0]
        steps = 20
        dt = 1.0 / steps

        got = diffeq.odeint(
            lambda t, y: lucid.matmul(
                lucid.tensor(matrix, dtype=lucid.float64), y.reshape(2, 1)
            ).reshape(2),
            lucid.tensor(y0_val, dtype=lucid.float64),
            _grid(steps),
            method=method,
            return_trajectory=False,
        )

        # Same construction as above, one Kronecker level up: the stage system
        # is (I - dt*(A kron M)) K = (1 kron M) y.
        tableau = _METHODS[method]
        stages = tableau.stages
        m = ref.tensor(matrix, dtype=ref.float64)
        a = ref.tensor([list(row) for row in tableau.a], dtype=ref.float64)
        b = ref.tensor(list(tableau.b), dtype=ref.float64)
        y = ref.tensor(y0_val, dtype=ref.float64)
        big_eye = ref.eye(2 * stages, dtype=ref.float64)

        for _ in range(steps):
            lhs = big_eye - dt * ref.kron(a, m)
            rhs = ref.cat([m @ y for _ in range(stages)])
            k = ref.linalg.solve(lhs, rhs).reshape(stages, 2)
            y = y + dt * (b @ k)

        assert got.tolist() == pytest.approx(y.tolist(), rel=1e-8, abs=1e-12)
