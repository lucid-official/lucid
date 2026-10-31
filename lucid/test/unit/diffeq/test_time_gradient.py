"""``t`` is differentiable: the time inputs of a solve stay in the graph (LCD-679).

Every solver reads its grid to the host as floats, so ``t.requires_grad`` used
to be honoured nowhere and ``t.grad`` stayed ``None`` while ``y0`` and every
parameter of the same solve got a gradient.  Two derivatives now exist, and
the tests pin each against the thing it is the derivative *of*:

* a fixed-step solve on the caller's own grid differentiates its arithmetic,
  step sizes included -- so it is checked against finite differences of that
  very solver, which is exact for it;
* every other route (adaptive, implicit, multistep, the adjoint, a decoupled
  grid) attaches the exact solution's sensitivity ``f(t_i, y_i)`` -- checked
  against the closed form and against finite differences of a tightly
  converged solve.
"""

import math
from collections.abc import Callable, Sequence

import pytest

import lucid
import lucid.diffeq as diffeq

_F64 = lucid.float64
_GRID = [0.0, 0.5, 1.0]
_UNEVEN = [0.0, 0.3, 0.7, 1.0]


def _decay(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    return -y


def _forced(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    """An oscillator driven explicitly in time, so ``f`` depends on ``t``."""
    return lucid.stack([y[1], -y[0] + lucid.sin(t)])


def _y0() -> lucid.Tensor:
    return lucid.tensor([1.0, 2.0], dtype=_F64)


Solve = Callable[[lucid.Tensor | list[float]], lucid.Tensor]


def _grad_of_t(solve: Solve, grid: Sequence[float]) -> list[float]:
    t = lucid.tensor(list(grid), dtype=_F64, requires_grad=True)
    solve(t).sum().backward()
    assert t.grad is not None, "t received no gradient"
    return [float(v) for v in t.grad.tolist()]


def _finite_difference(solve: Solve, grid: Sequence[float], eps: float = 1e-6) -> list[float]:
    out: list[float] = []
    for i in range(len(grid)):
        up, down = list(grid), list(grid)
        up[i] += eps
        down[i] -= eps
        with lucid.no_grad():
            out.append(float((solve(up).sum() - solve(down).sum()).item()) / (2 * eps))
    return out


def _tensor(out: object) -> lucid.Tensor:
    assert isinstance(out, lucid.Tensor)
    return out


def _odeint(
    func: Callable[..., lucid.Tensor],
    *,
    method: str | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    options: dict[str, object] | None = None,
    return_trajectory: bool = True,
) -> Solve:
    return lambda t: _tensor(
        diffeq.odeint(
            func,
            _y0(),
            t,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
            return_trajectory=return_trajectory,
        )
    )


def _adjoint(
    func: Callable[..., lucid.Tensor],
    *,
    method: str | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> Solve:
    return lambda t: _tensor(
        diffeq.odeint_adjoint(func, _y0(), t, method=method, rtol=rtol, atol=atol)
    )


class TestFixedStepIsTheDerivativeOfItsArithmetic:
    @pytest.mark.parametrize("method", ["euler", "midpoint", "heun3", "rk4", "rk4_classic"])
    @pytest.mark.parametrize("grid", [_GRID, _UNEVEN], ids=["even", "uneven"])
    def test_matches_finite_differences_of_the_same_solver(
        self, method: str, grid: list[float]
    ) -> None:
        solve = _odeint(_decay, method=method)
        assert _grad_of_t(solve, grid) == pytest.approx(
            _finite_difference(solve, grid), rel=1e-6, abs=1e-8
        )

    @pytest.mark.parametrize("method", ["euler", "rk4"])
    def test_an_explicit_time_dependence_is_differentiated_too(self, method: str) -> None:
        solve = _odeint(_forced, method=method)
        assert _grad_of_t(solve, _UNEVEN) == pytest.approx(
            _finite_difference(solve, _UNEVEN), rel=1e-6, abs=1e-8
        )

    def test_euler_on_a_coarse_grid_is_the_discretisations_derivative(self) -> None:
        # Not the exact solution's: dL/dt_1 of the exact solution is -3 e^{-1/2}
        # = -1.82, while Euler's own arithmetic gives -3.  This is the value
        # the reference ODE library reports for the same call.
        got = _grad_of_t(_odeint(_decay, method="euler"), _GRID)
        assert got == pytest.approx([4.5, -3.0, -1.5])

    def test_descending_grid(self) -> None:
        grid = [1.0, 0.5, 0.0]
        solve = _odeint(_decay, method="rk4")
        assert _grad_of_t(solve, grid) == pytest.approx(
            _finite_difference(solve, grid), rel=1e-6, abs=1e-8
        )

    def test_final_state_only(self) -> None:
        solve = _odeint(_decay, method="rk4", return_trajectory=False)
        assert _grad_of_t(solve, _UNEVEN) == pytest.approx(
            _finite_difference(solve, _UNEVEN), rel=1e-6, abs=1e-8
        )


class TestSolutionSensitivityElsewhere:
    """Adaptive, implicit and multistep: ``dL/dt_i = <dL/dy_i, f(t_i, y_i)>``."""

    @staticmethod
    def _closed_form() -> list[float]:
        # y = y0 e^{-(t - t0)}, L = sum_i sum(y_i); sum(y0) = 3.
        e1, e2 = math.exp(-0.5), math.exp(-1.0)
        return [3 * (e1 + e2), -3 * e1, -3 * e2]

    @pytest.mark.parametrize("method", ["dopri5", "tsit5", "bosh3", "dopri8"])
    def test_adaptive_matches_the_closed_form(self, method: str) -> None:
        got = _grad_of_t(_odeint(_decay, method=method, rtol=1e-10, atol=1e-12), _GRID)
        assert got == pytest.approx(self._closed_form(), rel=1e-5)

    @pytest.mark.parametrize("method", ["dopri5", "tsit5"])
    def test_adaptive_matches_finite_differences_with_a_time_dependent_rhs(
        self, method: str
    ) -> None:
        solve = _odeint(_forced, method=method, rtol=1e-11, atol=1e-13)
        assert _grad_of_t(solve, _UNEVEN) == pytest.approx(
            _finite_difference(solve, _UNEVEN), rel=1e-5, abs=1e-7
        )

    @pytest.mark.parametrize("method", ["gl6", "radauIIA5", "explicit_adams", "implicit_adams"])
    def test_other_families_track_the_closed_form(self, method: str) -> None:
        grid = [i / 16 for i in range(17)]
        got = _grad_of_t(_odeint(_decay, method=method), grid)
        e = [math.exp(-x) for x in grid]
        expect = [3 * sum(e[1:]) - 3 * 0.0] + [-3 * v for v in e[1:]]
        # Output rows carry f(t_i, y_i); the start carries the whole adjoint.
        assert got[1:] == pytest.approx(expect[1:], rel=2e-3)
        assert got[0] == pytest.approx(3 * sum(e[1:]), rel=2e-3)

    def test_decoupled_fixed_grid_uses_the_solution_sensitivity(self) -> None:
        # step_size splits the integration grid from t, so t_i are only
        # interpolation points and the exact solution's derivative applies.
        got = _grad_of_t(
            _odeint(_decay, method="rk4", options={"step_size": 0.01, "interp": "cubic"}), _GRID
        )
        assert got == pytest.approx(TestSolutionSensitivityElsewhere._closed_form(), rel=1e-5)

    def test_final_state_only(self) -> None:
        got = _grad_of_t(_odeint(_decay, rtol=1e-10, atol=1e-12, return_trajectory=False), _GRID)
        assert got == pytest.approx([3 * math.exp(-1.0), 0.0, -3 * math.exp(-1.0)], rel=1e-7)


class TestAdjoint:
    @pytest.mark.parametrize("method", ["dopri5", "tsit5", "rk4"])
    def test_matches_the_direct_gradient(self, method: str) -> None:
        tol = {} if method == "rk4" else {"rtol": 1e-10, "atol": 1e-12}
        grid = [i / 32 for i in range(33)] if method == "rk4" else _UNEVEN
        direct = _grad_of_t(_odeint(_forced, method=method, **tol), grid)
        got = _grad_of_t(_adjoint(_forced, method=method, **tol), grid)
        if method == "rk4":
            # The direct rk4 gradient is the discretisation's; the adjoint's is
            # the exact solution's.  They agree to the grid's order.
            assert got == pytest.approx(direct, rel=1e-4, abs=1e-6)
        else:
            assert got == pytest.approx(direct, rel=1e-6, abs=1e-8)

    def test_matches_finite_differences(self) -> None:
        solve = _adjoint(_forced, rtol=1e-11, atol=1e-13)
        assert _grad_of_t(solve, _UNEVEN) == pytest.approx(
            _finite_difference(solve, _UNEVEN), rel=1e-5, abs=1e-7
        )

    def test_start_time_uses_the_adjoint(self) -> None:
        # dL/dt_0 = -<a(t_0), f(t_0, y_0)>: a start-time shift moves every
        # later row, which only a gradient through the solve can account for.
        got = _grad_of_t(_adjoint(_decay, rtol=1e-10, atol=1e-12), _GRID)
        assert got[0] == pytest.approx(3 * (math.exp(-0.5) + math.exp(-1.0)), rel=1e-6)

    def test_parameter_and_state_gradients_are_unchanged_by_a_differentiable_t(self) -> None:
        def run(grad_t: bool) -> tuple[list[float], list[float]]:
            k = lucid.tensor([0.7], dtype=_F64, requires_grad=True)
            y0 = _y0().requires_grad_(True)
            t = lucid.tensor(_GRID, dtype=_F64, requires_grad=grad_t)
            ys = diffeq.odeint_adjoint(
                lambda s, y: -k * y, y0, t, adjoint_params=[k], rtol=1e-10, atol=1e-12
            )
            ys.sum().backward()
            assert k.grad is not None and y0.grad is not None
            return k.grad.tolist(), y0.grad.tolist()

        assert run(True) == pytest.approx(run(False), rel=1e-9)


class TestScope:
    def test_state_gradient_is_unchanged(self) -> None:
        def run(grad_t: bool) -> list[float]:
            y0 = _y0().requires_grad_(True)
            t = lucid.tensor(_GRID, dtype=_F64, requires_grad=grad_t)
            diffeq.odeint(_decay, y0, t, method="rk4").sum().backward()
            assert y0.grad is not None
            return y0.grad.tolist()

        assert run(True) == pytest.approx(run(False), rel=1e-12)

    def test_no_gradient_when_t_does_not_ask(self) -> None:
        t = lucid.tensor(_GRID, dtype=_F64)
        y = diffeq.odeint(_decay, _y0(), t)
        assert not y.requires_grad and t.grad is None

    def test_no_graph_under_no_grad(self) -> None:
        t = lucid.tensor(_GRID, dtype=_F64, requires_grad=True)
        with lucid.no_grad():
            y = diffeq.odeint(_decay, _y0(), t)
        assert not y.requires_grad

    def test_first_row_is_the_input_and_values_are_unchanged(self) -> None:
        t_plain = lucid.tensor(_GRID, dtype=_F64)
        t_grad = lucid.tensor(_GRID, dtype=_F64, requires_grad=True)
        for method in ("dopri5", "rk4"):
            a = diffeq.odeint(_decay, _y0(), t_plain, method=method)
            b = diffeq.odeint(_decay, _y0(), t_grad, method=method)
            # The differentiable fixed-step path trades the fused kernel for
            # plain ops, so the last bit may differ; the first row may not.
            assert b[0].tolist() == a[0].tolist()
            assert float((b - a).abs().max().item()) < 1e-13

    def test_tuple_state(self) -> None:
        t = lucid.tensor(_GRID, dtype=_F64, requires_grad=True)
        pos, vel = diffeq.odeint(
            lambda s, y: (y[1], -y[0]),
            (lucid.tensor([1.0], dtype=_F64), lucid.tensor([0.0], dtype=_F64)),
            t,
            rtol=1e-10,
            atol=1e-12,
        )
        (pos.sum() + vel.sum()).backward()
        assert t.grad is not None
        # y = cos t, v = -sin t: d/dt_2 of (cos t + -sin t) at t = 1.
        assert float(t.grad[2].item()) == pytest.approx(-math.sin(1.0) - math.cos(1.0), rel=1e-6)

    def test_float32_grid_against_a_float64_state(self) -> None:
        t = lucid.tensor(_GRID, requires_grad=True)
        diffeq.odeint(_decay, _y0(), t, method="rk4").sum().backward()
        assert t.grad is not None and t.grad.dtype == t.dtype


class TestEventStartTime:
    def test_autonomous_event_shifts_one_for_one_with_the_start(self) -> None:
        def fall(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], -9.8 * lucid.ones_like(y[1])])

        t0 = lucid.tensor(0.0, dtype=_F64, requires_grad=True)
        y0 = lucid.tensor([2.0, 3.0], dtype=_F64)
        event_t, solution = diffeq.odeint_event(
            fall, y0, t0, event_fn=lambda t, y: y[0], rtol=1e-11, atol=1e-13
        )
        event_t.backward()
        assert t0.grad is not None
        assert float(t0.grad.item()) == pytest.approx(1.0, rel=1e-6)
        assert not solution.requires_grad or solution[0].tolist() == y0.tolist()

    def test_reverse_time(self) -> None:
        t0 = lucid.tensor(1.0, dtype=_F64, requires_grad=True)
        event_t, _ = diffeq.odeint_event(
            _decay,
            lucid.tensor([1.0], dtype=_F64),
            t0,
            event_fn=lambda t, y: y[0] - 2.0,
            reverse_time=True,
            rtol=1e-11,
            atol=1e-13,
        )
        event_t.backward()
        assert t0.grad is not None
        assert float(t0.grad.item()) == pytest.approx(1.0, rel=1e-6)


class TestDenseQueryTime:
    def test_matches_the_closed_form(self) -> None:
        y0 = _y0()
        dense = diffeq.odeint_dense(lambda s, y: -1.3 * y, y0, 0.0, 1.0, rtol=1e-11, atol=1e-13)
        q = lucid.tensor(0.7, dtype=_F64, requires_grad=True)
        dense(q).sum().backward()
        assert q.grad is not None
        assert float(q.grad.item()) == pytest.approx(-1.3 * 3.0 * math.exp(-1.3 * 0.7), rel=1e-7)

    def test_a_host_query_is_untouched(self) -> None:
        dense = diffeq.odeint_dense(_decay, _y0(), 0.0, 1.0)
        assert not dense(0.5).requires_grad


class TestAcrossDevices:
    def test_t_gradient_matches_on_metal(self, device_gpu_only: str) -> None:
        results = {}
        for device in ("cpu", device_gpu_only):
            for name, solve in (("direct", diffeq.odeint), ("adjoint", diffeq.odeint_adjoint)):
                t = lucid.tensor(_GRID, requires_grad=True, device=device)
                y0 = lucid.tensor([1.0, 2.0], device=device)
                solve(_decay, y0, t).sum().backward()
                assert t.grad is not None and str(t.grad.device) == str(t.device)
                results[(device, name)] = t.grad.tolist()
        for name in ("direct", "adjoint"):
            assert results[(device_gpu_only, name)] == pytest.approx(
                results[("cpu", name)], rel=1e-4
            )

    def test_fixed_step_on_metal(self, device_gpu_only: str) -> None:
        t = lucid.tensor(_GRID, requires_grad=True, device=device_gpu_only)
        y0 = lucid.tensor([1.0, 2.0], device=device_gpu_only)
        diffeq.odeint(_decay, y0, t, method="euler").sum().backward()
        assert t.grad is not None
        assert t.grad.tolist() == pytest.approx([4.5, -3.0, -1.5])
