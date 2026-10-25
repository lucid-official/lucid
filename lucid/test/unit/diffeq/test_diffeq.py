"""``lucid.diffeq`` — Butcher tableaux and fixed-step Runge-Kutta integration."""

import math
import warnings

import pytest

import lucid
import lucid.diffeq as diffeq
from lucid.diffeq import _collocation, _fused, _multistep
from lucid.diffeq._solvers import _combine
from lucid.diffeq._tableau import _METHODS

METHOD_ORDERS = [
    ("euler", 1),
    ("midpoint", 2),
    ("heun2", 2),
    ("heun3", 3),
    ("rk4", 4),
    ("rk4_classic", 4),
]


def _grid(n: int, t1: float = 1.0) -> list[float]:
    return [i * t1 / n for i in range(n + 1)]


def _decay(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    """RHS of ``y' = -y``; solution ``y(t) = y0 * exp(-t)``."""
    return -y


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    """Largest elementwise gap, for shapes pytest.approx cannot nest into."""
    return float((a - b).abs().max().item())


class TestButcherTableau:
    def test_builtin_stage_counts(self) -> None:
        assert diffeq.EULER.stages == 1
        assert diffeq.MIDPOINT.stages == 2
        assert diffeq.HEUN2.stages == 2
        assert diffeq.HEUN3.stages == 3
        assert diffeq.RK4.stages == 4
        assert diffeq.RK4_CLASSIC.stages == 4

    def test_builtin_orders(self) -> None:
        tableaux = (diffeq.EULER, diffeq.MIDPOINT, diffeq.HEUN2, diffeq.HEUN3)
        assert [t.order for t in tableaux] == [1, 2, 2, 3]
        assert diffeq.RK4.order == 4
        assert diffeq.RK4_CLASSIC.order == 4

    def test_registry_names_match_the_reference_library(self) -> None:
        # Method-name strings are exposed API, so they track the reference
        # ODE library exactly: it has heun2 / heun3 and no bare "heun".
        #
        # "rk4_classic" is the one deliberate addition.  The reference gives
        # "rk4" to the 3/8 rule and has no name at all for the classical
        # tableau, so keeping the classical one reachable needs a name the
        # reference does not define.  See test_rk4_is_the_reference_spelling.
        assert sorted(_METHODS) == [
            "adaptive_heun",
            "bosh3",
            "dopri5",
            "dopri8",
            "euler",
            "fehlberg2",
            "gl4",
            "gl6",
            "heun2",
            "heun3",
            "implicit_euler",
            "implicit_midpoint",
            "midpoint",
            "radauIIA3",
            "radauIIA5",
            "rk4",
            "rk4_classic",
            "sdirk2",
            "trapezoid",
            "trbdf2",
            "tsit5",
        ]
        for name, tableau in _METHODS.items():
            assert tableau.name == name

    def test_rk4_is_the_reference_spelling_not_the_textbook_one(self) -> None:
        # Two different fourth-order methods share the popular name.  The
        # reference ODE library resolves "rk4" to Kutta's 3/8 rule, so Lucid
        # does too -- a method name is exposed API, and code ported across has
        # to produce the same numbers.  Both are fourth order, so an order
        # convergence test cannot tell them apart; only the coefficients can.
        assert diffeq.RK4.b == (1 / 8, 3 / 8, 3 / 8, 1 / 8)
        assert diffeq.RK4.c == (0.0, 1 / 3, 2 / 3, 1.0)
        assert diffeq.RK4.a == ((), (1 / 3,), (-1 / 3, 1.0), (1.0, -1.0, 1.0))

        assert diffeq.RK4_CLASSIC.b == (1 / 6, 1 / 3, 1 / 3, 1 / 6)
        assert diffeq.RK4_CLASSIC.c == (0.0, 0.5, 0.5, 1.0)
        assert diffeq.RK4_CLASSIC.a == ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0))

    def test_the_two_rk4_tableaux_actually_disagree(self) -> None:
        # Guards against a future edit quietly making them the same tableau,
        # which would leave both names but lose the distinction they encode.
        #
        # The problem has to be non-autonomous or nonlinear.  On y' = -y every
        # four-stage fourth-order explicit method collapses to the same
        # stability polynomial 1 + z + z^2/2 + z^3/6 + z^4/24, so the two
        # tableaux agree to round-off there and the guard would never fire.
        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return -y + lucid.sin(t)

        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64)
        args = dict(return_trajectory=False)
        a = diffeq.odeint(rhs, y0, _grid(8), method="rk4", **args)
        b = diffeq.odeint(rhs, y0, _grid(8), method="rk4_classic", **args)
        # Same order, so they agree to about the fifth-order term and no more.
        gap = _maxdiff(a, b)
        assert gap > 1e-12, "the two rk4 tableaux collapsed into one"
        assert gap < 1e-4, "these should still be two fourth-order methods"

    def test_heun3_coefficients(self) -> None:
        assert diffeq.HEUN3.a == ((), (1 / 3,), (0.0, 2 / 3))
        assert diffeq.HEUN3.b == (0.25, 0.0, 0.75)
        assert diffeq.HEUN3.c == (0.0, 1 / 3, 2 / 3)

    def test_sequences_are_coerced_to_tuples(self) -> None:
        tab = diffeq.ButcherTableau(
            a=[[], [1.0]], b=[0.5, 0.5], c=[0, 1], order=2, name="heun-ish"
        )
        assert tab.a == ((), (1.0,))
        assert tab.b == (0.5, 0.5)
        assert tab.c == (0.0, 1.0)
        # Frozen + fully tuple-ised means the tableau is hashable.
        assert isinstance(hash(tab), int)

    def test_is_frozen(self) -> None:
        with pytest.raises(Exception):
            diffeq.RK4.order = 5  # type: ignore[misc]

    def test_rejects_non_triangular_row(self) -> None:
        # One entry per row is neither the ragged shape an explicit tableau
        # has nor the square one an implicit tableau has.
        with pytest.raises(ValueError, match="ragged .* or square"):
            diffeq.ButcherTableau(
                a=((0.0,), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="bad"
            )

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(0.5, 0.4), c=(0.0, 1.0), order=2, name="bad"
            )

    def test_rejects_inconsistent_stage_time(self) -> None:
        with pytest.raises(ValueError, match="consistency violated"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 0.5), order=2, name="bad"
            )

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="stage counts disagree"):
            diffeq.ButcherTableau(
                a=((), (1.0,)), b=(1.0,), c=(0.0,), order=1, name="bad"
            )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            diffeq.ButcherTableau(a=(), b=(), c=(), order=1, name="bad")

    def test_rejects_bad_order_and_name(self) -> None:
        with pytest.raises(ValueError, match="order must be >= 1"):
            diffeq.ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=0, name="bad")
        with pytest.raises(ValueError, match="non-empty string"):
            diffeq.ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=1, name="")


class TestOrderOfConvergence:
    """Halving the step must shrink the global error by ``2 ** order``.

    This is the test that catches a mistyped tableau coefficient: a wrong
    entry usually still integrates, just at a lower order than advertised.
    Run in float64 — RK4 at these step counts is already near the float32
    round-off floor, which would swamp the measurement.
    """

    @pytest.mark.parametrize(("method", "order"), METHOD_ORDERS)
    def test_observed_order_matches_theory(self, method: str, order: int) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        errors = []
        for n in (10, 20, 40):
            y = diffeq.odeint(
                _decay, y0, _grid(n), method=method, return_trajectory=False
            )
            errors.append(abs(float(y.item()) - math.exp(-1.0)))

        observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for rate in observed:
            assert abs(rate - order) < 0.2, f"{method}: observed {observed}"


class TestAnalyticSolutions:
    def test_exponential_decay(self) -> None:
        y0 = lucid.tensor([2.0], dtype=lucid.float64)
        y = diffeq.odeint(_decay, y0, _grid(64), method="rk4", return_trajectory=False)
        assert abs(float(y.item()) - 2.0 * math.exp(-1.0)) < 1e-8

    def test_exponential_growth_matches_at_every_grid_point(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(64, t1=2.0)
        traj = diffeq.odeint(lambda t, y: y, y0, grid)
        values = [row[0] for row in traj.tolist()]
        for t, v in zip(grid, values):
            assert abs(v - math.exp(t)) < 1e-6

    def test_harmonic_oscillator_conserves_energy(self) -> None:
        # y = (x, v), y' = (v, -x): a unit circle in phase space, so
        # x^2 + v^2 is invariant.
        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], -y[0]], dim=0)

        y0 = lucid.tensor([1.0, 0.0], dtype=lucid.float64)
        grid = [i * 2 * math.pi / 200 for i in range(201)]
        y = diffeq.odeint(rhs, y0, grid, method="rk4", return_trajectory=False)

        x_end, v_end = y.tolist()
        assert abs(x_end**2 + v_end**2 - 1.0) < 1e-8
        # One full period returns to the start.
        assert abs(x_end - 1.0) < 1e-6
        assert abs(v_end) < 1e-6

    def test_linear_system(self) -> None:
        # y' = A y with A = [[0, 1], [0, 0]] is nilpotent, so the exact
        # solution y(t) = (y0 + t * y1, y1) is a polynomial RK4 integrates
        # without truncation error.
        a = lucid.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=lucid.float64)

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return a @ y

        y0 = lucid.tensor([[1.0], [3.0]], dtype=lucid.float64)
        y = diffeq.odeint(rhs, y0, _grid(8, t1=2.0), return_trajectory=False)
        assert y.shape == (2, 1)
        (x_end,), (v_end,) = y.tolist()
        assert abs(x_end - 7.0) < 1e-10
        assert abs(v_end - 3.0) < 1e-10


class TestTrajectory:
    def test_trajectory_shape_and_endpoints(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        grid = _grid(5)
        traj = diffeq.odeint(_decay, y0, grid)

        assert traj.shape == (len(grid), 2, 2)
        # Index 0 is y0 verbatim.
        assert traj[0].tolist() == y0.tolist()

    def test_final_only_shape(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = diffeq.odeint(_decay, y0, _grid(5), return_trajectory=False)
        assert y.shape == y0.shape

    def test_final_state_agrees_with_trajectory_tail(self) -> None:
        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64)
        grid = _grid(9)
        traj = diffeq.odeint(_decay, y0, grid)
        final = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert traj[-1].tolist() == pytest.approx(final.tolist())


class TestMethodResolution:
    @pytest.mark.parametrize("name", [m for m, _ in METHOD_ORDERS])
    def test_registered_names_run(self, name: str) -> None:
        y0 = lucid.tensor([1.0])
        y = diffeq.odeint(_decay, y0, _grid(4), method=name, return_trajectory=False)
        assert y.shape == (1,)

    def test_tableau_instance_accepted(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        by_name = diffeq.odeint(
            _decay, y0, _grid(8), method="rk4", return_trajectory=False
        )
        by_tableau = diffeq.odeint(
            _decay, y0, _grid(8), method=diffeq.RK4, return_trajectory=False
        )
        assert by_name.tolist() == pytest.approx(by_tableau.tolist())

    def test_custom_tableau_reproduces_heun2(self) -> None:
        custom = diffeq.ButcherTableau(
            a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="custom"
        )
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        got = diffeq.odeint(
            _decay, y0, _grid(8), method=custom, return_trajectory=False
        )
        want = diffeq.odeint(
            _decay, y0, _grid(8), method="heun2", return_trajectory=False
        )
        assert got.tolist() == pytest.approx(want.tolist())

    def test_unknown_method_lists_alternatives(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="unknown method"):
            diffeq.odeint(_decay, y0, _grid(2), method="radauIIA7")

    def test_non_method_type_rejected(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(TypeError, match="str or ButcherTableau"):
            diffeq.odeint(_decay, y0, _grid(2), method=4)  # type: ignore[arg-type]


class TestGrid:
    def test_tensor_grid_matches_list_grid(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(8)
        from_list = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        from_tensor = diffeq.odeint(
            _decay, y0, lucid.tensor(grid, dtype=lucid.float64), return_trajectory=False
        )
        assert from_list.tolist() == pytest.approx(from_tensor.tolist())

    def test_descending_grid_integrates_backwards(self) -> None:
        # Integrating y' = -y from t=1 back to t=0 recovers y0 * e^{+1}.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [1.0 - i / 32 for i in range(33)]
        y = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert abs(float(y.item()) - math.exp(1.0)) < 1e-7

    def test_non_uniform_spacing(self) -> None:
        # Spacing is read straight off the grid, so a stretched grid is a
        # legal (if less accurate) integration schedule.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [0.0, 0.1, 0.15, 0.5, 0.6, 1.0]
        y = diffeq.odeint(_decay, y0, grid, return_trajectory=False)
        assert abs(float(y.item()) - math.exp(-1.0)) < 1e-4

    def test_rejects_too_few_points(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="at least 2 time points"):
            diffeq.odeint(_decay, y0, [0.0])

    def test_rejects_non_monotonic(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="strictly monotonic"):
            diffeq.odeint(_decay, y0, [0.0, 0.5, 0.25, 1.0])

    def test_rejects_repeated_point(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="strictly monotonic"):
            diffeq.odeint(_decay, y0, [0.0, 0.5, 0.5, 1.0])

    def test_rejects_multidimensional_tensor_grid(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="1-D grid"):
            diffeq.odeint(_decay, y0, lucid.tensor([[0.0, 1.0]]))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_point(self, bad: float) -> None:
        # NaN compares False against everything, so without an explicit
        # check it would pass the ordering test and surface only as an
        # all-NaN result many RHS evaluations later.
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="finite time points"):
            diffeq.odeint(_decay, y0, [0.0, bad, 1.0])


class TestInputValidation:
    def test_rejects_integer_state(self) -> None:
        y0 = lucid.tensor([1, 2])
        with pytest.raises(ValueError, match="floating dtype"):
            diffeq.odeint(_decay, y0, _grid(2))

    def test_rejects_rhs_shape_drift(self) -> None:
        y0 = lucid.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="func returned shape"):
            diffeq.odeint(lambda t, y: lucid.tensor([1.0]), y0, _grid(2))

    def test_rejects_rhs_device_drift(self, device_gpu_only: str) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(ValueError, match="func returned a tensor on"):
            diffeq.odeint(lambda t, y: (-y).to(device_gpu_only), y0, _grid(2))

    def test_rejects_non_tensor_rhs(self) -> None:
        y0 = lucid.tensor([1.0])
        with pytest.raises(TypeError, match="func must return a Tensor"):
            diffeq.odeint(lambda t, y: 1.0, y0, _grid(2))  # type: ignore[arg-type,return-value]


class TestDtypeAndDevice:
    @pytest.mark.parametrize("dt", [lucid.float32, lucid.float64])
    def test_dtype_is_preserved(self, dt: lucid.dtype) -> None:
        y0 = lucid.tensor([1.0], dtype=dt)
        y = diffeq.odeint(_decay, y0, _grid(4), return_trajectory=False)
        assert y.dtype == dt

    def test_stage_time_matches_state_dtype_and_device(self) -> None:
        seen: list[tuple[object, object, tuple[int, ...]]] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            seen.append((t.dtype, t.device, t.shape))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(rhs, y0, _grid(2), method="rk4")
        assert len(seen) == 2 * 4  # 2 steps x 4 stages
        for dtype, device, shape in seen:
            assert dtype == y0.dtype
            assert device == y0.device
            assert shape == ()

    def test_stage_times_follow_the_tableau(self) -> None:
        times: list[float] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            times.append(float(t.item()))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(rhs, y0, [0.0, 1.0], method="rk4")
        assert times == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

        times.clear()
        diffeq.odeint(rhs, y0, [0.0, 1.0], method="rk4_classic")
        assert times == pytest.approx([0.0, 0.5, 0.5, 1.0])

    def test_mixed_dtype_rhs_is_promoted(self) -> None:
        # The engine op is strict about dtype, so the solver has to promote
        # exactly as ``Tensor.__add__`` does.  A float16 RHS over a float32
        # state is the autocast case, not a contrived one.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        y = diffeq.odeint(
            lambda t, yy: (-yy).to(lucid.float16),
            y0,
            _grid(4),
            return_trajectory=False,
        )
        assert y.dtype == lucid.float32

    def test_trajectory_is_uniform_after_promotion(self) -> None:
        # A float64 RHS lifts the state above y0's dtype; trajectory[0] is
        # still y0, so stacking has to reconcile them.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        traj = diffeq.odeint(lambda t, yy: (-yy).to(lucid.float64), y0, _grid(4))
        assert traj.dtype == lucid.float64
        assert traj.shape == (5, 1)

    def test_autocast_on_metal(self, device_gpu_only: str) -> None:
        import lucid.nn as nn

        net = nn.Linear(3, 3).to(device_gpu_only)
        y0 = lucid.ones(1, 3, device=device_gpu_only)
        with lucid.amp.autocast(device_type=device_gpu_only, dtype=lucid.float16):
            y = diffeq.odeint(
                lambda t, yy: net(yy), y0, _grid(4), return_trajectory=False
            )
        assert y.shape == y0.shape
        assert y.dtype == y0.dtype

    def test_runs_on_metal(self, device_gpu_only: str) -> None:
        y0 = lucid.tensor([1.0], device=device_gpu_only)
        y = diffeq.odeint(_decay, y0, _grid(8), return_trajectory=False)
        assert str(y.device) == str(y0.device)
        assert abs(float(y.item()) - math.exp(-1.0)) < 1e-5


class TestGradientFlow:
    def test_grad_through_solver(self) -> None:
        # y' = -k y integrated to t=1 gives y(1) = y0 * exp(-k), so
        # dy/dy0 = exp(-k) exactly.  This only holds if the fused op's
        # backward is wired — a fused forward alone would cut the graph.
        k = 0.5
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, y: -k * y, y0, _grid(64), method="rk4", return_trajectory=False
        )
        y.sum().backward()

        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-k), abs=1e-8)

    def test_grad_wrt_rhs_parameter(self) -> None:
        # d/dk of y0 * exp(-k) at k=0.5 is -exp(-0.5).
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            lambda t, y: -k * y, y0, _grid(64), method="rk4", return_trajectory=False
        )
        (grad_k,) = lucid.autograd.grad(y.sum(), k)

        assert grad_k is not None
        assert float(grad_k.item()) == pytest.approx(-math.exp(-0.5), abs=1e-8)

    def test_grad_through_trajectory(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        traj = diffeq.odeint(_decay, y0, _grid(32))
        traj.sum().backward()

        # Every grid point contributes exp(-t_i), including t_0 = 0.
        expected = sum(math.exp(-t) for t in _grid(32))
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(expected, abs=1e-6)

    def test_second_order_grad_through_solver(self) -> None:
        # y(1) = y0 * exp(-k), so d2/(dy0 dk) = -exp(-k).  The fused op has
        # to opt into graph-recording backward for this to work at all —
        # the unfused ``y + dt * c * k`` spelling supports it, so the
        # fusion must not silently cost the caller second-order gradients.
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        out = diffeq.odeint(lambda t, y: -k * y, y0, _grid(8), return_trajectory=False)

        (g_y0,) = lucid.autograd.grad(out.sum(), y0, create_graph=True)
        assert g_y0 is not None
        (mixed,) = lucid.autograd.grad(g_y0.sum(), k)

        assert mixed is not None
        assert float(mixed.item()) == pytest.approx(-math.exp(-0.5), abs=1e-6)

    @pytest.mark.parametrize("method", [m for m, _ in METHOD_ORDERS])
    def test_gradcheck_combine(self, method: str) -> None:
        # Finite-difference check of the fused op's backward, for every
        # tableau's final-combination weights.
        tableau = _METHODS[method]
        ks = [
            lucid.tensor([0.5 + i, -1.25 - i], dtype=lucid.float64, requires_grad=True)
            for i in range(tableau.stages)
        ]
        y0 = lucid.tensor([1.0, -2.0], dtype=lucid.float64, requires_grad=True)

        def fn(*args: lucid.Tensor) -> lucid.Tensor:
            # Squared so the gradient varies with the inputs — a linear
            # objective would pass on the scale factors alone.
            out = _combine(args[0], list(args[1:]), tableau.b, 0.3)
            return (out * out).sum()

        assert lucid.autograd.gradcheck(fn, [y0, *ks])


ADAPTIVE_METHODS = [
    "dopri5",
    "dopri8",
    "tsit5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
]


class TestAdaptiveTableaux:
    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_carries_error_and_mid_weights(self, name: str) -> None:
        tab = _METHODS[name]
        assert tab.is_adaptive
        assert tab.b_error is not None and len(tab.b_error) == tab.stages
        assert tab.mid is not None and len(tab.mid) == tab.stages

    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_error_weights_sum_to_zero(self, name: str) -> None:
        # Both embedded solutions are consistent, so their weights differ by
        # something that sums to zero.  A transcription slip in the tableau
        # almost always breaks this before it breaks anything else.
        tab = _METHODS[name]
        assert tab.b_error is not None
        assert sum(tab.b_error) == pytest.approx(0.0, abs=1e-12)

    def test_fsal_detection(self) -> None:
        # dopri5 and bosh3 end on a stage that already evaluated the new
        # state, so that derivative is reused as the next step's first stage.
        assert diffeq.DOPRI5.is_fsal
        assert diffeq.BOSH3.is_fsal
        assert not diffeq.ADAPTIVE_HEUN.is_fsal
        assert not diffeq.FEHLBERG2.is_fsal
        # tsit5 is FSAL as published.  This asserted the opposite, which
        # was true only because the fourth-order embedded weights had
        # been put in ``b``: those end on 1/66, so the last stage was not
        # the new state's derivative and the reuse was given up.
        assert diffeq.TSIT5.is_fsal
        assert not diffeq.RK4.is_adaptive

    def test_rejects_error_weights_that_do_not_cancel(self) -> None:
        with pytest.raises(ValueError, match="b_error must sum to 0"):
            diffeq.ButcherTableau(
                a=((), (1.0,)),
                b=(0.5, 0.5),
                c=(0.0, 1.0),
                order=2,
                name="bad",
                b_error=(0.5, 0.5),
                mid=(0.5, 0.0),
            )

    def test_error_weights_require_mid(self) -> None:
        with pytest.raises(ValueError, match="also needs mid"):
            diffeq.ButcherTableau(
                a=((), (1.0,)),
                b=(0.5, 0.5),
                c=(0.0, 1.0),
                order=2,
                name="bad",
                b_error=(0.5, -0.5),
            )


class TestAdaptiveIntegration:
    # What a caller reads back at an output time is limited by the
    # interpolant, not by the steps, and the interpolant's quality is not
    # uniform across these methods.  A single output time on a two-point grid
    # is reached by interpolating inside the final step, so this asks each
    # method for what its own interpolant can deliver.  The reference library
    # lands on the same values to ~1e-14, so these are properties of the
    # methods rather than of Lucid.
    ANALYTIC_TOL = {
        "dopri5": 1e-8,
        # Eighth-order steps, but the same quartic interpolant as the rest,
        # and an error estimate that at tight tolerances is small enough to be
        # dominated by cancellation -- so its step sequence, and with it the
        # distance interpolated back from the final step, is not reproducible
        # between implementations.
        "dopri8": 1e-6,
        "tsit5": 1e-8,
        "bosh3": 1e-7,
        "fehlberg2": 1e-8,
        "adaptive_heun": 1e-8,
    }

    @pytest.mark.parametrize("name", ADAPTIVE_METHODS)
    def test_reaches_analytic_solution(self, name: str) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method=name,
            rtol=1e-10,
            atol=1e-12,
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(
            math.exp(-1.0), abs=self.ANALYTIC_TOL[name]
        )

    def test_default_method_is_dopri5(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        implicit = diffeq.odeint(_decay, y0, [0.0, 1.0], return_trajectory=False)
        explicit = diffeq.odeint(
            _decay, y0, [0.0, 1.0], method="dopri5", return_trajectory=False
        )
        assert implicit.tolist() == explicit.tolist()

    def test_tightening_tolerance_reduces_error(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        errors = []
        for tol in (1e-3, 1e-6, 1e-9):
            y = diffeq.odeint(
                _decay, y0, [0.0, 1.0], rtol=tol, atol=tol, return_trajectory=False
            )
            errors.append(abs(float(y.item()) - math.exp(-1.0)))
        assert errors[0] > errors[1] > errors[2]

    def test_output_grid_does_not_change_the_answer(self) -> None:
        # The defining property of an adaptive solve: t is a set of output
        # times, so refining it must not change the value at a shared time.
        # A fixed-step solver would answer differently for each grid.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        coarse = diffeq.odeint(_decay, y0, [0.0, 1.0])
        fine = diffeq.odeint(_decay, y0, _grid(37))
        assert coarse[-1].tolist() == pytest.approx(fine[-1].tolist(), abs=1e-9)

    def test_dense_output_matches_analytic_everywhere(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(8)
        traj = diffeq.odeint(_decay, y0, grid, rtol=1e-10, atol=1e-12)
        assert traj.shape == (len(grid), 1)
        for t, row in zip(grid, traj.tolist()):
            assert row[0] == pytest.approx(math.exp(-t), abs=1e-9)

    def test_descending_grid(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay, y0, [1.0, 0.0], rtol=1e-10, atol=1e-12, return_trajectory=False
        )
        assert float(y.item()) == pytest.approx(math.exp(1.0), abs=1e-7)

    def test_harmonic_oscillator_period(self) -> None:
        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], -y[0]], dim=0)

        y0 = lucid.tensor([1.0, 0.0], dtype=lucid.float64)
        y = diffeq.odeint(
            rhs,
            y0,
            [0.0, 2 * math.pi],
            rtol=1e-11,
            atol=1e-13,
            return_trajectory=False,
        )
        x_end, v_end = y.tolist()
        assert x_end == pytest.approx(1.0, abs=1e-7)
        assert v_end == pytest.approx(0.0, abs=1e-7)

    def test_grad_flows_through_adaptive_solve(self) -> None:
        k = 0.5
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, y: -k * y,
            y0,
            [0.0, 1.0],
            rtol=1e-11,
            atol=1e-13,
            return_trajectory=False,
        )
        y.sum().backward()
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-k), abs=1e-7)

    def test_mixed_dtype_rhs_is_promoted(self) -> None:
        # Under autocast the derivatives come back below the state's
        # precision; the interpolant is built from derivatives alone in one
        # of its coefficients, so it has to promote as a group.
        y0 = lucid.tensor([1.0], dtype=lucid.float32)
        traj = diffeq.odeint(lambda t, y: (-y).to(lucid.float16), y0, _grid(4))
        assert traj.dtype == lucid.float32
        assert traj.shape == (5, 1)


class TestAdaptiveOptions:
    def test_first_step_and_max_step_are_honoured(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            options={"first_step": 0.01, "max_step": 0.05},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-8)

    def test_step_t_is_landed_on(self) -> None:
        seen: list[float] = []

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            seen.append(float(t.item()))
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint(
            rhs,
            y0,
            [0.0, 1.0],
            options={"step_t": [0.3, 0.7]},
            return_trajectory=False,
        )
        assert any(abs(t - 0.3) < 1e-12 for t in seen)
        assert any(abs(t - 0.7) < 1e-12 for t in seen)

    def test_max_num_steps_is_enforced(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(RuntimeError, match="max_num_steps"):
            diffeq.odeint(
                _decay,
                y0,
                [0.0, 1.0],
                rtol=1e-13,
                atol=1e-15,
                options={"max_num_steps": 2},
            )

    def test_rejects_unknown_option(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options={"nope": 1})

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"ifactor": 0.5}, "ifactor must be"),
            ({"dfactor": 0.0}, "dfactor must lie"),
            ({"max_num_steps": 0}, "max_num_steps must be"),
            ({"min_step": 1.0, "max_step": 0.1}, "must not exceed"),
            ({"first_step": 0.0}, "first_step must be non-zero"),
        ],
    )
    def test_rejects_out_of_range_options(
        self, bad: dict[str, float], match: str
    ) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match=match):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options=bad)

    def test_dtype_is_accepted_and_ignored(self) -> None:
        # Upstream this asks for the precision step control runs at; here that
        # is always host-double, so the key is accepted for signature
        # compatibility and has nothing to do.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            options={"dtype": lucid.float64},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-7)

    def test_norm_is_refused_rather_than_ignored(self) -> None:
        # The error norm is a fused engine kernel, so a caller-supplied one
        # cannot be honoured.  Accepting the key and quietly doing nothing --
        # which is what this used to do -- leaves the caller believing they
        # changed the step control when they did not, so it is refused.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(NotImplementedError, match="norm"):
            diffeq.odeint(
                _decay,
                y0,
                [0.0, 1.0],
                options={"norm": "rms"},
                return_trajectory=False,
            )


class TestFixedStepOptions:
    def test_step_size_decouples_the_integration_grid(self) -> None:
        # With step_size the solver walks its own grid and interpolates to t,
        # so a two-point t is as accurate as a fine one.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method="rk4",
            options={"step_size": 1 / 64, "interp": "cubic"},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-9)

    def test_cubic_interpolation_beats_linear_off_grid(self) -> None:
        # Linear interpolation caps accuracy at O(h^2) no matter how good the
        # stepper is; the cubic Hermite recovers the method's own order.  The
        # gap is the whole reason interp is an option.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [0.0, 0.3, 1.0]

        def worst(interp: str) -> float:
            traj = diffeq.odeint(
                _decay,
                y0,
                grid,
                method="rk4",
                options={"step_size": 1 / 64, "interp": interp},
            )
            return max(
                abs(row[0] - math.exp(-t)) for t, row in zip(grid, traj.tolist())
            )

        assert worst("cubic") < worst("linear") / 100

    def test_grid_constructor_overrides_step_size(self) -> None:
        seen: list[object] = []

        def build(func: object, y0_arg: object, t: object) -> list[float]:
            seen.append(t)
            return [i / 32 for i in range(33)]

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        y = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method="rk4",
            options={"grid_constructor": build, "step_size": 0.5, "interp": "cubic"},
            return_trajectory=False,
        )
        assert seen  # the constructor really was consulted
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-8)

    def test_default_grid_is_the_output_grid(self) -> None:
        # No options at all: t is the integration grid, exactly as before.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = _grid(64)
        implicit = diffeq.odeint(
            _decay, y0, grid, method="rk4", return_trajectory=False
        )
        explicit = diffeq.odeint(
            _decay, y0, grid, method="rk4", options={}, return_trajectory=False
        )
        assert implicit.tolist() == explicit.tolist()

    def test_perturb_only_nudges(self) -> None:
        # A one-ulp shift must not move the answer meaningfully; it exists so a
        # discontinuity sitting on a grid point is sampled from one side.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        plain = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method="rk4",
            options={"step_size": 1 / 64},
            return_trajectory=False,
        )
        nudged = diffeq.odeint(
            _decay,
            y0,
            [0.0, 1.0],
            method="rk4",
            options={"step_size": 1 / 64, "perturb": True},
            return_trajectory=False,
        )
        assert float(nudged.item()) == pytest.approx(float(plain.item()), abs=1e-12)

    def test_trajectory_shape_with_step_size(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0]], dtype=lucid.float64)
        grid = [0.0, 0.25, 0.5, 1.0]
        traj = diffeq.odeint(
            _decay, y0, grid, method="rk4", options={"step_size": 1 / 16}
        )
        assert traj.shape == (len(grid), 1, 2)
        assert traj[0].tolist() == y0.tolist()

    def test_rejects_adaptive_options_on_a_fixed_method(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(
                _decay, y0, _grid(4), method="rk4", options={"first_step": 0.1}
            )

    def test_rejects_fixed_options_on_an_adaptive_method(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint(_decay, y0, [0.0, 1.0], options={"step_size": 0.1})

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"step_size": 0.0}, "finite and non-zero"),
            ({"interp": "quartic"}, "interp must be one of"),
            ({"perturb": 1}, "must be a bool"),
            ({"grid_constructor": 5}, "must be callable"),
        ],
    )
    def test_rejects_out_of_range_options(
        self, bad: dict[str, object], match: str
    ) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises((ValueError, TypeError), match=match):
            diffeq.odeint(_decay, y0, _grid(4), method="rk4", options=bad)

    def test_grid_constructor_must_span_the_interval(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="must return a grid spanning"):
            diffeq.odeint(
                _decay,
                y0,
                [0.0, 1.0],
                method="rk4",
                options={"grid_constructor": lambda f, y, t: [0.0, 0.5]},
            )


class TestOdeintDense:
    def test_matches_analytic_across_the_interval(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0, rtol=1e-11, atol=1e-13)
        for i in range(101):
            t = i / 100
            assert float(dense(t).item()) == pytest.approx(math.exp(-t), abs=1e-9)

    def test_agrees_with_odeint_at_the_same_times(self) -> None:
        # The continuous solution and the grid solve run the same stepper, so
        # they must not disagree about any time.
        y0 = lucid.tensor([1.0, -0.5], dtype=lucid.float64)
        rhs = lambda t, y: -y + lucid.sin(t)  # noqa: E731
        grid = [0.0, 0.13, 0.5, 0.77, 1.0]
        traj = diffeq.odeint(rhs, y0, grid, rtol=1e-11, atol=1e-13)
        dense = diffeq.odeint_dense(rhs, y0, 0.0, 1.0, rtol=1e-11, atol=1e-13)
        for t, row in zip(grid, traj.tolist()):
            assert dense(t).tolist() == pytest.approx(row, abs=1e-9)

    def test_endpoints_are_exact(self) -> None:
        y0 = lucid.tensor([2.0, 3.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0)
        assert dense(0.0).tolist() == pytest.approx(y0.tolist(), abs=1e-12)
        final = diffeq.odeint(_decay, y0, [0.0, 1.0], return_trajectory=False)
        assert dense(1.0).tolist() == pytest.approx(final.tolist(), abs=1e-9)

    def test_backwards_interval(self) -> None:
        # Start at t=1 and integrate down; y(t) = y1 * exp(1 - t).
        y1 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y1, 1.0, 0.0, rtol=1e-11, atol=1e-13)
        for i in range(11):
            t = i / 10
            assert float(dense(t).item()) == pytest.approx(math.exp(1.0 - t), abs=1e-8)

    def test_accepts_a_tensor_query(self) -> None:
        # float64 on purpose: a float32 tensor cannot hold 0.4 exactly, so the
        # query time itself would shift by ~6e-9 and the answer with it.  The
        # callable reads whatever precision it is handed, by design.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0)
        query = lucid.tensor(0.4, dtype=lucid.float64)
        assert dense(query).tolist() == pytest.approx(dense(0.4).tolist(), abs=1e-12)

    def test_preserves_shape_and_dtype(self) -> None:
        y0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0)
        out = dense(0.5)
        assert out.shape == y0.shape
        assert out.dtype == y0.dtype

    @pytest.mark.parametrize(
        ("method", "tol", "want"),
        [
            ("dopri5", 1e-11, 1e-8),
            ("tsit5", 1e-11, 1e-8),
            ("dopri8", 1e-11, 1e-6),
            # A third-order method runs into its step floor long before a
            # fifth-order one does, so asking it for 1e-11 buys nothing.
            ("bosh3", 1e-9, 1e-4),
            ("adaptive_heun", 1e-9, 1e-6),
        ],
    )
    @pytest.mark.filterwarnings("ignore:method .*interpolates:RuntimeWarning")
    def test_every_adaptive_method(self, method: str, tol: float, want: float) -> None:
        # Two of the parametrised methods interpolate at order 1 and warn
        # about it; that is the point of the tolerances chosen above, not a
        # surprise, so the warning is filtered rather than left to accumulate.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(
            _decay, y0, 0.0, 1.0, method=method, rtol=tol, atol=tol * 100
        )
        assert float(dense(0.5).item()) == pytest.approx(math.exp(-0.5), abs=want)

    def test_fixed_method_with_step_size(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(
            _decay,
            y0,
            0.0,
            1.0,
            method="rk4",
            options={"step_size": 1 / 32, "interp": "cubic"},
        )
        for i in range(21):
            t = i / 20
            assert float(dense(t).item()) == pytest.approx(math.exp(-t), abs=1e-8)

    def test_rejects_query_outside_the_interval(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0)
        for bad in (-0.001, 1.001):
            with pytest.raises(ValueError, match="outside the solved interval"):
                dense(bad)

    def test_rejects_non_finite_query(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(_decay, y0, 0.0, 1.0)
        with pytest.raises(ValueError, match="must be finite"):
            dense(float("nan"))

    def test_rejects_degenerate_interval(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="must differ"):
            diffeq.odeint_dense(_decay, y0, 1.0, 1.0)

    def test_rejects_non_finite_bounds(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="must be finite"):
            diffeq.odeint_dense(_decay, y0, 0.0, float("inf"))

    def test_rejects_integer_state(self) -> None:
        with pytest.raises(ValueError, match="floating dtype"):
            diffeq.odeint_dense(_decay, lucid.tensor([1, 2]), 0.0, 1.0)

    def test_propagates_option_validation(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown option"):
            diffeq.odeint_dense(_decay, y0, 0.0, 1.0, options={"nope": 1})

    def test_grad_flows_through_a_query(self) -> None:
        k = 0.5
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        dense = diffeq.odeint_dense(
            lambda t, y: -k * y, y0, 0.0, 1.0, rtol=1e-11, atol=1e-13
        )
        dense(1.0).sum().backward()
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-k), abs=1e-7)

    @pytest.mark.filterwarnings("ignore:method .*interpolates:RuntimeWarning")
    def test_endpoints_are_tighter_than_interior_points(self) -> None:
        # Worth knowing rather than rediscovering: a dense query inside a step
        # is only as good as the interpolant, which is anchored on the
        # tableau's midpoint weights.  bosh3's are coarse ([0, 1/2, 0, 0]), so
        # its interior error runs well above its endpoint error; dopri5 has
        # proper mid coefficients and shows no such gap.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        dense = diffeq.odeint_dense(
            _decay, y0, 0.0, 1.0, method="bosh3", rtol=1e-11, atol=1e-13
        )
        endpoint = abs(float(dense(1.0).item()) - math.exp(-1.0))
        interior = max(
            abs(float(dense(t).item()) - math.exp(-t)) for t in (0.25, 0.5, 0.75)
        )
        assert endpoint < 1e-7
        assert interior > 5 * endpoint


class TestOdeintAdjoint:
    """``y' = -k y`` has ``y(1) = y0 exp(-k)``, so both gradients are closed form."""

    @staticmethod
    def _solve(**kwargs: object) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        ys = diffeq.odeint_adjoint(
            lambda t, y: -k * y,
            y0,
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
            adjoint_params=[k],
            **kwargs,  # type: ignore[arg-type]
        )
        return k, y0, ys

    def test_forward_matches_odeint(self) -> None:
        k, y0, ys = self._solve()
        direct = diffeq.odeint(
            lambda t, y: -k * y, y0, [0.0, 1.0], rtol=1e-12, atol=1e-14
        )
        assert _maxdiff(ys, direct) < 1e-12

    def test_gradients_match_the_closed_form(self) -> None:
        k, y0, ys = self._solve()
        ys[-1].sum().backward()
        assert y0.grad is not None and k.grad is not None
        assert float(y0.grad.item()) == pytest.approx(math.exp(-0.5), abs=1e-9)
        assert float(k.grad.item()) == pytest.approx(-math.exp(-0.5), abs=1e-9)

    def test_agrees_with_direct_differentiation(self) -> None:
        # The adjoint solves for the gradient instead of differentiating the
        # discretisation, so the two agree only up to solver tolerance.
        k_a = lucid.tensor([0.7], dtype=lucid.float64, requires_grad=True)
        y0_a = lucid.tensor([1.5], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint_adjoint(
            lambda t, y: -k_a * y + lucid.sin(t),
            y0_a,
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
            adjoint_params=[k_a],
        )[-1].sum().backward()

        k_d = lucid.tensor([0.7], dtype=lucid.float64, requires_grad=True)
        y0_d = lucid.tensor([1.5], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint(
            lambda t, y: -k_d * y + lucid.sin(t),
            y0_d,
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
        )[-1].sum().backward()

        assert float(y0_a.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            float(y0_d.grad.item()), abs=1e-7  # type: ignore[union-attr]
        )
        assert float(k_a.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            float(k_d.grad.item()), abs=1e-7  # type: ignore[union-attr]
        )

    def test_gradient_accumulates_over_every_output_time(self) -> None:
        # Summing the whole trajectory means each output time contributes;
        # dL/dy0 is then the sum of exp(-t_i).
        grid = _grid(8)
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint_adjoint(_decay, y0, grid, rtol=1e-12, atol=1e-14).sum().backward()
        assert y0.grad is not None
        assert float(y0.grad.item()) == pytest.approx(
            sum(math.exp(-t) for t in grid), abs=1e-7
        )

    def test_adjoint_settings_default_to_the_forward_ones(self) -> None:
        k, y0, ys = self._solve(adjoint_rtol=1e-12, adjoint_atol=1e-14)
        ys[-1].sum().backward()
        assert float(k.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            -math.exp(-0.5), abs=1e-9
        )

    def test_a_looser_adjoint_tolerance_costs_accuracy(self) -> None:
        # The backward solve is itself numerical, so its tolerance is what
        # controls gradient accuracy — not the forward one.
        def run(adj_tol: float) -> float:
            k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
            y0 = lucid.tensor([1.0], dtype=lucid.float64)
            diffeq.odeint_adjoint(
                lambda t, y: -k * y,
                y0,
                [0.0, 1.0],
                rtol=1e-12,
                atol=1e-14,
                adjoint_rtol=adj_tol,
                adjoint_atol=adj_tol * 1e-2,
                adjoint_params=[k],
            )[-1].sum().backward()
            return abs(float(k.grad.item()) + math.exp(-0.5))  # type: ignore[union-attr]

        assert run(1e-4) > run(1e-12)

    def test_params_default_to_the_modules_parameters(self) -> None:
        # Found by duck-typing: lucid.diffeq must not import lucid.nn.
        class Field:
            def __init__(self) -> None:
                self.k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)

            def parameters(self) -> list[lucid.Tensor]:
                return [self.k]

            def __call__(self, t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
                return -self.k * y

        field = Field()
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint_adjoint(field, y0, [0.0, 1.0], rtol=1e-12, atol=1e-14)[
            -1
        ].sum().backward()
        assert field.k.grad is not None
        assert float(field.k.grad.item()) == pytest.approx(-math.exp(-0.5), abs=1e-9)

    def test_does_not_pollute_unrelated_parameter_grads(self) -> None:
        # Regression: the adjoint calls autograd.grad once per stage, and a
        # tensor left outside its ``inputs`` has its .grad silently
        # accumulated on every one of those calls.  Passing the real
        # parameters (not detached stand-ins) is what keeps that from
        # happening — the first attempt here reported +73.996 for a gradient
        # whose true value is -0.607.  See debug-autograd-grad-leaks-into-grad.
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint_adjoint(
            lambda t, y: -k * y,
            y0,
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
            adjoint_params=[k],
        )[-1].sum().backward()
        assert float(k.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            -math.exp(-0.5), abs=1e-9
        )

    def test_works_without_any_parameters(self) -> None:
        y0 = lucid.tensor([1.0, 2.0], dtype=lucid.float64, requires_grad=True)
        diffeq.odeint_adjoint(_decay, y0, [0.0, 1.0], rtol=1e-12, atol=1e-14)[
            -1
        ].sum().backward()
        assert y0.grad is not None
        assert y0.grad.tolist() == pytest.approx([math.exp(-1.0)] * 2, abs=1e-9)

    def test_multidimensional_state(self) -> None:
        y0 = lucid.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=lucid.float64, requires_grad=True
        )
        ys = diffeq.odeint_adjoint(_decay, y0, [0.0, 1.0], rtol=1e-12, atol=1e-14)
        assert ys.shape == (2, 2, 2)
        ys[-1].sum().backward()
        assert y0.grad is not None
        assert _maxdiff(y0.grad, lucid.full_like(y0, math.exp(-1.0))) < 1e-9

    def test_rejects_event_fn(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(NotImplementedError, match="event_fn"):
            diffeq.odeint_adjoint(_decay, y0, [0.0, 1.0], event_fn=lambda t, y: y)

    def test_rejects_non_tensor_params(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(TypeError, match="must be a Tensor"):
            diffeq.odeint_adjoint(
                _decay, y0, [0.0, 1.0], adjoint_params=[1.0]  # type: ignore[list-item]
            )

    def test_rejects_bad_method_before_integrating(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="unknown method"):
            diffeq.odeint_adjoint(_decay, y0, [0.0, 1.0], method="radauIIA7")


def _fall(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    """Free fall: state is (height, velocity), gravity 9.8."""
    return lucid.stack([y[1], lucid.tensor(-9.8, dtype=y.dtype)], dim=0)


class TestEventTimeGradient:
    """The event time is differentiable through the implicit relation.

    Bisection itself has no derivative -- it compares signs -- so every case
    here is checking the rerouting, not the search.  Each has a closed form
    to compare against, because a gradient that is merely plausible is the
    thing this whole mechanism is most likely to produce.
    """

    def test_matches_the_closed_form_through_the_initial_state(self) -> None:
        # Dropped from height h under gravity: t* = sqrt(2h/g), so
        # dt*/dh = 1 / sqrt(2 h g).
        h_val, g = 10.0, 9.8
        h = lucid.tensor([h_val], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.cat([h, lucid.tensor([0.0], dtype=lucid.float64)], dim=0)

        event_t, _ = diffeq.odeint_event(
            _fall, y0, 0.0, event_fn=lambda t, y: y[0], rtol=1e-12, atol=1e-14
        )
        assert event_t.requires_grad
        event_t.backward()

        assert h.grad is not None
        assert float(h.grad.item()) == pytest.approx(
            1.0 / math.sqrt(2 * h_val * g), rel=1e-9
        )

    def test_matches_the_closed_form_through_a_parameter(self) -> None:
        # y' = -k y from 1, event at y = 1/2: t* = ln 2 / k, dt*/dk = -ln 2 / k^2.
        k = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        event_t, _ = diffeq.odeint_event(
            lambda t, y: -k * y,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            event_fn=lambda t, y: y[0] - 0.5,
            rtol=1e-12,
            atol=1e-14,
        )
        event_t.backward()

        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(-math.log(2) / 4.0, rel=1e-9)

    def test_accounts_for_the_event_function_depending_on_time(self) -> None:
        # g(t, y) = y - t, so the denominator picks up a dg/dt term that a
        # state-only derivation would drop.  Here t* solves e^{-k t} = t and
        # dt*/dk = -t* e^{-k t*} / (1 + k e^{-k t*}).
        k_val = 2.0
        k = lucid.tensor([k_val], dtype=lucid.float64, requires_grad=True)
        event_t, _ = diffeq.odeint_event(
            lambda t, y: -k * y,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            event_fn=lambda t, y: y[0] - t,
            rtol=1e-12,
            atol=1e-14,
        )
        t_star = float(event_t.item())
        event_t.backward()

        decay = math.exp(-k_val * t_star)
        want = -t_star * decay / (1.0 + k_val * decay)
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(want, rel=1e-8)

    def test_the_state_at_the_event_moves_with_the_event_time(self) -> None:
        # The state is read off the interpolant at a host float, so nothing
        # records that shifting t* shifts it too.  Differentiating the
        # solution as well as the time exercises that correction; without it
        # this gradient is short by grad_state . f.
        def solve(k_val: float) -> tuple[lucid.Tensor, lucid.Tensor]:
            k = lucid.tensor([k_val], dtype=lucid.float64, requires_grad=True)
            event_t, sol = diffeq.odeint_event(
                lambda t, y: -k * y,
                lucid.tensor([1.0], dtype=lucid.float64),
                0.0,
                event_fn=lambda t, y: y[0] - 0.5,
                rtol=1e-13,
                atol=1e-15,
            )
            return k, event_t + sol.sum()

        k, loss = solve(2.0)
        loss.backward()

        eps = 1e-6
        hi = float(solve(2.0 + eps)[1].item())
        lo = float(solve(2.0 - eps)[1].item())
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx((hi - lo) / (2 * eps), rel=1e-5)

    def test_reverse_time_carries_the_gradient_too(self) -> None:
        # Backwards to y = 2: t* = -ln 2 / k, so the derivative flips sign.
        k = lucid.tensor([2.0], dtype=lucid.float64, requires_grad=True)
        event_t, _ = diffeq.odeint_event(
            lambda t, y: -k * y,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            event_fn=lambda t, y: y[0] - 2.0,
            reverse_time=True,
            rtol=1e-12,
            atol=1e-14,
        )
        event_t.backward()

        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(math.log(2) / 4.0, rel=1e-8)

    def test_odeint_with_event_fn_is_differentiable_as_well(self) -> None:
        # The same rerouting has to reach the tuple-returning form of odeint,
        # not just the odeint_event convenience wrapper.
        a = lucid.tensor([1.5], dtype=lucid.float64, requires_grad=True)
        event_t, _ = diffeq.odeint(
            _decay,
            a,
            [0.0, 1.0],
            event_fn=lambda t, y: y[0] - 1.0,
            rtol=1e-12,
            atol=1e-14,
        )
        event_t.backward()

        # y = a e^{-t} reaches 1 at t* = ln a, so dt*/da = 1/a.
        assert a.grad is not None
        assert float(a.grad.item()) == pytest.approx(1.0 / 1.5, rel=1e-9)

    def test_costs_nothing_when_no_gradient_is_wanted(self) -> None:
        # The rerouting evaluates the right-hand side, so it must not run at
        # all on a graph-free solve.
        calls = [0]

        def counted(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            calls[0] += 1
            return -y

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, _ = diffeq.odeint_event(
            counted, y0, 0.0, event_fn=lambda t, y: y[0] - 0.5
        )
        assert not event_t.requires_grad
        before = calls[0]

        y0_grad = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        calls[0] = 0
        diffeq.odeint_event(counted, y0_grad, 0.0, event_fn=lambda t, y: y[0] - 0.5)
        # One extra call: the slope used by the state's correction term.
        assert calls[0] == before + 1


class TestOdeintEvent:
    def test_falling_body_hits_the_ground_at_the_analytic_time(self) -> None:
        y0 = lucid.tensor([10.0, 0.0], dtype=lucid.float64)
        event_t, sol = diffeq.odeint_event(
            _fall, y0, 0.0, event_fn=lambda t, y: y[0], rtol=1e-12, atol=1e-14
        )
        assert float(event_t.item()) == pytest.approx(
            math.sqrt(2 * 10.0 / 9.8), abs=1e-9
        )
        assert sol.shape == (2, 2)
        # Height is zero there, velocity is -g*t.
        assert sol[-1].tolist()[0] == pytest.approx(0.0, abs=1e-8)
        assert sol[-1].tolist()[1] == pytest.approx(-14.0, abs=1e-7)

    def test_solution_starts_at_y0(self) -> None:
        y0 = lucid.tensor([10.0, 0.0], dtype=lucid.float64)
        _, sol = diffeq.odeint_event(_fall, y0, 0.0, event_fn=lambda t, y: y[0])
        assert sol[0].tolist() == y0.tolist()

    def test_odeint_takes_event_fn_directly(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, sol = diffeq.odeint(
            _decay,
            y0,
            [0.0, 5.0],
            event_fn=lambda t, y: y[0] - 0.5,
            rtol=1e-12,
            atol=1e-14,
        )
        assert float(event_t.item()) == pytest.approx(math.log(2.0), abs=1e-9)
        assert sol.shape == (2, 1)

    def test_only_the_first_time_matters(self) -> None:
        # The grid stops being an output grid; only t[0] and the direction
        # survive, so a different end time must not change the answer.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        fn = lambda t, y: y[0] - 0.5  # noqa: E731
        a, _ = diffeq.odeint(
            _decay, y0, [0.0, 5.0], event_fn=fn, rtol=1e-12, atol=1e-14
        )
        b, _ = diffeq.odeint(
            _decay, y0, [0.0, 0.9], event_fn=fn, rtol=1e-12, atol=1e-14
        )
        assert float(a.item()) == pytest.approx(float(b.item()), abs=1e-12)

    def test_reverse_time(self) -> None:
        # y' = -y backwards from y(0)=1 reaches 2 at t = -ln 2.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, _ = diffeq.odeint_event(
            _decay,
            y0,
            0.0,
            event_fn=lambda t, y: y[0] - 2.0,
            reverse_time=True,
            rtol=1e-12,
            atol=1e-14,
        )
        assert float(event_t.item()) == pytest.approx(-math.log(2.0), abs=1e-8)

    @pytest.mark.parametrize("method", ["dopri5", "tsit5", "bosh3"])
    @pytest.mark.filterwarnings("ignore:method .*interpolates:RuntimeWarning")
    def test_every_adaptive_method_finds_it(self, method: str) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, _ = diffeq.odeint_event(
            _decay,
            y0,
            0.0,
            event_fn=lambda t, y: y[0] - 0.5,
            method=method,
            rtol=1e-11,
            atol=1e-13,
        )
        assert float(event_t.item()) == pytest.approx(math.log(2.0), abs=1e-7)

    def test_fixed_method_with_step_size(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, _ = diffeq.odeint_event(
            _decay,
            y0,
            0.0,
            event_fn=lambda t, y: y[0] - 0.5,
            method="rk4",
            options={"step_size": 0.05, "interp": "cubic"},
        )
        assert float(event_t.item()) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_fixed_method_needs_a_step_size(self) -> None:
        # There is no end time to build a grid from, so the caller must say
        # how big a step to take.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="step_size"):
            diffeq.odeint_event(
                _decay, y0, 0.0, event_fn=lambda t, y: y[0] - 0.5, method="rk4"
            )

    def test_event_already_satisfied_fires_immediately(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        event_t, sol = diffeq.odeint_event(
            _decay, y0, 0.0, event_fn=lambda t, y: y[0] - 1.0
        )
        assert float(event_t.item()) == 0.0
        assert sol[-1].tolist() == y0.tolist()

    def test_tightening_tolerance_sharpens_the_event_time(self) -> None:
        # Bisection narrows the bracket to machine precision, so what limits
        # the answer is the interpolant — i.e. the solver tolerance.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        fn = lambda t, y: y[0] - 0.5  # noqa: E731

        def err(tol: float) -> float:
            event_t, _ = diffeq.odeint_event(
                _decay, y0, 0.0, event_fn=fn, rtol=tol, atol=tol * 1e-2
            )
            return abs(float(event_t.item()) - math.log(2.0))

        assert err(1e-4) > err(1e-12)

    def test_gradient_flows_through_the_event_state(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        _, sol = diffeq.odeint_event(
            _decay,
            y0,
            0.0,
            event_fn=lambda t, y: y[0] - 0.5,
            rtol=1e-12,
            atol=1e-14,
        )
        sol[-1].sum().backward()
        assert y0.grad is not None
        # Zero, and not by accident: the event fires exactly when y reaches
        # 1/2, so the state there is 1/2 whatever y0 was.  Raising y0 only
        # moves the event later, and the two effects cancel exactly.
        #
        # Holding the event time fixed instead gives 1/2, which is what this
        # asserted while the event time carried no gradient.  The whole point
        # of the implicit-function rerouting is that the time is no longer
        # held fixed, so the total derivative is what a caller now gets.
        assert float(y0.grad.item()) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_a_non_scalar_event_fn(self) -> None:
        y0 = lucid.tensor([1.0, 2.0], dtype=lucid.float64)
        with pytest.raises(ValueError, match="single-element"):
            diffeq.odeint_event(_decay, y0, 0.0, event_fn=lambda t, y: y)

    def test_rejects_a_non_tensor_event_fn(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(TypeError, match="event_fn must return a Tensor"):
            diffeq.odeint_event(_decay, y0, 0.0, event_fn=lambda t, y: 1.0)

    def test_respects_max_num_steps_when_the_event_never_fires(self) -> None:
        # y decays towards 0 and never reaches 2 going forwards.  The
        # controller's own budget is the guard a caller reaches for.
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(RuntimeError, match="max_num_steps"):
            diffeq.odeint_event(
                _decay,
                y0,
                0.0,
                event_fn=lambda t, y: y[0] - 2.0,
                options={"max_step": 1e-2, "max_num_steps": 50},
            )

    def test_step_budget_backstops_an_unbounded_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without any budget in options the search would run forever, so
        # there is a backstop.  Patched down here rather than paying for
        # the real ceiling.
        from lucid.diffeq import _event

        monkeypatch.setattr(_event, "_MAX_EVENT_STEPS", 20)
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        with pytest.raises(RuntimeError, match="did not change sign"):
            diffeq.odeint_event(
                _decay,
                y0,
                0.0,
                event_fn=lambda t, y: y[0] - 2.0,
                options={"max_step": 1e-2},
            )

    def test_accepts_a_tensor_start_time(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        a, _ = diffeq.odeint_event(
            _decay,
            y0,
            lucid.tensor(0.0, dtype=lucid.float64),
            event_fn=lambda t, y: y[0] - 0.5,
        )
        b, _ = diffeq.odeint_event(_decay, y0, 0.0, event_fn=lambda t, y: y[0] - 0.5)
        assert float(a.item()) == pytest.approx(float(b.item()), abs=1e-12)

    def test_delegates_to_a_custom_interface(self) -> None:
        seen: list[str] = []

        def spy(*args: object, **kwargs: object) -> object:
            seen.append("called")
            return diffeq.odeint(*args, **kwargs)  # type: ignore[arg-type]

        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        diffeq.odeint_event(
            _decay, y0, 0.0, event_fn=lambda t, y: y[0] - 0.5, odeint_interface=spy
        )
        assert seen == ["called"]


class TestInterpolantQuality:
    """Dense output is only as good as the tableau's midpoint weights.

    Worth stating outright because it is not what you would assume: a
    method's interpolant does *not* inherit its step accuracy.  Everything
    read off the interpolant — ``odeint`` at an off-step output time,
    ``odeint_dense``, and the event time from ``odeint_event`` — inherits
    the midpoint quality instead.
    """

    @staticmethod
    def _midpoint_height(method: str, dt: float) -> float:
        """Height the tableau's ``mid`` weights predict for a free fall."""
        from lucid.diffeq import _fused
        from lucid.diffeq._tableau import _METHODS

        gravity = 9.8
        y0 = lucid.tensor([12.5, 0.0], dtype=lucid.float64)

        def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.stack([y[1], lucid.tensor(-gravity, dtype=y.dtype)], dim=0)

        tableau = _METHODS[method]
        ks = [rhs(lucid.tensor(0.0, dtype=lucid.float64), y0)]
        for stage in range(1, tableau.stages):
            stage_y = _fused.combine(y0, ks, tableau.a[stage], dt)
            ks.append(
                rhs(lucid.tensor(tableau.c[stage] * dt, dtype=lucid.float64), stage_y)
            )
        assert tableau.mid is not None
        return _fused.combine(y0, ks, tableau.mid, dt).tolist()[0]

    def test_midpoint_accuracy_varies_by_tableau(self) -> None:
        # Free-fall height is exactly quadratic, so an accurate midpoint is
        # reproduced to round-off and a first-order one is not.
        dt = 1.0
        exact = 12.5 - 0.5 * 9.8 * (dt / 2) ** 2

        for method in ("dopri5", "tsit5"):
            assert self._midpoint_height(method, dt) == pytest.approx(exact, abs=1e-12)

        # These carry first-order mid weights upstream; the gap is real and
        # is why they are excluded from event-time parity.
        for method in ("bosh3", "fehlberg2"):
            assert abs(self._midpoint_height(method, dt) - exact) > 1.0
        # adaptive_heun's midpoint estimate is just y0 for this problem.
        assert self._midpoint_height("adaptive_heun", dt) == pytest.approx(12.5)

    def test_event_time_is_accurate_for_the_good_tableaux(self) -> None:
        exact = math.sqrt(2 * 12.5 / 9.8)
        y0 = lucid.tensor([12.5, 0.0], dtype=lucid.float64)
        for method in ("dopri5", "tsit5"):
            event_t, _ = diffeq.odeint_event(
                _fall,
                y0,
                0.0,
                event_fn=lambda t, y: y[0],
                method=method,
                rtol=1e-12,
                atol=1e-14,
            )
            assert float(event_t.item()) == pytest.approx(exact, abs=1e-9)


def _osc_tuple(
    t: lucid.Tensor, y: tuple[lucid.Tensor, ...]
) -> tuple[lucid.Tensor, ...]:
    """Harmonic oscillator over a two-component state of differing shapes."""
    pos, vel = y
    return (vel.reshape(1, 2), -pos.reshape(2))


class TestTupleState:
    """``y0`` may be a tuple of differently-shaped tensors.

    The solvers integrate one flat vector either way — the tuple is packed on
    the way in and split on the way out — so this is about the boundary, not
    about a second solver.
    """

    @staticmethod
    def _initial() -> tuple[lucid.Tensor, lucid.Tensor]:
        return (
            lucid.tensor([[1.0, 0.0]], dtype=lucid.float64),
            lucid.tensor([0.0, 1.0], dtype=lucid.float64),
        )

    def test_trajectory_keeps_component_shapes(self) -> None:
        y0 = self._initial()
        traj = diffeq.odeint(_osc_tuple, y0, [0.0, math.pi / 2], rtol=1e-12, atol=1e-14)
        assert isinstance(traj, tuple) and len(traj) == 2
        assert traj[0].shape == (2, 1, 2)
        assert traj[1].shape == (2, 2)

    def test_matches_the_analytic_oscillator(self) -> None:
        # p(t) = p0 cos t + v0 sin t, v(t) = -p0 sin t + v0 cos t.  A quarter
        # period swaps them.
        y0 = self._initial()
        traj = diffeq.odeint(_osc_tuple, y0, [0.0, math.pi / 2], rtol=1e-12, atol=1e-14)
        assert traj[0].tolist()[-1][0] == pytest.approx([0.0, 1.0], abs=1e-9)
        assert traj[1].tolist()[-1] == pytest.approx([-1.0, 0.0], abs=1e-9)

    def test_first_entry_is_y0(self) -> None:
        y0 = self._initial()
        traj = diffeq.odeint(_osc_tuple, y0, [0.0, 1.0])
        assert traj[0].tolist()[0] == y0[0].tolist()
        assert traj[1].tolist()[0] == y0[1].tolist()

    def test_final_only_keeps_component_shapes(self) -> None:
        y0 = self._initial()
        out = diffeq.odeint(_osc_tuple, y0, [0.0, 1.0], return_trajectory=False)
        assert isinstance(out, tuple)
        assert [tuple(x.shape) for x in out] == [(1, 2), (2,)]

    def test_matches_the_equivalent_flat_solve(self) -> None:
        # Packing is supposed to be invisible: the same problem written as one
        # flat tensor must give the same numbers.
        y0 = self._initial()
        traj = diffeq.odeint(_osc_tuple, y0, [0.0, 1.0], rtol=1e-12, atol=1e-14)

        flat0 = lucid.tensor([1.0, 0.0, 0.0, 1.0], dtype=lucid.float64)

        def flat_rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return lucid.concat([y[2:4], -y[0:2]])

        flat = diffeq.odeint(flat_rhs, flat0, [0.0, 1.0], rtol=1e-12, atol=1e-14)
        assert traj[0].tolist()[-1][0] == pytest.approx(
            flat.tolist()[-1][:2], abs=1e-10
        )
        assert traj[1].tolist()[-1] == pytest.approx(flat.tolist()[-1][2:], abs=1e-10)

    def test_fixed_step_method(self) -> None:
        y0 = self._initial()
        traj = diffeq.odeint(_osc_tuple, y0, _grid(64, t1=math.pi / 2), method="rk4")
        assert traj[0].tolist()[-1][0] == pytest.approx([0.0, 1.0], abs=1e-8)

    def test_gradient_reaches_every_component(self) -> None:
        a0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        b0 = lucid.tensor([[2.0]], dtype=lucid.float64, requires_grad=True)

        def rhs(
            t: lucid.Tensor, y: tuple[lucid.Tensor, ...]
        ) -> tuple[lucid.Tensor, ...]:
            return (-y[0], -y[1])

        traj = diffeq.odeint(rhs, (a0, b0), [0.0, 1.0], rtol=1e-12, atol=1e-14)
        (traj[0][-1].sum() + traj[1][-1].sum()).backward()
        assert a0.grad is not None and b0.grad is not None
        assert float(a0.grad.item()) == pytest.approx(math.exp(-1.0), abs=1e-9)
        assert float(b0.grad.item()) == pytest.approx(math.exp(-1.0), abs=1e-9)

    def test_dense_output(self) -> None:
        y0 = self._initial()
        dense = diffeq.odeint_dense(
            _osc_tuple, y0, 0.0, math.pi / 2, rtol=1e-12, atol=1e-14
        )
        out = dense(math.pi / 2)
        assert isinstance(out, tuple)
        assert out[0].tolist()[0] == pytest.approx([0.0, 1.0], abs=1e-9)

    def test_event(self) -> None:
        y0 = self._initial()
        event_t, sol = diffeq.odeint_event(
            _osc_tuple,
            y0,
            0.0,
            event_fn=lambda t, y: y[0].reshape(2)[0],
            rtol=1e-12,
            atol=1e-14,
        )
        assert float(event_t.item()) == pytest.approx(math.pi / 2, abs=1e-8)
        assert isinstance(sol, tuple)
        assert [tuple(x.shape) for x in sol] == [(2, 1, 2), (2, 2)]

    def test_adjoint(self) -> None:
        k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
        a0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        b0 = lucid.tensor([[2.0]], dtype=lucid.float64, requires_grad=True)

        def rhs(
            t: lucid.Tensor, y: tuple[lucid.Tensor, ...]
        ) -> tuple[lucid.Tensor, ...]:
            return (-k * y[0], -k * y[1])

        out = diffeq.odeint_adjoint(
            rhs, (a0, b0), [0.0, 1.0], rtol=1e-12, atol=1e-14, adjoint_params=[k]
        )
        assert isinstance(out, tuple)
        (out[0][-1].sum() + out[1][-1].sum()).backward()
        assert float(a0.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            math.exp(-0.5), abs=1e-8
        )
        # d/dk of (a0 + b0) exp(-k) at k=0.5.
        assert float(k.grad.item()) == pytest.approx(  # type: ignore[union-attr]
            -3.0 * math.exp(-0.5), abs=1e-8
        )

    @pytest.mark.parametrize(
        ("bad", "exc", "match"),
        [
            ((), ValueError, "must not be empty"),
            (("x",), TypeError, "y0\\[0\\] must be a Tensor"),
            (123, TypeError, "Tensor or a tuple"),
        ],
    )
    def test_rejects_a_malformed_state(
        self, bad: object, exc: type[Exception], match: str
    ) -> None:
        with pytest.raises(exc, match=match):
            diffeq.odeint(_osc_tuple, bad, [0.0, 1.0])  # type: ignore[arg-type]

    def test_rejects_a_rhs_that_does_not_return_a_tuple(self) -> None:
        y0 = self._initial()
        with pytest.raises(TypeError, match="must return a tuple of tensors"):
            diffeq.odeint(lambda t, y: y[0], y0, [0.0, 1.0])

    def test_rejects_a_component_count_mismatch(self) -> None:
        y0 = self._initial()
        with pytest.raises(ValueError, match="returned 1 components but y0 has 2"):
            diffeq.odeint(lambda t, y: (y[0],), y0, [0.0, 1.0])


class TestAdamsCoefficients:
    """The Adams weights are derived, so the derivation itself needs a check.

    Deriving beats transcribing a wall of integer tables, but only if the
    derivation is right — hence the published low-order values below.
    """

    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (1, [1.0]),
            (2, [3 / 2, -1 / 2]),
            (3, [23 / 12, -16 / 12, 5 / 12]),
            (4, [55 / 24, -59 / 24, 37 / 24, -9 / 24]),
            (5, [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720]),
        ],
    )
    def test_bashforth_matches_the_published_table(
        self, order: int, expected: list[float]
    ) -> None:
        got = _multistep.coefficients(order, implicit=False)
        assert list(got) == pytest.approx(expected, rel=1e-14)

    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (1, [1.0]),
            (2, [1 / 2, 1 / 2]),
            (3, [5 / 12, 8 / 12, -1 / 12]),
            (4, [9 / 24, 19 / 24, -5 / 24, 1 / 24]),
            (5, [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]),
        ],
    )
    def test_moulton_matches_the_published_table(
        self, order: int, expected: list[float]
    ) -> None:
        got = _multistep.coefficients(order, implicit=True)
        assert list(got) == pytest.approx(expected, rel=1e-14)

    def test_weights_are_memoised(self) -> None:
        # The exact-rational derivation is slow enough that recomputing it per
        # step dominated the whole solve (96% of a 2000-step run) before this
        # cache existed.  Identity, not equality: a fresh tuple means the
        # derivation ran again.
        first = _multistep.coefficients(12, implicit=True)
        assert _multistep.coefficients(12, implicit=True) is first

    @pytest.mark.parametrize("order", range(1, 13))
    @pytest.mark.parametrize("implicit", [False, True])
    def test_weights_sum_to_one(self, order: int, implicit: bool) -> None:
        # Consistency: a constant derivative must advance the state by exactly
        # dt.  Any order that fails this is not an Adams method at all.
        got = _multistep.coefficients(order, implicit=implicit)
        assert sum(got) == pytest.approx(1.0, abs=1e-12)


class TestMultistep:
    """Adams methods, reachable through ``odeint`` like any other."""

    DECAY_EXACT = math.exp(-1.0)

    @staticmethod
    def _decay(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
        return -y

    @staticmethod
    def _uniform(n: int) -> list[float]:
        return [i / n for i in range(n + 1)]

    @pytest.mark.parametrize(
        "method", ["explicit_adams", "implicit_adams", "fixed_adams"]
    )
    def test_solves_exponential_decay(self, method: str) -> None:
        y = diffeq.odeint(
            self._decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            self._uniform(200),
            method=method,
            options={"max_order": 5},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(self.DECAY_EXACT, abs=1e-11)

    def test_implicit_and_fixed_adams_name_the_same_solver(self) -> None:
        y0 = lucid.tensor([1.0, 2.0], dtype=lucid.float64)
        grid = self._uniform(50)
        a = diffeq.odeint(self._decay, y0, grid, method="implicit_adams")
        b = diffeq.odeint(self._decay, y0, grid, method="fixed_adams")
        assert _maxdiff(a, b) == 0.0

    @pytest.mark.parametrize("order", [4, 5])
    def test_convergence_order(self, order: int) -> None:
        # Halving the step must shrink the error by ~2**order.  The RK4
        # startup contributes its own O(h^5), so above order 5 the observed
        # rate saturates and this check would stop discriminating.
        errs = []
        for n in (100, 200):
            y = diffeq.odeint(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                self._uniform(n),
                method="explicit_adams",
                options={"max_order": order},
                return_trajectory=False,
            )
            errs.append(abs(float(y.item()) - self.DECAY_EXACT))
        observed = math.log2(errs[0] / errs[1])
        assert observed == pytest.approx(order, abs=0.5)

    def test_corrector_beats_the_bare_predictor(self) -> None:
        grid = self._uniform(60)
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        opts = {"max_order": 5}
        pred = diffeq.odeint(
            self._decay,
            y0,
            grid,
            method="explicit_adams",
            options=opts,
            return_trajectory=False,
        )
        corr = diffeq.odeint(
            self._decay,
            y0,
            grid,
            method="implicit_adams",
            options=opts,
            return_trajectory=False,
        )
        assert abs(float(corr.item()) - self.DECAY_EXACT) < abs(
            float(pred.item()) - self.DECAY_EXACT
        )

    def test_a_non_uniform_grid_stays_accurate(self) -> None:
        # The weights assume even spacing, so an uneven grid has to fall back
        # to Runge-Kutta rather than apply them anyway.
        grid = [0.0, 0.1, 0.15, 0.4, 0.5, 0.62, 0.8, 1.0]
        y = diffeq.odeint(
            self._decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            grid,
            method="explicit_adams",
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(self.DECAY_EXACT, abs=1e-4)

    def test_step_size_decouples_the_output_grid(self) -> None:
        y = diffeq.odeint(
            self._decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            [0.0, 0.37, 1.0],
            method="implicit_adams",
            options={"step_size": 0.005, "max_order": 5},
        )
        got = y.tolist()
        assert got[1][0] == pytest.approx(math.exp(-0.37), abs=1e-6)
        assert got[2][0] == pytest.approx(self.DECAY_EXACT, abs=1e-11)

    @pytest.mark.parametrize("interp", ["linear", "cubic"])
    def test_interp_option_applies(self, interp: str) -> None:
        y = diffeq.odeint(
            self._decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            [0.0, 0.37, 1.0],
            method="implicit_adams",
            options={"step_size": 0.01, "interp": interp, "max_order": 5},
        )
        assert y.tolist()[1][0] == pytest.approx(math.exp(-0.37), abs=1e-4)

    def test_first_entry_is_y0(self) -> None:
        y0 = lucid.tensor([1.0, -3.0], dtype=lucid.float64)
        traj = diffeq.odeint(self._decay, y0, self._uniform(20), method="fixed_adams")
        assert traj.tolist()[0] == y0.tolist()

    def test_integrates_backwards(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [1.0 - i / 200 for i in range(201)]
        y = diffeq.odeint(
            self._decay,
            y0,
            grid,
            method="implicit_adams",
            options={"max_order": 5},
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(math.e, abs=1e-9)

    @pytest.mark.parametrize("method", ["explicit_adams", "implicit_adams"])
    def test_is_differentiable(self, method: str) -> None:
        k = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, s: -k * s,
            lucid.tensor([1.0], dtype=lucid.float64),
            self._uniform(100),
            method=method,
            options={"max_order": 5},
            return_trajectory=False,
        )
        y.sum().backward()
        assert k.grad is not None
        # d/dk exp(-k) at k=1.
        assert float(k.grad.item()) == pytest.approx(-math.exp(-1.0), abs=1e-8)

    def test_works_under_the_adjoint(self) -> None:
        k = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        traj = diffeq.odeint_adjoint(
            lambda t, s: -k * s,
            lucid.tensor([1.0], dtype=lucid.float64),
            self._uniform(100),
            method="implicit_adams",
            options={"max_order": 5},
            adjoint_params=[k],
        )
        traj[-1].sum().backward()
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(-math.exp(-1.0), abs=1e-7)

    def test_accepts_a_tuple_state(self) -> None:
        y0 = (
            lucid.tensor([[1.0, 0.0]], dtype=lucid.float64),
            lucid.tensor([0.0, 1.0], dtype=lucid.float64),
        )
        grid = [i * (math.pi / 2) / 400 for i in range(401)]
        traj = diffeq.odeint(
            _osc_tuple, y0, grid, method="fixed_adams", options={"max_order": 5}
        )
        assert [tuple(x.shape) for x in traj] == [(401, 1, 2), (401, 2)]
        assert traj[0].tolist()[-1][0] == pytest.approx([0.0, 1.0], abs=1e-9)

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"max_order": 0}, r"max_order must lie in \[1, 12\]"),
            ({"max_order": 13}, r"max_order must lie in \[1, 12\]"),
            ({"max_order": 2.5}, "must be an int"),
            ({"max_iters": 0}, "max_iters must be >= 1"),
            ({"nope": 1}, "unknown option"),
        ],
    )
    def test_rejects_bad_options(self, options: dict[str, object], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            diffeq.odeint(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                [0.0, 1.0],
                method="implicit_adams",
                options=options,
            )

    def test_the_unknown_method_error_lists_the_adams_names(self) -> None:
        with pytest.raises(ValueError, match="explicit_adams"):
            diffeq.odeint(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                [0.0, 1.0],
                method="nope",
            )

    @pytest.mark.parametrize(
        "method", ["explicit_adams", "implicit_adams", "fixed_adams"]
    )
    def test_dense_output_refuses_an_adams_method(self, method: str) -> None:
        # Better a clear refusal than a quietly cruder interpolant.
        with pytest.raises(NotImplementedError, match="no dense output"):
            diffeq.odeint_dense(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                0.0,
                1.0,
                method=method,
            )

    def test_event_detection_refuses_an_adams_method(self) -> None:
        with pytest.raises(NotImplementedError, match="no dense output"):
            diffeq.odeint_event(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                0.0,
                event_fn=lambda t, y: y[0] - 0.5,
                method="fixed_adams",
            )


IMPLICIT_ORDERS = [
    ("implicit_euler", 1),
    ("implicit_midpoint", 2),
    ("trapezoid", 2),
    ("radauIIA3", 3),
    ("gl4", 4),
    ("radauIIA5", 5),
    ("gl6", 6),
    ("sdirk2", 2),
    ("trbdf2", 2),
]


class TestDerivedTableaux:
    """The implicit tableaux are derived at import, so the derivation is tested.

    A mistyped coefficient does not raise — it quietly costs an order, which
    is the failure mode these checks exist to catch.
    """

    @pytest.mark.parametrize(("name", "order"), IMPLICIT_ORDERS)
    def test_reaches_its_stated_order(self, name: str, order: int) -> None:
        tableau = _METHODS[name]
        assert _collocation.quadrature_order(tableau.b, tableau.c) == order

    @pytest.mark.parametrize(("name", "_order"), IMPLICIT_ORDERS)
    def test_rows_are_consistent(self, name: str, _order: int) -> None:
        # c[i] == sum(a[i]) is what makes the stage times agree with the stage
        # states; ButcherTableau enforces it, so this pins the derivation.
        tableau = _METHODS[name]
        for i, row in enumerate(tableau.a):
            assert sum(row) == pytest.approx(tableau.c[i], abs=1e-12)
        assert sum(tableau.b) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(("name", "_order"), IMPLICIT_ORDERS)
    def test_is_classified_implicit(self, name: str, _order: int) -> None:
        assert _METHODS[name].is_implicit

    @pytest.mark.parametrize(
        ("name", "dirk"),
        [
            ("implicit_euler", True),
            ("implicit_midpoint", True),
            ("trapezoid", True),
            ("sdirk2", True),
            ("trbdf2", True),
            ("radauIIA3", False),
            ("radauIIA5", False),
            ("gl4", False),
            ("gl6", False),
        ],
    )
    def test_dirk_classification(self, name: str, dirk: bool) -> None:
        # Decides sequential versus coupled stage solves, so getting it wrong
        # is the difference between an n-by-n Jacobian and an sn-by-sn one.
        assert _METHODS[name].is_dirk is dirk

    @pytest.mark.parametrize("name", ["euler", "rk4", "dopri5", "tsit5"])
    def test_explicit_methods_are_not_implicit(self, name: str) -> None:
        assert not _METHODS[name].is_implicit
        assert not _METHODS[name].is_dirk

    def test_radau_iia3_matches_its_closed_form(self) -> None:
        t = _METHODS["radauIIA3"]
        assert t.a[0] == pytest.approx([5 / 12, -1 / 12], abs=1e-14)
        assert t.a[1] == pytest.approx([3 / 4, 1 / 4], abs=1e-14)
        assert t.b == pytest.approx([3 / 4, 1 / 4], abs=1e-14)
        assert t.c == pytest.approx([1 / 3, 1.0], abs=1e-14)

    def test_gauss_legendre_4_matches_its_closed_form(self) -> None:
        t = _METHODS["gl4"]
        root3 = math.sqrt(3.0)
        assert t.c == pytest.approx([0.5 - root3 / 6, 0.5 + root3 / 6], abs=1e-14)
        assert t.a[0] == pytest.approx([1 / 4, 1 / 4 - root3 / 6], abs=1e-14)
        assert t.a[1] == pytest.approx([1 / 4 + root3 / 6, 1 / 4], abs=1e-14)

    def test_radau_iia5_nodes_match_their_closed_form(self) -> None:
        root6 = math.sqrt(6.0)
        assert _METHODS["radauIIA5"].c == pytest.approx(
            [(4 - root6) / 10, (4 + root6) / 10, 1.0], abs=1e-14
        )

    def test_gauss_legendre_6_nodes_match_their_closed_form(self) -> None:
        root15 = math.sqrt(15.0)
        assert _METHODS["gl6"].c == pytest.approx(
            [0.5 - root15 / 10, 0.5, 0.5 + root15 / 10], abs=1e-14
        )

    def test_trapezoid_is_the_trapezoidal_rule(self) -> None:
        t = _METHODS["trapezoid"]
        assert t.a[0] == pytest.approx([0.0, 0.0], abs=1e-14)
        assert t.a[1] == pytest.approx([0.5, 0.5], abs=1e-14)

    @pytest.mark.parametrize("stages", [1, 2, 3, 4, 5])
    def test_gauss_nodes_reach_double_their_count(self, stages: int) -> None:
        nodes = _collocation.gauss_nodes(stages)
        _a, b = _collocation.collocation_tableau(nodes)
        c = [float(node) for node in nodes]
        assert _collocation.quadrature_order(b, c, limit=2 * stages) == 2 * stages

    @pytest.mark.parametrize("stages", [1, 2, 3, 4, 5])
    def test_radau_nodes_reach_one_less(self, stages: int) -> None:
        nodes = _collocation.radau_nodes(stages)
        _a, b = _collocation.collocation_tableau(nodes)
        c = [float(node) for node in nodes]
        assert c[-1] == pytest.approx(1.0, abs=1e-14)
        assert _collocation.quadrature_order(b, c, limit=2 * stages) == 2 * stages - 1

    def test_lobatto_needs_two_nodes(self) -> None:
        with pytest.raises(ValueError, match="at least two nodes"):
            _collocation.lobatto_nodes(1)

    def test_a_square_stage_matrix_is_accepted(self) -> None:
        t = diffeq.ButcherTableau(
            a=[[1.0]], b=[1.0], c=[1.0], order=1, name="backward_euler"
        )
        assert t.is_implicit and t.is_dirk and t.stages == 1

    def test_mixed_row_widths_are_rejected(self) -> None:
        # Neither ragged nor square: every coefficient after the bad row would
        # line up against the wrong stage.
        with pytest.raises(ValueError, match="ragged .* or square"):
            diffeq.ButcherTableau(
                a=[[], [1.0, 2.0, 3.0]],
                b=[0.5, 0.5],
                c=[0.0, 1.0],
                order=1,
                name="lopsided",
            )


class TestImplicitMethods:
    """The nine implicit methods, reachable through ``odeint`` like any other."""

    DECAY_EXACT = math.exp(-1.0)

    @pytest.mark.parametrize(("method", "_order"), IMPLICIT_ORDERS)
    def test_solves_exponential_decay(self, method: str, _order: int) -> None:
        y = diffeq.odeint(
            _decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            _grid(100),
            method=method,
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(self.DECAY_EXACT, abs=2e-3)

    @pytest.mark.parametrize(("method", "order"), IMPLICIT_ORDERS)
    def test_convergence_order(self, method: str, order: int) -> None:
        # Coarse grids on purpose: at fine ones the nonlinear solve's residual
        # tolerance, not the method, sets the error and the rate goes to noise.
        n1, n2 = (4, 8) if order >= 5 else (8, 16) if order >= 3 else (16, 32)
        errs = []
        for n in (n1, n2):
            y = diffeq.odeint(
                _decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                _grid(n),
                method=method,
                return_trajectory=False,
            )
            errs.append(abs(float(y.item()) - self.DECAY_EXACT))
        assert math.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.35)

    @pytest.mark.parametrize("method", ["implicit_euler", "radauIIA5", "trbdf2"])
    def test_survives_a_stiff_problem_that_kills_rk4(self, method: str) -> None:
        # y' = -1000(y - cos t) - sin t has solution cos t, but a decay mode
        # 1000x faster.  At h = 0.02 that puts h*lambda at -20, far outside
        # any explicit method's stability region -- which is the entire reason
        # this family exists.
        def stiff(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return -1000.0 * (y - lucid.cos(t)) - lucid.sin(t)

        y0 = lucid.tensor([0.0], dtype=lucid.float64)
        got = diffeq.odeint(
            stiff, y0, _grid(50), method=method, return_trajectory=False
        )
        assert float(got.item()) == pytest.approx(math.cos(1.0), abs=1e-4)

        blown = diffeq.odeint(
            stiff, y0, _grid(50), method="rk4", return_trajectory=False
        )
        assert abs(float(blown.item())) > 1e10

    def test_first_entry_is_y0(self) -> None:
        y0 = lucid.tensor([1.0, -3.0], dtype=lucid.float64)
        traj = diffeq.odeint(_decay, y0, _grid(10), method="gl4")
        assert traj.tolist()[0] == y0.tolist()

    def test_integrates_backwards(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.float64)
        grid = [1.0 - i / 50 for i in range(51)]
        y = diffeq.odeint(_decay, y0, grid, method="radauIIA3", return_trajectory=False)
        assert float(y.item()) == pytest.approx(math.e, abs=1e-5)

    # Differentiating the discretisation gives the gradient of the discrete
    # solution, which trails the true one by the method's own order -- so the
    # tolerance has to track the method, not be one number for all of them.
    @pytest.mark.parametrize(("method", "tol"), [("gl4", 1e-6), ("sdirk2", 5e-5)])
    def test_is_differentiable(self, method: str, tol: float) -> None:
        k = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        y = diffeq.odeint(
            lambda t, s: -k * s,
            lucid.tensor([1.0], dtype=lucid.float64),
            _grid(50),
            method=method,
            return_trajectory=False,
        )
        y.sum().backward()
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(-math.exp(-1.0), abs=tol)

    def test_works_under_the_adjoint(self) -> None:
        k = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        traj = diffeq.odeint_adjoint(
            lambda t, s: -k * s,
            lucid.tensor([1.0], dtype=lucid.float64),
            _grid(50),
            method="radauIIA3",
            adjoint_params=[k],
        )
        traj[-1].sum().backward()
        assert k.grad is not None
        assert float(k.grad.item()) == pytest.approx(-math.exp(-1.0), abs=1e-5)

    def test_accepts_a_tuple_state(self) -> None:
        y0 = (
            lucid.tensor([[1.0, 0.0]], dtype=lucid.float64),
            lucid.tensor([0.0, 1.0], dtype=lucid.float64),
        )
        grid = [i * (math.pi / 2) / 100 for i in range(101)]
        traj = diffeq.odeint(_osc_tuple, y0, grid, method="gl6")
        assert [tuple(x.shape) for x in traj] == [(101, 1, 2), (101, 2)]
        assert traj[0].tolist()[-1][0] == pytest.approx([0.0, 1.0], abs=1e-9)

    def test_dense_output(self) -> None:
        dense = diffeq.odeint_dense(
            _decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            1.0,
            method="gl4",
            options={"step_size": 0.02, "interp": "cubic"},
        )
        assert float(dense(0.37).item()) == pytest.approx(math.exp(-0.37), abs=1e-8)
        assert float(dense(1.0).item()) == pytest.approx(self.DECAY_EXACT, abs=1e-9)

    def test_event_detection(self) -> None:
        event_t, sol = diffeq.odeint_event(
            _decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            0.0,
            event_fn=lambda t, y: y[0] - 0.5,
            method="radauIIA5",
            options={"step_size": 0.01, "interp": "cubic"},
        )
        assert float(event_t.item()) == pytest.approx(math.log(2.0), abs=1e-8)
        assert sol[-1].tolist()[0] == pytest.approx(0.5, abs=1e-9)

    def test_step_size_decouples_the_output_grid(self) -> None:
        y = diffeq.odeint(
            _decay,
            lucid.tensor([1.0], dtype=lucid.float64),
            [0.0, 0.37, 1.0],
            method="radauIIA3",
            options={"step_size": 0.01, "interp": "cubic"},
        )
        got = y.tolist()
        assert got[1][0] == pytest.approx(math.exp(-0.37), abs=1e-6)
        assert got[2][0] == pytest.approx(self.DECAY_EXACT, abs=1e-8)

    def test_warns_when_the_solve_will_not_converge(self) -> None:
        # One iteration cannot resolve a stiff step, and a step that silently
        # used an unconverged iterate would look like a successful solve.
        def stiff(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return -1000.0 * y

        with pytest.warns(RuntimeWarning, match="did not converge"):
            diffeq.odeint(
                stiff,
                lucid.tensor([1.0], dtype=lucid.float64),
                _grid(4),
                method="gl6",
                options={"max_iters": 1},
                return_trajectory=False,
            )

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"max_iters": 0}, "max_iters must be >= 1"),
            ({"max_iters": 1.5}, "must be an int"),
            ({"max_order": 4}, "unknown option"),
        ],
    )
    def test_rejects_bad_options(self, options: dict[str, object], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            diffeq.odeint(
                _decay,
                lucid.tensor([1.0], dtype=lucid.float64),
                [0.0, 1.0],
                method="gl4",
                options=options,
            )


class TestBroydenProbe:
    """The fused probe behind every implicit step.

    It exists to answer an iteration's questions in one device round-trip, so
    what matters is that the answers stay separable -- a fused reduction that
    merged two of them would still hand back the same count of numbers.
    """

    def test_reports_each_norm_separately(self) -> None:
        residual = lucid.tensor([1.0, -2.0, 2.0], dtype=lucid.float64)
        step = lucid.tensor([3.0, 0.0, 4.0], dtype=lucid.float64)
        state = lucid.tensor([0.0, 6.0, 8.0], dtype=lucid.float64)
        info = lucid.zeros((), dtype=lucid.float64)

        residual_sq, step_sq, state_sq, singular = _fused.broyden_probe(
            residual, step, state, info
        )
        assert residual_sq == pytest.approx(9.0)
        assert step_sq == pytest.approx(25.0)
        assert state_sq == pytest.approx(100.0)
        assert singular == 0.0

    def test_carries_the_solve_status_through(self) -> None:
        # Non-zero info is how the solver learns the Jacobian was singular.
        one = lucid.ones((3,), dtype=lucid.float64)
        info = lucid.ones((), dtype=lucid.float64) * 3.0
        assert _fused.broyden_probe(one, one, one, info)[3] == pytest.approx(3.0)

    def test_accepts_the_column_the_linear_solve_returns(self) -> None:
        # solve_ex hands back (n, 1) while the residual is flat; matching on
        # element count is what lets the caller skip a per-iteration reshape.
        residual = lucid.ones((4,), dtype=lucid.float64)
        column = lucid.ones((4, 1), dtype=lucid.float64) * 2.0
        residual_sq, step_sq, _, _ = _fused.broyden_probe(residual, column, residual)
        assert residual_sq == pytest.approx(4.0)
        assert step_sq == pytest.approx(16.0)

    def test_info_is_optional(self) -> None:
        # The seeding call happens before any solve exists to report on.
        residual = lucid.tensor([2.0, 2.0], dtype=lucid.float64)
        residual_sq, _, _, singular = _fused.broyden_probe(residual, residual, residual)
        assert residual_sq == pytest.approx(8.0)
        assert singular == 0.0

    def test_agrees_with_the_unfused_spelling(self) -> None:
        # The arithmetic the op replaces, written out.
        rng = [0.3, -1.25, 4.0, 0.0, -2.5]
        residual = lucid.tensor(rng, dtype=lucid.float64)
        step = lucid.tensor([v * 2.0 - 1.0 for v in rng], dtype=lucid.float64)
        state = lucid.tensor([v + 7.0 for v in rng], dtype=lucid.float64)

        residual_sq, step_sq, state_sq, _ = _fused.broyden_probe(residual, step, state)
        assert residual_sq == pytest.approx(float((residual * residual).sum().item()))
        assert step_sq == pytest.approx(float((step * step).sum().item()))
        assert state_sq == pytest.approx(float((state * state).sum().item()))

    def test_rejects_a_mismatched_element_count(self) -> None:
        a = lucid.ones((4,), dtype=lucid.float64)
        b = lucid.ones((5,), dtype=lucid.float64)
        with pytest.raises(Exception):
            _fused.broyden_probe(a, b, a)
        with pytest.raises(Exception):
            _fused.broyden_probe(a, a, b)


class TestFailureModes:
    """What a solve does when it cannot succeed.

    Every parity test before this one compares a problem that converges.  That
    left the failure paths unexamined, and they were wrong in the way silence
    always is: a diverging problem drove the state to NaN, every comparison
    against NaN is false, so the step controller *accepted* each bad step and
    the solve returned NaN without a warning.  The reference raises for the
    same input.  These pin the behaviour the reference has and this did not.
    """

    @staticmethod
    def _grid() -> lucid.Tensor:
        return lucid.linspace(0.0, 1.0, 5, dtype=lucid.float64)

    @staticmethod
    def _one() -> lucid.Tensor:
        return lucid.ones(1, dtype=lucid.float64)

    def test_a_diverging_problem_raises_instead_of_returning_nan(self) -> None:
        # y' = 1e4 y^2 escapes to infinity in finite time.  The step shrinks
        # until it can no longer move t, which is the moment to give up.
        with pytest.raises(RuntimeError, match="underflow in dt"):
            diffeq.odeint(
                lambda t, y: y * y * 1e4,
                self._one(),
                self._grid(),
                return_trajectory=False,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_right_hand_side_raises(self, bad: float) -> None:
        with pytest.raises(RuntimeError, match="underflow in dt"):
            diffeq.odeint(
                lambda t, y: y * bad,
                self._one(),
                self._grid(),
                return_trajectory=False,
            )

    def test_a_non_finite_initial_state_raises(self) -> None:
        with pytest.raises(RuntimeError, match="underflow in dt"):
            diffeq.odeint(
                _decay,
                lucid.tensor([float("nan")], dtype=lucid.float64),
                self._grid(),
                return_trajectory=False,
            )

    def test_the_step_budget_is_enforced(self) -> None:
        with pytest.raises(RuntimeError, match="max_num_steps"):
            diffeq.odeint(
                lambda t, y: y * y * 1e4,
                self._one(),
                self._grid(),
                options={"max_num_steps": 10},
                return_trajectory=False,
            )

    def test_a_step_stuck_at_min_step_raises(self) -> None:
        # With a floor under the step the controller cannot shrink its way out,
        # so the rejection would otherwise repeat forever.
        with pytest.raises(RuntimeError, match="min_step"):
            diffeq.odeint(
                lambda t, y: y * y * 1e4,
                self._one(),
                self._grid(),
                options={"min_step": 1e-3},
                return_trajectory=False,
            )

    def test_a_fixed_grid_method_marches_through_a_blow_up(self) -> None:
        # Nothing to detect: a fixed grid has no step control, so it takes the
        # steps it was told to and reports whatever they produce.  The
        # reference does the same, to the same value.
        y = diffeq.odeint(
            lambda t, y: y * y * 1e4,
            self._one(),
            self._grid(),
            method="euler",
            return_trajectory=False,
        )
        assert float(y.item()) == pytest.approx(9.343075802364849e50, rel=1e-12)

    def test_a_well_posed_problem_is_untouched(self) -> None:
        # The guards are NaN-shaped; none of them may fire on a normal solve.
        y = diffeq.odeint(_decay, self._one(), self._grid(), return_trajectory=False)
        assert float(y.item()) == pytest.approx(math.exp(-1.0), abs=1e-7)


class TestAdjointSeminorm:
    """``adjoint_options={"norm": "seminorm"}``.

    The augmented backward state is ``[y | adjoint | parameter gradients]``.
    The gradient block is a quadrature -- it never feeds back into the
    dynamics -- so holding it to the same tolerance costs steps and buys no
    accuracy.  This used to be accepted and silently ignored, which is the
    worst of the three options.
    """

    @staticmethod
    def _problem() -> tuple:
        k = lucid.tensor([0.7], dtype=lucid.float64, requires_grad=True)
        y0 = lucid.tensor([1.0], dtype=lucid.float64, requires_grad=True)
        grid = [0.0, 1.0]

        class Field:
            def parameters(self) -> list[lucid.Tensor]:
                return [k]

            def __call__(self, t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
                return -k * y

        return Field(), y0, grid, k

    def test_it_matches_the_default_norm(self) -> None:
        # y(1) = y0 exp(-k), so dL/dk = -exp(-k) for L = y(1).  Dropping the
        # parameter block from the *step control* must not change the answer.
        for seminorm in (False, True):
            field, y0, grid, k = self._problem()
            out = diffeq.odeint_adjoint(
                field,
                y0,
                grid,
                rtol=1e-10,
                atol=1e-12,
                adjoint_options={"norm": "seminorm"} if seminorm else None,
            )
            out[-1].sum().backward()
            assert k.grad is not None
            assert float(k.grad.item()) == pytest.approx(
                -math.exp(-0.7), abs=1e-8
            ), f"seminorm={seminorm}"

    def test_any_other_norm_is_refused(self) -> None:
        field, y0, grid, _ = self._problem()
        with pytest.raises(NotImplementedError, match="seminorm"):
            diffeq.odeint_adjoint(
                field, y0, grid, adjoint_options={"norm": lambda x: 0.0}
            )


class TestInterpolantOrder:
    """``mid_order`` and the caution built on it.

    A tableau's step order says nothing about how well it interpolates
    between steps, and nothing at the call site reveals the difference.
    """

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("adaptive_heun", 1),
            ("fehlberg2", 1),
            ("bosh3", 1),
            ("dopri5", 4),
            ("tsit5", 4),
            ("dopri8", 5),
        ],
    )
    def test_midpoint_order_is_measured_from_the_tableau(
        self, method: str, expected: int
    ) -> None:
        assert _METHODS[method].mid_order == expected

    def test_a_fixed_grid_tableau_has_no_midpoint(self) -> None:
        assert _METHODS["rk4"].mid is None
        assert _METHODS["rk4"].mid_order == 0

    @pytest.mark.parametrize("method", ["bosh3", "fehlberg2", "adaptive_heun"])
    def test_a_first_order_interpolant_is_flagged(self, method: str) -> None:
        with pytest.warns(RuntimeWarning, match="interpolates between them at order 1"):
            diffeq.odeint_dense(
                _decay, lucid.ones(1, dtype=lucid.float64), 0.0, 1.0, method=method
            )

    @pytest.mark.parametrize("method", ["dopri5", "tsit5", "dopri8"])
    def test_a_derived_interpolant_is_not_flagged(self, method: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            diffeq.odeint_dense(
                _decay, lucid.ones(1, dtype=lucid.float64), 0.0, 1.0, method=method
            )


class TestRkErrorNorm:
    """The fused embedded-error norm behind adaptive step control.

    Both device streams take more than one route through it — the CPU stream
    materialises the error vector only when the stage count and the state size
    make that pay, and the GPU stream reuses one compiled kernel across every
    shape, stage count and dtype it is called with.  What these pin is that
    the routes agree with the arithmetic they stand in for, since a reduction
    that silently used a stale trace would still return a plausible number.
    """

    @staticmethod
    def _reference(
        y0: lucid.Tensor,
        y1: lucid.Tensor,
        ks: list[lucid.Tensor],
        coeffs: list[float],
        dt: float,
        rtol: float,
        atol: float,
    ) -> float:
        # The op written out: RMS of the embedded estimate over its tolerance.
        err = None
        for k, c in zip(ks, coeffs):
            term = (dt * c) * k
            err = term if err is None else err + term
        assert err is not None
        tol = atol + rtol * lucid.maximum(y0.abs(), y1.abs())
        ratio = err / tol
        return float((ratio * ratio).mean().item()) ** 0.5

    @staticmethod
    def _operands(
        shape: tuple, stages: int, device: str
    ) -> tuple[lucid.Tensor, lucid.Tensor, list[lucid.Tensor], list[float]]:
        # The GPU stream has no float64 at all, so the device picks the dtype.
        dt = lucid.float64 if device == "cpu" else lucid.float32
        n = math.prod(shape) if shape else 1
        y0 = lucid.tensor(
            [1.0 + 0.5 * i for i in range(n)], dtype=dt, device=device
        ).reshape(*shape)
        y1 = lucid.tensor(
            [1.1 + 0.5 * i for i in range(n)], dtype=dt, device=device
        ).reshape(*shape)
        ks = [
            lucid.tensor(
                [0.25 * (j + 1) - 0.03 * i for i in range(n)], dtype=dt, device=device
            ).reshape(*shape)
            for j in range(stages)
        ]
        return y0, y1, ks, [(-1.0) ** j / (j + 3.0) for j in range(stages)]

    # 1 and 2 stages take the register-only route, 3+ the materialised one;
    # the state sizes straddle the element-count guard and the stack scratch.
    @pytest.mark.parametrize("stages", [1, 2, 3, 7, 14])
    @pytest.mark.parametrize("n", [1, 8, 64, 300, 4096])
    def test_agrees_with_the_unfused_spelling(self, stages: int, n: int) -> None:
        y0, y1, ks, coeffs = self._operands((n,), stages, "cpu")
        got = _fused.error_ratio(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
        want = self._reference(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
        assert got == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("stages", [1, 3, 7])
    @pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4)])
    def test_holds_for_every_state_rank(
        self, device: str, stages: int, shape: tuple
    ) -> None:
        y0, y1, ks, coeffs = self._operands(shape, stages, device)
        got = _fused.error_ratio(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
        want = self._reference(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
        assert got == pytest.approx(want, rel=1e-6)

    def test_one_call_does_not_poison_the_next(self, device: str) -> None:
        # The GPU stream compiles the reduction once and reuses it across
        # shapes and stage counts.  A trace that captured either would keep
        # returning the first call's answer, so the shapes and stage counts
        # are deliberately interleaved rather than grouped.
        plan = [(3, 8), (7, 64), (3, 8), (7, 300), (5, 8), (3, 4096), (14, 9), (3, 8)]
        for stages, n in plan:
            y0, y1, ks, coeffs = self._operands((n,), stages, device)
            got = _fused.error_ratio(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
            want = self._reference(y0, y1, ks, coeffs, 0.25, 1e-6, 1e-9)
            assert got == pytest.approx(want, rel=1e-6), f"stages={stages} n={n}"

    def test_tolerances_are_not_baked_into_the_trace(self, device: str) -> None:
        # rtol and atol are inputs, not constants; a compiled kernel that
        # folded them in would ignore a later change.
        y0, y1, ks, coeffs = self._operands((32,), 4, device)
        loose = _fused.error_ratio(y0, y1, ks, coeffs, 0.25, 1e-2, 1e-4)
        tight = _fused.error_ratio(y0, y1, ks, coeffs, 0.25, 1e-8, 1e-10)
        assert tight > loose
        assert loose == pytest.approx(
            self._reference(y0, y1, ks, coeffs, 0.25, 1e-2, 1e-4), rel=1e-6
        )

    def test_all_zero_coefficients_accept_the_step(self, device: str) -> None:
        y0, y1, ks, _ = self._operands((6,), 3, device)
        assert _fused.error_ratio(y0, y1, ks, [0.0, 0.0, 0.0], 0.25, 1e-6, 1e-9) == 0.0


class TestRkCombine:
    """The fused affine combination every method's steps reduce to.

    Each device family evaluates it in one pass rather than a chain of
    scalar multiplies and adds, so what these pin is that the one pass
    still agrees with the arithmetic it stands in for -- on every shape
    the pass has a special case for, and on both devices, since the two
    implementations share nothing but their contract.
    """

    @staticmethod
    def _unfused(
        y0: lucid.Tensor,
        ks: list[lucid.Tensor],
        coeffs: list[float],
        dt: float,
    ) -> lucid.Tensor:
        out = y0
        for k, c in zip(ks, coeffs):
            out = out + (dt * c) * k
        return out

    @staticmethod
    def _operands(
        shape: tuple[int, ...], stages: int, device: str
    ) -> tuple[lucid.Tensor, list[lucid.Tensor], list[float]]:
        n = math.prod(shape)
        y0 = lucid.tensor(
            [0.5 - 0.25 * i for i in range(n)], device=device, dtype=lucid.float32
        ).reshape(*shape)
        ks = [
            lucid.tensor(
                [1.0 + 0.5 * j - 0.125 * i for i in range(n)],
                device=device,
                dtype=lucid.float32,
            ).reshape(*shape)
            for j in range(stages)
        ]
        return y0, ks, [1.0 / (j + 2.0) for j in range(stages)]

    # 1 stage takes the two-node shortcut, 2+ the packed reduction; 14 is
    # dopri8's row, the widest any method asks for.
    @pytest.mark.parametrize("stages", [1, 2, 3, 7, 14])
    def test_agrees_with_the_unfused_spelling(self, device: str, stages: int) -> None:
        y0, ks, coeffs = self._operands((6,), stages, device)
        fused = _fused.combine(y0, ks, coeffs, 0.125)
        plain = self._unfused(y0, ks, coeffs, 0.125)
        assert fused.shape == y0.shape
        assert float((fused - plain).abs().max().item()) < 1e-5

    @pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 3, 4)])
    def test_holds_for_every_state_rank(self, device: str, shape: tuple) -> None:
        # The packed form reshapes the terms into a leading axis, so a state
        # with no leading axis of its own (0-d) takes a different route.
        y0, ks, coeffs = self._operands(shape, 4, device)
        fused = _fused.combine(y0, ks, coeffs, -0.5)
        plain = self._unfused(y0, ks, coeffs, -0.5)
        assert fused.shape == y0.shape
        assert float((fused - plain).abs().max().item()) < 1e-5

    def test_skips_stages_whose_coefficient_is_zero(self, device: str) -> None:
        # Butcher rows are strictly lower triangular, so most entries are
        # zero; dropping them must not shift the remaining weights.
        y0, ks, _ = self._operands((5,), 4, device)
        coeffs = [0.0, 0.25, 0.0, -0.75]
        fused = _fused.combine(y0, ks, coeffs, 0.5)
        plain = self._unfused(y0, ks, coeffs, 0.5)
        assert float((fused - plain).abs().max().item()) < 1e-5

    def test_all_zero_coefficients_return_the_base(self, device: str) -> None:
        y0, ks, _ = self._operands((5,), 3, device)
        fused = _fused.combine(y0, ks, [0.0, 0.0, 0.0], 0.5)
        assert float((fused - y0).abs().max().item()) == 0.0

    def test_accepts_non_contiguous_operands(self, device: str) -> None:
        # A stage arriving as a view has to be materialised; the packed form
        # reads raw buffers, so a stride left unhandled would read garbage.
        base = lucid.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=lucid.float32
        )
        y0 = base.T[0]
        ks = [base.T[1], base.T[2]]
        coeffs = [0.5, -0.25]
        fused = _fused.combine(y0, ks, coeffs, 2.0)
        plain = self._unfused(y0, ks, coeffs, 2.0)
        assert float((fused - plain).abs().max().item()) < 1e-5

    def test_backward_still_scales_each_stage(self, device: str) -> None:
        # The forward is one pass; the backward is not, and fusing the one
        # must not quietly detach the other.
        y0 = lucid.ones((3,), device=device, requires_grad=True)
        ks = [
            lucid.tensor(
                [j + 1.0] * 3, device=device, dtype=lucid.float32, requires_grad=True
            )
            for j in range(3)
        ]
        coeffs = [0.5, 0.0, -1.5]
        _fused.combine(y0, ks, coeffs, 0.25).sum().backward()

        assert y0.grad is not None
        assert float((y0.grad - 1.0).abs().max().item()) == pytest.approx(0.0, abs=1e-6)
        for k, c in zip(ks, coeffs):
            assert k.grad is not None
            assert float(k.grad.max().item()) == pytest.approx(0.25 * c, abs=1e-6)


class TestPropagatedOrder:
    """The weights that advance the state must earn the declared order.

    ``tsit5`` shipped with its two embedded solutions swapped: the
    fourth-order weights sat in ``b`` and propagated, the fifth-order
    ones sat unused in the last row of ``a``.  Nothing caught it —
    ``b`` still summed to 1, ``b_error`` still summed to 0, the stage
    consistency conditions still held, and the declared ``order`` is
    just an integer nobody checked against the coefficients.  So the
    method took fourth-order steps, called itself fifth order, and fed
    the step controller an exponent for an order it did not have.
    """

    # ``dopri8`` is left out on purpose: at double precision its error
    # reaches the rounding floor before the asymptotic regime starts, so
    # a measured slope there says more about the floor than the method.
    MEASURABLE = [
        ("euler", 1),
        ("midpoint", 2),
        ("heun2", 2),
        ("heun3", 3),
        ("rk4", 4),
        ("rk4_classic", 4),
        ("adaptive_heun", 2),
        ("fehlberg2", 2),
        ("bosh3", 3),
        ("dopri5", 5),
        ("tsit5", 5),
    ]

    @staticmethod
    def _global_error(tab, n: int) -> float:
        """``y' = -y`` on ``[0, 1]`` in ``n`` equal steps, stepped by hand.

        Driven straight off the tableau rather than through ``odeint``, so
        an adaptive controller cannot mask the order of the weights.
        """
        h, y = 1.0 / n, 1.0
        stages = len(tab.b)
        for _ in range(n):
            ks: list[float] = []
            for i in range(stages):
                inner = sum(tab.a[i][j] * ks[j] for j in range(len(tab.a[i])))
                ks.append(-(y + h * inner))
            y += h * sum(tab.b[i] * ks[i] for i in range(stages))
        return abs(y - math.exp(-1.0))

    @pytest.mark.parametrize(
        "method,declared", MEASURABLE, ids=[m for m, _ in MEASURABLE]
    )
    def test_measured_order_matches_the_declared_one(
        self, method: str, declared: int
    ) -> None:
        tab = _METHODS[method]
        assert tab.order == declared
        coarse = self._global_error(tab, 20)
        fine = self._global_error(tab, 40)
        measured = math.log(coarse / fine) / math.log(2.0)
        assert measured > declared - 0.3, (
            f"{method}: declared order {declared}, measured {measured:.2f} — "
            f"the propagating weights deliver less than advertised"
        )

    def test_tsit5_propagates_the_published_fifth_order_weights(self) -> None:
        # The two unmistakable entries of the Tsitouras 5th-order vector:
        # b[1] is exactly 1/100, and b[6] is zero (which is what makes the
        # pair FSAL).  The 4th-order embedded solution has neither.
        tab = _METHODS["tsit5"]
        assert tab.b[1] == pytest.approx(0.01, abs=1e-15)
        assert tab.b[6] == 0.0
        assert tab.is_fsal
        assert tab.a[-1] == tab.b[:-1]


class TestDiffeqKernelDtypesAndHalfPrecision:
    """Every diffeq kernel and solver across floating-point dtypes on CPU and Metal."""

    @pytest.mark.parametrize(
        "dt",
        [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64],
        ids=["f16", "bf16", "f32", "f64"],
    )
    def test_every_diffeq_kernel_supports_every_floating_dtype(
        self, device: str, dt: lucid.Dtype
    ) -> None:
        if device == "metal" and dt == lucid.float64:
            pytest.skip("Metal has no float64")
        tol = 5e-2 if dt == lucid.bfloat16 else (1e-2 if dt == lucid.float16 else 1e-5)

        y0 = lucid.tensor([1.0, -2.0, 0.5, 3.0], dtype=dt, device=device)
        y1 = lucid.tensor([1.25, -1.75, 0.75, 2.5], dtype=dt, device=device)
        k0 = lucid.tensor([0.5, -1.0, 2.0, -0.5], dtype=dt, device=device)
        k1 = lucid.tensor([-0.25, 0.5, -1.0, 1.5], dtype=dt, device=device)

        # 1. rk_combine (forward + backward)
        y0_g = y0.detach().clone().requires_grad_(True)
        k0_g = k0.detach().clone().requires_grad_(True)
        k1_g = k1.detach().clone().requires_grad_(True)
        combined = _fused.combine(y0_g, [k0_g, k1_g], [0.5, -0.25], 0.5)
        assert combined.dtype == dt
        assert combined.device == device
        expected_combine = y0.to(lucid.float32) + 0.5 * (
            0.5 * k0.to(lucid.float32) - 0.25 * k1.to(lucid.float32)
        )
        assert (
            float((combined.to(lucid.float32) - expected_combine).abs().max().item())
            < tol
        )
        combined.sum().backward()
        assert y0_g.grad is not None and y0_g.grad.dtype == dt
        assert k0_g.grad is not None and k0_g.grad.dtype == dt
        assert k1_g.grad is not None and k1_g.grad.dtype == dt

        # 2. rk_error_norm (via _fused.error_ratio and raw _C_engine.diffeq.rk_error_norm)
        ratio = _fused.error_ratio(y0, y1, [k0, k1], [0.5, -0.5], 0.25, 1e-3, 1e-3)
        raw_ratio = lucid._C.engine.diffeq.rk_error_norm(
            y0._impl, y1._impl, [k0._impl, k1._impl], [0.5, -0.5], 0.25, 1e-3, 1e-3
        )
        ref_ratio = TestRkErrorNorm._reference(
            y0.to(lucid.float32),
            y1.to(lucid.float32),
            [k0.to(lucid.float32), k1.to(lucid.float32)],
            [0.5, -0.5],
            0.25,
            1e-3,
            1e-3,
        )
        assert ratio == pytest.approx(ref_ratio, rel=tol)
        assert raw_ratio == pytest.approx(ref_ratio, rel=tol)

        # 3. broyden_probe (via _fused.broyden_probe and raw _C_engine.diffeq.broyden_probe)
        info = lucid.tensor(2.0, dtype=dt, device=device)
        r_sq, s_sq, x_sq, info_val = _fused.broyden_probe(y0, k0, y1, info)
        raw_r_sq, raw_s_sq, raw_x_sq, raw_info = lucid._C.engine.diffeq.broyden_probe(
            y0._impl, k0._impl, y1._impl, info._impl
        )
        assert r_sq == pytest.approx(
            float((y0.to(lucid.float32) ** 2).sum().item()), rel=tol
        )
        assert s_sq == pytest.approx(
            float((k0.to(lucid.float32) ** 2).sum().item()), rel=tol
        )
        assert x_sq == pytest.approx(
            float((y1.to(lucid.float32) ** 2).sum().item()), rel=tol
        )
        assert info_val == pytest.approx(2.0, abs=tol)
        assert (raw_r_sq, raw_s_sq, raw_x_sq, raw_info) == pytest.approx(
            (r_sq, s_sq, x_sq, info_val), rel=1e-6
        )

    @pytest.mark.parametrize(
        ("dt", "atol"),
        [(lucid.float16, 1e-2), (lucid.bfloat16, 5e-2)],
        ids=["f16", "bf16"],
    )
    @pytest.mark.parametrize("adjoint", [False, True], ids=["direct", "adjoint"])
    @pytest.mark.parametrize("method", ["euler", "rk4", "dopri5"])
    def test_odeint_half_precision_forward_and_backward(
        self, device: str, dt: lucid.Dtype, atol: float, adjoint: bool, method: str
    ) -> None:
        y0_f64 = lucid.tensor([1.0, 2.0], dtype=lucid.float64, requires_grad=True)
        out_f64 = diffeq.odeint(
            lambda t, y: -y, y0_f64, [0.0, 1.0], method=method, rtol=1e-6, atol=1e-8
        )
        out_f64[-1].sum().backward()
        assert y0_f64.grad is not None
        want_y = out_f64[-1].to(lucid.float32)
        want_g = y0_f64.grad.to(lucid.float32)

        y0 = lucid.tensor([1.0, 2.0], dtype=dt, device=device, requires_grad=True)
        scale = lucid.nn.Parameter(lucid.ones((1,), dtype=dt, device=device))

        class _LinearRHS(lucid.nn.Module):
            def __init__(self, s: lucid.nn.Parameter) -> None:
                super().__init__()
                self.s = s

            def forward(self, t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
                return -self.s * y

        rhs = _LinearRHS(scale)
        solver = diffeq.odeint_adjoint if adjoint else diffeq.odeint
        out = solver(
            rhs,
            y0,
            [0.0, 1.0],
            method=method,
            rtol=1e-2,
            atol=1e-3,
        )
        assert out.dtype == dt
        assert out.device == device
        out[-1].sum().backward()
        assert y0.grad is not None
        assert y0.grad.dtype == dt
        assert y0.grad.device == device
        assert scale.grad is not None
        assert scale.grad.dtype == dt

        got_y = out[-1].to("cpu").to(lucid.float32)
        got_g = y0.grad.to("cpu").to(lucid.float32)
        assert float((got_y - want_y).abs().max().item()) < atol
        assert float((got_g - want_g).abs().max().item()) < atol
