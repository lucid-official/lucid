"""Implicit solvers on stiff problems, tolerance-vs-dtype warnings, method spellings.

An implicit method earns its keep on exactly the problems an explicit one cannot
touch, so these are the problems it is tested on: Robertson's kinetics (a
three-species chain whose rates span nine orders of magnitude) and the Van der
Pol oscillator at ``mu = 100``.  The solver used to start its nonlinear solve
from an identity Jacobian, and on both problems it took steps wrong by orders of
magnitude, then accepted the "best iterate" with a warning -- most methods ended
in ``NaN``.  What these pin is the behaviour a stiff solver has to have: finite
and accurate where the step is within the method's reach, and a typed error
where it is not, never a quiet wrong answer.
"""

import math
import warnings

import pytest

import lucid
import lucid.diffeq as diffeq
from lucid.diffeq import ImplicitSolveError
from lucid.diffeq._solvers import _resolve_method

# Every implicit method, with the error each is held to on the two problems
# below.  Bounds are about three times the error measured at the time of
# writing, so they track the method's order rather than the solver's noise.
ROBERTSON_BOUNDS = {
    "implicit_euler": 2e-2,
    "implicit_midpoint": 2e-4,
    "trapezoid": 6e-4,
    "radauIIA3": 5e-6,
    "radauIIA5": 1e-9,
    "gl4": 1e-7,
    "gl6": 1e-10,
    "sdirk2": 2e-4,
    "trbdf2": 3e-4,
}
VAN_DER_POL_BOUNDS = {
    "implicit_euler": 2.5e-2,
    "implicit_midpoint": 5e-3,
    "trapezoid": 1.5e-2,
    "radauIIA3": 1e-6,
    "radauIIA5": 1e-8,
    "gl4": 2e-4,
    "gl6": 2e-6,
    "sdirk2": 6e-5,
    "trbdf2": 2e-4,
}

# Reference values for the two problems.  Offline, once: scipy's Radau
# integrator at rtol=1e-12, atol=1e-14.  Pinned as constants so the test needs
# no second ODE library, and so it checks against something the solver under
# test did not produce.
ROBERTSON_AT_1000 = (0.336874530660999, 2.0137023182639965e-06, 0.6631234556366817)
VAN_DER_POL_AT_70 = (1.3430543973310172, -0.016699920194837223)


def _robertson(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    a, b, c = y[0:1], y[1:2], y[2:3]
    fast = 3e7 * b * b
    return lucid.concat([-0.04 * a + 1e4 * b * c, 0.04 * a - 1e4 * b * c - fast, fast])


def _van_der_pol(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    x, v = y[0:1], y[1:2]
    return lucid.concat([v, 100.0 * (1.0 - x * x) * v - x])


def _robertson_grid(steps: int = 200) -> list[float]:
    """``steps`` points, log spaced over ``[1e-6, 1e3]`` after the start at 0."""
    return [0.0] + [10 ** (-6 + 9 * i / (steps - 2)) for i in range(steps - 1)]


def _f64(values: list[float]) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.float64)


class TestStiffProblems:
    """Every implicit method, on a problem built to break a poor nonlinear solve."""

    @pytest.mark.parametrize("method", sorted(ROBERTSON_BOUNDS))
    def test_robertson_stays_finite_and_accurate(self, method: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            traj = diffeq.odeint(
                _robertson, _f64([1.0, 0.0, 0.0]), _robertson_grid(), method=method
            )
        rows = traj.tolist()
        assert all(math.isfinite(v) for row in rows for v in row)
        # Concentrations: non-negative to rounding, and the chain conserves mass.
        assert min(min(row) for row in rows) > -1e-9
        assert max(abs(sum(row) - 1.0) for row in rows) < 1e-9
        err = max(abs(a - b) for a, b in zip(rows[-1], ROBERTSON_AT_1000, strict=True))
        assert err < ROBERTSON_BOUNDS[method]

    @pytest.mark.parametrize("method", sorted(VAN_DER_POL_BOUNDS))
    def test_van_der_pol_mu_100_stays_finite_and_accurate(self, method: str) -> None:
        # Stiff phase of the oscillation: the decay rate is ~ -mu * (x^2 - 1),
        # a few hundred, against steps of 1.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            traj = diffeq.odeint(
                _van_der_pol, _f64([2.0, 0.0]), [float(i) for i in range(71)], method=method
            )
        rows = traj.tolist()
        assert all(math.isfinite(v) for row in rows for v in row)
        err = max(abs(a - b) for a, b in zip(rows[-1], VAN_DER_POL_AT_70, strict=True))
        assert err < VAN_DER_POL_BOUNDS[method]

    @pytest.mark.parametrize("method", ["gl4", "radauIIA5", "sdirk2"])
    def test_works_on_a_float32_state(self, method: str) -> None:
        y0 = lucid.tensor([1.0, 0.0, 0.0], dtype=lucid.float32)
        traj = diffeq.odeint(_robertson, y0, _robertson_grid(), method=method)
        end = traj.tolist()[-1]
        assert all(math.isfinite(v) for v in end)
        assert max(abs(a - b) for a, b in zip(end, ROBERTSON_AT_1000, strict=True)) < 5e-4


class TestNonConvergence:
    """A fixed-step method has no step to shrink, so a failed solve must raise."""

    @staticmethod
    def _cubic(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
        return -1000.0 * y * y * y

    @pytest.mark.parametrize("method", ["gl6", "implicit_euler", "sdirk2"])
    def test_an_exhausted_iteration_budget_raises_a_typed_error(self, method: str) -> None:
        with pytest.raises(ImplicitSolveError) as caught:
            diffeq.odeint(
                self._cubic,
                _f64([1.0]),
                [0.0, 0.5, 1.0],
                method=method,
                options={"max_iters": 1},
                return_trajectory=False,
            )
        error = caught.value
        assert isinstance(error, RuntimeError)
        assert error.method == method
        assert error.t == 0.0
        assert method in str(error) and "t=0.0" in str(error)

    def test_a_coupled_solve_reports_no_stage(self) -> None:
        with pytest.raises(ImplicitSolveError) as caught:
            diffeq.odeint(
                self._cubic,
                _f64([1.0]),
                [0.0, 1.0],
                method="gl4",
                options={"max_iters": 1},
                return_trajectory=False,
            )
        assert caught.value.stage is None
        assert ", stage" not in str(caught.value)

    def test_a_diagonally_implicit_solve_names_its_stage(self) -> None:
        with pytest.raises(ImplicitSolveError) as caught:
            diffeq.odeint(
                self._cubic,
                _f64([1.0]),
                [0.0, 1.0],
                method="trbdf2",
                options={"max_iters": 1},
                return_trajectory=False,
            )
        assert caught.value.stage == 1
        assert "stage 1" in str(caught.value)

    def test_the_failure_is_at_the_step_that_failed(self) -> None:
        # A step that is easy early and impossible late: the error carries the
        # time of the step that broke, not the first one.
        def blowup(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
            return y * y * 1e4

        with pytest.raises(ImplicitSolveError) as caught:
            diffeq.odeint(
                blowup,
                _f64([1.0]),
                [0.0, 1e-5, 2e-5, 1.0],
                method="implicit_euler",
                return_trajectory=False,
            )
        assert caught.value.t == pytest.approx(2e-5)

    def test_a_singular_stage_system_raises(self) -> None:
        # y' = y with h = 1 under implicit Euler: (1 - h) y = y0 has no solution.
        with pytest.raises(ImplicitSolveError, match="singular"):
            diffeq.odeint(
                lambda t, y: y,
                _f64([1.0]),
                [0.0, 1.0],
                method="implicit_euler",
                return_trajectory=False,
            )

    def test_the_same_failure_surfaces_through_event_detection(self) -> None:
        with pytest.raises(ImplicitSolveError):
            diffeq.odeint_event(
                self._cubic,
                _f64([1.0]),
                0.0,
                event_fn=lambda t, y: y[0] - 0.5,
                method="gl4",
                options={"step_size": 0.5, "max_iters": 1},
            )


class TestGradientsThroughTheSolve:
    """The implicit function's derivative, checked against finite differences."""

    @staticmethod
    def _final(k: lucid.Tensor, method: str) -> lucid.Tensor:
        return diffeq.odeint(
            lambda t, y: -k * y * y - 0.3 * y,
            _f64([1.0]),
            [0.0, 0.25, 0.5, 0.75, 1.0],
            method=method,
            return_trajectory=False,
        ).sum()

    @pytest.mark.parametrize("method", ["implicit_euler", "gl4", "sdirk2", "radauIIA5"])
    def test_first_and_second_derivative_match_finite_differences(self, method: str) -> None:
        k = _f64([1.5]).requires_grad_(True)
        value = self._final(k, method)
        (grad,) = lucid.autograd.grad(value, k, create_graph=True)
        (hess,) = lucid.autograd.grad(grad, k)

        h = 1e-4
        up = float(self._final(_f64([1.5 + h]), method).item())
        down = float(self._final(_f64([1.5 - h]), method).item())
        assert float(grad.item()) == pytest.approx((up - down) / (2 * h), rel=1e-6)
        assert float(hess.item()) == pytest.approx(
            (up - 2 * float(value.item()) + down) / h**2, rel=2e-3
        )

    def test_a_stiff_gradient_matches_finite_differences(self) -> None:
        # d y3(1e3) / d k1 on Robertson, through a coupled solve over 60 steps.
        def final_y3(k1: lucid.Tensor) -> lucid.Tensor:
            def rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
                a, b, c = y[0:1], y[1:2], y[2:3]
                fast = 3e7 * b * b
                return lucid.concat([-k1 * a + 1e4 * b * c, k1 * a - 1e4 * b * c - fast, fast])

            return diffeq.odeint(
                rhs, _f64([1.0, 0.0, 0.0]), _robertson_grid(60), method="radauIIA3"
            )[-1, 2]

        k1 = _f64([0.04]).requires_grad_(True)
        (grad,) = lucid.autograd.grad(final_y3(k1), k1)

        h = 1e-6
        up = float(final_y3(_f64([0.04 + h])).item())
        down = float(final_y3(_f64([0.04 - h])).item())
        assert float(grad.item()) == pytest.approx((up - down) / (2 * h), rel=1e-4)


class TestToleranceAgainstDtype:
    """A tolerance below the state's unit roundoff is steering by noise."""

    @staticmethod
    def _decay(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
        return -y

    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
    def test_the_default_tolerances_warn_on_a_half_precision_state(
        self, dtype: lucid.dtype
    ) -> None:
        y0 = lucid.tensor([1.0, 2.0], dtype=dtype)
        with pytest.warns(RuntimeWarning, match="unit roundoff") as record:
            diffeq.odeint(self._decay, y0, [0.0, 1.0])
        assert len(record) == 1
        message = str(record[0].message)
        assert str(dtype) in message
        assert "rtol=1e-07" in message and "atol=1e-09" in message
        roundoff = 2.0**-11 if dtype == lucid.float16 else 2.0**-8
        assert f"{roundoff:.3g}" in message
        # Attributed to the caller of odeint, not to the library.
        assert record[0].filename == __file__

    @pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
    def test_the_defaults_are_fine_for_float32_and_float64(self, dtype: lucid.dtype) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            diffeq.odeint(self._decay, lucid.tensor([1.0], dtype=dtype), [0.0, 1.0])

    def test_tolerances_the_dtype_can_meet_do_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            diffeq.odeint(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float16),
                [0.0, 1.0],
                rtol=1e-2,
                atol=1e-3,
            )

    def test_a_tight_tolerance_warns_even_in_float32(self) -> None:
        with pytest.warns(RuntimeWarning, match="float32"):
            diffeq.odeint(
                self._decay,
                lucid.tensor([1.0], dtype=lucid.float32),
                [0.0, 1.0],
                rtol=1e-9,
                atol=1e-12,
            )

    @pytest.mark.parametrize("method", ["rk4", "euler", "implicit_euler", "gl4"])
    def test_methods_that_ignore_the_tolerances_do_not_warn(self, method: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            diffeq.odeint(
                self._decay, lucid.tensor([1.0], dtype=lucid.float16), [0.0, 1.0], method=method
            )

    @pytest.mark.parametrize("method", ["explicit_adams", "dopri5", "tsit5"])
    def test_methods_that_read_the_tolerances_warn(self, method: str) -> None:
        with pytest.warns(RuntimeWarning, match="unit roundoff"):
            diffeq.odeint(
                self._decay, lucid.tensor([1.0], dtype=lucid.float16), [0.0, 1.0], method=method
            )

    def test_dense_output_warns_too(self) -> None:
        with pytest.warns(RuntimeWarning, match="unit roundoff"):
            diffeq.odeint_dense(
                self._decay, lucid.tensor([1.0], dtype=lucid.float16), 0.0, 1.0, method="dopri5"
            )

    def test_each_call_warns_once(self) -> None:
        y0 = lucid.tensor([1.0], dtype=lucid.bfloat16)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            diffeq.odeint(self._decay, y0, [0.0, 0.5, 1.0])
        assert sum("unit roundoff" in str(w.message) for w in record) == 1


class TestMethodSpellings:
    """The registered names mix conventions; the constants that carry them do not."""

    @staticmethod
    def _solve(method: str) -> list[float]:
        return diffeq.odeint(
            lambda t, y: -y, _f64([1.0, 2.0]), [0.0, 0.5, 1.0], method=method
        ).tolist()[-1]

    @pytest.mark.parametrize(
        ("spelling", "registered"),
        [
            ("radau_iia3", "radauIIA3"),
            ("RADAU_IIA3", "radauIIA3"),
            ("RadauIIA3", "radauIIA3"),
            ("radau-iia5", "radauIIA5"),
            ("radau_iia5", "radauIIA5"),
            ("Implicit_Euler", "implicit_euler"),
            ("implicitmidpoint", "implicit_midpoint"),
            ("GL4", "gl4"),
            ("DOPRI5", "dopri5"),
            ("rk4_classic", "rk4_classic"),
            ("Explicit_Adams", "explicit_adams"),
            ("fixedadams", "fixed_adams"),
        ],
    )
    def test_a_spelling_selects_the_registered_method(self, spelling: str, registered: str) -> None:
        assert self._solve(spelling) == self._solve(registered)

    def test_every_constant_name_selects_its_own_tableau(self) -> None:
        # `diffeq.RADAU_IIA3` is the documented spelling of the constant, so
        # the constant's own name must be a valid method string.
        constants = {
            name: getattr(diffeq, name)
            for name in diffeq.__all__
            if isinstance(getattr(diffeq, name), diffeq.ButcherTableau)
        }
        assert len(constants) == 21
        for name, tableau in constants.items():
            assert self._solve(name) == self._solve(tableau.name)

    def test_the_rk4_variants_stay_distinct(self) -> None:
        assert _resolve_method("rk4").name == "rk4"
        assert _resolve_method("RK4_Classic").name == "rk4_classic"

    def test_dense_output_and_the_adjoint_accept_them_too(self) -> None:
        dense = diffeq.odeint_dense(
            lambda t, y: -y, _f64([1.0]), 0.0, 1.0, method="radau_iia3", options={"step_size": 0.1}
        )
        assert float(dense(1.0).item()) == pytest.approx(math.exp(-1.0), abs=1e-3)

        k = _f64([1.0]).requires_grad_(True)
        traj = diffeq.odeint_adjoint(
            lambda t, y: -k * y,
            _f64([1.0]),
            [i / 20 for i in range(21)],
            method="radau_iia3",
            adjoint_params=[k],
        )
        traj[-1].sum().backward()
        assert k.grad is not None

    def test_an_unknown_name_still_raises_and_lists_the_registered_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown method 'radau_iia7'") as caught:
            diffeq.odeint(lambda t, y: -y, _f64([1.0]), [0.0, 1.0], method="radau_iia7")
        assert "radauIIA3" in str(caught.value)
