"""Keeping the time inputs of a solve in the autograd graph.

Every solver reads its grid to the host once, as plain floats, because the step
loop must branch on step sizes and a device read per step would stall the
queue.  That is the right call for the loop and a silent loss for the caller:
``t.requires_grad`` was honoured nowhere, so ``t.grad`` stayed ``None`` while
every other input of the same solve received a gradient.

Two different derivatives are needed, and which applies depends on what the
solve computed.

**The solution as a function of its output times.**  For an exact solution
:math:`y(t)` started at :math:`(t_0, y_0)`,

.. math::

    \\frac{\\partial y(t_i)}{\\partial t_i} = f(t_i, y_i), \\qquad
    \\frac{\\partial y(t_i)}{\\partial t_0}
        = -\\frac{\\partial y(t_i)}{\\partial y_0}\\, f(t_0, y_0).

The first is attached to each output row by adding
``f(t_i, y_i) * (t_i - t_i.detach())`` -- zero in value, the right edge in the
graph.  The second is a derivative *through the solve*, which only reverse mode
through the solver itself (or the adjoint) can supply, so instead the start is
moved: a trajectory begun at :math:`(t_0 + \\delta, y_0)` is, to first order,
the one begun at :math:`(t_0, y_0 - \\delta f(t_0, y_0))`.  Feeding the solver
that shifted state routes :math:`\\partial L / \\partial t_0` through whatever
gradient machinery it already has -- direct backprop or the adjoint -- with no
new backward pass.

**The discretisation, for a fixed-step solve on the caller's own grid.**  There
the output times *are* the step boundaries, so the honest derivative is that of
the arithmetic the solver performed, step sizes included.  That is what
:func:`lucid.diffeq._fixed.integrate_on_tensor_grid` does, by running the same
steps with the times kept as tensors.

The two agree as the grid refines and differ at a coarse one: Euler on a
three-point grid is not the derivative of the exact solution.
"""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import final

import lucid
from lucid._tensor.tensor import Tensor
from lucid.diffeq._typing import RightHandSide, ScalarFactory, StageCheck

__all__: list[str] = []


@final
@dataclass(frozen=True)
class RhsProbe:
    """Evaluates the right-hand side outside the integration loop.

    The time-gradient terms need ``f`` at the output points, and they need it
    validated the way the loop validates it: a mistake in ``func`` should be
    reported against ``func`` regardless of which code path first called it.
    """

    func: RightHandSide
    scalar: ScalarFactory
    check: StageCheck

    def slope(self, host_time: float, state: Tensor) -> Tensor:
        """Return ``f(host_time, state)`` with no graph behind it."""
        with lucid.no_grad():
            return self.check(self.func(self.scalar(host_time), state.detach()), 0, 0).detach()

    def tie(self, host_time: float, state: Tensor, time: Tensor) -> Tensor:
        """Give ``state`` the edge ``d state / d time = f(host_time, state)``.

        Parameters
        ----------
        host_time : float
            The value of ``time``, already on the host.
        state : Tensor
            State at that time.
        time : Tensor
            0-D time tensor carrying the graph.

        Returns
        -------
        Tensor
            ``state`` unchanged in value.
        """
        delta = (time - time.detach()).to(dtype=state.dtype, device=state.device)
        return state + self.slope(host_time, state).to(state.dtype) * delta


@final
class TimeGrid:
    """The caller's time tensor, kept in the graph beside its host floats.

    Parameters
    ----------
    times : Tensor
        1-D grid that requires gradient.
    host : sequence of float
        The same values as Python floats, which is what the loops step on.
    """

    def __init__(self, times: Tensor, host: Sequence[float]) -> None:
        self._times = times
        self._host = list(host)

    @staticmethod
    def of(t: object, host: Sequence[float]) -> TimeGrid | None:
        """Wrap ``t`` when it asks for a gradient and gradients are on."""
        if isinstance(t, Tensor) and t.requires_grad and lucid.is_grad_enabled():
            return TimeGrid(t, host)
        return None

    def point(self, index: int, like: Tensor) -> Tensor:
        """One grid point as a 0-D tensor in ``like``'s dtype and device."""
        return self._times[index].to(dtype=like.dtype, device=like.device)

    def points(self, like: Tensor) -> list[Tensor]:
        """Every grid point, as :meth:`point` returns it."""
        return [self.point(i, like) for i in range(len(self._host))]

    def shifted_start(self, y0: Tensor, probe: RhsProbe) -> Tensor:
        """Return the initial state moved so that ``t[0]`` has a gradient.

        The value is ``y0``; the graph additionally says that moving ``t[0]``
        by ``delta`` moves the trajectory by ``-delta * f(t[0], y0)``.

        Parameters
        ----------
        y0 : Tensor
            The caller's initial state.
        probe : RhsProbe
            Right-hand side access.

        Returns
        -------
        Tensor
            ``y0`` in value, with an edge to ``t[0]``.
        """
        t0 = self.point(0, y0)
        return y0 - probe.slope(self._host[0], y0) * (t0 - t0.detach())

    def tie_outputs(self, rows: Tensor, y0: Tensor, probe: RhsProbe) -> Tensor:
        """Attach the output-time edges to a stacked trajectory.

        Parameters
        ----------
        rows : Tensor
            Shape ``(len(t), *state)``; row 0 is the initial state.
        y0 : Tensor
            The caller's own initial state.  Row 0 is *replaced* by it: the
            trajectory's first row is the input, which does not depend on any
            time, whereas the solver was handed the shifted copy.
        probe : RhsProbe
            Right-hand side access.

        Returns
        -------
        Tensor
            Same shape and values as ``rows``.
        """
        out = [y0.to(rows.dtype)]
        out.extend(
            probe.tie(self._host[i], rows[i], self.point(i, y0)) for i in range(1, len(self._host))
        )
        return lucid.stack(out, dim=0)

    def tie_final(self, state: Tensor, y0: Tensor, probe: RhsProbe) -> Tensor:
        """Attach the last output time's edge to a final-state-only result."""
        return probe.tie(self._host[-1], state, self.point(len(self._host) - 1, y0))
