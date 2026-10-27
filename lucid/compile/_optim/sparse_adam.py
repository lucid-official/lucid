"""Compiled :class:`~lucid.optim.SparseAdam` — the lazy per-row Adam.

:func:`~lucid.compile._optim.compiler.compile_optimizer` dispatches here.
The eager optimizer picks its touched rows with ``nonzero`` (a host read a
trace cannot hold), so the compiled update masks them instead
(``_touched_rows``) — the same rows, kept in place by ``where``.
"""

import math
from typing import final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.compile._optim.compiler import _CompiledStepBase, _Flags, _ScalarFn, _State, _Update
from lucid.compile._optim.spec import _hp


@final
class _CompiledSparseAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.SparseAdam`.

    The eager rule, lazy per row: only the rows (slices along dim 0, the
    whole tensor for a 0-d parameter) whose gradient holds a nonzero value
    update their moments and values, by
    :math:`\eta\sqrt{1-\beta_2^t}/(1-\beta_1^t) \cdot m / (\sqrt v + \varepsilon)`
    — ``eps`` is not bias-corrected, unlike Adam.  The step count is one
    per parameter and advances whenever the parameter has a gradient, as
    in eager.  Parameters without a gradient are skipped, which the base
    class does for every optimizer.

    See Also
    --------
    :class:`lucid.optim.SparseAdam` : eager counterpart.
    """

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        # Eager's arithmetic order (``m += (1 - beta1)(g - m)``) so the two
        # paths agree to the bit; untouched rows keep moments and values.
        m0, v0 = state["exp_avg"], state["exp_avg_sq"]
        m = m0 + (g - m0) * gs("one_minus_beta1")
        v = v0 + (g * g - v0) * gs("one_minus_beta2")
        new_p = p + (m / (v.sqrt() + gs("eps"))) * ps("neg_step_size")
        touched = _touched_rows(g)
        return lucid.where(touched, new_p, p), {
            "exp_avg": lucid.where(touched, m, m0),
            "exp_avg_sq": lucid.where(touched, v, v0),
        }

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("exp_avg", "exp_avg_sq")

    @override
    def _step_entry(self, i: int) -> object:
        # The eager SparseAdam checkpoints ``step`` as a plain int.
        return self._steps[i]

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("one_minus_beta1", "one_minus_beta2", "eps")

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("neg_step_size",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        betas = group.get("betas", (0.9, 0.999))
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TypeError(f"SparseAdam: betas must be a pair, got {betas!r}")
        b1, b2 = float(betas[0]), float(betas[1])
        return {
            "lr": _hp(group, "lr", 1e-3),
            "eps": _hp(group, "eps", 1e-8),
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": b2,
            "one_minus_beta2": 1.0 - b2,
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        # The eager ``_sparse_adam_rows`` step size, to the same rounding.
        bc1 = 1.0 - gv["beta1"] ** step
        bc2 = 1.0 - gv["beta2"] ** step
        return {"neg_step_size": -(gv["lr"] * math.sqrt(bc2) / bc1)}


def _touched_rows(g: Tensor) -> Tensor:
    """Bool mask, broadcastable to ``g``: rows (dim-0 slices) holding a nonzero.

    The eager SparseAdam's row rule (``_sparse_adam_rows``); a 0-d gradient
    is one row.
    """
    shape = tuple(g.shape)
    if not shape:
        return g != 0
    n = shape[0]
    rows = (g.reshape(n, -1) != 0).any(dim=1)
    return rows.reshape(n, *([1] * (len(shape) - 1)))
