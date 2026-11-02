"""Compiled :class:`~lucid.optim.LBFGS` — the closure-less, per-element subset.

:func:`~lucid.compile._optim.compiler.compile_optimizer` dispatches here.
The eager optimizer's closure-driven line search has an iteration count that
depends on the loss, which a fixed executable cannot hold; this is a
different algorithm, so what it does not read it refuses (``_check_options``).
"""

import functools
from typing import final, override

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.compile._optim.compiler import _CompiledStepBase, _Flags, _ScalarFn, _State, _Update
from lucid.compile._optim.spec import _hp


@functools.cache
def _lbfgs_defaults() -> dict[str, object]:
    """LBFGS's own group defaults — the one statement of what "unset" means."""
    import lucid as _lucid
    from lucid.nn.parameter import Parameter
    from lucid.optim.lbfgs import LBFGS

    return dict(LBFGS([Parameter(_lucid.zeros(1))]).defaults)


@final
class _CompiledLBFGS(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.LBFGS` — single-iteration, no-line-search variant.

    L-BFGS in general needs a closure-driven line search whose iteration
    count depends on the loss value — incompatible with a fixed MPSGraph
    executable.  This path runs a restricted subset: one step per call,
    no closure, and a **per-element** Barzilai-Borwein direction from the
    last curvature pair instead of the flat-vector two-loop recursion:

    ::

        s     = param - prev_param
        y     = grad  - prev_grad
        alpha = clamp(|s| / (|y| + 1e-10), 0, 10)
        d     = -alpha * g        (steepest descent -g on a parameter's first two steps)
        param = param + lr * d

    Only ``lr`` is read, so every other option (``max_iter``, ``max_eval``,
    the tolerances, ``history_size``, ``line_search_fn``) is refused when set
    off its default (:meth:`_check_options`) rather than ignored.

    It is not the eager algorithm, so its checkpoint (``step``,
    ``prev_param``, ``prev_grad``) is its own, read through the wrapper;
    the eager optimizer keeps its own state, ``step`` and checkpoint
    methods (it neither hands its state over nor answers for this one).

    See Also
    --------
    :class:`lucid.optim.LBFGS` : eager counterpart (full closure-driven
        line search).
    """

    _EPS_HISTORY: float = 1e-10
    # A different algorithm from the eager LBFGS: ``opt`` keeps its state.
    _OWNS_EAGER_STATE = False
    # The one option the compiled step reads.
    _HONOURED_OPTIONS = frozenset({"lr"})

    @override
    def _check_options(self, group: dict[str, object]) -> None:
        defaults = dict(_lbfgs_defaults())
        # ``max_eval`` unset follows ``max_iter`` (LBFGS.__init__): a changed
        # ``max_iter`` is one option set, not two.
        max_iter = group.get("max_iter", defaults["max_iter"])
        if isinstance(max_iter, int):
            defaults["max_eval"] = max_iter * 5 // 4
        changed = {
            k: group[k]
            for k, default in defaults.items()
            if k not in self._HONOURED_OPTIONS and k in group and group[k] != default
        }
        if changed:
            named = ", ".join(f"{k}={v!r} (default {defaults[k]!r})" for k, v in changed.items())
            raise _C_engine.NotImplementedError(
                f"compile_optimizer: the compiled LBFGS is a per-element Barzilai-Borwein "
                f"step that honours only {', '.join(sorted(self._HONOURED_OPTIONS))}; it "
                f"would ignore {named}. Leave them at their defaults, or keep the eager LBFGS "
                f"with a closure."
            )

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("prev_param", "prev_grad")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("lr",)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("use_history",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {"lr": _hp(group, "lr", 1.0)}

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        return {"use_history": 1.0 if step >= 2 else 0.0}

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        s = p - state["prev_param"]
        y = g - state["prev_grad"]
        alpha = (_lucid.abs(s) / (_lucid.abs(y) + self._EPS_HISTORY)).clip(0.0, 10.0)
        d = _lucid.where(ps("use_history") > 0.5, -alpha * g, -g)
        new_p = p + gs("lr") * d
        return new_p, {"prev_param": p, "prev_grad": g}
