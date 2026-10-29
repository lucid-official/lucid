"""Compiled Adam family — Adam, AdamW, NAdam and RAdam.

:func:`~lucid.compile._optim.compiler.compile_optimizer` dispatches here.
These four share one weight-decay rule — coupled L2 folded into the
gradient, or, when the group sets ``decoupled_weight_decay`` (always, for
AdamW), a decay of the parameter itself — which ``_AdamLike._prepared`` states once.
"""

import math
from typing import final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.compile._optim.compiler import (
    _CompiledStepBase,
    _Flags,
    _round_state_scalar,
    _ScalarFn,
    _State,
    _Update,
    _wd_on,
)
from lucid.compile._optim.spec import _hp


def _decoupled(group: dict[str, object]) -> bool:
    """Whether ``group`` decays its weights apart from the gradient."""
    return bool(group.get("decoupled_weight_decay", False))


class _AdamLike(_CompiledStepBase):
    """What Adam, NAdam and RAdam share: the betas, two moments and one weight-decay rule.

    The structural flags are ``(weight decay on, decoupled)``; a subclass
    with more appends them.
    """

    # What a group without its own gets.
    _DEFAULT_LR: float = 1e-3
    _DEFAULT_WD: float = 0.0
    # AdamW decays its weights decoupled whatever its group says: its eager
    # engine is the decoupled one.
    _ALWAYS_DECOUPLED: bool = False
    _BETA_SCALARS: tuple[str, ...] = ("beta1", "one_minus_beta1", "beta2", "one_minus_beta2")

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group, self._DEFAULT_WD), self._ALWAYS_DECOUPLED or _decoupled(group))

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("exp_avg", "exp_avg_sq")

    @staticmethod
    def _decay_names(flags: _Flags) -> tuple[str, ...]:
        """The group scalar weight decay reads: ``1 - lr * wd`` (decoupled) or ``wd``."""
        if not flags[0]:
            return ()
        return ("wd_factor",) if flags[1] else ("weight_decay",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        lr = _hp(group, "lr", self._DEFAULT_LR)
        b1 = _hp(group, "beta1", 0.9)
        b2 = _hp(group, "beta2", 0.999)
        wd = _hp(group, "weight_decay", self._DEFAULT_WD)
        return {
            "lr": lr,
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": b2,
            "one_minus_beta2": 1.0 - b2,
            "eps": _hp(group, "eps", 1e-8),
            "weight_decay": wd,
            "wd_factor": 1.0 - lr * wd,
        }

    @staticmethod
    def _prepared(
        p: Tensor, g: Tensor, state: _State, gs: _ScalarFn, flags: _Flags
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(base, g, m, v)``: the value the update steps from, the gradient it
        sees, and the two moments after it.

        Weight decay comes first.  Decoupled decay shrinks the parameter and
        leaves the gradient alone; coupled decay (L2) adds ``wd * p`` to the
        gradient, and so to the moments.
        """
        base = p
        if flags[0]:
            if flags[1]:
                base = p * gs("wd_factor")
            else:
                g = g + gs("weight_decay") * p
        m = gs("beta1") * state["exp_avg"] + gs("one_minus_beta1") * g
        v = gs("beta2") * state["exp_avg_sq"] + gs("one_minus_beta2") * (g * g)
        return base, g, m, v


# ── Adam / AdamW ────────────────────────────────────────────────────


class _CompiledAdam(_AdamLike):
    r"""Compiled :class:`~lucid.optim.Adam` (AMSGrad included).

    Follows the eager GPU kernel's arrangement, which folds both bias
    corrections into two per-parameter scalars:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
        \theta_t &= \theta_{t-1} - \eta_{\text{eff}} \,
                    m_t / (\sqrt{v_t} + \varepsilon_{\text{eff}}),
        \quad \eta_{\text{eff}} = \eta \sqrt{1-\beta_2^t} / (1-\beta_1^t),
        \ \varepsilon_{\text{eff}} = \varepsilon \sqrt{1-\beta_2^t}

    Weight decay is folded into the gradient (coupled L2), or — when the
    group sets ``decoupled_weight_decay`` — applied to the parameter as in
    :class:`_CompiledAdamW`.  With ``amsgrad=True`` the denominator uses
    the running maximum of ``v``.

    See Also
    --------
    :class:`lucid.optim.Adam` : eager counterpart.
    :class:`_CompiledAdamW` : decoupled-weight-decay variant.
    """

    _OPTIONAL_STATE = ("max_exp_avg_sq",)

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (*super()._flags(group), bool(group.get("amsgrad", False)))

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        if flags[2]:
            return ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
        return ("exp_avg", "exp_avg_sq")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return self._BETA_SCALARS + self._decay_names(flags)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("lr_eff", "eps_eff")

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        bc1 = 1.0 - gv["beta1"] ** float(step)
        bc2 = 1.0 - gv["beta2"] ** float(step)
        sqrt_bc2 = math.sqrt(bc2)
        return {"lr_eff": gv["lr"] * sqrt_bc2 / bc1, "eps_eff": gv["eps"] * sqrt_bc2}

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        base, g, m, v = self._prepared(p, g, state, gs, flags)
        new_state = {"exp_avg": m, "exp_avg_sq": v}
        v_used = v
        if flags[2]:
            v_used = lucid.maximum(state["max_exp_avg_sq"], v)
            new_state["max_exp_avg_sq"] = v_used
        denom = v_used.sqrt() + ps("eps_eff")
        return base - ps("lr_eff") * (m / denom), new_state


@final
class _CompiledAdamW(_CompiledAdam):
    r"""Compiled :class:`~lucid.optim.AdamW` — Adam with decoupled weight decay.

    The parameter is first scaled by :math:`1 - \eta\lambda` (as the
    eager kernel does), then takes the Adam step; the decay never enters
    the moments.

    See Also
    --------
    :class:`lucid.optim.AdamW` : eager counterpart.
    """

    _ALWAYS_DECOUPLED = True
    _DEFAULT_WD = 1e-2


# ── NAdam ───────────────────────────────────────────────────────────


@final
class _CompiledNAdam(_AdamLike):
    r"""Compiled :class:`~lucid.optim.NAdam` — Adam with Nesterov lookahead.

    With :math:`\mu_t = \beta_1(1 - \tfrac12 \cdot 0.96^{t\psi})`
    (:math:`\psi = 0.004`, the engine's fixed momentum decay) and the
    per-parameter running product :math:`\Pi_t = \prod_{k\le t}\mu_k`:

    .. math::

        \theta_t = \theta_{t-1}
            - \frac{\eta(1-\mu_t)}{1-\Pi_t} \frac{g_t}{d_t}
            - \frac{\eta\mu_{t+1}}{1-\Pi_t\mu_{t+1}} \frac{m_t}{d_t},
        \quad d_t = \sqrt{v_t / (1-\beta_2^t)} + \varepsilon

    :math:`\Pi_t` is per-parameter scalar state rounded to float32 every
    step, as the engine keeps it (``mu_product`` in the state dict).

    See Also
    --------
    :class:`lucid.optim.NAdam` : eager counterpart.
    """

    _DEFAULT_LR = 2e-3
    _MOMENTUM_DECAY: float = 0.004
    _PSTATE_NAMES = ("mu_product",)

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return self._BETA_SCALARS + ("eps",) + self._decay_names(flags)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("inv_bc2", "c1", "c2")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        values = super()._group_values(group, flags)
        values["momentum_decay"] = _hp(group, "momentum_decay", self._MOMENTUM_DECAY)
        return values

    def _mu(self, beta1: float, step: int, momentum_decay: float = _MOMENTUM_DECAY) -> float:
        return beta1 * (1.0 - 0.5 * float(0.96 ** (float(step) * momentum_decay)))

    def _next_mu_product(
        self, i: int, step: int, beta1: float, momentum_decay: float = _MOMENTUM_DECAY
    ) -> float:
        prod = self._pstate[i].get("mu_product", 1.0)
        return _round_state_scalar(
            prod * self._mu(beta1, step, momentum_decay), self._params[i]._impl.dtype
        )

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        b1 = gv["beta1"]
        md = gv["momentum_decay"]
        mu = self._mu(b1, step, md)
        mu_next = self._mu(b1, step + 1, md)
        mu_prod = self._next_mu_product(i, step, b1, md)
        mu_prod_next = mu_prod * mu_next
        bc2 = 1.0 - gv["beta2"] ** float(step)
        lr = gv["lr"]
        return {
            "inv_bc2": 1.0 / bc2,
            "c1": lr * (1.0 - mu) / (1.0 - mu_prod),
            "c2": lr * mu_next / (1.0 - mu_prod_next),
        }

    @override
    def _init_pstate(self, i: int, group: dict[str, object]) -> None:
        self._pstate[i] = {"mu_product": 1.0}

    @override
    def _advance(self, i: int, step: int, gv: dict[str, float]) -> None:
        self._pstate[i]["mu_product"] = self._next_mu_product(
            i, step, gv["beta1"], gv["momentum_decay"]
        )

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        base, g, m, v = self._prepared(p, g, state, gs, flags)
        denom = (ps("inv_bc2") * v).sqrt() + gs("eps")
        new_p = (base - ps("c1") * (g / denom)) - ps("c2") * (m / denom)
        return new_p, {"exp_avg": m, "exp_avg_sq": v}


# ── RAdam ───────────────────────────────────────────────────────────


@final
class _CompiledRAdam(_AdamLike):
    r"""Compiled :class:`~lucid.optim.RAdam` — Rectified Adam.

    The rectification term :math:`r_t` and the "variance tractable"
    test :math:`\rho_t > 5` depend only on the parameter's step, so both
    are per-parameter scalars and the branch is a ``where`` on a 0-D
    flag:

    .. math::

        \hat m_t &= m_t / (1-\beta_1^t) \\
        \theta_t &= \theta_{t-1} - \begin{cases}
            \eta r_t \hat m_t \sqrt{1-\beta_2^t} / (\sqrt{v_t} + \varepsilon)
                & \rho_t > 5 \\
            \eta \hat m_t & \text{otherwise}
        \end{cases}

    See Also
    --------
    :class:`lucid.optim.RAdam` : eager counterpart.
    """

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("lr",) + self._BETA_SCALARS + ("eps",) + self._decay_names(flags)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("inv_bc1", "bc2_sqrt", "lr_rt", "use_rect")

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        b1, b2 = gv["beta1"], gv["beta2"]
        bc1 = 1.0 - b1 ** float(step)
        bc2 = 1.0 - b2 ** float(step)
        rho_inf = 2.0 / (1.0 - b2) - 1.0
        rho_t = rho_inf - 2.0 * step * b2 ** float(step) / bc2
        use_rect = rho_t > 5.0
        r_t = 0.0
        if use_rect:
            r_t = math.sqrt(
                (rho_t - 4.0)
                * (rho_t - 2.0)
                * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
        return {
            "inv_bc1": 1.0 / bc1,
            "bc2_sqrt": math.sqrt(bc2),
            "lr_rt": gv["lr"] * r_t,
            "use_rect": 1.0 if use_rect else 0.0,
        }

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        base, g, m, v = self._prepared(p, g, state, gs, flags)
        m_hat = ps("inv_bc1") * m
        adaptive = ps("bc2_sqrt") / (v.sqrt() + gs("eps"))
        p_rect = base - ps("lr_rt") * (m_hat * adaptive)
        p_sgd = base - gs("lr") * m_hat
        new_p = lucid.where(ps("use_rect") > 0.5, p_rect, p_sgd)
        return new_p, {"exp_avg": m, "exp_avg_sq": v}
