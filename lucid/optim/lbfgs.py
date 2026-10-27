"""
L-BFGS optimizer (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
"""

import math
import numbers
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, NamedTuple, Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.nn.parameter import Parameter
from lucid._dispatch import _wrap
from lucid.optim.optimizer import Optimizer, _follow

from lucid._C import engine as _C_engine

# Strong-Wolfe constants: sufficient decrease (c1) and curvature (c2).
_C1 = 1e-4
_C2 = 0.9
# The reference's line search stops shrinking its bracket at this width; it
# does not take the group's ``tolerance_change`` for it, so neither does this.
_LS_TOLERANCE_CHANGE = 1e-9
# A curvature pair with ``yᵀs`` at or below this is not stored.
_CURVATURE_EPS = 1e-10
_LINE_SEARCHES = (None, "strong_wolfe")

# A step, slope or scale.  Where the reference holds one as a 0-d tensor of
# the gradient's dtype this does too, so every comparison rounds the way the
# reference's does — a float64 Armijo test rejects, at the loss plateau, the
# float32 point the reference accepts, and the two runs part there.
type _Scalar = float | Tensor


# ── flat views of the parameters ─────────────────────────────────────────────
#
# L-BFGS treats every parameter as one slice of a single flat vector; these
# are the only places that map between the two.


def _flat_grad(params: list[Tensor]) -> Tensor:
    views = [
        (
            lucid.zeros(p.numel(), dtype=p.dtype, device=p.device)
            if p.grad is None
            else p.grad.detach().reshape(-1)
        )
        for p in params
    ]
    return lucid.cat(views)


def _add_step(params: list[Tensor], t: _Scalar, d: Tensor) -> None:
    """``params += t * d`` in place, so a parameter keeps its buffer and stays a leaf."""
    offset = 0
    for p in params:
        n = p.numel()
        p.copy_(p + d[offset : offset + n].reshape(p.shape) * t)
        offset += n


def _snapshot(params: list[Tensor]) -> list[Tensor]:
    return [p.detach().clone() for p in params]


def _restore(params: list[Tensor], saved: list[Tensor]) -> None:
    for p, value in zip(params, saved, strict=True):
        p.copy_(value)


def _evaluate(closure: Callable[[], Tensor], params: list[Tensor]) -> tuple[float, Tensor]:
    with lucid.enable_grad():
        loss = closure()
    return float(loss), _flat_grad(params)


# ── scalar arithmetic of the line search ─────────────────────────────────────


def _max2(a: _Scalar, b: _Scalar) -> _Scalar:
    """``max(a, b)`` — written out because a tensor comparison is not a ``bool``."""
    return b if b > a else a


def _min2(a: _Scalar, b: _Scalar) -> _Scalar:
    """``min(a, b)`` — written out because a tensor comparison is not a ``bool``."""
    return b if b < a else a


def _quotient(a: float, b: _Scalar) -> _Scalar:
    """``a / b``; a zero float ``b`` gives inf or nan as a tensor would, not an error."""
    if isinstance(b, Tensor):
        return a / b
    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


class _LinePoint(NamedTuple):
    """One evaluation along the search line: step, loss, gradient, slope."""

    t: _Scalar
    f: float
    g: Tensor
    gtd: Tensor


def _cubic_interpolate(
    p1: _LinePoint, p2: _LinePoint, bounds: tuple[_Scalar, _Scalar] | None = None
) -> _Scalar:
    """Minimiser of the cubic through two points with slopes, clamped to ``bounds``."""
    x1, f1, g1 = p1.t, p1.f, p1.gtd
    x2, f2, g2 = p2.t, p2.f, p2.gtd
    if bounds is not None:
        lo, hi = bounds
    else:
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * _quotient(f1 - f2, x1 - x2)
    d2_square = d1**2 - g1 * g2
    if not d2_square >= 0:
        return (lo + hi) / 2.0
    d2 = d2_square.sqrt()
    if x1 <= x2:
        min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    else:
        min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
    return _min2(_max2(min_pos, lo), hi)


type _Objective = Callable[[_Scalar], _LinePoint]


def _wolfe_bracket(
    obj: _Objective, start: _LinePoint, t: _Scalar, max_ls: int
) -> tuple[list[_LinePoint], bool, int]:
    """Grow the step until an interval holds a strong-Wolfe point.

    Returns the bracket, whether its single point already satisfies the
    conditions, and the number of evaluations made after the first.
    """
    prev, new = start, obj(t)
    ls_iter = 0
    while ls_iter < max_ls:
        if new.f > start.f + _C1 * new.t * start.gtd or (ls_iter > 1 and new.f >= prev.f):
            return [prev, new], False, ls_iter
        if abs(new.gtd) <= -_C2 * start.gtd:
            return [new], True, ls_iter
        if new.gtd >= 0:
            return [prev, new], False, ls_iter
        bounds = (new.t + 0.01 * (new.t - prev.t), new.t * 10)
        prev, new = new, obj(_cubic_interpolate(prev, new, bounds))
        ls_iter += 1
    return [start, new], False, ls_iter


def _safeguard(t: _Scalar, a: _Scalar, b: _Scalar, insuf_progress: bool) -> tuple[_Scalar, bool]:
    """Move a zoom trial ``t`` off the ends of ``[a, b]`` when progress stalls."""
    hi, lo = _max2(a, b), _min2(a, b)
    eps = 0.1 * (hi - lo)
    if not _min2(hi - t, t - lo) < eps:
        return t, False
    if insuf_progress or t >= hi or t <= lo:
        return (hi - eps if abs(t - hi) < abs(t - lo) else lo + eps), False
    return t, True


def _low_high(bracket: list[_LinePoint]) -> tuple[int, int]:
    return (0, 1) if bracket[0].f <= bracket[-1].f else (1, 0)


def _wolfe_zoom(
    obj: _Objective,
    start: _LinePoint,
    bracket: list[_LinePoint],
    ls_iter: int,
    max_ls: int,
    d_norm: Tensor,
) -> tuple[_LinePoint, int]:
    """Shrink ``bracket`` until a point satisfies the strong-Wolfe conditions."""
    insuf_progress = False
    low, high = _low_high(bracket)
    while ls_iter < max_ls:
        if abs(bracket[1].t - bracket[0].t) * d_norm < _LS_TOLERANCE_CHANGE:
            break
        t, insuf_progress = _safeguard(
            _cubic_interpolate(bracket[0], bracket[1]), bracket[0].t, bracket[1].t, insuf_progress
        )
        new = obj(t)
        ls_iter += 1
        if new.f > start.f + _C1 * t * start.gtd or new.f >= bracket[low].f:
            bracket[high] = new
            low, high = _low_high(bracket)
            continue
        if abs(new.gtd) <= -_C2 * start.gtd:
            return new, ls_iter
        if new.gtd * (bracket[high].t - bracket[low].t) >= 0:
            bracket[high] = bracket[low]
        bracket[low] = new
    return bracket[low], ls_iter


def _strong_wolfe(
    obj: _Objective, d: Tensor, start: _LinePoint, t: _Scalar, max_ls: int
) -> tuple[_LinePoint, int]:
    """Strong-Wolfe line search by cubic interpolation (bracket, then zoom).

    ``start`` is the point at step 0.  Returns the accepted point and the
    number of objective evaluations it took.
    """
    bracket, done, ls_iter = _wolfe_bracket(obj, start, t, max_ls)
    if done:
        return bracket[0], ls_iter + 1
    point, ls_iter = _wolfe_zoom(obj, start, bracket, ls_iter, max_ls, d.abs().max())
    return point, ls_iter + 1


# ── the two-loop recursion ───────────────────────────────────────────────────


def _two_loop(
    flat_grad: Tensor,
    old_dirs: list[Tensor],
    old_stps: list[Tensor],
    ro: list[Tensor],
    h_diag: _Scalar,
) -> Tensor:
    """``-H g`` for the L-BFGS inverse-Hessian estimate ``H`` of the stored pairs.

    The coefficients stay 0-d tensors, so the recursion does not wait on the
    device once per pair.
    """
    q = flat_grad.neg()
    al: dict[int, Tensor] = {}
    for i in range(len(old_dirs) - 1, -1, -1):
        al[i] = lucid.linalg.dot(old_stps[i], q) * ro[i]
        q = q - old_dirs[i] * al[i]
    r = q * h_diag
    for i, (y, s) in enumerate(zip(old_dirs, old_stps, strict=True)):
        be = lucid.linalg.dot(y, r) * ro[i]
        r = r + s * (al[i] - be)
    return r


# ── state ────────────────────────────────────────────────────────────────────


@dataclass
class _LBFGSState:
    """The optimizer's global state, under the reference framework's keys."""

    func_evals: int = 0
    n_iter: int = 0
    d: Tensor | None = None
    t: _Scalar | None = None
    old_dirs: list[Tensor] = field(default_factory=list)
    old_stps: list[Tensor] = field(default_factory=list)
    ro: list[Tensor] = field(default_factory=list)
    H_diag: _Scalar = 1.0
    prev_flat_grad: Tensor | None = None
    prev_loss: float | None = None

    def snapshot(self, to_host: Callable[[Tensor], object]) -> dict[str, object]:
        """Every entry by name, each tensor passed through ``to_host``."""
        out: dict[str, object] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                out[f.name] = to_host(v)
            elif isinstance(v, list):
                out[f.name] = [to_host(x) for x in v]
            else:
                out[f.name] = v
        return out

    def follow(self, like: Tensor) -> None:
        """Bring every stored tensor to ``like``'s dtype and device.

        The parameters may have been converted since the last step
        (``module.to()``, ``.double()``, ``.half()``); the curvature pairs,
        direction and scalars then follow them by the rule every optimizer's
        state follows (:func:`~lucid.optim.optimizer._follow`), so the next
        iteration does not mix the old dtype or device with the new.

        Every stored tensor is computed from the same flat gradient, so they
        share one dtype and one device: the last gradient stands for all of
        them, and the history is walked only when it is out of place —
        walking it on every step cost a deep history 60% of ``step``.
        """
        probe = self.prev_flat_grad
        if probe is None or (probe.dtype == like.dtype and probe.device == like.device):
            return
        for name, v in self.snapshot(partial(_follow_tensor, like=like)).items():
            setattr(self, name, v)

    @classmethod
    def restore(cls, snap: dict[str, object], to_tensor: Callable[[object], Tensor | None]) -> Self:
        """Rebuild from :meth:`snapshot`; a wrongly typed entry raises ``TypeError``."""

        def tensor(v: object, key: str) -> Tensor:
            out = to_tensor(v)
            if out is None:
                raise TypeError(f"LBFGS state '{key}' must hold arrays, got {type(v).__name__}")
            return out

        def tensors(key: str) -> list[Tensor]:
            v = snap.get(key, [])
            if not isinstance(v, list):
                raise TypeError(f"LBFGS state '{key}' must be a list, got {type(v).__name__}")
            return [tensor(x, key) for x in v]

        def optional_tensor(key: str) -> Tensor | None:
            v = snap.get(key)
            return None if v is None else tensor(v, key)

        def scalar(key: str, default: _Scalar | None) -> _Scalar | None:
            v = snap.get(key, default)
            if v is None or isinstance(v, (float, Tensor)):
                return v
            return _number(v, key) if isinstance(v, numbers.Real) else tensor(v, key)

        h_diag = scalar("H_diag", 1.0)
        prev_loss = snap.get("prev_loss")
        return cls(
            func_evals=int(_number(snap.get("func_evals", 0), "func_evals")),
            n_iter=int(_number(snap.get("n_iter", 0), "n_iter")),
            d=optional_tensor("d"),
            t=scalar("t", None),
            old_dirs=tensors("old_dirs"),
            old_stps=tensors("old_stps"),
            ro=tensors("ro"),
            H_diag=1.0 if h_diag is None else h_diag,
            prev_flat_grad=optional_tensor("prev_flat_grad"),
            prev_loss=None if prev_loss is None else _number(prev_loss, "prev_loss"),
        )


def _follow_tensor(v: Tensor, like: Tensor) -> Tensor:
    if v.dtype == like.dtype and v.device == like.device:
        return v
    return _wrap(_follow(v._impl, like._impl))


def _number(v: object, key: str) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"LBFGS state '{key}' must be a number, got {type(v).__name__}")
    return float(v)


class _Settings(NamedTuple):
    """One group's options, read once per :meth:`LBFGS.step`."""

    lr: float
    max_iter: int
    max_eval: int
    tolerance_grad: float
    tolerance_change: float
    history_size: int
    line_search_fn: str | None


def _should_stop(
    cfg: _Settings, st: _LBFGSState, n_iter: int, current_evals: int, flat_grad: Tensor, loss: float
) -> bool:
    """The reference's stopping rules, checked after each iteration's move."""
    if n_iter == cfg.max_iter or current_evals >= cfg.max_eval:
        return True
    if flat_grad.abs().max() <= cfg.tolerance_grad:
        return True
    if (
        st.d is not None
        and st.t is not None
        and ((st.d * st.t).abs().max() <= cfg.tolerance_change)
    ):
        return True
    return st.prev_loss is not None and abs(loss - st.prev_loss) < cfg.tolerance_change


class LBFGS(Optimizer):
    r"""Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.

    L-BFGS is a quasi-Newton method that approximates the inverse Hessian
    using a limited history of gradient and parameter difference vectors.
    One call to :meth:`step` runs up to ``max_iter`` iterations.  Each
    iteration computes a search direction :math:`d_k` via the two-loop
    recursion:

    .. math::

        d_k = -H_k \nabla L(\theta_k)

    where :math:`H_k` is the L-BFGS inverse-Hessian approximation built from
    the last ``history_size`` curvature pairs
    :math:`\{(s_i, y_i)\}`:

    .. math::

        s_i &= \theta_{i+1} - \theta_i \\
        y_i &= \nabla L(\theta_{i+1}) - \nabla L(\theta_i)

    with the initial diagonal scaling

    .. math::

        H_{\text{diag}} = \frac{s_{k-1}^\top y_{k-1}}{y_{k-1}^\top y_{k-1}}.

    A pair is stored only when :math:`y^\top s > 10^{-10}`.  The parameters
    then move to :math:`\theta_{k+1} = \theta_k + t_k d_k`.  With
    ``line_search_fn=None`` the step is fixed, :math:`t_k = \text{lr}`
    (on the very first iteration :math:`\text{lr} \cdot
    \min(1, 1/\lVert\nabla L\rVert_1)`).  With ``"strong_wolfe"``,
    :math:`t_k` is found by cubic interpolation so that

    .. math::

        L(\theta_k + t d_k) &\le L(\theta_k) + c_1 t \, \nabla L(\theta_k)^\top d_k \\
        \lvert \nabla L(\theta_k + t d_k)^\top d_k \rvert
        &\le c_2 \lvert \nabla L(\theta_k)^\top d_k \rvert

    with :math:`c_1 = 10^{-4}` and :math:`c_2 = 0.9`.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise.  Only one parameter group is supported.
    lr : float, optional
        Step size (default: ``1.0``).  The fixed step without a line search,
        and the first trial step of the strong-Wolfe search.
    max_iter : int, optional
        Maximum number of L-BFGS iterations per :meth:`step` call
        (default: ``20``).
    max_eval : int or None, optional
        Maximum number of closure evaluations per :meth:`step` call
        (default: ``None``, meaning ``max_iter * 5 // 4``).
    tolerance_grad : float, optional
        :meth:`step` stops once :math:`\lVert\nabla L\rVert_\infty \le`
        ``tolerance_grad`` (default: ``1e-7``).
    tolerance_change : float, optional
        :meth:`step` stops once the directional derivative, the parameter
        change :math:`\lVert t_k d_k\rVert_\infty` or the change of the loss
        falls below this (default: ``1e-9``).
    history_size : int, optional
        Number of :math:`(s, y)` curvature pairs retained in memory
        (default: ``100``).
    line_search_fn : str or None, optional
        ``"strong_wolfe"`` or ``None`` for a fixed step (default: ``None``).

    Attributes
    ----------
    param_groups : list of dict
        The single parameter group.
    defaults : dict
        Default hyperparameter values.

    Raises
    ------
    ValueError
        If ``params`` holds more than one parameter group, or an option is
        out of range (``lr < 0``, ``history_size < 1``, an unknown
        ``line_search_fn``).

    Notes
    -----
    Unlike first-order optimizers, L-BFGS **requires a closure** argument
    in :meth:`step` that clears gradients, computes the loss, and calls
    ``loss.backward()``.  Without a closure the method raises
    :exc:`ValueError`.

    The state (the last direction ``d`` and step ``t``, the curvature pairs
    ``old_dirs`` / ``old_stps`` with their ``ro = 1 / yᵀs``, ``H_diag``,
    ``prev_flat_grad``, ``prev_loss``, ``n_iter`` and ``func_evals``) carries
    over from one :meth:`step` to the next and round-trips through
    :meth:`state_dict`.

    L-BFGS is best suited for full-batch or large-batch training where the
    curvature information is reliable.  It is not recommended for
    stochastic mini-batch training because noisy gradients corrupt the
    Hessian approximation.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> x = lucid.randn(1, 4)
    >>> y = lucid.zeros((1, 2))
    >>> criterion = nn.MSELoss()
    >>> optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
    >>> def closure():
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(x), y)
    ...     loss.backward()
    ...     return loss
    >>> loss = optimizer.step(closure)   # step returns the closure's first loss
    """

    _HYPERPARAMS = _Settings._fields

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
    ) -> None:
        """Initialise the LBFGS.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_iter * 5 // 4 if max_eval is None else max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        self._lbfgs_state = _LBFGSState()
        super().__init__(params, defaults)

    # LBFGS is a pure-Python optimizer; no C++ engine optim needed.
    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        pass

    @override
    def add_param_group(self, group: dict[str, object]) -> None:
        """Refuse a second group: L-BFGS optimises one flat vector of parameters.

        Raises
        ------
        ValueError
            When the optimizer already has its parameter group.
        """
        if self.param_groups:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")
        super().add_param_group(group)

    @override
    @classmethod
    def _check_hyperparams(cls, values: dict[str, Any]) -> None:
        """Hold a group to the reference framework's LBFGS rule: ``lr >= 0``.

        Written so a NaN fails, as the reference's ``not 0.0 <= lr`` does.
        ``history_size`` and ``line_search_fn`` are checked here too, where
        the reference takes a bad value and fails inside a later ``step``.
        """
        if not 0.0 <= values["lr"]:
            raise _C_engine.InvalidArgument("LBFGS: lr must be >= 0")
        if values["history_size"] < 1:
            raise _C_engine.InvalidArgument("LBFGS: history_size must be >= 1")
        if values["line_search_fn"] not in _LINE_SEARCHES:
            raise _C_engine.InvalidArgument(
                "LBFGS: line_search_fn must be None or 'strong_wolfe', "
                f"got {values['line_search_fn']!r}"
            )

    def _params(self) -> list[Tensor]:
        params = self.param_groups[0]["params"]
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("LBFGS: the parameter group must hold a list of tensors")
        return params

    # ── state_dict round-trip ────────────────────────────────────────────────
    #
    # LBFGS has only global state.  Like the reference, it is filed under the
    # first parameter's index (0), with tensors as host arrays so a checkpoint
    # carries no autograd plumbing.

    @override
    def _save_state(self) -> dict[int, dict[str, object]]:
        st = self._lbfgs_state
        if st.func_evals == 0:
            return {}
        return {0: st.snapshot(lambda v: v.detach().numpy().copy())}

    @override
    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        import numpy as np

        snap = state.get(0)
        if snap is None:
            self._lbfgs_state = _LBFGSState()
            return
        anchor = self._params()[0]

        def to_tensor(v: object) -> Tensor | None:
            if isinstance(v, Tensor):
                return v.detach().to(anchor.device)
            if isinstance(v, np.ndarray):
                return lucid.tensor(v, device=anchor.device)
            return None

        self._lbfgs_state = _LBFGSState.restore(snap, to_tensor)

    # ── one iteration's pieces ────────────────────────────────────────────────

    def _direction(self, flat_grad: Tensor, history_size: int) -> Tensor:
        """Store the last curvature pair, then return the L-BFGS direction."""
        st = self._lbfgs_state
        if st.n_iter == 1:
            st.old_dirs, st.old_stps, st.ro, st.H_diag = [], [], [], 1.0
            return flat_grad.neg()
        if st.prev_flat_grad is None or st.d is None or st.t is None:
            raise RuntimeError(
                f"LBFGS state is incomplete: n_iter is {st.n_iter - 1} but the "
                "previous direction, step or gradient is missing"
            )
        y = flat_grad - st.prev_flat_grad
        s = st.d * st.t
        ys = lucid.linalg.dot(y, s)
        if ys > _CURVATURE_EPS:
            while len(st.old_dirs) >= history_size:
                st.old_dirs.pop(0)
                st.old_stps.pop(0)
                st.ro.pop(0)
            st.old_dirs.append(y)
            st.old_stps.append(s)
            st.ro.append(1.0 / ys)
            st.H_diag = ys / lucid.linalg.dot(y, y)
        return _two_loop(flat_grad, st.old_dirs, st.old_stps, st.ro, st.H_diag)

    def _probe(
        self,
        closure: Callable[[], Tensor],
        params: list[Tensor],
        x: list[Tensor],
        d: Tensor,
        t: _Scalar,
    ) -> _LinePoint:
        """Evaluate at ``x + t d`` and put the parameters back to ``x``."""
        _add_step(params, t, d)
        f, g = _evaluate(closure, params)
        _restore(params, x)
        return _LinePoint(t, f, g, lucid.linalg.dot(g, d))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Run up to ``max_iter`` L-BFGS iterations.

        Each iteration takes the two-loop-recursion direction, moves along it
        by the fixed step ``lr`` or by a strong-Wolfe line search, and stores
        the new curvature pair.  The loop ends after ``max_iter`` iterations,
        after ``max_eval`` closure evaluations, or when a ``tolerance_grad`` /
        ``tolerance_change`` rule is met.

        Parameters
        ----------
        closure : callable
            A zero-argument callable that:

            1. Calls ``optimizer.zero_grad()`` to clear stale gradients.
            2. Runs the forward pass and computes the scalar loss.
            3. Calls ``loss.backward()`` to populate gradients.
            4. Returns the loss tensor.

            This argument is **required** — passing ``None`` raises
            :exc:`ValueError`.

        Returns
        -------
        Tensor
            The closure's loss at the parameters :meth:`step` started from
            (its first evaluation), as the reference framework returns it.

        Raises
        ------
        ValueError
            If ``closure`` is ``None``.

        Notes
        -----
        The closure is called up to ``max_eval`` times per :meth:`step`
        call, always with gradients enabled.  Ensure that any side effects
        (e.g. batch norm running stats) are handled appropriately if this
        matters for your use-case.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> model = nn.Linear(4, 2)
        >>> x = lucid.randn(1, 4)
        >>> y = lucid.zeros((1, 2))
        >>> criterion = nn.MSELoss()
        >>> optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     output = model(x)
        ...     loss = criterion(output, y)
        ...     loss.backward()
        ...     return loss
        >>> loss = optimizer.step(closure)   # step returns the closure's first loss
        """
        if closure is None:
            raise ValueError("L-BFGS requires a closure that reevaluates the model")
        self._refuse_phantom_step()

        # The group is read below as it is; an edited one is checked first.
        self._sync_hyperparams()
        cfg = _Settings(**self._hyperparam_dict(self._hyperparams_of(self.param_groups[0])))
        params = self._params()
        with lucid.no_grad():
            with lucid.enable_grad():
                orig_loss = closure()
            self._refuse_phantom_step()
            self._lbfgs_state.func_evals += 1
            flat_grad = _flat_grad(params)
            self._lbfgs_state.follow(flat_grad)
            if not flat_grad.abs().max() <= cfg.tolerance_grad:
                self._iterate(closure, params, cfg, float(orig_loss), flat_grad)
        return orig_loss

    def _iterate(
        self,
        closure: Callable[[], Tensor],
        params: list[Tensor],
        cfg: _Settings,
        loss: float,
        flat_grad: Tensor,
    ) -> None:
        """The iteration loop of :meth:`step`, after its first evaluation."""
        st = self._lbfgs_state
        current_evals = 1
        n_iter = 0
        while n_iter < cfg.max_iter:
            n_iter += 1
            st.n_iter += 1
            d = st.d = self._direction(flat_grad, cfg.history_size)
            st.prev_flat_grad, st.prev_loss = flat_grad, loss
            # The first step of all is scaled down by the gradient's 1-norm.
            t = _min2(1.0, 1.0 / flat_grad.abs().sum()) * cfg.lr if st.n_iter == 1 else cfg.lr
            st.t = t
            gtd = lucid.linalg.dot(flat_grad, d)
            if gtd > -cfg.tolerance_change:
                break
            evals = 0
            if cfg.line_search_fn is not None:
                obj = partial(self._probe, closure, params, _snapshot(params), d)
                start = _LinePoint(0.0, loss, flat_grad, gtd)
                point, evals = _strong_wolfe(obj, d, start, t, cfg.max_eval - current_evals)
                loss, flat_grad, st.t = point.f, point.g, point.t
                _add_step(params, point.t, d)
            else:
                _add_step(params, t, d)
                if n_iter != cfg.max_iter:
                    # No point re-evaluating after the last move: the
                    # caller's next step evaluates there first anyway.
                    loss, flat_grad = _evaluate(closure, params)
                    evals = 1
            current_evals += evals
            st.func_evals += evals
            if _should_stop(cfg, st, n_iter, current_evals, flat_grad, loss):
                break
