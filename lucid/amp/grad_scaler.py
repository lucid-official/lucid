"""
GradScaler for mixed-precision training.

Pure Python implementation — no engine changes required.
"""

import enum
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.optim.optimizer import Optimizer

# Gradient dtypes ``unscale_`` divides.  float16 is refused with the reference
# framework's ``ValueError``.  Every other dtype is refused as unsupported.
_UNSCALABLE = frozenset({_C_engine.BF16, _C_engine.F32, _C_engine.F64})


class _Stage(enum.Enum):
    """Where one optimizer stands in the current scale → step → update cycle."""

    READY = "ready"  # nothing has touched its gradients since the last update()
    UNSCALED = "unscaled"  # unscale_() divided its gradients by the scale
    STEPPED = "stepped"  # step() ran (or skipped) its update


class _OptimizerState:
    """What the scaler recorded about one optimizer since the last ``update()``.

    ``found_inf`` is per optimizer because the skip is: an overflow in one
    optimizer's gradients skips that optimizer's step only, while
    :meth:`GradScaler.update` backs the scale off if *any* of them overflowed.
    """

    __slots__ = ("stage", "found_inf")

    def __init__(self) -> None:
        """Start a fresh cycle: nothing unscaled, nothing found."""
        self.stage = _Stage.READY
        self.found_inf = False


class GradScaler:
    r"""Dynamic loss-scaling helper for mixed-precision training.

    Mixed-precision training runs the forward pass in fp16 to halve
    memory bandwidth and exploit fp16-fast hardware paths, but
    fp16's narrow dynamic range causes small gradients to underflow to
    zero — the network stops learning.  :class:`GradScaler` works
    around this by multiplying the loss by a large constant :math:`s`
    before backpropagation:

    .. math::

        \tilde{L} = s \cdot L, \qquad
        \frac{\partial \tilde{L}}{\partial \theta}
        = s \cdot \frac{\partial L}{\partial \theta}.

    The scaled gradients sit comfortably inside fp16's representable
    range; before the optimizer step they are unscaled by :math:`1/s`
    in fp32 so the update is mathematically equivalent to ordinary
    training.

    The scale itself is adapted dynamically.  After every step the
    unscaled gradients are checked for ``inf`` / ``NaN``:

    * **Overflow detected** — the step is skipped and :math:`s` is
      multiplied by ``backoff_factor`` (typically ``0.5``).
    * **No overflow for ``growth_interval`` consecutive steps** —
      :math:`s` is multiplied by ``growth_factor`` (typically ``2.0``).

    This produces a sawtooth schedule that tracks the largest scale
    the current gradient distribution can tolerate.

    Parameters
    ----------
    init_scale : float, default=2**16
        Initial loss scaling factor applied by :meth:`scale`.
    growth_factor : float, default=2.0
        Multiplier applied to the scale after ``growth_interval``
        consecutive non-overflowing steps.  Must be ``> 1.0``.
    backoff_factor : float, default=0.5
        Multiplier applied when an ``inf`` / ``NaN`` gradient is
        detected.  Must be in ``(0, 1)``.
    growth_interval : int, default=2000
        Number of overflow-free steps required before the scale grows.
    enabled : bool, default=True
        When ``False`` the scaler degenerates into a transparent
        pass-through — :meth:`scale` returns its input unchanged,
        :meth:`step` calls the optimizer directly, and :meth:`update`
        is a no-op.

    Notes
    -----
    The canonical training-loop pattern is *scale-loss, then step,
    then update*:

    1. :meth:`scale` multiplies the loss by :math:`s` before
       ``backward()`` so the gradients land safely inside fp16 range.
    2. :meth:`step` unscales the gradients, checks for ``inf`` /
       ``NaN``, and either runs ``optimizer.step()`` or skips the
       update.
    3. :meth:`update` adjusts :math:`s` according to the
       growth / backoff schedule for the next iteration.

    To work on the real gradients between ``backward()`` and the step —
    gradient clipping is the usual reason — call :meth:`unscale_` first.
    The scaler remembers, per optimizer, that its gradients are already
    unscaled, so the following :meth:`step` does not divide them a second
    time.  Each optimizer moves through *ready → unscaled → stepped* once
    per iteration and :meth:`update` returns every optimizer to *ready*;
    calling :meth:`unscale_` twice, :meth:`unscale_` after :meth:`step`, or
    :meth:`step` twice before :meth:`update` raises :class:`RuntimeError`.

    With several optimizers on one scaler, an overflow in one optimizer's
    gradients skips only that optimizer's step; :meth:`update` backs the
    scale off if any optimizer overflowed.

    The parameters themselves stay float32: :class:`lucid.amp.autocast`
    moves the forward pass into half precision, not the weights, so the
    gradients arrive in float32 too.  bfloat16 and float64 gradients are
    unscaled as well, and unscaling never changes a gradient's dtype.  A
    float16 gradient — from a float16 parameter — makes :meth:`unscale_`
    and :meth:`step` raise :class:`ValueError`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> from lucid.amp import GradScaler, autocast
    >>> model = nn.Linear(4, 1).to("metal")
    >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
    >>> loss_fn = nn.MSELoss()
    >>> dataloader = [
    ...     (lucid.randn(8, 4, device="metal"), lucid.randn(8, 1, device="metal"))
    ...     for _ in range(3)
    ... ]
    >>> scaler = GradScaler()
    >>> for x, y in dataloader:
    ...     optimizer.zero_grad()
    ...     with autocast():
    ...         out = model(x)
    ...     loss = loss_fn(out.float(), y)
    ...     scaler.scale(loss).backward()
    ...     scaler.step(optimizer)
    ...     scaler.update()

    Clipping the unscaled gradients — :meth:`step` sees that
    :meth:`unscale_` already ran and leaves them alone:

    >>> from lucid.nn.utils import clip_grad_norm_
    >>> x, y = dataloader[0]
    >>> optimizer.zero_grad()
    >>> scaler.scale(loss_fn(model(x), y)).backward()
    >>> scaler.unscale_(optimizer)
    >>> norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    >>> scaler.step(optimizer)
    >>> scaler.update()
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        """Initialize the scaler state.

        Parameters
        ----------
        init_scale : float, default=2**16
            Initial loss scaling factor applied by :meth:`scale`.
        growth_factor : float, default=2.0
            Multiplier applied to the scale after ``growth_interval``
            consecutive non-overflowing steps.
        backoff_factor : float, default=0.5
            Multiplier applied when an inf/NaN gradient is detected.
        growth_interval : int, default=2000
            Number of overflow-free steps required before the scale grows.
        enabled : bool, default=True
            When ``False`` the scaler is a transparent pass-through.
        """
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        # Whether any optimizer overflowed since the last update() — the one
        # bit update() reads.  ``lucid.compile``'s fused step, which unscales
        # inside its own graph, writes it directly before calling update().
        self._found_inf = False
        # Keyed by ``id(optimizer)``; emptied by update().
        self._per_optimizer_states: dict[int, _OptimizerState] = {}

    def scale(self, outputs: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        """Multiply outputs by the current scale factor.

        Args:
            outputs: A Tensor or list of Tensors to scale.

        Returns:
            Scaled Tensor(s) — same structure as input.
        """
        if not self._enabled:
            return outputs

        from lucid._tensor.tensor import Tensor

        if isinstance(outputs, Tensor):
            return outputs * self._scale
        return [o * self._scale for o in outputs]

    def _state_for(self, optimizer: Optimizer) -> _OptimizerState:
        """Return this cycle's record for ``optimizer``, creating it if new."""
        key = id(optimizer)
        state = self._per_optimizer_states.get(key)
        if state is None:
            state = _OptimizerState()
            self._per_optimizer_states[key] = state
        return state

    def unscale_(self, optimizer: Optimizer) -> None:
        """Divide the optimizer's gradients by the current scale, in place.

        Call it between ``backward()`` and :meth:`step` when the real
        gradients are needed — to clip them, or to inspect their norm.  The
        scaler records that this optimizer's gradients are unscaled, so the
        :meth:`step` that follows does not divide them again.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer whose parameters' gradients are unscaled.

        Raises
        ------
        RuntimeError
            If :meth:`unscale_` already ran for ``optimizer`` since the last
            :meth:`update`, or :meth:`step` already ran for it.
        ValueError
            If any of ``optimizer``'s gradients is float16.  Nothing is
            unscaled and the optimizer's stage does not change.
        NotImplementedError
            If any of ``optimizer``'s gradients is complex, again before
            anything is unscaled.

        Notes
        -----
        Every gradient is divided, including those of a parameter whose
        gradient holds an ``inf`` / ``NaN`` (those entries stay non-finite);
        the overflow is recorded for this optimizer and its :meth:`step`
        is skipped.

        Every gradient keeps its dtype.  A bfloat16 gradient is cast to
        float32, divided and finiteness-checked there, then rounded back to
        bfloat16 once.  float32 and float64 gradients are unscaled in their
        own dtype.

        Float16 gradients are refused, as the reference framework refuses
        them.  They come from float16 *parameters*, and loss scaling
        assumes float32 master weights: :class:`lucid.amp.autocast` runs
        the forward in half precision while the parameters and their
        gradients stay float32.  Dividing a float16 gradient by the default
        scale ``2**16`` puts most of its entries below float16's smallest
        normal value ``6.1e-5``.  Metal flushes those subnormals to zero.
        Keep the parameters in float32 and let :class:`lucid.amp.autocast`
        choose the half-precision ops.
        """
        if not self._enabled:
            return
        state = self._state_for(optimizer)
        if state.stage is _Stage.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer "
                "since the last update()."
            )
        if state.stage is _Stage.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        found_inf = self._unscale_grads(optimizer)
        state.found_inf = found_inf
        state.stage = _Stage.UNSCALED
        self._found_inf = self._found_inf or found_inf

    def _unscale_grads(self, optimizer: Optimizer) -> bool:
        """Multiply every gradient of ``optimizer`` by ``1 / scale``.

        Returns whether any gradient held an ``inf`` or ``NaN``.  The check
        reads one flag per device rather than one per parameter, so a model
        on a single device costs a single host synchronisation.

        Each gradient is unscaled in place and keeps its dtype, so a tensor
        read from ``p.grad`` before the call reads the unscaled values too.
        The check reads the values before they are unscaled, as the
        reference's fused unscale does.  All dtypes are checked before any
        gradient is touched, so a refused call leaves the optimizer's
        gradients and stage as they were.
        """
        from lucid._dispatch import _unwrap, _wrap
        from lucid.autograd._grad_inplace import write_grad_

        grads: list[tuple[Tensor, _C_engine.TensorImpl]] = []
        for group in optimizer.param_groups:
            for p in group["params"]:  # type: ignore[attr-defined]
                if p.grad is None:
                    continue
                g_impl = _unwrap(p.grad)
                if g_impl.dtype == _C_engine.F16:
                    raise ValueError("Attempting to unscale FP16 gradients.")
                if g_impl.dtype not in _UNSCALABLE:
                    # Complex: an autocast-eligible multiply would cast it
                    # to a real dtype.  The reference framework has no
                    # complex unscale kernel either.
                    raise NotImplementedError(
                        f"GradScaler cannot unscale {p.grad.dtype} gradients."
                    )
                grads.append((p, g_impl))

        inv_scale = 1.0 / self._scale
        nonfinite: dict[_C_engine.Device, _C_engine.TensorImpl] = {}
        for p, g_impl in grads:
            grad_dtype = g_impl.dtype
            # bfloat16 is unscaled in float32 and rounded back once, as the
            # reference framework's fused unscale computes in its float
            # op-math type.  float32 / float64 are unscaled in their own
            # dtype: mixed-dtype multiply is not supported (BinaryKernel
            # validates same-dtype operands).
            work_dtype = _C_engine.F32 if grad_dtype == _C_engine.BF16 else grad_dtype
            g_work = (
                g_impl
                if grad_dtype == work_dtype
                else _C_engine.astype(g_impl, work_dtype)
            )
            bad = _C_engine.any(_C_engine.logical_not(_C_engine.isfinite(g_work)))
            seen = nonfinite.get(g_work.device)
            nonfinite[g_work.device] = (
                bad if seen is None else _C_engine.logical_or(seen, bad)
            )
            coef = _C_engine.full(
                list(g_work.shape), inv_scale, work_dtype, g_work.device
            )
            unscaled = _C_engine.mul(g_work, coef)
            if g_work is not g_impl:
                unscaled = _C_engine.astype(unscaled, grad_dtype)
            write_grad_(p, unscaled, "GradScaler.unscale_")
        return any(bool(_wrap(flag).item()) for flag in nonfinite.values())

    def step(
        self, optimizer: Optimizer, *args: object, **kwargs: object
    ) -> Tensor | None:
        """Unscale the gradients if needed, then step unless they overflowed.

        If :meth:`unscale_` already ran for ``optimizer`` in this iteration
        its gradients are used as they are; otherwise they are unscaled
        here.  The optimizer's ``step()`` is skipped when its gradients held
        an ``inf`` or ``NaN``.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to step.
        *args : object
            Forwarded to ``optimizer.step()``.
        **kwargs : object
            Forwarded to ``optimizer.step()``.  ``closure`` is rejected
            while the scaler is enabled: a closure re-runs ``backward()``
            and its gradients would never be unscaled.

        Returns
        -------
        Tensor or None
            What ``optimizer.step()`` returned, or ``None`` when the step
            was skipped.

        Raises
        ------
        RuntimeError
            If :meth:`step` already ran for ``optimizer`` since the last
            :meth:`update`, or ``closure`` is passed while enabled.
        ValueError
            If the gradients still need unscaling and one of them is
            float16 — see :meth:`unscale_`.
        NotImplementedError
            If the gradients still need unscaling and one of them is
            complex.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)  # type: ignore[arg-type]

        if "closure" in kwargs:
            raise RuntimeError(
                "Closure use is not currently supported if GradScaler is enabled."
            )
        state = self._state_for(optimizer)
        if state.stage is _Stage.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )
        if state.stage is _Stage.READY:
            self.unscale_(optimizer)
        retval = (
            None
            if state.found_inf
            else optimizer.step(*args, **kwargs)  # type: ignore[arg-type]
        )
        state.stage = _Stage.STEPPED
        return retval

    def update(self, new_scale: float | None = None) -> None:
        """Update the scale factor and start the next iteration.

        If a scale is provided, it is set directly. Otherwise, the scale
        is reduced by ``backoff_factor`` if any optimizer's gradients
        overflowed since the last update, or grown by ``growth_factor``
        after ``growth_interval`` overflow-free updates.  Either way every
        optimizer returns to the state where :meth:`unscale_` and
        :meth:`step` may be called again.

        Parameters
        ----------
        new_scale : float, optional
            Explicit new scale value.
        """
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
        elif self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

        self._found_inf = False
        self._per_optimizer_states.clear()

    def get_scale(self) -> float:
        """Return the current scale factor."""
        return self._scale

    def state_dict(self) -> dict[str, float]:
        """Return serializable state dict."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        """Load state from a dict."""
        self._scale = float(state_dict["scale"])
        self._growth_factor = float(state_dict["growth_factor"])
        self._backoff_factor = float(state_dict["backoff_factor"])
        self._growth_interval = int(state_dict["growth_interval"])
        self._growth_tracker = int(state_dict["growth_tracker"])
