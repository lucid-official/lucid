"""
Gradient clipping utilities.
All computation goes through the C++ engine — no numpy.
"""

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING, override
import math
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._factories.creation import zeros
from lucid.autograd._grad_inplace import write_grad_
from lucid.nn.utils._checks import _as_tensor_list

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_D = _C_engine.Dtype
_NORMABLE = frozenset({_D.F16, _D.BF16, _D.F32, _D.F64, _D.C64})
_REAL_OF_COMPLEX = {_D.C64: _D.F32}

_Group = list[_C_engine.TensorImpl]


def _by_device(impls: _Group) -> dict[_C_engine.Device, _Group]:
    """``impls`` split by device, first-seen order.

    A tensor is reduced where it lives; only the per-device scalar is read back.
    Stacking tensors of different devices raised ``DeviceMismatch``.
    """
    groups: dict[_C_engine.Device, _Group] = {}
    for impl in impls:
        groups.setdefault(impl.device, []).append(impl)
    return groups


def _group_dtype(group: _Group) -> _C_engine.Dtype:
    """The dtype a group's norm accumulates in.

    ``float32`` for everything below it: the reference's half-precision norm
    kernels accumulate in ``float32`` too, and squaring in ``float16``
    overflows at 256 (``|g| = 300`` summed to ``inf`` and zeroed the gradients).
    """
    return _D.F64 if any(i.dtype == _D.F64 for i in group) else _D.F32


def _magnitude(
    impl: _C_engine.TensorImpl, dt: _C_engine.Dtype, take_abs: bool
) -> _C_engine.TensorImpl:
    """``|impl|`` (or ``impl`` itself when ``take_abs`` is off and it is real) as ``dt``."""
    if take_abs or impl.dtype in _REAL_OF_COMPLEX:
        impl = _C_engine.abs(impl)
    return impl if impl.dtype == dt else _C_engine.astype(impl, dt)


def _item(impl: _C_engine.TensorImpl) -> float:
    return float(_wrap(impl).item())


def _safe_root(value: float, order: float) -> float:
    """``value ** (1 / order)`` where Python would raise instead of answering.

    ``0 ** negative`` is ``inf`` and an overflowing power is ``inf``, as in the
    float arithmetic the reference runs.
    """
    if order == 2.0:
        return math.sqrt(value)
    try:
        return math.pow(value, 1.0 / order)
    except ValueError:  # 0 ** negative
        return math.inf
    except OverflowError:  # past the float range
        return math.inf


class _NormOrder(ABC):
    """One order :math:`p` of the joint norm, reduced device by device.

    A norm over tensors that live on several devices is the combination of
    one partial result per device, so each order says how to take a partial
    from a group on one device, how partials combine, and how the combined
    value becomes the norm.
    """

    @abstractmethod
    def partial(self, group: _Group, dt: _C_engine.Dtype) -> float:
        """The reduction of the tensors of ``group``, all on one device."""

    @abstractmethod
    def combine(self, partials: list[float]) -> float:
        """The reduction of per-device partial results."""

    def finish(self, value: float) -> float:
        """The norm, given the combined reduction."""
        return value


class _Extremum(_NormOrder):
    """:math:`\\pm\\infty` orders: the largest / smallest magnitude."""

    def __init__(self, largest: bool) -> None:
        self._largest = largest

    @override
    def partial(self, group: _Group, dt: _C_engine.Dtype) -> float:
        reduce = _C_engine.max if self._largest else _C_engine.min
        # An empty tensor has no extreme; it is left out, not an error.
        norms = [
            reduce(_magnitude(i, dt, take_abs=True), [], False)
            for i in group
            if math.prod(i.shape) != 0
        ]
        if not norms:
            return self._identity()
        return _item(norms[0] if len(norms) == 1 else reduce(_C_engine.stack(norms, 0), [0], False))

    def _identity(self) -> float:
        return -math.inf if self._largest else math.inf

    @override
    def combine(self, partials: list[float]) -> float:
        if any(math.isnan(v) for v in partials):
            return math.nan
        return max(partials) if self._largest else min(partials)


class _NonzeroTensors(_NormOrder):
    """Order 0: the reference takes the order-0 norm of the per-tensor order-0
    norms, which counts the tensors holding a non-zero entry."""

    @override
    def partial(self, group: _Group, dt: _C_engine.Dtype) -> float:
        counts = [
            _C_engine.sum(_C_engine.sign(_magnitude(i, dt, take_abs=True)), [], False)
            for i in group
        ]
        return _item(_C_engine.sum(_C_engine.sign(_C_engine.stack(counts, 0)), [0], False))

    @override
    def combine(self, partials: list[float]) -> float:
        return sum(partials)


class _PowerSum(_NormOrder):
    """Every other order: :math:`(\\sum |x|^p)^{1/p}`."""

    def __init__(self, order: float) -> None:
        self._order = order

    @override
    def partial(self, group: _Group, dt: _C_engine.Dtype) -> float:
        sums = [_C_engine.sum(self._term(i, dt), [], False) for i in group]
        return _item(
            sums[0] if len(sums) == 1 else _C_engine.sum(_C_engine.stack(sums, 0), [0], False)
        )

    def _term(self, impl: _C_engine.TensorImpl, dt: _C_engine.Dtype) -> _C_engine.TensorImpl:
        """The entries of ``impl`` as they enter the sum."""
        if self._order == 2.0:
            return _C_engine.square(_magnitude(impl, dt, take_abs=False))
        return _C_engine.pow_scalar(_magnitude(impl, dt, take_abs=True), self._order)

    @override
    def combine(self, partials: list[float]) -> float:
        return sum(partials)

    @override
    def finish(self, value: float) -> float:
        return _safe_root(value, self._order)


def _norm_order(norm_type: float) -> _NormOrder:
    if math.isinf(norm_type):
        return _Extremum(largest=norm_type > 0)
    if norm_type == 0.0:
        return _NonzeroTensors()
    return _PowerSum(norm_type)


def _total_norm(impls: _Group, norm_type: float, error_if_nonfinite: bool) -> float:
    """The :math:`\\ell_p` norm of ``impls`` taken jointly, as a Python float."""
    order = _norm_order(norm_type)
    partials = [order.partial(g, _group_dtype(g)) for g in _by_device(impls).values()]
    total_norm = order.finish(order.combine(partials))
    if error_if_nonfinite and not math.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


def _norm_tensor(impls: _Group, total_norm: float) -> Tensor:
    """The 0-d result of a joint norm over ``impls``.

    It lives where the first tensor does and has the dtype the reference's
    stack of per-tensor norms promotes to (``float16`` beside ``float32`` gives
    ``float32``; a complex tensor's norm is real).
    """
    dt = _REAL_OF_COMPLEX.get(impls[0].dtype, impls[0].dtype)
    for impl in impls[1:]:
        dt = _promotion.promote_types(dt, _REAL_OF_COMPLEX.get(impl.dtype, impl.dtype))
    return _wrap(_C_engine.full([], total_norm, dt, impls[0].device))


def _gather_grads(params: list[Tensor]) -> tuple[list[Tensor], _Group]:
    """The parameters that hold a gradient, and those gradients."""
    held: list[Tensor] = []
    grads: _Group = []
    for p in params:
        g_impl = p._impl.grad_as_impl() or p._impl.grad_to_tensor()
        if g_impl is not None:
            held.append(p)
            grads.append(g_impl)
    return held, grads


def _scale_clipped_grads(params_with_grad: list[Tensor], grads: _Group, clip_coef: float) -> None:
    amp_active = _C_engine.amp_is_active()
    coefs: dict[tuple[_C_engine.Dtype, _C_engine.Device], _C_engine.TensorImpl] = {}
    for p, g_impl in zip(params_with_grad, grads, strict=True):
        g_dt = g_impl.dtype
        w_dt = _D.F32 if (amp_active and g_dt in (_D.F16, _D.BF16)) else g_dt
        g_w = g_impl if g_dt == w_dt else _C_engine.astype(g_impl, w_dt)
        key = (w_dt, g_w.device)
        coef = coefs.get(key)
        if coef is None:
            coef = _C_engine.full([], clip_coef, w_dt, g_w.device)
            coefs[key] = coef
        scaled = _C_engine.mul(g_w, coef)
        if g_w is not g_impl:
            scaled = _C_engine.astype(scaled, g_dt)
        write_grad_(p, scaled, "clip_grad_norm_")


def clip_grad_norm_(
    parameters: Tensor | Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> Tensor:
    r"""Clip the global gradient norm of ``parameters`` in place.

    Rescales every gradient so that the total :math:`\ell_p` norm —
    computed across all parameters jointly, as if they were one long
    concatenated vector — is at most ``max_norm``.  A staple of stable
    Transformer / RNN training: prevents the occasional huge gradient
    from derailing optimisation.

    Parameters
    ----------
    parameters : Tensor or iterable of Tensor
        Parameters whose ``.grad`` should be clipped; one ``Tensor`` counts
        as a one-element list.  Entries with ``grad is None`` are silently
        skipped.
    max_norm : float
        Maximum allowed norm of the combined gradient vector.  The
        scaling factor never exceeds ``1`` — gradients smaller than
        ``max_norm`` are untouched.
    norm_type : float, optional
        Order :math:`p` of the norm.  Default ``2.0`` (Euclidean);
        ``math.inf`` is the max-norm over all gradients and ``-math.inf``
        the smallest magnitude; ``0`` counts the gradients with a non-zero
        entry, as the reference does.  Gradients on different devices are
        reduced where they live.
    error_if_nonfinite : bool, optional
        If ``True``, raise :class:`RuntimeError` when the computed total
        norm is ``inf`` or ``nan`` instead of silently scaling by a
        non-finite coefficient.
    foreach : bool, optional
        Accepted for API compatibility; ignored.

    Returns
    -------
    Tensor
        Scalar tensor holding the *pre-clipping* total norm, in the dtype
        the gradients promote to — useful for logging even when nothing was
        clipped.  Half-precision gradients are summed in ``float32``.

    Notes
    -----
    With combined norm :math:`\|g\|_p = \left(\sum_i |g_i|^p\right)^{1/p}`
    taken over every element of every gradient, the update is

    .. math::

        g \;\mapsto\; g \cdot \min\!\left(1,\,
            \frac{\text{max\_norm}}{\|g\|_p + \epsilon}\right),

    where the :math:`\epsilon = 10^{-6}` guards against division by zero
    when all gradients vanish.  Because every parameter is scaled by the
    *same* coefficient the direction of the global update is preserved —
    only its magnitude is bounded.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import clip_grad_norm_
    >>> model = nn.Linear(4, 2)
    >>> (model(lucid.randn(8, 4)) * 100.0).sum().backward()
    >>> total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    >>> total_norm.shape                        # a 0-d scalar
    ()
    >>> float(total_norm) > 1.0                 # the norm before clipping
    True
    >>> after = clip_grad_norm_(model.parameters(), max_norm=1.0)
    >>> abs(float(after) - 1.0) < 1e-3          # and after it
    True
    """
    del foreach  # reference-signature parity
    params = _as_tensor_list(parameters, "clip_grad_norm_")
    params_with_grad, grads = _gather_grads(params)
    if not params_with_grad:
        # Preserve the parameters' device/dtype for the zero-norm result —
        # defaulting to CPU/float32 here breaks multi-device / mixed-precision
        # callers.  Fall back to the global default only when there are no
        # parameters at all (nothing to infer from).
        if not params:
            return zeros()
        impl0 = params[0]._impl
        return _wrap(_C_engine.full([], 0.0, impl0.dtype, impl0.device))

    norm_type = float(norm_type)
    total_norm = _total_norm(grads, norm_type, error_if_nonfinite)

    clip_coef = float(max_norm) / (total_norm + 1e-6)
    # ``not >=`` rather than ``<``: a NaN coefficient scales too, as the
    # reference's clamp-then-multiply does.
    if not clip_coef >= 1.0:
        _scale_clipped_grads(params_with_grad, grads, clip_coef)

    return _norm_tensor(grads, total_norm)


def clip_grad_value_(
    parameters: Tensor | Iterable[Tensor], clip_value: float, foreach: bool | None = None
) -> None:
    r"""Clamp every gradient element to :math:`[-\text{clip\_value}, \text{clip\_value}]` in place.

    Unlike :func:`clip_grad_norm_`, which preserves the direction of the
    full gradient vector, this operates element-wise — each scalar entry
    is independently clipped to the symmetric interval.  Cheap, and a
    useful safety net when only a handful of weights tend to blow up
    (e.g. embedding tables on rare tokens).

    Parameters
    ----------
    parameters : Tensor or iterable of Tensor
        Parameters whose ``.grad`` should be clipped; one ``Tensor`` counts
        as a one-element list.  Entries with ``grad is None`` are skipped.
    clip_value : float
        Symmetric magnitude bound; gradients are clamped to
        ``[-clip_value, +clip_value]``.  A negative value inverts the
        interval, and every element becomes ``clip_value``, as in the
        reference.
    foreach : bool, optional
        Accepted for API compatibility; ignored.

    Returns
    -------
    None
        The clipping happens in place via ``Parameter.grad``.

    Notes
    -----
    The element-wise update is

    .. math::

        g_i \;\mapsto\; \mathrm{clip}(g_i,\, -c,\, +c),

    with :math:`c = \text{clip\_value}`.  Because each component is
    treated independently the direction of the gradient vector is *not*
    preserved — large coordinates are flattened toward zero while small
    ones pass through unchanged.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import clip_grad_value_
    >>> model = nn.Linear(4, 2)
    >>> (model(lucid.randn(8, 4)) * 100.0).sum().backward()
    >>> clip_grad_value_(model.parameters(), clip_value=0.5)
    >>> max(float(p.grad.abs().max()) for p in model.parameters()) <= 0.5
    True
    """
    del foreach  # reference-signature parity
    bound = float(clip_value)
    # An inverted interval (bound < 0) sends every element to ``bound``; the
    # engine kernels disagree on one, so it is collapsed to a point here.
    lower = min(-bound, bound)
    params_with_grad, grads = _gather_grads(_as_tensor_list(parameters, "clip_grad_value_"))
    for p, g_impl in zip(params_with_grad, grads, strict=True):
        write_grad_(p, _C_engine.clip(g_impl, lower, bound), "clip_grad_value_")


def get_total_norm(
    tensors: Iterable[Tensor] | Tensor,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> Tensor:
    r"""Compute the :math:`\ell_p` norm of ``tensors``, taken jointly.

    The measurement step of :func:`clip_grad_norm_`, which hands it the
    parameters' gradients.  It measures the tensors it is given, as the
    reference framework's does — it used to read ``.grad`` from them, so
    the usual call, ``get_total_norm([p.grad for p in model.parameters()])``,
    read the gradients' own (absent) gradients and answered 0.

    Parameters
    ----------
    tensors : iterable of Tensor, or Tensor
        The tensors whose elements make up the norm — typically the
        gradients.  ``None`` entries are skipped; the empty case returns a
        zero tensor.
    norm_type : float, optional
        Order :math:`p` of the norm.  Default ``2.0``; use ``math.inf``
        for the max-norm.
    error_if_nonfinite : bool, optional
        Raise on ``inf`` / ``nan`` results instead of returning them.
    foreach : bool, optional
        Accepted for API compatibility; ignored — Lucid processes the
        tensors sequentially through the C++ engine.

    Returns
    -------
    Tensor
        Scalar tensor with the combined norm.

    Notes
    -----
    The returned scalar is

    .. math::

        \|g\|_p \;=\; \left(\sum_i |g_i|^p\right)^{1/p}

    taken over every element of every tensor — the quantity
    :func:`clip_grad_norm_` thresholds against ``max_norm``.  For
    ``norm_type=0`` it is the reference's norm of the per-tensor norms of
    order 0, which counts the tensors with any non-zero entry.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.clip_grad import get_total_norm
    >>> get_total_norm([lucid.tensor([3.0, 4.0])]).item()
    5.0
    >>> model = nn.Linear(4, 2)
    >>> model(lucid.randn(8, 4)).sum().backward()
    >>> g_norm = get_total_norm([p.grad for p in model.parameters()])
    >>> g_norm.shape
    ()
    >>> float(g_norm) > 0.0
    True
    """
    del foreach  # reference-signature parity
    impls = [_unwrap(t) for t in _as_tensor_list(tensors, "get_total_norm", skip_none=True)]
    if not impls:
        return zeros()
    # A norm is a real number, and integer tensors have no norm kernel on
    # every device — the CPU refused int64 while Metal computed it.  The
    # reference refuses them too; so does this, the same way everywhere.
    for impl in impls:
        if impl.dtype not in _NORMABLE:
            raise TypeError(
                f"get_total_norm: expected floating-point or complex tensors, got {impl.dtype}"
            )
    # 0-d, like :func:`clip_grad_norm_`'s result.
    return _norm_tensor(impls, _total_norm(impls, float(norm_type), error_if_nonfinite))


clip_grad_norm = clip_grad_norm_
