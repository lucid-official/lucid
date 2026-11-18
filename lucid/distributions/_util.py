"""Tiny helpers shared across the distribution implementations."""

from collections.abc import Callable
from numbers import Real
from typing import Protocol, Self, overload, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dtype import finfo
from lucid._engine_scope import untraced
from lucid._tensor._indexing import broadcast_shapes
from lucid._tensor.tensor import Tensor


def _require_parameter(value: object) -> None:
    """Raise ``TypeError`` unless ``value`` can be a distribution parameter.

    The one place the rule lives: a parameter is a tensor or a real number.
    Left to ``float``, a string would parse and a one-element array would
    convert, so a single-parameter law took what a multi-parameter one refused.
    """
    if not isinstance(value, (lucid.Tensor, Real)):
        raise TypeError(
            f"a distribution parameter must be a Tensor or a number, got {type(value).__name__}"
        )


def _as_tensor(x: Tensor | float | int) -> Tensor:
    """Promote a Python scalar to a 0-dim Lucid tensor — matches the
    reference framework's convention so distributions parameterised by
    scalars get ``batch_shape == ()`` rather than ``(1,)``.

    Raises
    ------
    TypeError
        If ``x`` is neither a tensor nor a real number (the rule
        :func:`_promote_params` applies to the same parameters).
    """
    if isinstance(x, lucid.Tensor):
        return x
    _require_parameter(x)
    with untraced():
        t = lucid.tensor(float(x))
        if t.ndim == 1 and t.shape == (1,):
            t = t.squeeze()
    return t


def _promote_params(*values: Tensor | float | int) -> tuple[Tensor, ...]:
    """Tensors for the parameters of one distribution, numbers following the tensors.

    A parameter given as a Python number takes the floating dtype and the
    device of the first tensor parameter among ``values`` — what the reference
    framework does when it broadcasts a distribution's parameters together —
    and the default dtype when there is none.  Converted on its own, a number
    was a default-dtype tensor before anything said what it would meet:
    ``0.1`` was rounded to float32 ahead of the float64 arithmetic, and a
    float32 count beside a float64 probability of the same shape was never
    promoted, so ``lgamma(n + 1)`` ran in float32.

    A tensor is returned as it is, so an integer tensor keeps its dtype and
    does not turn a number beside it into an integer: the reference would
    truncate the ``0.3`` of ``Binomial(int_count, probs=0.3)`` to ``0`` and then
    raise ``TypeError`` from ``finfo`` the first time it scored a value.

    Parameters
    ----------
    *values : Tensor | float | int
        One distribution's parameters, in any mixture of tensors and numbers.

    Returns
    -------
    tuple of Tensor
        One tensor per value, in order.  Numbers are 0-dim.

    Raises
    ------
    TypeError
        If a value is neither a tensor nor a real number.
    """
    for v in values:
        _require_parameter(v)
    tensors = [v for v in values if isinstance(v, lucid.Tensor)]
    if not tensors:
        return tuple(_as_tensor(v) for v in values)
    like = next((t for t in tensors if t.is_floating_point()), tensors[0])
    dtype = like.dtype if like.is_floating_point() else lucid.get_default_dtype()
    with untraced():
        return tuple(
            (
                v
                if isinstance(v, lucid.Tensor)
                else lucid.full((), float(v), dtype=dtype, device=like.device)
            )
            for v in values
        )


def _broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """The right-aligned broadcast of two shapes, as a shape."""
    return broadcast_shapes(a, b)


def _broadcast_to(t: Tensor, shape: tuple[int, ...]) -> Tensor:
    """``t`` as a view of ``shape`` — ``t`` itself when it already has that shape.

    The one way a distribution brings a parameter to a larger shape.  A view
    does no arithmetic, so every value survives bit for bit (an infinity or a
    NaN in one lane cannot reach another), and it joins the autograd graph only
    if ``t`` is in it: a detached parameter stays detached, a leaf that is
    broadcast keeps its gradient, and a parameter that needs no broadcast stays
    the very tensor it was.
    """
    return t if tuple(t.shape) == shape else t.broadcast_to(shape)


def _broadcast_pair(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast ``a`` and ``b`` to a common shape, as views.

    Arithmetic is no way to broadcast: ``a * 0 + b * 0`` made a finite parameter
    NaN beside an infinite one (``inf * 0``), tied every broadcast parameter to
    the autograd graph, and left a distribution that could not be deep-copied.
    Each of the pair is :func:`_broadcast_to` the joint shape instead.

    Parameters
    ----------
    a, b : Tensor
        Inputs to align.  Must be broadcast-compatible per the standard
        rules; otherwise the underlying engine raises.

    Returns
    -------
    tuple of Tensor
        ``(a_broadcast, b_broadcast)`` with identical shapes.

    Raises
    ------
    lucid._C.engine.DeviceMismatch
        If they live on different devices and neither is a 0-dim host tensor.
    """
    # A 0-dim host tensor is a scalar in every other sense, so beside a Metal
    # parameter it follows it — the allowance the reference framework makes.
    # This is the single point where the parameters of one distribution meet.
    if a.device != b.device:
        if a.ndim == 0 and a.device.type == "cpu":
            if not a.requires_grad:
                with untraced():
                    val = float(a.item())
                a = lucid.full(tuple(b.shape), val, dtype=b.dtype, device=b.device)
            else:
                a = a.to(b.device.type)
        elif b.ndim == 0 and b.device.type == "cpu":
            if not b.requires_grad:
                with untraced():
                    val = float(b.item())
                b = lucid.full(tuple(a.shape), val, dtype=a.dtype, device=a.device)
            else:
                b = b.to(a.device.type)
    # A view would not notice the mismatch until the first op that meets them.
    if a.device != b.device:
        raise _C_engine.DeviceMismatch(
            f"distribution parameters must share a device, got {a.device} and {b.device}"
        )
    shape = _broadcast_shapes(tuple(a.shape), tuple(b.shape))
    return _broadcast_to(a, shape), _broadcast_to(b, shape)


def _broadcast_params(*values: Tensor | float | int) -> tuple[Tensor, ...]:
    """One distribution's parameters as tensors of a common shape.

    The reference framework's ``broadcast_all``: :func:`_promote_params` turns
    numbers into tensors that follow the tensors beside them, and
    :func:`_broadcast_pair` brings every pair to one shape.  A constructor
    that scored its parameters together calls this rather than converting
    each on its own, which is how a number lost the dtype of its neighbour.

    Parameters
    ----------
    *values : Tensor | float | int
        The parameters, in any mixture of tensors and numbers.

    Returns
    -------
    tuple of Tensor
        One tensor per value, in order, all of the broadcast shape.
    """
    tensors = list(_promote_params(*values))
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            tensors[i], tensors[j] = _broadcast_pair(tensors[i], tensors[j])
    return tuple(tensors)


def _align_device(t: Tensor, ref: Tensor) -> Tensor:
    """``t`` on ``ref``'s device when ``t`` is a 0-dim CPU tensor.

    A transform or a parameter given as a Python number is held as a 0-dim
    host tensor (:func:`_as_tensor` cannot know the device the values will
    arrive on), and combining it with a Metal tensor raised
    ``DeviceMismatch``.  A 0-dim CPU tensor is a scalar in every other sense,
    so it follows the operand it meets — the allowance the reference
    framework makes for 0-dim CPU tensors.  Anything with a shape (or a
    0-dim Metal tensor against a CPU operand) is left alone: a real mismatch
    between two arrays still raises.

    Parameters
    ----------
    t : Tensor
        The tensor that may need to move.
    ref : Tensor
        The tensor whose device it should share.

    Returns
    -------
    Tensor
        ``t``, moved to ``ref.device`` if it is a 0-dim CPU tensor and
        ``ref`` is elsewhere.
    """
    if t.ndim == 0 and t.device.type == "cpu" and t.device != ref.device:
        if not t.requires_grad:
            with untraced():
                val = float(t.item())
            return lucid.full((), val, dtype=t.dtype, device=ref.device)
        return t.to(ref.device.type)
    return t


def _operand(t: Tensor, number: float | None, like: Tensor) -> Tensor | float:
    """The operand a transform combines with ``like``: a held number, else ``t`` aligned.

    A parameter given as a Python number is kept as that number (``number``)
    next to its 0-dim tensor ``t``.  Combined as a Python scalar it takes the
    dtype and device of ``like`` — a float64 input stays exact, where the
    float32 scalar ``t`` would round it — and the reference framework keeps the
    number the same way.  A tensor parameter (``number is None``) goes through
    :func:`_align_device` as before.
    """
    return _align_device(t, like) if number is None else number


def _sum_rightmost(value: Tensor, dim: int) -> Tensor:
    """``value`` with its ``dim`` rightmost dimensions summed out.

    A ``value`` with fewer than ``dim`` dimensions is summed whole, as in
    the reference framework.  The Wishart KL depends on this: its
    degrees-of-freedom term is 0-dim but is reduced by the rank of the
    matrix event.

    One deliberate departure from the reference: an empty kept dimension.
    The reference flattens through ``reshape(..., -1)``, which cannot
    infer the size when a kept dimension is 0, so a ``(0, 3)`` batch with
    ``dim=1`` raises there.  Summing over the axes returns the empty
    ``(0,)`` result instead, so an empty batch scores to an empty batch.
    """
    if dim <= 0 or value.ndim == 0:
        return value
    return value.sum(list(range(-min(dim, value.ndim), 0)))


class _ImageScorer(Protocol):
    """The base a pushed-forward family scores through."""

    def cdf(self, value: Tensor) -> Tensor: ...

    def icdf(self, value: Tensor) -> Tensor: ...


def _image_cdf(base: _ImageScorer, preimage: Tensor, *, fold: bool = False) -> Tensor:
    """The CDF of :math:`Y = g(X)` at ``y``, from ``preimage`` = :math:`g^{-1}(y)`.

    One rule for every family defined as a transform of a base
    distribution :math:`X`.  For an increasing ``g`` — ``LogNormal`` is
    ``exp`` of a ``Normal``, ``Pareto`` and ``Weibull`` maps of an
    ``Exponential`` — :math:`F_Y(y) = F_X(g^{-1}(y))`.  With ``fold``,
    :math:`Y = |X|` for an :math:`X` symmetric about 0 (``HalfNormal``,
    ``HalfCauchy``): :math:`F_Y(y) = F_X(y) - F_X(-y) = 2F_X(y) - 1`,
    held at 0 below the support, where the difference would go negative.

    The base must not validate: a value is checked once, against the
    family's own support, by the family's ``cdf``.
    """
    u = base.cdf(preimage)
    if not fold:
        return u
    return (2.0 * u - 1.0).clamp(min=0.0)


def _image_icdf(base: _ImageScorer, value: Tensor, *, fold: bool = False) -> Tensor:
    """The base quantile matching probability ``value`` under :func:`_image_cdf`.

    The caller maps it through ``g`` itself; for ``fold`` that map is the
    identity, since the quantile of :math:`|X|` at :math:`p` is
    :math:`X`'s at :math:`(1 + p)/2`.
    """
    return base.icdf((1.0 + value) / 2.0 if fold else value)


def _clamp_probs(probs: Tensor) -> Tensor:
    """``probs`` held one machine epsilon inside ``[0, 1]``.

    What the reference framework does before the logarithm that turns
    probabilities into logits: an exact 0 or 1 would otherwise give an
    infinite logit, and a derived ``logits`` attribute is expected to be
    finite whatever ``probs`` it came from.
    """
    eps = float(finfo(probs.dtype).eps)
    return probs.clip(eps, 1.0 - eps)


class _lazy_param[T](property):
    """A parameter a distribution either stores or derives from its dual.

    ``Bernoulli`` takes ``probs`` *or* ``logits``, and the reference
    framework answers both attributes whichever one was given.  Assigning
    stores the value on the instance; reading an attribute that was never
    assigned derives it from the other one.

    The derivation runs on every read rather than once.  A cached value
    goes stale the moment the stored parameter is trained in place, and it
    ties every later use to the autograd graph of whichever step happened
    to read it first.

    It is a ``property`` subclass so that it is one wherever a property is
    looked for: ``MultivariateNormal.covariance_matrix`` was a plain
    property before it could also be given, and introspection that lists a
    class's properties still finds it.

    Parameters
    ----------
    derive : callable
        ``derive(dist)`` computes the value from the stored dual.
    """

    def __init__(self, derive: Callable[[T], Tensor]) -> None:
        super().__init__(derive)
        self._derive = derive
        self._name = derive.__name__
        self.__doc__ = derive.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: T, owner: type | None = None) -> Tensor: ...

    @override
    def __get__(self, instance: T | None, owner: type | None = None) -> Self | Tensor:
        if instance is None:
            return self
        stored: Tensor | None = vars(instance).get(self._name)
        return stored if stored is not None else self._derive(instance)

    @override
    def __set__(self, instance: object, value: Tensor) -> None:
        vars(instance)[self._name] = value

    def is_stored(self, instance: object) -> bool:
        """Whether ``instance`` was given this parameter rather than its dual."""
        return self._name in vars(instance)


def _pick_probs_or_logits(
    cls_name: str, probs: Tensor | float | int | None, logits: Tensor | float | int | None
) -> tuple[Tensor | float | int, bool]:
    """Return ``(param, is_logits)`` — the one of ``probs`` / ``logits`` given, as given.

    The parameter is left unconverted, so a caller that scores it beside other
    parameters (a count, a temperature) can let a number follow their dtype
    through :func:`_promote_params`.
    """
    if probs is not None and logits is None:
        return probs, False
    if logits is not None and probs is None:
        return logits, True
    raise ValueError(f"{cls_name}: pass exactly one of `probs` or `logits`.")


def _require_probs_or_logits(
    cls_name: str, probs: Tensor | float | int | None, logits: Tensor | float | int | None
) -> tuple[Tensor, bool]:
    """Return ``(param, is_logits)`` when exactly one of ``probs`` or ``logits`` is passed."""
    param, is_logits = _pick_probs_or_logits(cls_name, probs, logits)
    return _as_tensor(param), is_logits


def _set_probs_or_logits(
    dist: object,
    cls_name: str,
    probs: Tensor | float | int | None,
    logits: Tensor | float | int | None,
) -> tuple[int, ...]:
    """Validate and store ``probs`` or ``logits`` on ``dist`` and return ``batch_shape``."""
    param, is_logits = _require_probs_or_logits(cls_name, probs, logits)
    vars(dist)["_is_logits"] = is_logits
    vars(dist)["logits" if is_logits else "probs"] = param
    return tuple(param.shape)
