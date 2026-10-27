"""
The boundary validators of the losses in :mod:`lucid.nn.functional.loss`.

Each public loss calls these at its top, before it computes anything, and
a malformed argument is refused with a typed exception, the same one on
the CPU and on Metal.  All but :func:`_check_unit_interval` read only
shapes, dtypes and Python values, so a refusal dispatches no op and waits
on no device.

Operand dtypes
--------------
Which dtypes a loss takes for its input and its target, and which dtype it
computes in, is decided here too (``_OPERAND_DTYPES``).  The losses used to
leave that to whichever op first touched the operand, so the answer
depended on the device and on the code path:

- ``nll_loss`` took an integer input on the CPU and returned an integer
  loss, and on Metal it raised ``ValueError`` about NaN;
- ``margin_ranking_loss`` refused the integer label that the reference
  takes;
- ``huber_loss`` returned a wrong number on the CPU for a float16 target
  beside a float32 input.

An operand of a dtype the loss does not take is now refused with a
``TypeError`` that names the loss and the dtype (:func:`_check_dtypes`).
The accepted operands are cast to one dtype (:func:`_loss_dtype`), so no
kernel sees a mixed pair.
"""

import math
import warnings
from collections.abc import Callable
from typing import Literal, TYPE_CHECKING, override, overload

import lucid as _lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid._tensor._indexing import broadcast_shapes
from lucid._types import Reduction, ReductionKL
from lucid.nn.functional._index_checks import _host_read
from lucid.nn.module import Module

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

#: The reductions every loss defines.  ``kl_div`` adds ``"batchmean"``.
_REDUCTIONS: tuple[str, ...] = ("none", "mean", "sum")


# ── boundary validators ─────────────────────────────────────────────────────
#
# Each raises before the loss computes.  All but ``_check_unit_interval``
# read shapes and Python values only, so a refusal dispatches no op at all.
# A wrong argument used to reach the arithmetic: a class weight of the wrong
# length was cut short or read out of bounds (a silent answer on the CPU,
# NaN or 0 on Metal), a BCE target of another shape was broadcast into a
# different loss, and nine losses looked at ``reduction`` only after
# computing.


def _check_reduction(reduction: object, op: str, *, batchmean: bool = False) -> None:
    """Refuse a ``reduction`` that the loss ``op`` does not define.

    ``reduction`` is typed ``object``: this is the boundary, and a caller
    may pass anything, so the ``str`` check is part of the contract."""
    valid = _REDUCTIONS + ("batchmean",) if batchmean else _REDUCTIONS
    if not isinstance(reduction, str) or reduction not in valid:
        raise ValueError(f"{op}: reduction must be one of {valid}, got {reduction!r}")


@overload
def _legacy_reduction(
    size_average: bool | None,
    reduce: bool | None,
    reduction: object,
    op: str,
    *,
    batchmean: Literal[False] = False,
) -> Reduction: ...


@overload
def _legacy_reduction(
    size_average: bool | None,
    reduce: bool | None,
    reduction: object,
    op: str,
    *,
    batchmean: bool,
) -> ReductionKL: ...


def _legacy_reduction(
    size_average: bool | None,
    reduce: bool | None,
    reduction: object,
    op: str,
    *,
    batchmean: bool = False,
) -> ReductionKL:
    """Resolve legacy ``size_average`` / ``reduce`` slots into ``reduction``
    (matching the reference's legacy reduction rule without a deprecation
    warning) and validate the resulting mode."""
    if size_average is not None or reduce is not None:
        sa = True if size_average is None else bool(size_average)
        red = True if reduce is None else bool(reduce)
        reduction = "mean" if (sa and red) else ("sum" if red else "none")
    _check_reduction(reduction, op, batchmean=batchmean)
    if reduction == "none":
        return "none"
    if reduction == "sum":
        return "sum"
    if reduction == "batchmean":
        return "batchmean"
    return "mean"


def _check_same_weight(weight: Tensor | None, x: Tensor, op: str) -> None:
    if weight is not None and _shape(weight) != _shape(x):
        raise ValueError(
            f"{op}: weight of shape {_shape(weight)} must have the same size as input {_shape(x)}"
        )


def _check_beta(beta: float, op: str = "smooth_l1_loss") -> None:
    if isinstance(beta := _promotion.as_number(beta), bool) or not beta >= 0.0:
        raise ValueError(f"{op}: beta must be non-negative, got {beta!r}")


def _check_delta(delta: float, op: str = "huber_loss") -> None:
    if isinstance(delta := _promotion.as_number(delta), bool) or not isinstance(
        delta, (int, float)
    ):
        raise TypeError(f"{op}: delta must be a float or int, got {type(delta).__name__}")
    if not delta > 0.0:
        raise ValueError(f"{op}: delta must be positive, got {delta!r}")


def _check_label_smoothing(label_smoothing: float, op: str = "cross_entropy") -> None:
    if (
        isinstance(label_smoothing := _promotion.as_number(label_smoothing), bool)
        or not 0.0 <= label_smoothing <= 1.0
    ):
        raise ValueError(f"{op}: label_smoothing must be in [0, 1], got {label_smoothing!r}")


def _check_multi_margin(p: int, weight: Tensor | None, op: str = "multi_margin_loss") -> None:
    if isinstance(p := _promotion.as_number(p), bool) or (p != 1 and p != 2):
        raise ValueError(f"{op}: only p == 1 and p == 2 supported, got {p!r}")
    if weight is not None and weight.ndim != 1:
        raise ValueError(f"{op}: expected weight to be None or 1D tensor, got {weight.ndim}D")


def _check_triplet_margin(margin: float, op: str) -> None:
    if isinstance(margin := _promotion.as_number(margin), bool) or not margin > 0.0:
        raise ValueError(f"{op}: expected margin to be greater than 0, got {margin!r}")


def _check_device(op: str, **tensors: Tensor | None) -> None:
    """Refuse a loss ``op``'s operands if two of them live on different
    devices, before a kernel dispatches assuming they share one storage
    and fails deep inside — naming the op that touched it last, not the
    loss (:func:`mse_loss` on a CPU/Metal pair raised a phantom-storage
    error from the fused kernel; :func:`l1_loss` named ``sub``).

    Pass every operand the loss takes, by the name its message should
    use — ``input``, ``target``, ``weight``, ``pos_weight``, ``var`` —
    including the ones that are ``None`` for this call; ``None`` is
    skipped, and so is a 0-d CPU tensor, which is a scalar that follows
    the other operands' device, as a binary op's own device check does
    (:func:`lucid._ops.composite._shared._binary_device`)."""
    cpu = _C_engine.Device.CPU
    ref: tuple[str, Tensor] | None = None
    for name, t in tensors.items():
        if t is None or (t.ndim == 0 and t._impl.device == cpu):
            continue
        if ref is None:
            ref = (name, t)
            continue
        ref_name, ref_t = ref
        if t._impl.device != ref_t._impl.device:
            raise _C_engine.DeviceMismatch(
                f"{op}: expected every operand on one device ({ref_name} on "
                f"{ref_t.device.type}), got {name} on {t.device.type}"
            )


def _shape(t: Tensor) -> tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The shape ``shapes`` broadcast to, or ``None`` if they do not."""
    try:
        return broadcast_shapes(*shapes)
    except _C_engine.ShapeMismatch:
        return None


def _check_broadcasts_to(t: Tensor | None, shape: tuple[int, ...], what: str, op: str) -> None:
    """``t`` scales a loss of ``shape`` element by element: it has to
    broadcast to ``shape`` without making it any larger."""
    if t is not None and _broadcast_shape(_shape(t), shape) != shape:
        raise ValueError(
            f"{op}: {what} of shape {_shape(t)} does not broadcast to the " f"input's shape {shape}"
        )


#: How each elementwise loss's ``target`` has to relate to its input's
#: shape — one rule per loss, the reference's:
#:
#: - ``"same"``: equal shapes.  Another shape is refused, not broadcast:
#:   broadcasting a BCE target scores a different problem.
#: - ``"within"``: the target broadcasts to the input's shape without
#:   enlarging it.  (For ``multilabel_soft_margin_loss`` this is stricter
#:   than the reference, which broadcasts the loss up.)
#: - ``"broadcast"``: input and target broadcast together.
#: - ``"broadcast-warn"``: the same, with a ``UserWarning`` when the shapes
#:   differ, as the reference's regression losses give — an ``(N, 1)``
#:   prediction against an ``(N,)`` target scores ``N * N`` pairs.
#:
#: The class-index losses take one index per sample instead
#: (:func:`_check_class_target`); the pair losses check their operands
#: together (:func:`_check_same_rank`).
_TARGET_SHAPE: dict[str, str] = {
    "binary_cross_entropy": "same",
    "binary_cross_entropy_with_logits": "same",
    "multilabel_margin_loss": "same",
    "soft_margin_loss": "within",
    "multilabel_soft_margin_loss": "within",
    "gaussian_nll_loss": "broadcast",
    "hinge_embedding_loss": "broadcast",
    "kl_div": "broadcast",
    "poisson_nll_loss": "broadcast",
    "huber_loss": "broadcast-warn",
    "l1_loss": "broadcast-warn",
    "mse_loss": "broadcast-warn",
    "smooth_l1_loss": "broadcast-warn",
}


def _check_target(x: Tensor, target: Tensor, op: str) -> tuple[int, ...]:
    """Check ``target``'s shape against ``x``'s by ``op``'s rule in
    ``_TARGET_SHAPE``, and return the shape of the elementwise loss."""
    rule = _TARGET_SHAPE[op]
    x_shape, t_shape = _shape(x), _shape(target)
    if rule == "same":
        if t_shape != x_shape:
            raise ValueError(
                f"{op}: target size {t_shape} must be the same as input size " f"{x_shape}"
            )
        return x_shape
    shape = _broadcast_shape(x_shape, t_shape)
    if shape is None or (rule == "within" and shape != x_shape):
        raise ValueError(
            f"{op}: target of shape {t_shape} does not broadcast "
            f"{'to' if rule == 'within' else 'with'} the input's shape {x_shape}"
        )
    if rule == "broadcast-warn" and t_shape != x_shape:
        warnings.warn(
            f"{op}: using a target size {t_shape} that is different to the "
            f"input size {x_shape}; the two are broadcast together, which "
            "is rarely what a regression means",
            UserWarning,
            stacklevel=3,
        )
    return shape


def _broadcast_operands(x: Tensor, target: Tensor, shape: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    """``x`` and ``target`` at the loss's ``shape``, for a fused kernel that
    takes operands of one shape.  An operand already at ``shape`` is passed
    through untouched, so the ordinary call dispatches nothing extra."""
    if _shape(x) != shape:
        x = _lucid.broadcast_to(x, shape)
    if _shape(target) != shape:
        target = _lucid.broadcast_to(target, shape)
    return x, target


def _check_same_rank(op: str, *, broadcast: bool = True, **tensors: Tensor) -> None:
    """The inputs of a pairwise loss line up element for element: one
    device (:func:`_check_device`), one rank, and (with ``broadcast``)
    shapes that broadcast together.  A user's distance function decides
    for itself which shapes it pairs, so ``triplet_margin_with_distance_loss``
    checks the rank only — but every triplet loss reaches this, unconditionally
    and before a custom distance function ever runs, for the device check."""
    _check_device(op, **tensors)
    shapes = {name: _shape(t) for name, t in tensors.items()}
    if len({len(s) for s in shapes.values()}) > 1 or (
        broadcast and _broadcast_shape(*shapes.values()) is None
    ):
        got = ", ".join(f"{name} {s}" for name, s in shapes.items())
        raise ValueError(
            f"{op}: the inputs are expected to have the same number of "
            f"dimensions{' and broadcastable shapes' if broadcast else ''}, "
            f"got {got}"
        )


def _num_classes(x: Tensor, op: str, *, max_ndim: int | None = None) -> int:
    """The class count of class scores ``x`` — dim 1 of ``(N, C, *)``, dim 0
    of one unbatched sample ``(C,)`` — refusing a rank the loss does not
    define."""
    if x.ndim == 0 or (max_ndim is not None and x.ndim > max_ndim):
        shapes = "(C,) or (N, C)" if max_ndim == 2 else "(C,), (N, C) or (N, C, *)"
        raise ValueError(f"{op}: expected an input of shape {shapes}, got {_shape(x)}")
    return int(x.shape[0] if x.ndim == 1 else x.shape[1])


def _check_class_weight(
    weight: Tensor | None, num_classes: int, x: Tensor, target: Tensor, op: str
) -> None:
    """A per-class ``weight`` has exactly one entry per class, and ``x``,
    ``target`` and ``weight`` all share one device
    (:func:`_check_device`) — the class-index losses call this
    unconditionally, before their first gather, so it is where their
    device check lives too (``cross_entropy``'s falls here rather than
    its own dtype check, which skips ``target`` for a class-probability
    one).

    The weight's shape is checked by value only — no entry is read, so
    this costs nothing on Metal.  A longer weight was silently cut to
    its first ``C`` entries, and a shorter one read out of bounds for
    the classes past its end."""
    _check_device(op, input=x, target=target, weight=weight)
    if weight is not None and (weight.ndim != 1 or int(weight.shape[0]) != num_classes):
        raise ValueError(
            f"{op}: weight tensor should be defined either for all {num_classes} "
            f"classes or no classes but got weight tensor of shape: "
            f"{list(_shape(weight))}"
        )


def _check_class_target(x: Tensor, target: Tensor, op: str) -> None:
    """``target`` holds one class index per sample of ``x``: the shape of
    ``x`` without its class dimension.  One unbatched ``(C,)`` sample takes a
    0-d target, or one of shape ``(1,)``."""
    want = (int(x.shape[0]),) + _shape(x)[2:] if x.ndim > 1 else ()
    got = _shape(target)
    if got != want and not (x.ndim == 1 and got == (1,)):
        raise ValueError(
            f"{op}: expected a target of shape {want} for an input of shape "
            f"{_shape(x)}, got {got}"
        )


def _check_unit_interval(t: Tensor, what: str, op: str) -> None:
    """Refuse a CPU ``t`` with an entry outside ``[0, 1]`` — NaN included —
    as the reference's CPU kernel does.

    A CPU tensor is on the host already, so the read costs no sync.  A
    Metal tensor is not read back (a host read in every loss call stalls
    every step), so it is not checked, and an entry outside the interval
    makes the loss NaN there instead."""
    if t.device != "cpu":
        return
    if not _host_read(lambda: bool(((t >= 0.0) & (t <= 1.0)).all().item())):
        raise ValueError(f"{op}: all elements of {what} should be between 0 and 1")


# ── operand dtypes ──────────────────────────────────────────────────────────

#: The dtypes an operand of each kind may hold:
#:
#: - ``"floating"``: floating point only;
#: - ``"real"``: floating point or integer, but not bool;
#: - ``"numeric"``: floating point, integer or bool;
#: - ``"index"``: integer class indices (no bool, no floating point).
_KINDS: dict[str, frozenset[str]] = {
    "floating": frozenset({"floating"}),
    "real": frozenset({"floating", "integer"}),
    "numeric": frozenset({"floating", "integer", "bool"}),
    "index": frozenset({"integer"}),
}

_DESCRIBE: dict[str, str] = {
    "floating": "a floating point",
    "real": "a floating point or integer (not bool)",
    "numeric": "a floating point, integer or bool",
    "index": "an integer (class index)",
}

#: Each loss's ``(input kind, target kind)``, the reference's rule,
#: measured against it (``test_loss_operand_dtypes``).  An empty target
#: kind means the loss has no target here, or checks it elsewhere
#: (``ctc_loss``'s labels are checked with its lengths).
#:
#: A loss with a floating point input has no integer form in the
#: reference: it has no integer kernel, or its result is a probability or
#: a logarithm.  The pair and distance losses compute a non-floating
#: input in the default floating dtype instead, as the reference does.
_OPERAND_DTYPES: dict[str, tuple[str, str]] = {
    "binary_cross_entropy": ("floating", "floating"),
    "binary_cross_entropy_with_logits": ("floating", "floating"),
    "kl_div": ("floating", "floating"),
    "cross_entropy": ("floating", "index"),
    "nll_loss": ("floating", "index"),
    "multi_margin_loss": ("floating", "index"),
    "multilabel_margin_loss": ("floating", "index"),
    "ctc_loss": ("floating", ""),
    "hinge_embedding_loss": ("floating", "numeric"),
    "huber_loss": ("floating", "numeric"),
    "soft_margin_loss": ("floating", "numeric"),
    "multilabel_soft_margin_loss": ("floating", "real"),
    "mse_loss": ("numeric", "numeric"),
    "smooth_l1_loss": ("numeric", "numeric"),
    "l1_loss": ("real", "real"),
    "gaussian_nll_loss": ("real", "real"),
    "poisson_nll_loss": ("real", "numeric"),
    "margin_ranking_loss": ("real", "real"),
    "cosine_embedding_loss": ("numeric", "numeric"),
    "triplet_margin_loss": ("real", ""),
    "triplet_margin_with_distance_loss": ("real", ""),
}

#: The regression losses: each takes a non-floating input or target, but
#: not both.  They compute in the dtype the two promote to, and an
#: integer pair has no loss to compute — the reference has no integer
#: kernel for them.
_FLOATING_RESULT: frozenset[str] = frozenset({"mse_loss", "smooth_l1_loss", "l1_loss"})


def _kind_of(t: Tensor) -> str:
    """``"floating"``, ``"integer"``, ``"bool"`` or ``"complex"``."""
    if t.is_floating_point():
        return "floating"
    d = _unwrap(t).dtype
    if d == _C_engine.Dtype.Bool:
        return "bool"
    return "integer" if _promotion.is_integer(d) else "complex"


def _check_kind(t: Tensor, kind: str, what: str, op: str) -> None:
    """Refuse ``t``, the ``what`` operand of loss ``op``, unless its dtype
    is of ``kind`` (see ``_KINDS``)."""
    if _kind_of(t) not in _KINDS[kind]:
        raise TypeError(f"{op}: expected {_DESCRIBE[kind]} {what}, got {t.dtype}")


def _check_gaussian_var(var: Tensor, x: Tensor, op: str) -> None:
    """``var`` of :func:`~lucid.nn.functional.loss.gaussian_nll_loss`:
    floating point, and on ``x``'s device (``target``'s is already
    checked against ``x``'s by the ``_check_dtypes`` call above this
    one)."""
    _check_kind(var, "floating", "var", op)
    _check_device(op, input=x, var=var)


def _check_dtypes(op: str, target: Tensor | None = None, **inputs: Tensor) -> None:
    """Refuse an operand whose dtype loss ``op`` does not take, by its row
    of ``_OPERAND_DTYPES`` — after refusing one on another device
    (:func:`_check_device`): every loss that is not routed through
    :func:`_operands` / :func:`_pair_operands` calls this unconditionally,
    so it is the device check's other reach into this module.

    ``inputs`` are the loss's inputs, by the names the message uses for
    them; ``target`` is its target, when it has one."""
    _check_device(op, target=target, **inputs)
    input_kind, target_kind = _OPERAND_DTYPES[op]
    for what, t in inputs.items():
        _check_kind(t, input_kind, what, op)
    if target is None or not target_kind:
        return
    _check_kind(target, target_kind, "target", op)
    if op in _FLOATING_RESULT and not any(
        t.is_floating_point() for t in (*inputs.values(), target)
    ):
        dtypes = ", ".join(str(t.dtype) for t in (*inputs.values(), target))
        raise TypeError(f"{op}: expected a floating point input or target, got {dtypes}")


def _loss_dtype(*operands: Tensor) -> _C_engine.Dtype:
    """The dtype a loss over ``operands`` computes in: the dtype they
    promote to, or the default floating dtype when that is not floating
    point (an integer or bool pair loss)."""
    impls = [_unwrap(t) for t in operands]
    d = _promotion.result_type(*impls)
    if _promotion.is_integral(d):
        return _promotion.default_float(impls[0].device)
    return d


def _cast(t: Tensor, d: _C_engine.Dtype) -> Tensor:
    """``t`` at the engine dtype ``d``; ``t`` itself when it is already."""
    return t if _unwrap(t).dtype == d else t.to(dtype=d)


def _operands(
    op: str,
    x: Tensor,
    target: Tensor,
    *,
    promote: bool = True,
    **extra: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """``x`` and ``target`` checked for loss ``op`` (:func:`_check_device`,
    :func:`_check_dtypes`) and cast to the dtype the loss computes in.

    That dtype is the one the two promote to.  With ``promote=False`` it is
    ``x``'s alone: a target that is only compared, never multiplied, does
    not widen the loss.  ``extra`` is a loss's other tensor operands —
    ``weight``, ``pos_weight`` — checked against ``x``'s device too, but
    not cast: a weight is only ever multiplied through the ordinary
    operator overloads, which promote and broadcast on their own."""
    _check_device(op, input=x, target=target, **extra)
    _check_dtypes(op, target, input=x)
    d = _loss_dtype(x, target) if promote else _loss_dtype(x)
    return _cast(x, d), _cast(target, d)


def _pair_operands(
    op: str, x1: Tensor, x2: Tensor, y: Tensor, *, promote: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """The two inputs and the label ``y`` of a pair loss, checked for loss
    ``op`` (:func:`_check_device`, :func:`_check_dtypes`) and cast to the
    dtype it computes in: the inputs' dtype, joined with the label's when
    ``promote`` (a label that is only compared does not widen the loss)."""
    _check_device(op, input1=x1, input2=x2, target=y)
    _check_dtypes(op, y, input1=x1, input2=x2)
    d = _loss_dtype(x1, x2, y) if promote else _loss_dtype(x1, x2)
    return _cast(x1, d), _cast(x2, d), _cast(y, d)


def _check_cross_entropy_target(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None,
    ignore_index: int,
    op: str,
) -> tuple[int, bool]:
    """Validate ``x``, ``target``, ``weight`` and ``ignore_index`` for
    :func:`~lucid.nn.functional.loss.cross_entropy`, returning
    ``(num_classes, soft)``."""
    num_classes: int = _num_classes(x, op)
    _check_class_weight(weight, num_classes, x, target, op)
    # A target of the input's own shape holds class probabilities, as in
    # the reference; the docstring said so, and the gather path refused it
    # with a rank mismatch.
    soft: bool = _shape(target) == _shape(x)
    _check_dtypes(op, None if soft else target, input=x)
    if not soft:
        _check_class_target(x, target, op)
    elif not target.is_floating_point():
        raise TypeError(
            f"{op}: a target of the input's shape holds class probabilities "
            f"and must be floating point, got {target.dtype}"
        )
    elif ignore_index >= 0:
        raise ValueError(f"{op}: ignore_index is not supported for a target of class probabilities")
    return num_classes, soft


def _check_cosine_shapes(x1: Tensor, x2: Tensor, y: Tensor, op: str) -> None:
    """Validate rank and broadcast compatibility for :func:`cosine_embedding_loss`."""
    # A (N,) target pairs (N, D) inputs; a 0-d one pairs one unbatched (D,)
    # pair, which raised IndexError ("axis out of range") here.
    if y.ndim > 1:
        raise ValueError(f"{op}: expected a 0-d or 1-D target, got {_shape(y)}")
    pair = _broadcast_shape(_shape(x1), _shape(x2))
    if x1.ndim != y.ndim + 1 or x2.ndim != y.ndim + 1 or pair is None:
        raise ValueError(
            f"{op}: a {y.ndim}-D target expects two {y.ndim + 1}-D inputs of "
            f"broadcastable shapes, got {_shape(x1)} and {_shape(x2)}"
        )
    # The target broadcasts with the pairs' batch, either way: a batch of
    # one against N labels is N losses, as in the reference.
    if _broadcast_shape(_shape(y), pair[:-1]) is None:
        raise ValueError(
            f"{op}: target of shape {_shape(y)} does not broadcast with the "
            f"inputs' batch shape {pair[:-1]}"
        )


def _align_gaussian_var(x: Tensor, var: Tensor | float) -> Tensor:
    """Materialise and broadcast-align ``var`` to ``x`` for :func:`gaussian_nll_loss`."""
    if isinstance(var, (int, float)):
        if var < 0:
            raise ValueError("var has negative entry/entries")
        var = _lucid.full_like(x, float(var))
    if tuple(var.shape) != tuple(x.shape):
        if tuple(x.shape[:-1]) == tuple(var.shape):
            var = var.unsqueeze(-1)
        elif x.ndim != var.ndim or (
            sum(v for d, v in zip(x.shape, var.shape, strict=True) if d != v) != 1
        ):
            raise ValueError("var is of incorrect size")
    if var.device == "cpu" and _host_read(lambda: bool((var < 0.0).any().item())):
        raise ValueError("var has negative entry/entries")
    return var


def _triplet_distances(
    anchor: Tensor, positive: Tensor, negative: Tensor, p: float, eps: float, swap: bool
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """Validate and compute ``(d_ap, d_an)`` for :func:`triplet_margin_loss`."""
    from lucid.nn.functional.activations import pairwise_distance

    op = "triplet_margin_loss"
    _check_dtypes(op, anchor=anchor, positive=positive, negative=negative)
    d = _loss_dtype(anchor, positive, negative)
    anchor, positive, negative = (_cast(t, d) for t in (anchor, positive, negative))
    _check_same_rank(op, anchor=anchor, positive=positive, negative=negative)
    d_ap = _unwrap(pairwise_distance(anchor, positive, p=p, eps=eps))
    d_an = _unwrap(pairwise_distance(anchor, negative, p=p, eps=eps))
    if swap:
        d_pn = _unwrap(pairwise_distance(positive, negative, p=p, eps=eps))
        d_an = _C_engine.minimum(d_an, d_pn)
    return d_ap, d_an


def _poisson_stirling(ti: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Stirling's approximation to ``log(target!)`` for :func:`poisson_nll_loss`."""
    # ``full`` was accepted and did nothing: passing it changed the
    # answer by exactly zero.  The term it names is Stirling's
    # approximation to the ``log(target!)`` that the non-full form
    # drops as constant in the parameters —
    #
    #     target·log(target) − target + ½·log(2π·target)
    #
    # applied only where ``target > 1``, since at 0 and 1 the true
    # ``log(target!)`` is 0 and the approximation is not.
    one = _C_engine.full(ti.shape, 1.0, ti.dtype, ti.device)
    safe = _C_engine.maximum(ti, one)  # keeps log finite at target == 0
    two_pi_safe = _C_engine.mul(_C_engine.full(ti.shape, 2.0 * math.pi, ti.dtype, ti.device), safe)
    stirling = _C_engine.add(
        _C_engine.sub(_C_engine.mul(safe, _C_engine.log(safe)), safe),
        _C_engine.mul(
            _C_engine.full(ti.shape, 0.5, ti.dtype, ti.device),
            _C_engine.log(two_pi_safe),
        ),
    )
    zero = _C_engine.zeros(ti.shape, ti.dtype, ti.device)
    return _C_engine.where(_C_engine.greater(ti, one), stirling, zero)


class _Loss(Module):
    """Base of the loss modules in :mod:`lucid.nn.modules.loss`."""

    reduction: Reduction

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the loss module's ``reduction`` attribute."""
        super().__init__()
        self.reduction = _legacy_reduction(size_average, reduce, reduction, self.__class__.__name__)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"reduction={self.reduction!r}"


def _default_distance(
    distance_function: Callable[[Tensor, Tensor], Tensor] | None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return ``distance_function`` or the default L2 pairwise distance."""
    if distance_function is not None:
        return distance_function
    from lucid.nn.functional.activations import pairwise_distance

    def _default(a: Tensor, b: Tensor) -> Tensor:
        return pairwise_distance(a, b, p=2.0)

    return _default
