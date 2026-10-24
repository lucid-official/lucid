"""
nn.functional loss functions.

Every public loss here checks its arguments at the top, before it computes
anything, and refuses a malformed one with a typed exception — the same
one on the CPU and on Metal.  The checks live in the private validators of
:mod:`lucid.nn.functional._loss_checks` (``_check_*``); they read shapes
and dtypes, so they never wait on the GPU.  Every loss then applies ``reduction`` through :func:`_reduce`.
The loss modules in :mod:`lucid.nn.modules.loss` delegate here, so they
inherit both.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry, _amp_op
from lucid._dtype import bfloat16 as _bfloat16, dtype as _dtype, finfo as _finfo
from lucid._dtype import float16 as _float16, float32 as _float32
from lucid._types import Reduction, ReductionKL
from lucid.nn.functional._loss_checks import (
    _REDUCTIONS,
    _broadcast_operands,
    _broadcast_shape,
    _cast,
    _check_broadcasts_to,
    _check_class_target,
    _check_class_weight,
    _check_dtypes,
    _check_gaussian_var,
    _check_reduction as _check_reduction,
    _check_same_rank,
    _check_target,
    _check_unit_interval,
    _loss_dtype,
    _num_classes,
    _operands,
    _pair_operands,
    _shape,
)
from lucid.nn.functional._index_checks import (
    _ClassTargets,
    _check_ctc,
    _class_index,
    _class_targets,
    _host_read,
    _poison,
)
from lucid.nn.functional._inplace import _keep_in_graph

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

#: The engine's code for each reduction, for the two losses whose fused
#: kernel takes it as an argument (see :func:`_reduce`).
_KERNEL_REDUCTION: dict[str, int] = {"none": 0, "mean": 1, "sum": 2}


# ── reduction ───────────────────────────────────────────────────────────────


def _reduce(
    t: Tensor,
    reduction: str,
    out_dtype: _dtype | None = None,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Apply ``reduction`` to the per-element losses ``t`` — the one place a
    loss reduces.

    ``"none"`` returns ``t``, ``"mean"`` its mean and ``"sum"`` its sum over
    every element; the engine accumulates a half-precision reduction in
    float32 on both devices.  With ``weight`` (the class-index losses'
    per-sample weights) ``"mean"`` divides by the total weight kept rather
    than by the count (:func:`_weighted_mean`), summed in float32 for a half
    ``t``.  ``out_dtype`` is the dtype to round the result to — by default
    ``t``'s own — for a loss that computed a half input in float32.  The
    caller has checked ``reduction`` at its boundary; an unknown one raises
    here too rather than falling through to ``"none"``.

    Autocast is not applied here.  A loss with a row takes it at its entry
    (``_amp_op``) and runs with autocast off; one without, and a weighted
    ``mse_loss`` / ``l1_loss``, is a composite in the reference too, so its
    sums and means apply their own rows.

    Three more reductions are part of a loss's own definition, and are
    built on this one rather than beside it:

    - ``ctc_loss``'s ``"mean"`` divides each sample by its target length
      before taking this mean;
    - ``kl_div``'s ``"batchmean"`` is this sum divided by the batch size;
    - ``mse_loss`` and ``huber_loss`` are each one fused kernel that takes
      the reduction as an argument (``_KERNEL_REDUCTION``), so a training
      step built on them dispatches one op, not two.
    """
    if reduction not in _REDUCTIONS:
        _check_reduction(reduction, "loss")
    if reduction == "mean" and weight is not None:
        # Half is widened, as the count of kept samples is a sum of its own;
        # not under a live autocast, where a weighted mse / l1 decomposes as
        # the reference's does and its sums apply their own rows.
        if not _C_engine.amp_is_active(t._impl.device):
            out_dtype = t.dtype if out_dtype is None else out_dtype
            if (acc := _accumulation_dtype(t)) != t.dtype:
                t, weight = t.to(dtype=acc), weight.to(dtype=acc)
        t = _weighted_mean(t, weight)
    elif reduction != "none":
        t = t.sum() if reduction == "sum" else t.mean()
    return t if out_dtype is None or t.dtype == out_dtype else t.to(dtype=out_dtype)


def _accumulation_dtype(t: Tensor) -> _dtype:
    """The dtype a loss over ``t`` is computed and summed in.

    Half precision is widened to float32.  A float16 sum overflows at
    65504, so a mean over a large batch came out ``inf / inf = nan`` even
    though every term and the answer were small; and constants a loss
    needs (an ``eps`` of 1e-12, a gradient floor of 1e12) are not float16
    numbers at all.  The result is rounded back to the input's dtype.
    """
    return _float32 if t.dtype in (_float16, _bfloat16) else t.dtype


def _check_same_weight(weight: Tensor | None, x: Tensor, op: str) -> None:
    if weight is not None and _shape(weight) != _shape(x):
        raise ValueError(
            f"{op}: weight of shape {_shape(weight)} must have the same size as input {_shape(x)}"
        )


def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    r"""Mean-squared-error (L2) loss between input and target.

    The workhorse loss for regression problems.  Penalises large
    errors *quadratically*, which makes it sensitive to outliers but
    yields a well-conditioned optimisation surface — the gradient is
    linear in the residual, so SGD updates scale gracefully near the
    optimum.  Compare with :func:`l1_loss` (constant gradient,
    robust to outliers) and :func:`huber_loss` (a smooth blend of
    the two).

    Parameters
    ----------
    input : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values that broadcast with ``input``.  A target of another
        shape is broadcast with a ``UserWarning``, as the reference
        does; one that does not broadcast raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    weight : Tensor or None, optional
        Optional element-wise weight tensor of the same shape as ``input``.

    Returns
    -------
    Tensor
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-element
        tensor of ``input`` and ``target``'s broadcast shape for
        ``"none"``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = (x_i - y_i)^2

    With reduction:

    .. math::

        L = \begin{cases}
            \tfrac{1}{N}\sum_i L_i & \text{``mean''} \\
            \sum_i L_i & \text{``sum''} \\
            L_i & \text{``none''}
        \end{cases}

    The gradient w.r.t. ``input`` is :math:`2(x_i - y_i) / N` under
    ``"mean"`` reduction — proportional to the error, so updates
    naturally shrink near the optimum.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import mse_loss
    >>> pred = lucid.tensor([1.0, 2.0, 3.0])
    >>> target = lucid.tensor([1.5, 2.5, 2.5])
    >>> mse_loss(pred, target)
    tensor(0.25)
    """
    op = "mse_loss"
    _check_reduction(reduction, op)
    _check_same_weight(weight, input, op)
    x, target = _operands(op, input, target, weight=weight)
    # The fused kernel takes operands of one shape; a target that
    # broadcasts (as l1_loss's and the reference's do) is expanded first.
    x, target = _broadcast_operands(x, target, _check_target(x, target, op))
    if weight is not None:
        sq = (x - target) ** 2 * weight
        return _reduce(sq, reduction, weight=weight)
    red: int = _KERNEL_REDUCTION[reduction]
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(
    input: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    r"""Mean-absolute-error (L1) loss between input and target.

    Robust regression loss whose gradient is constant in magnitude
    (:math:`\pm 1`) and therefore *insensitive to outliers* —
    extreme residuals do not dominate the update direction as they
    do under :func:`mse_loss`.  The trade-off is a non-smooth
    optimisation surface at :math:`x = y` (sub-gradient at zero),
    which can produce slightly slower convergence for small errors.
    Often paired with :func:`smooth_l1_loss` to recover smoothness
    while keeping outlier robustness.

    Parameters
    ----------
    input : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values that broadcast with ``input``.  A target of another
        shape is broadcast with a ``UserWarning``, as the reference
        does; one that does not broadcast raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    weight : Tensor or None, optional
        Optional element-wise weight tensor of the same shape as ``input``.

    Returns
    -------
    Tensor
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-element
        tensor of ``input`` and ``target``'s broadcast shape for
        ``"none"``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = |x_i - y_i|

    Gradient w.r.t. ``input`` is :math:`\operatorname{sign}(x_i - y_i)`
    (zero at the origin).  Used heavily in image-to-image regression
    (super-resolution, denoising) where outlier-robustness matters.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import l1_loss
    >>> pred = lucid.tensor([1.0, 2.0, 3.0])
    >>> target = lucid.tensor([1.5, 2.5, 2.5])
    >>> l1_loss(pred, target)
    tensor(0.5)
    """
    _check_reduction(reduction, "l1_loss")
    _check_same_weight(weight, input, "l1_loss")
    x, target = _operands("l1_loss", input, target, weight=weight)
    _check_target(x, target, "l1_loss")
    # Weighted, the reference composes it of ops without a row, and so does
    # this; unweighted it is one op, whose row applies at the entry.
    if weight is not None:
        return _reduce((x - target).abs() * weight, reduction, weight=weight)
    with _autocast_entry("l1_loss", _unwrap(x), _unwrap(target)) as (xi, ti):
        return _reduce(_wrap(_C_engine.abs(_C_engine.sub(xi, ti))), reduction)


def _check_beta(beta: float, op: str = "smooth_l1_loss") -> None:
    if isinstance(beta := _promotion.as_number(beta), bool) or not beta >= 0.0:
        raise ValueError(f"{op}: beta must be non-negative, got {beta!r}")


def _check_delta(delta: float, op: str = "huber_loss") -> None:
    if isinstance(delta := _promotion.as_number(delta), bool) or not isinstance(
        delta, (int, float)
    ):
        raise TypeError(
            f"{op}: delta must be a float or int, got {type(delta).__name__}"
        )
    if not delta > 0.0:
        raise ValueError(f"{op}: delta must be positive, got {delta!r}")


def _check_label_smoothing(label_smoothing: float, op: str = "cross_entropy") -> None:
    if (
        isinstance(label_smoothing := _promotion.as_number(label_smoothing), bool)
        or not 0.0 <= label_smoothing <= 1.0
    ):
        raise ValueError(
            f"{op}: label_smoothing must be in [0, 1], got {label_smoothing!r}"
        )


def _check_multi_margin(
    p: int, weight: Tensor | None, op: str = "multi_margin_loss"
) -> None:
    if isinstance(p := _promotion.as_number(p), bool) or (p != 1 and p != 2):
        raise ValueError(f"{op}: only p == 1 and p == 2 supported, got {p!r}")
    if weight is not None and weight.ndim != 1:
        raise ValueError(
            f"{op}: expected weight to be None or 1D tensor, got {weight.ndim}D"
        )


def _check_triplet_margin(margin: float, op: str) -> None:
    if isinstance(margin := _promotion.as_number(margin), bool) or not margin > 0.0:
        raise ValueError(f"{op}: expected margin to be greater than 0, got {margin!r}")


@_amp_op("smooth_l1_loss")
def smooth_l1_loss(
    input: Tensor, target: Tensor, reduction: Reduction = "mean", beta: float = 1.0
) -> Tensor:
    r"""Smooth L1 loss — a quadratic-near-zero, linear-far-from-zero hybrid.

    Combines the best of :func:`mse_loss` and :func:`l1_loss`: the
    quadratic region near the origin gives a smooth gradient and
    fast convergence, while the linear tails make the loss robust
    to outliers.  This is the standard regression head used in
    Fast R-CNN-style object detection (bounding-box regression),
    where outlier bounding boxes would otherwise dominate the
    training signal.

    The function is a thin wrapper around :func:`huber_loss` with
    ``delta = beta`` and an additional ``1/beta`` scaling inside the
    quadratic region (so the loss has unit slope at the transition).

    Parameters
    ----------
    input : Tensor
        Predicted values, any shape.
    target : Tensor
        Target values that broadcast with ``input``.  A target of another
        shape is broadcast with a ``UserWarning``, as the reference
        does; one that does not broadcast raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    beta : float, optional
        Transition point between quadratic and linear regions
        (default ``1.0``).  Smaller ``beta`` makes the loss behave
        more like :func:`l1_loss`; larger ``beta`` makes it behave
        more like :func:`mse_loss`.  A negative (or NaN) ``beta`` raises
        ``ValueError``.

    Returns
    -------
    Tensor
        Scalar (``"mean"``/``"sum"``) or full-shape (``"none"``).

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            \tfrac{1}{2}(x_i - y_i)^2 / \beta & |x_i - y_i| < \beta \\
            |x_i - y_i| - \tfrac{1}{2}\beta & \text{otherwise}
        \end{cases}

    Continuously differentiable everywhere; gradient is
    :math:`(x_i - y_i)/\beta` inside the quadratic region and
    :math:`\operatorname{sign}(x_i - y_i)` outside.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import smooth_l1_loss
    >>> pred = lucid.tensor([0.0, 2.0])
    >>> target = lucid.tensor([0.5, 5.0])
    >>> smooth_l1_loss(pred, target, beta=1.0)
    tensor(1.312)
    """
    # ``huber_loss`` alone is the wrong function: Huber's quadratic region
    # is ``0.5 x²`` where smooth L1's is ``0.5 x² / beta``, and its linear
    # tail is ``beta(|x| - 0.5 beta)`` where smooth L1's is ``|x| - 0.5
    # beta``.  The two differ by a factor of ``beta`` everywhere — which is
    # 1 at the default, so the default was right and every other ``beta``
    # was scaled.  The ``1/beta`` this restores is the one the docstring
    # above has always described.
    op = "smooth_l1_loss"
    _check_reduction(reduction, op)
    # A negative beta reached huber_loss as its delta, whose refusal named
    # the wrong function and the wrong argument.  Written so NaN fails too.
    _check_beta(beta, op)
    x, target = _operands(op, input, target)
    # Broadcast here, once, so the loss below sees operands of one shape
    # (and warns about the caller's shapes, not its own).
    x, target = _broadcast_operands(x, target, _check_target(x, target, op))
    if beta == 0.0:
        # The degenerate limit is plain L1, as the reference also answers.
        return l1_loss(x, target, reduction=reduction)
    loss: Tensor = huber_loss(x, target, reduction=reduction, delta=beta)
    return loss / beta


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
    delta: float = 1.0,
    weight: Tensor | None = None,
) -> Tensor:
    r"""Huber loss — robust regression with a tunable transition point.

    Identical in shape to :func:`smooth_l1_loss` but parameterised
    by the *slope-clip* point :math:`\delta` rather than the
    quadratic-region scale :math:`\beta`.  Inside the
    :math:`|x-y| < \delta` region the loss is quadratic; outside,
    the gradient saturates at :math:`\pm\delta` so a single huge
    residual cannot dominate the update.

    Originally proposed by Peter Huber (1964) as the maximum-
    likelihood estimator for a contaminated Gaussian model — i.e.,
    "mostly Gaussian noise but with occasional outliers".  Use it
    when you suspect a small fraction of your residuals come from
    a heavy-tailed distribution.

    Parameters
    ----------
    input : Tensor
        Predicted values.
    target : Tensor
        Target values that broadcast with ``input``.  A target of another
        shape is broadcast with a ``UserWarning``, as the reference
        does; one that does not broadcast raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    delta : float, optional
        Threshold at which the loss transitions from quadratic to
        linear (default ``1.0``).
    weight : Tensor or None, optional
        Optional element-wise weight tensor of the same shape as ``input``.

    Returns
    -------
    Tensor
        Scalar or full-shape, per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            \tfrac{1}{2}(x_i - y_i)^2 & |x_i - y_i| \le \delta \\
            \delta\,\big(|x_i - y_i| - \tfrac{1}{2}\delta\big) & \text{otherwise}
        \end{cases}

    Unlike :func:`smooth_l1_loss`, the quadratic region is *not*
    rescaled by :math:`1/\delta`, so the loss magnitude itself
    grows with :math:`\delta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import huber_loss
    >>> pred = lucid.tensor([0.0, 5.0])
    >>> target = lucid.tensor([0.5, 0.0])
    >>> huber_loss(pred, target, delta=1.0)
    tensor(2.312)
    """
    op = "huber_loss"
    _check_reduction(reduction, op)
    _check_delta(delta, op)
    _check_same_weight(weight, input, op)
    # The loss keeps the input's dtype, as the reference's does.
    x, target = _operands(op, input, target, promote=False, weight=weight)
    x, target = _broadcast_operands(x, target, _check_target(x, target, op))
    if weight is not None:
        raw = _C_engine.nn.huber_loss(
            _unwrap(x), _unwrap(target), delta, _KERNEL_REDUCTION["none"]
        )
        return _reduce(_wrap(raw) * weight, reduction)
    red: int = _KERNEL_REDUCTION[reduction]
    return _wrap(_C_engine.nn.huber_loss(_unwrap(x), _unwrap(target), delta, red))


def _class_nll(
    log_p: Tensor,
    targets: _ClassTargets,
    weight: Tensor | None,
    reduction: str,
    label_smoothing: float,
) -> Tensor:
    """Negative log-likelihood of integer class targets — the shared body of
    :func:`cross_entropy` and :func:`nll_loss`.

    ``log_p`` holds log-probabilities of shape ``(N, C, *)``.  ``targets``
    are the class indices of shape ``(N, *)``, checked before ``log_p`` was
    computed (:func:`~lucid.nn.functional._index_checks._class_targets`).
    Each sample is scaled by the weight of its class, ``ignore_index``
    samples are dropped, and ``"mean"`` divides by the total weight of the
    samples kept.
    """
    num_classes: int = int(log_p.shape[1])
    safe, counted, usable = targets

    # Gathered with the clamped targets, not masked after a raw gather.
    #
    # ``ignore_index`` defaults to -100, and those sentinels used to be
    # handed straight to ``gather``, which read that far outside the
    # logits and returned whatever was in memory; on the CPU, once
    # ``gather`` learned to bounds-check, the same call raised.  The class
    # weight was gathered with the raw index too, and with a rank-matched
    # ``gather`` that refused a K-d target outright.
    #
    # The value read at a clamped position is discarded by the mask below,
    # so which valid index is used does not matter; that it is valid does.
    nll: Tensor = -_lucid.gather(log_p, 1, safe.unsqueeze(1)).squeeze(1)

    # ``keep`` is 1 for a sample that counts and 0 for an ignored one — and
    # NaN for an out-of-range target on Metal, so a bad label poisons the
    # loss rather than being scored as class 0 or C - 1.
    scale: Tensor | None = None if counted is None else counted.to(dtype=log_p.dtype)
    keep: Tensor | None = _poison(scale, usable, safe, log_p.dtype)

    # Each sample's share of the mean: its class weight times ``keep``.
    # ``None`` when every sample counts once.
    sample_weight: Tensor | None = None
    if weight is not None:
        sample_weight = _lucid.index_select(weight, 0, safe.reshape(-1)).reshape(
            list(safe.shape)
        )
    if keep is not None:
        sample_weight = keep if sample_weight is None else sample_weight * keep
    if sample_weight is not None:
        nll = nll * sample_weight

    if label_smoothing > 0.0:
        # Smoothing term — uniform distribution NLL = -mean over classes
        # of log_softmax, which is -sum/C.  When weight is set, the uniform
        # term is weighted by (mean class weight) following the reference
        # framework's behaviour.
        smooth: Tensor = -log_p.mean(dim=1)  # (N, ...)
        if weight is not None:
            # Weighted-uniform: −Σ_c w_c · log_softmax / C
            w_shape = [1, num_classes] + [1] * (log_p.ndim - 2)
            smooth = -(log_p * weight.reshape(w_shape)).sum(dim=1) / num_classes
        if keep is not None:
            smooth = smooth * keep
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth

    # The mean divides by the total weight of the samples kept.
    return _reduce(nll, reduction, weight=sample_weight)


def _weighted_mean(nll: Tensor, sample_weight: Tensor) -> Tensor:
    """``nll.sum() / sample_weight.sum()`` — :func:`_reduce`'s weighted
    ``"mean"``, the one of :func:`_class_nll`, whose divisor is the total
    weight of the samples kept."""
    total: Tensor = nll.sum()
    denom: Tensor = sample_weight.sum()
    # A batch with nothing kept — all padding, as one micro-batch of an
    # MLM or seq2seq run can be — has the value 0 / 0 = NaN, as in the
    # reference.  Its gradient is 0, also as in the reference: dividing by
    # the zero itself sent inf to every masked position, inf * 0 = NaN
    # reached every parameter, and one such batch under gradient
    # accumulation ruined the whole step.
    #
    # Written without a comparison: a compiled autocast step makes a bare
    # scalar a float16 constant, and ``denom == 0.0`` against the float32
    # denominator aborted the graph compiler.  ``1 - |sign(denom)|`` is 1
    # only for an empty batch; ``(denom - denom) / denom`` is NaN only
    # there, and carries no gradient to the losses.
    safe_denom: Tensor = denom + (1.0 - denom.sign().abs())
    return total / safe_denom + (denom - denom) / denom


@_amp_op("cross_entropy_loss")
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""Cross-entropy loss for multi-class classification.

    The canonical training objective for categorical classifiers.
    Combines :func:`~lucid.nn.functional.log_softmax` and
    :func:`nll_loss` into a single, numerically stable expression —
    operating directly on raw logits avoids the catastrophic
    cancellation that arises when softmax probabilities are taken
    to ``log()``.  Implements the full contract: per-class
    ``weight`` rescaling, ``ignore_index`` masking, and
    label-smoothing regularisation.

    Parameters
    ----------
    input : Tensor
        Raw logits of shape :math:`(N, C)` or :math:`(N, C, d_1, \dots, d_k)`,
        or :math:`(C,)` for one unbatched sample (with a 0-d target, and a
        0-d loss under ``"none"``).
    target : Tensor
        Either integer class indices of shape :math:`(N,)` /
        :math:`(N, d_1, \dots, d_k)` or per-class probabilities of
        shape matching ``input``; any other shape raises ``ValueError``.  An
        index outside :math:`[0, C)` that is not ``ignore_index`` raises
        ``IndexError`` for a CPU tensor; on Metal, where reading the
        target back would stall every step, it makes the loss NaN
        instead.
    weight : Tensor or None, optional
        Per-class weight vector of shape :math:`(C,)` — useful for
        class-imbalanced training.  Any other shape raises
        ``ValueError``.
    ignore_index : int, optional
        Class index whose samples are skipped entirely (default ``-100``).
        Common for masked / padded targets in sequence models.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.  Under
        ``"mean"``, the divisor is the sum of effective sample weights
        (after ``weight`` and ``ignore_index``), not the raw element
        count — for class-index targets.  For class-probability targets
        it is the number of samples, and ``ignore_index`` must be left
        negative.
    label_smoothing : float, optional
        Interpolation factor :math:`\alpha \in [0, 1]` between hard
        one-hot targets and a uniform distribution
        (Szegedy et al. 2016) — at ``1.0`` the target is the uniform
        distribution itself.  Acts as a regulariser by discouraging
        over-confident predictions.  A value outside :math:`[0, 1]`
        raises ``ValueError``.

    Returns
    -------
    Tensor
        Scalar (``"mean"``/``"sum"``) or per-sample tensor (``"none"``).

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = -\sum_c w_c \, y_{i,c} \, \log \mathrm{softmax}(x_i)_c

    where :math:`y_{i,c}` is the (smoothed) target distribution.
    With ``label_smoothing = \alpha``,

    .. math::

        y_{i,c} = (1-\alpha)\,\mathbb{1}[c = t_i] + \alpha / C.

    Gradient w.r.t. the logits has the well-known clean form
    :math:`\mathrm{softmax}(x_i) - y_i` (up to weighting), which is
    the reason cross-entropy is preferred over MSE for classification:
    no sigmoid saturation, no vanishing gradient.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cross_entropy
    >>> logits = lucid.tensor([[2.0, 0.5, 0.1], [0.0, 1.5, 0.2]])
    >>> target = lucid.tensor([0, 1])
    >>> cross_entropy(logits, target)
    tensor(0.3597)
    """
    from lucid.nn.functional.activations import log_softmax as _log_softmax

    op = "cross_entropy"
    _check_reduction(reduction, op)
    # 1.0 is a uniform target, which the reference takes; it was refused.
    _check_label_smoothing(label_smoothing, op)
    x = input
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
        raise ValueError(
            f"{op}: ignore_index is not supported for a target of class probabilities"
        )

    # An unbatched (C,) input is one sample; it raised in log_softmax.
    unbatched: bool = x.ndim == 1
    if unbatched:
        x = x.unsqueeze(0)
        target = target.unsqueeze(0) if soft else target.reshape([1])
    out: Tensor
    if soft:
        # Class dim is 1 for both (N, C) and (N, C, *) inputs.
        out = _soft_target_nll(
            _log_softmax(x, dim=1), target, weight, reduction, label_smoothing
        )
    else:
        # The targets are checked before the first kernel runs.
        targets = _class_targets(target, num_classes, op, ignore_index=ignore_index)
        out = _class_nll(
            _log_softmax(x, dim=1), targets, weight, reduction, label_smoothing
        )
    return out.reshape([]) if unbatched and reduction == "none" else out


def _soft_target_nll(
    log_p: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: str,
    label_smoothing: float,
) -> Tensor:
    """Cross-entropy against per-class probabilities ``target`` of the
    shape of ``log_p``: ``-sum_c w_c y_c log p_c`` per sample, and a
    ``"mean"`` over the samples (not over their weights).  The target's
    dtype and ``ignore_index`` were checked by :func:`cross_entropy`."""
    num_classes: int = int(log_p.shape[1])
    if label_smoothing > 0.0:
        target = target * (1.0 - label_smoothing) + label_smoothing / num_classes
    prod: Tensor = log_p * target
    if weight is not None:
        prod = prod * weight.reshape([1, num_classes] + [1] * (log_p.ndim - 2))
    return _reduce(-prod.sum(dim=1), reduction)


@_amp_op("nll_loss")
def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Negative log-likelihood loss for multi-class classification.

    The "back-half" of :func:`cross_entropy`: assumes the input is
    already a tensor of log-probabilities (typically produced by
    :func:`~lucid.nn.functional.log_softmax`).  Provided as a
    separate entry-point so models that need log-probabilities for
    downstream use (e.g., beam search) can avoid recomputing them.

    Parameters
    ----------
    input : Tensor
        Log-probabilities of shape :math:`(N, C)` or
        :math:`(N, C, d_1, \dots, d_k)`, or :math:`(C,)` for one unbatched
        sample (with a 0-d target, and a 0-d loss under ``"none"``).
    target : Tensor
        Integer class indices of shape :math:`(N,)` /
        :math:`(N, d_1, \dots, d_k)`; any other shape raises
        ``ValueError``.  An index outside :math:`[0, C)` that is not
        ``ignore_index`` raises ``IndexError`` for a CPU tensor and makes
        the loss NaN on Metal.
    weight : Tensor or None, optional
        Per-class weight vector :math:`(C,)`.  Any other shape raises
        ``ValueError``.
    ignore_index : int, optional
        Class index whose samples are excluded (default ``-100``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor depending on ``reduction``.

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = -w_{t_i}\,x_{i, t_i}

    Under ``"mean"`` reduction, the divisor is the sum of effective
    sample weights — i.e., :math:`\sum_i w_{t_i}\,\mathbb{1}[t_i \ne \text{ignore}]`
    — not the raw count.  This matches the standard convention so
    that the loss is invariant to a global rescaling of ``weight``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import nll_loss, log_softmax
    >>> logits = lucid.tensor([[2.0, 0.5, 0.1], [0.0, 1.5, 0.2]])
    >>> target = lucid.tensor([0, 1])
    >>> nll_loss(log_softmax(logits, dim=1), target)
    tensor(0.3597)
    """
    op = "nll_loss"
    _check_reduction(reduction, op)
    x = input
    _check_dtypes(op, target, input=x)
    num_classes: int = _num_classes(x, op)
    _check_class_weight(weight, num_classes, x, target, op)
    _check_class_target(x, target, op)
    # An unbatched (C,) input is one sample.
    unbatched: bool = x.ndim == 1
    if unbatched:
        x = x.unsqueeze(0)
        target = target.reshape([1])
    targets = _class_targets(target, num_classes, op, ignore_index=ignore_index)
    out: Tensor = _class_nll(x, targets, weight, reduction, 0.0)
    return _keep_in_graph(
        out.reshape([]) if unbatched and reduction == "none" else out, input
    )


@_amp_op("bce_loss")
def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Binary cross-entropy between predicted probabilities and targets.

    The standard objective for binary classification and for
    multi-label classification with independent class predictions.
    Operates on *probabilities* in :math:`(0, 1)` — typically the
    output of a :func:`~lucid.nn.functional.sigmoid`.  When working
    with raw logits, prefer :func:`binary_cross_entropy_with_logits`
    for numerical stability.

    Each logarithm is clamped below at :math:`-100`, so a probability
    of exactly ``0`` or ``1`` costs at most ``100`` instead of an
    infinite or undefined loss, and the input gradient's denominator
    :math:`p(1-p)` is floored at :math:`\varepsilon = 10^{-12}`, so the
    gradient stays finite at the boundaries too.  For a float16 input the
    floor is :math:`1/65504`, the smallest that keeps a float16 gradient
    finite.  Under autocast the loss is computed in float32.

    Parameters
    ----------
    input : Tensor
        Predicted probabilities in :math:`[0, 1]`, any shape.  For a CPU
        tensor an entry outside :math:`[0, 1]` (or NaN) raises
        ``ValueError``, as the reference's CPU kernel does; a Metal
        tensor is not read back, which would stall every step, and such
        an entry makes the loss NaN instead.
    target : Tensor
        Target probabilities (typically binary) in :math:`[0, 1]`, of
        the same shape — another shape raises ``ValueError`` rather than
        being broadcast.  Its range is checked as ``input``'s is.
    weight : Tensor or None, optional
        Element-wise rescaling factor that broadcasts to ``input``'s shape;
        one that does not, or that would enlarge it, raises
        ``ValueError``.  Use to up-weight rare classes or hard examples.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = -\big(y_i\,\max(\log p_i, -100)
              + (1 - y_i)\,\max(\log(1 - p_i), -100)\big)

    Gradient w.r.t. ``input`` is
    :math:`(p_i - y_i) / \max(p_i(1 - p_i), \varepsilon)`, which grows
    without bound as :math:`p_i \to 0` or :math:`1` until the floor
    caps it — the reason the logits-form (which yields a clean
    :math:`\sigma(x) - y` gradient) is preferred when training
    stability is critical.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import binary_cross_entropy
    >>> p = lucid.tensor([0.9, 0.2, 0.7])
    >>> y = lucid.tensor([1.0, 0.0, 1.0])
    >>> binary_cross_entropy(p, y)
    tensor(0.2284)
    """
    op = "binary_cross_entropy"
    _check_reduction(reduction, op)
    # In the input's dtype: the gradient floor below is chosen for it.
    x, target = _operands(op, input, target, promote=False, weight=weight)
    _check_target(x, target, op)
    _check_broadcasts_to(weight, _shape(x), "weight", op)
    # Last: the only check that reads values, and only on the CPU.
    _check_unit_interval(x, "input", op)
    _check_unit_interval(target, "target", op)
    # The probability used to be clamped to [1e-12, 1 - 1e-12], and in
    # float32 ``1 - 1e-12`` rounds to 1.0: a sigmoid of a logit above ~17
    # is exactly 1, ``log(1 - 1)`` is -inf, and ``0 * -inf`` made the loss
    # NaN.  The clamp also zeroed the gradient at both ends.
    #
    # The definition now is the reference's: each log term clamped at
    # -100 in the value, and an input gradient of
    # ``(p - y) / max(p (1 - p), eps)``.  Where ``p (1 - p) >= eps`` the
    # plain expression is differentiated as written (which also keeps the
    # second derivative); elsewhere — within ~1e-12 of 0 or 1, where the
    # logs are clamped or infinite — the value is computed off the graph
    # and the floored gradient is attached as a straight-through term.
    #
    # Half precision is computed in float32 and rounded back: ``eps`` is 0
    # in float16, and the floored gradient (1e12) is not a float16 number.
    # ``bce_loss`` is ``KEEP`` on Metal, so a float16 input there is rounded
    # back to float16 under autocast as well.
    out_dtype = x.dtype
    acc = _accumulation_dtype(x)
    p, y = x.to(dtype=acc), target.to(dtype=acc)
    # The floor is 1e-12, or 1 / 65504 for a float16 input: a float16
    # gradient of 1e12 is inf and sends ``inf * 0 = NaN`` to saturated logits.
    eps: float = max(1e-12, 1.0 / float(_finfo(out_dtype).max))
    p_d: Tensor = p.detach()
    inside: Tensor = p_d * (1.0 - p_d) >= eps
    p_in: Tensor = _lucid.where(inside, p, _lucid.full_like(p_d, 0.5))
    interior: Tensor = -(y * p_in.log() + (1.0 - y) * (1.0 - p_in).log())
    log_p: Tensor = p_d.log().clamp(min=-100.0)
    log_q: Tensor = (1.0 - p_d).log().clamp(min=-100.0)
    grad_edge: Tensor = (p_d - y.detach()) / eps
    edge: Tensor = -(y * log_p + (1.0 - y) * log_q) + (p - p_d) * grad_edge
    bce: Tensor = _lucid.where(inside, interior, edge)
    if weight is not None:
        bce = bce * weight.to(dtype=acc)
    return _reduce(bce, reduction, out_dtype)


@_amp_op("bce_with_logits")
def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
    pos_weight: Tensor | None = None,
) -> Tensor:
    r"""Binary cross-entropy from raw logits (numerically stable).

    Mathematically equivalent to
    ``binary_cross_entropy(sigmoid(input), target)`` but evaluated in a
    log-sum-exp-style form that avoids overflow/underflow when
    ``|input|`` is large.  This is the preferred binary classification
    loss for training — composing a separate sigmoid with BCE risks
    catastrophic cancellation in :math:`\log(1 - \sigma(x))` for
    large positive logits.

    Parameters
    ----------
    input : Tensor
        Raw logits (un-bounded reals), any shape.
    target : Tensor
        Target probabilities (typically binary), same shape as ``input`` —
        another shape raises ``ValueError`` rather than being broadcast.
    weight : Tensor or None, optional
        Element-wise rescaling factor that broadcasts to ``input``'s shape;
        one that does not, or that would enlarge it, raises
        ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    pos_weight : Tensor or None, optional
        Per-class weight applied to the *positive* term only —
        useful for highly-imbalanced binary tasks, where setting
        ``pos_weight = n_neg / n_pos`` recovers the prevalence-
        balanced gradient.  It broadcasts to ``input``'s shape, as
        ``weight`` does.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    The numerically stable form is

    .. math::

        L_i = (1 - y_i)\,x_i + \operatorname{softplus}(-x_i),

    equivalent to :math:`-(y\log\sigma(x) + (1-y)\log(1-\sigma(x)))`
    but free of overflow, and smooth at :math:`x = 0` (the
    :math:`\max(x, 0) + \log(1 + e^{-|x|})` spelling has the same value
    but a kink in each piece there).  With ``pos_weight``:

    .. math::

        L_i = (1 - y_i)\,x_i
              + \big(1 + (w^{+} - 1) y_i\big)\operatorname{softplus}(-x_i).

    Gradient w.r.t. ``input`` is the clean :math:`\sigma(x_i) - y_i`
    (modulo weighting) — the canonical reason this form is used in
    practice instead of the explicit sigmoid + BCE composition.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import binary_cross_entropy_with_logits
    >>> logits = lucid.tensor([2.0, -1.0, 0.5])
    >>> target = lucid.tensor([1.0, 0.0, 1.0])
    >>> binary_cross_entropy_with_logits(logits, target)
    tensor(0.3048)
    """
    op = "binary_cross_entropy_with_logits"
    _check_reduction(reduction, op)
    x, target = _operands(op, input, target, weight=weight, pos_weight=pos_weight)
    _check_target(x, target, op)
    _check_broadcasts_to(weight, _shape(x), "weight", op)
    _check_broadcasts_to(pos_weight, _shape(x), "pos_weight", op)
    # ``max(x, 0) - x y + log(1 + exp(-|x|))`` has the right value and the
    # wrong derivative at x = 0: the subgradients there are clamp' = 1 and
    # sign(0) = 0, so d/dx came out 1 - y instead of 1/2 - y, and a
    # zero-initialised head got no gradient at all from its positive
    # labels.  ``(1 - y) x + softplus(-x)`` is the same function written
    # with one smooth piece, and softplus is stable for any |x|.
    out_dtype = x.dtype
    acc = _accumulation_dtype(x)
    xa: Tensor = x.to(dtype=acc)
    y: Tensor = target.to(dtype=acc)
    softplus_neg: Tensor = _wrap(_C_engine.softplus(_unwrap(-xa)))
    if pos_weight is None:
        loss: Tensor = (1.0 - y) * xa + softplus_neg
    else:
        loss = (1.0 - y) * xa + (1.0 + (pos_weight - 1.0) * y) * softplus_neg
    if weight is not None:
        loss = loss * weight
    return _reduce(loss, reduction, out_dtype)


@_amp_op("kl_div")
def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduction: ReductionKL = "mean",
    log_target: bool = False,
) -> Tensor:
    r"""Kullback-Leibler divergence between two distributions.

    Measures the "information gain" from approximating distribution
    :math:`p` (the target) with distribution :math:`q` (the input).
    Used heavily in knowledge distillation (matching student logits
    to a teacher's), variational inference (the ELBO's KL term),
    and policy-gradient regularisation in RL.

    The convention here matches the reference framework: ``input`` is
    :math:`\log q` (log-predicted), and ``target`` is :math:`p`
    (target probability) — or :math:`\log p` if ``log_target=True``.
    Note this is asymmetric in its arguments — KL is *not* a metric.

    Parameters
    ----------
    input : Tensor
        Log-probabilities of the *predicted* distribution
        :math:`\log q`, any shape.
    target : Tensor
        Probabilities of the *target* distribution :math:`p`,
        or its log when ``log_target=True``.  Same shape as ``input``, or
        one that broadcasts with it (another raises ``ValueError``).
    size_average : bool or None, optional
        Deprecated; overrides ``reduction`` to ``"mean"`` or ``"sum"``.
    reduction : str, optional
        ``"none"``, ``"mean"``, ``"sum"``, or ``"batchmean"`` (default
        ``"mean"``). ``"batchmean"`` divides the summed loss by the
        leading (batch) dimension (or the sum for a 0-d input) and is
        the only reduction that yields the mathematically correct KL
        value in expectation.
    log_target : bool, optional
        When ``True``, treat ``target`` as already-logged (:math:`\log p`).

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = p_i \cdot (\log p_i - \log q_i)

    with :math:`0 \log 0 = 0`, so a target with zero entries (one-hot,
    sparse) contributes nothing there rather than ``nan``.  Globally:

    .. math::

        D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \log \tfrac{p_i}{q_i} \ge 0,

    with equality iff :math:`p = q` almost everywhere.  The standard
    ``"mean"`` reduction divides by the element count, not the
    batch size, so it under-reports the divergence value — prefer
    ``"batchmean"`` whenever the absolute scale matters.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import kl_div, log_softmax
    >>> log_q = log_softmax(lucid.tensor([[2.0, 0.5, 0.1]]), dim=1)
    >>> p = lucid.tensor([[0.8, 0.15, 0.05]])
    >>> kl_div(log_q, p, reduction="batchmean")
    tensor(0.02391)
    """
    if size_average is not None:
        reduction = "mean" if size_average else "sum"
    _check_reduction(reduction, "kl_div", batchmean=True)
    x, target = _operands("kl_div", input, target)
    _check_target(x, target, "kl_div")
    # `x` is log_q (log of predicted probability) per the standard contract.
    # When log_target=False, target is the raw probability p; when True it
    # is log(p).  Loss elementwise = target * (log(target) - log_q).
    xi, ti = _unwrap(x), _unwrap(target)
    if log_target:
        # log_target=True → target itself is log(p); use exp(t) as the weight.
        kl = _C_engine.mul(_C_engine.exp(ti), _C_engine.sub(ti, xi))
    else:
        # ``target * (log(target) - x)`` is ``0 * -inf = nan`` wherever the
        # target is 0 — every off-class entry of a one-hot or sparse
        # distillation target.  ``xlogy`` takes ``0 log 0`` as 0, the limit
        # the divergence is defined with.
        xlogy = _unwrap(_lucid.xlogy(target, target))
        kl = _C_engine.sub(xlogy, _C_engine.mul(ti, xi))
    if reduction != "batchmean":
        return _reduce(_wrap(kl), reduction)
    # The sum over the leading (batch) dimension's size.  A 0-d input has
    # no batch dimension, and its batchmean is the sum, as in the
    # reference; reading ``x.shape[0]`` there raised IndexError.
    total: Tensor = _reduce(_wrap(kl), "sum")
    return total / int(xi.shape[0]) if xi.shape else total


@_amp_op("triplet_margin_loss")
def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Triplet margin loss for metric learning.

    Pulls an "anchor" embedding toward a "positive" example (same
    class / matching pair) and pushes it away from a "negative"
    example by at least a fixed ``margin``.  This is the workhorse
    objective for learning embeddings used in face verification
    (FaceNet), image retrieval, and contrastive representation
    learning.

    Optionally applies the "anchor-swap" trick of Balntas et al.
    2016 — when ``d(p, n) < d(a, n)`` the positive is closer to the
    negative than the anchor is, so we swap roles and use the
    *harder* of the two distances, focusing gradient on the more
    informative triplet.

    Parameters
    ----------
    anchor : Tensor
        Embedding of shape :math:`(N, D)`.
    positive : Tensor
        Positive sample embedding of the same shape.
    negative : Tensor
        Negative sample embedding of the same shape.  The three
        embeddings have one rank and broadcastable shapes; otherwise
        ``ValueError`` is raised.
    margin : float, optional
        Minimum desired gap between positive and negative distances
        (default ``1.0``).  Triplets satisfying the margin already
        receive zero loss / zero gradient.
    p : float, optional
        Norm degree of the pairwise distance (default ``2.0`` —
        Euclidean).
    eps : float, optional
        Numerical floor inside the distance to avoid zero-derivative
        at coincident points (default ``1e-6``).
    swap : bool, optional
        Enable the Balntas-anchor-swap trick (default ``False``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-triplet tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-triplet loss:

    .. math::

        L_i = \max\!\big(0,\; d(a_i, p_i) - d(a_i, n_i) + \text{margin}\big),

    where :math:`d(\cdot, \cdot)` is the :math:`L_p` pairwise
    distance.  Easy triplets (already separated by ``margin``)
    contribute exactly zero — only "semi-hard" / "hard" triplets
    drive the update, which is why batch mining strategies matter
    so much in practice.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import triplet_margin_loss
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> p = lucid.tensor([[1.0, 0.1]])
    >>> n = lucid.tensor([[0.0, 1.0]])
    >>> triplet_margin_loss(a, p, n, margin=1.0)
    tensor(0.)
    """
    from lucid.nn.functional.activations import pairwise_distance

    op = "triplet_margin_loss"
    _check_reduction(reduction, op)
    _check_dtypes(op, anchor=anchor, positive=positive, negative=negative)
    d = _loss_dtype(anchor, positive, negative)
    anchor, positive, negative = (_cast(t, d) for t in (anchor, positive, negative))
    _check_same_rank(op, anchor=anchor, positive=positive, negative=negative)
    d_ap = _unwrap(pairwise_distance(anchor, positive, p=p, eps=eps))
    d_an = _unwrap(pairwise_distance(anchor, negative, p=p, eps=eps))
    if swap:
        d_pn = _unwrap(pairwise_distance(positive, negative, p=p, eps=eps))
        d_an = _C_engine.minimum(d_an, d_pn)
    margin_t = _C_engine.full(d_ap.shape, margin, d_ap.dtype, d_ap.device)
    loss = _C_engine.clip(
        _C_engine.add(_C_engine.sub(d_ap, d_an), margin_t), 0.0, math.inf
    )
    return _reduce(_wrap(loss), reduction)


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: object | None = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Triplet margin loss with a user-supplied distance function.

    Identical in form to :func:`triplet_margin_loss` but lets the
    caller plug in any binary distance callable — useful when the
    embedding space is non-Euclidean (e.g., learned Mahalanobis
    distances, cosine distance, or hyperbolic embeddings).  When
    no ``distance_function`` is supplied it defaults to the
    :math:`L_2` pairwise distance, matching the reference framework
    semantics.

    Parameters
    ----------
    anchor : Tensor
        Anchor embedding of shape :math:`(N, D)`.
    positive : Tensor
        Positive sample embedding of the same shape.
    negative : Tensor
        Negative sample embedding of the same shape.  The three
        embeddings have one rank, or ``ValueError`` is raised.
    distance_function : callable or None, optional
        Function ``(x, y) -> Tensor`` returning a non-negative
        distance of shape :math:`(N,)`.  Defaults to :math:`L_2`
        pairwise distance.
    margin : float, optional
        Minimum desired margin between positive and negative
        distances (default ``1.0``).
    swap : bool, optional
        Enable the Balntas-2016 anchor-swap heuristic: replace
        :math:`d(a, n)` with :math:`\min\!\big(d(a, n), d(p, n)\big)`
        so the harder negative drives the gradient (default ``False``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-triplet tensor.

    Notes
    -----
    Per-triplet loss:

    .. math::

        L_i = \max\!\big(0,\; d(a_i, p_i) - d(a_i, n_i) + \text{margin}\big)

    The Lucid module wrapper
    :class:`lucid.nn.TripletMarginWithDistanceLoss` forwards into
    this function; both surfaces are valid entry-points.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import (
    ...     triplet_margin_with_distance_loss,
    ...     pairwise_distance,
    ... )
    >>> def manhattan(a, b):
    ...     return pairwise_distance(a, b, p=1.0)
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> p = lucid.tensor([[1.0, 0.1]])
    >>> n = lucid.tensor([[0.0, 1.0]])
    >>> triplet_margin_with_distance_loss(a, p, n, distance_function=manhattan)
    tensor(0.)
    """
    from lucid.nn.functional.activations import pairwise_distance

    op = "triplet_margin_with_distance_loss"
    _check_reduction(reduction, op)
    _check_same_rank(
        op, broadcast=False, anchor=anchor, positive=positive, negative=negative
    )
    df: object = distance_function
    if df is None:
        # A distance function of the caller's own decides which dtypes it
        # takes; the default one is the reference's, with its rule.
        _check_dtypes(op, anchor=anchor, positive=positive, negative=negative)

        def df(a: Tensor, b: Tensor) -> Tensor:
            """Derivative helper used inside the loss-function gradient computation."""
            return pairwise_distance(a, b, p=2.0)

    d_ap: Tensor = df(anchor, positive)
    d_an: Tensor = df(anchor, negative)
    if swap:
        d_pn: Tensor = df(positive, negative)
        d_an = d_an.minimum(d_pn)

    return _reduce((d_ap - d_an + margin).clamp(min=0.0), reduction)


@_amp_op("cosine_embedding_loss")
def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Cosine embedding loss for pairwise similarity learning.

    Encourages "similar" pairs (label :math:`y = 1`) to align in
    direction and "dissimilar" pairs (:math:`y = -1`) to be at
    least ``margin`` cosine-units apart.  Operates purely on
    angular (direction) information — magnitudes are normalised
    away, which is useful when the relevant signal is the
    *direction* of embeddings (e.g., word vectors, learned
    representations).

    Parameters
    ----------
    input1 : Tensor
        First embedding of shape :math:`(N, D)`, or :math:`(D,)` for one
        unbatched pair (with a 0-d ``target``, and a 0-d loss under
        ``"none"``).
    input2 : Tensor
        Second embedding of the same shape.
    target : Tensor
        Label tensor of shape :math:`(N,)` with values :math:`\pm 1`, or
        0-d for one unbatched pair.  A ``target`` of any other rank, inputs of
        a rank that does not match it, or shapes that do not line up
        raise ``ValueError``.
    margin : float, optional
        Minimum desired cosine gap for dissimilar pairs, typically
        in :math:`[-1, 1]` (default ``0.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-pair tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-pair loss:

    .. math::

        L_i = \begin{cases}
            1 - \cos(x_1^{(i)}, x_2^{(i)}) & y_i = +1 \\
            \max\!\big(0,\; \cos(x_1^{(i)}, x_2^{(i)}) - \text{margin}\big) & y_i = -1
        \end{cases}

    With ``margin = 0``, dissimilar pairs are only penalised when
    they have positive cosine similarity — i.e., the loss is
    satisfied as long as the angle between them exceeds :math:`90°`.
    Increasing ``margin`` toward 1 demands more separation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cosine_embedding_loss
    >>> x1 = lucid.tensor([[1.0, 0.0]])
    >>> x2 = lucid.tensor([[0.5, 0.5]])
    >>> y = lucid.tensor([1.0])
    >>> cosine_embedding_loss(x1, x2, y)
    tensor(0.2929)
    """
    from lucid.nn.functional.activations import cosine_similarity

    op = "cosine_embedding_loss"
    _check_reduction(reduction, op)
    x1, x2, y = _pair_operands(op, input1, input2, target, promote=False)
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
    cos: Tensor = cosine_similarity(x1, x2, dim=-1)
    zero: Tensor = _lucid.zeros_like(cos)
    loss: Tensor = _lucid.where(
        y == 1, 1.0 - cos, _lucid.where(y == -1, (cos - margin).clamp(min=0.0), zero)
    )
    return _reduce(loss, reduction)


@_amp_op("margin_ranking_loss")
def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Pairwise ranking hinge loss.

    Trains a scoring function so that for each pair :math:`(x_1, x_2)`
    the *signed* score gap :math:`y\,(x_1 - x_2)` exceeds the
    ``margin``.  Used for learning-to-rank (search, recommendation),
    Bradley-Terry style preference modelling, and reward-model
    training for RLHF — wherever the supervision signal is a
    pairwise preference rather than a target value.

    Parameters
    ----------
    input1 : Tensor
        Scores for the first item of each pair, any shape.
    input2 : Tensor
        Scores for the second item, same shape as ``input1``.
    target : Tensor
        Pairwise preference label :math:`\pm 1`: :math:`+1` if
        :math:`input1` should rank higher, :math:`-1` otherwise.  The three
        tensors have one rank and broadcastable shapes; otherwise
        ``ValueError`` is raised.
    margin : float, optional
        Required minimum score gap (default ``0.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-pair tensor.

    Notes
    -----
    Per-pair loss:

    .. math::

        L_i = \max\!\big(0,\; -y_i\,(x_1^{(i)} - x_2^{(i)}) + \text{margin}\big)

    Pairs already satisfying the margin (:math:`y(x_1 - x_2) \ge \text{margin}`)
    contribute zero loss and zero gradient — the hinge structure
    naturally focuses learning on the violating pairs, akin to a
    pairwise SVM.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import margin_ranking_loss
    >>> s1 = lucid.tensor([2.0, 0.5])
    >>> s2 = lucid.tensor([1.0, 1.0])
    >>> y = lucid.tensor([1.0, 1.0])
    >>> margin_ranking_loss(s1, s2, y, margin=1.0)
    tensor(0.75)
    """
    op = "margin_ranking_loss"
    _check_reduction(reduction, op)
    x1, x2, y = _pair_operands(op, input1, input2, target, promote=True)
    # A (1, N) label against (N,) scores broadcast to an (N, N) loss.
    _check_same_rank(op, x1=x1, x2=x2, y=y)
    diff = _C_engine.sub(_unwrap(x1), _unwrap(x2))
    margin_t = _C_engine.full(diff.shape, margin, diff.dtype, diff.device)
    neg_y_diff = _C_engine.mul(_C_engine.neg(_unwrap(y)), diff)
    loss = _C_engine.relu(_C_engine.add(neg_y_diff, margin_t))
    return _reduce(_wrap(loss), reduction)


@_amp_op("hinge_embedding_loss")
def hinge_embedding_loss(
    input: Tensor, target: Tensor, margin: float = 1.0, reduction: Reduction = "mean"
) -> Tensor:
    r"""Hinge embedding loss.

    Designed for similarity learning on pre-computed distances:
    given a (typically non-negative) score ``input`` representing a
    pairwise distance, push positive pairs (label :math:`+1`)
    toward small distances and negative pairs (label :math:`-1`)
    above a fixed ``margin``.  Common in Siamese network training
    and energy-based dissimilarity models.

    Parameters
    ----------
    input : Tensor
        Per-pair score (distance) tensor, any shape.
    target : Tensor
        Label tensor :math:`\pm 1` with the same shape as ``input`` (one
        that does not broadcast with it raises ``ValueError``).
    margin : float, optional
        Margin enforced for negative pairs (default ``1.0``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \begin{cases}
            x_i & y_i = +1 \\
            \max(0,\; \text{margin} - x_i) & y_i = -1
        \end{cases}

    The positive branch simply minimises the distance; the negative
    branch is a one-sided hinge that pushes apart only those pairs
    whose distance is below the margin — pairs already far apart
    contribute nothing.  This asymmetric structure prevents the
    loss from collapsing all embeddings into a single point.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hinge_embedding_loss
    >>> dist = lucid.tensor([0.2, 0.8])
    >>> y = lucid.tensor([1.0, -1.0])
    >>> hinge_embedding_loss(dist, y, margin=1.0)
    tensor(0.2)
    """
    _check_reduction(reduction, "hinge_embedding_loss")
    x, y = _operands("hinge_embedding_loss", input, target, promote=False)
    _check_target(x, y, "hinge_embedding_loss")
    zero: Tensor = _lucid.zeros_like(x)
    loss: Tensor = _lucid.where(y != -1, x, zero) + _lucid.where(
        y != 1, (margin - x).clamp(min=0.0), zero
    )
    return _reduce(loss, reduction)


@_amp_op("poisson_nll_loss")
def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Poisson negative log-likelihood loss for count regression.

    The maximum-likelihood objective when targets are non-negative
    integer counts modelled as :math:`y \sim \mathrm{Poisson}(\lambda)`.
    Standard for forecasting tasks (web clicks, event counts,
    biological cell counts) where the variance scales with the
    mean.  Unlike :func:`mse_loss`, this loss respects the
    heteroscedasticity inherent in count data.

    Parameters
    ----------
    input : Tensor
        Predicted Poisson rate.  By default (``log_input=True``)
        treated as :math:`\log \lambda` for numerical stability;
        set ``log_input=False`` to pass the rate :math:`\lambda`
        directly.
    target : Tensor
        Observed counts that broadcast with ``input``; a shape that does not
        raises ``ValueError``.
    log_input : bool, optional
        Whether ``input`` is :math:`\log \lambda` (default) or
        :math:`\lambda`.  The log-form avoids exponentiating an
        unbounded prediction in inner loops.
    full : bool, optional
        Include the Stirling approximation term
        :math:`\log(y!) \approx y\log y - y + \tfrac{1}{2}\log(2\pi y)`
        in the loss, where ``target > 1`` (at 0 and 1 the true
        :math:`\log(y!)` is 0).  Has no effect on gradients (constant in
        ``input``) but yields the correct log-likelihood value.
    eps : float, optional
        Small constant added before :math:`\log` when
        ``log_input=False`` (default ``1e-8``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss (constant-in-:math:`x` terms dropped):

    .. math::

        L_i = \begin{cases}
            e^{x_i} - y_i\,x_i & \text{log\_input = True} \\
            x_i - y_i \log(x_i + \varepsilon) & \text{log\_input = False}
        \end{cases}

    Gradient w.r.t. :math:`x` is :math:`e^x - y` (log-input form)
    or :math:`1 - y/(x + \varepsilon)` (rate form).  Both push
    :math:`\lambda` toward :math:`y` in expectation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import poisson_nll_loss
    >>> log_lam = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([1.0, 2.0, 5.0])
    >>> poisson_nll_loss(log_lam, y, log_input=True)
    tensor(-0.2976)
    """
    _check_reduction(reduction, "poisson_nll_loss")
    x, target = _operands("poisson_nll_loss", input, target)
    _check_target(x, target, "poisson_nll_loss")
    xi, ti = _unwrap(x), _unwrap(target)
    if log_input:
        # Formula — exp(x) - target * x
        loss = _C_engine.sub(_C_engine.exp(xi), _C_engine.mul(ti, xi))
    else:
        # Formula — x - target * log(x + eps)
        log_xeps = _C_engine.log(
            _C_engine.add(xi, _C_engine.full(xi.shape, eps, xi.dtype, xi.device))
        )
        loss = _C_engine.sub(xi, _C_engine.mul(ti, log_xeps))

    if full:
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
        two_pi_safe = _C_engine.mul(
            _C_engine.full(ti.shape, 2.0 * math.pi, ti.dtype, ti.device), safe
        )
        stirling = _C_engine.add(
            _C_engine.sub(_C_engine.mul(safe, _C_engine.log(safe)), safe),
            _C_engine.mul(
                _C_engine.full(ti.shape, 0.5, ti.dtype, ti.device),
                _C_engine.log(two_pi_safe),
            ),
        )
        zero = _C_engine.zeros(ti.shape, ti.dtype, ti.device)
        loss = _C_engine.add(
            loss, _C_engine.where(_C_engine.greater(ti, one), stirling, zero)
        )

    return _reduce(_wrap(loss), reduction)


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


@_amp_op("gaussian_nll_loss")
def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor | float,
    full: bool = False,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Gaussian negative log-likelihood for heteroscedastic regression.

    Maximum-likelihood objective when the prediction is a
    *distribution* :math:`\mathcal{N}(\mu, \sigma^2)` over the
    target rather than a point estimate.  Training a network with
    two heads (one for :math:`\mu`, one for :math:`\sigma^2`)
    against this loss recovers calibrated predictive uncertainty
    — useful for active learning, decision-aware regression, and
    Bayesian deep ensembles.

    The variance ``var`` is clamped below by ``eps`` to prevent
    division by zero and runaway log-terms when the network
    initially predicts near-zero variance.  The clamp acts on the value
    only: the gradient reaches ``var`` as if it had not been clamped, so
    a variance head stuck below ``eps`` is still pushed back up.

    Parameters
    ----------
    input : Tensor
        Predicted means :math:`\mu`, any shape.
    target : Tensor
        Observed values :math:`y` that broadcast with ``input``; a shape that
        does not raises ``ValueError``.
    var : Tensor or float
        Predicted variances :math:`\sigma^2 \ge 0`.  Either the shape of
        ``input``; or that shape without its last dimension — one variance
        per sample, e.g. ``(N,)`` for an ``(N, D)`` input — which is
        unsqueezed to broadcast over that dimension; or the shape of
        ``input`` with exactly one dimension of size 1.  A float is one
        variance for every element.  Any other shape raises
        ``ValueError``.  A negative variance raises ``ValueError`` for a
        CPU tensor; on Metal, where reading it back would stall the
        step, it makes the loss NaN there instead.
    full : bool, optional
        Include the constant :math:`\tfrac{1}{2}\log(2\pi)` term in
        the loss value.  Has no effect on gradients; it makes the value
        an actual negative log-likelihood rather than one shifted by a
        constant.
    eps : float, optional
        Lower bound applied to ``var`` for numerical stability
        (default ``1e-6``).
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss (constant terms dropped):

    .. math::

        L_i = \tfrac{1}{2}\!\left(\log \sigma_i^2 + \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right)

    The first term penalises over-confidence (small variance), the
    second term rewards accuracy weighted by precision.  Together
    they give the model a clean trade-off: when it cannot reduce
    :math:`(y-\mu)^2`, increasing :math:`\sigma^2` decreases the
    loss — this is what produces calibrated uncertainty estimates.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gaussian_nll_loss
    >>> mu = lucid.tensor([0.0, 1.0])
    >>> y = lucid.tensor([0.5, 1.0])
    >>> var = lucid.tensor([1.0, 0.25])
    >>> gaussian_nll_loss(mu, y, var)
    tensor(-0.2841)
    """
    _check_reduction(reduction, "gaussian_nll_loss")
    _check_target(x := input, target, "gaussian_nll_loss")
    _check_dtypes("gaussian_nll_loss", target, input=x)
    if not isinstance(var := _promotion.as_number(var), (int, float)):
        _check_gaussian_var(var, x, "gaussian_nll_loss")
        d = _loss_dtype(x, target, var)
        var = _cast(var, d)
    else:
        d = _loss_dtype(x, target)
    x, target = _cast(x, d), _cast(target, d)
    var_t = _align_gaussian_var(x, var)
    # Clamped at ``eps`` in value, not in gradient: the reference clamps a
    # copy under ``no_grad``, so the gradient flows to ``var`` unchanged.
    var_d: Tensor = var_t.detach()
    var_c: Tensor = var_t + (var_d.clamp(min=eps) - var_d)
    loss: Tensor = 0.5 * (var_c.log() + (x - target) ** 2 / var_c)
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    if var_t.device != "cpu":
        # Negative variance makes the loss NaN on Metal off the graph.
        loss = loss + var_d.clamp(max=0.0).sqrt() * 0.0
    return _reduce(loss, reduction)


@_amp_op("ctc_loss")
def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | Sequence[int],
    target_lengths: Tensor | Sequence[int],
    blank: int = 0,
    reduction: Reduction = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""Connectionist Temporal Classification (CTC) loss.

    The standard training objective for unaligned sequence
    prediction — used in speech recognition, handwriting
    recognition, and any task where the input sequence is much
    longer than the target and no per-frame alignment is provided.
    Introduced by Graves et al. 2006.

    Internally marginalises over every valid alignment of a
    :math:`T`-frame prediction onto an :math:`S`-symbol target by
    inserting "blank" symbols and allowing each target symbol to
    span one or more frames, computing the negative log of the
    total path probability via dynamic programming.

    Parameters
    ----------
    log_probs : Tensor
        Log-probabilities of shape :math:`(T, N, C)` where
        :math:`T` is the input sequence length, :math:`N` is the
        batch size, and :math:`C` is the number of classes
        (including the blank).  Typically produced by
        :func:`~lucid.nn.functional.log_softmax` over the class
        axis.  :math:`(T, C)` is one unbatched sequence, with a
        :math:`(S,)` target, 0-d lengths, and a 0-d loss under ``"none"``.
    targets : Tensor
        Target indices, shape :math:`(N, S)` (padded) or
        :math:`(\sum_i \text{target\_lengths}_i,)` (concatenated).
        ``int32``.  Every label read must lie in :math:`[0, C)`, else
        ``IndexError``.  This is stricter than the reference, which reads
        past the class axis.  The padding past a row's target length is
        not read and may hold anything.
    input_lengths : Tensor
        Effective input lengths :math:`(N,)`, ``int32``.  Enables
        padding-aware batching.  Each must lie in :math:`[1, T]`, else
        ``IndexError``.  A length of 0 raises ``NotImplementedError`` for
        now.
    target_lengths : Tensor
        Effective target lengths :math:`(N,)`, ``int32``.  Each must lie in
        :math:`[0, S]` for padded targets, and they must add up to the
        number of concatenated targets.  A length outside its range
        raises ``IndexError``; a count that does not match raises
        ``ValueError``.
    blank : int, optional
        Index of the blank symbol (default ``0``), in :math:`[0, C)`,
        else ``IndexError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.  Under
        ``"mean"``, each sample's loss is divided by its target
        length (at least 1) and the results are averaged across the
        batch, as the reference framework defines it.
    zero_infinity : bool, optional
        When ``True``, infinite losses (which arise when a target
        cannot fit in the available input frames) and their
        gradients are set to zero, effectively skipping those
        samples (default ``False``).

    Returns
    -------
    Tensor
        Scalar (``"mean"`` / ``"sum"``) or per-sample tensor of
        shape :math:`(N,)`.

    Notes
    -----
    The CTC objective is the negative log of the total alignment
    probability:

    .. math::

        L = -\log \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})}
            \prod_{t=1}^{T} p_t(\pi_t),

    where :math:`\mathcal{B}` is the "many-to-one" alignment map
    that collapses repeats and removes blanks.  The forward and
    backward recursions run in the log domain on the CPU; on metal
    the inputs make a round trip.  The gradient with respect to
    ``log_probs`` follows the reference framework's formula, which is
    the true gradient with respect to the logits once a
    :func:`~lucid.nn.functional.log_softmax` sits in front — the way
    the loss is meant to be fed.  It is differentiable once.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import ctc_loss, log_softmax
    >>> # T=4 frames, N=1 batch, C=3 classes (blank=0)
    >>> logits = lucid.randn(4, 1, 3)
    >>> log_p = log_softmax(logits, dim=2)
    >>> targets = lucid.tensor([[1, 2]], dtype=lucid.int32)
    >>> il = lucid.tensor([4], dtype=lucid.int32)
    >>> tl = lucid.tensor([2], dtype=lucid.int32)
    >>> ctc_loss(log_p, targets, il, tl)  # doctest: +SKIP
    Tensor(...)
    """
    op = "ctc_loss"
    _check_reduction(reduction, op)
    _check_dtypes(op, log_probs=log_probs)

    def _as_lengths(v: Tensor | Sequence[int]) -> Tensor:
        return (
            v
            if isinstance(v, _lucid.Tensor)
            else _lucid.tensor(list(v), dtype=_lucid.int64)
        )

    input_lengths = _as_lengths(input_lengths)
    target_lengths = _as_lengths(target_lengths)

    # An unbatched (T, C) input is one sequence, with a (S,) target and
    # 0-d (or one-element) lengths; it raised "must be 3-D".
    unbatched: bool = log_probs.ndim == 2
    if unbatched:
        log_probs = log_probs.unsqueeze(1)
        targets = targets.reshape([1, -1])
        input_lengths = input_lengths.reshape([1])
        target_lengths = target_lengths.reshape([1])

    # Every length, label and ``blank`` is checked before the kernel: an
    # ``input_lengths`` past the input, a ``target_lengths`` past its row
    # and a label or ``blank`` outside the classes were all read as given,
    # past the end of their axis.
    rows: list[int] = _check_ctc(
        log_probs, targets, input_lengths, target_lengths, blank, op
    )

    tgt_impl = _unwrap(targets)
    if len(list(tgt_impl.shape)) > 1:
        # Padded (N, S): row b's first target_lengths[b] entries are its
        # target.  The kernel reads one concatenated list, so the rows are
        # gathered into one — flattening the padding in with them made every
        # sample after a short one read the padding as its target.
        width = int(tgt_impl.shape[1])
        keep = [b * width + i for b, n in enumerate(rows) for i in range(n)]
        flat = _wrap(_C_engine.reshape(tgt_impl, [-1]))
        tgt_impl = _unwrap(flat[_lucid.tensor(keep, dtype=_lucid.int64)])

    # Ensure integer dtype for lengths and targets.
    def _to_i32(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
        return (
            impl
            if getattr(impl, "dtype", None) == _C_engine.I32
            else _C_engine.astype(impl, _C_engine.I32)
        )

    tgt_impl = _to_i32(tgt_impl)
    il_impl = _to_i32(_unwrap(input_lengths))
    tl_impl = _to_i32(_unwrap(target_lengths))
    # Targets and lengths usually live on the CPU whatever the device of
    # log_probs; the kernel reads all four from one device, and a metal
    # log_probs beside CPU integers failed with a bad_variant_access.
    device = log_probs.device
    tgt_impl, il_impl, tl_impl = (
        _unwrap(_wrap(t).to(device)) for t in (tgt_impl, il_impl, tl_impl)
    )

    per_sample: Tensor = _wrap(
        _C_engine.nn.ctc_loss(
            _unwrap(log_probs), tgt_impl, il_impl, tl_impl, blank, zero_infinity
        )
    )
    if reduction == "mean":
        # Each sample over its own target length first — ctc's definition
        # of the mean (see ``_reduce``).
        lengths = target_lengths.to(per_sample.dtype).to(per_sample.device)
        per_sample = per_sample / lengths.clamp(min=1.0)
    elif unbatched:
        per_sample = per_sample.reshape([])
    return _reduce(per_sample, reduction)


@_amp_op("multi_margin_loss")
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Multi-class hinge (margin) loss — Crammer-Singer SVM objective.

    A non-probabilistic alternative to :func:`cross_entropy` for
    multi-class classification: instead of fitting a softmax
    distribution, it requires the true-class score to exceed every
    other class score by at least ``margin``.  Frequently used in
    structured prediction and as a drop-in for hinge-style losses
    in metric learning.

    Parameters
    ----------
    input : Tensor
        Class scores of shape :math:`(N, C)`, or :math:`(C,)` for one
        unbatched sample (the ``"none"`` loss then takes the target's
        shape, ``()`` or ``(1,)``).  Any other rank raises
        ``ValueError``.
    target : Tensor
        Integer class indices of shape :math:`(N,)`; any other shape
        raises ``ValueError``.  An index outside :math:`[0, C)` raises
        ``IndexError`` for a CPU tensor and makes the loss NaN on Metal.
    p : int, optional
        Power applied to each hinge term — ``1`` for the standard
        hinge loss, ``2`` for the smoother squared-hinge variant
        (default ``1``).  Any other value raises ``ValueError``.
    margin : float, optional
        Required minimum score gap between the true class and
        every competitor (default ``1.0``).
    weight : Tensor or None, optional
        Per-class weight vector of shape :math:`(C,)`.  Each sample
        contribution is scaled by the weight of its *true* class.  Any
        other shape raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-sample loss:

    .. math::

        L_i = \frac{1}{C} \sum_{j \ne t_i}
              \max\!\big(0,\; \text{margin} - x_{i, t_i} + x_{i, j}\big)^p

    Samples whose true-class score already dominates all
    competitors by ``margin`` produce zero loss and zero gradient
    — like the binary SVM hinge, only the *support vectors*
    contribute to the update.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multi_margin_loss
    >>> scores = lucid.tensor([[2.0, 0.5, 0.1]])
    >>> target = lucid.tensor([0], dtype=lucid.int32)
    >>> multi_margin_loss(scores, target)
    tensor(0.)
    """
    op = "multi_margin_loss"
    _check_reduction(reduction, op)
    x = input
    _check_dtypes(op, target, input=x)
    # Any other power was computed as written; the reference defines these
    # two.
    _check_multi_margin(p, weight, op)
    num_classes: int = _num_classes(x, op, max_ndim=2)
    _check_class_weight(weight, num_classes, x, target, op)
    _check_class_target(x, target, op)
    # An unbatched (C,) input is one sample; its loss takes the target's
    # shape, () or (1,), as in the reference.  It raised in the gather.
    unbatched: bool = x.ndim == 1
    target_shape: list[int] = list(target.shape)
    if unbatched:
        x = x.unsqueeze(0)
    # The targets are checked before the first gather.
    safe, _, usable = _class_targets(target.reshape(-1), num_classes, op)
    poison: Tensor | None = _poison(None, usable, safe, x.dtype)
    tgt_col: Tensor = safe.reshape([-1, 1])  # (N, 1)

    # margin - x[i, y_i] + x[i, j] for every j, hinged and raised to p.
    correct: Tensor = _lucid.gather(x, 1, tgt_col)  # (N, 1)
    hinge: Tensor = (margin - correct + x).relu()
    if p == 2:
        hinge = hinge**2.0

    if weight is not None:
        # The weight of each sample's true class.  This was a gather of a
        # (1, C) weight with an (N, 1) index, which the CPU refused for any
        # N > 1 ("index is larger than the operand on a non-gathered axis").
        hinge = hinge * _lucid.index_select(weight, 0, safe).reshape([-1, 1])

    # The true class is not one of its own competitors.
    classes: Tensor = _lucid.arange(num_classes, device=x.device).reshape([1, -1])
    is_target: Tensor = classes == tgt_col.to(dtype=classes.dtype)
    hinge = _lucid.where(is_target, _lucid.zeros_like(hinge), hinge)

    loss_n: Tensor = hinge.sum(dim=1) / num_classes  # (N,)
    if poison is not None:
        loss_n = loss_n * poison
    if unbatched:
        loss_n = loss_n.reshape(target_shape)
    return _reduce(loss_n, reduction)


@_amp_op("multilabel_margin_loss")
def multilabel_margin_loss(
    input: Tensor, target: Tensor, reduction: Reduction = "mean"
) -> Tensor:
    r"""Multi-label hinge loss for set-valued targets.

    The multi-label counterpart of :func:`multi_margin_loss`: each
    sample can belong to *several* classes (the "positives"), and
    the objective requires every positive class score to exceed
    every non-positive (negative) class score by at least 1.  Used
    for set prediction tasks such as image tagging where labels
    are not mutually exclusive.

    Targets are encoded as a fixed-width index list with ``-1`` as
    a padding sentinel — entries up to the first ``-1`` mark the
    positive classes for that sample; that entry and every one after it
    are ignored, whatever they hold.

    Parameters
    ----------
    input : Tensor
        Class scores of shape :math:`(N, C)` or :math:`(C,)`; any other
        rank raises ``ValueError``.
    target : Tensor
        Same shape as ``input`` (any other raises ``ValueError``), any
        integer dtype.  The entries before the first negative one are
        the positive class indices.  A listed class outside
        :math:`[0, C)` raises ``IndexError`` for a CPU tensor and makes
        that sample's loss NaN on Metal.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar, or per-sample tensor of shape :math:`(N,)` (a 0-d tensor
        for a 1-D ``input``) under ``"none"``.

    Notes
    -----
    Per-sample loss, summed over positive labels :math:`t` and
    non-positive labels :math:`j`:

    .. math::

        L_i = \frac{1}{C} \sum_{t \in P_i} \sum_{j \notin P_i}
              \max\!\big(0,\; 1 - x_{i, t} + x_{i, j}\big),

    where :math:`P_i` is the list of positive labels for sample
    :math:`i` (a class listed twice is counted twice).  Equivalently,
    it is the average of multi-class hinge losses obtained by treating
    each positive label as *the* correct one against the full set of
    non-positives.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multilabel_margin_loss
    >>> scores = lucid.tensor([[1.0, 0.5, -0.3, 0.2]])
    >>> target = lucid.tensor([[0, 1, -1, -1]])
    >>> multilabel_margin_loss(scores, target)
    tensor(0.275)
    """
    op = "multilabel_margin_loss"
    _check_reduction(reduction, op)
    x = input
    _check_dtypes(op, target, input=x)
    num_classes: int = _num_classes(x, op, max_ndim=2)
    # A target of another shape was read column by column against x's
    # classes, and scored whatever lined up.
    _check_target(x, target, op)
    unbatched: bool = x.ndim == 1
    # Any integer dtype is taken: a ``lucid.tensor([...ints...])`` target
    # is int64, and an explicitly int32 one is just as usual.
    tgt: Tensor = target.reshape([1, -1]) if unbatched else target

    # The labels of a sample are its entries up to the first negative one.
    # Every column used to be read with ``index >= 0``, so a label after
    # the first -1 still counted as a positive.  Read at the width the
    # check below reads, so the two agree on which entries are labels.
    index: Tensor = _class_index(tgt)
    listed: Tensor = _lucid.cumprod((index >= 0).to(dtype=_lucid.int32), dim=1) == 1
    # A listed class outside [0, C) raises for a CPU target, before the
    # scatter, and makes the sample's loss NaN on Metal (its count is NaN),
    # as for cross_entropy.  What follows the pad is never read.
    safe, _, usable = _class_targets(index, num_classes, op, counted=listed)
    xb: Tensor = x.reshape([1, -1]) if unbatched else x
    src: Tensor = _poison(listed.to(dtype=xb.dtype), usable, safe, xb.dtype)

    # How many times each class is listed.  A class listed twice counts
    # twice as a positive, as in the reference; it is a target class (not
    # one of the negatives) once listed at all.
    counts: Tensor = _lucid.scatter_add(
        _lucid.zeros_like(xb), 1, safe.to(dtype=_lucid.int64), src
    )
    negative: Tensor = 1.0 - counts.clamp(max=1.0)

    # hinge[i, t, j] = max(0, 1 - x[i, t] + x[i, j]), for each positive t
    # (weighted by its count) against each negative j.
    hinge: Tensor = (1.0 - xb.unsqueeze(2) + xb.unsqueeze(1)).relu()
    pairs: Tensor = hinge * counts.unsqueeze(2) * negative.unsqueeze(1)
    loss_n: Tensor = pairs.sum(dim=[1, 2]) / num_classes  # (N,)
    if unbatched:
        loss_n = loss_n.reshape([])
    return _reduce(loss_n, reduction)


# ── P3 fills: soft_margin_loss / multilabel_soft_margin_loss ───────────────


@_amp_op("soft_margin_loss")
def soft_margin_loss(
    input: Tensor, target: Tensor, reduction: Reduction = "mean"
) -> Tensor:
    r"""Logistic (softplus) loss for binary classification with ±1 labels.

    A "soft" variant of the binary hinge loss: instead of the
    piecewise-linear :math:`\max(0, 1 - y\,x)`, it uses the smooth
    surrogate :math:`\log(1 + e^{-y\,x})` — which is the
    negative-log-likelihood of a logistic model with labels in
    :math:`\{-1, +1\}`.  Equivalent to
    :func:`binary_cross_entropy_with_logits` with the ``{0, 1}``
    labels re-coded as :math:`\{-1, +1\}`.

    The implementation evaluates :math:`\mathrm{softplus}(-y\,x)`,
    which is numerically stable for large ``|x|`` (no overflow,
    no log of near-zero values).

    Parameters
    ----------
    input : Tensor
        Raw scores (logits), any shape.
    target : Tensor
        Target tensor of the same shape, or one that broadcasts to it,
        conventionally holding :math:`\pm 1` (any real values are
        accepted).  A target that does not broadcast to ``input``'s shape,
        or would enlarge it, raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or full-shape per ``reduction``.

    Notes
    -----
    Per-element loss:

    .. math::

        L_i = \log\!\big(1 + \exp(-y_i\,x_i)\big).

    Unlike the (non-smooth) hinge, every sample contributes a
    non-zero gradient — even correctly classified ones — but the
    contribution decays exponentially as :math:`y\,x` grows.  This
    softness improves optimisation behaviour with first-order
    methods at the cost of a slightly less sparse solution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import soft_margin_loss
    >>> x = lucid.tensor([2.0, -1.0])
    >>> y = lucid.tensor([1.0, -1.0])
    >>> soft_margin_loss(x, y)
    tensor(0.2201)
    """
    _check_reduction(reduction, "soft_margin_loss")
    input, target = _operands("soft_margin_loss", input, target, promote=False)
    _check_target(input, target, "soft_margin_loss")
    return _reduce(_lucid.nn.functional.softplus(-target * input), reduction)


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Per-class logistic loss averaged over labels (multi-label BCE).

    The standard objective for multi-label classification with
    *independent* per-class probabilities: each class gets its own
    binary logistic regression head, and the total loss is the
    mean of the per-class binary cross-entropies.  Mathematically
    equivalent to applying
    :func:`binary_cross_entropy_with_logits` per class and
    averaging across the class axis.

    Computed via the numerically stable identity
    :math:`\log \sigma(x) = -\mathrm{softplus}(-x)`, which avoids
    overflow / underflow for large ``|x|``.

    Parameters
    ----------
    input : Tensor
        Raw logits of shape :math:`(N, C)`.
    target : Tensor
        Target probabilities (typically binary) of shape :math:`(N, C)`,
        or one that broadcasts to it.
    weight : Tensor or None, optional
        Per-class weight broadcast against the per-class loss
        tensor before averaging.  A ``target`` or ``weight`` that does
        not broadcast to the input's shape, or that would enlarge it,
        raises ``ValueError``.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    Tensor
        Scalar or per-sample tensor of shape :math:`(N,)`.

    Notes
    -----
    Per-sample loss, averaged across the :math:`C` classes:

    .. math::

        L_i = -\frac{1}{C} \sum_c \Big[
            t_{i,c}\,\log \sigma(x_{i,c})
            + (1 - t_{i,c})\,\log(1 - \sigma(x_{i,c}))
        \Big]

    Because the per-class predictions are independent (no softmax
    coupling), the gradient through each class is exactly that of
    a single binary logistic regression — convenient for highly
    multi-label problems where the active label set is sparse.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import multilabel_soft_margin_loss
    >>> logits = lucid.tensor([[2.0, -1.0, 0.5]])
    >>> target = lucid.tensor([[1.0, 0.0, 1.0]])
    >>> multilabel_soft_margin_loss(logits, target)
    tensor(0.3048)
    """
    op = "multilabel_soft_margin_loss"
    _check_reduction(reduction, op)
    input, target = _operands(op, input, target, weight=weight)
    _check_target(input, target, op)
    _check_broadcasts_to(weight, _shape(input), "weight", op)
    # logσ(x)   = -softplus(-x);  log(1-σ(x)) = -softplus(x).  Both forms
    # are numerically stable for large |x|.
    log_sig = -_lucid.nn.functional.softplus(-input)
    log_one_minus_sig = -_lucid.nn.functional.softplus(input)
    per_class = -(target * log_sig + (1.0 - target) * log_one_minus_sig)
    if weight is not None:
        per_class = per_class * weight
    return _reduce(_lucid.mean(per_class, dim=-1, keepdim=False), reduction)
