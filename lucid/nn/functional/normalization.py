"""
nn.functional normalization operations.
"""

import math
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry, _autocast_suspended
from lucid._ops._dims import _to_axes
from lucid._ops._reduce import _require_normable
from lucid._types import DimLike, DTypeLike
from lucid.linalg import vector_norm

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _check_batch_norm_sample_count(x: Tensor, training: bool) -> None:
    """Reject training batches with no variance estimate per channel."""
    if training and x.ndim in (2, 3, 4, 5) and x.shape[1] > 0 and x.numel() == x.shape[1]:
        raise ValueError(
            "batch_norm: expected more than one value per channel when training, "
            f"got input size {tuple(x.shape)}"
        )


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Batch normalization (Ioffe & Szegedy, 2015).

    Normalises each channel using statistics computed across the batch
    and all spatial axes, then applies a learnable per-channel affine
    transform.  Acts as a strong regulariser and an enabler of higher
    learning rates by reducing "internal covariate shift".

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *)`` with 2–5 dimensions:
        ``(N, C)``, ``(N, C, L)``, ``(N, C, H, W)``, or
        ``(N, C, D, H, W)``.
    running_mean : Tensor or None
        Running mean buffer of shape ``(C,)``; consulted in eval mode.
    running_var : Tensor or None
        Running variance buffer of shape ``(C,)``; consulted in eval mode.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.  Defaults to
        ones (no scaling).
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.  Defaults to
        zeros (no shift).
    training : bool, optional
        When ``True``, statistics come from the current batch; when
        ``False`` and running buffers are supplied, those are used.
    momentum : float, optional
        Exponential-moving-average coefficient for the running buffers.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    Math (over batch :math:`\mathcal{B}` and spatial dims):

    .. math::

        \mu_{\mathcal{B}} &= \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} x_i \\
        \sigma_{\mathcal{B}}^2 &= \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (x_i - \mu_{\mathcal{B}})^2 \\
        \hat{x}_i &= \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
        y_i &= \gamma\,\hat{x}_i + \beta

    Dispatches internally by ``ndim`` to the matching engine kernel:
    ``ndim==2`` is treated as ``(N, C, 1)``; ``ndim==3`` uses
    ``batch_norm1d``; ``ndim==4`` uses the 2-D op; ``ndim==5`` uses
    ``batch_norm3d``.  Eval mode bypasses statistics computation
    entirely and uses ``running_mean`` / ``running_var`` directly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import batch_norm
    >>> x = lucid.randn(8, 16, 32, 32)
    >>> rm = lucid.zeros(16); rv = lucid.ones(16)
    >>> y = batch_norm(x, rm, rv, training=True)
    >>> y.shape
    (8, 16, 32, 32)
    """
    from lucid._factories.creation import ones, zeros

    _check_batch_norm_sample_count(input, training)

    C = input.shape[1]
    ndim = input.ndim
    xi = _unwrap(input)
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=input.device, dtype=input.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=input.device, dtype=input.dtype))
    )

    # ── Eval mode: use precomputed running statistics ─────────────────────────
    if not training and running_mean is not None and running_var is not None:
        rm = _unwrap(running_mean)
        rv = _unwrap(running_var)
        out_impl = _C_engine.nn.batch_norm_eval(xi, rm, rv, w, b, eps)
        return _wrap(out_impl)

    # ── Training mode: dispatch by dimensionality ─────────────────────────────
    # When ``running_mean`` / ``running_var`` are supplied the C++ kernel
    # additionally fuses the EMA update — reusing the BN forward's
    # saved_mean / saved_rstd instead of doing a second mean+var reduction
    # over ``x``.  Pass None when the caller is not tracking running stats.
    rm_impl = _unwrap(running_mean) if running_mean is not None else None
    rv_impl = _unwrap(running_var) if running_var is not None else None

    if ndim == 2:
        # (N, C) → unsqueeze to (N, C, 1), batch_norm1d, squeeze back
        xi_3d = _C_engine.unsqueeze(xi, 2)
        out_3d = _C_engine.nn.batch_norm1d(xi_3d, w, b, eps, rm_impl, rv_impl, momentum)
        return _wrap(_C_engine.squeeze(out_3d, 2))
    elif ndim == 3:
        return _wrap(_C_engine.nn.batch_norm1d(xi, w, b, eps, rm_impl, rv_impl, momentum))
    elif ndim == 4:
        return _wrap(_C_engine.nn.batch_norm(xi, w, b, eps, rm_impl, rv_impl, momentum))
    elif ndim == 5:
        return _wrap(_C_engine.nn.batch_norm3d(xi, w, b, eps, rm_impl, rv_impl, momentum))
    else:
        raise ValueError(f"batch_norm: expected 2–5D input, got ndim={ndim}")


def layer_norm(
    input: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Layer normalization (Ba, Kiros & Hinton, 2016).

    Normalises each sample independently across the **last**
    ``normalized_shape`` dimensions.  Unlike :func:`batch_norm`, no
    batch statistics are involved — making LayerNorm the default
    choice for transformers and other models where batches may be
    small or sequence lengths variable.

    Parameters
    ----------
    input : Tensor
        Input whose trailing dims match ``normalized_shape``.
    normalized_shape : list of int or tuple of int
        Trailing dims to normalise over.  E.g. ``(d,)`` for a
        token-wise normalisation in a transformer with hidden size
        ``d``.
    weight : Tensor, optional
        Per-element scale :math:`\gamma` of shape ``normalized_shape``.
        Defaults to ones.
    bias : Tensor, optional
        Per-element shift :math:`\beta` of shape ``normalized_shape``.
        Defaults to zeros.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    Math (reduction taken over the last :math:`k` axes,
    :math:`k = \mathrm{len}(\text{normalized\_shape})`):

    .. math::

        \mu &= \frac{1}{|S|}\sum_{j \in S} x_j \\
        \sigma^2 &= \frac{1}{|S|}\sum_{j \in S} (x_j - \mu)^2 \\
        y &= \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

    Because the reduction is per-sample, behaviour is identical at
    train and eval time — no running statistics needed.  This is what
    makes LayerNorm so prevalent in sequence models (RNNs,
    transformers).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import layer_norm
    >>> x = lucid.randn(2, 10, 64)
    >>> y = layer_norm(x, normalized_shape=(64,))
    >>> y.shape
    (2, 10, 64)
    """
    from lucid._factories.creation import ones, zeros

    shape: tuple[int, ...] = tuple(normalized_shape)
    w: _C_engine.TensorImpl = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(*shape, device=input.device, dtype=input.dtype))
    )
    b: _C_engine.TensorImpl = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(*shape, device=input.device, dtype=input.dtype))
    )
    # Engine API: layer_norm(x, gamma, beta, eps) — no normalized_shape arg
    return _wrap(_C_engine.nn.layer_norm(_unwrap(input), w, b, eps))


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Group normalization (Wu & He, 2018).

    Splits the channel dimension into ``num_groups`` contiguous groups
    and normalises each ``(sample, group)`` slice independently across
    its channels and spatial axes.  Combines the spatial reduction of
    BatchNorm with the per-sample stability of LayerNorm — performance
    is therefore largely independent of batch size, which makes it the
    go-to choice for detection / segmentation models trained with very
    small batches.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *spatial)`` where ``C`` must be
        divisible by ``num_groups``.
    num_groups : int
        Number of channel groups.  Two limiting cases: ``num_groups ==
        C`` reduces to InstanceNorm; ``num_groups == 1`` reduces to
        LayerNorm over channels + spatial axes.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    Math (let :math:`G_g` be the channel set of group :math:`g` and
    :math:`S` the spatial axes):

    .. math::

        \mu_{n,g} &= \frac{1}{|G_g||S|} \sum_{c \in G_g} \sum_{s \in S} x_{n,c,s} \\
        \sigma^2_{n,g} &= \frac{1}{|G_g||S|} \sum_{c \in G_g} \sum_{s \in S} (x_{n,c,s} - \mu_{n,g})^2 \\
        y_{n,c,s} &= \gamma_c \cdot \frac{x_{n,c,s} - \mu_{n,g(c)}}{\sqrt{\sigma^2_{n,g(c)} + \epsilon}} + \beta_c

    Independence from batch size avoids the train/eval mismatch that
    BatchNorm requires running buffers to fix.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import group_norm
    >>> x = lucid.randn(2, 32, 16, 16)
    >>> y = group_norm(x, num_groups=8)
    >>> y.shape
    (2, 32, 16, 16)
    """
    from lucid._factories.creation import ones, zeros

    C = input.shape[1]
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=input.device, dtype=input.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=input.device, dtype=input.dtype))
    )
    return _wrap(_C_engine.nn.group_norm(_unwrap(input), w, b, num_groups, eps))


def rms_norm(
    input: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    r"""Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    A simplified variant of :func:`layer_norm` that drops mean
    centering and the additive bias.  Cheaper to compute and shown to
    perform competitively in large language models (LLaMA, T5,
    PaLM, ...), where its lower latency and memory footprint matter at
    scale.

    Parameters
    ----------
    input : Tensor
        Input whose last dimension matches ``normalized_shape``.
    normalized_shape : list of int or tuple of int
        Trailing dims to normalise over.  Typically ``(d,)`` where
        ``d`` is the model hidden size.
    weight : Tensor, optional
        Per-element scale :math:`\gamma`.  Defaults to ones.
    eps : float, optional
        Numerical safety added inside the square root.  ``None`` (the
        default) uses the machine epsilon of ``input``'s dtype, as the
        reference framework does — a fixed ``1e-8`` rounds to zero in
        float16, so a half-precision export divided by an exact zero on
        an all-zero row.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    Math:

    .. math::

        \text{RMS}(x) &= \sqrt{\frac{1}{|S|}\sum_{j \in S} x_j^2 + \epsilon} \\
        y &= \gamma \cdot \frac{x}{\text{RMS}(x)}

    The dropped mean-centering step costs LayerNorm one extra reduction
    and a subtraction; RMSNorm trades that for a small amount of
    expressivity (no shift invariance).  In practice the loss is
    negligible at scale.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import rms_norm
    >>> x = lucid.randn(2, 128, 512)
    >>> y = rms_norm(x, normalized_shape=(512,))
    >>> y.shape
    (2, 128, 512)
    """
    from lucid._factories.creation import ones

    if eps is None:
        from lucid._dtype import finfo

        eps = float(finfo(input.dtype).eps)
    norm_shape = tuple(int(d) for d in normalized_shape)
    if not norm_shape:
        raise ValueError("rms_norm: normalized_shape must not be empty")
    trailing = tuple(int(d) for d in input.shape[-len(norm_shape) :])
    if trailing != norm_shape:
        raise ValueError(
            f"rms_norm: normalized_shape {norm_shape} does not match the "
            f"trailing dimensions of x {trailing}"
        )

    # The engine normalises over the last axis only.  RMS over the last k axes
    # is identical to RMS over those axes flattened into one, so collapse them,
    # normalise, and restore the shape.  Without this ``normalized_shape`` was
    # accepted and ignored: every call normalised over the final axis whatever
    # was asked for.
    C = 1
    for d in norm_shape:
        C *= d
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=input.device, dtype=input.dtype))
    )
    if len(norm_shape) == 1:
        return _wrap(_C_engine.nn.rms_norm(_unwrap(input), w, eps))

    orig_shape = tuple(int(d) for d in input.shape)
    lead = orig_shape[: len(orig_shape) - len(norm_shape)]
    flat = input.reshape(*lead, C)
    w_flat = w if len(w.shape) == 1 else _unwrap(_wrap(w).reshape(C))
    out = _wrap(_C_engine.nn.rms_norm(_unwrap(flat), w_flat, eps))
    return out.reshape(*orig_shape)


def _fold_instance_stats(
    x: Tensor, running_mean: Tensor, running_var: Tensor, momentum: float
) -> None:
    """Fold ``x``'s per-instance statistics, averaged over the batch, into the buffers.

    Written into the buffers in place, so a view's base and anything else
    holding them sees the update, and each one's version moves, as the
    reference's do.  Both or neither: a buffer whose elements overlap
    refuses the write, and when the variance refuses it after the mean took
    its own, the mean is put back.
    """
    import lucid as _lucid

    spatial_dims = list(range(2, x.ndim))
    # Under autocast the statistics stay at the buffers' dtype: ``x`` is
    # cast up before the reductions, since ``var`` keeps its input's dtype
    # and a half variance loses the small spread to cancellation.
    with _lucid.no_grad():
        x_s = x if x.dtype == running_mean.dtype else x.to(dtype=running_mean.dtype)
        batch_mean = x_s.mean([0, *spatial_dims])
        batch_var = x_s.var(spatial_dims, correction=1).mean(0)
        new_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
        new_var = (1.0 - momentum) * running_var + momentum * batch_var
        kept = running_mean.clone()
        running_mean.copy_(new_mean)
        try:
            running_var.copy_(new_var)
        except RuntimeError:
            running_mean.copy_(kept)
            raise
    # Metal: schedule the new values now, or each step's update keeps the
    # previous one's lazy graph — the activations it was computed from
    # included — and the chain grows until it runs out of memory.  Under a
    # tracer the buffers are restored as soon as the trace closes and the
    # compiled executable (or eager fallback) owns the real write.
    if _C_engine.compile.current_tracer() is None:
        _C_engine.eval_tensors_async([running_mean._impl, running_var._impl])


def _check_instance_norm_input(
    x: Tensor,
    use_input_stats: bool,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
) -> None:
    """Refuse an input instance normalisation has no statistics for."""
    for name, peer in (
        ("running_mean", running_mean),
        ("running_var", running_var),
        ("weight", weight),
        ("bias", bias),
    ):
        if peer is not None and peer.device != x.device:
            raise _C_engine.DeviceMismatch(
                f"DeviceMismatch (instance_norm: x/{name}): "
                f"expected {x.device.type}, got {peer.device.type}"
            )
    if not use_input_stats and (running_mean is None or running_var is None):
        raise RuntimeError(
            "Expected running_mean and running_var to be defined when use_input_stats is false"
        )
    if x.ndim < 3:
        raise ValueError(
            f"instance_norm: expected at least 3-D input (N, C, *spatial), " f"got ndim={x.ndim}"
        )
    if use_input_stats and math.prod(x.shape[2:]) == 1:
        # One element is its own mean with zero variance, so the output is
        # ``bias`` whatever the input; the reference refuses it too.
        raise ValueError(
            "instance_norm: expected more than 1 spatial element when training, "
            f"got input size {tuple(x.shape)}"
        )


def _cast_if_amp(t: Tensor | None, dt: DTypeLike, active: bool) -> Tensor | None:
    return t.to(dt) if active and t is not None and t.dtype != dt else t


def instance_norm(
    input: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Instance normalization (Ulyanov, Vedaldi & Lempitsky, 2016).

    Normalises each ``(sample, channel)`` slice against its **own**
    spatial mean and variance — i.e. statistics are reduced only over
    the spatial axes.  Originally introduced for fast neural style
    transfer; widely used wherever per-image contrast should be made
    invariant (image generation, image-to-image translation).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *spatial)`` with ``ndim >= 3``.
    running_mean : Tensor or None, optional
        Running per-channel mean of shape ``(C,)``.  Normalises with it
        when ``use_input_stats=False``; otherwise, given with
        ``running_var``, the batch's mean is folded into it in place.
    running_var : Tensor or None, optional
        Running per-channel variance of shape ``(C,)``, used and updated
        as ``running_mean`` is, with each instance's unbiased variance
        averaged over the batch.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.
    use_input_stats : bool, optional
        When ``True`` (default) statistics come from the current
        instance, which needs more than one spatial element (else
        ``ValueError``).  When ``False``, ``running_mean`` and
        ``running_var`` must be defined (else ``RuntimeError``).
    momentum : float, optional
        Weight the batch takes when it is folded into the running
        buffers: ``running = (1 - momentum) * running + momentum * batch``.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    Math (reduction taken only over spatial axes :math:`S`):

    .. math::

        \mu_{n,c} &= \frac{1}{|S|}\sum_{s \in S} x_{n,c,s} \\
        \sigma^2_{n,c} &= \frac{1}{|S|}\sum_{s \in S} (x_{n,c,s} - \mu_{n,c})^2 \\
        y_{n,c,s} &= \gamma_c \cdot \frac{x_{n,c,s} - \mu_{n,c}}{\sqrt{\sigma^2_{n,c} + \epsilon}} + \beta_c

    InstanceNorm equals :func:`group_norm` with ``num_groups == C``.
    The lack of batch coupling also makes it the natural normaliser
    when adapting per-sample statistics matters more than population
    coherence — exactly the case in style transfer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import instance_norm
    >>> x = lucid.randn(4, 3, 64, 64)
    >>> y = instance_norm(x)
    >>> y.shape
    (4, 3, 64, 64)
    """
    _check_instance_norm_input(input, use_input_stats, running_mean, running_var, weight, bias)
    amp_was_active = _C_engine.amp_is_active(_unwrap(input).device)
    with _autocast_entry("instance_norm", _unwrap(input)) as (xi,):
        input = _wrap(xi)
        weight = _cast_if_amp(weight, input.dtype, amp_was_active)
        bias = _cast_if_amp(bias, input.dtype, amp_was_active)
        spatial_dims = list(range(2, input.ndim))
        bcast_shape = [1, int(input.shape[1])] + [1] * (input.ndim - 2)
        if use_input_stats or running_mean is None or running_var is None:
            mean = input.mean(spatial_dims, keepdim=True)
            var = input.var(spatial_dims, keepdim=True, correction=0)
            if running_mean is not None and running_var is not None and momentum != 0.0:
                _fold_instance_stats(input, running_mean, running_var, momentum)
        else:
            rm = _cast_if_amp(running_mean, input.dtype, amp_was_active)
            rv = _cast_if_amp(running_var, input.dtype, amp_was_active)
            assert rm is not None and rv is not None
            mean, var = rm.reshape(bcast_shape), rv.reshape(bcast_shape)
        from lucid._factories.creation import full as _full_factory

        eps_t = _full_factory((), eps, device=input.device, dtype=var.dtype)
        y = (input - mean) / (var + eps_t).sqrt()
        y = y * weight.reshape(bcast_shape) if weight is not None else y
        return y + bias.reshape(bcast_shape) if bias is not None else y


# ── P3 fill: local_response_norm (functional form of LocalResponseNorm) ────


def local_response_norm(
    input: Tensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> Tensor:
    r"""Local response normalization (Krizhevsky, Sutskever & Hinton, 2012).

    Implements lateral inhibition: each activation is divided by a
    function of the squared activations of its channel neighbours.
    Introduced with AlexNet and largely superseded by batch / layer
    normalisation, but still used in some classic architectures.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *spatial)`` (any number of spatial
        dims).  Inputs with fewer than 2 dimensions are returned
        unchanged.
    size : int
        Number of neighbouring channels (the size of the local window)
        used in the normalisation.
    alpha : float, optional
        Multiplicative scaling factor on the squared-activation sum.
        Default ``1e-4``.
    beta : float, optional
        Exponent applied to the normalising factor.  Default ``0.75``.
    k : float, optional
        Additive constant guarding the division.  Default ``1.0``.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    Notes
    -----
    The scale is the *mean* of the squared activations over a window
    of :math:`n = \text{size}` channels, so :math:`\alpha` is divided
    by :math:`n`:

    .. math::

        y_{i,c,\ldots} = \frac{x_{i,c,\ldots}}{\left(k + \frac{\alpha}{n}
            \sum_{c'=c-\lfloor n/2 \rfloor}^{c+\lfloor (n-1)/2 \rfloor}
            x_{i,c',\ldots}^2 \right)^\beta}

    Channels outside ``[0, C)`` count as zero and stay in the
    denominator :math:`n`.  The window reaches ``size // 2`` channels
    below :math:`c` and ``(size - 1) // 2`` above it, so for an even
    ``size`` it leans towards the lower channels.  This is the
    reference framework's definition (an average pool of :math:`x^2`
    along the channel axis); the AlexNet paper writes the same rule
    with the :math:`1/n` folded into :math:`\alpha`.  LRN predates
    BatchNorm and provides only lateral inhibition (no learnt affine),
    which is why modern architectures usually do not include it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import local_response_norm
    >>> x = lucid.randn(1, 96, 27, 27)   # AlexNet conv1 output shape
    >>> y = local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0)
    >>> y.shape
    (1, 96, 27, 27)
    """
    xi = _unwrap(input)
    if len(xi.shape) < 3:
        raise ValueError(
            f"Expected 3D or higher dimensionality input (got {len(xi.shape)} dimensions)"
        )
    if size <= 0:
        raise RuntimeError(
            f"local_response_norm: kernel size should be greater than zero, but got size={size}"
        )
    # The row applies once, here: its parts (a FP32 ``sum`` and a KEEP
    # ``div``) would otherwise meet in two dtypes under Metal autocast.
    with _autocast_entry("local_response_norm", _lrn_operand(xi)) as (xi,):
        return _wrap(_lrn_body(xi, size, alpha, beta, k))


def _lrn_operand(xi: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """:func:`local_response_norm`'s input, floating and at its pooling part's row."""
    # The normaliser is a fractional power of a scaled sum, so the result
    # is a real number whatever the input was — the same statement the
    # engine's ``real_valued`` schemas make.  Computing it in an integer
    # dtype does not approximate the answer, it erases it: ``alpha`` is
    # 1e-4, which truncates to zero, so every scale collapses to ``k`` and
    # the op becomes the identity.  Promote once, here, because this is a
    # composite and there is no schema for the engine to read.
    if xi.dtype in (
        _C_engine.Dtype.Bool,
        _C_engine.Dtype.I8,
        _C_engine.Dtype.I16,
        _C_engine.Dtype.I32,
        _C_engine.Dtype.I64,
    ):
        xi = _C_engine.astype(xi, _C_engine.Dtype.F32)
    # The reference averages a rank-3 input's window with ``avg_pool2d`` and
    # a deeper one's with ``avg_pool3d``, whose CPU row is FP32, so under a
    # CPU autocast its rank-4+ result is float32 and its rank-3 one is not.
    # A row cannot depend on the rank, so the deeper input also takes the
    # pooling part's row (a no-op on Metal, where it keeps).
    if len(xi.shape) > 3:
        (xi,) = _C_engine.amp_cast_inputs("avg_pool3d", [xi])
    return xi


def _lrn_body(
    xi: _C_engine.TensorImpl, size: int, alpha: float, beta: float, k: float
) -> _C_engine.TensorImpl:
    """:func:`local_response_norm` on a floating input of rank 3 or more."""
    ndim = len(xi.shape)
    C = int(xi.shape[1])
    # Pad totals to ``size - 1`` so the post-pad unfold yields exactly C
    # windows.  The window for channel c starts ``size // 2`` below it, so
    # for an even ``size`` the extra channel sits on the low side.
    pad_l = size // 2
    pad_r = (size - 1) // 2

    x_sq = _C_engine.mul(xi, xi)
    pad_pairs: list[tuple[int, int]] = [(0, 0)] + [(pad_l, pad_r)] + [(0, 0)] * (ndim - 2)
    x_sq_pad = _C_engine.pad(x_sq, pad_pairs, 0.0)

    spatial_size = 1
    for d in range(2, ndim):
        spatial_size *= int(xi.shape[d])
    flat = _C_engine.reshape(x_sq_pad, [int(xi.shape[0]), C + pad_l + pad_r, spatial_size])
    # Transpose to (N, S, C+2h) so unfold_dim slides along the last axis.
    flat_t = _C_engine.permute(flat, [0, 2, 1])
    unf = _C_engine.unfold_dim(flat_t, 2, size, 1)  # (N, S, C, size)
    window_sum = _C_engine.sum(unf, [3], False)  # (N, S, C)
    window_sum_t = _C_engine.permute(window_sum, [0, 2, 1])  # (N, C, S)
    out_shape = list(xi.shape)
    window_sum_rs = _C_engine.reshape(window_sum_t, out_shape)

    # A mean, not a sum: padded zeros stay in the denominator, so every
    # channel divides by ``size``.  Mean, then ``* alpha``, then ``+ k`` is
    # the reference's order, which keeps float64 results bit-close.
    size_t = _C_engine.full(out_shape, float(size), xi.dtype, xi.device)
    k_t = _C_engine.full(out_shape, k, xi.dtype, xi.device)
    alpha_t = _C_engine.full(out_shape, alpha, xi.dtype, xi.device)
    window_mean = _C_engine.div(window_sum_rs, size_t)
    scale = _C_engine.pow_scalar(_C_engine.add(_C_engine.mul(window_mean, alpha_t), k_t), beta)
    return _C_engine.div(xi, scale)


def normalize(input: Tensor, p: float = 2.0, dim: DimLike = 1, eps: float = 1e-12) -> Tensor:
    r"""Normalize a tensor to unit :math:`L_p` norm along a dimension.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    p : float, optional
        Exponent of the norm (default: 2.0 for Euclidean norm).
    dim : int, sequence of int, or None, optional
        Dimension(s) along which to normalize (default: 1).  A sequence
        takes the norm over all of them together; ``None`` over every
        element.
    eps : float, optional
        Lower bound on the denominator, for numerical stability.

    Returns
    -------
    Tensor
        Normalized tensor with unit :math:`L_p` norm along ``dim``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> x = lucid.tensor([[3.0, 4.0]])
    >>> F.normalize(x, p=2, dim=1)
    tensor([[0.6, 0.8]])
    >>> F.normalize(lucid.tensor([[3.0], [4.0]]), dim=None)
    tensor([[0.6], [0.8]])

    Notes
    -----
    .. math::

        \hat{x}_i = \frac{x_i}{\max\!\big(\|x\|_p,\, \epsilon\big)},
        \qquad \|x\|_p = \Big(\sum_i |x_i|^p\Big)^{1/p}

    along the chosen dimension.  The ``eps`` clamp avoids division by
    zero when the slice is all zeros.  Use ``p=2`` for cosine-similarity
    style embeddings, ``p=1`` for probability-simplex projection-like
    behaviour, and ``p=inf`` for max-norm clipping.  An integer or bool
    ``input`` is refused with a ``DtypeMismatch``, as the reference refuses it.
    """
    xi = _require_normable(_unwrap(input), "normalize")
    _to_axes(dim, "normalize")  # refuses a bool or a non-index dim up front
    if isinstance(dim, int):
        return _wrap(_C_engine.nn.lp_normalize(xi, p, dim, eps))
    # Several dims, or all of them, have no kernel: the norm over them,
    # kept for broadcasting, divides as the kernel's does.  ``normalize`` has
    # no autocast row (nor has the reference's ``Tensor.norm`` it divides
    # by), so it keeps the input, with autocast off so ``vector_norm`` does
    # not apply its own FP32 Metal row inside.
    with _autocast_suspended(xi.device):
        denom = vector_norm(input, ord=p, dim=dim, keepdim=True)
        return input / denom.clamp(min=eps)
