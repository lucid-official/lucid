"""
nn.functional dropout operations.
All implementations use the C++ engine ops; no numpy.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._types import DeviceLike
from lucid.nn.functional._inplace import _in_place

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# SELU affine constants (from the SELU paper)
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_ALPHA_PRIME = -_SELU_ALPHA * _SELU_SCALE  # ≈ -1.7580993408473766


def _check_probability(p: float) -> None:
    """Every dropout entry point takes ``p`` in ``[0, 1]``, as the reference does.

    Raises:
        ValueError: ``p`` is outside ``[0, 1]`` (or NaN).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")


def _drop_everything(x: Tensor, p: float, training: bool) -> Tensor | None:
    """Check ``p``; at ``p = 1`` or identity cases return without drawing a mask.

    Dropping every element leaves nothing to rescale — ``1 / (1 - p)`` is
    undefined — so the engine kernels refuse ``p = 1``.  The reference
    answers ``x * 0`` instead: zeros that keep the autograd edge, so the
    gradient reaching ``x`` is zero rather than missing.  Returns ``None``
    when a mask has to be drawn.
    """
    _check_probability(p)
    if p == 1.0:
        return x * 0.0 if training else x
    if (
        (not training or p == 0.0)
        and _C_engine.compile.current_tracer() is None
        and not (x.requires_grad and _C_engine.grad_enabled())
    ):
        return x
    return None


def _alloc_philox_state(device: DeviceLike) -> Tensor:
    """Allocate a fresh MPSGraph Philox state tensor seeded from Lucid's RNG.

    MPSGraph's stateful ``randomTensorWithShape:descriptor:stateTensor:``
    API expects a ``int32[7]`` state — the Philox-4x32 internal layout.
    We initialise it with 7 random ``int32`` values drawn from Lucid's
    global :class:`Generator` so distinct dropout call sites
    (potentially within the same model) start from uncorrelated
    sequences.

    Critically the allocation runs with the active :func:`_tracing`
    tracer temporarily detached.  Without that, the inner
    :func:`lucid.randint` call would record a ``"randint"`` op into the
    trace and the resulting tensor would be the *output* of that op
    rather than an external feed — making it ineligible for the
    ``compile_generic_fused_step_with_vars`` variable promotion which
    requires every ``feed_id`` in ``variable_pairs`` to be in
    :attr:`Tracer.external_feeds`.  Suppressing the tracer keeps the
    state buffer "outside" the captured DAG; when
    :func:`_C_engine.nn.dropout_stateful` consumes it as input, the
    tracer mints a fresh feed id and the variable promotion works.
    """
    import lucid as _lucid
    from lucid._engine_scope import untraced

    with untraced():
        # int32 max is 2**31 - 1.  Sample over the full positive range.
        return _lucid.randint(0, 2**31 - 1, (7,), dtype=_lucid.int32, device=device)


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""Element-wise Bernoulli dropout — the canonical Srivastava 2014 form.

    During training, each scalar in ``input`` is independently zeroed with
    probability ``p`` and survivors are scaled by :math:`1 / (1 - p)`
    so the expected activation magnitude stays unchanged.  At inference
    (``training=False``) the call is a no-op identity.

    Parameters
    ----------
    input : Tensor
        Input activations of any shape.
    p : float, optional
        Drop probability in :math:`[0, 1]`.  ``p = 0`` short-circuits
        to identity; ``p = 1`` returns ``input * 0`` in training.
        Default ``0.5``.
    training : bool, optional
        When ``False``, returns ``input`` unchanged — identity at
        inference time.  Modules built on top (``nn.Dropout``) pass
        ``self.training`` here.  Default ``True``.
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``.
        Default ``False``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    For each element :math:`x_i`, draw an independent
    :math:`m_i \sim \mathrm{Bernoulli}(1 - p)` and apply

    .. math::

        y_i = \frac{m_i}{1 - p}\, x_i.

    Backward uses the *same* saved mask, so gradients also propagate
    only through surviving positions.  For correlated activations
    (sequence / image channels) prefer :func:`dropout1d` /
    :func:`dropout2d` / :func:`dropout3d`, which drop whole feature
    maps and give stronger regularisation.

    **Compile path routing.**  When called inside a
    :func:`lucid.compile._tracing` context with ``training=True`` and
    ``p > 0``, this wrapper allocates a Philox state buffer and routes
    through the sibling engine op ``_C_engine.nn.dropout_stateful``.
    The captured trace then carries the 2-input/2-output schema the
    compile path needs to plumb MPSGraph's stateful Philox RNG —
    giving genuinely-per-dispatch varying masks where the stateless
    seed path produces dispatch-deterministic ones.  Eager-only
    callers (no active tracer) keep using the standard ``nn.dropout``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout
    >>> x = lucid.ones(2, 4)
    >>> y = dropout(x, p=0.5, training=True)
    >>> # ~50 % of entries are 0; survivors are 2.0 (= 1 / (1 - 0.5))

    See Also
    --------
    lucid.nn.Dropout : ``nn.Module`` wrapper that owns the ``training`` flag.
    dropout1d, dropout2d, dropout3d : whole-channel variants.
    alpha_dropout : SELU-compatible dropout that preserves the
        self-normalising statistics.

    References
    ----------
    .. [1] Srivastava et al., *Dropout: A Simple Way to Prevent Neural
       Networks from Overfitting*, JMLR 2014.
    """
    if inplace:
        return _dropout_in_place(input, "dropout", p, training, lambda t: dropout(t, p, training))
    dropped = _drop_everything(input, p, training)
    if dropped is not None:
        return dropped
    if training and p > 0.0 and _C_engine.compile.current_tracer() is not None:
        # Only route through the stateful sibling when we know the
        # downstream compile path can promote the state buffer to an
        # MPSGraph variable — i.e. inside :func:`fused_step`'s trace
        # block.  The forward-only :func:`lucid.compile` path uses
        # ``compile_trace``, which has no variable-promotion hook, so
        # routing through ``dropout_stateful`` there would compile
        # successfully but emit the *same* mask every dispatch (the
        # very regression the prior X2 prototype hit).  Fall back to
        # eager via the plain ``dropout`` op there instead — the
        # ``CompiledModule`` cache will register the signature as
        # eager-only and subsequent calls skip the recompile attempt.
        from lucid.compile._entry.fused_step import _is_fused_step_tracing

        if _is_fused_step_tracing():
            # Compile path: thread an explicit Philox state buffer so
            # the MPSGraph emitter can use the stateful RNG API.  The
            # state buffer's TensorImpl identity is captured as an
            # external feed — ``_fused_step.py`` walks the trace for
            # ``dropout_stateful`` ops, builds the matching
            # ``(state_in_tid, state_out_tid)`` variable pairs, and
            # routes the compile through
            # ``compile_generic_fused_step_with_vars``.
            xi = _unwrap(input)
            state = _alloc_philox_state(xi.device)
            y, _ = _C_engine.nn.dropout_stateful(xi, _unwrap(state), p, training)
            return _wrap(y)
    return _wrap(_C_engine.nn.dropout(_unwrap(input), p, training))


def _alpha_affine(p: float) -> tuple[float, float]:
    """``(a, b)`` that restore zero mean and unit variance after alpha dropout.

    With keep probability ``q = 1 - p`` a dropped element takes
    ``alpha'``; ``a = (q (1 + p alpha'^2))^(-1/2)`` and ``b = -a alpha' p``.
    """
    keep = 1.0 - p
    a = (keep * (1.0 + p * _ALPHA_PRIME**2)) ** -0.5
    return a, -a * _ALPHA_PRIME * p


def _dropout_in_place(
    x: Tensor, name: str, p: float, training: bool, drop: Callable[[Tensor], Tensor]
) -> Tensor:
    """``inplace=True`` for every dropout: ``drop(x)`` written into ``x``.

    Nothing is drawn when nothing would be dropped (``p = 0``, or not
    training), so ``x`` comes back untouched — even a leaf that requires
    grad, as the reference leaves it.
    """
    _check_probability(p)
    if not training or p == 0.0:
        return x
    return _in_place(x, f"{name}(inplace=True)", drop)


def _channel_dropout(
    x: Tensor, p: float, training: bool, inplace: bool, name: str, unbatched: bool = False
) -> Tensor:
    """Whole-channel dropout -- the one body of dropout1d / 2d / 3d.

    ``unbatched`` is an input with no batch dimension: the channel axis is
    its first, so a batch axis of one is added for the draw and dropped
    again, as the reference does.
    """

    def drop(t: Tensor) -> Tensor:
        t = t.unsqueeze(0) if unbatched else t
        out = _drop_everything(t, p, training)
        if out is None:
            out = _wrap(_C_engine.nn.dropoutnd(_unwrap(t), p, training))
        return out.squeeze(0) if unbatched else out

    if inplace:
        return _dropout_in_place(x, name, p, training, drop)
    return drop(x)


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""Channel-wise dropout for 1-D sequence / feature inputs.

    Drops entire 1-D feature maps (channels) of an :math:`(N, C, L)`
    tensor with probability ``p``, scaling the survivors by
    :math:`1/(1-p)` to keep the activation magnitude unbiased.
    Standard per-element :func:`dropout` is statistically weak when
    applied within strongly-correlated feature maps (consecutive
    positions in a sequence are highly correlated), so masking the
    *whole* channel at once provides stronger regularisation in
    1-D CNNs and temporal feature extractors.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape :math:`(N, C, L)`, or :math:`(C, L)` for one
        sample with no batch axis.  Any other rank raises.
    p : float, optional
        Channel-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, returns ``input`` unchanged — identity at
        inference time (default ``True``).
    inplace : bool, optional
        If ``True``, drop channels of ``input`` itself and return ``input``
        (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` independently and
    apply

    .. math::

        y_{n, c, \ell} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, \ell}.

    The :math:`1/(1-p)` scale ("inverted dropout") preserves
    :math:`\mathbb{E}[y] = \mathbb{E}[x]` so no rescaling is needed
    at inference.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout1d
    >>> x = lucid.ones((2, 4, 5))
    >>> y = dropout1d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    _check_probability(p)
    if input.ndim not in (2, 3):
        raise _C_engine.ShapeMismatch(
            f"dropout1d: expected a 2-D (C, L) or 3-D (N, C, L) input, got a {input.ndim}-D one"
        )
    return _channel_dropout(input, p, training, inplace, "dropout1d", unbatched=input.ndim == 2)


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""Channel-wise dropout for 2-D (spatial) feature maps.

    Drops entire 2-D feature maps of an :math:`(N, C, H, W)` tensor
    with probability ``p`` and rescales the survivors by
    :math:`1/(1-p)`.  This is the dropout variant of choice for
    convolutional networks (Tompson et al. 2015 "SpatialDropout"):
    adjacent pixels within a feature map are spatially correlated,
    so elementwise dropout removes very little information.
    Channel-wise masking forces independence between feature maps
    instead.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape :math:`(N, C, H, W)`.
    p : float, optional
        Channel-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).
    inplace : bool, optional
        If ``True``, zero the dropped feature maps of ``input`` itself
        and return ``input`` (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` and broadcast
    over the spatial dimensions:

    .. math::

        y_{n, c, h, w} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, h, w}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout2d
    >>> x = lucid.ones((2, 8, 4, 4))
    >>> y = dropout2d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    return _channel_dropout(input, p, training, inplace, "dropout2d")


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""Channel-wise dropout for 3-D (volumetric) feature maps.

    Drops entire 3-D feature maps of an :math:`(N, C, D, H, W)`
    tensor with probability ``p`` and rescales survivors by
    :math:`1/(1-p)`.  Used in 3-D ConvNets for video and volumetric
    medical imaging, where spatio-temporal locality makes per-
    element dropout statistically inefficient (same rationale as
    :func:`dropout2d`).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape :math:`(N, C, D, H, W)`.  Any other rank is
        taken as one sample with no batch axis, its channels on axis 0.
    p : float, optional
        Channel-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).
    inplace : bool, optional
        If ``True``, ``input`` itself is written and returned
        (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` and broadcast
    over the volumetric dimensions:

    .. math::

        y_{n, c, d, h, w} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, d, h, w}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout3d
    >>> x = lucid.ones((1, 4, 3, 4, 4))
    >>> y = dropout3d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    return _channel_dropout(input, p, training, inplace, "dropout3d", unbatched=input.ndim != 5)


def alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    r"""Alpha dropout — variance-preserving dropout for SELU networks.

    Designed to be paired with the SELU activation (Klambauer
    et al. 2017 "Self-Normalizing Neural Networks").  Standard
    dropout breaks the carefully-tuned zero-mean / unit-variance
    propagation that makes SELU networks self-normalising; alpha
    dropout fixes this by replacing dropped activations with the
    SELU saturation value :math:`\alpha' = -\alpha \cdot \text{scale}`
    (rather than zero), then applying an affine rescaling chosen so
    that the post-dropout activations again have zero mean and unit
    variance.

    Parameters
    ----------
    input : Tensor
        Input tensor, any shape.
    p : float, optional
        Element-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``False``).
    inplace : bool, optional
        If ``True``, write the result into ``input`` and return ``input``
        (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    With keep probability :math:`q = 1 - p`, each element is
    independently set to its original value (with probability
    :math:`q`) or to :math:`\alpha'` (with probability :math:`p`).
    The result is then affine-transformed:

    .. math::

        y = a \cdot \mathrm{mix}(x, \alpha') + b,
        \quad
        a = \big(q + p\,(\alpha')^2\,q\big)^{-1/2},
        \quad
        b = -a\,\alpha'\,p,

    chosen so that :math:`\mathbb{E}[y] = 0` and
    :math:`\mathrm{Var}(y) = 1` when the input has zero mean and
    unit variance.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import alpha_dropout
    >>> x = lucid.randn(8)
    >>> y = alpha_dropout(x, p=0.1, training=True)  # doctest: +SKIP
    """
    if inplace:
        return _dropout_in_place(
            input, "alpha_dropout", p, training, lambda t: alpha_dropout(t, p, training)
        )
    dropped = _drop_everything(input, p, training)
    if dropped is not None:
        return dropped
    if not training or p == 0.0:
        return input
    return _wrap(_C_engine.nn.alpha_dropout(_unwrap(input), p, training))


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    r"""Channel-wise alpha dropout for SELU convolutional networks.

    Combines the channel-granularity of :func:`dropout2d` /
    :func:`dropout3d` with the variance-preserving substitution of
    :func:`alpha_dropout`.  Whole channels are dropped (replaced
    rather than zeroed) and the result is affine-rescaled to keep
    the self-normalising property — appropriate for SELU-activated
    CNNs where spatially-correlated activations would make per-
    element alpha dropout ineffective.

    For inputs of rank less than 2, falls back to per-element
    :func:`alpha_dropout`.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape :math:`(N, C, *)` (one or more
        spatial dimensions).
    p : float, optional
        Channel-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``False``).
    inplace : bool, optional
        If ``True``, ``input`` is overwritten with the result and
        returned (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Notes
    -----
    A Bernoulli mask of shape :math:`(N, C)` is drawn and
    broadcast across the spatial axes.  A dropped channel takes the
    SELU saturation value :math:`\alpha'` in every position, and the
    result goes through the same affine map as :func:`alpha_dropout`:
    a kept channel becomes :math:`a x + b`, a dropped one the constant
    :math:`a \alpha' + b`.  Variance preservation holds in expectation
    when the input statistics already satisfy the SELU fixed-point
    assumptions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import feature_alpha_dropout
    >>> x = lucid.randn(2, 4, 3, 3)
    >>> y = feature_alpha_dropout(x, p=0.25, training=True)  # doctest: +SKIP
    """
    if inplace:
        return _dropout_in_place(
            input,
            "feature_alpha_dropout",
            p,
            training,
            lambda t: feature_alpha_dropout(t, p, training),
        )
    dropped = _drop_everything(input, p, training)
    if dropped is not None:
        return dropped
    if not training or p == 0.0:
        return input

    xi = _unwrap(input)
    if len(xi.shape) < 2:
        return alpha_dropout(input, p, training)

    N, C = int(xi.shape[0]), int(xi.shape[1])
    keep_prob = 1.0 - p
    a_coeff, b_coeff = _alpha_affine(p)

    # (N, C) Bernoulli mask, broadcast over the feature dimensions.
    rand_nc = _C_engine.rand([N, C], xi.dtype, xi.device)
    thresh = _C_engine.full([N, C], keep_prob, xi.dtype, xi.device)
    keep_nc = _C_engine.less(rand_nc, thresh)
    mask_shape = [N, C] + [1] * (len(xi.shape) - 2)
    keep = _C_engine.broadcast_to(_C_engine.reshape(keep_nc, mask_shape), list(xi.shape))

    # A kept channel is ``a * x + b``; a dropped one is the SELU saturation
    # value pushed through the same affine map, ``a * alpha' + b``.
    a_t = _C_engine.full(xi.shape, a_coeff, xi.dtype, xi.device)
    b_t = _C_engine.full(xi.shape, b_coeff, xi.dtype, xi.device)
    kept = _C_engine.add(_C_engine.mul(a_t, xi), b_t)
    dropped_t = _C_engine.full(xi.shape, a_coeff * _ALPHA_PRIME + b_coeff, xi.dtype, xi.device)
    return _wrap(_C_engine.where(keep, kept, dropped_t))
