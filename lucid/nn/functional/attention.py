"""
nn.functional attention operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.dropout import _check_probability

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import DeviceLike, DTypeLike

_NEG_INF: float = float("-inf")


def _keep_mask_to_additive(keep: Tensor, dtype: DTypeLike) -> Tensor:
    """A boolean keep-mask as the additive mask it stands for: ``0`` / ``-inf``.

    ``-inf`` rather than a large finite negative: a finite stand-in lets a
    row with no key left normalise over the banned keys, and it averages V.
    """
    import lucid as _lucid

    zeros = _lucid.zeros(keep.shape, dtype=dtype, device=keep.device)
    return _lucid.masked_fill(zeros, _lucid.logical_not(keep), _NEG_INF)


def _causal_additive_mask(
    lq: int, lk: int, offset: int, dtype: DTypeLike, device: DeviceLike
) -> Tensor:
    """``(lq, lk)`` additive mask letting query ``i`` see keys ``0..i + offset``."""
    import lucid as _lucid

    keep = _lucid.tril(_lucid.ones((lq, lk), dtype=_lucid.bool_, device=device), offset)
    return _keep_mask_to_additive(keep, dtype)


def _fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    scale: float,
    is_causal: bool,
) -> Tensor:
    """The engine's attention kernel.

    On metal the forward's extra memory is linear in the sequence lengths
    at every head dim up to 128.  With more than 8 queries, MLX fuses head
    dims 64, 72, 80, 96 and 128 and forms no ``(Lq, Lk)`` buffer.  The
    engine zero-pads a narrower q, k and v to the next of those once the
    scores would be at least twice the padded copies; below that the
    scores are themselves linear in the lengths.  The full scores are
    formed for a wider head dim, by the metal backward, which MLX
    differentiates unfused, and by the CPU kernel, which keeps the weights
    for its backward.
    """
    return _wrap(
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(query),
            _unwrap(key),
            _unwrap(value),
            _unwrap(attn_mask) if attn_mask is not None else None,
            scale,
            is_causal,
        )
    )


def _masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    scale: float,
    *,
    dropout_p: float = 0.0,
    need_weights: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """The masked-softmax core every attention entry point runs through.

    ``attn_mask`` is additive, with any causal triangle already folded in.
    A query row whose every key is ``-inf`` has no distribution to
    normalise; it attends to nothing, so its output, its weights and every
    gradient through it are ``0``.  The row's mask is zeroed before the
    softmax and its output after, so no kernel ever sees an all ``-inf``
    row — the backends disagreed on one (the Metal kernel returned NaN,
    the CPU one ``0`` or the mean of V by dtype).

    The fused kernel runs unless the weights are wanted or dropout is on,
    since it never materialises the probabilities either needs.  The
    weights returned are the ones applied to ``value`` — after dropout.

    An additive mask in another float dtype — a float32 bias against
    half-precision queries, which the reference accepts — is cast to the
    query's dtype here.  The cast is recorded, so a learnable mask's
    gradient comes back in the mask's own dtype.  The engine kernels
    refuse a mismatched mask rather than read it in the wrong dtype.
    """
    import lucid as _lucid

    dead: Tensor | None = None
    if attn_mask is not None:
        if attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)
        dead = attn_mask.detach().amax(dim=-1, keepdim=True) == _NEG_INF
        attn_mask = _lucid.masked_fill(attn_mask, dead, 0.0)

    weights: Tensor | None = None
    if need_weights or dropout_p > 0.0:
        from lucid.nn.functional.activations import softmax
        from lucid.nn.functional.dropout import dropout

        scores = _lucid.matmul(query, key.mT) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = softmax(scores, dim=-1)
        if dead is not None:
            weights = _lucid.masked_fill(weights, dead, 0.0)
        if dropout_p > 0.0:
            weights = dropout(weights, p=dropout_p, training=True)
        out = _lucid.matmul(weights, value)
    else:
        out = _fused_attention(query, key, value, attn_mask, scale, False)
    if dead is not None:
        out = _lucid.masked_fill(out, dead, 0.0)
    return out, weights if need_weights else None


def _broadcast_leading(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """The shape the leading (batch, head) axes of Q, K and V broadcast to."""
    rank = max(len(s) for s in shapes)
    out: list[int] = []
    for axis in range(rank):
        sizes = {s[axis - rank + len(s)] for s in shapes if axis - rank + len(s) >= 0}
        sizes.discard(1)
        if len(sizes) > 1:
            raise _C_engine.InvalidArgument(
                "scaled_dot_product_attention: the leading dimensions of query, "
                f"key and value do not broadcast: {shapes}; for fewer key / value "
                "heads than query heads, pass enable_gqa=True"
            )
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


def _attention_operands(
    query: Tensor, key: Tensor, value: Tensor, enable_gqa: bool, dropout_p: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Q, K and V with one leading shape, as the reference's math reads them.

    With ``enable_gqa`` each key / value head serves ``H_q / H_kv``
    consecutive query heads, so it is repeated that many times.  The
    leading axes then broadcast against each other as a matmul's would —
    a key shared across the batch is a ``1`` there.  The engine kernels
    take equal leading shapes only, and refused both.

    The call is checked here, before anything is read from the shapes: an
    operand below 2-D has no sequence axis, and a ``dropout_p`` outside
    ``[0, 1]`` is refused even when the fused path would not draw a mask.
    """
    _check_probability(dropout_p)
    if min(query.ndim, key.ndim, value.ndim) < 2:
        raise _C_engine.InvalidArgument(
            "scaled_dot_product_attention: query, key and value must each be at "
            f"least 2-D, got {query.ndim}-D, {key.ndim}-D and {value.ndim}-D"
        )
    if enable_gqa and min(query.ndim, key.ndim, value.ndim) >= 3:
        heads = int(query.shape[-3])
        grouped: list[Tensor] = []
        for name, t in (("key", key), ("value", value)):
            kv_heads = int(t.shape[-3])
            if heads % kv_heads != 0:
                raise ValueError(
                    "scaled_dot_product_attention: with enable_gqa, the "
                    f"{name} heads ({kv_heads}) must divide the query heads "
                    f"({heads})"
                )
            if kv_heads != heads:
                t = t.repeat_interleave(heads // kv_heads, dim=-3)
            grouped.append(t)
        key, value = grouped
    q_lead = query._impl.shape[:-2]
    k_lead = key._impl.shape[:-2]
    v_lead = value._impl.shape[:-2]
    if q_lead == k_lead == v_lead:
        return query, key, value
    shapes = [tuple(q_lead), tuple(k_lead), tuple(v_lead)]
    lead = _broadcast_leading(shapes)
    query, key, value = (
        t if shape == lead else t.expand(*lead, *t.shape[-2:])
        for t, shape in zip((query, key, value), shapes, strict=True)
    )
    return query, key, value


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    r"""Scaled dot-product attention — the core of every Transformer block.

    Computes the attention-weighted aggregation of value vectors using
    query-key dot products as similarity scores:

    .. math::

        \mathrm{Attention}(Q, K, V) =
            \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V

    where :math:`M` is an optional additive mask (``-inf`` to disallow
    attention at a position, ``0`` to allow).  The :math:`1/\sqrt{d_k}`
    factor keeps the softmax in a usable temperature regime as the head
    dimension :math:`d_k` grows — without it, large dot products saturate
    the softmax into a near one-hot distribution and gradients vanish.

    Parameters
    ----------
    query : Tensor
        Shape ``(B, H, T, E)``.  ``B`` batch, ``H`` heads, ``T`` query
        positions, ``E`` head dimension :math:`d_k`.
    key : Tensor
        Shape ``(B, H, S, E)``.
    value : Tensor
        Shape ``(B, H, S, V)``.  ``V`` may differ from ``E``.
    attn_mask : Tensor, optional
        Mask broadcast-compatible with ``(B, H, T, S)``, in either of two
        forms.  **Additive** (any float dtype): the values are added to
        the scores, so use large negatives (or ``-inf``) at positions to
        mask out.  **Boolean**: a *keep*-mask — ``True`` attends, ``False``
        is masked out — converted internally to the additive form it
        stands for, so the two spellings are exactly equivalent on every
        path, fused or not.  Combined with ``is_causal``, both apply.
        A query row whose every key is masked out (``False`` or ``-inf``
        throughout) attends to nothing: its output row is ``0`` and no
        gradient flows through it, on every device and dtype.  A large
        *finite* negative is not a mask-out — such a row still normalises,
        to the mean of the values.
    dropout_p : float, optional
        Dropout probability applied to attention weights during training.
        Default ``0.0``.
    is_causal : bool, optional
        If ``True``, apply an upper-triangular causal mask so each query
        position only attends to keys at the same or earlier positions
        (autoregressive decoder self-attention).  For a non-square score
        matrix the triangle is aligned top-left — query ``i`` sees keys
        ``0..i`` — so decoding against a KV cache wants an explicit mask.
    scale : float, optional
        Override the default :math:`1/\sqrt{d_k}` scale factor.
    enable_gqa : bool, optional
        Grouped-query attention: ``key`` and ``value`` may have fewer heads
        than ``query``, as long as theirs divides its; each key / value head
        then serves that many consecutive query heads.  Default ``False``.

    Returns
    -------
    Tensor
        Attention output of shape ``(B, H, T, V)``.  The leading axes of
        ``query``, ``key`` and ``value`` broadcast against each other, so a
        ``key`` of batch ``1`` serves every query in the batch.

    Notes
    -----
    Introduced in *Attention Is All You Need* (Vaswani et al., 2017).
    The implementation uses the log-sum-exp form of softmax for numerical
    stability under aggressive masking, and fuses the scale into the
    score matrix prior to softmax.  Causal masking enables efficient
    autoregressive decoding when combined with a key/value cache.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import scaled_dot_product_attention
    >>> q = lucid.randn(2, 8, 16, 64)          # (B, H, T, E)
    >>> k = lucid.randn(2, 8, 16, 64)
    >>> v = lucid.randn(2, 8, 16, 64)
    >>> out = scaled_dot_product_attention(q, k, v, is_causal=True)
    >>> out.shape
    (2, 8, 16, 64)
    """
    import math
    import lucid as _lucid

    query, key, value = _attention_operands(query, key, value, enable_gqa, dropout_p)
    if query.is_metal and query.ndim != 4 and key.ndim == value.ndim == query.ndim:
        # Metal's fused attention takes (B, H, L, D) and nothing else, and
        # raised "expected to be rank 4" on the (B, L, D) that MobileNet-v4's
        # hybrid attention hands it — the CPU takes any leading shape.  Fold
        # the leading axes into those four with views, attend, and unfold;
        # the views carry every gradient back, at any order.
        leading = tuple(query.shape[:-2])

        def fold(t: Tensor) -> Tensor:
            if t.ndim <= 3:
                while t.ndim < 4:
                    t = t.unsqueeze(-3)
                return t
            return t.reshape(-1, *t.shape[-3:])

        folded_mask = attn_mask
        if folded_mask is not None:
            while folded_mask.ndim < query.ndim:
                folded_mask = folded_mask.unsqueeze(0)
            if query.ndim > 4:
                # Folding merges the leading axes, so each must be real.
                folded_mask = folded_mask.expand(*leading, *folded_mask.shape[-2:])
            folded_mask = fold(folded_mask)
        out = scaled_dot_product_attention(
            fold(query),
            fold(key),
            fold(value),
            attn_mask=folded_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return out.reshape(*leading, *out.shape[-2:])

    if attn_mask is not None and attn_mask.dtype == _lucid.bool_:
        # Normalise a boolean mask to the additive form *once*, before
        # either path can form its own opinion about what ``True`` means.
        # They had opposite ones: the fused kernel treated ``True`` as
        # "mask this out", while the dropout branch treated it as "this
        # attends" — so the same mask inverted the moment dropout was
        # switched on.  Keep-semantics wins because it is what a boolean
        # mask conventionally means for scaled-dot-product attention.
        attn_mask = _keep_mask_to_additive(attn_mask, query.dtype)

    lq, lk = int(query.shape[-2]), int(key.shape[-2])
    if is_causal and (attn_mask is not None or lq != lk or dropout_p > 0.0):
        # The engine gives ``is_causal`` a meaning only for a square score
        # matrix with no other mask.  Beyond that its backends disagreed: the
        # Metal kernel let a mask win over causality and aligned a non-square
        # triangle bottom-right, while the CPU applied both, top-left — so the
        # same call attended to the future on one device only.  Fold the
        # top-left triangle into the additive mask so every device, and the
        # explicit form dropout needs, compute one thing.
        causal = _causal_additive_mask(lq, lk, 0, query.dtype, query.device)
        attn_mask = causal if attn_mask is None else attn_mask + causal
        is_causal = False

    if attn_mask is not None:
        # Match the scores' rank.  Given a learnable (L_q, L_k) bias against
        # (B, H, L_q, L_k) scores, the Metal backward returned a one-element
        # mask cotangent and the reshape back to the mask's shape raised; a
        # rank-matched view broadcasts identically and ``unsqueeze`` sums the
        # gradient back to the caller's shape.
        while attn_mask.ndim < query.ndim:
            attn_mask = attn_mask.unsqueeze(0)

    if attn_mask is not None and attn_mask.ndim >= 2:
        # The engine SDPA kernels (fused and manual) mishandle a mask that is
        # broadcast over the query dimension (query dim == 1, e.g. a (B,1,1,S)
        # key-padding mask) for some inputs — the mask lands on the wrong rows.
        # Expand the query dim to the number of query positions so every row
        # gets its mask, then materialize it.
        tq = query.shape[-2]
        if attn_mask.shape[-2] == 1 and tq != 1:
            target = list(attn_mask.shape)
            target[-2] = tq
            attn_mask = attn_mask.expand(*target).contiguous()

        # The CPU (Accelerate) SDPA kernel flattens the leading dims to one batch
        # and only broadcasts a mask that is either fully shared over that batch
        # — numel == Lq·Lk — or fully materialized (numel == B·…·Lq·Lk).  A
        # *partial* broadcast (e.g. a (1,H,Lq,Lk) relative-position bias, or a
        # (B,1,Lq,Lk) padding mask) has neither numel and is mis-indexed past the
        # buffer → silently wrong scores.  Metal (MLX) broadcasts a mask of any
        # rank correctly, so materialize to the full score shape on the CPU
        # stream only — it is free there (the scores are already dense; there is
        # no fused kernel to preserve a memory win for).
        if not query.is_metal:
            lead = list(query.shape[:-2])
            per_batch = lq * lk
            flat = 1
            for d in lead:
                flat *= int(d)
            if attn_mask.numel() not in (per_batch, flat * per_batch):
                full = lead + [lq, lk]
                while attn_mask.ndim < len(full):
                    attn_mask = attn_mask.unsqueeze(0)
                attn_mask = attn_mask.expand(*full).contiguous()

    head_dim = query.shape[-1]
    scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    if is_causal:
        # Square, unmasked and without dropout: every row keeps key 0, so
        # there is no empty row to guard and the kernel's own triangle runs.
        return _fused_attention(query, key, value, None, scale_val, True)
    out, _ = _masked_attention(query, key, value, attn_mask, scale_val, dropout_p=dropout_p)
    return out


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    r"""Repeat each key/value head ``n_rep`` times along the head axis.

    The expansion underlying grouped-query / multi-query attention: a model
    projects fewer key/value heads (``num_kv_heads``) than query heads
    (``num_heads``) — sharing each K/V head across a group of
    ``n_rep = num_heads // num_kv_heads`` query heads — which shrinks the K/V
    projection and, crucially, the K/V cache.  Before the attention scores are
    computed, each K/V head is repeated ``n_rep`` times so the head count matches
    the queries.  Head ``h`` of the result maps to source head ``h // n_rep``, so
    query head ``q`` attends to K/V head ``q // n_rep`` — the grouping GQA
    expects.  ``n_rep == 1`` (standard multi-head attention) is a no-op.

    Parameters
    ----------
    hidden_states : Tensor
        Key or value tensor of shape ``(B, num_kv_heads, T, head_dim)``.
    n_rep : int
        Repeat factor ``num_heads // num_kv_heads`` (``>= 1``).

    Returns
    -------
    Tensor
        Shape ``(B, num_kv_heads * n_rep, T, head_dim)``.

    Notes
    -----
    Equivalent to ``unsqueeze`` → ``expand`` → ``reshape`` (an
    ``interleave``-by-group, NOT a tile): the ``num_kv_heads`` heads expand to
    consecutive blocks, so the output head order is
    ``[kv0, kv0, …, kv1, kv1, …]`` with ``n_rep`` copies per block.  This matches
    the contiguous query-head grouping used by GQA models, where query heads
    ``[g·n_rep, (g+1)·n_rep)`` share key/value head ``g``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> kv = lucid.randn(1, 2, 4, 8)        # (B, num_kv_heads=2, T=4, head_dim=8)
    >>> out = F.repeat_kv(kv, 3)            # 8 query heads / 2 kv heads
    >>> out.shape
    (1, 6, 4, 8)
    >>> bool((out[:, 0] == kv[:, 0]).all().item())   # heads 0,1,2 ← kv head 0
    True
    >>> bool((out[:, 3] == kv[:, 1]).all().item())   # heads 3,4,5 ← kv head 1
    True
    """
    if n_rep == 1:
        return hidden_states
    b, h, t, d = hidden_states.shape
    expanded = hidden_states.unsqueeze(2).expand(b, h, n_rep, t, d)
    return expanded.reshape(b, h * n_rep, t, d)
