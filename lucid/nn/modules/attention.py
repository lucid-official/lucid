r"""
Multi-head attention module.
"""

import math
from typing import final, override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
from lucid._dispatch import _wrap
from lucid._C import engine as _C_engine
import lucid as _lucid
import lucid.nn.init as init
from lucid.nn.functional.linear import linear
from lucid.nn.functional.attention import (
    _causal_additive_mask,
    _keep_mask_to_additive,
    _masked_attention,
    repeat_kv,
    scaled_dot_product_attention,
)

from lucid.utils.cache import Cache, EncoderDecoderCache, StaticCache

# ``Tensor`` and ``Cache`` are imported at runtime, not under TYPE_CHECKING,
# so that the annotations on ``forward`` can actually be resolved.  Under
# PEP 649 a deferred annotation is only lazy until something asks for its
# value — ``inspect.signature``, ``help()`` and ``typing.get_type_hints`` all
# do, and each raised ``NameError`` here.  ``transformer.py`` already imports
# ``Tensor`` this way, so there is no cycle to avoid.


def _pad_mask_columns(mask: Tensor | None, n: int) -> Tensor | None:
    """Append ``n`` key columns every query may attend to (``0`` / ``False``)."""
    if mask is None:
        return None
    pad = _lucid.zeros(*mask.shape[:-1], n, dtype=mask.dtype, device=mask.device)
    return _lucid.cat([mask, pad], -1)


def _to_additive_mask(mask: Tensor, float_dtype: DTypeLike) -> Tensor:
    """Convert a bool mask (True = mask out) to an additive float mask
    (-inf where True, 0 where False).  Already-float masks pass through."""
    if mask.dtype == _lucid.bool_:
        return _keep_mask_to_additive(mask.logical_not(), float_dtype)
    return mask


def _update_mha_cache(
    kh: Tensor,
    vh: Tensor,
    past_key_value: Cache,
    layer_idx: int,
    extra: tuple[bool, int, Tensor | None],
) -> tuple[Tensor, Tensor]:
    """Update ``past_key_value`` for self- or cross-attention and return ``(kh, vh)``."""
    is_cross_attention, Tq, cache_position = extra
    if is_cross_attention and isinstance(past_key_value, EncoderDecoderCache):
        kh, vh = past_key_value.cross_attention_cache.update(kh, vh, layer_idx)
        past_key_value.is_updated[layer_idx] = True
        return kh, vh
    self_cache = (
        past_key_value.self_attention_cache
        if isinstance(past_key_value, EncoderDecoderCache)
        else past_key_value
    )
    past_len = self_cache.get_seq_length(layer_idx)
    cache_kwargs: dict[str, object] = {"cache_position": cache_position}
    if isinstance(self_cache, StaticCache):
        from lucid.compile._entry.decode_step import is_compiled_decode_tracing

        if not is_compiled_decode_tracing():
            cache_kwargs["read_len"] = past_len + Tq
    return self_cache.update(kh, vh, layer_idx, cache_kwargs=cache_kwargs)


def _mha_causal_mask(
    qh: Tensor, Tq: int, Tk: int, cached: bool, extra: tuple[bool, Cache | None, Tensor | None]
) -> Tensor:
    """Build the ``(1, 1, Tq, Tk)`` additive causal mask for MHA."""
    is_cross_attention, past_key_value, cache_position = extra
    self_cache = (
        past_key_value.self_attention_cache
        if isinstance(past_key_value, EncoderDecoderCache)
        else past_key_value
    )
    if (
        cached
        and not is_cross_attention
        and isinstance(self_cache, StaticCache)
        and cache_position is not None
    ):
        from lucid._engine_scope import untraced

        with untraced():
            full_causal = _causal_additive_mask(
                self_cache.max_cache_len, Tk, 0, qh.dtype, qh.device
            )
        return full_causal.index_select(0, cache_position.reshape(-1)).reshape(1, 1, Tq, Tk)
    return _causal_additive_mask(Tq, Tk, Tk - Tq if cached else 0, qh.dtype, qh.device).reshape(
        1, 1, Tq, Tk
    )


def _kv_head_count(embed_dim: int, num_heads: int, num_kv_heads: int | None) -> int:
    """The key/value head count, once the sizes that fix it are known to fit.

    Raises:
        ValueError: ``embed_dim`` or ``num_heads`` is not positive, ``embed_dim``
            does not split into ``num_heads`` heads, or ``num_heads`` does not
            split into ``num_kv_heads`` groups.
    """
    if embed_dim <= 0 or num_heads <= 0:
        raise ValueError(
            "embed_dim and num_heads must be greater than 0, "
            f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
        )
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
    kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    if kv_heads <= 0 or num_heads % kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({kv_heads})")
    return kv_heads


class MultiheadAttention(Module):
    r"""Multi-head scaled dot-product attention.

    Implements the multi-head attention mechanism introduced in
    "Attention Is All You Need" (Vaswani et al., 2017).  Each head
    independently computes scaled dot-product attention over a
    learned linear projection of the query, key and value inputs;
    the per-head outputs are then concatenated and projected once
    more to produce the final result.

    **Scaled dot-product attention** for a single head:

    .. math::

        \text{Attention}(Q, K, V)
        = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V

    where :math:`d_k` is the per-head key dimension (``head_dim``).
    The :math:`1/\sqrt{d_k}` scaling prevents the dot-products from
    growing so large that the softmax function is pushed into regions
    with extremely small gradients.

    **Multi-head attention** uses :math:`h` parallel heads:

    .. math::

        \text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)

    .. math::

        \text{MultiHead}(Q, K, V)
        = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O

    where :math:`W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}`,
    :math:`W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}`,
    :math:`W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}`, and
    :math:`W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}` are
    learned projection matrices.

    When ``kdim`` and ``vdim`` both equal ``embed_dim``, the three
    input projections are stored as a single fused weight
    ``in_proj_weight`` of shape ``(3 * embed_dim, embed_dim)`` and
    split at runtime, which is more cache-friendly on Apple Silicon.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model, :math:`d_{\text{model}}`.
        Must be divisible by ``num_heads``.
    num_heads : int
        Number of parallel attention heads :math:`h`.
        Each head operates on a subspace of dimension
        ``head_dim = embed_dim // num_heads``.
    dropout : float, optional
        Dropout probability applied to the attention weight matrix
        during training.  Default: ``0.0``.
    bias : bool, optional
        If ``True``, learnable bias terms are added to all input and
        output projection layers.  Default: ``True``.
    add_bias_kv : bool, optional
        If ``True``, learnable bias rows ``bias_k`` and ``bias_v``
        (each of shape ``(1, 1, embed_dim)``) are appended to the
        key and value sequences along the sequence dimension before
        the attention computation.  Useful for cross-attention
        scenarios where extra context tokens are desired.
        Default: ``False``.
    add_zero_attn : bool, optional
        If ``True``, a zero-valued row is appended to the key and
        value sequences.  This can stabilise training in early steps
        by providing an "attend to nothing" option.  Default: ``False``.
    kdim : int or None, optional
        Feature dimension of the key input.  When ``None`` (default),
        falls back to ``embed_dim`` and a fused ``in_proj_weight``
        is used.
    vdim : int or None, optional
        Feature dimension of the value input.  When ``None``
        (default), falls back to ``embed_dim``.
    num_kv_heads : int or None, optional
        Number of key/value heads for **grouped-query attention** (GQA) /
        **multi-query attention** (MQA).  Must divide ``num_heads``.  ``None``
        (default) → ``num_heads`` (standard multi-head attention).  With fewer
        K/V heads the model projects a smaller key/value space — shared across
        ``num_heads // num_kv_heads`` query heads — which shrinks the K/V
        projection and, during incremental decoding, the K/V cache; each K/V head
        is repeated to match the query heads just before attention.  ``1`` is MQA
        (all query heads share one K/V head).  GQA forces the separate-projection
        layout (no fused ``in_proj_weight``).
    batch_first : bool, optional
        Controls the expected layout of all input and output tensors.

        * ``False`` (default): ``(seq_len, batch, embed_dim)``  — the
          classic sequence-first convention.
        * ``True``: ``(batch, seq_len, embed_dim)`` — more intuitive
          for most modern use-cases.

    device : DeviceLike, optional
        Device on which to allocate parameters.  ``None`` defaults to
        the current default device.
    dtype : DTypeLike, optional
        Data type for all parameters.  ``None`` defaults to the
        current default floating-point type.

    Attributes
    ----------
    embed_dim : int
        Total model dimension passed at construction.
    num_heads : int
        Number of attention (query) heads.
    num_kv_heads : int
        Number of key/value heads (``== num_heads`` for standard MHA; fewer for
        GQA/MQA).
    head_dim : int
        Per-head dimension: ``embed_dim // num_heads``.
    kdim : int
        Effective key feature dimension.
    vdim : int
        Effective value feature dimension.
    dropout : float
        Attention weight dropout probability.
    batch_first : bool
        Whether inputs are ``(batch, seq, feature)``.
    in_proj_weight : Parameter or None
        Fused ``(3 * embed_dim, embed_dim)`` projection weight used
        when ``kdim == vdim == embed_dim``.  Sliced at runtime into
        Q, K, V sub-weights.  ``None`` when using separate weights.
    q_proj_weight : Parameter or None
        Separate query projection weight ``(embed_dim, embed_dim)``.
        Non-``None`` only when ``kdim`` or ``vdim`` differs from
        ``embed_dim``.
    k_proj_weight : Parameter or None
        Separate key projection weight ``(embed_dim, kdim)``.
    v_proj_weight : Parameter or None
        Separate value projection weight ``(embed_dim, vdim)``.
    in_proj_bias : Parameter or None
        Bias for the fused input projection ``(3 * embed_dim,)``.
        ``None`` when ``bias=False``.
    out_proj_weight : Parameter
        Output projection weight ``(embed_dim, embed_dim)``.
    out_proj_bias : Parameter or None
        Output projection bias ``(embed_dim,)``.
        ``None`` when ``bias=False``.
    bias_k : Parameter or None
        Learnable key bias row ``(1, 1, embed_dim)``.
        Non-``None`` when ``add_bias_kv=True``.
    bias_v : Parameter or None
        Learnable value bias row ``(1, 1, embed_dim)``.
        Non-``None`` when ``add_bias_kv=True``.
    add_zero_attn : bool
        Whether a zero row is appended to K and V.

    Shape
    -----
    The shapes below use the following notation:

    * :math:`N` — batch size
    * :math:`L` — target (query) sequence length
    * :math:`S` — source (key / value) sequence length
    * :math:`E` — ``embed_dim``

    When ``batch_first=False`` (default):

    * ``query``: :math:`(L, N, E)`
    * ``key``: :math:`(S, N, E_k)` where :math:`E_k` = ``kdim``
    * ``value``: :math:`(S, N, E_v)` where :math:`E_v` = ``vdim``
    * Output ``attn_output``: :math:`(L, N, E)`
    * Output ``attn_weights``: :math:`(N, L, S)` when
      ``need_weights=True`` and ``average_attn_weights=True``;
      :math:`(N, h, L, S)` when ``average_attn_weights=False``.

    When ``batch_first=True``:

    * ``query``: :math:`(N, L, E)`
    * ``key`` / ``value``: :math:`(N, S, E_{k/v})`
    * Output ``attn_output``: :math:`(N, L, E)`

    Notes
    -----
    **Why scale by** :math:`1/\sqrt{d_k}`?
        As :math:`d_k` grows, the dot-products :math:`QK^\top`
        accumulate over more dimensions and their magnitude grows
        like :math:`\sqrt{d_k}` under the assumption of unit-variance
        inputs.  Without the scale factor the softmax would saturate,
        producing near-one-hot distributions and vanishingly small
        gradients.  Dividing by :math:`\sqrt{d_k}` restores
        roughly unit variance before the softmax.

    **Causal masking** (``is_causal=True``):
        An upper-triangular :math:`-\infty` mask is added to the
        score matrix so that position :math:`i` cannot attend to any
        position :math:`j > i`.  This implements the autoregressive
        constraint needed for language model decoding.

    **A query with every key masked out** (a fully padded sequence, say):
        attends to nothing.  Its output row, before ``out_proj``, is
        ``0``, its attention weights are ``0``, and no gradient flows
        through it — on every device and dtype, and whether or not
        ``need_weights`` is set.  The reference agrees on its fused path;
        its ``need_weights=True`` path returns NaN for such a row instead.

    **Fused vs. separate projections**:
        When ``kdim == vdim == embed_dim``, the Q/K/V projections
        share a single ``(3E, E)`` weight matrix.  This layout
        allows a single ``linear`` call plus a cheap ``split_at``
        on the result, which amortises kernel-launch overhead and
        improves cache locality on the MLX / Accelerate backends.

    **Checkpoint compatibility**:
        State-dicts from the reference framework store the output
        projection under the key ``out_proj.weight`` / ``out_proj.bias``
        (a sub-module named ``out_proj``).  Lucid's ``_load_from_state_dict``
        hook transparently remaps those keys to the flat
        ``out_proj_weight`` / ``out_proj_bias`` attributes used here,
        so pre-trained weights can be loaded directly.

    Examples
    --------
    **Basic self-attention** (sequence-first layout):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)
    >>> # Sequence-first: (seq_len, batch, embed_dim)
    >>> x = lucid.randn(10, 2, 64)          # 10 tokens, batch=2
    >>> out, weights = mha(x, x, x)
    >>> out.shape
    (10, 2, 64)
    >>> weights.shape                        # averaged over heads
    (2, 10, 10)

    **Cross-attention with batch_first layout and causal mask**:

    >>> mha = nn.MultiheadAttention(embed_dim=64, num_heads=8,
    ...                             batch_first=True)
    >>> q = lucid.randn(2, 6, 64)           # batch=2, 6 query tokens
    >>> kv = lucid.randn(2, 10, 64)         # 10 key/value tokens
    >>> out, _ = mha(q, kv, kv, need_weights=False)
    >>> out.shape
    (2, 6, 64)

    **Cross-modal attention with different key/value dimensions**:

    >>> mha = nn.MultiheadAttention(embed_dim=128, num_heads=4,
    ...                             kdim=64, vdim=64)
    >>> q = lucid.randn(5, 1, 128)
    >>> kv = lucid.randn(7, 1, 64)
    >>> out, weights = mha(q, kv, kv)
    >>> out.shape
    (5, 1, 128)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: DeviceLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        num_kv_heads: int | None = None,
    ) -> None:
        """Initialise the MultiheadAttention module. See the class docstring for parameter semantics."""
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        # Grouped-query / multi-query attention: project ``num_kv_heads`` <=
        # ``num_heads`` key/value heads (``None`` → standard MHA = num_heads).
        # Each K/V head is shared by ``num_heads // num_kv_heads`` query heads,
        # shrinking the K/V projection and the K/V cache.
        self.num_kv_heads: int = _kv_head_count(embed_dim, num_heads, num_kv_heads)
        self.dropout: float = dropout
        self.batch_first: bool = batch_first
        self.head_dim: int = embed_dim // num_heads
        self._kv_dim: int = self.num_kv_heads * self.head_dim  # K/V projected width
        self.kdim: int = kdim if kdim is not None else embed_dim
        self.vdim: int = vdim if vdim is not None else embed_dim
        self.add_zero_attn: bool = add_zero_attn
        # The combined ``in_proj_weight`` only applies when Q/K/V all map
        # ``embed_dim → embed_dim`` from one source; GQA (fewer K/V heads) or a
        # differing ``kdim``/``vdim`` forces the separate-projection layout.
        self._qkv_same_embed_dim: bool = (
            self.kdim == embed_dim and self.vdim == embed_dim and self.num_kv_heads == num_heads
        )

        if self._qkv_same_embed_dim:
            self.in_proj_weight: Parameter | None = Parameter(
                empty(3 * embed_dim, embed_dim, dtype=dtype, device=device)
            )
            self.q_proj_weight: Parameter | None = None
            self.k_proj_weight: Parameter | None = None
            self.v_proj_weight: Parameter | None = None
        else:
            # Separate projection weights when Q/K/V come from sources with
            # different feature dims, or under GQA where K/V project to fewer
            # heads (``_kv_dim`` < ``embed_dim``).  ``in_proj_weight`` is unset.
            self.in_proj_weight = None
            self.q_proj_weight = Parameter(empty(embed_dim, embed_dim, dtype=dtype, device=device))
            self.k_proj_weight = Parameter(
                empty(self._kv_dim, self.kdim, dtype=dtype, device=device)
            )
            self.v_proj_weight = Parameter(
                empty(self._kv_dim, self.vdim, dtype=dtype, device=device)
            )

        # Q bias is embed_dim wide; K/V biases are _kv_dim wide (== embed_dim
        # for standard MHA, so this is 3*embed_dim then).
        self.in_proj_bias: Parameter | None = (
            Parameter(empty(embed_dim + 2 * self._kv_dim, dtype=dtype, device=device))
            if bias
            else None
        )

        if add_bias_kv:
            self.bias_k: Parameter | None = Parameter(
                empty(1, 1, self._kv_dim, dtype=dtype, device=device)
            )
            self.bias_v: Parameter | None = Parameter(
                empty(1, 1, self._kv_dim, dtype=dtype, device=device)
            )
        else:
            self.bias_k = None
            self.bias_v = None

        # Output projection — kept as flat parameters (``out_proj_weight``
        # / ``out_proj_bias``) for backwards compatibility with checkpoints
        # saved by earlier Lucid versions.  External reference checkpoints
        # use ``out_proj.weight`` / ``out_proj.bias``; the load hook below
        # remaps either form.
        self.out_proj_weight: Parameter = Parameter(
            empty(embed_dim, embed_dim, dtype=dtype, device=device)
        )
        self.out_proj_bias: Parameter | None = (
            Parameter(empty(embed_dim, dtype=dtype, device=device)) if bias else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform the projections, zero the biases, Xavier-normal ``bias_k`` / ``bias_v``."""
        if self.in_proj_weight is not None:
            init.xavier_uniform_(self.in_proj_weight)
        if self.q_proj_weight is not None:
            init.xavier_uniform_(self.q_proj_weight)
            assert self.k_proj_weight is not None
            assert self.v_proj_weight is not None
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            init.zeros_(self.in_proj_bias)
        if self.out_proj_bias is not None:
            init.zeros_(self.out_proj_bias)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
            assert self.bias_v is not None
            init.xavier_normal_(self.bias_v)

    def _reset_parameters(self) -> None:
        """:meth:`reset_parameters`, under the name the reference gives it."""
        self.reset_parameters()

    # ── reference-checkpoint loading: accept ``out_proj.weight`` / ``out_proj.bias``
    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Inline-rename ``out_proj.weight`` → ``out_proj_weight`` (and bias)
        # so external checkpoints using the reference framework's MHA
        # layout load directly.  Both forms are tolerated.
        """Internal helper for the MultiheadAttention module."""
        for src_name, dst_name in (
            ("out_proj.weight", "out_proj_weight"),
            ("out_proj.bias", "out_proj_bias"),
        ):
            src_key: str = f"{prefix}{src_name}"
            dst_key: str = f"{prefix}{dst_name}"
            if src_key in state_dict and dst_key not in state_dict:
                state_dict[dst_key] = state_dict.pop(src_key)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the input projections, returning ``(q, k, v)`` each shaped
        ``(B, T*, embed_dim)``."""
        d: int = self.embed_dim
        if self._qkv_same_embed_dim:
            wt = self.in_proj_weight._impl  # type: ignore[union-attr]
            parts = _C_engine.split_at(wt, [d, 2 * d], 0)
            q_w: Tensor = _wrap(parts[0])
            k_w: Tensor = _wrap(parts[1])
            v_w: Tensor = _wrap(parts[2])
        else:
            q_w = self.q_proj_weight  # type: ignore[assignment]
            k_w = self.k_proj_weight  # type: ignore[assignment]
            v_w = self.v_proj_weight  # type: ignore[assignment]

        if self.in_proj_bias is not None:
            bt = self.in_proj_bias._impl
            # Q bias is embed_dim wide; K/V biases _kv_dim wide (GQA: < embed_dim).
            b_parts = _C_engine.split_at(bt, [d, d + self._kv_dim], 0)
            q_b: Tensor | None = _wrap(b_parts[0])
            k_b: Tensor | None = _wrap(b_parts[1])
            v_b: Tensor | None = _wrap(b_parts[2])
        else:
            q_b = k_b = v_b = None

        q: Tensor = linear(query, q_w, q_b)
        k: Tensor = linear(key, k_w, k_b)
        v: Tensor = linear(value, v_w, v_b)
        return q, k, v

    def _project_q(self, query: Tensor) -> Tensor:
        """Project only the query, ``(B, Tq, embed_dim)``.

        Cross-attention reusing an already-filled cache reads its key/value
        straight from the cache, so projecting K/V from ``memory`` would be
        wasted work (discarded immediately).  This is the query-only path.
        """
        d: int = self.embed_dim
        if self._qkv_same_embed_dim:
            q_w: Tensor = _wrap(
                _C_engine.split_at(self.in_proj_weight._impl, [d, 2 * d], 0)[0]  # type: ignore[union-attr]
            )
        else:
            q_w = self.q_proj_weight  # type: ignore[assignment]
        q_b: Tensor | None = None
        if self.in_proj_bias is not None:
            q_b = _wrap(_C_engine.split_at(self.in_proj_bias._impl, [d], 0)[0])
        return linear(query, q_w, q_b)

    def _split_heads(
        self, x: Tensor, batch_size: int, seq_len: int, heads: int | None = None
    ) -> Tensor:
        """Reshape ``(B, T, heads*head_dim)`` → ``(B, heads, T, head_dim)``.

        ``heads`` defaults to ``num_heads`` (query); pass ``num_kv_heads`` for the
        key/value projections under grouped-query attention.
        """
        h = self.num_heads if heads is None else heads
        return x.reshape(batch_size, seq_len, h, self.head_dim).permute([0, 2, 1, 3])

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Inverse of :meth:`_split_heads`."""
        return x.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, self.embed_dim)

    def _build_attn_mask(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        batch_size: int,
        target_len: int,
        source_len: int,
        float_dtype: DTypeLike,
    ) -> Tensor | None:
        """Combine ``attn_mask`` and ``key_padding_mask`` into a single
        ``(B, H, T, S)`` additive float mask suitable for SDPA."""
        merged: Tensor | None = None

        if attn_mask is not None:
            am: Tensor = _to_additive_mask(attn_mask, float_dtype)
            if am.ndim == 2:
                am = am.reshape(1, 1, target_len, source_len)
            elif am.ndim == 3:
                # ``(B*H, T, S)`` is the reference convention for per-head masks.
                am = am.reshape(batch_size, self.num_heads, target_len, source_len)
            elif am.ndim == 4:
                pass
            else:
                raise ValueError(f"attn_mask must be 2-D, 3-D, or 4-D; got {am.ndim}-D")
            merged = am

        if key_padding_mask is not None:
            kpm: Tensor = _to_additive_mask(key_padding_mask, float_dtype)
            # (B, S) → (B, 1, 1, S) so it broadcasts over heads & target length.
            kpm = kpm.reshape(batch_size, 1, 1, source_len)
            merged = kpm if merged is None else merged + kpm

        return merged

    @override
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        *,
        past_key_value: Cache | None = None,
        layer_idx: int = 0,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
        is_cross_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # Internally operate in (B, T, E) layout regardless of batch_first.
        """Run the forward pass of the module.

        Parameters
        ----------
        query : Tensor
            See the class docstring.
        key : Tensor
            See the class docstring.
        value : Tensor
            See the class docstring.
        key_padding_mask : Tensor
            See the class docstring.
        need_weights : Tensor
            See the class docstring.
        attn_mask : Tensor
            See the class docstring.
        average_attn_weights : Tensor
            See the class docstring.
        is_causal : Tensor
            See the class docstring.
        past_key_value : Cache or None, optional, keyword-only
            Key/value cache for incremental decoding.  ``None`` disables
            caching.
        layer_idx : int, optional, keyword-only, default=0
            Index of this attention layer within the cache.
        cache_position : Tensor or None, optional, keyword-only
            Absolute positions of the query tokens; accepted for API parity.
        use_cache : bool, optional, keyword-only, default=False
            Read/write ``past_key_value`` when ``True``.
        is_cross_attention : bool, optional, keyword-only, default=False
            ``True`` for decoder cross-attention (keys/values from the encoder
            memory, cached once); ``False`` for self-attention (cache grows).

        Returns
        -------
        Tensor
            Output tensor; refer to the class docstring for the exact shape.
        """
        is_batched = query.ndim == 3
        if query.ndim not in (2, 3):
            raise ValueError(
                "MultiheadAttention: expected query to be 2-D or 3-D, " f"got {query.ndim}-D"
            )
        if is_batched:
            if key.ndim != 3 or value.ndim != 3:
                raise ValueError(
                    "MultiheadAttention: for 3-D (batched) query, expected "
                    f"key and value to be 3-D, got {key.ndim}-D and {value.ndim}-D"
                )
            if key_padding_mask is not None and key_padding_mask.ndim != 2:
                raise ValueError(
                    "MultiheadAttention: for 3-D (batched) query, expected "
                    f"key_padding_mask to be 2-D, got {key_padding_mask.ndim}-D"
                )
            if not self.batch_first:
                query = query.permute([1, 0, 2])
                key = key.permute([1, 0, 2])
                value = value.permute([1, 0, 2])
        else:
            if key.ndim != 2 or value.ndim != 2:
                raise ValueError(
                    "MultiheadAttention: for 2-D (unbatched) query, expected "
                    f"key and value to be 2-D, got {key.ndim}-D and {value.ndim}-D"
                )
            if key_padding_mask is not None:
                if key_padding_mask.ndim != 1:
                    raise ValueError(
                        "MultiheadAttention: for 2-D (unbatched) query, expected "
                        f"key_padding_mask to be 1-D, got {key_padding_mask.ndim}-D"
                    )
                key_padding_mask = key_padding_mask.unsqueeze(0)
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        B: int = query.shape[0]
        Tq: int = query.shape[1]

        # Keys/values are laid out (B, num_heads, T, head_dim).  Cross-attention
        # that reuses an already-filled cache reads K/V straight from it, so it
        # needs ONLY the query projection — skip the (otherwise discarded) memory
        # K/V projection (saves it eagerly; drops it from the compiled graph).
        cross_reuse: bool = (
            use_cache
            and is_cross_attention
            and isinstance(past_key_value, EncoderDecoderCache)
            and past_key_value.is_updated.get(layer_idx, False)
        )
        qh: Tensor
        kh: Tensor
        vh: Tensor
        Tk: int
        if cross_reuse:
            assert isinstance(past_key_value, EncoderDecoderCache)
            qh = self._split_heads(self._project_q(query), B, Tq)
            kh, vh = past_key_value.cross_attention_cache[layer_idx]
            Tk = int(kh.shape[2])
        else:
            q, k, v = self._project_qkv(query, key, value)

            # ── add_bias_kv: prepend learnable bias rows to K / V ───────────
            # K/V are ``_kv_dim`` wide (== embed_dim for standard MHA).
            if self.bias_k is not None:
                assert self.bias_v is not None
                bk: Tensor = self.bias_k.expand(B, 1, self._kv_dim)
                bv: Tensor = self.bias_v.expand(B, 1, self._kv_dim)
                k = _lucid.cat([k, bk], 1)
                v = _lucid.cat([v, bv], 1)

            # ── add_zero_attn: append a zero row to K / V ───────────────────
            if self.add_zero_attn:
                zero_kv: Tensor = _lucid.zeros(B, 1, self._kv_dim, dtype=k.dtype, device=k.device)
                k = _lucid.cat([k, zero_kv], 1)
                v = _lucid.cat([v, zero_kv], 1)

            # The rows appended above are keys every query may attend to.
            # Widen the caller's masks to cover them, as the reference does:
            # left alone, a mask no longer lines up with K, and the reshape in
            # _build_attn_mask failed for every mask given with either option.
            # Only what this call appended is padded, so a mask written
            # against a KV cache keeps its meaning.
            appended = int(self.bias_k is not None) + int(self.add_zero_attn)
            if appended:
                attn_mask = _pad_mask_columns(attn_mask, appended)
                key_padding_mask = _pad_mask_columns(key_padding_mask, appended)

            Tk = k.shape[1]
            qh = self._split_heads(q, B, Tq)
            # K/V split into num_kv_heads (GQA); repeated to num_heads below.
            kh = self._split_heads(k, B, Tk, self.num_kv_heads)
            vh = self._split_heads(v, B, Tk, self.num_kv_heads)

            # ── KV cache ────────────────────────────────────────────────────
            # Self-attention grows the cache; cross-attention fills it once from
            # the (constant) encoder memory (the reuse path is handled above).
            if use_cache and past_key_value is not None:
                kh, vh = _update_mha_cache(
                    kh, vh, past_key_value, layer_idx, (is_cross_attention, Tq, cache_position)
                )
                Tk = int(kh.shape[2])

        # GQA/MQA: the cache stores the smaller ``num_kv_heads`` K/V; repeat each
        # head to match the ``num_heads`` query heads before attention.  No-op for
        # standard MHA (``num_kv_heads == num_heads``).
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            kh = repeat_kv(kh, n_rep)
            vh = repeat_kv(vh, n_rep)

        merged_mask: Tensor | None = self._build_attn_mask(
            attn_mask, key_padding_mask, B, Tq, Tk, qh.dtype
        )

        dropout_p: float = self.dropout if self.training else 0.0
        explicit: bool = need_weights or dropout_p > 0.0
        cached: bool = use_cache and past_key_value is not None
        if is_causal and (cached or explicit):
            causal_add = _mha_causal_mask(
                qh, Tq, Tk, cached, (is_cross_attention, past_key_value, cache_position)
            )
            merged_mask = causal_add if merged_mask is None else merged_mask + causal_add
            is_causal = False

        attn_weights: Tensor | None
        if explicit:
            attn_out, attn_weights = _masked_attention(
                qh,
                kh,
                vh,
                merged_mask,
                1.0 / math.sqrt(qh.shape[-1]),
                dropout_p=dropout_p,
                need_weights=need_weights,
            )
        else:
            attn_out = scaled_dot_product_attention(
                qh, kh, vh, attn_mask=merged_mask, is_causal=is_causal
            )
            attn_weights = None

        # (B, H, Tq, head_dim) → (B, Tq, embed_dim)
        out: Tensor = self._merge_heads(attn_out, B, Tq)

        out = linear(out, self.out_proj_weight, self.out_proj_bias)

        if not is_batched:
            out = out.squeeze(0)
        elif not self.batch_first:
            out = out.permute([1, 0, 2])

        if need_weights and attn_weights is not None:
            if average_attn_weights:
                # (B, H, Tq, Tk) → (B, Tq, Tk)
                attn_weights = attn_weights.mean(dim=1)
            if not is_batched:
                attn_weights = attn_weights.squeeze(0)

        return out, attn_weights

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s: str = (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
        if self.num_kv_heads != self.num_heads:
            s += f", num_kv_heads={self.num_kv_heads}"
        if self.kdim != self.embed_dim:
            s += f", kdim={self.kdim}"
        if self.vdim != self.embed_dim:
            s += f", vdim={self.vdim}"
        if self.bias_k is not None:
            s += ", add_bias_kv=True"
        if self.add_zero_attn:
            s += ", add_zero_attn=True"
        return s


@final
class GroupedQueryAttention(MultiheadAttention):
    r"""Grouped-query attention — multi-head attention with fewer key/value heads.

    A thin :class:`MultiheadAttention` with ``num_kv_heads`` promoted to a
    **required** argument (the parameter that defines GQA; on the base class it is
    optional and defaults to ``num_heads``).  The ``num_kv_heads`` key/value heads
    are split into groups, each shared by ``num_heads // num_kv_heads`` query heads,
    so the model projects a smaller key/value space and — during incremental
    decoding — carries a smaller K/V cache.  This is the attention used by Llama 2 /
    3, Mistral, Qwen, and Gemma.  Exactly equivalent to
    ``MultiheadAttention(..., num_kv_heads=num_kv_heads)`` (forward, shapes, and the
    KV-cache contract are inherited verbatim); this subclass only reshapes the
    constructor for discoverability.

    Parameters
    ----------
    embed_dim : int
        Total model dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads — the defining grouped-query parameter.  Must
        divide ``num_heads``.  ``1`` is multi-query attention
        (:class:`MultiQueryAttention`); passing ``num_heads`` degenerates to
        standard multi-head attention.
    dropout : float, optional, default=0.0
        Dropout probability applied to the attention weights.
    bias : bool, optional, default=True
        If ``True``, add a learnable bias to the input and output projections.
    add_bias_kv : bool, optional, default=False
        If ``True``, prepend a learnable bias row to the keys and values.
    add_zero_attn : bool, optional, default=False
        If ``True``, append a zero row to the keys and values so a query can attend
        to "nothing".
    kdim : int or None, optional
        Feature dimension of the key input; defaults to ``embed_dim``.
    vdim : int or None, optional
        Feature dimension of the value input; defaults to ``embed_dim``.
    batch_first : bool, optional, default=False
        If ``True``, inputs/outputs are ``(batch, seq, feature)``; otherwise
        ``(seq, batch, feature)``.
    device : DeviceLike, optional
        Device on which the parameters are allocated.
    dtype : DTypeLike, optional
        Data type of the parameters.

    Notes
    -----
    The key/value projections are sized ``(num_kv_heads * head_dim, …)`` instead of
    ``(embed_dim, …)`` (so ``k_proj_weight`` / ``v_proj_weight`` and the K/V cache
    are smaller — the cache shrink is the main inference win: less memory bandwidth
    per decode step).  Just before the scores are formed, each K/V head is repeated
    ``num_heads // num_kv_heads`` times via
    :func:`~lucid.nn.functional.repeat_kv` so the head counts match; the output is
    bit-identical to standard attention with K/V heads tied within each group.

    Reference: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, and Sanghai,
    *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
    Checkpoints"*, EMNLP 2023 (arXiv:2305.13245).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> gqa = nn.GroupedQueryAttention(embed_dim=512, num_heads=8, num_kv_heads=2,
    ...                                batch_first=True)
    >>> x = lucid.randn(1, 10, 512)
    >>> out, _ = gqa(x, x, x, need_weights=False)
    >>> out.shape
    (1, 10, 512)
    >>> tuple(gqa.k_proj_weight.shape)   # 2 K/V heads × head_dim 64 = 128
    (128, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise grouped-query attention; see the class docstring."""
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            num_kv_heads=num_kv_heads,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )


@final
class MultiQueryAttention(MultiheadAttention):
    r"""Multi-query attention — multi-head attention with a single key/value head.

    A thin :class:`MultiheadAttention` that fixes ``num_kv_heads = 1`` (so it takes
    no ``num_kv_heads`` argument): all ``num_heads`` query heads share one key/value
    head — the extreme of grouped-query attention.  This minimises the key/value
    projection and, decisively, the K/V cache (one head's worth of K/V per layer
    instead of ``num_heads``), trading some quality for the smallest decode-memory
    footprint.  Exactly equivalent to ``MultiheadAttention(..., num_kv_heads=1)``
    (forward, shapes, and the KV-cache contract are inherited verbatim); this
    subclass only reshapes the constructor for discoverability.

    Parameters
    ----------
    embed_dim : int
        Total model dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of query heads (the single K/V head is shared across all of them).
    dropout : float, optional, default=0.0
        Dropout probability applied to the attention weights.
    bias : bool, optional, default=True
        If ``True``, add a learnable bias to the input and output projections.
    add_bias_kv : bool, optional, default=False
        If ``True``, prepend a learnable bias row to the keys and values.
    add_zero_attn : bool, optional, default=False
        If ``True``, append a zero row to the keys and values so a query can attend
        to "nothing".
    kdim : int or None, optional
        Feature dimension of the key input; defaults to ``embed_dim``.
    vdim : int or None, optional
        Feature dimension of the value input; defaults to ``embed_dim``.
    batch_first : bool, optional, default=False
        If ``True``, inputs/outputs are ``(batch, seq, feature)``; otherwise
        ``(seq, batch, feature)``.
    device : DeviceLike, optional
        Device on which the parameters are allocated.
    dtype : DTypeLike, optional
        Data type of the parameters.

    Notes
    -----
    The single key/value head is projected once (``k_proj_weight`` /
    ``v_proj_weight`` of shape ``(head_dim, …)``) and repeated ``num_heads`` times
    via :func:`~lucid.nn.functional.repeat_kv` before the attention scores, so the
    output is bit-identical to standard attention whose K/V heads are all tied.

    Reference: Shazeer, *"Fast Transformer Decoding: One Write-Head is All You
    Need"*, 2019 (arXiv:1911.02150).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> mqa = nn.MultiQueryAttention(embed_dim=512, num_heads=8, batch_first=True)
    >>> mqa.num_kv_heads
    1
    >>> tuple(mqa.k_proj_weight.shape)   # 1 K/V head × head_dim 64
    (64, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise multi-query attention (``num_kv_heads = 1``)."""
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            num_kv_heads=1,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
