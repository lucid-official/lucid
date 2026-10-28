"""Text-model helpers shared across text families.

Every transformer-LM family in :mod:`lucid.models.text` (BERT, GPT, GPT-2,
RoFormer, and future T5 / LLaMA / Mistral) needs the same two pieces of
boilerplate:

    * Activation dispatch — map a config-supplied string (one of the
      :data:`lucid.models.text.TextActivation` literals) to the correct
      :mod:`lucid.nn.functional` call.
    * Attention-mask normalisation — turn a ``(B, T)`` 0/1 padding mask
      into the additive ``(B, 1, 1, T)`` form that scaled-dot-product
      attention expects, with masked positions zeroed via ``-1e4`` bias.

Keeping them here avoids byte-identical duplication across every family
file and gives future text models a single canonical implementation.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.utils.cache import Cache, StaticCache

__all__ = [
    "text_activation",
    "extended_attention_mask",
    "token_at",
    "last_kept_token",
    "init_text_module_weights",
    "init_text_weights",
    "update_causal_kv_and_bias",
    "resolve_causal_positions",
]


def text_activation(name: str, x: Tensor) -> Tensor:
    """Apply the activation referenced by a :data:`TextActivation` literal.

    Args:
        name: One of ``"gelu"`` / ``"gelu_new"`` / ``"relu"`` / ``"silu"``
            / ``"swish"``.  Comes from a text config's ``hidden_act`` field.
        x:    Input tensor; activation is elementwise.

    Returns:
        Activated tensor of the same shape as ``x``.

    Raises:
        ValueError: If ``name`` is not a supported activation alias.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._text import text_activation
    >>> [round(v, 4) for v in text_activation("gelu", lucid.tensor([-1.0, 1.0])).tolist()]
    [-0.1587, 0.8413]

    Unlike ReLU this leaves a little of the negative side alive, which is
    what transformer stacks were tuned against.
    """
    # Divergence, deliberate: ``"gelu"`` is the exact erf form here, which is
    # what the checkpoint-publishing reference implementation does — but BERT's
    # original release used the tanh approximation under that same name.  The
    # two references disagree, so no single mapping satisfies both; we follow
    # the one the weights were converted from.  The gap is bounded: over
    # x in [-6, 6] the two forms differ by at most 4.7e-4 absolute
    # (1.3e-4 relative to |x| + 1), well inside float32 activation noise.
    # Configs that need the original elementwise curve can ask for it by name
    # with ``hidden_act="gelu_new"``.
    if name == "gelu":
        return F.gelu(x, approximate="none")
    if name == "gelu_new":
        return F.gelu(x, approximate="tanh")
    if name == "relu":
        return F.relu(x)
    if name in ("silu", "swish"):
        return F.silu(x)
    raise ValueError(f"Unsupported activation {name!r}")


def extended_attention_mask(
    attention_mask: Tensor | None, input_shape: tuple[int, ...], key_length: int | None = None
) -> Tensor | None:
    """Normalise a padding mask to the additive ``(B, 1, 1, T)`` form.

    Scaled-dot-product attention adds the resulting mask to the
    ``(B, H, T, T)`` score tensor; masked positions get ``-1e4`` added so
    that softmax drives their probabilities to ~0 without producing NaNs.

    Args:
        attention_mask: ``(B, T)`` integer / float mask with 1 for "attend"
            and 0 for "ignore", or already pre-broadcast at higher rank.
            ``None`` skips masking entirely.
        input_shape: ``(B, T, …)`` — the leading dims of the model input;
            only ``(B, T)`` are used.
        key_length: Number of *key* positions the scores span.  With a KV
            cache the queries are the ``T`` new tokens but the keys cover
            ``past_len + T`` positions, so the padding mask must be that
            wide.  Defaults to ``T`` (no cache).

    Returns:
        ``(B, 1, 1, key_length)`` additive mask, or ``None`` when no mask
        supplied.

    Raises:
        ValueError: If a rank-2 mask's width does not match ``key_length``.
            A mask covering only the new tokens cannot describe which of the
            *cached* positions were padding, and silently treating the past
            as all-real would attend to padded history.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._text import extended_attention_mask
    >>> extended_attention_mask(lucid.ones(2, 5), (2, 5)).shape
    (2, 1, 1, 5)

    Two axes are inserted so the mask broadcasts across heads and query
    positions. The values become 0 and a large negative rather than 1 and
    0, because this is added to the scores before the softmax, not
    multiplied after.
    """
    if attention_mask is None:
        return None
    B, T = input_shape[0], input_shape[1]
    key_len = T if key_length is None else key_length
    if attention_mask.ndim == 2:
        given = int(attention_mask.shape[1])
        if given != key_len:
            hint = (
                " With a KV cache the mask must cover the cached history too "
                f"(past + new = {key_len})."
                if key_len != T
                else ""
            )
            raise ValueError(
                f"attention_mask has width {given} but the attention scores "
                f"span {key_len} key positions.{hint}"
            )
        mask = attention_mask.reshape(B, 1, 1, key_len)
    elif attention_mask.ndim == 3:
        # Caller pre-broadcast over heads already.
        mask = attention_mask.unsqueeze(1)
    else:
        mask = attention_mask
    return (1.0 - mask.float()) * -1e4


def token_at(hidden: Tensor, positions: Tensor) -> Tensor:
    """The hidden state at ``positions`` along the sequence axis, per row.

    Args:
        hidden:    ``(*lead, T, H)`` hidden states.
        positions: ``(*lead)`` integer positions into ``T``.

    Returns:
        ``(*lead, H)`` — one hidden state per leading index.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._text import token_at
    >>> hidden = lucid.arange(12).reshape(2, 3, 2).float()
    >>> token_at(hidden, lucid.tensor([2, 0])).tolist()
    [[4.0, 5.0], [6.0, 7.0]]

    A gather rather than an index read on the host: the position stays a
    tensor, so a compiled or exported graph picks it from each call's
    inputs, and the gradient reaches the gathered rows.
    """
    lead = tuple(int(d) for d in positions.shape)
    index = positions.long().reshape(*lead, 1, 1)
    picked = lucid.take_along_dim(hidden, index, dim=-2)
    return picked.reshape(*lead, int(hidden.shape[-1]))


def last_kept_token(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
    """The hidden state at the last position ``attention_mask`` keeps.

    Args:
        hidden:         ``(B, T, H)`` hidden states.
        attention_mask: ``(B, T)`` padding mask, a value above 0.5 kept;
            ``None`` means every position is real.

    Returns:
        ``(B, H)``.  A row whose mask keeps nothing pools position 0.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._text import last_kept_token
    >>> hidden = lucid.arange(8).reshape(2, 4, 1).float()
    >>> mask = lucid.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    >>> last_kept_token(hidden, mask).tolist()
    [[3.0], [5.0]]

    The last kept position, not ``count - 1``: under left padding
    (``[0, 0, 1, 1]``) the count says index 1, which is a pad slot.
    """
    if attention_mask is None:
        return hidden[:, -1, :]
    T = int(attention_mask.shape[-1])
    kept = (attention_mask.float() > 0.5).long()
    positions = lucid.arange(T, device=attention_mask.device)
    # Unkept slots score 0, so an all-pad row lands on position 0.
    return token_at(hidden, (kept * positions).max(dim=-1).values)


def init_text_module_weights(
    module: nn.Module, std: float, *, reset_layer_norm: bool = False
) -> None:
    """Initialise a single Linear, Embedding, or LayerNorm module in place.

    Args:
        module: Submodule to initialise (`nn.Linear`, `nn.Embedding`, or
            `nn.LayerNorm`).
        std: Standard deviation for the Gaussian weight initialisation.
        reset_layer_norm: When ``True``, also reset `nn.LayerNorm` scale and
            bias parameters to ones and zeros.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            with lucid.no_grad():
                module.weight[module.padding_idx] = lucid.zeros(
                    (int(module.weight.shape[1]),), device=module.weight.device.type
                )
    elif reset_layer_norm and isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_text_weights(model: nn.Module, std: float) -> None:
    """Initialise every Linear and Embedding layer in ``model`` from ``N(0, std^2)``.

    Args:
        model: Root transformer module whose submodules are traversed.
        std: Standard deviation for the Gaussian weight initialisation.
    """
    for module in model.modules():
        init_text_module_weights(module, std)


def update_causal_kv_and_bias(
    k: Tensor,
    v: Tensor,
    causal_mask: Tensor,
    attention_mask: Tensor | None,
    cache_ctx: tuple[Cache | None, int, Tensor | None, int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Update ``past_key_value`` (if any) and build the causal + padding bias."""
    past_key_value, layer_idx, cache_position, T = cache_ctx
    past_len = 0
    if past_key_value is not None:
        past_len = past_key_value.get_seq_length(layer_idx)
        cache_kwargs: dict[str, object] = {"cache_position": cache_position}
        if isinstance(past_key_value, StaticCache):
            from lucid.compile._entry.decode_step import is_compiled_decode_tracing

            if not is_compiled_decode_tracing():
                cache_kwargs["read_len"] = past_len + T
        k, v = past_key_value.update(k, v, layer_idx, cache_kwargs=cache_kwargs)
    t_total = int(k.shape[2])
    if isinstance(past_key_value, StaticCache) and cache_position is not None:
        causal = causal_mask.index_select(2, cache_position)[:, :, :, :t_total]
    else:
        causal = causal_mask[:, :, past_len : past_len + T, :t_total]
    bias: Tensor = (1.0 - causal) * -1e4
    if attention_mask is not None:
        mask_len = int(attention_mask.shape[-1])
        pad_mask = (
            F.pad(attention_mask, (0, t_total - mask_len)) if mask_len < t_total else attention_mask
        )
        bias = bias + pad_mask
    return k, v, bias


def resolve_causal_positions(
    input_ids: Tensor,
    position_ids: Tensor,
    max_pos: int,
    past_key_values: Cache | None,
    cache_position: Tensor | None,
) -> tuple[int, Tensor, Tensor]:
    """Validate sequence length and return ``(past_len, cache_position, pos_ids)``."""
    T = int(input_ids.shape[1])
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    if past_len + T > max_pos:
        raise ValueError(
            f"Sequence length {past_len + T} exceeds max_position_embeddings {max_pos}"
        )
    if cache_position is None:
        cache_position = lucid.arange(past_len, past_len + T, device=input_ids.device.type).long()
    pos_ids = (
        cache_position.reshape(1, T)
        if isinstance(past_key_values, StaticCache)
        else position_ids[:, past_len : past_len + T]
    )
    return past_len, cache_position, pos_ids
