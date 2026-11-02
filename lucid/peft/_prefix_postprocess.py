"""How a prefix-tuning prompt is reshaped into a model family's own cache layout."""

from collections.abc import Callable

import lucid
from lucid._tensor.tensor import Tensor


def bloom_model_postprocess_past_key_value(
    past_key_values: tuple[Tensor, ...],
) -> tuple[tuple[Tensor, Tensor], ...]:
    r"""Reshape a prefix-tuning prompt into the cache layout of the bloom family.

    The bloom family keeps its keys as ``(batch * heads, head_dim, tokens)`` and its
    values as ``(batch * heads, tokens, head_dim)``.  This joins the prompt's tensors
    along their first axis, takes the first half of the result for keys and the second
    half for values, and reshapes each to that layout — the reference's rule, split
    point included.

    Parameters
    ----------
    past_key_values : tuple of Tensor
        The prompt's tensors, each of shape ``(n, batch, heads, tokens, head_dim)``;
        the ``n`` of all of them add up to an even number of layers' keys and values.

    Returns
    -------
    tuple of tuple of Tensor
        One ``(keys, values)`` pair per layer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import bloom_model_postprocess_past_key_value
    >>> prompt = tuple(lucid.zeros(1, 2, 4, 3, 8) for _ in range(4))  # 2 layers: keys, then values
    >>> pairs = bloom_model_postprocess_past_key_value(prompt)
    >>> len(pairs), pairs[0][0].shape, pairs[0][1].shape
    (2, (8, 8, 3), (8, 3, 8))
    """
    joined = lucid.cat(list(past_key_values))
    total_layers, batch_size, num_heads, num_tokens, head_dim = joined.shape
    half = total_layers // 2
    keys = joined[:half].transpose(2, 3).reshape(half, batch_size * num_heads, head_dim, num_tokens)
    values = joined[half:].reshape(half, batch_size * num_heads, num_tokens, head_dim)
    return tuple(zip(keys, values, strict=True))


#: A model family's ``model_type`` -> the function that turns a prefix-tuning prompt into that
#: family's cache layout.  A plain dict, as the reference's: add a family by assigning to it.
TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING: dict[
    str, Callable[[tuple[Tensor, ...]], tuple[tuple[Tensor, Tensor], ...]]
] = {"bloom": bloom_model_postprocess_past_key_value}
