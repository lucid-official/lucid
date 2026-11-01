"""What CPT does with the batch's labels and token types: which positions are labels, and how they are weighted (private).

Every token of a CPT call has a *type*, read modulo 4: an output token (type 4, 8, ...) is a
label, anything else is context.  The prompt's own tokens carry the types the config gives; the
input's are the caller's ``input_type_mask`` (all outputs by default), shifted past the prompt's
largest type so the two do not collide — as the reference does.
"""

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners.cpt._config import CPTConfig

#: The label the loss skips.
_IGNORE = -100
#: A type that is a multiple of this (and positive) marks an output token.
_OUTPUT_PERIOD = 4


def _is_output(types: Tensor) -> Tensor:
    """Which positions of ``types`` are output tokens: a positive multiple of four."""
    return (types > 0) & (types % _OUTPUT_PERIOD == 0)


def _call_types(
    config: CPTConfig, input_type_mask: Tensor | None, batch: int, tokens: int, like: Tensor
) -> Tensor:
    """The type of each position of the prompt followed by the input: ``(batch, prompt + tokens)``.

    The caller's mask is not changed; its positive entries are shifted up by the largest type of
    the prompt.
    """
    prompt = lucid.tensor(config.cpt_tokens_type_mask, dtype=lucid.int64, device=like.device)
    if input_type_mask is None:
        given = lucid.full((batch, tokens), _OUTPUT_PERIOD, dtype=lucid.int64, device=like.device)
    else:
        if tuple(input_type_mask.shape) != (batch, tokens):
            raise ValueError(
                f"input_type_mask is {tuple(input_type_mask.shape)}, "
                f"but the labels are {(batch, tokens)}"
            )
        given = input_type_mask.to(like.device).long()
    shifted = lucid.where(given > 0, given + int(prompt.max().item()), given)
    return lucid.cat([prompt.reshape(1, -1).repeat(batch, 1), shifted], dim=1)


def _call_labels(config: CPTConfig, labels: Tensor, types: Tensor) -> Tensor:
    """The labels of the prompt followed by the input, with every non-output position skipped."""
    batch = int(labels.shape[0])
    ids = lucid.tensor(config.cpt_token_ids, dtype=lucid.int64, device=labels.device)
    joined = lucid.cat([ids.reshape(1, -1).repeat(batch, 1), labels.long()], dim=1)
    return joined.masked_fill(~_is_output(types), _IGNORE)


def _decay_weights(types: Tensor, factor: float) -> Tensor:
    """Per position, ``factor`` to the number of output segments of its row that come after it.

    An output segment is the positions sharing one output type; the last segment of a row has
    weight one, the one before it ``factor``, and so on.  Positions of no segment weigh one.  A
    segment is counted once, at its first position, by comparing every pair of positions.
    """
    tokens = int(types.shape[1])
    equal = types.unsqueeze(2) == types.unsqueeze(1)  # (batch, p, q): same type
    earlier = lucid.tril(lucid.ones(tokens, tokens, device=types.device), diagonal=-1) > 0
    first = ~(equal & earlier).any(dim=2)  # q is the first of its type in the row
    counted = _is_output(types) & first  # (batch, q): one flag per segment
    later = types.unsqueeze(1) > types.unsqueeze(2)  # (batch, p, q): q's type is above p's
    segments_after = (later & counted.unsqueeze(1)).sum(dim=2)
    base = lucid.full(tuple(segments_after.shape), factor, dtype=lucid.float32, device=types.device)
    return lucid.pow(base, segments_after.float())


def _cpt_loss(logits: Tensor, labels: Tensor, types: Tensor, config: CPTConfig) -> Tensor:
    """Next-token cross-entropy over the labelled positions, each weighted, averaged over them.

    Raises
    ------
    ValueError
        No position of the call is a label (the mean would be undefined).
    """
    shifted_labels = labels[:, 1:]
    batch, length = int(shifted_labels.shape[0]), int(shifted_labels.shape[1])
    vocab = int(logits.shape[-1])
    per_token = F.cross_entropy(
        logits[:, :-1].reshape(batch * length, vocab),
        shifted_labels.reshape(batch * length),
        ignore_index=_IGNORE,
        reduction="none",
    ).reshape(batch, length)
    labelled = shifted_labels != _IGNORE
    count = int(labelled.sum().item())
    if count == 0:
        raise ValueError(
            "no position of the call is an output token (a positive multiple of 4 in the type "
            "mask), so the CPT loss has nothing to average"
        )
    weight = labelled.to(per_token.dtype)
    if config.opt_weighted_loss_type == "decay":
        weight = weight * _decay_weights(types[:, 1:], config.opt_loss_decay_factor).to(
            per_token.dtype
        )
    return (per_token * weight).sum() / count
