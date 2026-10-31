"""Where a prompt enters a call: forward pre-hooks that rewrite one module's arguments.

A prompt-learning adapter adds no layer to the model.  It is applied by a
*site* — a forward pre-hook on one module of the model that returns that
module's arguments with the prompt in them: input embeddings with the
virtual tokens in front, a key/value cache that already holds the prefix,
an attention mask and labels as wide as the longer sequence.  Because the hook
sits on the module the model's own code calls, ``model(...)`` and the
model's ``generate`` (which never goes through any wrapper's ``forward``)
see the same prompt, and an adapter that is disabled or deleted is a hook
that finds nothing to apply.

A site is told what to apply by ``_Applied`` — the active adapter's config and
encoder — and decides from the arguments whether the prompt is already in
the call: a decoding step that reads a cache which holds the prompt gets
none.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final, Protocol, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _PromptEncoder
from lucid.utils.cache import Cache, DynamicCache, EncoderDecoderCache

type _Call = dict[str, object]

#: The label the loss skips; the positions of the virtual tokens get it.
_IGNORE: Final = -100


@dataclass(frozen=True, slots=True)
class _Applied:
    """The adapter in use: its config, and the encoder that makes its prompt."""

    config: PromptLearningConfig
    encoder: _PromptEncoder
    #: The task of each batch row, for a prompt that is made per task.
    tasks: Tensor | None = None

    @property
    def virtual(self) -> int:
        """How many positions the prompt adds in front of the input."""
        assert self.config.num_virtual_tokens is not None  # the tuner resolved it
        return self.config.num_virtual_tokens

    def embeddings(self, batch: int, dtype: lucid.dtype) -> Tensor:
        """The prompt as ``(batch, rows, token_dim)`` rows to put before input embeddings."""
        return self.encoder.prompts(batch, self.tasks).to(dtype)

    def key_values(self, batch: int, dtype: lucid.dtype) -> list[tuple[Tensor, Tensor]]:
        """The prefix as one ``(key, value)`` pair per layer, each ``(batch, heads, virtual, head_dim)``."""
        config = self.config
        assert config.num_attention_heads is not None and config.token_dim is not None
        assert config.num_layers is not None
        heads, virtual = config.num_attention_heads, self.virtual
        rows = self.encoder.prompts(batch, self.tasks)
        rows = rows.reshape(batch, virtual, config.num_layers * 2, heads, config.token_dim // heads)
        planes = rows.permute(2, 0, 3, 1, 4).to(dtype).unbind(0)
        return [(planes[i], planes[i + 1]) for i in range(0, len(planes), 2)]


class _Prompting(Protocol):
    """What a site asks of the tuner that installed it."""

    def _applying(self) -> _Applied | None: ...

    def _model_dtype(self) -> lucid.dtype: ...


def _named_arguments(module: nn.Module, args: tuple[object, ...], kwargs: _Call) -> _Call:
    """The call as ``{parameter name: value}``, whatever mix of positional and keyword it was."""
    return dict(inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments)


def _widen(mask: object, virtual: int) -> object:
    """A 2-D padding mask with ``virtual`` kept columns in front; anything else as it is."""
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        return mask
    front = lucid.ones((int(mask.shape[0]), virtual), dtype=mask.dtype, device=mask.device.type)
    return lucid.cat([front, mask], dim=1)


def _pad(rows: Tensor, virtual: int, fill: float) -> Tensor:
    """``rows`` ((B, T) ids) with ``virtual`` columns of ``fill`` in front."""
    block = lucid.full(
        (int(rows.shape[0]), virtual), fill, dtype=rows.dtype, device=rows.device.type
    )
    return lucid.cat([block, rows], dim=1)


def _batch_of(call: _Call, names: tuple[str, ...]) -> int:
    for name in names:
        value = call.get(name)
        if isinstance(value, Tensor):
            return int(value.shape[0])
    raise ValueError(
        f"a prompt-learning adapter needs one of {', '.join(names)} to size the prompt"
    )


def _holds_prompt(cache: object) -> bool:
    """Whether ``cache`` already has entries — the prompt went in on an earlier step."""
    return isinstance(cache, Cache) and cache.get_seq_length() > 0


def _put_prefix(cache: Cache, pairs: list[tuple[Tensor, Tensor]], virtual: int) -> None:
    """Write the prefix keys and values into an empty ``cache``, layer by layer."""
    positions = lucid.arange(virtual, device=pairs[0][0].device.type).long()
    for layer, (key, value) in enumerate(pairs):
        cache.update(key, value, layer, cache_kwargs={"cache_position": positions})


class _Hook(ABC):
    """A forward pre-hook that applies the active adapter's prompt to one module's call."""

    __slots__ = ("_tuner",)

    def __init__(self, tuner: _Prompting) -> None:
        self._tuner = tuner

    def __call__(
        self, module: object, args: tuple[object, ...], kwargs: _Call
    ) -> tuple[tuple[object, ...], _Call] | None:
        applied = self._tuner._applying()
        if applied is None:
            return None
        assert isinstance(module, nn.Module)  # a hook is called by its module
        return self.apply(applied, module, args, kwargs)

    @abstractmethod
    def apply(
        self, applied: _Applied, module: nn.Module, args: tuple[object, ...], kwargs: _Call
    ) -> tuple[tuple[object, ...], _Call]:
        """The arguments ``module`` is called with instead, the prompt in them."""


class _Site(_Hook):
    """A hook that rewrites the module's arguments by their names (positional or keyword alike)."""

    __slots__ = ()

    @override
    def apply(
        self, applied: _Applied, module: nn.Module, args: tuple[object, ...], kwargs: _Call
    ) -> tuple[tuple[object, ...], _Call]:
        call = _named_arguments(module, args, kwargs)
        self.rewrite(applied, call)
        return (), call

    @abstractmethod
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        """Change ``call`` — the module's arguments by name — so the prompt is in it."""


class _RequestSite(_Site):
    """On the model, for a prompt in front of the embeddings: what describes the whole input, widened.

    The attention mask gets the virtual positions, kept; the token types
    get zeros there and the labels (a language model's, when ``pad_labels``)
    are skipped there; positions the caller gave are dropped, since the model
    numbers the longer sequence itself.  (A head that reads the mask — the last
    kept token — sees the sequence as long as the trunk's output.)
    """

    __slots__ = ("_pad_labels",)

    def __init__(self, tuner: _Prompting, pad_labels: bool) -> None:
        super().__init__(tuner)
        self._pad_labels = pad_labels

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        virtual = applied.virtual
        if "attention_mask" in call:
            call["attention_mask"] = _widen(call["attention_mask"], virtual)
        if call.get("position_ids") is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. "
                "Ignoring position ids.",
                stacklevel=4,
            )
            call["position_ids"] = None
        types = call.get("token_type_ids")
        if isinstance(types, Tensor):
            call["token_type_ids"] = _pad(types, virtual, 0)
        labels = call.get("labels")
        if self._pad_labels and isinstance(labels, Tensor):
            call["labels"] = _pad(labels, virtual, _IGNORE)


class _EmbeddingsSite(_Site):
    """On the trunk: input ids become embeddings with the prompt rows in front."""

    __slots__ = ("_table",)

    def __init__(self, tuner: _Prompting, table: nn.Module) -> None:
        super().__init__(tuner)
        self._table = table

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        if _holds_prompt(call.get("past_key_values")):
            return
        embeds = call.get("inputs_embeds")
        if not isinstance(embeds, Tensor):
            ids = call.pop("input_ids", None)
            if ids is None:
                raise ValueError("a prompt-learning adapter needs input_ids or inputs_embeds")
            if not isinstance(ids, Tensor):
                raise TypeError(f"input_ids must be a Tensor, got {type(ids).__name__}")
            embeds = self._table(ids)
        call.pop("input_ids", None)
        prompt = applied.embeddings(int(embeds.shape[0]), embeds.dtype)
        call["inputs_embeds"] = lucid.cat([prompt, embeds], dim=1)


class _PrefixSite(_Site):
    """On the trunk: a cache that already holds the prefix, when the call starts from none.

    The keys the trunk attends over are the prefix's as well as the input's, on
    every step, so its attention mask gets the virtual positions.  Positions are
    the trunk's to number (after the cache's length, as the reference does); the
    ones the caller gave are used as they are.  The model's heads see the real
    sequence.
    """

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        if "attention_mask" in call:
            call["attention_mask"] = _widen(call["attention_mask"], applied.virtual)
        cache = call.get("past_key_values")
        if _holds_prompt(cache):
            return
        batch = _batch_of(call, ("input_ids", "inputs_embeds"))
        pairs = applied.key_values(batch, self._tuner._model_dtype())
        cache = cache if isinstance(cache, Cache) else DynamicCache()
        _put_prefix(cache, pairs, applied.virtual)
        call["past_key_values"] = cache


class _EncoderSite(_Site):
    """On an encoder-decoder's encoder stack: the prompt rows in front of its input embeddings."""

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        embeds = call["embeddings"]
        assert isinstance(embeds, Tensor)
        virtual = applied.virtual
        # The encoder-decoder's table holds one block of rows per stack; the encoder reads the first.
        prompt = applied.embeddings(int(embeds.shape[0]), embeds.dtype)[:, :virtual]
        call["embeddings"] = lucid.cat([prompt, embeds], dim=1)
        call["attention_mask"] = _widen(call.get("attention_mask"), virtual)


class _MemoryMaskSite(_Site):
    """On the decoder stack: the source padding mask, as wide as the encoder output it masks."""

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        memory, mask = call.get("memory"), call.get("memory_mask")
        if (
            isinstance(memory, Tensor)
            and isinstance(mask, Tensor)
            and int(mask.shape[1]) + applied.virtual == int(memory.shape[1])
        ):
            call["memory_mask"] = _widen(mask, applied.virtual)


class _DecoderPrefixSite(_Site):
    """On the decoder stack: self-attention keys and values already holding the prefix."""

    @override
    def rewrite(self, applied: _Applied, call: _Call) -> None:
        virtual = applied.virtual
        call["attention_mask"] = _widen(call.get("attention_mask"), virtual)
        cache = call.get("cache")
        if _holds_prompt(cache):
            return
        embeds = call["embeddings"]
        assert isinstance(embeds, Tensor)
        pairs = applied.key_values(int(embeds.shape[0]), self._tuner._model_dtype())
        if not isinstance(cache, EncoderDecoderCache):
            cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        _put_prefix(cache, pairs, virtual)
        call["cache"] = cache


class _HeadSite(_Hook):
    """On a token-level head: its input without the rows the prompt added, so the output is the input's length.

    A classifier or span head maps every position of the trunk's output to a
    row of logits.  With the prompt in front of the input the trunk's output
    has ``virtual`` more positions than the input did; cutting them off at the
    head's input — rather than from its logits — leaves the model's own loss
    (its labels, its start and end positions) working on the sequence the
    caller gave.
    """

    __slots__ = ()

    @override
    def apply(
        self, applied: _Applied, module: nn.Module, args: tuple[object, ...], kwargs: _Call
    ) -> tuple[tuple[object, ...], _Call]:
        features = args[0] if len(args) == 1 and not kwargs else None
        if not isinstance(features, Tensor) or features.ndim < 3:
            raise TypeError(
                f"{type(module).__name__} is called with something other than one "
                "(batch, positions, features) tensor, which is what a token-level head reads"
            )
        virtual = applied.virtual
        if int(features.shape[1]) <= virtual:
            raise ValueError(
                f"the head reads {int(features.shape[1])} positions, "
                f"no more than the {virtual} virtual tokens in front"
            )
        return (features[:, virtual:],), {}
