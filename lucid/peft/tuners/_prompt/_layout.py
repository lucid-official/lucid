"""Which modules of a model carry the prompt, by the task the adapter is for.

A layout is the knowledge of one kind of model: a language model's loss
needs its labels widened, a classifier's does not; an encoder-decoder takes
the prompt on its encoder and the prefix on its decoder, a tagger's head loses
the rows the prompt added.  Each layout names
itself for the tasks it serves in its class statement
(``class _CausalLayout(..., task_types=(TaskType.CAUSAL_LM,))``) and is
found by ``_Layout.for_task``; the tuner never branches on the task.
"""

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import TaskType
from lucid.peft._heads import CLASSIFIER_HEADS, SPAN_HEADS
from lucid.peft.tuners._prompt._encoder import _Injection
from lucid.peft.tuners._prompt._sites import (
    _DecoderPrefixSite,
    _EmbeddingsSite,
    _EncoderSite,
    _HeadSite,
    _Hook,
    _MemoryMaskSite,
    _PrefixSite,
    _Prompting,
    _RequestSite,
)

type _Placement = list[tuple[nn.Module, _Hook]]

# The layouts register themselves as they are defined; naming them here is what keeps
# a reader (and dead-code detection) from taking an unreferenced class for unused.
__all__ = [
    "_CausalLayout",
    "_Layout",
    "_Seq2SeqLayout",
    "_SingleStackLayout",
    "_SpanLayout",
    "_TokenClassLayout",
    "_TokenLevelLayout",
]


def _takes(module: nn.Module, name: str) -> bool:
    return name in inspect.signature(module.forward).parameters


def _trunk(model: nn.Module) -> nn.Module:
    """The module under ``model`` that embeds and attends (the model itself when it is one)."""
    prefix = getattr(model, "base_model_prefix", "")
    trunk = getattr(model, prefix, None) if isinstance(prefix, str) and prefix else None
    return trunk if isinstance(trunk, nn.Module) else model


class _Layout(ABC):
    """The modules of a model that a prompt-learning adapter hooks, and the hook for each."""

    _layouts: ClassVar[dict[TaskType | None, type[_Layout]]] = {}
    #: Whether the model's labels get the virtual positions (a language model's loss runs over all of them).
    pads_labels: ClassVar[bool] = False

    @override
    def __init_subclass__(
        cls, *, task_types: tuple[TaskType | None, ...] = (), **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        for task in task_types:
            _Layout._layouts[task] = cls

    @staticmethod
    def for_task(task_type: TaskType | str | None) -> type[_Layout]:
        """The layout that serves ``task_type``; a model with no task is a single stack."""
        task = None if task_type is None else TaskType(task_type)
        return _Layout._layouts.get(task, _SingleStackLayout)

    @classmethod
    @abstractmethod
    def sites(cls, tuner: _Prompting, model: nn.Module, injection: _Injection) -> _Placement:
        """``(module, hook)`` pairs, in the order the hooks must run."""


class _SingleStackLayout(_Layout, task_types=(None, TaskType.SEQ_CLS, TaskType.FEATURE_EXTRACTION)):
    """An encoder-only or decoder-only model: one stack takes the prompt, the model's inputs describe it."""

    @classmethod
    @override
    def sites(cls, tuner: _Prompting, model: nn.Module, injection: _Injection) -> _Placement:
        trunk = _trunk(model)
        if injection == _Injection.EMBEDDINGS:
            table = getattr(model, "get_input_embeddings", lambda: None)()
            if not isinstance(table, nn.Module) or not _takes(trunk, "inputs_embeds"):
                raise NotImplementedError(
                    f"{type(trunk).__name__} cannot take inputs_embeds, which a prompt in front "
                    "of the input embeddings needs"
                )
            request = _RequestSite(tuner, cls.pads_labels)
            return [(model, request), (trunk, _EmbeddingsSite(tuner, table))]
        if not _takes(trunk, "past_key_values"):
            raise NotImplementedError(
                f"{type(trunk).__name__} cannot take cached keys and values, which prefix "
                "tuning puts its prefix in"
            )
        return [(trunk, _PrefixSite(tuner))]


class _CausalLayout(_SingleStackLayout, task_types=(TaskType.CAUSAL_LM,)):
    """A language model: the loss covers every position, so the virtual ones are skipped."""

    pads_labels = True


class _Seq2SeqLayout(_Layout, task_types=(TaskType.SEQ_2_SEQ_LM,)):
    """An encoder-decoder: the prompt goes in front of the encoder input, the prefix on the decoder.

    The model, or its trunk, hands out its two stacks (``get_encoder()``,
    ``get_decoder()``); a stack takes ``embeddings`` and ``attention_mask`` (and
    on the decoder ``memory``, ``memory_mask`` and ``cache``).  The model's own
    inputs are untouched, so ``generate`` — which calls the stacks directly —
    is covered as well as ``forward``.
    """

    @classmethod
    @override
    def sites(cls, tuner: _Prompting, model: nn.Module, injection: _Injection) -> _Placement:
        owner = model if hasattr(model, "get_encoder") else _trunk(model)
        if not (hasattr(owner, "get_encoder") and hasattr(owner, "get_decoder")):
            raise NotImplementedError(
                f"{type(model).__name__} does not hand out its encoder and decoder stacks "
                "(get_encoder / get_decoder), which an encoder-decoder prompt is put on"
            )
        encoder, decoder = owner.get_encoder(), owner.get_decoder()
        if injection == _Injection.EMBEDDINGS:
            return [(encoder, _EncoderSite(tuner)), (decoder, _MemoryMaskSite(tuner))]
        return [(decoder, _DecoderPrefixSite(tuner))]


class _TokenLevelLayout(_SingleStackLayout):
    """A model whose output has a row per input position (token tags, answer spans).

    With the prompt in front of the input embeddings the trunk's output is
    ``virtual`` positions longer than the input.  The head's input is cut back
    to the input's length (``_HeadSite``), so the logits, the labels and the
    start and end positions are all in the coordinates of the sequence the
    caller gave.  A prefix leaves the sequence as long as it was and needs no cut.
    """

    #: The names the head can have; the task's subclass says them.
    heads: ClassVar[tuple[str, ...]]

    @classmethod
    @override
    def sites(cls, tuner: _Prompting, model: nn.Module, injection: _Injection) -> _Placement:
        placement = super().sites(tuner, model, injection)
        if injection != _Injection.EMBEDDINGS:
            return placement
        for name in cls.heads:
            head = getattr(model, name, None)
            if isinstance(head, nn.Module):
                return [*placement, (head, _HeadSite(tuner))]
        raise NotImplementedError(
            f"{type(model).__name__} has no {' / '.join(cls.heads)} module, the head whose "
            "input the prompt's rows are cut off"
        )


class _TokenClassLayout(_TokenLevelLayout, task_types=(TaskType.TOKEN_CLS,)):
    """Token classification: a ``classifier`` / ``score`` head over every position."""

    heads = CLASSIFIER_HEADS


class _SpanLayout(_TokenLevelLayout, task_types=(TaskType.QUESTION_ANS,)):
    """Extractive question answering: a ``qa_outputs`` head giving a start and an end logit per position."""

    heads = SPAN_HEADS
