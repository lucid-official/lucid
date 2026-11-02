"""``PeftModelFor*``: the model class of each task, and the table that finds it.

A wrapper is a ``PeftModel`` subclass that names its task in its class
statement — ``class PeftModelForCausalLM(..., task_type=TaskType.CAUSAL_LM)``
— and is registered for it as it is defined (``PeftModel.__init_subclass__``).
``get_peft_model`` and ``PeftModel.from_pretrained`` look the class up by
the config's ``task_type``; nothing here is a branch they take.

What a wrapper adds to ``PeftModel`` is what the task means for the model:

* a classification or question-answering task trains the model's head, so
  the wrapper names it (``_head_names``) and the head joins
  ``modules_to_save`` of every adapter the model takes;
* its ``forward`` names the arguments the reference's does (``input_ids``,
  ``attention_mask``, ``labels`` …) and hands the ones given to the model by
  keyword, the first positional one staying positional, so a call reads the
  same on both sides.  A model whose ``forward`` has no parameter of such a
  name raises ``TypeError`` for it, as any call with an unknown keyword does.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, TaskType, _require_config
from lucid.peft._heads import CLASSIFIER_HEADS, SPAN_HEADS
from lucid.peft._peft_model import PeftModel
from lucid.peft._refusals import (
    NO_FLOAT_AUTOCAST_OFF,
    NO_META_DEVICE,
    reject_unexpected,
    refuse_value,
)

#: Wrapper class by the task it serves, as the reference's table of the same name.  Read-only.
MODEL_TYPE_TO_PEFT_MODEL_MAPPING: Final[Mapping[TaskType, type[PeftModel]]] = MappingProxyType(
    PeftModel._task_models
)


class _TaskPeftModel(PeftModel):
    """A wrapper for a task: the constructor the reference's wrappers share, and the call that names its arguments."""

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        **kwargs: object,
    ) -> None:
        """Adapt ``model`` in place with ``peft_config`` and wrap it for this task.

        ``kwargs`` are the options of :class:`~lucid.peft.PeftModel` the reference's
        wrappers pass on (``autocast_adapter_dtype``, ``low_cpu_mem_usage``); another key
        raises ``TypeError``.
        """
        where = type(self).__name__
        autocast = kwargs.pop("autocast_adapter_dtype", True)
        refuse_value(where, "autocast_adapter_dtype", autocast, True, NO_FLOAT_AUTOCAST_OFF)
        refuse_value(
            where,
            "low_cpu_mem_usage",
            kwargs.pop("low_cpu_mem_usage", False),
            False,
            NO_META_DEVICE,
        )
        reject_unexpected(where, kwargs)
        super().__init__(model, peft_config, adapter_name)

    def _called(
        self,
        input_ids: Tensor | None,
        named: Mapping[str, object],
        task_ids: Tensor | None,
        kwargs: dict[str, object],
    ) -> object:
        """Run the model on what was given: ``input_ids`` positionally, the other named arguments by keyword."""
        args = () if input_ids is None else (input_ids,)
        given = {name: value for name, value in named.items() if value is not None}
        return PeftModel.forward(self, *args, task_ids=task_ids, **given, **kwargs)


class _TextForward(_TaskPeftModel):
    """The call of a task whose model reads ``input_ids`` and answers with logits or a loss."""

    @override
    def forward(  # type: ignore[override]  # narrows the base call to the reference task signature
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        task_ids: Tensor | None = None,
        **kwargs: object,
    ) -> object:
        r"""Run the adapted model; the arguments given go to it as they are.

        Parameters
        ----------
        input_ids : Tensor, optional
            The token ids.
        attention_mask : Tensor, optional
            Which positions are tokens (1) and which padding (0).
        inputs_embeds : Tensor, optional
            Embeddings in place of ``input_ids``.
        labels : Tensor, optional
            The targets, for a model that computes its own loss.
        output_attentions : bool, optional
            Ask the model for its attention weights.
        output_hidden_states : bool, optional
            Ask the model for its hidden states.
        return_dict : bool, optional
            Ask the model for a result object rather than a tuple.
        task_ids : Tensor, optional
            For multitask prompt tuning: the task of each row of the batch, ``(batch,)``
            integers in ``[0, num_tasks)``.
        **kwargs : object
            Further keyword arguments of the model's ``forward``, and ``adapter_names``
            as for :meth:`~lucid.peft.PeftModel.forward`.

        Returns
        -------
        object
            What the model's ``forward`` returns.

        Raises
        ------
        TypeError
            The model's ``forward`` has no parameter for an argument that was given.
        """
        named = {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        return self._called(input_ids, named, task_ids, kwargs)


class PeftModelForCausalLM(_TextForward, task_type=TaskType.CAUSAL_LM):
    r"""An adapted causal language model: the ``PeftModel`` for ``task_type="CAUSAL_LM"``.

    What :func:`~lucid.peft.get_peft_model` returns for a ``CAUSAL_LM``
    config.  Built, called and trained as :class:`~lucid.peft.PeftModel`
    is, and ``generate`` runs the adapted model's own, so the adapter
    shapes every decoded token.

    Parameters
    ----------
    model, peft_config, adapter_name, **kwargs
        As in :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModelForCausalLM
    >>> class Counter(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.proj = nn.Linear(4, 4)
    ...     def generate(self, x, steps=2):
    ...         for _ in range(steps):
    ...             x = self.proj(x)
    ...         return x
    >>> peft = PeftModelForCausalLM(Counter(), LoraConfig(r=2, target_modules=["proj"]))
    >>> peft.generate(lucid.randn(1, 4), steps=3).shape
    (1, 4)
    """


class PeftModelForSeq2SeqLM(_TaskPeftModel, task_type=TaskType.SEQ_2_SEQ_LM):
    r"""An adapted encoder-decoder model: the ``PeftModel`` for ``task_type="SEQ_2_SEQ_LM"``.

    What :func:`~lucid.peft.get_peft_model` returns for a ``SEQ_2_SEQ_LM``
    config.  Calling it is calling the model — which takes the decoder's
    input as well — and ``generate`` runs the adapted model's own.

    Parameters
    ----------
    model, peft_config, adapter_name, **kwargs
        As in :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_peft_model
    >>> config = LoraConfig(r=2, target_modules=["0"], task_type="SEQ_2_SEQ_LM")
    >>> type(get_peft_model(nn.Sequential(nn.Linear(4, 4)), config)).__name__
    'PeftModelForSeq2SeqLM'
    """

    @override
    def forward(  # type: ignore[override]  # narrows the base call to the reference task signature
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        decoder_inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        task_ids: Tensor | None = None,
        **kwargs: object,
    ) -> object:
        r"""Run the adapted encoder-decoder; the arguments given go to it as they are.

        Parameters
        ----------
        input_ids, attention_mask, inputs_embeds
            The encoder's inputs, as in :meth:`~lucid.peft.PeftModelForCausalLM.forward`.
        decoder_input_ids : Tensor, optional
            The decoder's token ids.
        decoder_attention_mask : Tensor, optional
            Which decoder positions are tokens (1) and which padding (0).
        decoder_inputs_embeds : Tensor, optional
            Embeddings in place of ``decoder_input_ids``.
        labels, output_attentions, output_hidden_states, return_dict, task_ids, **kwargs
            As in :meth:`~lucid.peft.PeftModelForCausalLM.forward`.

        Returns
        -------
        object
            What the model's ``forward`` returns.

        Raises
        ------
        TypeError
            The model's ``forward`` has no parameter for an argument that was given.
        """
        named = {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        return self._called(input_ids, named, task_ids, kwargs)

    @override
    def generate(self, **kwargs: object) -> object:  # type: ignore[override]  # the reference takes keywords only
        r"""Generate with the adapted model: its own ``generate``, called with these keywords.

        Parameters
        ----------
        **kwargs : object
            Keyword arguments of the model's ``generate`` (``input_ids``, ``max_new_tokens`` …),
            and ``adapter_names`` / ``task_ids`` as for :meth:`~lucid.peft.PeftModel.generate`.

        Returns
        -------
        object
            What the model's ``generate`` returns.

        Raises
        ------
        AttributeError
            The adapted model has no ``generate``.
        """
        return super().generate(**kwargs)


class PeftModelForSequenceClassification(_TextForward, task_type=TaskType.SEQ_CLS):
    r"""An adapted sequence classifier: the ``PeftModel`` for ``task_type="SEQ_CLS"``.

    The classification head is trained in full beside the adapter: a
    ``classifier`` or ``score`` module of the model joins every adapter's
    ``modules_to_save`` (a head the model lacks is left out), so the new
    task's labels are learned whatever the base model's head was.  The
    config passed in is not changed; the wrapper keeps a copy with the head
    added.

    Parameters
    ----------
    model, peft_config, adapter_name, **kwargs
        As in :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModelForSequenceClassification
    >>> net = nn.Sequential(OrderedDict(encoder=nn.Linear(8, 8), classifier=nn.Linear(8, 2)))
    >>> config = LoraConfig(r=2, target_modules=["encoder"])
    >>> PeftModelForSequenceClassification(net, config).peft_config["default"].modules_to_save
    ['classifier']
    """

    _head_names: ClassVar[tuple[str, ...]] = CLASSIFIER_HEADS


class PeftModelForTokenClassification(_TextForward, task_type=TaskType.TOKEN_CLS):
    r"""An adapted token classifier: the ``PeftModel`` for ``task_type="TOKEN_CLS"``.

    As for :class:`~lucid.peft.PeftModelForSequenceClassification`, the
    model's ``classifier`` / ``score`` head is trained in full beside the
    adapter.  With a prompt (prompt tuning, p-tuning) in front of the input, the
    head reads the input's positions alone, so the logits and the labels are as
    long as the input the caller gave.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, changed in place.
    peft_config : PeftConfig
        What to adapt and how; required, though the reference's signature gives it a default.
    adapter_name : str, default="default"
        Name of the adapter.
    **kwargs : object
        ``autocast_adapter_dtype`` and ``low_cpu_mem_usage`` of
        :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModelForTokenClassification
    >>> net = nn.Sequential(OrderedDict(body=nn.Linear(6, 6), score=nn.Linear(6, 9)))
    >>> peft = PeftModelForTokenClassification(net, LoraConfig(r=2, target_modules=["body"]))
    >>> peft.peft_config["default"].modules_to_save
    ['score']
    """

    _head_names: ClassVar[tuple[str, ...]] = CLASSIFIER_HEADS

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig | None = None,
        adapter_name: str = "default",
        **kwargs: object,
    ) -> None:
        """Adapt ``model`` in place with ``peft_config`` and wrap it for token classification.

        ``peft_config`` has a default only because the reference's signature does; leaving it
        out raises ``TypeError``.
        """
        super().__init__(model, _require_config("peft_config", peft_config), adapter_name, **kwargs)


class PeftModelForQuestionAnswering(_TaskPeftModel, task_type=TaskType.QUESTION_ANS):
    r"""An adapted question-answering model: the ``PeftModel`` for ``task_type="QUESTION_ANS"``.

    The span head — the model's ``qa_outputs`` module — is trained in full
    beside the adapter, so the start and end positions are learned
    whatever the base model's head was.  With a prompt in front of the input,
    the head reads the input's positions alone, so the start and end logits
    and the answer positions are in the caller's coordinates.

    Parameters
    ----------
    model, peft_config, adapter_name, **kwargs
        As in :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModelForQuestionAnswering
    >>> net = nn.Sequential(OrderedDict(reader=nn.Linear(5, 5), qa_outputs=nn.Linear(5, 2)))
    >>> peft = PeftModelForQuestionAnswering(net, LoraConfig(r=2, target_modules=["reader"]))
    >>> peft.peft_config["default"].modules_to_save
    ['qa_outputs']
    """

    _head_names: ClassVar[tuple[str, ...]] = SPAN_HEADS

    @override
    def forward(  # type: ignore[override]  # narrows the base call to the reference task signature
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        start_positions: Tensor | None = None,
        end_positions: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        task_ids: Tensor | None = None,
        **kwargs: object,
    ) -> object:
        r"""Run the adapted reader; the arguments given go to it as they are.

        Parameters
        ----------
        input_ids, attention_mask
            As in :meth:`~lucid.peft.PeftModelForCausalLM.forward`.
        token_type_ids : Tensor, optional
            Which segment each position belongs to.
        position_ids : Tensor, optional
            The position of each token.
        inputs_embeds
            As in :meth:`~lucid.peft.PeftModelForCausalLM.forward`.
        start_positions : Tensor, optional
            The answer's first token, for a model that computes its own loss.
        end_positions : Tensor, optional
            The answer's last token, for a model that computes its own loss.
        output_attentions, output_hidden_states, return_dict, task_ids, **kwargs
            As in :meth:`~lucid.peft.PeftModelForCausalLM.forward`.

        Returns
        -------
        object
            What the model's ``forward`` returns.

        Raises
        ------
        TypeError
            The model's ``forward`` has no parameter for an argument that was given.
        """
        named = {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        return self._called(input_ids, named, task_ids, kwargs)


class PeftModelForFeatureExtraction(_TaskPeftModel, task_type=TaskType.FEATURE_EXTRACTION):
    r"""An adapted feature encoder: the ``PeftModel`` for ``task_type="FEATURE_EXTRACTION"``.

    There is no head to train: the adapter alone shapes the representation
    the model returns.  Calling it calls the model with the arguments given.

    Parameters
    ----------
    model, peft_config, adapter_name, **kwargs
        As in :class:`~lucid.peft.PeftModel`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModelForFeatureExtraction
    >>> net = nn.Sequential(nn.Linear(4, 6))
    >>> peft = PeftModelForFeatureExtraction(net, LoraConfig(r=2, target_modules=["0"]))
    >>> peft(lucid.randn(3, 4)).shape
    (3, 6)
    """

    @override
    def forward(  # type: ignore[override]  # narrows the base call to the reference task signature
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        task_ids: Tensor | None = None,
        **kwargs: object,
    ) -> object:
        r"""Run the adapted encoder; the arguments given go to it as they are.

        Parameters
        ----------
        input_ids, attention_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict, task_ids, **kwargs
            As in :meth:`~lucid.peft.PeftModelForCausalLM.forward`; ``input_ids`` is the
            model's input where it is not text.

        Returns
        -------
        object
            What the model's ``forward`` returns.

        Raises
        ------
        TypeError
            The model's ``forward`` has no parameter for an argument that was given.
        """
        named = {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        return self._called(input_ids, named, task_ids, kwargs)
