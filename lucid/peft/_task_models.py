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
* a language-modelling task decodes, so the wrapper has ``generate``.

``forward`` is the base class's: the call is passed to the model as it is.
The reference spells each task's arguments out because its models share
them; the zoo's forwards differ by family (an encoder-decoder takes
``decoder_input_ids`` where a causal model does not), so a wrapper that
named them would be wrong for half of its models.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import ClassVar, Final

from lucid._tensor.tensor import Tensor
from lucid.peft._config import TaskType
from lucid.peft._heads import CLASSIFIER_HEADS, SPAN_HEADS
from lucid.peft._peft_model import PeftModel

#: Wrapper class by the task it serves, as the reference's table of the same name.  Read-only.
MODEL_TYPE_TO_PEFT_MODEL_MAPPING: Final[Mapping[TaskType, type[PeftModel]]] = MappingProxyType(
    PeftModel._task_models
)


class _GeneratingPeftModel(PeftModel):
    """A wrapper for a task that decodes: ``generate`` runs the model's own."""

    def generate(
        self,
        *args: object,
        adapter_names: Sequence[str] | None = None,
        task_ids: Tensor | None = None,
        **kwargs: object,
    ) -> object:
        r"""Generate with the adapted model: its own ``generate``, called as given.

        The adapter applies at every step, since the model's layers are
        the adapted ones; ``with model.disable_adapter():`` generates with
        the base model alone.

        Parameters
        ----------
        *args : object
            Positional arguments of the model's ``generate``.
        adapter_names : sequence of str, optional
            One adapter per prompt, as for :meth:`~lucid.peft.PeftModel.forward`:
            every step of prompt ``i`` uses ``adapter_names[i]`` alone
            (``"__base__"``: the base model).  With beam search
            (``num_beams`` in ``kwargs``) each name covers that prompt's
            beams; an encoder, if the model has one, sees the prompts once.
        task_ids : Tensor, optional
            For multitask prompt tuning: the task of each prompt of the batch,
            as for :meth:`~lucid.peft.PeftModel.forward`.
        **kwargs : object
            Keyword arguments of the model's ``generate``.

        Returns
        -------
        object
            What the model's ``generate`` returns.

        Raises
        ------
        AttributeError
            The adapted model has no ``generate``.
        """
        model = self.get_base_model()
        generate = getattr(model, "generate", None)
        if generate is None:
            raise AttributeError(f"{type(model).__name__} has no generate() to call")
        beams = kwargs.get("num_beams")
        routed = self._routed(adapter_names, beams if isinstance(beams, int) and beams > 1 else 1)
        with routed, self._tasked(task_ids):
            return generate(*args, **kwargs)


class PeftModelForCausalLM(_GeneratingPeftModel, task_type=TaskType.CAUSAL_LM):
    r"""An adapted causal language model: the ``PeftModel`` for ``task_type="CAUSAL_LM"``.

    What :func:`~lucid.peft.get_peft_model` returns for a ``CAUSAL_LM``
    config.  Built, called and trained as :class:`~lucid.peft.PeftModel`
    is, and ``generate`` runs the adapted model's own, so the adapter
    shapes every decoded token.

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


class PeftModelForSeq2SeqLM(_GeneratingPeftModel, task_type=TaskType.SEQ_2_SEQ_LM):
    r"""An adapted encoder-decoder model: the ``PeftModel`` for ``task_type="SEQ_2_SEQ_LM"``.

    What :func:`~lucid.peft.get_peft_model` returns for a ``SEQ_2_SEQ_LM``
    config.  Calling it is calling the model — which takes the decoder's
    input as well — and ``generate`` runs the adapted model's own.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_peft_model
    >>> config = LoraConfig(r=2, target_modules=["0"], task_type="SEQ_2_SEQ_LM")
    >>> type(get_peft_model(nn.Sequential(nn.Linear(4, 4)), config)).__name__
    'PeftModelForSeq2SeqLM'
    """


class PeftModelForSequenceClassification(PeftModel, task_type=TaskType.SEQ_CLS):
    r"""An adapted sequence classifier: the ``PeftModel`` for ``task_type="SEQ_CLS"``.

    The classification head is trained in full beside the adapter: a
    ``classifier`` or ``score`` module of the model joins every adapter's
    ``modules_to_save`` (a head the model lacks is left out), so the new
    task's labels are learned whatever the base model's head was.  The
    config passed in is not changed; the wrapper keeps a copy with the head
    added.

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


class PeftModelForTokenClassification(PeftModel, task_type=TaskType.TOKEN_CLS):
    r"""An adapted token classifier: the ``PeftModel`` for ``task_type="TOKEN_CLS"``.

    As for :class:`~lucid.peft.PeftModelForSequenceClassification`, the
    model's ``classifier`` / ``score`` head is trained in full beside the
    adapter.  With a prompt (prompt tuning, p-tuning) in front of the input, the
    head reads the input's positions alone, so the logits and the labels are as
    long as the input the caller gave.

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


class PeftModelForQuestionAnswering(PeftModel, task_type=TaskType.QUESTION_ANS):
    r"""An adapted question-answering model: the ``PeftModel`` for ``task_type="QUESTION_ANS"``.

    The span head — the model's ``qa_outputs`` module — is trained in full
    beside the adapter, so the start and end positions are learned
    whatever the base model's head was.  With a prompt in front of the input,
    the head reads the input's positions alone, so the start and end logits
    and the answer positions are in the caller's coordinates.

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


class PeftModelForFeatureExtraction(PeftModel, task_type=TaskType.FEATURE_EXTRACTION):
    r"""An adapted feature encoder: the ``PeftModel`` for ``task_type="FEATURE_EXTRACTION"``.

    There is no head to train: the adapter alone shapes the representation
    the model returns.  Calling it calls the model with the arguments given.

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
