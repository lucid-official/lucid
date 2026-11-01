"""``CPTModel``: the CPT tuner — prompt tuning whose call knows the prompt's token types and its own loss."""

import dataclasses
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.cpt._config import CPTConfig
from lucid.peft.tuners.cpt._embedding import CPTEmbedding
from lucid.peft.tuners.cpt._loss import _call_labels, _call_types, _cpt_loss


class CPTModel(_PromptLearningTuner, peft_type=PeftType.CPT):
    r"""The tuner of context-aware prompt tuning: a ``CPTEmbedding`` per adapter, in front of the input.

    The prompt goes in front of the input embeddings as in prompt tuning.  A call that passes
    ``labels`` is the one that differs: the model is run without them, and the loss is CPT's — the
    next-token cross-entropy over the *output* tokens of the prompt followed by the input, as the
    type mask marks them, averaged and optionally decayed (``CPTConfig``).  ``input_type_mask``
    (``(batch, tokens)`` integers, all outputs when omitted) gives the types of the input's
    tokens; give ``labels`` and ``input_type_mask`` as keywords.  A call without labels, and
    ``generate``, are prompt tuning's.

    Parameters
    ----------
    model : nn.Module
        The causal language model to prompt.
    config : CPTConfig
        What to prompt with.
    adapter_name : str, default="default"
        Name of the adapter.

    Raises
    ------
    ValueError
        The model already holds adapters.
    NotImplementedError
        The model cannot take ``inputs_embeds``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import create_model
    >>> from lucid.peft import CPTConfig, CPTModel
    >>> lm = create_model("gpt2_small_lm", vocab_size=32, hidden_size=16, num_hidden_layers=1,
    ...                   num_attention_heads=2, intermediate_size=32, max_position_embeddings=16)
    >>> tuned = CPTModel(lm, CPTConfig(cpt_token_ids=[3, 4, 5], cpt_tokens_type_mask=[1, 2, 4],
    ...                                task_type="CAUSAL_LM"))
    >>> ids = lucid.randint(0, 32, (2, 5))
    >>> tuned(ids, labels=ids).logits.shape   # the prompt's 3 positions come first
    (2, 8, 32)
    """

    _config_class: ClassVar[type[PromptLearningConfig]] = CPTConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = CPTEmbedding

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        """Call the model; with ``labels``, return its output with the CPT loss in place of its own."""
        types = kwargs.pop("input_type_mask", None)
        labels = kwargs.get("labels")
        applied = self._applying()
        if applied is None or labels is None:
            return self.model(*args, **kwargs)
        config = applied.config
        assert isinstance(config, CPTConfig)  # the adapter was admitted as a CPT one
        if not isinstance(labels, Tensor) or not (types is None or isinstance(types, Tensor)):
            raise TypeError("labels and input_type_mask must be Tensors")
        del kwargs["labels"]
        output = self.model(*args, **kwargs)
        logits = getattr(output, "logits", None)
        if not (
            isinstance(logits, Tensor)
            and dataclasses.is_dataclass(output)
            and not isinstance(output, type)
        ):
            raise TypeError("CPT needs a model whose output is a dataclass with `logits`")
        batch, tokens = int(labels.shape[0]), int(labels.shape[1])
        type_mask = _call_types(config, types, batch, tokens, labels)
        loss = _cpt_loss(logits, _call_labels(config, labels, type_mask), type_mask, config)
        return dataclasses.replace(output, loss=loss)
