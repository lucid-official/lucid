"""GPT-2: every module is named alike on both sides; the sequence-classification head is not.

The reference's ``GPT2ForSequenceClassification`` calls its head ``score``; the
zoo's calls it ``classifier``, as BERT's does.  ``score`` is the only name
that changes: ``transformer.h.N.attn.c_attn`` and the rest are read as they
are (the ``Conv1D`` projections' ``fan_in_fan_out`` flag only describes the
base weight's layout, and the factors have the same shapes either way).
"""

from typing import override

from lucid.peft._convert._base import AdapterConverter


class GPT2Converter(AdapterConverter, model_type="gpt2"):
    """GPT-2 adapters: ``score`` is the zoo's ``classifier``.

    No other module is renamed and no tensor changes, so an adapter without a task head
    needs no conversion at all (``convert="auto"`` reads it as it is).
    """

    reference_classes = frozenset(
        {
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
            "GPT2ForQuestionAnswering",
        }
    )

    @override
    def map_module(self, module: str) -> str | None:
        return "classifier" if module == "score" else module
