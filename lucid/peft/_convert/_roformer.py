"""RoFormer: the zoo is laid out like the reference library, so a file converts by identity.

``roformer.encoder.layer.N.attention.self.{query,key,value}`` and the rest of
the encoder are named alike on both sides, and so is the sequence-classification
head: ``classifier.dense`` then ``classifier.out_proj`` (one module, which
``modules_to_save=["classifier"]`` trains whole).  Nothing is renamed; the
converter exists to claim the reference classes and run the shared checks.
"""

from typing import override

from lucid.peft._convert._base import AdapterConverter


class RoFormerConverter(AdapterConverter, model_type="roformer"):
    """RoFormer adapters: every module path is the reference's own.

    The encoder and the nested ``classifier.dense`` / ``classifier.out_proj``
    head are named alike in the reference library and in the zoo, so a file
    converts by identity.
    """

    reference_classes = frozenset(
        {
            "RoFormerModel",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerForMaskedLM",
            "RoFormerForQuestionAnswering",
            "RoFormerForMultipleChoice",
        }
    )

    @override
    def map_module(self, module: str) -> str | None:
        return module
