"""RoFormer: the encoder's names are the reference library's; only a classification head differs.

``roformer.encoder.layer.N.attention.self.{query,key,value}`` and the rest of
the encoder are named alike on both sides.  The sequence-classification head
is not: the reference nests it, ``classifier.dense`` then ``classifier.out_proj``
(one module, which ``modules_to_save=["classifier"]`` trains whole); the zoo's
``roformer_cls`` keeps a top-level ``dense`` beside a ``classifier`` that is only
the output projection (LCD-709).

Where the model has the reference's nested head, a file converts by identity.
Where it has the flat one, ``classifier.out_proj`` becomes ``classifier`` and
a file that trained ``classifier.dense`` is refused: the zoo could hold the trained
layer only as a ``modules_to_save`` entry named ``dense``, and a name matches by
dotted suffix, so it would wrap every ``….dense`` of the encoder too.  The
adapters of the other tasks (token classification's single-``Linear`` head,
adapters that train no head) load as they are.
"""

from typing import Final, override

import lucid.nn as nn
from lucid.peft._convert._base import MODEL_PREFIX, AdapterConverter, AdapterFile, split_key

_FLAT_HEAD: Final = {"classifier.dense": "dense", "classifier.out_proj": "classifier"}
_TRAINED_DENSE: Final = "classifier.dense"


class RoFormerConverter(AdapterConverter, model_type="roformer"):
    """RoFormer adapters: the encoder is read as it is, the classification head is laid out for the model's head."""

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

    @property
    def _nested_head(self) -> bool:
        """Whether the model's classification head is the reference's ``classifier.{dense,out_proj}``."""
        head = getattr(self.context.model, "classifier", None)
        return isinstance(head, nn.Module) and hasattr(head, "dense")

    @override
    def map_module(self, module: str) -> str | None:
        return module if self._nested_head else _FLAT_HEAD.get(module, module)

    @override
    def convert(self, adapter: AdapterFile) -> AdapterFile:
        trained = sorted(
            key
            for key in adapter.tensors
            if key.startswith(MODEL_PREFIX)
            and split_key(key.removeprefix(MODEL_PREFIX))[0] == _TRAINED_DENSE
        )
        if trained and not self._nested_head:
            raise NotImplementedError(
                "this adapter trains the classification head's dense layer "
                f"({trained[0]!r}...), which the zoo's roformer_cls holds as a top-level `dense` "
                "that modules_to_save cannot name without also wrapping every encoder dense "
                "layer; it loads once the head is nested like the reference's (LCD-709)"
            )
        return super().convert(adapter)
