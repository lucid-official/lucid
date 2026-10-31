"""T5: the reference library's ``block.N.layer.M.<Module>`` nesting → the zoo's roles.

The reference names a block's sub-layers by position and by the Mesh
TensorFlow port's module names; the zoo names them by what they do:

====================================================  ==========================================
reference                                              zoo
====================================================  ==========================================
``{enc,dec}.block.N.layer.0.SelfAttention.{q,k,v,o}``  ``{enc,dec}.blocks.N.self_attention.*``
``decoder.block.N.layer.1.EncDecAttention.*``          ``decoder.blocks.N.cross_attention.*``
``….layer.M.DenseReluDense.{wi,wo}``                   ``….blocks.N.feed_forward.*``
``….layer.M.layer_norm``                               ``….blocks.N.<sub-layer>_norm``
``{enc,dec}.block.0.layer.0.SelfAttention.``           ``{enc,dec}.relative_attention_bias``
``relative_attention_bias``
``shared`` / ``{enc,dec}.final_layer_norm``            unchanged
``lm_head``                                            unchanged (tied to ``shared``)
====================================================  ==========================================

The text-to-text model nests the trunk under ``t5.`` in the zoo
(``t5.encoder.blocks…``) where the reference's ``ForConditionalGeneration`` does not;
the trunk-only model has no prefix on either side.  Tensors are not
touched — ``nn.Linear`` on both sides, no transposes.

The map is the one the weight converter (``tools/convert_weights/t5.py``)
renames a released checkpoint by; it lives here so that the tool and an
adapter converter cannot disagree.
"""

import re
from typing import Final, NamedTuple, override

from lucid.peft._convert._base import AdapterConverter, ConversionContext

_TRUNKED: Final = "T5ForConditionalGeneration"
_TRUNK_ONLY: Final = ("T5Model", "T5EncoderModel")


class _Rule(NamedTuple):
    """A reference name pattern and the zoo name it becomes; ``trunk`` rules sit under the text-to-text ``t5.``."""

    pattern: re.Pattern[str]
    template: str
    trunk: bool = True


def _sublayer_rules() -> tuple[_Rule, ...]:
    """The rules of a block's sub-layers: the reference module at layer index M of a stack, and its role."""
    stacks = {
        "encoder": ((0, "SelfAttention", "self_attention"), (1, "DenseReluDense", "feed_forward")),
        "decoder": (
            (0, "SelfAttention", "self_attention"),
            (1, "EncDecAttention", "cross_attention"),
            (2, "DenseReluDense", "feed_forward"),
        ),
    }
    rules: list[_Rule] = []
    for stack, sublayers in stacks.items():
        for index, module, role in sublayers:
            block = rf"{stack}\.block\.(?P<n>\d+)\.layer\.{index}"
            if module == "SelfAttention":
                rules.append(
                    _Rule(
                        re.compile(
                            rf"{stack}\.block\.0\.layer\.0\.SelfAttention\."
                            r"relative_attention_bias(?P<tail>\..+)?"
                        ),
                        rf"{stack}.relative_attention_bias\g<tail>",
                    )
                )
            rules.append(
                _Rule(
                    re.compile(rf"{block}\.{module}\.(?P<tail>.+)"),
                    rf"{stack}.blocks.\g<n>.{role}.\g<tail>",
                )
            )
            rules.append(
                _Rule(
                    re.compile(rf"{block}\.layer_norm(?P<tail>\..+)?"),
                    rf"{stack}.blocks.\g<n>.{role}_norm\g<tail>",
                )
            )
    return tuple(rules)


_RULES: Final = (
    _Rule(re.compile(r"shared(?P<tail>\..+)?"), r"shared\g<tail>"),
    _Rule(
        re.compile(r"(?P<stack>encoder|decoder)\.final_layer_norm(?P<tail>\..+)?"),
        r"\g<stack>.final_layer_norm\g<tail>",
    ),
    _Rule(re.compile(r"lm_head(?P<tail>\..+)?"), r"lm_head\g<tail>", trunk=False),
    *_sublayer_rules(),
)


def t5_reference_name(name: str, *, trunked: bool = False) -> str | None:
    """The zoo's name for the reference T5 module or parameter ``name``; ``None`` if the layout has none.

    A tied alias (``encoder.embed_tokens``) has no zoo name: it is the shared table.

    Parameters
    ----------
    name : str
        A dotted module path or parameter key of the reference library's T5.
    trunked : bool, default=False
        Put the trunk under ``t5.``, as the zoo's text-to-text model does.

    Returns
    -------
    str or None
        The zoo's name, or ``None``.
    """
    for rule in _RULES:
        match = rule.pattern.fullmatch(name)
        if match is not None:
            mapped = match.expand(rule.template)
            return f"t5.{mapped}" if trunked and rule.trunk else mapped
    return None


class T5Converter(AdapterConverter, model_type="t5"):
    """The reference library's T5 adapters, renamed onto the zoo's stacks.

    Whether the zoo model nests its trunk under ``t5.`` (the text-to-text model does, the
    trunk-only model does not) is read off the model when there is one, and otherwise off
    the reference class the adapter's ``auto_mapping`` names.  The tensors are not touched.
    """

    reference_classes = frozenset({_TRUNKED, *_TRUNK_ONLY})

    def __init__(self, context: ConversionContext) -> None:
        super().__init__(context)
        self._trunked = self._has_trunk(context)

    @staticmethod
    def _has_trunk(context: ConversionContext) -> bool:
        if context.model is not None:
            return hasattr(context.model, "t5")
        if context.reference_class == _TRUNKED:
            return True
        if context.reference_class in _TRUNK_ONLY:
            return False
        raise ValueError(
            "cannot tell whether the T5 model nests its trunk under 't5.': the adapter's "
            f"auto_mapping names {context.reference_class!r}; give the reference class "
            f"({_TRUNKED}, {', '.join(_TRUNK_ONLY)}) or load into the model"
        )

    @override
    def map_module(self, module: str) -> str | None:
        return t5_reference_name(module, trunked=self._trunked)
