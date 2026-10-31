"""Adapters written for the reference library's models, converted to the zoo's layout.

``PeftModel.from_pretrained(..., convert=...)`` and ``tools/convert_adapters.py``
are the entry points; a converter per model type is registered in the modules
imported here.
"""

from lucid.peft._convert._base import (
    MODEL_PREFIX,
    AdapterConverter,
    AdapterFile,
    ConversionContext,
    ConvertMode,
    ModulePair,
    UnmappedKey,
    convert_adapter,
    split_key,
)
from lucid.peft._convert._gpt2 import GPT2Converter
from lucid.peft._convert._roformer import RoFormerConverter
from lucid.peft._convert._t5 import T5Converter, t5_reference_name
from lucid.peft._convert._vit import ViTConverter

__all__ = [
    "MODEL_PREFIX",
    "AdapterConverter",
    "AdapterFile",
    "ConversionContext",
    "ConvertMode",
    "GPT2Converter",
    "ModulePair",
    "RoFormerConverter",
    "T5Converter",
    "UnmappedKey",
    "ViTConverter",
    "convert_adapter",
    "split_key",
    "t5_reference_name",
]
