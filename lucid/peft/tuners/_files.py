"""A tuner whose adapter file is not just its layers' tensors.

Most methods write exactly what their layers hold, under the layers' keys.
Two do not: AdaLoRA writes the tensors without the singular triplets its
allocator pruned (and resizes its layers to fit such a file on load), and
VeRA writes one pair of projections for the whole model beside its per-layer
vectors.  A tuner of that kind also derives from ``_FileFormat`` and overrides
the hooks the state-dict helpers (``lucid.peft._state_dict``) call; one that
does not is read and written by the layers' own keys.
"""

from collections.abc import Mapping
from typing import NamedTuple

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig
from lucid.peft.tuners._base import _BaseTunerLayer, _dotted, _tuner_layers


class _FileContext(NamedTuple):
    """Where an adapter file's tensors live in the model a state-dict helper was handed.

    ``prefix`` is the path of the user's model inside ``model`` — ``""`` for a
    model adapted in place, ``"base_model.model."`` inside a ``PeftModel`` — so
    a layer at ``path`` (relative to the user's model) keeps its tensors at
    ``prefix + path + "." + table`` in the file, as the reference lays it out.
    """

    model: nn.Module
    prefix: str
    config: PeftConfig
    adapter: str

    @property
    def layers(self) -> list[tuple[str, _BaseTunerLayer]]:
        """The tuner layers, by path relative to the user's model."""
        cut = len(self.prefix)
        return [(name[cut:], layer) for name, layer in _tuner_layers(self.model)]

    def key(self, path: str, table: str) -> str:
        """The file's key of the tensor in ``table`` of the layer at ``path``."""
        return f"{self.prefix}{_dotted(path, table)}"


class _FileFormat:
    """The hooks of a tuner with a file format of its own; the defaults are the layers' keys and tensors."""

    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """In-memory key → file key of tensors the adapter owns that no slot holds (VeRA's projections)."""
        del context  # the default adapter keeps everything in its layers
        return {}

    @classmethod
    def _to_file(cls, context: _FileContext, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """``state`` — the adapter's tensors by file key — as the file holds them (AdaLoRA drops the ranks it pruned)."""
        del context  # the default writes what the model holds
        return state

    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """The tensors of ``state`` to load, once the model is ready for them: layers resized to fit (AdaLoRA), projections required (VeRA)."""
        del context  # the default takes any file whose shapes fit
        return state
