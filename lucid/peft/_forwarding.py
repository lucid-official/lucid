"""A module that answers for the module it wraps.

``PeftModel`` wraps ``LoraModel``, which wraps the user's model, and a
module trained in full sits behind ``ModulesToSaveWrapper``.  Code written
against the bare model — ``model.generate(...)``, ``model.config``, a zoo
task wrapper reading its head's ``out_features`` — keeps working through
any of them because a public name the wrapper lacks is looked up on what
it wraps.  Private and dunder names are not forwarded: those are the
wrapper's own business, and forwarding them would let machinery that
probes ``getattr(m, "_x", None)`` read the inner module's state as if it
were the wrapper's.
"""

from typing import override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter


def _as_module(value: object) -> nn.Module | None:
    return value if isinstance(value, nn.Module) else None


class _ForwardsAttributes(nn.Module):
    """Base for wrappers: a missing public attribute is read from ``_wrapped``."""

    def _wrapped(self) -> nn.Module | None:
        """The module answering for this one, or ``None`` before it is set."""
        raise NotImplementedError

    @override
    def __getattr__(self, name: str) -> Tensor | Parameter | nn.Module:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_") or (inner := self._wrapped()) is None:
                raise
            # The static type is the registry's; a forwarded method or
            # config object is returned as it is.
            found: Tensor | Parameter | nn.Module = getattr(inner, name)
            return found
