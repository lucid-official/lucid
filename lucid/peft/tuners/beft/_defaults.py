"""The modules a BEFT adapter takes when its config does not say: the value projections.

BEFT trains one bias, and the paper puts it on the value projection of each attention block.  The
reference keeps a table of that module's name per model type, with no row for a model whose
attention has no separate value projection.  This is that table for the model types Lucid's zoo
registers and the reference has a row for (``bert``, ``t5``); a model type it lacks must name its
``target_modules``.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: ``model_type`` → the module names a BEFT adapter takes by default.  Read-only.
BEFT_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {"bert": frozenset({"value"}), "t5": frozenset({"v"})}
)
