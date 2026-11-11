"""``lucid.peft.tuners.beft`` — BEFT: a trained bias added to chosen projections, and nothing else.

BEFT (Huang et al., 2025, arXiv:2509.15974) trains nothing but a bias added to chosen linear layers,
the value projections by default, starting at zero.  Pick it for a very small adapter when there is
little training data.

Not implemented yet.  ``BeftConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``BeftModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-826, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.BEFT`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.beft._config import BeftConfig
from lucid.peft.tuners.beft._model import BeftModel

__all__ = ["BeftConfig", "BeftModel"]
