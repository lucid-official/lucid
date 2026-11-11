"""``lucid.peft.tuners.deft`` — DEFT: knowledge injected after removing a learned subspace of the weight.

DEFT (Kumar et al., 2025, arXiv:2509.22793) removes a learned subspace from the weight and injects
new knowledge into another, through a residual projection.  Pick it for editing what a layer already
encodes, rather than only adding to it.

Not implemented yet.  ``DeftConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``DeftModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.DEFT`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.deft._config import DeftConfig
from lucid.peft.tuners.deft._model import DeftModel

__all__ = ["DeftConfig", "DeftModel"]
