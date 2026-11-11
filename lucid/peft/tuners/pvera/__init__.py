"""``lucid.peft.tuners.pvera`` — PVeRA: VeRA whose rank-r latent is a sampled Gaussian.

PVeRA (Fillioux et al., 2025, arXiv:2512.07703) makes VeRA probabilistic: shared frozen random
projections and two learned vectors per layer, with the rank-r latent between them the mean and log-
variance of a Gaussian.  Pick it for VeRA-sized adapters with a learned uncertainty over the update.

Not implemented yet.  ``PveraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``PveraModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-824, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.PVERA`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.pvera._config import PveraConfig
from lucid.peft.tuners.pvera._model import PveraModel

__all__ = ["PveraConfig", "PveraModel"]
