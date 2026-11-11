r"""``lucid.peft.tuners.pvera`` — PVeRA: VeRA whose rank-r latent is a sampled Gaussian.

PVeRA (Fillioux et al., 2025, arXiv:2512.07703) makes VeRA probabilistic: two frozen random
projections shared by every layer and two learned vectors per layer, with the rank-``r`` latent
between them the mean and log-variance of a Gaussian, sampled while training and taken at its mean
for inference (which then folds into the weight).  Pick it for VeRA-sized adapters with a learned
uncertainty over the update.

Where the reference library keeps them, so ``isinstance(m, PveraLayer)`` and ``from
lucid.peft.tuners.pvera import PveraConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``PveraConfig`` and ``PveraModel`` (which holds the
shared projections) are here and exported from ``lucid.peft``.  Importing the package registers the
method (its config under ``PeftType.PVERA``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.pvera._config import PveraConfig
from lucid.peft.tuners.pvera._layer import Linear, PveraLayer
from lucid.peft.tuners.pvera._model import PveraModel

__all__ = ["Linear", "PveraConfig", "PveraLayer", "PveraModel"]
