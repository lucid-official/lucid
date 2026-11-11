r"""``lucid.peft.tuners.peanut`` — PEANuT: a small non-linear network that reads the frozen weight and tweaks it.

PEANuT (Zhong et al., 2024, arXiv:2410.01870 — later versions of the paper call it Neat) produces
the weight update from the frozen weight itself, :math:`W + f(W;\theta)`, through a small trained
bottleneck network with a non-linearity, not a fixed low-rank product; ``depth`` adds encoder and
decoder layers with residuals between its ends.  The update reads the weight, so a merge keeps the
weight it was folded into and an unmerge gives it back exactly.  Pick it for a non-linear update
that can follow the structure of the weight it adapts, with LoRA-like parameter counts.

Where the reference library keeps them, so ``isinstance(m, PeanutLayer)`` and ``from
lucid.peft.tuners.peanut import PeanutConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``.  ``PeanutConfig`` and ``PeanutModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.PEANUT``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.peanut._config import PeanutConfig
from lucid.peft.tuners.peanut._layer import Linear, PeanutLayer
from lucid.peft.tuners.peanut._model import PeanutModel

__all__ = ["Linear", "PeanutConfig", "PeanutLayer", "PeanutModel"]
