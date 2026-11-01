r"""``lucid.peft.tuners.hra`` — HRA: a chain of learned Householder reflections.

HRA (Yuan et al., 2024, arXiv:2405.17484 — Householder Reflection Adaptation) multiplies a weight
from the right by :math:`r` Householder reflections,
:math:`W' = W\prod_i (I - 2\hat u_i \hat u_i^\top)`, an orthogonal transform of the input space
whose only parameters are the :math:`r` vectors :math:`u_i`.  With an even ``r`` it starts as the
identity, and it folds into the weight.  The paper presents it as a bridge between low-rank and
orthogonal adaptation: pick it for an orthogonal method that costs :math:`r\,d_{in}` parameters
per layer.

Where the reference library keeps them, so ``isinstance(m, HRALayer)`` and ``from
lucid.peft.tuners.hra import HRAConfig, Linear`` read as they do there.  One layer type per module
the method adapts: ``Linear`` and ``Conv2d``.  ``HRAConfig`` and ``HRAModel`` are here and exported
from ``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.HRA``,
its tuner and layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know
it.
"""

from lucid.peft.tuners.hra._config import HRAConfig
from lucid.peft.tuners.hra._layer import Conv2d, HRALayer, Linear
from lucid.peft.tuners.hra._model import HRAModel

__all__ = ["Conv2d", "HRAConfig", "HRALayer", "HRAModel", "Linear"]
