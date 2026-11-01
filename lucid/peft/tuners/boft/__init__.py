r"""``lucid.peft.tuners.boft`` — BOFT: OFT as a butterfly product of sparse rotations.

BOFT (Liu et al., 2023, arXiv:2311.06243 — Butterfly Orthogonal Finetuning) factors the
block-diagonal orthogonal matrix of OFT into a product of :math:`m` sparse orthogonal factors in
butterfly order, :math:`M = M_{m-1}\cdots M_0`, so few parameters span a dense rotation of the
input space; a learned row scale rides on it.  With ``boft_n_butterfly_factor=1`` it is OFT.  Pick
it over OFT when block-diagonal rotations are too local and a dense one costs too much.

Where the reference library keeps them, so ``isinstance(m, BOFTLayer)`` and ``from
lucid.peft.tuners.boft import BOFTConfig, Linear`` read as they do there.  One layer type per module
the method adapts: ``Linear`` and ``Conv2d``.  ``BOFTConfig`` and ``BOFTModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.BOFT``, its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.boft._config import BOFTConfig
from lucid.peft.tuners.boft._layer import BOFTLayer, Conv2d, Linear
from lucid.peft.tuners.boft._model import BOFTModel

__all__ = ["BOFTConfig", "BOFTLayer", "BOFTModel", "Conv2d", "Linear"]
