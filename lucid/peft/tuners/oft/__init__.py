r"""``lucid.peft.tuners.oft`` — OFT: a learned rotation of each layer's input space.

OFT (Qiu et al., 2023, arXiv:2306.07280 — Orthogonal Finetuning) leaves the weight's spectrum alone
and rotates its input space instead, :math:`y = (W R^\top) x + b`, with :math:`R` block-diagonal and
each block an orthogonal matrix built by a Cayley transform of a learned skew-symmetric matrix.  The
rotation preserves the pairwise angles between the weight's neurons, which the paper found to keep a
pretrained model's behaviour intact while it is steered.  It starts as the identity and folds into
the weight.  Pick it when the pretrained weight's geometry should survive the fine-tune (the paper
tuned text-to-image diffusion models); BOFT and HRA are the same idea parameterised otherwise.

Where the reference library keeps them, so ``isinstance(m, OFTLayer)`` and ``from
lucid.peft.tuners.oft import OFTConfig, Linear`` read as they do there.  One layer type per module
the method adapts: ``Linear`` and ``Conv2d``.  ``OFTConfig`` and ``OFTModel`` are here and exported
from ``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.OFT``,
its tuner and layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know
it.
"""

from lucid.peft.tuners.oft._config import OFTConfig
from lucid.peft.tuners.oft._layer import Conv2d, Linear, OFTLayer
from lucid.peft.tuners.oft._model import OFTModel

__all__ = ["Conv2d", "Linear", "OFTConfig", "OFTLayer", "OFTModel"]
