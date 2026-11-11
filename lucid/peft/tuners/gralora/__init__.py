r"""``lucid.peft.tuners.gralora`` — GraLoRA: block-wise low-rank updates that exchange information between blocks.

GraLoRA (Jung et al., 2025, arXiv:2505.20355 — Granular Low-Rank Adaptation) cuts a weight into a
``k x k`` grid of sub-blocks and gives each its own low-rank pair of rank ``r / k``, the pairs
sharing their factors along rows and columns of the grid so that information moves between blocks.
The parameter count and cost are those of a rank-``r`` LoRA while the update can reach rank
``k r``; with ``hybrid_r > 0`` a plain LoRA is added beside the grid.  Pick it for a LoRA-sized
adapter on layers where plain LoRA underfits, such as at higher ranks, or when a few input channels
dominate the gradient.

Where the reference library keeps them, so ``isinstance(m, GraloraLayer)`` and ``from
lucid.peft.tuners.gralora import GraloraConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``.  ``GraloraConfig`` and ``GraloraModel`` are here
and exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.GRALORA``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.gralora._config import GraloraConfig
from lucid.peft.tuners.gralora._layer import GraloraLayer, Linear
from lucid.peft.tuners.gralora._model import GraloraModel

__all__ = ["GraloraConfig", "GraloraLayer", "GraloraModel", "Linear"]
