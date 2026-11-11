r"""``lucid.peft.tuners.frod`` — FRoD: a full-rank update from a sparse trainable matrix over a shared basis.

FRoD (Wan et al., 2025, arXiv:2512.23485 — Full-Rank Efficient Fine-Tuning with Rotational Degrees)
writes the weights of every layer of one role in the network as :math:`W_i = U_i\Sigma_i V^\top` over a basis
:math:`V` they share (the query projections of all blocks, say), and trains :math:`\Sigma_i` plus a sparse
matrix :math:`S_i` of a few off-diagonal entries: :math:`W_i' = U_i(\Sigma_i + S_i)V^\top`.  The off-diagonal
entries rotate the weight's axes, so the update is full-rank with few parameters.  Pick it for converging faster
than a low-rank adapter at a similar number of trainable parameters.

Where the reference library keeps them, so ``isinstance(m, FrodLayer)`` and ``from
lucid.peft.tuners.frod import FrodConfig, Linear`` read as they do there.  One layer type for the one module the
method adapts, ``nn.Linear``.  ``FrodConfig`` and ``FrodModel`` are here and exported from ``lucid.peft``.
Importing the package registers the method (its config under ``PeftType.FROD``, its tuner and layer), which
is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.frod._config import FrodConfig
from lucid.peft.tuners.frod._layer import FrodLayer, Linear
from lucid.peft.tuners.frod._model import FrodModel

__all__ = ["FrodConfig", "FrodLayer", "FrodModel", "Linear"]
