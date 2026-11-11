r"""``lucid.peft.tuners.hira`` — HiRA: a low-rank matrix that modulates the weight elementwise.

HiRA (Huang et al., 2025, ICLR — Hadamard High-Rank Adaptation) multiplies the frozen weight
elementwise by a low-rank matrix, :math:`\Delta W = W \odot (BA)`, so the update can have a rank
far above the ``r`` it is built from while costing :math:`r\,(d_{in} + d_{out})` parameters.  The
update reads the weight, so a merge keeps the weight it was folded into and an unmerge gives it back
exactly.  Pick it for more expressive updates than LoRA at the same parameter count.

Where the reference library keeps them, so ``isinstance(m, HiraLayer)`` and ``from
lucid.peft.tuners.hira import HiraConfig, Linear`` read as they do there.  One layer type per module
the method adapts: ``Linear``, ``Embedding``, ``Conv1d``, ``Conv2d`` and ``Conv3d``.  ``HiraConfig``
and ``HiraModel`` are here and exported from ``lucid.peft``.  Importing the package registers the
method (its config under ``PeftType.HIRA``, its tuner and layers), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.hira._config import HiraConfig
from lucid.peft.tuners.hira._layer import Conv1d, Conv2d, Conv3d, Embedding, HiraLayer, Linear
from lucid.peft.tuners.hira._model import HiraModel

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "HiraConfig",
    "HiraLayer",
    "HiraModel",
    "Linear",
]
