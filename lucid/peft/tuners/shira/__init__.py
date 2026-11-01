r"""``lucid.peft.tuners.shira`` — SHiRA: train a sparse subset of the weight itself.

SHiRA (Bhardwaj et al., 2024, arXiv:2406.13175 — Sparse High Rank Adapters) fixes a mask of
:math:`r\,(d_{out}+d_{in})` entries of each adapted weight — as many as a rank-:math:`r` LoRA pair
has — and trains the values of those entries alone.  The update is sparse but not low rank.  The
mask is a function of the weight's shape and ``random_seed`` (or of your own ``mask_fn``, see
``random_mask``), so an adapter is a short list of indices and values.  Pick it when a high-rank,
sparse change suits the task.

Where the reference library keeps them, so ``isinstance(m, ShiraLayer)`` and ``from
lucid.peft.tuners.shira import ShiraConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``ShiraConfig`` and ``ShiraModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.SHIRA``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.shira._config import ShiraConfig
from lucid.peft.tuners.shira._layer import Linear, ShiraLayer
from lucid.peft.tuners.shira._mask import random_mask
from lucid.peft.tuners.shira._model import ShiraModel

__all__ = ["Linear", "ShiraConfig", "ShiraLayer", "ShiraModel", "random_mask"]
