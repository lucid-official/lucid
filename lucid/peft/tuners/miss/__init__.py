"""``lucid.peft.tuners.miss`` — MiSS: one small matrix shared across the shards of a weight.

MiSS (Kang et al., 2024, arXiv:2409.15371 — Matrix Shard Sharing, the successor of Bone) learns
one ``(r, d_out)`` matrix per layer and adds it to the weight in shards: the input axis is cut into
runs of ``r`` columns and every run receives the same update.  A layer costs ``r * d_out``
parameters whatever its input width.  The ``mini`` variant shrinks the matrix further; ``bat`` is
Bone's rule, applied to the weight itself.  Pick it for a cheap structured update; new work should
use MiSS rather than Bone.

Where the reference library keeps them, so ``isinstance(m, MissLayer)`` and ``from
lucid.peft.tuners.miss import MissConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``MissConfig`` and ``MissModel`` are here and exported
from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.MISS``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.miss._config import MissConfig
from lucid.peft.tuners.miss._layer import Linear, MissLayer
from lucid.peft.tuners.miss._model import MissModel

__all__ = ["Linear", "MissConfig", "MissLayer", "MissModel"]
