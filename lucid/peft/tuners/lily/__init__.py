r"""``lucid.peft.tuners.lily`` — Lily: low-rank adapters shared across layers and mixed by a router.

Lily (Zhong et al., 2024, arXiv:2407.09946) shares one low-rank down-projection among several
consecutive layers and one pool of up-projections among all layers, mixed in each layer by a learned
router.  Pick it for fewer, larger adapters than one LoRA per layer, which can then carry a higher
rank.  The mixture depends on the batch, so a Lily adapter cannot be folded into the weights.

Where the reference library keeps them, so ``isinstance(m, LilyLayer)`` and ``from
lucid.peft.tuners.lily import LilyConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``.  ``LilyConfig`` and ``LilyModel`` (which shares the
projections between layers) are here and exported from ``lucid.peft``.  Importing the package
registers the method (its config under ``PeftType.LILY``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.lily._config import LilyConfig
from lucid.peft.tuners.lily._layer import LilyLayer, Linear
from lucid.peft.tuners.lily._model import LilyModel

__all__ = ["LilyConfig", "LilyLayer", "LilyModel", "Linear"]
