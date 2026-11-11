r"""``lucid.peft.tuners.adamss`` — AdaMSS: independent low-rank updates inside clustered subspaces of the SVD.

AdaMSS (Zheng et al., 2025, NeurIPS) takes the SVD of each weight, clusters the rows of its leading
left singular directions into subspaces and learns an independent low-rank update inside each,
optionally freezing the less important subspaces while training.  Pick it for a low-rank adapter
whose capacity is spent where the weight's spectrum needs it: a handful of tiny factors per layer
instead of one large pair.

Where the reference library keeps them, so ``isinstance(m, AdamssLayer)`` and ``from
lucid.peft.tuners.adamss import AdamssConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``; ``SubspaceAllocator`` is the adaptive allocation's
state, which ``AdamssModel.update_and_allocate`` drives.  ``AdamssConfig`` and ``AdamssModel`` are
here and exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.ADAMSS``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.adamss._allocator import SubspaceAllocator
from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._layer import AdamssLayer, Linear
from lucid.peft.tuners.adamss._model import AdamssModel

__all__ = ["AdamssConfig", "AdamssLayer", "AdamssModel", "Linear", "SubspaceAllocator"]
