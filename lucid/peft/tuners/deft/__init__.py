r"""``lucid.peft.tuners.deft`` — DEFT: knowledge injected after removing a learned subspace of the weight.

DEFT (Kumar et al., 2025, arXiv:2509.22793 — Decompositional Efficient Fine-Tuning) removes a
learned subspace from the weight and injects new knowledge into it through a residual projection,
:math:`W' = (I - P_{proj})\,W + Q_P R`.  The projector comes from a learned direction matrix by a
ReLU or a QR decomposition; ``para=True`` keeps the removal alone (PaRa).  The update reads the
weight, so a merge keeps the weight it was folded into and an unmerge gives it back exactly.  Pick
it for editing what a layer already encodes, rather than only adding to it, or for long
fine-tunes that must keep the base model's instruction following.

Where the reference library keeps them, so ``isinstance(m, DeftLayer)`` and ``from
lucid.peft.tuners.deft import DeftConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``DeftConfig`` and ``DeftModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.DEFT``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.deft._config import DeftConfig
from lucid.peft.tuners.deft._layer import DeftLayer, Linear
from lucid.peft.tuners.deft._model import DeftModel

__all__ = ["DeftConfig", "DeftLayer", "DeftModel", "Linear"]
