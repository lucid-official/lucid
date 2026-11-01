r"""``lucid.peft.tuners.vera`` — VeRA: two learned vectors per layer over shared random projections.

VeRA (Kopiczko et al., 2023, arXiv:2310.11454 — Vector-based Random Matrix Adaptation) draws one
random pair :math:`A`, :math:`B` once and shares it, frozen, across every adapted layer; a layer
learns only a vector :math:`\lambda_d` of length ``r`` and a vector :math:`\lambda_b` of its output
width, :math:`y = Wx + \lambda_b\odot B(\lambda_d\odot Ax)`.  Far fewer parameters than LoRA at the
same rank, so ranks run much higher; the projections can be regenerated from ``projection_prng_key``
(``save_projection=False``), so a file can hold only the vectors.  Pick it for the smallest
adapters, or many adapters on one base.

Where the reference library keeps them, so ``isinstance(m, VeraLayer)`` and ``from
lucid.peft.tuners.vera import VeraConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``VeraConfig`` and ``VeraModel`` (which holds the
shared projections) are here and exported from ``lucid.peft``.  Importing the package registers the
method (its config under ``PeftType.VERA``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.vera._config import VeraConfig
from lucid.peft.tuners.vera._layer import Linear, VeraLayer
from lucid.peft.tuners.vera._model import VeraModel

__all__ = ["Linear", "VeraConfig", "VeraLayer", "VeraModel"]
