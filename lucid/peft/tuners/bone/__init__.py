"""``lucid.peft.tuners.bone`` — Bone, the method MiSS replaced: its config, tuner and layer types.

Bone (Kang, 2024, arXiv:2409.15371 — Block Affine) learns one block per run of output rows and
applies it to the weight itself, :math:`W' = W(I + S) + S` blockwise, so it is non-linear in the
blocks.  The paper was revised as MiSS, and the reference library no longer ships Bone.  It is kept
here because adapter files written under the name exist: a Bone adapter is a MiSS adapter of the
default or ``"bat"`` variant with the block stored as ``bone_block``.  New work uses
``lucid.peft.tuners.miss``.

One layer type for the one module the method adapts, ``nn.Linear``: ``Linear``, on ``BoneLayer``.
``BoneConfig`` and ``BoneModel`` are here and exported from ``lucid.peft``.  Importing the package
registers the method (its config under ``PeftType.BONE``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.bone._config import BoneConfig
from lucid.peft.tuners.bone._layer import BoneLayer, Linear
from lucid.peft.tuners.bone._model import BoneModel

__all__ = ["BoneConfig", "BoneLayer", "BoneModel", "Linear"]
