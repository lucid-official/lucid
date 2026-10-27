"""Models whose input spans more than one modality.

The domain exists because the alternative is a lie about where a model
belongs.  A CLIP filed under ``vision`` would hide a text Transformer
that is half its parameters and all of its transfer story; filed under
``text`` it would hide the image tower.  The zoo classifies by
structure, and two encoders trained to land in one representation are a
structure neither single-modality domain describes.

Concrete families live in sub-packages (``clip``).  What is exported
here is the base config — the arrangement
:mod:`lucid.models.generative` and :mod:`lucid.models.text` already use.

Nothing else is here yet, and that is the rule rather than an accident:
a module at this level is for what **several** families share.  CLIP's
Transformer blocks are generic enough to belong here eventually and sit
in ``clip/_towers.py`` until a second family wants them, because a
shared module with one consumer is an abstraction guessed rather than
observed.
"""

from lucid.models.multimodal._common._config import MultimodalActivation, MultimodalModelConfig

__all__ = ["MultimodalActivation", "MultimodalModelConfig"]
