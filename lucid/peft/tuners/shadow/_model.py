"""``ShadowModel``: the stand-in of the ShadowPEFT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _RefusedTuner


class ShadowModel(
    _RefusedTuner,
    peft_type=PeftType.SHADOW,
    reason="it drives the base model's decoder loop and key-value cache and builds a second backbone next to it, which lucid.peft has no counterpart for",
):
    r"""The ShadowPEFT tuner: refused.

    ``ShadowModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply ShadowPEFT: a parallel shadow network whose state rides the decoder loop.  Its
    config is ``ShadowConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` saying why,
    before any model is touched.
    """
