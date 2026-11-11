"""``CartridgeModel``: the stand-in of the Cartridge tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class CartridgeModel(_PendingTuner, peft_type=PeftType.CARTRIDGE, issue="LCD-826"):
    r"""The Cartridge tuner: registered, not written yet.

    ``CartridgeModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner
    that would apply Cartridge: a trained key-value cache that stands in for a long document.  Its
    config is ``CartridgeConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-826, which tracks the tuner, before any model is touched.
    """
