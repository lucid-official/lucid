"""``PeanutModel``: the stand-in of the PEANuT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class PeanutModel(_PendingTuner, peft_type=PeftType.PEANUT, issue="LCD-823"):
    r"""The PEANuT tuner: registered, not written yet.

    ``PeanutModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply PEANuT: a small non-linear network that reads the frozen weight and tweaks it.  Its
    config is ``PeanutConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-823, which tracks the tuner, before any model is touched.
    """
