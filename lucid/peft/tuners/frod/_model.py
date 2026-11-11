"""``FrodModel``: the stand-in of the FRoD tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class FrodModel(_PendingTuner, peft_type=PeftType.FROD, issue="LCD-825"):
    r"""The FRoD tuner: registered, not written yet.

    ``FrodModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply FRoD: a full-rank update from a sparse trainable rotation.  Its config is
    ``FrodConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-825, which tracks the tuner, before any
    model is touched.
    """
