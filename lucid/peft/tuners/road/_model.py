"""``RoadModel``: the stand-in of the RoAd tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class RoadModel(_PendingTuner, peft_type=PeftType.ROAD, issue="LCD-825"):
    r"""The RoAd tuner: registered, not written yet.

    ``RoadModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply RoAd: a rotation and scaling of pairs of output features.  Its config is
    ``RoadConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-825, which tracks the tuner, before any
    model is touched.
    """
