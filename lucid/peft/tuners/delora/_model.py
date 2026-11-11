"""``DeloraModel``: the stand-in of the DeLoRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class DeloraModel(_PendingTuner, peft_type=PeftType.DELORA, issue="LCD-823"):
    r"""The DeLoRA tuner: registered, not written yet.

    ``DeloraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply DeLoRA: a low-rank update whose strength is bounded apart from its direction.  Its
    config is ``DeloraConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-823, which tracks the tuner, before any model is touched.
    """
