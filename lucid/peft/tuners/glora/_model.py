"""``GloraModel``: the stand-in of the GLoRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class GloraModel(_PendingTuner, peft_type=PeftType.GLORA, issue="LCD-823"):
    r"""The GLoRA tuner: registered, not written yet.

    ``GloraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply GLoRA: learned corrections to both the weight and the bias of a layer.  Its config
    is ``GloraConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or
    the class itself, raises ``NotImplementedError`` naming LCD-823, which tracks the tuner, before
    any model is touched.
    """
