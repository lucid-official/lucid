"""``GraloraModel``: the stand-in of the GraLoRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class GraloraModel(_PendingTuner, peft_type=PeftType.GRALORA, issue="LCD-823"):
    r"""The GraLoRA tuner: registered, not written yet.

    ``GraloraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner
    that would apply GraLoRA: block-wise low-rank updates that exchange information between blocks.
    Its config is ``GraloraConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-823, which tracks the tuner, before any model is touched.
    """
