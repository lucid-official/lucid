"""``UniLoraModel``: the stand-in of the Uni-LoRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _ConfigKeyword, _PendingTuner


class UniLoraModel(_ConfigKeyword, _PendingTuner, peft_type=PeftType.UNILORA, issue="LCD-824"):
    r"""The Uni-LoRA tuner: registered, not written yet.

    ``UniLoraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner
    that would apply Uni-LoRA: every LoRA factor of the model read out of one trainable vector.  Its
    config is ``UniLoraConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-824, which tracks the tuner, before any model is touched.
    """
