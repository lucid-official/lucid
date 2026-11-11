"""``OSFModel``: the stand-in of the OSF tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _ConfigKeyword, _PendingTuner


class OSFModel(_ConfigKeyword, _PendingTuner, peft_type=PeftType.OSF, issue="LCD-825"):
    r"""The OSF tuner: registered, not written yet.

    ``OSFModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply OSF: constrained full fine-tuning in the weight's low singular directions.  Its
    config is ``OSFConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-825, which tracks the tuner, before any model is touched.
    """
