"""``SupertuningModel``: the stand-in of the Super-Tuning tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class SupertuningModel(_PendingTuner, peft_type=PeftType.SUPERTUNING, issue="LCD-826"):
    r"""The Super-Tuning tuner: registered, not written yet.

    ``SupertuningModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner
    that would apply Super-Tuning: a sparse subset of the weight's entries, chosen by magnitude.
    Its config is ``SupertuningConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-826, which tracks the tuner, before any model is touched.
    """
