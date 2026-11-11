"""``AdamssModel``: the stand-in of the AdaMSS tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _ConfigKeyword, _PendingTuner


class AdamssModel(_ConfigKeyword, _PendingTuner, peft_type=PeftType.ADAMSS, issue="LCD-828"):
    r"""The AdaMSS tuner: registered, not written yet.

    ``AdamssModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply AdaMSS: independent low-rank updates inside clustered subspaces of the SVD.  Its
    config is ``AdamssConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-828, which tracks the tuner, before any model is touched.
    """
