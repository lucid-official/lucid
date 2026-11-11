"""``PsoftModel``: the stand-in of the PSOFT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class PsoftModel(_PendingTuner, peft_type=PeftType.PSOFT, issue="LCD-825"):
    r"""The PSOFT tuner: registered, not written yet.

    ``PsoftModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply PSOFT: orthogonal fine-tuning inside the weight's principal subspace.  Its config is
    ``PsoftConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-825, which tracks the tuner, before any
    model is touched.
    """
