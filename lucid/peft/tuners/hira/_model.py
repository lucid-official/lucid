"""``HiraModel``: the stand-in of the HiRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class HiraModel(_PendingTuner, peft_type=PeftType.HIRA, issue="LCD-823"):
    r"""The HiRA tuner: registered, not written yet.

    ``HiraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply HiRA: a low-rank matrix that modulates the weight elementwise.  Its config is
    ``HiraConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-823, which tracks the tuner, before any
    model is touched.
    """
