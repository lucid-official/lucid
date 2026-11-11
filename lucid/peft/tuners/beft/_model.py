"""``BeftModel``: the stand-in of the BEFT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class BeftModel(_PendingTuner, peft_type=PeftType.BEFT, issue="LCD-826"):
    r"""The BEFT tuner: registered, not written yet.

    ``BeftModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply BEFT: a trained bias added to chosen projections, and nothing else.  Its config is
    ``BeftConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-826, which tracks the tuner, before any
    model is touched.
    """
