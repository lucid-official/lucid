"""``WaveFTModel``: the stand-in of the WaveFT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class WaveFTModel(_PendingTuner, peft_type=PeftType.WAVEFT, issue="LCD-826"):
    r"""The WaveFT tuner: registered, not written yet.

    ``WaveFTModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply WaveFT: a few coefficients of the update in a wavelet basis.  Its config is
    ``WaveFTConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or
    the class itself, raises ``NotImplementedError`` naming LCD-826, which tracks the tuner, before
    any model is touched.
    """
