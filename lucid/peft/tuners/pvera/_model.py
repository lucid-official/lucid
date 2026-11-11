"""``PveraModel``: the stand-in of the PVeRA tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class PveraModel(_PendingTuner, peft_type=PeftType.PVERA, issue="LCD-824"):
    r"""The PVeRA tuner: registered, not written yet.

    ``PveraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply PVeRA: VeRA whose rank-r latent is a sampled Gaussian.  Its config is
    ``PveraConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-824, which tracks the tuner, before any
    model is touched.
    """
