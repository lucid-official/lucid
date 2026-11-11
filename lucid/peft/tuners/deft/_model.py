"""``DeftModel``: the stand-in of the DEFT tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class DeftModel(_PendingTuner, peft_type=PeftType.DEFT, issue="LCD-823"):
    r"""The DEFT tuner: registered, not written yet.

    ``DeftModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply DEFT: knowledge injected after removing a learned subspace of the weight.  Its
    config is ``DeftConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-823, which tracks the tuner, before any model is touched.
    """
