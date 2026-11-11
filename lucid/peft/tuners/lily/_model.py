"""``LilyModel``: the stand-in of the Lily tuner."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._host import _PendingTuner


class LilyModel(_PendingTuner, peft_type=PeftType.LILY, issue="LCD-824"):
    r"""The Lily tuner: registered, not written yet.

    ``LilyModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner that
    would apply Lily: low-rank adapters shared across layers and mixed by a router.  Its config is
    ``LilyConfig``.  Building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the
    class itself, raises ``NotImplementedError`` naming LCD-824, which tracks the tuner, before any
    model is touched.
    """
