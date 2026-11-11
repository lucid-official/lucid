"""``TinyLoraModel``: the stand-in of the TinyLoRA tuner."""

from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft.tuners._host import _PendingTuner


class TinyLoraModel(_PendingTuner, peft_type=PeftType.TINYLORA, issue="LCD-824"):
    r"""The TinyLoRA tuner: registered, not written yet.

    ``TinyLoraModel`` stands in the tuner registry (``PEFT_TYPE_TO_TUNER_MAPPING``) for the tuner
    that would apply TinyLoRA: a trainable vector of a few numbers projected into the SVD basis.
    Its config is ``TinyLoraConfig``.  Building one, through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself, raises ``NotImplementedError`` naming
    LCD-824, which tracks the tuner, before any model is touched.
    """

    @override
    def __init__(
        self,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        **kwargs: object,
    ) -> None:
        """Refuse, with the config named ``config`` and the reference's extra keywords accepted unread."""
        del kwargs
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)
