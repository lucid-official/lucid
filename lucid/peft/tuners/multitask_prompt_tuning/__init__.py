"""``lucid.peft.tuners.multitask_prompt_tuning`` — multitask prompt tuning: ``MultitaskPromptEmbedding`` and its tuner.

The config (``MultitaskPromptTuningConfig``, ``MultitaskPromptTuningInit``) is exported from
``lucid.peft`` as well.
"""

from lucid.peft.tuners.multitask_prompt_tuning._config import (
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
)
from lucid.peft.tuners.multitask_prompt_tuning._model import (
    MultitaskPromptEmbedding,
    MultitaskPromptTuningModel,
)

__all__ = [
    "MultitaskPromptEmbedding",
    "MultitaskPromptTuningConfig",
    "MultitaskPromptTuningInit",
    "MultitaskPromptTuningModel",
]
