r"""``lucid.peft.tuners.multitask_prompt_tuning`` — multitask prompt tuning, a prompt per task.

Multitask prompt tuning (Wang et al., 2023, arXiv:2303.02861) learns one prompt shared by every task
and, per task :math:`k`, a low-rank pair :math:`u_k`, :math:`v_k`; the task's prompt is the shared
one multiplied elementwise by :math:`u_k v_k`.  A new task adds :math:`r\,(L + d)` numbers, and a
prompt trained on source tasks can start a target task (``MultitaskPromptTuningInit``).  Which task
each row of a batch is comes from the ``task_ids`` argument of the model's call.  Pick it when
several related tasks share one model and each should cost little.

The prompt is applied by hooks on the model's own call.  Here: ``MultitaskPromptEmbedding`` (what is
learned) and ``MultitaskPromptTuningModel`` (the tuner).  ``MultitaskPromptTuningConfig`` and
``MultitaskPromptTuningInit`` are here too and exported from ``lucid.peft``.
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
