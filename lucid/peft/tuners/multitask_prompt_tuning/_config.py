"""``MultitaskPromptTuningConfig``: prompt tuning's fields, plus the tasks and where a prompt starts from."""

import enum
from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import PeftType, _check_count, _check_rank, _Check, _choice, _optional_str
from lucid.peft.tuners.prompt_tuning._config import PromptTuningConfig


class MultitaskPromptTuningInit(enum.StrEnum):
    """How a multitask prompt starts: from text, at random, or from a prompt trained on source tasks."""

    #: From the embeddings of ``prompt_tuning_init_text``'s tokens, as prompt tuning does.
    TEXT = "TEXT"
    #: From the embedding table's random initialisation.
    RANDOM = "RANDOM"
    #: The shared prompt, and the mean of the source tasks' low-rank task factors.
    AVERAGE_SOURCE_TASKS = "AVERAGE_SOURCE_TASKS"
    #: The shared prompt, and the task factors of the one source task ``prompt_tuning_init_task``.
    EXACT_SOURCE_TASK = "EXACT_SOURCE_TASK"
    #: The shared prompt alone; the task factors start at random.
    ONLY_SOURCE_SHARED = "ONLY_SOURCE_SHARED"


@dataclass(kw_only=True)
class MultitaskPromptTuningConfig(PromptTuningConfig, peft_type=PeftType.MULTITASK_PROMPT_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with multitask prompt tuning.

    Multitask prompt tuning (Wang et al., 2023, arXiv:2303.02861) learns one
    prompt shared by every task and, per task, a low-rank factor pair
    :math:`u_k \in \mathbb{R}^{L \times r}`, :math:`v_k \in \mathbb{R}^{r \times d}`.
    Task :math:`k`'s prompt is the shared one multiplied elementwise by
    :math:`u_k v_k`, so a new task adds :math:`r\,(L + d)` numbers, and a prompt
    trained on source tasks can start a target task (``prompt_tuning_init`` and
    ``prompt_tuning_init_state_dict_path``).  Which task each row of a batch is
    comes from the ``task_ids`` argument of the model's call.

    Field names, defaults and ``adapter_config.json`` are the reference adapter
    library's.  Every field is keyword-only; the prompt tuning fields
    (``num_virtual_tokens``, ``tokenizer_name_or_path``, …) hold as for
    :class:`~lucid.peft.PromptTuningConfig`.

    Parameters
    ----------
    prompt_tuning_init : MultitaskPromptTuningInit or str, default="RANDOM"
        ``"RANDOM"`` and ``"TEXT"`` start the shared prompt as prompt tuning does;
        ``"AVERAGE_SOURCE_TASKS"``, ``"EXACT_SOURCE_TASK"`` and ``"ONLY_SOURCE_SHARED"``
        start from the adapter file at ``prompt_tuning_init_state_dict_path``.
    prompt_tuning_init_state_dict_path : str, optional
        A ``.safetensors`` adapter file of a multitask prompt trained on the source
        tasks; required by the three source initialisations.
    prompt_tuning_init_task : int, default=0
        With ``"EXACT_SOURCE_TASK"``, the source task whose factors are taken.
    num_ranks : int, default=1
        The rank :math:`r` of each task's factor pair.
    num_tasks : int, default=1
        How many tasks the prompt serves.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A count is not positive, or ``"TEXT"`` lacks its text or tokenizer.

    Examples
    --------
    >>> from lucid.peft import MultitaskPromptTuningConfig
    >>> config = MultitaskPromptTuningConfig(num_virtual_tokens=8, num_tasks=3, num_ranks=2)
    >>> config.peft_type, config.num_tasks
    (<PeftType.MULTITASK_PROMPT_TUNING: 'MULTITASK_PROMPT_TUNING'>, 3)
    >>> MultitaskPromptTuningConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "prompt_tuning_init": _choice(tuple(MultitaskPromptTuningInit)),
        "prompt_tuning_init_state_dict_path": _optional_str,
        "prompt_tuning_init_task": _check_count,
        "num_ranks": _check_rank,
        "num_tasks": _check_rank,
    }

    prompt_tuning_init: MultitaskPromptTuningInit | str = MultitaskPromptTuningInit.RANDOM
    prompt_tuning_init_state_dict_path: str | None = None
    prompt_tuning_init_task: int = 0
    num_ranks: int = 1
    num_tasks: int = 1
