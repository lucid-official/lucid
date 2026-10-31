"""``MultitaskPromptEmbedding``: the prompt of multitask prompt tuning, and ``MultitaskPromptTuningModel``, its tuner."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import ClassVar, Final, Self, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.multitask_prompt_tuning._config import (
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
)
from lucid.peft.tuners.prompt_tuning._model import PromptEmbedding, _TextEncoder
from lucid.serialization import load_safetensors

_COLUMNS: Final = "prefix_task_cols"
_ROWS: Final = "prefix_task_rows"
_SHARED: Final = "prompt_embeddings"
_ONLY_SHARED: Final = MultitaskPromptTuningInit.ONLY_SOURCE_SHARED


def _mean_of_tasks(factors: Tensor, task: int) -> Tensor:
    del task  # every source task takes part
    mean: Tensor = factors.mean(dim=0, keepdim=True)
    return mean


def _one_task(factors: Tensor, task: int) -> Tensor:
    if not 0 <= task < int(factors.shape[0]):
        raise IndexError(
            f"prompt_tuning_init_task={task}, but the source adapter has {int(factors.shape[0])} tasks"
        )
    one: Tensor = factors[task].unsqueeze(0)
    return one


#: How a source adapter's per-task factors become this adapter's: the mean over the source tasks,
#: or the factors of the one task asked for.  ``ONLY_SOURCE_SHARED`` takes none, so is not here.
_FACTORS: Final[Mapping[MultitaskPromptTuningInit, Callable[[Tensor, int], Tensor]]] = (
    MappingProxyType(
        {
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS: _mean_of_tasks,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK: _one_task,
        }
    )
)
_FROM_SOURCE: Final = (*_FACTORS, _ONLY_SHARED)


def _source_tensors(path: str) -> dict[str, Tensor]:
    """The tensors of a source adapter file (``.safetensors``)."""
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"{path!r}: only a .safetensors source adapter can be read; "
            "save the source with save_pretrained, which writes one"
        )
    loaded = load_safetensors(path)
    return {key: value for key, value in loaded.items() if isinstance(value, Tensor)}


def _fit(value: Tensor, key: str, shape: tuple[int, ...]) -> Tensor:
    """``value`` (the source's ``key``), which must have ``shape``."""
    if tuple(value.shape) != shape:
        raise ValueError(
            f"the source {key!r} gives shape {tuple(value.shape)}, but this adapter needs {shape}"
        )
    return value


class MultitaskPromptEmbedding(PromptEmbedding):
    r"""The prompt of multitask prompt tuning: a shared prompt scaled by a low-rank factor pair per task.

    For a row of task :math:`k` the prompt is ``embedding(ids) * (cols[k] @ rows[k])`` —
    the shared ``(num_virtual_tokens, token_dim)`` prompt times the product of
    ``(num_virtual_tokens, num_ranks)`` and ``(num_ranks, token_dim)`` factors.

    Parameters
    ----------
    config : MultitaskPromptTuningConfig
        Sizes (``num_virtual_tokens``, ``num_transformer_submodules`` and
        ``token_dim`` set), the tasks and ranks, and the initialisation.
    word_embeddings : nn.Module
        The model's token table, which ``"TEXT"`` copies rows of.
    tokenizer : object with ``encode(text) -> list[int]``, optional
        As for :class:`~lucid.peft.PromptEmbedding`.

    Attributes
    ----------
    embedding : nn.Embedding
        The shared prompt.
    prefix_task_cols : Parameter
        ``(num_tasks, num_virtual_tokens * num_transformer_submodules, num_ranks)``.
    prefix_task_rows : Parameter
        ``(num_tasks, num_ranks, token_dim)``.

    Raises
    ------
    ValueError
        A source initialisation without ``prompt_tuning_init_state_dict_path``, a source file
        that is not ``.safetensors``, lacks a tensor, or whose shapes do not fit this adapter.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import MultitaskPromptEmbedding, MultitaskPromptTuningConfig
    >>> config = MultitaskPromptTuningConfig(
    ...     num_virtual_tokens=3, token_dim=8, num_transformer_submodules=1, num_tasks=2
    ... )
    >>> prompt = MultitaskPromptEmbedding(config, nn.Embedding(10, 8))
    >>> prompt(lucid.arange(3).long().reshape(1, 3), lucid.tensor([1])).shape
    (1, 3, 8)
    """

    per_task: ClassVar[bool] = True
    stored: ClassVar[Mapping[str, str]] = MappingProxyType(
        {**PromptEmbedding.stored, _COLUMNS: _COLUMNS, _ROWS: _ROWS}
    )

    def __init__(
        self,
        config: MultitaskPromptTuningConfig,
        word_embeddings: nn.Module | None,
        tokenizer: _TextEncoder | None = None,
    ) -> None:
        super().__init__(config, word_embeddings, tokenizer)
        assert config.token_dim is not None  # the tuner resolved it
        self.num_tasks = config.num_tasks
        self.prefix_task_cols = Parameter(
            lucid.empty(config.num_tasks, self.num_embeddings, config.num_ranks)
        )
        self.prefix_task_rows = Parameter(
            lucid.empty(config.num_tasks, config.num_ranks, config.token_dim)
        )
        nn.init.normal_(self.prefix_task_cols, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_task_rows, mean=0.0, std=0.02)
        # A built adapter loads its own file next; reading the source again would need it to stay put.
        if not config.inference_mode and config.prompt_tuning_init in _FROM_SOURCE:
            self._start_from_source(config)

    def _start_from_source(self, config: MultitaskPromptTuningConfig) -> None:
        """Copy the shared prompt, and the task factors the init asks for, out of the source adapter."""
        path = config.prompt_tuning_init_state_dict_path
        if path is None:
            raise ValueError(
                "prompt_tuning_init_state_dict_path needs to be specified with "
                f"{config.prompt_tuning_init} init method"
            )
        source = _source_tensors(path)
        wanted = (
            [_SHARED] if config.prompt_tuning_init == _ONLY_SHARED else [_SHARED, _COLUMNS, _ROWS]
        )
        absent = [key for key in wanted if key not in source]
        if absent:
            raise ValueError(f"the source adapter file has no {absent}; it holds {sorted(source)}")
        picked = _FACTORS.get(MultitaskPromptTuningInit(config.prompt_tuning_init))
        with lucid.no_grad():
            self.embedding.weight.copy_(
                _fit(source[_SHARED], _SHARED, tuple(self.embedding.weight.shape))
            )
            if picked is not None:
                for name in (_COLUMNS, _ROWS):
                    param = getattr(self, name)
                    value = picked(source[name], config.prompt_tuning_init_task)
                    param.copy_(_fit(value, name, tuple(param.shape)))

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        assert isinstance(config, MultitaskPromptTuningConfig)
        return cls(config, word_embeddings)

    @override
    def prompts(self, batch: int, tasks: Tensor | None) -> Tensor:
        return self(self.indices().repeat(batch, 1), tasks)

    @override
    def forward(self, indices: Tensor, task_ids: Tensor | None = None) -> Tensor:
        """The prompt of each row's task: ``(B, rows)`` ids and ``(B,)`` task ids to ``(B, rows, token_dim)``.

        Raises
        ------
        ValueError
            ``task_ids`` is missing, or is not one integer per row.
        IndexError
            A task id is not in ``[0, num_tasks)``.
        """
        if task_ids is None:
            raise ValueError("task_ids cannot be None")
        if task_ids.ndim != 1 or int(task_ids.shape[0]) != int(indices.shape[0]):
            raise ValueError(
                f"task_ids must hold one task per row ({int(indices.shape[0])}), "
                f"got shape {tuple(task_ids.shape)}"
            )
        if task_ids.is_floating_point() or task_ids.is_complex():
            raise ValueError(f"task_ids must be integers, got {task_ids.dtype}")
        if int(task_ids.shape[0]) and (
            int(task_ids.min().item()) < 0 or int(task_ids.max().item()) >= self.num_tasks
        ):
            raise IndexError(f"task_ids must lie in [0, {self.num_tasks}), got {task_ids.tolist()}")
        tasks = task_ids.long()
        factors = self.prefix_task_cols.index_select(0, tasks) @ self.prefix_task_rows.index_select(
            0, tasks
        )
        prompt: Tensor = self.embedding(indices) * factors
        return prompt


class MultitaskPromptTuningModel(_PromptLearningTuner, peft_type=PeftType.MULTITASK_PROMPT_TUNING):
    """The tuner of multitask prompt tuning: a ``MultitaskPromptEmbedding`` per adapter, in front of the input."""

    _config_class: ClassVar[type[PromptLearningConfig]] = MultitaskPromptTuningConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = MultitaskPromptEmbedding
