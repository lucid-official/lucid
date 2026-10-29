"""The two ways in: ``get_peft_model`` (wrapped) and ``inject_adapter_in_model`` (bare)."""

import dataclasses

import lucid.nn as nn
from lucid.peft._config import LoraConfig, TaskType
from lucid.peft._peft_model import PeftModel
from lucid.peft._targeting import _module_names_matching
from lucid.peft.tuners.lora._model import LoraModel

#: Heads the reference's task wrappers train in full, by task.
_TASK_HEADS: dict[TaskType, tuple[str, ...]] = {
    TaskType.SEQ_CLS: ("classifier", "score"),
    TaskType.TOKEN_CLS: ("classifier", "score"),
    TaskType.QUESTION_ANS: ("qa_outputs",),
}


def _with_task_heads(model: nn.Module, config: LoraConfig) -> list[str] | None:
    """``modules_to_save`` plus the task's conventional head, if ``model`` has one."""
    saved = config.modules_to_save
    if config.task_type is None:
        return saved
    heads = _TASK_HEADS.get(TaskType(config.task_type), ())
    present = [head for head in heads if _module_names_matching(model, [head])]
    added = [head for head in present if head not in (saved or [])]
    return [*(saved or []), *added] if added else saved


def get_peft_model(
    model: nn.Module, peft_config: LoraConfig, adapter_name: str = "default"
) -> PeftModel:
    r"""Adapt ``model`` with LoRA and return it wrapped as a :class:`~lucid.peft.PeftModel`.

    The model is changed in place: each module ``peft_config`` selects
    becomes a LoRA layer around the original, and every parameter but the
    adapter's is frozen.  ``peft_config`` is not changed — the wrapper
    keeps a copy — with two additions the reference makes too: a
    ``base_model_name_or_path`` read from the model's ``name_or_path``,
    when it has one, and, for a classification or question-answering
    ``task_type``, the model's ``classifier`` / ``score`` / ``qa_outputs``
    head added to ``modules_to_save`` so that it trains.

    Parameters
    ----------
    model : nn.Module
        The model to adapt — any module; a zoo model keeps its
        ``generate`` and heads through the wrapper.
    peft_config : LoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``lora_A`` / ``lora_B``.

    Returns
    -------
    PeftModel
        The adapted model.

    Raises
    ------
    TypeError
        ``peft_config`` is not a ``LoraConfig``.
    ValueError
        No module matches ``target_modules``, or a match is not an
        ``nn.Linear``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_peft_model
    >>> net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    >>> peft = get_peft_model(net, LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"]))
    >>> peft.print_trainable_parameters()
    trainable params: 208 || all params: 548 || trainable%: 37.9562
    >>> peft(lucid.randn(2, 16)).shape
    (2, 4)
    """
    if not isinstance(peft_config, LoraConfig):
        raise TypeError(
            f"peft_config must be a LoraConfig (lucid.peft implements LoRA), "
            f"got {type(peft_config).__name__}"
        )
    name_or_path = model.__dict__.get("name_or_path")
    config = dataclasses.replace(
        peft_config,
        base_model_name_or_path=(
            name_or_path if isinstance(name_or_path, str) else peft_config.base_model_name_or_path
        ),
        modules_to_save=_with_task_heads(model, peft_config),
    )
    return PeftModel(model, config, adapter_name)


def inject_adapter_in_model(
    peft_config: LoraConfig, model: nn.Module, adapter_name: str = "default"
) -> nn.Module:
    r"""Put LoRA layers into ``model`` in place and return it, without a wrapper.

    For code that manages the model itself: the model is adapted and
    frozen exactly as by :func:`~lucid.peft.get_peft_model`, but returned
    bare, so its state dict keeps its own key names (plus ``base_layer`` /
    ``lora_A`` / ``lora_B`` under each adapted layer).
    :func:`~lucid.peft.get_peft_model_state_dict` reads the adapter out
    of it.  Note the argument order, the reference's: config first.

    Parameters
    ----------
    peft_config : LoraConfig
        What to adapt and how.
    model : nn.Module
        The model to adapt.
    adapter_name : str, default="default"
        Name of the adapter.

    Returns
    -------
    nn.Module
        ``model`` itself, adapted.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, inject_adapter_in_model
    >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    >>> inject_adapter_in_model(LoraConfig(r=2, target_modules=["1"]), net) is net
    True
    >>> sorted(net.state_dict())
    ['0.bias', '0.weight', '1.base_layer.bias', '1.base_layer.weight', '1.lora_A.default.weight', '1.lora_B.default.weight']
    """
    LoraModel(model, peft_config, adapter_name)
    return model
