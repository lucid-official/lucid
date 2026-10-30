"""An adapter's own tensors: what ``save_pretrained`` writes and ``from_pretrained`` reads.

Keys on disk are the reference library's: the in-memory key with the
adapter name taken out (``...lora_A.default.weight`` becomes
``...lora_A.weight``, ``...classifier.modules_to_save.default.weight``
becomes ``...classifier.weight``).  The map between the two is read off
the model's structure rather than by string surgery, so a module whose
own name contains ``lora_`` or the adapter's name is not misread.
"""

import warnings
from collections.abc import Mapping

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import LoraConfig
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft.tuners.lora._layer import LoraLayer
from lucid.peft.tuners.lora._model import _join, _learnable_bias_names, _named


def _peft_config_of(model: nn.Module, adapter_name: str) -> LoraConfig:
    configs = getattr(model, "peft_config", None)
    config = configs.get(adapter_name) if isinstance(configs, dict) else None
    if isinstance(config, LoraConfig):
        return config
    raise ValueError(
        f"the model holds no LoRA adapter named {adapter_name!r} "
        "(add one with get_peft_model or inject_adapter_in_model)"
    )


def _adapter_keys(model: nn.Module, adapter_name: str) -> dict[str, str]:
    """In-memory state-dict key → on-disk key, for every tensor the adapter owns.

    v1 holds one adapter per model, so every LoRA layer and every
    ``modules_to_save`` copy belongs to the adapter the config names.
    """
    config = _peft_config_of(model, adapter_name)
    layers = _named(model, LoraLayer)
    if not layers:
        raise RuntimeError(
            f"adapter {adapter_name!r} has been merged or unloaded: the model holds no "
            "LoRA layer left to save or load"
        )
    keys: dict[str, str] = {}
    for path, layer in layers:
        for kind, adapters in (("lora_A", layer.lora_A), ("lora_B", layer.lora_B)):
            for leaf in adapters[adapter_name].state_dict():
                keys[_join(path, f"{kind}.{adapter_name}.{leaf}")] = _join(path, f"{kind}.{leaf}")
    for path, wrapper in _named(model, ModulesToSaveWrapper):
        for leaf in wrapper.modules_to_save[adapter_name].state_dict():
            keys[_join(path, f"modules_to_save.{adapter_name}.{leaf}")] = _join(path, leaf)
    keys.update({name: name for name in _learnable_bias_names(model, config.bias)})
    return keys


def get_peft_model_state_dict(
    model: nn.Module, state_dict: Mapping[str, Tensor] | None = None, adapter_name: str = "default"
) -> dict[str, Tensor]:
    r"""Return the adapter's tensors under the keys an adapter file uses.

    That is ``lora_A`` / ``lora_B`` of every adapted layer, the trained
    copy of every ``modules_to_save`` module and the biases ``bias``
    trains — keyed as the reference writes them, the adapter name taken
    out.  Called on a ``PeftModel`` the keys start ``base_model.model.``,
    which is the reference's file layout.

    Parameters
    ----------
    model : nn.Module
        A ``PeftModel``, or a model adapted by ``inject_adapter_in_model``.
    state_dict : Mapping of str to Tensor, optional
        Read the values from here instead of ``model.state_dict()`` —
        keyed as that would be.
    adapter_name : str, default="default"
        The adapter to extract.

    Returns
    -------
    dict of str to Tensor
        On-disk key to tensor.

    Raises
    ------
    ValueError
        ``model`` holds no adapter named ``adapter_name``.
    RuntimeError
        The adapter was merged or unloaded, so no LoRA layer is left.
    KeyError
        ``state_dict`` lacks one of the adapter's tensors.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    >>> peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
    >>> sorted(get_peft_model_state_dict(peft))
    ['base_model.model.0.lora_A.weight', 'base_model.model.0.lora_B.weight']
    """
    keys = _adapter_keys(model, adapter_name)
    source = model.state_dict() if state_dict is None else state_dict
    absent = [memory for memory in keys if memory not in source]
    if absent:
        raise KeyError(f"state_dict lacks the adapter's tensors {absent}")
    return {disk: source[memory] for memory, disk in keys.items()}


def set_peft_model_state_dict(
    model: nn.Module, peft_model_state_dict: Mapping[str, Tensor], adapter_name: str = "default"
) -> object:
    r"""Load an adapter's tensors, keyed as an adapter file keys them, into ``model``.

    Each key is mapped back to where the tensor lives in memory (the
    adapter name put back in) and copied into the existing parameter,
    which keeps its dtype, device and identity.  Every key is checked
    before anything is copied: one that lands nowhere, or a shape that
    differs, raises with the model untouched — an adapter loaded into the
    wrong model must not pass for a fresh one.  An adapter tensor the dict
    does not supply is reported in a warning and keeps its value.

    Parameters
    ----------
    model : nn.Module
        A ``PeftModel``, or a model adapted by ``inject_adapter_in_model``.
    peft_model_state_dict : Mapping of str to Tensor
        On-disk key to tensor, as ``get_peft_model_state_dict`` returns.
    adapter_name : str, default="default"
        The adapter to load into.

    Returns
    -------
    object
        What ``model.load_state_dict(..., strict=False)`` returns, typed as
        that method declares it: the ``(missing_keys, unexpected_keys)``
        named tuple, whose ``missing_keys`` lists the base parameters (no
        adapter file holds them) and whose ``unexpected_keys`` is empty.

    Raises
    ------
    ValueError
        A key matches no tensor of the model, or a shape differs.
    TypeError
        A value is not a Tensor.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import (LoraConfig, get_peft_model,
    ...                         get_peft_model_state_dict, set_peft_model_state_dict)
    >>> make = lambda: get_peft_model(nn.Sequential(nn.Linear(4, 2)),
    ...                               LoraConfig(r=2, target_modules=["0"], init_lora_weights=False))
    >>> source, target = make(), make()
    >>> _ = set_peft_model_state_dict(target, get_peft_model_state_dict(source))
    >>> a = target.base_model.model[0].lora_A["default"].weight
    >>> bool((a == source.base_model.model[0].lora_A["default"].weight).all().item())
    True
    """
    keys = _adapter_keys(model, adapter_name)
    to_memory = {disk: memory for memory, disk in keys.items()}
    current = model.state_dict()
    translated: dict[str, Tensor] = {}
    refused: list[str] = []
    for key, value in peft_model_state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{key!r} holds {type(value).__name__}, not a Tensor")
        memory = to_memory.get(key, key)
        if memory not in current:
            refused.append(f"{key} (no such tensor in the model)")
        elif tuple(current[memory].shape) != tuple(value.shape):
            refused.append(
                f"{key} (shape {tuple(value.shape)}, model has {tuple(current[memory].shape)})"
            )
        else:
            translated[memory] = value
    if refused:
        raise ValueError(
            f"{len(refused)} adapter tensor(s) do not fit this model: " + "; ".join(refused[:8])
        )
    missing = [disk for memory, disk in keys.items() if memory not in translated]
    if missing:
        warnings.warn(
            f"Found missing adapter keys while loading the checkpoint: {missing}.", stacklevel=2
        )
    return model.load_state_dict(translated, strict=False)
