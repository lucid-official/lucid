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
from lucid.peft._config import PeftConfig
from lucid.peft._slots import _adapter_slots
from lucid.peft.tuners._base import _AdapterHost, _join, _tuner_for
from lucid.peft.tuners._files import _FileContext, _FileFormat


def _peft_config_of(model: nn.Module, adapter_name: str) -> PeftConfig:
    configs = getattr(model, "peft_config", None)
    held = configs if isinstance(configs, dict) else {}
    config = held.get(adapter_name)
    if isinstance(config, PeftConfig):
        return config
    names = ", ".join(f"{name!r} ({c.peft_type})" for name, c in held.items())
    detail = f" (it holds {names})" if names else ""
    raise ValueError(
        f"the model holds no adapter named {adapter_name!r}{detail}; "
        "add one with get_peft_model or inject_adapter_in_model"
    )


def _file_context(model: nn.Module, adapter_name: str) -> _FileContext:
    """Where ``adapter_name``'s tensors sit in ``model``, which may be a ``PeftModel``, a tuner or a bare model."""
    config = _peft_config_of(model, adapter_name)
    wrapper = getattr(model, "base_model", None)
    if isinstance(wrapper, _AdapterHost):
        host, prefix = "base_model.", f"base_model.{wrapper._model_path}"
    elif isinstance(model, _AdapterHost):
        host, prefix = "", model._model_path
    else:
        host = prefix = ""
    return _FileContext(model, prefix, config, adapter_name, host)


def _adapter_keys(context: _FileContext) -> dict[str, str]:
    """In-memory state-dict key → on-disk key, for every tensor the adapter owns.

    Each slot — tuner layer or ``modules_to_save`` copy — maps its own
    keys for the adapter, and none if it holds nothing of it: an adapter
    need not reach every layer another one does.  The parameters the method
    also trains (the biases ``bias`` picks) keep their names, and what the
    method keeps outside its layers (``_shared_keys``) is added.
    """
    model, config, adapter_name = context.model, context.config, context.adapter
    tuner = _tuner_for(config)
    tuner._require_adapter_state(model, adapter_name)
    keys = {
        _join(path, own): _join(path, disk)
        for path, slot in _adapter_slots(model)
        for own, disk in slot._adapter_keys(adapter_name).items()
    }
    keys.update({n: n for n in tuner._extra_trainable_names(model, config)})
    if issubclass(tuner, _FileFormat):
        keys.update(tuner._shared_keys(context))
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
        The adapter was merged or unloaded, so no adapter layer is left.
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
    context = _file_context(model, adapter_name)
    keys = _adapter_keys(context)
    source = model.state_dict() if state_dict is None else state_dict
    absent = [memory for memory in keys if memory not in source]
    if absent:
        raise KeyError(f"state_dict lacks the adapter's tensors {absent}")
    state = {disk: source[memory] for memory, disk in keys.items()}
    tuner = _tuner_for(context.config)
    return tuner._to_file(context, state) if issubclass(tuner, _FileFormat) else state


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
    context = _file_context(model, adapter_name)
    tuner = _tuner_for(context.config)
    if issubclass(tuner, _FileFormat):
        peft_model_state_dict = tuner._from_file(context, peft_model_state_dict)
    keys = _adapter_keys(context)
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
