"""The two ways in (``get_peft_model`` wrapped, ``inject_adapter_in_model`` bare) and the method tables."""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import _CONFIG_CLASSES, PeftConfig, PeftType, _require_config
from lucid.peft._peft_model import PeftModel
from lucid.peft._refusals import refuse_value
from lucid.peft._targeting import _config_for_model
from lucid.peft.tuners._base import _TUNERS, _AdapterHost, _tuner_for

#: The config class of each method, by ``PeftType``: a read-only view of the registry that
#: ``PeftConfig`` subclasses join as they are defined, so a method added later appears here.
PEFT_TYPE_TO_CONFIG_MAPPING: Mapping[PeftType, type[PeftConfig]] = MappingProxyType(_CONFIG_CLASSES)

#: The tuner (the model class that puts a method's adapters into a model) of each method, by
#: ``PeftType``: a read-only view of the tuner registry, like ``PEFT_TYPE_TO_CONFIG_MAPPING``.
PEFT_TYPE_TO_TUNER_MAPPING: Mapping[PeftType, type[_AdapterHost]] = MappingProxyType(_TUNERS)


def get_peft_config(config_dict: dict[str, object]) -> PeftConfig:
    r"""Build the adapter config a dict describes, of the class its ``peft_type`` names.

    The dict is the shape ``PeftConfig.to_dict`` returns and ``adapter_config.json``
    holds.  The class is looked up in ``PEFT_TYPE_TO_CONFIG_MAPPING`` and reads the dict
    as :meth:`~lucid.peft.PeftConfig.from_dict` does, so a key from the reference's file
    that this library has no field for is accepted at a value that turns its feature off.

    Parameters
    ----------
    config_dict : dict of str to object
        The config's fields, by name, with the method under ``"peft_type"``.

    Returns
    -------
    PeftConfig
        The validated config, of the method's own class.

    Raises
    ------
    KeyError
        ``config_dict`` has no ``"peft_type"``, or it names a method this library lacks.
    NotImplementedError
        The dict turns on a feature the library does not implement.

    Examples
    --------
    >>> from lucid.peft import get_peft_config
    >>> config = get_peft_config({"peft_type": "LORA", "r": 4, "target_modules": ["q"]})
    >>> type(config).__name__, config.r
    ('LoraConfig', 4)
    """
    named = config_dict["peft_type"]
    if not isinstance(named, str):
        raise TypeError(f"peft_type must be a str, got {type(named).__name__}")
    try:
        peft_type = PeftType(named)
    except ValueError:
        raise KeyError(
            f"{named!r} is not a method of lucid.peft; "
            f"the methods are {', '.join(PEFT_TYPE_TO_CONFIG_MAPPING)}"
        ) from None
    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_dict(config_dict)


def get_peft_model(
    model: nn.Module,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: str | None = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel:
    r"""Adapt ``model`` with the method ``peft_config`` names and return it wrapped as a :class:`~lucid.peft.PeftModel`.

    The model is changed in place: each module ``peft_config`` selects
    becomes an adapter layer of that method (LoRA, (IA)^3, LoHa, LoKr) around the
    original, and every parameter but the adapter's is frozen.  ``peft_config``
    is not changed — the wrapper keeps a copy — with two additions the reference makes too: a
    ``base_model_name_or_path`` read from the model's ``name_or_path``,
    when it has one, and, for a classification or question-answering
    ``task_type``, the model's ``classifier`` / ``score`` / ``qa_outputs``
    head added to ``modules_to_save`` so that it trains.

    The class returned follows ``peft_config.task_type``:
    ``PeftModelForCausalLM`` for ``"CAUSAL_LM"``, ``PeftModelForSeq2SeqLM``,
    ``PeftModelForSequenceClassification``, ``PeftModelForTokenClassification``,
    ``PeftModelForQuestionAnswering`` and ``PeftModelForFeatureExtraction``
    for theirs (``MODEL_TYPE_TO_PEFT_MODEL_MAPPING`` is the table), and a
    plain ``PeftModel`` when the config has no task.

    Parameters
    ----------
    model : nn.Module
        The model to adapt — any module; a zoo model keeps its
        ``generate`` and heads through the wrapper.
    peft_config : PeftConfig
        What to adapt and how — a ``LoraConfig``, ``IA3Config``, ``LoHaConfig`` or
        ``LoKrConfig``; its method picks the tuner.
    adapter_name : str, default="default"
        Name of the adapter, the key in each of its tables (``lora_A``, ``ia3_l`` …).
    mixed : bool, default=False
        The reference returns a ``PeftMixedModel`` for ``True``, which holds adapters of
        different methods at once.  Not implemented: ``True`` raises ``NotImplementedError``.
    autocast_adapter_dtype : bool, default=True
        Cast a half-precision base's adapter weights to float32.  Always the case here;
        ``False`` raises ``NotImplementedError``.
    revision : str, optional
        The revision of the base model, recorded in the config's ``revision`` (metadata) when
        given.
    low_cpu_mem_usage : bool, default=False
        Build the adapter on the meta device.  Not implemented: ``True`` raises
        ``NotImplementedError``.

    Returns
    -------
    PeftModel
        The adapted model.

    Raises
    ------
    TypeError
        ``peft_config`` is not an adapter config.
    ValueError
        No module matches ``target_modules``, or a match is not an
        ``nn.Linear``.
    NotImplementedError
        ``mixed``, ``low_cpu_mem_usage`` or ``autocast_adapter_dtype`` holds a value the
        library cannot honour.

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
    refuse_value("get_peft_model", "mixed", mixed, False, "PeftMixedModel is not implemented")
    config = _config_for_model(model, _require_config("peft_config", peft_config))
    if revision is not None:
        config = dataclasses.replace(config, revision=revision)
    return PeftModel._model_class(config.task_type)(
        model,
        config,
        adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def inject_adapter_in_model(
    peft_config: PeftConfig,
    model: nn.Module,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
    state_dict: dict[str, Tensor] | None = None,
) -> nn.Module:
    r"""Put adapter layers into ``model`` in place and return it, without a wrapper.

    For code that manages the model itself: the model is adapted and
    frozen exactly as by :func:`~lucid.peft.get_peft_model`, but returned
    bare, so its state dict keeps its own key names (plus ``base_layer`` /
    ``lora_A`` / ``lora_B`` under each adapted layer).
    :func:`~lucid.peft.get_peft_model_state_dict` reads the adapter out
    of it.  Note the argument order, the reference's: config first.

    Parameters
    ----------
    peft_config : PeftConfig
        What to adapt and how — a ``LoraConfig``.
    model : nn.Module
        The model to adapt.
    adapter_name : str, default="default"
        Name of the adapter.
    low_cpu_mem_usage : bool, default=False
        Build the adapter on the meta device.  Not implemented: ``True`` raises
        ``NotImplementedError``.
    state_dict : dict of str to Tensor, optional
        The reference matches targets against these keys instead of the model's modules
        (for a model on the meta device).  Not implemented: anything but ``None`` raises
        ``NotImplementedError``.

    Returns
    -------
    nn.Module
        ``model`` itself, adapted.

    Raises
    ------
    NotImplementedError
        ``low_cpu_mem_usage`` or ``state_dict`` holds a value the library cannot honour.

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
    config = _require_config("peft_config", peft_config)
    _tuner_for(config)._adapt_in_place(model, config, adapter_name, low_cpu_mem_usage, state_dict)
    return model
