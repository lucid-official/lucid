"""The two ways in: ``get_peft_model`` (wrapped) and ``inject_adapter_in_model`` (bare)."""

import lucid.nn as nn
from lucid.peft._config import PeftConfig, _require_config
from lucid.peft._peft_model import PeftModel
from lucid.peft._targeting import _config_for_model
from lucid.peft.tuners._base import _tuner_for


def get_peft_model(
    model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default"
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
    config = _config_for_model(model, _require_config("peft_config", peft_config))
    return PeftModel._model_class(config.task_type)(model, config, adapter_name)


def inject_adapter_in_model(
    peft_config: PeftConfig, model: nn.Module, adapter_name: str = "default"
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
    config = _require_config("peft_config", peft_config)
    _tuner_for(config)._adapt_in_place(model, config, adapter_name)
    return model
