"""``LoraModel``: the LoRA half of a tuner — its layers, ranks and trainable biases.

Everything that is not LoRA — planning, applying, freezing, merging,
unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``.
"""

import re
import warnings
from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftConfig, PeftType, _pattern_key_regex, _TargetedConfig
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft.tuners._base import _BaseTuner, _join, _named, _Target
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.lora._layer import LoraLayer


def _as_lora(config: PeftConfig) -> LoraConfig:
    # ``_BaseTuner.__init__`` has checked it against ``_config_class``.
    assert isinstance(config, LoraConfig)
    return config


def _pattern_value[V](patterns: Mapping[str, V], key: str, default: V) -> V:
    if key in patterns:
        return patterns[key]
    for pattern, value in patterns.items():
        if re.match(_pattern_key_regex(pattern), key):
            return value
    return default


class LoraModel(_BaseTuner, peft_type=PeftType.LORA):
    r"""A model with LoRA layers put into it, and the tools to take them out.

    Construction replaces each module ``config`` selects with a
    :class:`~lucid.peft.tuners.lora.Linear` around it and each
    ``modules_to_save`` entry with a frozen-original / trainable-copy
    pair, then freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is
    read from it.  ``config`` itself is not changed: ``peft_config``
    holds a copy, with ``"all-linear"`` spelled out as the names it chose.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : LoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``lora_A`` / ``lora_B``.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an
        ``nn.Linear``, or ``modules_to_save`` matches nothing.
    NotImplementedError
        A match is a layer type the reference adapts but v1 does not
        (``nn.Embedding``, convolutions, ``nn.MultiheadAttention``), or the
        model already holds a LoRA adapter (v1 keeps one per model).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, LoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> lora = LoraModel(net, LoraConfig(r=2, target_modules=["0"]))
    >>> lora.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].lora_A["default"].weight.requires_grad
    (False, True)
    >>> lora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = LoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = LoraConfig
    _later_types: ClassVar[tuple[type[nn.Module], ...]] = (
        nn.Embedding,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
    )

    def __init__(self, model: nn.Module, config: LoraConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_lora(super()._prepare_config(config, model))
        if stored.fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
                "Setting fan_in_fan_out to False.",
                stacklevel=4,
            )
            stored.fan_in_fan_out = False
        return stored

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        lora = _as_lora(config)
        layer, module = target.layer, target.module
        # Planning took both from ``LoraLayer``'s table, which holds nn.Linear layers.
        assert issubclass(layer, LoraLayer) and isinstance(module, nn.Linear)
        return layer(
            module,
            adapter_name,
            r=_pattern_value(lora.rank_pattern, target.key, lora.r),
            lora_alpha=_pattern_value(lora.alpha_pattern, target.key, lora.lora_alpha),
            lora_dropout=lora.lora_dropout,
            init_lora_weights=lora.init_lora_weights,
            use_rslora=lora.use_rslora,
        )

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """The parameters ``bias`` makes trainable, by name under ``model``.

        ``"all"`` is every parameter named ``bias`` outside a module trained
        in full (that copy trains whole anyway, and its original stays
        frozen); ``"lora_only"`` the bias of each adapted layer.
        """
        bias = _as_lora(config).bias
        if bias == "lora_only":
            return [
                _join(name, "base_layer.bias")
                for name, layer in model.named_modules()
                if isinstance(layer, LoraLayer) and layer.bias is not None
            ]
        if bias == "none":
            return []
        wrapped = tuple(f"{name}." for name, _ in _named(model, ModulesToSaveWrapper))
        return [
            name
            for name, _ in model.named_parameters()
            if name.rpartition(".")[2] == "bias" and not name.startswith(wrapped)
        ]

    @override
    def disable_adapter_layers(self) -> None:
        """Run the base model alone: LoRA updates off, original ``modules_to_save``.

        With ``bias`` other than ``"none"`` the trained biases are base
        parameters and stay as trained, so the output is not the base
        model's; a warning says so.
        """
        bias = _as_lora(self.peft_config[self.active_adapter]).bias
        if bias != "none":
            warnings.warn(
                f"Careful, disabling adapter layers with bias configured to be '{bias}' does not "
                "produce the same output as the base model would without adaption.",
                stacklevel=2,
            )
        super().disable_adapter_layers()
