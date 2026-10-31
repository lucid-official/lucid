"""``FourierFTModel``: the FourierFT tuner — its config and layer family; the rest is the generic tuner's."""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._base import _Target
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.fourierft._config import FourierFTConfig
from lucid.peft.tuners.fourierft._defaults import FOURIERFT_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.fourierft._layer import FourierFTLayer


class FourierFTModel(_ConfigTuner, peft_type=PeftType.FOURIERFT):
    r"""A model with FourierFT layers in it, and the tools to take them out again.

    Modules named by ``config`` are wrapped in a
    :class:`~lucid.peft.tuners.fourierft.Linear` that adds the spectral update to
    their output while every other parameter stays frozen.  The model is edited
    in place and remains available as ``model``; ``peft_config`` stores ``config``
    with the model type's default targets resolved.  Adapters coexist under their own
    names (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : FourierFTConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``fourierft_spectrum``.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match is not an ``nn.Linear``,
        ``n_frequency`` is larger than a match's weight, or ``modules_to_save``
        matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import FourierFTConfig, FourierFTModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> fourier = FourierFTModel(net, FourierFTConfig(n_frequency=10, target_modules=["0"]))
    >>> fourier.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].fourierft_spectrum["default"].requires_grad
    (False, True)
    >>> fourier.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = FourierFTLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = FourierFTConfig
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = FOURIERFT_DEFAULT_TARGET_MODULES

    def __init__(
        self, model: nn.Module, config: FourierFTConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = super()._prepare_config(config, model)
        assert isinstance(stored, FourierFTConfig)
        _reset_fan_in_fan_out(stored)
        return stored

    @staticmethod
    def _config_for(config: _TargetedConfig, key: str) -> _TargetedConfig:
        """``config`` with the ``n_frequency`` the module named ``key`` is given (``n_frequency_pattern``, else the default)."""
        assert isinstance(config, FourierFTConfig)
        count = _pattern_value(config.n_frequency_pattern, key, config.n_frequency)
        return config if count == config.n_frequency else dataclasses.replace(config, n_frequency=count)

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return super()._create_new_module(cls._config_for(config, target.full_name), adapter_name, target)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(cls._config_for(config, target.full_name), adapter_name, target)
