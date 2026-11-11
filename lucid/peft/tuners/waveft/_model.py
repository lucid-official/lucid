"""``WaveFTModel``: the WaveFT tuner — its config and layer family, and how many coefficients each layer is given."""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._plan import _Placement, _Target
from lucid.peft.tuners._shared import _base_linear
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.waveft._config import WaveFTConfig
from lucid.peft.tuners.waveft._defaults import WAVEFT_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.waveft._layer import WaveFTLayer


def _allocated(config: WaveFTConfig, targets: list[_Target]) -> WaveFTConfig:
    """``config``, with ``n_frequency_pattern`` naming each target's share when ``proportional_parameters`` asks for one.

    Every target gets ``n_frequency`` times the number of targets, split in proportion to the size of
    its weight and at least one: a larger layer gets more coefficients, and the total is the same as
    giving each ``n_frequency``.  The shares are keyed by the targets' own names, which an exact key
    reads before any pattern does.
    """
    if not config.proportional_parameters:
        return config
    sizes = {}
    for target in targets:
        weight = _base_linear(target.module).weight
        sizes[target.full_name] = int(weight.shape[0]) * int(weight.shape[1])
    budget = config.n_frequency * len(sizes)
    total = sum(sizes.values())
    shares = {name: max(1, round(size / total * budget)) for name, size in sizes.items()}
    return dataclasses.replace(config, n_frequency_pattern=shares)


def _for_module(config: WaveFTConfig, key: str) -> WaveFTConfig:
    """``config`` with the number of coefficients the module named ``key`` is given by ``n_frequency_pattern``."""
    return dataclasses.replace(
        config,
        n_frequency=_pattern_value(config.n_frequency_pattern or {}, key, config.n_frequency),
        n_frequency_pattern={},
        proportional_parameters=False,
    )


class WaveFTModel(_ConfigTuner[WaveFTConfig], peft_type=PeftType.WAVEFT):
    r"""A model whose chosen linear layers train a few wavelet coefficients of their weight update, and the tools to undo that.

    Each ``nn.Linear`` that ``config`` selects becomes a :class:`~lucid.peft.tuners.waveft.Linear`
    that adds an update rebuilt by an inverse wavelet transform from ``n_frequency`` trained
    coefficients; every other parameter is frozen.  A layer's coefficient count is the config's unless
    ``n_frequency_pattern`` names it, or ``proportional_parameters`` shares the total by weight
    size.  The model is edited in place and stays reachable as ``model``; several adapters coexist
    (``add_adapter``, ``set_adapter``, ``delete_adapter``) and their updates add.

    Parameters
    ----------
    model : nn.Module
        The model whose weight updates are parametrised.
    peft_config : WaveFTConfig
        Which modules, the wavelet and the coefficient counts.
    adapter_name : str
        Name of the first adapter, the key in every ``waveft_spectrum``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or a module that is not an ``nn.Linear``; a layer is given
        more coefficients than its weight has entries; ``use_idwt`` names a wavelet without filters;
        ``modules_to_save`` selects nothing.
    NotImplementedError
        The model already holds layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import WaveFTConfig, WaveFTModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> wave = WaveFTModel(net, WaveFTConfig(n_frequency=6, target_modules=["0"]), "default")
    >>> wave.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].waveft_spectrum["default"].requires_grad
    (False, True)
    >>> wave.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = WaveFTLayer._layer_types
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = WAVEFT_DEFAULT_TARGET_MODULES

    @override
    @classmethod
    def _prepare_config(cls, config: WaveFTConfig, model: nn.Module) -> WaveFTConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    def _new_slots(
        self, config: WaveFTConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        return super()._new_slots(_allocated(config, targets), adapter_name, targets, saved)

    @override
    def _attach_adapter(
        self,
        config: WaveFTConfig,
        adapter_name: str,
        targets: list[_Target],
        saved: list[str],
        placements: list[_Placement],
    ) -> None:
        super()._attach_adapter(
            _allocated(config, targets), adapter_name, targets, saved, placements
        )

    @override
    @classmethod
    def _create_new_module(
        cls, config: WaveFTConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return super()._create_new_module(
            _for_module(config, target.full_name), adapter_name, target
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: WaveFTConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(_for_module(config, target.full_name), adapter_name, target)
