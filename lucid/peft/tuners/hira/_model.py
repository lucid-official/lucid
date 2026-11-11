"""``HiraModel``: the HiRA tuner — its config and layer family; the rest is the generic tuner's."""

import dataclasses
from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.hira._config import HiraConfig
from lucid.peft.tuners.hira._layer import HiraLayer


def _with_rank_of(config: HiraConfig, key: str) -> HiraConfig:
    """``config`` with the rank ``rank_pattern`` gives the module named ``key``."""
    rank = _pattern_value(config.rank_pattern or {}, key, config.r)
    return dataclasses.replace(config, r=rank, rank_pattern={})


class HiraModel(_ConfigTuner[HiraConfig], peft_type=PeftType.HIRA):
    r"""A model with HiRA layers in it, and the tools to take them out again.

    A linear, embedding or convolution module ``config`` selects is wrapped in the layer of its
    type (:class:`~lucid.peft.tuners.hira.Linear`, ``Embedding``, ``Conv1d`` / ``Conv2d`` /
    ``Conv3d``), which adds the frozen weight modulated by a low-rank matrix,
    :math:`W \odot BA`; the rest of the model is frozen.  A module's rank is the config's unless
    ``rank_pattern`` names it.  The model is edited in place and stays reachable as ``model``;
    several adapters coexist (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to put HiRA layers into.
    peft_config : HiraConfig
        Rank, dropout, initialisation and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``hira_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` matches no module, a match is none of the supported types, or
        ``modules_to_save`` names nothing.
    NotImplementedError
        Another method's layers are already in the model, or a match is a grouped convolution.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import HiraConfig, HiraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> hira = HiraModel(net, HiraConfig(r=4, target_modules=["0"]), "default")
    >>> hira.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].hira_A["default"].requires_grad
    (False, True)
    >>> hira.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = HiraLayer._layer_types

    @override
    @classmethod
    def _prepare_config(cls, config: HiraConfig, model: nn.Module) -> HiraConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    @classmethod
    def _create_new_module(
        cls, config: HiraConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return super()._create_new_module(
            _with_rank_of(config, target.full_name), adapter_name, target
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: HiraConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(_with_rank_of(config, target.full_name), adapter_name, target)
