"""``C3AModel``: the C3A tuner — its config and layer family, and the block size each module is given."""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.c3a._config import C3AConfig
from lucid.peft.tuners.c3a._defaults import C3A_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.c3a._layer import C3ALayer
from typing import override


class C3AModel(_ConfigTuner, peft_type=PeftType.C3A):
    r"""A model with C3A layers in it, and the tools to take them out again.

    A module ``config`` selects becomes a :class:`~lucid.peft.tuners.c3a.Linear` around it
    with a block-circulant update; everything else is frozen.  The model is edited in place
    (reachable as ``model``; public names this wrapper lacks are read from it) and
    ``peft_config`` keeps ``config`` with the default targets spelled out.  A module's block
    size is ``block_size_pattern``'s entry for it, else ``block_size``.  Adapters coexist
    under names (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    peft_config : C3AConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in every ``c3a_kernel``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or a non-linear module, a block size does not
        divide a module's sizes, or ``modules_to_save`` matches nothing.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import C3AConfig, C3AModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> c3a = C3AModel(net, C3AConfig(block_size=2, target_modules=["0"]), "default")
    >>> tuple(net[0].c3a_kernel["default"].shape)
    (4, 4, 2)
    >>> net[0].base_layer.weight.requires_grad, net[0].c3a_kernel["default"].requires_grad
    (False, True)
    >>> c3a.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = C3ALayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = C3AConfig
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = C3A_DEFAULT_TARGET_MODULES

    @staticmethod
    def _sized(config: _TargetedConfig, key: str) -> _TargetedConfig:
        """``config`` with the block size the module named ``key`` gets."""
        assert isinstance(config, C3AConfig)
        size = _pattern_value(config.block_size_pattern, key, config.block_size)
        return config if size == config.block_size else dataclasses.replace(config, block_size=size)

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return super()._create_new_module(
            cls._sized(config, target.full_name), adapter_name, target
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(cls._sized(config, target.full_name), adapter_name, target)
