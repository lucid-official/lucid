"""``LNTuningModel``: the LN-tuning tuner — its config and layer; the rest is the generic tuner's."""

from collections.abc import Mapping
from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.ln_tuning._config import LNTuningConfig
from lucid.peft.tuners.ln_tuning._defaults import LN_TUNING_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.ln_tuning._layer import LNTuningLayer


class LNTuningModel(_ConfigTuner, peft_type=PeftType.LN_TUNING):
    """A model whose chosen normalisation layers train as copies, and the tools to take them out again.

    Each module ``config`` selects is replaced by an
    :class:`~lucid.peft.tuners.ln_tuning.LNTuningLayer` around it — the module
    and a trainable copy of it — and every other parameter is frozen.  The model
    is changed in place and kept as ``model``; a public name this wrapper lacks is
    read from it.  ``peft_config`` holds a copy of ``config`` with the default
    ``target_modules`` spelled out; ``config`` is untouched.

    Any number of adapters live side by side, each under its own name, but only
    one applies (and only one can be merged) at a time: see ``add_adapter``,
    ``set_adapter`` and ``delete_adapter``.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : LNTuningConfig
        What to adapt.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``ln_tuning_layers``.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, the model type has no default
        ``target_modules`` and the config names none, or ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LNTuningConfig, LNTuningModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    >>> tuned = LNTuningModel(net, LNTuningConfig(target_modules=["1"]))
    >>> tuned.targeted_module_names
    ['1']
    >>> net[1].base_layer.weight.requires_grad, net[1].ln_tuning_layers["default"].weight.requires_grad
    (False, True)
    >>> tuned.unload() is net, type(net[1]) is nn.LayerNorm
    (True, True)
    """

    _layer_types = LNTuningLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = LNTuningConfig
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = LN_TUNING_DEFAULT_TARGET_MODULES

    def __init__(
        self, model: nn.Module, config: LNTuningConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)
