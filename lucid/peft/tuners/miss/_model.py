"""``MissModel``: the MiSS tuner — its config and layer family; the rest is the generic tuner's."""

from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.miss._config import MissConfig
from lucid.peft.tuners.miss._layer import MissLayer


class MissModel(_ConfigTuner, peft_type=PeftType.MISS):
    r"""A model with MiSS layers in it, and the tools to take them out again.

    A module ``config`` selects becomes a :class:`~lucid.peft.tuners.miss.Linear` around it,
    its weight rewritten by the shard block; everything else is frozen.  The model is edited
    in place (reachable as ``model``; public names this wrapper lacks are read from it) and
    ``peft_config`` keeps ``config`` with the default targets spelled out.  Adapters
    coexist under names (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : MissConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``miss_block``.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing, selects a non-linear module, selects one whose
        sizes the chosen variant cannot take, or ``modules_to_save`` matches nothing.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import MissConfig, MissModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> miss = MissModel(net, MissConfig(r=4, target_modules=["0"]))
    >>> miss.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].miss_block["default"].requires_grad
    (False, True)
    >>> miss.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = MissLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = MissConfig

    def __init__(self, model: nn.Module, config: MissConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)
