"""``ShiraModel``: the SHiRA tuner — its config and layer family, and the index tensors its file carries."""

from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.shira._config import ShiraConfig
from lucid.peft.tuners.shira._layer import ShiraLayer


class ShiraModel(_FileFormat, _ConfigTuner, peft_type=PeftType.SHIRA):
    r"""A model with SHiRA layers in it, and the tools to take them out again.

    A module ``config`` selects becomes a :class:`~lucid.peft.tuners.shira.Linear` around it
    that trains a sparse subset of its weight; everything else is frozen.  The model is
    edited in place (reachable as ``model``; public names this wrapper lacks are read from
    it) and ``peft_config`` keeps ``config`` with the default targets spelled out.  Adapters
    coexist under names (``add_adapter``, ``set_adapter``, ``delete_adapter``).  The adapter
    file holds, per layer, the trained values and the indices they belong to.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : ShiraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``shira_weight``.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or a non-linear module, a mask does not hold
        ``r (d_out + d_in)`` entries, or ``modules_to_save`` matches nothing.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import ShiraConfig, ShiraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> shira = ShiraModel(net, ShiraConfig(r=1, random_seed=0, target_modules=["0"]))
    >>> shira.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].shira_weight["default"].requires_grad
    (False, True)
    >>> shira.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = ShiraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = ShiraConfig

    def __init__(
        self, model: nn.Module, config: ShiraConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = super()._prepare_config(config, model)
        assert isinstance(stored, ShiraConfig)
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """Each layer's indices, which no slot table holds: ``<layer>.shira_indices`` in the file."""
        keys: dict[str, str] = {}
        for path, layer in context.layers:
            if isinstance(layer, ShiraLayer) and context.adapter in layer.shira_indices:
                where = f"{context.prefix}{path}.shira_indices"
                keys[f"{where}.{context.adapter}"] = where
        return keys
