"""``OFTModel``: the OFT tuner — its config, its layer family; the rest is the generic tuner's."""

from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.oft._config import OFTConfig
from lucid.peft.tuners.oft._layer import OFTLayer


class OFTModel(_ConfigTuner, peft_type=PeftType.OFT):
    r"""A model with OFT layers in it, and the tools to take them out again.

    The modules ``config`` selects each become an
    :class:`~lucid.peft.tuners.oft.Linear` or ``Conv2d`` (by their type) that
    rotates its input; everything else is frozen.  ``model`` is changed in place
    and kept; a public name this wrapper lacks is read from it.  The stored
    ``peft_config`` has the default targets spelled out, ``config`` itself is left alone.
    Adapters are added, switched and removed with ``add_adapter``, ``set_adapter``
    and ``delete_adapter``.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : OFTConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``oft_R``.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match has no OFT layer type, or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method, or a match is a grouped convolution.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import OFTConfig, OFTModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> oft = OFTModel(net, OFTConfig(oft_block_size=4, target_modules=["0"]))
    >>> oft.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].oft_R["default"].requires_grad
    (False, True)
    >>> oft.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = OFTLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = OFTConfig

    def __init__(self, model: nn.Module, config: OFTConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = super()._prepare_config(config, model)
        assert isinstance(stored, OFTConfig)
        _reset_fan_in_fan_out(stored)
        return stored
