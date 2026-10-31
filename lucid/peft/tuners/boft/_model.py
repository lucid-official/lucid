"""``BOFTModel``: the BOFT tuner — its config and layer family; the rest is the generic tuner's."""

from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.boft._config import BOFTConfig
from lucid.peft.tuners.boft._layer import BOFTLayer


class BOFTModel(_ConfigTuner, peft_type=PeftType.BOFT):
    r"""A model with BOFT layers in it, and the tools to take them out again.

    A module that ``config`` selects is replaced by a
    :class:`~lucid.peft.tuners.boft.Linear` or ``Conv2d`` (chosen by its type) that
    holds the butterfly rotation and the row scale; all other parameters are
    frozen.  The change is made in place, ``model`` keeps the wrapped model
    and answers for its public names, and ``peft_config`` holds ``config`` with the
    default targets resolved.  Add further adapters with ``add_adapter`` and
    choose among them with ``set_adapter``.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : BOFTConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in ``boft_R`` and ``boft_s``.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match has no BOFT layer type, the
        block sizes do not fit a match, or ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers of another method, or a match is a grouped convolution.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import BOFTConfig, BOFTModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> boft = BOFTModel(net, BOFTConfig(boft_block_size=4, target_modules=["0"]))
    >>> boft.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].boft_R["default"].requires_grad
    (False, True)
    >>> boft.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = BOFTLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = BOFTConfig

    def __init__(self, model: nn.Module, config: BOFTConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = super()._prepare_config(config, model)
        assert isinstance(stored, BOFTConfig)
        _reset_fan_in_fan_out(stored)
        return stored
