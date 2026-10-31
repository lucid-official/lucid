"""``HRAModel``: the HRA tuner — its config and layer family; the rest is the generic tuner's."""

from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.hra._config import HRAConfig
from lucid.peft.tuners.hra._layer import HRALayer


class HRAModel(_ConfigTuner, peft_type=PeftType.HRA):
    r"""A model with HRA layers in it, and the tools to take them out again.

    Every module ``config`` selects is wrapped in an
    :class:`~lucid.peft.tuners.hra.Linear` or ``Conv2d`` (by its type) whose
    weight a chain of reflections multiplies, and the rest of the model is frozen.
    The model is changed in place and stays reachable as ``model``, with unknown
    public names read from it.  ``peft_config`` stores ``config`` with its default
    targets filled in.  Several adapters can coexist (``add_adapter``,
    ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : HRAConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``hra_u``.

    Raises
    ------
    ValueError
        ``target_modules`` matches no module, or a match is neither linear nor
        convolutional, or ``modules_to_save`` names nothing.
    NotImplementedError
        Another method's layers are already in the model, or a match is a grouped convolution.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import HRAConfig, HRAModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> hra = HRAModel(net, HRAConfig(r=4, target_modules=["0"]))
    >>> hra.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].hra_u["default"].requires_grad
    (False, True)
    >>> hra.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = HRALayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = HRAConfig

    def __init__(self, model: nn.Module, config: HRAConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)
