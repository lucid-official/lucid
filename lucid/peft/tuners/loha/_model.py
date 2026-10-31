"""``LoHaModel``: the LoHa tuner — its config, its layer family; the rest is the LyCORIS tuner's."""

from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._lycoris import _LycorisTuner
from lucid.peft.tuners.loha._config import LoHaConfig
from lucid.peft.tuners.loha._layer import LoHaLayer


class LoHaModel(_LycorisTuner, peft_type=PeftType.LOHA):
    r"""A model with LoHa layers in it, and the tools to take them out again.

    Each module ``config`` selects is replaced by a LoHa layer around it —
    :class:`~lucid.peft.tuners.loha.Linear`, ``Conv1d`` or ``Conv2d`` by the
    module's type — and every other parameter is frozen.  The model is changed
    in place and kept as ``model``; a public name this wrapper lacks is read
    from it.  ``peft_config`` holds a copy of ``config`` with the default
    ``target_modules`` and ``"all-linear"`` spelled out; ``config`` is untouched.

    Any number of adapters live side by side, each under its own name: see
    ``add_adapter``, ``set_adapter`` and ``delete_adapter``.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : LoHaConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``hada_*`` table.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match has no LoHa layer type, or
        ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers; wrap it once, then ``add_adapter``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoHaConfig, LoHaModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> loha = LoHaModel(net, LoHaConfig(r=2, target_modules=["0"]))
    >>> loha.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].hada_w1_a["default"].requires_grad
    (False, True)
    >>> loha.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = LoHaLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = LoHaConfig

    def __init__(self, model: nn.Module, config: LoHaConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)
