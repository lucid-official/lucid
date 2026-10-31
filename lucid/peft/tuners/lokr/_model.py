"""``LoKrModel``: the LoKr tuner — its config, its layer family; the rest is the LyCORIS tuner's."""

from typing import ClassVar

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._lycoris import _LycorisTuner
from lucid.peft.tuners.lokr._config import LoKrConfig
from lucid.peft.tuners.lokr._layer import LoKrLayer


class LoKrModel(_LycorisTuner, peft_type=PeftType.LOKR):
    r"""A model whose chosen modules carry a Kronecker-product update, and the tools to undo it.

    ``config`` picks the modules; each becomes a LoKr layer —
    :class:`~lucid.peft.tuners.lokr.Linear`, ``Conv1d`` or ``Conv2d`` by the
    module's type — wrapped around the original, which keeps its parameters
    and, like everything else outside the adapter, is frozen.  The model is
    modified in place and held as ``model``; public names this wrapper lacks
    are read from it.  ``peft_config`` stores a copy of ``config`` (the default
    ``target_modules`` and any ``"all-linear"`` written out as names).

    Adapters coexist under their own names (``add_adapter``, ``set_adapter``,
    ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : LoKrConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``lokr_*`` table.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a matched module is of a type LoKr
        has no layer for, or no module matches ``modules_to_save``.
    NotImplementedError
        The model already holds adapter layers; wrap it once, then ``add_adapter``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoKrConfig, LoKrModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> lokr = LoKrModel(net, LoKrConfig(r=2, target_modules=["0"]))
    >>> lokr.targeted_module_names
    ['0']
    >>> sorted(net[0].state_dict())
    ['base_layer.bias', 'base_layer.weight', 'lokr_w1.default', 'lokr_w2.default']
    >>> lokr.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = LoKrLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = LoKrConfig

    def __init__(self, model: nn.Module, config: LoKrConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)
