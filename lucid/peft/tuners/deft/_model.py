"""``DeftModel``: the DEFT tuner — its config and layer family; the rest is the generic tuner's."""

from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.deft._config import DeftConfig
from lucid.peft.tuners.deft._layer import DeftLayer


class DeftModel(_ConfigTuner[DeftConfig], peft_type=PeftType.DEFT):
    r"""A model with DEFT layers in it, and the tools to take them out again.

    Each ``nn.Linear`` ``config`` selects becomes a :class:`~lucid.peft.tuners.deft.Linear` whose
    weight loses a learned subspace and gains a low-rank injection into it,
    :math:`(I - P_{proj})\,W + Q_P R`; the rest of the model is frozen.  The rank is capped at
    a module's output size.  The model is edited in place and stays reachable as ``model``;
    several adapters coexist (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to put DEFT layers into.
    peft_config : DeftConfig
        Rank, decomposition, scale, dropout and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``deft_P``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear``, or no module matches
        ``modules_to_save``.
    NotImplementedError
        The model is already adapted by another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import DeftConfig, DeftModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> deft = DeftModel(net, DeftConfig(r=4, target_modules=["0"]), "default")
    >>> deft.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].deft_P["default"].requires_grad
    (False, True)
    >>> deft.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = DeftLayer._layer_types

    @override
    @classmethod
    def _prepare_config(cls, config: DeftConfig, model: nn.Module) -> DeftConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored
