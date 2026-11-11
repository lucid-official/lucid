"""``GraloraModel``: the GraLoRA tuner — its config and layer family; the rest is the generic tuner's."""

from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.gralora._config import GraloraConfig
from lucid.peft.tuners.gralora._layer import GraloraLayer


class GraloraModel(_ConfigTuner[GraloraConfig], peft_type=PeftType.GRALORA):
    r"""A model with GraLoRA layers in it, and the tools to take them out again.

    Each module ``config`` selects becomes a :class:`~lucid.peft.tuners.gralora.Linear` that adds
    a grid of ``gralora_k`` by ``gralora_k`` low-rank blocks (and, with ``hybrid_r``, a plain LoRA)
    to the module's output; the rest of the model is frozen.  A selected module's input and output
    sizes must be divisible by ``gralora_k``.  The model is edited in place and stays reachable as
    ``model``; several adapters coexist (``add_adapter``, ``set_adapter``, ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to put GraLoRA layers into.
    peft_config : GraloraConfig
        Rank, grid size, hybrid rank, dropout and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``gralora_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` matches no module, a match is not ``nn.Linear``, ``gralora_k`` does not
        divide a match's sizes, or ``modules_to_save`` names nothing.
    NotImplementedError
        Another method's layers are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import GraloraConfig, GraloraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> gralora = GraloraModel(net, GraloraConfig(r=4, gralora_k=2, target_modules=["0"]), "default")
    >>> gralora.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].gralora_A["default"].requires_grad
    (False, True)
    >>> gralora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = GraloraLayer._layer_types

    @override
    @classmethod
    def _prepare_config(cls, config: GraloraConfig, model: nn.Module) -> GraloraConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored
