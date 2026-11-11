"""``BeftModel``: the BEFT tuner — its config, layer family and default targets; the rest is the generic tuner's."""

from collections.abc import Mapping
from typing import ClassVar

from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.beft._config import BeftConfig
from lucid.peft.tuners.beft._defaults import BEFT_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.beft._layer import BeftLayer


class BeftModel(_ConfigTuner[BeftConfig], peft_type=PeftType.BEFT):
    r"""A model whose chosen linear layers carry a trained bias, and the tools to take them out again.

    Each ``nn.Linear`` that ``config`` selects becomes a :class:`~lucid.peft.tuners.beft.Linear`
    that adds one trained vector to its output; every other parameter is frozen, so the adapter is
    one vector of the output width per chosen layer.  Without ``target_modules`` the value
    projections of the model types the reference's table covers are taken.  The model is edited in
    place and stays reachable as ``model``; several adapters coexist (``add_adapter``,
    ``set_adapter``, ``delete_adapter``) and their biases add.

    Parameters
    ----------
    model : nn.Module
        The model to give the biases to.
    peft_config : BeftConfig
        Which modules, and whether the bias starts at zero.
    adapter_name : str
        Name of the first adapter, the key in every ``beft_bias``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` finds nothing (or is left out for a model type the table does not
        cover), or ``modules_to_save`` finds nothing.
    NotImplementedError
        Layers of a different method are already in ``model``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import BeftConfig, BeftModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> beft = BeftModel(net, BeftConfig(target_modules=["0"]), "default")
    >>> beft.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].beft_bias["default"].requires_grad
    (False, True)
    >>> beft.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = BeftLayer._layer_types
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = BEFT_DEFAULT_TARGET_MODULES
