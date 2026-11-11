"""``PeanutModel``: the PEANuT tuner — its config and layer family; the rest is the generic tuner's."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.peanut._config import PeanutConfig
from lucid.peft.tuners.peanut._layer import PeanutLayer


class PeanutModel(_ConfigTuner[PeanutConfig], peft_type=PeftType.PEANUT):
    r"""A model with PEANuT layers in it, and the tools to take them out again.

    Each ``nn.Linear`` ``config`` selects becomes a :class:`~lucid.peft.tuners.peanut.Linear` that
    adds, to the module's output, the input times an update a small network makes of the frozen
    weight, :math:`W + f(W;\theta)`; the rest of the model is frozen.  The model is edited in place
    and stays reachable as ``model``; several adapters coexist (``add_adapter``, ``set_adapter``,
    ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to put PEANuT layers into.
    peft_config : PeanutConfig
        Width, depth, activation, scale and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``peanut_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` finds nothing, a module it finds is not an ``nn.Linear``, or
        ``modules_to_save`` finds nothing.
    NotImplementedError
        Layers of a different method are already in ``model``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import PeanutConfig, PeanutModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> peanut = PeanutModel(net, PeanutConfig(r=4, target_modules=["0"]), "default")
    >>> peanut.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].peanut_A["default"].weight.requires_grad
    (False, True)
    >>> peanut.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = PeanutLayer._layer_types
