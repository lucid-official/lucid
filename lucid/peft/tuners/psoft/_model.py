"""``PsoftModel``: the PSOFT tuner — its config and layer family; the rest is the generic tuner's."""

from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner, _reset_fan_in_fan_out
from lucid.peft.tuners.psoft._config import PsoftConfig
from lucid.peft.tuners.psoft._layer import PsoftLayer


class PsoftModel(_ConfigTuner[PsoftConfig], peft_type=PeftType.PSOFT):
    r"""A network whose linear layers rotate their weights inside the principal subspace, and the tools to undo that.

    Each ``nn.Linear`` that ``config`` selects is replaced by a
    :class:`~lucid.peft.tuners.psoft.Linear` that adds :math:`U_r\Sigma_r (C - I) V_r^\top` to its
    weight, with the top-``r`` singular triplets frozen and :math:`C` a trained ``r x r`` matrix
    kept orthogonal up to two diagonal scalings; every other parameter is frozen.  The
    network is edited in place and stays reachable as ``model``.  Adapters added later
    (``add_adapter``, ``set_adapter``, ``delete_adapter``) coexist and add up.

    Parameters
    ----------
    model : nn.Module
        The network that receives the PSOFT layers; it is edited in place.
    peft_config : PsoftConfig
        The rank and scale, the SVD backend, the options of the rotation and the target modules.
    adapter_name : str
        The first adapter's name, the key of its ``psoft_R`` entry in every layer.
    low_cpu_mem_usage, state_dict
        Options of the reference that this library cannot honour: anything but the default raises,
        as in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``; a match is not an ``nn.Linear`` or is narrower than
        ``r``; or ``modules_to_save`` matches nothing.
    NotImplementedError
        The network already carries layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import PsoftConfig, PsoftModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> psoft = PsoftModel(net, PsoftConfig(r=4, target_modules=["0"]), "default")
    >>> psoft.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].psoft_R["default"].weight.requires_grad
    (False, True)
    >>> psoft.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = PsoftLayer._layer_types

    @override
    @classmethod
    def _prepare_config(cls, config: PsoftConfig, model: nn.Module) -> PsoftConfig:
        stored = super()._prepare_config(config, model)
        _reset_fan_in_fan_out(stored)
        return stored
