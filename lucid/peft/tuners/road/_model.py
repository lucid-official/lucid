"""``RoadModel``: the RoAd tuner — its config and layer family; the rest is the generic tuner's."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.road._config import RoadConfig
from lucid.peft.tuners.road._layer import RoadLayer


class RoadModel(_ConfigTuner[RoadConfig], peft_type=PeftType.ROAD, mixed=True):
    r"""A network whose linear layers turn their outputs by learned 2-D rotations, and the tools to undo that.

    Each ``nn.Linear`` that ``config`` selects is replaced by a
    :class:`~lucid.peft.tuners.road.Linear` computing :math:`y = R(Wx + b)`, with :math:`R` made
    of one rotation-and-scale block per pair of output features; every other parameter is frozen.
    The network is edited in place and stays reachable as ``model``.  Adapters added later
    (``add_adapter``, ``set_adapter``, ``delete_adapter``) coexist, and the active ones turn the
    output one after the other in the order they are listed.

    Parameters
    ----------
    model : nn.Module
        The network to adapt.
    peft_config : RoadConfig
        Which modules, the variant and the group size.
    adapter_name : str
        What the first adapter is called: the key of every ``road_theta`` entry.
    low_cpu_mem_usage, state_dict
        The reference's options, refused unless left at their defaults, as in
        :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an ``nn.Linear`` or has an output width
        that ``group_size`` does not divide, or ``modules_to_save`` matches nothing.
    NotImplementedError
        The network already holds layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import RoadConfig, RoadModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> road = RoadModel(net, RoadConfig(group_size=2, target_modules=["0"]), "default")
    >>> road.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].road_theta["default"].requires_grad
    (False, True)
    >>> road.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = RoadLayer._layer_types
