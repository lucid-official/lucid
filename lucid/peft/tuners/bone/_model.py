"""``BoneModel``: the tuner for Bone adapters — MiSS's, under Bone's names."""

from typing import ClassVar

from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.bone._config import BoneConfig
from lucid.peft.tuners.bone._layer import BoneLayer


class BoneModel(_ConfigTuner, peft_type=PeftType.BONE):
    r"""A model with Bone layers in it, and the tools to take them out again.

    As :class:`~lucid.peft.MissModel`, with the block kept as ``bone_block`` and only
    the balance and ``"bat"`` variants (see :class:`~lucid.peft.tuners.bone.BoneConfig`).

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    peft_config : BoneConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in every ``bone_block``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing is selected, a selected module is not linear, the sizes do not suit
        ``r`` (``"bat"`` wants both divisible by it), or ``modules_to_save`` is empty-handed.
    NotImplementedError
        The model already holds another method's layers.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.bone import BoneConfig, BoneModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> bone = BoneModel(net, BoneConfig(r=4, target_modules=["0"]), "default")
    >>> net[0].bone_block["default"].requires_grad
    True
    >>> bone.unload() is net
    True
    """

    _layer_types = BoneLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = BoneConfig
