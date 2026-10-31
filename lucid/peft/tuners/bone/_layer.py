"""The Bone layers: MiSS layers that keep their block under Bone's name, ``bone_block``."""

from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._support import _config_as
from lucid.peft.tuners.bone._config import BoneConfig
from lucid.peft.tuners.miss._config import MissConfig
from lucid.peft.tuners.miss._layer import MissLayer


class BoneLayer(MissLayer, family=True):
    """Adapter bookkeeping shared by the Bone layers: a MiSS layer whose block is ``bone_block``.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : BoneConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.r`` exceeds the layer's input size, the ``"bat"`` variant needs sizes
        divisible by ``r``, or ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("bone_block",)
    _repr_prefix = "bone."
    bone_block: nn.ParameterDict

    @override
    @classmethod
    def _miss_config(cls, config: _TargetedConfig) -> MissConfig:
        bone = _config_as(BoneConfig, config)
        return MissConfig(r=bone.r, init_weights=bone.init_weights)


class Linear(BoneLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a Bone update: a MiSS update of the balance or ``"bat"`` variant.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`BoneLayer`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.bone import BoneConfig, Linear
    >>> layer = Linear(nn.Linear(8, 4), "default", BoneConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.bone_block["default"].shape)
    (2, 4)
    """

    base_layer: nn.Linear
