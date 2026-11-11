"""``LilyModel``: the Lily tuner — which layers share a down-projection and which share the up-projections."""

from collections.abc import Set
from typing import override

from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.lily._config import LilyConfig
from lucid.peft.tuners.lily._layer import LilyLayer
from lucid.peft.tuners.unilora._common import _SettlingTuner


def _suffix(key: str, target_modules: str | Set[str] | list[str] | None) -> str | None:
    """The ``target_modules`` entry that selected the module named ``key``, if one did.

    A string is a pattern that selects every module it matches, so it is its own entry.  Of several
    entries ending the name, the longest wins (the reference takes the last of an unordered set).
    """
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        return target_modules
    ending = [entry for entry in target_modules if key.endswith(entry)]
    return max(sorted(ending), key=len) if ending else None


class LilyModel(_SettlingTuner, _ConfigTuner[LilyConfig], peft_type=PeftType.LILY):
    r"""A model with Lily layers put into it, and the tools to take them out again.

    Construction replaces each ``nn.Linear`` ``config`` selects with a Lily layer around it and freezes
    every other parameter.  The model is changed in place and kept as ``model``; a public name this
    wrapper lacks is read from it.  ``config`` itself is not changed: ``peft_config`` holds a copy with
    the default ``target_modules`` spelled out.

    Layers are grouped by the ``target_modules`` entry that selected them and the shape of their
    weight, in the order the model lists them.  Within a group every ``stride_A`` consecutive layers
    share the down-projection of the first of the run, and all layers share the up-projections of
    the first layer — one module, found in each layer's ``lily_A`` and ``lily_B``.  The router of every
    layer is its own.  A Lily adapter cannot be merged: ``merge_adapter`` raises, ``unload`` works.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    peft_config : LilyConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in every ``lily_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or something that is not an ``nn.Linear``, or
        ``modules_to_save`` names nothing.
    NotImplementedError
        Layers of another method are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LilyConfig, LilyModel
    >>> net = nn.ModuleList([nn.ModuleDict({"q": nn.Linear(8, 8)}) for _ in range(3)])
    >>> lily = LilyModel(net, LilyConfig(r=4, stride_A=2, target_modules=["q"]), "default")
    >>> q = [block["q"] for block in net]                 # the three layers the entry "q" selects
    >>> q[1].lily_A["default"] is q[0].lily_A["default"]   # one down-projection per two layers
    True
    >>> q[2].lily_A["default"] is q[0].lily_A["default"]
    False
    >>> q[2].lily_B["default"] is q[0].lily_B["default"]   # the up-projections are the group's
    True
    >>> lily.unload() is net, type(net[0]["q"]) is nn.Linear
    (True, True)
    """

    _layer_types = LilyLayer._layer_types

    @override
    def _settle(self, config: _TargetedConfig, adapter_name: str, targets: list[_Target]) -> None:
        del targets  # the layers that hold the adapter are what is shared
        assert isinstance(config, LilyConfig)
        groups: dict[tuple[str, tuple[int, ...]], list[LilyLayer]] = {}
        for key, layer in _tuner_layers(self.model):
            if not isinstance(layer, LilyLayer) or adapter_name not in layer._adapter_names():
                continue
            suffix = _suffix(key, config.target_modules)
            if suffix is not None:
                shape = tuple(int(n) for n in layer.weight.shape)
                groups.setdefault((suffix, shape), []).append(layer)
        for layers in groups.values():
            shared_up = layers[0].lily_B[adapter_name]
            for index, layer in enumerate(layers):
                layer.lily_B[adapter_name] = shared_up
                lead = layers[(index // config.stride_A) * config.stride_A]
                layer.lily_A[adapter_name] = lead.lily_A[adapter_name]
