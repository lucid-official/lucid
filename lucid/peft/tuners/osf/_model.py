"""``OSFModel``: the OSF tuner — its config and layer family; the rest is the generic tuner's."""

import dataclasses
import re
from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft._target_defaults import _model_type
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.osf._config import OSFConfig
from lucid.peft.tuners.osf._defaults import OSF_DEFAULT_TARGET_MODULES
from lucid.peft.tuners.osf._layer import OSFLayer


def _for_module(config: OSFConfig, key: str) -> OSFConfig:
    """``config`` with the frozen rank the first ``rank_pattern`` regex that matches the module named ``key`` gives it."""
    rank = config.effective_rank
    for pattern, value in (config.rank_pattern or {}).items():
        if re.search(pattern, key):
            rank = value
            break
    return dataclasses.replace(config, effective_rank=rank, rank_pattern=None)


class OSFModel(_ConfigTuner[OSFConfig], peft_type=PeftType.OSF):
    r"""A network whose weights train in their low singular directions, with the gradients kept off the high ones.

    Each ``nn.Linear`` that ``config`` selects is replaced by a :class:`~lucid.peft.tuners.osf.Linear`
    that splits its weight by SVD, freezes the top ``effective_rank`` triplets and trains the rest,
    its gradients projected onto the complement of the frozen directions; the number of frozen
    triplets is the config's unless ``rank_pattern`` names the module.  Nothing is added to the network
    beyond the split's own tensors.  The network is edited in place and stays reachable as ``model``.
    Adapters added later (``add_adapter``, ``set_adapter``, ``delete_adapter``) coexist, and the active
    ones add their updates to the weight.

    Parameters
    ----------
    model : nn.Module
        The network whose layers are split.
    config : OSFConfig
        The frozen rank, its per-module overrides and which modules.  Named ``config``, not
        ``peft_config``, as in the reference.
    adapter_name : str
        What the first adapter is called: the key of its ``osf_svd_params`` entry in every layer.
    low_cpu_mem_usage, state_dict
        Options of the reference that this library cannot honour: anything but the default raises,
        as in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``; a match is not an ``nn.Linear``; the frozen rank comes out
        as zero; or ``modules_to_save`` matches nothing.
    NotImplementedError
        The network already carries layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import OSFConfig, OSFModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> osf = OSFModel(net, OSFConfig(effective_rank=2, target_modules=["0"]), "default")
    >>> osf.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].osf_svd_params["default"]["U_low"].requires_grad
    (False, True)
    >>> osf.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = OSFLayer._layer_types
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = OSF_DEFAULT_TARGET_MODULES

    @override
    def __init__(
        self,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt ``model`` in place with ``config``, building the OSF layers as ``adapter_name``."""
        super().__init__(model, config, adapter_name, low_cpu_mem_usage, state_dict)

    @override
    def unmerge_adapter(self, *args: object, **kwargs: object) -> None:
        """Take every merged adapter update back out of its base weight.

        The reference refuses to unmerge an OSF adapter and declares ``(*args, **kwargs)``; here the
        update is a plain difference added to the weight, so it can be taken out again, and the
        signature stays the reference's.

        Raises
        ------
        TypeError
            An argument is given: there is nothing to pass.
        """
        if args or kwargs:
            raise TypeError("OSFModel.unmerge_adapter takes no arguments")
        super().unmerge_adapter()

    @override
    @classmethod
    def _prepare_config(cls, config: OSFConfig, model: nn.Module) -> OSFConfig:
        # A model type the table does not know takes every linear layer, as the reference does.
        if config.target_modules is None and _model_type(model) not in cls._default_targets:
            config = dataclasses.replace(config, target_modules="all-linear")
        return super()._prepare_config(config, model)

    @override
    @classmethod
    def _create_new_module(cls, config: OSFConfig, adapter_name: str, target: _Target) -> nn.Module:
        return super()._create_new_module(
            _for_module(config, target.full_name), adapter_name, target
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: OSFConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(_for_module(config, target.full_name), adapter_name, target)
