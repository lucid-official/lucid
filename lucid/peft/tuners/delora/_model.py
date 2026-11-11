"""``DeloraModel``: the DeLoRA tuner — its config and layer family; the rest is the generic tuner's."""

import dataclasses
from typing import override

import lucid.nn as nn
from lucid.peft._config import PeftType
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._plan import _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.delora._config import DeloraConfig
from lucid.peft.tuners.delora._layer import DeloraLayer


def _for_module(config: DeloraConfig, key: str) -> DeloraConfig:
    """``config`` with the rank and boundary the module named ``key`` is given by ``rank_pattern`` / ``lambda_pattern``."""
    return dataclasses.replace(
        config,
        r=_pattern_value(config.rank_pattern or {}, key, config.r),
        delora_lambda=_pattern_value(config.lambda_pattern or {}, key, config.delora_lambda),
        rank_pattern={},
        lambda_pattern={},
    )


class DeloraModel(_ConfigTuner[DeloraConfig], peft_type=PeftType.DELORA):
    r"""A model with DeLoRA layers in it, and the tools to take them out again.

    Each module ``config`` selects becomes a :class:`~lucid.peft.tuners.delora.Linear` that adds a
    normalised low-rank update, bounded by a learned :math:`\lambda`, to the module's output; the
    rest of the model is frozen.  A module's rank and :math:`\lambda` are the config's unless
    ``rank_pattern`` / ``lambda_pattern`` name it.  The model is edited in place and stays
    reachable as ``model``; several adapters coexist (``add_adapter``, ``set_adapter``,
    ``delete_adapter``).

    Parameters
    ----------
    model : nn.Module
        The model to put DeLoRA layers into.
    peft_config : DeloraConfig
        Ranks, boundaries, dropout and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``delora_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` matches no module, a match is not ``nn.Linear``, or ``modules_to_save``
        names nothing.
    NotImplementedError
        Another method's layers are already in the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import DeloraConfig, DeloraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> delora = DeloraModel(net, DeloraConfig(r=4, target_modules=["0"]), "default")
    >>> delora.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].delora_A["default"].requires_grad
    (False, True)
    >>> delora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = DeloraLayer._layer_types

    @override
    @classmethod
    def _create_new_module(
        cls, config: DeloraConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return super()._create_new_module(
            _for_module(config, target.full_name), adapter_name, target
        )

    @override
    @classmethod
    def _add_to_layer(cls, config: DeloraConfig, adapter_name: str, target: _Target) -> None:
        super()._add_to_layer(_for_module(config, target.full_name), adapter_name, target)
