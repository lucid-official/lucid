"""``SupertuningModel``: the Super-Tuning tuner — its config and layer family, and the positions its file may carry."""

from typing import override

from lucid.peft._config import PeftConfig, PeftType
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.supertuning._config import SupertuningConfig
from lucid.peft.tuners.supertuning._layer import SupertuningLayer


class SupertuningModel(
    _FileFormat, _ConfigTuner[SupertuningConfig], peft_type=PeftType.SUPERTUNING
):
    r"""A model whose chosen linear layers train a sparse subset of their weight, and the tools to undo that.

    Each ``nn.Linear`` that ``config`` selects becomes a :class:`~lucid.peft.tuners.supertuning.Linear`
    that trains the ``1 - sparsity`` fraction of its entries with the largest magnitude (and a LoRA pair
    beside them when ``r`` is set); every other parameter is frozen.  The model is edited in place and
    stays reachable as ``model``; several adapters coexist (``add_adapter``, ``set_adapter``,
    ``delete_adapter``) and their updates add.  The adapter file holds, per layer, the trained values,
    the positions they belong to (unless ``save_precomputed_indices`` is ``False``) and the LoRA pair.

    Parameters
    ----------
    model : nn.Module
        The model whose weights are partly trained.
    peft_config : SupertuningConfig
        Which modules, how sparse, and the LoRA part.
    adapter_name : str
        Name of the first adapter, the key in every ``supertuning_values``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` selects nothing or a module that is not an ``nn.Linear``; ``sparsity``
        leaves a weight with no entry to train; two adapters differ on ``save_precomputed_indices``;
        ``modules_to_save`` selects nothing.
    NotImplementedError
        The model already holds layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import SupertuningConfig, SupertuningModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> tuned = SupertuningModel(net, SupertuningConfig(sparsity=0.9, target_modules=["0"]), "default")
    >>> tuned.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].supertuning_values["default"].requires_grad
    (False, True)
    >>> tuned.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = SupertuningLayer._layer_types

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        super()._check_new_adapter_config(config)
        assert isinstance(config, SupertuningConfig)
        held = (c for c in self.peft_config.values() if isinstance(c, SupertuningConfig))
        saved = {c.save_precomputed_indices for c in held} | {config.save_precomputed_indices}
        if len(saved) > 1:
            raise ValueError(
                "Super-Tuning indices must be saved for all adapters or none, but got multiple "
                f"different values: {sorted(saved)}"
            )

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """Each layer's positions, which no slot table holds: ``<layer>.supertuning_indices`` in the file, when saved."""
        keys: dict[str, str] = {}
        for path, layer in context.layers:
            if (
                isinstance(layer, SupertuningLayer)
                and layer.supertuning_indices.persistent
                and context.adapter in layer.supertuning_indices
            ):
                where = f"{context.prefix}{path}.supertuning_indices"
                keys[f"{where}.{context.adapter}"] = where
        return keys
