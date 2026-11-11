"""``AdamssModel``: the AdaMSS tuner — its layers, the subspace allocation and the adapter file's frozen projection.

Everything that is not AdaMSS — planning, applying, freezing, switching, merging, unloading — is
``_BaseTuner``'s, in ``lucid.peft.tuners._tuner``.
"""

from collections.abc import Mapping
from typing import override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.adamss._allocator import SubspaceAllocator
from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._layer import AdamssLayer, _parameters

_NEW_B = "adamss_newB"


class AdamssModel(_FileFormat, _ConfigTuner[AdamssConfig], peft_type=PeftType.ADAMSS):
    r"""A model whose chosen linear layers train a low-rank update inside each subspace of their rows.

    Each ``nn.Linear`` that ``config`` selects becomes a :class:`~lucid.peft.tuners.adamss.Linear`:
    the SVD of its weight (with the bias as one more column) is clustered into
    ``num_subspaces`` subspaces of rows, and each subspace gets a pair of small trained factors that
    start as the identity; every other parameter is frozen.  The model is edited in place and stays
    reachable as ``model``; several adapters coexist (``add_adapter``, ``set_adapter``,
    ``delete_adapter``) and their updates add.  Without ``target_modules`` the attention projections
    the reference's table covers are taken.  ``config`` is named ``config``, not ``peft_config``, as
    in the reference.

    The split into subspaces is derived from the weight, not saved: an adapter file holds the
    factors and the projection, and loading it into a model whose weights split another way (a
    different base, or an adapter trained with the reference library on a weight that its randomised
    SVD split differently) raises ``ValueError`` naming the layer.

    **Training.**  With ``use_asa=True`` call :meth:`update_and_allocate` once per step, after
    ``loss.backward()`` and before the gradients are cleared; it records how much each subspace
    matters and, on the schedule ``init_warmup`` / ``final_warmup`` / ``mask_interval`` define,
    freezes the least important ones until about ``asa_target_subspaces`` remain.  A step's
    ``set_adapter`` makes every subspace of the adapter trainable again.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : AdamssConfig
        What to adapt and how.
    adapter_name : str
        Name of the first adapter, the key in every ``adamss_*`` table.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        ``target_modules`` finds nothing (or is left out for a model type the table does not
        cover), a match is not an ``nn.Linear``, or ``modules_to_save`` finds nothing.
    NotImplementedError
        Layers of a different method are already in ``model``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import AdamssConfig, AdamssModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> adamss = AdamssModel(net, AdamssConfig(r=4, num_subspaces=2, target_modules=["0"]), "default")
    >>> adamss.targeted_module_names, net[0].num_subspaces["default"]
    (['0'], 2)
    >>> net[0].base_layer.weight.requires_grad, net[0].adamss_B["default"]["0"].requires_grad
    (False, True)
    >>> adamss.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = AdamssLayer._layer_types

    @override
    def __init__(
        self,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt ``model`` in place with ``config``, building the AdaMSS layers as ``adapter_name``."""
        super().__init__(model, config, adapter_name, low_cpu_mem_usage, state_dict)
        self._allocators: dict[str, SubspaceAllocator] = {}

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name`` from every layer and forget what the allocator recorded for it."""
        super().delete_adapter(adapter_name)
        self._allocators.pop(adapter_name, None)

    # ── training ──────────────────────────────────────────────────────────

    def allocator(self, adapter_name: str) -> SubspaceAllocator:
        """The :class:`SubspaceAllocator` of ``adapter_name`` (built on first use).

        Parameters
        ----------
        adapter_name : str
            An adapter this model holds.

        Raises
        ------
        ValueError
            The model holds no adapter of that name.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"the model holds no adapter named {adapter_name!r}")
        config = self.peft_config[adapter_name]
        assert isinstance(config, AdamssConfig)
        if adapter_name not in self._allocators:
            self._allocators[adapter_name] = SubspaceAllocator(self.model, config, adapter_name)
        return self._allocators[adapter_name]

    def update_and_allocate(self, global_step: int) -> None:
        """Record the importance of every subspace and, on schedule, freeze the least important ones.

        Call it every training step, after ``loss.backward()`` and before the gradients are
        cleared.  Each active adapter whose config has ``use_asa`` is allocated; the others are
        left alone.

        Parameters
        ----------
        global_step : int
            The current training step.
        """
        for name in self.active_adapters:
            config = self.peft_config[name]
            if isinstance(config, AdamssConfig) and config.use_asa:
                self.allocator(name).update_and_allocate(global_step)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """Each layer's frozen projection, which no slot table holds: ``<layer>.adamss_newB`` in the file."""
        return context.buffer_keys(_NEW_B)

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Refuse a file whose subspaces are not the ones this model's weights split into, naming the layer."""
        for path, layer in context.layers:
            if not isinstance(layer, AdamssLayer) or context.adapter not in layer._specs:
                continue
            for index, held in enumerate(_parameters(layer.adamss_B, context.adapter)):
                loaded = state.get(context.key(path, f"adamss_B.{index}"))
                if loaded is not None and tuple(loaded.shape) != tuple(held.shape):
                    raise ValueError(
                        f"the AdaMSS tensors of {path!r} do not fit its subspaces: subspace {index} "
                        f"of the file has shape {tuple(loaded.shape)} where the model's has "
                        f"{tuple(held.shape)}.  The split follows the base weights and the "
                        "reference's randomised SVD and clustering; load the adapter into the "
                        "weights it was trained on"
                    )
        return state
