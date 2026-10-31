"""``AdaLoraModel``: the AdaLoRA half of a tuner — its layers, the rank allocation and the loss penalty.

Everything that is not AdaLoRA — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``.
"""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft.tuners._base import _NO_PARAMETERS, _BaseTuner, _Target, _tuner_layers
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._support import _reset_fan_in_fan_out
from lucid.peft.tuners.adalora._allocator import RankAllocator
from lucid.peft.tuners.adalora._checks import _check_rank_pattern
from lucid.peft.tuners.adalora._config import AdaLoraConfig
from lucid.peft.tuners.adalora._layer import AdaLoraLayer

_E: str = "lora_E"
_A: str = "lora_A"
_B: str = "lora_B"


def _as_adalora(config: PeftConfig) -> AdaLoraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, AdaLoraConfig)
    return config


_LOSS: Final = "loss"


def _with_loss(outputs: object, loss: Tensor) -> object:
    """``outputs`` carrying ``loss`` in place of the one it had: a copy for a dataclass, the object itself otherwise."""
    if dataclasses.is_dataclass(outputs) and not isinstance(outputs, type):
        return dataclasses.replace(outputs, **{_LOSS: loss})
    setattr(outputs, _LOSS, loss)
    return outputs


def _kept_indices(pattern: list[bool]) -> list[int]:
    return [i for i, kept in enumerate(pattern) if kept]


class AdaLoraModel(_FileFormat, _BaseTuner, peft_type=PeftType.ADALORA):
    r"""A model with AdaLoRA layers put into it, the rank allocation and the tools to take them out.

    Construction replaces each module ``config`` selects with an AdaLoRA layer
    around it — :class:`~lucid.peft.tuners.adalora.SVDLinear` or ``SVDConv2d``,
    by the module's type — and freezes every other parameter.  The model is
    changed in place and kept as ``model``; a public name this wrapper lacks is
    read from it.  ``config`` itself is not changed: ``peft_config`` holds a
    copy with the default ``target_modules`` spelled out.

    **Training.**  Call :meth:`update_and_allocate` once per step, after
    ``loss.backward()`` and before the gradients are cleared; it records how
    much each singular triplet matters and, on the schedule ``tinit`` /
    ``tfinal`` / ``deltaT`` / ``total_step`` define, masks the least important
    ones until ``target_r`` per layer remain.  The forward pass adds the
    orthogonality penalty ``orth_reg_weight * mean(||Q Q^T - I||, ||P^T P - I||)``
    to the output's ``loss`` when it has one, so the model has to be called
    through this wrapper (or a ``PeftModel``) to be penalised.

    **One trainable adapter.**  Several adapters may be held, but only one may
    train: give every other config ``inference_mode=True``.  The rank allocator
    follows the one that trains.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : AdaLoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``lora_*`` table.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is a module type with no
        AdaLoRA layer, ``modules_to_save`` matches nothing, or a second
        trainable adapter is asked for.
    NotImplementedError
        The model already holds adapter layers (wrap it once; add adapters
        with ``add_adapter``).

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.adalora import AdaLoraConfig, AdaLoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> config = AdaLoraConfig(init_r=4, target_r=2, total_step=10, target_modules=["0"])
    >>> ada = AdaLoraModel(net, config)
    >>> ada.targeted_module_names, net[0].r["default"]
    (['0'], 4)
    >>> ada(lucid.randn(3, 8)).sum().backward()
    >>> ada.update_and_allocate(1)
    >>> ada.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = AdaLoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = AdaLoraConfig
    # The LoRA table, as the reference does; its raw-parameter defaults are LoRA's alone.
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS

    def __init__(
        self, model: nn.Module, config: AdaLoraConfig, adapter_name: str = "default"
    ) -> None:
        super().__init__(model, config, adapter_name)
        self._allocator: RankAllocator | None = None

    # ── the adapters ──────────────────────────────────────────────────────

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        training = [name for name, held in self.peft_config.items() if not held.inference_mode]
        if not config.inference_mode:
            training.append("<new>")
        if len(training) > 1:
            raise ValueError(
                f"{type(self).__name__} supports only 1 trainable adapter. When using multiple "
                "adapters, set inference_mode to True for all adapters except the one you want "
                "to train."
            )

    @property
    def trainable_adapter_name(self) -> str | None:
        """The one adapter whose config is not ``inference_mode``, or ``None`` if every adapter is for inference."""
        names = [name for name, held in self.peft_config.items() if not held.inference_mode]
        return names[0] if names else None

    @property
    def rankallocator(self) -> RankAllocator:
        """The :class:`RankAllocator` of the trainable adapter (built on first use, rebuilt if that adapter changes).

        Raises
        ------
        RuntimeError
            Every adapter is for inference: there is nothing to allocate.
        """
        name = self.trainable_adapter_name
        if name is None:
            raise RuntimeError(
                "every adapter is in inference_mode; there is no trainable adapter to allocate"
            )
        if self._allocator is None or self._allocator.adapter_name != name:
            self._allocator = RankAllocator(self.model, _as_adalora(self.peft_config[name]), name)
        return self._allocator

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_adalora(super()._prepare_config(config, model))
        _reset_fan_in_fan_out(stored)
        return stored

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        # Planning took it from ``AdaLoraLayer``'s table.
        assert layer is not None and issubclass(layer, AdaLoraLayer)
        ada = _as_adalora(config)
        return layer(target.module, adapter_name, ada, r=ada.init_r, lora_alpha=ada.lora_alpha)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, AdaLoraLayer)
        ada = _as_adalora(config)
        layer.update_layer(adapter_name, ada.init_r, ada.lora_alpha, ada)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # an AdaLoRA adapter trains its own tensors only
        return []

    def add_weighted_adapter(self, *args: object, **kwargs: object) -> None:
        """Not supported for AdaLoRA: its pruned, masked triplets have no weighted sum.

        Raises
        ------
        TypeError
            Always.
        """
        del args, kwargs  # the reference's signature; no arguments make it supported
        raise TypeError(f"{type(self).__name__} does not support add_weighted_adapter method.")

    # ── training ──────────────────────────────────────────────────────────

    def _orthogonality(self, adapter: str) -> Tensor | None:
        """The mean over the adapter's ``lora_A`` and ``lora_B`` of their departure from orthonormality, or ``None`` if it has none."""
        penalties = [
            penalty
            for _, layer in _tuner_layers(self.model)
            if isinstance(layer, AdaLoraLayer) and adapter in layer._specs
            for penalty in layer.orthogonality(adapter)
        ]
        if not penalties:
            return None
        total = penalties[0]
        for penalty in penalties[1:]:
            total = total + penalty
        return total / len(penalties)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        outputs = self.model(*args, **kwargs)
        adapter = self.trainable_adapter_name
        loss = getattr(outputs, _LOSS, None)
        if adapter is None or not isinstance(loss, Tensor):
            return outputs
        weight = _as_adalora(self.peft_config[adapter]).orth_reg_weight
        penalty = self._orthogonality(adapter) if weight > 0 else None
        return outputs if penalty is None else _with_loss(outputs, loss + weight * penalty)

    def update_and_allocate(self, global_step: int) -> None:
        """Record the importance of every triplet and, on schedule, mask the least important ones.

        Call it every training step, after ``loss.backward()`` and before the
        gradients are cleared.  ``tinit``, ``tfinal`` and ``deltaT`` are
        handled here: until ``total_step - tfinal`` the allocator masks down to
        the falling budget; at that step it masks to ``target_r`` per layer,
        stores the rank pattern in the config and forgets the importances;
        afterwards it keeps the dropped triplets at zero.

        Parameters
        ----------
        global_step : int
            The current training step.

        Raises
        ------
        RuntimeError
            The adapter has no gradient (``backward`` was not called), or every
            adapter is for inference.
        """
        allocator = self.rankallocator
        config = allocator.peft_config
        final = allocator.final_step
        if global_step < final:
            _, pattern = allocator.update_and_allocate(self.model, global_step)
            if pattern:
                config.rank_pattern = pattern
        elif global_step == final:
            _, pattern = allocator.update_and_allocate(self.model, global_step, force_mask=True)
            config.rank_pattern = pattern
            allocator.reset_ipt()
        elif config.rank_pattern is not None:
            allocator.mask_using_rank_pattern(self.model, config.rank_pattern)

    # ── the rank pattern ──────────────────────────────────────────────────

    def resize_modules_by_rank_pattern(
        self, rank_pattern: Mapping[str, list[bool]], adapter_name: str
    ) -> None:
        """Drop the triplets a rank pattern marks as pruned from the layers, for good.

        The scale of what is kept does not change.  A layer already as small as
        its pattern asks is left alone.

        Parameters
        ----------
        rank_pattern : dict of str to list of bool
            ``"<module>.lora_E"`` to one flag per initial triplet, as the config stores it.
        adapter_name : str
            The adapter to resize.

        Raises
        ------
        ValueError
            A pattern entry names no layer, or its length is neither the
            layer's rank nor the number of triplets kept.
        """
        pattern = _check_rank_pattern("rank_pattern", dict(rank_pattern)) or {}
        layers = {
            f"{path}.{_E}": layer
            for path, layer in _tuner_layers(self.model)
            if isinstance(layer, AdaLoraLayer) and adapter_name in layer._specs
        }
        unknown = sorted(set(pattern) - set(layers))
        if unknown:
            raise ValueError(f"the rank pattern names layers the model does not have: {unknown}")
        for key, flags in pattern.items():
            layer, kept = layers[key], _kept_indices(flags)
            rank = layer._specs[adapter_name].r
            if len(kept) == rank:
                continue  # pruned already
            if len(flags) != rank:
                raise ValueError(
                    f"the rank pattern of {key} has {len(flags)} flags ({len(kept)} kept) but "
                    f"the layer holds {rank} triplets"
                )
            layer.prune(adapter_name, kept)
        self._allocator = None  # its budgets were measured on the layers as they were

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _to_file(cls, context: _FileContext, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Drop the triplets the rank pattern pruned from the tensors written (a layer already pruned is written whole)."""
        pattern = _as_adalora(context.config).rank_pattern
        if pattern is None:
            return state
        written = dict(state)
        for path, layer in context.layers:
            flags = pattern.get(f"{path}.{_E}")
            if (
                flags is None
                or not isinstance(layer, AdaLoraLayer)
                or context.adapter not in layer._specs
            ):
                continue
            kept = _kept_indices(flags)
            rank = layer._specs[context.adapter].r
            if len(kept) == rank:
                continue
            if len(flags) != rank:
                raise ValueError(
                    f"the rank pattern of {path}.{_E} has {len(flags)} flags but the layer "
                    f"holds {rank} triplets"
                )
            index = lucid.tensor(
                kept, dtype=lucid.int64, device=state[context.key(path, _E)].device
            )
            for table, axis in ((_E, 0), (_A, 0), (_B, 1)):
                key = context.key(path, table)
                written[key] = written[key].index_select(axis, index)
        return written

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Resize each layer to the rank the file's tensors have, after checking that they agree with one another."""
        resizes: list[tuple[AdaLoraLayer, int]] = []
        for path, layer in context.layers:
            if not isinstance(layer, AdaLoraLayer) or context.adapter not in layer._specs:
                continue
            singular, left, right = (state.get(context.key(path, t)) for t in (_E, _B, _A))
            if singular is None or left is None or right is None:
                continue  # missing tensors are reported by the loader
            rank = singular.shape[0]
            if left.ndim != 2 or right.ndim != 2 or left.shape[1] != rank or right.shape[0] != rank:
                raise ValueError(
                    f"the AdaLoRA tensors of {path!r} disagree on the rank (shapes: lora_E "
                    f"{tuple(singular.shape)}, lora_B {tuple(left.shape)}, lora_A {tuple(right.shape)})"
                )
            if rank != layer._specs[context.adapter].r:
                resizes.append((layer, rank))
        for layer, rank in resizes:
            layer.resize(context.adapter, rank)
        return state
