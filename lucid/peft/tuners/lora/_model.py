"""``LoraModel``: the LoRA half of a tuner — its layers, ranks, trainable biases and weighted adapters.

Everything that is not LoRA — planning, applying, freezing, switching,
merging, unloading — is ``_BaseTuner``'s, in ``lucid.peft.tuners._base``.
"""

import dataclasses
import functools
import operator
import re
import warnings
from collections.abc import Mapping
from typing import ClassVar, Literal, TypedDict, override

import lucid
import lucid.nn as nn
from lucid.peft._config import PeftConfig, PeftType, _pattern_key_regex, _TargetedConfig
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft.tuners._base import (
    _BaseTuner,
    _check_adapter_name,
    _join,
    _named,
    _Target,
    _tuner_layers,
)
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.lora._layer import LoraLayer
from lucid.peft.tuners.lora._merging import _Combination, _CombinationType, _MergeOptions


def _as_lora(config: PeftConfig) -> LoraConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, LoraConfig)
    return config


def _pattern_value[V](patterns: Mapping[str, V], key: str, default: V) -> V:
    if key in patterns:
        return patterns[key]
    for pattern, value in patterns.items():
        if re.match(_pattern_key_regex(pattern), key):
            return value
    return default


class _LayerArguments(TypedDict):
    """What a LoRA layer is told of an adapter: ``update_layer``'s hyper-parameters."""

    r: int
    lora_alpha: float
    lora_dropout: float
    init_lora_weights: bool | str
    use_rslora: bool


def _layer_arguments(config: LoraConfig, key: str) -> _LayerArguments:
    return _LayerArguments(
        r=_pattern_value(config.rank_pattern, key, config.r),
        lora_alpha=_pattern_value(config.alpha_pattern, key, config.lora_alpha),
        lora_dropout=config.lora_dropout,
        init_lora_weights=config.init_lora_weights,
        use_rslora=config.use_rslora,
    )


class LoraModel(_BaseTuner, peft_type=PeftType.LORA):
    r"""A model with LoRA layers put into it, and the tools to take them out.

    Construction replaces each module ``config`` selects with a
    :class:`~lucid.peft.tuners.lora.Linear` around it and each
    ``modules_to_save`` entry with a frozen-original / trainable-copy
    pair, then freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is
    read from it.  ``config`` itself is not changed: ``peft_config``
    holds a copy, with ``"all-linear"`` spelled out as the names it chose.

    The model can hold any number of adapters, each under its own name:
    ``add_adapter`` puts another beside the first (frozen and inactive),
    ``set_adapter`` chooses which one — or which several — the forward pass
    applies, ``add_weighted_adapter`` makes one out of others and
    ``delete_adapter`` removes one.  ``active_adapter`` is the name, or the
    list of names, in use.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    config : LoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter, the key in every ``lora_A`` / ``lora_B``.

    Raises
    ------
    ValueError
        No module matches ``target_modules``, a match is not an
        ``nn.Linear``, or ``modules_to_save`` matches nothing.
    NotImplementedError
        A match is a layer type the reference adapts but lucid.peft does not
        (``nn.Embedding``, convolutions, ``nn.MultiheadAttention``), or the
        model already holds adapter layers (wrap it once; add adapters
        with ``add_adapter``).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, LoraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> lora = LoraModel(net, LoraConfig(r=2, target_modules=["0"]))
    >>> lora.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].lora_A["default"].weight.requires_grad
    (False, True)
    >>> lora.add_adapter("other", LoraConfig(r=4, target_modules=["0"]))
    >>> lora.active_adapters, sorted(lora.peft_config)
    (['default'], ['default', 'other'])
    >>> lora.set_adapter("other")
    >>> lora.active_adapters, net[0].lora_A["default"].weight.requires_grad
    (['other'], False)
    >>> lora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = LoraLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = LoraConfig
    _later_types: ClassVar[tuple[type[nn.Module], ...]] = (
        nn.Embedding,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
    )

    def __init__(self, model: nn.Module, config: LoraConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_lora(super()._prepare_config(config, model))
        if stored.fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
                "Setting fan_in_fan_out to False.",
                stacklevel=5,
            )
            stored.fan_in_fan_out = False
        return stored

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        biased = [name for name, held in self.peft_config.items() if _as_lora(held).bias != "none"]
        if _as_lora(config).bias != "none":
            biased.append("<new>")
        if len(biased) > 1:
            raise ValueError(
                f"{type(self).__name__} supports only 1 adapter with bias. When using multiple "
                "adapters, set bias to 'none' for all adapters."
            )

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer, module = target.layer, target.module
        # Planning took both from ``LoraLayer``'s table, which holds nn.Linear layers.
        assert layer is not None and issubclass(layer, LoraLayer) and isinstance(module, nn.Linear)
        return layer(module, adapter_name, **_layer_arguments(_as_lora(config), target.key))

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, LoraLayer)
        layer.update_layer(adapter_name, **_layer_arguments(_as_lora(config), target.key))

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """The parameters ``bias`` makes trainable, by name under ``model``.

        ``"all"`` is every parameter named ``bias`` outside a module trained
        in full (that copy trains whole anyway, and its original stays
        frozen); ``"lora_only"`` the bias of each adapted layer.
        """
        bias = _as_lora(config).bias
        if bias == "lora_only":
            return [
                _join(name, "base_layer.bias")
                for name, layer in model.named_modules()
                if isinstance(layer, LoraLayer) and layer.bias is not None
            ]
        if bias == "none":
            return []
        wrapped = tuple(f"{name}." for name, _ in _named(model, ModulesToSaveWrapper))
        return [
            name
            for name, _ in model.named_parameters()
            if name.rpartition(".")[2] == "bias" and not name.startswith(wrapped)
        ]

    @override
    def disable_adapter_layers(self) -> None:
        """Run the base model alone: LoRA updates off, original ``modules_to_save``.

        With ``bias`` other than ``"none"`` the trained biases are base
        parameters and stay as trained, so the output is not the base
        model's; a warning says so.
        """
        for name in self.active_adapters:
            bias = _as_lora(self.peft_config[name]).bias
            if bias != "none":
                warnings.warn(
                    f"Careful, disabling adapter layers with bias configured to be '{bias}' does "
                    "not produce the same output as the base model would without adaption.",
                    stacklevel=2,
                )
        super().disable_adapter_layers()

    # ── weighted adapters ─────────────────────────────────────────────────

    def _lora_config(self, adapter_name: str) -> LoraConfig:
        return _as_lora(self.peft_config[adapter_name])

    def _weighted_target_modules(self, adapters: list[str]) -> str | set[str]:
        """The ``target_modules`` of an adapter that covers every module of ``adapters``."""
        targets = [self._lora_config(name).target_modules for name in adapters]
        if len({type(t) for t in targets}) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and "
                "string is not supported."
            )
        if all(isinstance(t, str) for t in targets):
            return "|".join(f"({t})" for t in targets)
        if all(isinstance(t, set) for t in targets):
            return functools.reduce(operator.or_, (t for t in targets if isinstance(t, set)))
        raise TypeError(f"Invalid type {type(targets[0])} found in target_modules")

    def _check_weighted_adapters(
        self, adapters: list[str], weights: list[float], majority_sign_method: str
    ) -> None:
        if not adapters:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        for name in adapters:
            if name not in self.peft_config:
                raise ValueError(f"Adapter {name} does not exist")
        if len(weights) != len(adapters):
            raise ValueError(
                f"add_weighted_adapter needs one weight per adapter, got {len(weights)} "
                f"weights for {len(adapters)} adapters"
            )
        if majority_sign_method not in ("total", "frequency"):
            raise ValueError(
                f"majority_sign_method must be 'total' or 'frequency', got {majority_sign_method!r}"
            )
        # Two copies of one module cannot be averaged.
        for key, wrapper in _named(self.model, ModulesToSaveWrapper):
            holders = [name for name in adapters if name in wrapper.modules_to_save]
            if len(holders) > 1:
                raise ValueError(
                    "Cannot add weighted adapters if they target the same module with "
                    f"modules_to_save, but {key!r} is trained by {holders}."
                )

    def _weighted_config(self, adapters: list[str], rank: int) -> LoraConfig:
        """The first source's config, widened to the union of the sources' modules, at ``rank``."""
        return dataclasses.replace(
            self._lora_config(adapters[0]),
            r=rank,
            lora_alpha=rank,
            # alpha = r only makes the scale 1 without rank stabilisation.
            use_rslora=False,
            target_modules=self._weighted_target_modules(adapters),
            alpha_pattern={},
            rank_pattern={},
        )

    def _fill_weighted_adapter(
        self, adapter_name: str, folding: _Combination, adapters: list[str], weights: list[float]
    ) -> None:
        with lucid.no_grad():
            for _, layer in _tuner_layers(self.model):
                if not isinstance(layer, LoraLayer) or adapter_name not in layer.lora_A:
                    continue
                lora_a, lora_b = folding.weights_for(layer, adapters, weights)
                target_a = layer.lora_A[adapter_name].weight
                target_b = layer.lora_B[adapter_name].weight
                target_a.copy_(lora_a.to(target_a.dtype))
                target_b.copy_(lora_b.to(target_b.dtype))

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: _CombinationType = "svd",
        svd_rank: int | None = None,
        svd_clamp: float | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        r"""Add an adapter that is a weighted combination of existing ones.

        The new adapter is frozen and inactive, like any added adapter, and
        has :math:`\alpha = r`, so its scale is 1 and the weights live in
        its matrices.  It targets the union of the sources' modules; a
        layer one source does not reach combines the sources it has.

        Parameters
        ----------
        adapters : list of str
            Names of the adapters to combine.
        weights : list of float
            One weight per adapter; negative weights subtract.
        adapter_name : str
            Name of the new adapter.
        combination_type : str, default="svd"
            ``"svd"``: the rank-``svd_rank`` truncation of
            :math:`\sum_i w_i s_i B_i A_i`, the nearest rank-``r`` update to
            the exact sum.  ``"cat"``: the exact sum, stacked, at the sum of
            the sources' ranks.  ``"linear"``: weights ``A`` and ``B`` apart
            (adding cross terms :math:`B_i A_j`); the sources share a rank.
            With one adapter every type is ``"linear"``.  The TIES / DARE /
            magnitude-prune types of the reference raise ``NotImplementedError``.
        svd_rank : int, optional
            Rank of an ``"svd"`` result; the largest source rank by default.
        svd_clamp : float, optional
            A quantile in ``[0, 1]``; ``"svd"`` factors are clamped to the
            symmetric range holding that share of their values.
        svd_full_matrices : bool, default=True
            Whether the SVD computes full ``U`` / ``Vh``.
        svd_driver : str, optional
            The SVD routine; ``lucid.linalg.svd`` accepts ``None`` only.
        density : float, optional
            For the TIES / DARE types; unused by the ones implemented.
        majority_sign_method : {"total", "frequency"}, default="total"
            For the TIES types; unused by the ones implemented.

        Raises
        ------
        ValueError
            The name is taken, a source does not exist, the weights do not
            match the sources, they cannot be combined that way (``"linear"``
            needs one rank; two train one ``modules_to_save`` module), or
            ``combination_type`` is unknown.
        NotImplementedError
            ``combination_type`` is in the TIES / DARE family.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, LoraModel
        >>> net = nn.Sequential(nn.Linear(6, 5))
        >>> config = LoraConfig(r=2, target_modules=["0"], init_lora_weights=False)
        >>> lora = LoraModel(net, config, adapter_name="a")
        >>> lora.add_adapter("b", config)
        >>> lora.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix", combination_type="cat")
        >>> net[0].r["mix"]
        4
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        self._check_weighted_adapters(adapters, weights, majority_sign_method)
        del density  # the TIES / DARE family's knob; none of the implemented types has a density
        # A single adapter has nothing to combine: it is scaled.
        strategy = _Combination.named("linear" if len(adapters) == 1 else combination_type)
        # The tensors allocated for the new adapter must hold the largest rank any layer uses.
        configs = [self._lora_config(name) for name in adapters]
        rank = strategy.new_rank([max([c.r, *c.rank_pattern.values()]) for c in configs], svd_rank)
        self.add_adapter(adapter_name, self._weighted_config(adapters, rank))
        folding = strategy(_MergeOptions(rank, svd_clamp, svd_full_matrices, svd_driver))
        try:
            self._fill_weighted_adapter(adapter_name, folding, adapters, weights)
        except BaseException:
            # A half-filled adapter must not stay behind under a name that looks complete.
            self.delete_adapter(adapter_name)
            raise
