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
from lucid.peft._slots import _AdapterSlot, _outside_slots
from lucid.peft.tuners._base import (
    _BaseTuner,
    _BaseTunerLayer,
    _check_adapter_name,
    _join,
    _named,
    _Target,
    _tuner_layers,
)
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.lora._layer import Linear, LoraLayer
from lucid.peft.tuners.lora._merging import _Combination, _CombinationType, _MergeOptions
from lucid.peft.tuners.lora._parameter import ParamWrapper


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


def _wrapper_of(module: nn.Module, parameter: str) -> ParamWrapper | None:
    """The wrapper among ``module``'s nested ``ParamWrapper``\\ s that adapts ``parameter``."""
    while isinstance(module, ParamWrapper):
        if module.parameter_name == parameter:
            return module
        module = module.base_layer
    return None


def _parameter_targets(model: nn.Module, names: list[str], claimed: set[str]) -> list[_Target]:
    """A target for each raw parameter named by ``names``: its dotted name equals one or ends with ``"." + name``.

    The module that holds the parameter is the target's ``module`` — the
    wrapper already adapting it for an earlier adapter, or else the
    outermost of the wrappers adapting its other parameters, or else the
    module itself.  The top-level module's own parameters cannot be
    adapted (the wrapper would have to hold itself), and neither can a
    parameter of a module the config adapts as a module.
    """
    wanted = sorted(set(names))
    targets: list[_Target] = []
    for key, module in _outside_slots(model):
        if isinstance(module, _BaseTunerLayer):
            holder = module.get_base_layer()
        elif isinstance(module, _AdapterSlot):
            continue  # a modules_to_save copy trains whole
        else:
            holder = module
        for parameter, _ in holder.named_parameters(recurse=False):
            full = _join(key, parameter)
            if not any(full == name or full.endswith(f".{name}") for name in wanted):
                continue
            if not key:
                raise ValueError(
                    f"Targeting an nn.Parameter on the top-level module is not supported "
                    f"(parameter '{parameter}')."
                )
            if key in claimed or (
                isinstance(module, _BaseTunerLayer) and not isinstance(module, ParamWrapper)
            ):
                raise ValueError(
                    f"Trying to wrap the nn.Parameter {full!r} of {type(holder).__name__} "
                    f"{key!r}, which is also adapted as a module; target it either through "
                    "`target_modules` or through `target_parameters`, not both."
                )
            existing = _wrapper_of(module, parameter)
            targets.append(
                _Target(key, module, ParamWrapper, parameter)
                if existing is None
                else _Target(key, existing, None, parameter)
            )
    return targets


class _LayerArguments(TypedDict):
    """What a LoRA layer is told of an adapter: ``update_layer``'s hyper-parameters."""

    r: int
    lora_alpha: float
    lora_dropout: float
    init_lora_weights: bool | str
    use_rslora: bool
    use_dora: bool
    lora_bias: bool


def _layer_arguments(config: LoraConfig, key: str) -> _LayerArguments:
    return _LayerArguments(
        r=_pattern_value(config.rank_pattern, key, config.r),
        lora_alpha=_pattern_value(config.alpha_pattern, key, config.lora_alpha),
        lora_dropout=config.lora_dropout,
        init_lora_weights=config.init_lora_weights,
        use_rslora=config.use_rslora,
        use_dora=config.use_dora,
        lora_bias=config.lora_bias,
    )


class LoraModel(_BaseTuner, peft_type=PeftType.LORA):
    r"""A model with LoRA layers put into it, and the tools to take them out.

    Construction replaces each module ``config`` selects with a LoRA layer
    around it — :class:`~lucid.peft.tuners.lora.Linear`, ``Embedding``,
    ``Conv1d`` / ``Conv2d`` / ``Conv3d`` or ``MultiheadAttention``, by the
    module's type — each module holding a parameter ``target_parameters``
    names with a :class:`~lucid.peft.tuners.lora.ParamWrapper` for that
    parameter, and each ``modules_to_save`` entry with a frozen-original /
    trainable-copy pair, then freezes every other parameter.  The model is
    changed in place and kept as ``model``; a public name this wrapper
    lacks is read from it.  ``config`` itself is not changed:
    ``peft_config`` holds a copy, with ``"all-linear"`` spelled out as the
    names it chose.

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
        No module matches ``target_modules`` (and no parameter
        ``target_parameters``), a match is a module type with no LoRA
        layer, or ``modules_to_save`` matches nothing.
    NotImplementedError
        The model already holds adapter layers (wrap it once; add
        adapters with ``add_adapter``).

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
        layer = target.layer
        # Planning took it from ``LoraLayer``'s table, or made it ``ParamWrapper``.
        assert layer is not None and issubclass(layer, LoraLayer)
        arguments = _layer_arguments(_as_lora(config), target.full_name)
        if target.parameter is not None:
            return ParamWrapper(target.module, adapter_name, target.parameter, **arguments)
        return layer(target.module, adapter_name, **arguments)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, LoraLayer)
        layer.update_layer(adapter_name, **_layer_arguments(_as_lora(config), target.full_name))

    @override
    @classmethod
    def _adapts_parameters(cls, config: _TargetedConfig) -> bool:
        return bool(_as_lora(config).target_parameters)

    @override
    @classmethod
    def _plan_parameter_targets(
        cls, model: nn.Module, config: _TargetedConfig, claimed: set[str]
    ) -> list[_Target]:
        names = _as_lora(config).target_parameters
        if not names:
            return []
        targets = _parameter_targets(model, names, claimed)
        if not targets and not claimed:
            raise ValueError(
                f"Target parameters {names} not found in the base model. "
                "Please check the target parameters and try again."
            )
        if not targets:
            warnings.warn(
                f"target_parameters={names} were set but no parameter was matched.",
                RuntimeWarning,
                stacklevel=6,
            )
        return targets

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
            # A module whose parameters several nested layers adapt answers once.
            return list(
                dict.fromkeys(
                    _join(name, f"{layer._base_path()}.bias")
                    for name, layer in model.named_modules()
                    if isinstance(layer, LoraLayer) and layer.bias is not None
                )
            )
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

    def _refuse_variants(self, adapters: list[str]) -> None:
        """Raise ``NotImplementedError`` if a source adapter is DoRA or has ``lora_bias``: their extra state has no weighted sum."""
        for name in adapters:
            config = self._lora_config(name)
            if config.use_dora or config.lora_bias:
                raise NotImplementedError(
                    f"add_weighted_adapter does not combine DoRA or lora_bias adapters: "
                    f"{name!r} has use_dora={config.use_dora}, lora_bias={config.lora_bias}"
                )

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
        self._refuse_variants(adapters)
        for key, layer in _tuner_layers(self.model):
            held = [name for name in adapters if name in layer._adapter_names()]
            if held and not isinstance(layer, Linear):
                raise NotImplementedError(
                    f"add_weighted_adapter combines nn.Linear adapters only; {held} also adapt "
                    f"{key!r}, a {type(layer).__name__}"
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
            # Filled from the sources right after: nothing may touch the base weights first.
            init_lora_weights=True,
            target_modules=self._weighted_target_modules(adapters),
            alpha_pattern={},
            rank_pattern={},
        )

    def _fill_weighted_adapter(
        self, adapter_name: str, folding: _Combination, adapters: list[str], weights: list[float]
    ) -> None:
        with lucid.no_grad():
            for _, layer in _tuner_layers(self.model):
                if not isinstance(layer, Linear) or adapter_name not in layer.lora_A:
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
            ``combination_type`` is in the TIES / DARE family, or a source is
            a DoRA or ``lora_bias`` adapter.

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
