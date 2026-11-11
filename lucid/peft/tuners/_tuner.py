"""The generic half of a tuner: put a method's layers into a model, freeze the rest, take them out.

A method's tuner is a ``_BaseTuner`` subclass registered by ``peft_type=``;
it supplies its layer family, how a layer is built for a target and what
else trains, and inherits the rest — planning an injection, applying it
atomically, switching, freezing, deleting, merging and unloading.

``_BaseTuner`` is generic in the config class its method takes.  A subclass
that parametrises it (``class LoraModel(_BaseTuner[LoraConfig], ...)``) has
``_config_class`` derived from the argument; one that does not names
``_config_class`` itself, and sees the base's ``_TargetedConfig`` in its hooks.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import get_original_bases
from typing import ClassVar, Final, TypeIs, get_args, get_origin, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, _TargetedConfig
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._slots import _adapter_slots, _as_names, _outermost_slots
from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES, DEFAULT_LORA_TARGET_PARAMETERS
from lucid.peft._targeting import _parameter_paths, _shared_outside
from lucid.peft.tuners._host import _AdapterHost
from lucid.peft.tuners._layer import _BaseTunerLayer, _check_adapter_name, _join, _tuner_layers
from lucid.peft.tuners._plan import (
    _Injected,
    _Placement,
    _Target,
    _named,
    _not_found,
    _plan_modules_to_save,
    _put_back,
    _selected_modules,
    _set_submodule,
    _stored_config,
    _values_of,
)

#: Where a model adapted in place keeps the ``requires_grad`` it had before its first adapter.
_FLAGS_KEY: Final = "_peft_requires_grad_before"


class _BaseTuner[C: _TargetedConfig = _TargetedConfig](_AdapterHost, ABC):
    """Put a method's layers into a model, freeze the rest, take them out again.

    The generic half of a tuner.  Construction checks the first adapter,
    plans its injection, applies it and freezes; ``add_adapter`` does the
    same for each further one, which starts frozen and inactive until
    ``set_adapter`` names it.  A method supplies its layer family
    (``_layer_types``), how to build a layer for a target
    (``_create_new_module``) or add an adapter to one it already built
    (``_add_to_layer``), and may find raw parameters to adapt
    (``_plan_parameter_targets``), adjust the stored config
    (``_prepare_config``), check a new adapter against the others
    (``_check_new_adapter_config``) or add parameters that train
    (``_extra_trainable_names``).

    ``C`` is the config class the method takes.  Parametrising the base —
    ``class FooModel(_BaseTuner[FooConfig], peft_type=...)`` — types the
    ``config`` every hook receives and sets ``_config_class`` to ``FooConfig``;
    a subclass of the bare base keeps the default ``_TargetedConfig`` and names
    ``_config_class`` itself.
    """

    #: The method's layer classes by the module type each wraps — its
    #: family root's table (``LoraLayer._layer_types``).
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]]
    #: The config class this tuner takes, e.g. ``LoraConfig``.
    _config_class: ClassVar[type[_TargetedConfig]]
    #: What a config that names no ``target_modules`` adapts, by the model's ``model_type``.
    _default_targets: ClassVar[Mapping[str, frozenset[str]]] = DEFAULT_LORA_TARGET_MODULES
    #: The raw parameters it adapts instead, for a model type whose weights are not layers'.
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = DEFAULT_LORA_TARGET_PARAMETERS

    @override
    def __init_subclass__(cls, *, peft_type: PeftType | None = None, **kwargs: object) -> None:
        derived = _parametrised_config(cls)
        if derived is not None:
            named = cls.__dict__.get("_config_class", derived)
            if named is not derived:
                raise TypeError(
                    f"{cls.__name__} is parametrised by {derived.__name__} but names "
                    f"_config_class = {named.__name__}; give one"
                )
            cls._config_class = derived
        super().__init_subclass__(peft_type=peft_type, **kwargs)

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt ``model`` in place with ``peft_config``, building the method's layers as ``adapter_name``.

        Parameters
        ----------
        model : nn.Module
            The model to adapt.
        peft_config : PeftConfig
            What to adapt and how: the config class of this tuner's method.
        adapter_name : str
            Name of the adapter, the key in every one of its tables.
        low_cpu_mem_usage : bool, default=False
            The reference's option to build the adapter on the meta device; ``True`` raises
            ``NotImplementedError``.
        state_dict : dict of str to Tensor, optional
            The reference's option to match targets against state-dict keys rather than the
            model's modules; anything but ``None`` raises ``NotImplementedError``.

        Raises
        ------
        TypeError
            ``model`` is not a module or ``peft_config`` is another method's config.
        ValueError
            The adapter cannot be applied; ``model`` is left as it was.
        NotImplementedError
            ``low_cpu_mem_usage`` or ``state_dict`` holds a value the library cannot honour.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not self._accepts(peft_config):
            kind = type(self)._config_class
            raise TypeError(f"config must be a {kind.__name__}, got {type(peft_config).__name__}")
        adapter_name = _check_adapter_name(adapter_name)
        earlier = self._adopted_configs(model, adapter_name)
        # By name, which a deep copy or a pickle of the model keeps; a model
        # adapted before keeps what it had before the first adapter.
        trainable_now = {
            name: param.requires_grad
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        flags = model.__dict__.get(_FLAGS_KEY) if earlier is not None else None
        if flags is None:
            flags = trainable_now
        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage, state_dict)
        self.model = model
        self.peft_config: dict[str, PeftConfig] = {} if earlier is None else earlier
        self.active_adapter: str | list[str] = adapter_name
        self.targeted_module_names: list[str] = []
        self.targeted_parameter_names: list[str] = []
        self._requires_grad_before = flags
        injected = self._inject(adapter_name, peft_config)
        try:
            if earlier is not None:
                # The newest adapter is the one in use; a merged earlier one would hide it.
                self.unmerge_adapter()
            # Where ``get_peft_model_state_dict`` finds the config of a model
            # adapted in place, as the reference's does.
            model.peft_config = self.peft_config
            model.__dict__[_FLAGS_KEY] = flags
            self._mark_only_adapters_as_trainable()
        except BaseException:
            # A constructor that raises leaves the model it was given as it found it.
            self._rollback(adapter_name, injected)
            for name, param in model.named_parameters(remove_duplicate=False):
                param.requires_grad_(trainable_now[name])
            if earlier is None:
                model.__dict__.pop("peft_config", None)
                model.__dict__.pop(_FLAGS_KEY, None)
            raise

    @property
    @override
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""
        return _as_names(self.active_adapter)

    def _accepts(self, config: object) -> TypeIs[C]:
        """Whether ``config`` is an instance of the config class this tuner takes.

        ``C`` is erased at run time, so ``_config_class`` stands in for it; this is the one
        place the two are tied, and the checks that narrow a ``PeftConfig`` to ``C`` go through it.
        """
        return isinstance(config, type(self)._config_class)

    @classmethod
    def _adopted_configs(cls, model: nn.Module, adapter_name: str) -> dict[str, PeftConfig] | None:
        """The configs of the adapters ``model`` holds, if it already holds some of this method; else ``None``.

        A model adapted before takes the new adapter beside the ones it
        has, as the reference does when ``get_peft_model`` is called on it
        again, with the same warning; the new tuner then answers for all of
        them.

        Raises
        ------
        ValueError
            The model holds another method's adapters, or one named
            ``adapter_name``.
        """
        slots = _adapter_slots(model)
        if not slots:
            return None
        for key, slot in slots:
            # A family's layers share one table, which tells a layer's method.
            if isinstance(slot, _BaseTunerLayer) and slot._layer_types is not cls._layer_types:
                raise ValueError(
                    "Cannot combine adapters with different peft types: the model already holds "
                    f"{type(slot).__name__} (at {key!r}), which is not a layer of "
                    f"{cls.__name__}."
                )
        held = {name for _, slot in slots for name in slot._adapter_names()}
        known: dict[str, PeftConfig] = dict(model.__dict__.get("peft_config", {}))
        if adapter_name in held | set(known):
            raise ValueError(
                f"Adapter with name '{adapter_name}' already exists; the model holds "
                f"{sorted(held | set(known))}.  Pass another adapter_name to add a new one."
            )
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload "
            "the model with a different config, make sure to call `.unload()` before.",
            stacklevel=6,
        )
        return known

    # ── a method's hooks ──────────────────────────────────────────────────

    @classmethod
    def _prepare_config(cls, config: C, model: nn.Module) -> C:
        """The config to store: a copy, its default targets and ``"all-linear"`` spelled out."""
        return _stored_config(
            config,
            model,
            default_targets=cls._default_targets,
            default_parameters=cls._default_parameters,
            adapts_parameters=cls._adapts_parameters,
            adapted_linear=cls._adapted_linear,
        )

    @classmethod
    def _adapted_linear(cls, module: nn.Module) -> nn.Module | None:
        """The linear module ``module`` wraps, if it is one of this method's layers around one."""
        if isinstance(module, tuple(cls._layer_types.values())):
            base = module.get_base_layer()
            return base if isinstance(base, nn.Linear) else None
        return None

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """Raise ``ValueError`` if ``config`` cannot be added beside the adapters already held."""

    @classmethod
    @abstractmethod
    def _create_new_module(cls, config: C, adapter_name: str, target: _Target) -> nn.Module:
        """The layer that takes ``target``'s place."""

    @classmethod
    @abstractmethod
    def _add_to_layer(cls, config: C, adapter_name: str, target: _Target) -> None:
        """Give the layer ``target`` already is one more adapter."""

    @classmethod
    @abstractmethod
    @override
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """Parameters beyond the adapters' that train, by name under ``model`` (LoRA's biases)."""

    @classmethod
    @override
    def _require_adapter_state(cls, model: nn.Module, adapter_name: str) -> None:
        if not _tuner_layers(model):
            raise RuntimeError(
                f"adapter {adapter_name!r} has been merged or unloaded: the model holds no "
                f"{cls._config_class._peft_type} layer left to save or load"
            )

    @classmethod
    def _adapts_parameters(cls, config: C) -> bool:
        """Whether ``config`` names raw parameters to adapt.

        Then a config naming no ``target_modules`` adapts no module, rather
        than the model type's default ones.
        """
        del config  # the default adapts modules only
        return False

    @classmethod
    def _plan_parameter_targets(
        cls, model: nn.Module, config: C, claimed: set[str]
    ) -> list[_Target]:
        """The raw parameters ``config`` adapts, as targets that carry a ``parameter``.

        None by default: a method that adapts parameters finds them here.
        ``claimed`` holds the names of the modules already planned as
        modules — a module cannot be both.
        """
        del model, config, claimed  # the default has no parameter targets
        return []

    # ── injecting an adapter ──────────────────────────────────────────────

    @classmethod
    def _rewrites_base(cls, config: C) -> bool:
        """Whether building an adapter from ``config`` writes into the base weights (LoRA's PiSSA, OLoRA).

        Those writes happen as the layers are built, ahead of the tree being touched, so
        an injection that fails afterwards has to put the values back (``_Injected``).
        """
        del config  # the default leaves the base as it is
        return False

    def _inject(self, adapter_name: str, config: C) -> _Injected:
        """Plan, then apply, ``config`` as adapter ``adapter_name``; the model is as it was on failure."""
        model = self.model
        stored = self._prepare_config(config, model)
        self._check_new_adapter_config(stored)
        targets = self._plan_targets(model, stored)
        saved = _plan_modules_to_save(model, stored)
        # Everything new to the tree is built before the first change to it; an
        # initialisation that rewrites the base weights is the one thing built
        # in place, so its targets' values are kept.
        kept = _values_of(t.module for t in targets) if self._rewrites_base(stored) else []
        try:
            placements = self._new_slots(stored, adapter_name, targets, saved)
        except BaseException:
            _put_back(kept)
            raise
        injected = _Injected(placements, kept)
        self.peft_config[adapter_name] = stored
        try:
            self._attach_adapter(stored, adapter_name, targets, saved, placements)
            # A slot new to the tree began with its one adapter active.
            for _, slot in _adapter_slots(model):
                slot._activate(self.active_adapter)
            # The one place an adapter's dtype is settled, whatever its layer allocated: a
            # method cannot leave half precision on a half-precision base by accident.
            for _, layer in _tuner_layers(model):
                layer._widen_adapter(adapter_name)
        except BaseException:
            self._rollback(adapter_name, injected)
            raise
        for target in targets:
            names = (
                self.targeted_module_names
                if target.parameter is None
                else self.targeted_parameter_names
            )
            if target.full_name not in names:
                names.append(target.full_name)
        return injected

    def _new_slots(
        self, config: C, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        """The slots the adapter adds to the tree, built and not yet placed.

        One per module: where several parameters of a module are adapted
        the later layers wrap the earlier ones, and the outermost is placed.
        """
        outermost: dict[str, _Placement] = {}
        for target in targets:
            if target.layer is None:
                continue
            earlier = outermost.get(target.key)
            below = target if earlier is None else target._replace(module=earlier.slot)
            original = target.module if earlier is None else earlier.original
            slot = self._create_new_module(config, adapter_name, below)
            outermost[target.key] = _Placement(target.key, slot, original)
        placements = list(outermost.values())
        for key in saved:
            held = self.model.get_submodule(key)
            if not isinstance(held, ModulesToSaveWrapper):
                placements.append(_Placement(key, ModulesToSaveWrapper(held, adapter_name), held))
        return placements

    def _attach_adapter(
        self,
        config: C,
        adapter_name: str,
        targets: list[_Target],
        saved: list[str],
        placements: list[_Placement],
    ) -> None:
        """Give the adapter to the slots that exist, then put the new slots in the tree."""
        for target in targets:
            if target.layer is None:
                self._add_to_layer(config, adapter_name, target)
        for key in saved:
            wrapper = self.model.get_submodule(key)
            if isinstance(wrapper, ModulesToSaveWrapper):
                wrapper.update(adapter_name)
        for placement in placements:
            _set_submodule(self.model, placement.key, placement.slot)

    def _rollback(self, adapter_name: str, injected: _Injected) -> None:
        """Take ``injected`` back out: the adapter out of the tree, the base weights as they were."""
        self._withdraw(adapter_name, injected.placements)
        _put_back(injected.overwritten)

    def _withdraw(self, adapter_name: str, placements: list[_Placement]) -> None:
        """Undo a failed ``_inject``: the adapter out of every slot, the new slots out of the tree."""
        model = self.model
        for _, slot in _adapter_slots(model):
            slot.delete_adapter(adapter_name)
        for placement in placements:
            if model.get_submodule(placement.key) is placement.slot:
                _set_submodule(model, placement.key, placement.original)
        self.peft_config.pop(adapter_name, None)

    @classmethod
    def _plan_targets(cls, model: nn.Module, config: C) -> list[_Target]:
        targets = [cls._target_for(key, module) for key, module in _selected_modules(model, config)]
        targets += cls._plan_parameter_targets(model, config, {t.key for t in targets})
        if not targets:
            raise _not_found(config)
        return targets

    @classmethod
    def _target_for(cls, key: str, module: nn.Module) -> _Target:
        if isinstance(module, tuple(cls._layer_types.values())):
            return _Target(key, module, None)
        return _Target(key, module, cls._target_layer(key, module))

    @classmethod
    def _target_layer(cls, key: str, module: nn.Module) -> type[_BaseTunerLayer]:
        for kind in type(module).__mro__:
            layer = cls._layer_types.get(kind)
            if layer is not None:
                return layer
        supported = ", ".join(f"nn.{k.__name__}" for k in cls._layer_types)
        raise ValueError(
            f"Target module {key} ({type(module).__name__}) is not supported. "
            f"Currently, only {supported} is supported."
        )

    @override
    def add_adapter(self, adapter_name: str, config: PeftConfig) -> None:
        """Add an adapter beside the ones the model holds.

        The new adapter starts frozen and inactive: the forward pass keeps
        applying the adapters it applied, and every parameter keeps the
        ``requires_grad`` it had.  ``set_adapter`` switches to it.  A
        module an earlier adapter already wraps takes the new one beside
        it; any other target is wrapped.  The model is left as it was if
        the adapter cannot be added.

        Parameters
        ----------
        adapter_name : str
            Name of the new adapter; unique among the model's adapters.
        config : PeftConfig
            What to adapt and how — the config class of this tuner's method.

        Raises
        ------
        ValueError
            The name is taken or malformed, the config is another
            method's, no module matches it, or it cannot sit beside the
            adapters held (two adapters training biases).
        TypeError
            ``config`` is not an adapter config.
        """
        adapter_name = self._admit(adapter_name, config)
        assert self._accepts(config)  # _admit checked the class
        self._inject(adapter_name, config)
        self.set_requires_grad(adapter_name, False)

    # ── switching, freezing, deleting ─────────────────────────────────────

    def _check_weighted_sources(
        self, adapters: list[str], weights: list[float], adapter_name: str
    ) -> str:
        """The name of an adapter to be combined from ``adapters``, once the sources are checked.

        The rules every ``add_weighted_adapter`` shares: a free, well-formed name; sources that
        exist, one weight each; and no ``modules_to_save`` module trained by two sources (two
        copies of one module cannot be averaged).
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
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
        for key, wrapper in _named(self.model, ModulesToSaveWrapper):
            holders = [name for name in adapters if name in wrapper.modules_to_save]
            if len(holders) > 1:
                raise ValueError(
                    "Cannot add weighted adapters if they target the same module with "
                    f"modules_to_save, but {key!r} is trained by {holders}."
                )
        return adapter_name

    def _add_combined_adapter(
        self, adapter_name: str, config: PeftConfig, fill: Callable[[], None]
    ) -> None:
        """Add ``adapter_name``, then ``fill`` it from other adapters; withdrawn if ``fill`` fails.

        A half-filled adapter must not stay behind under a name that looks complete.
        """
        self.add_adapter(adapter_name, config)
        try:
            fill()
        except BaseException:
            self.delete_adapter(adapter_name)
            raise

    def _require_known(self, adapter_names: Sequence[str]) -> None:
        for name in adapter_names:
            if name not in self.peft_config:
                raise ValueError(f"Adapter {name} not found.")

    @override
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapter(s) in the forward pass, and train exactly those.

        Every other adapter is frozen; the activated ones train unless
        ``inference_mode``.  A merged layer is unmerged first.  Layers that
        hold none of the named adapters run their base alone.  The
        parameters ``bias`` makes trainable follow the adapter that asked
        for them.

        Parameters
        ----------
        adapter_name : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the activated adapters too.

        Raises
        ------
        ValueError
            A name is not one of the model's adapters, or two named
            adapters train the same ``modules_to_save`` module.
        """
        names = _as_names(adapter_name)
        self._require_known(names)
        slots = _adapter_slots(self.model)
        for _, slot in slots:
            slot._check_activation(names)
        for _, slot in slots:
            slot.set_adapter(adapter_name, inference_mode)
        self.active_adapter = adapter_name if isinstance(adapter_name, str) else [*adapter_name]
        params = dict(self.model.named_parameters())
        for held, config in self.peft_config.items():
            on = held in names and not inference_mode
            for extra in self._extra_trainable_names(self.model, config):
                params[extra].requires_grad_(on)

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the parameters of the named adapter(s).

        Raises
        ------
        ValueError
            A name is not one of the model's adapters.
        """
        names = _as_names(adapter_names)
        self._require_known(names)
        for _, slot in _adapter_slots(self.model):
            slot.set_requires_grad(names, requires_grad)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter: its parameters, its config, its ``modules_to_save`` copies.

        If it was active, the adapters still active stay so; when none
        remain the first of the others (by name) takes over, with a
        warning, and trains.  Deleting the last adapter leaves the layers
        in the model, applying nothing; ``unload()`` removes them.

        Raises
        ------
        ValueError
            The adapter does not exist, or it is merged — unmerge first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")
        merged = {name for _, layer in _tuner_layers(self.model) for name in layer.merged_adapters}
        if adapter_name in merged:
            raise ValueError(
                f"Cannot delete adapter(s) [{adapter_name!r}] while they are merged. "
                "Please unmerge them first."
            )
        del self.peft_config[adapter_name]
        for _, slot in _adapter_slots(self.model):
            slot.delete_adapter(adapter_name)
        active = self.active_adapters
        if adapter_name not in active:
            return
        remaining = [name for name in active if name != adapter_name]
        if not remaining and self.peft_config:
            successor = sorted(self.peft_config)[0]
            warnings.warn(
                f"Adapter {adapter_name} was active which is now deleted. "
                f"Setting active adapter to {successor}.",
                stacklevel=3,
            )
            remaining = [successor]
        self.set_adapter(remaining[0] if len(remaining) == 1 else remaining)

    def _mark_only_adapters_as_trainable(self) -> None:
        model = self.model
        trainable: set[int] = set()
        params = dict(model.named_parameters())
        for name in self.active_adapters:
            config = self.peft_config[name]
            if config.inference_mode:
                continue
            for _, slot in _adapter_slots(model):
                trainable.update(id(p) for p in slot._adapter_parameters(name))
            trainable.update(id(params[n]) for n in self._extra_trainable_names(model, config))
        for param in model.parameters():
            param.requires_grad_(id(param) in trainable)

    @override
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        """A context in which each batch row goes through the adapter ``adapter_names`` names for it.

        Only a method that can answer it implements this (LoRA).

        Raises
        ------
        NotImplementedError
            This method cannot route the rows of a batch to different adapters.
        """
        del adapter_names, num_beams
        raise NotImplementedError(
            f"{type(self).__name__} cannot send the rows of a batch through different adapters"
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        """Call the adapted model; its adapter layers do their work inside the call."""
        return self.model(*args, **kwargs)

    # ── merging and unloading ─────────────────────────────────────────────

    def _mergeable_layers(self) -> list[tuple[str, _BaseTunerLayer]]:
        """The tuner layers, after refusing any whose base weight another module shares."""
        layers = _tuner_layers(self.model)
        paths = _parameter_paths(self.model)
        for key, layer in layers:
            for leaf, param in layer._merge_targets().items():
                own = _join(key, leaf)
                others = [path for path in paths[id(param)] if path != own]
                if others:
                    raise RuntimeError(
                        f"cannot merge into {own}: the same Parameter is also "
                        f"{', '.join(others)}, which merging would change too — keep this "
                        "adapter unmerged, or untie the weight first"
                    )
        return layers

    def merge_adapter(
        self, adapter_names: list[str] | None = None, safe_merge: bool = False
    ) -> None:
        r"""Fold adapter updates into their base weights, keeping the adapter layers.

        A merged model runs at the base model's cost; ``unmerge_adapter``
        takes the updates out again.  Every refusal is made before any
        weight is written: nothing is merged if a layer's base weight is
        shared with another module (a tied LM head — merging would change
        that module too) or if the adapter is disabled.

        Parameters
        ----------
        adapter_names : list of str, optional
            Adapters to merge; the active ones by default.  A layer holding
            only some of them merges those.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or the adapter is
            disabled (inside ``disable_adapter()``).
        ValueError
            A name is not one of the model's adapters, or ``safe_merge``
            found a non-finite merged weight.
        """
        names = self.active_adapters if adapter_names is None else list(adapter_names)
        self._require_known(names)
        pending = []
        for _, layer in self._mergeable_layers():
            held = [name for name in names if name in layer._adapter_names()]
            pending.append((layer, held, layer._pending_merges(held)))
        if safe_merge:
            for layer, _, adapters in pending:
                for adapter in adapters:
                    layer._check_finite_merge(adapter)
        for layer, held, _ in pending:
            if held:
                layer.merge(adapter_names=held)

    def unmerge_adapter(self) -> None:
        """Take every merged adapter update back out of its base weight."""
        for _, layer in _tuner_layers(self.model):
            if layer.merged:
                layer.unmerge()

    def _unload(self, keep_trained: bool) -> nn.Module:
        model = self.model
        for key, slot in _outermost_slots(model):
            _set_submodule(model, key, slot._unwrap(keep_trained))
        for name, param in model.named_parameters(remove_duplicate=False):
            before = self._requires_grad_before.get(name)
            if before is not None:
                param.requires_grad_(before)
        if "peft_config" in model.__dict__:
            del model.peft_config
        model.__dict__.pop(_FLAGS_KEY, None)
        return model

    @override
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Merge the adapters into the base weights and return the plain model.

        Each adapter layer is replaced by its merged base module — the base
        model's own module, holding its own ``Parameter`` objects — and each
        ``modules_to_save`` entry by the active adapter's trained copy (the
        original when that adapter has none).  Adapters not merged are
        dropped with their layers.  Every base parameter gets back the
        ``requires_grad`` it had before injection.

        Parameters
        ----------
        progressbar : bool, default=False
            Accepted for the reference's signature; there is no progress bar.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones by default.

        Returns
        -------
        nn.Module
            The model, merged, with no adapter layer left in it.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or a
            ``modules_to_save`` module shared one (its trained copy would
            break the tie).
        """
        del progressbar  # the reference's signature; nothing to report progress on
        self._refuse_untying()
        self.merge_adapter(adapter_names=adapter_names, safe_merge=safe_merge)
        return self._unload(keep_trained=True)

    def _refuse_untying(self) -> None:
        """Refuse to swap in a trained copy of a module whose weight another module shares.

        The copy trained apart from the tie, so the merged model would hold
        two different tensors under names the base model ties into one —
        and a loader that re-ties them (the zoo's ``from_pretrained``)
        keeps whichever it reads last.
        """
        paths = _parameter_paths(self.model)
        for key, wrapper in _named(self.model, ModulesToSaveWrapper):
            tied = _shared_outside(paths, key, wrapper.original_module)
            if tied:
                raise RuntimeError(
                    f"cannot merge_and_unload: modules_to_save {key!r} was trained apart from "
                    f"{', '.join(tied)}, which it shares a weight with; the merged model would "
                    "hold two tensors where the base ties one.  Keep the adapter "
                    "(save_pretrained / PeftModel.from_pretrained), or untie the weight before "
                    "injecting"
                )

    @override
    def unload(self) -> nn.Module:
        """Remove every adapter without merging and return the original model.

        Every adapter layer is replaced by the module it wrapped (a merged
        one is unmerged first), every ``modules_to_save`` entry by its
        original module, and every base parameter gets back the
        ``requires_grad`` it had before injection.
        """
        return self._unload(keep_trained=False)

    @override
    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(True)

    @override
    def disable_adapter_layers(self) -> None:
        """Run the base model alone: adapter updates off, original ``modules_to_save``."""
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(False)

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        """Whether every adapted layer in this tuner supports conversion to LoRA."""
        return all(
            module.supports_lora_conversion(adapter_name)
            for module in self.modules()
            if isinstance(module, _BaseTunerLayer)
        )


def _parametrised_config(tuner: type[_BaseTuner]) -> type[_TargetedConfig] | None:
    """The config class ``tuner`` names by parametrising ``_BaseTuner`` (or a subclass of it), if it does.

    A type variable is not a class: ``_ConfigTuner[C]`` passes its own ``C`` on and names
    nothing, so only the subclass that fills it in sets ``_config_class``.

    Raises
    ------
    TypeError
        The argument is a class that is not a ``_TargetedConfig``.
    """
    for base in get_original_bases(tuner):
        origin = get_origin(base)
        if not (isinstance(origin, type) and issubclass(origin, _BaseTuner)):
            continue
        for argument in get_args(base)[:1]:
            if not isinstance(argument, type):
                continue
            if not issubclass(argument, _TargetedConfig):
                raise TypeError(
                    f"{tuner.__name__} is parametrised by {argument.__name__}, "
                    "which is not a _TargetedConfig"
                )
            return argument
    return None
