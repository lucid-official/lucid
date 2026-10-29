"""``LoraModel``: put LoRA layers into a model, freeze the rest, take them out again.

Injection is planned in full before the model is touched — every target
found, typed and given its rank, every ``modules_to_save`` entry found —
so a config that cannot be applied raises with the model as it was.  The
layers then go in by ``setattr`` on their parents, which keeps each
child's slot in place (a ``Sequential`` still runs it in the same turn).
"""

import copy
import warnings
from typing import NamedTuple, override

import lucid.nn as nn
from lucid.peft._config import LoraConfig
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._targeting import (
    _module_names_matching,
    _parameter_paths,
    _rank_and_alpha,
    _resolve_all_linear,
    _selects,
)
from lucid.peft.tuners.lora._layer import Linear, LoraLayer, _check_adapter_name

#: Layer types the reference adapts and v1 does not yet.
_LATER_TYPES = (nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention)


class _Target(NamedTuple):
    key: str
    module: nn.Linear
    r: int
    alpha: float


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _named[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind)]


def _set_submodule(model: nn.Module, key: str, module: nn.Module) -> None:
    parent, _, child = key.rpartition(".")
    setattr(model.get_submodule(parent), child, module)


def _learnable_bias_names(model: nn.Module, bias: str) -> list[str]:
    """The parameters ``bias`` makes trainable, by name under ``model``.

    ``"all"`` is every parameter named ``bias`` outside a module trained in
    full (that copy trains whole anyway, and its original stays frozen);
    ``"lora_only"`` the bias of each adapted layer.
    """
    if bias == "lora_only":
        return [
            _join(name, "base_layer.bias")
            for name, layer in _named(model, LoraLayer)
            if layer.bias is not None
        ]
    if bias != "all":
        return []
    wrapped = tuple(f"{name}." for name, _ in _named(model, ModulesToSaveWrapper))
    return [
        name
        for name, _ in model.named_parameters()
        if name.rpartition(".")[2] == "bias" and not name.startswith(wrapped)
    ]


def _linear_target(key: str, module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, _LATER_TYPES):
        raise NotImplementedError(
            f"Target module {key} ({type(module).__name__}) is not supported in "
            "lucid.peft v1, which adapts nn.Linear only"
        )
    raise ValueError(
        f"Target module {key} ({type(module).__name__}) is not supported. "
        "Currently, only nn.Linear is supported."
    )


def _not_found(config: LoraConfig) -> ValueError:
    targets = config.target_modules
    if targets is None:
        # The reference falls back to a table of per-architecture defaults;
        # Lucid's zoo has none yet, so the choice is the caller's.
        return ValueError(
            "Please specify `target_modules` in `peft_config`: a list of module names, "
            "a regex, or 'all-linear'"
        )
    shown = sorted(targets) if isinstance(targets, set) else targets
    message = (
        f"Target modules {shown} not found in the base model. "
        "Please check the target modules and try again."
    )
    if config.layers_to_transform is not None:
        message += f" Note: You specified 'layers_to_transform': {config.layers_to_transform}."
    if config.layers_pattern is not None:
        message += f" You also specified 'layers_pattern': {config.layers_pattern}."
    return ValueError(message)


def _plan_targets(model: nn.Module, config: LoraConfig) -> list[_Target]:
    targets = []
    for key, module in model.named_modules():
        if key and _selects(config, key):
            r, alpha = _rank_and_alpha(config, key)
            targets.append(_Target(key, _linear_target(key, module), r, alpha))
    if not targets:
        raise _not_found(config)
    return targets


def _plan_modules_to_save(model: nn.Module, config: LoraConfig) -> list[str]:
    names = config.modules_to_save
    if not names:
        return []
    keys = _module_names_matching(model, names)
    if not keys:
        raise ValueError(f"modules_to_save {names} matched no module of the model")
    return keys


def _stored_config(config: LoraConfig, model: nn.Module) -> LoraConfig:
    """A copy of ``config`` with ``"all-linear"`` spelled out for ``model``."""
    stored = copy.deepcopy(config)
    resolved = _resolve_all_linear(stored, model)
    if resolved is not None:
        stored.target_modules = resolved
    if stored.fan_in_fan_out:
        warnings.warn(
            "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
            "Setting fan_in_fan_out to False.",
            stacklevel=3,
        )
        stored.fan_in_fan_out = False
    return stored


def _refuse_existing_adapters(model: nn.Module) -> None:
    held = [name for name, _ in _named(model, LoraLayer)]
    held += [name for name, _ in _named(model, ModulesToSaveWrapper)]
    if held:
        raise RuntimeError(
            f"the model already holds a LoRA adapter (at {held[0]}); call unload() or "
            "merge_and_unload() first — lucid.peft v1 keeps one adapter per model"
        )


class LoraModel(_ForwardsAttributes):
    r"""A model with LoRA layers put into it, and the tools to take them out.

    Construction replaces each module ``config`` selects with a
    :class:`~lucid.peft.tuners.lora.Linear` around it and each
    ``modules_to_save`` entry with a frozen-original / trainable-copy
    pair, then freezes every other parameter.  The model is changed in
    place and kept as ``model``; a public name this wrapper lacks is
    read from it.  ``config`` itself is not changed: ``peft_config``
    holds a copy, with ``"all-linear"`` spelled out as the names it chose.

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
        A match is a layer type the reference adapts but v1 does not
        (``nn.Embedding``, convolutions, ``nn.MultiheadAttention``).
    RuntimeError
        The model already holds a LoRA adapter.

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
    >>> lora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    def __init__(self, model: nn.Module, config: LoraConfig, adapter_name: str = "default") -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not isinstance(config, LoraConfig):
            raise TypeError(f"config must be a LoraConfig, got {type(config).__name__}")
        adapter_name = _check_adapter_name(adapter_name)
        _refuse_existing_adapters(model)
        stored = _stored_config(config, model)
        targets = _plan_targets(model, stored)
        saved = _plan_modules_to_save(model, stored)
        flags = {id(param): param.requires_grad for param in model.parameters()}
        layers: list[tuple[str, nn.Module]] = [
            (
                target.key,
                Linear(
                    target.module,
                    adapter_name,
                    r=target.r,
                    lora_alpha=target.alpha,
                    lora_dropout=stored.lora_dropout,
                    init_lora_weights=stored.init_lora_weights,
                    use_rslora=stored.use_rslora,
                ),
            )
            for target in targets
        ]
        layers += [
            (key, ModulesToSaveWrapper(model.get_submodule(key), adapter_name)) for key in saved
        ]
        super().__init__()
        self.model = model
        self.peft_config: dict[str, LoraConfig] = {adapter_name: stored}
        self.active_adapter: str = adapter_name
        self.targeted_module_names: list[str] = [target.key for target in targets]
        self._requires_grad_before = flags
        for key, layer in layers:
            _set_submodule(model, key, layer)
        # Where ``get_peft_model_state_dict`` finds the config of a model
        # adapted in place, as the reference's does.
        model.peft_config = self.peft_config
        self._mark_only_adapters_as_trainable()

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("model")) if modules else None

    def _mark_only_adapters_as_trainable(self) -> None:
        config = self.peft_config[self.active_adapter]
        model = self.model
        trainable: set[int] = set()
        if not config.inference_mode:
            for _, layer in _named(model, LoraLayer):
                trainable.update(id(p) for p in layer.lora_A.parameters())
                trainable.update(id(p) for p in layer.lora_B.parameters())
            for _, wrapper in _named(model, ModulesToSaveWrapper):
                trainable.update(id(p) for p in wrapper.modules_to_save.parameters())
            params = dict(model.named_parameters())
            trainable.update(id(params[name]) for name in _learnable_bias_names(model, config.bias))
        for _, param in model.named_parameters():
            param.requires_grad_(id(param) in trainable)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.model(*args, **kwargs)

    def _mergeable_layers(self) -> list[tuple[str, Linear]]:
        """The LoRA layers, after refusing any whose base weight another module shares."""
        layers = _named(self.model, Linear)
        paths = _parameter_paths(self.model)
        for key, layer in layers:
            shared = paths[id(layer.weight)]
            if len(shared) > 1:
                others = [path for path in shared if path != _join(key, "base_layer.weight")]
                raise RuntimeError(
                    f"cannot merge into {_join(key, 'base_layer.weight')}: the same Parameter is "
                    f"also {', '.join(others)}, which merging would change too — keep this "
                    "adapter unmerged, or untie the weight first"
                )
        return layers

    def merge_adapter(
        self, adapter_names: list[str] | None = None, safe_merge: bool = False
    ) -> None:
        r"""Fold every LoRA update into its base weight, keeping the LoRA layers.

        A merged model runs at the base model's cost; ``unmerge_adapter``
        takes the updates out again.  Nothing is merged if any layer's base
        weight is shared with another module (a tied LM head): merging
        would change that module too.

        Parameters
        ----------
        adapter_names : list of str, optional
            Adapters to merge; the active one by default.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module.
        ValueError
            ``safe_merge`` found a non-finite merged weight.
        """
        layers = self._mergeable_layers()
        if safe_merge:
            for _, layer in layers:
                for adapter in layer._pending_merges(adapter_names):
                    layer._check_finite_merge(adapter)
        for _, layer in layers:
            layer.merge(adapter_names=adapter_names)

    def unmerge_adapter(self) -> None:
        """Take every merged LoRA update back out of its base weight."""
        for _, layer in _named(self.model, Linear):
            if layer.merged:
                layer.unmerge()

    def _unload(self, keep_trained: bool) -> nn.Module:
        model = self.model
        for key, layer in _named(model, LoraLayer):
            _set_submodule(model, key, layer.base_layer)
        for key, wrapper in _named(model, ModulesToSaveWrapper):
            if keep_trained:
                _set_submodule(model, key, wrapper.modules_to_save[wrapper.active_adapter])
            else:
                _set_submodule(model, key, wrapper.original_module)
        for param in model.parameters():
            before = self._requires_grad_before.get(id(param))
            if before is not None:
                param.requires_grad_(before)
        if "peft_config" in model.__dict__:
            del model.peft_config
        return model

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Merge the adapter into the base weights and return the plain model.

        Each LoRA layer is replaced by its merged ``nn.Linear`` — the base
        model's own module, holding its own ``Parameter`` objects — and each
        ``modules_to_save`` entry by its trained copy.  Every base parameter
        gets back the ``requires_grad`` it had before injection.

        Parameters
        ----------
        progressbar : bool, default=False
            Accepted for the reference's signature; there is no progress bar.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.
        adapter_names : list of str, optional
            Adapters to merge; the active one by default.

        Returns
        -------
        nn.Module
            The model, merged, with no LoRA layer left in it.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module.
        """
        del progressbar  # the reference's signature; nothing to report progress on
        self.merge_adapter(adapter_names=adapter_names, safe_merge=safe_merge)
        return self._unload(keep_trained=True)

    def unload(self) -> nn.Module:
        """Remove the adapter without merging it and return the original model.

        Every LoRA layer is replaced by the ``nn.Linear`` it wrapped (a
        merged one is unmerged first), every ``modules_to_save`` entry by
        its original module, and every base parameter gets back the
        ``requires_grad`` it had before injection.
        """
        self.unmerge_adapter()
        return self._unload(keep_trained=False)

    def _set_adapter_layers(self, enabled: bool) -> None:
        for _, layer in _named(self.model, LoraLayer):
            layer.enable_adapters(enabled)
        for _, wrapper in _named(self.model, ModulesToSaveWrapper):
            wrapper.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""
        self._set_adapter_layers(True)

    def disable_adapter_layers(self) -> None:
        """Run the base model alone: LoRA updates off, original ``modules_to_save``.

        With ``bias`` other than ``"none"`` the trained biases are base
        parameters and stay as trained, so the output is not the base
        model's; a warning says so.
        """
        bias = self.peft_config[self.active_adapter].bias
        if bias != "none":
            warnings.warn(
                f"Careful, disabling adapter layers with bias configured to be '{bias}' does not "
                "produce the same output as the base model would without adaption.",
                stacklevel=2,
            )
        self._set_adapter_layers(False)
