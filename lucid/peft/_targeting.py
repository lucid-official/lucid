"""Which modules a targeting config (``LoraConfig`` and its kind) selects.

The reference library's matching rules, kept so a config means the same
thing on both sides: a list of names matches by dotted suffix, a string
is a regex over the whole name, and ``layers_to_transform`` filters by
the index after the layer list's name.  Where v1 differs is stated on
``LoraConfig``'s fields, which is where a user reads it.
"""

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping

import lucid.nn as nn
from lucid.peft._config import (
    PeftConfig,
    TaskType,
    _layer_regex,
    _pattern_key_regex,
    _TargetedConfig,
)
from lucid.peft._slots import _outside_slots

_ALL_LINEAR = "all-linear"

#: Root attributes the reference treats as a sequence-classification head.
_SEQ_CLS_HEADS = ("score", "classifier")


def _suffix_match(key: str, names: Iterable[str]) -> bool:
    """``key`` is one of ``names`` or ends with ``"." + name``."""
    return any(key == name or key.endswith(f".{name}") for name in names)


def _excluded(config: _TargetedConfig, key: str) -> bool:
    excluded = config.exclude_modules
    if isinstance(excluded, str):
        if re.fullmatch(excluded, key):
            return True
    elif excluded and _suffix_match(key, excluded):
        return True
    # An adapter inside a module trained in full would fight its copy.
    return any(f".{name}." in f".{key}." for name in config.modules_to_save or ())


def _layer_index(key: str, layers_pattern: str | list[str] | None) -> int | None:
    """The index after the layer list's name in ``key``, or ``None``."""
    if not layers_pattern:
        found = re.match(_layer_regex(None), key)
        return None if found is None else int(found.group(1))
    patterns = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
    for pattern in patterns:
        found = re.match(_layer_regex(pattern), key)
        if found is not None:
            return int(found.group(1))
    return None


def _in_selected_layer(config: _TargetedConfig, key: str) -> bool:
    layers = config.layers_to_transform
    if layers is None or layers == []:
        return True
    index = _layer_index(key, config.layers_pattern)
    if index is None:
        return False
    return index == layers if isinstance(layers, int) else index in layers


def _matches_target(config: _TargetedConfig, key: str) -> bool:
    targets = config.target_modules
    if targets is None:
        return False
    if isinstance(targets, str):
        return re.fullmatch(targets, key) is not None
    # An exact name is taken as asked for, past the layer filter, as the
    # reference does; a suffix match is filtered.
    if key in targets:
        return True
    return _suffix_match(key, targets) and _in_selected_layer(config, key)


def _selects(config: _TargetedConfig, key: str) -> bool:
    """Whether the module named ``key`` receives an adapter under ``config``."""
    return not _excluded(config, key) and _matches_target(config, key)


def _pattern_value[V](patterns: Mapping[str, V], key: str, default: V) -> V:
    """``patterns``' value for the module named ``key`` (exact key, else a regex-suffix key), else ``default``."""
    if key in patterns:
        return patterns[key]
    for pattern, value in patterns.items():
        if re.match(_pattern_key_regex(pattern), key):
            return value
    return default


def _parameter_paths(model: nn.Module) -> dict[int, list[str]]:
    """Every dotted name each parameter is reachable by, keyed by ``id``."""
    paths: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        paths.setdefault(id(param), []).append(name)
    return paths


def _output_head(model: nn.Module) -> nn.Module | None:
    """The module that maps hidden states to vocabulary logits, if the model has one."""
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    return head if isinstance(head, nn.Module) else None


def _head_names(model: nn.Module, task_type: TaskType | str | None) -> set[str]:
    head = _output_head(model)
    if head is None and task_type == TaskType.SEQ_CLS:
        for name in _SEQ_CLS_HEADS:
            candidate = getattr(model, name, None)
            if isinstance(candidate, nn.Module):
                head = candidate
                break
    if head is None:
        return set()
    return {name for name, module in model.named_modules() if module is head}


def _resolve_all_linear(
    config: _TargetedConfig,
    model: nn.Module,
    adapted_linear: Callable[[nn.Module], nn.Linear | None] | None = None,
) -> set[str] | None:
    """The module names ``"all-linear"`` stands for in ``model``; ``None`` if not used.

    Every ``nn.Linear`` — and every layer ``adapted_linear`` maps to the
    ``nn.Linear`` it wraps, which already holds an adapter and takes one
    more — except the output head, any layer whose weight another module
    shares (a tied head is the same matrix as the input table, and adapting
    it would adapt the table too) and any the config excludes
    (``exclude_modules``, ``modules_to_save``), so the names stored are
    exactly the names adapted.  The insides of an adapter slot
    (``q.base_layer``, ``q.lora_A.default``) are never candidates.
    """
    targets = config.target_modules
    if not (isinstance(targets, str) and targets.lower() == _ALL_LINEAR):
        return None
    paths = _parameter_paths(model)
    linear: set[str] = set()
    for name, module in _outside_slots(model):
        base = module if isinstance(module, nn.Linear) else None
        if base is None and adapted_linear is not None:
            base = adapted_linear(module)
        if name and base is not None and len(paths[id(base.weight)]) == 1:
            linear.add(name)
    return {
        name
        for name in linear - _head_names(model, config.task_type)
        if not _excluded(config, name)
    }


def _shared_outside(paths: dict[int, list[str]], key: str, module: nn.Module) -> list[str]:
    """Names outside ``key`` that reach one of ``module``'s parameters — a tie."""
    inside = f"{key}."
    return [
        path
        for param in module.parameters()
        for path in paths[id(param)]
        if not path.startswith(inside)
    ]


def _module_names_matching(model: nn.Module, names: Iterable[str]) -> list[str]:
    """Outermost modules whose dotted name ends with one of ``names``."""
    wanted = list(names)
    chosen: list[str] = []
    for key, _ in _outside_slots(model):
        inside = any(key.startswith(f"{outer}.") for outer in chosen)
        if key and not inside and _suffix_match(key, wanted):
            chosen.append(key)
    return chosen


def _with_heads(model: nn.Module, config: PeftConfig, heads: Iterable[str]) -> PeftConfig:
    """``config`` with those of ``heads`` that ``model`` has added to ``modules_to_save``.

    A head the model lacks is left out rather than named, so the stored
    config lists modules that exist.  A config that selects no modules
    (it has no ``modules_to_save``) is returned as it is.
    """
    if not isinstance(config, _TargetedConfig):
        return config
    saved = config.modules_to_save or []
    added = [head for head in heads if head not in saved and _module_names_matching(model, [head])]
    if not added:
        return config
    return dataclasses.replace(config, modules_to_save=[*saved, *added])


def _config_for_model(model: nn.Module, config: PeftConfig) -> PeftConfig:
    """``config`` as it is stored for ``model``: the base model's name filled in.

    A ``base_model_name_or_path`` read from the model's ``name_or_path``
    when it has one, as the reference does.  ``config`` itself is left as
    it is.
    """
    name_or_path = model.__dict__.get("name_or_path")
    return dataclasses.replace(
        config,
        base_model_name_or_path=(
            name_or_path if isinstance(name_or_path, str) else config.base_model_name_or_path
        ),
    )
