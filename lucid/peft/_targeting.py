"""Which modules a ``LoraConfig`` selects.

The reference library's matching rules, kept so a config means the same
thing on both sides: a list of names matches by dotted suffix, a string
is a regex over the whole name, and ``layers_to_transform`` filters by
the index after the layer list's name.  Two places differ, both where
the reference's rule was a bare string test: ``modules_to_save`` entries
match on a dotted boundary (``classifier`` no longer catches
``pre_classifier``), and a layer list at the model's root (``h.0.attn``)
is indexed too.
"""

import re
from collections.abc import Iterable, Mapping

import lucid.nn as nn
from lucid.peft._config import LoraConfig, TaskType

_ALL_LINEAR = "all-linear"

#: Root attributes the reference treats as a sequence-classification head.
_SEQ_CLS_HEADS = ("score", "classifier")


def _suffix_match(key: str, names: Iterable[str]) -> bool:
    """``key`` is one of ``names`` or ends with ``"." + name``."""
    return any(key == name or key.endswith(f".{name}") for name in names)


def _excluded(config: LoraConfig, key: str) -> bool:
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
        found = re.match(r"(?:.*?\.)?[^.]*\.(\d+)\.", key)
        return None if found is None else int(found.group(1))
    patterns = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
    for pattern in patterns:
        found = re.match(rf"(?:.*?\.)?{pattern}\.(\d+)\.", key)
        if found is not None:
            return int(found.group(1))
    return None


def _in_selected_layer(config: LoraConfig, key: str) -> bool:
    layers = config.layers_to_transform
    if layers is None or layers == []:
        return True
    index = _layer_index(key, config.layers_pattern)
    if index is None:
        return False
    return index == layers if isinstance(layers, int) else index in layers


def _matches_target(config: LoraConfig, key: str) -> bool:
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


def _selects(config: LoraConfig, key: str) -> bool:
    """Whether the module named ``key`` receives an adapter under ``config``."""
    return not _excluded(config, key) and _matches_target(config, key)


def _pattern_value[V](patterns: Mapping[str, V], key: str, default: V) -> V:
    if key in patterns:
        return patterns[key]
    for pattern, value in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", key):
            return value
    return default


def _rank_and_alpha(config: LoraConfig, key: str) -> tuple[int, float]:
    """The rank and alpha for ``key``, after ``rank_pattern`` / ``alpha_pattern``."""
    return (
        _pattern_value(config.rank_pattern, key, config.r),
        _pattern_value(config.alpha_pattern, key, config.lora_alpha),
    )


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


def _resolve_all_linear(config: LoraConfig, model: nn.Module) -> set[str] | None:
    """The module names ``"all-linear"`` stands for in ``model``; ``None`` if not used.

    Every ``nn.Linear`` except the output head and any layer whose weight
    another module shares — a tied head is the same matrix as the input
    table, and adapting it would adapt the table too.
    """
    targets = config.target_modules
    if not (isinstance(targets, str) and targets.lower() == _ALL_LINEAR):
        return None
    paths = _parameter_paths(model)
    linear = {
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and len(paths[id(module.weight)]) == 1
    }
    return linear - _head_names(model, config.task_type)


def _module_names_matching(model: nn.Module, names: Iterable[str]) -> list[str]:
    """Outermost modules whose dotted name ends with one of ``names``."""
    wanted = list(names)
    chosen: list[str] = []
    for key, _ in model.named_modules():
        inside = any(key.startswith(f"{outer}.") for outer in chosen)
        if key and not inside and _suffix_match(key, wanted):
            chosen.append(key)
    return chosen
