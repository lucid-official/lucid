"""The shape every adapter converter takes: records, a registry keyed by model type, and the renaming template.

An adapter file written for the reference library's model names its tensors
after that library's modules (``encoder.block.0.layer.0.SelfAttention.q``).
The zoo names the same modules otherwise (``t5.encoder.blocks.0.self_attention.q``),
and in places lays them out otherwise (one fused ``qkv`` where the reference
has three projections).  A converter is the bridge for one model type: it
says which Lucid module each reference module is (``map_module``), and
where the layouts differ in more than a name it rewrites the tensors
(``convert``).

A converter registers itself under the zoo ``model_type`` it serves, as the
configs and tuners do (``class T5Converter(AdapterConverter, model_type="t5")``);
a type with no converter is read as it is.
"""

import dataclasses
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import ClassVar, Final, Literal, NamedTuple, Self, overload, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, _pattern_key_regex, _TargetedConfig
from lucid.peft.tuners._base import _BaseTunerLayer

#: Where an adapter file keeps the tensors that sit on the user's model.
MODEL_PREFIX: Final = "base_model.model."

#: What ``PeftModel.from_pretrained(convert=...)`` takes.
ConvertMode = bool | Literal["auto"]


class UnmappedKey(ValueError):
    """The file holds tensors the converter has no rule for: another layout, or a module the zoo lacks."""


@dataclasses.dataclass(frozen=True, slots=True)
class AdapterFile:
    """An adapter as its directory holds it: the parsed config and every tensor, by file key."""

    config: PeftConfig
    tensors: Mapping[str, Tensor]


@dataclasses.dataclass(frozen=True, slots=True)
class ConversionContext:
    """What a converter may need to know about the model the adapter is for.

    ``model`` is the zoo model the adapter will be loaded into — its layout
    (a task head, a trunk) and, for the few conversions that need them, its
    weights.  A conversion done on files alone (the command line) has none
    and says what it knows of the reference side in ``reference_class``,
    the class name the adapter's ``auto_mapping`` records.
    """

    model_type: str | None = None
    model: nn.Module | None = None
    reference_class: str | None = None

    @classmethod
    def of(cls, model: nn.Module, config: PeftConfig) -> Self:
        """The context of loading ``config``'s adapter into ``model``."""
        kind = getattr(getattr(model, "config", None), "model_type", None)
        return cls(
            model_type=kind if isinstance(kind, str) else None,
            model=model,
            reference_class=(config.auto_mapping or {}).get("base_model_class"),
        )


class ModulePair(NamedTuple):
    """One module of the file, in the reference's name and in the zoo's.

    ``adapted`` tells a module that carries adapter tensors (a ``lora_A``)
    from one the adapter keeps a trained copy of (``modules_to_save``).
    """

    reference: str
    lucid: str
    adapted: bool


def _table_names() -> frozenset[str]:
    """Every per-adapter table a tuner layer keeps: the key part that follows the module's path."""
    names: set[str] = set()
    pending = _BaseTunerLayer.__subclasses__()
    while pending:
        layer = pending.pop()
        names.update(layer.adapter_layer_names)
        pending.extend(layer.__subclasses__())
    return frozenset(names)


def split_key(path: str) -> tuple[str, str]:
    """``path`` — a file key without ``base_model.model.`` — as ``(module, rest)``; ``rest`` keeps its dot.

    The module ends before the first adapter table (``lora_A``, ``ia3_l`` …)
    and, in a key that has none (a trained head's ``classifier.weight``), before
    the last part.
    """
    parts = path.split(".")
    tables = _table_names()
    cut = next((i for i, part in enumerate(parts) if i and part in tables), len(parts) - 1)
    return ".".join(parts[:cut]), "." + ".".join(parts[cut:])


def _names_module(entry: str, module: str) -> bool:
    """Whether the name ``entry`` selects ``module`` (equal, a dotted suffix, or an ancestor of it)."""
    return f".{entry}." in f".{module}." or f".{module}".endswith(f".{entry}")


def _unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _translated(entries: Iterable[str], pairs: Iterable[ModulePair]) -> list[str]:
    """``entries`` (module names as the reference spells them) as the zoo spells them.

    An entry is replaced by the zoo name of each module of the file it
    selects; one that selects none stays as it is.
    """
    pairs = list(pairs)
    out: list[str] = []
    for entry in entries:
        found = [p.lucid for p in pairs if _names_module(entry, p.reference)]
        out.extend(found or [entry])
    return _unique(out)


class AdapterConverter(ABC):
    """Reference-layout adapter → the zoo's layout, for one model type.

    A subclass names the type it serves (``model_type=``) and says
    ``map_module``; the template here renames every tensor by it and
    rewrites the config's module names to match.  A conversion that needs
    more than renaming overrides ``convert`` and uses the helpers
    (``_renamed``, ``_config``).
    """

    _registry: ClassVar[dict[str, type[AdapterConverter]]] = {}
    #: The ``model_type`` served; empty for the converter that reads files as they are.
    model_type: ClassVar[str] = ""
    #: The reference library's class names whose adapters this converter reads.
    reference_classes: ClassVar[frozenset[str]] = frozenset()

    @override
    def __init_subclass__(cls, *, model_type: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if model_type is not None:
            cls.model_type = model_type
            cls._registry[model_type] = cls

    @overload
    @classmethod
    def lookup(cls, *, model_type: str | None) -> type[AdapterConverter] | None: ...
    @overload
    @classmethod
    def lookup(cls, *, reference_class: str | None) -> type[AdapterConverter] | None: ...
    @classmethod
    def lookup(
        cls, *, model_type: str | None = None, reference_class: str | None = None
    ) -> type[AdapterConverter] | None:
        """The converter registered for ``model_type``, or the one that reads ``reference_class``.

        Parameters
        ----------
        model_type : str, optional
            A zoo ``model_type``; when given, it alone decides.
        reference_class : str, optional
            The reference library's class name, as an adapter's ``auto_mapping`` records it.

        Returns
        -------
        type[AdapterConverter] or None
            The converter class, or ``None`` when none serves the type or class.
        """
        if model_type is not None:
            return cls._registry.get(model_type)
        return next(
            (c for c in cls._registry.values() if reference_class in c.reference_classes), None
        )

    @classmethod
    def registered(cls) -> list[str]:
        """The model types that have a converter."""
        return sorted(cls._registry)

    def __init__(self, context: ConversionContext) -> None:
        self.context = context

    @abstractmethod
    def map_module(self, module: str) -> str | None:
        """The zoo's path of the reference module ``module`` (relative to the model); ``None`` if not a reference path."""

    def _check_tuner(self, config: PeftConfig) -> None:
        """Hook: refuse a method this converter cannot convert."""
        del config  # renaming is method-independent

    def _pairs(self, tensors: Mapping[str, Tensor]) -> list[ModulePair]:
        """Each module of the file with its zoo name; keys outside the model are not modules."""
        pairs: dict[tuple[str, bool], ModulePair] = {}
        unmapped: list[str] = []
        for key in tensors:
            if not key.startswith(MODEL_PREFIX):
                continue
            module, rest = split_key(key.removeprefix(MODEL_PREFIX))
            lucid = self.map_module(module)
            if lucid is None:
                unmapped.append(key)
                continue
            adapted = rest.split(".")[1] in _table_names()
            pairs.setdefault((module, adapted), ModulePair(module, lucid, adapted))
        if unmapped:
            raise UnmappedKey(
                f"{len(unmapped)} adapter key(s) are not in the reference layout of "
                f"model type {self.model_type or '?'!r}: {unmapped[:6]}"
            )
        return list(pairs.values())

    def needs_conversion(self, adapter: AdapterFile) -> bool:
        """Whether ``adapter`` is in a layout this converter changes (a Lucid-layout file is not)."""
        for key in adapter.tensors:
            if key.startswith(MODEL_PREFIX):
                module, _ = split_key(key.removeprefix(MODEL_PREFIX))
                mapped = self.map_module(module)
                if mapped is not None and mapped != module:
                    return True
        return False

    def _renamed(self, tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """``tensors`` under the zoo's keys, unchanged in value."""
        out: dict[str, Tensor] = {}
        for key, tensor in tensors.items():
            new = key
            if key.startswith(MODEL_PREFIX):
                module, rest = split_key(key.removeprefix(MODEL_PREFIX))
                lucid = self.map_module(module)
                if lucid is None:
                    raise UnmappedKey(f"{key!r} is not in the reference layout")
                new = f"{MODEL_PREFIX}{lucid}{rest}"
            if new in out:
                raise ValueError(f"two adapter tensors land on the key {new!r}")
            out[new] = tensor
        return out

    def _config(
        self, config: PeftConfig, pairs: list[ModulePair], keep: frozenset[str] = frozenset()
    ) -> PeftConfig:
        """``config`` with its module names in the zoo's spelling.

        When any module is renamed, ``target_modules`` becomes the exact
        list of adapted modules (every name selector the reference allows
        — a suffix, a regex — resolves against the file, so nothing is
        left to match differently) and ``exclude_modules`` is dropped,
        since that list is already net of it.  ``keep`` are the zoo
        modules whose ``rank_pattern`` / ``alpha_pattern`` entries the
        converter writes itself.
        """
        if all(p.reference == p.lucid for p in pairs) or not isinstance(config, _TargetedConfig):
            return config
        adapted = [p for p in pairs if p.adapted]
        changes: dict[str, object] = {
            "target_modules": _unique(p.lucid for p in adapted),
            "exclude_modules": None,
        }
        if config.modules_to_save:
            changes["modules_to_save"] = _translated(
                config.modules_to_save, (p for p in pairs if not p.adapted)
            )
        names = {f.name for f in dataclasses.fields(config)}
        for field in ("feedforward_modules", "rank_pattern", "alpha_pattern"):
            if field not in names or not getattr(config, field):
                continue
            current = getattr(config, field)
            changes[field] = (
                _translated(current, adapted)
                if field == "feedforward_modules"
                else self._pattern(current, adapted, keep)
            )
        return type(config).from_dict({**config.to_dict(), **changes})

    @staticmethod
    def _pattern[V](
        pattern: Mapping[str, V], adapted: list[ModulePair], keep: frozenset[str]
    ) -> dict[str, V]:
        """A ``rank_pattern`` / ``alpha_pattern`` keyed by the zoo's module names."""
        out: dict[str, V] = {}
        for key, value in pattern.items():
            matched = [p.lucid for p in adapted if re.match(_pattern_key_regex(key), p.reference)]
            for lucid in matched or [key]:
                if lucid in keep:
                    continue
                if lucid in out and out[lucid] != value:
                    raise ValueError(f"the patterns give module {lucid!r} two different values")
                out[lucid] = value
        return out

    def convert(self, adapter: AdapterFile) -> AdapterFile:
        """The adapter in the zoo's layout."""
        self._check_tuner(adapter.config)
        pairs = self._pairs(adapter.tensors)
        return AdapterFile(self._config(adapter.config, pairs), self._renamed(adapter.tensors))


def convert_adapter(
    adapter: AdapterFile, context: ConversionContext, mode: ConvertMode = "auto"
) -> AdapterFile:
    """``adapter`` in the layout of ``context``'s model, as ``mode`` asks.

    ``False`` leaves it be.  ``"auto"`` converts when the model type has a
    converter and the file is in the reference layout it changes, and
    otherwise leaves it be.  ``True`` converts, and refuses a model type
    with no converter.

    Parameters
    ----------
    adapter : AdapterFile
        The adapter in the reference library's layout.
    context : ConversionContext
        The model type, and the model (or the reference class) the adapter is for.
    mode : bool or "auto", default="auto"
        Whether to convert.

    Returns
    -------
    AdapterFile
        The adapter in the zoo's layout, or ``adapter`` itself when nothing is converted.

    Raises
    ------
    ValueError
        ``mode`` is none of the three, or ``True`` for a model type with no converter.
    UnmappedKey
        The file holds a tensor the converter has no rule for.
    NotImplementedError
        The adapter's method cannot be converted for this layout.
    """
    if not (isinstance(mode, bool) or mode == "auto"):
        raise ValueError(f"convert must be True, False or 'auto', got {mode!r}")
    if mode is False:
        return adapter
    found = AdapterConverter.lookup(model_type=context.model_type)
    if found is None and context.model_type is None:
        found = AdapterConverter.lookup(reference_class=context.reference_class)
    if found is None:
        if mode is True:
            raise ValueError(
                f"no adapter converter for model type {context.model_type!r}; "
                f"registered: {AdapterConverter.registered()}"
            )
        return adapter
    converter = found(context)
    if mode == "auto" and not converter.needs_conversion(adapter):
        return adapter
    return converter.convert(adapter)
