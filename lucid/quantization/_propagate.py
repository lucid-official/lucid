"""Per-module :class:`QConfig` resolution shared by every quantization entry point.

``prepare``, ``prepare_qat`` and ``quantize_dynamic`` used to each decide
"which recipe applies to this module" on their own, and only the first of
those three read a :class:`QConfigMapping` at all — a ``module.qconfig``
attribute was ignored, so ``child.qconfig = None`` could not exclude a
layer and a model-level ``qconfig`` could not pick the bit width.  The rule
now lives here, once, with the reference framework's ``propagate_qconfig_``
semantics:

1. the module's own ``.qconfig`` attribute — ``None`` is a real answer
   ("leave this module and everything under it in float");
2. otherwise a per-name / per-type rule of the explicit
   :class:`QConfigMapping`;
3. otherwise what the parent resolved to (the root inherits the mapping's
   global tier).

A child that sets its own ``.qconfig`` therefore re-enters quantization
underneath an excluded parent.
"""

from collections.abc import Callable, Collection, Iterator, Mapping
from typing import NamedTuple

import lucid.nn as nn
from lucid.quantization._qscheme import QDtype, per_channel_symmetric, quint8
from lucid.quantization.observer import PerChannelMinMaxObserver, PlaceholderObserver
from lucid.quantization.qconfig import QConfig, QConfigMapping

_MISSING = object()


class Resolved(NamedTuple):
    """One module of a model together with the recipe it resolved to (``None`` = stay float)."""

    name: str  # attribute name in its parent ("" for the root)
    module: nn.Module
    path: str  # dotted path from the root
    qconfig: QConfig | None


class QConfigResolver:
    """Resolves the effective :class:`QConfig` of every module in a model.

    The single owner of "which recipe applies here" for ``prepare``,
    ``prepare_qat`` and ``quantize_dynamic``: each module's own ``.qconfig``
    attribute, then a per-name / per-type rule of the wrapped
    :class:`QConfigMapping`, then what its parent resolved to.
    """

    def __init__(self, mapping: QConfigMapping, *, implicit: bool = False) -> None:
        self._mapping = mapping
        self._implicit = implicit  # ``mapping`` is the entry point's stock default

    @classmethod
    def from_argument(
        cls, qconfig: QConfig | QConfigMapping | None, default: Callable[[], QConfigMapping]
    ) -> QConfigResolver:
        """Build a resolver from an entry point's ``qconfig`` argument.

        A bare :class:`QConfig` becomes the global tier, ``None`` falls back to
        ``default()`` — but only for a model that carries no ``.qconfig``
        attribute of its own: once the user has configured modules, the stock
        recipe must not quietly cover the rest.

        Parameters
        ----------
        qconfig : QConfig or QConfigMapping or None
            The entry point's ``qconfig`` argument.
        default : Callable[[], QConfigMapping]
            Builds the entry point's stock mapping (static PTQ or QAT).

        Returns
        -------
        QConfigResolver
            A resolver over the mapping the argument denotes.

        Raises
        ------
        TypeError
            If ``qconfig`` is none of the accepted forms.
        """
        if qconfig is None:
            return cls(default(), implicit=True)
        if isinstance(qconfig, QConfig):
            return cls(QConfigMapping().set_global(qconfig))
        if isinstance(qconfig, QConfigMapping):
            return cls(qconfig)
        raise TypeError(
            f"qconfig must be a QConfig, a QConfigMapping or None; got {type(qconfig).__name__}"
        )

    def is_targeted(self, module: nn.Module, path: str) -> bool:
        """Whether a per-name / per-type rule names this module with a recipe."""
        return (
            self._mapping._has_rule(type(module), path)
            and self._mapping.get_qconfig(type(module), path) is not None
        )

    def resolve(self, module: nn.Module, path: str, inherited: QConfig | None) -> QConfig | None:
        """The effective recipe for ``module`` given what its parent resolved to."""
        own = getattr(module, "qconfig", _MISSING)
        if own is not _MISSING:
            if own is not None and not isinstance(own, QConfig):
                raise TypeError(
                    f"{path or type(module).__name__}.qconfig must be a QConfig or None; "
                    f"got {type(own).__name__}"
                )
            return own
        if self._mapping._has_rule(type(module), path):
            return self._mapping.get_qconfig(type(module), path)
        return inherited

    def root(self, model: nn.Module) -> Resolved:
        """Resolve the root module, which inherits the mapping's global tier."""
        seed = self._mapping._fallback()
        if self._implicit and any(hasattr(m, "qconfig") for m in model.modules()):
            seed = None
        return Resolved("", model, "", self.resolve(model, "", seed))

    def children(self, module: nn.Module, path: str, qconfig: QConfig | None) -> Iterator[Resolved]:
        """Resolve the direct children of ``module`` (whose own recipe is ``qconfig``)."""
        for name, child in list(module.named_children()):
            child_path = f"{path}.{name}" if path else name
            yield Resolved(name, child, child_path, self.resolve(child, child_path, qconfig))

    def walk(self, model: nn.Module) -> Iterator[Resolved]:
        """Every module once, parents before children, with its resolved recipe."""
        seen: set[int] = set()
        stack = [self.root(model)]
        while stack:
            entry = stack.pop()
            if id(entry.module) in seen:
                continue
            seen.add(id(entry.module))
            yield entry
            stack.extend(reversed(list(self.children(entry.module, entry.path, entry.qconfig))))


def dynamic_qconfig(dtype: QDtype) -> QConfig:
    """The weight-only recipe ``quantize_dynamic`` uses for ``dtype``.

    Activations carry a placeholder (they are quantized at run time); weights
    are observed per output channel, symmetric, on ``dtype``'s grid.
    """
    return QConfig(
        activation=PlaceholderObserver.with_args(qdtype=quint8),
        weight=PerChannelMinMaxObserver.with_args(
            ch_axis=0, qscheme=per_channel_symmetric, qdtype=dtype
        ),
    )


def weight_dtype(qconfig: QConfig, fallback: QDtype) -> QDtype:
    """The weight :class:`QDtype` a recipe asks for, or ``fallback`` if it names none.

    Builds one weight observer from ``qconfig`` and reads its ``qdtype``.

    Parameters
    ----------
    qconfig : QConfig
        The resolved recipe.
    fallback : QDtype
        Used when the weight factory builds something without a ``qdtype``.

    Returns
    -------
    QDtype
        The grid the weights are quantized on.
    """
    found = getattr(qconfig.weight(), "qdtype", None)
    return found if isinstance(found, QDtype) else fallback


class DynamicSpec(NamedTuple):
    """A normalised ``quantize_dynamic`` ``qconfig_spec``: the rules plus what they target."""

    mapping: QConfigMapping
    targets: tuple[str, ...]  # what the spec asks to quantize, for messages


def normalize_dynamic_spec(
    spec: object,
    dtype: QDtype,
    default_types: Collection[type],
) -> DynamicSpec:
    """Fold every accepted ``qconfig_spec`` form into one :class:`QConfigMapping`.

    ``None`` targets ``default_types``; a collection of module types and / or
    dotted names targets each with the default recipe for ``dtype``; a mapping
    pairs each type / name with a :class:`QConfig`, a :class:`QDtype` or
    ``None`` (skip).

    Parameters
    ----------
    spec : collection or mapping or None
        The user's ``qconfig_spec``.
    dtype : QDtype
        Weight dtype for entries that name none.
    default_types : Collection[type]
        What ``spec=None`` targets.

    Returns
    -------
    DynamicSpec
        The equivalent :class:`QConfigMapping` and the names it targets.

    Raises
    ------
    TypeError
        If ``spec`` or one of its keys / values is not an accepted form.
    """
    default = dynamic_qconfig(dtype)
    entries: dict[object, QConfig | None]
    if spec is None:
        entries = dict.fromkeys(default_types, default)
    elif isinstance(spec, Mapping):
        entries = {key: _spec_value(key, value) for key, value in spec.items()}
    elif isinstance(spec, Collection) and not isinstance(spec, (str, type)):
        entries = dict.fromkeys(spec, default)
    else:
        raise TypeError(
            "qconfig_spec must be a set of module types / names or a dict mapping them to a "
            f"QConfig, a QDtype or None; got {type(spec).__name__}"
        )
    mapping = QConfigMapping()
    targets: list[str] = []
    for key, qconfig in entries.items():
        if isinstance(key, str):
            mapping.set_module_name(key, qconfig)
            label = key
        elif isinstance(key, type) and issubclass(key, nn.Module):
            mapping.set_object_type(key, qconfig)
            label = key.__name__
        else:
            raise TypeError(
                f"qconfig_spec keys must be nn.Module subclasses or module names; got {key!r}"
            )
        if qconfig is not None:
            targets.append(label)
    return DynamicSpec(mapping, tuple(sorted(targets)))


def _spec_value(key: object, value: object) -> QConfig | None:
    if value is None or isinstance(value, QConfig):
        return value
    if isinstance(value, QDtype):
        return dynamic_qconfig(value)
    raise TypeError(
        f"qconfig_spec[{key!r}] must be a QConfig, a QDtype or None; got {type(value).__name__}"
    )
