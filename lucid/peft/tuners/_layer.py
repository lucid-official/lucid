"""The layer half of a tuner: the base every method's layers share, and the helpers on its tables.

A tuner family's layers are ``_BaseTunerLayer`` subclasses, one per kind of
module the method can wrap, found by the wrapped module's type — the
``base_type=`` each concrete layer joins its family's table with.  A layer
holds any number of adapters, one entry per name in each of the tables
``adapter_layer_names`` lists; the helpers here read and name those entries.
"""

import functools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._slots import _as_names, _SlotState


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


@functools.cache
def _module_dict_attributes() -> frozenset[str]:
    return frozenset(dir(nn.ModuleDict()))


def _check_adapter_name(adapter_name: object) -> str:
    """``adapter_name`` if it can be the key of an adapter in a layer's ``ModuleDict``\\ s."""
    if not isinstance(adapter_name, str):
        raise TypeError(f"adapter_name must be a str, got {adapter_name!r}")
    if not adapter_name or "." in adapter_name:
        raise ValueError(f"adapter_name must be a non-empty name without '.', got {adapter_name!r}")
    if adapter_name in _module_dict_attributes():
        raise ValueError(
            f"adapter_name {adapter_name!r} is an attribute of nn.ModuleDict, which holds the "
            "adapters of a layer; choose another name"
        )
    return adapter_name


def _dotted(*parts: str) -> str:
    """The non-empty ``parts`` joined with dots."""
    return ".".join(part for part in parts if part)


def _entry_parameters(entry: nn.Module | Parameter | None) -> list[Parameter]:
    """The parameters of one table entry: a module's own, or the bare parameter."""
    if isinstance(entry, nn.Module):
        return list(entry.parameters())
    return [] if entry is None else [entry]


def _parameter_at(table: nn.ParameterDict, name: str) -> Parameter:
    """``table[name]``, which a ``ParameterDict`` types as optional: it must hold that parameter."""
    entry = table[name]
    if entry is None:
        raise KeyError(f"{name!r} holds no parameter")
    return entry


def _entry_leaves(entry: nn.Module | Parameter | None) -> list[str]:
    """The state-dict suffixes under one table entry: a module's keys, or ``""`` for a bare parameter."""
    return list(entry.state_dict()) if isinstance(entry, nn.Module) else [""]


def _tuner_layers(model: nn.Module) -> list[tuple[str, _BaseTunerLayer]]:
    # Not ``_named``: a ``type[T]`` parameter takes concrete classes only.
    return [(name, m) for name, m in model.named_modules() if isinstance(m, _BaseTunerLayer)]


class _BaseTunerLayer(nn.Module, ABC):
    """The base of a tuner's layers: the wrapped module and the state forward branches on.

    A tuner family's root class — the direct subclass of this one — starts
    a table of layer classes; each concrete layer joins it with
    ``base_type=`` — the kind of module it wraps — and the tuner finds a
    target's layer class there.  A layer names its per-adapter tables in
    ``adapter_layer_names``: ``ModuleDict``\\ s, or ``ParameterDict``\\ s
    where the factors are bare parameters.  It holds any number of
    adapters, one entry per name in each of those tables.

    What the forward pass branches on — ``active_adapter``,
    ``disable_adapters``, ``merged`` — are plain attributes in the
    instance ``__dict__``, not properties or descriptors as in the
    reference: ``lucid.compile`` keys an executable on each module's
    public scalar ``__dict__`` attributes, and state kept anywhere else
    would let a compiled model replay a branch it has left (LCD-703).
    ``active_adapter`` is one name or a list of names; a layer that holds
    none of them runs its base alone.  They change only through
    ``set_adapter``, ``delete_adapter``, ``enable_adapters``, ``merge``
    and ``unmerge``.
    """

    #: The per-adapter table attributes (``ModuleDict`` / ``ParameterDict``).
    adapter_layer_names: ClassVar[tuple[str, ...]] = ()
    #: Wrapped module type → layer class, one table per family.
    _layer_types: ClassVar[dict[type[nn.Module], type[_BaseTunerLayer]]] = {}
    #: The module type (or tuple of module types) this layer wraps, from ``base_type=``.
    _base_type: ClassVar[type[nn.Module] | tuple[type[nn.Module], ...]] = nn.Module

    @override
    def __init_subclass__(
        cls, *, base_type: type[nn.Module] | None = None, family: bool = False, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        # A direct subclass starts a table; ``family=True`` starts one for a
        # class that shares a base with other methods (the LyCORIS layers).
        if family or _BaseTunerLayer in cls.__bases__:
            cls._layer_types = {}
        if base_type is not None:
            cls._base_type = base_type
            cls._layer_types[base_type] = cls

    def __init__(self, base_layer: nn.Module) -> None:
        kind = type(self)._base_type
        if not isinstance(base_layer, kind):
            shown = (
                ", ".join(f"nn.{k.__name__}" for k in kind)
                if isinstance(kind, tuple)
                else f"nn.{kind.__name__}"
            )
            raise TypeError(
                f"{type(self).__name__} adapts {shown} in lucid.peft, "
                f"got {type(base_layer).__name__}"
            )
        if base_layer.has_uninitialized_parameters():
            raise ValueError(
                f"{type(base_layer).__name__} builds its parameters on its first input; "
                "run it once before adapting it"
            )
        super().__init__()
        self.base_layer = base_layer
        self.merged_adapters: list[str] = []
        #: Name (or names) of the adapter the forward pass applies.
        self.active_adapter: str | list[str] = ""
        #: Whether the forward pass runs the base layer alone.
        self.disable_adapters = False
        #: Whether an adapter is folded into the base weight (``merged_adapters`` non-empty).
        self.merged = False

    @property
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""
        return [name for name in _as_names(self.active_adapter) if name]

    def get_base_layer(self) -> nn.Module:
        """Return the wrapped module, whose parameters are the base model's own."""
        return self.base_layer

    @property
    def weight(self) -> Parameter:
        """The base layer's weight (unchanged unless an adapter is merged into it)."""
        weight = self.get_base_layer().weight
        if isinstance(weight, Parameter):
            return weight
        raise AttributeError(f"{type(self.get_base_layer()).__name__} has no weight parameter")

    @property
    def bias(self) -> Tensor | None:
        """The base layer's bias (a buffer on a quantized base), or ``None``."""
        bias = getattr(self.get_base_layer(), "bias", None)
        return bias if isinstance(bias, (Parameter, Tensor)) else None

    def enable_adapters(self, enabled: bool) -> None:
        """Apply the adapter in the forward pass (``True``) or run the base alone (``False``).

        Disabling a merged layer unmerges it first, here rather than on
        the next call, so the forward pass never writes a weight — a
        compiled forward must not replay that write.

        Parameters
        ----------
        enabled : bool
            Whether the adapter takes part.  Gradient flags are left as
            they are: a frozen adapter stays frozen when it is re-enabled.
        """
        if not enabled and self.merged:
            self.unmerge()
        self.disable_adapters = not enabled

    def _activate(self, adapter_names: str | list[str]) -> None:
        self.active_adapter = adapter_names if isinstance(adapter_names, str) else [*adapter_names]

    def _check_activation(self, adapter_names: list[str]) -> None:
        del adapter_names  # any set of adapters may apply together in a layer

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapters, and train exactly those unless ``inference_mode``.

        The names this layer holds nothing for are skipped by the forward
        pass.  A merged layer is unmerged first, since the active adapters
        are no longer the merged ones.

        Parameters
        ----------
        adapter_names : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the activated adapters as well.
        """
        if self.merged:
            warnings.warn(
                "Adapter cannot be set when the model is merged. Unmerging the model first.",
                stacklevel=2,
            )
            self.unmerge()
        wanted = _as_names(adapter_names)
        for held in self._adapter_names():
            self.set_requires_grad(held, held in wanted and not inference_mode)
        self._activate(adapter_names)

    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the parameters of the named adapters this layer holds."""
        wanted = _as_names(adapter_names)
        for table in self._tables():
            for name, module in table.items():
                if name in wanted:
                    module.requires_grad_(requires_grad)

    def delete_adapter(self, adapter_name: str) -> None:
        """Drop ``adapter_name`` from this layer; nothing happens if it holds no such adapter.

        An active adapter that is deleted leaves the active set; the tuner
        chooses what takes its place, the same on every layer.
        """
        for table in self._tables():
            if adapter_name in table:
                del table[adapter_name]
        active = self.active_adapters
        if adapter_name in active:
            self._activate([name for name in active if name != adapter_name])

    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        """The requested adapters (the active ones by default) not merged yet.

        A disabled layer runs its base alone, so merging into it would make
        it run the adapter while it reads as disabled; it is refused.  With
        no names given, the active adapters this layer holds are taken.
        """
        if self.disable_adapters:
            raise RuntimeError(
                "the adapter is disabled; enable it (leave disable_adapter()) before merging"
            )
        held = self._adapter_names()
        if adapter_names is None:
            names = [name for name in self.active_adapters if name in held]
        else:
            names = list(adapter_names)
            unknown = [name for name in names if name not in held]
            if unknown:
                raise ValueError(f"unknown adapter(s) {unknown}; this layer holds {held}")
        return [name for name in names if name not in self.merged_adapters]

    def _adapter_names(self) -> list[str]:
        """The adapter names this layer holds."""
        tables = self._tables()
        return list(tables[0]) if tables else []

    def _state(self) -> _SlotState:
        return _SlotState(not self.disable_adapters, self.active_adapters, [*self.merged_adapters])

    def _tables(self) -> list[nn.ModuleDict[nn.Module] | nn.ParameterDict]:
        """The per-adapter tables ``adapter_layer_names`` names."""
        return [
            table
            for name in self.adapter_layer_names
            if isinstance(table := getattr(self, name), (nn.ModuleDict, nn.ParameterDict))
        ]

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        return [
            p
            for table in self._tables()
            if adapter_name in table
            for p in _entry_parameters(table[adapter_name])
        ]

    def _disk_table(self, table_name: str) -> str:
        """The name an adapter file gives the table ``table_name``; the reference's layout."""
        return table_name

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        return {
            _dotted(name, adapter_name, leaf): _dotted(self._disk_table(name), leaf)
            for name, table in zip(self.adapter_layer_names, self._tables(), strict=True)
            if adapter_name in table
            for leaf in _entry_leaves(table[adapter_name])
        }

    def _unwrap(self, keep_trained: bool) -> nn.Module:
        if self.merged and not keep_trained:
            self.unmerge()
        return self.base_layer

    # ``update_layer`` — creating an adapter — is each family's own, not an
    # abstract method here: its parameters are the method's hyper-parameters,
    # and one signature for all would be either ``**kwargs`` or wrong.

    @abstractmethod
    def get_delta_weight(self, adapter: str) -> Tensor:
        """The change ``adapter`` makes to the base weight, shaped like it."""

    @abstractmethod
    def _merge_targets(self) -> dict[str, Parameter]:
        """The base parameters ``merge`` writes, by their name under this layer."""

    @abstractmethod
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        """Each merge target as it would read with ``adapter`` folded in now, in the target's dtype."""

    @abstractmethod
    def _fold(self, adapter: str) -> None:
        """Write ``adapter`` into the merge targets, in place."""

    @abstractmethod
    def _unfold(self) -> None:
        """Take every merged adapter back out of the merge targets, in place."""

    def _check_finite_merge(self, adapter: str) -> None:
        """Raise ``ValueError`` if merging ``adapter`` would write NaN or inf."""
        with lucid.no_grad():
            finite = all(
                bool(lucid.isfinite(value).all().item())
                for value in self._merged_values(adapter).values()
            )
        if not finite:
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {adapter} seems to be broken"
            )

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Fold the update into the base weights, in place.

        The weights keep their identity — they are written, not replaced —
        so whatever holds them (an optimizer, a tied embedding) sees the
        merged values.  That is also why a weight shared with another
        module must not be merged into; the tuner's ``merge_adapter`` and
        ``merge_and_unload`` refuse it, as this layer alone cannot see the
        rest of the model.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the merged weights for NaN / inf first and leave the base
            untouched if there are any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones this layer holds by default.

        Raises
        ------
        ValueError
            An adapter name this layer does not hold, or ``safe_merge``
            found a non-finite merged weight.
        RuntimeError
            The adapter is disabled, or the layer's own rules refuse the merge
            (``lora_bias`` on a base with no bias).
        NotImplementedError
            This layer's update cannot be written into its base weights.
        """
        pending = self._pending_merges(adapter_names)
        if not pending:
            warnings.warn(f"Already merged: {self.merged_adapters}. Nothing to do.", stacklevel=2)
        if safe_merge:
            for adapter in pending:
                self._check_finite_merge(adapter)
        with lucid.no_grad():
            for adapter in pending:
                self._fold(adapter)
                self.merged_adapters.append(adapter)
                self.merged = True

    def unmerge(self) -> None:
        """Take every merged update back out of the base weights, in place.

        The result equals the original weights up to the rounding of the
        round trip.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.", stacklevel=2)
            return
        with lucid.no_grad():
            self._unfold()
        self.merged_adapters.clear()
        self.merged = False

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        """Whether ``adapter_name`` on this layer can be converted to a plain LoRA update."""
        del adapter_name
        return False
