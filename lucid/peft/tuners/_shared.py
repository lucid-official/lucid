"""What the methods that share tensors between their layers have in common: VeRA, RandLoRA, VB-LoRA, Uni-LoRA and TinyLoRA.

Those methods keep one set of tensors for the whole model (random projections,
a vector bank) and give each layer only a few small parameters of its own.  The
shared tensors are *tables* — one entry per adapter name, like a layer's
``ParameterDict`` — held once by the tuner and read by every layer.

A table is in the module tree exactly once: the first layer bound to the tuner's
tables registers them as its attributes (``_Shared.attach``), so ``.to()``,
``state_dict`` and ``named_buffers`` see each tensor once, and the other layers
read them through a plain reference.  The adapter file holds each table once, as
the reference writes it (``base_model.<table>``), not once per layer.

A table may hold the adapter's trainable parameters (VB-LoRA's bank, Uni-LoRA's vector, TinyLoRA's
vectors): ``_TrainedSharedLayer`` trains and freezes them with the adapter of each layer that reads
them.  ``_SettlingTuner`` is for the methods whose layers need to know about one another once all are
in the tree.
"""

import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar, Final, Protocol, override, runtime_checkable

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftConfig, _Check, _check_bool, _TargetedConfig
from lucid.peft._slots import _as_names
from lucid.peft.tuners._layer import _BaseTunerLayer, _entry_parameters
from lucid.peft.tuners._plan import _NO_PARAMETERS, _Injected, _Placement, _Target
from lucid.peft.tuners._tuner import _BaseTuner
from lucid.peft.tuners._files import _FileContext, _FileFormat

#: Where the reference's adapter file keeps a shared table: ``base_model.<table>``, one tensor for the whole model.
_FILE_PREFIX: Final = "base_model."


def _check_key(name: str, value: object) -> int:
    """The PRNG key: any ``int`` (``bool`` excluded) the generator takes as a seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    return value


class _ProjectionConfig(_TargetedConfig):
    """The checks of a method whose shared tables are random projections: the key they are drawn from, and whether a file holds them.

    Each method's config declares the two fields itself, in the reference's order.

    Parameters
    ----------
    projection_prng_key : int, default=0
        Seed of the generator the shared projections are drawn from (on the CPU, then
        moved), so the same key gives the same projections on every device.
    save_projection : bool, default=True
        Write the projections into the adapter file.  ``False`` makes the file small and
        redraws them from ``projection_prng_key`` on load (warns).
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset({"exclude_modules"})
    _field_checks: ClassVar[dict[str, _Check]] = {
        "projection_prng_key": _check_key,
        "save_projection": _check_bool,
    }
    #: The projections' names as a warning spells them (``"vera_A and vera_B"``).
    _projection_names: ClassVar[str] = ""

    projection_prng_key: int
    save_projection: bool

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if not self.save_projection:
            warnings.warn(
                f"Specified to not save {self._projection_names} within the state dictionary, "
                "instead they will be restored using the PRNG key store in "
                "`config.projection_prng_key`. Consider setting `config.save_projection` to "
                "`True` to guarantee restoring the checkpoint correctly on all system "
                "configurations.",
                stacklevel=2,
            )


def _take_entries(
    container: _BufferDict, state_dict: dict[str, Tensor], prefix: str, *rest: object
) -> None:
    """Load pre-hook: a container holds whatever entries it is given, so a fresh one makes a buffer per name in a checkpoint.

    As ``ModuleDict`` takes the keys of its children's: without this a layer's own checkpoint
    (which lists the tables it holds) would not load into a fresh instance of the layer.
    """
    del rest  # missing / unexpected keys and the rest of the hook's arguments: the default loader's
    if container.persistent:
        for key, value in state_dict.items():
            name = key[len(prefix) :]
            if key.startswith(prefix) and "." not in name and name not in container:
                container[name] = lucid.zeros_like(value)


class _BufferDict(nn.Module):
    """Buffers by adapter name: tensors a layer reads and does not train, as ``ParameterDict`` holds parameters.

    A buffer saved in ``state_dict`` as ``<name>`` (so ``vera_A.default``) unless the
    container is not ``persistent``, in which case it is never saved — the
    projections are then restored from the PRNG key.
    """

    def __init__(self, persistent: bool) -> None:
        super().__init__()
        self._persistent = persistent
        self.register_load_state_dict_pre_hook(_take_entries)

    @property
    def persistent(self) -> bool:
        """Whether the buffers are saved in ``state_dict``."""
        return self._persistent

    def __setitem__(self, name: str, tensor: Tensor) -> None:
        self.register_buffer(name, tensor, persistent=self._persistent)

    def __getitem__(self, name: str) -> Tensor:
        buffer = self._buffers.get(name)
        if buffer is None:
            raise KeyError(name)
        return buffer

    def __delitem__(self, name: str) -> None:
        if name not in self._buffers:
            raise KeyError(name)
        # ``delattr`` also drops the name from the persistence bookkeeping.
        delattr(self, name)

    def __contains__(self, name: object) -> bool:
        return name in self._buffers

    def __iter__(self) -> Iterator[str]:
        return iter(self._buffers)

    def __len__(self) -> int:
        return len(self._buffers)


def _base_linear(module: nn.Module) -> nn.Linear:
    """The ``nn.Linear`` a target is: itself, or what the layer already around it wraps."""
    base = module.get_base_layer() if isinstance(module, _BaseTunerLayer) else module
    # Planning has refused every other module type.
    assert isinstance(base, nn.Linear)
    return base


class _Shared:
    """The tables a tuner shares between its layers, and whether a layer holds them in its tree.

    ``tables`` maps the attribute name a layer holds a table under (``"vera_A"``)
    to the table.  Exactly one layer *owns* them (``attach``).
    """

    __slots__ = ("owned", "tables")

    def __init__(self, tables: Mapping[str, nn.Module]) -> None:
        self.tables = dict(tables)
        self.owned = False

    def buffers(self, name: str) -> _BufferDict:
        """The table ``name``, which is a ``_BufferDict``."""
        table = self.tables[name]
        assert isinstance(table, _BufferDict)
        return table

    def parameters_of(self, name: str) -> nn.ParameterDict:
        """The table ``name``, which is a ``ParameterDict``."""
        table = self.tables[name]
        assert isinstance(table, nn.ParameterDict)
        return table

    def attach(self, layer: nn.Module) -> bool:
        """Register the tables as ``layer``'s attributes if no layer holds them yet; whether ``layer`` now does."""
        if self.owned:
            return False
        for name, table in self.tables.items():
            setattr(layer, name, table)
        self.owned = True
        return True


@runtime_checkable
class _Binds(Protocol):
    """A layer that reads the tables its tuner shares."""

    def bind(self, shared: _Shared) -> None:
        """Read the tables from ``shared`` from now on."""
        ...


class _SharingLayer(_BaseTunerLayer):
    """A layer that reads tables its tuner shares, and holds tables of its own until it is bound.

    A layer used alone (outside a tuner) works on tables it holds itself, so it
    saves and loads as any layer; ``bind`` points it at the tuner's instead, and the
    first layer bound registers those in its module tree.  A subclass calls
    ``_hold`` from ``_prepare`` with its own empty tables and reads every table
    through ``_table``.
    """

    #: The attribute names of the tables the layer reads, as the tuner names them.
    _shared_names: ClassVar[tuple[str, ...]] = ()

    def _hold(self, tables: Mapping[str, nn.Module]) -> None:
        for name, table in tables.items():
            setattr(self, name, table)
        self._bound: _Shared | None = None

    def bind(self, shared: _Shared) -> None:
        """Read the tables from the ones the tuner shares; the first layer bound holds them in its tree."""
        if self._bound is shared:
            return
        if not shared.attach(self):
            # Another layer holds the tables in its tree; the ones this layer began with are dead weight.
            for name in self._shared_names:
                delattr(self, name)
        self._bound = shared
        self._rebound()

    def _rebound(self) -> None:
        """Called after ``bind``: drop what was derived from the layer's own tables."""

    def _table(self, name: str) -> nn.Module:
        """The table ``name`` this layer reads: the tuner's if bound, else its own."""
        if self._bound is not None:
            return self._bound.tables[name]
        table = getattr(self, name)
        assert isinstance(table, nn.Module)
        return table

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``, and its entries in the shared tables if the layer owns them."""
        super().delete_adapter(adapter_name)
        if self._bound is None:  # a bound layer's tables are the tuner's to trim
            for name in self._shared_names:
                table = getattr(self, name)
                if isinstance(table, (nn.ParameterDict, _BufferDict)) and adapter_name in table:
                    del table[adapter_name]


class _TrainedSharedLayer(_SharingLayer):
    """A sharing layer whose shared tables hold the adapter's trainable parameters.

    ``_shared_names`` names the tables; each maps an adapter name to a ``Parameter`` or to a module of
    them.  Such a table is the adapter's trainable tensor though no layer owns it, so a layer that
    holds an adapter reports the parameters its shared tables hold for it, and switches their
    ``requires_grad`` with the adapter's.
    """

    def _shared_parameters(self, adapter_name: str) -> list[Parameter]:
        """The parameters the shared tables hold for ``adapter_name``."""
        found: list[Parameter] = []
        for name in self._shared_names:
            table = self._table(name)
            if isinstance(table, (nn.ParameterDict, nn.ModuleDict)) and adapter_name in table:
                found.extend(_entry_parameters(table[adapter_name]))
        return found

    @override
    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        own = super()._adapter_parameters(adapter_name)
        if adapter_name not in self._adapter_names():
            return own
        return [*own, *self._shared_parameters(adapter_name)]

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the named adapters' own tensors and on the shared parameters they read."""
        super().set_requires_grad(adapter_names, requires_grad)
        held = self._adapter_names()
        for name in _as_names(adapter_names):
            if name in held:
                for parameter in self._shared_parameters(name):
                    parameter.requires_grad_(requires_grad)


class _SettlingTuner(_BaseTuner):
    """A tuner that finishes an adapter once its layers are in the model.

    What a layer needs to know about the others — the slice of one draw it takes (Uni-LoRA), the group
    whose vector it reads (TinyLoRA), the projections it shares with its neighbours (Lily) — cannot be
    known while layers are made one at a time.  ``_settle`` runs after every layer of the adapter is in
    place, inside the injection that rolls back if it raises.
    """

    @override
    def _attach_adapter(
        self,
        config: _TargetedConfig,
        adapter_name: str,
        targets: list[_Target],
        saved: list[str],
        placements: list[_Placement],
    ) -> None:
        super()._attach_adapter(config, adapter_name, targets, saved, placements)
        self._settle(config, adapter_name, targets)

    @abstractmethod
    def _settle(self, config: _TargetedConfig, adapter_name: str, targets: list[_Target]) -> None:
        """Bind ``adapter_name``'s layers to one another, now that all of them are in the tree.

        Parameters
        ----------
        config : _TargetedConfig
            The adapter's stored config.
        adapter_name : str
            The adapter just attached.
        targets : list of _Target
            Everything the adapter targets, in the order the model lists it.
        """


class _SharedTablesTuner(_FileFormat, _BaseTuner):
    """A tuner whose layers read tables the tuner owns: provide them per adapter, bind the layers, write them once.

    A subclass names its tables (``_table_names``), builds them (``_new_tables``),
    fills one entry per adapter (``_provide``) and, for the file, says which
    tables the file holds (``_saved_tables``).  Everything else — dropping an
    adapter's entries when it is withdrawn or deleted, binding the layers it
    placed, mapping the tables to the file's keys — is here once.
    """

    #: The attribute names of the shared tables, on the tuner and on the layer that holds them.
    _table_names: ClassVar[tuple[str, ...]]
    #: The tables, made when the first adapter is injected.
    _shared: _Shared
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS

    @classmethod
    @abstractmethod
    def _new_tables(cls, config: _TargetedConfig) -> dict[str, nn.Module]:
        """The empty tables, by attribute name, for the first adapter's ``config``."""

    @abstractmethod
    def _provide(self, adapter_name: str, config: _TargetedConfig, targets: list[_Target]) -> None:
        """Enter ``adapter_name``'s entry in every table, sized for the largest of ``targets``."""

    @classmethod
    def _saved_tables(cls, config: PeftConfig) -> tuple[str, ...]:
        """The tables an adapter file holds under ``config``."""
        del config  # by default every table is saved
        return cls._table_names

    def _table(self, name: str) -> nn.Module:
        """The shared table ``name``."""
        return self._shared.tables[name]

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del (
            model,
            config,
        )  # the layers' own tensors train; a trainable shared table is its layers' (`_adapter_parameters`)
        return []

    # ── the adapters ──────────────────────────────────────────────────────

    @override
    def _inject(self, adapter_name: str, config: _TargetedConfig) -> _Injected:
        if "_shared" not in self.__dict__:
            self._shared = _Shared(self._new_tables(config))
        return super()._inject(adapter_name, config)

    @override
    def _new_slots(
        self, config: _TargetedConfig, adapter_name: str, targets: list[_Target], saved: list[str]
    ) -> list[_Placement]:
        self._provide(adapter_name, config, targets)
        try:
            placements = super()._new_slots(config, adapter_name, targets, saved)
        except BaseException:
            self._drop(adapter_name)
            raise
        for placement in placements:
            if isinstance(placement.slot, _Binds):
                placement.slot.bind(self._shared)
        return placements

    def _drop(self, adapter_name: str) -> None:
        for table in self._shared.tables.values():
            if isinstance(table, (nn.ParameterDict, _BufferDict)) and adapter_name in table:
                del table[adapter_name]

    @override
    def _withdraw(self, adapter_name: str, placements: list[_Placement]) -> None:
        super()._withdraw(adapter_name, placements)
        self._drop(adapter_name)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name`` from the model, and its entries in the tuner's shared tables."""
        super().delete_adapter(adapter_name)
        self._drop(adapter_name)

    # ── the adapter file ──────────────────────────────────────────────────

    @override
    @classmethod
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        """The saved tables, as the file's model-wide tensors (``base_model.<table>``)."""
        names = cls._saved_tables(context.config)
        if not names:
            return {}
        for path, layer in context.layers:
            held = getattr(layer, names[0], None)
            if isinstance(layer, _Binds) and held is not None and context.adapter in held:
                # The layer holding the tables in its tree: the others only read them.
                for name in names:
                    table = getattr(layer, name)
                    if isinstance(table, _BufferDict) and not table.persistent:
                        raise ValueError(
                            f"Model was initialised to not save {' and '.join(names)} but config "
                            "now specifies to save projection! Set `config.save_projection` to "
                            "`False`."
                        )
                where = f"{context.prefix}{path}"
                return {
                    f"{where}.{name}.{context.adapter}": f"{_FILE_PREFIX}{name}" for name in names
                }
        return {}

    @classmethod
    def _file_keys(cls) -> dict[str, str]:
        """Table name → the key the file holds it under."""
        return {name: f"{_FILE_PREFIX}{name}" for name in cls._table_names}

    @classmethod
    def _covers(
        cls, config: PeftConfig, loaded: Mapping[str, Tensor], layers: list[_BaseTunerLayer]
    ) -> bool:
        """Whether the file's tables (by table name) can stand in for the ones built here for ``layers``.

        The file holds the tables drawn for the largest layer of the model that
        saved it; a model that adapts fewer layers drew smaller ones.  By default
        a file's tables must be the size the model drew (nothing is resized).
        """
        del config, loaded, layers  # the default accepts no other size
        return False

    @classmethod
    def _fit_tables(cls, context: _FileContext, state: Mapping[str, Tensor]) -> None:
        """Give the adapter's buffer tables the size the file's have, if every layer still fits them."""
        layers = [layer for _, layer in context.layers if context.adapter in layer._adapter_names()]
        loaded = {name: state[key] for name, key in cls._file_keys().items()}
        owner = next(
            (l for l in layers if context.adapter in getattr(l, cls._table_names[0], ())), None
        )
        if owner is None or not cls._covers(context.config, loaded, layers):
            return
        for name, tensor in loaded.items():
            table = getattr(owner, name)
            if isinstance(table, _BufferDict):
                current = table[context.adapter]
                if tuple(current.shape) != tuple(tensor.shape):
                    table[context.adapter] = lucid.zeros(
                        *tensor.shape, device=current.device, dtype=current.dtype
                    )

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Require the saved tables, resize the model's to them where they fit, and drop the per-layer copies a file may carry."""
        keys = cls._file_keys()
        # A file the reference wrote holds a copy of each table under every layer.
        kept = {
            key: value
            for key, value in state.items()
            if key in keys.values() or key.rpartition(".")[2] not in keys
        }
        saved = cls._saved_tables(context.config)
        if saved:
            if not all(key in kept for key in keys.values()):
                raise ValueError(
                    f"Specified to load {' and '.join(saved)} from state dictionary however they "
                    "were not present!"
                )
            cls._fit_tables(context, kept)
        return kept


class _ProjectionTuner(_SharedTablesTuner):
    """A shared-table tuner whose tables are random projections: a file may leave them out and the PRNG key restores them.

    A config of such a method has ``save_projection`` and ``projection_prng_key``.
    With ``save_projection`` the file holds the tables (``_SharedTablesTuner``);
    without it the file does not, and the model's own — drawn from the key — stay.
    """

    @classmethod
    @abstractmethod
    def _saves_projection(cls, config: PeftConfig) -> bool:
        """Whether ``config`` asks for the projections to be written into the file."""

    @classmethod
    @abstractmethod
    def _prng_key(cls, config: PeftConfig) -> int:
        """The key the projections are drawn from."""

    @override
    @classmethod
    def _saved_tables(cls, config: PeftConfig) -> tuple[str, ...]:
        return cls._table_names if cls._saves_projection(config) else ()

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        for held in self.peft_config.values():
            if self._prng_key(held) != self._prng_key(config):
                raise ValueError(
                    f"{type(self).__name__[:-5]} PRNG initialisation key must be the same for all "
                    f"adapters. Got projection_prng_key={self._prng_key(config)} but previous "
                    f"config had {self._prng_key(held)}."
                )
        saved = sorted({self._saves_projection(c) for c in [*self.peft_config.values(), config]})
        if len(saved) > 1:
            raise ValueError(
                f"{type(self).__name__[:-5]} projection weights must be saved for all adapters "
                f"or none, but got multiple different values: {saved}"
            )

    @override
    @classmethod
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        if cls._saves_projection(context.config):
            return super()._from_file(context, state)
        keys = set(cls._file_keys().values())
        names = " and ".join(cls._table_names)
        if keys <= state.keys():
            warnings.warn(
                f"Specified to not load {names} from state dictionary however they are present in "
                "state dictionary! Consider using them to ensure checkpoint loading is correct on "
                "all platforms using `peft_config.save_projection = True`",
                stacklevel=4,
            )
        else:
            warnings.warn(
                f"Specified to not load {names} from state dictionary. This means we will be "
                "relying on PRNG initialisation to restore these projections using "
                "`config.projection_prng_key`, which may not be accurate on all system "
                "configurations.",
                stacklevel=4,
            )
        return {key: value for key, value in state.items() if key not in keys}
