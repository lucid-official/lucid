"""``Optimizer.state``: the per-parameter state, read from where it lives.

An engine optimizer keeps its buffers in the engine, so ``state`` cannot be a
dict the optimizer fills: whatever it held would be a copy that goes stale at
the next step.  :class:`EngineStateView` reads the engine each time it is
asked, keyed by :class:`~lucid.nn.Parameter` as the reference framework keys
``optimizer.state``, and hands out lucid tensors and Python numbers.  An
optimizer that keeps its state in Python (``SparseAdam``, a user subclass)
holds an ordinary ``defaultdict`` keyed the same way; its checkpoint form is
converted here too, so there is one rule for how a state entry crosses between
the two.

The tensors an engine hands over are copies taken at the read: the engine
replaces its buffers rather than updating them in place, so a tensor held
across a step shows the state of the moment it was read.  Index ``state``
again to see the present.
"""

from collections import defaultdict
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from dataclasses import dataclass
from typing import Protocol, override

from lucid._dispatch import _wrap
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.optim._state_load import _group_params, _loaded_state_like

from lucid._C import engine as _C_engine

StateEntry = MutableMapping[str, object]
StateMap = MutableMapping[Parameter, StateEntry]


class _BufferEngine(Protocol):
    """The part of an engine optimizer its state is read and written through."""

    def state_buffers(self) -> Sequence[tuple[str, Sequence[_C_engine.TensorImpl | None]]]: ...

    def load_state_buffers(
        self, bufs: Sequence[tuple[str, Sequence[_C_engine.TensorImpl | None]]]
    ) -> None: ...


class StateOwner(Protocol):
    """The optimizer a view reads: its groups, its engines and its not-yet-built state."""

    param_groups: list[dict[str, object]]
    _engines_built: bool

    @property
    def _engine_optims(self) -> Sequence[_BufferEngine]: ...

    def _save_state(self) -> dict[int, dict[str, object]]: ...


def slot_entry(
    buffers: Sequence[tuple[str, Sequence[_C_engine.TensorImpl | None]]],
    step_count: int,
    slot: int,
    convert: Callable[[_C_engine.TensorImpl], object],
) -> dict[str, object]:
    """Parameter ``slot`` of one group's state, each buffer passed through ``convert``.

    Buffers shaped like the parameter (``exp_avg`` ...) and per-parameter
    scalars (``step``, ``mu_product``) come under the engine's names, which are
    the reference framework's.  A parameter never updated has no entry.  An
    engine with one count per group, not per parameter, gives it as ``step``.

    Parameters
    ----------
    buffers : sequence of (str, sequence of TensorImpl or None)
        What the engine's ``state_buffers()`` returned for the group.
    step_count : int
        The engine's group-wide step count, ``0`` when it has none.
    slot : int
        The parameter's position in the group.
    convert : callable
        Applied to each buffer: a host copy for a checkpoint, a tensor for the view.

    Returns
    -------
    dict
        Name to converted buffer.
    """
    entry: dict[str, object] = {}
    for name, tensors in buffers:
        buf = tensors[slot] if slot < len(tensors) else None
        if buf is not None:
            entry[name] = convert(buf)
    if step_count != 0 and "step" not in entry:
        entry["step"] = step_count
    return entry


def group_entries(
    engine: _BufferEngine, count: int, convert: Callable[[_C_engine.TensorImpl], object]
) -> list[dict[str, object]]:
    """The state of each of one group's ``count`` parameters, one :func:`slot_entry` apiece."""
    buffers = engine.state_buffers()
    steps = int(getattr(engine, "step_count", 0) or 0)
    return [slot_entry(buffers, steps, slot, convert) for slot in range(count)]


def _keep(buf: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Leave a buffer as the engine gave it (only whether it exists matters)."""
    return buf


@dataclass(frozen=True, slots=True)
class _Where:
    """Where a parameter sits in its optimizer: group, slot in the group, flat index."""

    group: int
    slot: int
    flat: int
    param: Parameter


def flat_params(groups: Sequence[dict[str, object]]) -> list[Parameter]:
    """Every parameter of ``groups``, in the flat order a checkpoint's ``state`` is indexed by."""
    return [p for group in groups for p in _group_params(group)]


def _locate(owner: StateOwner, param: object) -> _Where | None:
    """``param``'s place in ``owner``, ``None`` when it is not one of its parameters."""
    flat = 0
    for g, group in enumerate(owner.param_groups):
        for slot, p in enumerate(_group_params(group)):
            if p is param:
                return _Where(g, slot, flat, p)
            flat += 1
    return None


class EngineEntry(MutableMapping[str, object]):
    """One parameter's state, read from the engine on every access.

    Assigning an existing key writes through to the engine.  A key the
    parameter has no state under cannot be added: the engine decides what it
    keeps.
    """

    def __init__(self, owner: StateOwner, where: _Where) -> None:
        self._owner = owner
        self._where = where

    def _read(self) -> dict[str, object]:
        owner, where = self._owner, self._where
        if not owner._engines_built:
            # Nothing is bound yet: the state is a checkpoint waiting for the
            # engines, or the seed an optimizer gives its parameters at build.
            saved = owner._save_state().get(where.flat, {})
            return {
                name: _wrap(_loaded_state_like(name, value, where.param._impl))
                for name, value in saved.items()
            }
        engine = owner._engine_optims[where.group]
        buffers = engine.state_buffers()
        steps = int(getattr(engine, "step_count", 0) or 0)
        return slot_entry(buffers, steps, where.slot, _wrap)

    @override
    def __getitem__(self, name: str) -> object:
        return self._read()[name]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    @override
    def __len__(self) -> int:
        return len(self._read())

    @override
    def items(self) -> ItemsView[str, object]:
        """The entries, read once."""
        return self._read().items()

    @override
    def values(self) -> ValuesView[object]:
        """The values, read once."""
        return self._read().values()

    @override
    def __setitem__(self, name: str, value: object) -> None:
        owner, where = self._owner, self._where
        if not owner._engines_built:
            raise RuntimeError(
                f"optimizer state is written once the optimizer has stepped; before "
                f"that, load it with load_state_dict() (state[{name!r}] assigned)"
            )
        engine = owner._engine_optims[where.group]
        count = len(_group_params(owner.param_groups[where.group]))
        held = {
            n
            for n, tensors in engine.state_buffers()
            if where.slot < len(tensors) and tensors[where.slot] is not None
        }
        if name not in held:
            raise KeyError(
                f"this parameter has no state entry {name!r} to replace "
                f"(it has {sorted(held)}); the optimizer decides what it keeps"
            )
        slots: list[_C_engine.TensorImpl | None] = [None] * count
        slots[where.slot] = _loaded_state_like(name, value, where.param._impl)
        engine.load_state_buffers([(name, slots)])

    @override
    def __delitem__(self, name: str) -> None:
        raise TypeError(
            f"cannot delete state[{name!r}]: an engine optimizer's state is replaced "
            "as a whole through load_state_dict()"
        )

    @override
    def __repr__(self) -> str:
        return repr(self._read())


class EngineStateView(MutableMapping[Parameter, StateEntry]):
    """The state of an engine optimizer, keyed by parameter, read live.

    A parameter of the optimizer maps to its :class:`EngineEntry`, empty until
    the parameter has been updated; iteration, ``len`` and ``in`` cover the
    parameters that have state.  Each read copies the group's buffers out of
    the engine, so inspect it, do not poll it in a hot loop.  Entries cannot
    be added or removed; use ``load_state_dict`` to replace the state.
    """

    def __init__(self, owner: StateOwner) -> None:
        self._owner = owner

    def _with_state(self) -> list[Parameter]:
        owner = self._owner
        if not owner._engines_built:
            flat = flat_params(owner.param_groups)
            return [flat[i] for i, entry in sorted(owner._save_state().items()) if entry]
        out: list[Parameter] = []
        for group, engine in zip(owner.param_groups, owner._engine_optims, strict=True):
            params = _group_params(group)
            out += [
                p
                for p, entry in zip(params, group_entries(engine, len(params), _keep), strict=True)
                if entry
            ]
        return out

    @override
    def __getitem__(self, param: Parameter) -> StateEntry:
        where = _locate(self._owner, param)
        if where is None:
            raise KeyError("the parameter is not one of this optimizer's")
        return EngineEntry(self._owner, where)

    @override
    def __contains__(self, param: object) -> bool:
        where = _locate(self._owner, param)
        return where is not None and bool(EngineEntry(self._owner, where)._read())

    @override
    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._with_state())

    @override
    def __len__(self) -> int:
        return len(self._with_state())

    @override
    def __setitem__(self, param: Parameter, entry: StateEntry) -> None:
        raise TypeError(
            "an engine optimizer's state is replaced through load_state_dict(); "
            "assign to a key of an existing entry to replace one buffer"
        )

    @override
    def __delitem__(self, param: Parameter) -> None:
        raise TypeError("an engine optimizer's state is replaced through load_state_dict()")

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} parameters with state)"


class ReadOnlyMap[K, V](MutableMapping[K, V]):
    """A mapping that refuses writes, for state an optimizer keeps in its own fields.

    ``absent`` makes the value a key without an entry reads as, without adding
    it, as the reference's ``defaultdict`` state reads a parameter with none.
    """

    def __init__(
        self, data: Mapping[K, V], what: str, absent: Callable[[], V] | None = None
    ) -> None:
        self._data = data
        self._what = what
        self._absent = absent

    @override
    def __getitem__(self, key: K) -> V:
        if key not in self._data and self._absent is not None:
            return self._absent()
        return self._data[key]

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._data

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __setitem__(self, key: K, value: V) -> None:
        raise TypeError(self._what)

    @override
    def __delitem__(self, key: K) -> None:
        raise TypeError(self._what)

    @override
    def __repr__(self) -> str:
        return repr(dict(self._data))


def held_checkpoint(
    held: Mapping[Parameter, StateEntry],
    groups: Sequence[dict[str, object]],
    to_host: Callable[[_C_engine.TensorImpl], object],
) -> dict[int, dict[str, object]]:
    """State an optimizer holds in Python, in checkpoint form: flat index, host arrays.

    A tensor becomes ``to_host`` of its impl; any other value is kept as it is.
    A parameter with no entries is left out.

    Parameters
    ----------
    held : mapping
        The optimizer's state, by parameter.
    groups : sequence of dict
        Its param groups, which fix the flat indices.
    to_host : callable
        Makes a host copy of a tensor's impl.

    Returns
    -------
    dict
        Flat parameter index to entry.
    """
    index = {id(p): i for i, p in enumerate(flat_params(groups))}
    return {
        index[id(p)]: {k: to_host(v._impl) if isinstance(v, Tensor) else v for k, v in e.items()}
        for p, e in held.items()
        if e and id(p) in index
    }


def held_from_checkpoint(
    state: Mapping[int, Mapping[str, object]], groups: Sequence[dict[str, object]]
) -> defaultdict[Parameter, StateEntry]:
    """The inverse of :func:`held_checkpoint`: an array becomes a tensor on its parameter's device.

    Parameters
    ----------
    state : mapping
        A checkpoint's ``state``, by flat parameter index.
    groups : sequence of dict
        The optimizer's param groups.

    Returns
    -------
    defaultdict
        The state by parameter, ready to be an optimizer's own.

    Raises
    ------
    IndexError
        An entry for a parameter index the optimizer does not have.
    """
    params = flat_params(groups)
    out: defaultdict[Parameter, StateEntry] = defaultdict(dict)
    for idx, entry in state.items():
        if not 0 <= idx < len(params):
            raise IndexError(
                f"load_state_dict: state for parameter {idx}, "
                f"but the optimizer has {len(params)} parameters"
            )
        device = params[idx]._impl.device
        out[params[idx]] = {k: _revive(v, device) for k, v in entry.items()}
    return out


def _revive(value: object, device: _C_engine.Device) -> object:
    """A checkpointed array as a tensor on ``device``; anything else unchanged."""
    if isinstance(value, Tensor) or not hasattr(value, "shape"):
        return value
    impl = _C_engine.TensorImpl(value, _C_engine.Device.CPU, False)
    return _wrap(impl if device == _C_engine.Device.CPU else impl.transfer_to_device(device, False))
