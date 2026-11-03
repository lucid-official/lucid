"""A checkpoint, staged for a compiled optimizer: every entry checked before any is applied.

``load_state_dict`` of a compiled optimizer replaces its state with the
checkpoint's (see :meth:`_CompiledStepBase.load_state_dict`).  Replacing is
only safe when nothing in the checkpoint can fail halfway, so each parameter's
entry is converted and checked here first — a scalar that is not a number, a
buffer not shaped like its parameter — and the optimizer is touched only with
what came out.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import SupportsFloat, SupportsInt

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid._dtype import _float32_or_wider
from lucid.optim._state_load import _loaded_state_like
from lucid.optim._state_view import ReadOnlyMap, StateEntry, StateMap, flat_params


@dataclass(frozen=True, slots=True)
class StagedEntry:
    """What a checkpoint says about one parameter, converted and checked.

    ``step`` is its step count (``None`` when the entry has none), ``scalars``
    its per-parameter scalar state rounded as the eager engine keeps it, and
    ``buffers`` its state buffers in the parameter's dtype and on its device.
    """

    step: int | None
    scalars: dict[str, float]
    buffers: dict[str, _C_engine.TensorImpl]


def stage_entry(
    index: int,
    entry: dict[str, object],
    param: Tensor,
    *,
    scalar_names: Collection[str],
    buffer_names: Collection[str],
) -> StagedEntry:
    """Convert and check parameter ``index``'s checkpoint ``entry`` against ``param``.

    Parameters
    ----------
    index : int
        The parameter's flat index, for the messages.
    entry : dict
        The parameter's checkpoint entry.
    param : Tensor
        The live parameter the entry is for.
    scalar_names : collection of str
        The per-parameter scalars the optimizer keeps next to ``step``.
    buffer_names : collection of str
        The state buffers it can hold.  An entry the optimizer has no use for
        (another optimizer's buffer) is left out.

    Raises
    ------
    TypeError
        A ``step`` or scalar that is not a number.
    lucid._C.engine.ShapeMismatch
        A buffer that is not shaped like the parameter.
    """
    from lucid.compile._optim.compiler import _round_state_scalar

    step: int | None = None
    scalars: dict[str, float] = {}
    buffers: dict[str, _C_engine.TensorImpl] = {}
    for name, value in entry.items():
        if name == "step":
            if not isinstance(value, SupportsInt):
                raise TypeError(f"load_state_dict: step of parameter {index} is not a number")
            step = int(value)
        elif name in scalar_names:
            if not isinstance(value, SupportsFloat):
                raise TypeError(f"load_state_dict: {name!r} of parameter {index} is not a number")
            scalars[name] = _round_state_scalar(float(value), param._impl.dtype)
        elif name in buffer_names:
            src = _loaded_state_like(name, value, param._impl)
            if list(src.shape) != list(param.shape):
                raise _C_engine.ShapeMismatch(
                    f"load_state_dict: state {name!r} of parameter {index} has "
                    f"shape {tuple(src.shape)}, expected {tuple(param.shape)}"
                )
            buffers[name] = src
    return StagedEntry(step, scalars, buffers)


def stage_checkpoint(
    state: dict[int, dict[str, object]],
    params: Sequence[Tensor],
    scalar_names: Collection[str],
    buffer_names: Callable[[int], Collection[str]],
) -> dict[int, StagedEntry]:
    """Every parameter's checkpoint entry, converted and checked, touching nothing.

    Parameters
    ----------
    state : dict
        The checkpoint's per-parameter state, keyed by flat parameter index.
        An empty entry, or an index the optimizer has no parameter for, is
        left out.
    params : sequence of Tensor
        The optimizer's parameters, by flat index.
    scalar_names : collection of str
        The per-parameter scalars the optimizer keeps next to ``step``.
    buffer_names : callable
        For a parameter index, the state buffers it can hold.
    """
    staged: dict[int, StagedEntry] = {}
    for raw_idx, entry in state.items():
        i = int(raw_idx)
        if 0 <= i < len(params) and entry:
            staged[i] = stage_entry(
                i, entry, params[i], scalar_names=scalar_names, buffer_names=buffer_names(i)
            )
    return staged


_READ_ONLY_STATE = "a compiled optimizer's state is replaced through load_state_dict()"


def _empty_compiled_entry() -> StateEntry:
    return ReadOnlyMap({}, _READ_ONLY_STATE)


def export_compiled_entries(
    arrays: tuple[
        Sequence[Tensor],
        Sequence[bool],
        Sequence[int],
        Sequence[dict[str, float]],
        Sequence[dict[str, Tensor]],
    ],
    hooks: tuple[bool, Callable[[int], object], Sequence[str], Callable[[int, str], bool]],
    *,
    convert: Callable[[Tensor], object] | None = None,
) -> dict[int, dict[str, object]]:
    """Per-parameter state by flat index: converted host buffers or live Tensors.

    Parameters
    ----------
    arrays : tuple
        Flat parameter list, initialized flags, step counts, scalar state, and buffers.
    hooks : tuple
        Step export flag, step value hook, scalar state names, and buffer export predicate.
    convert : callable or None, optional
        Optional tensor-to-host converter; when ``None``, returns live detached tensors.
    """
    import lucid as _lucid

    params, initialized, steps, pstate, state_bufs = arrays
    exports_step, step_entry, pstate_names, exports_state = hooks

    out: dict[int, dict[str, object]] = {}
    for i, p in enumerate(params):
        if not initialized[i]:
            continue
        entry: dict[str, object] = {}
        if exports_step:
            step_val = step_entry(i)
            entry["step"] = (
                step_val
                if convert is not None or isinstance(step_val, int)
                else _lucid.tensor(steps[i], dtype=_lucid.int64)
            )
        scalar_dt = _float32_or_wider(p.dtype)
        for name in pstate_names:
            if (value := pstate[i].get(name)) is not None:
                t = _lucid.tensor(value, dtype=scalar_dt)
                entry[name] = convert(t) if convert is not None else t
        for name, buf in state_bufs[i].items():
            if exports_state(i, name):
                entry[name] = convert(buf) if convert is not None else buf.detach().clone()
        if entry:
            out[i] = entry
    return out


def compiled_state_view(
    groups: Sequence[dict[str, object]], entries: dict[int, dict[str, object]]
) -> StateMap:
    """Live read-only view over the compiled state buffers, keyed by parameter.

    Parameters
    ----------
    groups : Sequence[dict[str, object]]
        Live parameter groups from the wrapped optimizer.
    entries : dict[int, dict[str, object]]
        Per-parameter exported state dicts keyed by flat parameter index.
    """
    params = flat_params(groups)
    return ReadOnlyMap(
        {params[i]: ReadOnlyMap(entry, _READ_ONLY_STATE) for i, entry in entries.items()},
        _READ_ONLY_STATE,
        _empty_compiled_entry,
    )
