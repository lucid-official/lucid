"""
Tensor indexing: __getitem__ and __setitem__ with full reference-framework parity.

Supported forms
───────────────
Basic (no autograd break):
  t[int]                   scalar selection (removes dimension)
  t[slice]                 slice selection
  t[Ellipsis]              Ellipsis expansion
  t[None, ...]             None / newaxis → unsqueeze
  t[int, slice, ...]       multi-dimensional basic

Advanced (any Tensor in the index key):
  t[bool_mask_1d]          row-selection where mask is True   → (n_true, *t.shape[1:])
  t[bool_mask_full]        element-selection, mask == t.shape → (n_true,)
  t[int_tensor]            fancy row selection                → (*idx.shape, *t.shape[1:])
  t[int_t0, int_t1, ...]   coordinate selection               → broadcast(*idx_shapes)
  t[slice, int_tensor]     basic prefix + fancy suffix
  t[int_tensor, slice]     fancy prefix + basic suffix
  t[None, int_tensor, ...] None/newaxis anywhere

  t[True] / t[False]       a new leading axis of length 1 / 0     → (1|0, *t.shape)
  t[[0, 2]], t[ndarray]    a list or array is an index tensor

In-place assignment:
  t[key] = value           every form above; the value is broadcast to
                           what ``t[key]`` reads and written there

Two stages own every decision about what a key or a value means, and both
paths — reading and writing — go through them:

* :func:`_normalize_key` turns any key into one :class:`_Key`: ``...``
  expanded, bool scalars turned into a new axis plus an index, masks into
  their coordinates, lists and arrays into index tensors, every index tensor
  on the indexed tensor's device at its own integer width.  Bounds are left
  to the engine, which checks int64 at full width.
* :func:`_normalize_value` turns a value into a tensor of the destination's
  dtype, on its device, at the shape it is written at — a Python scalar
  keeping its kind and precision.

Every write lands through :func:`_rebind`, which keeps what the tensor
carries besides its values (``requires_grad``, ``retain_grad``, a gradient
slot it reads, the views it has).
"""

import operator
from bisect import bisect_left
from typing import Sequence, SupportsIndex, TYPE_CHECKING, cast

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._factories._build import _fill
from lucid._factories.converters import _to_impl

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import _IndexType, TensorOrScalar


# ── low-level helpers ──────────────────────────────────────────────────────────


def _prod(seq: Sequence[int]) -> int:
    """Integer product of a sequence."""
    result = 1
    for v in seq:
        result *= int(v)
    return result


def _select_int(impl: _C_engine.TensorImpl, dim: int, i: int) -> _C_engine.TensorImpl:
    """Select a single integer index along dim, removing that dimension."""
    length = impl.shape[dim]
    normalized = i + length if i < 0 else i
    if normalized < 0 or normalized >= length:
        raise IndexError(f"index {i} is out of bounds for dimension {dim} with size {length}")
    hi = normalized + 1
    if normalized == 0 and hi == length:
        sliced = impl
    elif normalized == 0:
        sliced = _C_engine.split_at(impl, [hi], dim)[0]
    elif hi == length:
        sliced = _C_engine.split_at(impl, [normalized], dim)[1]
    else:
        sliced = _C_engine.split_at(impl, [normalized, hi], dim)[1]
    return _C_engine.squeeze(sliced, dim)


def _narrow(
    impl: _C_engine.TensorImpl, dim: int, start: int, stop: int, length: int
) -> _C_engine.TensorImpl:
    """``impl[start:stop]`` along ``dim`` with a unit step — a view on the CPU."""
    if start == 0 and stop == length:
        return impl
    if start == 0:
        return _C_engine.split_at(impl, [stop], dim)[0]
    if stop == length:
        return _C_engine.split_at(impl, [start], dim)[1]
    return _C_engine.split_at(impl, [start, stop], dim)[1]


def _select_slice(impl: _C_engine.TensorImpl, dim: int, s: slice) -> _C_engine.TensorImpl:
    """Slice along dim using a Python slice object."""
    length = impl.shape[dim]
    start, stop, step = s.indices(length)

    # Empty result.  Must be handled before the step == 1 fast path: a
    # backwards range like ``x[5:2]`` reached ``split_at(impl, [5, 2])`` with
    # DESCENDING split points and segfaulted.  (``x[2:2]`` survived because the
    # points were equal.)  ``s.indices`` has already clamped both bounds into
    # [0, length], so this single test covers every empty case for both signs.
    # The result is a zero-width narrow of ``impl``, not a fresh ``zeros``: a
    # new constant has no grad_fn, so the slice cut the autograd edge and a
    # module that sliced its input at batch 0 lost its weights' gradients.
    # For a positive step, narrowing at ``[start:start]`` preserves the slice's
    # start offset in the underlying allocation (LCD-517).
    if step > 0 and stop <= start:
        return _narrow(impl, dim, start, start, length)
    if step < 0 and stop >= start:
        return _C_engine.contiguous(_narrow(impl, dim, 0, 0, length))

    if step == 1:
        return _narrow(impl, dim, start, stop, length)
    if step > 0:
        # A positive step is a view, as the reference's is: the run the slice
        # covers, one-element windows every ``step`` along ``dim``, and the
        # window axis squeezed away.  split_at, unfold_dim and squeeze each
        # make a view, and each already has an emitter on every path.  A
        # negative step copies below, as the reference does not have one.
        windows = _C_engine.unfold_dim(_narrow(impl, dim, start, stop, length), dim, 1, step)
        return _C_engine.squeeze(windows, len(windows.shape) - 1)
    else:
        # Element count of the slice.  The hand-rolled ceil-division dropped
        # the ``step`` term from the numerator, so it was wrong for **both**
        # signs once |step| > 1, and wrong for *every* negative step: the
        # canonical ``x[::-1]`` over-counted by one, asked ``arange`` for n+1
        # indices, and raised ShapeMismatch. (Positive steps mostly survived
        # because the error only bites when the range does not divide evenly.)
        # ``range`` computes the count exactly; do not re-derive it.
        n = len(range(start, stop, step))
        idx_1d = _C_engine.arange(start, stop, step, _C_engine.I32, impl.device)
        out_shape = list(impl.shape)
        out_shape[dim] = n
        bcast_shape = [1] * len(out_shape)
        bcast_shape[dim] = n
        idx_rs = _C_engine.reshape(idx_1d, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, out_shape)
        return _C_engine.gather(impl, idx_bc, dim)


# ── key normalisation ─────────────────────────────────────────────────────────

_BOOL = _C_engine.Dtype.Bool
_I64 = _C_engine.Dtype.I64
_CPU = _C_engine.Device.CPU

#: The dtypes an index tensor may hold: the ones the engine's gather and
#: scatter kernels read.  Every index door — a key here, the ``index``
#: argument of the composite index ops — refuses the rest by this set.
_INDEX_DTYPES = (_C_engine.Dtype.I32, _I64)


class _Key:
    """A key as every indexing path reads it.

    ``items`` holds only ``None``, ``int``, ``slice`` and integer index
    tensors (``TensorImpl``), one per dim of the source after ``unsqueeze``
    — ``...`` expanded, a mask split into one index per dim it covers.
    ``unsqueeze`` lists the axes a bool scalar inserts into the source, in
    the order they are inserted.  ``unchecked`` says an index tensor came
    from the caller — one a mask or a bool scalar made is in range by
    construction — so a value of it may be out of range.
    """

    __slots__ = ("items", "unsqueeze", "advanced", "unchecked")

    def __init__(self) -> None:
        self.items: list[object] = []
        self.unsqueeze: list[int] = []
        self.advanced = False
        self.unchecked = False


def _is_bool_scalar(token: object) -> bool:
    if isinstance(token, bool):
        return True
    return isinstance(token, _C_engine.TensorImpl) and token.dtype == _BOOL and not token.shape


def _token(part: object, op: str) -> object:
    """One part of a key as ``None``, ``...``, ``bool``, ``int``, ``slice``
    or a ``TensorImpl`` — an ``int32`` / ``int64`` index or a bool mask.

    A 0-d integer tensor of any width is the integer it holds, as Python
    reads it through ``__index__``; an index tensor with dims must be one
    the engine's kernels read, and is refused here, under ``op``'s name,
    rather than by the gather or scatter it would reach.
    """
    if isinstance(part, (type(None), type(Ellipsis), bool, int, slice)):
        return part
    impl = _promotion.try_unwrap(part)
    if impl is None:
        if isinstance(part, SupportsIndex) and getattr(part, "ndim", 0) == 0:
            try:
                return operator.index(part)  # a NumPy integer
            except TypeError:
                pass  # a NumPy bool, which has no integer value: converted below
        # A list or a NumPy array is an index tensor; an empty list indexes
        # nothing, as an empty int64 index.  A NumPy scalar goes in as the
        # array it stands for: converted as itself, it took the default
        # float dtype.
        to_array = getattr(part, "__array__", None)
        if to_array is not None:
            part = to_array()
        elif not isinstance(part, list):
            raise IndexError(
                "only integers, slices, ..., None, bools, integer or bool tensors "
                f"and lists or arrays of them are valid indices, got "
                f"{type(part).__name__}"
            )
        try:
            impl = _to_impl(part, dtype=_I64 if isinstance(part, list) and not part else None)
        except (TypeError, ValueError) as err:
            raise IndexError(
                "a list index holds only integers or only bools; to index "
                "several dims, use a tuple — x[i, j], not x[[i, j]]"
            ) from err
    return _index_part(impl, op)


def _index_part(impl: _C_engine.TensorImpl, op: str) -> object:
    """A tensor in a key as ``_token`` answers it: a mask, the integer a
    0-d integer tensor holds, or an index tensor of a width the kernels read."""
    if impl.dtype == _BOOL:
        return impl
    integer = _promotion.is_integer(impl.dtype)
    if (
        integer
        and not impl.shape
        and (op in ("__getitem__", "__setitem__") or impl.dtype in _INDEX_DTYPES)
    ):
        return int(impl.item())
    if not integer or impl.dtype not in _INDEX_DTYPES:
        raise IndexError(
            f"{op}: tensors used as indices must be int32, int64 or bool "
            f"tensors, got {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    return impl


def _consumes(token: object) -> int:
    """How many dims of the indexed tensor ``token`` addresses."""
    if token is None or token is Ellipsis or _is_bool_scalar(token):
        return 0
    if isinstance(token, _C_engine.TensorImpl) and token.dtype == _BOOL:
        return len(token.shape)
    return 1


def _on_device(index: _C_engine.TensorImpl, device: _C_engine.Device) -> _C_engine.TensorImpl:
    """``index`` on ``device``.

    Index tensors are built on the CPU far more often than not — a list
    turned into a tensor, a mask from ``nonzero`` — and the reference
    accepts them against a tensor on any device.  The other direction is
    refused, as the reference refuses it: a metal index cannot address a
    CPU tensor without a round trip the caller did not ask for.
    """
    if index.device == device:
        return index
    if device != _CPU:
        return _C_engine.to_device(index, device)
    raise RuntimeError(
        "indexing: the index is on metal but the indexed tensor is on the CPU "
        "— move the index to the CPU first"
    )


def _mask_indices(mask: _C_engine.TensorImpl) -> list[_C_engine.TensorImpl]:
    """The int64 coordinates of ``mask``'s True elements, one tensor per dim."""
    nz = _C_engine.nonzero(mask)  # (n_true, mask.ndim)
    k = nz.shape[1] if len(nz.shape) > 1 else 1
    if k == 1:
        return [_C_engine.reshape(nz, [-1])]
    return [
        _C_engine.contiguous(_C_engine.squeeze(_C_engine.split_at(nz, [d, d + 1], 1)[1], 1))
        for d in range(k)
    ]


def _bool_scalar_index(
    token: bool | _C_engine.TensorImpl, device: _C_engine.Device
) -> _C_engine.TensorImpl:
    """The index a bool scalar puts on the axis it inserts: ``[0]`` or ``[]``."""
    if isinstance(token, bool):
        return _C_engine.zeros([1 if token else 0], _I64, device)
    return _mask_indices(_C_engine.reshape(_on_device(token, device), [1]))[0]


def _normalize_key(key: object, shape: Sequence[int], device: _C_engine.Device, *, op: str) -> _Key:
    """Normalise ``key`` against a tensor of ``shape`` on ``device`` for ``op``.

    The single owner of what an index key means: ``Tensor.__getitem__``,
    ``Tensor.__setitem__`` and ``index_put``, which takes keys of the same
    form, all read their keys through here.  For reading and writing alike:

    * ``...`` stands for as many full slices as the other parts leave dims;
    * a bool scalar — ``True``, ``False`` or a 0-d bool tensor — inserts an
      axis of length 1 and indexes it with ``[0]`` or ``[]``, so ``x[True]``
      has shape ``(1, *x.shape)`` and ``x[False]`` ``(0, *x.shape)``;
    * a bool mask covers as many dims as it has, must match their sizes, and
      becomes the coordinates of its True elements, wherever it stands;
    * a list or a NumPy array is one index tensor — ``x[[[0, 1], [2, 3]]]``
      is a ``(2, 2)`` index, as NumPy and the reference's announced
      semantics read it, not the tuple ``x[[0, 1], [2, 3]]`` the reference
      still reads with a deprecation warning;
    * an integer index tensor moves to ``device`` and keeps its width, so
      the engine checks an int64 index at full width (``2**40`` is out of
      range, not 0); one with dims must be ``int32`` or ``int64``.

    ``op`` names the caller in a refusal — ``__getitem__``, ``__setitem__``
    or ``index_put`` — rather than the engine op the key would reach.

    Raises
    ------
    IndexError
        A part that is not a valid index — an index tensor with dims that
        is neither ``int32``, ``int64`` nor bool among them — a second
        ``...``, more indices than ``shape`` has dims, or a mask whose
        shape does not match.
    """
    parts = key if isinstance(key, tuple) else (key,)
    tokens = [_token(part, op) for part in parts]
    if sum(1 for token in tokens if token is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    ndim = len(shape)
    used = sum(_consumes(token) for token in tokens)
    if used > ndim:
        raise IndexError(f"too many indices for tensor of dimension {ndim} ({used} given)")
    out = _Key()
    sizes = list(shape)  # the source's sizes, axes inserted so far included
    dim = 0  # the source dim the next token addresses
    for token in tokens:
        if token is Ellipsis:
            out.items.extend([slice(None)] * (ndim - used))
            dim += ndim - used
        elif token is None:
            out.items.append(None)
        elif isinstance(token, (bool, _C_engine.TensorImpl)) and _is_bool_scalar(token):
            out.unsqueeze.append(dim)
            sizes.insert(dim, 1)
            out.items.append(_bool_scalar_index(token, device))
            dim += 1
        elif isinstance(token, _C_engine.TensorImpl):
            index = _on_device(token, device)
            if index.dtype != _BOOL:
                out.items.append(index)
                out.unchecked = True
                dim += 1
                continue
            covered = sizes[dim : dim + len(index.shape)]
            if list(index.shape) != covered:
                raise IndexError(
                    f"the shape of the mask {list(index.shape)} does not match the "
                    f"shape {covered} of the dims it indexes, starting at dim {dim}"
                )
            out.items.extend(_mask_indices(index))
            dim += len(index.shape)
        else:
            out.items.append(token)
            dim += 1
    out.advanced = any(isinstance(item, _C_engine.TensorImpl) for item in out.items)
    return out


def _inserted(impl: _C_engine.TensorImpl, axes: list[int]) -> _C_engine.TensorImpl:
    """``impl`` with a length-1 axis inserted at each of ``axes``, in order."""
    for axis in axes:
        impl = _C_engine.unsqueeze(impl, axis)
    return impl


# ── basic indexing (int / slice / None only) ──────────────────────────────────


def _apply_basic_index(impl: _C_engine.TensorImpl, idx_list: list[object]) -> _C_engine.TensorImpl:
    """Apply a list of basic (int/slice/None) indices to impl."""
    dim = 0
    for i in idx_list:
        if i is None:
            impl = _C_engine.unsqueeze(impl, dim)
            dim += 1
        elif isinstance(i, int):
            impl = _select_int(impl, dim, i)
        elif isinstance(i, slice):
            impl = _select_slice(impl, dim, i)
            dim += 1
        else:
            raise IndexError(f"Unsupported index type: {type(i).__name__}")
    return impl


# ── advanced indexing helpers ─────────────────────────────────────────────────


def _fancy_select(
    impl: _C_engine.TensorImpl, dim: int, idx_impl: _C_engine.TensorImpl
) -> _C_engine.TensorImpl:
    """
    Advanced selection along a single dim.
    idx_impl has any shape (m0, m1, ...).
    Result shape: (*impl.shape[:dim], *idx_impl.shape, *impl.shape[dim+1:])
    """
    idx_flat = _C_engine.reshape(idx_impl, [-1])  # (M,)
    dim_size = impl.shape[dim]
    rest = list(impl.shape[dim + 1 :])

    if dim == 0:
        # Fast path: index_select directly on dim 0.
        selected = _C_engine.index_select(impl, 0, idx_flat)  # (M, *rest)
        out_shape = list(idx_impl.shape) + rest
        return _C_engine.reshape(selected, out_shape)

    # General case: fold dims before `dim` into the first axis.
    outer = _prod(impl.shape[:dim])
    t2d = _C_engine.reshape(impl, [outer, dim_size] + rest)  # (outer, dim_size, *rest)
    selected = _C_engine.index_select(t2d, 1, idx_flat)  # (outer, M, *rest)
    out_shape = list(impl.shape[:dim]) + list(idx_impl.shape) + rest
    return _C_engine.reshape(selected, out_shape)


def _coordinate_select(
    impl: _C_engine.TensorImpl,
    int_indices: list[_C_engine.TensorImpl],
    unchecked: bool,
) -> _C_engine.TensorImpl:
    """
    Pure coordinate selection: result[*i] = impl[int_indices[0][*i], int_indices[1][*i], ...]
    Result has shape = broadcast(int_indices) + impl.shape[n_indexed:]

    The indexed dims are folded into one and read through a single flat
    index.  Each index is checked against its own dim first: folded as it
    is, ``x[[0], [-1]]`` on a ``(2, 4)`` read element 7 rather than 3, and
    ``x[[0], [5]]`` read element 5 rather than refusing.  A negative index
    wraps within its dim, and an out-of-range one is sent past the end of
    the folded axis, where the engine refuses it on the CPU and isolates it
    on Metal, as for any other index.  Indices a mask made (``unchecked``
    false) are in range and non-negative by construction, and skip both.
    """
    n_indexed = len(int_indices)
    shape = impl.shape
    rest = list(shape[n_indexed:])  # dims not indexed
    bcast_shape = _broadcast_shape([list(idx.shape) for idx in int_indices])
    indexed_total = _prod(shape[:n_indexed])
    dev = impl.device

    def scalar(v: int) -> _C_engine.TensorImpl:
        return _C_engine.full([], float(v), _I64, dev)

    zero = scalar(0)
    flat_idx: _C_engine.TensorImpl | None = None
    outside: _C_engine.TensorImpl | None = None
    stride = 1
    for k in reversed(range(n_indexed)):
        idx = int_indices[k]
        if idx.dtype != _I64:
            idx = _C_engine.astype(idx, _I64)
        if not unchecked:
            term = _C_engine.mul(idx, scalar(stride)) if stride != 1 else idx
            flat_idx = term if flat_idx is None else _C_engine.add(flat_idx, term)
            stride *= shape[k]
            continue
        size = scalar(shape[k])
        idx = _C_engine.where(_C_engine.less(idx, zero), _C_engine.add(idx, size), idx)
        bad = _C_engine.logical_or(_C_engine.less(idx, zero), _C_engine.greater_equal(idx, size))
        if dev == _CPU and not bad.is_phantom:
            # Only decides whether to raise — ``bad`` reaches the result
            # through ``outside`` below, not through this read — so a trace
            # does not count it as a data dependency.  Imported here:
            # ``lucid.compile`` imports this package.
            from lucid.compile import _host_check

            with _host_check():
                if _C_engine.any(bad).item():
                    _refuse_out_of_range(int_indices[k], shape[k])
        term = _C_engine.mul(idx, scalar(stride)) if stride != 1 else idx
        flat_idx = term if flat_idx is None else _C_engine.add(flat_idx, term)
        outside = bad if outside is None else _C_engine.logical_or(outside, bad)
        stride *= shape[k]
    assert flat_idx is not None
    if outside is not None:
        flat_idx = _C_engine.where(outside, scalar(indexed_total), flat_idx)
    flat_idx_1d = _C_engine.reshape(
        _C_engine.contiguous(_C_engine.broadcast_to(flat_idx, bcast_shape)), [-1]
    )

    # Flatten the indexed dims of impl: (D0*...*Dk-1, *rest)
    impl_flat = _C_engine.reshape(impl, [indexed_total] + rest)

    # index_select along dim=0: (M, *rest)
    selected = _C_engine.index_select(impl_flat, 0, flat_idx_1d)

    # Reshape to (*bcast_shape, *rest)
    return _C_engine.reshape(selected, bcast_shape + rest)


def _refuse_out_of_range(index: _C_engine.TensorImpl, size: int) -> None:
    """Raise for the first value of ``index`` outside ``[-size, size)``.

    The CPU says which index and which size; the folded index the engine
    would see names neither.
    """
    values = _C_engine.reshape(index, [-1]).tolist()
    assert isinstance(values, list)
    first = next(v for v in values if not -size <= v < size)
    raise IndexError(f"index {first} is out of bounds for an indexed dimension with size {size}")


# ── main advanced getitem ─────────────────────────────────────────────────────


def _advanced_getitem(
    impl: _C_engine.TensorImpl, idx_list: list[object], unchecked: bool
) -> _C_engine.TensorImpl:
    """Read ``impl`` through ``idx_list``, a normalised key's items
    (:class:`_Key`) with at least one index tensor among them; ``unchecked``
    is the key's flag of the same name."""
    # Phase 1: tag each item with its kind.
    expanded: list[tuple[str, object]] = []
    for item in idx_list:
        if isinstance(item, _C_engine.TensorImpl):
            expanded.append(("__tensor__", item))
        elif item is None:
            expanded.append(("__none__", None))
        elif isinstance(item, int):
            expanded.append(("__int__", item))
        else:
            expanded.append(("__slice__", item))

    # Phase 2: Find the span of tensor indices (first to last)
    tensor_positions = [k for k, (kind, _) in enumerate(expanded) if kind == "__tensor__"]
    first_t = tensor_positions[0]
    last_t = tensor_positions[-1]

    # Phase 3: Apply prefix (basic ops before first tensor)
    result = impl
    pre = expanded[:first_t]
    mid = expanded[first_t : last_t + 1]
    post = expanded[last_t + 1 :]

    # Apply prefix (None/int/slice), tracking the "next available dim" in result.
    # `adv_start_dim` = the dim in result where the tensor block begins.
    adv_start_dim = 0
    for kind, val in pre:
        if kind == "__none__":
            result = _C_engine.unsqueeze(result, adv_start_dim)
            adv_start_dim += 1
        elif kind == "__int__":
            # Int removes a dim: adv_start_dim stays the same
            result = _select_int(result, adv_start_dim, cast(int, val))
        elif kind == "__slice__":
            result = _select_slice(result, adv_start_dim, cast(slice, val))
            adv_start_dim += 1

    # Phase 4: process mid block.
    # We need to know, for each dim in `result` starting at adv_start_dim, whether
    # it is tensor-indexed or basic (slice/int/None).
    #
    # Build a list of (result_local_dim, kind, val) for each element in mid,
    # where result_local_dim is relative to adv_start_dim.

    # First pass: assign local dims, tracking int-removal.
    mid_entries: list[tuple[int, str, object]] = []  # (local_dim, kind, val)
    local_dim = 0
    tensor_impls: list[_C_engine.TensorImpl] = []
    for kind, val in mid:
        if kind == "__tensor__":
            mid_entries.append((local_dim, "__tensor__", val))
            tensor_impls.append(cast(_C_engine.TensorImpl, val))
            local_dim += 1
        elif kind == "__slice__":
            mid_entries.append((local_dim, "__slice__", val))
            local_dim += 1
        elif kind == "__int__":
            mid_entries.append((local_dim, "__int__", val))
            # int removes this dim; subsequent dims still increment local_dim
            # BUT: int selection happens in post-processing, not here.
            # For simplicity, int between tensor dims: apply now, don't track.
            result = _select_int(result, adv_start_dim + local_dim, cast(int, val))
            # After removing dim, subsequent local_dims shift down — but we've
            # already recorded the tensor positions above. This interleaved-int
            # case is rare; skip adjusting for now.
        elif kind == "__none__":
            mid_entries.append((local_dim, "__none__", None))
            local_dim += 1

    # Tensor local dims and basic local dims
    tensor_local_dims = [d for d, k, _ in mid_entries if k == "__tensor__"]
    basic_local_dims = [(d, k, v) for d, k, v in mid_entries if k != "__tensor__"]

    n_tensors = len(tensor_impls)
    bc_shape = (
        _broadcast_shape([list(t.shape) for t in tensor_impls])
        if n_tensors > 1
        else list(tensor_impls[0].shape)
    )
    adv_out_ndim = len(bc_shape)

    # Check if tensor dims form a contiguous block starting at tensor_local_dims[0].
    contiguous = tensor_local_dims == list(
        range(tensor_local_dims[0], tensor_local_dims[0] + n_tensors)
    )

    # Set by the branch that cannot express its post-block position as
    # ``anchor + adv_out_ndim + mid_basic`` (see phase 5).
    post_start: int | None = None

    if n_tensors == 1:
        # Single tensor: direct fancy select at adv_start_dim + tensor_local_dims[0]
        # Apply basic ops that come before the tensor dim
        for ld, kind, val in basic_local_dims:
            if ld < tensor_local_dims[0]:
                if kind == "__slice__":
                    result = _select_slice(result, adv_start_dim + ld, cast(slice, val))
                elif kind == "__none__":
                    result = _C_engine.unsqueeze(result, adv_start_dim + ld)
        result = _fancy_select(result, adv_start_dim + tensor_local_dims[0], tensor_impls[0])
        # Apply basic ops after the tensor dim
        post_base = adv_start_dim + tensor_local_dims[0] + adv_out_ndim
        offset = 0
        for ld, kind, val in basic_local_dims:
            if ld > tensor_local_dims[0]:
                effective_dim = post_base + (ld - tensor_local_dims[0] - 1) + offset
                if kind == "__slice__":
                    result = _select_slice(result, effective_dim, cast(slice, val))
                    offset += 1
                elif kind == "__none__":
                    result = _C_engine.unsqueeze(result, effective_dim)
                    offset += 1
        # Adjust cur_dim for phase 5
        adv_result_anchor = adv_start_dim + tensor_local_dims[0]

    elif contiguous:
        # All tensor dims are consecutive: standard coordinate select.
        t_start = adv_start_dim + tensor_local_dims[0]
        # Apply any basic ops in the mid block (non-tensor)
        for ld, kind, val in basic_local_dims:
            effective = adv_start_dim + ld
            if kind == "__slice__":
                result = _select_slice(result, effective, cast(slice, val))
            elif kind == "__none__":
                result = _C_engine.unsqueeze(result, effective)
        # Move tensor dims to front if needed, apply, move back
        if t_start > 0:
            pre_d = list(range(t_start))
            coord_d = list(range(t_start, t_start + n_tensors))
            post_d = list(range(t_start + n_tensors, len(result.shape)))
            perm = coord_d + pre_d + post_d
            result = _C_engine.permute(result, perm)
        coord_result = _coordinate_select(result, tensor_impls, unchecked)
        if t_start > 0:
            n_bc = len(bc_shape)
            n_pre = t_start
            n_post = len(coord_result.shape) - n_bc - n_pre
            perm_back = (
                list(range(n_bc, n_bc + n_pre))
                + list(range(n_bc))
                + list(range(n_bc + n_pre, n_bc + n_pre + n_post))
            )
            coord_result = _C_engine.permute(coord_result, perm_back)
        result = coord_result
        adv_result_anchor = t_start  # where adv result dims sit

    else:
        # Non-contiguous advanced indexing: advanced dims are NOT consecutive.
        # The reference framework places the advanced result dims at the FRONT
        # of the output, followed by *every* surviving dim in its original
        # order — the ones a slice touched and the ones no index mentioned
        # alike.  Splitting those two groups reorders the output silently:
        # ``a[:, i, :, j]`` on a ``(2, 3, 4, 5)`` gives ``(bc, 2, 4)``, and
        # putting the sliced dim first yields ``(bc, 4, 2)`` — same element
        # count, wrong axes, and identical shapes whenever the two happen to
        # match.  So the permutation keeps them in one ascending run.
        #
        # ``local_dim`` in ``mid_entries`` counts a ``None`` as occupying a
        # slot, which it does in the *index expression* but not in ``result``;
        # re-derive the real dim each entry sits on from a separate counter.
        real = adv_start_dim
        targets: list[int] = []
        for _, kind, _ in mid_entries:
            targets.append(real)
            if kind in ("__tensor__", "__slice__"):
                real += 1

        t_dims_abs = [
            t for t, (_, k, _) in zip(targets, mid_entries, strict=True) if k == "__tensor__"
        ]
        indexed = set(t_dims_abs)
        keep = [d for d in range(len(result.shape)) if d not in indexed]
        result = _C_engine.permute(result, t_dims_abs + keep)

        # Coordinate select on first n_tensors dims → (*bc_shape, *keep)
        result = _coordinate_select(result, tensor_impls, unchecked)

        # Apply the mid block's basic ops where their dim landed.  ``keep`` is
        # ascending, so a dim's output position is its rank in ``keep`` — and
        # ``bisect_left`` gives the same answer for a ``None``, whose slot is
        # wherever the next real dim would have gone.
        n_bc = len(bc_shape)
        inserted = 0
        for (_, kind, val), target in zip(mid_entries, targets, strict=True):
            if kind == "__slice__":
                pos = n_bc + bisect_left(keep, target) + inserted
                result = _select_slice(result, pos, cast(slice, val))
            elif kind == "__none__":
                pos = n_bc + bisect_left(keep, target) + inserted
                result = _C_engine.unsqueeze(result, pos)
                inserted += 1

        # For non-contiguous, adv result goes to front (position 0)
        adv_result_anchor = 0
        post_start = n_bc + bisect_left(keep, real) + inserted

    # Phase 5: apply post block (after last tensor in original idx).
    # Result dims: [*pre_dims, *adv_dims, *mid_basic, *post]
    # post ops start after all mid dims
    n_mid_basic_kept = sum(1 for _, k, _ in basic_local_dims if k in ("__slice__", "__none__"))
    cur_dim = (
        post_start
        if post_start is not None
        else adv_result_anchor + adv_out_ndim + n_mid_basic_kept
    )
    for kind, val in post:
        if kind == "__none__":
            result = _C_engine.unsqueeze(result, cur_dim)
            cur_dim += 1
        elif kind == "__int__":
            result = _select_int(result, cur_dim, cast(int, val))
        elif kind == "__slice__":
            result = _select_slice(result, cur_dim, cast(slice, val))
            cur_dim += 1

    return result


def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]:
    r"""Return the common shape to which ``shapes`` broadcast.

    Applies the standard right-aligned broadcasting rules across zero or
    more shape sequences: two dimensions are compatible when they are
    equal or one of them is ``1``, in which case the output dimension is
    their non-unit size (or ``0`` when an axis has length ``0`` paired
    with ``0`` or ``1``).

    Parameters
    ----------
    *shapes : Sequence[int]
        Zero or more shape tuples or lists of non-negative dimension sizes.

    Returns
    -------
    tuple[int, ...]
        The broadcasted shape tuple (``()`` when called with no arguments).

    Raises
    ------
    ValueError
        If any dimension size is negative.
    lucid._C.engine.ShapeMismatch
        If any two shapes have incompatible non-unit dimensions along the
        same right-aligned axis.

    Examples
    --------
    >>> import lucid
    >>> lucid.broadcast_shapes((2, 1, 3), (4, 1))
    (2, 4, 3)
    >>> lucid.broadcast_shapes()
    ()
    """
    norm: list[tuple[int, ...]] = [
        (
            (operator.index(s),)
            if isinstance(s, SupportsIndex)
            else tuple(operator.index(d) for d in s)
        )
        for s in shapes
    ]
    for s in norm:
        if any(d < 0 for d in s):
            raise ValueError(f"Attempting to broadcast a dimension with negative length in {s}!")
    rank = max((len(s) for s in norm), default=0)
    out: list[int] = []
    for axis in range(-rank, 0):
        sizes = {s[axis] for s in norm if len(s) >= -axis} - {1}
        if len(sizes) > 1:
            raise _C_engine.ShapeMismatch(
                "shape mismatch: shapes could not be broadcast together "
                f"with shapes {', '.join(str(list(s)) for s in norm)}"
            )
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


def broadcast_tensors(*tensors: Tensor) -> tuple[Tensor, ...]:
    r"""Broadcast one or more tensors against each other to a common shape.

    Computes the joint shape via :func:`lucid.broadcast_shapes` and returns
    a tuple of views produced by :meth:`Tensor.broadcast_to`, preserving
    autograd tracking on every input tensor.

    Parameters
    ----------
    *tensors : Tensor
        Zero or more tensors whose shapes are broadcast-compatible.

    Returns
    -------
    tuple[Tensor, ...]
        Tuple of broadcasted tensor views sharing the common shape.

    Raises
    ------
    TypeError
        If any argument is not a :class:`Tensor`.
    lucid._C.engine.ShapeMismatch
        If the input tensor shapes cannot be broadcast together.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(3).reshape(3, 1)
    >>> y = lucid.arange(2).reshape(1, 2)
    >>> bx, by = lucid.broadcast_tensors(x, y)
    >>> bx.shape, by.shape
    ((3, 2), (3, 2))
    """
    for i, t in enumerate(tensors):
        if _promotion.try_unwrap(t) is None or not hasattr(t, "broadcast_to"):
            raise TypeError(
                f"broadcast_tensors(): expected Tensor at position {i}, " f"got {type(t).__name__}"
            )
    target_shape = broadcast_shapes(*(t.shape for t in tensors))
    return tuple(t.broadcast_to(target_shape) for t in tensors)


def _broadcast_shape(shapes: list[list[int]]) -> list[int]:
    """The shape index tensors of ``shapes`` broadcast to together.

    Raises
    ------
    IndexError
        If two of them cannot be broadcast together.
    """
    try:
        return list(broadcast_shapes(*shapes))
    except _C_engine.ShapeMismatch:
        raise IndexError(
            "shape mismatch: indexing tensors could not be broadcast together "
            f"with shapes {', '.join(str(list(s)) for s in shapes)}"
        ) from None


# ── value normalisation ───────────────────────────────────────────────────────


def _normalize_value(
    value: object, dst: _C_engine.TensorImpl, shape: list[int]
) -> _C_engine.TensorImpl:
    """``value`` as what is written into ``dst`` at ``shape``.

    In ``dst``'s dtype, on ``dst``'s device — moved there, as the reference
    moves it, and differentiably — and broadcast to ``shape``; size-1 dims
    in front of the value's shape are dropped as the reference drops them.
    A Python number keeps its kind and precision (:func:`_fill`);
    anything else that is not a tensor is converted as ``lucid.tensor``
    would convert it.
    """
    impl = _promotion.try_unwrap(value)
    if impl is None:
        value = _promotion.as_number(value)
        if isinstance(value, _promotion.NUMBER):
            return _fill(shape, value, dst.dtype, dst.device)
        impl = _to_impl(value, device=dst.device)
    if impl.device != dst.device:
        impl = _C_engine.to_device(impl, dst.device)
    if impl.dtype != dst.dtype:
        impl = _C_engine.astype(impl, dst.dtype)
    value_shape = list(impl.shape)
    if value_shape == shape:
        return impl
    lead = 0
    while len(value_shape) - lead > len(shape) and value_shape[lead] == 1:
        lead += 1
    value_shape = value_shape[lead:]
    pad = len(shape) - len(value_shape)
    impl = _C_engine.reshape(impl, [1] * pad + value_shape)
    return _C_engine.broadcast_to(impl, shape)


# ── landing a write ───────────────────────────────────────────────────────────


def _take(t: Tensor, impl: _C_engine.TensorImpl) -> None:
    """Make ``impl`` ``t``'s tensor, keeping what ``t`` asked of autograd.

    The single owner of rebinding a tensor's impl, so a flag that lives on
    the impl is carried in one place.  Assignment, an in-place op whose
    dtype promotion produced a new impl (:func:`_adopt_inplace`) and the
    in-place index ops in ``lucid._ops.composite.indexing`` (``index_fill_``,
    ``index_put_`` and the rest) all go through it.

    ``retain_grad`` is registered on the slot of the tensor's producer, so
    a new impl does not have it: ``y.retain_grad(); y[0] = v`` left
    ``y.grad`` None (LCD-296).  The reference moves it to the new place in
    the graph, as the engine's in-place ops do.
    """
    old = t._impl
    retains = old.retains_grad
    if isinstance(old, _C_engine.TensorImpl):
        # Bumps ``old``'s version and carries the bumped version (plus any
        # saved-output baseline on ``impl``'s own grad_fn) onto ``impl`` so
        # ``t._version`` advances on non-leaf in-place rebinds as well.
        impl.rebind_version_from(old)
    t._impl = impl
    if retains:
        impl.retain_grad_()


def _rebind(t: Tensor, impl: _C_engine.TensorImpl) -> None:
    """Make ``impl`` — same shape, dtype and device — ``t``'s values.

    Every write lands here, and it keeps what lives on ``t`` besides its
    values:

    * a write that records no graph goes into ``t``'s own buffer, so ``t``
      stays the tensor it was — a Parameter keeps its flag and its place in
      the graph that already used it (``weight[pad] = 0`` under
      ``no_grad``), a tensor read from ``.grad`` writes the gradient
      (``p.grad[0] = v``), a Metal shared buffer and a tensor a compile
      trace reads see the values, as does the array a NumPy-backed tensor
      shares;
    * a tensor with live views takes the values into its buffer, where the
      views read them, by the engine's in-place rules, graph included;
    * otherwise ``t`` takes ``impl`` and with it ``impl``'s place in the
      graph, keeping ``retain_grad`` (:func:`_take`).

    ``impl`` may be a view (a broadcast value) for the first two, which
    read it through its strides; it must be a buffer of its own for the
    third, or ``t`` would share the value's storage.  Callers that write a
    tensor requiring grad under autograd must hand over a result that
    records the write (the scatter): copied in, ``t`` would keep the graph
    of the values the write replaced, and send them a gradient.
    """
    old = t._impl
    records = impl.requires_grad and _C_engine.grad_enabled()
    if old.is_metal_shared and not old.requires_grad and not records:
        old.copy_from(impl)
        return
    if old.is_aliased() or not old.is_contiguous() or old.storage_offset_bytes() != 0:
        _C_engine.assign_inplace(old, impl, "__setitem__")
        return
    if not records:
        old.copy_from(impl)
        return
    _take(t, impl)


def _read_for_write(t: Tensor, value: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """``t``'s values, for a write into ``t`` to be computed from.

    A write that records a graph bumps ``t``'s version counter (either in
    place or in :func:`_take`), so it must not read ``t`` through an engine
    op: the op records ``t``'s version, the write bumps it, and backward
    refused the very write it was computing —
    ``out[0] = a; out[1] = b; out.sum().backward()``.  It reads a snapshot
    instead, which stands in for ``t`` in the graph and records nothing.
    """
    impl = t._impl
    records = _C_engine.grad_enabled() and (impl.requires_grad or value.requires_grad)
    if not records:
        return impl
    from lucid._ops.composite.indexing import _Snapshot
    from lucid._tensor.tensor import Tensor as _Tensor

    snapshot = _Snapshot.apply(t)
    assert isinstance(snapshot, _Tensor)
    return snapshot._impl


def _adopt_inplace(t: Tensor, impl: _C_engine.TensorImpl, name: str) -> Tensor:
    """Write ``impl``, an in-place op's result, into ``t`` at ``t``'s dtype; return ``t``.

    An engine in-place op hands back ``t``'s own impl, written in place.  A
    different one means same-kind dtype promotion ran the op on a wider cast
    copy (e.g. ``float16 += float32``); the result is cast back to ``t``'s
    dtype and written into ``t`` in place so ``t`` and its views keep their
    dtype and storage.
    """
    if impl is not t._impl:
        if impl.dtype != t._impl.dtype:
            impl = _C_engine.astype(impl, t._impl.dtype)
        if (
            t._impl.is_aliased()
            or not t._impl.is_contiguous()
            or t._impl.storage_offset_bytes() != 0
        ):
            _C_engine.assign_inplace(t._impl, impl, name)
        elif (
            impl.requires_grad
            and not (t._impl.requires_grad and t._impl.is_leaf)
            and _C_engine.grad_enabled()
        ):
            _take(t, impl)
        else:
            t._impl.assign_from(impl, name)
    return t


# ── public entry points ───────────────────────────────────────────────────────


def _getitem(t: Tensor, idx: _IndexType) -> Tensor:
    """Top-level dispatcher for ``Tensor.__getitem__``.

    The key is normalised once (:func:`_normalize_key`), then read by the
    basic path (ints, slices, ``None``) or the advanced one (index tensors
    among them).

    Parameters
    ----------
    t : Tensor
        The tensor being indexed.
    idx : _IndexType
        Index spec — either a single index element or a tuple of them.

    Returns
    -------
    Tensor
        The selected sub-tensor (a view where possible, a copy otherwise).
    """
    impl = t._impl
    key = _normalize_key(idx, impl.shape, impl.device, op="__getitem__")
    source = _inserted(impl, key.unsqueeze)
    if key.advanced:
        out = _advanced_getitem(source, key.items, key.unchecked)
    else:
        out = _apply_basic_index(source, key.items)
    if out is impl:
        # An index that selects everything (``x[:]``, ``x[...]``, ``x[()]``,
        # ``x[0:n]``) came back as ``t``'s own TensorImpl, so the result
        # was ``t`` under another name: ``x[:].requires_grad_(True)`` turned
        # on ``x``'s flag and ``x[:].grad`` was ``x.grad``.  The reference's
        # result is a view — a tensor of its own over the same storage, in
        # ``t``'s graph when ``t`` requires grad — and so is this one.
        out = _C_engine.view(impl, list(impl.shape))
    return _wrap(out)


def _written_positions(
    shape: list[int], key: _Key, device: _C_engine.Device
) -> tuple[_C_engine.TensorImpl, list[int]]:
    """The flat positions ``t[key]`` reads, and the shape it reads them in.

    Read through the reading path itself, from a map of positions, so a
    write can never name other elements than the same key reads: deriving
    them a second time wrote the whole ``rows x cols`` rectangle for
    ``t[rows, cols] = v``, and every touched row for a full-shape mask.

    On Metal, a key with an index tensor from the caller reads a map that
    counts from 1.  An integer gather there reads an out-of-range index as
    0, and counted from 0 that is the first element: ``x[[1, 99]] = v``
    wrote ``x[0]`` (LCD-209).  Counted from 1, the 0 is told apart and sent
    past the end, where the scatter drops it, as every Metal scatter drops
    an out-of-range write.  The CPU refuses such an index in the gather.
    """
    total = _prod(shape)
    dtype = _C_engine.Dtype.I32 if total < 2**31 - 1 else _I64
    shifted = key.unchecked and device != _CPU
    first = 1 if shifted else 0
    positions = _C_engine.reshape(_C_engine.arange(first, total + first, 1, dtype, device), shape)
    positions = _inserted(positions, key.unsqueeze)
    if key.advanced:
        target = _advanced_getitem(positions, key.items, key.unchecked)
    else:
        target = _apply_basic_index(positions, key.items)
    flat = _C_engine.reshape(_C_engine.contiguous(target), [-1])
    if shifted:
        one = _C_engine.full([], 1.0, dtype, device)
        flat = _C_engine.where(
            _C_engine.equal(flat, _C_engine.full([], 0.0, dtype, device)),
            _C_engine.full([], float(total), dtype, device),
            _C_engine.sub(flat, one),
        )
    return flat, list(target.shape)


def _refuse_illegal_setitem_target(impl: _C_engine.TensorImpl) -> None:
    """Refuse in-place assignment to a grad leaf or custom Function view."""
    if not _C_engine.grad_enabled():
        return
    if impl.requires_grad and impl.is_leaf:
        # Both paths in _setitem rebind ``t`` to the written result, which
        # under autograd carries a grad_fn: a Parameter assigned this way
        # stopped being a leaf and never received ``.grad`` again, so the
        # optimiser silently left it alone.  The reference refuses the same
        # write.
        raise RuntimeError(
            "__setitem__: a leaf tensor that requires grad cannot be assigned "
            "in place — wrap the assignment in lucid.no_grad(), or build a new "
            "tensor"
        )
    if impl.is_custom_function_output:
        raise RuntimeError(
            "__setitem__: Output of a custom Function is a view and is being "
            "modified in place — custom Function output views cannot be "
            "modified in place; clone() first"
        )


def _setitem(t: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
    """
    In-place assignment using Lucid engine ops only — no numpy.

    The key and the value are normalised once (:func:`_normalize_key`,
    :func:`_normalize_value`).  Whole-tensor assignment (``t[:] = v``) has
    its own path; everything else scatters the value into the positions
    the reading path names (:func:`_written_positions`).  Either way the
    result lands through :func:`_rebind`.
    """
    impl = t._impl
    _refuse_illegal_setitem_target(impl)
    shape = list(impl.shape)
    key = _normalize_key(idx, shape, impl.device, op="__setitem__")

    # ``t[:] = v`` and ``t[...] = v`` select every element, and scattering
    # through the flat index of the entire tensor costs random-access writes
    # over every element to express a straight copy: 51.7 ms against
    # 0.63 ms on a 64x3x128x128 tensor.  A value with a graph is made a
    # buffer of its own (``contiguous``) before ``t`` takes it, so a later
    # write to the value cannot rewrite ``t``; one without is copied in as
    # it is.  A tensor that requires grad taking a value without a graph
    # goes through the scatter, which records that its old values no longer
    # reach the result — copied in, ``t`` kept their graph, and
    # ``w * 2`` assigned ``5`` sent ``w`` a gradient of 2.
    whole = bool(key.items) and not key.unsqueeze
    whole = whole and all(isinstance(item, slice) and item == slice(None) for item in key.items)
    if whole:
        val = _normalize_value(value, impl, shape)
        records = _C_engine.grad_enabled()
        if records and val.requires_grad:
            _rebind(t, _C_engine.contiguous(val))
            return
        if not (records and impl.requires_grad):
            _rebind(t, val)
            return

    flat_idx, target_shape = _written_positions(shape, key, impl.device)
    flat_val = _C_engine.reshape(
        _C_engine.contiguous(_normalize_value(value, impl, target_shape)), [-1]
    )
    # The scatter writes a buffer of its own, so a dense ``t`` is read in
    # place; a copy first only cost another pass over every element.
    base = _read_for_write(t, flat_val)
    if not base.is_contiguous():
        base = _C_engine.contiguous(base)
    flat_out = _C_engine.scatter(_C_engine.reshape(base, [_prod(shape)]), 0, flat_idx, flat_val)
    _rebind(t, _C_engine.reshape(flat_out, shape))
