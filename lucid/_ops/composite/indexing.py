"""Index-based write and scatter operations.

All ops here follow the reference-framework API surface:

* ``index_fill``  — fill elements at 1-D index positions with a scalar.
* ``index_add``   — accumulate scaled source into input at 1-D index positions.
* ``index_copy``  — copy source into input at 1-D index positions.
* ``scatter_reduce`` — scatter-reduce src into input (sum / mean / prod / amax / amin).
* ``masked_scatter`` — copy source elements into positions where mask is True.

All implementations use only engine primitives — no numpy at the Python level.

Index tensors reach the engine at the width the caller gave them: the
engine checks an ``int64`` index at full width, refusing an out-of-range one
on the CPU and dropping its write on Metal.  Narrowed to ``int32`` first,
``2**40`` wrapped to 0 and wrote the first element.  A ``source`` / ``src``
/ ``values`` of another dtype than ``input``'s is refused, as the reference
refuses it.

An op that flattens or broadcasts its index before an engine kernel sees
it (the ``index_*`` family, ``put``) checks the index here, at the public
op, before any data is touched (:func:`_positions`, :func:`_require_index`).
Left to the kernels, a 2-D index for ``index_copy`` was flattened in
silence, or refused by the ``reshape`` that flattened it.  That refusal,
like the dtype and device refusals of ``scatter_set`` and ``scatter_add``,
named an op the caller never called.  ``scatter_reduce`` hands the caller's
index to the engine unchanged, and the engine's scatter door owns its rules.

The in-place forms (``Tensor.index_add_`` and the rest, and
:func:`index_put_`) run the out-of-place op and write its result into the
destination through :func:`_write_inplace`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, final, override

import lucid
from lucid import _promotion
from lucid._dispatch import _cannot_cast_inplace, _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry
from lucid._ops._dims import _dim_size, _wrap_dim
import lucid._C.engine as _C_engine
from lucid._tensor._indexing import (
    _INDEX_DTYPES,
    _broadcast_shape,
    _normalize_key,
    _normalize_value,
    _take,
    _written_positions,
)
from lucid.autograd.function import Function, FunctionCtx

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── helpers ────────────────────────────────────────────────────────────────


def _recorded_snapshot(x: Tensor) -> Tensor:
    """What a compile recording reads for a :class:`_Snapshot` of ``x``.

    A copy, as the node's own ``forward`` reads, but one that carries
    ``x``'s graph — its derivative is the identity — instead of a detached
    one, which a compiled backward would take for a gradient sink.  ``x``
    itself will not do: the op computing the write would save the very
    tensor the write bumps, and an eager backward run inside the recording
    (``autograd.grad`` of an energy, a gradient penalty) would refuse it,
    or ``assign_inplace`` would refuse the write to storage a saved tensor
    holds.  Each such read records one ``contiguous`` that a recording
    under ``no_grad`` does not.
    """
    return x.clone()


@final
class _Snapshot(Function):
    """``x``'s values in a buffer of their own, standing in for ``x`` in the graph.

    What an in-place index op reads its destination through while autograd
    records.  An engine op applied to the destination itself records its
    version, and the write that follows bumps it, so the op's own backward
    refused the write as an in-place modification of a tensor it saved.
    This node saves nothing, and its derivative is the identity.

    A compile recording reads a copy that carries ``x``'s graph instead
    (:attr:`Function._trace_form`, :func:`_recorded_snapshot`): one extra
    ``contiguous``, and no node noted as a custom backward.
    """

    _trace_form = staticmethod(_recorded_snapshot)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        return x.detach().clone()

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        return grad


def _require_input_dtype(name: str, input: Tensor, source: Tensor, what: str) -> None:
    """Refuse a ``source`` whose dtype is not ``input``'s.

    The write keeps ``input``'s dtype, and the reference refuses rather
    than cast the values written into it; so does the engine's own scatter
    door (``require_scatter_dtypes``), whose exception this raises, a
    ``RuntimeError`` and a ``TypeError`` both.  ``Tensor.__setitem__`` is
    the one write that casts, and it does not come through here.
    """
    if source._impl.dtype != input._impl.dtype:
        raise _C_engine.DtypeMismatch(
            f"{name}: {what} must have input's dtype {input.dtype}, got "
            f"{source.dtype} — cast it first"
        )


_KERNEL_PREFIXES: tuple[str, ...] = (
    "scatter_add: ",
    "scatter_set: ",
    "scatter_amax: ",
    "scatter_amin: ",
    "scatter_prod: ",
    "scatter: ",
)


def _call_engine[R](name: str, fn: Callable[[], R]) -> R:
    """Run ``fn()``, naming ``name`` in any engine refusal it raises."""
    try:
        return fn()
    except _C_engine.LucidError as exc:
        msg = str(exc)
        for prefix in _KERNEL_PREFIXES:
            if msg.startswith(prefix):
                msg = msg.removeprefix(prefix)
                break
        if not msg.startswith(f"{name}:"):
            msg = f"{name}: {msg}"
        raise type(exc)(msg) from None


def _scatter_add(base: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """``base`` with ``src`` added at ``index`` along ``dim``, by the engine.

    The engine op directly, not ``Tensor.scatter_add``, whose adapter
    narrows an ``int64`` index to ``int32`` on the way.
    """
    return _wrap(
        _C_engine.scatter_add(_unwrap(base), _unwrap(index), _unwrap(src), dim)
    )


def _require_device(name: str, input: Tensor, other: Tensor, what: str) -> None:
    """Refuse an operand that lives on another device than ``input``.

    The reference refuses it as a ``RuntimeError``, which ``DeviceMismatch``
    is.
    """
    if other._impl.device != input._impl.device:
        raise _C_engine.DeviceMismatch(
            f"{name}: {what} must be on input's device "
            f"{input.device.type}, got {other.device.type}"
        )


def _require_index(name: str, input: Tensor, index: Tensor) -> None:
    """Refuse an ``index`` that is not an ``int32`` / ``int64`` tensor on ``input``'s device.

    Asked by the composites that flatten or broadcast the index before an
    engine kernel sees it: the ``index_*`` family and ``put``.  The kernel
    does refuse a bad index, but under its own name (``scatter_set``,
    ``scatter_add``, ``scatter``), and only once the index is no longer the
    caller's.  An op that hands the caller's index to the engine unchanged,
    as ``scatter_reduce`` does, leaves these checks to the engine's door.

    The dtype refusal is the engine's ``IndexError``, an ``IndexError`` and
    a ``RuntimeError`` both.  The reference refuses the dtype with the
    first for ``index_fill`` and with the second for the rest, so either
    ``except`` catches this.
    """
    if not isinstance(index, lucid.Tensor):
        raise TypeError(f"{name}: index must be a Tensor, got {type(index).__name__}")
    _require_device(name, input, index, "index")
    if index._impl.dtype not in _INDEX_DTYPES:
        raise _C_engine.IndexError(
            f"{name}: index must be an int32 or int64 tensor, got {index.dtype}"
        )


def _positions(
    name: str, input: Tensor, dim: int, index: Tensor
) -> tuple[int, _C_engine.TensorImpl]:
    """``dim`` resolved against ``input``, and ``index`` as 1-D positions.

    The index argument of ``index_add`` / ``index_copy`` / ``index_fill``,
    in place or not: a 0-d or 1-D ``int32`` / ``int64`` tensor, a 0-d one
    naming a single position.  The reference refuses a 2-D index as an
    ``IndexError`` (``index_add``, ``index_copy``) or a ``RuntimeError``
    (``index_fill``); the engine's ``IndexError`` is both.  An index into
    an axis of size 0 is out of range whatever its values, so it is refused
    from the shapes alone, on Metal as well as on the CPU.

    Raises
    ------
    IndexError
        ``dim`` is out of range for ``input``.
    TypeError
        ``index`` is not a Tensor.
    lucid._C.engine.DeviceMismatch
        ``index`` is on another device than ``input``.
    lucid._C.engine.IndexError
        ``index`` has more than one dim, a dtype other than ``int32`` /
        ``int64``, or elements while ``input``'s ``dim`` has size 0.
    """
    dim = _wrap_dim(dim, input.ndim, name)
    _require_index(name, input, index)
    if index.ndim > 1:
        raise _C_engine.IndexError(
            f"{name}: index must be 0-D or 1-D, got shape {tuple(index.shape)}"
        )
    count = index.numel()
    if count and _dim_size(list(input.shape), dim) == 0:
        raise _C_engine.IndexError(
            f"{name}: index out of range: dim {dim} of input has size 0, got "
            f"{count} indices"
        )
    return dim, _C_engine.reshape(_unwrap(index), [-1])


def _require_source_slices(
    name: str, input: Tensor, dim: int, count: int, source: Tensor, *, same_rank: bool
) -> None:
    """Refuse a ``source`` that is not ``count`` slices of ``input`` along ``dim``.

    ``source`` must have ``input``'s shape with ``dim`` sized ``count``, the
    number of indices.  A 0-d ``input`` or ``source`` reads as a one-element
    vector, as the reference's ``index_copy`` reads it.  ``same_rank`` adds
    the reference's ``index_add`` rule: there a 0-d ``source`` goes only with
    a 0-d ``input``.

    Raises
    ------
    lucid._C.engine.DeviceMismatch
        ``source`` is on another device than ``input``.
    lucid._C.engine.IndexError
        ``source`` and ``input`` differ in their number of dims, or
        ``source.shape[dim]`` is not ``count``.  The reference raises an
        ``IndexError`` here for ``index_copy`` and a ``RuntimeError`` for
        ``index_add``.
    lucid._C.engine.ShapeMismatch
        ``source`` and ``input`` differ on a dim other than ``dim``.
    """
    _require_device(name, input, source, "source")
    want = list(input.shape) or [1]
    got = list(source.shape) or [1]
    if len(got) != len(want) or (same_rank and source.ndim != input.ndim):
        raise _C_engine.IndexError(
            f"{name}: source must have as many dims as input "
            f"{tuple(input.shape)}, got source of shape {tuple(source.shape)}"
        )
    if got[dim] != count:
        raise _C_engine.IndexError(
            f"{name}: {count} indices need source.shape[{dim}] == {count}, got "
            f"source of shape {tuple(source.shape)}"
        )
    if got[:dim] + got[dim + 1 :] != want[:dim] + want[dim + 1 :]:
        raise _C_engine.ShapeMismatch(
            f"{name}: source must match input's shape {tuple(input.shape)} on "
            f"every dim but {dim}, got {tuple(source.shape)}"
        )


def _as_vector(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """``impl``, or the one-element vector a 0-d ``impl`` is to a slice write."""
    return _C_engine.reshape(impl, [1]) if not list(impl.shape) else impl


def _write_slices(
    name: str,
    input: Tensor,
    dim: int,
    positions: _C_engine.TensorImpl,
    source: Tensor,
    write: Callable[
        [_C_engine.TensorImpl, _C_engine.TensorImpl, _C_engine.TensorImpl],
        _C_engine.TensorImpl,
    ],
) -> Tensor:
    """``source``'s slices along ``dim`` written into ``input``'s at ``positions``.

    ``write(base, index, src)`` is the engine scatter, given the 1-D
    positions broadcast to ``src``'s shape.  A 0-d ``input`` or ``source``
    is scattered as a one-element vector, and the result has ``input``'s
    shape.  Their shapes were checked by :func:`_require_source_slices`.
    """
    src = _as_vector(_unwrap(source))
    out = _call_engine(
        name,
        lambda: write(
            _as_vector(_unwrap(input)), _along(positions, dim, list(src.shape)), src
        ),
    )
    return _wrap(_C_engine.reshape(out, []) if input.ndim == 0 else out)


def _along(
    positions: _C_engine.TensorImpl, dim: int, shape: list[int]
) -> _C_engine.TensorImpl:
    """1-D ``positions`` along ``dim``, broadcast to ``shape``.

    The index every slice-wise write (``index_add``, ``index_copy``)
    scatters with: ``shape[dim]`` must be the number of positions.
    """
    along = [1] * len(shape)
    along[dim] = shape[dim]
    return _C_engine.broadcast_to(_C_engine.reshape(positions, along), shape)


def _in_place_result(
    input: Tensor, name: str, result_of: Callable[[Tensor], Tensor], base: Tensor
) -> _C_engine.TensorImpl:
    """``result_of(base)`` as an in-place op computes it, at ``input``'s dtype.

    The rules every engine in-place op follows (``inplace::run`` in
    ``lucid/_C/ops/utils/InplaceGraph.h``):

    * autocast does not apply to an in-place op, as the reference's does not,
      so the op computes in ``input``'s own dtype.  Run under it,
      ``F.silu(x, inplace=True)`` of a float32 ``x`` computed in float16 and
      the write refused the result — or, while autograd recorded, made ``x``
      float16;
    * a result of another dtype is cast to ``input``'s when the in-place rule
      allows it (:func:`lucid._promotion.can_cast`) and refused otherwise.
    """
    guard = _C_engine.AutocastGuard(_C_engine.Dtype.F32, input._impl.device, False)
    try:
        result = result_of(base)
    finally:
        del guard  # the last reference — ~AutocastGuard restores the prior state
    if result.shape != input.shape:
        raise _C_engine.ShapeMismatch(
            f"{name}: result shape {tuple(result.shape)} does not match "
            f"destination shape {tuple(input.shape)}"
        )
    impl = result._impl
    target = input._impl.dtype
    if impl.dtype == target:
        return impl
    if not _promotion.can_cast(impl.dtype, target):
        raise _cannot_cast_inplace(impl.dtype, target)
    return _C_engine.astype(impl, target)


def _write_inplace[T: Tensor](
    input: T, name: str, result_of: Callable[[Tensor], Tensor], *operands: Tensor
) -> T:
    """Run an out-of-place index op and make its result ``input``'s values.

    The op runs outside autocast and its result keeps ``input``'s dtype
    (:func:`_in_place_result`).  A leaf that requires grad is refused while
    autograd records, as every in-place op refuses it; under
    :func:`lucid.no_grad` it is written and stays a trainable leaf.
    Otherwise the result lands one of three ways, the ones
    ``Tensor.__setitem__`` takes:

    * ``input`` shares its buffer with a live view: the values go into the
      buffer, where the views read them, and the engine moves ``input``
      and its views to the result's place in the graph;
    * autograd records: ``input`` takes the result's tensor, and with it
      the result's place in the graph, keeping ``retain_grad``
      (``lucid._tensor._indexing._take``).  Copied into ``input``'s buffer
      instead, a tensor that only *received* a gradient-carrying ``source``
      kept its leaf flag, and every later in-place op refused it;
    * otherwise the values are copied into ``input``'s buffer.

    In the first case, while autograd records, the op reads ``input``
    through a :class:`_Snapshot`.  Its nodes save what they read — ``where``
    does — and the write bumps the buffer's version, so read directly,
    ``input`` made backward refuse (``VersionMismatch``) the very op that
    wrote it.

    Parameters
    ----------
    input : Tensor
        The destination, written in place and returned.
    name : str
        The in-place op's name, for errors.
    result_of : callable
        The out-of-place op, given the tensor to read ``input`` from.
    *operands : Tensor
        The op's other differentiable inputs (``source``, ``values``).

    Returns
    -------
    Tensor
        ``input``, now holding the result.

    Raises
    ------
    RuntimeError
        If ``input`` is a leaf that requires grad and autograd is recording,
        or the result's dtype cannot be cast to ``input``'s.
    """
    records = _C_engine.grad_enabled() and (
        input.requires_grad or any(o.requires_grad for o in operands)
    )
    if records and input.requires_grad and input.is_leaf:
        raise RuntimeError(
            f"{name}: a leaf tensor that requires grad cannot be modified in place — "
            "wrap the call in lucid.no_grad(), or use the out-of-place form"
        )
    aliased = (
        input._impl.is_aliased()
        or not input.is_contiguous()
        or input._impl.storage_offset_bytes() != 0
    )
    base: Tensor = input
    if records:
        snapshot = _Snapshot.apply(input)
        assert isinstance(snapshot, lucid.Tensor)
        base = snapshot
    result = _in_place_result(input, name, result_of, base)
    if aliased:
        _C_engine.assign_inplace(input._impl, result, name)
    elif records:
        _take(input, result)
    else:
        input._impl.assign_from(result, name)
    return input


# ── public API ─────────────────────────────────────────────────────────────


def index_fill(
    input: Tensor,
    dim: int,
    index: Tensor,
    value: float,
) -> Tensor:
    """Return a copy of ``input`` with positions ``index`` along ``dim`` set to ``value``.

    Autograd flows through the *unmasked* positions; filled positions
    receive zero gradient (they're overwritten by a constant).

    Parameters
    ----------
    input : Tensor
        Source tensor; not mutated.
    dim : int
        Axis along which ``index`` addresses slices.
    index : Tensor
        0-D or 1-D ``int32`` / ``int64`` tensor of positions along ``dim``.
    value : float
        Scalar to write into every indexed position.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; indexed slices replaced by
        ``value``, others unchanged.

    Raises
    ------
    IndexError
        ``dim`` is out of range, or, on the CPU, an ``index`` value lies
        outside ``[-size, size)`` of ``dim``.  Metal drops such a write.
    lucid._C.engine.IndexError
        ``index`` has more than one dim or is not ``int32`` / ``int64``,
        or has elements while ``input``'s ``dim`` has size 0 (an
        ``IndexError`` and a ``RuntimeError``).
    lucid._C.engine.DeviceMismatch
        ``index`` is not on ``input``'s device.

    Examples
    --------
    >>> import lucid
    >>> lucid.index_fill(lucid.zeros(2, 3), 1, lucid.tensor([0, 2]), 1.0)
    tensor([[1., 0., 1.], [1., 0., 1.]])
    """
    shape = list(input.shape)
    if not shape:
        shape = [1]
    dim, positions = _positions("index_fill", input, dim, index)
    n = shape[dim]
    device = input._impl.device
    hits = _call_engine(
        "index_fill",
        lambda: _C_engine.scatter_add(
            _C_engine.zeros([n], _C_engine.I32, device),
            positions,
            _C_engine.full([int(positions.shape[0])], 1.0, _C_engine.I32, device),
            0,
        ),
    )
    along = [1] * len(shape)
    along[dim] = n
    mask = _wrap(_C_engine.reshape(hits, along)) > 0
    if input.ndim == 0:
        mask = mask.reshape(())
    return lucid.where(mask, lucid.full_like(input, value), input)


def index_add(
    input: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    *,
    alpha: float = 1.0,
) -> Tensor:
    """Return ``input`` with ``alpha * source`` accumulated at ``index`` positions along ``dim``.

    Differentiable through both ``input`` and ``source``.  ``alpha`` is
    accumulated as a Python constant — gradients pass through cleanly
    as if the multiplication were inlined.

    Parameters
    ----------
    input : Tensor
        Source tensor; not mutated.
    dim : int
        Axis along which ``index`` addresses slices.
    index : Tensor
        0-D or 1-D ``int32`` / ``int64`` tensor of :math:`m` positions
        along ``dim`` to accumulate into.
    source : Tensor
        Per-slice update tensor; same shape as ``input`` except
        ``source.shape[dim] == m`` (matching ``index`` length), and
        ``input``'s dtype.  A 0-d ``source`` goes with a 0-d ``input``.
    alpha : float, optional
        Scalar multiplier applied to ``source`` before accumulation.
        Default ``1.0``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; positions listed in
        ``index`` carry ``input[..., index[i], ...] + alpha * source[..., i, ...]``.

    Raises
    ------
    lucid._C.engine.DtypeMismatch
        ``source``'s dtype is not ``input``'s (a ``RuntimeError`` and a
        ``TypeError``).
    IndexError
        ``dim`` is out of range, or, on the CPU, an ``index`` value lies
        outside ``[-size, size)`` of ``dim``.  Metal drops such a write.
    lucid._C.engine.IndexError
        ``index`` is not a 0-D or 1-D ``int32`` / ``int64`` tensor, it has
        elements while ``input``'s ``dim`` has size 0 (the reference adds
        nothing there instead), or ``source`` lacks ``input``'s number of
        dims or one slice per index along ``dim`` (an ``IndexError`` and a
        ``RuntimeError``).
    lucid._C.engine.ShapeMismatch
        ``source`` and ``input`` differ on a dim other than ``dim``.
    lucid._C.engine.DeviceMismatch
        ``index`` or ``source`` lives on another device than ``input``.

    Examples
    --------
    >>> import lucid
    >>> lucid.index_add(lucid.ones(4), 0, lucid.tensor([3, 0, 3]), lucid.ones(3))
    tensor([2., 1., 1., 3.])
    """
    _require_input_dtype("index_add", input, source, "source")
    dim, positions = _positions("index_add", input, dim, index)
    count = int(positions.shape[0])
    _require_source_slices("index_add", input, dim, count, source, same_rank=True)
    scaled = source * alpha if alpha != 1 else source
    return _write_slices(
        "index_add",
        input,
        dim,
        positions,
        scaled,
        lambda base, at, src: _C_engine.scatter_add(base, at, src, dim),
    )


def index_copy(
    input: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
) -> Tensor:
    """Return a copy of ``input`` with slices at ``index`` replaced by ``source``.

    Single set-scatter: the 1-D ``index`` is broadcast along ``dim`` to
    ``source``'s shape and applied through the engine ``scatter_set`` primitive
    — one MPSGraph ``scatterAlongAxis`` (Set mode) op when compiled, or
    ``mlx.put_along_axis`` eager on GPU.  Differentiable through both ``input``
    and ``source``.

    Parameters
    ----------
    input : Tensor
        Destination tensor; not mutated (a fresh copy is returned).
    dim : int
        Axis along which slices are addressed.
    index : Tensor
        0-D or 1-D ``int32`` / ``int64`` tensor of positions along ``dim``.
        Its number of elements must equal ``source.shape[dim]``.
    source : Tensor
        Replacement slices.  All non-``dim`` dimensions must match
        ``input``; ``source.shape[dim]`` must equal ``index.numel()``.
        A 0-d ``input`` or ``source`` counts as a one-element vector.
        Must have ``input``'s dtype.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; values at the indexed
        positions are taken from ``source``, others from ``input``.

    Raises
    ------
    lucid._C.engine.DtypeMismatch
        ``source``'s dtype is not ``input``'s.
    IndexError
        ``dim`` is out of range, or, on the CPU, an ``index`` value lies
        outside ``[-size, size)`` of ``dim``.  Metal drops such a write.
    lucid._C.engine.IndexError
        ``index`` is not a 0-D or 1-D ``int32`` / ``int64`` tensor, points
        into a ``dim`` of size 0, or ``source``, read as a vector when 0-d,
        has another number of dims than ``input`` or another number of
        slices along ``dim`` than ``index`` has elements (an ``IndexError``
        and a ``RuntimeError``).
    lucid._C.engine.ShapeMismatch
        A dim of ``source`` other than ``dim`` differs from ``input``'s.
    lucid._C.engine.DeviceMismatch
        ``input``, ``index`` and ``source`` are not all on one device.

    Examples
    --------
    >>> import lucid
    >>> lucid.index_copy(lucid.zeros(4), 0, lucid.tensor([3, 0]), lucid.tensor([1.0, 2.0]))
    tensor([2., 0., 0., 1.])
    """
    with _autocast_entry("index_copy", _unwrap(input), _unwrap(source)) as (
        in_impl,
        src_impl,
    ):
        input, source = _wrap(in_impl), _wrap(src_impl)
        _require_input_dtype("index_copy", input, source, "source")
        dim, positions = _positions("index_copy", input, dim, index)
        count = int(positions.shape[0])
        _require_source_slices("index_copy", input, dim, count, source, same_rank=False)
        return _write_slices(
            "index_copy",
            input,
            dim,
            positions,
            source,
            lambda base, at, src: _C_engine.scatter_set(base, at, src, dim),
        )


def _scatter_into(
    base: Tensor, dim: int, index: Tensor, src: Tensor, reduce: str
) -> Tensor:
    """``src`` reduced into ``base`` by ``reduce`` — a ``'mean'`` as its sum."""
    if reduce in ("sum", "mean"):
        return _call_engine(
            "scatter_reduce", lambda: _scatter_add(base, dim, index, src)
        )
    _fn = {
        "amax": _C_engine.scatter_amax,
        "amin": _C_engine.scatter_amin,
        "prod": _C_engine.scatter_prod,
    }[reduce]
    return _call_engine(
        "scatter_reduce",
        lambda: _wrap(_fn(_unwrap(base), _unwrap(index), _unwrap(src), dim)),
    )


def _divide(total: Tensor, count: Tensor) -> Tensor:
    """``total / count`` in ``total``'s dtype, floored for an integer dtype."""
    if total.is_floating_point() or total.is_complex():
        return total / count
    # ``//`` widens int32 to int64.
    return (total // count).to(total.dtype)


def _scatter_count(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """How many entries ``index`` sends to each position of ``input``.

    Counted in ``int64`` whatever ``input``'s dtype: counted in it, a
    complex count could not be compared with 0, and a bool one saturated.
    """
    zeros = lucid.zeros(tuple(input.shape), dtype=lucid.int64, device=input.device)
    ones = lucid.ones(tuple(index.shape), dtype=lucid.int64, device=input.device)
    return _scatter_add(zeros, dim, index, ones)


def _reduce_identity(reduce: str, input: Tensor) -> bool | float:
    """The value a ``reduce`` leaves unchanged, within ``input``'s dtype.

    What an ``include_self=False`` reduction starts from, so a position
    ``src`` reaches comes out as the reduction of those entries alone.
    An integer dtype holds no infinity, so ``amax`` / ``amin`` start from
    its bounds, and a bool from ``False`` / ``True``.

    Raises
    ------
    NotImplementedError
        ``amax`` / ``amin`` of a complex tensor, which has no order.
    """
    if reduce in ("sum", "mean"):
        return 0.0
    if reduce == "prod":
        return 1.0
    if input.is_complex():
        raise NotImplementedError(
            f"scatter_reduce: {reduce} has no order on complex values"
        )
    if input.is_floating_point():
        return float("-inf") if reduce == "amax" else float("inf")
    if input.dtype == lucid.bool_:
        return reduce == "amin"
    info = lucid.iinfo(input.dtype)
    return info.min if reduce == "amax" else info.max


def scatter_reduce(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    *,
    include_self: bool = True,
) -> Tensor:
    """Reduce ``src`` into ``input`` along ``dim`` at positions given by ``index``.

    Multi-reduction sibling of :func:`scatter_add`.  When several entries
    of ``src`` target the same position the chosen ``reduce`` op decides
    how they combine.  ``include_self`` controls whether the existing
    value in ``input`` participates in the reduction or is replaced.

    Every reduction treats a position no ``index`` entry names the same
    way: it keeps ``input``'s value, and the gradient reaches ``input``
    there.  With ``include_self=False`` a position ``index`` does name
    takes nothing from ``input`` — neither its value nor a gradient.
    For ``'amax'`` / ``'amin'`` the gradient of a position splits evenly
    among the values tied for its result; with ``include_self=False``
    ``input``'s value is not among them even when it equals the result,
    so the shares still add up to the incoming gradient.  This differs
    from the reference framework, which counts that value as a tie and
    then drops its share.

    Parameters
    ----------
    input : Tensor
        Destination tensor; not mutated (a fresh copy is returned).
    dim : int
        Axis along which ``index`` / ``src`` are scattered.
    index : Tensor
        ``int32`` / ``int64`` tensor of ``input``'s number of dims, no larger
        than ``src`` on any dim nor than ``input`` on any dim but ``dim``
        (it does not broadcast).  Each entry names the position along
        ``dim`` of ``input`` that the matching ``src`` entry updates.
    src : Tensor
        Values to scatter into ``input`` at the positions named by
        ``index``.  Must have ``input``'s dtype.
    reduce : str, optional
        Reduction op applied when multiple ``src`` values collide on
        the same target.  One of ``'sum'`` (default), ``'mean'``,
        ``'prod'``, ``'amax'``, ``'amin'``.  ``'mean'`` divides by the
        number of values reduced — the scattered ones, plus ``input``'s
        own when ``include_self`` — and floors on an integer dtype.
    include_self : bool, optional
        When ``True`` (default) the existing value in ``input`` is part
        of the reduction set; when ``False`` it is overwritten at every
        position ``index`` names and only the scattered values count.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.

    Raises
    ------
    ValueError
        If ``reduce`` is not one of the five reductions.
    lucid._C.engine.DtypeMismatch
        ``src``'s dtype is not ``input``'s.
    NotImplementedError
        ``'amax'`` / ``'amin'`` on a complex tensor, which has no order.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> index = lucid.tensor([[0, 0, 1]])
    >>> src = lucid.tensor([[10.0, 20.0, 30.0]])
    >>> lucid.scatter_reduce(x, 1, index, src, "sum", include_self=False)
    tensor([[30., 30., 3., 4.]])
    >>> lucid.scatter_reduce(x, 1, index, src, "mean")
    tensor([[10.33, 16., 3., 4.]])
    """
    if reduce not in ("sum", "mean", "prod", "amax", "amin"):
        raise ValueError(
            f"scatter_reduce: unknown reduce={reduce!r}; "
            "expected 'sum', 'mean', 'prod', 'amax', or 'amin'."
        )
    _require_input_dtype("scatter_reduce", input, src, "src")
    if include_self:
        out = _scatter_into(input, dim, index, src, reduce)
        if reduce != "mean":
            return out
    else:
        identity = lucid.full_like(input, _reduce_identity(reduce, input))
        out = _scatter_into(identity, dim, index, src, reduce)
    count = _scatter_count(input, dim, index)
    if reduce == "mean":
        # A position no index names divides by 1 rather than 0: it is
        # replaced by ``input`` below, and 0 / 0 there would turn its
        # masked-off gradient into NaN.
        out = _divide(out, count + 1 if include_self else count.clamp(min=1))
    return out if include_self else lucid.where(count > 0, out, input)


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """Fill the positions where ``mask`` holds with ``source``'s leading elements.

    ``input`` and ``mask`` broadcast together; the ``k``-th position the
    mask selects, in row-major order, takes ``source.reshape(-1)[k]``, and
    every other position keeps ``input``'s value.  Differentiable through
    ``input`` (where the mask is False) and ``source`` (its elements that
    were written).

    Parameters
    ----------
    input : Tensor
        Destination values; not mutated.
    mask : Tensor
        Bool tensor broadcasting against ``input``.
    source : Tensor
        Values to write, of ``input``'s dtype and any shape, with at least
        as many elements as the mask selects.

    Returns
    -------
    Tensor
        A new tensor of the broadcast shape of ``input`` and ``mask``, in
        ``input``'s dtype — never ``input`` itself, even when the mask
        selects nothing.

    Raises
    ------
    lucid._C.engine.DtypeMismatch
        ``mask`` is not bool, or ``source``'s dtype is not ``input``'s.
    lucid._C.engine.ShapeMismatch
        ``source`` has fewer elements than the mask selects.
    IndexError
        ``input`` and ``mask`` do not broadcast together.

    Examples
    --------
    >>> import lucid
    >>> mask = lucid.tensor([True, False])
    >>> lucid.masked_scatter(lucid.zeros(2, 2), mask, lucid.tensor([1.0, 2.0, 3.0]))
    tensor([[1., 0.], [2., 0.]])
    """
    if mask.dtype != lucid.bool_:
        raise _C_engine.DtypeMismatch(
            f"masked_scatter: the mask must be a bool tensor, got {mask.dtype}"
        )
    _require_input_dtype("masked_scatter", input, source, "source")
    shape = _broadcast_shape([list(input.shape), list(mask.shape)])
    values = input.broadcast_to(tuple(shape)).contiguous().reshape(-1)
    selected = mask.broadcast_to(tuple(shape)).reshape(-1)
    # nonzero is the one host round trip: how many positions the mask
    # selects decides how much of source is read, and a source too short
    # for them is refused rather than read past its end.
    positions = lucid.nonzero(selected).reshape(-1)
    count = int(positions.shape[0])
    available = source.numel()
    if count > available:
        raise _C_engine.ShapeMismatch(
            f"masked_scatter: the mask selects {count} elements, but source "
            f"has only {available}"
        )
    picked = source.reshape(-1).narrow(0, 0, count)
    out = _C_engine.scatter_set(_unwrap(values), _unwrap(positions), _unwrap(picked), 0)
    return _wrap(_C_engine.reshape(out, shape))


def index_put(
    input: Tensor,
    indices: list[Tensor] | tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """Out-of-place advanced-indexing write.

    Equivalent to ``out = input.clone(); out[tuple(indices)] = values``
    (adding at each position when ``accumulate=True``): ``indices`` is
    read as the key ``Tensor.__setitem__`` reads, by the same owner
    (``lucid._tensor._indexing._normalize_key``).  A bool mask selects its
    True positions; integer index tensors broadcast together and address
    the leading dimensions, the rest taken whole —
    ``index_put(x, (i,), v)`` on a ``(4, 3)`` tensor writes whole rows.

    Parameters
    ----------
    input : Tensor
        Destination tensor; not mutated.
    indices : sequence of Tensors
        One ``int32`` / ``int64`` index tensor per leading dimension of
        ``input``, or a bool mask covering as many dims as it has.
    values : Tensor
        Values of ``input``'s dtype, broadcastable to the shape
        ``input[tuple(indices)]`` reads.
    accumulate : bool, default False
        If True, add at each position — a position named twice receives
        both; otherwise overwrite — a position named twice keeps the last
        write on the CPU (and on Metal under
        :func:`lucid.use_deterministic_algorithms`), and an unspecified
        one of its values on Metal otherwise.

    Returns
    -------
    Tensor
        A new tensor of ``input``'s shape and dtype.

    Raises
    ------
    ValueError
        ``indices`` is not a non-empty sequence.
    IndexError
        An index tensor of another integer width (``int8``, ``int16``),
        more indices than ``input`` has dims, a mask whose shape does not
        match, index tensors that do not broadcast together, or — on the
        CPU — an index out of range; Metal drops such a write.
    lucid._C.engine.DtypeMismatch
        ``values``'s dtype is not ``input``'s.

    Examples
    --------
    >>> import lucid
    >>> mask = lucid.tensor([False, False, True, True])
    >>> lucid.index_put(lucid.zeros(4), (mask,), lucid.tensor(5.0))
    tensor([0., 0., 5., 5.])
    """
    return _index_put(input, indices, values, accumulate, "index_put")


def _index_put(
    input: Tensor,
    indices: object,
    values: Tensor,
    accumulate: bool,
    name: str,
) -> Tensor:
    """:func:`index_put`, its refusals under ``name`` — ``index_put_`` runs it too.

    ``name`` is for messages only: both spellings take the autocast table's
    ``index_put`` row.  ``indices`` is whatever the caller passed, and is
    checked here.
    """
    if not isinstance(indices, (list, tuple)) or len(indices) == 0:
        raise ValueError(f"{name}: `indices` must be a non-empty sequence of Tensors")
    with _autocast_entry("index_put", _unwrap(input), _unwrap(values)) as (
        impl,
        val_impl,
    ):
        input, values = _wrap(impl), _wrap(val_impl)
        _require_input_dtype(name, input, values, "values")
        shape = list(impl.shape)
        key = _normalize_key(tuple(indices), shape, impl.device, op=name)
        positions, target_shape = _written_positions(shape, key, impl.device)
        flat_values = _C_engine.reshape(
            _C_engine.contiguous(_normalize_value(values, impl, target_shape)), [-1]
        )
        return _put_flat(name, input, positions, flat_values, accumulate)


def _put_flat(
    name: str,
    input: Tensor,
    positions: _C_engine.TensorImpl,
    values: _C_engine.TensorImpl,
    accumulate: bool,
) -> Tensor:
    """``input`` with 1-D ``values`` written at its flat ``positions``."""
    impl = _unwrap(input)
    shape = list(impl.shape)
    if not impl.is_contiguous():
        impl = _C_engine.contiguous(impl)
    flat = _C_engine.reshape(impl, [int(input.numel())])
    if accumulate:
        out = _call_engine(
            name, lambda: _C_engine.scatter_add(flat, positions, values, 0)
        )
    else:
        out = _call_engine(name, lambda: _C_engine.scatter(flat, 0, positions, values))
    return _wrap(_C_engine.reshape(out, shape))


def put(
    input: Tensor,
    index: Tensor,
    source: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """Write ``source`` into ``input`` at the *flat* positions in ``index``.

    Mirrors the reference framework's ``Tensor.put`` semantics: indices
    refer to the row-major linearisation of ``input``, regardless of its
    shape, and a negative one counts from the end.  ``accumulate=True``
    performs additive scatter (duplicates add).  Otherwise a position
    named twice keeps the last write on the CPU (and on Metal under
    :func:`lucid.use_deterministic_algorithms`), and one of its values,
    which one unspecified, on Metal otherwise (``scatter`` semantics).

    Parameters
    ----------
    input : Tensor
        Destination — its shape is preserved in the output.
    index : Tensor
        ``int32`` / ``int64`` tensor of any shape of flat positions in
        ``[-input.numel(), input.numel())``.
    source : Tensor
        Values of ``input``'s dtype, as many as ``index`` has.
    accumulate : bool, default False
        If True, add to the existing value at each position; otherwise
        overwrite.

    Returns
    -------
    Tensor
        A new tensor of ``input``'s shape and dtype.

    Raises
    ------
    lucid._C.engine.DtypeMismatch
        ``source``'s dtype is not ``input``'s.
    lucid._C.engine.DeviceMismatch
        ``index`` or ``source`` is on another device than ``input``.
    lucid._C.engine.IndexError
        ``index`` is not ``int32`` / ``int64``, or has elements while
        ``input`` has none.
    IndexError
        ``source`` and ``index`` hold different numbers of elements, or —
        on the CPU — an index is out of range; Metal drops such a write.
    """
    _require_input_dtype("put", input, source, "source")
    # put flattens the index before the engine sees it, so the engine's
    # refusals name its ``scatter``; these are asked here, under put's name.
    _require_index("put", input, index)
    _require_device("put", input, source, "source")
    if source.numel() != index.numel():
        raise IndexError(
            f"put: source and index must have the same number of elements, "
            f"got {source.numel()} and {index.numel()}"
        )
    if index.numel() and input.numel() == 0:
        raise _C_engine.IndexError(
            f"put: index out of range: input has no elements, got "
            f"{index.numel()} indices"
        )
    positions = _C_engine.reshape(_unwrap(index), [-1])
    values = _C_engine.reshape(_C_engine.contiguous(_unwrap(source)), [-1])
    return _put_flat("put", input, positions, values, accumulate)


def index_put_(
    input: Tensor,
    indices: list[Tensor] | tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """In-place variant of :func:`index_put`: writes into ``input``'s storage.

    The values land in ``input``'s own buffer, so a view of ``input`` (or
    the tensor ``input`` is a view of) sees them, and ``input`` takes the
    write's place in the autograd graph, as the engine's in-place ops do.
    A leaf that requires grad is refused while autograd records — wrap the
    write in :func:`lucid.no_grad`, which leaves a Parameter a trainable
    leaf.

    Parameters
    ----------
    input : Tensor
        Destination; written in place.
    indices : list of Tensor or tuple of Tensor
        Per-axis index tensors (one per leading dimension of ``input``);
        same contract as :func:`index_put`.
    values : Tensor
        Values to write at the addressed positions, of ``input``'s dtype;
        another dtype is refused (``lucid._C.engine.DtypeMismatch``).
    accumulate : bool, optional
        When ``True`` add to existing values (duplicate indices sum);
        when ``False`` (default) overwrite — a duplicate index keeps the
        last write on the CPU (and on Metal under
        :func:`lucid.use_deterministic_algorithms`), and an unspecified
        one of its values on Metal otherwise.

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding the updated values.

    Raises
    ------
    RuntimeError
        If ``input`` is a leaf that requires grad and autograd is recording.
    """
    return _write_inplace(
        input,
        "index_put_",
        lambda base: _index_put(base, indices, values, accumulate, "index_put_"),
        values,
    )


def argwhere(input: Tensor) -> Tensor:
    """Return the coordinates of every non-zero element in ``input``.

    Thin alias for :func:`lucid.nonzero` named to match the NumPy /
    reference-framework convention.  The output is a synchronisation
    point on GPU streams — the kernel can't know how many non-zeros
    there are without a device→host count.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape and dtype; zero is determined by the
        usual truthiness rules (``0`` for numeric dtypes, ``False`` for
        bool).

    Returns
    -------
    Tensor
        Shape ``(N, x.ndim)`` ``int64`` tensor; row ``i`` lists the
        multi-dimensional index of the ``i``-th non-zero element in
        row-major order.

    See Also
    --------
    lucid.nonzero : the underlying engine call.
    """
    return lucid.nonzero(input)


__all__ = [
    "index_fill",
    "index_add",
    "index_copy",
    "scatter_reduce",
    "masked_scatter",
    "put",
    "index_put",
    "index_put_",
    "argwhere",
]
