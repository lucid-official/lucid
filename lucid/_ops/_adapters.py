"""Adapter functions for ops whose Python signature differs from the engine's.

Every function here normalises a flexible Python calling convention (e.g.
``dim=None | int | list``, ``keepdim=...``, variadic shape, list-as-positional)
into the strict positional signature that the underlying engine kernel
expects.  ``_registry.py`` references these adapters as ``engine_fn`` on the
relevant ``OpEntry``; ``_make_method`` and ``_make_free_fn`` then forward
both ``*args`` and ``**kwargs`` through, so the adapter sees exactly what
the user wrote at the call site.
"""

import inspect
import itertools
import math
import operator
from typing import Callable, Sequence, SupportsIndex, TYPE_CHECKING, TypeIs, cast

from lucid._C import engine as _C_engine
from lucid import _promotion
from lucid._dispatch import (
    _REAL_FLOATS,
    _binary_operands,
    _promote_impls,
    _refuse_bool_subtraction,
    _scalar_power,
    _unwrap,
    _unwrap_or_scalar,
)
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._ops._dims import (
    _dim_size,
    _kept_sizes,
    _require_real_axis,
    _to_axes,
    _wrap_dim,
    _wrap_dims,
)
from lucid._ops._reduce import _ValuesIndices
from lucid._types import Scalar, TensorOrScalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Convenience alias for the engine's TensorImpl class — it's the lingua
# franca of every adapter signature.  Adapters consume the engine's raw
# storage type, while user-facing args are typed as ``Tensor``.
_Impl = _C_engine.TensorImpl


def _own_impl(out: _Impl, a_impl: _Impl) -> _Impl:
    """``out``, or a same-shape view of ``a_impl`` when an op handed it back.

    A no-op view (``narrow`` over a whole dimension, ``squeeze`` of a dim
    that is not size 1) returned its input's own ``TensorImpl``, so two
    Python tensors shared one: flipping ``requires_grad`` on the result
    flipped it on the input, and a gradient accumulated into either landed
    on both.  A view is a tensor of its own — the reference framework's
    result is a distinct view too — that still reads and writes the input's
    storage and, when the input requires grad, carries it through autograd.
    """
    if out is a_impl:
        return _C_engine.view(a_impl, list(a_impl.shape))
    return out


def _keep_engine_signature(
    adapter: Callable[..., object], engine_fn: Callable[..., object]
) -> None:
    """Let ``adapter`` stand in for ``engine_fn`` wherever a signature is read.

    The pybind11 docstring carries the canonical signature line, and
    ``__wrapped__`` points ``_signature_for_entry`` back at it, so
    ``inspect.signature``, ``help()`` and ``gen_pyi.py`` see the op's own
    Tensor-typed parameters rather than the adapter's ``_Impl``-typed ones:
    wrapping an op leaves its public signature and its stub as they were.
    """
    adapter.__doc__ = getattr(engine_fn, "__doc__", None)
    adapter.__name__ = getattr(engine_fn, "__name__", adapter.__name__)
    adapter.__wrapped__ = engine_fn  # type: ignore[attr-defined]


#: The engine ops that compute a half tensor against a 0-d float32 operand
#: in float32 (op-math); the reference does so for mul, div and floor
#: division only.  ``div`` has an adapter of its own (``_make_div_adapter``).
_OPMATH_OPS = (
    _C_engine.mul,
    _C_engine.mul_,
    _C_engine.floordiv,
)


#: ``(input, other)`` without annotations: either operand may be a tensor or
#: a Python number, which the adapter's ``_Impl`` annotations do not say.
_BINARY_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("other", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
)

#: ``(input, other, *, alpha=1)`` for ``add``, ``add_``, ``sub``, ``sub_``.
_ADD_SUB_SIGNATURE = _BINARY_SIGNATURE.replace(
    parameters=[
        *_BINARY_SIGNATURE.parameters.values(),
        inspect.Parameter("alpha", inspect.Parameter.KEYWORD_ONLY, default=1, annotation=Scalar),
    ]
)


def _scale_by_alpha(a: _Impl, b: _Impl, alpha: Scalar) -> _Impl:
    """Scale ``b`` by ``alpha`` for ``add`` / ``sub``, matching the reference's dtype rules."""
    alpha = _promotion.as_number(alpha)
    if isinstance(alpha, bool):
        if a.dtype != _C_engine.Bool:
            raise RuntimeError("Boolean alpha only supported for Boolean results.")
        return b if alpha else _C_engine.zeros_like(b)
    if isinstance(alpha, float) and _promotion.is_integral(a.dtype):
        raise RuntimeError(
            "For integral input tensors, argument alpha must not be a floating point number."
        )
    if alpha == 1 or (a.dtype == _C_engine.Bool and alpha != 0):
        return b
    if a.dtype == _C_engine.Bool:
        return _C_engine.zeros_like(b)
    scale = _unwrap_or_scalar(alpha, b)
    return _C_engine.mul(b, _C_engine.astype(scale, b.dtype) if scale.dtype != b.dtype else scale)


def _make_arith_adapter(
    engine_fn: Callable[[_Impl, _Impl], _Impl],
    *,
    inplace: bool = False,
) -> Callable[..., _Impl]:
    """Wrap a binary engine function with scalars and dtype promotion.

    Both operands go through :func:`lucid._dispatch._binary_operands`, the
    path the operators take too, so ``a.add(2)``, ``lucid.add(a, 2)`` and
    ``a + 2`` cannot drift apart: a Python number becomes a 0-d constant, and
    the pair meets at the dtype :mod:`lucid._promotion` gives it.  Two tensors
    of one dtype pass straight through.

    Used by the arithmetic ops (add/sub/mul/pow/maximum/minimum,
    floor_divide) and by the elementwise binaries the reference framework
    also accepts a scalar for: the comparisons, fmod/remainder, and the
    bitwise and shift ops.  Those return their own dtype — Bool for a
    comparison — but they still want both operands at a common one first.

    matmul/dot/inner/outer deliberately bypass it: a scalar operand is
    meaningless there.  ``outer`` promotes its two tensors through
    ``_outer_adapter`` instead; the others have dtype constraints of their
    own that the engine enforces.

    mul and floor division — in place or not — hand a half tensor's scalar
    partner over unrounded, for the engine's op-math.  ``div`` and ``div_``,
    whose ``rounding_mode`` picks the kernel, have :func:`_make_div_adapter`.

    ``inplace`` marks the trailing-underscore form, which writes the result
    back into ``input``: a result dtype that ``input`` cannot hold is refused.
    """
    name = getattr(engine_fn, "__name__", "_arith_adapter")

    subtracts = engine_fn in (_C_engine.sub, _C_engine.sub_)
    adds_or_subtracts = subtracts or engine_fn in (_C_engine.add, _C_engine.add_)
    powers = engine_fn is _C_engine.pow
    opmath = engine_fn in _OPMATH_OPS

    # Positional-only: the wrappers bind the advertised names to position
    # (``exponent`` for ``pow``), so ``other=`` must not reach the body too.
    def _adapter(input: _Impl, other: _Impl, /, *, alpha: Scalar = 1) -> _Impl:
        if subtracts:
            _refuse_bool_subtraction(input, other)
        if powers and isinstance(input, _C_engine.TensorImpl):
            scalar = _scalar_power(input, other)
            if scalar is not None:
                return scalar
        a, b = _binary_operands(input, other, opmath=opmath, inplace=inplace)
        if adds_or_subtracts:
            alpha = _promotion.as_number(alpha)
            if alpha != 1 or isinstance(alpha, (bool, float)):
                b = _scale_by_alpha(a, b, alpha)
        return engine_fn(a, b)

    # The adapter's own ``(input, other)`` is the op's signature — not the
    # engine's names, which for ``atan2(y, x)`` came out as ``(other, input)``
    # and advertised the operands in the wrong order.
    _adapter.__doc__ = getattr(engine_fn, "__doc__", None)
    _adapter.__name__ = name
    _adapter.__dict__["__signature__"] = (
        _ADD_SUB_SIGNATURE if adds_or_subtracts else _BINARY_SIGNATURE
    )
    return _adapter


_complex_parts = _make_arith_adapter(_C_engine.complex)


def _complex_adapter(real: _Impl, imag: _Impl) -> _Impl:
    """complex(real, imag) — the reference's names for the two parts."""
    return _complex_parts(real, imag)


def _make_join_adapter(engine_fn: Callable[..., _Impl]) -> Callable[..., _Impl]:
    """Wrap a joining engine op (cat, stack, ...) so it takes mixed dtypes.

    The engine joins tensors of a single dtype only, so ``cat`` of an
    ``int64`` and a ``float32`` raised ``DtypeMismatch``, where the
    reference joins them at their common dtype.  The inputs are brought
    there first; inputs that already agree pass straight through.
    """

    def _adapter(tensors: Sequence[_Impl], *args: object, **kwargs: object) -> _Impl:
        return engine_fn(_promote_impls(tensors), *args, **kwargs)

    _keep_engine_signature(_adapter, engine_fn)
    return _adapter


# ── Engine-arg-order adapters ────────────────────────────────────────────────


def _detach_adapter(x_impl: _Impl) -> _Impl:
    """detach(x): the same storage, cut from the graph (see Tensor.detach)."""
    return x_impl.data_alias()


def _scatter_add_adapter(
    x_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> _Impl:
    """scatter_add(x, dim, index, src) — Python order → engine order.

    The index goes over at full width: the engine checks an ``int64`` index
    against the axis as it is, and narrowing it to ``int32`` first wrapped
    ``2**40`` round to ``0`` and added into the wrong slot.
    """
    return _C_engine.scatter_add(x_impl, _unwrap(index), _unwrap(src), dim)


def _as_dim(value: object) -> int:
    """``value`` as a ``gather`` dim: an int, or a 0-d integer tensor."""
    impl = _promotion.try_unwrap(value)
    if impl is not None:
        if not impl.shape and _promotion.is_integer(impl.dtype):
            return int(cast(int, impl.item()))
        raise TypeError(
            "gather(): dim must be an int or a 0-d integer tensor, got a tensor "
            f"of shape {tuple(impl.shape)} and dtype {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    try:
        return operator.index(value)  # type: ignore[arg-type]  # refused below
    except TypeError:
        raise TypeError(f"gather(): dim must be an int, got {type(value).__name__}") from None


def _gather_adapter(input: _Impl, dim: int, index: Tensor, *, sparse_grad: bool = False) -> _Impl:
    """gather(input, dim, index, *, sparse_grad=False).

    ``dim`` may be a 0-d integer tensor, as the reference reads one.
    """
    if sparse_grad:
        raise _C_engine.NotImplementedError("gather(): sparse_grad=True is not supported")
    index_impl = _promotion.try_unwrap(index)
    if index_impl is None:
        raise TypeError(f"gather(): index must be a Tensor, got {type(index).__name__}")
    d = dim if type(dim) is int else _as_dim(dim)
    return _C_engine.gather(_unwrap(input), index_impl, d)


# ── Composite indexing adapters ──────────────────────────────────────────────


def _take_adapter(a_impl: _Impl, index: Tensor) -> _Impl:
    """take(a, indices) — ``a`` read as 1-D, in the shape of ``index``.

    The engine op takes a 1-D index only, so any other shape — a matrix of
    positions, a 0-d one — raised ``ShapeMismatch`` where the reference
    answers in the index's shape.
    """
    idx = _unwrap(index)
    shape = list(idx.shape)
    if len(shape) == 1:
        return _C_engine.take(a_impl, idx)
    flat = _C_engine.take(a_impl, _C_engine.reshape(idx, [-1]))
    return _C_engine.reshape(flat, shape)


def _index_select_adapter(a_impl: _Impl, dim: int, index: Tensor) -> _Impl:
    """index_select(a, dim, index) — third arg is a tensor."""
    return _C_engine.index_select(a_impl, int(dim), _unwrap(index))


#: ``scatter``'s ``reduce=`` modes, each its own engine kernel.
_SCATTER_REDUCE = {"add": _C_engine.scatter_add, "multiply": _C_engine.scatter_prod}


def _scatter_adapter(
    x_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor | Scalar,
    reduce: str | None = None,
) -> _Impl:
    """scatter(base, dim, index, src, reduce=None).

    ``src`` may be a Python number, written at every position the index
    names.  A tensor ``src`` must have ``base``'s dtype: the engine's
    overwrite scatter casts it, as an assignment inside Lucid wants, but
    the reference refuses here as it does for every scatter — and as the
    engine's ``reduce`` kernels already do.  The index goes over at full
    width, so one past the axis is refused rather than wrapped.
    """
    idx_impl = _unwrap(index)
    src = _promotion.as_number(src)
    if isinstance(src, _promotion.NUMBER):
        src_impl = _C_engine.full(list(idx_impl.shape), src, x_impl.dtype, x_impl.device)
    else:
        src_impl = _unwrap(src)
        if src_impl.dtype != x_impl.dtype:
            raise _C_engine.DtypeMismatch(
                f"scatter: src must have self's dtype "
                f"{_ENGINE_TO_DTYPE[x_impl.dtype]}, got "
                f"{_ENGINE_TO_DTYPE[src_impl.dtype]} — cast it first"
            )
    if reduce is None:
        return _C_engine.scatter(x_impl, int(dim), idx_impl, src_impl)
    kernel = _SCATTER_REDUCE.get(reduce)
    if kernel is None:
        raise ValueError(f"scatter: reduce must be 'add' or 'multiply', got {reduce!r}")
    return kernel(x_impl, idx_impl, src_impl, int(dim))


def _broadcast_with_mask(op: str, a_impl: _Impl, mask: _Impl) -> tuple[_Impl, _Impl]:
    """``a_impl`` and ``mask`` broadcast to their common shape, as the reference's are.

    The engine's masked kernels walk the two buffers side by side, so each
    operand has to arrive at the shape the other one sees.
    """
    a_shape = list(a_impl.shape)
    m_shape = list(mask.shape)
    rank = max(len(a_shape), len(m_shape))
    a_full = [1] * (rank - len(a_shape)) + a_shape
    m_full = [1] * (rank - len(m_shape)) + m_shape
    shape: list[int] = []
    for i, (x, y) in enumerate(zip(a_full, m_full, strict=True)):
        if x != y and 1 not in (x, y):
            raise ValueError(
                f"{op}: input shape {tuple(a_shape)} and mask shape "
                f"{tuple(m_shape)} do not broadcast (dimension {i}: {x} vs {y})"
            )
        shape.append(0 if 0 in (x, y) else max(x, y))
    if a_shape != shape:
        a_impl = _C_engine.broadcast_to(a_impl, shape)
    if m_shape != shape:
        mask = _C_engine.broadcast_to(mask, shape)
    return a_impl, mask


def _masked_select_adapter(a_impl: _Impl, mask: _Impl) -> _Impl:
    """masked_select(a, mask) — ``a``'s elements where ``mask`` holds, as 1-D.

    Both operands broadcast to their common shape first, as the reference
    framework's do.  The engine kernel did not: given a mask of a different
    shape it walked the two buffers side by side, so a row mask returned
    the first row's picks only and a broadcast input read past its end.
    The kernel also records no graph, so an input that tracks gradients
    goes through boolean indexing instead, which scatters the gradient
    back to the selected positions.
    """
    a_impl, mask = _broadcast_with_mask("masked_select", a_impl, mask)
    if mask.dtype != _C_engine.Bool:
        mask = _C_engine.astype(mask, _C_engine.Bool)
    if a_impl.requires_grad and _C_engine.grad_enabled():
        from lucid._dispatch import _wrap

        return _unwrap(_wrap(a_impl)[_wrap(mask)])
    return _C_engine.masked_select(_C_engine.contiguous(a_impl), _C_engine.contiguous(mask))


def _sort_adapter(
    a_impl: _Impl, dim: int = -1, descending: bool = False, *, stable: bool = False
) -> _ValuesIndices[_Impl]:
    """sort(a, dim=-1, descending=False, *, stable=False) — ``(values, indices)`` along ``dim``."""
    del stable
    d = _wrap_dim(int(dim), len(a_impl.shape), "sort")
    indices = _argsort_adapter(a_impl, d, descending)
    if not (descending and a_impl.shape):
        return _ValuesIndices((_C_engine.sort(a_impl, d), indices))
    if a_impl.requires_grad and _C_engine.grad_enabled():
        return _ValuesIndices((_C_engine.gather(a_impl, indices, d), indices))
    return _ValuesIndices((_C_engine.flip(_C_engine.sort(a_impl, d), [d]), indices))


def _argsort_adapter(
    a_impl: _Impl,
    dim: int = -1,
    descending: bool = False,
    *,
    stable: bool = False,
) -> _Impl:
    """argsort(a, dim=-1, descending=False, *, stable=False).

    Descending keeps equal keys in their input order, as ascending does:
    sort the reversed input ascending, map each position ``p`` of it back
    to index ``n - 1 - p``, and reverse.  Reversing the ascending indices
    instead would put ties last-first.
    """
    del stable
    d = _wrap_dim(int(dim), len(a_impl.shape), "argsort")
    if not descending or not a_impl.shape:
        return _C_engine.argsort(a_impl, d)
    reversed_order = _C_engine.argsort(_C_engine.flip(a_impl, [d]), d)
    n = _dim_size(a_impl.shape, d)
    last = _C_engine.full_like(reversed_order, float(n - 1))
    return _C_engine.flip(_C_engine.sub(last, reversed_order), [d])


def _topk_adapter(
    a_impl: _Impl,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> _ValuesIndices[_Impl]:
    """topk(a, k, dim=-1, largest=True, sorted=True) — ``(values, indices)``, best first.

    The smallest ``k`` are the head of the stable ascending order, so equal
    keys keep their input order and NaN, which sorts last, is taken only
    when fewer than ``k`` numbers remain.
    """
    del sorted
    d = _wrap_dim(int(dim), len(a_impl.shape), "topk")
    # A 0-d tensor's one element is both its largest and its smallest, and
    # ``narrow`` below has no real axis to cut.
    if largest or not a_impl.shape:
        values, indices = _C_engine.topk(a_impl, int(k), d)
        return _ValuesIndices((values, indices))
    order = _C_engine.narrow(_C_engine.argsort(a_impl, d), d, 0, int(k))
    return _ValuesIndices((_C_engine.gather(a_impl, order, d), order))


def _chunk_adapter(a_impl: _Impl, chunks: int, dim: int = 0) -> list[_Impl]:
    """chunk(a, chunks, dim=0).

    Pieces of ``ceil(n / chunks)`` along ``dim``, the last one shorter —
    so fewer than ``chunks`` pieces come back when that size leaves nothing
    for the tail (7 split into 4 is 2, 2, 2, 1; 6 split into 4 is 2, 2, 2).
    """
    count = int(chunks)
    if count <= 0:
        raise _C_engine.InvalidArgument(
            f"chunk expects `chunks` to be greater than 0, got: {count}"
        )
    d = _wrap_dim(int(dim), len(a_impl.shape), "chunk")
    _require_real_axis(len(a_impl.shape), "chunk expects at least a 1-dimensional tensor")
    size = _dim_size(a_impl.shape, d)
    if size % count == 0:
        return _C_engine.chunk(a_impl, count, d)
    step = (size + count - 1) // count
    return _C_engine.split_at(a_impl, list(range(step, size, step)), d)


def _swapaxes_adapter(a_impl: _Impl, axis0: int, axis1: int) -> _Impl:
    """swapaxes(a, axis0, axis1) — the free function's parameter names."""
    return _C_engine.swapaxes(a_impl, int(axis0), int(axis1))


def _kthvalue_adapter(
    a_impl: _Impl,
    k: int,
    dim: int = -1,
    keepdim: bool = False,
) -> _ValuesIndices[_Impl]:
    """kthvalue(a, k, dim=-1, keepdim=False) — ``(values, indices)``."""
    values = _C_engine.kthvalue(a_impl, int(k), int(dim), bool(keepdim))
    if not a_impl.shape:
        indices = _C_engine.zeros([], _C_engine.I64, a_impl.device)
    else:
        d = _wrap_dim(int(dim), len(a_impl.shape), "kthvalue")
        indices = _C_engine.narrow(_C_engine.argsort(a_impl, d), d, int(k) - 1, 1)
        if not keepdim:
            indices = _C_engine.squeeze(indices, d)
    return _ValuesIndices((values, indices))


def _narrow_adapter(a_impl: _Impl, dim: int, start: int, length: int) -> _Impl:
    """narrow(a, dim, start, length)."""
    d = int(dim)
    s = int(start)
    ndim = len(a_impl.shape)
    if s < 0 and -ndim <= d < ndim:
        s += a_impl.shape[d]
    return _own_impl(_C_engine.narrow(a_impl, d, s, int(length)), a_impl)


def _is_count(value: object) -> TypeIs[SupportsIndex]:
    return isinstance(value, SupportsIndex) and not isinstance(value, bool)


def _integers(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(
            f"as_strided(): argument {name!r} must be a sequence of ints, not "
            f"{type(value).__name__}"
        )
    for at, item in enumerate(value):
        if not _is_count(item):
            raise TypeError(
                f"as_strided(): argument {name!r} must be a sequence of ints, but "
                f"found an element of type {type(item).__name__} at position {at}"
            )
    return tuple(operator.index(item) for item in value)


def _as_strided_adapter(
    x_impl: _Impl,
    size: Sequence[int],
    stride: Sequence[int],
    storage_offset: int | None = None,
) -> _Impl:
    """as_strided(x, size, stride, storage_offset=None), all in elements."""
    if storage_offset is not None and not _is_count(storage_offset):
        raise TypeError(
            "as_strided(): argument 'storage_offset' must be int or None, not "
            f"{type(storage_offset).__name__}"
        )
    at = None if storage_offset is None else operator.index(storage_offset)
    if at is not None and at < 0:
        raise _C_engine.LucidError(f"as_strided: invalid storage offset {at}")
    sz = _integers("size", size)
    st = _integers("stride", stride)
    return _C_engine.as_strided(x_impl, list(sz), list(st), at)


_as_strided_adapter.__name__ = _as_strided_adapter.__qualname__ = "as_strided"


def _contiguous_adapter(a_impl: _Impl) -> _Impl:
    """contiguous(a) — return the input itself when already contiguous outside a trace."""
    if (
        a_impl.is_contiguous()
        and _C_engine.compile.current_tracer() is None
        and (not a_impl.is_phantom or _C_engine.phantom_mode())
    ):
        return a_impl
    return _C_engine.contiguous(a_impl)


# ── Layout / shape adapters ──────────────────────────────────────────────────


def _movedim_adapter(
    a_impl: _Impl,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> _Impl:
    """movedim(a, source, destination) — accept int or list for either arg."""
    return _C_engine.movedim(
        a_impl,
        _int_list(source, "movedim", "source"),
        _int_list(destination, "movedim", "destination"),
    )


def _unflatten_adapter(a_impl: _Impl, dim: int, sizes: Sequence[int]) -> _Impl:
    """unflatten(a, dim, sizes)."""
    return _C_engine.unflatten(a_impl, int(dim), [int(s) for s in sizes])


def _int_list(
    value: int | Sequence[int],
    name: str = "op",
    what: str = "dims",
) -> list[int]:
    """An int-list argument as a list: one int is a list of one.

    The method spellings ``x.view(2, 3)`` reach these adapters as one tuple
    (``lucid._tensor._methods``), the reference's own reading.  Routes
    through :func:`_to_axes` so ``SupportsIndex`` scalars (such as
    ``np.int64``) are accepted and ``bool`` / ``None`` are refused.
    """
    if value is None:
        raise TypeError(f"{name}(): {what} must be an int or a sequence of ints, got None")
    return _to_axes(value, name, what)


def _view_adapter(a_impl: _Impl, size: int | Sequence[int]) -> _Impl:
    """view(input, size) — ``x.view(2, 3)`` is ``x.view((2, 3))``."""
    return _C_engine.view(a_impl, _int_list(size, "view", "size"))


def _concat_adapter(tensors: Sequence[Tensor], dim: int = 0) -> _Impl:
    """concat(tensors, dim=0) — first arg is a list of tensors."""
    return _C_engine.concat(_promote_impls([_unwrap(t) for t in tensors]), int(dim))


def _reshape_adapter(x_impl: _Impl, shape: int | Sequence[int]) -> _Impl:
    """reshape(input, shape) — ``x.reshape(2, 3)`` is ``x.reshape((2, 3))``."""
    return _C_engine.reshape(x_impl, _int_list(shape, "reshape", "shape"))


def _transpose_adapter(x_impl: _Impl, dim0: int | None = None, dim1: int | None = None) -> _Impl:
    """transpose(x) swaps the last two axes; transpose(x, dim0, dim1) swaps those.

    The two-axis form is the reference framework's, and code written for it
    raised a TypeError here — the engine op takes no axes.
    """
    if dim0 is None and dim1 is None:
        return _C_engine.transpose(x_impl)
    if dim0 is None or dim1 is None:
        raise TypeError("transpose takes no axes, or two: transpose(dim0, dim1)")
    return _C_engine.swapaxes(x_impl, int(dim0), int(dim1))


def _permute_adapter(x_impl: _Impl, dims: int | Sequence[int]) -> _Impl:
    """permute(input, dims) — ``x.permute(1, 0)`` is ``x.permute((1, 0))``."""
    return _C_engine.permute(x_impl, _int_list(dims, "permute", "dims"))


def _broadcast_to_adapter(x_impl: _Impl, size: int | Sequence[int]) -> _Impl:
    """broadcast_to(input, size) — ``-1`` keeps a dim's size, as in ``expand``."""
    raw = _int_list(size, "broadcast_to", "size")
    return _C_engine.expand(x_impl, _kept_sizes(x_impl.shape, raw, "broadcast_to"))


def _expand_adapter(x_impl: _Impl, size: int | Sequence[int], *, implicit: bool = False) -> _Impl:
    """expand(input, size, *, implicit=False) — ``x.expand(2, 3)`` is ``x.expand((2, 3))``."""
    del implicit
    raw = _int_list(size, "expand", "size")
    return _C_engine.expand(x_impl, _kept_sizes(x_impl.shape, raw, "expand"))


def _flip_adapter(x_impl: _Impl, dims: int | Sequence[int]) -> _Impl:
    """flip(x, dims) — accept ``dims=int`` or list/tuple of ints."""
    return _C_engine.flip(x_impl, _int_list(dims, "flip", "dims"))


def _roll_adapter(
    a_impl: _Impl,
    shifts: int | Sequence[int],
    dims: int | Sequence[int] | None = None,
) -> _Impl:
    """roll(input, shifts, dims=None) — no ``dims`` rolls the flat order.

    The engine takes per-axis shifts only, so a call without ``dims`` — or
    with one int for either — reached the binding and failed on its
    signature.  As in the reference, no ``dims`` rolls the elements in
    their flattened order and puts the shape back, a 0-d tensor included.
    """
    steps = _int_list(shifts, "roll", "shifts")
    if dims is None:
        if len(steps) != 1:
            raise ValueError(f"roll: without dims, shifts must be a single int, got {shifts!r}")
        flat = _C_engine.reshape(a_impl, [a_impl.numel()])
        rolled = _C_engine.roll(flat, steps, [0])
        return _C_engine.reshape(rolled, list(a_impl.shape))
    axes = _to_axes(dims, "roll")
    if len(steps) != len(axes):
        raise ValueError(
            f"roll: shifts and dims must align, got {len(steps)} shift(s) and {len(axes)} dim(s)"
        )
    return _C_engine.roll(a_impl, steps, axes)


def _fliplr_adapter(x_impl: _Impl) -> _Impl:
    """fliplr(x) — flip along axis 1.  ``x`` must be at least 2-D."""
    if len(x_impl.shape) < 2:
        raise ValueError(f"fliplr: input must be at least 2-D, got shape {tuple(x_impl.shape)}")
    return _C_engine.flip(x_impl, [1])


def _flipud_adapter(x_impl: _Impl) -> _Impl:
    """flipud(x) — flip along axis 0.  ``x`` must be at least 1-D."""
    if len(x_impl.shape) < 1:
        raise ValueError(f"flipud: input must be at least 1-D, got shape {tuple(x_impl.shape)}")
    return _C_engine.flip(x_impl, [0])


def _tril_adapter(a_impl: _Impl, diagonal: int = 0) -> _Impl:
    """tril(input, diagonal=0)."""
    return _C_engine.tril(a_impl, int(diagonal))


def _triu_adapter(a_impl: _Impl, diagonal: int = 0) -> _Impl:
    """triu(input, diagonal=0)."""
    return _C_engine.triu(a_impl, int(diagonal))


def _flatten_adapter(a_impl: _Impl, start_dim: int = 0, end_dim: int = -1) -> _Impl:
    """flatten(input, start_dim=0, end_dim=-1)."""
    return _C_engine.flatten(a_impl, int(start_dim), int(end_dim))


def _tile_adapter(a_impl: _Impl, dims: int | Sequence[int]) -> _Impl:
    """tile(input, dims) — ``x.tile(2, 2)`` is ``x.tile((2, 2))``.

    A single int is one count, as ``x.tile(2)`` reads in the reference.
    Fewer counts than axes are padded with leading 1s, as documented and as
    the reference does; the engine refuses them.
    """
    counts = _int_list(dims, "tile", "dims")
    short = len(a_impl.shape) - len(counts)
    return _C_engine.tile(a_impl, [1] * short + counts)


def _squeeze_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
) -> _Impl:
    """squeeze(x, dim=None) — None drops all size-1; list squeezes multiple.

    A dim that is not size 1 is left alone, as the reference does — and the
    result is still a view of its own (see :func:`_own_impl`).
    """
    if dim is None:
        return _own_impl(_C_engine.squeeze_all(x_impl), x_impl)
    shape = x_impl.shape
    result = x_impl
    # Descending, so each ``squeeze`` leaves the indices still to come valid.
    # A 0-d tensor's notional axis has nothing to drop.
    dims = _wrap_dims(_to_axes(dim, "squeeze"), len(shape), "squeeze")
    for d in sorted(dims, reverse=True):
        if shape and int(shape[d]) == 1:
            result = _C_engine.squeeze(result, d)
    return _own_impl(result, x_impl)


# ── Repeat / split adapters ──────────────────────────────────────────────────


def _repeat_adapter(x_impl: _Impl, repeats: int, dim: int | None = None) -> _Impl:
    """``lucid.repeat(x, repeats, dim=None)`` — interleaved replication along
    axis 0 when ``dim`` is None, else along the given axis.

    Distinct from ``Tensor.repeat`` (below): the free function follows
    NumPy-style ``repeat`` semantics, while the method tiles like the
    reference framework's ``Tensor.repeat``.
    """
    axis = 0 if dim is None else int(dim)
    return _C_engine.repeat(x_impl, int(repeats), axis)


def _repeat_method_adapter(x_impl: _Impl, repeats: int | Sequence[int]) -> _Impl:
    """``Tensor.repeat(repeats)`` — tile copies along each dim.

    Routes to ``engine.tile`` so the method semantics match the reference
    framework's ``Tensor.repeat`` (and stay separated from the free
    function above).
    """
    return _C_engine.tile(x_impl, _int_list(repeats, "repeat", "repeats"))


def _repeat_interleave_tensor(base: _Impl, rep_impl: _Impl, axis: int) -> _Impl:
    """Interleave ``base`` along ``axis`` by a 0-D or 1-D integer tensor ``rep_impl``."""
    if not _promotion.is_integer(rep_impl.dtype) or rep_impl.dtype == _C_engine.Bool:
        raise RuntimeError("repeats has to be Long or Int tensor")
    if not rep_impl.shape:
        return _C_engine.repeat(base, int(rep_impl.item()), axis)
    if len(rep_impl.shape) != 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    raw_list = rep_impl.tolist()
    assert isinstance(raw_list, list)
    counts = [int(c) for c in raw_list if isinstance(c, (int, float))]
    axis_len = int(base.shape[axis])
    if len(counts) == 1 and axis_len != 1:
        counts = counts * axis_len
    if len(counts) != axis_len:
        raise RuntimeError("repeats must have the same size as input along dim")
    if any(c < 0 for c in counts):
        raise RuntimeError("repeats can not be negative")
    pieces = [
        _C_engine.repeat(_C_engine.narrow(base, axis, i, 1), c, axis)
        for i, c in enumerate(counts)
        if c > 0
    ]
    if pieces:
        return _C_engine.concatenate(pieces, axis)
    empty_shape = list(base.shape)
    empty_shape[axis] = 0
    return _C_engine.zeros(empty_shape, base.dtype, base.device)


def _repeat_interleave_adapter(
    a_impl: _Impl,
    repeats: int | Tensor,
    dim: int | None = None,
    *,
    output_size: int | None = None,
) -> _Impl:
    """repeat_interleave(x, repeats, dim=None, *, output_size=None)."""
    base = _C_engine.reshape(a_impl, [int(a_impl.numel())]) if dim is None else a_impl
    axis = 0 if dim is None else _wrap_dim(int(dim), len(a_impl.shape), "repeat_interleave")
    repeats = _promotion.as_number(repeats)
    out = (
        _C_engine.repeat(base, int(repeats), axis)
        if isinstance(repeats, int)
        else _repeat_interleave_tensor(base, _unwrap(repeats), axis)
    )
    if output_size is not None and int(out.shape[axis]) != int(output_size):
        raise RuntimeError(
            f"Allocated size does not match required size: expected {out.shape[axis]} but got {output_size}"
        )
    return out


def _split_pieces(x_impl: _Impl, sections: int | Sequence[int], dim: int, what: str) -> list[_Impl]:
    """The pieces of ``x_impl`` along ``dim`` for a split size or size list.

    The one place a split's sizes become the cut points of the engine's
    ``split_at``, whose own check is only that each cut lies inside the
    axis.  A size list that did not sum to the axis passed that check, and
    the last piece took the remainder or ran short: ``[2, 3]`` of a length-6
    axis came back ``[2, 4]``, of a length-4 one ``[2, 2]``.  So the sizes
    are checked here, before a cut is made, as the reference checks them.

    ``what`` names the size argument in a type refusal, since the free
    function and the method spell it differently.

    Raises
    ------
    TypeError
        A size that is not an integer (a ``bool`` included).
    lucid._C.engine.InvalidArgument
        A 0-d tensor, a negative size, a zero ``split_size`` on a non-empty
        axis, or a size list that does not sum to the axis.
    """
    wrapped = _wrap_dim(int(dim), len(x_impl.shape), "split")
    _require_real_axis(len(x_impl.shape), "split expects at least a 1-dimensional tensor")
    axis_size = _dim_size(x_impl.shape, wrapped)
    if isinstance(_promotion.as_number(sections), bool):
        raise TypeError(f"split(): {what} must be an int or a sequence of ints, got {sections!r}")
    if not isinstance(sections, Sequence):
        size = operator.index(sections)
        if size < 0:
            raise _C_engine.InvalidArgument(
                f"split expects split_size be non-negative, but got split_size={size}"
            )
        if size == 0 and axis_size:
            raise _C_engine.InvalidArgument(
                "split_size can only be 0 if dimension size is 0, but got "
                f"dimension size of {axis_size}"
            )
        if size and axis_size and axis_size % size == 0:
            return _C_engine.split(x_impl, axis_size // size, wrapped)
        cuts = list(range(size, axis_size, size)) if size else []
        return _C_engine.split_at(x_impl, cuts, wrapped)
    sizes = _int_list(sections, "split", what)
    if any(s < 0 for s in sizes):
        raise _C_engine.InvalidArgument(
            "split_with_sizes expects split_sizes have only non-negative entries, "
            f"but got split_sizes={sizes}"
        )
    if sum(sizes) != axis_size:
        raise _C_engine.InvalidArgument(
            f"split_with_sizes expects split_sizes to sum exactly to {axis_size} "
            f"(input tensor's size at dimension {dim}), but got split_sizes={sizes}"
        )
    if not sizes:
        return []
    return _C_engine.split_at(x_impl, list(itertools.accumulate(sizes[:-1])), wrapped)


def _split_adapter(
    x_impl: _Impl,
    split_size_or_sections: int | Sequence[int],
    dim: int = 0,
) -> list[_Impl]:
    """split(x, sections, dim=0) — int → pieces of that size, the last one
    shorter when it does not divide; list → exactly those sizes."""
    return _split_pieces(x_impl, split_size_or_sections, dim, "split_size_or_sections")


def _split_method_adapter(
    x_impl: _Impl, split_size: int | Sequence[int], dim: int = 0
) -> list[_Impl]:
    """``Tensor.split(split_size, dim=0)`` — the method's name for the sizes."""
    return _split_pieces(x_impl, split_size, dim, "split_size")


# ── tensordot / meshgrid / where / masked_fill ─────────────────────────────


def _tensordot_adapter(
    a_impl: _Impl,
    b_impl: _Impl,
    dims: int | Sequence[Sequence[int]] = 2,
) -> _Impl:
    """tensordot(a, b, dims=2) — an int, or a pair of axis lists."""
    dims = _promotion.as_number(dims)
    if isinstance(dims, int):
        n, ra = int(dims), len(a_impl.shape)
        axes_a = [ra] if n < 0 or n > ra else list(range(ra - n, ra))
        axes_b = list(range(n))
    else:
        axes_a = [int(d) for d in dims[0]]
        axes_b = [int(d) for d in dims[1]]
    return _C_engine.tensordot(a_impl, b_impl, axes_a, axes_b)


def _outer_adapter(a_impl: _Impl, b_impl: _Impl) -> _Impl:
    """outer(a, b) — the two vectors at their common dtype.

    The engine multiplies operands of one dtype only, so an ``int64`` vector
    beside a ``float32`` one raised ``DtypeMismatch``; the reference answers
    in ``float32``, as ``a * b`` would.  Shared by ``Tensor.outer`` and
    ``lucid.linalg.outer`` so the two cannot drift.

    Both callers hand over impls already, and they are not unwrapped again:
    shadow allocation's phantom impls pass one ``_unwrap`` but not a second.
    A Python number is refused rather than coerced, as the bare engine op
    refused it, since an outer product with one is not an outer product.
    """
    for operand in (a_impl, b_impl):
        if isinstance(_promotion.as_number(operand), _promotion.NUMBER):
            raise TypeError(f"outer: expected a tensor, got {type(operand).__name__}")
    a_impl, b_impl = _promote_impls([a_impl, b_impl])
    return _C_engine.outer(a_impl, b_impl)


_keep_engine_signature(_outer_adapter, _C_engine.outer)


def _kron_adapter(a_impl: _Impl, b_impl: _Impl) -> _Impl:
    """kron(a, b) — the two tensors at their common dtype."""
    for operand in (a_impl, b_impl):
        if isinstance(_promotion.as_number(operand), _promotion.NUMBER):
            raise TypeError(f"kron: expected a tensor, got {type(operand).__name__}")
    a_impl, b_impl = _promote_impls([a_impl, b_impl])
    return _C_engine.kron(a_impl, b_impl)


_keep_engine_signature(_kron_adapter, _C_engine.kron)


def _meshgrid_adapter(*tensors: Tensor, indexing: str = "ij") -> list[_Impl]:
    """meshgrid(*tensors, indexing='ij') — variadic input; ``indexing`` kwarg
    selects between matrix ('ij') and Cartesian ('xy') ordering."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    impls = [_unwrap(t) for t in tensors]
    return _C_engine.meshgrid(impls, indexing == "xy")


def _clip_bound_impl(
    bound: TensorOrScalar | None, x_impl: _Impl, tgt: _C_engine.Dtype
) -> _Impl | None:
    """Unwrap an optional bound and cast to ``tgt``."""
    if bound is None:
        return None
    b = _unwrap_or_scalar(bound, x_impl)
    return _C_engine.astype(b, tgt) if b.dtype != tgt else b


def _clip_with_nan(out: _Impl, any_nan: _Impl, is_float: bool) -> _Impl:
    """Propagate NaN onto ``out`` when ``is_float`` holds."""
    if not is_float:
        return out
    nan_c = _C_engine.full([], math.nan, out.dtype, out.device)
    return _C_engine.where(any_nan, nan_c, out)


def _clip_both_impls(x_impl: _Impl, lo_impl: _Impl, hi_impl: _Impl, is_float: bool) -> _Impl:
    """Two-sided tensor-bound clamp matching the reference's tie and NaN rules."""
    use_x = _C_engine.logical_and(
        _C_engine.greater_equal(x_impl, lo_impl),
        _C_engine.less_equal(x_impl, hi_impl),
    )
    use_lo = _C_engine.logical_and(
        _C_engine.less(x_impl, lo_impl),
        _C_engine.less(lo_impl, hi_impl),
    )
    use_hi = _C_engine.logical_or(
        _C_engine.greater(x_impl, hi_impl),
        _C_engine.greater(lo_impl, hi_impl),
    )
    out = _C_engine.where(
        use_x,
        x_impl,
        _C_engine.where(
            use_lo,
            lo_impl,
            _C_engine.where(use_hi, hi_impl, hi_impl.data_alias()),
        ),
    )
    if not is_float:
        return out
    any_nan = _C_engine.logical_or(
        _C_engine.logical_or(_C_engine.isnan(x_impl), _C_engine.isnan(lo_impl)),
        _C_engine.isnan(hi_impl),
    )
    return _clip_with_nan(out, any_nan, True)


def _clip_scalar_bounds(
    x_impl: _Impl,
    min: TensorOrScalar | None,
    max: TensorOrScalar | None,
    is_float: bool,
    clip: Callable[[_Impl, float, float], _Impl] = _C_engine.clip,
) -> _Impl | None:
    """Fast path when ``min`` and ``max`` are Python numbers or ``None``.

    ``clip`` is the engine kernel — ``clip_`` for ``Tensor.clamp_``, so the
    in-place form takes the fast path exactly when this one does.
    """
    min = _promotion.as_number(min)
    max = _promotion.as_number(max)
    if not isinstance(min, (int, float, type(None))) or not isinstance(
        max, (int, float, type(None))
    ):
        return None
    if is_float:
        lo_f = -math.inf if min is None else float(min)
        hi_f = math.inf if max is None else float(max)
        return clip(x_impl, lo_f, hi_f)
    if min is not None and max is not None and abs(min) <= (1 << 53) and abs(max) <= (1 << 53):
        return clip(x_impl, float(min), float(max))
    return None


def _clip_dtype(
    x_impl: _Impl,
    min: TensorOrScalar | None,
    max: TensorOrScalar | None,
) -> _C_engine.Dtype:
    """The dtype ``clip(x, min, max)`` computes in: ``x``'s, promoted with its bounds.

    One rule for ``clamp`` and ``clamp_``: the in-place form refuses a dtype
    ``x`` cannot hold where this one returns it.
    """
    if min is None and max is None:
        raise ValueError("clip: at least one of min or max must be given")
    bounds: list[_promotion.Operand] = [
        (bn if isinstance(bn := _promotion.as_number(b), _promotion.NUMBER) else _unwrap(bn))
        for b in (min, max)
        if b is not None
    ]
    return _promotion.result_type(x_impl, *bounds)


def _clip_adapter(
    x_impl: _Impl,
    min: TensorOrScalar | None = None,
    max: TensorOrScalar | None = None,
) -> _Impl:
    """clip(x, min, max) — either bound may be omitted or a tensor.

    Scalar bounds on floating-point tensors route through ``_C_engine.clip``
    (with ``-inf`` / ``+inf`` stand-ins for omitted ends) so the tie gradient
    at the boundary is asymmetric (0 at ``x == min`` or ``x == max``) rather
    than the 0.5/0.5 tie split of ``maximum`` / ``minimum``.  Tensor bounds
    and integer one-sided bounds use asymmetric ``where`` comparisons (and
    propagate NaN across floating-point operands) to match the reference
    framework's ``clamp(Tensor, min=Tensor, max=Tensor)`` overload.
    """
    tgt = _clip_dtype(x_impl, min, max)
    if tgt != x_impl.dtype:
        x_impl = _C_engine.astype(x_impl, tgt)

    is_float = tgt in _REAL_FLOATS
    fast = _clip_scalar_bounds(x_impl, min, max, is_float)
    if fast is not None:
        return fast

    lo_impl = _clip_bound_impl(min, x_impl, tgt)
    hi_impl = _clip_bound_impl(max, x_impl, tgt)
    if lo_impl is not None and hi_impl is None:
        out = _C_engine.where(_C_engine.less(x_impl, lo_impl), lo_impl, x_impl)
        return _clip_with_nan(
            out,
            _C_engine.logical_or(_C_engine.isnan(x_impl), _C_engine.isnan(lo_impl)),
            is_float,
        )
    if hi_impl is not None and lo_impl is None:
        out = _C_engine.where(_C_engine.greater(x_impl, hi_impl), hi_impl, x_impl)
        return _clip_with_nan(
            out,
            _C_engine.logical_or(_C_engine.isnan(x_impl), _C_engine.isnan(hi_impl)),
            is_float,
        )
    assert lo_impl is not None and hi_impl is not None
    return _clip_both_impls(x_impl, lo_impl, hi_impl, is_float)


def _clamp_min_adapter(x_impl: _Impl, min: TensorOrScalar) -> _Impl:
    """clamp_min(x, min) — raise elements below ``min`` to ``min``."""
    if min is None:
        raise TypeError("clamp_min(): argument 'min' must not be None")
    return _clip_adapter(x_impl, min, None)


def _clamp_max_adapter(x_impl: _Impl, max: TensorOrScalar) -> _Impl:
    """clamp_max(x, max) — lower elements above ``max`` to ``max``."""
    if max is None:
        raise TypeError("clamp_max(): argument 'max' must not be None")
    return _clip_adapter(x_impl, None, max)


def _where_adapter(
    condition: Tensor,
    input: TensorOrScalar | None = None,
    other: TensorOrScalar | None = None,
) -> _Impl | tuple[_Impl, ...]:
    """where(cond, x, y) — bool-cast the condition, promote the branches.

    ``where(cond)`` alone is ``nonzero(cond, as_tuple=True)``, as in the
    reference: the indices of the true elements, one tensor per dim.

    Either branch may be a Python scalar, which is the common spelling for
    a masked constant: ``where(x > 0, 1.0, -1.0)``.  Each is promoted
    against whichever operand is already a tensor, so the pair keeps the
    dtype it would have had written the long way.

    The two branches then meet at their common dtype, as in ``x + y``: the
    engine selects between tensors of one dtype only, so an ``int64``
    branch beside a ``float32`` one — or beside ``1.5`` — raised
    ``DtypeMismatch`` where the reference answers in float.
    """
    if input is None and other is None:
        return _nonzero_adapter(_unwrap(condition), as_tuple=True)
    if input is None or other is None:
        raise TypeError("where() takes either cond alone, or cond, x and y")
    c = _unwrap(condition)
    if c.dtype != _C_engine.Bool:
        c = _C_engine.astype(c, _C_engine.Bool)
    # Tensor is a TYPE_CHECKING-only name here, so the test is for
    # scalar-ness rather than for the wrapper type.
    input = _promotion.as_number(input)
    other = _promotion.as_number(other)
    ref_impl = c
    for branch in (input, other):
        if not isinstance(branch, _promotion.NUMBER):
            ref_impl = _unwrap(branch)
            break
    x_impl, y_impl = _promote_impls(
        [_unwrap_or_scalar(input, ref_impl), _unwrap_or_scalar(other, ref_impl)]
    )
    return _C_engine.where(c, x_impl, y_impl)


def _nonzero_adapter(a_impl: _Impl, *, as_tuple: bool = False) -> _Impl | tuple[_Impl, ...]:
    """nonzero(a, *, as_tuple=False) — the indices of ``a``'s nonzero elements.

    ``as_tuple`` returns one 1-D index tensor per dim rather than one
    ``[count, ndim]`` matrix; a 0-d input counts as 1-D there, as in the
    reference.  How many elements are nonzero is read back to size the
    result — the one host round trip this op needs.
    """
    if not as_tuple:
        return _C_engine.nonzero(a_impl)
    if not a_impl.shape:
        a_impl = _C_engine.reshape(a_impl, [1])
    return tuple(_C_engine.unbind(_C_engine.nonzero(a_impl), 1))


def _masked_fill_adapter(x_impl: _Impl, mask: Tensor, value: float) -> _Impl:
    """masked_fill(x, mask, value) — auto-cast mask to bool.

    ``x`` and the mask broadcast to their common shape, as the reference's
    do: the engine kernel reads them side by side, so a ``[2, 1, 1, 4]``,
    ``[4]`` or 0-d mask against a ``[2, 3, 4, 4]`` input raised
    ``ShapeMismatch`` instead.

    An integer or bool tensor has no infinity and no NaN, and the engine's
    cast wrote whatever the conversion gave: ``-inf`` became ``INT64_MIN``
    with nothing to say so, which then reads as a huge negative number
    rather than a mask.  The reference refuses such a value, and so does
    this; a finite one is cast to the tensor's dtype as before.
    """
    m = _unwrap(mask)
    if m.dtype != _C_engine.Bool:
        m = _C_engine.astype(m, _C_engine.Bool)
    x_impl, m = _broadcast_with_mask("masked_fill", x_impl, m)
    fill = float(value)
    if not math.isfinite(fill) and _promotion.is_integral(x_impl.dtype):
        dtype = _ENGINE_TO_DTYPE.get(x_impl.dtype, x_impl.dtype)
        raise RuntimeError(
            f"masked_fill: {fill} cannot be written into a {dtype} tensor, "
            "which has no infinity or NaN — cast it to a floating dtype first"
        )
    return _C_engine.masked_fill(x_impl, m, fill)


# ── Stats / search / combinatorial ───────────────────────────────────────────


def _histc_adapter(
    a_impl: _Impl,
    bins: int = 100,
    min: float = 0.0,
    max: float = 0.0,
) -> _Impl:
    """histc(a, bins=100, min=0, max=0) — counts over ``[min, max]``, as the reference.

    ``min == max`` takes the range from the data; a range that is still a
    single point widens to ``[v - 1, v + 1]``, which puts ``v`` in the
    middle bin — the engine widened it to ``[v, v + 1]``, putting every
    element in the first.  A NaN lies in no bin: the kernel's bin index for
    one came out as 0, so it was counted there.  It is sent below the range
    instead, where elements are skipped.
    """
    lo, hi = float(min), float(max)
    if lo > hi:
        raise ValueError(f"histc: max must be at least min, got [{lo}, {hi}]")
    if lo == hi and a_impl.numel() > 0:
        # The range is the data's: two scalars read back, as the engine did.
        lo = float(_C_engine.min(a_impl, [], False).item())
        hi = float(_C_engine.max(a_impl, [], False).item())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"histc: range of [{lo}, {hi}] is not finite")
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    if a_impl.dtype in _REAL_FLOATS:
        a_impl = _C_engine.nan_to_num(a_impl, -math.inf, math.inf, -math.inf)
    return _C_engine.histc(a_impl, int(bins), lo, hi)


#: All non-complex dtypes: the engine's ``unique`` kernel stably orders NaN
#: at the end, collapses ``-0.0`` and ``0.0`` to the first seen, and covers
#: half, sub-int32, and bool inputs directly.
_UNIQUE_KERNEL_DTYPES = (
    _C_engine.Bool,
    _C_engine.I8,
    _C_engine.I16,
    _C_engine.I32,
    _C_engine.I64,
    _C_engine.F16,
    _C_engine.BF16,
    _C_engine.F32,
    _C_engine.F64,
)


def _attach_unique_grad(a_impl: _Impl, values: _Impl, op_name: str) -> None:
    """Attach a backward node to ``values`` that refuses differentiation, as the reference does."""
    if not (a_impl.requires_grad and _C_engine.grad_enabled()):
        return

    def _refuse(_ctx: object, *_grads: _Impl) -> list[_Impl | None]:
        raise NotImplementedError(f"the derivative for '{op_name}' is not implemented.")

    node = _C_engine._PythonBackwardNode()
    node.ctx = _C_engine.FunctionCtx()
    node.backward_fn = _refuse
    _C_engine._register_python_backward_node([values], node, [a_impl])


def _prepare_unique_rows(
    a_impl: _Impl, dim: int | None, op_name: str
) -> tuple[_Impl, int | None, list[int], list[int]]:
    if dim is None:
        return (
            _C_engine.reshape(a_impl, [a_impl.numel(), 1]),
            None,
            [],
            list(a_impl.shape),
        )
    ndim = len(a_impl.shape)
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    d = _wrap_dim(int(dim), ndim, op_name)
    if any(int(s) == 0 for ax, s in enumerate(a_impl.shape) if ax != d):
        raise RuntimeError(
            "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied"
        )
    moved = _C_engine.contiguous(_C_engine.movedim(a_impl, [d], [0]))
    rest = [int(s) for s in moved.shape[1:]]
    shape = [int(moved.shape[0])]
    # The width spelled out: ``-1`` cannot be inferred with no rows.
    return _C_engine.reshape(moved, [shape[0], math.prod(rest)]), d, rest, shape


def _run_unique(
    a_impl: _Impl,
    return_inverse: bool,
    return_counts: bool,
    dim: int | None,
    *,
    consecutive: bool,
) -> _Impl | tuple[_Impl, ...]:
    op_name = "unique_consecutive" if consecutive else "unique"
    plain = dim is None and not (return_inverse or return_counts)
    if not consecutive and plain and a_impl.dtype in _UNIQUE_KERNEL_DTYPES:
        with _C_engine.NoGradGuard():
            values = _C_engine.unique(_C_engine.reshape(a_impl, [-1]))
        _attach_unique_grad(a_impl, values, op_name)
        return values
    with _C_engine.NoGradGuard():
        rows, d, rest, shape = _prepare_unique_rows(a_impl, dim, op_name)
        values, inverse, counts = _unique_rows(
            rows, return_inverse, return_counts, consecutive=consecutive
        )
        if d is None:
            values = _C_engine.reshape(values, [-1])
        else:
            values = _C_engine.movedim(
                _C_engine.reshape(values, [int(values.shape[0]), *rest]), [0], [d]
            )
        out = [values]
        if inverse is not None:
            out.append(_C_engine.reshape(inverse, shape))
        if counts is not None:
            out.append(counts)
    _attach_unique_grad(a_impl, values, op_name)
    return out[0] if len(out) == 1 else tuple(out)


def _unique_adapter(
    a_impl: _Impl,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> _Impl | tuple[_Impl, ...]:
    """unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None).

    The values always come back sorted, which the reference allows for
    ``sorted=False`` too.  ``dim=None`` reads ``a`` as 1-D and the inverse
    takes ``a``'s shape; with ``dim`` the unique values are whole slices
    along it and the inverse has one entry per slice.  How many there are
    is read back to size the result — the one host round trip.
    """
    del sorted  # sorted either way; the reference leaves the order open
    return _run_unique(a_impl, return_inverse, return_counts, dim, consecutive=False)


def _unique_consecutive_adapter(
    a_impl: _Impl,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> _Impl | tuple[_Impl, ...]:
    """unique_consecutive(a, return_inverse=False, return_counts=False, dim=None)."""
    return _run_unique(a_impl, return_inverse, return_counts, dim, consecutive=True)


def _unique_rows(
    rows: _Impl, inverse: bool, counts: bool, *, consecutive: bool = False
) -> tuple[_Impl, _Impl | None, _Impl | None]:
    """The distinct rows of a 2-D ``rows``, sorted; each row's group; each group's size.

    Rows are ordered lexicographically by a stable sort per column, last
    column first; a single column is one sort, and needs no order at all
    when nobody asks where each row went.  Neighbouring rows that differ
    anywhere start a group, so NaN, which equals nothing, is a group of its
    own each time, and ``-0.0`` joins ``0.0``.  The group and size of each
    are computed only when asked for.
    """
    n, width = int(rows.shape[0]), int(rows.shape[1])
    device = rows.device
    order: _Impl | None = None
    if consecutive:
        ranked = rows
    elif inverse or width != 1:
        order = _lexicographic_order(rows)
        ranked = _C_engine.index_select(rows, 0, order)
    else:
        ranked = _C_engine.sort(rows, 0)
    if n == 0:
        empty = _C_engine.zeros([0], _C_engine.I64, device)
        return ranked, empty if inverse else None, empty if counts else None
    differs = _C_engine.not_equal(
        _C_engine.narrow(ranked, 0, 1, n - 1), _C_engine.narrow(ranked, 0, 0, n - 1)
    )
    if width != 1:
        differs = _C_engine.greater(
            _C_engine.sum(_C_engine.astype(differs, _C_engine.I64), [1], False),
            _C_engine.zeros([n - 1], _C_engine.I64, device),
        )
    starts = _C_engine.concatenate(
        [_C_engine.ones([1], _C_engine.Bool, device), _C_engine.reshape(differs, [-1])],
        0,
    )
    first = _C_engine.reshape(_C_engine.nonzero(starts), [-1])
    groups: _Impl | None = None
    if inverse:
        group = _C_engine.sub(
            _C_engine.cumsum(_C_engine.astype(starts, _C_engine.I64), 0),
            _C_engine.ones([n], _C_engine.I64, device),
        )
        groups = (
            _C_engine.scatter(_C_engine.zeros([n], _C_engine.I64, device), 0, order, group)
            if order is not None
            else group
        )
    sizes: _Impl | None = None
    if counts:
        ends = _C_engine.concatenate(
            [
                _C_engine.narrow(first, 0, 1, int(first.shape[0]) - 1),
                _C_engine.full([1], n, _C_engine.I64, device),
            ],
            0,
        )
        sizes = _C_engine.sub(ends, first)
    return _C_engine.index_select(ranked, 0, first), groups, sizes


def _lexicographic_order(rows: _Impl) -> _Impl:
    """The ``int64`` permutation that sorts ``rows`` lexicographically, stably."""
    n, width = int(rows.shape[0]), int(rows.shape[1])
    if width == 1:
        return _C_engine.argsort(_C_engine.reshape(rows, [n]), 0)
    order = _C_engine.arange(0, n, 1, _C_engine.I64, rows.device)
    for col in range(width - 1, -1, -1):
        key = _C_engine.index_select(
            _C_engine.reshape(_C_engine.narrow(rows, 1, col, 1), [n]), 0, order
        )
        order = _C_engine.index_select(order, 0, _C_engine.argsort(key, 0))
    return order


def _cartesian_prod_adapter(*tensors: Tensor) -> _Impl:
    """cartesian_prod(*tensors) — accept variadic tensor args.

    The registry's list-arg path requires the user to pass a list;
    ``cartesian_prod(t1, t2)`` is the more natural calling form, so we
    register with ``n_tensor_args=0`` and unwrap each operand here.
    """
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    return _C_engine.cartesian_prod([_unwrap(t) for t in tensors])


def _searchsorted_adapter(
    sorted_sequence: Tensor,
    input: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: Tensor | None = None,
) -> _Impl:
    """searchsorted(sorted_seq, values, *, out_int32=False, right=False, side=None, sorter=None)."""
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(f"searchsorted(): side can only be 'left' or 'right', got {side!r}")
        if right and side == "left":
            raise RuntimeError(
                "searchsorted(): side and right can't be set to opposites, got side of left while right was True"
            )
        right = side == "right"
    seq_impl = _unwrap(sorted_sequence)
    if sorter is not None:
        s_impl = _unwrap(sorter)
        if s_impl.dtype != _C_engine.I64 or list(s_impl.shape) != list(seq_impl.shape):
            raise RuntimeError(
                "searchsorted: sorter must be int64 dtype and match sorted_sequence shape"
            )
        seq_impl = _C_engine.gather(seq_impl, s_impl, len(seq_impl.shape) - 1)
    out = _C_engine.searchsorted(seq_impl, _unwrap(input), bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out


def _bucketize_adapter(
    input: Tensor, boundaries: Tensor, *, out_int32: bool = False, right: bool = False
) -> _Impl:
    """bucketize(values, boundaries, *, out_int32=False, right=False)."""
    out = _C_engine.bucketize(_unwrap(input), _unwrap(boundaries), bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out


def _isclose_adapter(
    a_impl: _Impl | Scalar,
    b_impl: _Impl | Scalar,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> _Impl:
    """isclose(a, b, rtol, atol, equal_nan) — equal_nan branch handled here.

    Either operand may be a Python scalar; comparing a tensor against a
    literal tolerance target is the ordinary use.
    """
    ref = a_impl if isinstance(a_impl, _C_engine.TensorImpl) else None
    b_impl = _unwrap_or_scalar(b_impl, ref)
    a_impl = _unwrap_or_scalar(a_impl, b_impl)
    if equal_nan:
        # ``isclose ∨ (isnan(a) ∧ isnan(b))`` — we lift to bitwise on bool
        # tensors so the result stays a single bool tensor.
        out = _C_engine.isclose(a_impl, b_impl, float(rtol), float(atol))
        nan_a = _C_engine.isnan(a_impl)
        nan_b = _C_engine.isnan(b_impl)
        both_nan = _C_engine.bitwise_and(nan_a, nan_b)
        return _C_engine.bitwise_or(out, both_nan)
    return _C_engine.isclose(a_impl, b_impl, float(rtol), float(atol))


# Sub-module forwarders for ``cross`` / ``norm`` / ``einsum`` were removed
# (2026-05-08).  Lucid's API tree exposes those ops only via their canonical
# sub-package paths — ``lucid.linalg.cross``, ``lucid.linalg.norm``,
# ``lucid.einops.einsum`` — so the adapters that used to forward them at the
# top level were dead code under the new policy.
