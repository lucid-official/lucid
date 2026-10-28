"""
Auto-inject Tensor methods from the ops registry, and the in-place methods
no engine kernel writes.

Every registry method is generated from an ``OpEntry`` in ``_registry.py``;
signature translation (``dim=None|int|list``, ``keepdim``, ``correction``
...) lives in ``_adapters.py``.

The in-place forms whose op has no in-place kernel — the index writes
(``masked_fill_``, ``scatter_`` ...), ``bitwise_or_``, ``remainder_``,
``floor_divide_``, ``lerp_``, ``addcmul_`` and the rest — and the augmented
assignments that spell them (``|=``, ``%=``, ``//=`` ...) compute the
out-of-place op and write its result into ``self`` through
``lucid._ops.composite.indexing._write_inplace``: into ``self``'s storage,
refusing a result dtype ``self`` cannot hold, bumping its version.  Before,
``|=`` and the others had no in-place dunder, so Python rebound the name to
a new tensor and every view and alias kept the old values; and ``clamp_``
called the engine kernel without the promotion ``clamp`` does, so
``int64.clamp_(0.5, 1.5)`` wrote ``[0, 1, 1, 1]``.

Called once at module import time by ``tensor.py``.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _REAL_FLOATS, _binary_operands, _cannot_cast_inplace, _unwrap, _wrap
from lucid._tensor._indexing import _adopt_inplace
from lucid._ops._adapters import _clip_adapter, _clip_dtype, _clip_scalar_bounds
from lucid._ops._registry import _REGISTRY, OpEntry, _bind_leading

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import DTypeLike, TensorOrScalar

_Impl = _C_engine.TensorImpl


_OPERANDS: dict[str, tuple[str, ...]] = {}

#: Methods whose one argument is a list of ints, which the reference also
#: reads spread out — ``x.permute(1, 0)`` is ``x.permute((1, 0))`` — where
#: its free functions take the list alone (``lucid.permute(x, (1, 0))``).
_SPREAD_INT_LIST = frozenset(
    {"permute", "reshape", "view", "expand", "broadcast_to", "repeat", "tile", "flip"}
)


def _operand_names(e: OpEntry) -> tuple[str, ...]:
    """The advertised names of ``e``'s tensor operands after ``self``.

    Read from the free function's signature, so the method and
    ``lucid.<op>`` bind the same keywords — once per op, on its first
    keyword call: ``lucid._ops`` imports this package, so it cannot be
    imported while the methods are made.
    """
    names = _OPERANDS.get(e.name)
    if names is None:
        from lucid._ops import _leading_tensor_names

        names = _OPERANDS[e.name] = _leading_tensor_names(e)[1:]
    return names


def _wrap_method_result(self: Tensor, result: _Impl) -> Tensor:
    if isinstance(result, (list, tuple)):
        return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
    return self if result is self._impl else _wrap(result)


def _unary_tensor_method(e: OpEntry) -> object:
    spreads = e.method_name in _SPREAD_INT_LIST
    engine_fn = e.engine_fn

    def method(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        if spreads and (len(args) > 1 or not (args or kwargs)):
            args = (args,)
        return _wrap_method_result(self, engine_fn(self._impl, *args, **kwargs))

    method.__name__ = e.method_name or e.name
    return method


def _fixed_arity_method(e: OpEntry, tensor_cls: type) -> object:
    """The Tensor method for ``e`` when ``self`` is followed by fixed arguments.

    Unwraps the leading ``Tensor`` positional args (up to the op's declared
    tensor-arg count), dispatches to the engine op, and either wraps the
    result as a Tensor or returns the raw value per ``e.returns_tensor``.
    """
    if e.n_tensor_args == 1 and not e.inplace and e.returns_tensor:
        return _unary_tensor_method(e)

    binds = e.n_tensor_args > 1
    n_after_self = e.n_tensor_args - 1
    spreads = e.method_name in _SPREAD_INT_LIST
    engine_fn = e.engine_fn

    def method(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Standard-arity method body for registry-driven Tensor methods."""
        if spreads and (len(args) > 1 or not (args or kwargs)):
            args = (args,)
        if kwargs and binds:
            args, kwargs = _bind_leading(method.__name__, _operand_names(e), args, kwargs)
        proc = [
            _unwrap(a) if i < n_after_self and isinstance(a, tensor_cls) else a  # type: ignore[arg-type]
            for i, a in enumerate(args)
        ]
        result = engine_fn(self._impl, *proc, **kwargs)
        if e.inplace:
            return _adopt_inplace(self, result, e.name)
        if e.returns_tensor:
            return _wrap_method_result(self, result)
        return cast("Tensor", result)

    method.__name__ = e.method_name or e.name
    return method


def _masked_fill_(self: Tensor, mask: Tensor, value: float | Tensor) -> Tensor:
    """Fill elements of ``self`` where ``mask`` is ``True`` with ``value``, in place."""
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    if _is_tensor(value):
        if value.ndim != 0:
            raise RuntimeError("masked_fill_ only supports a 0-dimensional value tensor")
        if value.requires_grad:
            import lucid

            return _write_inplace(
                self,
                "masked_fill_",
                lambda base: lucid.where(mask, value.to(base.dtype), base),
                value,
            )
        scalar = float(value.item())
    else:
        scalar = float(value)
    return _write_inplace(self, "masked_fill_", lambda base: base.masked_fill(mask, scalar))


def _scatter_(
    self: Tensor, dim: int, index: Tensor, src: Tensor | float, reduce: str | None = None
) -> Tensor:
    """Write ``src`` into ``self`` along ``dim`` at ``index``, in place."""
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    if _is_tensor(src):
        return _write_inplace(
            self, "scatter_", lambda base: base.scatter(dim, index, src, reduce=reduce), src
        )
    return _write_inplace(
        self, "scatter_", lambda base: base.scatter(dim, index, src, reduce=reduce)
    )


def _scatter_add_(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Add ``src`` into ``self`` along ``dim`` at ``index``, in place."""
    from lucid._ops.composite.indexing import _write_inplace

    return _write_inplace(self, "scatter_add_", lambda base: base.scatter_add(dim, index, src), src)


def _scatter_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    *,
    include_self: bool = True,
) -> Tensor:
    """Out-of-place ``scatter_reduce`` along ``dim``."""
    from lucid._ops.composite.indexing import scatter_reduce as _sr

    return _sr(self, dim, index, src, reduce=reduce, include_self=include_self)


def _scatter_reduce_(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    *,
    include_self: bool = True,
) -> Tensor:
    """In-place ``scatter_reduce_`` along ``dim``."""
    from lucid._ops.composite.indexing import _write_inplace, scatter_reduce as _sr

    return _write_inplace(
        self,
        "scatter_reduce_",
        lambda base: _sr(base, dim, index, src, reduce=reduce, include_self=include_self),
        src,
    )


def _put_(self: Tensor, index: Tensor, source: Tensor, accumulate: bool = False) -> Tensor:
    """Write ``source`` into ``self`` at flat ``index`` positions, in place."""
    from lucid._ops.composite.indexing import _write_inplace, put as _put

    return _write_inplace(
        self, "put_", lambda base: _put(base, index, source, accumulate=accumulate), source
    )


def _masked_scatter(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """Out-of-place ``masked_scatter`` where ``mask`` is ``True``."""
    from lucid._ops.composite.indexing import masked_scatter as _ms

    return _ms(self, mask, source)


def _masked_scatter_(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """In-place ``masked_scatter_`` where ``mask`` is ``True``."""
    from lucid._ops.composite.indexing import _write_inplace, masked_scatter as _ms

    return _write_inplace(self, "masked_scatter_", lambda base: _ms(base, mask, source), source)


def _bincount(self: Tensor, weights: Tensor | None = None, minlength: int = 0) -> Tensor:
    """Count occurrences of each value in 1-D non-negative integer ``self``."""
    from lucid._ops.composite.statistics import bincount as _bc

    return _bc(self, weights=weights, minlength=minlength)


def _take_along_dim(self: Tensor, indices: Tensor, dim: int | None = None) -> Tensor:
    """Select values from ``self`` at ``indices`` along ``dim``."""
    from lucid._ops.composite.shape import take_along_dim as _tad

    return _tad(self, indices, dim=dim)


def _median(
    self: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Median of ``self`` along ``dim`` (or over the whole tensor when ``None``)."""
    from lucid._ops.composite.reductions import median as _med

    return _med(self, dim=dim, keepdim=keepdim)


def _nanmedian(
    self: Tensor, dim: int | None = None, keepdim: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Median of ``self``, ignoring NaN entries."""
    from lucid._ops.composite.reductions import nanmedian as _nanmed

    return _nanmed(self, dim=dim, keepdim=keepdim)


def _mode(self: Tensor, dim: int = -1, keepdim: bool = False) -> tuple[Tensor, Tensor]:
    """Modal value and index of each slice of ``self`` along ``dim``."""
    from lucid._ops.composite.reductions import mode as _md

    return _md(self, dim=dim, keepdim=keepdim)


def _quantile(
    self: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> Tensor:
    """Compute the ``q``-th quantile of ``self``."""
    from lucid._ops.composite.statistics import quantile as _qt

    return _qt(self, q, dim=dim, keepdim=keepdim, interpolation=interpolation)


def _nanquantile(
    self: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> Tensor:
    """Compute the NaN-ignoring ``q``-th quantile of ``self``."""
    from lucid._ops.composite.statistics import nanquantile as _nqt

    return _nqt(self, q, dim=dim, keepdim=keepdim, interpolation=interpolation)


def _msort(self: Tensor) -> Tensor:
    """Sort ``self`` along its first dimension in ascending order."""
    from lucid._ops.composite.reductions import msort as _ms

    return _ms(self)


def _select(self: Tensor, dim: int, index: int) -> Tensor:
    """Slice ``self`` along ``dim`` at ``index``, removing that dimension."""
    from lucid._ops.composite.shape import select as _sel

    return _sel(self, dim=dim, index=index)


def _tensor_split(
    self: Tensor, indices_or_sections: int | Sequence[int], dim: int = 0
) -> list[Tensor]:
    """Split ``self`` along ``dim`` into sections or at cut indices."""
    from lucid._ops.composite.shape import tensor_split as _ts

    return _ts(self, indices_or_sections, dim=dim)


def _vsplit(self: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    """Split ``self`` vertically along dimension 0."""
    from lucid._ops.composite.shape import vsplit as _vs

    return _vs(self, indices_or_sections)


def _hsplit(self: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    """Split ``self`` horizontally along dimension 1 (or 0 for 1-D)."""
    from lucid._ops.composite.shape import hsplit as _hs

    return _hs(self, indices_or_sections)


def _dsplit(self: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    """Split ``self`` along dimension 2."""
    from lucid._ops.composite.shape import dsplit as _ds

    return _ds(self, indices_or_sections)


def _aminmax(
    self: Tensor, *, dim: int | None = None, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """Compute the minimum and maximum of ``self``."""
    from lucid._ops.composite.reductions import aminmax as _amm

    return _amm(self, dim=dim, keepdim=keepdim)


def _renorm(self: Tensor, p: float, dim: int, maxnorm: float) -> Tensor:
    """Rescale sub-tensors of ``self`` along ``dim`` whose ``p``-norm exceeds ``maxnorm``."""
    from lucid._ops.composite.reductions import renorm as _rn

    return _rn(self, p=p, dim=dim, maxnorm=maxnorm)


def _logcumsumexp(self: Tensor, dim: int) -> Tensor:
    """Logarithm of the cumulative sum of exponentials of ``self`` along ``dim``."""
    from lucid._ops.composite.reductions import logcumsumexp as _lce

    return _lce(self, dim=dim)


def _addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    """Element-wise ``self + value * tensor1 * tensor2``."""
    from lucid._ops.composite.blas import addcmul as _acm

    return _acm(self, tensor1, tensor2, value=value)


def _addcdiv(self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    """Element-wise ``self + value * tensor1 / tensor2``."""
    from lucid._ops.composite.blas import addcdiv as _acd

    return _acd(self, tensor1, tensor2, value=value)


def _is_set_to(self: Tensor, tensor: Tensor) -> bool:
    """Return ``True`` when ``self`` and ``tensor`` share the same storage, offset, shape, and stride."""
    from lucid._ops.composite._shared import _is_tensor

    if not _is_tensor(tensor):
        raise TypeError(
            f"is_set_to(): argument 'tensor' (position 1) must be Tensor, not {type(tensor).__name__}"
        )
    if (
        self.device != tensor.device
        or self.dtype != tensor.dtype
        or self.shape != tensor.shape
        or self.stride() != tensor.stride()
        or self.storage_offset() != tensor.storage_offset()
    ):
        return False
    if self.is_metal:
        return self._impl is tensor._impl or (
            (fid := self._impl.family_id()) != 0 and fid == tensor._impl.family_id()
        )
    ptr_a = self.data_ptr()
    ptr_b = tensor.data_ptr()
    if ptr_a == 0 or ptr_b == 0:
        return self._impl is tensor._impl
    return ptr_a == ptr_b


# ── in-place forms with no engine kernel ─────────────────────────────────────


def _write(self: Tensor, name: str, result_of: Callable[[Tensor], Tensor], *args: object) -> Tensor:
    """Make ``result_of(self)`` the values of ``self`` in place; return ``self``.

    ``args`` are the op's other arguments: the tensors among them may carry
    gradient, which decides whether the write records.
    """
    from lucid._ops.composite._shared import _is_tensor
    from lucid._ops.composite.indexing import _write_inplace

    return _write_inplace(self, name, result_of, *(a for a in args if _is_tensor(a)))


def _binary_inplace(
    self: Tensor,
    other: TensorOrScalar,
    name: str,
    op: Callable[[_Impl, _Impl], _Impl],
    *,
    opmath: bool = False,
    promotes: bool = True,
) -> Tensor:
    """``self`` written with ``op(self, other)``; ``self``.

    The operands meet at their common dtype through :func:`_binary_operands`,
    as they do for the operator and for ``lucid.<op>``.  When the op answers
    in that dtype (``promotes``), a dtype ``self`` cannot hold is refused
    before the kernel runs — ``int64 |= 2.5`` raises what ``int64 += 2.5``
    raises.  A logical op answers in bool whatever its operands meet at, and
    bool casts to every dtype, so its result is written as it comes.
    """

    def result_of(base: Tensor) -> Tensor:
        a, b = _binary_operands(base, other, opmath=opmath, inplace=promotes)
        return _wrap(op(a, b))

    return _write(self, name, result_of, other)


def _bitwise_and_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`bitwise_and`: ``self`` becomes ``self & other``.

    The operands meet at their common dtype; one ``self`` cannot hold
    (``int64`` with ``2.5``, ``bool`` with ``1``) raises ``RuntimeError``
    before anything is written.

    >>> import lucid
    >>> x = lucid.tensor([12, 10])
    >>> x.bitwise_and_(6)
    tensor([4, 2], dtype=lucid.int64)
    """
    return _binary_inplace(self, other, "bitwise_and_", _C_engine.bitwise_and)


def _bitwise_or_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`bitwise_or`: ``self`` becomes ``self | other``.

    Refused like ``bitwise_and_`` when the operands meet at a dtype
    ``self`` cannot hold.

    >>> import lucid
    >>> x = lucid.tensor([12, 10])
    >>> x.bitwise_or_(1)
    tensor([13, 11], dtype=lucid.int64)
    """
    return _binary_inplace(self, other, "bitwise_or_", _C_engine.bitwise_or)


def _bitwise_xor_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`bitwise_xor`: ``self`` becomes ``self ^ other``.

    Refused like ``bitwise_and_`` when the operands meet at a dtype
    ``self`` cannot hold.

    >>> import lucid
    >>> x = lucid.tensor([True, False])
    >>> x.bitwise_xor_(True)
    tensor([False, True], dtype=lucid.bool)
    """
    return _binary_inplace(self, other, "bitwise_xor_", _C_engine.bitwise_xor)


def _bitwise_left_shift_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`bitwise_left_shift`: ``self`` becomes ``self << other``.

    A float shift amount makes the result float, which an integer ``self``
    refuses before anything is written.

    >>> import lucid
    >>> x = lucid.tensor([1, 3])
    >>> x.bitwise_left_shift_(2)
    tensor([4, 12], dtype=lucid.int64)
    """
    return _binary_inplace(self, other, "bitwise_left_shift_", _C_engine.bitwise_left_shift)


def _bitwise_right_shift_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`bitwise_right_shift`: ``self`` becomes ``self >> other``.

    Refused like ``bitwise_left_shift_`` for a float shift amount.

    >>> import lucid
    >>> x = lucid.tensor([4, 12])
    >>> x.bitwise_right_shift_(2)
    tensor([1, 3], dtype=lucid.int64)
    """
    return _binary_inplace(self, other, "bitwise_right_shift_", _C_engine.bitwise_right_shift)


def _remainder_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`remainder`: ``self`` becomes ``self % other``.

    The result takes the divisor's sign, as ``%`` does.  An integer ``self``
    refuses a float divisor before anything is written.

    >>> import lucid
    >>> x = lucid.tensor([-7.0, 7.0])
    >>> x.remainder_(3)
    tensor([2., 1.])
    """
    return _binary_inplace(self, other, "remainder_", _C_engine.remainder)


def _fmod_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`fmod`: the C remainder, with the sign of the dividend.

    An integer ``self`` refuses a float divisor, as ``remainder_`` does.

    >>> import lucid
    >>> x = lucid.tensor([-7.0, 7.0])
    >>> x.fmod_(3)
    tensor([-1., 1.])
    """
    return _binary_inplace(self, other, "fmod_", _C_engine.fmod)


def _floor_divide_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`floor_divide`: ``self`` becomes ``self // other``.

    An integer ``self`` refuses a float divisor, as ``remainder_`` does.

    >>> import lucid
    >>> x = lucid.tensor([7.0, -7.0])
    >>> x.floor_divide_(2)
    tensor([3., -4.])
    """
    return _binary_inplace(self, other, "floor_divide_", _C_engine.floordiv, opmath=True)


def _logical_and_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`logical_and`: ``1`` where both are nonzero, ``0`` elsewhere.

    The answer is written at ``self``'s own dtype, which can hold any bool.

    >>> import lucid
    >>> x = lucid.tensor([0.0, 2.0, 3.0])
    >>> x.logical_and_(lucid.tensor([1.0, 1.0, 0.0]))
    tensor([0., 1., 0.])
    """
    return _binary_inplace(self, other, "logical_and_", _C_engine.logical_and, promotes=False)


def _logical_or_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`logical_or`: ``1`` where either is nonzero, ``0`` elsewhere.

    >>> import lucid
    >>> x = lucid.tensor([0, 0, 5])
    >>> x.logical_or_(lucid.tensor([0, 7, 0]))
    tensor([0, 1, 1], dtype=lucid.int64)
    """
    return _binary_inplace(self, other, "logical_or_", _C_engine.logical_or, promotes=False)


def _logical_xor_(self: Tensor, other: TensorOrScalar) -> Tensor:
    """In-place :meth:`logical_xor`: ``1`` where exactly one is nonzero.

    >>> import lucid
    >>> x = lucid.tensor([True, True, False])
    >>> x.logical_xor_(lucid.tensor([True, False, False]))
    tensor([False, True, False], dtype=lucid.bool)
    """
    return _binary_inplace(self, other, "logical_xor_", _C_engine.logical_xor, promotes=False)


def _logical_not_(self: Tensor) -> Tensor:
    """In-place :meth:`logical_not`: ``1`` where ``self`` was zero, else ``0``.

    >>> import lucid
    >>> x = lucid.tensor([0.0, 2.5])
    >>> x.logical_not_()
    tensor([1., 0.])
    """
    return _write(self, "logical_not_", lambda base: base.logical_not())


def _bitwise_not_(self: Tensor) -> Tensor:
    """In-place :meth:`bitwise_not`: ``~self``, for integer and bool tensors.

    >>> import lucid
    >>> x = lucid.tensor([0, 5])
    >>> x.bitwise_not_()
    tensor([-1, -6], dtype=lucid.int64)
    """
    return _write(self, "bitwise_not_", lambda base: base.bitwise_not())


def _lerp_(self: Tensor, end: Tensor, weight: float | Tensor) -> Tensor:
    """In-place :meth:`lerp`: ``self`` becomes ``self + weight * (end - self)``.

    A non-float ``self`` or ``end`` raises the ``TypeError`` :meth:`lerp`
    raises, before anything is written.

    >>> import lucid
    >>> x = lucid.tensor([0.0, 10.0])
    >>> x.lerp_(lucid.tensor([10.0, 20.0]), 0.5)
    tensor([5., 15.])
    """
    return _write(self, "lerp_", lambda base: base.lerp(end, weight), end, weight)


def _addcmul_(self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    """In-place :func:`lucid.addcmul`: ``self`` gains ``value * tensor1 * tensor2``.

    A float product into an integer ``self`` raises ``RuntimeError`` before
    anything is written.

    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0])
    >>> x.addcmul_(lucid.tensor([2.0, 3.0]), lucid.tensor([0.5, 0.5]), value=2)
    tensor([3., 5.])
    """
    from lucid._ops.composite.blas import addcmul

    def result_of(base: Tensor) -> Tensor:
        return addcmul(base, tensor1, tensor2, value=value)

    return _write(self, "addcmul_", result_of, tensor1, tensor2)


def _addcdiv_(self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    """In-place :func:`lucid.addcdiv`: ``self`` gains ``value * tensor1 / tensor2``.

    The quotient is floating point, so an integer ``self`` is refused, as
    ``addcmul_`` refuses a float product.

    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0])
    >>> x.addcdiv_(lucid.tensor([4.0, 9.0]), lucid.tensor([2.0, 3.0]), value=0.5)
    tensor([2., 3.5])
    """
    from lucid._ops.composite.blas import addcdiv

    def result_of(base: Tensor) -> Tensor:
        return addcdiv(base, tensor1, tensor2, value=value)

    return _write(self, "addcdiv_", result_of, tensor1, tensor2)


def _clamp_inplace(
    self: Tensor, min: TensorOrScalar | None, max: TensorOrScalar | None, name: str
) -> Tensor:
    """``self`` clamped to ``[min, max]`` in place, by :meth:`Tensor.clamp`'s rules.

    The bounds promote ``self``'s dtype as they do out of place
    (:func:`lucid._ops._adapters._clip_dtype`), and a dtype ``self`` cannot
    hold is refused before anything runs, with the error ``add_`` gives.
    Number bounds that leave the dtype alone take the engine's ``clip_``
    where ``clamp`` takes its ``clip``; anything else is ``clamp``'s result,
    written through :func:`_write`.  One helper for ``clamp_``, ``clip_``,
    ``clamp_min_`` and ``clamp_max_``, so each public spelling is one
    in-place write and names itself in its errors.
    """
    impl = self._impl
    dt = _clip_dtype(impl, min, max)
    if dt != impl.dtype:
        if not _promotion.can_cast(dt, impl.dtype):
            raise _cannot_cast_inplace(dt, impl.dtype)
    else:
        is_float = dt in _REAL_FLOATS
        kernel = _clip_scalar_bounds(impl, min, max, is_float, _C_engine.clip_)
        if kernel is not None:
            return _adopt_inplace(self, kernel, name)

    def result_of(base: Tensor) -> Tensor:
        return _wrap(_clip_adapter(base._impl, min, max))

    return _write(self, name, result_of, min, max)


def _clip_(
    self: Tensor, min: Tensor | float | None = None, max: Tensor | float | None = None
) -> Tensor:
    """In-place :meth:`clip`, another name for :meth:`Tensor.clamp_`.

    >>> import lucid
    >>> x = lucid.tensor([-2.0, 0.5, 3.0])
    >>> x.clip_(-1.0, 2.0)
    tensor([-1., 0.5, 2.])
    """
    return _clamp_inplace(self, min, max, "clip_")


# ── table-driven in-place twins (LCD-525) ───────────────────────────────────

_UNARY_INPLACE_TWINS: tuple[str, ...] = (
    "absolute_",
    "acos_",
    "acosh_",
    "arccosh_",
    "arcsinh_",
    "arctanh_",
    "asin_",
    "asinh_",
    "atan_",
    "atanh_",
    "conj_physical_",
    "digamma_",
    "erf_",
    "erfc_",
    "erfinv_",
    "exp2_",
    "expm1_",
    "frac_",
    "i0_",
    "lgamma_",
    "log10_",
    "log1p_",
    "negative_",
    "rsqrt_",
    "sinc_",
    "trunc_",
)

_BINARY_INPLACE_TWINS: tuple[str, ...] = (
    "arctan2_",
    "atan2_",
    "copysign_",
    "eq_",
    "gcd_",
    "ge_",
    "greater_",
    "greater_equal_",
    "gt_",
    "hypot_",
    "lcm_",
    "ldexp_",
    "le_",
    "less_",
    "less_equal_",
    "lt_",
    "multiply_",
    "ne_",
    "nextafter_",
    "not_equal_",
    "true_divide_",
    "xlogy_",
)

#: Which of ``_BINARY_INPLACE_TWINS`` refuse a bare Python number -- their
#: free function (``lucid.gcd`` / ``lucid.lcm``) has no engine promotion
#: path for a scalar operand and raises ``TypeError`` itself (LCD-644).
#: Every other twin forwards to a free function that promotes a scalar the
#: way ``+`` does (``lucid._dispatch._binary_operands``), so the stub's
#: default ``other: TensorOrScalar`` already matches what it accepts.
#: ``tools/gen_pyi.py`` reads this set to stub each twin by what it truly
#: takes, rather than one blanket annotation for all 22 (see
#: ``lucid/test/unit/tensor/test_inplace_entry_points.py`` for the guard
#: that a scalar call matches the stub either way).
_BINARY_INPLACE_TENSOR_ONLY: frozenset[str] = frozenset({"gcd_", "lcm_"})

_SCAN_INPLACE_TWINS: tuple[str, ...] = ("cumsum_", "cumprod_")
_TRI_INPLACE_TWINS: tuple[str, ...] = ("tril_", "triu_")
_BMM_INPLACE_TWINS: tuple[str, ...] = ("addbmm_", "baddbmm_")


def _make_unary_inplace(name: str) -> Callable[[Tensor], Tensor]:
    op_name = name[:-1]

    def method(self: Tensor) -> Tensor:
        """Write the unary op result into ``self`` in place."""
        import lucid

        fn = getattr(lucid, op_name)
        return _write(self, name, fn)

    method.__name__ = name
    return method


def _make_binary_inplace(name: str) -> Callable[[Tensor, TensorOrScalar], Tensor]:
    op_name = name[:-1]

    def method(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Write the binary op result into ``self`` in place."""
        import lucid

        fn = getattr(lucid, op_name)
        return _write(self, name, lambda base: fn(base, other), other)

    method.__name__ = name
    return method


def _make_scan_inplace(name: str) -> Callable[..., Tensor]:
    op_name = name[:-1]

    def method(self: Tensor, dim: int = -1, *, dtype: DTypeLike | None = None) -> Tensor:
        """Write the cumulative scan along ``dim`` into ``self`` in place."""
        import lucid
        from lucid._dtype import to_engine_dtype

        if self.dtype == lucid.bool:
            raise _C_engine.NotImplementedError(f"{name} is not implemented for {self.dtype}")
        if dtype is not None:
            target = to_engine_dtype(dtype)
            if target != self._impl.dtype:
                raise RuntimeError(
                    f"Bad in-place call: input tensor dtype {self.dtype} and "
                    f"output tensor dtype {dtype} should match"
                )
        fn = getattr(lucid, op_name)
        return _write(self, name, lambda base: fn(base, dim, dtype=base.dtype))

    method.__name__ = name
    return method


def _make_tri_inplace(name: str) -> Callable[..., Tensor]:
    op_name = name[:-1]

    def method(self: Tensor, diagonal: int = 0) -> Tensor:
        """Write the triangular mask result into ``self`` in place."""
        import lucid

        fn = getattr(lucid, op_name)
        return _write(self, name, lambda base: fn(base, diagonal=diagonal))

    method.__name__ = name
    return method


def _make_bmm_inplace(name: str) -> Callable[..., Tensor]:
    op_name = name[:-1]

    def method(
        self: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
    ) -> Tensor:
        """Write the batched matrix-multiply accumulate into ``self`` in place."""
        import lucid

        fn = getattr(lucid, op_name)
        return _write(
            self,
            name,
            lambda base: fn(base, batch1, batch2, beta=beta, alpha=alpha),
            batch1,
            batch2,
        )

    method.__name__ = name
    return method


def _float_power_(self: Tensor, exponent: TensorOrScalar) -> Tensor:
    """Write ``lucid.float_power(self, exponent)`` into ``self`` in place."""
    import lucid

    if self._impl.dtype not in (_C_engine.Dtype.F64, _C_engine.Dtype.C128):
        raise RuntimeError(
            f"the base given to float_power_ has dtype {self.dtype} but the "
            "operation's result requires dtype lucid.float64"
        )
    return _write(self, "float_power_", lambda base: lucid.float_power(base, exponent), exponent)


def _heaviside_(self: Tensor, values: TensorOrScalar) -> Tensor:
    """Write ``lucid.heaviside(self, values)`` into ``self`` in place."""
    import lucid

    return _write(self, "heaviside_", lambda base: lucid.heaviside(base, values), values)


def _divide(self: Tensor, other: TensorOrScalar, *, rounding_mode: str | None = None) -> Tensor:
    """Element-wise division — :meth:`Tensor.div` under its longer name."""
    import lucid

    return lucid.divide(self, other, rounding_mode=rounding_mode)


def _true_divide(self: Tensor, other: TensorOrScalar) -> Tensor:
    """Element-wise true (floating-point) division."""
    import lucid

    return lucid.true_divide(self, other)


def _divide_(self: Tensor, other: TensorOrScalar, *, rounding_mode: str | None = None) -> Tensor:
    """Write ``lucid.divide(self, other, rounding_mode=rounding_mode)`` in place."""
    import lucid

    return _write(
        self, "divide_", lambda base: lucid.divide(base, other, rounding_mode=rounding_mode), other
    )


def _subtract_(self: Tensor, other: TensorOrScalar, *, alpha: float = 1.0) -> Tensor:
    """Write ``lucid.subtract(self, other, alpha=alpha)`` into ``self`` in place."""
    import lucid

    return _write(self, "subtract_", lambda base: lucid.subtract(base, other, alpha=alpha), other)


def _logit_(self: Tensor, eps: float | None = None) -> Tensor:
    """Write ``lucid.logit(self, eps=eps)`` into ``self`` in place."""
    import lucid

    return _write(self, "logit_", lambda base: lucid.logit(base, eps=eps))


def _nan_to_num_(
    self: Tensor, nan: float | None = None, posinf: float | None = None, neginf: float | None = None
) -> Tensor:
    """Write ``lucid.nan_to_num(self, nan, posinf, neginf)`` into ``self`` in place."""
    import lucid

    return _write(
        self,
        "nan_to_num_",
        lambda base: lucid.nan_to_num(base, nan=nan, posinf=posinf, neginf=neginf),
    )


def _renorm_(self: Tensor, p: float, dim: int, maxnorm: float) -> Tensor:
    """Write ``lucid.renorm(self, p, dim, maxnorm)`` into ``self`` in place."""
    import lucid

    return _write(self, "renorm_", lambda base: lucid.renorm(base, p=p, dim=dim, maxnorm=maxnorm))


def _addmm_(
    self: Tensor, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    """Write ``lucid.addmm(self, mat1, mat2, beta=beta, alpha=alpha)`` in place."""
    import lucid

    return _write(
        self,
        "addmm_",
        lambda base: lucid.addmm(base, mat1, mat2, beta=beta, alpha=alpha),
        mat1,
        mat2,
    )


def _addmv_(
    self: Tensor, mat: Tensor, vec: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    """Write ``lucid.addmv(self, mat, vec, beta=beta, alpha=alpha)`` in place."""
    import lucid

    return _write(
        self, "addmv_", lambda base: lucid.addmv(base, mat, vec, beta=beta, alpha=alpha), mat, vec
    )


def _addr_(
    self: Tensor, vec1: Tensor, vec2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    """Write ``lucid.addr(self, vec1, vec2, beta=beta, alpha=alpha)`` in place."""
    import lucid

    return _write(
        self, "addr_", lambda base: lucid.addr(base, vec1, vec2, beta=beta, alpha=alpha), vec1, vec2
    )


def _bernoulli(
    self: Tensor, p: float | None = None, *, generator: _C_engine.Generator | None = None
) -> Tensor:
    """Draw binary random numbers (0 or 1) from a Bernoulli distribution."""
    import lucid

    return lucid.bernoulli(self, p, generator=generator)


def _fill_drawn(self: Tensor, name: str, draw: Callable[[Tensor], Tensor]) -> Tensor:
    """Make ``draw(self)`` — random values shaped like ``self`` — ``self``'s values.

    The values do not depend on what ``self`` held: a tensor that requires
    grad keeps requiring it, as after the reference's in-place draws, and
    passes no gradient back through them.  ``self``'s old values are never
    read — ``self * 0 + out`` made an infinite or NaN entry NaN.  Under
    :func:`lucid.no_grad` (``nn.init``) the draw is written as it comes.
    """
    import lucid

    def result_of(base: Tensor) -> Tensor:
        out = draw(base)
        if not (base.requires_grad and _C_engine.grad_enabled()):
            return out
        keep_none = lucid.zeros(base.shape, dtype=lucid.bool, device=base.device)
        return lucid.where(keep_none, base, out)

    return _write(self, name, result_of)


def _bernoulli_(
    self: Tensor, p: Tensor | float = 0.5, *, generator: _C_engine.Generator | None = None
) -> Tensor:
    """Fill ``self`` with independent Bernoulli draws from ``p``, in place."""
    import lucid

    def _draw(base: Tensor) -> Tensor:
        if isinstance(p, lucid.Tensor):
            return lucid.bernoulli(p.expand(base.shape), generator=generator).to(base.dtype)
        return lucid.bernoulli(base, p, generator=generator)

    return _fill_drawn(self, "bernoulli_", _draw)


def _normal_(
    self: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Fill ``self`` with :math:`\mathcal{N}(\text{mean}, \text{std}^2)` samples.

    Parameters
    ----------
    mean : float, optional
        Mean of the distribution.  Default ``0.0``.
    std : float, optional
        Standard deviation; a negative one is refused
        (``lucid._C.engine.InvalidArgument``).  Default ``1.0``.
    generator : lucid._C.engine.Generator, optional
        The stream drawn from — when ``None``, the default stream
        :func:`lucid.manual_seed` controls.

    Notes
    -----
    A complex ``self`` draws each part with standard deviation
    :math:`\text{std}/\sqrt{2}` around ``mean``.  An integer or bool one
    has no such values (``NotImplementedError``); a leaf that requires
    grad is refused while autograd records, as every in-place op is.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> w = lucid.empty(1000).normal_(2.0, 0.5)
    >>> abs(float(w.mean()) - 2.0) < 0.1
    True
    """
    from lucid._factories.random import _normal_like

    return _fill_drawn(self, "normal_", lambda base: _normal_like(base, mean, std, generator))


def _uniform_(
    self: Tensor,
    from_: float = 0.0,
    /,
    to: float = 1.0,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Fill ``self`` with :math:`U[\text{from}, \text{to})` samples.

    The reference names the lower bound ``from``, a Python keyword no call
    can spell, so it goes by position — ``x.uniform_(-1, 1)`` — and ``to``
    by either.  A complex ``self`` draws both parts from the interval; an
    integer or bool one is refused, and so is a leaf that requires grad
    while autograd records.

    Parameters
    ----------
    from_ : float, optional
        Lower bound (inclusive), positional only.  Default ``0.0``.
    to : float, optional
        Upper bound (exclusive).  Equal bounds fill with ``from_``; a lower
        one is refused (``lucid._C.engine.InvalidArgument``).  Default
        ``1.0``.
    generator : lucid._C.engine.Generator, optional
        Which stream to read; ``None`` reads the default one.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> w = lucid.empty(2, 3).uniform_(-1.0, 1.0)
    >>> bool(((w >= -1.0) & (w < 1.0)).all())
    True
    """
    from lucid._factories.random import _uniform_like

    return _fill_drawn(self, "uniform_", lambda base: _uniform_like(base, from_, to, generator))


_GENERATED_INPLACE_TWINS: tuple[tuple[str, Callable[..., Tensor]], ...] = (
    *((n, _make_unary_inplace(n)) for n in _UNARY_INPLACE_TWINS),
    *((n, _make_binary_inplace(n)) for n in _BINARY_INPLACE_TWINS),
    *((n, _make_scan_inplace(n)) for n in _SCAN_INPLACE_TWINS),
    *((n, _make_tri_inplace(n)) for n in _TRI_INPLACE_TWINS),
    *((n, _make_bmm_inplace(n)) for n in _BMM_INPLACE_TWINS),
    ("float_power_", _float_power_),
    ("heaviside_", _heaviside_),
    ("divide_", _divide_),
    ("subtract_", _subtract_),
    ("logit_", _logit_),
    ("nan_to_num_", _nan_to_num_),
    ("renorm_", _renorm_),
    ("addmm_", _addmm_),
    ("addmv_", _addmv_),
    ("addr_", _addr_),
    ("bernoulli_", _bernoulli_),
    ("normal_", _normal_),
    ("uniform_", _uniform_),
)


# ── augmented assignment, through the methods above ─────────────────────────
# Each operator is its in-place method, called by attribute (``x &= y`` is
# ``x.bitwise_and_(y)``), so the two refuse and write alike.  A write through
# a view reaches its base on both devices in the engine's view family
# (``ViewFamily``, lucid/_C/core/TensorImpl.h); nothing here carries it back.


def __iand__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self &= other`` — bitwise / logical AND, written into ``self`` by ``bitwise_and_``.

    >>> import lucid
    >>> x = lucid.tensor([True, True])
    >>> x &= lucid.tensor([True, False])
    >>> x
    tensor([True, False], dtype=lucid.bool)
    """
    return self.bitwise_and_(other)


def __ior__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self |= other`` — bitwise / logical OR, written into ``self`` by ``bitwise_or_``.

    >>> import lucid
    >>> x = lucid.tensor([True, False])
    >>> x |= lucid.tensor([False, True])
    >>> x
    tensor([True, True], dtype=lucid.bool)
    """
    return self.bitwise_or_(other)


def __ixor__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self ^= other`` — bitwise / logical XOR, written into ``self`` by ``bitwise_xor_``.

    >>> import lucid
    >>> x = lucid.tensor([5, 6])
    >>> x ^= 3
    >>> x
    tensor([6, 5], dtype=lucid.int64)
    """
    return self.bitwise_xor_(other)


def __ilshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self <<= other`` — left shift, written into ``self`` by ``bitwise_left_shift_``.

    >>> import lucid
    >>> x = lucid.tensor([1, 2])
    >>> x <<= 3
    >>> x
    tensor([8, 16], dtype=lucid.int64)
    """
    return self.bitwise_left_shift_(other)


def __irshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self >>= other`` — right shift, written into ``self`` by ``bitwise_right_shift_``.

    >>> import lucid
    >>> x = lucid.tensor([8, 16])
    >>> x >>= 3
    >>> x
    tensor([1, 2], dtype=lucid.int64)
    """
    return self.bitwise_right_shift_(other)


def __imod__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self %= other`` — remainder with the divisor's sign, written by ``remainder_``.

    >>> import lucid
    >>> x = lucid.tensor([-7.0, 7.0])
    >>> x %= 3
    >>> x
    tensor([2., 1.])
    """
    return self.remainder_(other)


def __ifloordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    """``self //= other`` — floor division, written into ``self`` by ``floor_divide_``.

    >>> import lucid
    >>> x = lucid.tensor([7.0, 8.0])
    >>> x //= 2
    >>> x
    tensor([3., 4.])
    """
    return self.floor_divide_(other)


_EXTRA_METHODS = (
    ("masked_fill_", _masked_fill_),
    ("scatter_", _scatter_),
    ("scatter_add_", _scatter_add_),
    ("scatter_reduce", _scatter_reduce),
    ("scatter_reduce_", _scatter_reduce_),
    ("put_", _put_),
    ("masked_scatter", _masked_scatter),
    ("masked_scatter_", _masked_scatter_),
    ("bincount", _bincount),
    ("take_along_dim", _take_along_dim),
    ("median", _median),
    ("nanmedian", _nanmedian),
    ("mode", _mode),
    ("quantile", _quantile),
    ("nanquantile", _nanquantile),
    ("msort", _msort),
    ("select", _select),
    ("tensor_split", _tensor_split),
    ("vsplit", _vsplit),
    ("hsplit", _hsplit),
    ("dsplit", _dsplit),
    ("aminmax", _aminmax),
    ("renorm", _renorm),
    ("logcumsumexp", _logcumsumexp),
    ("addcmul", _addcmul),
    ("addcdiv", _addcdiv),
    ("is_set_to", _is_set_to),
    ("bernoulli", _bernoulli),
    ("divide", _divide),
    ("true_divide", _true_divide),
    ("bitwise_and_", _bitwise_and_),
    ("bitwise_or_", _bitwise_or_),
    ("bitwise_xor_", _bitwise_xor_),
    ("bitwise_left_shift_", _bitwise_left_shift_),
    ("bitwise_right_shift_", _bitwise_right_shift_),
    ("bitwise_not_", _bitwise_not_),
    ("remainder_", _remainder_),
    ("fmod_", _fmod_),
    ("floor_divide_", _floor_divide_),
    ("logical_and_", _logical_and_),
    ("logical_or_", _logical_or_),
    ("logical_xor_", _logical_xor_),
    ("logical_not_", _logical_not_),
    ("lerp_", _lerp_),
    ("addcmul_", _addcmul_),
    ("addcdiv_", _addcdiv_),
    ("clip_", _clip_),
    *_GENERATED_INPLACE_TWINS,
    ("__iand__", __iand__),
    ("__ior__", __ior__),
    ("__ixor__", __ixor__),
    ("__ilshift__", __ilshift__),
    ("__irshift__", __irshift__),
    ("__imod__", __imod__),
    ("__ifloordiv__", __ifloordiv__),
)


def _make_method(e: OpEntry, tensor_cls: type) -> object:
    """Build a Tensor method that dispatches to ``e.engine_fn``."""
    if e.n_tensor_args == -1:

        def method_list(
            self: Tensor, tensors: list[Tensor], *args: object, **kwargs: object
        ) -> Tensor:
            """Variadic-tensor method body — prepends ``self`` to the unwrapped tensor list."""
            all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
            result = e.engine_fn(all_tensors, *args, **kwargs)
            if e.returns_tensor:
                if isinstance(result, (list, tuple)):
                    return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
                return _wrap(result)
            return cast("Tensor", result)

        method_list.__name__ = e.method_name or e.name
        return method_list
    return _fixed_arity_method(e, tensor_cls)


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""
    globals().setdefault("Tensor", tensor_cls)

    for entry in _REGISTRY:
        if entry.method_name is None or entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry, tensor_cls))

    def eval(self: Tensor) -> Tensor:
        """Force immediate evaluation of this tensor."""
        self._impl.eval()
        return self

    for _name, _fn in (("eval", eval), *_EXTRA_METHODS):
        _fn.__name__ = _name
        setattr(tensor_cls, _name, _fn)
    type.__setattr__(tensor_cls, "_version", property(lambda self: self._impl.version))
