from collections.abc import Sequence
from typing import TYPE_CHECKING
import weakref

from lucid._C import engine as _C_engine
from lucid import _promotion
from lucid._dtype import dtype, to_engine_dtype, _ENGINE_TO_DTYPE
from lucid._device import device, _device_from_engine  # noqa: F401
from lucid._globals import get_default_dtype, get_default_device
from lucid._tree_version import _tree_changed

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# 3.2.2: cache the resolved default-dtype/device engine enums so that
# ``normalize_factory_kwargs`` — called once per op invocation (~200–300
# ops per ResNet-18 forward, ~65 k calls per LeNet-5/MNIST epoch) —
# doesn't re-do the lookup + dtype/device parsing on every call.
# Invalidated by ``lucid._globals.set_default_dtype/device``.
_CACHED_DEFAULT_DTYPE_ENUM: _C_engine.Dtype | None = None
_CACHED_DEFAULT_DEVICE_ENUM: _C_engine.Device | None = None


def _default_dtype_enum_cached() -> _C_engine.Dtype:
    global _CACHED_DEFAULT_DTYPE_ENUM
    if _CACHED_DEFAULT_DTYPE_ENUM is None:
        _CACHED_DEFAULT_DTYPE_ENUM = to_engine_dtype(get_default_dtype())
    return _CACHED_DEFAULT_DTYPE_ENUM


def _default_device_enum_cached() -> _C_engine.Device:
    global _CACHED_DEFAULT_DEVICE_ENUM
    if _CACHED_DEFAULT_DEVICE_ENUM is None:
        _CACHED_DEFAULT_DEVICE_ENUM = _parse_device(get_default_device())
    return _CACHED_DEFAULT_DEVICE_ENUM


#: Whether the default floating dtype is float32, the dtype the engine's
#: unary kernels promote a bool or integer operand to on their own.  Kept by
#: :func:`_invalidate_default_dtype_cache`; read by the float-valued unary
#: ops' fast path (see :func:`_float_operand`).
_DEFAULT_FLOAT_IS_F32: bool = True


def _invalidate_default_dtype_cache() -> None:
    """Called by ``lucid._globals.set_default_dtype`` after a change."""
    global _CACHED_DEFAULT_DTYPE_ENUM, _DEFAULT_FLOAT_IS_F32
    _CACHED_DEFAULT_DTYPE_ENUM = None
    _DEFAULT_FLOAT_IS_F32 = to_engine_dtype(get_default_dtype()) == _C_engine.Dtype.F32


def _invalidate_default_device_cache() -> None:
    """Called by ``lucid._globals.set_default_device`` after a change.

    Note: ``lucid._factories.converters`` keeps a *separate* device cache
    for the ndarray-fast-path hot loop; ``set_default_device`` invalidates
    both via the hook installed in ``lucid._globals``.
    """
    global _CACHED_DEFAULT_DEVICE_ENUM
    _CACHED_DEFAULT_DEVICE_ENUM = None


def _unwrap(t: _C_engine.TensorImpl | Tensor) -> _C_engine.TensorImpl:
    """Return the underlying TensorImpl from a Tensor or TensorImpl."""
    impl = _promotion.try_unwrap(t)
    if impl is not None:
        return impl
    raise TypeError(f"Expected Tensor or TensorImpl, got {type(t).__name__}")


# Dtypes ``pow_scalar`` computes in: the real floating ones.
_REAL_FLOATS: frozenset[_C_engine.Dtype] = frozenset(
    {_C_engine.Dtype.F16, _C_engine.Dtype.BF16, _C_engine.Dtype.F32, _C_engine.Dtype.F64}
)


def _scalar_power(base: _C_engine.TensorImpl, exponent: object) -> _C_engine.TensorImpl | None:
    """``base ** exponent`` through ``pow_scalar`` when that is the same op.

    A Python number as the exponent of a real floating tensor is a constant
    of the base's dtype, so the general tensor power only made a 0-d tensor
    of it and broadcast it.  ``pow_scalar`` carries the number on the node
    instead, and its kernels answer the common exponents (2, 1, 0, -1, 0.5)
    with the single correctly rounded operation each one is — ``x ** 2``
    on the CPU was 180 times ``x * x``.

    Returns ``None`` for anything else — a tensor, a bool, an integer or
    complex base — which keeps the general path and its promotion rules.
    When the computation stays in a half dtype (outside Metal autocast),
    ``PowScalarBackward::forward`` rounds the exponent to that half dtype.
    """
    exponent = _promotion.as_number(exponent)
    _promotion.check_pow_scalar(base.dtype, exponent)
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return None
    if base.dtype not in _REAL_FLOATS:
        return None
    return _C_engine.pow_scalar(base, float(exponent))


def _is_bool_operand(x: object) -> bool:
    if isinstance(_promotion.as_number(x), bool):
        return True
    impl = _promotion.try_unwrap(x)
    return impl is not None and impl.dtype == _C_engine.Bool


def _refuse_bool_subtraction(a: object, b: object) -> None:
    """Refuse subtraction involving a ``bool`` or ``Bool`` tensor, as the reference does."""
    a_bool = _is_bool_operand(a)
    b_bool = _is_bool_operand(b)
    if a_bool and b_bool:
        raise _C_engine.LucidError(
            "subtraction, the `-` operator, with two bool tensors is not "
            "supported; use `^` or logical_xor() instead"
        )
    if a_bool or b_bool:
        raise _C_engine.LucidError(
            "subtraction, the `-` operator, with a bool tensor is not "
            "supported; if you are trying to invert a mask, use the `~` "
            "or logical_not() operator instead"
        )


def _integer_impl(x: int, dt: _C_engine.Dtype, device: _C_engine.Device) -> _C_engine.TensorImpl:
    """A 0-d integer impl holding ``x`` exactly, wrapped to ``dt``'s width.

    ``full`` takes its fill value as a C double, so an integer past 2**53
    lost its low bits on the way in (``int64 + (2**60 + 1)`` was off by
    one), and one outside ``dt``'s range met an undefined double-to-integer
    conversion.  Those go in as bytes instead, wrapped as an integer cast
    wraps — what the reference answers for ``int8_tensor + 300``.  The
    rest, nearly every scalar, still take ``full``: it builds the value on
    the device, where bytes are a host buffer the GPU stream must copy.
    """
    bits = _promotion.bit_width(dt)
    if -(1 << (bits - 1)) <= x < (1 << (bits - 1)) and -(1 << 53) <= x <= (1 << 53):
        return _C_engine.full([], float(x), dt, device)
    if not -(1 << 63) <= x < (1 << 64):
        raise OverflowError(
            f"Python integer {x} does not fit in 64 bits, the widest integer "
            "a tensor operand can hold"
        )
    raw = (x & ((1 << bits) - 1)).to_bytes(bits // 8, "little")
    return _C_engine.TensorImpl.from_bytes(raw, [], dt, device, False)


def _scalar_impl(
    x: bool | int | float | complex, dt: _C_engine.Dtype, device: _C_engine.Device
) -> _C_engine.TensorImpl:
    """``x`` as a 0-d impl of ``dt`` on ``device`` — where a scalar crosses.

    A complex number is assembled from its parts through the engine's
    ``complex`` op, and an integer bound for an integer dtype goes in
    exactly; everything else is a ``full`` of the float value.
    """
    if isinstance(x, complex):
        part = _C_engine.Dtype.F64 if dt == _C_engine.Dtype.C128 else _C_engine.Dtype.F32
        out = _C_engine.complex(
            _C_engine.full([], x.real, part, device), _C_engine.full([], x.imag, part, device)
        )
        return out if out.dtype == dt else _C_engine.astype(out, dt)
    if not isinstance(x, bool) and isinstance(x, int) and _promotion.is_integer(dt):
        return _integer_impl(x, dt, device)
    return _C_engine.full([], float(x), dt, device)


def _unwrap_or_scalar(
    x: object, ref_impl: _C_engine.TensorImpl | None = None
) -> _C_engine.TensorImpl:
    """Return the TensorImpl for a Tensor, or build one from a Python scalar.

    Parameters
    ----------
    x : Tensor or TensorImpl or bool or int or float or complex
        Operand to normalise.
    ref_impl : TensorImpl or None, default=None
        Tensor the scalar is being combined with; supplies the device, and
        the dtype the pair promotes to (see :mod:`lucid._promotion`).
        Without one the scalar takes its own dtype on the CPU.

    Returns
    -------
    TensorImpl
        ``x`` itself when it already is one, else a 0-d constant.

    Raises
    ------
    TypeError
        If ``x`` is neither a tensor nor a Python number.

    Notes
    -----
    The scalar becomes a **0-d** constant rather than one materialised to
    ``ref_impl``'s full shape.  The engine's binary ops — arithmetic and
    comparison alike — broadcast it, and staying 0-d is what lets scalar
    arithmetic ride the symbolic-batch compile path: a full pinned to the
    trace-time batch would mismatch the symbolic input and abort MPSGraph's
    MLIR pass.  It also skips allocating a full-shape buffer in eager.

    A scalar is *weak*: it can lift the result's category (bool < int <
    float < complex) but never its width, so ``float16_tensor + 1.5`` stays
    float16 while ``int32_tensor + 1.5`` becomes floating.  The binary ops
    reach the same rule through :func:`_binary_operands`, which builds its
    scalar operands with the same :func:`_scalar_impl`.
    """
    impl = _promotion.try_unwrap(x)
    if impl is not None:
        return impl

    x = _promotion.as_number(x)
    if isinstance(x, _promotion.NUMBER):
        if ref_impl is None:
            return _scalar_impl(x, _promotion.result_type(x), _C_engine.Device.CPU)
        return _scalar_impl(x, _promotion.result_type(ref_impl, x), ref_impl.device)

    raise TypeError(f"Cannot convert {type(x).__name__} to TensorImpl")


def _is_opmath_operand(impl: _C_engine.TensorImpl, peer_device: _C_engine.Device) -> bool:
    """Whether a tensor operand may reach a half op-math op uncast.

    The engine's mul/div/floordiv read a 0-d float32 (float64 on the CPU)
    operand at its own width beside a half tensor on the same device (or a
    0-d CPU scalar beside a Metal half tensor), including when the 0-d
    operand requires grad (LCD-300).
    """
    if impl.shape:
        return False
    if impl.device != peer_device and impl.device != _C_engine.Device.CPU:
        return False
    dt = impl.dtype
    return dt == _C_engine.Dtype.F32 or (
        dt == _C_engine.Dtype.F64 and impl.device == _C_engine.Device.CPU
    )


_IMPL = _C_engine.TensorImpl


def _float_operand(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """``impl`` at the dtype a float-valued op computes in.

    The one rule for an op whose answer is floating whatever its input —
    ``exp``, ``sin``, ``sqrt``, ``sigmoid``, ``erf``, ``lgamma`` and the
    rest: a bool or integer operand is cast to the default floating dtype
    (narrowed to what its device holds, :func:`lucid._promotion.default_float`),
    as true division does; a floating or complex one comes back as it is.
    So ``exp2(int64)`` is ``exp2`` of the float, not of an integer the
    composite multiplied by a truncated ``log 2``, and the answer follows
    :func:`lucid.set_default_dtype` as the reference's does.

    The registry's float-valued unary entries (``_registry._float_valued``)
    and the Python composites (:func:`_float_input`) both cast through here.
    The cast comes first and any autocast policy after it: an integer input
    is made the default float, which autocast may then narrow, never the
    reverse.
    """
    if type(impl) is _IMPL and _promotion.is_integral(impl.dtype):
        return _C_engine.astype(impl, _promotion.default_float(impl.device))
    return impl


def _float_input(x: Tensor) -> Tensor:
    """``x`` at the dtype a float-valued op computes in — the Tensor form of
    :func:`_float_operand`, for the ops written in Python.

    Returns ``x`` itself when it needs no cast, so the common floating call
    allocates nothing; anything without an engine tensor is returned as is.
    """
    impl = _promotion.try_unwrap(x)
    if impl is None:
        return x
    out = _float_operand(impl)
    return x if out is impl else _wrap(out)


def _tensor_operand(x: object) -> _C_engine.TensorImpl | None:
    """``x``'s engine tensor, or None when ``x`` is a Python number."""
    if isinstance(x, _promotion.NUMBER):
        return None
    return _unwrap_or_scalar(x)


def _cast(
    impl: _C_engine.TensorImpl,
    current: _C_engine.Dtype,
    dt: _C_engine.Dtype,
    opmath: bool,
    peer_device: _C_engine.Device,
) -> _C_engine.TensorImpl:
    """``impl`` (of dtype ``current``) at ``dt`` — or uncast, for op-math."""
    if current == dt or (opmath and _is_opmath_operand(impl, peer_device)):
        return impl
    return _C_engine.astype(impl, dt)


def _cannot_cast_inplace(dt: _C_engine.Dtype, target: _C_engine.Dtype) -> RuntimeError:
    return RuntimeError(
        f"result type {_ENGINE_TO_DTYPE[dt]} can't be cast to the desired "
        f"output type {_ENGINE_TO_DTYPE[target]}"
    )


def _beside_number(
    t: _C_engine.TensorImpl,
    x: bool | int | float | complex,
    true_div: bool,
    opmath: bool,
    inplace: bool = False,
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """A tensor and a Python number, as ``(tensor operand, number operand)``.

    The common binary case gets its own path: the dtype is the tensor's own
    unless the number lifts its category, so in the usual ``x * 0.5`` the
    tensor is passed through untouched and only the 0-d constant is built.
    """
    d = t.dtype
    dt = _promotion.beside_number(t, d, x)
    device = t.device
    if true_div and _promotion.is_integral(dt):
        dt = _promotion.default_float(device)
    if inplace and not _promotion.can_cast(dt, d):
        raise _cannot_cast_inplace(dt, d)
    if opmath and _promotion.is_half(dt) and not isinstance(x, (bool, complex)):
        number = _scalar_impl(x, _C_engine.Dtype.F32, device)
    else:
        number = _scalar_impl(x, dt, device)
    return (t if dt is d else _C_engine.astype(t, dt)), number


def _tensor_pair(
    ai: _C_engine.TensorImpl,
    da: _C_engine.Dtype,
    bi: _C_engine.TensorImpl,
    db: _C_engine.Dtype,
    true_div: bool,
    opmath: bool,
    inplace: bool = False,
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """Two tensors of different dtypes, each brought to the pair's dtype."""
    dt = _promotion.tensor_pair_type(ai, da, bi, db)
    if true_div and _promotion.is_integral(dt):
        dt = _promotion.default_float(ai.device)
    if inplace and not _promotion.can_cast(dt, da):
        raise _cannot_cast_inplace(dt, da)
    wide = opmath and _promotion.is_half(dt)
    return (_cast(ai, da, dt, wide, bi.device), _cast(bi, db, dt, wide, ai.device))


def _number_pair(
    a: object, b: object, true_div: bool
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """Two Python numbers, as 0-d tensors on the CPU (``lucid.add(2, 3)``)."""
    if not (isinstance(a, _promotion.NUMBER) and isinstance(b, _promotion.NUMBER)):
        # _tensor_operand refuses anything else before this point.
        raise TypeError(f"cannot combine {type(a).__name__} and {type(b).__name__}")
    cpu = _C_engine.Device.CPU
    dt = _promotion.result_type(a, b)
    if true_div and _promotion.is_integral(dt):
        dt = _promotion.default_float(cpu)
    return _scalar_impl(a, dt, cpu), _scalar_impl(b, dt, cpu)


def _binary_operands(
    a: object, b: object, *, true_div: bool = False, opmath: bool = False, inplace: bool = False
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """``a`` and ``b`` as engine operands at the dtype their op computes in.

    The single promotion path behind every spelling of a binary op — the
    operator, its reflected and in-place forms, ``Tensor.<op>`` and
    ``lucid.<op>``.  The dtype comes from :mod:`lucid._promotion`; each
    tensor is cast to it and each Python number built at it.

    Parameters
    ----------
    a, b : Tensor or TensorImpl or bool or int or float or complex
        The operands, in the op's order.
    true_div : bool, default=False
        True division: an integral or bool result becomes the default
        floating dtype.
    opmath : bool, default=False
        The op is mul, div or floordiv, whose engine kernels compute a half
        tensor against a 0-d float32 operand in float32 and round once.  When
        the result is half, a real Python number is handed over as a 0-d
        float32 and a qualifying 0-d tensor uncast, rather than rounded to
        the half dtype first.
    inplace : bool, default=False
        Whether ``a`` is written in place: the promoted result dtype must be
        castable to ``a``'s dtype (:func:`lucid._promotion.can_cast`).

    Returns
    -------
    tuple[TensorImpl, TensorImpl]
        The two operands.

    Raises
    ------
    TypeError
        If an operand is neither a tensor nor a Python number.
    RuntimeError
        If ``inplace`` is True and the promoted dtype cannot be cast to ``a``'s.
    """
    # A Tensor's impl is one attribute away and a bare impl is itself; a
    # number takes the general route.
    ai = _promotion.try_unwrap(a)
    if ai is None:
        a = _promotion.as_number(a)
        ai = _tensor_operand(a)
    bi = _promotion.try_unwrap(b)
    if bi is None:
        b = _promotion.as_number(b)
        bi = _tensor_operand(b)
    if ai is not None and bi is not None:
        da = ai.dtype
        db = bi.dtype
        # Two tensors of one dtype, before any tier logic.
        if da == db and not (true_div and _promotion.is_integral(da)):
            return ai, bi
        return _tensor_pair(ai, da, bi, db, true_div, opmath, inplace)
    if ai is not None and isinstance(b, _promotion.NUMBER):
        return _beside_number(ai, b, true_div, opmath, inplace)
    if bi is not None and isinstance(a, _promotion.NUMBER):
        t, number = _beside_number(bi, a, true_div, opmath)
        return number, t
    return _number_pair(a, b, true_div)


def _promote_impls(impls: Sequence[_C_engine.TensorImpl]) -> list[_C_engine.TensorImpl]:
    """Cast ``impls`` to the one dtype they promote to together.

    The n-ary form of :func:`_binary_operands`, for the ops whose engine
    kernel wants every tensor operand at a single dtype — the joins,
    ``where``, ``outer``, ``einsum`` — and so raised ``DtypeMismatch`` for
    an ``int64`` beside a ``float32``, where the reference answers at their
    common dtype.  Operands that already agree come back uncast, so the
    common case costs one comparison each.
    """
    out = list(impls)
    if not out:
        return out
    first = out[0].dtype
    if all(t.dtype == first for t in out):
        return out
    dt = _promotion.result_type(*out)
    return [t if t.dtype == dt else _C_engine.astype(t, dt) for t in out]


def _wrap(impl: _C_engine.TensorImpl) -> Tensor:
    """Wrap a TensorImpl in a Tensor (zero-copy)."""
    from lucid._tensor.tensor import Tensor

    return Tensor.__new_from_impl__(impl)


def normalize_factory_kwargs(
    dt: dtype | type[dtype] | _C_engine.Dtype | str | None = None,
    dev: device | _C_engine.Device | str | None = None,
    requires_grad: bool = False,
) -> tuple[_C_engine.Dtype, _C_engine.Device, bool]:
    """Resolve user-facing ``dtype`` / ``device`` kwargs to engine enums.

    Single dispatch hot path used by every Python factory function
    (``tensor``, ``zeros``, ``randn``, …) to translate the variety of
    accepted argument types (string / class / instance / ``None``) into
    the engine's two enum values plus the ``requires_grad`` flag.
    ``None`` arguments fall through to the cached defaults set by
    :func:`set_default_dtype` / :func:`set_default_device`.

    Parameters
    ----------
    dt : dtype, dtype-class, engine.Dtype, str, or None, optional
        Requested dtype.  ``None`` (default) → current default dtype.
    dev : device, engine.Device, str, or None, optional
        Requested device.  ``None`` (default) → current default device.
    requires_grad : bool, optional
        Pass-through; included in the return tuple unchanged.  Default
        ``False``.

    Returns
    -------
    tuple
        ``(engine_dtype, engine_device, requires_grad)``.  The first two
        are C++ enum values consumable by ``_C_engine`` factories.
    """
    if dt is None:
        _dtype = _default_dtype_enum_cached()
    else:
        _dtype = to_engine_dtype(dt)
    if dev is None:
        _device = _default_device_enum_cached()
    else:
        _device = _parse_device(dev)
    return _dtype, _device, requires_grad


def _check_grad_dtype(dtype: _C_engine.Dtype, requires_grad: bool) -> None:
    """Refuse ``requires_grad=True`` on a bool or integer dtype.

    The one home of this rule.  The engine's flag setters do not check, so
    every Python path that sets the flag comes through here: an alias
    through :func:`_impl_with_grad`, the in-place flip through
    ``Tensor.requires_grad_`` and the property setter.  It checks the
    tensor's own dtype — checking a factory's *requested* dtype instead,
    which reads as the default float under ``dtype=None``, is how
    ``Tensor(int_tensor, requires_grad=True)`` used to get through.

    Raises
    ------
    RuntimeError
        If ``requires_grad`` is set and ``dtype`` is bool or an integer.
    """
    if requires_grad and _promotion.is_integral(dtype):
        raise RuntimeError("only Tensors of floating point and complex dtype can require gradients")


def _impl_with_grad(impl: _C_engine.TensorImpl, requires_grad: bool) -> _C_engine.TensorImpl:
    """An alias of ``impl`` — same storage, no copy — with ``requires_grad`` set.

    Python code takes a ``clone_with_grad`` alias only through here, so no
    alias can carry the flag on a dtype that has no gradient.

    Raises
    ------
    RuntimeError
        If ``requires_grad`` is set and ``impl`` is bool or an integer.
    """
    _check_grad_dtype(impl.dtype, requires_grad)
    return impl.clone_with_grad(requires_grad)


_GRAD_OWNERS: weakref.WeakKeyDictionary[
    _C_engine.TensorImpl, weakref.ReferenceType[_C_engine.TensorImpl]
] = weakref.WeakKeyDictionary()


def _read_grad(owner: _C_engine.TensorImpl) -> _C_engine.TensorImpl | None:
    """Read ``owner``'s ``.grad`` impl and record it as ``owner``'s handle."""
    grad_impl = owner.grad_as_impl() or owner.grad_to_tensor()
    if grad_impl is None:
        return None
    _GRAD_OWNERS[grad_impl] = weakref.ref(owner)
    return grad_impl


def _set_grad(owner: _C_engine.TensorImpl, v: Tensor | None) -> None:
    """Assign or clear ``owner``'s accumulated ``.grad`` slot."""
    if v is None:
        owner.zero_grad()
    else:
        owner.set_grad(_unwrap(v))


def _swap_data(old: _C_engine.TensorImpl, value: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """The impl a tensor holding ``old`` takes for ``.data = value``.

    The one home of that swap (``Tensor.data``'s setter, and ``Module._apply``
    through it).  The result aliases ``value``'s storage — its shape, dtype
    and device — and carries what ``old`` is to autograd: its
    ``requires_grad``, and its gradient while that still fits the new data.
    It has no ``grad_fn``, so the swap records nothing in a graph, and
    ``value``'s own flag and graph stay with ``value``.  When ``old`` is a
    tensor's current ``.grad`` handle, the owner's gradient slot is updated
    too (``p.grad.data = value``, LCD-598), and the module tree version moves
    so a compiled module or step sees the replaced storage.

    Raises
    ------
    RuntimeError
        If ``old`` is not a leaf — its place in the graph cannot move to new
        storage — or it requires grad and ``value`` is bool or an integer.
    """
    if not old.is_leaf:
        raise RuntimeError(
            "cannot set .data of a tensor that is not a leaf: the graph that "
            "computed it would be lost; set it on a detached tensor instead"
        )
    if not old.is_phantom and value.is_phantom:
        raise _C_engine.LucidError("phantom tensor has no values: .data")
    new = _impl_with_grad(value, old.requires_grad)
    grad = old.grad_to_tensor()
    # The reference keeps any gradient; here a gradient must match its
    # tensor, so one of the old shape, dtype or device is dropped.
    if grad is not None and (grad.shape, grad.dtype, grad.device) == (
        new.shape,
        new.dtype,
        new.device,
    ):
        new.set_grad(grad)
    owner_ref = _GRAD_OWNERS.get(old)
    owner = owner_ref() if owner_ref is not None else None
    if owner is not None and (owner.grad_as_impl() is old or owner.grad_to_tensor() is old):
        owner.set_grad(new)
        slot = _read_grad(owner)
        if slot is not None:
            new = slot
    _tree_changed()
    return new


def _parse_device(d: device | _C_engine.Device | str) -> _C_engine.Device:
    """Convert device/string/engine Device → engine Device enum."""
    if isinstance(d, _C_engine.Device):
        return d
    if isinstance(d, device):
        return d._engine
    if isinstance(d, str):
        return device(d)._engine
    raise TypeError(f"Cannot parse device: {d!r}")
