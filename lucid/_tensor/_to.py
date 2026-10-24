"""
Tensor.to() and device/dtype shortcut methods.
Injected into Tensor by _inject_to() after class definition.
"""

from typing import TYPE_CHECKING, cast
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dtype import dtype as _dtype_cls, to_engine_dtype
from lucid._dtype import bfloat16, bool_, complex64, complex128
from lucid._dtype import float16, float32, float64, int8, int16, int32, int64
from lucid._device import device as _device_cls
from lucid._dispatch import _impl_with_grad, _parse_device, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_builtin_bool = bool  # save before any local shadowing

# 3.2.0 fast paths: string→engine.Device lookup table, so the hot training
# loop ``x.to(device="metal")`` never allocates a Python ``device(...)``
# instance just to extract ``._engine``.  cProfile measured ~65 k .to()
# calls / epoch in LeNet-5/MNIST; each parsing pass through ``_parse_device``
# costs ~2.5 µs.  The dict miss path (unknown string) falls back to
# ``_parse_device`` which still raises a proper TypeError for garbage input.
_STR_TO_DEVICE_ENUM: dict[str, _C_engine.Device] = {
    "cpu": _C_engine.Device.CPU,
    "metal": _C_engine.Device.GPU,
    "gpu": _C_engine.Device.GPU,
}


_TO_KEYWORDS = frozenset({"device", "dtype", "non_blocking", "copy"})
_DEVICE_LIKE = (str, _C_engine.Device, _device_cls, type(None))
_DTYPE_LIKE = (_dtype_cls, _C_engine.Dtype)


def _to_operands(
    args: tuple[object, ...], kwargs: dict[str, object], tensor_cls: type
) -> tuple[object, object, object, bool]:
    """``(other, device, dtype, copy)`` of a ``.to(...)`` call, read as the reference reads it.

    The reference takes three shapes — ``(device=None, dtype=None,
    non_blocking=False, copy=False)``, ``(dtype, non_blocking=False,
    copy=False)`` and ``(other, non_blocking=False, copy=False)`` — and
    refuses everything else with ``TypeError``: an unknown keyword, an
    argument of no shape (``object()``, a list), a dtype before a device,
    a second dtype, a keyword for an argument already given by position.
    A leading ``True`` / ``False`` is read as the dtype ``bool``, as the
    reference reads it.  ``None`` is "keep" for a device or dtype.
    A device index (``x.to(3)``) is refused: Lucid has one device of each
    kind.  A transfer is synchronous, so ``non_blocking`` is read and
    dropped.
    """
    unknown = sorted(set(kwargs) - _TO_KEYWORDS)
    if unknown:
        raise TypeError(f"to() got an unexpected keyword argument {unknown[0]!r}")
    rest = list(args)
    other, device, dtype, taken = _to_target(rest, tensor_cls)
    if len(rest) > 2 or not all(isinstance(f, _builtin_bool) for f in rest):
        raise TypeError(
            "to() takes at most non_blocking and copy after the device / dtype / "
            f"tensor, got {tuple(rest)!r}"
        )
    taken.update(("non_blocking", "copy")[: len(rest)])
    clash = sorted(taken & set(kwargs))
    if clash:
        raise TypeError(f"to() got multiple values for argument {clash[0]!r}")
    device = kwargs.get("device", device)
    dtype = kwargs.get("dtype", dtype)
    copy = rest[1] if len(rest) > 1 else kwargs.get("copy", False)
    return other, device, dtype, _builtin_bool(copy)


def _to_engine_target(
    args: tuple[object, ...], kwargs: dict[str, object], tensor_cls: type
) -> tuple[_C_engine.Device | None, _C_engine.Dtype | None, bool]:
    """``(device, dtype, copy)`` of a ``.to(...)`` call as engine values.

    For a caller that applies one ``.to`` to many tensors (``Module.to``):
    the call is read once, as :func:`_to_operands` reads it, and the other
    tensor's device and dtype stand in for it.  ``None`` keeps a tensor's own.

    Raises
    ------
    TypeError
        If the call has no ``.to`` shape, or a keyword names no device / dtype.
    """
    other, device, dtype, copy = _to_operands(args, kwargs, tensor_cls)
    impl = _promotion.try_unwrap(other)
    if impl is not None:
        return impl.device, impl.dtype, copy
    if not (device is None or isinstance(device, (str, _C_engine.Device, _device_cls))):
        raise TypeError(f"to() takes a device as device=, got {device!r}")
    if not (dtype is None or isinstance(dtype, (str, _dtype_cls, _C_engine.Dtype))):
        raise TypeError(f"to() takes a dtype as dtype=, got {dtype!r}")
    return (
        None if device is None else _parse_device(device),
        None if dtype is None else to_engine_dtype(dtype),
        copy,
    )


def _to_target(
    rest: list[object], tensor_cls: type
) -> tuple[object, object, object, set[str]]:
    """Pop ``.to``'s leading tensor, dtype, or device and dtype off ``rest``.

    Returns ``(other, device, dtype, taken)``, ``taken`` the parameters the
    positions filled, which a keyword may not name again.
    """
    if not rest:
        return None, None, None, set()
    first = rest[0]
    if isinstance(first, tensor_cls):  # the other tensor's device and dtype
        return rest.pop(0), None, None, {"device", "dtype"}
    if isinstance(first, (_builtin_bool, *_DTYPE_LIKE)):
        dtype = bool_ if isinstance(rest.pop(0), _builtin_bool) else first
        return None, None, dtype, {"device", "dtype"}
    if not isinstance(first, _DEVICE_LIKE):
        raise TypeError(
            f"to() takes a device, a dtype or a tensor first, got {type(first).__name__}"
        )
    device = rest.pop(0)
    if not rest:
        return None, device, None, {"device"}
    if not (rest[0] is None or isinstance(rest[0], _DTYPE_LIKE)):
        raise TypeError(f"to() takes a dtype after the device, got {rest[0]!r}")
    return None, device, rest.pop(0), {"device", "dtype"}


# The reference's legacy type names, one per dtype Lucid holds.  Lucid has
# no uint8, so ``ByteTensor`` (unsigned in the reference) names no dtype
# here and ``int8`` is ``CharTensor``, as it is there.
_LEGACY_TYPE_NAME: dict[_C_engine.Dtype, str] = {
    _C_engine.Dtype.F32: "lucid.FloatTensor",
    _C_engine.Dtype.F64: "lucid.DoubleTensor",
    _C_engine.Dtype.F16: "lucid.HalfTensor",
    _C_engine.Dtype.BF16: "lucid.BFloat16Tensor",
    _C_engine.Dtype.I8: "lucid.CharTensor",
    _C_engine.Dtype.I16: "lucid.ShortTensor",
    _C_engine.Dtype.I32: "lucid.IntTensor",
    _C_engine.Dtype.I64: "lucid.LongTensor",
    _C_engine.Dtype.Bool: "lucid.BoolTensor",
    _C_engine.Dtype.C64: "lucid.ComplexFloatTensor",
    _C_engine.Dtype.C128: "lucid.ComplexDoubleTensor",
}
# Off the CPU the reference qualifies the name by the device — its Apple GPU
# tensors are ``<lib>.mps.FloatTensor`` — and a name casts to its device as
# well as its dtype, a plain one to the CPU.  Metal's names are
# ``lucid.metal.FloatTensor`` and so on; unlike the reference's mps names
# they are taken back too, as its other accelerators' names are.
_METAL_PREFIX = "lucid.metal."
_LEGACY_TYPE: dict[str, tuple[_C_engine.Device, _C_engine.Dtype]] = {
    **{v: (_C_engine.Device.CPU, k) for k, v in _LEGACY_TYPE_NAME.items()},
    **{
        _METAL_PREFIX + v.removeprefix("lucid."): (_C_engine.Device.GPU, k)
        for k, v in _LEGACY_TYPE_NAME.items()
    },
}


def _legacy_type_name(impl: _C_engine.TensorImpl) -> str:
    """``Tensor.type()``: the legacy name of ``impl``'s dtype on its device."""
    name = _LEGACY_TYPE_NAME[impl.dtype]
    if impl.device == _C_engine.Device.GPU:
        return _METAL_PREFIX + name.removeprefix("lucid.")
    return name


def _type_target(
    dst_type: object,
) -> tuple[_C_engine.Device | None, _C_engine.Dtype]:
    """The ``(device, dtype)`` ``Tensor.type(dst_type)`` casts to.

    Takes what the reference's ``type`` takes: a dtype, which keeps the
    device (``None``), or a legacy type name such as ``"lucid.DoubleTensor"``
    (the CPU) or ``"lucid.metal.HalfTensor"`` (Metal).  A dtype's own name
    (``"float64"``) or a Python type (``float``) is refused, as there.

    Raises
    ------
    ValueError
        If ``dst_type`` is a string or a type that names no legacy type.
    TypeError
        If ``dst_type`` is neither a dtype, a string nor a type.
    """
    if isinstance(dst_type, _DTYPE_LIKE):
        return None, to_engine_dtype(dst_type)
    if isinstance(dst_type, str):
        found = _LEGACY_TYPE.get(dst_type)
        if found is not None:
            return found
        name = dst_type
    elif isinstance(dst_type, type):
        name = dst_type.__name__
    else:
        raise TypeError(
            f"type() takes a dtype or a legacy type name, got {type(dst_type).__name__}"
        )
    raise ValueError(
        f"invalid type: {name!r}; type() takes a dtype or one of {sorted(_LEGACY_TYPE)}"
    )


def _fast_device_enum(dev_in: object) -> _C_engine.Device | None:
    if isinstance(dev_in, str):
        return _STR_TO_DEVICE_ENUM.get(dev_in)
    if isinstance(dev_in, _C_engine.Device):
        return dev_in
    if isinstance(dev_in, _device_cls):
        return dev_in._engine
    return None


def _resolve_to_targets(
    self: Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[_C_engine.Device, _C_engine.Dtype, bool]:
    target_device = self._impl.device
    target_dtype = self._impl.dtype
    # The class is read at the call, not stashed by the injector: running an
    # injector on another class (the shadowing guard does) must not change
    # what ``x.to(other)`` accepts.
    from lucid._tensor.tensor import Tensor as _Tensor

    other, device, dtype, copy = _to_operands(args, kwargs, _Tensor)
    if other is not None:
        t = cast("Tensor", other)
        target_device = t._impl.device
        target_dtype = t._impl.dtype
    if device is not None:
        target_device = _parse_device(
            cast(_device_cls | _C_engine.Device | str, device)
        )
    if dtype is not None:
        target_dtype = to_engine_dtype(
            cast(_dtype_cls | type[_dtype_cls] | _C_engine.Dtype | str, dtype)
        )
    return target_device, target_dtype, copy


def _transfer_impl(
    src_impl: _C_engine.TensorImpl,
    target_device: _C_engine.Device,
    target_dtype: _C_engine.Dtype,
) -> Tensor:
    # A shared tensor is already one whole contiguous buffer, and
    # ``contiguous`` copied it into an ordinary one, so the relabel below
    # was never reached and ``.to()`` of a shared tensor always copied.
    # Keep the buffer unless a cast has to copy anyway.
    #
    # A cast writes a fresh dense tensor from any layout, so the dense
    # copy is needed only when no cast follows; taking it first copied
    # every element twice (a third of ``.long()``).
    relabel = src_impl.is_metal_shared and target_dtype == src_impl.dtype
    casts = target_dtype != src_impl.dtype
    impl = src_impl if relabel or casts else _C_engine.contiguous(src_impl)
    if target_dtype != impl.dtype:
        impl = _C_engine.astype(impl, target_dtype)

    # The result keeps the source's flag — except under ``no_grad``, where
    # it is a new tensor outside any graph (LCD-486), and on bool or an
    # integer, where ``astype`` ends the graph (rounding's derivative is
    # zero): restoring the flag there made ``x.long()`` require grad with
    # a backward that fell over inside the backend.
    rg = src_impl.requires_grad and _C_engine.grad_enabled()
    rg = rg and not _promotion.is_integral(target_dtype)
    if target_device != impl.device:
        if rg and impl.requires_grad:
            # Inside a graph the move is differentiable: the gradient
            # goes back to the source device.
            return _wrap(_C_engine.to_device(impl, target_device))
        if impl.is_metal_shared:
            impl = _C_engine.transfer_storage(impl, target_device)
        else:
            impl = impl.transfer_to_device(target_device, rg)
        if impl.requires_grad != rg:
            impl = _impl_with_grad(impl, rg)
    elif impl.requires_grad != rg:
        impl = _impl_with_grad(impl, rg)
    return _wrap(impl)


def _to(
    self: Tensor,
    *args: _dtype_cls | _device_cls | _C_engine.Device | str | _builtin_bool,
    **kwargs: object,
) -> Tensor:
    """
    Move and/or cast tensor.

    Overloads:
      .to(device)                  → device conversion
      .to(dtype)                   → dtype conversion
      .to(device, dtype)           → both
      .to(other_tensor)            → match other's device & dtype
      .to(device, dtype, non_blocking, copy)
      .to(device=, dtype=, non_blocking=, copy=)

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(3)
    >>> x.to(lucid.float64).dtype
    lucid.float64
    >>> x.to("metal").device
    device('metal')

    Notes
    -----
    No-op when the source already matches the target device and dtype,
    unless ``copy=True``; in that case the same Tensor instance is
    returned without allocation.  Apple Silicon's unified memory means
    device transfers between CPU and Metal copy the dispatch label but
    share physical DRAM when the underlying buffer is a SharedStorage.
    """
    if not args and len(kwargs) == 1 and "device" in kwargs:
        target_enum = _fast_device_enum(kwargs["device"])
        if target_enum is not None and target_enum == self._impl.device:
            return self

    target_device, target_dtype, copy = _resolve_to_targets(self, args, kwargs)
    if (
        target_device == self._impl.device
        and target_dtype == self._impl.dtype
        and not copy
    ):
        return self
    return _transfer_impl(self._impl, target_device, target_dtype)


def _metal(self: Tensor) -> Tensor:
    """Move this tensor to Apple Metal GPU.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(3)
    >>> x.metal().device
    device('metal')

    Notes
    -----
    No-op when the tensor is already on Metal.  Under Apple Silicon's
    unified memory architecture the physical DRAM is shared with the
    CPU; this call only updates the dispatch target so subsequent ops
    run through the MLX/Metal backend.
    """
    if self._impl.device == _C_engine.Device.GPU:
        return self
    return _to(self, _C_engine.Device.GPU)


def _cpu(self: Tensor) -> Tensor:
    """Move this tensor to CPU.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(3).metal()
    >>> x.cpu().device
    device('cpu')

    Notes
    -----
    No-op when the tensor is already on CPU.  On Apple Silicon both
    CPU and Metal share the same physical DRAM (unified memory); this
    call relabels the dispatch target so subsequent ops route through
    Apple Accelerate (vDSP / vForce / BLAS / LAPACK).
    """
    if self._impl.device == _C_engine.Device.CPU:
        return self
    return _to(self, _C_engine.Device.CPU)


def _cast(self: Tensor, target: _dtype_cls) -> Tensor:
    """``self`` at ``target``'s dtype: ``self`` itself when it is there already."""
    if self._impl.dtype == to_engine_dtype(target):
        return self
    return _to(self, target)


def _float(self: Tensor) -> Tensor:
    """Cast to float32.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3, dtype=lucid.int64).float().dtype
    lucid.float32

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``float32``.  Otherwise performs a copy-cast via the engine's
    ``astype`` op; the device is preserved.
    """
    return _cast(self, float32)


def _double(self: Tensor) -> Tensor:
    """Cast to float64.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).double().dtype
    lucid.float64

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``float64``.  Otherwise allocates a new tensor with widened
    precision; device is preserved.  Metal has no float64, so a Metal
    tensor is refused with ``NotImplementedError``.
    """
    return _cast(self, float64)


def _half(self: Tensor) -> Tensor:
    """Cast to float16.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).half().dtype
    lucid.float16

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``float16``.  Lossy narrowing from float32/float64 may overflow to
    ``inf`` for magnitudes above ~65504 and underflow to 0 for
    magnitudes below ~6e-5.
    """
    return _cast(self, float16)


def _bfloat16(self: Tensor) -> Tensor:
    """Cast to bfloat16.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).bfloat16().dtype
    lucid.bfloat16

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``bfloat16``.  bfloat16 keeps float32's exponent range with an 8-bit
    significand, so narrowing rounds to about three significant decimal
    digits but does not overflow where float32 does not.
    """
    return _cast(self, bfloat16)


def _int(self: Tensor) -> Tensor:
    """Cast to int32.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.7, 2.3]).int().dtype
    lucid.int32

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``int32``.  Casts from floating point truncate toward zero;
    values outside ``[-2**31, 2**31 - 1]`` produce undefined results.
    """
    return _cast(self, int32)


def _long(self: Tensor) -> Tensor:
    """Cast to int64.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.7, 2.3]).long().dtype
    lucid.int64

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``int64``.  Casts from floating point truncate toward zero.
    Typically used for index tensors (e.g. ``gather`` / ``scatter``).
    """
    return _cast(self, int64)


def _short(self: Tensor) -> Tensor:
    """Cast to int16.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.7, -2.3]).short().tolist()
    [1, -2]

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``int16``.  Casts from floating point truncate toward zero.
    """
    return _cast(self, int16)


def _char(self: Tensor) -> Tensor:
    """Cast to int8.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.7, -2.3]).char().dtype
    lucid.int8

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``int8``, the signed byte.  Casts from floating point truncate
    toward zero.
    """
    return _cast(self, int8)


def _byte(self: Tensor) -> Tensor:
    """Cast to uint8: refused, since Lucid has no unsigned dtype.

    The reference's ``byte()`` is ``to(uint8)``.  Lucid's integer dtypes
    are all signed, and a silent stand-in would wrap every value above
    127 or read a negative one differently, so the call is refused.

    Raises
    ------
    NotImplementedError
        Always.  ``char()`` casts to int8, the signed byte, and
        ``short()`` to int16, which holds every uint8 value.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).byte()
    Traceback (most recent call last):
        ...
    NotImplementedError: Tensor.byte(): lucid has no uint8 dtype ...
    """
    del self  # reference-signature parity: the method takes the tensor
    raise NotImplementedError(
        "Tensor.byte(): lucid has no uint8 dtype — its integer dtypes are "
        "signed.  Use .char() for int8, or .short() for int16, which holds "
        "every uint8 value."
    )


def _cfloat(self: Tensor) -> Tensor:
    """Cast to complex64.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.0, -2.0]).cfloat().dtype
    lucid.complex64

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``complex64``.  A real tensor becomes the real part, with a zero
    imaginary part.
    """
    return _cast(self, complex64)


def _chalf(self: Tensor) -> Tensor:
    """Cast to complex32: refused, since Lucid has no half-precision complex.

    The reference's ``chalf()`` is ``to(complex32)``, a pair of float16
    halves.  Lucid's narrowest complex dtype is complex64, and handing one
    back would double the width the caller asked for, so the call is
    refused.

    Raises
    ------
    NotImplementedError
        Always.  ``cfloat()`` casts to complex64.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).chalf()
    Traceback (most recent call last):
        ...
    NotImplementedError: Tensor.chalf(): lucid has no complex32 dtype ...
    """
    del self  # reference-signature parity: the method takes the tensor
    raise NotImplementedError(
        "Tensor.chalf(): lucid has no complex32 dtype — its narrowest complex "
        "dtype is complex64.  Use .cfloat() for complex64."
    )


def _cdouble(self: Tensor) -> Tensor:
    """Cast to complex128.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.0, -2.0]).cdouble().dtype
    lucid.complex128

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``complex128``.  Metal has no complex128, so a Metal tensor is
    refused with ``NotImplementedError``.
    """
    return _cast(self, complex128)


def _bool(self: Tensor) -> Tensor:
    """Cast to bool.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([0.0, 1.5, -2.0]).bool()
    tensor([False, True, True], dtype=lucid.bool)

    Notes
    -----
    Returns ``self`` unchanged when the source dtype is already
    ``bool``.  Otherwise each element is reduced to ``True`` if
    nonzero and ``False`` if zero.
    """
    return _cast(self, bool_)


_TO_METHODS = (
    ("to", _to),
    ("metal", _metal),
    ("cpu", _cpu),
    ("float", _float),
    ("double", _double),
    ("half", _half),
    ("bfloat16", _bfloat16),
    ("int", _int),
    ("long", _long),
    ("short", _short),
    ("char", _char),
    ("byte", _byte),
    ("cfloat", _cfloat),
    ("chalf", _chalf),
    ("cdouble", _cdouble),
    ("bool", _bool),
)


def _inject_to(cls: type) -> None:
    """Attach .to(), .metal(), .cpu(), and dtype-cast methods to Tensor."""
    globals().setdefault("Tensor", cls)
    for name, fn in _TO_METHODS:
        fn.__name__ = name
        fn.__qualname__ = f"_inject_to.<locals>.{name}"
        setattr(cls, name, fn)
    _reset_class_annotation_cell(cls)


def _reset_class_annotation_cell(cls: type) -> None:
    """Keep injected ``int`` / ``bool`` / ``float`` methods out of ``cls``'s PEP 649 cell."""
    ann = getattr(cls.__dict__.get("__init__"), "__annotate__", None)
    closure = getattr(ann, "__closure__", None)
    if closure:
        closure[0].cell_contents = {"_UntypedStorage": cls.__dict__["_UntypedStorage"]}
