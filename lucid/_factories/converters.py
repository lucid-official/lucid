"""
Conversion utilities: Python objects / NumPy arrays -> TensorImpl.

This is one of the H4 numpy bridge sites — the only place where a
``np.ndarray`` is allowed to enter Lucid.  NumPy is imported lazily
(inside the functions that need it) so ``import lucid`` works
without numpy installed.

3.0.2: pure-Python inputs (``list``, ``tuple``, scalar) now take a
numpy-free fast path through ``struct.pack`` + ``TensorImpl.from_bytes``.
``lucid.tensor([1, 2, 3])`` no longer pulls numpy into the dependency
graph — only ``lucid.tensor(np_array)`` and explicit ``from_numpy``
keep the numpy bridge.
"""

from typing import TYPE_CHECKING, TypeGuard

from lucid._C import engine as _C_engine
from lucid._dispatch import _impl_with_grad, normalize_factory_kwargs
from lucid._factories._build import (
    _from_values,
    _infer_dtype,
    _is_scalar_leaf,
    _leaf,
)
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    import numpy as np
    from lucid._tensor.tensor import Tensor


# Cached default-device engine enum for the ndarray fast path.
# Default device is stable per process (set once by user or defaults to CPU);
# resolving it on every ``lucid.tensor(np_array)`` call cost ~50 ns × 120 k
# calls = 6 ms / epoch — small but eliminable.
_CACHED_DEFAULT_DEVICE_ENUM: _C_engine.Device | None = None


def _default_device_enum_cached() -> _C_engine.Device:
    """Return the resolved default-device engine enum from a per-process cache.

    Invalidation: cleared by :func:`lucid._globals.set_default_device` —
    user explicitly switching the global default is rare.  Code that
    passes an explicit ``device=`` kwarg bypasses this helper and goes
    through ``_parse_device`` as before.
    """
    global _CACHED_DEFAULT_DEVICE_ENUM
    if _CACHED_DEFAULT_DEVICE_ENUM is None:
        from lucid._dispatch import _parse_device
        from lucid._globals import get_default_device

        _CACHED_DEFAULT_DEVICE_ENUM = _parse_device(get_default_device())
    return _CACHED_DEFAULT_DEVICE_ENUM


def _invalidate_default_device_cache() -> None:
    """Called by ``lucid._globals.set_default_device`` after a change."""
    global _CACHED_DEFAULT_DEVICE_ENUM
    _CACHED_DEFAULT_DEVICE_ENUM = None


def _flatten_with_shape(data: object) -> tuple[list[int], list[object]] | None:
    """Walk a (possibly nested) list/tuple, returning ``(shape, flat)``.

    Returns ``None`` when the structure is ragged (different sub-lengths
    at the same level), signalling the caller to fall back to numpy.
    """
    if not isinstance(data, (list, tuple)):
        # Scalar — 0-d tensor.
        return [], [data]
    shape: list[int] = [len(data)]
    if len(data) == 0:
        return shape, []
    if isinstance(data[0], (list, tuple)):
        # Nested — recurse on first element to pick up the sub-shape,
        # then validate every sibling against it.
        first = _flatten_with_shape(data[0])
        if first is None:
            return None
        sub_shape, _ = first
        flat: list[object] = []
        for item in data:
            sub = _flatten_with_shape(item)
            if sub is None or sub[0] != sub_shape:
                return None
            flat.extend(sub[1])
        return shape + sub_shape, flat
    # Leaf row — must be uniform scalars.
    for item in data:
        if isinstance(item, (list, tuple)):
            return None
    return shape, list(data)


def _python_literal_impl(
    data: object,
    dtype_eng: _C_engine.Dtype | None,
    device_eng: _C_engine.Device,
) -> _C_engine.TensorImpl | None:
    """Build a Python literal (scalar or nested list/tuple) without numpy.

    NumPy *scalars* count as literal elements: they are numbers with a
    dtype of their own, which inference keeps (``np.bool_`` → bool,
    ``np.float16`` → float16).  Returns ``None`` — and the caller falls
    through to the numpy bridge — for ragged nesting or any element that
    is not a number (an ndarray, a Tensor, an object).
    """
    if not isinstance(data, (list, tuple)) and not _is_scalar_leaf(data):
        return None
    unpacked = _flatten_with_shape(data)
    if unpacked is None:
        return None
    shape, flat = unpacked
    if not all(_is_scalar_leaf(v) for v in flat):
        return None
    target = dtype_eng if dtype_eng is not None else _infer_dtype(flat)
    return _from_values(flat, shape, target, device_eng)


_NP_TO_ENGINE_DTYPE: dict[str, _C_engine.Dtype] = {
    "float16": _C_engine.Dtype.F16,
    "float32": _C_engine.Dtype.F32,
    "float64": _C_engine.Dtype.F64,
    "int8": _C_engine.Dtype.I8,
    "int16": _C_engine.Dtype.I16,
    "int32": _C_engine.Dtype.I32,
    "int64": _C_engine.Dtype.I64,
    "bool": _C_engine.Dtype.Bool,
    "complex64": _C_engine.Dtype.C64,
    "complex128": _C_engine.Dtype.C128,
}


def _np_dtype_to_engine(np_dtype: np.dtype) -> _C_engine.Dtype:
    name = np_dtype.name
    if name in _NP_TO_ENGINE_DTYPE:
        return _NP_TO_ENGINE_DTYPE[name]
    if name.startswith("float"):
        return _C_engine.Dtype.F64
    return _C_engine.Dtype.F32


def _is_ndarray(obj: object) -> TypeGuard[np.ndarray]:
    """Check ndarray-ness without importing numpy when it isn't loaded."""
    cls = type(obj)
    return cls.__module__ == "numpy" and cls.__name__ == "ndarray"


def _ndarray_impl(
    arr: np.ndarray, device_eng: _C_engine.Device
) -> _C_engine.TensorImpl:
    """The one handoff of an ndarray to the engine, at the array's own rank.

    Every ndarray ingest ends here.  ``np.ascontiguousarray`` returns an
    array of at least one dimension, so routing a 0-d array through it
    handed the engine shape ``(1,)`` — ``tensor(np.asarray(0.5),
    device="cpu")`` came back one-dimensional while the same call without
    ``device`` stayed 0-d.  ``asarray(order="C")`` copies only what is not
    already C-contiguous and keeps the rank.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        np = _require_numpy("lucid.tensor() ndarray ingest")
        arr = np.asarray(arr, order="C")  # type: ignore[attr-defined]
    return _C_engine.TensorImpl(arr, device_eng, False)


def _require_numpy(operation: str) -> object:
    """Lazy-import numpy with a clean error message when it isn't installed.

    Lucid runs without numpy by default; the bridge methods listed in H4
    (``tensor(np_array)``, ``Tensor.numpy()``, ``from_numpy``,
    ``from_dlpack`` / ``to_dlpack``) opt into numpy as the canonical
    interop library.  When the user reaches one of these without having
    installed numpy, raise an ``ImportError`` that points them at the
    correct extra rather than the generic ``ModuleNotFoundError``.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"{operation} requires numpy, but numpy is not installed.\n"
            "Install it explicitly:\n"
            "    pip install lucid[numpy]\n"
            "or\n"
            "    pip install numpy"
        ) from e
    return np


def _to_impl(
    data: object,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> _C_engine.TensorImpl:
    """Convert list/scalar/ndarray/Tensor -> TensorImpl."""
    from lucid._tensor.tensor import Tensor as _Tensor

    # 3.1.1: hottest path — bare ``lucid.tensor(np_array)`` with no dtype
    # override, default device, no grad.  Skips ``normalize_factory_kwargs``
    # and the literal path's isinstance gauntlet.  Profile of LeNet-5/MNIST training showed
    # ``lucid.tensor`` called 120k times per epoch via the DataLoader
    # per-sample tensorisation pattern — this fast path turns a ~9 µs
    # per-call hot loop into ~1 µs.
    if dtype is None and device is None and not requires_grad and _is_ndarray(data):
        if data.dtype.kind == "u":
            data = _widen_unsigned(data)
        return _ndarray_impl(data, _default_device_enum_cached())

    _dtype_eng, _device_eng, _rg = normalize_factory_kwargs(
        dtype, device, requires_grad
    )

    if isinstance(data, (_Tensor, _C_engine.TensorImpl)):
        return _source_impl(
            data,
            _dtype_eng if dtype is not None else None,
            _device_eng if device is not None else None,
            _rg,
        )

    # Everything below builds the value first and makes the leaf last: a
    # ``requires_grad`` set before the bfloat16 narrowing used to hand
    # back the narrowing's output, a non-leaf whose ``.grad`` never filled.
    explicit = _dtype_eng if dtype is not None else None
    literal = _python_literal_impl(data, explicit, _device_eng)
    if literal is not None:
        return _leaf(literal, _rg)
    return _leaf(_numpy_impl(data, explicit, _device_eng), _rg)


def _source_impl(
    data: Tensor | _C_engine.TensorImpl,
    dtype_eng: _C_engine.Dtype | None,
    device_eng: _C_engine.Device | None,
    requires_grad: bool,
) -> _C_engine.TensorImpl:
    """A Tensor or engine source, converted to ``dtype_eng`` / ``device_eng``.

    ``Tensor(t, dtype=float64, device="metal")`` used to drop both and hand
    back a float32 CPU tensor.  A dtype or device that differs converts the
    source — a new tensor, still in its graph; ``None`` or a match keeps the
    source's own.
    """
    # A Tensor's own impl, which an unconverted source must not be handed
    # back as; an engine source has none.
    own: _C_engine.TensorImpl | None = None
    if isinstance(data, _C_engine.TensorImpl):
        source = data
    else:
        own = source = data._impl
    if dtype_eng is not None and source.dtype != dtype_eng:
        source = _C_engine.astype(source, dtype_eng)
    if device_eng is not None and source.device != device_eng:
        source = _C_engine.to_device(source, device_eng)
    if source is own:
        # ``Tensor(x)`` used to hand back ``x``'s own TensorImpl here, so
        # the new tensor was ``x`` under another name: a flag flipped or a
        # gradient accumulated on either showed up on both.  A view shares
        # the storage without sharing the tensor, and keeps a source that
        # requires grad in the graph, as the reference's ``Tensor(x)`` does.
        if requires_grad and not source.requires_grad:
            return _impl_with_grad(source, True)
        return _C_engine.view(source, list(source.shape))
    if source.requires_grad != requires_grad:
        return _impl_with_grad(source, requires_grad)
    return source


def _numpy_impl(
    data: object,
    dtype_eng: _C_engine.Dtype | None,
    device_eng: _C_engine.Device,
) -> _C_engine.TensorImpl:
    """Build an ndarray — or anything ``np.array`` accepts — in its final dtype.

    Without an explicit dtype the array's own dtype is kept (unsigned
    widened to the next signed type), including for non-ndarray input
    such as ``range`` or a list of arrays.  The result does not require
    grad; the caller makes the leaf.
    """
    # Numpy is the sanctioned conversion library here, imported lazily so
    # ``import lucid`` doesn't need it; ``_require_numpy`` explains a miss.
    np = _require_numpy("lucid.tensor() with non-Tensor input")
    arr = data if isinstance(data, np.ndarray) else np.array(data)  # type: ignore[attr-defined]
    if dtype_eng is None:
        if arr.dtype.kind == "u":
            arr = _widen_unsigned(arr)
        dtype_eng = _np_dtype_to_engine(arr.dtype)
    else:
        arr = arr.astype(_engine_dtype_to_np(dtype_eng), copy=False)
    with np.errstate(invalid="ignore", over="ignore"):  # type: ignore[attr-defined]
        impl = _ndarray_impl(arr, device_eng)
    # bfloat16 has no NumPy counterpart, so the array above is float32 and
    # the engine narrows afterwards.  Doing it this way round rather than
    # picking the nearest NumPy dtype matters: float16 is *not* a
    # substitute — a bfloat16 value can exceed float16's maximum, and
    # routing through it would turn a representable number into infinity
    # before the engine ever saw it.
    if dtype_eng == _C_engine.Dtype.BF16 and impl.dtype != _C_engine.Dtype.BF16:
        impl = _C_engine.astype(impl, _C_engine.Dtype.BF16)
    return impl


def _engine_dtype_to_np(d: _C_engine.Dtype) -> str:
    _MAP: dict[_C_engine.Dtype, str] = {
        # bfloat16 is widened, not narrowed — see the note in
        # ``_to_impl``: float16 cannot hold every bfloat16 value.
        _C_engine.Dtype.BF16: "float32",
        _C_engine.Dtype.F16: "float16",
        _C_engine.Dtype.F32: "float32",
        _C_engine.Dtype.F64: "float64",
        _C_engine.Dtype.I8: "int8",
        _C_engine.Dtype.I16: "int16",
        _C_engine.Dtype.I32: "int32",
        _C_engine.Dtype.I64: "int64",
        _C_engine.Dtype.Bool: "bool",
        _C_engine.Dtype.C64: "complex64",
        _C_engine.Dtype.C128: "complex128",
    }
    return _MAP.get(d, "float32")


def _copy_tensor(
    src: Tensor,
    *,
    dtype: DTypeLike,
    device: DeviceLike,
    requires_grad: bool,
) -> _C_engine.TensorImpl:
    """A fresh copy of ``src`` for :func:`tensor`, as a leaf of its own.

    ``_to_impl`` wraps a Tensor's storage when nothing is converted, which
    is right for ``Tensor(t)`` and ``Parameter(t)`` — they wrap the data
    they are given — and wrong here: ``tensor`` is documented to copy, and
    to leave the source's graph.

    ``None`` keeps the source's own dtype and device rather than falling
    back to the global defaults, as the reference framework does.
    """
    dt, dev, rg = normalize_factory_kwargs(
        dtype if dtype is not None else src.dtype,
        device if device is not None else src.device,
        requires_grad,
    )
    # Starting from a detached view means none of the steps below is
    # recorded, so the copy does not hang off ``src``'s graph.
    # ``contiguous`` always allocates, which is what makes this a copy even
    # when neither the dtype nor the device changes.
    impl = _C_engine.contiguous(src.detach()._impl)
    if impl.dtype != dt:
        impl = _C_engine.astype(impl, dt)
    if impl.device != dev:
        impl = impl.transfer_to_device(dev, False)
    return _leaf(impl, rg)


def tensor(
    data: object,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Construct a new :class:`Tensor` from Python data, a NumPy array, or another Tensor.

    Always allocates a fresh storage and **copies** the source bytes into
    Lucid-owned memory.  This is the canonical entry point for creating
    tensors from heterogeneous Python inputs: scalars (``int`` / ``float`` /
    ``bool``), nested lists, NumPy ``ndarray``\s, and existing Lucid
    ``Tensor``\s.  Dtype is inferred from the source unless ``dtype`` is
    given; device defaults to the global default (typically ``"cpu"``) unless
    overridden.

    Parameters
    ----------
    data : object
        Source data.  Accepted forms:

        * Python scalar (``bool``, ``int``, ``float``, ``complex``) or
          NumPy scalar — produces a 0-d tensor.
        * Nested ``list`` / ``tuple`` of those — recursively converted;
          the nesting must be rectangular.
        * ``numpy.ndarray`` — bridge boundary
          (see :mod:`lucid._factories.converters`); the data is copied
          regardless of the source array's contiguity.
        * Existing :class:`Tensor` — copied to a new buffer and detached
          from its autograd graph, so the result is a leaf of its own (use
          :func:`as_tensor` to avoid the copy when dtype/device match).
    dtype : dtype | str | None, optional
        Target element type.  ``None`` (default) infers from ``data``:
        ``bool`` → ``bool``, ``int`` → ``int64``, ``float`` → the default
        dtype (`get_default_dtype`), ``complex`` → its complex counterpart
        (``complex128`` under a ``float64`` default, else ``complex64``);
        a NumPy scalar or array, or a Tensor, keeps its own dtype.  Mixed
        elements take the highest kind (bool < int < float < complex) at
        the widest width among them.  Integers are stored exactly at any
        magnitude; a value the dtype cannot hold raises — ``TypeError``
        for a complex into a real dtype, ``ValueError`` for NaN or
        infinity into an integer dtype, ``OverflowError`` for an integer
        out of range.
    device : device | str | None, optional
        Target device (``"cpu"`` or ``"metal"``).  ``None`` uses
        :func:`lucid.get_default_device`, except that a Tensor stays on its
        own device.
    requires_grad : bool, optional
        Whether the resulting tensor should record autograd operations.
        The result is always a leaf, whatever the dtype conversion.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        A freshly-allocated Lucid tensor.

    Notes
    -----
    This factory is one of the six "bridge" entry points in the **H4** rule
    — the only places where external libraries (NumPy here) may legitimately
    cross into Lucid's compute path.  Outside the bridges, Lucid composites
    must use engine primitives directly.

    For zero-copy conversion when the source is already an ``ndarray`` on
    CPU and shares dtype, prefer :func:`as_tensor`.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.0, 2.0, 3.0])
    tensor([1., 2., 3.])
    >>> lucid.tensor([[1, 2], [3, 4]], dtype=lucid.float32)
    tensor([[1., 2.],
            [3., 4.]])
    >>> import numpy as np
    >>> lucid.tensor(np.arange(6).reshape(2, 3))
    tensor([[0, 1, 2],
            [3, 4, 5]], dtype=lucid.int64)
    >>> src = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> copy = lucid.tensor(src, dtype=lucid.float64)
    >>> copy.dtype, copy.is_leaf, copy.requires_grad
    (lucid.float64, True, False)
    """
    from lucid._tensor.tensor import Tensor

    if isinstance(data, Tensor):
        return Tensor.__new_from_impl__(
            _copy_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        )
    return Tensor.__new_from_impl__(
        _to_impl(data, dtype=dtype, device=device, requires_grad=requires_grad)
    )


def as_tensor(
    data: object,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Convert data to a tensor, avoiding a copy when the source already matches.

    Unlike :func:`tensor`, ``as_tensor`` is "best-effort no-copy":

    * If ``data`` is already a :class:`Tensor` with the requested ``dtype``
      and ``device``, it is returned unchanged.
    * If ``data`` is a :class:`Tensor` whose dtype or device differs, it is
      converted as :meth:`Tensor.to` converts it, and the result stays
      connected to ``data``'s autograd graph — unlike :func:`tensor`, which
      copies into a detached leaf.
    * Anything else — Python data, NumPy arrays — goes through
      :func:`tensor`, which copies.

    Parameters
    ----------
    data : object
        Source data — Python scalar / list, NumPy array, or Tensor.
    dtype : dtype | str | None, optional
        Target element type.  ``None`` preserves the source dtype.
    device : device | str | None, optional
        Target device.  When the source already lives on a different device,
        a copy across the device boundary is performed.

    Returns
    -------
    Tensor
        ``data`` itself, a converted copy of it, or a freshly-constructed
        Lucid tensor.

    Notes
    -----
    ``as_tensor`` is the right choice in code that may be handed either a
    Tensor or raw data (e.g. a collate function): a Tensor that already has
    the requested dtype and device passes through with no allocation at
    all.  For semantic clarity in library code that should never share
    storage, use :func:`tensor`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.as_tensor(x) is x              # already a Tensor, returned as-is
    True
    >>> lucid.as_tensor(x, dtype=lucid.float64).dtype
    lucid.float64
    >>> w = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> lucid.as_tensor(w, dtype=lucid.float64).sum().backward()
    >>> w.grad                               # the conversion stays in the graph
    tensor([1., 1.])
    """
    from lucid._tensor.tensor import Tensor

    if isinstance(data, Tensor):
        # Nothing to convert: hand back the object itself, so identity,
        # storage and the autograd graph all carry over unchanged.
        _dt, _dev, _ = normalize_factory_kwargs(
            dtype if dtype is not None else data.dtype,
            device if device is not None else data.device,
        )
        if data._impl.dtype == _dt and data._impl.device == _dev:
            return data
        # A conversion goes through ``Tensor.to`` rather than ``tensor``:
        # ``tensor`` copies into a detached leaf, while ``as_tensor`` of a
        # Tensor is a cast that stays in the graph, as the reference
        # framework's is.
        return data.to(device=_dev, dtype=_dt)
    return tensor(data, dtype=dtype, device=device)


def _widen_unsigned(arr: np.ndarray) -> np.ndarray:
    """An unsigned integer array as the next signed width, losslessly.

    Lucid has no unsigned dtypes.  Refusing them outright left no way in
    for a uint8 image or a uint16 bit pattern short of the caller casting
    by hand; every value of uint8/16/32 fits the next signed width, and
    uint64 fits int64 unless it holds a value past int64's range.
    """
    np = _require_numpy("lucid.tensor() unsigned input")
    size = arr.dtype.itemsize
    target = {1: np.int16, 2: np.int32}.get(size, np.int64)  # type: ignore[attr-defined]
    if size == 8 and arr.size and int(arr.max()) > np.iinfo(np.int64).max:  # type: ignore[attr-defined]
        raise OverflowError(
            "lucid.tensor: a uint64 array holds values past int64's range, "
            "and Lucid has no unsigned dtype to keep them"
        )
    return arr.astype(target)


def from_numpy(ndarray: np.ndarray, /) -> Tensor:
    r"""Copy a NumPy ``ndarray`` into an owned Lucid tensor.

    The returned tensor owns a copy and inherits the array's dtype according
    to the canonical
    NumPy → Lucid mapping (``np.float32`` → ``lucid.float32``,
    ``np.int64`` → ``lucid.int64``, etc.). Mutations do not propagate between
    the source array and the result. Like :func:`tensor`, this bridge uses
    the active default device; use ``tensor(arr, device="cpu")`` to pin it.

    Parameters
    ----------
    ndarray : numpy.ndarray
        Source array.  Must reside in CPU memory.  Any layout (C / Fortran /
        strided) is accepted and copied to contiguous owned storage.

    Returns
    -------
    Tensor
        A tensor with copied values on the active default device.

    Raises
    ------
    RuntimeError
        If the array dtype has no corresponding Lucid dtype. The native
        ``DtypeMismatch`` exception derives from ``RuntimeError``; object
        arrays, for example, are rejected.
    OverflowError
        If a ``uint64`` array holds a value past ``int64``'s range.

    Notes
    -----
    Lucid has no unsigned integer dtypes, so unsigned arrays widen
    losslessly to the next signed width: ``uint8`` to ``int16``, ``uint16``
    to ``int32``, ``uint32`` and ``uint64`` to ``int64``.

    This is one of the documented **H4** bridge boundaries — the only places
    where Lucid is allowed to take a NumPy array as input.  To move the
    result onto a Metal device, chain :meth:`Tensor.to`::

        t = lucid.from_numpy(arr).to("metal")

    Examples
    --------
    >>> import lucid, numpy as np
    >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    >>> t = lucid.from_numpy(arr)
    >>> t.dtype
    lucid.float32
    >>> arr[0, 0] = 99.0          # mutate the source
    >>> t[0, 0].item()            # the owned copy is unchanged
    1.0
    """
    return tensor(ndarray)


# ── DLPack interop ─────────────────────────────────────────────────────────
#
# Lucid's DLPack bridge intentionally goes through NumPy:
#
#   * NumPy already implements the DLPack PyCapsule producer/consumer
#     correctly (lifetime, deleter, all dtypes).  A Lucid CPU tensor's
#     numpy view is itself zero-copy, so wrapping it via numpy's
#     ``__dlpack__`` adds no data movement.
#   * Metal tensors must download to CPU regardless — DLPack with
#     ``kDLMetal`` device type is supported by almost no consumers.
#   * Building our own DLPack ABI in C++ would duplicate ~350 lines of
#     subtle struct / lifetime code without any runtime benefit on top
#     of what numpy already provides.
#
# Calling ``from_dlpack`` / ``to_dlpack`` therefore opts the user into
# numpy as the canonical interop library — same H4 carve-out as
# ``tensor(np_array)``, ``Tensor.numpy()``, and ``from_numpy``.
# ``_require_numpy`` raises a clean ImportError pointing at
# ``pip install lucid[numpy]`` when numpy is absent.


# DLPack device type for Metal, as MLX tags its capsules and as
# ``Tensor.__dlpack_device__`` reports for a GPU-resident tensor.
_DLPACK_METAL = 8


def _dlpack_device_kind(ext_tensor: object) -> int | None:
    dl_device = getattr(ext_tensor, "__dlpack_device__", None)
    if not callable(dl_device):
        return None
    try:
        kind, _index = dl_device()
    except (
        AttributeError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return kind if isinstance(kind, int) else None


def _finish_dlpack_import(
    out: Tensor, *, device: DeviceLike, copy: bool | None
) -> Tensor:
    from lucid._dispatch import _parse_device

    if device is not None:
        target_dev = _parse_device(device)
        if target_dev != out._impl.device:
            if copy is False:
                raise BufferError(
                    "from_dlpack: cannot move across devices when copy=False"
                )
            return out.to(device=target_dev)
    return out.clone() if copy else out


def from_dlpack(
    ext_tensor: object,
    *,
    device: DeviceLike = None,
    copy: bool | None = None,
) -> Tensor:
    """Construct a Lucid tensor from any object exposing the DLPack
    protocol (``__dlpack__``) or from a raw PyCapsule.

    Two dialects, picked by what the producer says it is:

    * **Metal** (``__dlpack_device__() == (8, 0)``) — adopted natively.
      The producer's ``MTLBuffer`` becomes this tensor's storage with no
      copy at all, which is what makes an ``mlx.core.array`` and a Lucid
      Metal tensor two views of one allocation. NumPy cannot read a
      capsule of this device type, so this path bypasses it entirely.
    * **Everything else** — through NumPy, which shares host memory
      where it can. The result lives on the CPU by default unless
      ``device`` is explicitly specified.

    Requires numpy for the host dialect — install via
    ``pip install lucid[numpy]`` if missing. The Metal dialect needs
    nothing.

    Parameters
    ----------
    ext_tensor : object
        An object exposing ``__dlpack__`` or a raw DLPack ``PyCapsule``.
    device : DeviceLike, optional
        Target device for the returned tensor.
    copy : bool or None, optional
        If ``True``, always copy the data; if ``False``, raise ``BufferError``
        when a copy is required.
    """
    from lucid._dispatch import _parse_device, _wrap
    from lucid._engine_scope import untraced

    if _dlpack_device_kind(ext_tensor) == _DLPACK_METAL:
        capsule = ext_tensor.__dlpack__()  # type: ignore[attr-defined]
        out = _wrap(_C_engine.from_dlpack_metal(capsule))
        return _finish_dlpack_import(out, device=device, copy=copy)

    if isinstance(ext_tensor, Tensor):
        ext_tensor.numpy()
        with untraced():
            alias = _wrap(ext_tensor._impl.data_alias())
        return _finish_dlpack_import(alias, device=device, copy=copy)

    if (
        device is not None
        and copy is False
        and _parse_device(device) != _C_engine.Device.CPU
    ):
        raise BufferError("from_dlpack: cannot move across devices when copy=False")
    np = _require_numpy("lucid.from_dlpack")
    arr = np.from_dlpack(ext_tensor)  # type: ignore[attr-defined]
    if copy is False:
        raise BufferError(
            "from_dlpack: cannot avoid a copy when importing an external host buffer"
        )
    return tensor(arr, device="cpu" if device is None else device)


def to_dlpack(tensor: Tensor) -> object:
    """Export ``tensor`` as a DLPack PyCapsule.

    A Metal tensor exports natively — the capsule carries its
    ``MTLBuffer``, so an MLX consumer reads the same pages rather than a
    downloaded copy. A CPU tensor materialises through NumPy. Either
    capsule can be consumed exactly once.

    Requires numpy for the host dialect — install via
    ``pip install lucid[numpy]`` if missing.
    """
    if tensor.is_metal:
        return tensor.__dlpack__()
    _require_numpy("lucid.to_dlpack")
    return tensor.numpy().__dlpack__()
