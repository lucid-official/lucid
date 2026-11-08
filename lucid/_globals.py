import threading
from lucid._dtype import dtype, float32 as _f32
from lucid._device import device as _dev

_lock: threading.Lock = threading.Lock()
_default_dtype: dtype = _f32
_default_device_str: str = "cpu"


def set_default_dtype(d: dtype, /) -> None:
    """Set the dtype used by factory functions when none is supplied.

    Affects ``lucid.tensor``, ``lucid.zeros``, ``lucid.ones``,
    ``lucid.randn`` and every other constructor that defaults to
    ``dtype=None``.  Existing tensors are not converted.  Thread-safe
    (guarded by a module-level lock); the change becomes visible to
    every thread on the next factory call.

    Parameters
    ----------
    d : lucid.dtype
        New default — typically :data:`lucid.float32` (the framework
        default) or :data:`lucid.float64` for higher-precision research
        runs.  Also invalidates the dispatch-side default-dtype cache
        so subsequent kwargs normalisation re-resolves the engine enum.

    Raises
    ------
    TypeError
        If ``d`` is not a dtype.  Checked here rather than at first use:
        this writes process-wide state that every later factory call
        reads, so an unusable value has to be refused at the point it
        was passed.  Stored unchecked, it surfaced as a failure inside
        an unrelated ``lucid.tensor`` call an arbitrary distance away.
    """
    if not isinstance(d, dtype):
        raise TypeError(f"Default dtype must be a lucid.dtype, got {d!r}")
    global _default_dtype
    with _lock:
        _default_dtype = d
    # 3.2.2: invalidate the dispatch-side default-dtype cache so that
    # subsequent ``normalize_factory_kwargs`` calls re-resolve the
    # engine enum.  Imported lazily to avoid circular dependency at
    # module load time.
    from lucid._dispatch import _invalidate_default_dtype_cache

    _invalidate_default_dtype_cache()


def get_default_dtype() -> dtype:
    """Return the dtype currently used by factory functions.

    Reads the same global the dispatch path consults — so a freshly
    constructed ``lucid.zeros(3)`` (with no explicit ``dtype``) lands
    on the value returned here.

    Returns
    -------
    lucid.dtype
        Current default dtype.  Initialised to :data:`lucid.float32`;
        mutated by :func:`set_default_dtype`.
    """
    return _default_dtype


def set_default_device(device: _dev | str) -> None:
    """Set the device used by factory functions when none is supplied.

    Same scope as :func:`set_default_dtype` but for placement.
    Existing tensors are not moved; subsequent factory calls observe
    the new default on the next invocation.  Thread-safe under a
    module-level lock.  Invalidates two dispatch-side caches — the
    converter fast path and the kwargs normaliser — so the next
    factory call re-resolves placement from scratch.

    Parameters
    ----------
    device : lucid.device or str
        New default placement.  Either a :class:`~lucid._device.device`
        instance or a string like ``"cpu"`` / ``"metal"``.

    Raises
    ------
    TypeError
        If ``device`` is neither a device nor a string.
    ValueError
        If the string does not name a device.

    Notes
    -----
    Both are checked here rather than at first use.  The previous
    ``d if isinstance(d, str) else d.type`` accepted *any* object with a
    ``.type`` attribute — a tensor has one, and passing one stored a
    bound method as the process-wide default, after which every
    subsequent factory call failed with an unparseable device and no
    trace of where the value came from.
    """
    if isinstance(device, _dev):
        resolved = device.type
    elif isinstance(device, str):
        resolved = _dev(device).type  # raises ValueError on an unknown name
    else:
        raise TypeError(f"Default device must be a lucid.device or str, got {device!r}")
    global _default_device_str
    with _lock:
        _default_device_str = resolved
    # Invalidate BOTH device caches:
    #   - ``lucid._factories.converters`` for the ndarray-fast-path
    #   - ``lucid._dispatch`` for ``normalize_factory_kwargs``
    # Imported lazily to avoid circular dependency at module load time.
    from lucid._factories.converters import _invalidate_default_device_cache as _c_inv
    from lucid._dispatch import _invalidate_default_device_cache as _d_inv

    _c_inv()
    _d_inv()


def get_default_device() -> _dev:
    """Return the default device used by factory functions.

    Mirror of :func:`get_default_dtype` for placement.  The result is a
    :class:`~lucid._device.device`, so it round-trips through
    :func:`set_default_device` and can be passed straight to ``.to`` or a
    ``device=`` argument.

    Returns
    -------
    device
        ``device("cpu")`` or ``device("metal")``.  Initialised to the CPU;
        mutated by :func:`set_default_device`.

    Examples
    --------
    >>> import lucid
    >>> lucid.get_default_device()
    device('cpu')
    """
    return _dev(_default_device_str)
