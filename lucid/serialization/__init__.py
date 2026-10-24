"""
lucid.serialization: save and load tensors and modules.

The wire format goes through the engine's ``to_bytes`` / ``from_bytes``
contract — no numpy round-trip, so checkpoints can be loaded in a
numpy-free environment.  Format version 3 records the dtype as the
canonical Lucid name (``"float32"``, ``"int64"``, …) instead of the
numpy ``str(arr.dtype)`` it used in v1/v2.
"""

import io
import os
import pickle
import warnings
from collections.abc import Mapping
from typing import Any, Callable, TYPE_CHECKING, cast, final, override

from lucid._tensor.tensor import Tensor as _T
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._dtype import _ENGINE_TO_DTYPE, _resolve_dtype_name, to_engine_dtype
from lucid.serialization._storage_plan import (
    _StoragePlan,
    _build_storage_plan,
    _contiguous_stride,
)

# ── Allowed types for weights_only=True ──────────────────────────────────────

_SAFE_CLASSES = frozenset(
    {
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.int",
        "builtins.float",
        "builtins.complex",
        "builtins.str",
        "builtins.bool",
        "builtins.bytes",
        "builtins.bytearray",
        "builtins.set",
        "builtins.slice",
        "builtins.NoneType",
        "collections.OrderedDict",
        "collections.Counter",
        "collections.defaultdict",
        "lucid._dtype.dtype",
        "lucid._dtype.float16",
        "lucid._dtype.float32",
        "lucid._dtype.float64",
        "lucid._dtype.bfloat16",
        "lucid._dtype.int8",
        "lucid._dtype.int16",
        "lucid._dtype.int32",
        "lucid._dtype.int64",
        "lucid._dtype.bool_",
        "lucid._dtype.complex64",
        "lucid._dtype.complex128",
        "lucid._device.device",
        "lucid._C.engine.Dtype",
        "lucid._C.engine.Device",
        # Optimizer state_dict() stores momentum/variance buffers and step
        # counters as NumPy arrays/scalars via _host_copy (LCD-67).
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray._reconstruct",
        "numpy._core.multiarray.scalar",
    }
)


def _wrap_loaded_impl(
    impl: _C_engine.TensorImpl, is_param: bool, requires_grad: bool
) -> _T:
    """Wrap ``impl`` as a :class:`Parameter` or :class:`Tensor` with ``requires_grad``."""
    if is_param:
        from lucid.nn.parameter import Parameter

        param = object.__new__(Parameter)
        param._impl = impl
        if requires_grad != impl.requires_grad:
            param.requires_grad_(requires_grad)
        return param
    t = _wrap(impl)
    if requires_grad != impl.requires_grad:
        t.requires_grad_(requires_grad)
    return t


# ── Pickler / Unpickler ───────────────────────────────────────────────────────


@final
class _LucidPickler(pickle.Pickler):
    def __init__(
        self, file: io.BytesIO, protocol: int | None, *, plan: _StoragePlan
    ) -> None:
        super().__init__(file, protocol=protocol)
        self._plan = plan
        self._emitted_tensors: dict[int, int] = {}
        self._emitted_storages: set[int] = set()

    @override
    def persistent_id(self, obj: object) -> object:
        if not isinstance(obj, _T):
            return None
        oid = id(obj)
        existing_tid = self._emitted_tensors.get(oid)
        if existing_tid is not None:
            return ("tensor_ref", existing_tid)

        tid = len(self._emitted_tensors)
        self._emitted_tensors[oid] = tid
        meta = self._plan.meta.get(oid)
        if meta is not None:
            shape, stride_elems, offset_elems, sid = meta
            raw_bytes: bytes | None = None
            if sid not in self._emitted_storages:
                self._emitted_storages.add(sid)
                raw_bytes = self._plan.blobs[sid]
        else:
            # A tensor the plan did not reach (inside an object's custom
            # reduce, say) is written dense: ``to_bytes`` packs it row-major,
            # whatever its own strides.
            shape = [int(d) for d in obj.shape]
            stride_elems = _contiguous_stride(shape)
            offset_elems = 0
            sid = -1 - tid
            raw_bytes = bytes(obj._impl.to_bytes())

        return (
            "tensor_v3",
            tid,
            shape,
            stride_elems,
            offset_elems,
            obj.dtype._name,
            "metal" if obj.is_metal else "cpu",
            bool(getattr(obj, "_is_parameter", False)),
            bool(obj.requires_grad),
            sid,
            raw_bytes,
        )


def _restore_tensor(
    shape: list[int], dtype_name: str, device_str: str, raw_bytes: bytes
) -> _T:
    eng_device = _C_engine.Device.GPU if device_str == "metal" else _C_engine.Device.CPU
    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    impl = _C_engine.TensorImpl.from_bytes(
        raw_bytes, list(shape), eng_dtype, eng_device, False
    )
    return _wrap(impl)


def _resolve_target_device(
    device_str: str,
    map_location: object,
) -> str:
    from lucid._device import device as _device_cls

    if isinstance(map_location, (str, _device_cls)):
        loc = str(map_location)
        return "metal" if ("metal" in loc or "gpu" in loc) else "cpu"
    if isinstance(map_location, dict):
        mapped = str(map_location.get(device_str, device_str))
        return "metal" if ("metal" in mapped or "gpu" in mapped) else "cpu"
    return device_str


def _load_v3_11tuple(
    pid: tuple[Any, ...],
    map_location: object,
    storages: dict[int, _C_engine.TensorImpl],
    tensors: dict[int, _T],
) -> _T:
    (
        _,
        tid,
        shape,
        stride_elems,
        offset_elems,
        dtype_name,
        device_str,
        is_param,
        requires_grad,
        sid,
        raw_bytes,
    ) = pid
    target_dev = _resolve_target_device(device_str, map_location)
    eng_device = _C_engine.Device.GPU if target_dev == "metal" else _C_engine.Device.CPU
    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    itemsize = _ENGINE_TO_DTYPE[eng_dtype].itemsize
    base_impl = storages.get(int(sid))
    if base_impl is None:
        # The first record of a storage carries its bytes; a later one names
        # it by id.  Anything else is a damaged file, refused here before
        # the engine is handed a length it would misread.
        if not isinstance(raw_bytes, (bytes, bytearray)) or len(raw_bytes) % itemsize:
            raise pickle.UnpicklingError(
                f"tensor record {tid}: storage {sid} has no bytes, or a byte "
                f"count that is not a multiple of {dtype_name}'s {itemsize}"
            )
        base_impl = _C_engine.TensorImpl.from_bytes(
            bytes(raw_bytes),
            [len(raw_bytes) // itemsize],
            eng_dtype,
            eng_device,
            False,
        )
        storages[int(sid)] = base_impl
    view_impl = _C_engine.TensorImpl._make_view(
        base_impl,
        [int(d) for d in shape],
        [int(s) for s in stride_elems],
        int(offset_elems),
    )
    tensor = _wrap_loaded_impl(view_impl, bool(is_param), bool(requires_grad))
    tensors[int(tid)] = tensor
    return tensor


class _LucidUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file: io.BytesIO,
        *,
        map_location: object = None,
        **pickle_load_args: Any,
    ) -> None:
        super().__init__(file, **pickle_load_args)
        self._map_location = map_location
        self._storages: dict[int, _C_engine.TensorImpl] = {}
        self._tensors: dict[int, _T] = {}

    @override
    def persistent_load(self, pid: object) -> object:
        if isinstance(pid, tuple) and pid:
            tag = pid[0]
            if tag == "tensor_ref":
                return self._tensors[int(pid[1])]
            if tag == "tensor_v3" and len(pid) == 11:
                return _load_v3_11tuple(
                    pid, self._map_location, self._storages, self._tensors
                )
            if tag == "tensor_v3" and len(pid) == 5:
                _, shape, dtype_name, device_str, raw_bytes = pid
                target_dev = _resolve_target_device(device_str, self._map_location)
                return _restore_tensor(shape, dtype_name, target_dev, raw_bytes)
            if tag == "tensor":
                _, shape, dtype_name, device_str, raw_bytes, _np_dtype_str = pid
                target_dev = _resolve_target_device(device_str, self._map_location)
                return _restore_tensor(shape, dtype_name, target_dev, raw_bytes)
        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


@final
class _SafeUnpickler(_LucidUnpickler):
    """Restricted Unpickler that only allows safe types (weights_only=True mode)."""

    @override
    def find_class(self, module: str, name: str) -> type:
        full = f"{module}.{name}"
        if full in _SAFE_CLASSES:
            return cast(type, super().find_class(module, name))
        raise pickle.UnpicklingError(
            f"weights_only=True: refusing to deserialize {full!r}. "
            "Pass weights_only=False to allow arbitrary objects "
            "(potential security risk)."
        )

    @override
    def persistent_load(self, pid: object) -> object:
        return super().persistent_load(pid)

    @override
    def load(self) -> object:
        res = super().load()
        _check_no_frozenset(res, set())
        return res


# ── Public API ────────────────────────────────────────────────────────────────


def save(
    obj: object,
    f: str | bytes | os.PathLike[str] | os.PathLike[bytes] | io.IOBase,
    pickle_module: object = pickle,
    pickle_protocol: int = 4,
) -> None:
    r"""Serialise ``obj`` to a Lucid-format checkpoint file.

    Writes a pickle stream augmented with a custom persistent-id
    protocol that captures every embedded :class:`Tensor` as a typed
    byte blob rather than a numpy array. The resulting ``.lucid`` file
    can be reloaded in environments without numpy installed.

    Parameters
    ----------
    obj : object
        Object graph to serialise. ``dict`` / ``OrderedDict`` state
        dicts are the primary use case; arbitrary picklable objects are
        also supported.
    f : str, bytes, os.PathLike, or file-like
        Destination path or open binary file handle.
    pickle_module : module, optional
        Module used for pickling metadata and objects. Only the standard
        :mod:`pickle` module is supported.
    pickle_protocol : int, optional
        Pickle protocol version forwarded to the underlying
        ``pickle.Pickler``. Default ``4``.

    Returns
    -------
    None

    Notes
    -----
    State dicts produced by ``Module.state_dict()`` carry a hidden
    ``_metadata`` attribute storing version information per submodule.
    That attribute is detected, packed alongside ``obj`` in the
    container, and reattached on :func:`load`.

    Every tensor comes back with exactly the values it held.  Tensors
    saved together that share a storage — a tensor and its views, or
    overlapping views — are written once and share it again after
    :func:`load`; a tensor that shares with nothing else saved is written
    dense, so a view saved alone does not carry the rest of its base.  On
    Metal only the views taken through Lucid's view ops are known to
    share; two tensors that merely read the same buffer are saved apart.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(3, 3)}
    >>> lucid.serialization.save(sd, "/tmp/ckpt.lucid")
    """
    if pickle_module not in (None, pickle):
        raise NotImplementedError("lucid.save supports only the standard pickle module")
    sd_metadata: object | None = getattr(obj, "_metadata", None)
    container: dict[str, object] = {"_lucid_format": 2, "obj": obj}
    if sd_metadata is not None:
        container["_state_dict_metadata"] = sd_metadata

    keep_alive, plan = _build_storage_plan(container)
    buf = io.BytesIO()
    pickler = _LucidPickler(buf, pickle_protocol, plan=plan)
    pickler.dump(container)
    del keep_alive
    data = buf.getvalue()

    if isinstance(f, (str, bytes, os.PathLike)):
        with open(os.fspath(f), "wb") as fp:
            fp.write(data)
    else:
        f.write(data)


def load(
    f: str | bytes | os.PathLike[str] | os.PathLike[bytes] | io.IOBase,
    map_location: (
        str | Callable[[str, str], str] | dict[str, str] | object | None
    ) = None,
    pickle_module: object = None,
    *,
    weights_only: bool | None = True,
    mmap: bool | None = None,
    **pickle_load_args: Any,
) -> object:
    r"""Load an object saved by :func:`save` or :func:`save_safetensors`.

    By default uses a restricted unpickler that only permits a small
    whitelist of safe primitive types — sufficient for state dicts but
    not for arbitrary objects. Disable this with ``weights_only=False``
    only for trusted files; arbitrary pickle deserialisation can execute
    code.

    Parameters
    ----------
    f : str, bytes, os.PathLike, or file-like
        Source path or open binary file handle. Paths ending in
        ``.safetensors`` are delegated to :func:`load_safetensors`.
    map_location : str, device, dict, callable, or None, optional
        Device remapping applied to every loaded tensor. A bare string or
        :class:`device` moves all tensors to that device; a ``dict`` maps
        source device names to targets; a callable receives
        ``(tensor, source_device)`` and returns the relocated tensor.
    pickle_module : module, optional
        Module used for unpickling metadata and objects. Only the standard
        :mod:`pickle` module is supported.
    weights_only : bool, optional
        If ``True`` (default), restrict deserialisation to a safe type
        whitelist. Set to ``False`` only for fully trusted checkpoints.
    mmap : bool, optional
        Compatibility flag with the reference signature. ``mmap=True`` is
        refused with :class:`NotImplementedError`.
    **pickle_load_args
        Additional keyword arguments forwarded to :class:`pickle.Unpickler`
        when ``weights_only=False``.

    Returns
    -------
    object
        The deserialised object, with ``_metadata`` reattached for
        state-dict round-trips.

    Notes
    -----
    Implements two formats: the current persistent-id format
    (``"tensor_v3"``) plus a backward-compatible path for v1/v2
    checkpoints that previously round-tripped through numpy.

    Examples
    --------
    >>> import lucid
    >>> lucid.serialization.save({"w": lucid.ones(3)}, "/tmp/ckpt.lucid")
    >>> sd = lucid.serialization.load("/tmp/ckpt.lucid")
    >>> sd["w"].tolist()
    [1.0, 1.0, 1.0]
    """
    safe_only = _validate_load_options(
        pickle_module, weights_only=weights_only, mmap=mmap, extra=pickle_load_args
    )
    if isinstance(f, (str, bytes, os.PathLike)):
        raw_path = os.fspath(f)
        path_str = raw_path.decode() if isinstance(raw_path, bytes) else str(raw_path)
        if path_str.endswith(".safetensors"):
            return load_safetensors(path_str)
        with open(raw_path, "rb") as fp:
            data = fp.read()
    else:
        data = f.read()

    buf = io.BytesIO(data)
    if not safe_only:
        warnings.warn(
            "lucid.load() with weights_only=False allows arbitrary object "
            "deserialization which may execute untrusted code.",
            UserWarning,
            stacklevel=2,
        )
    unpickler_cls = _SafeUnpickler if safe_only else _LucidUnpickler
    unpickler = unpickler_cls(buf, map_location=map_location, **pickle_load_args)
    container = unpickler.load()
    unpickler._storages.clear()
    unpickler._tensors.clear()
    if not isinstance(container, dict) or container.get("_lucid_format") not in (1, 2):
        raise RuntimeError("File is not a valid Lucid checkpoint")

    obj = container["obj"]
    if callable(map_location) and not isinstance(map_location, str):
        obj = _apply_map_location(obj, map_location)
    return _with_state_dict_metadata(obj, container.get("_state_dict_metadata"))


def _validate_load_options(
    pickle_module: object | None,
    *,
    weights_only: bool | None,
    mmap: bool | None,
    extra: dict[str, object],
) -> bool:
    if pickle_module not in (None, pickle):
        raise NotImplementedError("lucid.load supports only the standard pickle module")
    if mmap is True:
        raise NotImplementedError("lucid.load does not support mmap=True")
    safe_only = True if weights_only is None else bool(weights_only)
    if safe_only and extra:
        raise RuntimeError("Cannot use ``pickle_load_args`` with ``weights_only=True``")
    return safe_only


def _with_state_dict_metadata(obj: object, sd_metadata: object) -> object:
    """Restore the ``_metadata`` attribute on state_dicts saved with format v2."""
    if sd_metadata is None or not isinstance(obj, dict):
        return obj
    try:
        obj._metadata = sd_metadata  # type: ignore[attr-defined]
    except AttributeError:
        from collections import OrderedDict as _OD

        new_obj = _OD(obj)
        new_obj._metadata = sd_metadata  # type: ignore[attr-defined]
        return new_obj
    return obj


def _check_no_frozenset(obj: object, seen: set[int]) -> None:
    """Refuse ``frozenset`` under ``weights_only=True`` (protocol 4+ uses opcode ``FROZENSET``)."""
    oid = id(obj)
    if oid in seen:
        return
    if isinstance(obj, frozenset):
        raise pickle.UnpicklingError(
            "weights_only=True: refusing to deserialize 'builtins.frozenset'. "
            "Pass weights_only=False to allow arbitrary objects "
            "(potential security risk)."
        )
    if isinstance(obj, dict):
        seen.add(oid)
        for k, v in obj.items():
            _check_no_frozenset(k, seen)
            _check_no_frozenset(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        seen.add(oid)
        for item in obj:
            _check_no_frozenset(item, seen)


def _remap_dict(
    obj: dict[object, object],
    map_location: str | Callable[..., Any] | dict[str, str],
    memo: dict[int, object],
) -> dict[object, object]:
    """Rebuild ``obj`` with remapped entries while preserving subtype and attrs."""
    from collections import defaultdict

    if type(obj) is dict:
        res_dict: dict[object, object] = {}
    elif isinstance(obj, defaultdict):
        res_dict = type(obj)(obj.default_factory)
    else:
        res_dict = type(obj)()
    memo[id(obj)] = res_dict
    for k, v in obj.items():
        res_dict[_apply_map_location(k, map_location, memo)] = _apply_map_location(
            v, map_location, memo
        )
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for attr_k, attr_v in obj_dict.items():
            setattr(res_dict, attr_k, _apply_map_location(attr_v, map_location, memo))
    return res_dict


def _remap_seq(
    obj: list[object] | tuple[object, ...] | set[object],
    map_location: str | Callable[..., Any] | dict[str, str],
    memo: dict[int, object],
) -> object:
    """Rebuild a sequence/set with remapped items while preserving namedtuple types."""
    import operator

    remapped = [_apply_map_location(v, map_location, memo) for v in obj]
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        res_seq: object = operator.methodcaller("_make", remapped)(type(obj))
    else:
        res_seq = type(obj)(remapped)
    memo[id(obj)] = res_seq
    return res_seq


def _apply_map_location(
    obj: object,
    map_location: str | Callable[..., Any] | dict[str, str],
    _memo: dict[int, object] | None = None,
) -> object:
    """Recursively apply map_location to all Tensors in obj."""
    if _memo is None:
        _memo = {}
    oid = id(obj)
    if oid in _memo:
        return _memo[oid]

    if isinstance(obj, _T):
        is_param = bool(getattr(obj, "_is_parameter", False))
        req_grad = bool(obj.requires_grad)
        if callable(map_location) and not isinstance(map_location, str):
            device_str = "metal" if obj.is_metal else "cpu"
            mapped = map_location(obj, device_str)
        elif isinstance(map_location, dict):
            device_str = "metal" if obj.is_metal else "cpu"
            target = map_location.get(device_str, device_str)
            mapped = obj.to(target)
        else:
            mapped = obj.to(map_location)
        if isinstance(mapped, _T):
            mapped = _wrap_loaded_impl(mapped._impl, is_param, req_grad)
        _memo[oid] = mapped
        return mapped

    if isinstance(obj, dict):
        return _remap_dict(obj, map_location, _memo)
    if isinstance(obj, (list, tuple, set)):
        return _remap_seq(obj, map_location, _memo)
    return obj


def _as_directory(path: object, name: str) -> str:
    """The checkpoint directory named by ``path``, or a refusal.

    The two sharded entry points used to reach their directory through a
    bare ``str(path)``, which is not a check but a coercion: anything at
    all has a ``__str__``, so a tensor passed by mistake became a
    directory literally named ``tensor([[1.225, 1.606, ...]])`` and the
    save reported success.  Nothing downstream can notice — the write
    succeeds, the index is valid, and the checkpoint is simply somewhere
    nobody will look for it.

    ``save`` and ``load`` never had the problem: both gate on
    ``isinstance(f, (str, bytes))`` and treat anything else as a file
    object, so a wrong argument fails at the first ``.write``.  This is
    the same gate, widened by ``os.PathLike`` because ``pathlib.Path``
    reaches here constantly and used to work through the coercion.
    """
    import os

    if isinstance(path, bytes):
        return path.decode()
    if isinstance(path, str):
        return path
    if isinstance(path, os.PathLike):
        return str(os.fspath(path))
    raise TypeError(
        f"{name}() expects a directory path as str, bytes or os.PathLike, "
        f"got {type(path).__name__}"
    )


def save_sharded(
    obj: object,
    path: str | bytes | os.PathLike[str],
    *,
    shard_size_mb: float = 1024.0,
    pickle_protocol: int = 4,
) -> None:
    r"""Save ``obj`` as a sharded, self-describing checkpoint directory.

    Splits a state dict across multiple shard files capped at
    ``shard_size_mb`` MiB each and emits a JSON index listing which keys
    live in which shard. Useful for very large models where a single
    monolithic file is impractical to host, partial-load, or
    download-resume.

    Parameters
    ----------
    obj : object
        Object to save. ``dict`` / ``OrderedDict`` state dicts are
        sharded; other types are written as a single shard for
        compatibility with :func:`load_sharded`.
    path : str, bytes or os.PathLike
        Destination *directory*. Created if it does not exist.
    shard_size_mb : float, optional
        Soft cap on per-shard size in MiB. A tensor exceeding the cap
        gets its own shard. Default ``1024.0``.
    pickle_protocol : int, optional
        Pickle protocol forwarded to every per-shard :func:`save`.
        Default ``4``.

    Returns
    -------
    None

    Notes
    -----
    The packing algorithm is a first-fit pass that never splits a
    single tensor across shards — preserving the property that any
    given key resolves to exactly one file. The index records the
    optional ``_metadata`` attribute when present on a state dict.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(4096, 4096)}
    >>> lucid.serialization.save_sharded(sd, "/tmp/ckpt_dir", shard_size_mb=64)
    """
    import json
    import os

    path_str = _as_directory(path, "save_sharded")
    os.makedirs(path_str, exist_ok=True)

    if not isinstance(obj, dict):
        fname = "shard-00000-of-00001.lucid"
        save(obj, os.path.join(path_str, fname), pickle_protocol=pickle_protocol)
        index: dict[str, object] = {
            "_lucid_sharded": 1,
            "shards": [{"file": fname, "keys": None}],
        }
        with open(os.path.join(path_str, "index.json"), "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2)
        return

    limit_bytes = int(shard_size_mb * 1024 * 1024)

    # Bucket (key, value) pairs into shards without splitting any single tensor.
    shards: list[dict[str, object]] = []
    cur: dict[str, object] = {}
    cur_bytes: int = 0

    for k, v in obj.items():
        v_sz = (v.numel() * v.element_size()) if isinstance(v, _T) else 0
        # Flush current shard if adding this tensor would exceed the limit
        # (but never emit an empty shard — always include at least one entry).
        if cur and cur_bytes + v_sz > limit_bytes:
            shards.append(cur)
            cur = {}
            cur_bytes = 0
        cur[k] = v
        cur_bytes += v_sz

    if cur:
        shards.append(cur)
    if not shards:
        shards = [{}]

    n = len(shards)
    index_shards: list[dict[str, object]] = []
    for i, shard_dict in enumerate(shards):
        fname = f"shard-{i:05d}-of-{n:05d}.lucid"
        save(shard_dict, os.path.join(path_str, fname), pickle_protocol=pickle_protocol)
        index_shards.append({"file": fname, "keys": list(shard_dict.keys())})

    index = {"_lucid_sharded": 1, "shards": index_shards}
    sd_meta = getattr(obj, "_metadata", None)
    if sd_meta is not None:
        try:
            # Only attach metadata that actually round-trips through the
            # json.dump below — a bare assignment can't fail, so validate
            # serializability here; otherwise a non-serializable value would
            # corrupt the whole index write further down.
            json.dumps(sd_meta)
            index["_state_dict_metadata"] = sd_meta
        except (TypeError, ValueError):  # fmt: skip
            # Non-JSON-serializable _metadata is omitted from index.json; shards still load.
            pass

    with open(os.path.join(path_str, "index.json"), "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)


def load_sharded(
    path: str | bytes | os.PathLike[str],
    *,
    map_location: str | Callable[[str, str], str] | dict[str, str] | None = None,
    weights_only: bool = True,
) -> object:
    r"""Load a sharded checkpoint directory written by :func:`save_sharded`.

    Reads ``index.json``, then deserialises each shard with :func:`load`
    and merges the results into a single ``OrderedDict`` preserving key
    order. The ``_metadata`` attribute, if stored in the index, is
    reattached to the merged dictionary.

    Parameters
    ----------
    path : str or bytes
        Directory containing ``index.json`` and the per-shard files.
    map_location : str, dict, callable, or None, optional
        Device remapping forwarded to each per-shard :func:`load` call.
    weights_only : bool, optional
        If ``True`` (default), restrict deserialisation to a safe type
        whitelist.

    Returns
    -------
    OrderedDict or object
        The merged state dict; for single-shard non-dict objects, the
        underlying object as written.

    Notes
    -----
    Shard order is taken from the index file — the function does not
    rely on filesystem listing order, so reproducibility is preserved
    across hosts.

    Examples
    --------
    >>> import lucid
    >>> lucid.serialization.save_sharded({"w": lucid.ones(3)}, "/tmp/ckpt_dir")
    >>> sd = lucid.serialization.load_sharded("/tmp/ckpt_dir")
    >>> sd["w"].tolist()
    [1.0, 1.0, 1.0]
    """
    import json
    import os
    from collections import OrderedDict

    path_str = _as_directory(path, "load_sharded")
    index_path = os.path.join(path_str, "index.json")

    with open(index_path, encoding="utf-8") as fp:
        index = json.load(fp)

    if not isinstance(index, dict) or index.get("_lucid_sharded") != 1:
        raise RuntimeError(
            f"{index_path!r} is not a valid Lucid sharded checkpoint index"
        )

    shards_meta: list[dict[str, object]] = index["shards"]

    # Single non-dict shard (fallback path written for non-dict objects).
    if len(shards_meta) == 1 and shards_meta[0].get("keys") is None:
        fname = str(shards_meta[0]["file"])
        return load(
            os.path.join(path_str, fname),
            map_location=map_location,
            weights_only=weights_only,
        )

    result: OrderedDict[str, object] = OrderedDict()
    for shard_meta in shards_meta:
        fname = str(shard_meta["file"])
        shard = load(
            os.path.join(path_str, fname),
            map_location=map_location,
            weights_only=weights_only,
        )
        if isinstance(shard, dict):
            result.update(shard)

    sd_meta = index.get("_state_dict_metadata")
    if sd_meta is not None and isinstance(sd_meta, dict):
        result._metadata = sd_meta  # type: ignore[attr-defined]

    return result


# SafeTensors dtype codes for the dtypes Lucid holds natively.  The file is
# read and written here rather than through the ``safetensors`` package's
# numpy backend, which has no bfloat16 — the default dtype of most language
# and diffusion checkpoints — and failed on every such file.
_ST_CODES: dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "bool": "BOOL",
    "complex64": "C64",
}
_ST_ENGINE: dict[str, _C_engine.Dtype] = {
    "F64": _C_engine.Dtype.F64,
    "F32": _C_engine.Dtype.F32,
    "F16": _C_engine.Dtype.F16,
    "BF16": _C_engine.Dtype.BF16,
    "I64": _C_engine.Dtype.I64,
    "I32": _C_engine.Dtype.I32,
    "I16": _C_engine.Dtype.I16,
    "I8": _C_engine.Dtype.I8,
    "BOOL": _C_engine.Dtype.Bool,
    "C64": _C_engine.Dtype.C64,
}
# Unsigned integers have no Lucid dtype; each widens losslessly to the next
# signed width, as :func:`lucid.from_numpy` does.
_ST_UNSIGNED: dict[str, tuple[str, str]] = {
    "U8": ("<u1", "int16"),
    "U16": ("<u2", "int32"),
    "U32": ("<u4", "int64"),
    "U64": ("<u8", "int64"),
}


def save_safetensors(
    state_dict: Mapping[str, object],
    path: str,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    r"""Save a flat state dict as a SafeTensors file.

    SafeTensors is the recommended interchange format for sharing model
    weights: the file layout is a small JSON header followed by raw
    tensor bytes, so loading is fast, zero-copy where possible, and free
    of pickle's code-execution surface area. Use this whenever a
    checkpoint may be shared with untrusted parties.

    Parameters
    ----------
    state_dict : Mapping of str to Tensor
        Flat mapping from parameter name to :class:`Tensor` — a ``dict``,
        or any read-only mapping such as a ``dict[str, Tensor]`` typed as
        one.  Nested dicts or non-tensor values raise ``TypeError``.
    path : str
        Destination file path; use a ``.safetensors`` suffix by
        convention.
    metadata : dict of str to str, optional
        Free-form string metadata stored in the file header (model
        version, training framework, etc.).

    Returns
    -------
    None

    Notes
    -----
    Every Lucid dtype is written as itself, bfloat16 included, so a
    checkpoint round-trips bit for bit and other SafeTensors readers see
    the dtype it was trained in.  Zero-rank scalars are promoted to shape
    ``(1,)`` on write and squeezed back to ``()`` on load via a private
    metadata key.

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.randn(3, 3)}
    >>> lucid.serialization.save_safetensors(sd, "/tmp/weights.safetensors")
    """
    import json
    import struct

    header: dict[str, object] = {}
    blobs: list[bytes] = []
    scalar_keys: list[str] = []
    offset = 0
    for name, value in state_dict.items():
        if not isinstance(value, _T):
            raise TypeError(
                f"save_safetensors: state_dict[{name!r}] is "
                f"{type(value).__name__}, expected Tensor"
            )
        code = _ST_CODES.get(value.dtype._name)
        if code is None:
            raise TypeError(
                f"save_safetensors: tensor {name!r} has dtype {value.dtype._name}, "
                "which SafeTensors cannot store"
            )
        shape = [int(d) for d in value.shape]
        if not shape:
            # Written as (1,) and tagged, so readers that predate zero-rank
            # entries still load it.
            shape = [1]
            scalar_keys.append(name)
        blob = bytes(_host_bytes(value))
        header[name] = {
            "dtype": code,
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    combined_meta: dict[str, str] = dict(metadata) if metadata else {}
    if scalar_keys:
        combined_meta["__lucid_scalar_keys__"] = ",".join(scalar_keys)
    if combined_meta:
        header["__metadata__"] = combined_meta
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The data section starts 8-aligned, as the reference writer pads it.
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)


def _host_bytes(value: _T) -> bytes:
    """A tensor's elements as contiguous row-major bytes, on the host."""
    from lucid._dispatch import _unwrap

    return bytes(_unwrap(value).to_bytes())


def load_safetensors(
    path: str,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    r"""Load a SafeTensors file into a flat state dict.

    Reads the header and each tensor's bytes directly, builds a Lucid
    :class:`Tensor` of the stored dtype — bfloat16 included — optionally
    relocates it to the requested device, and restores any zero-rank
    scalars that were promoted to shape ``(1,)`` during
    :func:`save_safetensors`.

    Parameters
    ----------
    path : str
        Path to a ``.safetensors`` file.
    device : str, optional
        Target device for every loaded tensor: ``"cpu"`` (default) or
        ``"metal"``.

    Returns
    -------
    dict of str to Tensor
        Flat state dict suitable for ``model.load_state_dict()``.

    Raises
    ------
    ValueError
        If the file is not a well-formed SafeTensors file, or stores a
        dtype Lucid cannot represent (the 8-bit floats).

    Notes
    -----
    Unsigned integers have no Lucid dtype and widen losslessly: ``U8`` to
    int16, ``U16`` to int32, ``U32`` to int64 (``U64`` to int64, refused
    past its range).

    Examples
    --------
    >>> import lucid
    >>> sd = {"w": lucid.ones(3)}
    >>> lucid.serialization.save_safetensors(sd, "/tmp/weights.safetensors")
    >>> loaded = lucid.serialization.load_safetensors("/tmp/weights.safetensors")
    >>> loaded["w"].tolist()
    [1.0, 1.0, 1.0]
    """
    import json
    import mmap
    import struct

    import lucid as _lucid

    result: dict[str, object] = {}
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValueError(
                f"load_safetensors: {path!r} is too short to be SafeTensors"
            )
        (header_len,) = struct.unpack("<Q", prefix)
        try:
            header = json.loads(fh.read(header_len))
        except ValueError as exc:
            raise ValueError(
                f"load_safetensors: {path!r} has no readable header"
            ) from exc
        base = 8 + header_len
        meta: dict[str, str] = header.pop("__metadata__", None) or {}
        scalar_keys = set(meta.get("__lucid_scalar_keys__", "").split(",")) - {""}
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for name, info in header.items():
                start, end = (int(v) for v in info["data_offsets"])
                shape = [int(d) for d in info["shape"]]
                raw = data[base + start : base + end]
                t = _tensor_from_safetensors(name, str(info["dtype"]), shape, raw)
                if name in scalar_keys:
                    t = _lucid.squeeze(t)
                if device in ("metal", "gpu"):
                    t = t.to("metal")
                result[name] = t
    return result


def _tensor_from_safetensors(name: str, code: str, shape: list[int], raw: bytes) -> _T:
    """One SafeTensors entry as a Lucid tensor of its own dtype."""
    count = 1
    for d in shape:
        count *= d
    engine_dtype = _ST_ENGINE.get(code)
    if engine_dtype is not None:
        impl = _C_engine.TensorImpl.from_bytes(
            raw, shape, engine_dtype, _C_engine.Device.CPU, False
        )
        return _wrap(impl)
    widening = _ST_UNSIGNED.get(code)
    if widening is None:
        raise ValueError(
            f"load_safetensors: tensor {name!r} is stored as {code}, which Lucid "
            "cannot represent"
        )
    import numpy as _np

    from lucid._factories.converters import from_numpy as _from_numpy

    stored, target = widening
    arr = _np.frombuffer(raw, dtype=stored, count=count).reshape(shape)
    if code == "U64" and count and int(arr.max()) > _np.iinfo(_np.int64).max:
        raise ValueError(
            f"load_safetensors: tensor {name!r} holds uint64 values past int64's range"
        )
    return _from_numpy(arr.astype(target))


__all__ = [
    "save",
    "load",
    "save_sharded",
    "load_sharded",
    "save_safetensors",
    "load_safetensors",
]
