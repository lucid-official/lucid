"""The SafeTensors file format: dtype tables, header validation, tensor decoding.

The file is read and written here rather than through the ``safetensors``
package's numpy backend, which has no bfloat16 — the default dtype of most
language and diffusion checkpoints — and failed on every such file.

A file is a little-endian ``u64`` header length, a JSON header, then the
tensors' bytes.  :class:`_SafeTensorsFile` owns the one place a header is
trusted: every offset it hands out has been checked against the file, so a
damaged or hostile header is refused with the file and tensor named, before
any bytes are sliced.
"""

import array
import json
import math
import mmap
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor.tensor import Tensor as _T

# SafeTensors dtype codes for the dtypes Lucid holds natively.
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
# signed width, as :func:`lucid.from_numpy` does.  ``(array typecode of the
# stored width, array typecode of the signed width, its engine dtype)``.
_ST_UNSIGNED: dict[str, tuple[str, str, _C_engine.Dtype]] = {
    "U8": ("B", "h", _C_engine.Dtype.I16),
    "U16": ("H", "i", _C_engine.Dtype.I32),
    "U32": ("I", "q", _C_engine.Dtype.I64),
    "U64": ("Q", "q", _C_engine.Dtype.I64),
}
# Bytes per element, for every dtype whose stored length can be checked
# against its shape.  A dtype outside the table (the 8-bit floats) is refused
# by name when it is decoded, not here.
_ST_ITEMSIZE: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "BOOL": 1,
    "C64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}

_INT64_MAX = 2**63 - 1

#: Header key of the free-form string metadata.
_METADATA_KEY = "__metadata__"
#: Metadata key Lucid reserves for the names of its zero-rank tensors, which
#: are stored as shape ``(1,)`` so that readers without zero-rank support load
#: them.  A JSON list of names; files written before LCD-689 held the names
#: joined by commas, which could not carry a name containing one.
SCALAR_KEYS_KEY = "__lucid_scalar_keys__"


def encode_scalar_keys(names: list[str]) -> str:
    """The value :data:`SCALAR_KEYS_KEY` holds for ``names``: a compact JSON list.

    A list, not a joined string, so a name may contain any character
    including the separator.  :func:`decode_scalar_keys` reads it back.
    """
    return json.dumps(names, separators=(",", ":"))


def decode_scalar_keys(text: str) -> frozenset[str]:
    """The names a :data:`SCALAR_KEYS_KEY` value lists, in either format.

    The current format is a JSON list of strings.  Anything else is the
    legacy comma-joined list; a legacy value that happens to be a JSON list
    of strings would need a tensor named ``["a"]``, which no model has.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(k, str) for k in parsed):
        return frozenset(parsed)
    return frozenset(k for k in text.split(",") if k)


def checked_metadata(metadata: Mapping[str, object] | None) -> dict[str, str]:
    """A copy of the metadata a caller passed to ``save_safetensors``, checked.

    Raises
    ------
    TypeError
        If a value is not a string, which no SafeTensors reader accepts.
    ValueError
        If the caller claims :data:`SCALAR_KEYS_KEY`, which the writer owns.
    """
    meta: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if not isinstance(value, str):
            raise TypeError(
                f"save_safetensors: metadata maps strings to strings, got {key!r}: {value!r}"
            )
        meta[key] = value
    if SCALAR_KEYS_KEY in meta:
        raise ValueError(f"save_safetensors: metadata key {SCALAR_KEYS_KEY!r} is reserved")
    return meta


@dataclass(frozen=True)
class _Entry:
    """One tensor's header record, offsets relative to the data section."""

    name: str
    code: str
    shape: tuple[int, ...]
    start: int
    end: int


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _SafeTensorsFile:
    """A SafeTensors file whose header has been read and validated.

    Parameters
    ----------
    path : str or os.PathLike
        The file.  Only the header is read here; tensor bytes are read by
        :meth:`read` and :meth:`read_all`.

    Raises
    ------
    ValueError
        If the header is not what the format allows: unreadable, longer than
        the file, a record that is not ``{dtype, shape, data_offsets}``,
        offsets out of range, overlapping, gapped or not covering the data
        section, or a stored length that disagrees with the shape.  The
        message names the file and, where one is at fault, the tensor.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        header, size = self._read_header()
        self._payload = size - self.base
        raw_meta = self._take_metadata(header)
        self.scalar_keys = decode_scalar_keys(raw_meta.get(SCALAR_KEYS_KEY, ""))
        #: The caller's own metadata, with Lucid's reserved key removed.
        self.metadata: dict[str, str] = {k: v for k, v in raw_meta.items() if k != SCALAR_KEYS_KEY}
        self.entries: dict[str, _Entry] = {}
        for name, info in header.items():
            entry = self._record(name, info)
            self._check_bytes(entry)
            self.entries[name] = entry
        self._check_layout()

    def _error(self, why: str, name: str | None = None) -> ValueError:
        subject = f"tensor {name!r} in " if name is not None else ""
        return ValueError(f"SafeTensors file: {subject}{self.path!r}: {why}")

    def _read_header(self) -> tuple[dict[str, object], int]:
        """``(the JSON header, the file's size)``; sets :attr:`base`, where the data begins."""
        with open(self.path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise self._error("too short to be SafeTensors")
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len > size - 8:
                raise self._error(
                    f"its header claims {header_len} bytes but the file holds only {size - 8} "
                    "after the length prefix"
                )
            try:
                header = json.loads(fh.read(header_len))
            except ValueError as exc:
                raise self._error("has no readable header") from exc
        if not isinstance(header, dict):
            raise self._error("its header is not a JSON object")
        self.base = 8 + header_len
        return header, size

    def _take_metadata(self, header: dict[str, object]) -> dict[str, str]:
        """Remove and return the header's string metadata."""
        raw = header.pop(_METADATA_KEY, None) or {}
        if not isinstance(raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
        ):
            raise self._error(f"{_METADATA_KEY} must map strings to strings")
        return raw

    def _record(self, name: str, info: object) -> _Entry:
        """One header record, checked for form."""
        if not isinstance(info, dict):
            raise self._error("its record is not a JSON object", name)
        code, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if not isinstance(code, str):
            raise self._error("its record has no dtype string", name)
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise self._error(f"its shape {shape!r} is not a list of non-negative ints", name)
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(map(_is_int, offsets))):
            raise self._error(f"its data_offsets {offsets!r} are not two ints", name)
        return _Entry(name, code, tuple(shape), offsets[0], offsets[1])

    def _check_bytes(self, entry: _Entry) -> None:
        """``entry``'s bytes lie inside the data section and are as long as its shape needs."""
        if entry.start < 0 or entry.end < entry.start:
            raise self._error(
                f"its data_offsets [{entry.start}, {entry.end}] are not a forward range",
                entry.name,
            )
        if entry.end > self._payload:
            raise self._error(
                f"its bytes [{entry.start}, {entry.end}) run past the {self._payload}-byte "
                "data section (the file is truncated or the header is wrong)",
                entry.name,
            )
        itemsize = _ST_ITEMSIZE.get(entry.code)
        if itemsize is None:
            return
        needed = math.prod(entry.shape) * itemsize
        if entry.end - entry.start != needed:
            raise self._error(
                f"shape {list(entry.shape)} of {entry.code} needs {needed} bytes but its "
                f"data_offsets span {entry.end - entry.start}",
                entry.name,
            )

    def _check_layout(self) -> None:
        """The records tile the data section: no overlap, no gap, no trailing bytes."""
        cursor, previous = 0, None
        for entry in sorted(self.entries.values(), key=lambda e: (e.start, e.end)):
            if entry.start < cursor:
                raise self._error(
                    f"its bytes [{entry.start}, {entry.end}) overlap {previous!r}", entry.name
                )
            if entry.start > cursor:
                raise self._error(
                    f"{entry.start - cursor} bytes before it belong to no tensor", entry.name
                )
            cursor, previous = entry.end, entry.name
        if cursor != self._payload:
            raise self._error(f"{self._payload - cursor} trailing bytes follow the last tensor")

    def read(self, name: str, device: _C_engine.Device) -> _T:
        """One tensor, read from the file."""
        entry = self.entries[name]
        with open(self.path, "rb") as fh:
            fh.seek(self.base + entry.start)
            raw = fh.read(entry.end - entry.start)
        return self._decode(entry, raw, device)

    def read_all(self, device: _C_engine.Device) -> dict[str, _T]:
        """Every tensor, in header order."""
        result: dict[str, _T] = {}
        with open(self.path, "rb") as fh:
            if not self.entries:
                return result
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
                for name, entry in self.entries.items():
                    raw = data[self.base + entry.start : self.base + entry.end]
                    result[name] = self._decode(entry, raw, device)
        return result

    def _decode(self, entry: _Entry, raw: bytes, device: _C_engine.Device) -> _T:
        if len(raw) != entry.end - entry.start:
            raise self._error("its bytes are no longer in the file (it changed)", entry.name)
        tensor = self._tensor(entry, raw)
        if entry.name in self.scalar_keys and entry.shape == (1,):
            tensor = tensor.reshape(())
        if device != _C_engine.Device.CPU:
            tensor = tensor.to(device)
        return tensor

    def _tensor(self, entry: _Entry, raw: bytes) -> _T:
        """One entry as a Lucid tensor of its own dtype, on the CPU."""
        shape = list(entry.shape)
        engine_dtype = _ST_ENGINE.get(entry.code)
        if engine_dtype is not None:
            return _wrap(
                _C_engine.TensorImpl.from_bytes(
                    raw, shape, engine_dtype, _C_engine.Device.CPU, False
                )
            )
        widening = _ST_UNSIGNED.get(entry.code)
        if widening is None:
            raise self._error(f"stored as {entry.code}, which Lucid cannot represent", entry.name)
        stored, signed, target = widening
        values = array.array(stored)
        values.frombytes(raw)
        if entry.code == "U64" and values and max(values) > _INT64_MAX:
            raise self._error("holds uint64 values past int64's range", entry.name)
        wide = array.array(signed, values)
        return _wrap(
            _C_engine.TensorImpl.from_bytes(
                wide.tobytes(), list(entry.shape), target, _C_engine.Device.CPU, False
            )
        )
