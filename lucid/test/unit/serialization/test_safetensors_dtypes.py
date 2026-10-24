"""SafeTensors files load and save in every dtype they carry, bfloat16 first.

``load_safetensors`` read through the ``safetensors`` package's numpy
backend, which has no bfloat16 — the default dtype of most language and
diffusion checkpoints on the Hub — so every such file failed with "data
type 'bfloat16' not understood", pretrained weights included (they load
through the same function).  ``save_safetensors`` refused bfloat16 too, and
``from_numpy`` refused unsigned arrays, which left no public way to load
bf16 bit patterns at all (reported while porting Self-Forcing).

The file is read and written directly now: each tensor comes back in its
stored dtype, bit for bit.  Unsigned integers, which Lucid has no dtype
for, widen losslessly to the next signed width.
"""

import struct
from pathlib import Path
from types import MappingProxyType
import json

import numpy as np
import pytest

import lucid
from lucid.serialization import load_safetensors, save_safetensors


def _write(path: str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """A SafeTensors file written by hand, independent of the code under test."""
    header: dict[str, object] = {}
    offset = 0
    blobs = []
    for name, (code, shape, raw) in tensors.items():
        header[name] = {
            "dtype": code,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _bf16_bits(values: np.ndarray) -> bytes:
    """bfloat16 is the top half of float32 — exact for values that fit it."""
    bits = values.astype(np.float32).view(np.uint32) >> 16
    return bits.astype(np.uint16).tobytes()


def test_a_bfloat16_file_loads_as_bfloat16(tmp_path) -> None:  # type: ignore[no-untyped-def]
    values = np.array([[1.0, -2.5, 0.15625], [3.0e38, -1.0e-3, 0.0]], dtype=np.float32)
    exact = (values.view(np.uint32) & 0xFFFF0000).view(np.float32)  # bf16-representable
    path = str(tmp_path / "bf16.safetensors")
    _write(path, {"w": ("BF16", [2, 3], _bf16_bits(exact))})
    loaded = load_safetensors(path)["w"]
    assert loaded.dtype == lucid.bfloat16
    np.testing.assert_array_equal(loaded.to(lucid.float32).numpy(), exact)


@pytest.mark.parametrize(
    "dtype",
    [
        lucid.bfloat16,
        lucid.float16,
        lucid.float32,
        lucid.float64,
        lucid.int8,
        lucid.int16,
        lucid.int32,
        lucid.int64,
        lucid.bool,
    ],
    ids=lambda d: str(d),
)
def test_every_dtype_round_trips_bit_for_bit(tmp_path, dtype) -> None:  # type: ignore[no-untyped-def]
    t = (lucid.randn(4, 3) * 50).to(dtype)
    path = str(tmp_path / "rt.safetensors")
    save_safetensors(
        {"t": t, "s": lucid.tensor(2.0).to(dtype)}, path, metadata={"k": "v"}
    )
    back = load_safetensors(path)
    assert back["t"].dtype == dtype and tuple(back["s"].shape) == ()
    assert bytes(back["t"]._impl.to_bytes()) == bytes(t._impl.to_bytes())


def test_any_read_only_mapping_saves(tmp_path: Path) -> None:
    # LCD-630: typed ``dict[str, object]``, it made a caller holding a
    # ``dict[str, Tensor]`` copy it, since a dict's value type is invariant.
    t = lucid.randn(2, 3)
    path = str(tmp_path / "mapping.safetensors")
    save_safetensors(MappingProxyType({"t": t}), path)
    assert bytes(load_safetensors(path)["t"]._impl.to_bytes()) == bytes(
        t._impl.to_bytes()
    )


def test_unsigned_entries_widen_losslessly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "u.safetensors")
    _write(
        path,
        {
            "u8": ("U8", [2], np.array([0, 255], np.uint8).tobytes()),
            "u16": ("U16", [2], np.array([1, 65535], np.uint16).tobytes()),
            "u32": ("U32", [1], np.array([4294967295], np.uint32).tobytes()),
        },
    )
    got = load_safetensors(path)
    assert (got["u8"].dtype, got["u8"].tolist()) == (lucid.int16, [0, 255])
    assert (got["u16"].dtype, got["u16"].tolist()) == (lucid.int32, [1, 65535])
    assert (got["u32"].dtype, got["u32"].tolist()) == (lucid.int64, [4294967295])


def test_an_eight_bit_float_is_refused_by_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "f8.safetensors")
    _write(path, {"q": ("F8_E4M3", [2], b"\x00\x01")})
    with pytest.raises(ValueError, match="F8_E4M3"):
        load_safetensors(path)


def test_from_numpy_widens_unsigned_arrays() -> None:
    for dtype, want in (
        (np.uint8, lucid.int16),
        (np.uint16, lucid.int32),
        (np.uint32, lucid.int64),
    ):
        arr = np.array([0, np.iinfo(dtype).max], dtype=dtype)
        t = lucid.from_numpy(arr)
        assert t.dtype == want and t.tolist() == arr.astype(np.int64).tolist()
    with pytest.raises(OverflowError):
        lucid.from_numpy(np.array([2**64 - 1], dtype=np.uint64))


def test_a_reference_writer_file_loads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    path = str(tmp_path / "np.safetensors")
    safetensors_numpy.save_file(
        {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, path
    )
    np.testing.assert_array_equal(
        load_safetensors(path)["a"].numpy(),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    # And a Lucid-written bfloat16 file reads back through the reference reader's header.
    save_safetensors(
        {"w": lucid.ones(2).to(lucid.bfloat16)}, str(tmp_path / "l.safetensors")
    )
    with open(tmp_path / "l.safetensors", "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    assert header["w"]["dtype"] == "BF16" and (8 + size) % 8 == 0
