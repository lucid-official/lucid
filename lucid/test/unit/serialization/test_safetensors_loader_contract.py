"""The SafeTensors and checkpoint loaders hold their inputs to account (LCD-689).

Three things were quietly wrong.  A device the loaders did not recognise
(``"Metal"``, ``"mps"``, ``"garbage"``, ``None``) loaded onto the CPU instead
of raising, because each loader tested the string for a substring.  A header
was trusted: a negative offset read the header region through Python's
negative slicing, overlapping or gapped records and trailing bytes loaded,
a header claiming 2**40 bytes of a tiny file returned ``{}``, and a short
payload failed with an engine message that named neither file nor tensor.
And the names of zero-rank tensors were stored comma-joined, so a name with
a comma came back as shape ``(1,)``, while the metadata a caller wrote could
not be read back at all.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import lucid
from lucid.serialization import (
    load,
    load_safetensors,
    load_sharded,
    safe_open,
    save,
    save_safetensors,
    save_sharded,
)
from lucid.test._fixtures.devices import metal_available

needs_metal = pytest.mark.skipif(not metal_available(), reason="Metal is unavailable")

#: Device specs ``lucid.device`` accepts, and the device each names.
VALID_SPECS = [
    pytest.param("cpu", "cpu", id="cpu"),
    pytest.param("cpu:0", "cpu", id="cpu-index"),
    pytest.param(lucid.device("cpu"), "cpu", id="device-cpu"),
    pytest.param("metal", "metal", id="metal", marks=needs_metal),
    pytest.param("metal:0", "metal", id="metal-index", marks=needs_metal),
    pytest.param(lucid.device("metal"), "metal", id="device-metal", marks=needs_metal),
]
#: Specs the loaders used to read as the CPU, or as Metal by luck of a substring.
INVALID_SPECS = [
    pytest.param("Metal", ValueError, id="capitalised"),
    pytest.param("mps", ValueError, id="mps"),
    pytest.param("garbage", ValueError, id="garbage"),
    pytest.param("gpu", ValueError, id="gpu"),
    pytest.param("metal:x", ValueError, id="bad-index"),
    pytest.param("", ValueError, id="empty"),
    pytest.param(3, TypeError, id="int"),
    pytest.param(None, TypeError, id="none"),
]


def _raw(path: Path, header: object, payload: bytes = b"", *, length: int | None = None) -> str:
    """A SafeTensors file written by hand: ``header`` verbatim, then ``payload``."""
    encoded = header if isinstance(header, bytes) else json.dumps(header).encode()
    prefix = struct.pack("<Q", len(encoded) if length is None else length)
    path.write_bytes(prefix + encoded + payload)
    return str(path)


def _entry(start: int, end: int, *, shape: list[int] | None = None, dtype: str = "F32") -> dict:
    count = (end - start) // 4
    return {
        "dtype": dtype,
        "shape": shape if shape is not None else [count],
        "data_offsets": [start, end],
    }


def _device_of(t: object) -> str:
    return "metal" if t.is_metal else "cpu"  # type: ignore[attr-defined]


# ── one device parser, shared by every loader ────────────────────────────────


class TestDeviceSpecs:
    @pytest.fixture
    def st_file(self, tmp_path: Path) -> str:
        path = str(tmp_path / "w.safetensors")
        save_safetensors({"w": lucid.ones(3), "s": lucid.tensor(2.0)}, path)
        return path

    @pytest.fixture
    def pickle_file(self, tmp_path: Path) -> str:
        path = str(tmp_path / "w.lucid")
        save({"w": lucid.ones(3)}, path)
        return path

    @pytest.mark.parametrize(("spec", "expected"), VALID_SPECS)
    def test_load_safetensors_places_tensors_on_the_named_device(
        self, st_file: str, spec: object, expected: str
    ) -> None:
        got = load_safetensors(st_file, device=spec)  # type: ignore[arg-type]
        assert {_device_of(v) for v in got.values()} == {expected}
        assert tuple(got["s"].shape) == ()  # type: ignore[attr-defined]

    @pytest.mark.parametrize(("spec", "expected"), VALID_SPECS)
    def test_safe_open_places_tensors_on_the_named_device(
        self, st_file: str, spec: object, expected: str
    ) -> None:
        assert _device_of(safe_open(st_file, device=spec).get_tensor("w")) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize(("spec", "expected"), VALID_SPECS)
    def test_load_map_location_places_tensors_on_the_named_device(
        self, st_file: str, pickle_file: str, spec: object, expected: str
    ) -> None:
        for path in (st_file, pickle_file):
            assert _device_of(load(path, map_location=spec)["w"]) == expected  # type: ignore[arg-type]

    @needs_metal
    def test_a_dict_map_location_names_a_device_by_the_same_rules(self, pickle_file: str) -> None:
        got = load(pickle_file, map_location={"cpu": "metal:0"})
        assert _device_of(got["w"]) == "metal"

    @pytest.mark.parametrize(("spec", "error"), INVALID_SPECS)
    def test_load_safetensors_refuses_a_device_it_does_not_know(
        self, st_file: str, spec: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            load_safetensors(st_file, device=spec)  # type: ignore[arg-type]

    @pytest.mark.parametrize(("spec", "error"), INVALID_SPECS)
    def test_safe_open_refuses_a_device_it_does_not_know(
        self, st_file: str, spec: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            safe_open(st_file, device=spec)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("spec", "error"), [p for p in INVALID_SPECS if p.values[0] is not None]
    )
    def test_load_refuses_a_map_location_it_does_not_know(
        self, st_file: str, pickle_file: str, spec: object, error: type[Exception]
    ) -> None:
        for path in (st_file, pickle_file):
            with pytest.raises(error):
                load(path, map_location=spec)  # type: ignore[arg-type]

    def test_a_map_location_with_nothing_to_move_is_still_checked(self, tmp_path: Path) -> None:
        # No tensor in the file means no tensor would have tripped the parser.
        path = str(tmp_path / "empty.lucid")
        save({"n": 1}, path)
        with pytest.raises(ValueError, match="garbage"):
            load(path, map_location="garbage")
        with pytest.raises(ValueError, match="garbage"):
            load(path, map_location={"cpu": "garbage"})
        with pytest.raises(TypeError, match="map_location"):
            load(path, map_location=3)  # type: ignore[arg-type]

    def test_load_sharded_checks_its_map_location_too(self, tmp_path: Path) -> None:
        directory = str(tmp_path / "shards")
        save_sharded({"w": lucid.ones(3)}, directory)
        with pytest.raises(ValueError, match="garbage"):
            load_sharded(directory, map_location="garbage")
        assert _device_of(load_sharded(directory, map_location="cpu")["w"]) == "cpu"  # type: ignore[index]

    def test_a_pickle_naming_an_unknown_device_is_refused_not_read_as_cpu(self) -> None:
        from lucid.serialization import _resolve_target_device

        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_target_device("garbage", None)


# ── one header validator ─────────────────────────────────────────────────────

_F32X2 = np.arange(2, dtype=np.float32).tobytes()  # 8 bytes
_F32X4 = np.arange(4, dtype=np.float32).tobytes()  # 16 bytes

#: ``(id, header, payload, header length override, tensor named, message fragment)``.
MALFORMED = [
    ("negative-start", {"w": _entry(-8, 0, shape=[0])}, _F32X2, None, "w", "forward range"),
    ("backwards-range", {"w": _entry(8, 0, shape=[0])}, _F32X2, None, "w", "forward range"),
    (
        "overlap",
        {"a": _entry(0, 8), "b": _entry(4, 12)},
        _F32X4 + _F32X2,
        None,
        "b",
        "overlap",
    ),
    (
        "identical-ranges",
        {"a": _entry(0, 8), "b": _entry(0, 8)},
        _F32X2,
        None,
        "b",
        "overlap",
    ),
    (
        "gap",
        {"a": _entry(0, 8), "b": _entry(12, 20)},
        _F32X4 + _F32X2,
        None,
        "b",
        "belong to no tensor",
    ),
    ("leading-gap", {"a": _entry(4, 12)}, _F32X4, None, "a", "belong to no tensor"),
    ("trailing-bytes", {"a": _entry(0, 8)}, _F32X4, None, None, "8 trailing bytes"),
    ("past-the-end", {"a": _entry(0, 16)}, _F32X2, None, "a", "run past"),
    ("short-payload", {"a": _entry(0, 8), "b": _entry(8, 16)}, _F32X2, None, "b", "run past"),
    (
        "length-disagrees-with-shape",
        {"a": _entry(0, 8, shape=[3])},
        _F32X2,
        None,
        "a",
        "needs 12 bytes",
    ),
    ("negative-dimension", {"a": _entry(0, 8, shape=[-2])}, _F32X2, None, "a", "non-negative"),
    ("record-not-an-object", {"a": [0, 8]}, _F32X2, None, "a", "not a JSON object"),
    ("no-dtype", {"a": {"shape": [2], "data_offsets": [0, 8]}}, _F32X2, None, "a", "no dtype"),
    (
        "one-offset",
        {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0]}},
        _F32X2,
        None,
        "a",
        "two ints",
    ),
    (
        "float-offset",
        {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0.0, 8]}},
        _F32X2,
        None,
        "a",
        "two ints",
    ),
    ("header-is-a-list", [], b"", None, None, "not a JSON object"),
    ("metadata-not-strings", {"__metadata__": {"k": 1}}, b"", None, None, "strings to strings"),
    ("huge-header-length", {}, b"", 2**40, None, "header claims"),
    ("header-longer-than-file", {"a": _entry(0, 8)}, _F32X2, 10_000, None, "header claims"),
    ("unreadable-header", b"{not json", b"", None, None, "no readable header"),
    ("empty-header", b"", b"", None, None, "no readable header"),
]


class TestHeaderValidation:
    @pytest.mark.parametrize(
        ("header", "payload", "length", "tensor", "fragment"),
        [pytest.param(*row[1:], id=row[0]) for row in MALFORMED],
    )
    def test_a_malformed_file_is_refused_with_the_file_and_tensor_named(
        self,
        tmp_path: Path,
        header: object,
        payload: bytes,
        length: int | None,
        tensor: str | None,
        fragment: str,
    ) -> None:
        path = _raw(tmp_path / "bad.safetensors", header, payload, length=length)
        for reader in (load_safetensors, safe_open):
            with pytest.raises(ValueError, match=fragment) as caught:
                reader(path)
            assert repr(path) in str(caught.value)
            if tensor is not None:
                assert f"tensor {tensor!r}" in str(caught.value)

    def test_a_file_shorter_than_its_length_prefix_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x01\x00")
        with pytest.raises(ValueError, match="too short") as caught:
            load_safetensors(str(path))
        assert repr(str(path)) in str(caught.value)

    def test_a_file_the_writer_made_passes_every_check(self, tmp_path: Path) -> None:
        path = str(tmp_path / "ok.safetensors")
        sd = {
            "a": lucid.randn(3, 4),
            "empty": lucid.zeros(0, 5),
            "b": lucid.ones(2).to(lucid.bfloat16),
            "i": lucid.arange(4),
        }
        save_safetensors(sd, path)
        back = load_safetensors(path)
        assert list(back) == list(sd)
        for name, t in sd.items():
            assert back[name].dtype == t.dtype and tuple(back[name].shape) == tuple(t.shape)
            assert bytes(back[name]._impl.to_bytes()) == bytes(t._impl.to_bytes())

    def test_an_empty_state_dict_round_trips(self, tmp_path: Path) -> None:
        path = str(tmp_path / "none.safetensors")
        save_safetensors({}, path)
        assert load_safetensors(path) == {}
        assert safe_open(path).keys() == []

    def test_a_failed_read_leaves_no_partial_result(self, tmp_path: Path) -> None:
        # A damaged tail must not hand back the tensors before it.
        path = _raw(
            tmp_path / "tail.safetensors",
            {"a": _entry(0, 8), "b": _entry(8, 24)},
            _F32X2,
        )
        with pytest.raises(ValueError, match="tensor 'b'"):
            load_safetensors(path)


# ── metadata ─────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_metadata_written_can_be_read_back(self, tmp_path: Path) -> None:
        path = str(tmp_path / "m.safetensors")
        meta = {"format": "lucid", "step": "12", "ünï": "çode"}
        save_safetensors({"w": lucid.ones(2), "s": lucid.tensor(1.0)}, path, metadata=meta)
        with safe_open(path) as f:
            assert f.metadata() == meta
            assert f.keys() == ["w", "s"]

    def test_a_file_without_metadata_reads_none(self, tmp_path: Path) -> None:
        path = str(tmp_path / "plain.safetensors")
        save_safetensors({"w": lucid.ones(2)}, path)
        assert safe_open(path).metadata() is None

    def test_lucids_own_bookkeeping_is_not_the_callers_metadata(self, tmp_path: Path) -> None:
        # Only a zero-rank tensor makes the writer add a private key.
        path = str(tmp_path / "scalar.safetensors")
        save_safetensors({"s": lucid.tensor(1.0)}, path)
        assert safe_open(path).metadata() is None
        save_safetensors({"s": lucid.tensor(1.0)}, path, metadata={"k": "v"})
        assert safe_open(path).metadata() == {"k": "v"}

    def test_the_metadata_a_reference_writer_stored_is_read(self, tmp_path: Path) -> None:
        path = _raw(
            tmp_path / "ref.safetensors",
            {"__metadata__": {"format": "pt"}, "a": _entry(0, 8)},
            _F32X2,
        )
        assert safe_open(path).metadata() == {"format": "pt"}

    def test_the_reference_reader_sees_the_same_metadata(self, tmp_path: Path) -> None:
        reference = pytest.importorskip("safetensors")
        path = str(tmp_path / "x.safetensors")
        save_safetensors({"s": lucid.tensor(1.0)}, path, metadata={"k": "v"})
        with reference.safe_open(path, framework="numpy") as f:
            assert f.metadata()["k"] == "v"

    def test_metadata_that_is_not_strings_is_refused_at_save(self, tmp_path: Path) -> None:
        path = str(tmp_path / "bad.safetensors")
        with pytest.raises(TypeError, match="strings to strings"):
            save_safetensors({}, path, metadata={"k": 1})  # type: ignore[dict-item]

    def test_the_reserved_key_cannot_be_claimed(self, tmp_path: Path) -> None:
        path = str(tmp_path / "bad.safetensors")
        with pytest.raises(ValueError, match="reserved"):
            save_safetensors({}, path, metadata={"__lucid_scalar_keys__": "x"})

    def test_asking_for_a_missing_tensor_names_it(self, tmp_path: Path) -> None:
        path = str(tmp_path / "w.safetensors")
        save_safetensors({"w": lucid.ones(2)}, path)
        with pytest.raises(KeyError, match="nope"):
            safe_open(path).get_tensor("nope")


# ── zero-rank names ──────────────────────────────────────────────────────────

AWKWARD_NAMES = ["a,b", ",", "x,", ",y", '"q"', '["a"]', "[1]", "null", "w e i r d", "ünï,çode"]


class TestScalarKeys:
    @pytest.mark.parametrize("name", AWKWARD_NAMES)
    def test_a_zero_rank_tensor_keeps_its_rank_whatever_it_is_called(
        self, tmp_path: Path, name: str
    ) -> None:
        path = str(tmp_path / "s.safetensors")
        save_safetensors(
            {name: lucid.tensor(3.0), "c": lucid.tensor(4.0), "v": lucid.ones(1)}, path
        )
        back = load_safetensors(path)
        assert tuple(back[name].shape) == () and back[name].item() == 3.0
        assert tuple(back["c"].shape) == () and tuple(back["v"].shape) == (1,)

    def test_every_awkward_name_at_once(self, tmp_path: Path) -> None:
        path = str(tmp_path / "all.safetensors")
        sd = {n: lucid.tensor(float(i)) for i, n in enumerate(AWKWARD_NAMES)}
        sd["vec"] = lucid.ones(1)
        save_safetensors(sd, path)
        back = load_safetensors(path)
        assert all(tuple(back[n].shape) == () for n in AWKWARD_NAMES)
        assert tuple(back["vec"].shape) == (1,)

    def test_the_names_are_stored_as_a_json_list(self, tmp_path: Path) -> None:
        path = str(tmp_path / "j.safetensors")
        save_safetensors({"a,b": lucid.tensor(1.0), "c": lucid.tensor(2.0)}, path)
        raw = Path(path).read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        stored = json.loads(raw[8 : 8 + n])["__metadata__"]["__lucid_scalar_keys__"]
        assert json.loads(stored) == ["a,b", "c"]

    @pytest.mark.parametrize(
        ("legacy", "scalars"),
        [
            pytest.param("s,t", ["s", "t"], id="two"),
            pytest.param("s", ["s"], id="one"),
            pytest.param("[1]", ["[1]"], id="looks-like-json-but-is-not-a-string-list"),
            pytest.param("null", ["null"], id="looks-like-json-null"),
        ],
    )
    def test_a_file_in_the_old_comma_format_still_loads(
        self, tmp_path: Path, legacy: str, scalars: list[str]
    ) -> None:
        header: dict[str, object] = {
            "__metadata__": {"__lucid_scalar_keys__": legacy, "kept": "yes"},
            "vec": _entry(0, 4, shape=[1]),
        }
        for i, name in enumerate(scalars):
            header[name] = _entry(4 * (i + 1), 4 * (i + 2), shape=[1])
        payload = np.arange(len(scalars) + 1, dtype=np.float32).tobytes()
        path = _raw(tmp_path / "old.safetensors", header, payload)
        back = load_safetensors(path)
        assert all(tuple(back[n].shape) == () for n in scalars)
        assert tuple(back["vec"].shape) == (1,)
        assert safe_open(path).metadata() == {"kept": "yes"}


class TestUnsignedWidening:
    def test_unsigned_values_widen_with_their_shape(self, tmp_path: Path) -> None:
        raw = struct.pack("<6Q", 0, 1, 2**63 - 1, 7, 8, 9)
        path = _raw(
            tmp_path / "u64.safetensors",
            {"u": {"dtype": "U64", "shape": [2, 3], "data_offsets": [0, 48]}},
            raw,
        )
        got = load_safetensors(path)["u"]
        assert (got.dtype, tuple(got.shape)) == (lucid.int64, (2, 3))
        assert got.tolist() == [[0, 1, 2**63 - 1], [7, 8, 9]]

    def test_a_uint64_past_int64_is_refused_by_name(self, tmp_path: Path) -> None:
        path = _raw(
            tmp_path / "big.safetensors",
            {"u": {"dtype": "U64", "shape": [1], "data_offsets": [0, 8]}},
            struct.pack("<Q", 2**63),
        )
        with pytest.raises(ValueError, match="tensor 'u'.*int64's range"):
            load_safetensors(path)
