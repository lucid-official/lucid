"""``save_sharded`` keeps keys that share a storage in one shard (LCD-657).

Each shard is its own file, so a storage written to two shards loads back as two
unrelated tensors and costs its bytes twice.  Keys that share a storage — a
tied weight, a view and its base — are therefore packed as one unit.
"""

import json
import tempfile
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn


def _shard_files(d: Path) -> list[Path]:
    return sorted(d.glob("shard-*.lucid"))


def _index(d: Path) -> dict:
    return json.loads((d / "index.json").read_text())


def _shard_of(d: Path, key: str) -> int:
    for i, s in enumerate(_index(d)["shards"]):
        if key in s["keys"]:
            return i
    raise AssertionError(key)


_KIB16 = 64 * 64 * 4


@pytest.mark.parametrize("shard_mb", [1e-6, 0.01, 0.03, 0.05, 1024.0])
def test_tie_across_would_be_boundary(shard_mb: float) -> None:
    w = nn.Parameter(lucid.randn(64, 64))
    sd = {"emb": w, **{f"f{i}": lucid.randn(64, 64) for i in range(3)}, "head": w}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ck"
        lucid.save_sharded(sd, d, shard_size_mb=shard_mb)
        assert _shard_of(d, "emb") == _shard_of(d, "head")
        got = lucid.load_sharded(d)
        assert list(got) == list(sd)  # key order survives the regrouping
        for k in sd:
            assert got[k].tolist() == sd[k].tolist()
        with lucid.no_grad():
            got["emb"][0, 0] = 77.0
        assert got["head"][0, 0].item() == 77.0
        assert got["f0"][0, 0].item() != 77.0


def test_view_and_base_in_one_shard() -> None:
    base = lucid.randn(64, 64)
    sd = {
        "base": base,
        "pad0": lucid.randn(64, 64),
        "pad1": lucid.randn(64, 64),
        "row": base[3],
        "pad2": lucid.randn(64, 64),
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ck"
        lucid.save_sharded(sd, d, shard_size_mb=0.02)
        assert _shard_of(d, "base") == _shard_of(d, "row")
        got = lucid.load_sharded(d)
        with lucid.no_grad():
            got["base"][3, 0] = 55.0
        assert got["row"][0].item() == 55.0
        assert got["row"].tolist() == got["base"][3].tolist()


def test_group_larger_than_shard_gets_its_own_shard() -> None:
    w = nn.Parameter(lucid.randn(64, 64))
    sd = {"a": lucid.randn(8), "emb": w, "b": lucid.randn(8), "head": w, "c": lucid.randn(8)}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ck"
        lucid.save_sharded(sd, d, shard_size_mb=0.001)  # ~1 KiB < 16 KiB
        shards = _index(d)["shards"]
        assert sorted(shards[_shard_of(d, "emb")]["keys"]) == ["emb", "head"]
        assert list(lucid.load_sharded(d)) == list(sd)


def test_file_not_doubled() -> None:
    w = nn.Parameter(lucid.randn(64, 64))
    sizes = []
    for second in (w, lucid.randn(64, 64)):
        sd = {"emb": w, "f0": lucid.randn(64, 64), "f1": lucid.randn(64, 64), "head": second}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "ck"
            lucid.save_sharded(sd, d, shard_size_mb=0.03)
            sizes.append(sum(f.stat().st_size for f in _shard_files(d)))
    assert sizes[0] < sizes[1] - _KIB16 // 2


def test_untied_layout_and_index_unchanged() -> None:
    sd = {f"t{i}": lucid.randn(64, 64) for i in range(5)}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ck"
        lucid.save_sharded(sd, d, shard_size_mb=0.04)  # two 16 KiB tensors per shard
        idx = _index(d)
        assert "key_order" not in idx
        assert [s["keys"] for s in idx["shards"]] == [["t0", "t1"], ["t2", "t3"], ["t4"]]
        assert list(lucid.load_sharded(d)) == list(sd)


def test_loads_a_checkpoint_written_before_grouping() -> None:
    # The previous writer: per-shard `save`, an index of file/keys only.
    a, b, c = lucid.randn(4, 4), lucid.randn(4), lucid.randn(2)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        lucid.save({"a": a, "b": b}, d / "shard-00000-of-00002.lucid")
        lucid.save({"c": c}, d / "shard-00001-of-00002.lucid")
        (d / "index.json").write_text(
            json.dumps(
                {
                    "_lucid_sharded": 1,
                    "shards": [
                        {"file": "shard-00000-of-00002.lucid", "keys": ["a", "b"]},
                        {"file": "shard-00001-of-00002.lucid", "keys": ["c"]},
                    ],
                }
            )
        )
        got = lucid.load_sharded(d)
        assert list(got) == ["a", "b", "c"]
        assert got["a"].tolist() == a.tolist() and got["c"].tolist() == c.tolist()


def test_new_checkpoint_keeps_the_index_fields_an_old_reader_needs() -> None:
    w = nn.Parameter(lucid.randn(8))
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ck"
        lucid.save_sharded({"a": w, "b": lucid.randn(8), "c": w}, d, shard_size_mb=1e-6)
        idx = _index(d)
        assert idx["_lucid_sharded"] == 1
        assert all({"file", "keys"} <= set(s) for s in idx["shards"])
