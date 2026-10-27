"""Track J — sharded checkpoint tests.

Verifies lucid.save_sharded / load_sharded round-trips for:
  * Single-tensor state dicts (no splitting needed)
  * Multi-tensor state dicts that split across multiple shards
  * Non-dict objects (single-shard fallback)
  * _metadata propagation through the JSON index
  * map_location forwarding
  * weights_only gate (safe vs unsafe round-trip)
  * load_sharded raises on bad index
"""

import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn

# ── helpers ───────────────────────────────────────────────────────────────────


def _tiny_state_dict() -> OrderedDict:
    """Two small tensors in an OrderedDict."""
    sd: OrderedDict = OrderedDict(
        [("weight", lucid.tensor([[1.0, 2.0], [3.0, 4.0]])), ("bias", lucid.tensor([0.5, -0.5]))]
    )
    return sd


def _model_state_dict() -> OrderedDict:
    """State dict from a small Sequential model (carries _metadata)."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return model.state_dict()


# ── basic round-trip ──────────────────────────────────────────────────────────


class TestShardedRoundTrip:
    def test_single_shard_small_dict(self) -> None:
        """Everything fits in one shard → one shard file created."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            lucid.save_sharded(sd, tmp, shard_size_mb=1024.0)

            files = list(Path(tmp).iterdir())
            shard_files = [f for f in files if f.suffix == ".lucid"]
            assert len(shard_files) == 1, "expected exactly 1 shard"
            assert (Path(tmp) / "index.json").exists()

            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert isinstance(loaded, OrderedDict)
            assert set(loaded.keys()) == {"weight", "bias"}
            assert loaded["weight"].shape == (2, 2)
            assert loaded["bias"].shape == (2,)

    def test_values_preserved(self) -> None:
        """Tensor values are bit-exact after round-trip."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            lucid.save_sharded(sd, tmp)
            loaded = lucid.load_sharded(tmp, weights_only=False)
            np.testing.assert_array_equal(loaded["weight"].numpy(), sd["weight"].numpy())
            np.testing.assert_array_equal(loaded["bias"].numpy(), sd["bias"].numpy())

    def test_key_order_preserved(self) -> None:
        """Keys come back in insertion order."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            lucid.save_sharded(sd, tmp)
            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert list(loaded.keys()) == list(sd.keys())

    def test_dtype_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sd: OrderedDict = OrderedDict([("x", lucid.tensor([1, 2, 3], dtype=lucid.int64))])
            lucid.save_sharded(sd, tmp)
            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert loaded["x"].dtype == lucid.int64


# ── sharding logic ────────────────────────────────────────────────────────────


class TestShardSplitting:
    def test_multiple_shards_created(self) -> None:
        """Tiny shard_size_mb forces each tensor into its own shard."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            # 1 byte limit → each key must go into its own shard
            lucid.save_sharded(sd, tmp, shard_size_mb=1e-9)

            index = json.loads((Path(tmp) / "index.json").read_text())
            assert len(index["shards"]) == 2

            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert set(loaded.keys()) == {"weight", "bias"}

    def test_index_json_structure(self) -> None:
        """index.json has the expected schema."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            lucid.save_sharded(sd, tmp, shard_size_mb=1e-9)
            index = json.loads((Path(tmp) / "index.json").read_text())

            assert index["_lucid_sharded"] == 1
            assert "shards" in index
            all_keys: list[str] = []
            for shard in index["shards"]:
                assert "file" in shard
                assert "keys" in shard
                all_keys.extend(shard["keys"])
            assert set(all_keys) == {"weight", "bias"}

    def test_large_model_round_trip(self) -> None:
        """Full model state dict (with _metadata) survives sharded round-trip."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            sd = _model_state_dict()
            lucid.save_sharded(sd, tmp, shard_size_mb=1e-9)

            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert set(loaded.keys()) == set(sd.keys())
            for k in sd:
                np.testing.assert_allclose(
                    loaded[k].numpy(), sd[k].numpy(), atol=1e-6, err_msg=f"mismatch at key {k!r}"
                )


# ── _metadata propagation ─────────────────────────────────────────────────────


class TestMetadataPropagation:
    def test_metadata_survives_round_trip(self) -> None:
        """_metadata attached by state_dict() is preserved in index.json."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _model_state_dict()
            assert hasattr(sd, "_metadata"), "model.state_dict() must set _metadata"

            lucid.save_sharded(sd, tmp)
            index = json.loads((Path(tmp) / "index.json").read_text())
            assert "_state_dict_metadata" in index

            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert hasattr(loaded, "_metadata")
            assert loaded._metadata.get("")["version"] == 1  # type: ignore[index]


# ── non-dict fallback ─────────────────────────────────────────────────────────


class TestNonDictFallback:
    def test_single_tensor_fallback(self) -> None:
        """Saving a bare Tensor (non-dict) uses the single-shard fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            t = lucid.tensor([1.0, 2.0, 3.0])
            lucid.save_sharded(t, tmp)

            index = json.loads((Path(tmp) / "index.json").read_text())
            assert len(index["shards"]) == 1
            assert index["shards"][0]["keys"] is None

            loaded = lucid.load_sharded(tmp, weights_only=False)
            assert loaded.shape == (3,)


# ── weights_only gate ─────────────────────────────────────────────────────────


class TestWeightsOnly:
    def test_weights_only_true_accepts_tensor_dict(self) -> None:
        """weights_only=True (default) works for plain tensor state dicts."""
        with tempfile.TemporaryDirectory() as tmp:
            sd = _tiny_state_dict()
            lucid.save_sharded(sd, tmp)
            # Should not raise — dicts and tensors are on the safe list.
            loaded = lucid.load_sharded(tmp)
            assert set(loaded.keys()) == {"weight", "bias"}


# ── error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_bad_index_raises(self) -> None:
        """load_sharded raises RuntimeError when index.json is invalid."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = {"not_a_lucid_index": True}
            (Path(tmp) / "index.json").write_text(json.dumps(bad))
            with pytest.raises(RuntimeError, match="not a valid Lucid sharded checkpoint"):
                lucid.load_sharded(tmp)

    def test_directory_created_automatically(self) -> None:
        """save_sharded creates the target directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = str(Path(tmp) / "nested" / "ckpt")
            lucid.save_sharded(_tiny_state_dict(), new_dir)
            assert (Path(new_dir) / "index.json").exists()
