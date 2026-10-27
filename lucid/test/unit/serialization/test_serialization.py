"""``lucid.save`` / ``lucid.load`` round-trip + Module state_dict."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestSaveLoadTensor:
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.lucid"
            t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
            lucid.save(t, str(path))
            loaded = lucid.load(str(path))
            np.testing.assert_array_equal(loaded.numpy(), t.numpy())

    def test_dtype_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.lucid"
            t = lucid.tensor([1, 2, 3], dtype=lucid.int64)
            lucid.save(t, str(path))
            loaded = lucid.load(str(path))
            assert loaded.dtype == lucid.int64

    def test_pathlike_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            t = lucid.tensor([1.0, 2.0, 3.0])
            lucid.save({"x": t}, path)
            loaded = lucid.load(path)
            np.testing.assert_array_equal(loaded["x"].numpy(), t.numpy())

            st_path = Path(tmp) / "ckpt.safetensors"
            lucid.serialization.save_safetensors({"x": t}, str(st_path))
            loaded_st = lucid.load(st_path)
            np.testing.assert_array_equal(loaded_st["x"].numpy(), t.numpy())


class TestSaveLoadStateDict:
    def test_module_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.lucid"
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
            sd = model.state_dict()
            lucid.save(sd, str(path))

            model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
            loaded = lucid.load(str(path), weights_only=False)
            model2.load_state_dict(loaded)

            # Compare any one parameter — they must now be equal.
            for k, v in sd.items():
                v2 = dict(model2.state_dict())[k]
                np.testing.assert_allclose(
                    v.numpy(), v2.numpy(), atol=1e-6, err_msg=f"parameter {k} drifted on round-trip"
                )


class TestSaveLoadDict:
    def test_dict_of_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.lucid"
            d = {"a": lucid.tensor([1.0]), "b": lucid.tensor([2.0, 3.0])}
            lucid.save(d, str(path))
            loaded = lucid.load(str(path), weights_only=False)
            np.testing.assert_array_equal(loaded["a"].numpy(), [1.0])
            np.testing.assert_array_equal(loaded["b"].numpy(), [2.0, 3.0])


class TestStateDictV2:
    """state-dict-v2: _metadata version propagation + assign= parameter."""

    def test_metadata_version_in_simple_module(self) -> None:
        import lucid.nn as nn

        m = nn.Linear(3, 2)
        sd = m.state_dict()
        assert hasattr(sd, "_metadata")
        assert sd._metadata.get("")["version"] == 1  # type: ignore[index]

    def test_metadata_propagates_to_children(self) -> None:
        import lucid.nn as nn

        seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 3))
        sd = seq.state_dict()
        meta = sd._metadata
        for key in ("", "0", "1", "2"):
            assert key in meta, f"'{key}' missing from _metadata"
            assert meta[key]["version"] == 1

    def test_batchnorm_metadata_version_2(self) -> None:
        import lucid.nn as nn

        bn = nn.BatchNorm2d(4)
        sd = bn.state_dict()
        assert sd._metadata[""]["version"] == 2

    def test_load_state_dict_returns_incompatible_keys(self) -> None:
        import lucid.nn as nn

        m = nn.Linear(3, 2)
        sd = m.state_dict()
        result = m.load_state_dict(sd)
        assert result.missing_keys == []
        assert result.unexpected_keys == []

    def test_assign_false_shape_mismatch_raises(self) -> None:
        import lucid.nn as nn
        import lucid

        m = nn.Linear(3, 2)
        bad_sd = {"weight": lucid.randn(4, 3), "bias": lucid.zeros(4)}
        with pytest.raises(RuntimeError, match="size mismatch"):
            m.load_state_dict(bad_sd, strict=True, assign=False)

    def test_assign_true_shape_mismatch_raises(self) -> None:
        # assign=True swaps the tensors but, as in the reference, not their
        # shapes (LCD-499).
        import lucid.nn as nn
        import lucid

        m = nn.Linear(3, 2)
        bad_sd = {"weight": lucid.randn(4, 3), "bias": lucid.zeros(4)}
        with pytest.raises(RuntimeError, match="size mismatch"):
            m.load_state_dict(bad_sd, strict=True, assign=True)
        assert m.weight.shape == (2, 3)

    def test_assign_true_copies_values(self) -> None:
        import lucid.nn as nn
        import numpy as np

        src = nn.Linear(3, 2)
        dst = nn.Linear(3, 2)
        dst.load_state_dict(src.state_dict(), assign=True)
        np.testing.assert_allclose(dst.weight.numpy(), src.weight.numpy(), atol=1e-6)

    def test_metadata_round_trip_via_save_load(self) -> None:
        """_metadata survives lucid.save / lucid.load."""
        import lucid.nn as nn
        import lucid
        import io

        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
        sd = m.state_dict()
        buf = io.BytesIO()
        lucid.save(sd, buf)
        buf.seek(0)
        loaded = lucid.load(buf, weights_only=False)
        assert hasattr(loaded, "_metadata")
        assert loaded._metadata.get("")["version"] == 1  # type: ignore[index]


class TestParameterGradAndStorageSharing:
    """LCD-104 & LCD-67: Parameter class, requires_grad, tied weights, shared storage."""

    @staticmethod
    def _roundtrip(obj: object, *, weights_only: bool = True) -> object:
        import io

        buf = io.BytesIO()
        lucid.save(obj, buf)
        buf.seek(0)
        return lucid.load(buf, weights_only=weights_only)

    def test_parameter_class_and_requires_grad_preserved(self, device: str) -> None:
        p_grad = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
        p_nograd = nn.Parameter(lucid.tensor([4.0, 5.0], device=device), requires_grad=False)
        t_grad = lucid.tensor([6.0, 7.0], device=device, requires_grad=True)
        t_nograd = lucid.tensor([8.0, 9.0], device=device, requires_grad=False)

        for wo in (True, False):
            r = self._roundtrip(
                {"p_grad": p_grad, "p_nograd": p_nograd, "t_grad": t_grad, "t_nograd": t_nograd},
                weights_only=wo,
            )
            assert isinstance(r, dict)
            assert type(r["p_grad"]) is nn.Parameter
            assert isinstance(r["p_grad"], nn.Parameter)
            assert r["p_grad"].requires_grad is True
            assert r["p_grad"].tolist() == [1.0, 2.0, 3.0]

            assert type(r["p_nograd"]) is nn.Parameter
            assert r["p_nograd"].requires_grad is False
            assert r["p_nograd"].tolist() == [4.0, 5.0]

            assert type(r["t_grad"]) is lucid.Tensor
            assert r["t_grad"].requires_grad is True
            assert r["t_grad"].tolist() == [6.0, 7.0]

            assert type(r["t_nograd"]) is lucid.Tensor
            assert r["t_nograd"].requires_grad is False
            assert r["t_nograd"].tolist() == [8.0, 9.0]

    def test_tied_weights_preserve_identity_and_save_once(self, device: str) -> None:
        import io

        w = nn.Parameter(lucid.randn(64, 64, device=device))
        single_buf = io.BytesIO()
        lucid.save({"a": w}, single_buf)
        tied_buf = io.BytesIO()
        lucid.save({"a": w, "b": w}, tied_buf)

        # Tied tensor should not duplicate the raw 16 KiB weight payload.
        assert len(tied_buf.getvalue()) < len(single_buf.getvalue()) + 256

        tied_buf.seek(0)
        r = lucid.load(tied_buf)
        assert isinstance(r, dict)
        assert r["a"] is r["b"]
        assert type(r["a"]) is nn.Parameter
        assert r["a"].requires_grad is True
        with lucid.no_grad():
            r["a"][0, 0] = 123.0
        assert r["b"][0, 0].item() == 123.0

    def test_slice_and_base_share_storage_after_roundtrip(self, device: str) -> None:
        b = lucid.arange(6, dtype=lucid.float32, device=device)
        s = b[2:5]
        r = self._roundtrip({"base": b, "slice": s})
        assert isinstance(r, dict)
        assert r["base"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert r["slice"].tolist() == [2.0, 3.0, 4.0]

        r["base"][3] = 99.0
        assert r["slice"].tolist() == [2.0, 99.0, 4.0]
        r["slice"][0] = -7.0
        assert r["base"].tolist() == [0.0, 1.0, -7.0, 99.0, 4.0, 5.0]

    def test_overlapping_slices_share_storage_without_base(self, device: str) -> None:
        b = lucid.arange(6, dtype=lucid.float32, device=device)
        r = self._roundtrip({"a": b[1:4], "b": b[2:5]})
        assert isinstance(r, dict)
        r["a"][2] = 77.0
        assert r["b"][1].item() == 77.0

    def test_transposed_view_and_base_share_storage(self, device: str) -> None:
        b = lucid.arange(6, dtype=lucid.float32, device=device).reshape(2, 3)
        r = self._roundtrip({"base": b, "t": b.T})
        assert isinstance(r, dict)
        assert r["t"].shape == (3, 2)
        assert r["t"].stride() == (1, 3)
        r["base"][0, 2] = 55.0
        assert r["t"][2, 0].item() == 55.0

    def test_optimizer_state_dict_loads_with_weights_only_true(self, device: str) -> None:
        w = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
        opt = lucid.optim.Adam([w], lr=0.1)
        (w * 2.0).sum().backward()
        opt.step()

        loaded_sd = self._roundtrip(opt.state_dict(), weights_only=True)
        assert isinstance(loaded_sd, dict)

        w2 = nn.Parameter(w.detach().clone())
        opt2 = lucid.optim.Adam([w2], lr=0.1)
        opt2.load_state_dict(loaded_sd)

        w.grad = None
        w2.grad = None
        (w * 3.0).sum().backward()
        (w2 * 3.0).sum().backward()
        opt.step()
        opt2.step()
        assert w2.tolist() == pytest.approx(w.tolist(), rel=1e-6)


class TestWeightsOnlyWhitelistAndMapLocationContainers:
    """LCD-106 & LCD-107: weights_only=True whitelist and map_location containers."""

    @staticmethod
    def _save_load(
        obj: object, *, weights_only: bool = True, map_location: object = None
    ) -> object:
        import io

        buf = io.BytesIO()
        lucid.save(obj, buf)
        buf.seek(0)
        return lucid.load(
            buf, weights_only=weights_only, map_location=map_location  # type: ignore[arg-type]
        )

    def test_weights_only_allows_dtypes_devices_and_builtins(self) -> None:
        from collections import Counter, defaultdict

        dd: defaultdict[str, int] = defaultdict(int, {"a": 1})
        payload = {
            "dtypes": [
                lucid.float16,
                lucid.float32,
                lucid.float64,
                lucid.bfloat16,
                lucid.int8,
                lucid.int16,
                lucid.int32,
                lucid.int64,
                lucid.bool_,
                lucid.complex64,
                lucid.complex128,
            ],
            "device_cpu": lucid.device("cpu"),
            "device_metal": lucid.device("metal"),
            "complex": 1 + 2j,
            "bytearray": bytearray(b"abc"),
            "set": {1, 2, 3},
            "slice": slice(1, 10, 2),
            "counter": Counter({"x": 3, "y": 1}),
            "defaultdict": dd,
        }
        loaded = self._save_load(payload, weights_only=True)
        assert isinstance(loaded, dict)
        for orig_dt, loaded_dt in zip(payload["dtypes"], loaded["dtypes"], strict=True):
            assert loaded_dt is orig_dt
            assert loaded_dt == orig_dt
            assert hash(loaded_dt) == hash(orig_dt)
        assert loaded["device_cpu"] == lucid.device("cpu")
        assert loaded["device_metal"] == lucid.device("metal")
        assert loaded["complex"] == 1 + 2j
        assert loaded["bytearray"] == bytearray(b"abc")
        assert loaded["set"] == {1, 2, 3}
        assert loaded["slice"] == slice(1, 10, 2)
        assert isinstance(loaded["counter"], Counter)
        assert loaded["counter"] == Counter({"x": 3, "y": 1})
        assert isinstance(loaded["defaultdict"], defaultdict)
        assert loaded["defaultdict"].default_factory is int
        assert loaded["defaultdict"]["missing"] == 0

    def test_weights_only_refuses_frozenset(self) -> None:
        import io
        import pickle

        buf = io.BytesIO()
        lucid.save({"fs": frozenset([1, 2, 3])}, buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="frozenset"):
            lucid.load(buf, weights_only=True)

    def test_map_location_preserves_ordered_dict_and_metadata(self) -> None:
        from collections import Counter, OrderedDict, defaultdict, namedtuple

        Pair = namedtuple("Pair", ["left", "right"])
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
        sd = m.state_dict()
        assert isinstance(sd, OrderedDict)
        assert hasattr(sd, "_metadata")

        for loc in ("cpu", {"cpu": "cpu", "metal": "cpu"}, lambda t, _d: t.to("cpu")):
            loaded_sd = self._save_load(sd, weights_only=True, map_location=loc)
            assert isinstance(loaded_sd, OrderedDict)
            assert hasattr(loaded_sd, "_metadata")
            assert loaded_sd._metadata == sd._metadata  # type: ignore[attr-defined]
            assert list(loaded_sd.keys()) == list(sd.keys())

        dd: defaultdict[str, lucid.Tensor] = defaultdict(lucid.Tensor)
        dd["x"] = lucid.tensor([1.0, 2.0])
        cnt: Counter[str] = Counter({"a": 2})
        nt = Pair(lucid.tensor([3.0]), lucid.tensor([4.0]))

        from lucid.serialization import _apply_map_location

        remapped = _apply_map_location(
            {"dd": dd, "cnt": cnt, "nt": nt, "s": {1, 2}}, lambda t, _d: t.to("cpu")
        )
        assert isinstance(remapped, dict)
        assert isinstance(remapped["dd"], defaultdict)
        assert remapped["dd"].default_factory is lucid.Tensor
        assert remapped["dd"]["x"].tolist() == [1.0, 2.0]
        assert isinstance(remapped["cnt"], Counter)
        assert remapped["cnt"]["a"] == 2
        assert isinstance(remapped["nt"], Pair)
        assert remapped["nt"].left.tolist() == [3.0]
        assert remapped["nt"].right.tolist() == [4.0]
        assert remapped["s"] == {1, 2}
