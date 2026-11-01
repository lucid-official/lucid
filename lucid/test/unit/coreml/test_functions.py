"""Several entry points in one package, over one copy of the weights.

A decoder wants two: one that reads a whole prompt, one that reads a
single token. They are the same network, and shipping them as two
packages ships the weights twice — which for a language model is the
whole download.
"""

import gc
import os

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.coreml._shared import Float as _Float
from lucid.coreml._shared import Shared as _Slots

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class Shared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x)


def _weight_bytes(package: str) -> int:
    return os.path.getsize(f"{package}/Data/com.apple.CoreML/weights/weight.bin")


class TestMultipleFunctions:
    def test_each_entry_point_runs_and_agrees(self, tmp_path: object) -> None:
        model = Shared().eval()
        handles = cml.export_functions(
            {"one": (model, lucid.randn(1, 64)), "two": (model, lucid.randn(4, 64))},
            str(tmp_path / "two.mlpackage"),
            default="one",
        )
        try:
            assert sorted(handles) == ["one", "two"]
            for name, rows in (("one", 1), ("two", 4)):
                x = lucid.randn(rows, 64)
                reference = model(x)
                got = handles[name].predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            for handle in handles.values():
                handle.close()

    def test_the_weights_are_written_once(self, tmp_path: object) -> None:
        """The reason to put them in one package rather than two."""
        model = Shared().eval()
        together = str(tmp_path / "together.mlpackage")
        handles = cml.export_functions(
            {"one": (model, lucid.randn(1, 64)), "two": (model, lucid.randn(4, 64))}, together
        )
        for handle in handles.values():
            handle.close()

        apart = str(tmp_path / "apart.mlpackage")
        single = cml.export(model, lucid.randn(1, 64), apart)
        single.close()

        assert _weight_bytes(together) == _weight_bytes(apart)

    def test_the_default_is_what_a_caller_gets_unasked(self, tmp_path: object) -> None:
        model = Shared().eval()
        package = str(tmp_path / "default.mlpackage")
        handles = cml.export_functions(
            {"one": (model, lucid.randn(1, 64)), "two": (model, lucid.randn(4, 64))},
            package,
            default="two",
        )
        for handle in handles.values():
            handle.close()

        # ``load`` names no function, so the package's own default decides.
        opened = cml.load(package)
        try:
            x = lucid.randn(4, 64)
            reference = model(x)
            got = opened.predict(x)
            assert got.shape == reference.shape
        finally:
            opened.close()


class TestMultipleFunctionRefusals:
    def test_no_functions_is_refused(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="at least one function"):
            cml.export_functions({}, str(tmp_path / "none.mlpackage"))

    def test_a_default_that_is_not_there_is_refused(self, tmp_path: object) -> None:
        model = Shared().eval()
        with pytest.raises(ValueError, match="not one of"):
            cml.export_functions(
                {"one": (model, lucid.randn(1, 64))},
                str(tmp_path / "bad.mlpackage"),
                default="missing",
            )


class _WithConstant(nn.Module):
    """A forward that builds a host constant (``k, 2k, 3k, 4k``) of its own."""

    def __init__(self, fc: nn.Linear, k: float) -> None:
        super().__init__()
        self.fc = fc
        self.k = k

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x) + lucid.tensor([self.k, 2 * self.k, 3 * self.k, 4 * self.k])


class TestSharingIsByIdentity:
    """A weight is shared when it is the same tensor, never when it is the same address.

    Function one's host constants are freed once it is traced, and the
    allocator hands their addresses to function two's, so an ``id()`` key
    gave a later function an earlier one's constant: a package that loads,
    predicts, and is silently wrong by the difference of the two.
    """

    def test_a_freed_tensor_does_not_lend_its_slot(self, tmp_path: object) -> None:
        """The deterministic form: no new tensor can alias a recorded one."""
        paths = _C_engine.coreml.prepare_package(str(tmp_path / "slots.mlpackage"))
        slots = _Slots(paths=paths, blob=_C_engine.coreml.BlobWriter(paths.weight_bin))
        first = lucid.tensor([1.0, 2.0])
        slots.place(_unwrap(first), _Float, (False,), lambda: _Float(111))
        del first
        gc.collect()

        keep = [lucid.tensor([float(i), 2.0 * i]) for i in range(2000)]
        placed = [slots.place(_unwrap(t), _Float, (False,), lambda: _Float(222)) for t in keep]
        assert all(slot == _Float(222) for slot in placed)

    def test_different_constants_in_different_functions_keep_their_values(
        self, tmp_path: object
    ) -> None:
        lucid.manual_seed(0)
        fc = nn.Linear(4, 4)
        ks = (1, 10, 100, 1000, 5, 7)
        models = {f"f{i}": _WithConstant(fc, k).eval() for i, k in enumerate(ks)}
        x = lucid.randn(2, 4)
        handles = cml.export_functions(
            {name: (model, x) for name, model in models.items()},
            str(tmp_path / "constants.mlpackage"),
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
        try:
            for name, model in models.items():
                got, want = handles[name].predict(x), model(x)
                gap = float((got - want).abs().max().item())
                assert gap < 1e-5 * max(1.0, float(want.abs().max().item())), (name, gap)
        finally:
            for handle in handles.values():
                handle.close()


class _Wide(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(256, 256)
        self.b = nn.Linear(256, 128)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.b(self.a(x).relu())


class TestCompressedWeightsAreWrittenOnce:
    """Every storage format goes through the one sharing path, not only float."""

    @pytest.mark.parametrize(
        "weights",
        [
            cml.WeightPrecision.FLOAT,
            cml.WeightPrecision.INT8,
            cml.Palettize(bits=4),
            cml.Sparsify(ratio=0.5),
        ],
        ids=["float", "int8", "palettize4", "sparsify50"],
    )
    def test_two_entry_points_weigh_what_one_does(
        self, tmp_path: object, weights: cml.WeightPrecision | cml.Palettize | cml.Sparsify
    ) -> None:
        lucid.manual_seed(1)
        model = _Wide().eval()
        alone = str(tmp_path / "alone.mlpackage")
        single = cml.export(model, lucid.randn(1, 256), alone, weights=weights)
        together = str(tmp_path / "together.mlpackage")
        handles = cml.export_functions(
            {"one": (model, lucid.randn(1, 256)), "two": (model, lucid.randn(4, 256))},
            together,
            weights=weights,
        )
        try:
            assert _weight_bytes(together) == _weight_bytes(alone)
            x = lucid.randn(1, 256)
            want = single.predict(x)
            gap = float((handles["one"].predict(x) - want).abs().max().item())
            assert gap <= 1e-4 * float(want.abs().max().item())
        finally:
            single.close()
            for handle in handles.values():
                handle.close()

    def test_compression_still_shrinks_the_shared_blob(self, tmp_path: object) -> None:
        """The comparison above is vacuous if the compressed form was never used."""
        model = _Wide().eval()
        sizes = {}
        for weights in (cml.WeightPrecision.FLOAT, cml.WeightPrecision.INT8):
            package = str(tmp_path / f"{weights.value}.mlpackage")
            handles = cml.export_functions(
                {"one": (model, lucid.randn(1, 256)), "two": (model, lucid.randn(4, 256))},
                package,
                weights=weights,
            )
            for handle in handles.values():
                handle.close()
            sizes[weights] = _weight_bytes(package)
        assert sizes[cml.WeightPrecision.INT8] < 0.35 * sizes[cml.WeightPrecision.FLOAT]
