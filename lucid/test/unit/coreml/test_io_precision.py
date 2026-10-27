"""The precision a package was written at, and the precision it talks in.

Two things a handle has to know about the file it opened. The body's
precision decides what a compute plan can say when the Neural Engine took
none of the work — and a reopened handle used to assume float32, so a
float16 program was told to export at float16. The interface's precision
decides what ``predict`` hands over and gets back: float32 either way by
default, with casts bracketing a float16 body, or float16 on both sides
for a caller that already holds half precision and wants neither
conversion.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.coreml import _build
from lucid.coreml._model import PlacementSummary

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


def _small() -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


def _half_package(tmp_path: object, name: str = "half") -> cml.CoreMLModel:
    return cml.export(
        _small(),
        lucid.randn(2, 8),
        f"{tmp_path}/{name}.mlpackage",
        precision=cml.Precision.FLOAT16,
        io_precision=cml.Precision.FLOAT16,
    )


class TestAFloat16Interface:
    def test_float16_in_and_out(self, tmp_path: object) -> None:
        model, x = _small(), lucid.randn(2, 8)
        with _half_package(tmp_path) as package:
            got = package.predict(x.half())
            assert got.dtype == lucid.float16
            assert got.shape == (2, 4)
            reference = model(x)
            scale = float(reference.abs().max().item())
            gap = float((got.float() - reference).abs().max().item())
            assert gap / scale < 1e-2

    def test_a_float32_input_is_converted_on_the_way_in(self, tmp_path: object) -> None:
        x = lucid.randn(2, 8)
        with _half_package(tmp_path) as package:
            from_single = package.predict(x)
            from_half = package.predict(x.half())
            assert from_single.dtype == lucid.float16
            assert bool((from_single == from_half).all().item())

    def test_verify_compares_at_the_eager_models_precision(self, tmp_path: object) -> None:
        x = lucid.randn(2, 8)
        with _half_package(tmp_path) as package:
            assert package.verify(_small(), x, relative=True) < 1e-2

    def test_the_default_interface_is_unchanged(self, tmp_path: object) -> None:
        x = lucid.randn(2, 8)
        with cml.export(
            _small(), x, f"{tmp_path}/body.mlpackage", precision=cml.Precision.FLOAT16
        ) as package:
            assert package.io_precision == "FLOAT32"
            assert package.predict(x).dtype == lucid.float32

    def test_a_reopened_package_keeps_it(self, tmp_path: object) -> None:
        _half_package(tmp_path).close()
        with cml.load(f"{tmp_path}/half.mlpackage") as reopened:
            assert reopened.io_precision == "FLOAT16"
            assert reopened.predict(lucid.randn(2, 8)).dtype == lucid.float16

    def test_several_entry_points_take_it_too(self, tmp_path: object) -> None:
        model, x = _small(), lucid.randn(2, 8)
        handles = cml.export_functions(
            {"batch": (model, x), "single": (model, x[:1])},
            f"{tmp_path}/fn.mlpackage",
            precision=cml.Precision.FLOAT16,
            io_precision=cml.Precision.FLOAT16,
        )
        try:
            assert handles["batch"].predict(x).dtype == lucid.float16
            assert handles["single"].predict(x[:1]).shape == (1, 4)
        finally:
            for handle in handles.values():
                handle.close()

    def test_around_a_float32_body_it_is_refused(self, tmp_path: object) -> None:
        with pytest.raises(ValueError, match="precision=Precision.FLOAT16"):
            cml.export(
                _small(),
                lucid.randn(2, 8),
                f"{tmp_path}/single.mlpackage",
                io_precision=cml.Precision.FLOAT16,
            )


class TestTheRecordedPrecision:
    @pytest.mark.parametrize("precision", [cml.Precision.FLOAT32, cml.Precision.FLOAT16])
    def test_load_reads_it_back(self, tmp_path: object, precision: cml.Precision) -> None:
        cml.export(
            _small(), lucid.randn(2, 8), f"{tmp_path}/m.mlpackage", precision=precision
        ).close()
        with cml.load(f"{tmp_path}/m.mlpackage") as reopened:
            assert reopened.precision == precision.value

    def test_a_package_that_does_not_record_it_is_unknown(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What a package from elsewhere — or from before — looks like."""
        monkeypatch.setattr(_build, "_PRECISION_KEY", "elsewhere.precision")
        monkeypatch.setattr(_build, "_IO_PRECISION_KEY", "elsewhere.io")
        cml.export(
            _small(), lucid.randn(2, 8), f"{tmp_path}/m.mlpackage", precision=cml.Precision.FLOAT16
        ).close()
        with cml.load(f"{tmp_path}/m.mlpackage") as reopened:
            assert reopened.precision == "UNKNOWN"
            assert reopened.io_precision == "UNKNOWN"
            assert reopened.predict(lucid.randn(2, 8)).dtype == lucid.float32


class TestWhatThePlanSays:
    """The note on a plan that reached no Neural Engine at all."""

    @staticmethod
    def _missed(precision: str) -> PlacementSummary:
        return PlacementSummary(
            [("conv", "CPU"), ("relu", "CPU"), ("const", "unknown")],
            precision=precision,
            units=cml.ComputeUnits.CPU_AND_NE,
        )

    def test_float32_is_named_when_it_is_the_cause(self) -> None:
        note = self._missed("FLOAT32").note
        assert "float32" in note and "FLOAT16" in note

    def test_a_float16_program_is_not_told_to_be_float16(self) -> None:
        note = self._missed("FLOAT16").note
        assert note
        assert "Precision.FLOAT16" not in note
        assert "too large" in note and "rank 4" in note

    def test_an_unrecorded_precision_is_not_guessed(self) -> None:
        note = self._missed("UNKNOWN").note
        assert "Precision.FLOAT16" not in note
        assert "does not record its precision" in note

    def test_a_reopened_float16_package_gets_the_neutral_note(self, tmp_path: object) -> None:
        """The reported case: a float16 package opened by ``load``."""
        _half_package(tmp_path).close()
        with cml.load(
            f"{tmp_path}/half.mlpackage", compute_units=cml.ComputeUnits.CPU_AND_NE
        ) as reopened:
            plan = reopened.compute_plan()
            if plan.total_compute == 0:
                pytest.skip("this macOS reports no compute plan")
            assert "because the program is float32" not in plan.note
