"""Several entry points in one package, over one copy of the weights.

A decoder wants two: one that reads a whole prompt, one that reads a
single token. They are the same network, and shipping them as two
packages ships the weights twice — which for a language model is the
whole download.
"""

import os

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine

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
