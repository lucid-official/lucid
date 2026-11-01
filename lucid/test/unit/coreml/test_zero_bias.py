"""A linear layer with no bias writes no bias constant.

``F.linear`` hands the kernel a ``zeros(out_features)`` when there is no
bias, because the kernel takes one.  Exported as it traced, that is a
constant per projection the model does not have: it bloats a bias-free
(LLaMA-style) network and is 12% of a LoRA adapter's package.
"""

import os
import struct

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

_ENTRY_MAGIC = 0xDEADBEEF


def _entries(package: str) -> list[int]:
    """Payload size of every entry in the package's weights file, in file order."""
    path = os.path.join(package, "Data", "com.apple.CoreML", "weights", "weight.bin")
    with open(path, "rb") as handle:
        raw = handle.read()
    sizes: list[int] = []
    at = 64
    while at < len(raw):
        magic, _dtype, size, offset = struct.unpack_from("<IIQQ", raw, at)
        assert magic == _ENTRY_MAGIC
        sizes.append(size)
        at = (offset + size + 63) // 64 * 64
    return sizes


def _agrees(model: nn.Module, package: str, x: lucid.Tensor) -> None:
    exported = cml.export(model, x, package, compute_units=cml.ComputeUnits.CPU_ONLY)
    try:
        want = model(x)
        gap = float((exported.predict(x) - want).abs().max().item())
        assert gap <= 1e-5 * max(1.0, float(want.abs().max().item()))
    finally:
        exported.close()


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 32, bias=False)
        self.down = nn.Linear(32, 8, bias=False)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.down(self.up(x).relu())


class TestBiasFreeLinear:
    def test_the_package_holds_the_weights_and_nothing_else(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        package = f"{tmp_path}/stack.mlpackage"
        _agrees(_Stack().eval(), package, lucid.randn(2, 16))
        assert sorted(_entries(package)) == sorted([4 * 16 * 32, 4 * 32 * 8])

    def test_a_bias_is_still_written(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        package = f"{tmp_path}/biased.mlpackage"
        _agrees(nn.Linear(16, 8).eval(), package, lucid.randn(2, 16))
        assert sorted(_entries(package)) == sorted([4 * 16 * 8, 4 * 8])

    def test_a_bias_that_is_all_zeros_is_left_out_too(self, tmp_path: object) -> None:
        """``x @ W.T + 0`` is ``x @ W.T``: the answer cannot differ."""
        lucid.manual_seed(0)
        layer = nn.Linear(16, 8).eval()
        with lucid.no_grad():
            layer.bias.fill_(0.0)
        package = f"{tmp_path}/zeroed.mlpackage"
        _agrees(layer, package, lucid.randn(2, 16))
        assert _entries(package) == [4 * 16 * 8]

    def test_a_computed_input_dependent_bias_is_kept(self, tmp_path: object) -> None:
        """Only constants qualify; a bias made from the input is a value the program needs."""

        class Computed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(lucid.randn(8, 16))

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return F.linear(x, self.weight, x[0, :8] * 2.0)

        lucid.manual_seed(0)
        _agrees(Computed().eval(), f"{tmp_path}/computed.mlpackage", lucid.randn(2, 16))

    def test_a_zero_that_something_else_reads_is_kept(self, tmp_path: object) -> None:
        """The same zeros tensor, read again past the bias slot, still has to exist."""

        class Reused(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(lucid.randn(8, 16))

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                zeros = lucid.zeros(8)
                return F.linear(x, self.weight, zeros) + zeros + 1.0

        lucid.manual_seed(0)
        _agrees(Reused().eval(), f"{tmp_path}/reused.mlpackage", lucid.randn(2, 16))

    def test_a_projection_that_is_not_a_constant_still_works(self, tmp_path: object) -> None:
        """The ``matmul`` form of linear takes the same two operands."""

        class Computed(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return F.linear(x, x[:8, :16] * 2.0)

        lucid.manual_seed(0)
        _agrees(Computed().eval(), f"{tmp_path}/matmul.mlpackage", lucid.randn(16, 16))

    def test_every_entry_point_of_a_package_drops_it(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        model = _Stack().eval()
        package = f"{tmp_path}/two.mlpackage"
        handles = cml.export_functions(
            {"one": (model, lucid.randn(1, 16)), "two": (model, lucid.randn(4, 16))}, package
        )
        for handle in handles.values():
            handle.close()
        assert sorted(_entries(package)) == sorted([4 * 16 * 32, 4 * 32 * 8])
