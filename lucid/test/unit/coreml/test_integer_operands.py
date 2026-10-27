"""Emitters for operations Lucid runs on integers have to accept int32.

Core ML does not promote. ``clip``, ``relu`` and ``relu6`` take floats
only, and a ``mul`` of an int32 by a float constant is refused, all when
the package is parsed — after the export reported success. Lucid runs
every one of them on integer tensors: ``lucid.histogram`` clamps its int64
bucket index, which is how this was first reached (LCD-338).

Each case casts a float input to an integer type in the middle of the
model, so every integer width Lucid has goes through the program body
(the package's own inputs are float32 or int32 only). The answer has to
match eager exactly: these are integers.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

_INTEGER_TYPES = ["int8", "int16", "int32", "int64"]

_OPERATIONS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "clamp": lambda t: lucid.clamp(t, -5, 5),
    "clip-lower-only": lambda t: lucid.clamp(t, 0, None),
    "clip-upper-only": lambda t: lucid.clamp(t, None, 3),
    "neg": lambda t: -t,
    "relu": lambda t: F.relu(t),
    "relu6": lambda t: F.relu6(t),
}


class _ThroughIntegers(nn.Module):
    def __init__(self, dtype: lucid.dtype, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.dtype = dtype
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        held = (x * 10).to(self.dtype)
        return self.fn(held).to(lucid.float32)


def _single(answer: object) -> lucid.Tensor:
    if isinstance(answer, lucid.Tensor):
        return answer
    assert isinstance(answer, dict) and len(answer) == 1
    return next(iter(answer.values()))


@pytest.mark.parametrize("dtype_name", _INTEGER_TYPES)
@pytest.mark.parametrize("op_name", list(_OPERATIONS))
def test_an_integer_operand_exports_and_predicts_exactly(
    op_name: str, dtype_name: str, tmp_path: object
) -> None:
    lucid.manual_seed(0)
    model = _ThroughIntegers(getattr(lucid, dtype_name), _OPERATIONS[op_name]).eval()
    x = lucid.randn(4, 8)
    want = model(x)
    exported = cml.export(model, x, f"{tmp_path}/{op_name}-{dtype_name}.mlpackage")
    got = _single(exported.predict(x))
    assert float((got - want).abs().max().item()) == 0.0


@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_an_integer_input_is_clamped_in_its_own_type(dtype_name: str, tmp_path: object) -> None:
    class Clamp(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return lucid.clamp(x, -5, 5)

    x = (lucid.randn(4, 8) * 10).to(getattr(lucid, dtype_name))
    want = Clamp()(x)
    exported = cml.export(Clamp().eval(), x, f"{tmp_path}/clamp-{dtype_name}.mlpackage")
    got = _single(exported.predict(x))
    assert got.dtype in (lucid.int32, lucid.int64)
    assert got.to(lucid.int64).tolist() == want.to(lucid.int64).tolist()


def test_a_histogram_exports_and_follows_its_input(tmp_path: object) -> None:
    """The reachable case: the bucket index is an int64 clamp."""

    class Histogram(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return lucid.histogram(x, bins=4)[0]

    lucid.manual_seed(0)
    first = lucid.randn(16)
    second = lucid.randn(16) * 7 + 3
    model = Histogram().eval()
    exported = cml.export(model, first, f"{tmp_path}/histogram.mlpackage")
    for x in (first, second):
        got = _single(exported.predict(x))
        assert got.tolist() == model(x).tolist()


#: Bounds past int32: the body is int32, so they saturate exactly. Written
#: as given, Core ML kept their low 32 bits — ``2**40`` became ``0``.
_WIDE_BOUNDS = [(0, 2**40), (-(2**40), 3), (0, 2**31), (-(2**31) - 1, 2**31)]


@pytest.mark.parametrize(("low", "high"), _WIDE_BOUNDS)
def test_a_bound_past_int32_saturates(low: int, high: int, tmp_path: object) -> None:
    model = _ThroughIntegers(lucid.int64, lambda t: lucid.clamp(t, low, high)).eval()
    lucid.manual_seed(0)
    x = lucid.randn(4, 8)
    want = model(x)
    exported = cml.export(model, x, f"{tmp_path}/wide.mlpackage")
    got = _single(exported.predict(x))
    assert got.tolist() == want.tolist()


def test_an_integer_constant_past_int32_is_refused(tmp_path: object) -> None:
    """A wider constant has no exact int32 form, so export says so."""

    class AddsWide(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("wide", lucid.tensor([2**40], dtype=lucid.int64))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            held = x.to(lucid.int64) + self.get_buffer("wide")
            return held.to(lucid.float32)

    with pytest.raises(ValueError, match="outside int32"):
        cml.export(AddsWide().eval(), lucid.randn(4), f"{tmp_path}/wide.mlpackage")


@pytest.mark.parametrize(("low", "high"), [(2**40, 2**41), (-(2**41), -(2**40))])
def test_a_clamp_whose_every_result_is_past_int32_is_refused(
    low: int, high: int, tmp_path: object
) -> None:
    model = _ThroughIntegers(lucid.int64, lambda t: lucid.clamp(t, low, high)).eval()
    with pytest.raises(ValueError, match="outside int32"):
        cml.export(model, lucid.randn(4, 8), f"{tmp_path}/past.mlpackage")
