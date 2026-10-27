"""Every way a traced function hands a tensor's values to Python.

Shared by the compile and Core ML tests of LCD-337: each entry reads a
value the result depends on — computed from the input, the input itself,
or the model's own parameter or buffer — and builds the result from it,
so a trace that kept the value it read would answer every later call
with the first one's result. LCD-345 added the reads that go through a
copy made outside the op dispatcher (a device move, a shared buffer) and
through bytes, pickle, ``lucid.save`` and DLPack.
"""

import copy
import io
import pickle
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class Apply(nn.Module):
    """``fn(self, x)``, with a parameter and a buffer the test may change."""

    def __init__(self, fn: Callable[[Apply, lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.scale = nn.Parameter(lucid.ones(()))
        self.register_buffer("offset", lucid.zeros(()))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(self, x)

    def change_state(self) -> None:
        """New weights, as an optimizer step or ``load_state_dict`` gives."""
        with lucid.no_grad():
            self.scale.copy_(lucid.full_like(self.scale, 5.0))
            offset = self.get_buffer("offset")
            offset.copy_(lucid.full_like(offset, 7.0))


def _numpy_total(t: lucid.Tensor) -> float:
    pytest.importorskip("numpy")
    return float(t.numpy().sum())


def _elsewhere(x: lucid.Tensor) -> str:
    """The device ``x`` is not on: a move there makes a copy."""
    return "cpu" if x.is_metal else "metal"


def _byte_total(blob: bytes) -> float:
    return float(sum(blob))


def _reloaded(x: lucid.Tensor) -> lucid.Tensor:
    # Read back rather than summed as bytes: the file also names the device.
    buffer = io.BytesIO()
    lucid.save(x, buffer)
    buffer.seek(0)
    loaded = lucid.load(buffer)
    assert isinstance(loaded, lucid.Tensor)
    return loaded


def _dlpack_total(x: lucid.Tensor) -> float:
    # The host dialect goes through NumPy; the Metal one needs nothing.
    if not x.is_metal:
        pytest.importorskip("numpy")
    return _list_total(lucid.from_dlpack(x))


def _list_total(t: lucid.Tensor) -> float:
    values = t.tolist()
    assert isinstance(values, list)
    return float(sum(v for v in values if isinstance(v, float)))


HOST_READS: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "item": lambda m, x: x * 0 + lucid.full_like(x, x.sum().item()),
    "tolist": lambda m, x: x * 0 + x.sum().tolist(),
    "numpy": lambda m, x: x * 0 + _numpy_total(x.sum()),
    "bool": lambda m, x: x * 2 if bool(x.sum() > 0) else x * 3,
    "if": lambda m, x: x * 2 if x.sum() > 0 else x * 3,
    "float": lambda m, x: x * 0 + float(x.sum()),
    "int": lambda m, x: x * 0 + int(x.sum()),
    # The argument itself, read before any operation has consumed it.
    "argument-tolist": lambda m, x: lucid.full_like(x, sum(x.tolist())) + x * 0,
    "argument-numpy": lambda m, x: lucid.full_like(x, _numpy_total(x)) + x * 0,
    # The model's own state, read before any operation has consumed it. A
    # Core ML package holds its weights as constants, so only the compile
    # path, which feeds them on every run, must not keep the value read.
    "parameter-item": lambda m, x: x * m.scale.item(),
    "buffer-item": lambda m, x: x + m.get_buffer("offset").item(),
    # A copy made outside the op dispatcher, then read (LCD-345): the trace
    # has to know the copy holds its value, or the read goes unnoticed.
    "moved-tolist": lambda m, x: x * 0 + _list_total(x.to(_elsewhere(x))),
    "moved-numpy": lambda m, x: x * 0 + _numpy_total(x.to(_elsewhere(x))),
    "detached-moved-tolist": lambda m, x: x * 0 + _list_total(x.detach().to(_elsewhere(x))),
    "shared-tolist": lambda m, x: x * 0 + _list_total(lucid.metal.to_shared(x)),
    "deepcopy": lambda m, x: x * 0 + _list_total(copy.deepcopy(x)),
    # The values handed out as bytes or a DLPack capsule.
    "to-bytes": lambda m, x: x * 0 + _byte_total(x._impl.to_bytes()),
    "pickle": lambda m, x: x * 0 + _list_total(pickle.loads(pickle.dumps(x))),
    "save": lambda m, x: x * 0 + _list_total(_reloaded(x)),
    "dlpack": lambda m, x: x * 0 + _dlpack_total(x),
}

_DOUBLE = """
#include <metal_stdlib>
using namespace metal;
kernel void dbl(
    device const float* inp [[buffer(0)]],
    device float* out       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = inp[gid] * 2.0f;
}
"""


def _metal_kernel(x: lucid.Tensor) -> lucid.Tensor:
    doubled = lucid.metal.run_kernel(_DOUBLE, "dbl", [x * 1], (16,), grid=(16, 1, 1))
    return x * 0 + doubled.to(x.device)


def _fused_linear(x: lucid.Tensor, fused: Callable[..., lucid.Tensor]) -> lucid.Tensor:
    # The weights are constants; the input is what the trace sees.
    weight = lucid.tensor(
        [[(i + j) % 5 - 2.0 for j in range(4)] for i in range(3)], device=x.device
    )
    bias = lucid.tensor([0.5, -0.5, 1.0], device=x.device)
    rows = x.reshape(4, 4)
    out = fused(rows, weight, bias)
    return out * 2 + x.sum()


#: A tensor made outside the op dispatcher that reaches the result through an
#: op, not a host read. Unrecorded, it was a feed pinned at its trace-time
#: value. A compiled call holding one runs eagerly: no executable spans two
#: devices, hands out a shared buffer, or runs a custom Metal kernel.
DERIVED_VALUES: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "moved-result": lambda m, x: (x * 3).to(_elsewhere(x)),
    "moved-twice": lambda m, x: x.to(_elsewhere(x)).to(x.device) * 3,
    "shared-operand": lambda m, x: lucid.metal.to_shared(x * 3) * 2,
    "metal-kernel": lambda m, x: _metal_kernel(x),
}

#: The fused inference kernels (``nn.FusedLinear``). Their output was a
#: tensor the trace had never seen, so a Core ML package kept the example's
#: result for every input (LCD-345). The trace records them now, and both the
#: compile and the Core ML emitters translate them (LCD-450).
FUSED_KERNELS: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "fused-linear-relu": lambda m, x: _fused_linear(x, F.fused_linear_relu),
    "fused-linear-gelu": lambda m, x: _fused_linear(x, F.fused_linear_gelu),
}

#: Entries whose read is of the model's weights rather than of its input.
STATE_READS = frozenset({"parameter-item", "buffer-item"})


def _made_on_the_spot(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    return x * lucid.tensor([2.0, 3.0]).tolist()[1]


def _validation_only(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    with lucid.compile._host_check():
        if bool((x < -1e6).any()):
            raise ValueError("unreachable on these inputs")
    return x * 3


def _advanced_index(m: Apply, x: lucid.Tensor) -> lucid.Tensor:
    # Two coordinate indices computed from the input: the CPU checks each
    # against its own dim by reading traced values — only to decide whether
    # to raise (LCD-336).  ``x[rows, cols]`` on the 4 x 4 view is ``x``.
    n = x.shape[0]
    zero = (x * 0).long()
    rows = zero + lucid.tensor([i // 4 for i in range(n)], device=x.device)
    cols = zero + lucid.tensor([i % 4 for i in range(n)], device=x.device)
    return x.reshape(n // 4, 4)[rows, cols] * 3


#: Reads that leave the trace whole: each answers ``x * 3``. A tensor built
#: from Python numbers holds no input data, and a read under
#: ``lucid.compile._host_check()`` only decides whether to raise.
CONSTANT_READS: dict[str, Callable[[Apply, lucid.Tensor], lucid.Tensor]] = {
    "made-on-the-spot": _made_on_the_spot,
    "validation-only": _validation_only,
    "advanced-index-bounds": _advanced_index,
}


def inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    """Two inputs whose sums differ in sign, so every entry's answer moves."""
    lucid.manual_seed(0)
    first = -lucid.randn(16).abs() - 1
    second = lucid.randn(16).abs() * 7 + 3
    return first, second
