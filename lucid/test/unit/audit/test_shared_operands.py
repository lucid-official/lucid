"""An axis that runs one call twice gives each run operands of its own.

LCD-532: once ``Tensor.bitwise_xor_`` and its siblings existed (LCD-519),
the device axis reported four of them as "cpu and metal differ".  The
devices agreed; the axis ran the CPU call on the probe's own tensors —
``.to("cpu")`` of a CPU tensor is that tensor — so Metal was handed what
the CPU run had written and applied the op a second time.  The idempotent
ones (``bitwise_and_``, ``clip_``) passed, and float operands were copied
by the float64-to-float32 narrowing, which is why only these four showed.
The entry axis ran its spellings back to back on the same arguments too.
:meth:`~lucid.test.audit._specs.Call.fresh` is the one rule both use.
"""

import functools
import inspect

import numpy as np
import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context, DeviceAxis, EntryPointAxis
from lucid.test.audit._result import Status


@functools.cache
def _symbols() -> dict[str, _surface.Symbol]:
    return {s.qualname: s for s in _surface.enumerate_surface()}


def test_a_fresh_call_owns_its_tensor_operands() -> None:
    x = lucid.tensor(np.array([1, 2, 3], dtype=np.int32))
    w = lucid.tensor(np.array([0.5, 1.5], dtype=np.float32)).requires_grad_()
    call = _specs.Call([x, 2, "mode"], {"weight": w, "dim": 0}, primary=0)
    fresh = call.fresh()

    fresh.args[0].add_(10)
    assert x.numpy().tolist() == [1, 2, 3]
    assert fresh.args[1:] == [2, "mode"] and fresh.kwargs["dim"] == 0
    assert fresh.args[0].dtype == x.dtype
    assert fresh.kwargs["weight"] is not w
    assert fresh.kwargs["weight"].requires_grad and fresh.kwargs["weight"].is_leaf
    assert (fresh.primary, fresh.note) == (call.primary, call.note)


@pytest.mark.parametrize(
    "qualname",
    [
        "Tensor.bitwise_xor_",
        "Tensor.bitwise_not_",
        "Tensor.bitwise_left_shift_",
        "Tensor.bitwise_right_shift_",
    ],
)
def test_the_devices_are_compared_on_the_same_input(qualname: str) -> None:
    """Each of these changes its answer when applied twice."""
    if not _probe.metal_available():
        pytest.skip("needs metal")
    symbol = _symbols().get(qualname)
    if symbol is None:
        pytest.skip(f"{qualname} is not on this surface")
    finding = DeviceAxis().run(symbol, Context(metal=True))
    assert finding.status is Status.PASS, finding.detail


def test_each_spelling_is_called_on_operands_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two spellings of one op that adds to its input in place.  Handed
    the same tensor in turn, the second spelling adds to what the first
    wrote — or, while the first answer is still read through a NumPy
    view of that storage, is refused, and the cell went to SKIP."""

    def audit_twice_(input: Tensor) -> Tensor:
        return input.add_(1.0)

    def method(self: Tensor) -> Tensor:
        return self.add_(1.0)

    audit_twice_.__module__ = method.__module__ = "lucid"
    monkeypatch.setattr(lucid, "audit_twice_", audit_twice_, raising=False)
    monkeypatch.setattr(Tensor, "audit_twice_", method, raising=False)
    symbol = _surface.Symbol(
        "Tensor.audit_twice_", "tensor", "method", inspect.getattr_static(Tensor, "audit_twice_")
    )
    finding = EntryPointAxis().run(symbol, Context())
    assert finding.status is Status.PASS, finding.detail
