"""The overwrite scatter under ``lucid.compile`` (CHA-158).

``x[key] = v``, ``lucid.scatter`` and ``index_copy`` trace as one
``scatter_set`` op.  It had a forward emitter and no manual VJP, so a
training step through ``index_copy`` ran eagerly whenever MPSGraph's own
autodiff was ruled out; once the other two stopped tracing as a
gather-subtract-scatter_add they would have too.

The gradient is the reference's rule, a repeated index included: the
written positions of the base take none, and every source element gathers
the gradient of the position it was aimed at — though only one of the
elements aimed at a position survives the write, and on the GPU which one
is unspecified.  The rule does not depend on which.
"""

import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.symbolic_gate import graph_symbolic_safe
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor, to_metal


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


_needs_metal = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

# Row r writes columns ``_IDX[r]``; rows 0 and 2 name a column twice.
_IDX = [[5, 5, 1], [0, 2, 4], [3, 3, 3], [1, 0, 2]]


def _probe() -> lucid.Tensor:
    return lucid.tensor(np.arange(24.0, dtype=np.float32).reshape(4, 6) * 0.1 + 0.3).to(
        COMPILE_DEVICE
    )


class _Writes(nn.Module):
    """``base`` and ``src`` both from parameters; the batch norm in train mode
    rules out MPSGraph's autodiff, so the step needs the manual VJP."""

    def __init__(self, how: str) -> None:
        super().__init__()
        lucid.manual_seed(3)
        self.w = nn.Parameter((lucid.rand(4, 6) + 0.5).to(COMPILE_DEVICE))
        self.v = nn.Parameter((lucid.rand(4, 3) + 0.5).to(COMPILE_DEVICE))
        self.bn = nn.BatchNorm1d(6).to(COMPILE_DEVICE)
        self.how = how
        # Made here, not in ``forward``: a constant built inside the trace
        # on the CPU would make it a mixed-device trace, which runs eagerly.
        self.probe = _probe()
        self.idx = lucid.tensor(_IDX).to(COMPILE_DEVICE)
        self.rows = lucid.arange(4).reshape(4, 1).to(COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        base = self.bn(x) * self.w
        src = self.v * 2.0
        if self.how == "scatter":
            out = lucid.scatter(base, 1, self.idx, src)
        else:
            out = base + 0.0
            out[self.rows, self.idx] = src
        return (out * self.probe).sum()


def _grads(model: _Writes) -> tuple[np.ndarray, np.ndarray]:
    assert model.w.grad is not None and model.v.grad is not None
    return model.w.grad.numpy().copy(), model.v.grad.numpy().copy()


@_needs_metal
@pytest.mark.parametrize("how", ["scatter", "setitem"])
def test_a_compiled_step_through_the_overwrite_has_eagers_gradient(how: str) -> None:
    x = metal_tensor(4, 6)
    model = _Writes(how)
    model(x).backward()
    want_w, want_v = _grads(model)

    step = make_step(model, lambda out: out)
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        for _ in range(2):  # trace, then replay
            model.w.grad = model.v.grad = None
            step(x).backward()
    finally:
        sys.stderr = old
    fallbacks = step.eager_only  # type: ignore[attr-defined]
    assert not (
        fallbacks.snapshot() if hasattr(fallbacks, "snapshot") else fallbacks
    ), err.getvalue()
    got_w, got_v = _grads(model)
    np.testing.assert_allclose(got_v, want_v, rtol=1e-5)
    np.testing.assert_allclose(got_w, want_w, rtol=1e-4, atol=1e-5)

    # The rule, written out: every element aimed at a position takes that
    # position's gradient, the duplicates included.
    probe = _probe().numpy()
    rows = np.arange(4)[:, None]
    np.testing.assert_allclose(got_v, 2.0 * probe[rows, np.array(_IDX)], rtol=1e-5)
    written = np.zeros((4, 6), bool)
    written[rows, np.array(_IDX)] = True
    assert np.all(got_w[written] == 0.0)


def _graph(*ops: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(ops=list(ops))


def _op(name: str, inputs: list[int], out: int, **attrs: object) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, inputs=inputs, outputs=[SimpleNamespace(id=out, shape=(8, 5))], attrs=attrs
    )


class TestTheSymbolicGateSeesTheOverwrite:
    """A write along the batch axis needs the traced batch's index: the gate
    has to route it to per-shape compiles.  ``x[key] = v`` always scatters
    along axis 0 of the flattened tensor."""

    def test_along_the_batch_axis_is_unsafe(self) -> None:
        graph = _graph(_op("scatter_set", [0, 1, 2], 3, dim=0))
        assert graph_symbolic_safe(graph, trace_batch=8, batch_ids={0}) is False

    def test_along_another_axis_is_safe(self) -> None:
        graph = _graph(_op("scatter_set", [0, 1, 2], 3, dim=1))
        assert graph_symbolic_safe(graph, trace_batch=8, batch_ids={0}) is True

    @_needs_metal
    def test_a_model_that_writes_into_its_input_compiles_per_shape(self) -> None:
        class _ZeroFirstColumn(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                y = x * 2.0
                y[:, 0] = 0.0
                return y

        m = to_metal(_ZeroFirstColumn()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 5)
            assert float((cm(x) - m(x)).abs().max().item()) == 0.0
        assert cm._symbolic_resolved is False  # type: ignore[attr-defined]
