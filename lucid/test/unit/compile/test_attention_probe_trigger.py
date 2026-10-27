"""Regression: the attention capability probe must fire for fused-SDPA graphs.

``maybe_probe_for_graph`` used to scan the trace for a ``softmax`` node only.
Manual attention (``softmax(QKᵀ) @ V``) has one; the **fused**
``scaled_dot_product_attention`` op does not — its whole computation is a
single node.  So a graph whose attention came from ``F.scaled_dot_product_
attention`` (i.e. every model-zoo family since the fused-SDPA migration, plus
``nn.MultiheadAttention``) never probed, the workaround flag stayed pinned at
its conservative ``-1``, and every such compile emitted the transposed
value-matmul forever — on hardware where the MPSGraph fusion bug does not even
exist.

Measured cost of that on an unaffected GPU (B=2, H=8, D=64, decomposed path):

    T= 128   0.687 ms -> 0.409 ms   (1.68x)
    T= 256   0.702 ms -> 0.508 ms   (1.38x)
    T= 512   1.213 ms -> 0.958 ms   (1.27x)
    T=1024   2.805 ms -> 2.606 ms   (1.08x)

The win shrinks as T grows because the transpose amortises against the O(T²)
matmul, so this mattered most for short-sequence inference.

Correctness was never at risk: ``-1`` means "apply the workaround", which is
the always-correct setting.  This is purely a performance gate that never
opened.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._C import engine as _C_engine
from lucid.compile._core import attention_probe
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _FusedSdpa(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class _ManualSdpa(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        scores = (q @ k.swapaxes(-1, -2)) * (int(q.shape[-1]) ** -0.5)
        return F.softmax(scores, dim=-1) @ v


def test_fused_sdpa_is_a_probe_trigger():
    """The fused op must be in the trigger set — it emits no softmax node."""
    assert "scaled_dot_product_attention" in attention_probe._PROBE_TRIGGER_OPS
    assert "softmax" in attention_probe._PROBE_TRIGGER_OPS


def _probe_runs_for(module: nn.Module) -> bool:
    """Compile ``module`` from an unprobed state; report whether it probed."""
    attention_probe._probed = False
    _C_engine.compile.set_attention_workaround_state(-1)
    q, k, v = (metal_tensor(2, 8, 32, 64) for _ in range(3))
    compiled = lucid.compile(module)
    compiled(q, k, v).numpy()
    return _C_engine.compile.attention_workaround_state() != -1


def test_probe_runs_for_fused_sdpa_graph():
    """This is the regression: it used to leave the flag at -1 forever."""
    assert _probe_runs_for(_FusedSdpa())


def test_probe_still_runs_for_manual_attention_graph():
    assert _probe_runs_for(_ManualSdpa())


def test_probe_skipped_for_attention_free_graph():
    """A plain MLP must not pay the one-time probe cost."""
    attention_probe._probed = False
    _C_engine.compile.set_attention_workaround_state(-1)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    mlp.to(COMPILE_DEVICE)
    compiled = lucid.compile(mlp)
    compiled(metal_tensor(4, 16)).numpy()
    assert _C_engine.compile.attention_workaround_state() == -1


def test_fused_sdpa_output_correct_in_both_workaround_states():
    """Flipping the gate must not change results — only speed."""
    q, k, v = (metal_tensor(2, 8, 32, 64) for _ in range(3))
    module = _FusedSdpa()
    eager = module(q, k, v).numpy()
    for state in (1, 0):
        _C_engine.compile.set_attention_workaround_state(state)
        compiled = lucid.compile(_FusedSdpa())
        got = compiled(q, k, v).numpy()
        assert abs(got - eager).max() < 1e-5, f"workaround_state={state}"
