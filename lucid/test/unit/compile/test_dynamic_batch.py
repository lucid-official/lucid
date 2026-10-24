"""Dynamic batch: ``dynamic=True`` attempts a shared symbolic-batch executable.

``lucid.compile(model, dynamic=True)`` declares that the module will be called
with varying batch sizes and **attempts** to compile a single symbolic-batch
executable shared across all of them (one cache entry for every batch). A safety
gate (``graph_symbolic_safe``) decides per-model on the first trace:

* **gate clears the graph** (no batch-baking broadcast / batch-axis join /
  batch-shaped factory) → one symbolic executable, reused for every batch size.
* **gate rejects it, or the symbolic lowering fails** (e.g. an off-dim-0 view) →
  robust per-shape static caching (one executable per distinct shape). Correct,
  never crashes — just recompiles per shape.

So `dynamic=True` never crashes a real model: it shares one executable where
provably safe and falls back to per-shape static otherwise. `LUCID_COMPILE_DYNAMIC=0`
forces pure static (no symbolic attempt). The compiled training step
(``make_step``) is always per-shape static.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_cache_hit,
    metal_tensor,
    to_metal,
)


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


@pytest.mark.usefixtures("inference")
class TestSymbolicByDefault:
    """Gate-clean graphs share ONE symbolic-batch executable, no env var needed."""

    def test_mlp_one_shared_executable(self) -> None:
        m = to_metal(
            nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (8, 16, 32):
            x = metal_tensor(bs, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert cm._symbolic_resolved is True
        assert_cache_hit(cm, 1)  # SHARED across every batch size, no env opt-in

    def test_transformer_one_shared_executable(self) -> None:
        m = to_metal(
            nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, batch_first=True)
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 7, 32)
            assert _maxdiff(cm(x), m(x)) < 1e-3
        assert cm._symbolic_resolved is True
        assert_cache_hit(cm, 1)

    def test_cnn_one_shared_executable(self) -> None:
        m = to_metal(
            nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(8, 10),
            )
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 3, 8, 8)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert cm._symbolic_resolved is True
        assert_cache_hit(cm, 1)

    def test_scalar_arithmetic_symbolic(self) -> None:
        # Scalar broadcast (`x*0.5`, `1.0-x`, `(x-μ)/2`), where, manual LayerNorm
        # all clear the gate and ride one symbolic executable.
        class _Net(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                y = (x - mu) * 0.5 + 1.0
                return lucid.where(y > 0.0, y, y * 0.25)

        m = to_metal(_Net()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 6, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert cm._symbolic_resolved is True
        assert_cache_hit(cm, 1)

    def test_feature_axis_concat_symbolic(self) -> None:
        # Concatenating on a NON-batch axis is gate-safe (only batch-axis joins
        # are rejected).
        class _Cat(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = nn.Linear(16, 8)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.cat([self.l(x), self.l(x)], dim=1)

        m = to_metal(_Cat()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert cm._symbolic_resolved is True
        assert_cache_hit(cm, 1)


@pytest.mark.usefixtures("inference")
class TestGateFallsBackToStatic:
    """Gate-unsafe graphs fall back to per-shape static — correct, never crashes."""

    def test_batch_shaped_factory_falls_back(self) -> None:
        # zeros_like(x) bakes the batch into a constant shape → gate rejects →
        # per-shape static (one executable per distinct batch), not a crash.
        class _Z(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + lucid.zeros_like(x)

        m = to_metal(_Z()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 8)
            assert _maxdiff(cm(x), m(x)) < 1e-5
        assert cm._symbolic_resolved is False
        assert_cache_hit(cm, 2)  # per-shape

    def test_rnn_zero_hidden_init_falls_back(self) -> None:
        # GRU's default zero hidden state is a batch-shaped factory → static.
        m = to_metal(nn.GRU(8, 16, batch_first=True))
        m.eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 5, 8)
            out_c, _ = cm(x)
            out_e, _ = m(x)
            assert _maxdiff(out_c, out_e) < 1e-4
        assert cm._symbolic_resolved is False

    def test_off_dim0_view_falls_back_not_corrupts(self) -> None:
        # A view that moves the batch off dim 0 (unsqueeze-at-front) clears the
        # gate but the emitter rejects the symbolic lowering (returns nil), so
        # the compile retries static — correct results, no silent mis-shaping.
        class _OffDim0(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                y = x.unsqueeze(0)  # (B, F) -> (1, B, F)
                y = y + 1.0
                return y.squeeze(0)

        m = to_metal(_OffDim0()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 5)
            assert _maxdiff(cm(x), m(x)) < 1e-5
        assert cm._symbolic_resolved is False


@pytest.mark.usefixtures("inference")
class TestForceStaticOptOut:
    """``LUCID_COMPILE_DYNAMIC=0`` forces pure per-shape static (no symbolic)."""

    def test_env_disables_symbolic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LUCID_COMPILE_DYNAMIC", "0")
        m = to_metal(
            nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        assert cm._symbolic is False  # symbolic attempt disabled
        for bs in (8, 16):
            x = metal_tensor(bs, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 2)  # per-shape


class TestMakeStepDynamicIsStatic:
    """``make_step(dynamic=True)`` is always per-shape static (backward unsafe)."""

    def test_cross_entropy_no_crash(self) -> None:
        m = to_metal(nn.Linear(16, 4))
        step = make_step(m, lambda y, t: F.cross_entropy(y, t), dynamic=True)
        for bs in (8, 16):
            x = metal_tensor(bs, 16)
            t = lucid.zeros(bs, dtype=lucid.int64).to(COMPILE_DEVICE)
            loss = step(x, t)
            loss.backward()
        assert len(step.cache) == 2  # one fwd+bwd executable per distinct shape
