"""Compiled training must preserve the observable BatchNorm counter."""

from typing import override
from types import SimpleNamespace

import pytest

import lucid
from lucid.compile import compiled_step, fused_step, make_step


class _Model(lucid.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active = lucid.nn.BatchNorm1d(4)
        self.unused = lucid.nn.BatchNorm1d(4, affine=False)

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:  # type: ignore[override]
        return self.active(x)


@pytest.mark.parametrize("entry", ["fused", "make", "direct"])
def test_only_executed_batchnorm_advances_once_per_step(entry: str) -> None:
    model = _Model().to("metal").train()
    x = lucid.randn(8, 4, device="metal")
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.001)
    if entry == "fused":
        step = fused_step(model, lambda out: out.square().mean(), optimizer)
    elif entry == "make":
        step = make_step(model, lambda out: out.square().mean())
    else:
        step = lambda value: compiled_step(model, value, lambda out: out.square().mean())
    for expected_count in range(1, 4):
        step(x).item()
        assert model.active.num_batches_tracked.item() == expected_count
        assert model.unused.num_batches_tracked.item() == 0


class _Head(lucid.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = lucid.nn.Linear(4, 3)
        self.bn = lucid.nn.BatchNorm1d(3)

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:  # type: ignore[override]
        return self.bn(self.fc(x))


@pytest.mark.parametrize(
    "entry, device",
    [
        ("make", "cpu"),
        ("make", "metal"),
        ("fused", "metal"),
        ("direct", "metal"),
    ],
)
def test_the_counter_advances_in_the_tensor_a_holder_already_has(entry: str, device: str) -> None:
    # The compiled step rebound the buffer to ``counter + count``: the live
    # attribute counted, a reference taken before the step stayed at 0
    # (LCD-503).  The eager layer counts in place.  A CPU make_step runs
    # eager, which pins that side of the rule.
    model = _Head().to(device).train()
    counter = model.bn.num_batches_tracked
    held = model.state_dict()["bn.num_batches_tracked"]
    x = lucid.randn(8, 4, device=device)
    loss = lambda out: out.square().mean()
    if entry == "fused":
        step = fused_step(model, loss, lucid.optim.SGD(model.parameters(), lr=0.01))
    elif entry == "make":
        step = make_step(model, loss)
    else:
        step = lambda value: compiled_step(model, value, loss)
    for expected in range(1, 4):
        step(x).item()
        assert model.bn.num_batches_tracked is counter
        assert counter.item() == expected
        assert held.item() == expected


def test_counter_plan_counts_repeated_nodes_not_registered_layers() -> None:
    from lucid._dispatch import _unwrap
    from lucid.compile._core.bn_runstats import bn_counter_targets, advance_bn_counters

    model = _Model()
    ext = {7: _unwrap(model.active.running_mean)}
    node = SimpleNamespace(name="batch_norm1d", inputs=[1, 2, 3, 7, 8], outputs=[])
    graph = SimpleNamespace(ops=[node, node])
    targets = bn_counter_targets(model, graph, ext)
    assert targets == [(model.active, 2)]
    advance_bn_counters(targets)
    assert model.active.num_batches_tracked.item() == 2
    assert model.unused.num_batches_tracked.item() == 0


def test_trace_keeps_live_buffers_unchanged_but_retains_chained_values() -> None:
    from lucid.compile import _tracing
    from lucid.compile._core.bn_runstats import bn_writeback_targets
    from lucid._dispatch import _wrap

    model = _Model().to("metal").train()
    eager = _Model().to("metal").train()
    eager.load_state_dict(model.state_dict())
    x = lucid.arange(32, dtype=lucid.float32, device="metal").reshape(8, 4)
    with lucid.no_grad():
        eager(x)
        eager(3 * x + 2)
        with _tracing() as trace:
            model(x)
            model(3 * x + 2)
    assert model.active.running_mean.tolist() == [0.0] * 4
    assert model.active.running_var.tolist() == [1.0] * 4
    assert model.active.num_batches_tracked.item() == 0
    targets = bn_writeback_targets(trace.graph, trace.external_feeds)
    assert len(targets) == 2
    retained = trace.retained_values
    mean, variance = [_wrap(retained[out_id]) for _, out_id, _ in targets]
    assert lucid.allclose(mean, eager.active.running_mean, atol=1e-5)
    assert lucid.allclose(variance, eager.active.running_var, atol=1e-4)


@pytest.mark.parametrize("entry", ["fused", "make", "direct"])
def test_shared_batchnorm_chains_running_stats_and_counts(entry: str) -> None:
    class Shared(_Model):
        @override
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:  # type: ignore[override]
            return self.active(x) + self.active(x * 3 + 2)

    model = Shared().to("metal").train()
    eager = Shared().to("metal").train()
    eager.load_state_dict(model.state_dict())
    x = lucid.arange(32, dtype=lucid.float32, device="metal").reshape(8, 4)
    loss = lambda out: out.square().mean()
    if entry == "fused":
        step = fused_step(model, loss, lucid.optim.SGD(model.parameters(), lr=0.001))
    elif entry == "make":
        step = make_step(model, loss)
    else:
        step = lambda value: compiled_step(model, value, loss)
    for index in range(3):
        values = x + index
        eager(values).sum().item()
        step(values).item()
        assert model.active.num_batches_tracked.item() == 2 * (index + 1)
        assert model.unused.num_batches_tracked.item() == 0
        assert lucid.allclose(model.active.running_mean, eager.active.running_mean, atol=1e-5)
        assert lucid.allclose(model.active.running_var, eager.active.running_var, atol=1e-4)
